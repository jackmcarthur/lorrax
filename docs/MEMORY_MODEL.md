# Memory Model and Chunk Size Optimization

The zeta-fitting pipeline allocates tensors in clearly defined stages.  Each
stage has a closed-form expression for its per-device footprint; the chunk
solver enforces those expressions.  This document lists every stage, the
associated arrays, and the formulas that drive the sizing.

LORRAX has **two** chunk choosers that can run in parallel:

1. **Heuristic (default)** — `gw/gw_init.py :: compute_optimal_chunks`.  Closed-form
   per-stage inversion of the hand-derived α coefficients below.  Always runs.
   Picks `band_chunk`, `chunk_r`, `q_chunk`, `q_gather`, `k_chunk`.
2. **AOT memory model (opt-in)** — `gw/aot_memory_model/`.  Per-kernel NNLS fits
   of `memory_analysis()` peaks against dimensional primitives, with an
   **analytic** closed-form chunk chooser and a **20/80** heuristic chooser
   that use those fits.  Enabled with `use_aot_chunk_chooser: true` in
   `cohsex.in` (default `false`).  When enabled, overrides the heuristic's
   `chunk_r` and `band_chunk`; keeps the heuristic's `q_chunk`, `q_gather`,
   `k_chunk`.  See [§AOT Memory Model](#aot-memory-model) below and the
   per-kernel artifact JSONs under
   [`src/gw/aot_memory_model/artifacts/`](../src/gw/aot_memory_model/artifacts/).

Conventions throughout this doc:

| Concept | Value |
|---|---|
| Element size | complex128 → 16 B (`_mem(…) = 16 · ∏dims / shard`) |
| Mesh axes | `'x'` = μ/centroid, `'y'` = r-chunk; `P = p_x · p_y` |
| Budget detection | `common.gpu_utils.get_device_memory_gb = 0.9 · bytes_available`; `bytes_available` from `jax.memory_stats().bytes_limit − bytes_in_use`, falls back to `nvidia-smi memory.free` |
| Target utilization | 0.97 of detected budget (`cohsex.in :: chunk_target_utilization`) |
| Safety cap (opt-in) | `ISDF_ZCT_STAGE_CAP_GB` / `ISDF_ZCT_STAGE_CAP_FRAC` env vars for the ZCT stage only |

Typical ranges (from production datasets):

| Symbol | Meaning                         | Typical Range |
|--------|---------------------------------|---------------|
| n_k    | total k-points                  | 1 – 2,000     |
| n_b    | union of bands loaded           | 20 – 5,000    |
| n_b^L  | left band count (b0→b3)         | 20 – 2,000    |
| n_b^R  | right band count (b1→b4)        | 20 – 5,000    |
| n_rmu  | ISDF interpolation points       | 200 – 50,000  |
| n_r    | real-space grid (nx*ny*nz)      | 20k – 2M      |
| n_q    | q-points (nkx*nky*nkz)          | 1 – 200       |

## Stage Summary

| Stage | Arrays (per device) | Bytes |
|-------|---------------------|-------|
| **Centroid load** | `psi_rmu_Y (n_k, n_b, n_s, n_rmu/p_y)` and `psi_rmuT_X (n_k, n_rmu/p_x, n_b, n_s)` | `M_full = 16 n_k n_s n_rmu n_b (1/p_x + 1/p_y)` |
| **Centroid copies** | Left+right slices (4 arrays) | `M_cent = 16 n_k n_s n_rmu (n_b^L+n_b^R)(1/p_x+1/p_y)` |
| **FFT workspace** | `psi_G` + `psi_r` + 2 FFT staging + `phase` | `4*16 n_k (B_b/P) n_s n_r + 16 n_k n_r` |
| **C_q build** | `P_l`, `P_r`, `C_q` | `M_cct = M_cent + 2*16 n_k (n_rmu/p_x)(n_rmu/p_y) + 16 n_q (n_rmu/p_x)(n_rmu/p_y)` |
| **Pair density (chunk)** | `psi_xchunk`, `P_l`, `P_r` | `M_pair = base + 16 n_k n_b n_s (B_r/p_y) + 2*16 n_k (n_rmu/p_x)(B_r/p_y)` |
| **ZCT** | `P_l`, `P_r`, FFT temps, `Z_q` | `M_zct = base + 4*16 n_k n_rmu B_r + 16 n_q n_rmu B_r` |
| **Solve** | `Z_col` input/output, triangular-solve temps, `L_q` temp, replicated `L` panels | `M_solve(B_q) = base + 4*16 n_q n_rmu (B_r/P) + M_L_q + B_q*16 n_rmu^2` |
| **V_q compute** | μ/ν chunks in r- and G-space | `M_vq ≈ 3 * 16 * μ_chunk * n_r` |

`base` in the chunk stages is `M_cent + M_L_q + cache`, i.e.
persistent centroids, the Cholesky factors, and (optionally) the cached
G-space wavefunctions.  The G-space cache can be turned off with
`use_phdf5_gspace: true` (cohsex.in), in which case `read_Gvecs_to_devices`
re-reads the G-space coefficients from the WFN via parallel HDF5 at each
r-chunk — zero device residency for `psiG_cache` at the cost of higher
per-r-chunk I/O.

## Band Chunk (`B_b`)

During centroid construction the solver must hold the FFT workspace plus the
persistent centroid arrays. The FFT peak is the dominant cost.

### FFT peak memory (measured)

The 3-D `ifftn` decomposes into three 1-D FFTs (x→y→z).  At peak four copies
of the shard `(n_k, B_b/P, n_s, n_r)` coexist — the G-space input, the
real-space output, and two staging buffers from the intermediate passes.
After the FFT finishes the staging buffers free and the phase multiply
briefly reuses their space, so the peak holds at 4× the shard:

```
M_fft_peak = 4 * 16 * n_k * (B_b / P) * n_s * n_r + 16 * n_k * n_r
```

The second term is the `(n_k, n_r)` phase array (replicated, <1% of the
first term).

**Measured validation** (each in a fresh process, 1 GPU):

| System | Shard (GB) | Peak (GB) | Ratio | 4× pred | Error |
|--------|-----------|-----------|-------|---------|-------|
| Si 24³, nk=64, nb=60 | 1.699 | 6.795 | 4.00× | 6.809 | 0.2% |
| 48³, nk=1, nb=80 | 0.283 | 1.132 | 4.00× | 1.134 | 0.2% |
| MoS2 24×24×80, nk=1, nb=160 | 0.236 | 1.074 | 4.55× | 0.944 | 14% |
| Si 24³, nk=216, nb=10 | 0.956 | 4.059 | 4.25× | 3.870 | 5% |

The 4× model is exact for large shards (>0.3 GB). For small shards,
a fixed overhead of ~0.03–0.1 GB adds ~10–15% above the 4× prediction.
This overhead comes from the cuFFT plan cache, the phase array broadcast,
and JIT compilation metadata.

### Band chunk constraint

```
M_full + 4 * 16 * n_k * (B_b / P) * n_s * n_r + 16 * n_k * n_r <= M_budget
```

so

```
B_b <= ((M_budget - M_full) - phase) * P / (4 * 16 * n_k * n_s * n_r)
```

`B_b` is clamped between 1 and `n_b`. If the numerator becomes negative the
system physically cannot fit; the solver raises a descriptive error.

## R-Chunk (`B_r = x_chunk_r`)

`B_r` is the number of contiguous r-points per chunk (`x_chunk · ny · nz`).
Four independent constraints must hold, with `base = M_cent + M_L_q + cache`.

**1. R-chunk wavefunction reshard.**  Between the FFT and the pair density
the wavefunctions reshard from band-sharded `{-, XY, -, -}` (FFT output) to
centroid-sharded `{-, -, -, Y}` (pair-density input) via all-gather along X
then all-to-all along Y.  The binding intermediate is the middle stage,
where each device holds `ceil(nb_pad / p_y)` bands on the full r-chunk:

```
M_reshard = 16 * n_k * ceil(nb_pad / p_y) * n_s * B_r
```

This is the largest single buffer in the r-chunk pipeline and often binds
`B_r`.  Concrete scale: Si 10×10×10 on a 4×4 mesh with `B_r = 12672` gives
`M_reshard = 16 · 1000 · 16 · 2 · 12672 = 6.5 GB`.  The reshard runs in a
separate JIT from the FFT; fusing them lets XLA's SPMD partitioner
rematerialize the FFT output to satisfy the output sharding (22+ GB on a
single device).

**2. Pair density build.**  `psi_xchunk_Y (n_k, n_b, n_s, B_r/p_y)` plus the
two `P_l / P_r (n_k, n_rmu/p_x, B_r/p_y)`:

```
base + 16 * (B_r / p_y) * [n_k·n_b·n_s + 2·n_k·(n_rmu / p_x)] <= M_budget
```

**3. ZCT pipeline.**  `P_l / P_r` and `Z_q (n_q, n_rmu/p_x, B_r/p_y)` — the
4× coefficient is the confirmed live-set model (4 concurrent pair-sized
temps, cf. AOT `zct_lr : β[PrBr] = 4`):

```
base + 16 * B_r * [(4·n_k + n_q) · n_rmu] <= M_budget
```

**4. Solve (with q_chunk = 1).**  `Z_col / zeta (n_q, n_rmu, B_r/P)` plus
one replicated `L_rep (n_rmu, n_rmu)`:

```
base + 4 * 16 * n_q * n_rmu * (B_r / P) + M_L_q + 16·n_rmu^2 <= M_budget
```

The solver takes the min over these four, rounds down to a multiple of
`P = p_x · p_y`, and halves `band_chunk` and retries if no feasible `B_r > 0`
exists (up to `bpd = 1`).  "Fill the GPU" behavior with no ad-hoc percentages.

## Q-Chunk (`B_q`)

With `B_r` fixed, the triangular solve baseline contains:

- `Z_col` input/output + triangular-solve temps: `4 * 16 * n_q * n_rmu * (B_r/P)`
- one local `L_q` temporary: `M_L_q`
- one replicated `L` panel for `q_chunk=1`: `16*n_rmu^2`

Additional `q_chunk` values add one replicated panel per extra q-point.
The bound is:

```
B_q <= 1 + (M_budget - (base + 4*M_Z_col + M_L_q + M_Lrep)) / M_Lrep
```

where `M_Lrep = 16*n_rmu^2`. `B_q` is clamped to `[1, n_q]`.

Bottleneck arrays: replicated Cholesky panels (`B_q` copies of `(n_rmu, n_rmu)`)
plus `Z_col`/`zeta` from the x-chunk stage.

## μ-Chunk for `V_q`

Building `V_q` from on-disk zeta holds three `(μ_chunk, n_r)` buffers
concurrently — `ζ_μ(r)`, its weighted G-space counterpart `ζ̃_μ(G)`, and a
second ν-block for off-diagonal tiles — plus the persistent `V_q (n_rmu,
n_rmu)` accumulator (on-GPU via `.at[].set()`, no device→host sync per block
since commit `b0e0f41`).  Available bytes are `effective_budget − M_cent`
because the centroids stay resident through COHSEX:

```
μ_chunk <= available_bytes / (3 * 16 * n_r)
```

The CLI reports this chunk alongside the others.

## Automatic Sizing Algorithm

The heuristic solver (`compute_optimal_chunks`) always runs.  When
`use_aot_chunk_chooser = true` the AOT chooser runs on top of it and
overrides the `chunk_r` / `band_chunk` it picked.

1. **Gather inputs**: `{n_k, n_b_full, n_b^L, n_b^R, n_rmu, n_q, fft_grid,
   memory_per_device_gb, mesh}`.

2. **Compute persistent costs**:
   - `M_full`, `M_cent`, `M_L_q`.
   - Validate `M_full + M_cent <= M_budget`.

3. **Band chunk**: Solve inequality above for `B_b`.

4. **C_q stage**: Ensure `M_cent + 2*M_pair(mumu) + M_C_q <= M_budget`.

5. **G-space cache**: Try enabling the cache, evaluate the chunk constraints
   with the cache bytes included, and fall back to “no cache” only if the
   memory system cannot fit even one x-slice otherwise.  The separate
   `use_phdf5_gspace: true` route skips this step entirely and reads G-space
   on demand from parallel HDF5.

6. **R-chunk**:
   - Derive analytic limits from the six per-stage linear models
     (`_max_cr` in `gw_init.py`).
   - Take the min over stages, round down to a multiple of `P = p_x · p_y`.
   - If no feasible `cr > 0` exists, halve `band_chunk` (and the `bpd`
     per-device share) and retry — up to `bpd = 1`.

7. **Q-chunk**: With `B_r` fixed, compute the available headroom for replicated
   `L` matrices and choose the largest integer chunk that fits.

8. **μ-chunk (V_q driver)**: Use the formula above with the persistent centroid
   cost to size the V_q μ-chunk.  The runtime also clamps this against a fresh
   `get_device_memory_gb` query and takes the minimum budget.

9. **AOT override (optional)**: If `use_aot_chunk_chooser = true`, call
   `choose_chunks_heuristic` (20/80 split) or `choose_chunks_analytic`
   (closed-form peak-bound inversion).  Both accept the same `SysDims`
   constructed from `meta` and rewrite `chunks['chunk_r']` / `chunks['band_chunk']`.
   The heuristic `_find_r_chunk` still populates `chunks['q_chunk']`,
   `chunks['q_gather']`, `chunks['k_chunk']`.

10. **Instrumentation**: the heuristic solver returns the chunk plan
    (`band_chunk`, `chunk_r`, `q_chunk`, `q_gather`, `k_chunk`,
    `use_gspace_cache`) plus a `memory_estimate` sub-dict holding
    `peak_estimate_gb` (max of all chunk-loop stages + pre-loop FFT + C_q),
    `budget_gb`, `bottleneck` (the binding stage name: `fft` / `pair` /
    `zct` / `reshard` / `solve` / `gather`), `available_vcoul_gb`
    (`m_budget − m_centroids`, headroom for V_q), and `limit_info` (per-stage
    peaks in GB).  The AOT chooser exposes the equivalent via
    `ChunkChoice.note` (`describe_chunks`), reporting either the α
    components (`α₀`, `α_cr`, `α_bc`, `α_crbc`) for the analytic path or the
    wfn/rchunk/persistent splits for the 20/80 heuristic.

## Working Backwards from a Failure

Solver errors always name the offending stage and the required GB.  Common
remedies: **centroid copy** — shrink `n_b_left`/`n_b_right` (sigma window) or
raise `memory_per_device_gb`.  **C_q build** — drop `n_rmu` or `n_q` (only
dials that touch the μ×μ stage).  **Chunk stage** (even `x_chunk = 1` and
`q_chunk = 1` don't fit) — lower `n_rmu`, use more devices, or raise the
budget.  **V_q μ-chunk** — grow the μ chunks or disable the G-space cache so
more memory is free when V_q runs.

## XProf Workflow

Recommended stack (June 2025 guidance): XProf + TensorBoard memory viewer.

1. Capture:
`uv run python tools/profile_gw_xprof.py -i <input.in> --workdir <run_dir> --logdir ./profiles/xprof --name <tag>`
2. Open UI:
`uv run xprof ./profiles/xprof`
3. Inspect:
Memory Viewer tab for `jit__compute_ZCT_LR(...)` and `jit__solve_all_q(...)`.

Recent production traces show:

- `jit__compute_ZCT_LR` peak dominated by five large buffers
  (`P_l`, `P_r`, two FFT temporaries, and output), which maps to the
  `4*n_k + n_q` coefficient in the ZCT constraint.
- `jit__solve_all_q` peak dominated by four `Z_col`-sized buffers plus
  an additional `L_q`-sized temporary.

## Current Bottleneck and Model Corrections

From the profiled production run
`profiles/xprof/cohsex_prod-20260303-112900/.../GPUtop.xplane.pb`:

- `jit__compute_ZCT_LR`: `peakHeapMib ~= 1794`
- `jit__solve_all_q`: `peakHeapMib ~= 1485`
- `jit__compute_P_traced`: `peakHeapMib ~= 480-493`

So the current bottleneck is the ZCT/solve region, not pair-density build.

Recent model corrections in `compute_optimal_chunks`:

1. ZCT stage coefficient updated from a too-optimistic `2*P + Z` estimate to
   the XProf-aligned `4*P + Z` live-set model.
2. Solve stage now explicitly includes replicated `L` panels in both
   feasibility checks and `q_chunk` sizing.
3. Breakdown output now reports ZCT and solve sub-components directly
   (`zct_pair_inputs`, `zct_fft_temps`, `solve_z_io`, `solve_tri_temps`, etc.).

The AOT memory model (§AOT Memory Model below) **independently confirms** each
of these coefficients by NNLS-fitting `memory_analysis()` against dimensional
primitives of the isolated jits — the per-kernel fits come out with integer
β's matching the hand-derived hypotheses (ZCT `β[PrBr]=4`, load-wfn
`β[psi_G]=3`, chi0 τ-step `β[Gbuf]=4`).

## AOT Memory Model

`src/gw/aot_memory_model/` (in-tree, entry points re-exported from the
package `__init__`).

The hand-derived formulas above are hypotheses about each stage's
allocation set.  When XLA schedules something the hand-model missed — remat,
allocator coalescing, a donation failure — the symptom is a runtime OOM at a
`chunk_r` the heuristic claimed would fit.  The AOT tool addresses this by
*measuring* each jit's peak via
`jax.jit(f).lower(specs).compile().memory_analysis()` at many `(sys, knobs,
mesh)` points, then NNLS-fitting a non-negative linear combination of
dimensional primitives:

```
peak(sys, knobs, mesh) = intercept + Σ_i  β_i · T_i(sys, knobs, mesh)
```

Each primitive `T_i` is the bytes of one physical shard (e.g. `(n_k, n_rmu,
B_r)/P`), and `β_i` counts concurrent copies at peak.  A clean fit gives
integer β's with MB-range residual RMS — that's the signal the primitive set
is complete.  Re-run the sweep when you modify a kernel; artifact JSONs are
checked in so every agent sees the same formulas.

### Architecture

```
aot_memory_model/
├── core.py                 SysDims, MeshSpec, Knobs, AotKernel, aot_measure,
│                           fit_nnls, predict_peak, save/load_fit
├── cost.py                 Analogous FLOPs fit: fit_cost_nnls,
│                           predict_flops_per_call, save/load_cost_fit
├── chooser.py              choose_chunks_analytic (closed-form)
│                           choose_chunks_heuristic (20/80 split)
│                           choose_chunks_aot (grid search)
├── doe.py                  build_doe_axes (one-at-a-time DoE generator)
├── presets.py              points_<kernel>() — per-kernel sample points
├── sweep.py                Rank-0 DoE driver: runs aot_measure + saves JSON
├── predict_cli.py          CLI: `python -m gw.aot_memory_model.predict_cli …`
├── kernels/                One module per covered jit (see table below)
└── artifacts/              Checked-in *__current__{samples,fit,cost_fit}.json
```

### Covered kernels (as of 2026-04-20)

| Kernel | Where in production | KNOBS | PRIMITIVES (fit β → integer count) |
|---|---|---|---|
| `cct_lr` | `common.isdf_fitting.compute_CCT_from_left_right` | — | `Pq` → 4 |
| `zct_lr` | `common.isdf_fitting.compute_ZCT_from_left_right_zchunk` | `chunk_r` | `PrBr` → **4** |
| `pair_density_traced` | `common.isdf_fitting.compute_pair_density_spin_traced` | — | `P`, `psiL`, `psiR` → 1,1,1 |
| `chi0_tau_step` | `gw.w_isdf._get_chi_minimax_kernel._tau_step` | — | `Gbuf`→**4**, `chi`→1, `psi`→1 |
| `solve_q` | `common.isdf_fitting.solve_zeta_from_L_q` | `chunk_r`, `q_chunk` | `Zcol`→3.6, `Lfull_rep`→2.4 (fractional — real-valued fit on a narrow DoE) |
| `slab_write` | `file_io.slab_io.SlabIO.write_slab` | `chunk_r` | `slab` → 1 |
| `load_psi_rchunk_fft` | `common.load_wfns.read_Gvecs_to_devices` + FFT | `k_chunk`, `band_chunk`, `chunk_r` | `psi_G`→**3**, `rchunk_xy`→1 |
| `load_psi_rchunk_reshard` | `common.load_wfns.iter_psi_rchunk_bandwise` reshard | `k_chunk`, `band_chunk`, `chunk_r` | `rchunk_xy`→1, `rchunk_y`→**2** |
| `sigma_kij` | `gw.ppm_sigma._get_sigma_kij_kernel` | — | `Gmid`→2.67, `Vmid`→2.33, `psi_X`→2, `psi_Y`→2, … |
| `vq_mu_chunk` | `gw.compute_vcoul.make_v_munu_chunked_kernel` | `mu_chunk` | `zeta`→**3**, `vphase`→1.5, `out`→1 |
| `fit_one_rchunk` | `common.isdf_fitting.fit_one_rchunk` (driver-level r-chunk body) | `chunk_r`, `band_chunk` | `pair`→**4**, `psiG_cache`→1, `centroid`→9, `Lq_rep`→1 |

Primitives in **bold** confirm the hand-derived α coefficients.  The
`fit_one_rchunk` composite kernel is the one the choosers consult — it
AOT-lowers the entire per-r-chunk driver body (FFT + reshard + pair-density
stream + ZCT + solve) in one HLO, so it captures *coexisting* buffers that
per-stage fits miss.

### NNLS fits in practice

```python
from gw.aot_memory_model import (
    SysDims, MeshSpec, Knobs, get_kernel, aot_measure, fit_nnls, save_fit
)
kernel = get_kernel("zct_lr")
points = [(SysDims(...), Knobs.of(chunk_r=cr), MeshSpec(2, 2)) for cr in (...)]
samples = [(sys, knobs, mesh,
            aot_measure(kernel, sys, knobs, mesh)["total"])
           for (sys, knobs, mesh) in points]
fit = fit_nnls(kernel, samples)
save_fit(fit, tag="current")
```

The `sweep` module wraps this end-to-end — rank 0 drives a preset DoE and
writes both `*_samples.json` and `*_fit.json` into `artifacts/`.  Sweeps
must run inside `lxrun` because they need a real JAX mesh, even though the
AOT lower/compile path allocates no GPU memory.

### Chooser modes

`gw_init.fit_zeta` consults the AOT chooser when
`cohsex.in :: use_aot_chunk_chooser = true`, picking one of three modes:

**Heuristic (default, `LORRAX_CHOOSER_MODE=heuristic`)** — `choose_chunks_heuristic`.
No DoE lookups; splits the budget 20/80 between wfn-FFT workspace and the
r-chunk body, then sizes against two calibrated integer β's:

```
wfn_budget    = 0.20 · budget           # k_chunk · bc · (3·16·n_s·n_r/P) ≤ wfn_budget
rchunk_budget = 0.80 · budget           # 4·16·n_k·n_rmu·cr/P + M_persistent ≤ rchunk_budget
```

The 3 is `β[psi_G]=3` from `load_psi_rchunk_fft`; the 4 is `β[pair]=4` from
`zct_lr` + `fit_one_rchunk`.  Prefers `chunk_r` / `band_chunk` values that
divide `n_rtot` / `n_b` so every loop iteration is one compile shape.

**Analytic (`LORRAX_CHOOSER_MODE=analytic`)** — `choose_chunks_analytic`.
Regroups the `fit_one_rchunk` β·T contributions into four scaling classes
(`PRIMITIVE_CLASSES` in `kernels/fit_one_rchunk.py` assigns each primitive
to `const` / `cr` / `bc` / `crbc`):

```
peak(chunk_r, bc) = α₀ + α_cr·chunk_r + α_bc·bc + α_crbc·(chunk_r·bc) ≤ M
```

Inverted in closed form per candidate `bc`; minimises total FLOPs via the
companion `cost_fit`.  Also enforces the post-jit allgather bound
(`16·q_gather·n_rmu·cr` bytes per device).

**Grid search** — `choose_chunks_aot`.  Coarse `(cr, bc)` grid + `predict_peak`.
For tests and manual exploration; not wired into the driver.

### γ calibration

`memory_analysis()` is an upper-bound *compiler* estimate — XLA schedules
tighter at runtime, and NCCL / cuFFT scratch sits outside the reported
peak.  Each `Fit` carries a scalar `γ = runtime_peak / aot_predicted`
applied uniformly to intercept and all β·T terms (fresh fits start at
`γ = 1.0`).  `gw_init.fit_zeta` prints `γ` after every ζ-fit so it can be
updated manually; a CLI to roll it into the artifact JSON is still TODO.

### When to trust which chooser

**Default heuristic** (`use_aot_chunk_chooser = false`) is the first choice:
α coefficients are physically motivated and stable across untested
(system, mesh) combinations.  The AOT **20/80 heuristic** is the natural
second stop — one free parameter (`wfn_workspace_frac`) and intentionally
conservative; useful for diagnosing "am I leaving memory on the table?".
The AOT **analytic** path is maximally precise *within the calibrated DoE
range* but can overfit on narrow sweeps (e.g. `solve_q` fractional β's from
7 points) and misses by ~10% when extrapolating far beyond them.

### Status (2026-04-20)

Scaffolded and calibrated at MoS2 3×3 and Si 4×4×4 scales; not the default
yet.  Known gaps: no multi-node DoE (some primitives are collinear at fixed
`total_devices` — e.g. `Lq_sharded` vs `Lq_rep` fold into one coefficient —
so predictions beyond 4 GPUs need validation); `q_chunk` / `q_gather` /
`k_chunk` stay on the hand-heuristic (the AOT chooser only overrides
`chunk_r` and `band_chunk`); `γ` is manual, not persisted to JSON, and is
currently wired into the analytic chooser only.  Per-kernel DoE caveats
(which primitives are collinear, which cross-terms were probed) live in the
`.notes` fields of the `*_fit.json` artifacts.
