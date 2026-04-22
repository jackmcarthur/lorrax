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

For automatic memory detection, `common.gpu_utils.get_device_memory_gb` uses
`budget = 0.9 * bytes_available`, with:
- `bytes_available = bytes_limit - bytes_in_use` from `jax.memory_stats()` when available
- fallback: `bytes_available = nvidia-smi memory.free`

That detector-side 10% guard band is the only default safety margin; the chunk
solver default is `target_utilization = 0.97` (configurable via
`cohsex.in :: chunk_target_utilization`).

The driver no longer applies an implicit ZCT soft cap.  If needed, set one
explicitly with `ISDF_ZCT_STAGE_CAP_GB` (absolute) or `ISDF_ZCT_STAGE_CAP_FRAC`
(fraction of total GPU memory).

All sizes below use complex128 storage (`bytes_per_complex = 16`).  Mesh axes
follow the code: `'x'` shards the μ/centroid axis, `'y'` shards r-chunks and
the band axis in FFTs, and the total processor count is `P = p_x * p_y`.

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

The 3D `ifftn` is decomposed into three 1D FFTs (x→y→z). At peak, four
copies of the per-device psi shard coexist:

1. **`psi_G`** (FFT input): `(n_k, B_b/P, n_s, n_r)` — the band-sharded
   G-space wavefunction being transformed
2. **`psi_r`** (FFT output): same shape — the real-space result
3. **FFT staging buffer 1**: same shape — intermediate after the x-FFT
4. **FFT staging buffer 2**: same shape — intermediate after the y-FFT

After the FFT completes, buffers 3–4 are freed and the phase multiply
allocates a fifth copy briefly (but by then the staging buffers are gone,
so the steady state is 3 copies). The peak is therefore **4×** the shard:

```
M_fft_peak = 4 * 16 * n_k * (B_b / P) * n_s * n_r + 16 * n_k * n_r
```

The second term is `phase_spatial` at `(n_k, n_r)`, replicated on all
devices. For most systems this is <1% of the first term.

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

`B_r` is the number of contiguous r-points processed per chunk
(`x_chunk * ny * nz`).  Four independent constraints must hold:

1. **R-chunk wavefunction reshard**

Before the pair density, the r-chunk wavefunctions must be resharded from
band-sharded `{-, XY, -, -}` (output of the FFT) to `{-, -, -, Y}` (input
for pair density). The reshard goes `{-, XY, -, -}` → `{-, Y, -, -}` →
`{-, -, -, Y}` via all-gather along X then all-to-all along Y.

The binding intermediate is at the `{-, Y, -, -}` stage, where each device
holds `ceil(nb_pad / p_y)` bands with the full r-chunk (not yet Y-sharded):

```
M_reshard = 16 * n_k * ceil(nb_pad / p_y) * n_s * B_r
```

This is the largest single buffer during the r-chunk pipeline and often
the binding constraint for `B_r`. For Si 10×10×10 on a 4×4 mesh with
`B_r = 12672`: `M_reshard = 16 × 1000 × 16 × 2 × 12672 = 6.5 GB`.

The reshard is executed in a separate JIT from the FFT to prevent XLA's
SPMD partitioner from rematerializing the FFT output to satisfy the
output sharding (which would require 22+ GB on a single device).

2. **Pair density build**

```
base + 16 * (B_r / p_y) * [n_k n_b n_s + 2 n_k (n_rmu / p_x)] <= M_budget
```

Bottleneck arrays: `psi_xchunk_Y (n_k, n_b, n_s, B_r/p_y)`,
`P_l`/`P_r (n_k, n_rmu/p_x, B_r/p_y)`.

3. **ZCT pipeline**

```
base + 16 * B_r * [ (4 n_k + n_q) * n_rmu ] <= M_budget
```

Bottleneck arrays: `P_l`/`P_r (n_k, n_rmu/p_x, B_r/p_y)`,
`Z_q (n_q, n_rmu/p_x, B_r/p_y)`.

4. **Solve (with q_chunk = 1)**

```
base + 4 * 16 * n_q * n_rmu * (B_r / P) + M_L_q + 16*n_rmu^2 <= M_budget
```

Bottleneck arrays: `Z_col`/`zeta (n_q, n_rmu, B_r/P)`,
`L_rep (n_rmu, n_rmu)` per q during the solve.

`base = M_cent + M_L_q + cache`.  The analytic upper bounds from these
inequalities provide an initial guess which is then rounded to an integer
number of x-slices, forced to be divisible by `p_y`, and iteratively reduced
until the evaluated stage peaks fall below the budget.  This preserves the
“fill the GPU” behavior without ad-hoc percentages.

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

When building `V_q` from the stored zeta HDF5 the code holds

1. `ζ_μ(r)` for the current μ-block,
2. its FFT/weighted counterpart `ζ̃_μ(G)`, and
3. a second ν-block when contracting off-diagonal tiles.

This leads to the simple bound used by the driver:

```
μ_chunk <= available_bytes / (3 * 16 * n_r)
```

The “available bytes” are `effective_budget - M_cent`, because the centroids
remain resident for the rest of the COHSEX calculation.  The CLI now reports
this chunk alongside the others so users can correlate V_q throughput with
memory settings.

Bottleneck arrays: `ζ_μ(r)` at `(μ_chunk, n_r)`, `ζ̃_μ(G)` at `(μ_chunk, n_r)`
after weighting, `ζ̃_ν(G)` for off-diagonal contraction, and `V_q` accumulator
at `(n_rmu, n_rmu)` on GPU. The accumulation is done entirely on GPU via
`.at[].set()` — no device→host sync per block (as of commit `b0e0f41`).

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

10. **Instrumentation**: The heuristic solver returns a top-level dict plus a
    `memory_estimate` sub-dict.  The fields `fit_zeta` prints:

    Top-level:

    - `band_chunk`, `chunk_r`, `q_chunk`, `q_gather`, `k_chunk`
    - `use_gspace_cache` (bool)

    Under `memory_estimate`:

    - `peak_estimate_gb` — max of the six chunk-loop stages + pre-loop FFT + C_q
    - `budget_gb` — `memory_per_device_gb`
    - `bottleneck` — name of the binding stage (`fft` / `pair` / `zct` /
      `reshard` / `solve` / `gather`)
    - `available_vcoul_gb` — `m_budget - m_centroids` (headroom for V_q μ-chunk)
    - `limit_info` — per-stage peaks in GB (`fft`, `pair`, `zct`, `reshard`,
      `solve`, `gather`)

    These map directly to the 6-stage table at the top.  For the AOT chooser
    the equivalent information comes from the `ChunkChoice.note` field
    (printed by `describe_chunks`), which lists the α components
    (`α₀`, `α_cr`, `α_bc`, `α_crbc`) for the analytic path, or the per-budget
    wfn / rchunk / persistent splits for the heuristic path.

## Working Backwards from a Failure

When the solver raises an error it always references the offending stage and
provides the required gigabytes.  Typical remedies are:

1. **Centroid copy failure**  
   Reduce `n_b_left`/`n_b_right` (e.g., shrink the sigma window) or increase
   `memory_per_device_gb`.

2. **C_q build failure**  
   Reduce `n_rmu` or `n_q`; these are the only dials that affect the μ×μ stage.

3. **Chunk stage failure**  
   This means even `x_chunk = 1` and `q_chunk = 1` would not fit.  Lower
   `n_rmu`, use more devices, or increase the budget.

4. **V_q μ-chunk failure**  
   Increase the μ chunks or disable caching so more memory is available to the
   V_q builder.

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

**Where**: `src/gw/aot_memory_model/` (Python package, in-tree).  Entry points
re-exported from `src.gw.aot_memory_model.__init__`.

**Motivation**: the hand-derived `compute_optimal_chunks` formulas above
describe a *hypothesis* about each stage's allocation set.  When XLA schedules
something we didn't predict (remat, allocator coalescing, donation failures),
the hand-model silently gets the peak wrong — the symptom is a runtime OOM at
a `chunk_r` the heuristic claimed would fit.  The AOT tool addresses this by
**measuring** the peak via
`jax.jit(f).lower(specs).compile().memory_analysis()` at many `(sys, knobs,
mesh)` points, then **fitting** a non-negative linear combination of
dimensional primitives:

```
peak(sys, knobs, mesh) = intercept + Σ_i  β_i · T_i(sys, knobs, mesh)
```

where each primitive `T_i` is bytes of one physical shard (e.g. `(n_k, n_rmu,
B_r)/P`), and `β_i` counts how many concurrent copies XLA holds alive at the
peak.  A clean fit gives integer β's; residual RMS in the MB range indicates
the primitive set is complete.  Update the fit by re-running the sweep when
you modify a kernel — the artifact JSONs are checked in so other agents see
the same formulas.

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
writes both `*_samples.json` and `*_fit.json` into `artifacts/`.  The
sweep **must** run inside `lxrun` (needs JAX + a mesh), even though the AOT
lower/compile path allocates no GPU memory.

### Chooser modes

`gw_init.fit_zeta` consults the AOT chooser when
`cohsex.in :: use_aot_chunk_chooser = true`:

```python
from gw.aot_memory_model import (
    choose_chunks_heuristic,   # default
    choose_chunks_analytic,    # LORRAX_CHOOSER_MODE=analytic
)
choice = choose_chunks_heuristic(  # or _analytic
    aot_sys, aot_mesh, budget_bytes=budget,
)
chunks['chunk_r']    = int(choice.chunk_r)
chunks['band_chunk'] = int(choice.band_chunk)
```

**Heuristic (default)** — `choose_chunks_heuristic` in `chooser.py`.
Budget-split without any regression:

```
wfn_budget   = 0.20 · budget       (wavefunction FFT workspace)
rchunk_budget = 0.80 · budget      (pair-density + ZCT + solve)
```

Pick `(k_chunk, band_chunk)` so `k_chunk · band_chunk · (3·16·n_s·n_r/P) ≤
wfn_budget` (the 3 is `β[psi_G]=3` from `load_psi_rchunk_fft`).  Pick
`chunk_r` so `4·16·n_k·n_rmu·chunk_r/P + M_persistent ≤ rchunk_budget` (the 4
is `β[pair]=4` from `zct_lr` + `fit_one_rchunk`).  Rounds to a divisor of
`n_rtot` / `n_b` when possible so every chunk is the same size (one compile
shape per loop).  No DoE lookups — the two integer β's are the only
calibrated constants.

**Analytic** — `choose_chunks_analytic` in `chooser.py`.  Regroups the
`fit_one_rchunk` primitive β·T contributions into four scaling classes:

```
peak(chunk_r, bc) = α₀ + α_cr·chunk_r + α_bc·bc + α_crbc·(chunk_r·bc)
```

(`PRIMITIVE_CLASSES` in `kernels/fit_one_rchunk.py` assigns each primitive to
a class — `pair`→cr, `psiG_bc`→bc, `psiY_bc`→crbc, everything else→const.)
Inverts the bilinear bound `peak ≤ M` in closed form for each candidate
`bc`, picks the `(cr, bc)` that minimizes total FLOPs (from the companion
`cost_fit`).  Applies a post-jit allgather bound (16·q_gather·n_rmu·cr
bytes per device must also fit).

**Grid-search fallback** — `choose_chunks_aot`.  Enumerates a coarse
`(cr, bc)` grid and calls `predict_peak`.  Used for unit tests and manual
exploration; not wired into the driver by default.

### γ calibration

`memory_analysis()` is an upper-bound *compiler estimate*.  XLA schedules
tighter at runtime — remat can happen even with
`xla_disable_hlo_passes=rematerialization` disabled if the initial plan
overflows, and NCCL / cuFFT scratch is outside the reported peak.  Each
`Fit` carries a scalar `gamma = runtime_peak / aot_predicted` applied
uniformly to the intercept and all β·T terms at prediction time.  Fresh fits
start at `γ = 1.0`.  `gw_init.fit_zeta` prints `γ = peak_gb / aot_peak_gb`
after every ζ-fit run so you can update it manually; a CLI to roll these
into the artifact JSON is TODO.

### When to trust which chooser

- **Default heuristic** (`use_aot_chunk_chooser = false`): stable across
  (system, mesh) combinations never seen in the DoE.  The α coefficients are
  physically motivated and change only when you modify the pipeline.  First
  choice unless you have evidence it OOMs.
- **AOT 20/80 heuristic** (`use_aot_chunk_chooser = true`,
  `LORRAX_CHOOSER_MODE=heuristic`): one free parameter
  (`wfn_workspace_frac`, default 0.2).  Good for investigating "why am I
  leaving memory on the table" — it's intentionally conservative.
- **AOT analytic** (`use_aot_chunk_chooser = true`,
  `LORRAX_CHOOSER_MODE=analytic`): maximally precise *within the DoE range*.
  Can over-fit on narrow sweeps (e.g. `solve_q__current__fit` fractional β's
  from a 7-point DoE).  Extrapolating far outside the calibrated range gives
  polite numbers that miss by ~10% on a new machine.

### Status (2026-04-20)

The AOT pipeline is **scaffolded and calibrated at MoS2 3×3 / Si 4×4×4
scales**.  Not yet the default.  Known gaps:

- Multi-node DoE missing — some primitives are collinear at fixed
  `total_devices` (e.g. `Lq_sharded` vs `Lq_rep` in `fit_one_rchunk`).  The
  fit absorbs them into one coefficient; validate before trusting
  predictions beyond 4 GPUs.
- `q_chunk`, `q_gather`, `k_chunk` are still picked by the hand-heuristic.
  `choose_chunks_aot` only overrides `chunk_r` and `band_chunk`.
- `γ` calibration is manual (printed each run, not persisted).
- The `γ` field in artifact JSON is not yet used by the choosers (added in
  `core.Fit`, wired through `_group_alpha` for the analytic chooser only).

See the per-kernel `.notes` fields in the `*_fit.json` artifacts for
DoE-specific caveats (which primitives are collinear, which cross-terms were
probed, etc.).
