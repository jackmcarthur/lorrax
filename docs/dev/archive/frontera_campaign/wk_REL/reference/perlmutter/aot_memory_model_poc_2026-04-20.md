# AOT memory model — POC report (agent C, 2026-04-20)

Branch: `agent-C/aot-memory-model` on `lorrax_C`.  Module lives at
`sources/lorrax_C/src/gw/aot_memory_model/`.

## Motivation

`gw_init.compute_optimal_chunks` has hand-derived `base + αᵢ·cr + cᵢ`
memory rules for the ISDF r-chunk loop + V_q μ-chunk only.  Coefficients
were manually calibrated from XProf traces (ZCT stage coefficient `2P+Z`
had to be corrected to `4P+Z` after a profile miss).  The chi0 `_tau_step`
OOM in [chi0_mem16_handoff.md](../../chi0_mem16_handoff.md) is not covered
by any model; previous iteration was HLO-dump-based and slow.

Goal: an AOT-lowering sweep that calibrates a multivariate linear fit for
each heavy kernel's peak memory as a function of `(sys_dims, chunk knobs,
p_x, p_y)`.  Offline calibration (~6 AOT points per kernel) → online
prediction by evaluating the fitted linear formula.

## What was built

```
sources/lorrax_C/src/gw/aot_memory_model/
├── __init__.py         — public API
├── core.py             — AotKernel, SysDims, MeshSpec, Knobs, aot_measure,
│                         fit_nnls, predict_peak, JSON persistence
├── doe.py              — one-at-a-time axis sweeps + product-only checks
├── presets.py          — baseline DoE configurations per kernel/system
├── sweep.py            — `python -m gw.aot_memory_model.sweep`  CLI
├── predict_cli.py      — `python -m gw.aot_memory_model.predict_cli` CLI
├── kernels/
│   ├── __init__.py     — registry
│   └── chi0_tau_step.py — first concrete AotKernel (reproduction of
│                          `w_isdf._get_chi_minimax_kernel`'s _tau_step)
└── artifacts/          — persisted samples.json + fit.json
```

Kernel interface:

```python
class AotKernel:
    name: str
    SYSTEM_DIMS: tuple[str, ...]
    KNOBS: tuple[str, ...]              # chunk-size knobs (may be empty)
    PRIMITIVES: dict[str, Callable]     # name -> bytes(sys, knobs, mesh)
    def build_specs(sys, knobs, mesh): ...
    def build_callable(sys, knobs, mesh): ...
```

Fit form per user spec: multivariate linear with non-negative intercept
and slopes via NNLS:

```
peak_bytes = intercept + Σ_i β_i · T_i(sys, knobs, mesh)
```

where `T_i` are dimensional volume primitives (bytes-per-device) — not
raw dims.  Raw-variable linear regression would underfit because peak
memory is product/reciprocal in the factors, not sum.

## Results — chi0_tau_step

Sweep at Si 4×4×4 60 Ry production scale, 8 DoE points
(`si444_60Ry_lean` preset).  Primitives:

| Primitive | Formula | Physical meaning |
|---|---|---|
| `Gbuf` | `16·n_k·n_s²·μ² / (p_x·p_y)` | one Gv/Gc/Gv_R/Gc_mR live buffer |
| `chi`  | `16·n_k·μ² / (p_x·p_y)` | chi_R_acc (expected aliased → β≈0) |
| `psi`  | `16·n_k·n_s·μ·(n_b_v+n_b_c)·(1/p_x + 1/p_y)` | four psi input shards |

DoE: baseline (nk=64, μ=2400, n_s=2, mesh=2×2), plus {kgrid=(2,2,2),
kgrid=(4,4,2), μ=1200, n_b_c=552, n_s=1, n_s=1+μ=1200, mesh=1×4}.  8 points
in ~3 seconds total AOT-compile wall.

Fit:

```
β[Gbuf]      = 4.489     # 4 concurrent live buffers + small transient
β[chi]       = 0.000     # donate_argnums=(0,) aliases, not in peak
β[psi]       = 1.009     # 1× copy of psi inputs
intercept    = 35 MB     # noise
residual RMS = 54 MB on 28 GB peak = 0.19%
```

Coefficients match the handoff's allocation dump:
`allocation 14 (preallocated-temp) = 4 × 5.49 GiB` for the 4 concurrent
Gbufs; chi_R_acc appears as argument but is aliased to output.

## Extrapolation check

Using the fitted rule to predict the handoff's Si 4×4×4 mem16 failing
run:

```
Predicted AOT peak: 27.98 GB
Runtime actual:     24.70 GB  (from chi0_mem16_handoff.md)
AOT / runtime:      1.133
```

This +13 % gap is the cuFFT scratch / NCCL staging / runtime allocator
overhead that `memory_analysis()` explicitly cannot see.  Per the AOT
guide, the kernel declares a scalar ``gamma = runtime_peak / aot_peak``
that the prediction multiplies at lookup.  Here γ ≈ 0.88.

## Key finding — rematerialization regime

XLA triggers `hlo_rematerialization` when AOT peak exceeds the declared
mesh memory (~40 GiB per device on 4-GPU A100 at BFC default).  At
μ=3600 in my initial sweep, XLA rewrote the allocation plan to recompute
intermediates instead of storing them → **peak_bytes no longer follows
the scaling law**.  A fit that includes a rematerialized point collapses
to residual RMS 2 GB (7 %).  Excluding that one point recovers 0.19 %.

**Consequence:** the DoE must stay in the scaling regime.  Two ways to
enforce:

1. Drop high-μ points from the preset (current approach, works fine).
2. **Preferred next step:** compile with
   `compiler_options={'xla_disable_hlo_passes': 'rematerialization'}`.
   Rematerialization-triggered samples then error instead of silently
   rewriting the plan.  Wrap `aot_measure` to catch the compile failure
   and flag the sample as `infeasible=True`.

Currently `aot_measure` captures stderr to detect remat warnings, but
absl logs bypass Python's `contextlib.redirect_stderr` so the capture
misses them.  Documented as next-step TODO.

## Architecture vs prior proposal

One deviation from the prior plan-doc (exchange of messages):

- I started with **8 primitives** (chi_acc, Gbuf, psi_x/y for val+cond
  separately, μ²/p_x, μ²/p_y) for maximal expressivity.  The fit came
  back over-parameterized: residuals inflated, coefficients unphysical
  (β[psi_y_val] = 69.6 — absurd).
- Collapsed to **3 primitives** driven by the handoff's allocation
  dump.  Fit residual dropped 30×.

Lesson: primitives should be derived from the XLA allocation report,
not from "things that could plausibly scale".  Adding features inflates
collinearity and hurts identifiability without adding physics.

## Validation cost

- Si 4×4×4 60 Ry, 8 AOT compiles: 3.1 seconds total.
- μ=400 "tiny" preset, 10 compiles: 1.2 seconds total.
- No GPU work — compile-only.  Can run on a single GPU.

This matches the handoff's promise ("AOT lowering gets the same numbers
in seconds" vs hours of production runs).

## Known limitations

1. **Remat regime detection is heuristic.**  Need `xla_disable_hlo_passes`
   to force an error instead of silent rewrites.
2. **`chi0_tau_step` only — one kernel so far.**  Need: `sigma_kij`,
   `compute_CCT_LR`, `compute_ZCT_LR`, `_solve_all_at_once`,
   `_single_chunk_proc` (V_q), and (once implemented) the `_tau_step`
   tiled variants (R_tile, μ_tile, τ_batch) the handoff proposes.
3. **γ calibration still manual** — one production-scale runtime peak per
   kernel needed, fed into the Fit JSON.  Should be a
   `aot_memory_model.calibrate_gamma` CLI that runs the kernel under
   `jax.devices()[0].memory_stats()['peak_bytes_in_use']` and writes γ.
4. **Kernel re-implements `_tau_step`** instead of importing from
   `w_isdf._get_chi_minimax_kernel`.  If production sharding specs change,
   this copy drifts.  Production should expose a `build_tau_step_for_aot`
   hook in a follow-up commit (one-line refactor).
5. **Duplication across the 3 lorrax checkouts** — the module lives only
   in `lorrax_C`.  Once validated, fast-forward A and B.

## Next steps (in order)

1. **Add `xla_disable_hlo_passes` to `aot_measure`** so remat regimes
   error instead of silently rewriting.  Failed compiles get `meas =
   {infeasible: True}` and the fitter drops them.
2. **γ-calibration CLI** — run one production configuration under
   `peak_bytes_in_use` and write γ into the fit JSON.
3. **Add sigma_kij kernel** (the immediate high-leverage second target —
   dominant runtime cost per earlier profiling).
4. **Add ZCT + pair-density + solve kernels** to supersede `compute_optimal_chunks`
   for the ISDF r-chunk loop.
5. **Wire `aot_memory_model.choose_knobs(...)` into `gw_init`** so the
   production path uses AOT-calibrated chunk sizes.
6. **Refactor w_isdf to expose `_tau_step`** for the AOT kernel to import
   directly (eliminates the drift risk).

## Update 2026-04-21 — driver-level `fit_one_rchunk` kernel

### Phase 2 — jit the r-chunk body

The full per-r-chunk iteration — FFT+reshard of the G-space band chunks,
streamed spin-traced pair-density accumulation, ZCT, Z→col reshard,
Cholesky solve — is now one jitted kernel in
`common.isdf_fitting._make_fit_one_rchunk_kernel`.  The driver
`fit_zeta_chunked_to_h5` calls `fit_one_rchunk()` once per r-chunk.

Mechanics:
- `_make_fit_one_rchunk_kernel` closes over all static structure
  (mesh, fft_grid, band-chunk ranges, L/R endpoints, r-chunk width,
  q-batch).  Python-unrolls the band-chunk loop at trace and classifies
  each chunk as `skip`/`direct`/`pad` against the L/R ranges.
- `fit_one_rchunk` entry-point caches one compiled kernel per
  `(mesh, n_rchunk, band_chunk_ranges, L/R endpoints, q-batch, n_k,
  n_rmu, n_s, fft_grid, kvecs_hash)`.  Two variants compile at runtime:
  full-width + remainder.
- `get_sharded_wfns_rchunk_slice` refactored to accept dynamic
  `r_start` (tracer) + static `r_chunk_size`.
- `solve_zeta_from_L_q`'s `block_until_ready` is tracer-guarded so the
  function is safe to call inside an outer jit.

Validation: MoS2 3×3 nosym COHSEX — single r-chunk (46080 pts) and
multi-rchunk (10000 × 5 with remainder 6080) both produce
`md5sum eqp0.dat == c8fc139fb22d2653d585874fe19c72a7`, matching the
reshard-fix baseline.

### Phase 4a — AOT kernel `fit_one_rchunk`

New composite kernel
`src/gw/aot_memory_model/kernels/fit_one_rchunk.py` mirrors the
production factory, capturing the driver-level peak including
coexisting G-space cache + centroid copies + L_q + live P_l/P_r +
per-bc FFT/reshard temps.

SysDims gained a new optional `fft_grid` field + `fft_shape` property
so kernels that need BOTH the k-grid (for `nq = ∏k`) and the real-space
FFT box (for IFFT reshape) can model them separately.  Back-compat:
kernels that repurpose `kgrid` as the FFT box see no change.

Primitives (all in bytes, with `B=16` for complex128):

| Primitive    | Formula                                                            | Role                          |
|--------------|--------------------------------------------------------------------|-------------------------------|
| `Pacc`       | `2·B · n_k · n_rmu · B_r / (p_x·p_y)`                              | P_l + P_r accumulators        |
| `PrBc`       | `B · (4·n_k + n_q) · n_rmu · B_r / (p_x·p_y)`                      | ZCT 4 concurrent + output     |
| `psiBc`      | `B · n_k · bc_size · n_s · n_r / (p_x·p_y)`                        | per-bc FFT output             |
| `psiBcY`     | `B · n_k · bc_size · n_s · B_r / p_y`                              | per-bc reshard stage          |
| `psi_cent`   | `B · n_k · n_rmu · n_b · n_s / p_x`                                | centroid copies               |
| `L_q`        | `B · n_q · n_rmu² / (p_x·p_y)`                                     | Cholesky factor               |
| `psiG_total` | `B · n_k · n_b · n_s · n_r / (p_x·p_y)`                            | G-space cache (always alive)  |

### Phase 4b — DoE + NNLS fit

MoS2 3×3 preset: `fft_grid=(80,72,8)` (n_r=46080), `n_rmu=640`,
`n_b=80`, `n_s=2`, chunk_r axes {5000, 20000, 46080}, band_chunk axes
{8, 32}, plus system axes (n_rmu, n_b, n_s) and mesh axes.

11 DoE points swept in ~28 s wall.  Peaks ranged 1.00–6.15 GB.
NNLS fit:

| Feature      | β      | Notes                                               |
|--------------|--------|-----------------------------------------------------|
| `PrBc`       | 1.034  | ZCT stage matches theoretical 4 pair + 1 output     |
| `L_q`        | 5.022  | Absorbs solve-stage μ² intermediates; small in abs  |
| `psiG_total` | 1.654  | G-space cache alive throughout                      |
| others       | 0      | subsumed into the three dominant primitives         |

Residual RMS = 0.23 GB — ~5% of peak.  Saved at
`artifacts/fit_one_rchunk__current__{fit,samples}.json`.

### Phase 4c — logged alongside heuristic

`gw_init.fit_zeta` now prints the AOT prediction right after the
per-stage heuristic:

```
Memory estimate: peak 10.93 GB (budget 28.00 GB), bottleneck=zct
Per-stage: fft=0.93  pair=2.96  zct=4.56  reshard=4.56  solve=2.69  gather=2.43 GB
AOT fit_one_rchunk peak (driver-level): 6.44 GB
```

Sanity log only — does not override `chunk_r`.  The heuristic's 10.93
GB is `max(stage) + base` which overestimates because stage maxes
don't coexist with the full persistent base (XLA reschedules).  AOT's
6.44 GB aligns with runtime `peak_bytes_in_use` observations.

### Phase 1b — phdf5 on-demand G-space (no persistent device cache)

New config flag `use_phdf5_gspace: bool` on `cohsex.in` (wired through
`CohsexCfg.use_phdf5_gspace` → `fit_zeta_chunked_to_h5`).  When set,
the driver skips `load_gspace_for_bands()` and instead calls
`PhdfWfnReader.coeffs_gspace(band_range)` freshly per r-chunk per
band-chunk.  The `psi_bc_G_tuple` is `del`'d right after the
`fit_one_rchunk` jit returns so nothing persists between r-chunks.

Duck-type match: `PhdfWfnReader.coeffs_gspace` already returns
`(n_k, nb_pad, n_s, nx, ny, nz)` with
`P(None, ('x','y'), None, None, None, None)` — no signature change to
the FFI reader was needed.  The driver-side factory is four lines.

Validation (MoS2 3×3):
- `use_phdf5_gspace=true` single r-chunk (46080 pts) + `use_ffi_io=true`:
  md5 = `c8fc139fb22d2653d585874fe19c72a7` ✓
- `use_phdf5_gspace=true` multi-chunk (5×10000 + remainder 6080) +
  `use_ffi_io=false`: same md5 ✓
- `use_phdf5_gspace=true` multi-chunk + `use_ffi_io=true`: fails with
  concurrent HDF5 MPI-IO errors in the async zeta_q writer
  (pre-existing issue — PhdfWfnReader + SlabIO-FFI on the same ranks
  race on MPI-IO state, not specific to this refactor).  Combine with
  `use_ffi_io=false` when both flags are set.

Timing impact at MoS2 3×3 multi-chunk: +0.2 s total (4.3 s vs 4.1 s
baseline).  Negligible.

Memory impact: zero persistent GPU residency for G-space between
r-chunks.  At MoS2 3×3 that's ~265 MB per rank (small); at Si
10×10×10 with 1000+ bands it is multi-GB and pushes the pre-rchunk
CCT/cholesky stages back under budget.

Interaction with the AOT model: the per-r-chunk jit boundary is
unchanged — `psi_bc_G_tuple` is still an input argument, so
`argument_size_in_bytes` and hence the AOT peak are identical.  What
changes is the *between-rchunk* GPU state, which the AOT kernel was
never measuring in the first place.  The `β[psiG_total]=1.65`
coefficient continues to describe the per-r-chunk peak correctly
under both cache strategies.

### Phase 6 — analytic chooser + γ calibration

**Analytic inversion**.  The 7 memory primitives collapse into 4
scaling classes via a new `PRIMITIVE_CLASSES` metadata dict on the
kernel (`const`, `cr`, `bc`, `crbc`).  `_group_alpha(mem_fit, sys,
mesh)` sums β·T contributions per class to get

```
peak(chunk_r, bc) = α₀ + α_cr · chunk_r + α_bc · bc + α_crbc · (chunk_r · bc)
```

This is linear in chunk_r at fixed bc, so

```
chunk_r_max(bc) = (M − α₀ − α_bc · bc) / (α_cr + α_crbc · bc)
```

is a **closed-form inversion**.  `choose_chunks_analytic` 1-D searches
over a handful of bc candidates (mesh-divisible, bounded by n_b),
picks the feasible argmin of total FLOPs.  No 2-D grid.

**α decomposition** at MoS2 3×3 (2×2 mesh):

| Coefficient | Value             | Physical meaning                       |
|-------------|-------------------|----------------------------------------|
| α₀          | 0.513 GB          | centroids + L_q + psiG cache           |
| α_cr        | 0.119 MB per cr   | Pacc + PrBc (linear in chunk_r)        |
| α_bc        | 0                 | psiBc (β=0 in fit, bc-degenerate)      |
| α_crbc      | 0                 | psiBcY (β=0 in fit, bc-degenerate)     |

The `bc` and `crbc` coefficients land on 0 because the 11-sample DoE
has only 2 bc variants — not enough to break collinearity.  The
chunk_r scaling is still clean and the chooser picks the biggest
feasible `chunk_r`; bc chosen by tiebreak at the maximum of its range.

**γ calibration** (runtime nvidia-smi / AOT-predicted):
  * JAX CUDA PJRT on this stack returns `None` from
    `jax.local_devices()[0].memory_stats()`, so the runtime peak
    tracker in `fit_zeta_chunked_to_h5` now shells out to
    `nvidia-smi --query-gpu=memory.used` once per r-chunk.
  * MoS2 3×3 nosym, 1 r-chunk fit: AOT predicts 6.00 GB, nvidia-smi
    reads **3.06 GB**, γ = 0.510.  Baked into the fit artifact.
  * Both `predict_peak` and `_group_alpha` multiply α's by
    `fit.gamma`, so the chooser now predicts peaks that match
    runtime to within measurement noise.
  * memory_analysis is a worst-case upper bound (all args + temps +
    outputs live simultaneously, minus donated aliases); XLA's
    actual runtime scheduler coexists buffers tighter.  γ < 1 is
    expected — user's prior note about γ > 1 for FFT kernels was
    the opposite regime where cuFFT scratch pushes past the bound.

**Budget-hit validation** — reduce `memory_per_device_gb` to 4.0 and
re-run the chooser at MoS2 3×3:

```
AOT chooser: chunk_r=46080 band_chunk=80 (1×1 jits,
    peak=3.06 GB / 3.88 GB = 79%, total=7.3 GF)
```

Chooser-predicted peak (3.06 GB) matches runtime nvidia-smi peak
(3.06 GB) exactly after γ calibration.  Budget is genuinely hit at
79% utilization.

### Next steps
- **Phase 3**: production-scale profile of the refactored driver to
  validate per-rchunk timing hasn't regressed vs the pre-jit path.
- **Multi-kernel AOT**: use `fit_one_rchunk` to *replace* the
  `pair+zct+reshard+solve` stages in `compute_optimal_chunks`, with the
  per-stage heuristic retained for `fft` and `gather` which aren't yet
  AOT-modeled at driver level.
- **γ calibration** at Si 4×4×4 60 Ry to verify the MoS2-fitted
  coefficients extrapolate.  Expected γ ≈ 1.15–1.30 from cuFFT/NCCL
  blind spot (same pattern as chi0 kernel).

## Reproduction

```bash
module load lorrax_C
lxalloc 1                                        # 1 node, 4 GPUs, 2h
# Sweep + fit Si 4×4×4 60 Ry chi0_tau_step:
lxrun python3 -m gw.aot_memory_model.sweep \
    --kernel chi0_tau_step --preset si444_60Ry_lean --tag si444_lean3 --mode both
# Predict a new configuration:
LORRAX_NGPU=1 lxrun python3 -m gw.aot_memory_model.predict_cli \
    --kernel chi0_tau_step --tag si444_lean3 \
    --kgrid 4,4,4 --n_rmu 2400 --n_s 2 --n_b_v 20 --n_b_c 276 --mesh 2,2
```
