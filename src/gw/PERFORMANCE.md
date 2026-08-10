# `gw.gw_jax` performance — a 2026-04-25 measurement record

> **Every number below is historical; the methodology is not (banner added
> 2026-08-09).** This document was last measured on 2026-04-25, before the
> services layer existed, so the timings, the compile counts and the
> next-target backlog all describe a tree that has since been restructured —
> `PROFILING_TOOLS.md` §12 in the BSE-perf campaign directory reaches the same
> verdict and is the reason this banner exists. Read the *approach* here (find
> the eager-pjit sites, jit-wrap the small helpers, re-measure the cache-miss
> count) and take none of the figures as a current baseline.
>
> The tooling this document invokes is also gone. `scripts/profiling/` now
> holds a single file, `aot_cufft_sanity.py`; the `pf.py`,
> `analyze_compile_log.py` and `run_profiled.py` scripts the commands below
> call no longer exist in the tree or in the sandbox path they are spelled
> with. The run directories (`runs/Si/…`) are gone too — there is no `runs/`
> tree — and the `lorrax_B` / `lorrax_X` modulefiles have been replaced by the
> single `lorrax` module under `config/modulefiles/`. The commands are kept
> as written because they record what was actually run; they will not execute
> as-is.

Last updated: 2026-04-25 (agent-B/sternheimer-solver).
Companion to `src/psp/PERFORMANCE.md` (sister doc on the Sternheimer
side).

## Reference workload

Si 4×4×4 COHSEX (static), 64 k-points no-symmetry, scalar (`bispinor=False`),
8v + 52c (60 bands), 480 ISDF centroids, `memory_per_device_gb=28`,
`use_chunked_isdf=true`, `use_ffi_io=true`, default `auto → high_mem`
ISDF mode.  Run dir: `runs/Si/B_compile_floor_check/`.  Canonical
invocation:

```bash
LORRAX_NGPU=4 lxrun python3 -u -m gw.gw_jax -i cohsex.in
```

## End-to-end timing

Wall, on a fresh-from-`lxalloc` 4×A100 node, all numbers from the
LORRAX-internal `Total recorded` block + run wall, after the perf
arc on this branch:

| Phase                         | Wall    | Notes |
| :---------------------------- | ------: | :---- |
| `load_centroid_wfns`          | 1.2 s   | scatter + make_global_array |
| `zeta_fit_chunked`            | 11.0 s  | 6 s of which is `close_io` (phdf5 finalize) |
| `V_q_compute`                 | 5.0 s   | includes batched FFI cusolverMp |
| `wavefunction_setup`          | 0.04 s  |  |
| `chi0_W`                      | 1.4 s   | 0.5 s compile + 0.9 s exec |
| `sigma`                       | 3.3 s   | Σ_X + Σ_C diagonals |
| **LORRAX-recorded total**     | **~22 s** | |
| **End-to-end wall**           | **~32 s** (warm) / **~33 s** (cold cache) | + JAX init + module import |

## Compilation breakdown

JAX persistent compile cache: `$JAX_COMPILATION_CACHE_DIR` (set by
`lorrax_X` modulefile to `$SCRATCH/.jax_cache`).  Override via env var
**before** `module load`:

```bash
export JAX_COMPILATION_CACHE_DIR=/path/to/fresh_cache
module load lorrax_B
```

Per `pf.attach_compile_log` + `analyze_compile_log.py` on the
canonical run:

| | Cold cache (fresh) | Warm cache | Pre-perf-arc warm (baseline) |
| :--- | ---: | ---: | ---: |
| Total wall | 33 s | 32 s | 36 s |
| **XLA compile wall** | **11.9 s** | **7.8–8.2 s** | **10.2 s** |
| jaxpr→MLIR | 0.85 s | 0.80 s | 1.07 s |
| trace+transform | 0.88 s | 0.65 s | 0.60 s |
| **Compile count** | **217** | **~242** | **369** |
| **Cache misses** | — | **~231** | **321** |

Profiling tooling:

```bash
LORRAX_NGPU=4 lxrun python3 -u \
    /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/run_profiled.py \
    --out gwjax_profile --no-trace --no-hlo \
    -m gw.gw_jax -i cohsex.in

python3 /pscratch/sd/j/jackm/lorrax_sandbox/scripts/profiling/analyze_compile_log.py \
    gwjax_profile --top 60 --out-md gwjax_profile/compile_summary.md
```

Output: `compile_summary.md` ranks XLA compiles by total wall, and
also lists every `TRACING CACHE MISS` location with its `because:`
reason — read this BEFORE editing code on the compile-time path.

The `restart=true` mode skips wfn load + ISDF zeta-fit and gives a
useful test of where the compile-cache load lives — about half the
cache misses live in those pre-stages.

## What lives in the remaining ~8 s of XLA compile

After the 5 fixes on this branch, the warm-cache hot list (top
non-trivial XLA compiles by total wall):

| jit name                | Count | Total s | Notes |
| :---------------------- | ----: | ------: | :---- |
| `_kernel`               | 2     | 0.99    | ISDF zeta-fit per-rchunk kernel |
| `gather`                | 15    | 1.0     | mostly slice helpers, partly pre-jit eager |
| `sigma_sx`              | 2     | 0.79    | bare exchange diagonal |
| `_compute_CCT_LR`       | 2     | 0.49    | ISDF Cholesky CC^T builder |
| `_batched_chol`         | 1     | 0.51    | Cholesky on the ISDF-fit matrix |
| `eigh`                  | 3     | 0.45    | spread across QP rotations + gauge fix |
| `_fft_gather_reshard`   | 1     | 0.42    | ζ-G FFT axis reshard |
| `sigma_coh`             | 1     | 0.41    | dynamic Σ-COH at frequencies |
| `minimax_tau_integrate_chi` | 1 | 0.37    | imag-axis χ → minimax W mapping |
| (long tail of `broadcast_in_dim`, `add`, `convert_element_type`, etc.) | — | ~3 s | residual eager-pjit |

The big functional kernels (`_kernel`, `sigma_sx`, `sigma_coh`,
`_batched_chol`, `_compute_CCT_LR`, `_fft_gather_reshard`,
`minimax_tau_integrate_chi`, `_solve_w`) are physics work — they each
compile once and persist across runs in the cache.  The ~3 s of
residual eager-pjit is the next floor to chip away at.

## Fixes applied on this branch (perf arc)

Same playbook as the Sternheimer side: jit-wrap small Python helpers
that emit eager-pjit primitives line-by-line, so each call hits a
single cached XLA module instead of re-tracing.

  1. `gw/vcoul.py:wrap_points_to_voronoi` — `@jax.jit(nmax static)`.  ~32 misses → 1 (Voronoi-wrap of QMC samples in `compute_vcoul` 3D path).
  2. `gw/head_correction.py:expand_band_diagonal_to_kij` — thin wrapper + `_expand_band_diagonal_to_kij_jit(nk_tot, nb static)`.  ~27 misses → 4.
  3. `common/chi_from_dipole.py:compute_S_omega` — thin wrapper + `_compute_S_omega_jit(nelec, nb, omegas_is_scalar static)`.  ~30 misses → 1.
  4. `gw/wavefunction_bundle.py:{xn, xr, yn, yr}` — `@jax.jit(bands static)`.  ~17 misses → ~4 (one per unique slice in chi/W/Σ).
  5. `gw/compute_vcoul.py:_sqrt_v_phase_batch` — replace python-for-loop + `jnp.stack` of jit results with single `numpy`-side stack + `jnp.asarray` + `jax.vmap` of the jit'd kernel.  ~12 misses → ~0.

Cumulative impact (Si 4×4×4 COHSEX `restart=false`, warm cache):

|                          | Before | After | Δ |
| :----------------------- | -----: | ----: | ----: |
| XLA compiles             | 369    | 242   | −127 (−34 %) |
| Cache misses             | 321    | 231   | −90  (−28 %) |
| XLA compile wall         | 10.16 s | 7.77 s | −2.39 s (−24 %) |
| End-to-end wall          | 35.97 s | 31.75 s | −4.22 s |

Cold persistent cache: ~19 s → ~12 s of XLA wall (−37 %).

## Next-target backlog

Ranked by likely return / risk; each comes with a hypothesis.

1. **`common/fft_helpers.py:214` — actual shape polymorphism in FFT.**
   Same logical FFT (`fft_impl` inside `_make_axis_wrapper`) compiled at
   multiple `(4, 4, 480, 480, 4)`-class shapes (~9 misses).  Fix would
   need to be at the *consumers* — either pad the input to a common
   axis-stack shape or hoist the per-axis shape into a static arg.
   Invasive — touches many call sites.

2. **`file_io/_slab_io_ffi.py:382-385` — phdf5 per-rank closure.**  ~9
   misses across the `write_slab` `_per_rank` shard_map path.  The
   closure id changes per call site since shard_map captures a fresh
   `_per_rank` per cache key.  Hard to deduplicate without restructuring
   the shard_map path or hoisting `_per_rank` to module scope.

3. **`gw/w_isdf.py:375` (`flatten_V_qmunu`) and `:386` (`build_static_quadrature`'s `wfns.enk[:, s.cond]`).**  ~7 misses combined.  Trivial slicing emitted as eager `gather`.  Could `@jit` either the helpers or the consuming functions.  Tiny gain; low priority.

4. **`gw/vcoul.py:322,324` — QMC mean-replicate Python loop.**  Python `for rep in range(qmc_reps)` over a body that does eager `jnp.einsum`/`jnp.mean`/`jnp.stack`.  Wrapping in `@jax.jit` (with `qmc_reps` static or via `lax.scan`) would collapse the per-rep eager primitives.  Medium impact.

5. **Reduce the residual `broadcast_in_dim` / `add` / `convert_element_type` long tail** (~30 misses combined across 10+ small helper sites).  Diminishing returns; would need a systematic sweep.

## Numerical-equivalence smoke test

Always run before considering a compile-cache change "done":

```bash
cd runs/Si/C_main_baseline_si4
diff <(head -100 ../B_compile_floor_check/eqp0.dat) <(head -100 eqp0.dat)
```

The Si 4×4×4 baseline `eqp0.dat` is an ISDF-converged COHSEX QP table;
it should be bit-identical (no floating-point drift expected from the
jit-wrap pattern of fixes since they don't change op order).

## Cross-references

  - `src/psp/PERFORMANCE.md` — Sternheimer side, parallel arc
  - `runs/Si/B_compile_floor_check/` — where the profile runs and summaries
    lived. The `runs/` tree no longer exists; this is a provenance note.
  - `scripts/profiling/{pf.py, analyze_compile_log.py, run_profiled.py}` — the
    tooling these measurements were taken with. None of the three survives;
    `scripts/profiling/` now holds only `aot_cufft_sanity.py`. For profiling
    today, start from `PROFILING_TOOLS.md` in the BSE-perf campaign directory.
