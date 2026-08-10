# `psp.run_sternheimer` performance — a 2026-04-25 measurement record

> **Every number below is historical; the methodology is not (banner added
> 2026-08-09).** Like its companion `src/gw/PERFORMANCE.md`, this was measured
> on 2026-04-25, before the services layer existed, and
> `PROFILING_TOOLS.md` §12 in the BSE-perf campaign directory flags both files
> as pre-services and their figures as historical. The profiling commands
> below call `scripts/profiling/analyze_compile_log.py`, which no longer
> exists — that directory now holds only `aot_cufft_sanity.py` — and the run
> directory the reference workload names is gone with the rest of the `runs/`
> tree. Read the method, not the milliseconds.

Last updated: 2026-04-25 (agent-B/sternheimer-solver).

## Reference workload

MoS₂ 3×3 no-symmorphic, FR/nspinor=2, 26 valence × 9 full-BZ k-points,
FFT box (24, 24, 80), `ngkmax = 1963`, `n_cond_bands = 20` (Schur on),
`tol = 1e-6`, `max_iter = 80`.  Run dir:
`runs/MoS2/02_mos2_3x3_nosym/B_sternheimer_smoke/`.

## End-to-end timing

Wall, warm persistent cache (subsequent runs):

| Phase                           | Wall    | Note |
| :------------------------------ | ------: | :--- |
| `ρ_val from full-BZ sum`        | 1.0 s   | full-BZ density integration |
| `Potentials:` block             | 1.4 s   | `build_ionic_and_core` + `compute_V_H_and_V_xc` + `build_vnl_setup` (all GPU) |
| ψ unfold + H_k cache (9 k)      | 0.8 s   | host→GPU stack, vmap'd `setup_H_k_from_kvec` |
| **Setup total**                 | **3.4 s** |  |
| Cold q[0] (first jit invocation)| 3.4 s   | ≈ irreducible `_chi_at_q_jit` compile |
| Warm q (steady-state, 0.48 s/q) | ~3.8 s  | 8 × 0.48 |
| S-tensor at q=0                 | 2.4 s   | one-shot vmap-over-k full kernel |
| **Total cold-start (9 q + S)**  | **~13 s** |  |

XLA: 80 jit modules, 4.0 s total compile wall, 85 cache misses.

## Profiling tooling

```bash
PF_OUT=profile_run LORRAX_NGPU=1 lxrun python3 -u profile_run_sternheimer.py
python3 .../scripts/profiling/analyze_compile_log.py profile_run \
        --top 60 --out-md profile_run/compile_summary.md
```

Output: `compile_summary.md` lists XLA compiles + cache-miss reasons,
ranked by total wall.  Cold-start XLA wall scales with `9 × 26 × 2 ×
(24, 24, 80)` FFT plan-finding cost.

## Scoreboard since 2026-04-24

| Iteration                                    | XLA compiles | XLA wall | Setup wall | Cold q[0] | S-tensor wall |
| :------------------------------------------- | -----------: | -------: | ---------: | --------: | ------------: |
| Baseline (post-batched-pipeline)             | 209          | 4.7 s    | 19.4 s     | 5.1 s     | 6.2 s         |
| `_ionic_gspace_jit` (`2969d08`)              | 209          | 4.7 s    | 19.4 s     | 5.1 s     | 6.2 s         |
| Per-k jits (`_assemble_Z_jit`, `setup_H_k`, `_per_k_psi_to_masked_G`) (`45f253e`) | 70  | 3.9 s    | 17.9 s     | 5.0 s     | ~2.5 s warm   |
| GPU-batched Hankel transforms (`bb2a116`)    | 80           | 4.0 s    | 3.4 s      | 3.4 s     | 2.4 s         |
| **Current (`c9b1193`)**                      | **80**       | **4.0 s**| **3.4 s**  | **3.4 s** | **2.4 s**     |

## What the remaining 4 s of XLA compile contains

`_chi_at_q_jit` is **3.4 s of the 4 s total** — single largest module.
MLIR breakdown (12 distinct FFT call sites at `(9 × 26 × 2 × 24 × 24 × 80)`,
23 dot_generals, one `_sternheimer_core` while-loop body inlined with
`apply_H_k_from_G` — 2 FFTs/iter):

  * `_build_stk_at_q`     – per-q stack builder (4 FFTs for V_pert phase + Vu source)
  * vmap-over-k(`_per_k_chi`) at 9-k batch = 9-way fan-out:
    * `build_sternheimer_op_at_kvec_traced` — T_diag + Z (via `_assemble_Z_jit`)
    * `_sternheimer_core` — while-loop CG body
      * `apply_H_k_from_G` — IFFT/FFT pair + V_NL einsums per iter
    * `accumulate_chi_density` + `project_density_to_Gsphere` — 2 more FFTs
  * sum-over-k

cuFFT plan-finding for the `(24, 24, 80)` 3D batched FFT dominates.  XLA
optimization passes on the inlined while-loop body add the rest.

## Future optimization targets

Ranked by likely return / risk.  Each comes with a hypothesis to test
before implementing.

1. **Pre-stack `vnl_Z_full` from `H_cache` to skip `_assemble_Z_jit` inside the chi trace.**
   Z is fully determined by `kvec_kmq[k]` and `Gk_int[k]`, both already
   constant-per-q.  Only the `--with-derivatives` / S-tensor paths need
   Z to be re-built inside the linearize trace (so the q-tangent flows
   through the projector); chi-only could read from H_cache.  Estimated
   save: ~0.5 s of chi compile, no steady-state effect.  Risk: must
   keep two code paths (chi-only + with-derivs) and make sure neither
   silently leaks the wrong Z into a JVP.

2. **Reduce FFT call-site count in `_build_stk_at_q`.**
   The phase-wrap / V_pert-source build uses 4 FFTs.  Some of these
   are for the (1, 1, 1) constant V_pert and could be replaced with
   an analytic Fourier expression (V_pert ≡ 1 → only G=0 component
   survives).  Save: 1-2 of the 12 FFT plan-find compiles, ~0.3 s.

3. **`compute_dZ=True` velocity path is still eager.**
   `_build_vnl_kdata_core` keeps the `compute_dZ=True` block on numpy
   because of a per-channel Python loop.  Vectorise the dZ assembly
   along the same lines as `_assemble_Z_jit` so `get_dipole_mtxels`
   benefits from the same compile-once batched path.  Save: per-k
   dipole compute on cold start; small but consistent.

4. **Reconsider Schur warm-start cost-benefit.**
   On MoS₂ 3×3 with α_pv shift, Schur saves ~10 % iterations
   (47 vs 52 iters) at the cost of ~0.8 s compile.  Either:
   - benchmark with substantially larger `n_cond_bands` (40-100) where
     the explicit T-block actually preconditions the small-denominator
     modes — Cancès-et-al. target was 40 iters total
   - or default to `n_cond_bands=0` for systems where it doesn't pay,
     making it opt-in.
   Both need a wider system sweep first.

5. **Persistent-cache friendliness.**
   Any change to the chi jaxpr (e.g. swapping projector definition,
   changing static args) invalidates the cache and forces a 10+ s cold
   compile until it warms.  Worth keeping the public chi API stable
   across cosmetic refactors so Perlmutter-side persistent caches stay
   warm across iteration cycles.

## Numerical-equivalence smoke tests

Always run these before considering a chi-pipeline change "done":

  * `test_vnl_jit_equivalence.py`        – padded vs natural Gk_int build
  * `test_hankel_jax_equiv.py`           – scipy `hankel_l` vs JAX batch
  * `test_jvp_through_hankel.py`         – G_table / Gp_table / dZ
  * Full `--with-derivatives --s-tensor` run, χ / S compared against
    reference (machine-eps target: ~1e-11 rel for the S-tensor).

All four currently pass on `agent-B/sternheimer-solver`.
