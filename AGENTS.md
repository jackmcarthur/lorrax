# LORRAX Agent Guide

**LORRAX** (**Lo**w-scaling **R**eal-space **R**eal-**A**xis e**X**cited state package) —
JAX-based GW with ISDF compression. The GW driver is called **GWJAX**.

You are most likely arriving here from the `lorrax_sandbox` project, where test runs
and BGW comparisons are organized. This repo contains the source code you may need to
read or modify. Read this file upon first inspection of the LORRAX source before editing any code.

## Where things are

| Path | What | When to read |
|------|------|-------------|
| `src/gw/gw_jax.py` | Main GW driver | Any GW debugging |
| `src/gw/ppm_sigma.py` | GN-PPM dynamic self-energy Σ^c(ω) | Frequency-dependent sigma issues |
| `src/gw/w_isdf.py` | χ₀ → W screening pipeline | Screening / epsilon issues |
| `src/gw/minimax_screening.py` | PPM extraction, minimax window helpers | PPM parameter issues |
| `src/gw/gw_init.py` | Input parsing, ISDF fitting, memory model | Input file questions, chunk sizing |
| `src/gw/vcoul.py` | Coulomb potential (2D slab, 0D box) | Head corrections, truncation |
| `src/common/minimax.py` | Minimax quadrature solvers | Quadrature node/weight issues |
| `src/common/wfnreader.py` | WFN.h5 reader | Wavefunction loading |
| `src/psp/` | Pseudopotentials, dipole matrix elements | `dipole.h5` or `kin_ion.h5` issues |
| `src/isdf_init/kmeans_isdf.py` | ISDF centroid generation | Centroid count / quality |

## Key documentation

| Doc | What it covers |
|-----|---------------|
| `docs/PHYSICS_COMPREHENSIVE.md` | ISDF theory, GW equations, COHSEX, CTSP formalism |
| `docs/CODEBASE_COMPREHENSIVE.md` | Module map, data flow, key classes, sharding patterns |
| `docs/MEMORY_MODEL.md` | Per-stage memory formulas, chunk sizing, bottleneck arrays |
| `docs/MINIMAX_QUADRATURE.md` | GL/HGL quadrature, error scaling, crossing windows |
| `docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md` | PPM sigma pipeline: phases, windows, evaluation loop |
| `docs/SIGMA_FREQ_AUDIT_STATUS.md` | Current BGW vs GWJAX comparison status, known offsets |

## How to run

```bash
# Preprocessing (centroids, dipole, kin+ion)
uv run python -m centroid.kmeans_isdf -i cohsex.in 600
uv run python -m psp.get_dipole_mtxels -i cohsex.in
uv run python -m gw.kin_ion_io -i cohsex.in

# GW calculation
uv run python -m gw.gw_jax -i cohsex.in


# Tests (~15s, JAX compilation overhead)
uv run python -m pytest -q
```

## GW code style guide

### Array layout: flat k-indices everywhere

All G, V, W, χ, Σ tensors use a **flat k/q index** as their leading dimension:
```
V(nq, μ, μ)      W(nq, μ, μ)      χ(nq, μ, μ)
G(nk, s, μ, s, μ)                  Σ(nk, s, μ, s, μ)
```

The 3D k-grid `(nkx, nky, nkz)` appears **only inside FFT helpers** that
reshape, FFT, and flatten back. This is preparation for non-uniform k-grids
and NUFFTs — restricting the 3D layout to FFT functions means swapping FFT
implementations touches one place.

Pass `kgrid = (nkx, nky, nkz)` as a single tuple, never three separate ints.

File I/O (restart files) may use 3D k-layout; flatten immediately after reading.

### Wavefunctions: the `Wavefunctions` bundle

The `Wavefunctions` dataclass stores four sharded copies of ψ_nk(r_μ), one per
`{device axis} × {memory layout}` combination:
```
psi_xn (nk, s, μ_X, n)  — bands fast, μ on X   → G/χ construction LHS (conj)
psi_xr (nk, n, s, μ_X)  — centroids fast, μ on X → Σ projection LHS (conj)
psi_yr (nk, n, s, μ_Y)  — centroids fast, μ on Y → G/χ construction RHS
psi_yn (nk, s, μ_Y, n)  — bands fast, μ on Y   → Σ projection RHS
```

`Wavefunctions` is registered as a JAX pytree and can be passed directly to
`@jax.jit` functions. Use `wfns.xn(s.sigma)`, `wfns.yr(s.full)` etc. to
slice — never pre-materialize 6 named views.

### Function signatures: minimal arguments

Top-level functions take container objects, not individual fields:
- `wfns: Wavefunctions` (not 4 separate psi arrays)
- `meta: Meta` (not nkx, nky, nkz, nk_tot, nspinor, bispinor)
- `opts` (not 25 unpacked option fields)

For innermost JIT kernels that need concrete arrays for tracing, individual
args are fine. But the driver and any function a human reads should have a
short signature that communicates intent.

### Sigma pipeline: one equation, one parameterized function

SX and COH differ only in which G is built, which interaction is used, and a
prefactor. Write one `_convolve(G_k, V_7d, prefactor)` and call it twice —
don't duplicate the FFT + multiply + FFT-back + project logic.

Write the equations once as a comment block where the pipeline is called.
The JIT function names are self-documenting; they don't need docstrings
restating the equation.

### JAX sharding rules

- Every `PartitionSpec` must be wrapped in `NamedSharding(mesh_xy, P(...))`.
  Bare `P(...)` in `with_sharding_constraint` will error without a mesh context.
  The only exception is `shard_map` `in_specs`/`out_specs` which take bare `P`.
- Never hard-code mesh shapes. Refer to mesh axes by name (`'x'`, `'y'`).
- Let XLA handle communication. No `np.concatenate` or host-side gathers of
  large arrays.
- We are memory constrained — most large arrays must be tiled over the XY
  processor grid. Avoid operations that rematerialize large arrays on subsets
  of all processors.

### General conventions

- Match existing formatting. Do not reformat unrelated lines.
- Self-documenting names over comments. If the function is `compute_sigma_sx`,
  it does not need a docstring saying "compute screened exchange self-energy."
- No backward-compatibility aliases. If something is renamed, update all callers.
  Let it break and fix the breakage — stale aliases accumulate and confuse.
- No fallback code paths (e.g. "if sharded_ifftn is not None ... else bare fft").
  There is one execution mode. Dead branches are dead weight.
- `ryd2ev` should be defined once and imported, not redefined in every file.

## Before committing

Run `uv run python -m pytest -q` after long running branches (5+ small commits); it takes 15 seconds.
Do not commit `__pycache__/`, `.venv/`, or `uv_cache/`, etc. directories.

## Environment

Use `uv` as the package manager. One `.venv/` (gitignored) per machine. No alternative
envs. Let uv use its global cache — do not create project-local uv cache directories.

On Perlmutter: use Shifter with the NVIDIA JAX container. See
`cluster_setup/README_CLUSTER.md`.
