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

## Coding standards

- Use NumPy-style docstrings. Document shapes, units, and shardings for array parameters.
- Match existing formatting. Do not reformat unrelated lines.
- Every function implementing a physics equation should reference what it is computing for human readability standards.

### JAX sharding rules

- Never hard-code mesh shapes. Refer to mesh axes by name (`'x'`, `'y'`).
- Use `NamedSharding` / `PartitionSpec` for all layouts. Let XLA handle communication, no `np.concatenate` or
  host-side gathers.
- We are very memory constrained and most large arrays can only be stored when tiled over the XY processor grid. Avoid at all costs operations that rematerialize large arrays on subsets of all processors.

## Before committing

Run `uv run python -m pytest -q` after long running branches (5+ small commits); it takes 15 seconds.
Do not commit `__pycache__/`, `.venv/`, or `uv_cache/`, etc. directories.

## Environment

Use `uv` as the package manager. One `.venv/` (gitignored) per machine. No alternative
envs. Let uv use its global cache — do not create project-local uv cache directories.

On Perlmutter: use Shifter with the NVIDIA JAX container. See
`config/README.md`.
