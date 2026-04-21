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
| `src/gw/gw_init.py` | Input parsing, chunking strategy, pipeline orchestration | Input file questions, chunk sizing |
| `src/gw/gw_config.py` | `LorraxConfig` runtime options dataclass | Flag plumbing, memory budget |
| `src/gw/gw_driver_helpers.py` | Screening/PPM setup helpers | Driver wiring |
| `src/gw/w_isdf.py` | χ₀ → W screening pipeline (CTSP, Dyson solve) | Screening / epsilon issues |
| `src/gw/ppm_sigma.py` | GN-PPM dynamic self-energy Σ^c(ω) | Frequency-dependent sigma issues |
| `src/gw/minimax_screening.py` | PPM extraction, minimax window helpers | PPM parameter issues |
| `src/gw/minimax_config.py` | Shared minimax / sigma quadrature config | Quadrature setup |
| `src/gw/head_correction.py` | q=0 head / wing correction | Head corrections |
| `src/gw/vcoul.py`, `compute_vcoul.py`, `compute_vcoul_0d.py` | Coulomb potential (3D / 2D slab / 0D box) | Truncation, V_q build |
| `src/gw/greens_function_kernel.py` | `build_G` occupied/all Green's function | G-matrix construction |
| `src/gw/projection_kernel.py` | Σ_μν → Σ_ij band projection | Band-basis projection |
| `src/gw/qsgw_utils.py` | QSGW fixed-point solver, Σ^xc I/O | Self-consistent GW |
| `src/gw/kin_ion_io.py`, `kin_ion_io_chunked.py` | Kinetic + ionic Hamiltonian I/O | `kin_ion.h5` issues |
| `src/common/isdf_fitting.py` | CCT/ZCT, pair-density kernels, zeta solve | Zeta fitting, pair density |
| `src/common/load_wfns.py` | Wavefunction loading + band-chunked FFT | WFN load path |
| `src/common/cholesky_2d.py` | 2D-blocked Cholesky for sharded CCT | Cholesky issues |
| `src/common/fft_helpers.py` | Flat-k FFT helpers | FFT plumbing |
| `src/common/gvec_fft_box.py` | Sphere ↔ FFT-box gather | V_q G-space build |
| `src/common/symmetry_maps.py` | `SymMaps`: IBZ→full BZ unfolding, spinor rotations | Symmetry / k-point unfolding |
| `src/common/minimax.py` | Minimax quadrature solvers | Quadrature node/weight issues |
| `src/common/meta.py` | `Meta` system-parameters dataclass | k/q-grid, band ranges |
| `src/common/phdf5_wfn_reader.py` | phdf5 (parallel HDF5) WFN reader | Async H5 reads |
| `src/common/gpu_utils.py` | Host-side GPU memory detection | Chunk auto-sizing |
| `src/file_io/wfnreader.py` | Canonical WFN.h5 reader (used by `gw_jax`) | Wavefunction loading |
| `src/file_io/slab_io.py` | `SlabIO`: phdf5 writer wrapper for zeta_q / V_qmunu | Big HDF5 writes |
| `src/file_io/sigma_output.py` | Σ output (eqp.dat, sigma.h5) | Output formats |
| `src/ffi/` | XLA FFI bridge: `cusolvermp`, `cusolvermg`, `phdf5`, `slate` | Native-library entry points |
| `src/solvers/` | Davidson, Lanczos, Chebyshev, pseudobands | Iterative eigensolvers |
| `src/centroid/kmeans_isdf.py` | ISDF centroid generation (k-means) | Centroid count / quality |
| `src/psp/` | Pseudopotentials, dipole / kin+ion generators | `dipole.h5` or `kin_ion.h5` issues |
| `src/bse/` | Bethe–Salpeter equation (experimental) | Optical spectra |
| `src/bandstructure/` | H-matrix interpolation (experimental) | Band-structure plots |

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

### Local dev (single machine, uv)

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

### Perlmutter (Shifter via Lmod module)

```bash
module load lorrax_X           # X = A | B | C
lxalloc                        # 1 node / 4 GPUs / 2 h
lxpre cohsex.in 640            # all 3 preprocessing steps (single-GPU)
lxrun python3 -u -m gw.gw_jax -i cohsex.in        # 4-GPU GW
LORRAX_NGPU=1 lxrun ...                           # single-GPU override
```

See [`config/README.md`](config/README.md) for the full cluster reference. Docs: [`docs/ENVIRONMENT_COMPREHENSIVE.md`](docs/ENVIRONMENT_COMPREHENSIVE.md).

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
