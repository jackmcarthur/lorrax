# ISDF COHSEX Utilities

This repository contains a small collection of Python scripts implementing the Interpolative Separable Density Fitting approach used by BerkeleyGW.

#### BerkeleyGW-pyISDF
### GW, GW Perturbation Theory (GWPT) and Time-Dependent Adiabatic GW (TD-aGW) via ISDF

The Interpolative Separable Density Fitting (ISDF) procedure is a low-rank procedure which allows the large Khatri-Rao pair-product tensor $`M_{mn}(k,q,r)=\psi^*_{mk-q}(r)\psi_{nk}(r)`$ needed in MBPT calculations 
to be approximated as $`M_{mn}(k,q,r)\approx\sum_{\mu}\zeta_q(r_\mu)\psi^*_{mk-q}(r_\mu)\psi_{nk}(r_\mu)`$, where the "interpolation points" $`r_\mu`$ are a small number (~10 times the 
number of bands) of points chosen in the unit cell, and the "interpolation vectors" $`\zeta_q(r_\mu)`$ are a basis chosen by a least-squares procedure to minimize the error in reconstructing the full $`M_{mn}(k,q,r)`$.

It turns out that the form of this procedure (a basis expansion for the pair-products with separable coefficients, $`C^\mu_{nmkq} = {C^\mu_{mk-q}}^*C^\mu_{nk}`$) reduces the prefactor of the $`O(N^3)`$-scaling 
"space-time GW" formalism by around four orders of magnitude. Full-rank space-time GW is normally only faster than the canonical $`O(N^4)`$ plane-wave formalism for systems with 100+ atoms. 
This makes it significantly faster than the canonical approach for the quasiparticle self-energy matrix elements $`\langle mk|\Sigma|nk\rangle`$ even for small systems, where it offers a 2-3 order of magnitude speedup.


This Python package implements the ISDF procedure for calculating quasiparticle self-energy matrix elements (GW bandstructures), self-energy contributions to electron-phonon coupling matrix elements (GWPT), and 
the time-dependent COHSEX method for nonequilibrium simulations. The code is heavily performance-optimized and is intended for MPI+GPU HPC systems; nearly all routines are written to take place on the GPU if available.

The package requires as input the BerkeleyGW format wavefunction files `WFN.h5` and `WFNq.h5`. It is currently only compatible with full-spinor wavefunctions, but it can be used with wavefunction k-grids that are reduced by symmetry using `kgrid.x`, in which case it will make use of a symmetry-reduced q-grid in self-energy matrix element calculations.

The main drivers are located in `src/isdf/`:
- **GW/COHSEX**: `gw_isdf/cohsex_jax.py` (main driver), `w_isdf.py` (screened interaction)
- **ISDF initialization**: `isdf_init/kmeans_isdf.py` (k-means clustering for interpolation points)
- **Wavefunction loading**: `common/load_wfns.py` (FFT transforms, CCT/ZCT fitting)

Available as console commands: `cohsex_isdf`, `kmeans_isdf`, `bse_isdf`.

## Documentation Guide

**For AI agents**: Read only the documentation relevant to your task to avoid context window overflow.

### Core Comprehensive Guides (START HERE)

- **[`docs/PHYSICS_COMPREHENSIVE.md`](docs/PHYSICS_COMPREHENSIVE.md)** — Physics & theory
  - §1-3: ISDF theory, k-means clustering, Galerkin CCT/ZCT derivation
  - §4: Complete 5-stage zeta fitting procedure
  - §5: Chunking strategy and memory-efficient implementation
  - §6: GW self-energy equations (COHSEX, head correction, self-consistency)
  - §7: JAX sharding reference tables

- **[`docs/CODEBASE_COMPREHENSIVE.md`](docs/CODEBASE_COMPREHENSIVE.md)** — Code structure & architecture
  - Module organization, key classes (Meta, WFNReader, etc.)
  - Data flow pipeline, file formats (HDF5)
  - Entry points, function call hierarchy
  - JAX sharding patterns, code location quick reference

- **[`docs/ENVIRONMENT_COMPREHENSIVE.md`](docs/ENVIRONMENT_COMPREHENSIVE.md)** — Setup & deployment
  - Dependencies, installation (uv, conda-forge, Docker)
  - JAX configuration, environment variables
  - Cluster usage (NERSC Perlmutter, SLURM)
  - Multi-host/multi-GPU setup, CUDA compatibility

### Specialized Topics

- **[`docs/MEMORY_MODEL.md`](docs/MEMORY_MODEL.md)** — Memory footprints and GPU/CPU allocation strategy
- **[`docs/CHUNK_BUDGETS.md`](docs/CHUNK_BUDGETS.md)** — Band/R/q chunking constraints and buffer sizes
- **[`docs/chi_omega_quadrature.md`](docs/chi_omega_quadrature.md)** — CTSP (Complex-Time Shredded Propagator) quadrature method for χ⁰(iω)

### Experimental/Reference

- **[`docs/advanced/`](docs/advanced/)** — Multi-host JAX, specialized derivations (GPP model)
- **[`docs/references/`](docs/references/)** — Reference papers (Kim 2020 CTSP, self-consistent GW)
- **[`docs/archive/`](docs/archive/)** — **OUTDATED**: superseded docs (see archive/README.md for mapping)
- **[`docs/NUFFT_BACKEND_STATUS.md`](docs/NUFFT_BACKEND_STATUS.md)** — NUFFT backend (CPU-only, not active)
- **[`JAX_FINUFFT_USAGE.md`](JAX_FINUFFT_USAGE.md)** — jax-finufft notes and CUDA 13.0 issues

### Agent Suggestions

- **[`docs/AGENT_TODO.md`](docs/AGENT_TODO.md)** — **Code improvement suggestions (NOT user priorities)**

### Task-Specific File Guide

| Task | Read These Files |
|------|------------------|
| **Understanding ISDF theory** | `docs/PHYSICS_COMPREHENSIVE.md` §1-3 |
| **Setting up environment** | `docs/ENVIRONMENT_COMPREHENSIVE.md` |
| **Understanding code structure** | `docs/CODEBASE_COMPREHENSIVE.md` |
| **Working on zeta fitting** | `docs/PHYSICS_COMPREHENSIVE.md` §4-5, `docs/CODEBASE_COMPREHENSIVE.md` §6, `src/isdf/common/load_wfns.py` |
| **Debugging memory issues** | `docs/MEMORY_MODEL.md`, `docs/CHUNK_BUDGETS.md` |
| **Modifying GW self-energy** | `docs/PHYSICS_COMPREHENSIVE.md` §6, `src/isdf/gw_isdf/cohsex_jax.py` |
| **χ⁰ or W calculations** | `docs/chi_omega_quadrature.md`, `src/isdf/gw_isdf/w_isdf.py` |
| **K-means clustering** | `docs/PHYSICS_COMPREHENSIVE.md` §1.2, `src/isdf/isdf_init/kmeans_isdf.py` |
| **JAX sharding/chunking** | `docs/PHYSICS_COMPREHENSIVE.md` §5,§7, `src/isdf/common/load_wfns.py` |
| **Multi-GPU/cluster setup** | `docs/ENVIRONMENT_COMPREHENSIVE.md` §4-5, `docs/advanced/jax_multihost.md` |
| **Build issues (CUDA, finufft)** | `docs/ENVIRONMENT_COMPREHENSIVE.md` §6-7, `JAX_FINUFFT_USAGE.md` |

## Requirements
- numpy
- scipy
- h5py
- matplotlib

JAX is used for CPU-only parallelism in `kmeans_isdf.py`. To create multiple
host devices for testing, set the environment variable:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
```

The JAX setup uses only the standard CPU backend. 

## Setup
To install the Python dependencies use uv/Docker setup.

### Perlmutter (Shifter) quickstart
If you're running on NERSC Perlmutter with Shifter, see [`cluster_shifter/README_CLUSTER.md`](cluster_shifter/README_CLUSTER.md) for batch and interactive workflows. The setup uses the NVIDIA JAX container—no venv or extra setup needed.


### Local uv usage
To create the environment and install dependencies:
```bash
uv venv
uv sync --no-install-project --locked
```
Run the code with:
```bash
uv run python cohsex_isdf.py
```
The provided `Dockerfile` mirrors these steps. If a GPU is available and cupy usage is desired, you can use `uv sync --locked --extra gpu`. *This is not available to LLMs and we will be phasing out cupy for JAX, which is more flexible.*


## Layout

- `src/isdf/` – Python package with all driver routines and utilities.
  - `isdf_init/` – k-means point generation and charge density tools.
  - `gw_isdf/` – GW/COHSEX calculations (`cohsex_jax.py`, `w_isdf.py`, `get_windows.py`).
  - `bse_isdf/` – Bethe-Salpeter driver and notes.
  - `common/` – shared helpers: wavefunction readers (`load_wfns.py`), symmetry maps, FFT/NUFFT wrappers, tagged arrays.
  - `io/` – HDF5 I/O for wavefunctions, centroids, self-energy output.
  - `mixing/` – Self-consistent GW acceleration (Anderson mixing).
- `docs/` – comprehensive documentation (see Documentation Guide above).
  - Core guides: `PHYSICS_COMPREHENSIVE.md`, `CODEBASE_COMPREHENSIVE.md`, `ENVIRONMENT_COMPREHENSIVE.md`
  - Specialized: `MEMORY_MODEL.md`, `CHUNK_BUDGETS.md`, `chi_omega_quadrature.md`
  - `archive/` – outdated documentation (superseded, see archive/README.md).
  - `references/` – reference papers (Kim 2020, self-consistent GW).
  - `advanced/` – multi-host JAX, specialized derivations.
  - `AGENT_TODO.md` – code improvement suggestions (not user priorities).
- `examples/` – runnable directories for test and production COHSEX setups.
- `tests/` – small regression tests.
- `misc/` – larger data files and archived scripts.
- `cohsex_isdf.py` – wrapper that launches the test example when run directly.

The main drivers are also available as console commands (`cohsex_isdf`, `kmeans_isdf`, `bse_isdf`) once the package is installed.

## Quick start

Install dependencies and run the unit tests:

```bash
uv pip install -e .
uv run -- python -m pytest -q
```

Run the end-to-end COHSEX regression test:

```bash
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax/isdf_cohsex"
export JAX_ENABLE_COMPILATION_CACHE=1
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
uv run -- python -m pytest -q tests/test_cohsex_jax_regression.py -m regression
```

Notes:
- Regression fixture files live in `tests/regression/cohsex_debug/`.
- The test runs `python -m isdf.gw_isdf.cohsex_jax -i cohsex_test.in` and compares `eqp_test.dat` to `eqp_ref.dat`.
- By default the regression runs on CPU for portability (`ISDF_COHSEX_TEST_PLATFORM=cpu`).
- To request GPU execution, set `ISDF_COHSEX_TEST_PLATFORM=gpu` before running pytest.

Run the sample COHSEX calculation:

```bash
uv run python cohsex_isdf.py
```
