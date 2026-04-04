# LORRAX: The LOw-scaling Real-space Real-Axis eXcited state package in BerkeleyGW

The LORRAX code is a new GW-BSE package to be made available in BerkeleyGW in 2026. LORRAX is a JAX multi-GPU(/CPU) implementation of an $O(N^3)$-scaling formalism for the full-frequency (WIP), plasmon-pole (WIP), and static COHSEX self-energies $\Sigma^{GW}$ for one-shot $G_{0}W_{0}$ and quasiparticle-self consistent QSGW calculations. This provides significant speedups, with potential memory tradeoffs, relative to the canonical $O(N^4)$ scaling plane-wave formalism in BerkeleyGW. Iterative diagonalization of the Electron-Hole Bethe-Salpeter Equation Hamiltonian is also under development. LORRAX will shortly contain a novel $O(N^4)$ $QSG\hat{W}$ implementation with ladder vertex corrections in the screened interaction. 

Besides these reduced scaling exponents, LORRAX gets its name from: 1.) our real-space framework for evaluating diagrammatic quantities, with the basis size reduced by the interpolative separable density fitting (ISDF) method, and 2.) real-frequency-axis integrations for the GW $\chi(\omega)$ and $\Sigma(\omega)$, avoiding ill-conditioned analytic continuation to recover real-axis quantities as in the majority of $O(N^3)$ scaling GW codes. The real-axis method has greater but comparable cost to imaginary-axis techniques, much improved by our novel real-axis minimax quadrature scheme.

LORRAX is developed primarily by Jack McArthur (myself) and supervised by Prof. Steven Louie at UC Berkeley, under whom the public BerkeleyGW package has been developed and maintained since 2011. Interoperability with the outputs of the BGW executables `epsilon.x`, `sigma.x`, `kernel.x`, and `absorption.x` is an active area of development. We use heavily and are actively expanding the roles of agentic coding platforms in the development of LORRAX, namely Anthropic's Claude Code and OpenAI's Codex. Both have been collaborators of great importance and are owed significant credit for the current state of LORRAX. We continue to explore applications of SOTA long-horizon schemes for scientific codebases, sandboxes for closed-loop development, and hierarchical tools like MCPs, subagents, and so forth.

#### BerkeleyGW-pyISDF
### GW, GW Perturbation Theory (GWPT) and Time-Dependent Adiabatic GW (TD-aGW) via ISDF

The Interpolative Separable Density Fitting (ISDF) procedure is a low-rank procedure which allows the large Khatri-Rao pair-product tensor $`M_{mn}(k,q,r)=\psi^*_{mk-q}(r)\psi_{nk}(r)`$ needed in MBPT calculations 
to be approximated as $`M_{mn}(k,q,r)\approx\sum_{\mu}\zeta_q(r_\mu)\psi^*_{mk-q}(r_\mu)\psi_{nk}(r_\mu)`$, where the "interpolation points" $`r_\mu`$ are a small number (~10 times the 
number of bands) of points chosen in the unit cell, and the "interpolation vectors" $`\zeta_q(r_\mu)`$ are a basis chosen by a least-squares procedure to minimize the error in reconstructing the full $`M_{mn}(k,q,r)`$.

It turns out that the form of this procedure (a basis expansion for the pair-products with separable coefficients, $`C^\mu_{nmkq} = {C^\mu_{mk-q}}^*C^\mu_{nk}`$) reduces the prefactor of the $`O(N^3)`$-scaling 
"space-time GW" formalism by around four orders of magnitude. Full-rank space-time GW is normally only faster than the canonical $`O(N^4)`$ plane-wave formalism for systems with 100+ atoms. 
This makes it significantly faster than the canonical approach for the quasiparticle self-energy matrix elements $`\langle mk|\Sigma|nk\rangle`$ even for small systems, where it offers a 2-3 order of magnitude speedup.

The package requires as input the BerkeleyGW format wavefunction files `WFN.h5` and `WFNq.h5`. It is currently only compatible with full-spinor wavefunctions, but it can be used with wavefunction k-grids that are reduced by symmetry using `kgrid.x`, in which case it will make use of a symmetry-reduced q-grid in self-energy matrix element calculations.

The main drivers are located in `src/`:
- **GW/COHSEX**: `gw_isdf/gw_jax.py` (main driver), `w_isdf.py` (screened interaction)
- **ISDF initialization**: `isdf_init/kmeans_isdf.py` (k-means clustering for interpolation points)
- **Wavefunction loading**: `common/load_wfns.py` (FFT transforms, CCT/ZCT fitting)

Available as console commands: `gw_jax`, `cohsex_isdf` (compatibility alias), `kmeans_isdf`, `bse_isdf`.

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

- **[`docs/MEMORY_MODEL.md`](docs/MEMORY_MODEL.md)** — Memory footprints, GPU/CPU allocation, and chunking constraints
- **[`docs/MINIMAX_QUADRATURE.md`](docs/MINIMAX_QUADRATURE.md)** — GL/HGL theory, minimax solver methods, and CTSP derivations

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
| **Debugging memory issues** | `docs/MEMORY_MODEL.md` |
| **Modifying GW self-energy** | `docs/PHYSICS_COMPREHENSIVE.md` §6, `src/gw_isdf/gw_jax.py` |
| **χ⁰ or W calculations** | `docs/MINIMAX_QUADRATURE.md`, `src/gw_isdf/w_isdf.py` |
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
If you're running on NERSC Perlmutter with Shifter, see [`cluster_setup/README_CLUSTER.md`](cluster_setup/README_CLUSTER.md) for batch and interactive workflows. The setup uses the NVIDIA JAX container—no venv or extra setup needed.


### Local uv usage
To create the environment and install dependencies:
```bash
uv venv
uv sync --no-install-project --locked
```
Run the code with:
```bash
uv run python -m gw_isdf.gw_jax -i cohsex.in
```
The provided `Dockerfile` mirrors these steps. If a GPU is available and cupy usage is desired, you can use `uv sync --locked --extra gpu`. *This is not available to LLMs and we will be phasing out cupy for JAX, which is more flexible.*


## Layout

- `src/` – Python package roots.
  - `gw_isdf/` – GW/COHSEX calculations (`gw_jax.py`, `gw_init.py`, `w_isdf.py`, `get_windows.py`).
- `src/isdf/` – Core ISDF/BSE/IO libraries.
  - `isdf_init/` – k-means point generation and charge density tools.
  - `bse_isdf/` – Bethe-Salpeter driver and notes.
  - `common/` – shared helpers: wavefunction readers (`load_wfns.py`), symmetry maps, FFT/NUFFT wrappers, tagged arrays.
  - `io/` – HDF5 I/O for wavefunctions, centroids, self-energy output.
  - `mixing/` – Self-consistent GW acceleration (Anderson mixing).
- `docs/` – comprehensive documentation (see Documentation Guide above).
  - Core guides: `PHYSICS_COMPREHENSIVE.md`, `CODEBASE_COMPREHENSIVE.md`, `ENVIRONMENT_COMPREHENSIVE.md`
  - Specialized: `MEMORY_MODEL.md`, `MINIMAX_QUADRATURE.md`, `GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md`
  - `archive/` – outdated documentation (superseded, see archive/README.md).
  - `references/` – reference papers (Kim 2020, self-consistent GW).
  - `advanced/` – multi-host JAX, specialized derivations.
  - `AGENT_TODO.md` – code improvement suggestions (not user priorities).
- `examples/` – runnable directories for test and production COHSEX setups.
- `tests/` – small regression tests.
- `misc/` – larger data files and archived scripts.
The main drivers are available as console commands (`gw_jax`, `cohsex_isdf`, `kmeans_isdf`, `bse_isdf`) once the package is installed.

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
uv run -- python -m pytest -q tests/test_gw_jax_regression.py -m regression
```

Notes:
- Regression fixture files live in `tests/regression/cohsex_debug/`.
- The test runs `python -m gw_isdf.gw_jax -i cohsex_test.in` and compares `eqp_test.dat` to `eqp_ref.dat`.
- By default the regression uses JAX auto-selection (`ISDF_COHSEX_TEST_PLATFORM=auto`), which will pick GPU on your test nodes.
- To force CPU or GPU explicitly, set `ISDF_COHSEX_TEST_PLATFORM=cpu` or `ISDF_COHSEX_TEST_PLATFORM=gpu`.

Run the sample COHSEX calculation:

```bash
uv run gw_jax -i tests/regression/cohsex_debug/cohsex_test.in
```
