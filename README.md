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

The current most important executable files are:
- `kmeans_isdf.py`
- `cohsex_main.py`

Which depend on I/O and support from
- `wfnreader.py`
- `symmetry_maps.py`
- `tagged_arrays.py`

And call routines important to the physics from
- `gamma_matrices.py`
- `w_isdf.py`
- `get_charge_density.py`
- `get_windows.py`

These supplementary scripts are now stored in the root and `test_scripts/` directories
to keep the repository root focused on the main COHSEX drivers.

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
  - `gw_isdf/` – GW/COHSEX calculations (`cohsex_isdf`, `w_isdf`, `get_windows`).
  - `bse_isdf/` – Bethe-Salpeter driver and notes.
  - `common/` – shared helpers: wavefunction readers, symmetry maps, tagged arrays and GPU utilities.
- `examples/` – runnable directories for the test and production COHSEX setups.
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

Run the sample COHSEX calculation:

```bash
uv run python cohsex_isdf.py
```
