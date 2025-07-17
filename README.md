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

Core routines include:
- `get_charge_density.py`
- `kmeans_isdf.py`
- `get_interp_vectors.py`
- `cohsex_main.py`

These supplementary scripts are now stored in the `test_scripts/` directory
to keep the repository root focused on the main COHSEX drivers.

## Requirements
- numpy
- scipy
- h5py
- matplotlib

Optional GPU acceleration is supported with CuPy and FFTX. The code will
automatically fall back to NumPy if these libraries are not installed.

JAX is used for CPU-only parallelism in `kmeans_isdf.py`. To create multiple
host devices for testing, set the environment variable:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
```

This repository avoids optional packages such as FFTX and Spiral, so the JAX
setup uses only the standard CPU backend. Real-space positions and the density
array are flattened and evenly sharded across the available devices so each
processor works on a contiguous slice of the grid while keeping a copy of all
centroids.


## Setup
To install the Python dependencies, run:
```bash
./run/setup.sh
```

## Docker environment
A basic Dockerfile is provided to make it easy to run the code in a clean environment. To build the image and start a shell inside it, run:
```bash
docker build -t isdf_cohsex .
docker run --rm -it isdf_cohsex bash
```
This installs the Python requirements defined in `run/setup.sh` and drops you into a shell where you can run the examples, tests, or your own scripts.
