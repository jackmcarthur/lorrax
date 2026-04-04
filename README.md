# LORRAX

**LORRAX** (**Lo**w-scaling **R**eal-space **R**eal-**A**xis e**X**cited state package) is a
JAX-based package for computing quasiparticle band structures via the GW approximation
with Interpolative Separable Density Fitting (ISDF). The main GW driver is called
**GWJAX** (`gw_isdf.gw_jax`).

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

## Quick start

```bash
uv sync
uv run python -m pytest -q                                    # unit tests (~30s)
uv run python -m gw_isdf.gw_jax -i cohsex.in                  # run GW calculation
```

On Perlmutter: use Shifter with the NVIDIA JAX container. See `cluster_setup/README_CLUSTER.md`.

## Main entry points

| Command | Module | Purpose |
|---------|--------|---------|
| `gw_jax` | `gw_isdf.gw_jax` | GW/COHSEX self-energy |
| `kmeans_isdf` | `isdf_init.kmeans_isdf` | ISDF centroid generation |
| — | `psp.get_dipole_mtxels` | Dipole matrix elements for head corrections |
| — | `gw_isdf.kin_ion_io` | Kinetic + ionic Hamiltonian for QP energies |
| `bse_isdf` | `bse_isdf.bse_isdf` | Bethe-Salpeter equation (experimental) |

All run via `uv run python -m <module>`.

## Documentation

Detailed physics, code architecture, and environment setup are in `docs/`:

- **[`docs/PHYSICS_COMPREHENSIVE.md`](docs/PHYSICS_COMPREHENSIVE.md)** — ISDF theory, GW equations, COHSEX, CTSP, zeta fitting, JAX sharding
- **[`docs/CODEBASE_COMPREHENSIVE.md`](docs/CODEBASE_COMPREHENSIVE.md)** — Module map, data flow, key classes, entry points
- **[`docs/ENVIRONMENT_COMPREHENSIVE.md`](docs/ENVIRONMENT_COMPREHENSIVE.md)** — Dependencies, installation, cluster usage, CUDA
- **[`docs/MEMORY_MODEL.md`](docs/MEMORY_MODEL.md)** — Per-stage memory formulas, chunk sizing
- **[`docs/MINIMAX_QUADRATURE.md`](docs/MINIMAX_QUADRATURE.md)** — Minimax solvers, GL/HGL quadrature, CTSP derivations
- **[`docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md`](docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md)** — GN-PPM sigma pipeline

For AI agents: read `AGENTS.md` first, then only the docs relevant to your task.
