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


This Python package implements the ISDF procedure for calculating quasiparticle self-energy matrix elements (GW bandstructures), self-energy contributions to electron-phonon coupling matrix elements (GWPT), and
the time-dependent COHSEX method for nonequilibrium simulations. The code is heavily performance-optimized and is intended for MPI+GPU HPC systems; nearly all routines are written to take place on the GPU if available.

The package requires as input the BerkeleyGW format wavefunction files `WFN.h5` and `WFNq.h5`. It is currently only compatible with full-spinor wavefunctions, but it can be used with wavefunction k-grids that are reduced by symmetry using `kgrid.x`, in which case it will make use of a symmetry-reduced q-grid in self-energy matrix element calculations.

The main drivers are located in `src/`:
- **GW/COHSEX**: `gw_isdf/gw_jax.py` (main driver), `w_isdf.py` (screened interaction)
- **ISDF initialization**: `isdf_init/kmeans_isdf.py` (k-means clustering for interpolation points)
- **Wavefunction loading**: `common/load_wfns.py` (FFT transforms, CCT/ZCT fitting)

Available as console commands: `gw_jax`, `lorrax-gw`, `kmeans_isdf`, `bse_isdf`.

## Quick start

```bash
uv sync
uv run python -m pytest -q                                    # unit tests (~30s)
uv run python -m gw_isdf.gw_jax -i cohsex.in                  # run GW calculation
```

On Perlmutter: use Shifter with the NVIDIA JAX container. See `cluster_setup/README_CLUSTER.md`.

## Documentation

Detailed physics, code architecture, and environment setup are in `docs/`:

- **[`docs/PHYSICS_COMPREHENSIVE.md`](docs/PHYSICS_COMPREHENSIVE.md)** — ISDF theory, GW equations, COHSEX, CTSP, zeta fitting, JAX sharding
- **[`docs/CODEBASE_COMPREHENSIVE.md`](docs/CODEBASE_COMPREHENSIVE.md)** — Module map, data flow, key classes, entry points
- **[`docs/ENVIRONMENT_COMPREHENSIVE.md`](docs/ENVIRONMENT_COMPREHENSIVE.md)** — Dependencies, installation, cluster usage, CUDA
- **[`docs/MEMORY_MODEL.md`](docs/MEMORY_MODEL.md)** — Per-stage memory formulas, chunk sizing
- **[`docs/MINIMAX_QUADRATURE.md`](docs/MINIMAX_QUADRATURE.md)** — Minimax solvers, GL/HGL quadrature, CTSP derivations
- **[`docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md`](docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md)** — GN-PPM sigma pipeline

For AI agents: read `AGENTS.md` first, then only the docs relevant to your task.
