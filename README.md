# LORRAX: The LOw-scaling Real-space Real-Axis eXcited state package in BerkeleyGW

The LORRAX code is a new GW-BSE package to be made available in BerkeleyGW in 2026. LORRAX is a JAX multi-GPU(/CPU) implementation of an $O(N^3)$-scaling formalism for the full-frequency (WIP), plasmon-pole (WIP), and static COHSEX self-energies $\Sigma^{GW}$ for one-shot $G_{0}W_{0}$ and quasiparticle-self consistent QSGW calculations. This provides significant speedups, with potential memory tradeoffs, relative to the canonical $O(N^4)$ scaling plane-wave formalism in BerkeleyGW. Iterative diagonalization of the Electron-Hole Bethe-Salpeter Equation Hamiltonian is also under development. LORRAX will shortly contain a novel $O(N^4)$ $QSG\hat{W}$ implementation with ladder vertex corrections in the screened interaction.

Besides these reduced scaling exponents, LORRAX gets its name from: 1.) our real-space framework for evaluating diagrammatic quantities, with the basis size reduced by the interpolative separable density fitting (ISDF) method, and 2.) real-frequency-axis integrations for the GW $\chi(\omega)$ and $\Sigma(\omega)$, avoiding ill-conditioned analytic continuation to recover real-axis quantities as in the majority of $O(N^3)$ scaling GW codes. The real-axis method has greater but comparable cost to imaginary-axis techniques, much improved by our novel real-axis minimax quadrature scheme.

LORRAX is developed primarily by Jack McArthur (myself) and supervised by Prof. Steven Louie at UC Berkeley, under whom the public BerkeleyGW package has been developed and maintained since 2011. Interoperability with the outputs of the BGW executables `epsilon.x`, `sigma.x`, `kernel.x`, and `absorption.x` is an active area of development. We use heavily and are actively expanding the roles of agentic coding platforms in the development of LORRAX, namely Anthropic's Claude Code and OpenAI's Codex. Both have been collaborators of great importance and are owed significant credit for the current state of LORRAX. We continue to explore applications of SOTA long-horizon schemes for scientific codebases, sandboxes for closed-loop development, and hierarchical tools like MCPs, subagents, and so forth.

The package requires as input the BerkeleyGW format wavefunction file `WFN.h5`. It is currently only compatible with full-spinor wavefunctions, but it can be used with wavefunction k-grids that are reduced by symmetry using BGW's `kgrid.x`. Symmetries are not used in the evaluation of the quasiparticle energies and will not reduce computational cost relative to unfolded k-grids.

The main drivers are located in `src/`:
- **ISDF initialization**: `centroid/kmeans_isdf.py` (k-means algorithm) + `centroid/kmeans_cli.py` (CLI entrypoint, `python -m centroid.kmeans_cli`)
- **Wavefunction loading**: `common/load_wfns.py` (ISDF basis fitting by least squares, memory bottlenecks)
- **GW quasiparticle energies**: `gw/gw_jax.py` (main driver), `gw/w_isdf.py` (screened interaction builder)

Available as console commands: `gw_jax`, `lorrax-gw`, `lorrax-centroids` (= `centroid.kmeans_cli`), `lorrax-bse`.

## Quick start

```bash
uv sync                                                       # editable install, no GPU/native build needed
uv run python -m pytest -q                                    # regression smoke test (CPU, ~1-2 min)
uv run python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in   # run a GW calculation
```

The third line runs a complete static-COHSEX calculation end-to-end on a fresh clone:
the bundled fixture sets `use_ffi_io = false` and ships its own wavefunction, so it needs
**no GPU and no native (FFI) build**. It is the fastest way to confirm LORRAX works on your
machine. Everything distributed (sharded HDF5, distributed `eigh`, SLATE) additionally
requires the native FFI stack — see [`docs/ENVIRONMENT_COMPREHENSIVE.md`](docs/ENVIRONMENT_COMPREHENSIVE.md).

On NERSC Perlmutter: `module load lorrax` then use `lxrun` / `lxpre`. See [`config/README.md`](config/README.md).

## Documentation

Detailed physics, code architecture, and environment setup are in `docs/`:

- **[`docs/PHYSICS_COMPREHENSIVE.md`](docs/PHYSICS_COMPREHENSIVE.md)** — ISDF theory, GW equations, ISDF basis (zeta) fitting, JAX sharding
- **[`docs/CODEBASE_COMPREHENSIVE.md`](docs/CODEBASE_COMPREHENSIVE.md)** — Module map, data flow, key classes, entry points
- **[`docs/ENVIRONMENT_COMPREHENSIVE.md`](docs/ENVIRONMENT_COMPREHENSIVE.md)** — Dependencies, installation, cluster usage, CUDA
- **[`docs/MEMORY_MODEL.md`](docs/MEMORY_MODEL.md)** — Per-stage memory usage formulas for the ISDF basis construction, heavy chunked operations
- **[`docs/MINIMAX_QUADRATURE.md`](docs/MINIMAX_QUADRATURE.md)** — Explanation of GW frequency integrals discretized via $\Sigma(omega) = \int dt e^{i \omega t} G(t)W(t)$, minimax quadrature 
- **[`docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md`](docs/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md)** — GN-PPM sigma pipeline, most recent dev push

Contributors and coding agents working in this repository should also read [`AGENTS.md`](AGENTS.md) for the module map and coding standards.
