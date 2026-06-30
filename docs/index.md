# LORRAX

**LO**w-scaling **R**eal-space **R**eal-**A**xis e**X**cited state package — a JAX
multi-GPU/CPU implementation of an $O(N^3)$-scaling GW formalism, accelerated by
Interpolative Separable Density Fitting (ISDF) and real-frequency-axis integration.
The main GW driver is **GWJAX** (`gw.gw_jax`): it reads BerkeleyGW-format plane-wave
DFT wavefunctions (`WFN.h5`) and computes quasiparticle corrections via static COHSEX
or GN-PPM (Godby–Needs Generalized Plasmon Pole) frequency dependence.

- **Input**: DFT wavefunctions on a plane-wave grid, symmetry maps, and k-point sampling
- **Core idea**: replace dense charge-density products with a compact ISDF basis defined by centroids $r_\mu$
- **Outcome**: exchange and screened-exchange self-energy matrix elements and band-edge corrections

## Try it in 60 seconds

The fastest way to confirm LORRAX works on your machine — runs a complete static-COHSEX
calculation end-to-end on a fresh clone with **no GPU and no native (FFI) build**
(the bundled fixture sets `use_ffi_io = false` and ships its own wavefunction):

```bash
uv sync
uv run python -m gw.gw_jax -i tests/regression/cohsex_debug/cohsex_test.in
```

See the [Quickstart](quickstart.md) for the worked example, and
[Installation](installation/index.md) for the GPU / distributed / from-source tracks.

## High-level pipeline

1. Charge density from selected bands → choose ISDF points $r_\mu$ via k-means/CVT
2. Read wavefunctions $c_{nk}(G)$, FFT to real space $\psi_{nk}(r)$
3. For each $q$, construct $\zeta_{q,\mu}(r)$ by solving $C_q \zeta_q = Z_q$ by least-squares
4. Compute $V_{q,\mu,\nu}$ from $\zeta_{q,\mu}$ in G-space with the Coulomb kernel $v_q(G)$
5. Build the Green's function $G$ and (optionally) $\chi_0$ and screened interaction $W$
6. Form $\Sigma_{X/SX/COH}$ and project to the band representation $\Sigma_{kij}$

## Documentation map

- **[Installation](installation/index.md)** — support matrix and the three install tracks
- **[Quickstart](quickstart.md)** — the bundled fixture, run end-to-end
- **Theory** — [overview](theory/overview.md), [physics](theory/physics.md),
  [ISDF / zeta–V(q)](theory/isdf-zeta-vq.md),
  [minimax quadrature](theory/minimax-quadrature.md), [symmetry](theory/symmetry.md)
- **Architecture** — [codebase](architecture/codebase.md),
  [memory model](architecture/memory-model.md), [multi-host](architecture/multihost.md)

Developer notes, plans, progress logs, and the frozen archive live under `docs/dev/`
(outside this rendered site). Contributors and coding agents should also read
`AGENTS.md` in the repository root for the module map and coding standards.
