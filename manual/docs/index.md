# LORRAX

`LORRAX` The Low-scaling Real-space Real-Axis eXcited-state module in BerkeleyGW.
{: .page-summary }

<div class="chain-map" data-driver="gw" data-page="deps"></div>

LORRAX computes quasiparticle energies $E^{\mathrm{QP}}_{n\bk}$ in the GW approximation and neutral excitations from the Bethe–Salpeter equation, starting from a Quantum ESPRESSO calculation in BerkeleyGW's `WFN.h5` format. Every two-particle object is compressed by interpolative separable density fitting (ISDF) onto $N_\mu$ real-space points $\br_\mu$, and Σ is evaluated on the real frequency axis from real-time Green's functions. It runs through JAX with one process per device, on GPUs or CPUs.
{: .lead }

## The chain {: #chain }

<div class="stages" markdown>

1. **Mean field**
   Quantum ESPRESSO `pw.x` scf and nscf, exported with `pw2bgw.x` to `WFN.h5`, or the LORRAX [`nscf`](drivers/nscf/index.md) driver from the QE scf `.save`.
2. **GW inputs**
   Three utilities read `WFN.h5`: [`centroids`](utilities/centroids.md) selects the ISDF interpolation points $\br_\mu$, [`kin-ion`](utilities/kin-ion.md) writes $\langle m\bk|T + V^{\mathrm{ion}}|n\bk\rangle$, and [`dipole`](utilities/dipole.md) writes the velocity matrix elements for the $\bq \to 0$ head.
3. **GW**
   [`gw`](drivers/gw/index.md) builds $V_\bq(\mu,\nu)$, $W_\bq(\mu,\nu;z)$ and $\Sigma^{xc}$, and writes `eqp0.dat`, `eqp1.dat`, `qp_wfn_rotations.h5` and the restart bundle `tmp/isdf_tensors_<N_mu>.h5`.
4. **Excitons and bands**
   [`bse`](drivers/bse/index.md) solves the optical BSE from the restart bundle; [`exciton-bands`](drivers/exciton-bands/index.md) solves it along a $\bQ$ path. [`htransform`](utilities/htransform.md) interpolates QP bands, and [`downfold`](utilities/downfold.md) downfolds $V_\bq(\mu,\nu)$ and $W_\bq(\mu,\nu;0)$ onto a subspace of the bands for the BSE.

</div>

## Drivers {: #drivers }

| Driver | Module | Computes | Pages |
|---|---|---|---|
| [`nscf`](drivers/nscf/index.md) | `psp.run_nscf` | Kohn–Sham bands from a QE scf density, written as `WFN.h5` | [keywords](drivers/nscf/keywords.md) · [theory](drivers/nscf/theory.md) · [files](drivers/nscf/files.md) · [dependencies](drivers/nscf/dependencies.md) |
| [`gw`](drivers/gw/index.md) | `gw.gw_jax` | One-shot and self-consistent GW quasiparticle energies | [keywords](drivers/gw/keywords.md) · [theory](drivers/gw/theory.md) · [files](drivers/gw/files.md) · [dependencies](drivers/gw/dependencies.md) |
| [`bse`](drivers/bse/index.md) | `bse.bse_jax` | Optical exciton energies and eigenvectors; the absorption spectrum through `bse.absorption_haydock` | [keywords](drivers/bse/keywords.md) · [theory](drivers/bse/theory.md) · [files](drivers/bse/files.md) · [dependencies](drivers/bse/dependencies.md) |
| [`exciton-bands`](drivers/exciton-bands/index.md) | `bse.exciton_bands` | Exciton dispersion $\Omega_S(\bQ)$ along a $\bQ$ path | [keywords](drivers/exciton-bands/keywords.md) · [theory](drivers/exciton-bands/theory.md) · [files](drivers/exciton-bands/files.md) · [dependencies](drivers/exciton-bands/dependencies.md) |

## Utilities {: #utilities }

| Utility | Module | Writes |
|---|---|---|
| [`centroids`](utilities/centroids.md) | `centroid.kmeans_cli` | `centroids_frac_<N>.txt`, the ISDF interpolation points $\br_\mu$ |
| [`kin-ion`](utilities/kin-ion.md) | `gw.kin_ion_io` | `kin_ion.h5`, the kinetic-plus-ionic single-particle Hamiltonian $\langle m\bk\vert T+V^{\mathrm{ion}}\vert n\bk\rangle$ |
| [`dipole`](utilities/dipole.md) | `psp.get_dipole_mtxels` | `dipole.h5`, the velocity matrix elements |
| [`htransform`](utilities/htransform.md) | `bandstructure.htransform` | `bandstructure.dat`, DFT or QP bands along a `K_POINTS crystal_b` path |
| [`downfold`](utilities/downfold.md) | `gw.downfold_cli` | `isdf_tensors_<N>.h5`, the ISDF tensors $V_\bq(\mu,\nu)$ and $W_\bq(\mu,\nu;0)$ downfolded onto a subspace of the bands |

## Theory {: #theory }

| Page | Covers |
|---|---|
| [Physics overview](theory/physics.md) | The equations LORRAX evaluates, from $G$ to $W$ to Σ to the BSE |
| [Frequency treatments of Σ](theory/frequency-treatments.md) | `x_only`, `cohsex`, `gn_ppm`, `hl_ppm`, `mpa` |
| [Time–frequency transforms](theory/time-frequency.md) | The frequency convolutions of χ⁰ and Σ as sums over time nodes: sampling, minimax rules and the windows of Σ |
| [Tangential W model](theory/tangential-w.md) | $W^c$ as one set of poles shared by every matrix element |
| [The q → 0 head](theory/q0-head.md) | The singular head of $V$ and $W$ as a mini-BZ cell average with local fields, from the velocity matrix elements |
| [Short- and long-range W](theory/w-range-splitting.md) | $W$ carried to a finer k grid: the smooth part interpolated through $\bR$, the long-range plane-wave channels screened in G space and added back (`c1`, `srlr`) |
| [Metals](theory/metals.md) | Fermi–Dirac occupations in χ⁰, the head and Σ, and the intraband term |
| [Band extrapolation](theory/band-extrapolation.md) | $\Sigma^c$ extrapolated over the conduction bands to the complete-basis limit |
| [Quasiparticle self-consistency](theory/qsgw.md) | The QSGW map and its mixing |
| [ISDF and V(q)](theory/isdf.md) | Fitting $\zeta_{\bq\mu}(\br)$ and building $V_\bq(\mu,\nu)$ |
| [Centroid selection](theory/centroid-selection.md) | Weighted k-means and pivoted-Cholesky pruning |
| [Symmetry](theory/symmetry.md) | Space-group operations, time reversal, the k and q wedges and the transport of each object to the full grid |
| [Velocity operator](theory/velocity-operator.md) | $\hat v = \partial_\bk H$ with the nonlocal term, and the Berry connection from parallel-transport links |
| [Orbital magnetization](theory/orbital-magnetization.md) | The modern-theory orbital moment per band and per cell, from the velocity matrix (under development) |
| [Parallel algorithms](theory/parallel-algorithms.md) | The device mesh, the k-convolution, the G-space ζ fit, time nodes one at a time, the per-q solves, memory and I/O |

## Where to start {: #start }

- New to LORRAX: [Installation](using/installation.md), then [Quickstart](using/quickstart.md) for the full sequence from `pw.x` to an absorption spectrum.
- Preparing a material: [Inputs from DFT](using/inputs-from-dft.md) states what `WFN.h5` must contain and which QE settings it needs.
- Coming from BerkeleyGW: [For BerkeleyGW users](using/berkeleygw-users.md) maps `epsilon.inp`, `sigma.inp`, `kernel.inp` and `absorption.inp` keywords to LORRAX keys.
- Looking up a deck key: the Input keywords page of each driver. Environment variables are on [Environment variables](using/environment.md).
