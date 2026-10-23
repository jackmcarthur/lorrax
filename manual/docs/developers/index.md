# Developer reference

The services and compute kernels LORRAX is built on, documented for developers who want to add new physics.
{: .page-summary }

LORRAX's GW and BSE drivers are thin layers over a small set of reusable pieces. The **services** are independently installable Python packages under `services/` in the repository, each with one public API (its top-level names) and no knowledge of GW: reading wavefunctions and ISDF tensors, crystal symmetry, the Coulomb interaction, time–frequency quadrature, and distributed dense linear algebra. The **kernels** are the compiled and JAX routines the drivers spend their time in: the k-axis convolutions that turn χ⁰ and Σ into products in time, and the GEMMs that build $G$ on the centroids and project Σ back onto bands. A new method that works with Green's functions, screened interactions or Bloch-periodic two-point operators on the ISDF basis — electron–phonon self-energies, forces and structural relaxation, new vertex corrections — can usually be assembled from these pieces. These pages describe what each one computes, its arguments with units, shapes and sharding, its errors, and where LORRAX calls it.
{: .lead }

## Where each piece sits {: #map }

| Stage of a GW/BSE workflow | Service or kernel |
|---|---|
| Read $\psi_{n\bk}$ from `WFN.h5`, sharded over the device mesh | [wfn_loader](wfn_loader.md) |
| Map the irreducible wedge to the full Brillouin zone; rotate wavefunctions and ISDF operators | [symmetry_maps](symmetry_maps.md) |
| Bare Coulomb $v(\bq+\bG)$ with 0D/2D/3D truncation and the $\bq\to0$ average | [vcoul](vcoul.md) |
| Read the interpolating vectors $\zeta_{\bq\mu}$ from the ISDF tensor file | [zeta_loader](zeta_loader.md) |
| Time and frequency nodes for χ⁰, $W$ and Σ | [minimax](minimax.md) |
| $G$ from $\psi$ at the centroids, χ⁰ and Σ as k-convolutions, the BSE kernel, band projections | [Kernels](kernels.md) |
| Distributed eigensolvers, polar factor, Cholesky, LU and GEMM on the device mesh | [distrib_la](distrib_la.md) |
| Environment dials, platform and library probing, placement, ordered reductions, memory model | [lxkit](lxkit.md) |

## Conventions {: #conventions }

- **Import the package, not its modules.** Each service's public API is its top-level names; importing `service.submodule` from outside fails the repository's layering lint.
- **Units.** Energies are in Rydberg and times in Ry⁻¹ unless a page says otherwise; lengths are in bohr. Input-file keys given in eV are converted at the driver.
- **Devices.** Arrays are JAX arrays on a two-axis device mesh, `('x', 'y')`, unless a page says host. Sharding is given as a `PartitionSpec` on that mesh.
- **Refusals.** A precondition that fails raises with a message that starts `GATE <name>:` and says what it got, what it wanted and how to fix it. Environment dials are read once, before tracing, through `lxkit.Gate`.

## Physics background {: #background }

The theory these pieces implement is in the Theory section: [ISDF and V(q)](../theory/isdf.md) for the centroid basis, [Time–frequency transforms](../theory/time-frequency.md) for the nodes, [Symmetry](../theory/symmetry.md) for the wedge and its unfolding, and [The q → 0 head](../theory/q0-head.md) for the Coulomb singularity.
