# G-space Hartree implementation

Status: current implementation. [Direct Hartree field](../theory/hartree.md)
owns the equations and physical conventions.

## Shared physics

`psp.get_DFT_mtxels.density_components_from_psi_r` is the only local
charge/current contraction. Both production schedules call it. Charge and
signed current therefore use the same spinor convention and occupation
weights.

`psp.dft_operators` owns the scalar Poisson solve and transverse projector.
`common.mtxel_sweep` owns the band-sharded local-potential contraction. A
driver must not copy any of these operations.

## Two execution schedules

The local physics is shared. The data schedule is selected by use case.

| use | schedule | strength | current limit |
|---|---|---|---|
| one-shot GW | stream fixed `(k, band)` chunks in `gw.kin_ion_io` | bounded wavefunction memory; one final grid reduction | work is divided by JAX process, not by every addressable device |
| density-self-consistent GW | scan resident band-sharded orbitals in `gw.qsgw_density` | reuses the rotated orbitals and runs on the full device mesh | retains the orbitals and performs mesh reductions during the scan |

The one-shot schedule is fully parallel for the required launch with one GPU
per process. A historical one-process/multi-GPU diagnostic showed that its
density/current stage uses only the process's first device; that launch
geometry is prohibited on Perlmutter and is not P=4 evidence. The later
matrix-element sweep did use the visible two-dimensional mesh. Extending the
stream over global devices would combine its low memory use with full-device
parallelism; it should reuse the same local contraction and finish with one
grid reduction.

The older resident density helper in `psp.get_DFT_mtxels` is a compatibility
path used by centroid selection. It is not the production Hartree schedule.
The current-density weight in `centroid.current_density` is also different:
it is a nonnegative squared-current weight for choosing centroids, not the
signed current that sources the transverse direct field.

## Matrix elements and self-consistency

Both scalar and bispinor one-shot calculations use
`gw.kin_ion_io.compute_hartree_matrix`. Scalar and vector actions are packed
into one `common.mtxel_sweep.sweep_matrix_elements` call. The matrix remains
sharded as `P(None, 'x', 'y')` over the two band axes.

`gw.sc_iteration.rebuild_hartree_dft_basis` rebuilds the field from the
current orbitals and occupations. It reuses the rotated, band-sharded
orbitals for the density and matrix-element steps. There is no Hartree cache
between iterations or runs.

Scissor corrections are downstream of the completed Hamiltonian, after the
final direct-field basis rotation.

## Required invariants

- Build charge and current from the same orbitals and occupations.
- Use the canonical WFN symmetry service.
- Keep the matrix sharded over both band axes.
- Keep the periodic zero mode at zero.
- Use the Coulomb truncation selected by `sys_dim`.
- Preserve Hermiticity after symmetry completion and basis rotation.
- Treat `kin_ion.h5` as kinetic plus ionic only.

## Owners

| purpose | owner |
|---|---|
| local charge/current contraction | `psp.get_DFT_mtxels` |
| scalar and transverse fields | `psp.dft_operators` |
| one-shot schedule | `gw.kin_ion_io.compute_hartree_matrix` |
| self-consistent schedule | `gw.qsgw_density` and `gw.sc_iteration` |
| band-sharded matrix sweep | `common.mtxel_sweep` |
| output schema | `file_io.sigma_output` |

Numerical evidence is in sandbox report
`reports/gspace_hartree_single_path_2026-08-29/report.md`, claim 500.
