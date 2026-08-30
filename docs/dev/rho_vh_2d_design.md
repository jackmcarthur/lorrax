# G-space Hartree implementation

Status: current implementation. [Direct Hartree field](../theory/hartree.md)
owns the equations and conventions; this page owns APIs and schedules.

## API contracts

| operation | owner | contract |
|---|---|---|
| local sources | `psp.get_DFT_mtxels.density_components_from_psi_r` | `(nb,ns,nx,ny,nz)` orbitals and band weights → `rho`, `(rho,Jx,Jy,Jz)`, or two-spinor `rho_ab`; sole production contraction |
| spin fields | `psp.get_DFT_mtxels.spin_density_matrix_to_pauli_fields` | raw `rho_ab` → real `(rho,mx,my,mz)`; spatial/antiunitary action remains with `symmetry_maps` |
| scalar/vector fields | `psp.get_DFT_mtxels.build_hartree_potential`, `psp.dft_operators.transverse_potential_from_current` | WFN FFT grid, run `sys_dim`; zero-mode and transverse-sign conventions come from the shared Coulomb service |
| band matrix | `common.mtxel_sweep.sweep_matrix_elements` | `(nk,nb,nb)` complex Ry, `P(None,'x','y')`; scalar and vector actions share one sweep |
| one-shot entry | `gw.kin_ion_io.compute_hartree_matrix` | streams fixed `(k,band)` chunks and returns charge plus optional transverse matrices |
| self-consistent entry | `gw.sc_iteration.rebuild_hartree_dft_basis` | rebuilds from current orbitals and returns full-BZ matrices in the DFT basis |

`kin_ion.h5` remains kinetic plus ionic. No driver may substitute another
Hartree builder or cache the field across self-consistent iterations.

## Schedules and evidence scope

| use | schedule | trade-off |
|---|---|---|
| one-shot GW | process-distributed stream; one final grid reduction | bounded wavefunction memory |
| density-self-consistent GW | resident band-sharded orbital scan | reuses rotations; retains orbitals and performs mesh reductions |

The one-shot schedule is parallel under the required one-process-per-GPU
launch. A historical one-process/four-visible-GPU diagnostic used only the
first device during source construction; it is single-rank evidence, not P=4.
The later matrix sweep did use the visible mesh, which does not upgrade the
source-stage evidence.

Both schedules use the canonical WFN symmetry service and must preserve
two-band-axis sharding and Hermiticity. Scissors are applied only after the
final direct-field map.

Evidence: sandbox
`reports/gspace_hartree_single_path_2026-08-29/report.md`, claim 500.
