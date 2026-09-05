# SUITEGUT: plain four-rank pytest (2026-09-05)

Evidence: `/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/139_suitegut_rank_fixtures_2026-09-05`.

The coordinator's shared-HDF5 create failures came from different pytest
rank directories (`...b0/A` versus `...b2/A`), not an HDF5 transport defect.
`lx test` runs one task plus xdist; `lx run -n 4 ... pytest` runs four
independent pytest sessions whose children inherit a common MPI world.
Core now stages once, shares the absolute path, joins every child result,
and preserves rank-zero stdout for every rank's assertions. Shared stages
live outside basetemp; explicit basetemps are rank-private. The real P4
linalg child uses the existing ranks. The socket regression exercises four
OS processes, a delayed writer, peer failure, and timeout propagation.

Actual P4 driver coverage exposed production defects that P1 could not:

* B has one occupied k/band work item. Idle density ranks bypassed the
  FFT compilation while rank zero waited for world-wide compile agreement.
  Idle ranks now execute one uniform-width item with zero quadrature weight.
  This fixes the live Hartree density caller, not every sanctioned empty
  `local_share` consumer in the older cache-contract ledger.
* The initial SC identity rotation was placed before padding its odd band
  axes. It now uses the canonical padding receipt and the same logical
  return seam as the SC eigensolver. Logical device rotations retain their
  placement until the contraction jit, or until the Sigma receipt pads them
  for final-output placement. Compile agreement stays enabled.

The excited-state fixtures had P1 Krylov limits. A's nine physical states
occupy 36 padded transitions at P4. The exciton cell uses 36 steps; the
standalone eigenvalue cell uses Davidson to avoid scalar-Lanczos exact
breakdown/ghosts. Both retain the original 0.02 meV numerical tolerance.

No frozen reference bytes changed. Dynamic quadrature comparisons have
explicit approximation budgets, measured with tenfold tighter P4 fits:

| Quantity | 1e-3 versus frozen, max meV | 1e-4 versus 1e-3, max meV | Test tolerance, meV |
|---|---:|---:|---:|
| A GN Eqp0/1 | 0.323 | 0.155 | 0.5 |
| B one-shot MPA Eqp0/1 | 0.076 | 0.101 | 0.2 |
| B SC Eqp0/1 | 0.0019 | 0.096 | 0.02 (same 1e-3 fixed-rule pin) |

The tighter SC fit is a convergence probe, not the frozen fixed-rule pin.
SC residuals remain at 2e-5 eV; the printed gain uses 1e-5 absolute tolerance
instead of an exact six-decimal string. Static energies remain at 2e-5 eV
and zeta at 2e-10. GN's 1e-5 probe refused the existing runtime-noise gate;
that refusal is retained in the evidence rather than weakened.

CPU evidence: 23 core tests passed, five GPU-only cells skipped with four
forced host devices; 69 focused cache/eigh/rank-evaluation tests passed.
This is not a claim that the GPU driver cells ran on CPU, nor that the
whole public-entry-point Pareto matrix or two-minute P4 target is complete.
The final GPU command receipts and collected counts live in `STATUS.md`
and `final_*.txt` in the evidence directory above.
