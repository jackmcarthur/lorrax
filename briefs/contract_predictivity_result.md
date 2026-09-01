# Contract predictivity result

Lane weight: **heavy**.  Five unique, attributable plans on the same frozen Na
measure give Pearson **r = -0.082** / Spearman **rho = 0.500** between
`sum(envelope*residual)` and delivered max error; for RMS error Pearson is
**r = 0.003**.  This sample does not support using the contract sum as a
predictor.

## Matched pairs (numbers first)

`P` is `sum(envelope*residual)/envelope_total`; all five have the same
`envelope_total = 3.790251517317762e13`, so the raw sums and normalized values
have identical correlations.

| plan | pairs | raw sum | P | max error (meV) | RMS (meV) |
|---|---:|---:|---:|---:|---:|
| scalar | 115 | 1.338478265e9 | 3.531370568e-5 | 0.195875 | 0.008834 |
| pointwise | 74 | 2.168590568e9 | 5.721495151e-5 | 0.235789 | 0.009575 |
| merged | 77 | 2.169418367e9 | 5.723679174e-5 | 0.760283 | 0.023515 |
| consolidated | 98 | 1.401039699e9 | 3.696429361e-5 | 0.645712 | 0.016876 |
| legacy | 137 | 2.622986921e9 | 6.920350560e-5 | 0.278609 | 0.010070 |

The strongest direct counterexample is 74 versus 77 pairs: contract spend
differs by **0.0382%**, while max error differs by **3.224x** and RMS by
**2.456x**.  Likewise, the 98-pair plan spends only 4.67% more than the scalar
plan but has 3.30x its max error.

## Window census

Mass shares are 2.50114%, 1.74578%, 33.74468%, 61.63011%, 0.05906%, and
0.31922% in the row order below.  Residuals are the selected-rule residuals
printed after the final plan summary, not the earlier `best_achieved_residual`
candidate diagnostic.

| window | scalar 115 | pointwise 74 | merged 77 | legacy 137 |
|---|---:|---:|---:|---:|
| cond:resonant | 2.14802e-5 | 8.97095e-4 | 8.16353e-4 | 8.03458e-4 |
| cond:state_tail | 4.33471e-6 | 4.33471e-6 | 1.94246e-5 | 2.67833e-4 |
| cond:pole_tail | 8.31831e-5 | 8.31831e-5 | 3.23359e-5 | 8.26827e-5 |
| val:bulk | 1.07204e-5 | 1.07204e-5 | 4.11083e-5 | 2.24551e-5 |
| val:resonant | 2.73166e-5 | 2.88256e-5 | 2.91597e-5 | 1.07688e-3 |
| val:pole_tail | 2.43819e-6 | 2.43819e-6 | 6.75348e-5 | 6.44066e-4 |

The consolidated plan keeps the three conduction rows above at 2.15136e-5,
4.33471e-6, and 8.31831e-5, and replaces the three valence rows (combined
mass 62.00840%) by one 1.33541e-5 residual.  No individual residual predicts
max error across all five: Pearson r is -0.020, -0.273, -0.723 for the three
common conduction rows; maximum unweighted residual gives r = -0.100.
`val:bulk` is the best observed candidate (r = 0.679, rho = 0.872), but this is
only five points and one is a reconstructed consolidated row, so it is not a
validated predictor.

## Worst elements and archive accounting

Worst indices `(omega,k,i,j)` are scalar `(20,0,8,8)`, pointwise
`(20,4,8,8)`, merged `(9,0,8,8)`, consolidated `(20,0,8,8)`, and legacy
`(19,4,9,9)`.  These are near-Fermi diagonal states; the merged failure is at
2.25 eV, while four others are at the 4.75/5.00 eV endpoint.  The archive does
not store per-window Sigma contributions, so it cannot attribute these
elements to high- or low-mass windows.  The 77-pair proxy is 44.3% val:bulk
and 35.7% cond:resonant, yet neither identifies its interior-frequency worst
element.

Five pairs above are the usable unique records after deduplicating repeat logs
and copied HDF5 outputs.  The 154-, 159-, and 167-pair wider-omega arms have
max errors 0.195847, 0.195849/0.195850, and 0.195916 meV, but their older logs
do not record envelope masses for their changed measures; the 102-pair direct
term plan records neither window residual rows nor masses.  They are reported
but excluded rather than assigning another plan's contract.  The 98- and
137-pair masses are reconstructed exactly from the later census of the same
frozen measure; the consolidated valence mass is the sum of its three source
windows.

No already-recorded cheap scalar predicts delivery.  The unmeasured plan-time
candidate is an elementwise maximum over the actual `(omega,k,i,j)` consumer
grid, retaining window attribution instead of mass averaging; the planner
already has the poles and evaluation energies, but this archive cannot score
that proposal.  No planner behavior was changed.

Evidence: archived logs and `sigma_mnk.h5` files under
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829`;
metric definitions were committed before scoring in
`briefs/contract_predictivity_prereg.md` at `5fd72b25`.  The prescribed CPU
gate completed **134 passed, 8 warnings in 102.92 s**; no GPU leg is owed
because this measurement lane changed documentation only (four-GPU rule CPU
exemption).
