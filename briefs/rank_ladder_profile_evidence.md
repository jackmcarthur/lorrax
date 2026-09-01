# Rank-ladder search-frequency evidence

Frozen measure:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/evidence/causal_hankel/na_reconstructed_problems_v1.npz`, arrays `p0_internal`, `p0_mass`, `p0_validation_internal`, and `p0_validation_mass`. Each arm used 21 evenly spaced frequencies over the stated symmetric eV interval, target `1e-4`, and `eta=0.01837465441237269 Ry`.

| support | A/gamma | arm | planner s | rank | residual | kappa p99 | kappa max |
|---|---:|---|---:|---:|---:|---:|---:|
| +/-5 eV | 65.516777 | all 21 search frequencies | 8.940584 | 70 | 5.1215783836e-5 | 16.8732967 | 293.6973141 |
| +/-5 eV | 65.516777 | 11 search frequencies | 8.330754 | 70 | 5.1215783836e-5 | 16.8732967 | 293.6973141 |
| +/-10 eV | 85.370787 | all 21 search frequencies | 10.106743 | 91 | 5.7165205822e-5 | 22.5255117 | 109.5637344 |
| +/-10 eV | 85.370787 | 11 search frequencies | 9.566066 | 91 | 5.7165205822e-5 | 22.5255117 | 109.5637344 |
| +/-15 eV | 105.224798 | all 21 search frequencies | 15.392582 | 134 | 4.5248336897e-5 | 29.0347329 | 156.7273116 |
| +/-15 eV | 105.224798 | 11 search frequencies | 13.325008 | 134 | 4.5248336897e-5 | 29.0347329 | 156.7273116 |

An exploratory 121-frequency +/-15 eV grid measured 67.223252 s before and 57.523147 s after (-14.43%); both returned rank 134, residual `5.3321138888e-5`, kappa p99 `28.2570978`, and kappa max `189.9202858`.

The rejected patience-cap experiment measured 67.223252 s at 16/3 iterations/stall and 67.222482 s at 8/2. All provisional errors were bit-identical: the existing patience rule had already stopped before either cap, so that change was discarded.

Verification command was the sprint's five-file CPU gate with `services/minimax/tests/test_roq_fit.py::test_rank_search_thins_only_provisional_fit_frequencies` appended. It collected 135 tests and reported `135 passed, 8 warnings in 112.76s`.
