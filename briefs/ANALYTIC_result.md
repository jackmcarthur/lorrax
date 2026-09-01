# ANALYTIC result — heavy lane

## Achieved numbers

On the four frozen-Na sign-definite windows, removing all four Lawson passes
reduced the fixed accepted-rank solve total from **0.820 s to 0.671 s (1.22x)**.
Times are medians of the last three of four single-core CPU runs.  Every rule
still passed the delivered residual and runtime-noise gates.

| window | R | rank | before (s) | after (s) | residual before -> after | kappa p99 before -> after |
|---|---:|---:|---:|---:|---:|---:|
| cond:state_tail | 25.740 | 6 | 0.090 | 0.060 | 1.170e-4 -> 1.269e-4 | 1.07658 -> 1.07648 |
| cond:pole_tail | 136.375 | 11 | 0.257 | 0.228 | 6.470e-5 -> 6.560e-5 | 1.08160 -> 1.08160 |
| val:bulk | 335.747 | 12 | 0.377 | 0.343 | 5.952e-6 -> 6.190e-6 | 1.01559 -> 1.01559 |
| val:pole_tail | 15.058 | 5 | 0.096 | 0.040 | 1.091e-4 -> 1.112e-4 | 1.08704 -> 1.08699 |

The fully constructed midpoint-Laplace rule, evaluated on demand from
`h = pi^2/(2 log(4R))`, `t_k = h(k-1/2)`, and `w_k = h`, took **10--17 us**
but failed all four windows: delivered residuals were **0.977, 0.973, 1.000,
and 0.310**.  Holding those nodes fixed and solving only the linear weights
also failed all four.  Thus this classical construction is a useful seed, not
a serving rule at ranks 6/11/12/5; nonlinear node relocation remains necessary
and milliseconds per window were not achieved.

## Change and evidence

The branch includes the sibling's analytic `(R, eps)` rank seed, then changes
the on-demand noncrossing default to stop after the initial VarPro solve.
Lawson refinement remains explicitly available to offline callers.  No table,
cache, deck dial, or executor path was added.  Raw measurements are in
`briefs/ANALYTIC_measurements.csv`; frozen input was
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/evidence/causal_hankel/na_reconstructed_problems_v1.npz`.

Required CPU gate: **134 passed in 84.18 s**.  Focused solver gate: **3 passed
in 1.59 s**.  GPU verification is not owed under the Four-GPU Rule's CPU/unit
exemption: this changes only the SciPy CPU fitting path.  Si's ten windows were
not reached inside the sprint.  Branch: `feat/analytic-sign-definite-2026-09-01`.
