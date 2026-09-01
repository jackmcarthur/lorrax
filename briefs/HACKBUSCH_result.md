# HACKBUSCH result — heavy lane

## Achieved numbers

On the four sign-definite windows in the frozen real-Na one-shot measure, the
analytic rank seed reduced total on-demand fit wall from **4.702 s to 1.356 s
(3.47x)**, or **1.176 s to 0.339 s per window**.  Each A/B used the same
single-core CPU environment and the same deterministic VarPro/Lawson solve.

| window | R | before (s) | after (s) | rank | refined residual | kappa p99 |
|---|---:|---:|---:|---:|---:|---:|
| cond:state_tail | 25.740 | 0.544 | 0.264 | 5 | 5.77943e-4 | 1.07928 |
| cond:pole_tail | 136.375 | 1.647 | 0.398 | 9 | 1.69921e-4 | 1.08180 |
| val:bulk | 335.747 | 2.324 | 0.510 | 11 | 1.83145e-5 | 1.01555 |
| val:pole_tail | 15.058 | 0.187 | 0.183 | 3 | 3.80088e-3 | 1.09182 |

The stricter frozen six-window benchmark, with on-demand noncrossing fitting
bypassed in for all four sign-definite windows and its normal two crossing
fits retained, took **13.000 s before and 4.380 s after (2.97x)** for the whole
fitting stage.  Its accepted sign-definite ranks were **6/11/12/5**; the
before/after node and weight arrays were bit-identical at every window, as
were residual and kappa.  Their SHA-256 prefixes were respectively
`3b69957a709e1912`, `8818e63e850c11b1`, `50ec6e6de5bc2c7b`, and
`7871da09837e09d8`.

This is the whole frozen-deck **fitting-stage** wall, not a claimed full-driver
planner wall: the named artifact contains six windows (four sign-definite),
and the base branch still has catalog wiring.  The on-demand path was bypassed
only in the benchmark as the lane brief permits.  A production-wide planner
wall remains to be measured after the separate on-demand-only wiring lands.

## Change

The per-rank solver already used the deterministic Hackbusch analytic node
distribution
`pi^2 (k-1/2) / (2 log(4R))`; the cost defect was the outer loop solving every
rank from 2 upward.  `noncrossing_grids` now seeds that ladder with the
existing deterministic `(R, eps)` error law, then walks up or down to the
first measured passing rank.  Node refinement, achieved-error acceptance,
noise/factor gates at the consumer, and the returned rule are unchanged.  No
table or new dial was added.

## Evidence and gates

Frozen input:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/evidence/causal_hankel/na_reconstructed_problems_v1.npz`.
The shipped benchmark harness was `tools/benchmark_delivered_lookup.py`, with
its sign-definite generator bypassed to the runtime solver for the A/B.

- Required CPU gate: **134 passed** in 88.17 s.
- New focused rank-seed gate: **2 passed** in 0.73 s.
- GPU gate: not owed; this changes and measures a CPU-only minimax solver, so
  the Four-GPU Rule's unit/CPU exemption applies.

Branch: `perf/hackbusch-seed-2026-08-31`.
