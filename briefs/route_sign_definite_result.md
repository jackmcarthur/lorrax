# route_sign_definite result — heavy lane

## Achieved numbers

The frozen-Na sign-definite fit sum fell from **48.981 s to 2.174 s
(22.53x)**.  The six-window cold parallel fitting wall was **2.261 s**
(imports and NPZ load excluded); the sequential six-window fitting-stage wall
was **3.245 s**.  The earlier 30.513 s number includes planner selection and is
therefore not presented as a like-for-like speedup against the 2.261 s fit-only
wall.

| window | physical R | before wall (s) | after wall (s) | rank before→after | residual before→after | kappa p99 before→after |
|---|---:|---:|---:|---:|---:|---:|
| `cond:state_tail` | 25.740 | 3.281 | 0.255 | 9→6 | 1.942e-5→1.170e-4 | 1.1073→1.0766 |
| `cond:pole_tail` | 136.375 | 9.908 | 0.977 | 14→11 | 3.234e-5→6.470e-5 | 1.0586→1.0816 |
| `val:bulk` | 335.747 | 27.470 | 0.740 | 11→12 | 2.216e-5→5.952e-6 | 1.0146→1.0156 |
| `val:pole_tail` | 15.058 | 8.323 | 0.202 | 9→5 | 6.753e-5→1.091e-4 | 1.0458→1.0870 |

All four after-rules beat the frozen incumbent residuals associated with the
0.196 meV deck and pass their unchanged refined residual/noise gates.  However,
the brief's stricter request for the supplied ROQ before-metrics to remain the
same or improve is **not achieved**: three residuals, three p99 kappas, and the
`val:bulk` rank regress.  This is therefore a measured performance candidate,
not a claim that the exact requested rule-parity condition is closed.

## Change and diagnosis

Production sign-definite windows now try the deterministic Hackbusch-seeded
noncrossing ladder first, with no table lookup.  If the real-axis family misses
the unchanged refined consumer gates, the existing measure-adapted fitter is
the accuracy-preserving fallback; a degenerate real support stops at roundoff
instead of grinding higher ranks.  Crossing routing is unchanged.

`val:bulk` is no longer pathological: **27.470→0.740 s (37.11x)** at
R=335.747.  Cost does not simply track R after seeding: `cond:pole_tail` at
R=136.375 is slower (0.977 s) because it needs ranks 9→10→11, while
`val:bulk` reaches its accepted rank 12 from the analytic seed in one ladder
fit.  The former pathology was the failed-prefix ladder, not large R alone.

## Evidence

Frozen input:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/DEV/80_minimax_delivered_error_toy_20260828/results/analysis/evidence/causal_hankel/na_reconstructed_problems_v1.npz`.
The checked-in replay is `tools/benchmark_delivered_lookup.py`.  Required CPU
gate: **134 passed** (final rerun recorded in the lane handoff).  GPU evidence
is not owed under the Four-GPU Rule's CPU-only solver exemption.

Branch: `fix/route-sign-definite-2026-09-01`.
