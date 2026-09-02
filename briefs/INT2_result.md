# INT2 — planner consolidation sprint result

**Achieved:** 7/8 requested branches merged; final required CPU gate **137 passed, 0 failed, 2 warnings in 80.32 s**. Branch `integ/planner-consolidated-2026-08-31` is pushed. No GPU/deck claim was attempted inside the funded line.

| Tree after merge | Gate result | Wall |
|---|---:|---:|
| `perf/plan-wall-2026-08-31` | 134 passed | 101.28 s |
| `perf/five-second-planner-2026-08-31` | 134 passed | 80.71 s |
| `feat/usable-rank-cap-v2-2026-08-31` | 134 passed | 80.16 s |
| `fix/pointwise-budget-2026-08-31` | 135 passed | 91.92 s |
| `fix/pointwise-dp-2026-08-31` | 136 passed | 89.80 s |
| `fix/pointwise-accuracy-2026-08-31` | 136 passed | 79.08 s |
| `fix/sc-anchor-cycle-2026-08-31`, first integration | 136 passed, 1 failed | 79.75 s |
| SC test-double integration fix | **137 passed** | **80.32 s** |

The first ROQ conflict was resolved by composing warm-start/log-error interpolation with deferred kappa scoring. The numerical-rank-cap conflict retains the usable-rank ceiling and honest miss path; successful brackets retain interpolation, deferred scoring, and warm-started final fits. Miss probes cannot be ranked by kappa because deferred scoring intentionally leaves it `NaN`, so their stable-rank choice uses achieved residual then rank.

The pointwise conflict keeps the exact MILP selection and removes scalar pruning while retaining the pointwise budget accounting. After the SC-anchor merge, its new tightening regression failed because its `_select_rules` test double still implemented the old four-argument API; accepting the required `pointwise_budget` keyword made the complete gate pass at 137/137. This is test composition only, not a production semantic change.

The merged tree is on the **pointwise delivered-error contract**. The inherited `fix/pointwise-accuracy` evidence records 74 `(window,tau)` pairs and 0.2359 meV versus 115 pairs and 0.1959 meV for the scalar plan; those deck numbers were not remeasured in this sprint.

Not completed before the 15-minute stop: `origin/fix/silent-thresholds-2026-08-31` is not an ancestor; no one-shot sodium deck, three-support planning-wall run, or signed SC deck was launched. Thus no combined-tree planning or sigma-c accuracy claim is made.
