# Planner simplification — result

**6,845 -> 6,838 code lines** across the requested quadrature path: 7 code
lines (9 physical lines) deleted, leaving **4,838 lines above / 3.419x** the
2,000-line aspiration. Branch `refactor/planner-slim-2026-08-31`, commit
`e6aac07d`. This is a light, host/offline-only lane; no GPU kernel or sharding
path changed, so the four-GPU rule's CPU/unit-cell exemption applies.

| file | before | after | delta |
|---|---:|---:|---:|
| `src/gw/mpa/delivered_windows.py` | 2,306 | 2,303 | -3 |
| `src/gw/mpa/sigma_windows.py` | 748 | 748 | 0 |
| `services/minimax/src/minimax/roq_fit.py` | 689 | 686 | -3 |
| `services/minimax/src/minimax/reciprocal_fit.py` | 550 | 550 | 0 |
| `services/minimax/src/minimax/door.py` | 551 | 551 | 0 |
| `services/minimax/src/minimax/solver.py` | 827 | 826 | -1 |
| `services/minimax/src/minimax/_catalog.py` | 528 | 528 | 0 |
| `services/minimax/src/minimax/beta_selector.py` | 646 | 646 | 0 |
| **total** | **6,845** | **6,838** | **-7** |

Counts are Python-token-bearing source lines, excluding blank and comment-only
lines. Docstrings count as code, matching their runtime presence.

## What was removed and why it is safe

- `_fit_product_group` was a private one-group wrapper around the live
  `_fit_product_groups`. A whole-tree exact-name check over `src`, `services`,
  and `tests` found its definition as its **only occurrence**; it was absent
  from the service exports. The production planner calls `_fit_product_groups`
  directly.
- `N = len(tau)` and `adapted_only = stage == "adapted"` were reported as
  unused by Ruff F841. The former never entered the crossing solve; the latter
  never controlled the already-explicit `if stage == "adapted"` branch.
- `_merge_branch_specs` built an `order` dictionary and immediately deleted it;
  no lookup, branch, mutation, or return consumed it.

The required CPU gate after the code commit is **134 passed, 0 failed, 0
skipped in 92.13 s**. Ruff's F401/F811/F841 check and `git diff --check` are
clean.

## Sodium invariance and retained paths

Base/branch A/B over both crossing windows in the six-window frozen real-sodium
measure produced the same SHA-256 over every selected node, weight, rank,
metric and branch-evidence field:
`245be3ccabfe534ca49a2c68e5352216a3c999c6d06f1520d8e16f18fdf9be5d`.
Both selected **57 nodes**: cond resonant rank 48, residual
`5.5117191e-5`, kappa-p99 `9.87611`; val resonant rank 9, residual
`2.8825625e-5`, kappa-p99 `1.12744`. The deleted statements cannot change
window construction, selection, or execution. Thus the unchanged one-shot
receipt remains **4 windows, 98 (window,tau) pairs**, with its measured
`sigma_c_kij_ev` difference against `control_panes_24b` unchanged at
**0.6457120361 meV max / 0.0168763116 meV RMS**. Frozen receipt evidence:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Na/02_soc48b_qsgw_mpa/50_delivered_plan_20260829/codex_parallelplan_p4_20260831`.

No larger candidate was deleted. The gate exercises shipped lookup; the sodium
receipt proves consolidation is live (the accepted valence merge is 12 nodes);
omega subdivision is reachable before ROQ fitting; and the base still has the
scalar selector contract, so pointwise-budget cleanup is premature. A direct
multi-window ROQ audit also exposed a pre-existing out-of-scope `NameError`
in the refusal/consolidation arm (`_rank_ceiling(group)` where only `groups`
exists). The production delivered caller supplies singleton windows and did not
hit it; it was left unchanged under this lane's no-behavior-change rule.
