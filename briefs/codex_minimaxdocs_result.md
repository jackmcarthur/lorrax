# Minimax documentation-accuracy result

| doc claim | code truth | action |
|---|---|---|
| `mpa_material_class` is a deck key | `gw_jax._infer_material_class` derives the class from WFN occupations | remove it as a live option; describe occupation-derived selection |
| `sigma_omega_layout` selects replicated or sharded Sigma storage | dynamic Sigma is unconditionally sharded | remove the key and describe the invariant |
| `mpa_sigma_crossing_target_error` separately controls crossing rules | `sigma_windows` uses `mpa_sigma_sector_target_error` for both sectors | remove the separate target |
| `mpa_sigma_omega_cluster_gap_ry` controls patch splitting | gaps wider than `1.5 * sigma_omega_step_ev` define clusters | replace the deleted key with the grid-step rule |
| `mpa_sigma_max_nodes` was deleted and the delivered ceiling is derived | `_derived_pair_ceiling` derives the delivered ceiling, but `_DEFAULTS`, `MPAConfig`, and `sigma_dispatch` still accept and consume the deck key | remove it from live docs; register the remaining code surface without changing behavior |
| the env registry is complete for delivered/minimax planning | code reads `LORRAX_SIGMA_PLAN`, `LORRAX_DELIVERED_PLAN_CACHE`, `LORRAX_DELIVERED_CENSUS_PROFILE`, `LORRAX_MINIMAX_ALLOW_RUNTIME_SOLVE`, `LORRAX_MINIMAX_CACHE_DIR`, and `LORRAX_DISABLE_MINIMAX_DISK_CACHE` | add the two missing delivered-planner rows; retain only migration prose for the removed tau-grid variable |
| the minimax service documents lookup/serve but not its new ROQ surface | `minimax.__init__` lazily exports the ROQ records, fit helpers, evidence functions, and `plan_measure_adapted_roq`; no service README exists | add a concise README and align module/API docstrings with the exports and product-window refusal contract |

The source branch is `docs/minimax-accuracy-2026-08-31`, based on
`fix/sc-selector-refusal-2026-08-31` at `a9999161`. The requested
`briefs/common_context.md` is absent from both tips; the supplied campaign
context was used as the contract.

## Page corrections

Pending final reconciliation and gate evidence.

## Verification

Pending `tests/test_layering.py` and the repository's documentation/link checks.
