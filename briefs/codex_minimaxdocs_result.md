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
| every name in `minimax.__all__` resolves | four deleted measure-window prototypes remain in `_MEASURE_WINDOW_NAMES`; the ROQ exports resolve | document the live ROQ surface and register the stale export metadata without changing behavior |
| the shipped catalog contains 31 rules (26 noncrossing, 5 crossing) | `catalog.json` contains 34 rules (28 noncrossing, 6 crossing); `catalog_complex_laplace.json` contains 55 | correct service docstrings and exact test pins, including the new exact-A=24 crossing selection |

The source branch is `docs/minimax-accuracy-2026-08-31`, based on
`fix/sc-selector-refusal-2026-08-31` at `a9999161`. The requested
`briefs/common_context.md` is absent from both tips; the supplied campaign
context was used as the contract.

## Page corrections

| page | stale claim | current statement |
|---|---|---|
| `docs/theory/minimax-quadrature.md` | MPA was outside the chapter; runtime solve and fitting ownership were ambiguous | separates panes from delivered ROQ, names lookup/serve behavior, the derived ceiling, and owners |
| `docs/theory/sigma-quadrature-problem.md` | no production ceiling/retry/refusal contract | records the support-derived ceiling, tightening retry, and terminal product-window refusal |
| `docs/theory/THEORY_mpa_implementation.md` | two Sigma targets, user node cap, material/layout dials, old cluster key | one target, occupation inference, always-sharded Sigma, grid-step clusters, and delivered ROQ section |
| `docs/theory/metallic-mpa-screening.md` | cluster gap came from a separate key | gaps larger than 1.5 Sigma grid steps split clusters |
| `docs/dev/delivered_plan.md` | fixed 200-pair ceiling and removed shared-tau mode | derived ceiling, two-pass tightening, consolidation reuse, receipt semantics, and no pair fallback |
| `docs/dev/crossing-rule-cost-law.md` | cluster-gap key owned patch splitting | `sigma_omega_step_ev` owns the threshold |
| `docs/dev/env_vars.md` | plan receipt and census profiler absent | all six `LORRAX_*` reads in the planner/service scope are registered; removed tau-grid env is migration-only |
| `docs/input_reference.md` | five retired controls appeared live or had stale semantics | live table contains target + eta policy; a separate migration table marks all five names removed |
| `docs/drivers.md`; `docs/dev/large_nmu_operation.md` | dynamic Sigma could be replicated | band sharding is invariant |
| `docs/architecture/codebase.md` | dead minimax function and old developer-note owner | live solver and theory owner are linked |
| `docs/architecture/fractional_chi0_response_face.md` | declared material class and low-memory metal refusal | WFN inference and the now-supported low-memory path |
| `docs/dev/notes/GN_PPM_MINIMAX_SIGMA_GUIDE_REVISED.md`; `NEW_WINDOW_MINIMAX_GUIDELINES.md` | historical sketches could be read as current API/geometry | banners point to the live owners while retaining the derivations |
| `services/minimax/README.md` and module docstrings | no service README; ROQ order/catalog counts stale | top-level lookup/serve/ROQ API, algorithm order, refusal contract, and current catalog counts |

Mechanical final scan (`docs`, `src`, `services`, and every `*.in`): no deck
uses any retired name. Active docs contain them only in the migration table;
the two extra `sigma_omega_layout` matches are a historical test filename.
Only `mpa_sigma_max_nodes` remains executable: four references in two source
files, registered as `SMALL_ISSUES.md` row 48. Three decks use the replacement
`sigma_omega_step_ev`.

## Verification

- Required CPU/doc gate: **108 passed** in 69.56 s (`test_layering`, env
  registry, `_DEFAULTS`/input-reference completeness, and focused doc rows).
- Minimax lookup/catalog/ROQ audit: **85 passed, 1 deselected** in 24.56 s.
  The deselected dead-name export test is registered as `SMALL_ISSUES.md` row
  49; changing `__all__` is outside this documentation-only lane.
- `mkdocs build`: **passed** in 3.20 s. `--strict` remains blocked by 23
  pre-existing repository-wide source-file links; no warning names a link
  added by this branch.
- `git diff --check` and Python compilation of edited module docstrings:
  **passed**. No GPU leg was run for this documentation-only/CPU gate.

Evidence directory: `/pscratch/sd/j/jackm/wt_minimax_docs_2026-08-31/briefs`.
