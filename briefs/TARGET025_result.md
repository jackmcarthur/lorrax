# TARGET025 result — heavy lane

## Numbers first

- **7/7 requested merge boundaries gated; 0 test failures at the final tree.**
- Gate sequence: planner baseline **137 passed / 87.88 s**; null-family
  **138 / 76.63 s**; on-demand-only **136 / 84.68 s**; Hackbusch seed
  **136 / 90.57 s**; sign-definite route **137 / 80.00 s**; crossing floor
  **137 / 77.61 s**; rank margin **138 / 78.77 s**.
- The route merge's first resolution gate was **136 passed, 1 failed** in
  **82.77 s**. The failure was an over-specific family assertion: the scalar
  result met its reported residual but correctly took the documented
  measure-adapted fallback. The corrected gate is the **137-pass** result above.
- Na one-shot measurements achieved in this sprint: **0**. Therefore this lane
  makes **no claim** for pair count, planning wall, `cond:resonant`
  nodes/residual/kappa, or max/RMS meV accuracy.

## Assembled tree

The branch was cut at `5f750a77` from
`fix/sc-selector-refusal-2026-08-31`, then merged the planner integration and
all six requested branches in order. Conflict resolution preserves all-on-demand
selection: Hackbusch-seeded sign-definite fits with a measure-adapted fallback,
measure-adapted crossing fits with the mass-weighted cost-law rank floor, the
usable-rank ceiling, pointwise allocation, and the +1 sign-definite rank margin.
No shipped-table or fixed-time crossing path was restored.

## Evidence and owed work

Evidence tree: `/pscratch/sd/j/jackm/wt_target025_2026-08-31` on
`integ/target-025-2026-09-01`. The exact prescribed CPU command was run after
each merge; the final collection is **138 passed, 2 warnings in 78.77 s**.

The sprint wall was exhausted by the required per-merge gates and semantic
conflict resolutions. A cold P=4 Na 0–5 eV run against `control_panes_24b`,
including the per-omega table and cold planning breakdown, remains owed. Si is
also owed because the Na accuracy gate was not reached. No GPU leg was launched,
so this result is not landing evidence for the driver.
