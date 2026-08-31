# Known LORRAX issues

## 2026-08-31 planner adversarial audit (AUD-P)

### Pointwise selector can discard a feasible plan before validation

**Affected tree:** `origin/fix/pointwise-budget-2026-08-31` at `eed87297`,
`src/gw/mpa/delivered_windows.py:1968-1990`.

The dynamic program retains only the lowest cost at one surrogate frequency
for each total node count.  Choices discarded on that scalar coordinate can be
the only choices that meet the budgets at other frequencies.  Executing the
branch's `_select_rules` directly gave a false `_BudgetShortfall` for this
two-frequency case:

| choice | nodes | pointwise cost | budget | result |
|---|---:|---:|---:|---|
| selector-retained | 3 | `[0.1, 0.9]` | `[1.0, 0.5]` | refused at index 1 |
| selector-discarded | 3 | `[0.9, 0.1]` | `[1.0, 0.5]` | feasible |

Each frequency has one window.  Each window offers a 1-node rule at relative
cost 0.9 and a 2-node rule at cost 0.1; the pair ceiling is 3.  The surrogate
is frequency 0 because its budget is largest.  At the 3-node state, lines
1988-1990 keep the accurate-frequency-0 combination and discard the only
combination that meets both budgets.  Pointwise checking at lines 1994-2003
therefore cannot recover the feasible plan.  The DP state must retain a
pointwise Pareto frontier (or solve the small exact integer problem without
scalar pruning).

### The advertised three tightening rounds stop after round one

**Affected tree:** `fix/sc-selector-refusal-2026-08-31` at `5f750a77`,
`src/gw/mpa/delivered_windows.py:2293-2294` and `:2428-2431`.

The stage tuple contains three `"tightened"` entries, but any budget shortfall
from the first one reaches `if stage == "tightened": raise`.  Thus the second
and third entries are unreachable on precisely the compounding case they are
intended to serve.

A controlled call through `build_delivered_sigma_windows` forced selection
calls 1-3 to report a 2.0/1.0 shortfall and configured call 4 to accept.  The
planner refused after exactly 3 selection calls, made 2 fit calls, and exposed
only allowances `0.00016` and `0.000072`; it never entered a second tightened
round.  This is a clean refusal with fitted candidates, not a fit exception.

### Per-map hard band classification admits a protected-set two-cycle

**Affected tree:** `fix/sc-selector-refusal-2026-08-31` at `5f750a77`,
`src/gw/sc_iteration.py:2470-2482`, with the hard all-k predicate at
`src/gw/scissor.py:325-350`.

There is no hysteresis or carry for a band at the Sigma-grid edge.  A two-band
case using the production `build_omega_band_partition` and
`apply_band_partition` primitives alternates forever:

| map | protected mask | upper eigenvalue (eV) | retained off-diagonal (eV) |
|---:|---|---:|---:|
| 0, 2, 4 | `[1, 1]` | 1.061267292017 | 0.2 |
| 1, 3, 5 | `[1, 0]` | 0.990000000000 | 0.0 |

The grid is `[-1, +1] eV`.  When both bands are protected, the fixed
`[[0.50, 0.20], [0.20, 0.99]] eV` block diagonalizes to an upper state outside
the grid.  The next map drops that band, zeros the mixing, and applies a
0.99 eV scissor value, putting it back inside.  This is a constructed
adversarial map rather than a reproduced material deck, but it demonstrates
that rebuilding the structural mask from each iterate is not a continuous map
and can itself create a period-two failure.

