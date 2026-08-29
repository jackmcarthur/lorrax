# Delivered hybrid Sigma planning

The delivered pathway is an opt-in quadrature planner shared by the MPA and
GN-PPM correlation-Sigma routes. It fits the reciprocal rule to the spectral
measure the causal branch actually delivers, rather than certifying every
point of rectangular panes that may contain almost no residue. The incumbent
pane/window planner is unchanged and remains the default.

Enable it with:

```bash
export LORRAX_SIGMA_PLAN=delivered
```

`gw.sigma_plan.resolve_sigma_plan` is the single selector used by both
drivers. Unset, blank, or `panes` selects the incumbent path. The grammar is
case-insensitive after stripping whitespace; every other value refuses before
planning.

## Hybrid plan

For each causal branch the planner forms the internal sums
`s = E + Omega` (conduction) or `s = E - Omega` (valence). Delivered mass is
the state occupation weight times the state amplitude magnitude, pole-residue
magnitude, and cloud-in-cell weight. Pole damping includes the requested Sigma
regularization exactly once.

The service's tail-refined lattice uses delivered-mass quantiles with the
fixed per-axis bands 0--1%, 1--4%, 4--8%, 8--96%, 96--99%, and 99--100%.
The executable leading pole axis is then split by delivered-mass quantiles
(up to four conduction and three valence windows). This restriction is
intentional: the existing separable `G(t) W(t)` executor can select poles and
states independently, but cannot apply arbitrary state-pole tuple masks. Each
pole window's fit still sees its complete state x residue measure.

One conservative absolute true-error envelope is multiplied by the 0.8 safety
factor and apportioned among windows in proportion to delivered mass times
measured inverse-gap difficulty. Sign-definite windows use the minimax
service's incumbent `fit_damped_reciprocal`. Crossing windows use the
amplification-capped fixed-phase service fit over the union of the incumbent
MPA positive-time rule, the pane/HGL crossing family, and the service time
dictionary. The refined lattice validates the selected rule; a missed budget
or the default p99 amplification cap of 10 refuses.

Every result is an ordinary `SharedSigmaWindow` consumed by the existing
window executor and `DeviceOmegaAccumulator`. No second tau kernel, Green's
function, symmetry action, or pole-store format exists. The adapter converts
the fitted time orientation to the executor convention, folds
`exp(-eta*t_exec)` into each weight once, and retains the executor's global
`-1` prefactor.

## GN-PPM and time reversal

GN-PPM presents its single fitted pole per spatial matrix element as a
singleton leading pole axis. Degenerate one-pole measures therefore produce
one executable window per causal branch without asking a quantile routine to
invent empty bins. MPA retains its fitted leading pole axis and may produce
several windows.

Positive- and negative-frequency causal branches are always separate planner
inputs. The adapter does not reconstruct one W half from the other, preserving
the current time-reversal-broken W producer seam.

## Scope of the error statement

The planner report records node counts, fit and refined residuals,
amplification, apportioned absolute budgets, and planning wall time. Its true
error envelope applies to the scalar residue-weighted reciprocal measure used
for planning. It is conservative for that measure; it is not a proof that
spatial projection cannot cancel or amplify errors in every Sigma matrix
element. A full real-material run still needs the normal P=4 integration and
reference-comparison gates.

MPA planning reads the complete fitted pole axis once in its native spatial
sharding and reduces each addressable shard before cross-process gathering.
The executor then returns to configured pole batches. GN-PPM reuses its
already-resident single pole. In both modes the default `panes` arm retains its
previous reads, plan, and execution order.
