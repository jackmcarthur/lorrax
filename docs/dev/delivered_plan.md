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
the state occupation weight times the projected state-amplitude envelope,
pole-residue magnitude, and cloud-in-cell weight. The production callers
measure that state envelope from the actual left/right intermediate and
output wavefunction carriers as
`||psi^L_kn|| ||psi^R_kn|| max_i||psi^L_ki|| max_j||psi^R_kj||`; they never
use the planner's unit-amplitude fallback. Pole damping includes the requested
Sigma regularization exactly once.

The service's tail-refined validation lattice uses unweighted count quantiles
with the fixed per-axis bands 0--1%, 1--4%, 4--8%, 8--96%, 96--99%, and
99--100%. The distributed spatial-pole pre-reduction instead uses a fixed
compact coordinate `x/(x+eta)` for positive pole energy and intrinsic width.
Each local shard deposits mass and complex first moments onto at most
`lattice_bins^2` cells in each of two fixed pole intervals before any
collective. Fixed-size `psum` operations produce the global centroids. Thus
the communicated cell count is independent of process count, state count,
and spatial pole extent; no raw state--spatial-pole support is gathered.

Each causal branch is covered by ordinary Cartesian product
windows, `state interval x pole interval` (a degenerate one-pole branch may
need only one). A window selects its states with a plain interval mask and its
leading poles with a plain interval bound. A window is specified by
`(E_min, E_max)` and `(Omega_min, Omega_max)` and nothing else. The crossing
band stays inside its resonant product window. Crossing supports first use the
measure-adapted ROQ route; sign-definite supports start from shipped
`noncrossing` tables. An exact small integer selection chooses one candidate
per window subject to the global delivered-error budget and the derived pair
ceiling.

The production target is **envelope-relative**. It multiplies the
noncancelling inverse-gap planning envelope by the unchanged 0.8 safety factor
and is apportioned among windows in proportion to delivered mass times
measured inverse-gap difficulty. It is not a physical relative Sigma-error
target. When a reference Sigma is supplied, the planner reports the measured
`inverse_gap_envelope / max|Sigma_reference|` exchange rate; without a
reference it reports that calibration as unavailable. Since the initial
difficulty is envelope divided by delivered mass, mass-times-difficulty
apportionment intentionally gives each retained window the same normalized
envelope residual target. A sign-definite lookup walks to the next tighter or
wider certified table when the first covered entry misses the measured gate.
For a crossing window, deterministic ROQ derives the contour, horizon, and
rank from the measured support. Shipped HGL entries and one deterministic
fixed-time fit are tightening candidates, not an explicit-pair fallback. The
refined lattice validates every selected rule.

Amplification multiplies runtime arithmetic noise, not the physical fit
error. Acceptance is therefore

```text
residual <= target
kappa_p99 * 6.0e-8 <= 0.05 * target
```

`6.0e-8` is the measured fp32-scale injected-noise figure from the trust
study; the executor itself uses fp64, so it is conservative. The factor 0.05
reserves five percent of the window's physical residual allowance for
runtime noise. For example, a target of `8.26e-5` admits
`kappa_p99 <= 68.8`. Peak amplification remains diagnostic and is not a
fixed-cap acceptance gate. The adapter separately refuses a rule when either
executed `G` or `W` exponential exceeds the log-growth cap, even if their
product would cancel.

Every window is served by a quadrature rule. A refusal names the product
window and the best achieved `(residual, kappa_p99)` pair; there is no
alternative route for a window that cannot be served, and no per-state or
per-pole evaluation exists anywhere in the planner or the executor.

Every result is an ordinary `SharedSigmaWindow` consumed by the existing MPA
window executor and `DeviceOmegaAccumulator`. The adapter converts the fitted
time orientation to the executor convention, folds `exp(-eta*t_exec)` into
each weight once, and retains the executor's global `-1` prefactor. The pole
interval and state mask remain independent executor selectors, so the
streamed pole-batch walk evaluates their Cartesian product without adding a
tuple axis to a large spatial tensor.

### Ceiling and tightening retry

There is one grid mode: each product window carries the nodes of its served
rule. The planner derives a global `(window, tau)` ceiling from the measured
supports. Its honest-cost estimate is `2A/eta` for each crossing window and 20
nodes for each sign-definite window; twice their sum, with a floor of 32, is the
runaway guard. This is derived policy, not a deck dial.

The first selection pass offers the measure-adapted crossing rule and the
lookup-served sign-definite rules. Only if their achieved absolute costs cannot
close the global budget does a second pass add the shipped crossing/tighter
lookup candidates. The adapted fit is reused. Branch consolidation is tried
only below the split plan's node count and is cached across this retry. If a
merge makes the global budget unaffordable, selection retries the already-fit
unmerged windows before refusing. No retry creates a coupled state--pole
selector or direct evaluator.

Reports retain `window_tau_pairs` and `distinct_tau_count`; the latter is a
census of exact node values by branch, not a selectable shared-grid plan.

`LORRAX_DELIVERED_PLAN_CACHE=/path/to/receipt.npz` stores fitted rules and the
measured-problem fingerprint. A complete matching receipt loads before the
pole census and survives a process-count change. A fit-only hit is validated
against the live measured problems before execution. Any fingerprint or gate
mismatch replans; the cache never makes a stale rule executable.

## GN-PPM and time reversal

GN-PPM presents its single fitted pole per spatial matrix element as a
singleton leading pole axis. Degenerate one-pole measures therefore produce
one Cartesian executable window per causal branch without asking a partition
routine to invent empty bins. MPA retains its fitted leading pole axis and may
produce several product windows. Both pathways assert that the delivered
plan contains no direct rows before execution.

Positive- and negative-frequency causal branches are always separate planner
inputs. The adapter does not reconstruct one W half from the other, preserving
the current time-reversal-broken W producer seam.

## Scope of the envelope statement

The planner report records node counts, fit and refined residuals,
amplification, apportioned envelope budgets, and planning wall time. Its
inverse-gap currency applies to the scalar residue-weighted reciprocal measure
used for planning. The projected state weights make that measure responsive to
the real requested band projection, but spatial phases and cancellations still
prevent the target from being called a physical relative Sigma error. A full
real-material run still needs the normal P=4 integration and
reference-comparison gates.

MPA planning reads the complete fitted pole axis once in its native spatial
sharding and reduces each addressable shard before fixed-size cross-process
reduction.
The executor then returns to configured pole batches. GN-PPM reuses its
already-resident single pole. In both modes the default `panes` arm retains its
previous reads, plan, and execution order.
