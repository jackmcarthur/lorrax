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

Each causal branch is covered by two to four ordinary Cartesian product
windows, `state interval x pole interval` (a degenerate one-pole branch may
need only one). A window selects its states with a plain interval mask and its
leading poles with a plain interval bound. There are no explicit state--pole
tuple lists, membership predicates, or frequency staircases. The crossing
band stays inside its resonant product window. Its `eta`-damped reciprocal is
integrated by the positive real-time quadrature; candidate density follows
the oscillatory bandwidth relative to the `eta` floor. An exact small integer
selection chooses one candidate rule per window while enforcing at most 200
total `(window, tau)` pairs across all branches.

The production target is **envelope-relative**. It multiplies the
noncancelling inverse-gap planning envelope by the unchanged 0.8 safety factor
and is apportioned among windows in proportion to delivered mass times
measured inverse-gap difficulty. It is not a physical relative Sigma-error
target. When a reference Sigma is supplied, the planner reports the measured
`inverse_gap_envelope / max|Sigma_reference|` exchange rate; without a
reference it reports that calibration as unavailable. Since the initial
difficulty is envelope divided by delivered mass, mass-times-difficulty
apportionment intentionally gives each retained window the same normalized
envelope residual target. Sign-definite windows use the minimax service's
incumbent `fit_damped_reciprocal`; crossing windows use the `eta`-damped
positive rule. The refined lattice validates every selected rule.

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

This construction emits no direct terms. A refusal names the product window
and the best achieved `(residual, kappa_p99)` pair. The separately configured
`LORRAX_DELIVERED_MAX_DIRECT_TERMS` ceiling (default 32) remains in the shared
driver contract and report as a fail-closed guard; the owner construction's
target and achieved direct count are both zero.

Every result is an ordinary `SharedSigmaWindow` consumed by the existing MPA
window executor and `DeviceOmegaAccumulator`. The adapter converts the fitted
time orientation to the executor convention, folds `exp(-eta*t_exec)` into
each weight once, and retains the executor's global `-1` prefactor. The pole
interval and state mask remain independent executor selectors, so the
streamed pole-batch walk evaluates their Cartesian product without adding a
tuple axis to a large spatial tensor.

### Shared tau grid

The delivered planner defaults to independent (`free`) window grids. Set

```bash
export LORRAX_DELIVERED_TAU_GRID=shared
```

to build one identical grid per causal branch and re-solve unrestricted
complex weights for every window on that grid. Candidate times are the exact
stable union of the already accepted free-window rules. Every refitted window
must still satisfy its refined residual, p99 noise budget, and factor-growth
gate.

Reports keep the two cost currencies separate: `window_tau_pairs` is
`sum_w n_tau(w)`, while `distinct_tau_count` sums the per-branch shared-grid
sizes. At execution, rows with the same exact tau grid and frequency block are
fused: all component `G/W` transforms remain explicit, but the Sigma-side
forward transform and band projection run once per distinct tau and resident
pole batch. The run log reports both actual tau dispatches and the saved Sigma
back-transforms. `direct_term_count` remains a separate reported currency and
is zero for this construction.
`max_nodes` is a global `window_tau_pairs` ceiling for the complete plan, not
a per-window allowance. Shared-grid selection converts the remaining global
budget to a per-branch grid ceiling before fitting, and a final fail-closed
check pins the total. Direct terms have their own separately reported ceiling
and must remain zero.

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
