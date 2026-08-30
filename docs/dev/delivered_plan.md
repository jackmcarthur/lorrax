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

The service's tail-refined validation lattice uses unweighted count quantiles with the
fixed per-axis bands 0--1%, 1--4%, 4--8%, 8--96%, 96--99%, and 99--100%.
The distributed spatial-pole pre-reduction instead uses a fixed compact
coordinate `x/(x+eta)` for positive pole energy and intrinsic width. Each
local shard deposits mass and complex first moments onto at most
`lattice_bins^2` cells before any collective. One fixed-size `psum` per
leading pole produces the global centroids. Thus the communicated cell count
is independent of process count, state count, and spatial pole extent; no raw
state--spatial-pole support is gathered.
The explicit state--leading-pole tuples are then split by delivered-mass
quantiles (up to four conduction and three valence windows). Each tuple block
is an executable selector, not merely a fitting surrogate. The MPA executor
factors it into a small coefficient table over states and the resident pole
batch, runs the canonical `build_G_tau` and screened-interaction synthesis for
those components, and sums them before the canonical Sigma transform and band
projection. No large spatial tensor acquires a tuple axis.

The production target is **envelope-relative**. It multiplies the
noncancelling inverse-gap planning envelope by the unchanged 0.8 safety factor
and is apportioned among windows in proportion to delivered mass times
measured inverse-gap difficulty. It is not a physical relative Sigma-error
target. When a reference Sigma is supplied, the planner reports the measured
`inverse_gap_envelope / max|Sigma_reference|` exchange rate; without a
reference it reports that calibration as unavailable. Since the initial
difficulty is envelope divided by delivered mass, mass-times-difficulty
apportionment intentionally gives each retained window the same normalized
envelope residual target; the later crossing-window redistribution remains a
separate safety step. Sign-definite windows use the minimax
service's incumbent `fit_damped_reciprocal`. Crossing windows use the
amplification-capped fixed-phase service fit over the union of the incumbent
MPA positive-time rule, the pane/HGL crossing family, and the service time
dictionary. The refined lattice validates the selected rule. Acceptance gates
the maximum amplification, while p99 remains diagnostic. The adapter also
refuses a rule when either separately executed `G` or `W` exponential exceeds
the log-growth cap, even if their product would cancel. A missed crossing
budget or safety gate may use exact
reciprocal summation when the failed block contains at most
`max_direct_terms` tuples. That fallback is reported in direct-term currency and does not
consume tau nodes; larger failed supports refuse.

Every result is an ordinary `SharedSigmaWindow` consumed by the existing MPA
window executor and `DeviceOmegaAccumulator`. The adapter converts the fitted
time orientation to the executor convention, folds `exp(-eta*t_exec)` into
each weight once, and retains the executor's global `-1` prefactor. Exact
fallback rows reuse the canonical Green's-function builder and band
projector, with the causal denominator evaluated at each required frequency
and `exp(-eta*t)` replaced by the same single `-i eta` fold in that
denominator.

### Shared tau grid

The delivered planner defaults to independent (`free`) window grids. Set

```bash
export LORRAX_DELIVERED_TAU_GRID=shared
```

to build one identical grid per causal branch and re-solve unrestricted
complex weights for every window on that grid. Candidate times are the union
of already accepted incumbent-discipline fits; progressively smaller shared
prefixes are retained only when every refined residual and p99 amplification
check still passes. The accepted free union, zero-padded per window, is the
deterministic fallback when it fits under the node ceiling.

Reports keep the two cost currencies separate: `window_tau_pairs` is
`sum_w n_tau(w)`, while `distinct_tau_count` sums the per-branch shared-grid
sizes. At execution, rows with the same exact tau grid and frequency block are
fused: all component `G/W` transforms remain explicit, but the Sigma-side
forward transform and band projection run once per distinct tau and resident
pole batch. The run log reports both actual tau dispatches and the saved Sigma
back-transforms. Exact reciprocal fallbacks are reported separately as
`direct_term_count` and never silently priced as tau nodes.
`max_nodes` is a global `window_tau_pairs` ceiling for the complete plan, not
a per-window allowance. Shared-grid selection converts the remaining global
budget to a per-branch grid ceiling before fitting, and a final fail-closed
check pins the total. Direct terms have their own separately reported ceiling.

## GN-PPM and time reversal

GN-PPM presents its single fitted pole per spatial matrix element as a
singleton leading pole axis. Degenerate one-pole measures therefore produce
one executable window per causal branch without asking a quantile routine to
invent empty bins. A tau-only one-pole tuple block is Cartesian-equivalent to
that ordinary GN window. Exact direct fallback remains an explicit GN
refusal; the direct executor currently belongs to the MPA streamed-pole
pathway. MPA retains its fitted leading pole axis and may produce several
windows.

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
