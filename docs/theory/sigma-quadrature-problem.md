# The dynamic Sigma(omega) denominator-box quadrature

Implementation: `gw/sigma_box_plan.py`. Executor:
`gw/mpa/sigma.py::_integrate_sigma_batches`. Numerical rule service:
`minimax.uniform_rule.build_uniform_rule`.

## 1. Problem and separability

For one causal branch, the MPA correlation self-energy contains terms

```
R_np / d_jnp,       d_jnp = omega_j - sigma_b (E_n + Omega_p),
```

where `sigma_b=+1` for conduction and `-1` for valence. The fitted retarded
poles have `Im Omega_p <= 0`; the requested broadening `eta` is applied once in
the executable weights.

The reciprocal is replaced by a short exponential sum,

```
1/d ~= sum_l w_l exp(i t_l d).
```

Each term then factors into independent external-frequency, electronic-state,
and screened-pole factors. One `(window,tau)` pair therefore costs one spatial
Green-function/screened-interaction transform, rather than an explicit
state-by-pole contraction. The total number of `(window,tau)` pairs is the
resource currency and is refused when it exceeds `mpa_sigma_max_nodes`.

### 1.1 Ordered one-pole completion on a magnet

GN/HL one-pole fits are persisted in the same pole store and execute through
this MPA planner and spatial kernel. On a measured-broken-TR GN deck the
imaginary probe is not replaced by its Hermitian half: ordered odd-kernel
nodes retain the anti-Hermitian, frequency-odd channel of
$\chi^0(i\omega_p)$. The two-point fit therefore produces two Hermitian pole
residues,

$$
B=\frac{R_++R_-}{2},\qquad D=\frac{R_+-R_-}{2},\qquad
R_+=B+D,\quad R_-=B-D.
$$

The algebraic store stamps this as an ordered fit and writes `B_odd_p = D`
beside `B_p = B`. Every executable row carries its Green-function space:
conduction rows contract $R_+$ and valence rows contract $R_-$. Thus the box
plan changes the reciprocal approximation, not the magnetic contour
completion. The planner uses `max(abs(R+), abs(R-))` as a branch-neutral
liveness and partition witness, so a pole live in either branch is retained;
the executor still receives the exact residue for that row.
`LORRAX_DEBUG_GN_ODD_RESIDUE_OFF=1` is the named diagnostic twin: it keeps the
ordered route but sets $D=0$, making both rows consume $B$ and making the
reported odd Sigma contribution vanish. It refuses on a single-residue/TRS
store. The response and residue derivation is in
[GN-PPM without global time reversal](../dev/notes/DERIVATION_gnppm_nonhermitian.md).

## 2. Product windows

Each of the four causal branches (positive/negative external frequency,
conduction/valence) first partitions its central pole rectangle into at most
three Cartesian products. Let

```
state_edge = sigma_window_edge_factor * eta
pole_edge  = max(abs(omega)) + state_edge + negative-state excursion.
```

For a crossing branch the products are:

```
resonant   : E <= pole_edge,  Re Omega <= pole_edge
state_tail : E >  pole_edge,  Re Omega <= pole_edge
pole_tail  : all E,           Re Omega >  pole_edge
```

For the other causal orientation they are:

```
bulk       : E >  state_edge, all poles
resonant   : E <= state_edge, Re Omega <= pole_edge
pole_tail  : E <= state_edge, Re Omega >  pole_edge
```

Before those products are built, residue-weighted CDFs of `Re Omega` and
`-Im Omega` identify the central 1st--99th percentile pole rectangle. Poles
outside it are not dropped: they enter disjoint outlier selectors, split by
the sign of the complete branch denominator where possible, and are crossed
with every live state. The resulting central and outlier products cover each
live state/pole tuple exactly once.

These are products because the executor windows `G` and `W` independently.
A diagonal predicate coupling one state to one pole would destroy that
separability. Empty products are omitted. Products are not merged: a
whole-branch rule widens cheap sign-definite tails into an expensive crossing
box and was measured to reintroduce the low-mass-state error the box rule
removed.

## 3. Direct support boxes

The caller first reduces distributed pole fields into the CDF described above,
then reduces exact live extrema for every central/outlier selector. Residue
magnitudes choose only that disjoint product partition; they never weight a
rule fit, its error, or its acceptance. For every product window, the real
support is the extrema of the literal corners

```
omega - sigma_b * (E + Re Omega)
```

over that window's external frequencies, live state energies, and selected
pole extrema. The positive imaginary support presented to the rule builder is
`[-Im Omega_min + eta, -Im Omega_max + eta]`.

The real interval is padded by 2% of `max(width,eta)`. A sign-definite edge is
allowed to move at most 30% toward zero, so padding never changes a
sign-definite box into a crossing box. This matters: widening a near-zero edge
changes the linear-in-bandwidth crossing rank, whereas far-edge widening of a
tail costs only logarithmically.

No histogram, lattice representative, envelope, or error apportionment enters
the construction or certification of a rule. The pole CDF partitions products
only, and every resulting product receives a uniform certificate on its
complete box. The former measure-adapted campaign produced small weighted
residuals while missing a low-mass Na state at the Fermi level by 0.95 meV.
The direct box makes the same statement for every live tuple.

## 4. Rule and error currency

`build_uniform_rule(box, eps, time_budget=...)` chooses the error currency:

- sign-definite boxes use `sup |d| |Q(d)-1/d| <= eps` (relative);
- boxes crossing `Re d=0` use
  `sup eta_min |Q(d)-1/d| <= eps` (peak-relative).

The distinction is physical. Peak-relative error on a far tail grows in
relative terms like `|d|/eta` and produced a 4 meV Na semicore error; the
relative tail rule reduced it to about 0.1 meV without extra asymptotic cost.
Conversely, relative error on a crossing box over-resolves harmless far edges
and added about 50% more nodes in the measured Na case.

The planner rechecks the rule's own sup certificate, the float32-runtime noise
allowance

```
kappa_p99 * 6e-8 <= 0.05 * eps,
```

where the percentile uses Voronoi area weights on the rule builder's own fine
certification cloud, not a physical histogram. The area weights remove the
cloud's adaptive sampling density, so the statistic remains a function of the
box and rule only. The planner also requires a maximum separately factored log
growth of 30. A refusal is final. It does not trigger a hidden tighter-`eps`
retry or a second quadrature family.

## 5. Causal conjugation and executor conventions

The rule service builds on `Im d > 0`. A lower-half-plane window is served by
`1/conj(d) = conj(1/d)`, implemented as `t -> -conj(t)`,
`w -> conj(w)`. The executable window is then assembled once with

```
time_exec = pole_sign * t
alpha_exec = w * exp(-eta * time_exec)
omega_sign = pole_sign * external_sign
project = "full"
prefactor = -1
```

and with state/pole reference shifts chosen at the bounded endpoint of each
factored exponential. `gw/sigma_box_plan.py` is the sole owner of these box
plan conventions; `gw/mpa/sigma.py` only dispatches the finished windows.

## 6. Cache and process independence

Rules depend only on `(box,eps,currency)`. Cache lookup is by containment: a
rule certified on a superset serves a requested subset. On a miss, only real
edges farther than `3*eta` from zero are widened by 1% and the far imaginary
edge by 1%; this lets nearby self-consistent iterations hit without raising the
crossing rank. The default cache lives below the run's `tmp` directory,
`sigma_quadrature_cache_dir=off` disables it, and a deck-relative or absolute
path may be supplied.

Independent window fits are assigned round-robin across processes. Only the
small fitted rules and receipts are gathered, so every rank assembles the same
plan without gathering a pole field or state-pole product.

## 7. Controls and dials

The production/default MPA and GN/HL-PPM route is the shared box plan.
`LORRAX_SIGMA_PLAN=panes` selects the frozen pane implementation only for
comparison controls; there is no campaign-planner route. Numerical policy is
carried by three deck keys:

```
sigma_quadrature_eps = 1e-4
sigma_quadrature_reduction_seconds = 120
sigma_quadrature_cache_dir = auto
```

The reduction budget trades planning wall for node count after an accepted
interpolatory rule exists; it does not weaken `eps`. Retarded broadening stays
under `sigma_regularization_ev` / `sigma_regularization_floor_ev`, and the total
pair ceiling stays `mpa_sigma_max_nodes`.
