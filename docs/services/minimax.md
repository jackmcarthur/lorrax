# minimax — numerical quadrature service

`services/minimax/` is an independently installable NumPy service. Consumers
use `import minimax`; importing its submodules from LORRAX is a layering
failure. Importing the package does not import JAX or SciPy. SciPy is loaded
only by the offline/runtime solvers.

## Caller contract

| Surface | Contract |
|---|---|
| `lookup(...)` | Searches shipped certified tables only and raises a named F1–F4 refusal on an invalid or uncovered request; it never solves. |
| `serve(...)` | Computes and announces a screening rule in process. It does not consult shipped tables. The result carries achieved error and provenance. |
| `catalog()` / `nearest_certified(...)` | Enumerate certified coverage and suggest nearby covered requests without solving. |
| `family_for_character(...)`, `TARGETS`, `FAMILIES` | Define the accepted target and family vocabulary as data. |
| `Quadrature`, `Provenance` | Return nodes, weights, measured error, certification state, source, and artifact identity. |
| `build_uniform_rule(box, eps)` | Builds and certifies one denominator-box rule for `1/d` on `[re_lo, re_hi] x [im_lo, im_hi]`. A production surface, not a lookup: it takes no clock and no pass count, and returns when its own boundary certificate is met, so the same box and tolerance give the same rule on any machine. `gw.sigma_box_plan` is the consumer. |
| `response_bank_rule(...)` / `response_laplace_rule(...)` | Build the shared-pole W time rule and remote inverse-moment rows, with separate value and derivative contracts. `gw.response_bank` is the consumer. |
| `matsubara_response_rule(...)` | Builds a finite-temperature KMS-paired imaginary-time rule. |
| `augment_odd_laplace(...)` | Adds the odd GN-PPM resolvent channel on the existing even rule's time nodes, refusing a missed sampled gate. |
| `damped_line_rule(...)` / `damped_rectangle_rule(...)` / `damped_rectangle_gauss_rule(...)` / `damped_rectangle_positive_rule(...)` | Build MPA's positive-time line and rectangle rules. The rectangle constructors retain their respective geometric and error contracts; GW passes scalar bounds and receives time nodes and weights. |
| `positive_reciprocal(...)` / `odd_reciprocal(...)` / `damped_line_reciprocal(...)` | Experimental analytic constructions for three specific one-dimensional reciprocal domains; none is a general replacement for a production target with a different kernel or error currency. |
| offline solver names | Lazily expose table-generation machinery; using them does not make the result a shipped certified rule. |

There is no escape hatch and no shipped-table branch: `serve` computes every
screening rule in process, announces it once naming the request, the achieved
error, the measured sum of |w| and kappa_0, and certifies it with a de la
Vallee Poussin alternation certificate. `LORRAX_MINIMAX_ALLOW_RUNTIME_SOLVE`
retired with that branch on 2026-09-16 and is read nowhere; its row in the
[environment-variable registry](../dev/env_vars.md) says so.

## One service, distinct target contracts

`lookup` remains available to inspect historical certified assets;
production `serve` computes the screening rule in process. Sigma's
deck-dependent denominator rectangles use `build_uniform_rule`, while the
shared-pole W bank needs both response values and derivatives at its current
complex frequencies. MPA's damped line and rectangle builders also live here;
`gw.mpa.evaluator` retains only the exact scalar kernel and compatibility
exports, and `gw.mpa.sigma_windows` calls the service directly. The three experimental reciprocal constructors cover
one-dimensional domains only. Keeping these targets distinct prevents an
apparently shorter API from applying a valid rule to the wrong function.

The damped builders accept physical frequencies and damping heights in one
energy unit and return positive times in the inverse unit. Their default
relative tolerance is `DEFAULT_DAMPED_REL_TOL`; the line builder also exposes
`DEFAULT_WAVELENGTHS_PER_PANEL`. The common Legendre interval nodes are cached
inside the service, so repeated panel widths do not redo the polynomial
eigensolve. The scalar MPA kernel remains in `gw.mpa.evaluator` as a physics
oracle, not a node selector.

## Catalog and selection

Catalog entries bind a family, target, range selector, error bound, node limit,
payload hash, achieved error, amplification, generator provenance, and backend.
Selectors such as `beta_selector` and `damped_line_selector` are public module
objects at the package door because their clauses are part of catalog
selection. An explicit malformed, missing, or insufficient entry refuses; it
is not treated as a silent cache miss.

## Boundary with LORRAX's odd imaginary-axis kernel

The magnetic GN-PPM odd channel is determined by
`minimax.augment_odd_laplace`: it retains the served even nodes and greedily
adds at most 16 nodes to fit `omega_p / (x**2 + omega_p**2)` to the requested
absolute error. Candidate exponential columns are built once and reused
across weight-only Lawson fits. `gw.minimax_screening` only converts physical
units and appends zero even weights on the added nodes; the service refuses
if the odd gate is missed. The augmentation is a sampled runtime fit, not a
new continuum-certified table.

`gw.w_isdf.compute_chi0_imag_ordered` requires `alpha_odd` and consumes the
even and odd weights on one node axis. The derivation and limiting identities
are owned by the [non-Hermitian GN-PPM memo](../dev/notes/DERIVATION_gnppm_nonhermitian.md).

## Experimental reciprocal constructors (not production dispatch)

The lazy package door exposes `positive_reciprocal(R, tolerance)`,
`odd_reciprocal(A, tolerance)`, and
`damped_line_reciprocal(bandwidth_over_broadening, tolerance)`. All three use dimensionless
**absolute** error and return an object with `times` and `evaluate(x)`.
The positive rule has positive `strengths` in the stable shifted convention
`sum strengths * exp(-(x-1)*times)`; the old unshifted coefficients are
`strengths * exp(times)`. The odd rule has `weights` in
`sum weights * sin(x*times)`. The line rule has complex `weights` in
`sum weights * exp(i*u*times)` for `1/(u+i)` on
`|u|<=bandwidth_over_broadening`; damping is already in the coefficients.
To approximate physical `1/(x+i*height)` to absolute error `eps`, construct
with `(span/height, height*eps)` and call `rule.rescaled(height)`.
No driver is routed to these constructors yet.

The positive constructor prescribes elliptic interpolation abscissae, solves
their moments in elevated precision, and optionally applies two measured
error-envelope corrections. Its reported extremum is a numerical audit, not
an interval certificate. The sine rule's pole-aware interpolation provides an
exact-arithmetic bound equal to its reported core plus correction bounds.
The line rule is a conservative composite Gauss-Legendre quadrature of the
causal positive-time integral: its tail is `exp(-height*T)/height` and its
panel bound follows the standard Gaussian `2n`-derivative remainder summed
over panels. It does **not** implement the discussion's proposed one-sided
csc correction, whose complete certificate has not been derived. The
comparison and its exact source pin are in the sandbox report
`reports/analytic_quadrature_2026-09-16/report.md`.

`import minimax` remains NumPy-only. Calling the positive or sine constructor
loads optional SciPy, and positive construction also needs mpmath (the `solve`
extra). These are exploratory rules, not catalog entries; the catalog's
provenance and production selection promises remain unchanged.

## Verification

The standalone package tests live in `services/minimax/tests/`; the monorepo
layering test enforces the top-level door. Lookup tests must run without SciPy,
while solver-generation tests may require it.


## Response-bank rule sessions

`response_bank_rule(z_ry, delta_max_ry, rel_tol=..., previous=None,
domain_pad_ry=0)` returns positive real-time nodes and weights, value and
s-derivative projections, a node digest, and continuum panel/tail bounds.
`response_laplace_rule(delta_lo_ry, delta_hi_ry, z_ry, ...)` returns positive
Laplace nodes and inverse-moment coefficient rows, with continuum row bounds
and current-frequency Taylor bounds. Both accept the previous in-memory result.
A hit retains its integration arrays exactly and regenerates projections for
all supplied frequencies. The bank norm is eta-scaled absolute value/derivative
error; the remote norm is relative error. Neither certifies W or Sigma accuracy.

`domain_pad_ry` enlarges a newly built transition domain. Remote lower padding
stops at positivity and cannot cross the Taylor convergence boundary. Reuse
requires current-domain containment, the same tolerance, an intact node digest
and passing current-frequency bounds. Otherwise the owner builds a new rule;
corrupt integration arrays refuse. Receipts report `reuse_status`,
`reuse_reason`, `node_digest` and the actual certified domain. No wavefunctions,
response matrices, physical samples or W models live in this session.

### Response-rule currencies and certificates

Frequencies and transition intervals are in Ry, times in inverse Ry, derivatives with respect to `s = z²`.

- `response_bank_rule` (real-time Hermite panels): positive `(t, h)` returning `h e^{izt}` and the derivative
  row `h e^{izt}·it/(2z)`. The currency is peak-scaled absolute error (`η·|value|`, `η³·|derivative|`,
  `η = min Im z`); the certificate bounds the full signed transition interval and both exponential branches,
  tail and panel budgets included. It does not certify relative W or Σ accuracy.
- `response_laplace_rule` (remote cells): a nonnegative NNLS fit of `δ/(δ²+η²)^{n+1}` on a positive time
  dictionary, refined until the interval certificate passes (endpoint errors plus a second-derivative bound
  from the log-derivatives of each positive term, never the training grid alone). With
  `q = (s+η²)/(δ²+η²)` the value Taylor remainder is bounded by `ρ^{N+1}` and the derivative remainder by
  `ρ^N((N+1)+Nρ)`, `ρ = sup|q|`; a nonconvergent Taylor domain refuses and must be repartitioned by the bank.

Neither rule sees band masks, occupations or response arrays. `tests/test_response_rules.py` checks analytic
kernels, the missing `1/(2z)` derivative red twin, positivity and refusals.

### Finite-temperature Matsubara rules

`matsubara_response_rule(beta_ry_inv, delta_max_ry, n_indices, rel_tol=...)` returns nodes `t` in
`(0, beta/2]` and complex weights `W[k, l]` for bosonic `nu_k = 2 pi k / beta`, used as
`sum_l W h(t_l) + conj(W) h(beta - t_l)` on any imaginary-time correlation whose terms are
`f_m (1-f_n) exp(-(e_n - e_m) tau)` (Fermi-Dirac, KMS-bounded by 1). Pairs of either sign are covered: KMS
maps a negative transition onto the mirrored node. The certificate is the sup over `y` in `[0, delta_max]` of
the even target `y(1-e^{-beta y})/(y^2+nu^2)` and odd target `nu(1-e^{-beta y})/(y^2+nu^2)` errors, each
relative to its own sup, on a dense grid with the largest samples refined locally; an amplification above
`0.1 rel_tol / eps_machine` refuses. The node set solves Kaltak and Kresse's finite-temperature problem
(PRB 101, 205145 (2020)) by pivoted-QR compression of a graded Gauss-Legendre pool with least-squares
weights, not by their nonlinear minimax optimization. `tests/test_matsubara_rules.py` checks the Lindhard
weight `(f_m - f_n)/(x - i nu)` pair by pair (including `-df/de` at `nu_0`) with one-particle factors in log
form, a wrong-frequency red twin, and refusals.
