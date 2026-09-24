# minimax — numerical quadrature service

`services/minimax/` is an independently installable NumPy service. Consumers
write `import minimax` and use top-level names; importing a submodule from
LORRAX is a layering failure. Importing the package loads neither JAX nor
SciPy. Each constructor loads its numerical dependencies when called; SciPy
and mpmath come from the `solve` extra.

The namespace holds several distinct rule families, one per kernel, domain
and error currency. A rule valid for one target is not valid for another with
the same bandwidth, so callers select the constructor that matches their
kernel. The disk cache and uniform-rule backend are controlled by
`LORRAX_MINIMAX_CACHE_DIR`, `LORRAX_DISABLE_MINIMAX_DISK_CACHE` and
`LORRAX_UNIFORM_RULE_BACKEND`, whose rules
[`docs/dev/env_vars.md`](../dev/env_vars.md) owns. A cached rule always carries
`source='cache'` and `certified=False`.

## Caller contract

| surface | contract | production consumer |
|---|---|---|
| `serve(*, family, target, range_value, error_bound, n_max, eps_q=None, omega_hat=None) -> Quadrature` | Solves a screening rule in process for families `noncrossing`, `noncrossing_imag` (needs `omega_hat`) and `crossing` (`eps_q`, target `hgl` or `fermi`). It never reads shipped tables and refuses any other selector keyword. | `gw.minimax_screening` |
| `lookup(...)`, `catalog()`, `nearest_certified(...)` | Search, enumerate and suggest shipped certified tables. Never solve. | tests and generators |
| `TARGETS`, `FAMILIES`, `CHARACTERS`, `family_for_character(...)` | Target and family vocabulary as data. | |
| `Quadrature`, `Provenance` | Nodes, weights, `max_error`, κ₀, certification state, source and artifact identity. | |
| `build_uniform_rule(box, eps, ...)` | Builds and certifies one rule for `1/d` on the denominator box `[re_lo, re_hi] × [im_lo, im_hi]`. It returns when its own boundary certificate is met, with no clock or pass-count input, so a given box and tolerance give the same rule on any machine. | `gw.sigma_box_plan` |
| `analytic_line_box_rule(box, eps)` | Fixed-height line rule in Σ's `Σ w e^{itd}` convention, checked to peak-relative error. Crossing boxes with one positive imaginary height only. | `gw.sigma_box_plan`, PPM real-pole crossing windows |
| `fit_damped_reciprocal(rectangles, *, target_error, ...)` | One positive rule `1/d ≈ Σ w e^{−d t}` for `1/(x − iγ)` over rectangles `(x_min, x_max, γ_min, γ_max)`, `x_min > 0`. | `gw.mpa.sigma_windows` |
| `damped_line_rule`, `damped_rectangle_rule`, `damped_rectangle_gauss_rule`, `damped_rectangle_positive_rule` | MPA positive-time line and rectangle rules, § [Damped MPA rules](#damped-mpa-rules). | `gw.mpa.model` |
| `augment_odd_laplace(times, x_min, x_max, omega, *, tolerance, max_extra=16)` | Odd GN-PPM channel on the even rule's times, § [Odd imaginary-axis channel](#odd-imaginary-axis-channel). | `gw.minimax_screening` |
| `response_group_rules`, `response_laplace_rule`, `response_bank_rule` | Shared-pole response rules, § [Response-bank rule sessions](#response-bank-rule-sessions). | `response_group_rules`: `gw.response_bank`; `response_laplace_rule`: `gw.minimax_screening`; `response_bank_rule`: none |
| `matsubara_response_rule(beta_ry_inv, delta_max_ry, n_indices, *, rel_tol=1e-8)` | Finite-temperature KMS-paired imaginary-time rule, § [Matsubara rules](#finite-temperature-matsubara-rules). | `gw.w_isdf` |
| `positive_reciprocal`, `odd_reciprocal`, `damped_line_reciprocal` | Analytic constructions on three one-dimensional reciprocal domains, § [Analytic reciprocal constructors](#analytic-reciprocal-constructors). | none directly; `analytic_line_box_rule` wraps the line rule |

`gw.mpa.evaluator` re-exports the four damped builders for import
compatibility and keeps the scalar MPA kernel as a physics oracle, not a node
selector.

**`serve`.** `noncrossing` is solved by `noncrossing_levelled` and bounded by a
de la Vallée Poussin alternation certificate; `noncrossing_imag` and
`crossing` return the smallest rule whose measured grid error meets the
target. When `n_max` nodes do not reach `error_bound`, the `n_max` rule is
returned with its larger `Quadrature.max_error`; the caller checks it. Every
distinct request is announced once (`RuntimeWarning`) with node count,
max error, Σ|w|, κ₀ and provenance. Node positions can differ in the last
digits between hosts, because the solve goes through the local LAPACK, so
compare rules by node count and error, not bytes.

**`lookup` refusals** are `MinimaxRefusal` subclasses (`RuntimeError`):
`NoCertifiedTable` (F1, outside the catalog), `AmplificationCap` (F2, κ₀ above
the declared cap), `UnknownTarget` (F3, outside the vocabulary; also raised by
`serve`), and `CatalogUnavailable` / `TableUnreadable` / `CatalogCorrupt`
(F4a–c). A malformed or insufficient entry refuses; it is never a silent cache
miss. Catalog entries bind family, target, range selector, error bound, node
limit, payload hash, achieved error, amplification, generator provenance and
backend; `beta_selector` and `damped_line_selector` are public modules because
their clauses are part of selection.

## Damped MPA rules

The damped builders take physical frequencies and damping heights in one
energy unit and return positive times in the inverse unit.
`damped_line_rule(varpi, freq_max, *, rel_tol=DEFAULT_DAMPED_REL_TOL,
wavelengths_per_panel=DEFAULT_WAVELENGTHS_PER_PANEL, max_order=256)` is a
composite Gauss–Legendre quadrature of the causal positive-time integral. The
rectangle constructors keep their own geometric and error contracts; GW passes
scalar bounds and receives times and weights. Legendre interval nodes are
cached inside the service, so repeated panel widths do not repeat the
polynomial eigensolve.

## Odd imaginary-axis channel

`augment_odd_laplace` keeps the served even nodes in order and greedily adds
at most `max_extra = 16` nodes to fit ω/(x² + ω²) on `[x_min, x_max]` to the
requested absolute error, reusing precomputed candidate exponential columns
across weight-only Lawson fits. It returns
`(times, odd_weights, added_count, sampled_max_error)` and raises if the
sampled gate is missed. It is a sampled fit, not a continuum certificate.
`gw.minimax_screening` converts units and appends zero even weights on the
added nodes; `gw.w_isdf.compute_chi0_imag_ordered` consumes even and odd
weights on one node axis. The derivation is in the
[non-Hermitian GN-PPM memo](../dev/notes/DERIVATION_gnppm_nonhermitian.md).

## Response-bank rule sessions

Frequencies and transition intervals are in Ry, times in Ry⁻¹, and
derivatives are with respect to s = z². No rule sees band masks, occupations
or response arrays, and none certifies W or Σ accuracy.

* **`response_group_rules(lo_ry, hi_ry, z_ry, *, rel_tol=1e-8, previous=None,
  decay_rate=0)`** (the shared-pole bank). A stacked Hankel shift pencil
  proposes complex Laplace times shared by every forward (z) and reverse
  (−z̄) pole of a sample group; linear projection fits 1/(d − p) and
  1/(d − p)² on those nodes. Each node is one Green-pair evaluation A(t):
  forward rows use exp[−(d − reference)t], reverse rows use conj(A(t)). A
  group whose shared fit fails is halved down to single samples. Each rule
  has `members`, `t[RESPONSE_NODE_CAPACITY]` (384; one pencil holds
  `RESPONSE_RULE_CAPACITY = 192`), `value`/`derivative` of shape
  `[members, 2, 384]`, `count`, `sampled_error`, `coefficient_mass` and
  `reference_ry`; zero coefficients mark inactive slots. Errors are sampled,
  not proven. A positive `decay_rate` bounds occupation products by
  min(1, e^{decay_rate·d}) and restricts 0 ≤ Re t ≤ decay_rate. `previous`
  rules with matching members are tried first.
* **`response_laplace_rule(delta_lo_ry, delta_hi_ry, z_ry, *, rel_tol=1e-8,
  previous=None, domain_pad_ry=0, ordered=False, reference_ry=None)`** (remote
  noncrossing cells). Elliptic decay rates, a small Lyapunov solve and a
  symmetric time-moment eigensolve prescribe positive real nodes; linear
  projection fits exact even value/ds (and, with `ordered`, K/dsK) targets on
  them. Projections include e^{−reference·t}; the consumer supplies
  e^{−(δ − reference)t}. A continuum polynomial/exponential residual bound
  certifies the rounded returned arrays, and the degree grows until it passes.
  The norm is relative error. [Derivation and accuracy
  scope](../theory/response-laplace.md).
* **`response_bank_rule(z_ry, delta_max_ry, *, rel_tol=1e-8, previous=None,
  domain_pad_ry=0)`** (no production consumer). Positive real-time Hermite
  panels `(t, h)` returning h e^{izt} and h e^{izt}·it/(2z). The currency is
  peak-scaled absolute error (η|value|, η³|derivative|, η = min Im z), and
  the certificate covers the full signed transition interval, both branches,
  tail and panels.

For the two real-time rules, `domain_pad_ry` enlarges a newly built domain
(remote lower padding stops at positivity and cannot cross
δ_lo > max|Re z|). Reuse of `previous` requires current-domain containment,
an intact node digest and passing current-frequency bounds at the requested
tolerance (the bank rule also needs an unchanged tolerance); otherwise a new
rule is built, and corrupt integration arrays refuse. Results report
`reuse_status`, `reuse_reason`, `node_digest` and the certified domain.
`services/minimax/tests/test_response_rules.py` checks analytic kernels, the missing-1/(2z)
derivative red twin, positivity and refusals.

## Finite-temperature Matsubara rules

`matsubara_response_rule(beta_ry_inv, delta_max_ry, n_indices, *,
rel_tol=1e-8)` returns nodes t ∈ (0, β/2] and complex weights W[k, l] for
bosonic ν_k = 2πk/β, applied as

$$\chi(i\nu_k) \approx \sum_l W_{kl}\, h(t_l) + \overline{W_{kl}}\, h(\beta - t_l)$$

to any imaginary-time correlation whose terms are
f_m(1 − f_n) e^{−(ε_n − ε_m)τ} (Fermi–Dirac, KMS-bounded by 1). KMS maps
negative transitions onto the mirrored node, so pairs of either sign are
covered. The certificate is the sup over y ∈ [0, δ_max] of the even target
y(1 − e^{−βy})/(y² + ν²) and odd target ν(1 − e^{−βy})/(y² + ν²) errors, each
relative to its own sup, on a dense grid refined near the largest samples; an
amplification above 0.1·rel_tol/ε_machine refuses. Nodes come from pivoted-QR
compression of a graded Gauss–Legendre pool with least-squares weights (the
Kaltak–Kresse finite-temperature problem, PRB 101, 205145 (2020), solved
without their nonlinear minimax optimization).
`services/minimax/tests/test_matsubara_rules.py` checks the Lindhard weight
(f_m − f_n)/(x − iν) pair by pair, including −∂f/∂ε at ν₀, with a
wrong-frequency red twin.

## Analytic reciprocal constructors

All three take a dimensionless **absolute** tolerance and return an object
with `times` and `evaluate(x)`.

| constructor | target | form | guarantee |
|---|---|---|---|
| `positive_reciprocal(R, tolerance)` | 1/x on 1 ≤ x ≤ R | Σ strengths·e^{−(x−1)t} (unshifted coefficients are strengths·e^{t}) | high-precision numerical extremum audit (mpmath), not an interval certificate |
| `odd_reciprocal(A, tolerance)` | 1/x on [−A, −1] ∪ [1, A] | Σ weights·sin(xt) | exact-arithmetic bound: reported core plus correction bounds |
| `damped_line_reciprocal(u_max, tolerance)` | 1/(u + i) on \|u\| ≤ u_max | Σ weights·e^{iut}, damping folded into the weights | composite Gauss–Legendre: tail e^{−T}, panel bound from the 2n-derivative remainder |

To approximate the physical 1/(x + i·height) to absolute error ε, construct
the line rule with `(span/height, height·ε)` and call
`rule.rescaled(height)`. `analytic_line_box_rule` removes e^{−height·t} from
the weights, because Σ's e^{itd} already carries that damping; it accepts only
a real interval crossing zero at one positive height. Sign-definite tails and
finite-height rectangles use `build_uniform_rule`.

## Verification

Package tests live in `services/minimax/tests/`; the monorepo layering test
enforces the top-level door. Lookup tests must pass without SciPy.
