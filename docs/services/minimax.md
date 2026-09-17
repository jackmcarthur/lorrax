# minimax — certified quadrature lookup

`services/minimax/` is an independently installable NumPy service. Consumers
use `import minimax`; importing its submodules from LORRAX is a layering
failure. Importing the package does not import JAX or SciPy. SciPy is loaded
only by the offline/runtime solvers.

## Caller contract

| Surface | Contract |
|---|---|
| `lookup(...)` | Searches shipped certified tables only and raises a named F1–F4 refusal on an invalid or uncovered request; it never solves. |
| `serve(...)` | Calls `lookup` first, then either announces and performs an uncertified runtime solve or raises F5 when that escape hatch is disabled. |
| `catalog()` / `nearest_certified(...)` | Enumerate certified coverage and suggest nearby covered requests without solving. |
| `family_for_character(...)`, `TARGETS`, `FAMILIES` | Define the accepted target and family vocabulary as data. |
| `Quadrature`, `Provenance` | Return nodes, weights, measured error, certification state, source, and artifact identity. |
| `build_uniform_rule(box, eps)` | Builds and certifies one denominator-box rule for `1/d` on `[re_lo, re_hi] x [im_lo, im_hi]`. A production surface, not a lookup: it takes no clock and no pass count, and returns when its own boundary certificate is met, so the same box and tolerance give the same rule on any machine. `gw.sigma_box_plan` is the consumer. |
| offline solver names | Lazily expose table-generation machinery; using them does not make the result a shipped certified rule. |

There is no escape hatch and no shipped-table branch: `serve` computes every
screening rule in process, announces it once naming the request, the achieved
error, the measured sum of |w| and kappa_0, and certifies it with a de la
Vallee Poussin alternation certificate. `LORRAX_MINIMAX_ALLOW_RUNTIME_SOLVE`
retired with that branch on 2026-09-16 and is read nowhere; its row in the
[environment-variable registry](../dev/env_vars.md) says so.

## Two surfaces, not one

The catalog half answers `lookup`/`serve` from shipped tables and is what the
rest of this page describes. The box-rule half (`build_uniform_rule`) solves a
fresh problem every call and ships no tables: Σ's windows are deck-dependent,
so there is nothing to tabulate. They share only the package door and the
convention that a rule carries its own measured error.

## Catalog and selection

Catalog entries bind a family, target, range selector, error bound, node limit,
payload hash, achieved error, amplification, generator provenance, and backend.
Selectors such as `beta_selector` and `damped_line_selector` are public module
objects at the package door because their clauses are part of catalog
selection. An explicit malformed, missing, or insufficient entry refuses; it
is not treated as a silent cache miss.

## Boundary with LORRAX's odd imaginary-axis kernel

The certified service tables and their bytes are unchanged by the magnetic
GN-PPM odd-kernel path. `gw.minimax_screening` is the adapter: when
`solve_laplace_minimax_imag_interval(..., with_odd_kernel=True)` cannot obtain
the odd component from a certified complex rule, it keeps the served even
nodes, greedily adds at most `ODD_KERNEL_MAX_EXTRA_NODES`, and fits
`omega_p / (x**2 + omega_p**2)` to the same requested error. The adapter marks
those extra weights in `LaplaceMinimaxQuadrature.alpha_odd`; it refuses if the
fit misses the gate. This augmentation is a runtime LORRAX fit, not a new
certified service table.

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
