# Compact noncrossing response quadrature

`minimax.response_laplace_rule` returns positive real times and projection
rows for response kernels on a transition interval [L, H] that no sample
frequency crosses. `minimax.laplace_ritz` owns the placement, the projection
and the continuum certificate. The production consumer is the GN-PPM
imaginary-axis probe ([Minimax quadrature §3](minimax-quadrature.md)). The
shared-pole response bank uses the grouped shared-node rules of
`minimax.response_group_rules` instead: complex times shared by a group of
samples, with sampled rather than continuum error bounds
([shared-pole model §2](../architecture/shared_pole_model.md)). Energies
share one unit (Ry in GW), times its inverse, and derivatives are taken with
respect to s = z².

## Targets

For samples z with W = max|Re z| < L and transitions d ∈ [L, H]:

$$
K_e=\frac{d}{d^2-z^2},\quad \partial_sK_e=\frac{d}{(d^2-z^2)^2},\qquad
K=\frac1{d^2-z^2},\quad \partial_sK=\frac1{(d^2-z^2)^2}.
$$

The last two serve ordered (broken time reversal) response through
K_o = zK and ∂_sK_o = K/(2z) + z∂_sK. Every target is represented on the same
times as Σ_j c_j e^{−(d−r)t_j}, where r ∈ (W, L] is the caller's reference:
the returned coefficients include e^{−(r−L)t} and the Green's-function factors
supply e^{−(d−r)t}. Physical transitions obey d ≥ r, so no production operand
grows.

## Times from geometry alone

For degree N set a = (L − W)/2, b = (H + W)/2, k′ = a/b, k = √(1 − k′²), and
take the elliptic rates

$$
\alpha_j=b\,\mathrm{dn}\!\left(\frac{(2j+1)K(k)}{2N},\,k\right),\qquad j=0,\dots,N-1,
$$

evaluated by a complementary-modulus product that never subtracts nearly
equal numbers. In the orthonormal basis of the exponentials e^{−α_j t},
differentiation has A_jj = −α_j and A_ij = −2√(α_iα_j) for i < j, and zero
below the diagonal. Solve AᵀM + MA = −I. The eigenvalues of the symmetric
time-moment matrix M are the times: the Ritz values of multiplication by t on
that subspace. They are positive because ∫t|f|²/∫|f|² > 0, and equal rates
give Gauss–Laguerre times divided by 2α. Energies are scaled by √(ab) before
this N × N solve. The ill-conditioned Cauchy Gram matrix is never formed, and
no nonlinear optimization or node bank is used.

The elliptic rates bound the Blaschke-product projection error of the
subspace. That bound does not certify the finite sum; the certificate below
does.

## Projection on the fixed times

Sample d at max(400, 16N) Chebyshev points in log(d − W) and form
B_lj = e^{−(d_l−L)t_j}. Each target is fitted independently by a column-scaled
real least-squares solve with relative row weights, the derivative targets
included, so no derivative of a frequency-dependent weighted solve is taken.
Coefficients may be complex; the times stay real and positive, so the
consumer's Green and FFT kernels are unchanged.

## Continuum certificate

The domain is split into geometric panels in d − W, with ratio 1.15 per
panel. On each panel every exponential is expanded to degree 20 in
y = (d − centre)/halfwidth, with remainder bounded by
e^{−(left−L)t}(halfwidth·t)^{21}/21!. The polynomial is multiplied by the
exact denominator (d² − z² or its square for K_e and K; d ∓ z or its square
for the ordered primitives), and the exact numerator is subtracted. The bound
is the l¹ norm of the residual's Chebyshev coefficients, plus the remainder
times the denominator bound, plus a floating-point guard.

The certificate covers every real d in [L, H] at every supplied z. It is built
from the rounded returned rows, including the odd rows' 1/(2z) term, with the
reference shift reversed in extended precision, and it checks the forward and
backward primitive sums separately. It bounds K and ∂_sK rather than the
relative error of the odd derivative, which is undefined at that
derivative's zeros. The finite sum is compared with the full rational target,
so no infinite-time tail is omitted. The arithmetic guard assumes ordinary
libm accuracy; it is not interval arithmetic.

## Degree, reuse and cost

Degree starts at min(64, max(4, ⌈ln(16ρ) ln(1/ε)/π²⌉)), with
ρ = (H + W)/(L − W), and rises by one until the certificate passes. A previous
rule is reused when its digest is intact, its certified domain contains
[L, H] and it passes the current frequencies; otherwise a new rule is built,
optionally on a padded domain. All of this is host scalar work, dominated per
degree by the (400 or 16N) × N least squares and one degree-20 expansion per
panel, of which there are ⌈ln((H − W)/(L − W))/ln 1.15⌉.

## Refusals

| refusal | cause |
|---|---|
| `remote Laplace integral does not converge` | L ≤ max\|Re z\|: the interval is crossed and belongs to a crossing rule |
| `remote reference must lie above \|Re(z)\| and at or below delta_lo` | reference r outside (W, L] |
| `remote Ritz certificate failed through 64 nodes` | tolerance too tight for the interval ratio |
| `response rule node digest mismatch` | a corrupted reused rule |
| `Loss of positivity in the time Ritz spectrum` | numerical breakdown of the Ritz solve |
| `Reference shift exceeds scalar certification range` | e^{(r−L)t} overflows extended precision; raise r toward L |
