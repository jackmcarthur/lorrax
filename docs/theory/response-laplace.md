# Compact noncrossing response quadrature

`minimax.response_laplace_rule` returns positive real times and projection
rows for response kernels on a transition interval [L, H] that no sample
frequency crosses. `minimax.laplace_ritz` owns the placement, the projection
and the continuum certificate. The production consumer is the GN-PPM
imaginary-axis probe ([Minimax quadrature §3](minimax-quadrature.md)). The
shared-pole response bank uses the grouped rules of
`minimax.response_group_rules` (§ [Grouped response rules](#grouped-response-rules)
below; [shared-pole model §2](../architecture/shared_pole_model.md)). Energies
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

## Grouped response rules

**The integral.** A sample group's rule serves, for every pole p of its
samples (forward z and reverse −z̄, both with Im p > 0) and every real
transition d ∈ [lo, hi] of the occupation support,

$$
\frac1{d-p}\simeq\sum_j c_j(p)\,e^{-(d-r)T_j},\qquad
\frac1{(d-p)^2}\simeq\sum_j c'_j(p)\,e^{-(d-r)T_j},
$$

with one Green pair per complex time T_j (r = lo on an insulator, μ on a
metal). The currency is the pole's own peak: Im p·|error| for the value and
(Im p)³|error| for ∂_s = (2z)⁻¹∂_z, each at rel_tol/2. The bank calls it at a
tenth of the tier's bank tolerance (`gw.response_bank._GROUP_RULE_MARGIN`):
the derived rule's error is uniform at its certificate over the whole
transition interval, and the shared-pole construction amplifies it (Na 8³
map 0: eqp moves 1.17, 0.43 and 0.04 meV against a 10⁻¹⁰ reference at
group tolerances 10⁻⁸, 10⁻⁹ and 10⁻¹⁰).

**It is the Σ box problem.** With D = p − d the targets are −1/D and 1/D² on
the horizontal segment {p − d} at height Im p, and T = it maps
e^{−(d−r)T} to e^{itD} up to the pole factor e^{it(p−r)}. So the poles of one
height form one thin box [min Re p − hi, max Re p − lo] × {Im p}, and the Σ
box rules of [minimax quadrature §7](minimax-quadrature.md#7-σ-denominator-box-rules)
apply unchanged in their node formulas.

**Families.** The group's levels split into families, each on the smallest
box holding its levels:

- sign-definite levels (Re D of one sign, e.g. imaginary samples on a gapped
  system): the elliptic sector rule (time-Ritz times of the rotated box,
  `laplace_ritz.place_times`);
- crossing levels: the bent contour.

A second candidate splits off the crossing levels whose narrow side lies
within one height (a high imaginary sample on a metal, −lo < Im z): they lie
in an open sector too and take the sector rule, keeping the highest such
levels whose rung-0 times hold 0 ≤ Re T ≤ β. The builder tries the candidates
in order of their rung-0 node union (a formula of the boxes) and takes the
first whose ladders certify: the split wins on Fe (188 against 228 nodes) and
is the only one that certifies a 320 eV level on the 1 keV Fe interval; the
merged partition wins on Na (144 against 165). A sign-definite or peak-narrow
level inside the bent contour's box pays the line's Nyquist density at that
box's lowest height. The node set is the union of the families' sets; every
node is one Green pair for every member.

**Three additions to the Σ bent contour**, each a formula, because the bank's
tolerance (5·10⁻⁹ against Σ's 10⁻⁴) and its ds rows expose what Σ's rule
leaves out:

- nodes at ε = tol/ln(4/tol): the ds target's time density is s·e^{isD}, one
  horizon factor larger than the value's;
- ⌈(c + ln(1/ε))/2⌉ Gauss–Legendre nodes on the leg [0, −ic/m]: there the
  narrow side grows as e^{cu}, which the geometric grading toward 0 does not
  resolve below 10⁻⁸;
- on a tall box (height ratio H), ⌈γS(H − 1)/π⌉ Gauss–Legendre nodes on the
  decaying image axis [0, −iS], S = (L + c)/(max(1 − m/B₀, 0.05)B₀): its
  members turn their phase through S(H − 1) radians there.

The bend is c = 6 (capped by the tall-box leg phase as in Σ): the executor
admits a term mass of 5000, and e⁶ leaves the least-squares cancellation a
factor of ten, as Σ's c = 4 sits under its cap of 83.

**Node count.** With η the crossing box's lowest height, m and M its narrow
and wide half-widths in η, L = ln(1/ε) and Λ = ln(4/ε), the crossing family
costs the Σ line count 1 + γI/2π (the I table of minimax quadrature §7) plus
⌈ln(max(Mc/m, 2))L/π²⌉ leg, two ⌈ln(16R)(L + c)/π²⌉ image sets and the three
additions above. On a complete-basis interval the box is saturated
(M > mΛ/c), and

$$
N_C\approx\frac{\gamma\,m\,\Lambda^2}{4\pi c}+\mathcal O\!\left(\ln\frac{M}{m}\,L\right),
\qquad m=\frac{\max\operatorname{Re}z-lo}{\eta},
$$

linear in the narrow side, logarithmic in the span: Fe's 1 keV interval
enters only through ln(M/m). The sign-definite family costs the extremal-length
law of §7, O(ln(span/η)·ln(1/ε)).

**Weights and certificate.** Nothing but the weights is solved: one linear
least-squares solve per pole height on the union (value and ds as two right
sides, columns scaled to unit maximum, ridge 0.005·tol per term; columns dead
below e⁻⁴⁰·ε on that level are left out). The certificate evaluates each
level's segment at six points per half wave of the union's largest |t| and
refines every local maximum by golden-section search; a level passes when
value and ds sups are ≤ tol and the term mass Im p·Σ|c e^{−(d−r)T}| ≤ 5000.
Admissibility is exact: |Re T|(hi − lo) ≤ 3 on growth-side times (no Green
factor grows past e³) and Re T ≤ β under a metal's occupation envelope
min(1, e^{βd}); an inadmissible rung is skipped. A family whose levels fail
climbs its fixed ladder (sector: six rungs; bent contour: c, c/2, c/4 with six
margin rungs each), and the union is refit. A group whose ladders run out, or
whose union exceeds `RESPONSE_NODE_CAPACITY` = 768, refuses as
`GATE response_rule_certificate`; no group is split and nothing is searched.

