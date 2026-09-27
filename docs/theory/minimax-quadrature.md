# Minimax quadrature for the space-time χ₀ and the Σ box rules

This page covers the gapped imaginary-time (Laplace) χ₀ behind static,
GN-PPM and HL-PPM screening and the rules it consumes (§1–6), and the
derived time rules for 1/d on Σ's denominator boxes (§7). Related material is
owned elsewhere:

- Σ's frequency integral, its windows, boxes, currencies and acceptance:
  [the dynamic Σ(ω) quadrature](sigma-quadrature-problem.md).
- MPA sampling and its damped line rules:
  [Multipole frequency integration](THEORY_mpa_implementation.md).
- Finite-occupation response: [Metallic MPA screening](metallic-mpa-screening.md)
  and its [carrier](../architecture/fractional_chi0_response_face.md).
- Service contracts: [`minimax`](../services/minimax.md).

## 1. Laplace kernels

A gapped transition Δ = ε_c − ε_v > 0 enters χ₀ through one scalar kernel of
Δ, and every kernel used here is a Laplace transform in imaginary time:

| response | kernel K(Δ) | k(τ) in K = ∫₀^∞ e^{−Δτ} k(τ) dτ |
|---|---|---|
| static | 1/Δ | 1 |
| GN probe, even | Δ/(Δ² + ω_p²) | cos ω_pτ |
| GN probe, odd (time reversal broken) | ω_p/(Δ² + ω_p²) | sin ω_pτ |
| HL probe (Ω > Δ_max) | Δ/(Δ² − Ω²) | two shifted 1/y kernels, §3 |

A rule K(Δ) ≈ Σ_l α_l e^{−Δτ_l} on [Δ_min, Δ_max] factors e^{−Δτ} into an
occupied and an empty factor, so the (v, c) pair sum becomes one product of
Green's functions per node:

$$
G^v_{\mathbf k}(\tau)=\sum_{v}\psi_{v\mathbf k}\,e^{-\tau(\varepsilon^{\max}_v-\varepsilon_{v\mathbf k})}\,\psi^\dagger_{v\mathbf k},
\qquad
G^c_{\mathbf k}(\tau)=\sum_{c}\psi_{c\mathbf k}\,e^{-\tau(\varepsilon_{c\mathbf k}-\varepsilon^{\min}_c)}\,\psi^\dagger_{c\mathbf k},
$$

$$
A_{\mu\nu}(\mathbf R,\tau)=\sum_{ss'}G^c_{\mu s,\nu s'}(\mathbf R,\tau)\,G^v_{\mu s,\nu s'}(\mathbf R,\tau)^*,
\qquad
\chi^0(\mathbf q)=-\sum_l\alpha_l\,e^{-\tau_lE_g}\,\mathcal F_{\mathbf R\to\mathbf q}\big[A(\tau_l)+A(\tau_l)^*\big],
$$

with E_g = ε_c^min − ε_v^max. G(R) is the lattice Fourier transform of G_k on
the ISDF centroids r_μ, so the Hadamard product in R is the convolution over
the k grid that pairs k with k − q, done by FFT. Both factors decay for every
τ ≥ 0. The two particle–hole orientations are A and A^*. When time reversal is
measured broken, the ordered route weights A by −(α_l − iβ_l) e^{−τ_lE_g},
with β the odd kernel's coefficients on the same times, and completes the
other orientation as the complex conjugate of that result at −q
([derivation](../dev/notes/DERIVATION_gnppm_nonhermitian.md)).

## 2. Cost

Per node, on P ranks, with N_k k-points (N_k^par symmetry parents), N_μ
centroids and n_s spinor components:

| step | cost |
|---|---|
| G^v and G^c: one complex GEMM per parent each, contracting only its own band support (active range) | 8 N_k^par (N_μn_s)² (N_v + N_c) / P flops |
| two FFTs over the k grid | O((N_μn_s)² N_k log N_k / P) |
| product and spin trace in R | O(N_k (N_μn_s)² / P) |

One final FFT R → q follows the sweep, and every q is produced at once. The
live set is two full-k Green tiles and the accumulator, each
O(N_k (N_μn_s)² / P). The total, O(N_τ N_k N_μ² (N_b + log N_k)), is cubic in
system size; the direct pair sum costs O(N_q N_k N_v N_c N_μ²). N_τ is the
only frequency-dependent factor.

## 3. The rules

**Static, 1/Δ.** The interval runs from Δ_min, the gap (floored at the
occupation smearing width when one is declared), to Δ_max = ε_c^max − ε_v^min.
The conduction side is the union of the χ and Σ band windows, so one interval
serves both. The rule is solved on [1, R], R = Δ_max/Δ_min, and rescaled:
τ = τ̂/Δ_min, α = ŵ/Δ_min. The physical absolute error target
`minimax_target_error` becomes the scaled target `minimax_target_error`·Δ_min.
The solver is a levelled Remez exchange that returns the smallest N whose best
uniform N-term error meets the target, and certifies it by alternation: if
1/x − Σ_l w_l e^{−t_l x} alternates in sign at 2N + 1 points with magnitude
≥ δ, no N-term exponential sum does better than δ (de la Vallée Poussin). Its
count is

$$
N=\mathcal O\!\left(\log R\,\log\frac1\epsilon\right).
$$

Rules are computed at run time in milliseconds, capped at `minimax_max_nodes`,
and cached by (log R, target, cap).

**GN probe at iω_p.** `minimax.response_laplace_rule` places positive times
on [Δ_min, Δ_max] at the single sample z = iω_p (reference Δ_min) and projects
the even target Δ/(Δ² + ω_p²) with a continuum certificate at relative
tolerance `minimax_target_error`
([Compact noncrossing response quadrature](response-laplace.md)). The odd
target is the same rule's ordered row, i times ω_p/(Δ² + ω_p²), on the same
times, so a broken-time-reversal deck adds no nodes.

**HL probe at Ω > Δ_max.**

$$
\frac{\Delta}{\Delta^2-\Omega^2}=\frac{1/2}{\Omega+\Delta}-\frac{1/2}{\Omega-\Delta}.
$$

Both denominators are positive on the interval, so each is a static 1/y rule,
on [Ω + Δ_min, Ω + Δ_max] and [Ω − Δ_max, Ω − Δ_min]; the second enters with
negated times. A real-axis probe carries only the even orientation, so on a
broken-time-reversal deck its odd channel is not represented.

MPA's insulating imaginary samples use the service's `noncrossing_imag` solve
of Δ/(Δ² + ϖ²) on the same interval. Its damped line rules and the metallic
rules belong to the pages above.

## 4. Error and amplification

Every rule is judged in its target's norm by two numbers:

$$
\epsilon_{\max}=\max_{\Delta\in[\Delta_{\min},\Delta_{\max}]}\Big|K(\Delta)-\sum_l\alpha_le^{-\Delta\tau_l}\Big|,
\qquad
\kappa=\sup_\Delta\frac{\sum_l|\alpha_le^{-\Delta\tau_l}|}{|K(\Delta)|}.
$$

κ multiplies roundoff, ISDF error and reduction-order differences between
ranks, so a small residual with a large κ is not a good rule. The service
reports both with each rule, and the driver prints the rule's provenance, node
count and achieved error. The energy reference (`midgap` by default) cancels
from χ₀, because only Δ enters.

## 5. Refusals

| refusal | fix |
|---|---|
| `GATE chi0_laplace_needs_gap`: ε_c^min ≤ ε_v^max over the χ band slices | a gapless system takes the finite-occupation routes (`compute_mode = mpa`; the material class is inferred from the WFN occupations) |
| HL probe with Ω ≤ Δ_max | HL-PPM is defined only above every transition |
| `GATE gn_ppm_analytic_probe_real`: complex even or odd probe coefficients | none; the rule is inconsistent with its real target |
| no certified GN probe rule through 64 nodes | a smaller Δ_max/Δ_min (band window) or a looser `minimax_target_error` |
| `GATE chi0_imag_ordered_needs_odd_kernel`: ordered χ₀ without odd weights | build the probe with the odd kernel |

A static rule that cannot reach its target within `minimax_max_nodes` is not
refused: the service returns the capped rule, and the printed achieved error
is the only record.

## 6. Ownership

`services/minimax` owns the targets, solvers, certificates and caches.
`gw.minimax_screening` owns the physical intervals, the energy reference, the
rescaling and the probe adapters. `gw.w_isdf` owns the τ sweep. The Green's
function builder and the FFT helpers carry no quadrature policy.

## 7. Σ denominator-box rules

`minimax.analytic_box_rule(box, ε)` returns Q(d) = Σ_k w_k e^{i t_k d} ≈ 1/d
on a box [a, b] × [η, y_max], η > 0, in the box's currency: peak-relative
η|Q − 1/d| on a crossing box (a < 0 < b), relative |d||Q − 1/d| on a
sign-definite one. Every node is a formula of (box, ε); the weights are one
linear least-squares solve on the box boundary with a ridge 0.05ε on each
term's largest contribution in the currency. Units below are η = 1,
L = ln(1/ε), Λ = ln(4/ε).

**The trapezoid identity.** For Im d > 0 and |Re d| < B,

$$
\frac1d=\frac{\pi}{B}\cot\frac{\pi d}{B}+\frac2B\int_0^\infty\frac{\sinh(td/B)}{e^t-1}\,dt ,
\qquad
\frac{\pi}{B}\cot\frac{\pi d}{B}=-\frac{2\pi i}{B}\Big[\tfrac12+\sum_{k\ge1}e^{ik(2\pi/B)d}\Big].
$$

Proof: π cot πz − 1/z = ψ(1−z) − ψ(1+z) = −2∫₀^∞ sinh(zt)/(e^t − 1) dt.
The cot term is the trapezoid rule of 1/d = −i∫₀^∞ e^{isd} ds at spacing
2π/B; its whole error is the integral, which splits into e^{±td/B}, two
Laplace integrals on the imaginary time axes s = ∓it/B, i.e. 1/d's images at
d = ±B. Each is sign-definite on the box and costs O(log(M/η)·L) Gauss nodes.
The line itself only has to meet the interior Nyquist condition.

**The bent contour (crossing boxes).** Orient the box so its wide half-width
M lies at x < 0 and its narrow half-width m at x > 0. At a node
s = σ − iτ a member d = x + iy has |e^{isd}| = e^{−σy+τx}, so it is live
while −σy + τx > −Λ, and Poisson summation folds a live x onto
x − 2πj/Δ: the local density must be γB(σ)/2π with B(σ) the largest live |x|.
The smallest τ that reaches the minimum B is

$$
\tau(\sigma)=\min\!\Big(\frac cm,\ \frac{\Lambda-\sigma}{m}\Big),\qquad c=4 ,
$$

so the contour is a vertical leg 0 → −ic/m (the wide side's Laplace part,
⌈ln(Mc/m)L/π²⌉ nodes graded geometrically toward 0), the capped line
σ − ic/m, and a linear fall to the real axis at σ = Λ, where every live member
is below ε/4. The corner amplifies the narrow edge by e^c, and the executor
admits a term mass of 5·10⁻⁶/6·10⁻⁸ = 83.3, so c must stay below
ln 83.3 = 4.42; c = 4 is a choice under that bound, not a derivation. The two
image sets sit at ±B₀,
B₀ = γ·max(B(0), min(M, Λm/c)); each has
K = ⌈ln(16R)(L + c)/π²⌉ Gauss–Legendre nodes, R = (B₀ + x_live)/(B₀ − m), and
the growth-side image is capped at |Im s|(b − a) ≤ 3. With κ = c/L the line
count is N_line = 1 + γI/2π,

| regime | I = ∫B dσ |
|---|---|
| m ≤ M ≤ m(1+κ) | m[(1+κ)Λ − cκ/2] |
| m(1+κ) < M ≤ mΛ/c | MΛ − cM²/(2m) + (mc/2)(1+κ+κ²) |
| M > mΛ/c (saturated) | mΛ²/(2c) + (mc/2)(1+κ+κ²) |

The margin is γ² − 1 = 8(L + c)/(πMΛ), clipped to
[1 + 0.01·max(2, L − cM/m), 1.2]. That expression minimizes a simplified
count, a straight line γMΛ/2π plus two image sets at R = (γ + 1)/(γ − 1); it
is not the minimizer of the bent count above, and the floor, not the
expression, sets γ on 290 of the 371 corpus crossing boxes (claim 2882). A
symmetric box bends too: the straight line leaves the far image to the
growth-capped set and misses on a tall box ([−20, 20] × [1, 10]η: 1.49ε),
where the bent contour certifies. On a miss the ladder raises γ by 1.1 and
adds one node to the leg and to each image set, four rungs.

A crossing box whose narrow side is 4–8η and whose height is 10η or more
certifies in neither family ((−60, 6) × (1, 20)η at 10⁻⁴ ends at 7ε), where
the fitted builder certified with 21–52 nodes; the planner refuses such a
window by name. No deck box has this shape today (`KNOWN_LORRAX_ISSUES`).

**The sector rule and the local extremal-length law.** A box with Im d ≥ η
lies in an open sector of the upper half plane. Rotate by the sector axis φ;
the times are the elliptic time-Ritz times of the rotated box's real extent
(`minimax.laplace_ritz.place_times`, a Lyapunov solve and one symmetric
eigensolve) rotated back. In w = ln d the box is a strip of local half-gap
g(ℓ) = π/2 − max|θ − φ| over the box's arc at radius e^ℓ, and the count is

$$
n=\frac{L}{\pi^2}\left[\int\frac{\pi/2}{g(\ell)}\,d\ell+\ln4\cdot\frac\pi2\Big(\frac1{g_{\rm near}}+\frac1{g_{\rm far}}\Big)\right],
$$

Zolotarev's ln 16 split into one ln 4 per end at that end's own gap; for a
constant gap it is the strip count ln(16R)L/(π²(1 − 2γ/π)). φ minimizes the
law on a fixed 200-point grid. The law has no calibrated constant, and it is
an estimate, not a bound. Over the 953 fitted sign-definite rules the median
fitted/⌈law⌉ ratio is 1.000 (claim 2873), and against the continuous law it
is 1.075. Rung 0 certifies 315 of those 953 boxes (45 of 120 in QAUDIT's
sample, claim 2882); the ladder n → max(n + 1, ⌈1.1n⌉), six rungs, carries
the rest, and the certified counts sit at a median 1.15× the rung-0 law.
Sign-definite boxes weight the fit geometrically in |Re d|, the relative
currency's measure.

**Which family.** A sign-definite box takes the sector rule. A crossing box
builds first the family with the smaller count, N_line + leg + images or the
sector law, and the other when the first ladder ends uncertified. The sector
rule wins when the narrow side lies inside the peak (m of a few η), where the
bent contour's fixed 10–16-node overhead dominates. A rule neither family
certifies is returned uncertified and the planner refuses the window by name.

**Chosen constants.** None of these is derived; each was fixed once and not
tuned per box. Corner exponent c = 4 (under ln 83.3); line end Λ = ln(4/ε);
growth-side image cap |Im s|(b − a) ≤ 3 (the fitted rules' off-ray cap); margin
floor slope 0.01 (the one constant calibrated against the corpus) and margin
cap 1.2; leg start 0.05/(Mτ_c); the image-horizon guard
1/max(1 − m/B₀, 0.05); ridge 0.05ε; fit density 2 points per half wave of the
largest |t| on the real edges and 40 geometric points on the sides; the
ladders (×1.1, four crossing and six sector rungs); the sector φ grid (200
points) and its gap guard (0.02 rad).

**Acceptance.** A rung is accepted when the boundary certificate
([§8 of the Σ page](sigma-quadrature-problem.md#8-acceptance)) reads sup ≤ ε and
the term mass ρΣ|w e^{itd}| ≤ the executor's 5·10⁻⁶/6·10⁻⁸ in the same
currency. The cancellation ratio Σ|term|/|Q| is not the gate: on a crossing
box it grows like |d|/η and reads 2.6·10⁴ on a certified Na box.

## References

- Rojas, Godby and Needs, *Phys. Rev. Lett.* **74**, 1827 (1995).
- Kim, Martyna and Ismail-Beigi, *Phys. Rev. B* **101**, 035139 (2020).
- Braess, *Nonlinear Approximation Theory* (1986).
- Hackbusch, *Hierarchical Matrices: Algorithms and Analysis* (2015).
- Beylkin and Monzón, *Appl. Comput. Harmon. Anal.* **28**, 131 (2010).
