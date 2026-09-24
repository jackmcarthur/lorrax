# Minimax quadrature for the space-time χ₀

This page covers the gapped imaginary-time (Laplace) χ₀ behind static,
GN-PPM and HL-PPM screening, and the rules it consumes. Related material is
owned elsewhere:

- Σ's frequency integral: [the dynamic Σ(ω) quadrature](sigma-quadrature-problem.md).
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
| `GATE chi0_laplace_needs_gap`: ε_c^min ≤ ε_v^max over the χ band slices | a gapless system takes the finite-occupation routes (`mpa_material_class = metal`) |
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

## References

- Rojas, Godby and Needs, *Phys. Rev. Lett.* **74**, 1827 (1995).
- Kim, Martyna and Ismail-Beigi, *Phys. Rev. B* **101**, 035139 (2020).
- Braess, *Nonlinear Approximation Theory* (1986).
- Hackbusch, *Hierarchical Matrices: Algorithms and Analysis* (2015).
- Beylkin and Monzón, *Appl. Comput. Harmon. Anal.* **28**, 131 (2010).
