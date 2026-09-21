# Time-domain quadrature for dynamic heads: scope and design

This is a design analysis, not a new production route. The numerical target is a
material head self-energy change at 0.2% relative accuracy with literal
`eta = 0.25 eV`; a contribution established below 1 micro-eV is a stop case.
The dense direct head remains the reference. The current source and measured
status are recorded in the sandbox
`reports/head_release_2026-09-17/time_domain_review.md`.

## 1. What can be separated in time

For a causal independent-particle response at `Im z > 0`, let
`a=(n,k)`, `b=(m,k-q)`, `Delta_ab=E_b-E_a`, and let `X_ab^{IJ}(q)` be the
**ordered** pair of charge/current vertices, including its actual left and
right orientation. With the source normalization factored into `C`,

$$
\chi^{0,IJ}_q(z)=C\sum_{ab}\frac{(f_a-f_b)X^{IJ}_{ab}(q)}{z-\Delta_{ab}}
=-iC\int_0^\infty e^{izt}[A^{IJ}_q(t)-B^{IJ}_q(t)]\,dt,
$$

$$
A_q(t)=\sum_{ab}f_a(1-f_b)X_{ab}(q)e^{-i\Delta_{ab}t},\qquad
B_q(t)=\sum_{ab}(1-f_a)f_bX_{ab}(q)e^{-i\Delta_{ab}t}.
$$

The identity `f_a-f_b=f_a(1-f_b)-(1-f_a)f_b` holds even for
Methfessel-Paxton occupation overshoot. The weights must remain linear;
square-rooting or clipping them changes the response. For charge density the
vertex adjoint gives `B_q(t)=A_{-q}(-t)^T`; this is an orientation identity,
not time reversal. Its implementation in `gw.response_bank` builds two
weighted one-particle Green sums, multiplies them after their band sums, and
transforms the real-space correlation to `q`. The ordered bank uses
`F_q[chi]`, not the time-reversal-only `F_q[chi^T]` orientation. See
[shared-pole model](../architecture/shared_pole_model.md#2-the-response-bank)
and [metallic MPA screening](metallic-mpa-screening.md#1-the-finite-occupation-response-and-its-cancellation-structure).

For a positive transition `Delta` write its two residues separately:

$$
\chi_\Delta(z)=\frac{R_+}{z-\Delta}-\frac{R_-}{z+\Delta}
=\frac{2\Delta R_e+2zR_o}{z^2-\Delta^2},\quad
R_e=\tfrac12(R_++R_-),\quad R_o=\tfrac12(R_+-R_-).
$$

Equivalently its positive-time commutator is
`2 R_o cos(Delta t) - 2i R_e sin(Delta t)`.
The sine and cosine terms both enter the causal integral. Under time
reversal the odd term may vanish, but no head/current or magnetic path may
discard it by assumption. At `q=-q`, the odd part can be an imaginary,
Hermitian antisymmetric tensor; contraction with `q_a q_b` annihilates the
antisymmetric *charge* part but says nothing about CT/TC or the ordered
wings. The left and right wing endpoints remain independent through the
Schur fold and only meet in the final observable. This is the same
particle-hole convention as the [ordered shared-pole model](../architecture/shared_pole_model.md#1-the-object).

The head response has a small Cartesian dimension, but its source may be a
large band-pair bank. Applying the above transform to that bank avoids a
separate pair denominator for every requested frequency. For centroid
body `chi0`, `gw.response_bank` already implements the distributed Green/FFT
form; a new pairwise implementation would duplicate it and reintroduce an
O(N^4)-class stage. To include current endpoints, the existing response
kernel must accept the authenticated charge/current vertices and produce
both ordered orientations. It is not enough to call the charge-only bank and
label its output a photon response.

## 2. Damping, quadrature rank, and the likely crossover

At fixed physical `eta`, `z=x+i eta`, so `e^{izt}` damps as `e^{-eta t}`.
A finite upper time `T` must cover the slowest active transition and the
largest cancellation amplification. A first design estimate is
`T ~ log(C/epsilon)/eta`. Resolving a transition bandwidth `B` without
aliasing needs `dt` of order `pi/B` or finer, giving
`N_t = O((B/eta) log(C/epsilon))`. The actual service rule can move complex
nodes, choose nonuniform spacing, and certify a smaller count; this
estimate is a cost envelope, not a node prescription or error certificate.
At `eta=0.25 eV`, `B=100 eV`, and `epsilon=0.002`, the crude estimate
`B log(1/epsilon)/(pi eta)` is about 790 positive-time points, before
constants, quadrature error, and the ordered endpoint cost. If a dense head
asks for 387–773 upper-line frequencies, a full time stream is not
automatically cheaper. It becomes promising when many more frequencies,
frequency derivatives, or repeated external-state queries reuse the same
correlation. The sandbox `reports/analytic_quadrature_2026-09-16/report.md`
found a normalized causal line constructor with `R=B/eta`; its conservative
Gaussian implementation uses 192/476 nodes at `R=50/120` for absolute
`1e-4` in normalized units. Those counts are for a scalar reciprocal on a
fixed line, not a demonstrated head-response or self-energy speedup.

For a retained transition bank with `N_pair` terms, `N_z` requested
frequencies, and `m` small head channels, direct response assembly is
roughly `O(N_pair N_z m)` with a cheap small Dyson solve afterward.
Transform assembly is `O(N_pair N_t m + N_t N_z m)` and still needs one
nonlinear Dyson/mini-BZ evaluation per `z`. For the full centroid body the
Green/FFT stream has its own sharded `O(N^3)` spatial work per time node;
the full `W(z)` also needs a distributed solve per sample. Therefore a
time rule does not turn hundreds of body solves into one. Compare measured
wall and peak bytes at fixed geometry, `P`, allocator, `eta`, and delivered
Sigma error before choosing it over direct sparse sampling. Do not infer a
speedup by comparing the 796-row frozen-body pilot with the 165-row
dynamic-body experiment: their workloads differ.

## 3. Screened head and self-energy are different transforms

The charge head is `S(z)` plus the exact ordered wing fold

$$ S_{\rm eff}(z)=S(z)+Y(z)W_b(\Gamma,z)Z(z)/\Omega. $$

The small screened head is formed by the existing mini-BZ average of
`[D_h(q)^{-1}-R_eff(q,z)]^{-1}`. This inverse and the body Dyson solve are
nonlinear in `chi0`; Fourier transforming a sampled `chi0(t)` does not
directly give `W_h(t)`. Build or model the complete screened head only after
the body and wings have been included, using the existing cubature and
Schur owners. The direct dense head is affordable and remains the baseline.
For costly full-body corrections, stream a bounded batch of frequency
points, immediately reduce `Y W_b Z` and the relevant small CC/CT/TC/TT
coefficients, and release the `mu^2` body before the next batch. The
measured CrI3 producer already uses this layout; its bank is a useful
reference for a future fit, not a full-frequency current-photon model.

For general off-shell Sigma, a causal model of `W_c=W-v` can be converted to
time factors and contracted with `G(t)`. The current
[denominator-box plan](sigma-quadrature-problem.md) already implements this
separation: product windows over states and poles, a certified reciprocal
exponential sum, and one spatial contraction per `(window,time)` pair.
The [shared-pole Sigma consumer](../architecture/shared_pole_model.md#8-the-%CF%83-consumer)
already synthesizes `W_+(q,t)` from common real poles and routes valence
through `W_+(-q,t)^T`. A new generic pairwise contour or time-grid Sigma
path is redundant and would violate the product-window scaling invariant.
The remaining head problem is to model or directly integrate the *small
completed head* accurately at the requested observable, not to rebuild the
whole Sigma executor.

## 4. Instantaneous and metallic pieces

Separate every nondecaying high-frequency term **before** finite-time
quadrature or a causal pole fit. A subtracted current response
`Pi_sub(z)=Pi_raw(z)-Pi_raw(0)` has
`Pi_sub(infinity)=-Pi_raw(0)`, which is a time-local `delta(t)` term in a
retarded kernel. A subtracted wing can leave a TT infinity coefficient even
though its dynamic part decays. Fold that wing with the *bare* high-frequency
body and evaluate the actual completed-head infinity through the same
small Dyson equation; do not assume that nonlinear screening leaves the
bare head unchanged. Pass the constant to the existing analytic/static
Sigma treatment with its occupation and contact convention, then transform
only the decaying remainder. An arbitrary finite pole used to mimic the
constant corrupts the contour and its high-frequency moment.

Metallic `D/z^2` and the Thomas-Fermi static analytic limit are also
separate analytic pieces. Their order of limits matters. For the leading
diagonal `q=0` electron line *on shell*, the spectral identity is
`Sigma^c_n(E_n)=(1/2-f_n)<W^c(q,0_analytic)>/(Omega N_k)`; there is no
frequency quadrature to accelerate for that observable. It does not hold
for a general off-shell or finite-`q` self-energy. A time-rule certificate
at `eta=0.25 eV` cannot certify the `z->0` metallic limit.

## 5. Implementation and acceptance sequence

1. Retain the dense direct head and the literal `eta=0.25 eV` as the
   reference. Price a candidate on the same state set, `z` domain,
   occupations, mini-BZ cubature, body, and left/right wing convention.
   Establish whether the requested difference is above 1 micro-eV before
   funding 0.2% convergence.
2. For a material charge/current response requiring many `z` values,
   extend `gw.response_bank`/the shared Green kernel with authenticated
   head vertices and ordered `+q/-q` orientations. Preserve the existing
   `f(1-f)` weighting, no pair tensor, all-P sharding, and response-rule
   certificate. Compare direct `S,Y,Z` at held positive, negative, lower,
   static and high-frequency points. Check both branch relations and the
   actual CT/TC parity; a same-frequency adjoint test alone is insufficient.
3. Stream `W_b(Γ,z)` and reduce each solve to the small Schur coefficients.
   Fit a causal small matrix model *after* this nonlinear fold, with
   conjugate/particle-hole closure, separate ordered residues, prescribed
   infinity constant, and current-state provenance. Reuse the
   shared-pole direction/gate machinery where its matrix passivity and
   moment assumptions apply. The current `gw.shared_pole_head` evaluates
   only the even `b(z²-Lambda)^-1b†` Γ body and explicitly refuses
   TR-broken and two-component full-head use; the signed ordered Γ
   evaluator and corresponding head fit must land before that route is
   enabled. Never feed an ordered store through the even evaluator.
4. Gate held completed-head values **and** the final complex head-Sigma
   difference, using independent direct dense frequencies. Refine a fit
   where its weighted Sigma defect is large; response norms or a uniform
   held-point percentage alone do not certify 0.2% Sigma. Record maximum
   absolute error, a denominator for relative error, nested-frequency
   change, high-frequency/contact closure, peak memory, and wall time.
   If the correction is below 1 micro-eV with an adequate cutoff margin,
   report the cutoff and stop without pursuing 0.2% for it.

The current CrI3 restricted current-wing change is such a stop case
(`0.703 micro-eV`, `0.031 micro-eV` promotion sensitivity). The Na finite-cell
head model is material against the analytic q-first Drude comparator, but its
leading diagonal on-shell integral is exactly static; its smaller difference
against the TF-consistent static comparator is below the stop line. These
are sandbox measurements on stated models, not a general bound on a future
full-frequency photon head.

## Published context

The [real-space/imaginary-time GW method of Rojas, Godby and Needs](https://www-users.york.ac.uk/~rwg3/Papers/Paper_34.pdf)
and [dual minimax imaginary grids of Kaltak, Klimeš and Kresse](https://pubs.acs.org/doi/10.1021/ct5001268)
target smooth imaginary-axis transforms and require a separate real-energy
continuation for a retarded head. The [complex-time shredded-propagator
method of Kim, Martyna and Ismail-Beigi](https://link.aps.org/accepted/10.1103/PhysRevB.101.035139)
is closer to the real-frequency product-window separation above. None of
those papers establishes the `eta=0.25 eV` Γ-head crossover or the
completed-head Sigma error in this code; those remain consumer measurements.
