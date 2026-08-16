# Metallic MPA screening

This chapter is the authoritative description of metallic (finite-occupation)
screening in LORRAX: the occupation-weighted response, the metal frequency
plan and the measured reasoning behind it, the two `q->0` heads and their
order of limits, the finite-`q` body, the occupation-weighted self-energy,
the per-iteration QSGW occupation state, and the metallic self-consistency
loop — what one map call rebuilds, the stop rule, and the measured
convergence. It subsumes the former
`finite-occupation-screening.md`, which is now a pointer here; the in-code
references to that filename remain valid through it.

Boundaries, per the [register](../index.md#register): sampling geometry, the
Loewner fit, window planning and the disk pipeline belong to
[Multipole frequency integration](THEORY_mpa_implementation.md); deck keys to
the [input reference](../input_reference.md); the `chi00 = q.S.q` head
convention to [S-tensor convention](s-tensor-convention.md); measured claims
to the sandbox ledger (`CLAIMS.md` rows cited as "claim NNN"). All equations
are in Ry, and the pole model is always

$$
W_c(z)=\sum_p\frac{2\Omega_pB_p}{z^2-\Omega_p^2},
\qquad \Omega_p=a_p-i\Gamma_p .
$$

Verification banner: every named symbol in this page was read at commit
`941db3a7` on `integ/metal-mpa-qsgw-2026-08-15` (2026-08-15; sections 1–5
unchanged since the `a5b1002b` reading); measured numbers carry their claim
row or probe record. `compute_mode = mpa` **no
longer refuses at driver entry**: `gw_config.UNIMPLEMENTED_MODES` was
emptied at `9c9b23dc` (2026-08-15) once the metal pipeline ran end to end.
Section 6.4 states exactly what that lift does and does not assert — it is
parseability, not convergence.

## 1. The finite-occupation response and its cancellation structure

### 1.1 The exact identity and the implemented form

With `a = (n,k)`, `b = (m,k-q)`, `Delta_ab = E_b - E_a`, and the
density-vertex outer product `X_ab(q)`, the independent-particle response at
`Im z > 0` is the finite-occupation Adler–Wiser sum over **all ordered band
pairs**,

$$
\chi_{0,q}(z)=C\sum_{ab}\frac{(f_a-f_b)\,X_{ab}(q)}{z-\Delta_{ab}} .
$$

The identity the implementation rests on is

$$
f_a-f_b=f_a(1-f_b)-(1-f_a)f_b ,
$$

exact for **any real** `f`, including Methfessel–Paxton overshoot `f<0`,
`f>1`. Define one positive-time product

$$
A_q(t)=\sum_{ab}f_a(1-f_b)\,e^{-i\Delta_{ab}t}\,X_{ab}(q),\qquad t\ge0 .
$$

Hermiticity of the density vertex, with the `q` and `-q` orientations kept
explicit (an orientation identity, **not** time reversal — valid for a fully
relativistic `no_t_rev` deck), gives the Keldysh partner without a second
pair build,

$$
B_q(t)=A_{-q}(-t)^{\mathsf T},
\qquad
\chi_{0,q}(z)=-i\int_0^\infty dt\,e^{izt}\left[A_q(t)-A_{-q}(-t)^{\mathsf T}\right],
$$

and in the flat-grid FFT convention
`FFT_R[conj(A_R(t))](q) = conj(A_{-q}(t)) = A_{-q}(-t)^T`. One time node
therefore costs **two weighted single-band Green sums** — one with
`band_weight = f`, one with `band_weight = 1-f` — that meet only after both
band axes are summed; no band-pair-by-node object exists. Implemented:
`gw.w_isdf.compute_chi0_contour_fractional` (public entry, refuses
`Im z <= 0`), kernel `w_isdf._get_chi_fractional_contour_kernel`, weight seam
`gw.greens_function_kernel.build_G_tau(band_weight=)` — the weight is applied
linearly after the phase and is never clipped or square-rooted. Support
selection is `w_isdf._exact_occupation_support_slices`: the smallest
contiguous `f` and `1-f` band supports, dropping only weights stored as
exactly 0 or exactly 1, with no tolerance; a partially occupied band belongs
to both supports. Gate: `tests/multi_device/fractional_chi_gate.py` pins the
kernel against a dense `O(N^4)` Adler–Wiser/Kubo oracle at `P=4`.

### 1.2 Where the cancellation lives: form A against the single-occupation split

Call the implemented weighting **form A**: pair weight `f_a(1-f_b)` on one
positive-time product, minus its mirror with `(1-f_a)f_b`. The textbook
alternative — **form B**, the single-occupation split — writes the same net
`f_a - f_b` as one term weighted `f_a` with the `b` propagator unweighted,
minus the mirror weighted `f_b`. Both are exact identities. The difference is
*where the cancellation between the two terms lives*, and that decides which
one a quadrature may safely discretize:

- **Form B.** Every occupied–occupied pair enters *both* terms with weight
  `~1` and must cancel between them: an `O(N_occ^2)` block of order-unity
  contributions canceling to zero, including exactly degenerate pairs sitting
  at `Delta = 0` — precisely the slowly decaying low-frequency content a
  damped-time rule resolves worst. The quadrature error is committed per
  term; the analytic parts cancel, the discretization errors do not.
- **Form A.** `f_a(1-f_b)` is *pointwise zero* on the occupied–occupied and
  empty–empty blocks: those pairs never enter either term. The only residual
  cancellation is the partial-times-partial Fermi shell — both weights
  fractional, each product `<= 1/4` for `f` in `[0,1]` — canceling to the net
  `f_a - f_b`. The block is small in pair count (the shell, not the occupied
  manifold) and bounded in weight. In the insulating limit (`f` exactly 0/1)
  form A has **zero** cancellation anywhere, while form B still cancels the
  entire occupied manifold against itself.

Quantitatively, on a shell pair the surviving fraction of the computed weight
is `|f_a-f_b| / f_a(1-f_b) >= 4|f_a-f_b|` (term weights `<= 1/4`), so the
cancellation is bounded both in absolute size (`<= 1/4` per pair, against
order unity in form B) and in depth. On the dynamic samples the kernel factor
is additionally bounded by `1/|z-Delta| <= 1/varpi`, so shell pairs are
regularized by the line height and the plan's four decades of `varpi`
(section 2) stay comfortably inside float64. That is the quantitative reason
the dynamic rows tolerate form A while the `z->0` row does not — see 1.5.

### 1.3 The CTSP comparison, stated precisely

The shredded-propagator method of Kim, Martyna and Ismail-Beigi
(PRB 101, 035139, §II D) is the nearest published relative of this time-domain
kernel, and it makes the opposite choice — safely, for a structural reason
worth recording exactly. Their gapless static polarizability is evaluated per
energy-window pair `(l,m)` as `D^{lm}E^{lm} - F^{lm}G^{lm}` with

$$
D^{lm}=\sum_{v\in l}f(E_v)\,e^{-\tau\zeta\Delta E_{vl}}\psi^*\psi,\quad
E^{lm}=\sum_{c\in m}e^{-\tau\zeta\Delta E_{cm}}\psi\psi^*,
$$

$$
F^{lm}=\sum_{v\in l}e^{-\tau\zeta\Delta E_{vl}}\psi^*\psi,\quad
G^{lm}=\sum_{c\in m}f(E_c)\,e^{-\tau\zeta\Delta E_{cm}}\psi\psi^* ,
$$

i.e. the occupation factors are carried **asymmetrically** — the valence-side
propagator weighted by `f(E_v)`, the conduction-side one by `f(E_c)`, as a
difference of two products: form B. What protects it is the paper's pair
structure: the `l` windows partition the *valence* energy range and the `m`
windows the *conduction* range, with `E_c >= E_v` for every formed pair, so
every window pair is cross-window by construction. The occupied–occupied
block that makes plain form B ill-conditioned is excluded **structurally by
the windowing**, never formed and never canceled. (Their remaining
`Delta -> 0` hazard — degenerate states at `E_F` inside one window pair — is
handled by noting `J_{cv} = [f(E_v)-f(E_c)]/(E_c-E_v) -> -f'(mu)` and
manually scissoring that window pair's gap to `1/beta`.)

Our positive-time scan has no windows. The `f(1-f)` weights themselves do the
exclusion — the same protection, achieved **pointwise** instead of
structurally, with no window machinery, no valence/conduction list split, and
no scissors at the Fermi surface: the degenerate limit is not dodged but
evaluated exactly, by the divided-difference kernel of 1.5. The windowless
form also keeps every ordered pair of the exact sum — including same-side
Fermi-shell pairs, which a disjoint valence-list-times-conduction-list
structure cannot form — with the exact net weight `f_a - f_b`.

### 1.4 Why MP1 overshoot is safe

First-order Methfessel–Paxton is not a positive distribution
(`gw.efermi.mp1_occupations`, BerkeleyGW width semantics
`x = (E-mu)/(2*broadening)`): `f` overshoots `[0,1]` in side lobes, so some
`f(1-f)` products fall slightly outside `[0,1/4]` and some branch weights
leave `[0,1]`. This is harmless *because* the weights only ever enter
linearly — `build_G_tau` multiplies the phase by `band_weight` and nothing
downstream clips, squares, or square-roots a weight. A scheme that factors
weights into square roots (e.g. `sqrt(f) G sqrt(f)` symmetrizations) is where
overshoot becomes fatal; this architecture has no such factorization, and the
support rule of 1.1 deliberately keeps overshoot bands
(`w_isdf.occupation_support_bandwidth` includes them at the support edge).
`tests/test_efermi.py` pins overshoot preservation through the
`OccupationState` constructor.

### 1.5 The `z->0` row has no cancellation at all

At `z=0` the shell block is where form-A cancellation would be numerically
worst: `|z-Delta|` no longer bounds the kernel, and the degenerate limit

$$
\frac{f(E)-f(E')}{E'-E}\;\longrightarrow\;-\frac{df}{dE}
$$

is finite but reached as `0/0`. The metal plan therefore routes exactly that
row — and only that row — through the exact divided-difference kernel
(section 2.3 for the measured decision): `w_isdf._static_fractional_pair_scan`
forms the **net** `(f_a-f_b)/(E_a-E_b)` per ordered pair tile *before* any
quadrature exists, with the analytic `-df/dE` midpoint limit on pairs closer
than floating-point energy resolution. There are no two terms to cancel; the
kernel is cancellation-free by construction, and it is exact (finite-band),
not a finite-`eta` contour in disguise. The split is: **form A for every
dynamic sample, divided difference for the static row** — each form used
exactly where its conditioning is best.

## 2. Frequency sampling: the metal plan, argued from measurement

### 2.1 The shifted-origin double-parallel plan

The metal grid is the insulating double-parallel protocol of
[Multipole frequency integration](THEORY_mpa_implementation.md) §3 with one
substitution: the near line's first point moves from the exact `z=0` to

$$
z_0^{(1)}=i\,\varpi_1,\qquad \varpi_1=10^{-5}\ \mathrm{Ha}=2\times10^{-5}\ \mathrm{Ry},
$$

with the far-line origin still `i*varpi_2`. This is Leon et al.'s published
metals protocol verbatim (PRB 107, 155130 §II C: the shift avoids "numerical
instabilities due to intra-band transitions with energies close to zero"; it
is a stability displacement, not a broadening), including the `alpha`
partition exponent of their Eq. (10): `alpha = 1` (linear) is their Na
choice and ours; `alpha = 2` concentrates samples near zero and was needed
for Al and Cu (Cu with `n_p = 15`), both out of scope here. Owners:
`gw.mpa.sampling.double_parallel_grid` (grid values, material-class
substitution), `gw.mpa.sample_plan.mpa_plan` (per-point analytic character
and route). Note the classification consequence: the metal origin is
`imag`, **not** `static` — the plan's character column is what dispatches it
(2.3), and the sample-grid geometry itself needed nothing new.

### 2.2 Rule bandwidths come from the occupation supports

The damped-line rule for a line at height `varpi` must resolve the fastest
beat of the integrand,
`F_chi = max|Re z| + Delta_max`. For an insulator
`Delta_max = omega_m = quad.x_max`, the maximum valence-to-conduction
transition of the gapped cut. For a metal that quantity is **undefined** —
there is no valence/conduction cut, the two supports overlap, and pair
energies `Delta_ab` run continuously through zero — so the bandwidth is taken
over the *occupation supports*:

$$
\Delta_{\max}
=\max_{b\in\mathrm{supp}(1-f)}E_b-\min_{a\in\mathrm{supp}(f)}E_a ,
$$

implemented as `w_isdf.occupation_support_bandwidth` under the same no-slop
support rule as the kernel (an MP1-overshoot band at a support edge is
included), consumed at `gw.mpa.model._evaluate_samples` in place of
`quad.x_max`. The difference matters in both directions: on a metal the
support bandwidth *exceeds* any gapped-cut estimate (the `f` support reaches
above `mu`, the `1-f` support below it, and the reach grows with the smearing
width), so sizing the rule by `quad.x_max` would under-resolve the fastest
oscillation and alias silently; and because the supports are the *exact*
weighted state set, the rule is never sized for transitions whose weight is
identically zero. Metal line calls pass the rule's **positive nodes only** —
the fractional kernel supplies both Keldysh terms itself (1.1), and applying
the insulating symmetric `±tau` doubling on top of it would double-count.

### 2.3 The origin row is a measurement, not a preference

Both design forks independently expected the contour route to fail at
`varpi_1`; the plan still demanded the number (SYNTHESIS §"measure before
building"). The probe
(`runs/records/metal_mpa_wave1_20260815/I1_origin_probe.md`, rung R1):

| rule request | result |
|---|---|
| `damped_line_rule(2e-5 Ry, 4.0 Ry, rel_tol=1e-6)` — the shifted origin over the 48-band Na spectral width | certifies at **1,007,048 nodes**, 28,865 panels, `t_max = 7.2543e5 Ry^-1` |
| same at `rel_tol = 1e-4` | **655,488 nodes** |
| far line, `damped_line_rule(1.0 Ry, 4.0 Ry, rel_tol=1e-6)` | **31 nodes**, 1 panel, `t_max = 14.5` |

The cost driver is the truncation horizon `t_max ~ log(2/eps)/varpi`, not the
tolerance — which is why loosening by two decades recovers only a third of
the nodes. Against the pre-registered threshold ("certifies at `<= 256`
nodes ⇒ contour permitted for origin rows") the contour route is ~4000×
over, so the finite-`q` divided-difference route was built (commit
`82f81933`): `w_isdf.compute_chi0_static_fractional`, dispatched from
`model._evaluate_samples` for the near line's first point, with the far
pure-imaginary point and every damped point on
`compute_chi0_contour_fractional` (the far line is cheap: `O(1)` Ry heights,
31 nodes). No metal point reaches the insulating `compute_chi0` or
`compute_chi0_contour`, whose positive-gap split is exactly the assumption a
metal breaks; the dispatch census test in `tests/test_chi_contour_kernel.py`
fails if one ever does.

The stored value at the shifted-origin slot is the exact static
`chi0(0)` while the fit reads `sample_z = i*varpi_1`. At finite `q` the
inconsistency is bounded by `(varpi_1/(q v_F))^2` — asserted on the oracle
fixtures at first order (fixture-level `2.556e-5` on a non-TRS random
fixture, `I1_origin_probe.md`); the deck-level quadratic estimate (~`4e-8`
on Na `8^3`) is a run-ladder result and **has not been measured** — do not
cite it as one. At `Gamma` the head overrides the row outright with the
static order-of-limits value (section 4.1), so the bound never applies there.

### 2.4 Alternatives considered and rejected, with their numbers

1. **Shifted origin through the contour.** The 1,007,048-node economics
   above: ~4000× the pre-registered 256-node ceiling, and four orders beyond
   the measured 31-node far line, for a single sample. Rejected on the
   measurement.
2. **A naive `varpi -> 0` limit of the contour.** Strictly worse:
   `t_max = log(2/eps)/varpi` diverges, so the node count is unbounded below
   any fixed tolerance; the probe's `varpi_1` row is already the practical
   image of this divergence at `1e-5` Ha.
3. **The separable resolvent-pair static target.** Approximate the smearing
   function by a rational form `f(E) ≈ sum_j a_j/(E-z_j)`; then

   $$
   \frac{f(E)-f(E')}{E-E'}\;\approx\;-\sum_j\frac{a_j}{(E-z_j)(E'-z_j)} ,
   $$

   two single-band *resolvent* sums per pole `j` — restoring `N^3` scaling
   and retaining the `-df/dE` diagonal naturally. This is the staged scaling
   path behind the same public API, and it is **deliberately not shipped**:
   the service must be certified for the actual smearing family, width,
   interval and absolute error before it may replace the exact kernel, and
   MP1 — sign-changing, non-monotone — cannot borrow a Fermi–Dirac/Matsubara
   pole certificate. At Na scale the exact tiled kernel is affordable for
   its single sample per fit (the TASTE-6 ruling with the per-step byte
   figure is restated in `_static_fractional_pair_scan`'s docstring, which
   also forbids extending the route to dynamic samples).

### 2.5 What "optimal given current theory" means

The plan is the optimum of a three-way trade, not a free choice:

- **Node economics.** The only unaffordable row of the certified-rule family
  is replaced by an exact evaluation that is simultaneously *cheaper* (one
  tile scan vs `10^6` nodes) and *more accurate* (exact finite-band vs
  `rel_tol`-certified quadrature). Everything else keeps the cheap rules.
- **Loewner conditioning.** The origin sample sits four decades below
  `varpi_near` — the Loewner pencil acquires a near-isolated row, which is
  the top-ranked fit-health risk (rung R4 sweeps `N_p` in {6,8,10} against
  the store's `fit_condition`/`condition_max_allowed` guards). Moving the
  origin *further* down (smaller `varpi_1`) buys nothing physically (the
  stored value is already exact static) and worsens both the conditioning
  and the would-be contour cost; moving it *up* violates the published
  protocol's stability rationale and enlarges the `(varpi_1/(q v_F))^2`
  inconsistency. `1e-5` Ha is the published, validated compromise.
- **The shared-grid contract.** The scalar head fit must use the *identical*
  complex grid as the body — `build_mpa_fit` refuses otherwise ("QSGW head
  and MPA body must use the identical stamped z grid") — because head and
  body residues are summed inside one `Sigma` and a mismatched grid would
  fit them to different models of the same screening. So the origin sample
  cannot be tuned per consumer; one plan serves both, which is precisely why
  its single problematic row is solved by changing the *evaluator*, not the
  *geometry*.

Within current theory — a certified damped-line family whose cost is
`t_max ~ log(2/eps)/varpi`, a Loewner fit on `2N_p` shared samples, and an
exact static kernel — no reachable rearrangement improves any leg without
paying more on another. The genuinely better static evaluator (the certified
separable target, 2.4.3) is staged, not skipped.

## 3. The finite-`q` body

Every stored wedge row of every dynamic sample is the fractional contour
kernel; the shifted-origin row is the finite-`q` divided difference. The
finite-`q` static kernel (`w_isdf.compute_chi0_static_fractional`) is the
`Gamma` kernel's sibling through one shared tile scan
(`_static_fractional_pair_scan`): for wedge row `j`, every `b`-side operand
— both centroid wavefunction copies, energies, occupations, surface weights
— is rolled by the caller's flat `k -> k-q_j` map
(`model._metal_kminq_rows`, which asserts the `Gamma` row's map is the
identity). The map is replicated and the `psi` k-axis is replicated on this
mesh, so the gather is rank-local: no collective is added over the `Gamma`
kernel. Cost is the ordered band-pair tile transient
(`nk*(nmu_x/P_x + nmu_y/P_y)*tile^2*16 B` per rank per step); one static
sample per fit rides it, never a dynamic route.

One deliberate asymmetry, stated so no one "fixes" it silently: the two
static kernels use **different diagonal `-df/dE` tables**. The `Gamma` kernel
(`compute_chi0_static_fractional_gamma`) consumes a caller-supplied surface
table — the QSGW path supplies periodic-tetrahedron weights
(`gw.fermi_surface.tetrahedron_delta_weights`), keeping the body diagonal
consistent with the head's `kappa_TF^2` anchor (4.1). The finite-`q` kernel
uses the analytic MP1 derivative internally
(`gw.efermi.mp1_negative_derivative`) and refuses any state whose
`smearing_family != "mp1"` by name (`GATE static_fractional_needs_mp1`); at
finite `q` the true diagonal `a=b` pair sits at `k` vs `k-q` and is only
*accidentally* degenerate, so the analytic midpoint limit is the correct
regularization there, while the `Gamma` diagonal is a genuine Fermi-surface
integral for which the tetrahedron table is the anchor (claim 182 for the
estimator-choice consequences). Off-diagonal pairs use the carried MP1
occupations in both kernels. Gate: the extended
`fractional_chi_gate.py` static row measured `max_rel = 3.720e-16` against
the dense divided-difference oracle over all `q` rows (P=4, JID 56986042).

Dyson, wedge storage, and the bounded column fit are unchanged from the
insulating pipeline and owned by
[Multipole frequency integration](THEORY_mpa_implementation.md) §5–6; the
metallic content of the body enters only through the sample values.

## 4. The two heads, and the order of limits

A metal's long-wavelength limit is direction- *and* order-sensitive: the
static screening limit takes `omega -> 0` **at fixed small q, then**
`q -> 0`, and yields Thomas–Fermi screening; the dynamic head takes
`q -> 0` **first** and stays `omega`-dependent, with the intraband (Drude)
term diverging as `1/omega^2` toward zero frequency. These are different
objects with different values, and the implementation keeps them separate by
construction rather than by numerical accident.

### 4.1 Static head: Thomas–Fermi, Schur-folded

`gw.qsgw_head.head_s_tensor_sharded` deliberately leaves the exact
zero-frequency slot of the dynamic expression untouched (its comment: "the
exact static metallic limit is Thomas-Fermi, not the omega->0 value of the
dynamic Drude expression"); `head_samples_from_s` substitutes the TF model
into the mini-BZ average for that slot,

$$
\kappa_{\mathrm{TF}}^2=\frac{8\pi N(E_F)}{V_{\mathrm{cell}}},
$$

with `N(E_F)` the periodic-tetrahedron `-df/dE` weight sum
(`build_iteration_head_response` computes it from
`fermi_surface.tetrahedron_delta_weights`). The Schur fold then couples the
scalar to the body: `qsgw_head._fold_static_kappa2` folds
`f00 = -kappa_TF^2/(8*pi)` through the *static density wings*
(`static_head_wings_sharded`) and the `Gamma` divided-difference body
(`IterationHeadResponse.static_chi_body_gamma`, built by
`compute_chi0_static_fractional_gamma`) via
`head_correction.fold_cartesian_head_wings_sharded`, reporting

$$
\kappa_{\mathrm{eff}}^2=-8\pi f_{00}^{\mathrm{eff}} ,
$$

which also **overrides the `Gamma` row of the body's origin sample**
(`model._evaluate_samples`, `static_gamma_override`) — the one row where the
`q`-first order of limits of section 2.3 would be wrong. Measured (claim
181, Na 48b): `kappa_TF^2 = 0.708586826 bohr^-2`; the fold correction on
this deck is relative `2.827e-7` (dynamic-wing tool scope), so
`kappa_eff^2 = kappa_TF^2` to well inside the value discrepancy of 4.4.
The **static-wing** fold has still not been measured: it was blocked on a
completed metallic QSGW iteration, one now exists (claim 200), and the
measurement is owed — see `§7.7`.

### 4.2 Dynamic head: Kubo + Drude + wings, on the body's grid

Per QSGW iteration `i`, using the state carried from `i-1`
(`build_iteration_head_response`):

1. **Velocities**: parallel-transport covariant derivative
   (`covariant_structured_delta` on the stored Berry connection) added to
   the DFT velocity, rotated to the current QP basis
   (`rotate_velocity_active_to_qp`); `validate_dft_velocity_identity` is the
   SOC-degeneracy guard in the per-iteration path.
2. **Interband Kubo tensor** `S(z)` (`head_s_tensor_sharded`) with the
   iteration's occupations, in the `chi00 = q.S.q` convention owned by
   [S-tensor convention](s-tensor-convention.md).
3. **Intraband Drude tensor** (`head_drude_tensor_sharded`) from the
   tetrahedron Fermi-surface weights, entering as `D/z^2` at `z = omega +
   i*eta` — with the exact-zero slot excluded (4.1).
4. **ISDF wings and Schur fold**: independent complex left/right centroid
   wings `Y`, `Z` (`head_wings_sharded`);
   `S_eff = S_direct + Y W_body,Gamma Z / V_cell`
   (`fold_cartesian_head_wings_sharded`), finalized per Dyson slab while the
   total-`W` body slab is resident (`model._solve_wc` →
   `finalize_iteration_head_sample`), so only the replicated `3x3` survives.
5. **Mini-BZ average and fit**: `head_samples_from_s` → scalar
   `Wc_head(z) = W_head(z) - v_head` fit with the same Loewner policy and
   guards as the body on the **identical** `z` grid
   (`model._fit_head_samples`); the head fit store carries a model string
   from `_HEAD_FIT_MODELS`
   (`dft_direct_loewner | qsgw_direct_loewner | qsgw_schur_loewner`, plus the
   insulating `fixed_dft_gn`) and the occupation stamps of 5.4.

### 4.3 What this replaces

Leon et al. anchor their `q -> 0` intraband limit with (1) a phenomenological
Drude pole `Y_D(omega) = omega_D^2/(omega(omega+i*gamma))` whose
`omega_D` is an input and `gamma` a free parameter (~0.1 eV), or (2) the
"constant approximation": `Y(q=0) ≈ Y(q_min)`, the nearest stored neighbor
(PRB 107, 155130 §II D, §III C). Both — and the whole
first/second-neighbor W-av reconstruction stack of the earlier design
discussions (Sesti et al., arXiv:2508.06930) — are owner-excluded here: the
ab-initio chain of 4.1–4.2 supplies the same limits with no free parameter
and no neighbor stencil, and strictly supersedes them at `q=0`. The neighbor
machinery remains in-tree but unwired.

### 4.4 Measured accuracy, and the `N(E_F)` caveat

**Dynamic (claim 180, R2 PASS, Na 48b band-matched at 46 bands):** the
`-Im eps^-1_00` loss peak lands at 5.9862 eV (LORRAX) vs 6.0801 eV (BGW
full-frequency), `delta = -93.9 meV` — improving on the accepted 24-band
precedent (−156.1 meV); dynamic-window RMS `|chi00_L - chi00_B| = 2.043e-4`
(band-converged by 23 bands); the tetrahedron Drude `omega_p = 6.0892 eV` is
band-count independent to `1.7e-6`; and `mu` from the fixed-N MP1 solve
agrees with QE's SCF `E_F` to `6.2e-7 eV` (the QE↔LORRAX smearing-consistency
risk closed at this deck). Below ~2 eV the comparison is *definitionally*
void: BGW evaluates a finite `q0` while LORRAX evaluates the exact `q -> 0`
tensor, different quantities with different `omega -> 0` values.

**Static (claims 181/182):** LORRAX's `kappa^2` is **+12.8%** above what
BGW's own static head implies at the same cell (0.7086 vs 0.6281 bohr^-2
Lindhard-extrapolated; free-electron analytic 0.6216). Adjudicated (claim
182): this is `N(E_F)` **estimator spread at 8×8×8**, not a defect — five
estimators of the same eigenvalue table at the same `mu` disagree by ±40%:

| estimator | `N(E_F)` (states/Ry/cell) |
|---|---|
| MP1 smeared delta (deck smearing) | 10.418 |
| MP0/Gaussian smeared delta | 8.825 |
| linear tetrahedron (**the anchor**) | 7.177 |
| BGW finite-q divided difference | 6.361 |
| free-electron analytic `4k_F/pi` | 6.296 |

A `sigma = 0.27 eV` smeared single `3s` crossing sampled at 29 IBZ points is
an intrinsically noisy DOS estimator. The tetrahedron anchor **stays** (the
most stable discrete estimator; "smearing consistency" via MP1 `-df/dE`
would move the anchor to +65%). Consequence, to be restated by any claim
that inherits it: the static head anchor carries `O(10%)`-of-head-channel
uncertainty on **absolute** QP energies at this mesh — an accuracy cap, not
a stability risk (sign and scale of `kappa^2` are right, and band
*differences* largely cancel it). The convergence path is a denser-NSCF
top-level run on which the tetrahedron and divided-difference estimators
must approach each other, optionally with Blöchl corrections — post-landing
work.

## 5. Sigma with finite occupations

### 5.1 Body branches: occupation becomes weight, energy stays signed

The four causal branches of the MPA body `Sigma_c`
([Multipole frequency integration](THEORY_mpa_implementation.md) §7) survive
a metal unchanged in topology; what changes is membership. With an
`OccupationState` (section 6), `gw.mpa.sigma._branches` replaces the
insulating `occ > 0.5` masks by the exact supports and weights — the
occupied branch sums every band with `f != 0` at weight `f`, the empty
branch every band with `f != 1` at weight `1-f`, nothing clipped — and
energies stay signed against the state's `mu_ry` (a caller `efermi_ry`
inconsistent with it refuses: one chemical potential per iteration).
`ppm_windows._SigmaBranch` carries `band_weight: Array | None`; `None` is
the incumbent bool-mask semantics, bit-exact, and the PPM path is untouched.
The executor folds support mask × fractional weight into one float selector
and dispatches it onto the **same** `build_G_tau(band_weight=)` seam the chi
kernel uses (`sigma._integrate_sigma_batches`) — one Green-function builder,
no metal copy. Structurally this matches the head-residue form of 5.3 summed
over `(m,q)` with `|M|^2` weights: per-state `f_m` and `1-f_m` on the two
branch Green functions, no new contraction.

### 5.2 The Fermi window split, and its motivating number

Fractional weights give the crossing branches a negative-`E_A` shell a few
smearing widths wide, so a nominally sign-definite deep-pole rectangle can
reach zero — the case the insulating planner refuses, and the case that,
un-split, silently mis-evaluates **exactly the Fermi-surface states a metal
run is for**. The measured size of that hazard is the number that made the
machinery worth building: on the Fermi-shell fixture of
`tests/test_sigma_fermi_split.py`, the `occ > 0.5` mask semantics disagree
with the exact fractional reference by relative error **1.14 / 0.72 / 0.22
at `omega` = 0 / 0.25 / 0.5 Ry** (commit `c560065c`; the suite permanently
pins a `> 5e-2` floor for the mask arm and `< 5e-4` for the weighted split —
a failing-first certificate, recorded against the pre-change planner). The
fix rides the existing crossing machinery rather than a new rule family:
`sigma_windows._geometry` deepens the shallow/deep pole edge by the crossing
branches' negative-`E_A` excursion, keeping every deep slab rectangle
sign-definite and routing the straddle into the crossing core. A
non-negative support contributes zero excursion, so insulating geometry is
bit-identical.

### 5.3 Head residues

The head `Sigma` injection already carries the occupation split
(`head_correction.compute_complex_pole_head_sigma_diag`):

$$
\Sigma^{\mathrm{head}}_{nk}(\omega)=\frac{1}{V_{\mathrm{cell}}N_k}\sum_p R_p
\left[\frac{f_{nk}}{\delta_{nk}+\Omega_p}
+\frac{1-f_{nk}}{\delta_{nk}-\Omega_p}\right],
\qquad \delta_{nk}=\omega-(E_{nk}-E_F),
$$

with per-`(k,band)` occupations accepted precisely so a window straddling
`E_F` stays valid. `sigma_dispatch` asserts at dispatch time that the head
fit's stamped occupations equal the body's live state
(`gw.mpa.sigma.assert_head_body_occupation_match`) — two occupation states
cannot leak into one iteration.

### 5.3b Broadening: MPA passes eta through, GN-PPM floors it

The two dynamic ansatzes do NOT share an effective broadening, and any
cross-ansatz comparison that ignores this is confounded (registered in the
sandbox KNOWN_LORRAX_ISSUES, 2026-08-15):

- **MPA** passes the deck's `regularization_ev` straight into the `Sigma`
  denominators — on the sodium campaign decks, eta = 0.25 eV, and every
  R5/R6 claim quotes it explicitly.
- **GN-PPM** silently floors its xi at

      xi_floor = 2*omega_max / (24 - 2*edge)

  which on this window class evaluates to 1.4286 eV — 5.7x the MPA value on
  the same deck. The floor is printed at runtime but was documented nowhere
  before this paragraph.

Consequence: an MPA-vs-GN-PPM energy difference mixes physics with a 5.7x
broadening mismatch. No such comparison may be claimed without either
equalizing xi/eta or carrying this caveat verbatim.

### 5.4 Exchange, SX, Hartree: `diag(f)` end to end

`gw.cohsex_sigma.build_Gij(occupation_state=...)` builds `G_ij = diag(f)`
over the `Sigma` window, with a metallized window-coverage guard (refuses a
window electron count off the state's target by more than `1e-8` — the same
`V_H`-silently-small hazard the integer `nb_sigma >= nelec` guard prevents),
and step occupations reproduce the integer projector bit for bit (the
insulator/metal unification done where it is exact; asserted in
`tests/test_mpa_sigma.py`). **Wiring status, landed at `bfa402a0`:** the
carried `OccupationState` reaches all three call sites through
`compute_cohsex_sigma`, `compute_v_h_sigma_x` and
`compute_ppm_sigma_pipeline` -> `compute_sigma_c_ppm_omega_grid` ->
`_compute_invalid_static_sigma`, with `None` the default at every link
(insulating behaviour byte-identical; 8 previously-failing threading cells
green, 133-passed focused suite on JID 57005734). Metallic
`Sigma_x`/SX/`V_H` take `diag(f)`.

### 5.5 Provenance stamps

Fit and head stores carry the occupation stamp group
(`file_io.mpa_store._OCC_STAMP_ORDER`:
`occ_hash, mu_ry, smearing_family, smearing_width_ry, occ_nelec`), written
by `stamp_occupation_provenance` from the live state. The stamps gate
**reuse**, not the same-run write path (a same-run assert would be
circular): `assert_occupation_stamps` refuses an unstamped store under
metallic reuse and names the mismatched field otherwise;
`read_head_fit_collective` returns the head stamps for the dispatch
cross-check of 5.3. `occ_hash` is bound to the bytes of `f_kn` in the
state's `__post_init__`, so a stamp can never describe a table it was not
computed from.

## 6. The QSGW occupation cycle

### 6.1 One state per map call, solved at entry

Every call of the QSGW map solves its own occupation state, **at entry**,
from the spectrum of the `H` it was handed: `gw.sc_iteration.gw_iteration_map`
rotates into the QP basis and immediately calls
`_solve_head_occupations(inputs, wfns_qp.enk)`, and that one state is what
this call's chi, `q->0` head and `Sigma` consume (commit `178f62b8`). There
is exactly one MP1 solve per map call and no carry: `SCState.occupation_state`
still holds the previous call's state, but only as the `|d mu|` drift
diagnostic, and nothing physical reads it. Section 7.2 owns why the rule
changed from the earlier end-of-iteration solve, and what it bought.

The solved state is one frozen `gw.efermi.OccupationState`
(`f_kn` unclipped, `mu_ry`, `smearing_family`, `smearing_width_ry`,
`n_electrons`, derived `occ_hash`), built from the **QP eigenvalues** by
`OccupationState.solve_mp1` — a safeguarded fixed-N bisection
(`solve_mp1_occupations`; `state_capacity` = 1 spinor / 2 scalar) whose
fixed-N invariant is asserted by both constructors
(`efermi.assert_fixed_n`, `|N_realized - N_target| <= 1e-10`). The same
object reaches screening (`gw.screening.compute_screening_model` →
`build_mpa_fit`), the head response, and `Sigma`
(`gw.sigma_dispatch.compute_sigma_xc`, which takes `sigma_efermi_ry` from
`state.mu_ry`, never the loader's midgap `efermi`), all gated metal-only so
insulating decks pass `None` and stay bit-exact. Per iteration the driver
logs `|d mu|` drift and the `occ_hash`; tetrahedron surface weights are
rebuilt beside the state as derived data. Startup consistency:
`efermi.assert_wfn_occupation_consistency` compares the solved state against
the WFN's own occupation table (`N_e` to `1e-8`, `max|df|` to `1e-6`) —
measured on the Na deck, `mu` agrees with QE's SCF `E_F` to `6.2e-7 eV`
(claim 180).

### 6.2 The convergence residual: map output vs map input

The R6 invariant is the residual of the QSGW map `F`, i.e.
`eigvalsh(F(H_in)) - spectrum(H_in)` **per map call** — not the difference
of accepted iterates. The two coincide only at `sc_mixing = 1.0`, where the
accepted state *is* the previous call's candidate, so
`candidate_i - candidate_{i-1}` is exactly map-output-vs-map-input. At any
`mixing != 1` the accepted iterate is a blend, the identity fails, and
accepted-iterate differences are **soft** — they under-report the true map
residual by up to the mixing factor and can show spurious convergence while
the map still moves. The ladder's analysis script
(`runs/Na/02_soc48b_qsgw_mpa/01_lorrax_metal_mpa/r6_residual.py`) refuses to
run unless the caller confirms `mixing = 1` for exactly this reason, and
deliberately does not reuse the run's printed RMS (a different norm over a
different band set). It no longer defaults to *all* bands either: it parses
the protected set from the snapshot's own `active_scissored_bands_1based`
comment — `{9,10}` on this deck — and reproduces the driver's criterion to
every printed digit (claim 198; the audit's finding A3 was that an
all-bands default re-counts the refit scissor diagonals). The driver now
applies this same output-vs-input rule itself, on both the linear and the
rCROP path; §7.3 owns the implemented predicate and the band set it runs
over. Alongside it, R6 asserts per iteration: electron count conserved by
the fixed-N solve (`<= 1e-8`), `mu` drift decreasing, fit certifications
green, head poles stable.

### 6.3 The smearing width: one owner, BGW semantics

Settled and landed at `bfa402a0` (evidence:
`runs/records/metal_mpa_wave2_20260815/WIDTH_KEY_EVIDENCE.md` in the
sandbox): `occ_smearing_width_ry` carries BerkeleyGW `occ_broadening`
semantics in Ry — the MP1 argument is `(E-mu)/(2*width)`, so the value is
**half the QE `degauss`** (BGW itself uses `degauss/2`,
`input_utils.f90:380`, reproducing QE occupations to `7.1e-12`). It is the
single width every MP1 solve consumes (`LorraxConfig.occ_broadening_ry`);
`occ_broadening` remains the zero/non-zero dial; a deck whose two keys
disagree beyond `1e-4` relative is refused at parse with the conversion
printed. Claims 0180/0181 ran at this same physical width (0.01 Ry) and
need no restatement.

### 6.4 Gate status

**The gate is LIFTED as of 2026-08-15** (commit `9c9b23dc` on
`integ/metal-mpa-qsgw-2026-08-15`, pushed; *not* an ancestor of
`origin/main`). `gw_config.UNIMPLEMENTED_MODES` is now empty:
`ComputeMode.MPA` passes the driver-entry check and the `Sigma` seam. The
row it used to hold was never a metal row — it refused metals and insulators
alike — so do not describe what was removed as a metal gate.

**What the lift asserts, and what it does not.** Removing the row was the
declared landing gesture for *one real metallic run traversing
chi/W/head/`Sigma`/QSGW end to end*, and that is what it was taken on: R6
ran three self-consistent iterations to `rc = 0` on the Na deck. **It was a
statement about parseability, not convergence** — that run did not converge
(`max|dE|` 5.433 eV against a 1e-4 eV criterion, `mu` moving 1.849 eV
between iterations), and it predates the three defects §7 records: the
zero-sample scissor (`bf57701b`), the midgap-anchored partition window
(`90b8275d`) and the mixed omega reference in the finalizer (`59d7ea20`).
**Convergence is now separately measured** — claims 198 and 200, `§7.4`:
undamped diverges, damped reaches 0.833 meV in 7 map calls. Read the two
statements as what they are, a parse gate and a physics result, not as one
another. The site-level refusals are the safety now and are untouched:
`mpa_metal_needs_occupations`, the deck-key cross-validation, and the
occupation-stamp assert.

The route it ran on is `sc_head_update = dft_velocity`, not parallel
transport. The PT velocity gate still fails on the Na deck for a reason
independent of the transport itself (claim 183 — the gate's diagonal
compares an FFT derivative of band-sorted `E_n(k)` against the exact
velocity and is not smooth at band crossings; measured `max_abs = 3.169`
against `atol = 5e-4`, already `3.9e-2` on the never-crossing semicore
bands, improved to 1.796 by the transported frame of claim 195 but still
refusing). Rungs R2 (claim 180) and R3 (claims 181/182) stand; R4's fit half
is claim 196 and its `eqp` half still awaits its rerun on the
omega-reference-fixed tip; R6 landed as claims 198/200 (`§7.4`), and R5 —
the one-shot metallic `Sigma` against the BerkeleyGW full-frequency
reference — has not. Velocities everywhere on this route are DFT p-matrix
elements, with the covariant-derivative upgrade parked on claim 183.

## 7. Self-consistency: the metallic QSGW map and its convergence

Sections 1–6 build one evaluation of the screening and the self-energy.
This section is about the *map* they compose into, and about the metallic
self-consistency results that exist: sodium diverges under the plain
undamped iteration, converges under damping (claims 198, 200), and
converges under the accelerator once the occupations are solved at entry
(claim 201) — which is the production route.

### 7.1 What one map call rebuilds

The QSGW map is

$$
F(H)=H_{\mathrm{KIH}}+\Delta H\!\left[\,H,\ \mathrm{occ}(H)\,\right],
$$

with `H_KIH` the immutable kinetic-plus-ion operator: `sc_iteration`
assembles `H_out = inputs.kin_ion_dft + delta_h_dft`, so the *input* `H`
never enters additively (claim 197(k)). On a metal, one call rebuilds
everything downstream of the spectrum:

1. **Occupations**, entry-solved from the spectrum of this call's `H`
   (6.1).
2. **Screening.** The MPA sample plan is re-evaluated and refit — chi
   samples, the Dyson solve, the bounded column Loewner fit — because the
   orbitals and transition energies moved; the per-iteration stores are
   written fresh under `sc_%04d` names and only one complete pair is kept
   on disk (`retain_iteration_artifacts`). Nothing model-shaped lives in
   `SCState`.
3. **The `q->0` head**, rebuilt by `build_iteration_head_response` on this
   call's velocities (rotated into the current QP basis by the same `U`)
   and this call's occupations, then mini-BZ averaged and refit with the
   scalar Loewner policy on the body's identical `z` grid (4.2). The
   converged sodium run took the `sc_head_update = dft_velocity` route
   (6.4), so those velocities are DFT `p`-matrix elements.
4. **`Sigma`**, with `sigma_efermi_ry` from the state's `mu_ry` (6.1), the
   band partition, the scissor of 7.5, and the Hermitian QSGW operator
   build.

**The one-omega-reference rule.** `Sigma_c(omega)`'s grid is *relative*, and
on a metal it is measured from the fixed-N `mu`; on the sodium deck that is
`2.7934 eV` away from the loader's midgap/VBM convention, so any consumer
sampling it against the wrong reference reads the cube in the wrong place.
Four sites had to agree, and each was fixed where it lived:

| site | rule | commit |
|---|---|---|
| grid build (`sigma_dispatch`) | one `mu` per iteration | (incumbent) |
| finalize interpolation (`dynamic_sigma.finalize_dynamic_sigma`, `eval_sigma_c_at_dft_energies`) | `efermi_ry` argument; `None` = the incumbent `wfn.efermi`, PPM bit-identical | `59d7ea20` |
| band partition window (`run_sc_driver`) | metallic decks anchor `[omega_min, omega_max]` at the fixed-N MP1 `mu` of the full-BZ table with uniform weights | `90b8275d`, `6fe3fcb8` |
| the from-disk assembler (`eqp_bgw.make_eqp_bgw`) | `sigma_mnk.h5` stamps `omega_reference_ev` **and its provenance** on `omega_ev` itself (`write_sigma_omega_h5`, read back by `read_omega_reference`); an unstamped *metallic* file is refused by name, an unstamped insulating one still falls back to midgap bit-identically | `cd5b0aa4` |

The two measured symptoms of getting this wrong are worth keeping, because
they look like physics: near-`E_F` QP corrections of `+2.0..+2.9 eV`
(mean `+2.42`, i.e. the 2.79 eV reference error) from the finalizer, and
`0/48` bands in range from the partition — which scissored every band with
a fit that had no samples and returned `diag(0)`, printing a 24.677 eV
"inter-iteration RMS" that is just `RMS|E_DFT|` against zeros (commits
`59d7ea20`, `90b8275d`; the zero-diagonal mechanism is 7.5). Claim 200's
converged run stamps `1.507789 eV (fixed-N mu)`, not the 4.44 eV midgap of
the same spectrum's VBM/CBM.

### 7.2 The entry-solve rule, and why it is what makes acceleration definable

Before `178f62b8`, the same MP1 solve ran at the *end* of a call and its
result was carried into the *next* one — a one-generation lag (the
metallic-invariant audit's finding A5). Two consequences, one numerical
and one structural:

- **Numerical.** Chi, head and `Sigma` at call `n+1` paired current
  energies with the occupations of the previous spectrum. Measured on the
  same undamped linear deck, the first three map calls give
  `0.871 / 0.514 / 0.316 eV` under the entry solve against
  `0.871 / 0.550 / 0.404 eV` under the end solve — identical at call 1,
  where both rules see the same starting state, and steadier thereafter:
  the entry-solve contraction ratio holds (0.59, 0.61) where the
  end-solve arm's was already climbing (0.63, 0.73). Claim 201 records
  this A/B as the enabling evidence for the rule; the end-solve
  trajectory is claim 198's.
- **Structural, and this is the load-bearing one.** With an end-solve
  carry, `F` depends on the *trajectory* that produced `H`, not on `H`:
  it is exact only along the `sc_mixing = 1` linear path, where the
  accepted iterate is the previous call's output. Solving at entry makes
  the occupations a function of the iterate, `occ = occ(H)`, so `F(H)` is
  a self-map of `H` alone — the contract `_run_rcrop`'s own header always
  stated (*"`gw_iteration_map` reads `state.iteration` and
  `state.H_qp_dft` and nothing else"*). Every evaluation, accepted or
  trial, gets occupations consistent with its own `H` by construction,
  which is exactly the property an acceleration trajectory needs to be
  well-defined.

Fixed points are unchanged: at `H* = F(H*)` the two rules see the same
spectrum. The cost is zero extra diagonalizations (the entry `eigh`
already exists) and one *fewer* solve per call. Insulating decks
(`occ_broadening = 0`) take the `None` branch and are bit-identical.

### 7.3 The criterion is `max|dE|` over the non-scissored bands

`sc_iteration.protected_band_convergence` is the stop rule, and it is
deliberately the crudest defensible one:

$$
\max_{(n,k)\ \in\ \mathcal{P}}
\left|E^{\mathrm{out}}_{nk}-E^{\mathrm{in}}_{nk}\right| < \texttt{sc\_tol\_ev},
\qquad
\mathcal{P}=\{\texttt{protected\_mask}\ \cup\ \texttt{in\_range\_mask}\},
$$

evaluated on **one map call's output against that same call's input**, on
non-trial calls only. Four choices in it, each with a reason:

- **`L-infinity`, not RMS.** "Every band moved less than the cutoff" is a
  statement about the worst band. The RMS figures are printed as
  diagnostics and are never compared to the cutoff; the run log says which
  number is the criterion, because the old prose calling an operator RMS
  and a band RMS "approximately equal" is what hid the defect below for as
  long as it did.
- **Output-vs-input, not iterate-vs-iterate.** The two coincide only at
  `sc_mixing = 1`; at any damping the accepted iterate is a blend and its
  differences are *soft* — a loop can then "converge" by damping rather
  than by solving (6.2 owns this argument; `_run_linear_mixing` and the
  rCROP path apply the identical predicate to the unmixed map output).
- **The union of the two masks.** `apply_band_partition` substitutes a
  scissor diagonal exactly where `in_range_mask` is false, so an in-range
  non-protected band keeps its own `Sigma`-derived diagonal and is a
  genuine degree of freedom. Scissored bands are excluded because their
  energies are `alpha*E_DFT + beta` with the coefficients refit each call
  *from* the in-range corrections — including them re-counts in-range drift
  through the fit. Zero non-scissored bands, or a mask length that
  disagrees with the active window, **refuse** rather than answer.
- **No second stopping rule.** `rcrop_nojit` is called with `tol = 0.0`:
  the accelerator accelerates and the caller decides, on the exact
  eigenvalue test that is free because the map already diagonalizes `H`.

**What it replaced, and the autopsy that closes (commit `4a6ef831`).** The
previous rule stopped on an `L2` norm of the `H` residual with the
per-band tolerance converted by `sqrt(n_elem)`. For Hermitian `H`, Weyl
gives `|dlambda_i| <= ||dH||_2 <= ||dH||_F = ||f||_2`, so the only sound
conversion is `tol_resid = tol_ry` with no factor at all. On Si `4x4x4`
SYM/SOC at `P=4` the carry is `(64, 24, 24)`, `sqrt(n_elem) = 192`; a 2 meV
request became a `2.8223e-02 Ry` threshold, rCROP returned converged at
`||f||_2 = 2.3618e-02 Ry`, and `max|dE|` over the non-scissored bands at
that very call was `0.120477 eV` — **60.2x** the cutoff. With the measured
Weyl slack `||dH||_F / max|dlambda| = 2.67`, the predicted looseness is
`192 / 2.67 = 72x` against `60.2x` observed; nothing is unaccounted for.
The same deck under plain linear mixing landed within `0.313 meV` of the
rCROP answer, so this was a stopping-rule defect and not a physics one.

**The floors are close, and where exactly they sit is not settled.** On
the sodium deck the criterion is an `L-infinity` over **two** bands — the
Fermi-crossing Kramers pair, the only non-scissored bands the frozen
window leaves (7.5) — so the tolerance is doing very little averaging, and
two banked numbers bracket what it can mean. Claims 198 and 200 read the
1 meV cutoff as sitting *at* the `omega`-grid half-step floor of
`1.42 meV` on band energies, i.e. at the edge of what the grid resolves.
Claim 201 reads the same pair differently and says why: that `1.42 meV`
was measured on a `0.5 eV` `omega` step, while this deck runs `0.25 eV`
(41 points over `[-5,+5] eV`), so the deck's own floor is *smaller and
unmeasured* — the rerun on the fixed tip is owed. Above the tolerance
there is a second, coarser floor: the Si-lineage `N_p` sensitivity of
`15 meV` between 8 and 10 poles, whose sodium rerun is also owed (claim
201). Do not average the two readings; state which floor a per-band claim
is being made against, and note that neither has been measured on this
deck at its own settings. The companion analysis script defaults to the
same band set, parsed from the snapshot's own
`active_scissored_bands_1based` comment rather than from a band window
written down somewhere else (claim 198, audit finding A3).

### 7.4 The measured characterization: damping is the whole difference

Same deck, same tree, same budget, one key changed (claims 198 and 200,
jobid 57038615, tip `81c99c95`; `P=4` on one node).

| arm | `max\|dE\|` per map call (eV) | verdict |
|---|---|---|
| undamped, `sc_mixing = 1.0` | 0.871369 0.549794 0.404293 0.290045 **\|** 0.344554 0.469577 0.788125 1.356513 | contracts four calls, then **diverges** |
| damped, `sc_mixing = 0.5` | 0.871369 0.164167 0.043921 0.014123 0.005356 0.002062 **0.000833** | **CONVERGED** in 7 calls |

The undamped arm turns and grows at ratios `1.19 / 1.36 / 1.68 / 1.72`,
ending *above* where it started; at the turn `RMS_all(48)` jumps 5x
(0.229 → 1.075), the argmax relocates from `k=292` to `k=355`, and `mu`
oscillates with growing amplitude (`|dmu|` 4.2e-2 → 1.0e-1 eV). That
trajectory reproduces **bit-identically on all eight points** at commit
`bf57701b`, i.e. before that session's HDF5-lifecycle work, which is what
licenses calling the divergence physics rather than I/O (claim 198).

The damped arm contracts geometrically with ratios
`0.188 / 0.268 / 0.322 / 0.379 / 0.385` — `rho` rising toward ~0.39 rather
than falling, the signature of a linear rate, not of superlinear
convergence — and stops **on the criterion at 7 calls of a cap of 8**, not
on the cap (claim 200). `mu` converges with it: `1.51693 → 1.51114 →
1.50877 → 1.50779 eV` with `|dmu|` falling
`1.41e-2 → 5.80e-3 → 2.37e-3 → 9.79e-4 eV`, against the undamped arm's
*growing* `|dmu|` — the sloshing is damped, not masked. The converged
protected pair (claim 200): `eqp0.dat` bands 9/10 span `+1.1332` to
`+6.9125 eV` with mean QP correction `+1.8831 eV`; `eqp1.dat` `+1.0072` to
`+6.6780 eV`, mean `+1.1701 eV`; 1392 rows each with **zero** zero-valued
rows (claim 196 §3's all-zero trap is absent on the multi-iteration path),
and bands 9 and 10 agree to `~5e-5 eV` — the Kramers degeneracy this
`nspinor = 2` deck should show, an unrequested internal check that passes.

What this does **not** say: that the amplifying mechanism is identified.
The audit's A5 (occupation lag) met its own deferral condition verbatim,
but A4 — band-index pairing across iterates under the `eigvalsh` sort, of
which the `k=292 -> 355` argmax relocation is a candidate symptom —
remains unseparated from it (claim 200). Damping fixing the amplitude is
not a diagnosis, and both arms in the table ran under the *end*-solve rule
that 7.2 replaced. Damped linear is therefore how metallic sodium was
*first* converged, not the recommended way to converge one: the production
ruling is entry-solve rCROP, and 7.6 is where that is argued and measured.

### 7.5 Two anchors that decide which bands are even being converged

**The scissor's no-information law (commit `bf57701b`).** The
out-of-range scissor fits `E_QP = alpha*E_DFT + beta` per class by weighted
least squares. An empty class must return the **identity**
(`alpha = 1, beta = 0`, scissored bands keep `E_DFT`) and a single sample a
**rigid shift** (`alpha = 1`, `beta` the one `dE`); the previous
`(0, 0)` return extrapolated every scissored band to `E_QP = 0` exactly.
A metal is where this fires, and sodium is the worst case of it: the only
protected bands *cross* `E_F`, so neither the valence nor the conduction
class had clean samples, all 46 scissored diagonals became `0.0`,
`eigvalsh`'s ascending sort interleaved 46 zeros with the two real
eigenvalues, and every downstream observable inherited the wreckage — the
migrating populated-band snapshots, a `max|dE|` that was the VBM to six
decimals (a difference against a zero cell), the `mu` walk to `-0.20 eV`
(MP1 solved on a three-quarters-zero table), and the iteration-2 MPA
conditioning trip (chi on that spectrum). The law is not metal-specific;
the crossing protected set is what makes a metal reach it.

**The frozen window (claim 197(k), BAND_SHIFT_ANALYSIS §7).** The
`Sigma` `omega` window is anchored on `mu`
(`omega_min_ev = config value + efermi_ev`) and the `in_range`/protected
masks are computed **once**, in `run_sc_driver`, from the **DFT** spectrum
— on a metal from the fixed-N MP1 `mu` of that spectrum (7.1) — and then
frozen for the whole SC run. They do not re-anchor as the QP spectrum
drifts. `protected_band_convergence` is written against an
*iteration-local* `BandPartition` by contract, but today's driver hands it
the same frozen pair every call, so the criterion inherits the freeze:
the set of bands being converged is the set the **DFT** spectrum put in
range.

This is a window sensitivity, not a gauge freedom, and the distinction is
argued from the source in claim 197(k): a uniform shift `H -> H + cI` is a
**true null direction** of `F` (`dF/dc = 0` exactly), because `H_KIH` pins
the absolute level and every channel by which `H` reaches `Sigma` is `mu`-
or difference-referenced — the fixed-N solve (`mu -> mu + c`, `f_kn`
unchanged), chi and `W` through energy differences, the `Sigma_c` argument
and its poles co-shifting, and the scissor's `beta -> beta + c` at fixed
`alpha`. The numerical probe of that statement was descoped from claim 197
and has not been run, so it is analytic. What the freeze *does* expose is
a real drift: on the Si ladder of claim 197 the QP spectrum moves over half
an eV on a window of default half-width 5 eV, and the mean QP correction on
sodium's converged protected pair is `+1.88 eV` (7.4). A deck whose QP
spectrum carries a band across the frozen window edge is the untested case
(audit A4).

### 7.6 rCROP on metals: refused, legalized, and now the production route

**Why it was refused, and it was a state-consistency argument, not a
tuning preference.** Under the end-solve rule the occupation state was a
*sequential* object: exact only along the `sc_mixing = 1` linear
trajectory, where the accepted iterate is the previous call's output.
rCROP mixes only the Hamiltonian; it cannot preserve that state, and its
trial iterates would be evaluated with occupations belonging to some other
`H`. `gw_config` therefore refused `sc_accelerator != linear` whenever
`occ_broadening > 0`, and claim 198 records the refusal as honoured rather
than tuned past — with the honest note that rCROP's soundness evidence at
that point was the **insulating** Si lineage.

**What the entry solve changes.** The refusal is deleted (`178f62b8`), and
a metallic rCROP deck parses, because the reason for it is gone: 7.2's
self-map property makes trial and accepted evaluations equally consistent.
Separately, `4a6ef831` removed rCROP's own early-stop defect (7.3), so a
metallic rCROP arm now stops on the same `L-infinity` criterion as the
linear arm rather than on an `L2` proxy.

**Measured, and it converges without a variant (claim 201).** The
entry-solve rCROP horizon run on the same sodium deck
(`runs/Na/02_soc48b_qsgw_mpa/05_rcrop_ab/`) reaches
`max|dE| = 0.389 meV` in **9 map calls** against the same 1 meV
tolerance, on the trajectory

```
0.871369  0.514006  0.154386  0.106831  0.040433
0.029043  0.002650  0.001527  0.000389   eV
```

which is **not monotone in its ratios** and should not be read as one: the
`0.107 -> 0.040 -> 0.029` plateau is rCROP spending early calls building
its secant space, and the quasi-Newton tail then contracts by up to 11x
per call. Against the damped-linear arm of 7.4 (7 calls to 0.833 meV) the
cost at the 1 meV tolerance is comparable, but there is no hand-chosen
damping parameter — `sc_mixing = 1.0` diverges and `0.5` was picked by
hand — and the accelerating tail is what wins at any tighter tolerance,
which a fixed `~0.3-0.4`-per-call linear contraction cannot follow. Claim
201's ruling, adopted here: **`sc_accelerator = rcrop` with the entry
solve is the production route for metallic QSGW.**

Its iterates differ from 7.4's by design — those arms ran the carried
occupation rule at `bf57701b`, this one the entry solve at `178f62b8` —
so the two are not an accelerator A/B on a fixed map, and nothing here
separates "rCROP beats damping" from "the entry solve improved the map".
The claim's own floor caveats (7.3) bound what may be read off the
converged energies.

### 7.7 What metallic self-consistency does not yet establish

Open, with the reason each is still open:

- **No BerkeleyGW comparison of the converged spectrum.** Claims 198/200
  are self-consistency of the LORRAX loop against its own criterion. The
  accuracy question — including the `O(10%)`-of-head-channel cap that 4.4
  puts on absolute QP energies at this mesh — is untouched by them.
- **Multi-node is broken, and pre-existing.** At 16 ranks on 4 nodes,
  12 of 16 die at `fit_driver.py:457` with *"file is already open for
  write"*, and an A/B at `bf57701b` fails identically (claim 198). Every
  number in 7.4 is 4 ranks on one node.
- **A4 vs A5 unseparated** (7.4).
- **The A1 hazard is reduced, not eliminated — and it owns less than it
  was charged with.** Two core libhdf5 instances are mapped in one process
  (cross-major), and the per-file churn was **measured** at 1027
  cross-library alternations on one file in one iteration against the
  audit's ~25 estimate (claim 198). The landed mitigation is a per-process
  one-owner registry (`file_io/hdf5_owner.py`, `LORRAX_HDF5_ONE_OWNER`), a
  single h5py door in `mpa_store`, and one `SlabIO` held across an
  iteration's pole batches; the sibling-file split that would remove the
  class is specified but not scheduled. The intermittent garbage-offset
  head-fit read that killed one attempt at the damped arm has since been
  root-caused **elsewhere**: an unordered control-operand copy on the
  legacy default CUDA stream, which the XLA stream does not order against,
  fixed at `ef98d47f` (stream-ordered copy) and `15eef55f` (live-context
  handle validation). That write-up's gates are measured but its ledger row
  and its acceptance arm (`07_damped_streamfix/`) were not landed when this
  section was written, so treat the reattribution as pending confirmation
  and the defect itself as still registered in the sandbox
  `KNOWN_LORRAX_ISSUES`.
- **The static-wing Schur fold is unblocked but unmeasured.** 4.1's
  `2.827e-7` fold correction is dynamic-wing tool scope; a completed
  metallic QSGW iteration now exists (claim 200), so the static-wing
  measurement can be made — it has not been.
- **R4's `eqp` half is still owed.** Claim 196 covers the fit half only
  and explicitly invalidates its own `eqp`-derived numbers (they predate
  the omega-reference fix of 7.1); the rerun on the fixed tip has not
  landed.
- **The velocity route is still `dft_velocity`.** The parallel-transport
  gate refuses on this artifact for reasons upstream of the frame
  (claims 183, 195; 6.4), so the converged run's head velocities are DFT
  `p`-matrix elements.

## 8. Claims ledger for this page

| statement in this page | evidence |
|---|---|
| origin contour rule 1,007,048 nodes; far line 31; thresholds and decision | `runs/records/metal_mpa_wave1_20260815/I1_origin_probe.md`, commit `82f81933` |
| finite-q static kernel `max_rel 3.720e-16` vs dense oracle; fixture origin-shift `2.556e-5` | same probe record, P=4 gate JID 56986042 |
| mask-semantics error 1.14/0.72/0.22; permanent `>5e-2` floor | commit `c560065c`, `tests/test_sigma_fermi_split.py` |
| dynamic head vs BGW: −93.9 meV peak, RMS 2.043e-4, `omega_p` 6.0892 eV, `mu` to 6.2e-7 eV | claim 180 (JID 57005734) |
| `kappa_TF^2 = 0.7086 bohr^-2`, +12.8% vs BGW, fold 2.8e-7 | claim 181 |
| five-estimator `N(E_F)` spread; tetrahedron anchor; `O(10%)` absolute-energy cap | claim 182 |
| velocity-gate failure blocking R4–R6; gate not lifted | claim 183, commit `a5b1002b` |
| transported-frame re-gate `3.169 -> 1.796`, still refusing | claim 195, commit `1bae7d73` |
| fit conditioning at the shifted origin; no `n_p` census pathology on Na | claim 196 |
| the 2.7934 eV omega-reference error and its four sites | commits `59d7ea20`, `90b8275d`, `6fe3fcb8`, `cd5b0aa4` |
| scissor identity law; the all-zero-diagonal wreckage it fixed | commit `bf57701b` |
| `max\|dE\|` criterion; the `sqrt(n_elem)` autopsy (60.2x vs 72x predicted) | commit `4a6ef831` |
| entry-solved occupations; the metallic rCROP refusal deleted | commit `178f62b8` |
| entry-solve vs end-solve first three calls (0.316 vs 0.404 eV); rCROP 0.389 meV in 9 calls; the production ruling; the two tolerance floors | claim 201 |
| undamped divergence trajectory; bit-identical pre-change A/B; multi-node break; 1027 alternations | claim 198 |
| damped convergence 0.833 meV in 7 calls; `mu` and `\|dmu\|`; converged QP pair; 1.42 meV grid floor | claim 200 |
| uniform shift a null direction of `F`; the frozen window and mask caveat | claim 197(k) |

Anything in this page not in that table is either code structure (verify by
symbol — every named symbol exists at `941db3a7`) or published literature
(references below).

## References

- Leon, Ferretti, Varsano, Molinari, Cardoso, *Efficient full frequency GW
  for metals using a multipole approach for the dielectric screening*,
  Phys. Rev. B 107, 155130 (2023). https://arxiv.org/abs/2301.02282
- Leon et al., *Frequency dependence in GW made simple using a multipole
  approximation*, Phys. Rev. B 104, 115157 (2021).
  https://link.aps.org/doi/10.1103/PhysRevB.104.115157
- Kim, Martyna, Ismail-Beigi, *Complex time, shredded propagator method for
  large-scale GW calculations*, Phys. Rev. B 101, 035139 (2020).
  https://doi.org/10.1103/PhysRevB.101.035139
- Sesti et al., *Efficient GW calculations for metals from an accurate ab
  initio polarizability*, arXiv:2508.06930 (2025).
  https://arxiv.org/abs/2508.06930
- Rojas, Godby, Needs, *Space-Time Method for Ab Initio Calculations of
  Self-Energies and Dielectric Response Functions of Solids*,
  Phys. Rev. Lett. 74, 1827 (1995).
  https://doi.org/10.1103/PhysRevLett.74.1827
