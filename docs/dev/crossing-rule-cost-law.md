# The crossing-rule cost law, and the ω-clustered core decomposition

**Status: design + diagnosis record, 2026-08-16.**  Companion theory text
lands in `docs/theory/metallic-mpa-screening.md` (quadrature section) and
`docs/theory/minimax-quadrature.md`.  Implementation:
`src/gw/mpa/sigma_windows.py` (planner), no executor changes.

## 1. The measured law and where it comes from

`damped_rectangle_positive_rule` (the positive causal crossing rule, tag
`global_gauss`, printed "(glo)") costs **n = 87·f_max + 10** nodes at the
production tolerance 2.0e-3 and η = 0.018375 Ry (measured,
`runs/Na/02_soc48b_qsgw_mpa/14_semicore_cond_window/rule_cost_scan.out`;
gamma-bracket insensitive).  The construction discretises the causal
Laplace identity `1/z = ∫₀^∞ e^{-zt} dt` on the REAL interval
`[0, t_max]`, `t_max = log(1/(tail·tol))/γ_min ≈ 340–375 Ry⁻¹`, with one
global Gauss–Legendre rule whose order search is seeded at the
oscillation floor `⌊f_max·t_max/4⌋` (π/2 nodes per wavelength — the
Bernstein-ellipse threshold stated in `_gauss_legendre_panel_bound`).
The measured slope IS that floor: `t_max/4 ≈ 85–94 per Ry`.

This is **not** a defect of node selection, and panels cannot fix it:

* The panelled fallback (`damped_rectangle_rule`) tiles the same real-t
  interval at fixed nodes-per-wavelength; the selector already prefers
  the global rule because it is always at least as small.
* Any representation the executor can run has the form
  `Q(x) = Σ_l α_l e^{-i x t_l}` (the GN tau kernel's separable algebra).
  Demanding `|1 − x·Q(x)| ≤ ε` uniformly over the sign-symmetric window
  `x ∈ [−f, f]` at η-regularisation, with a bounded stability constant
  (no cancellation — the positivity/κ certificate), runs into a
  Landau-density floor: the target `1/(γ_min − ix)` has ε-significant
  Laplace content filling `t ∈ [0, log(1/ε)/γ_min]`, and representing it
  uniformly on an interval of length `2f` needs a node density of
  ~`f/π` per unit t, i.e. **n ≳ (f/γ_min)·log(1/ε)/π**.  At f = 8 Ry,
  ε = 2e-3 that is ≈ 860; the incumbent delivers 709.  The incumbent is
  essentially AT the information-theoretic floor of its certified
  region.
* Complex (rotated-contour) nodes are exactly what makes the
  sign-definite family logarithmic — `minimax.fit_damped_reciprocal`
  fits on a contour `e^{iφ}` seeded from the log-R sector rule, and the
  17-nodes-at-R=6.9e12 result exists because sign-definite rectangles
  sit in a sector `|arg d| ≤ π/2 − β`, leaving room to rotate.  The
  crossing rectangle fills the full sector as `f/γ → ∞`: a ray that
  decays for `x > 0` grows like `e^{|x| τ sinψ}` for `x < 0`.  No single
  exponential family serves both signs sub-linearly.

**Conclusion of the diagnosis: the region is wrong, not the rule.**  The
core window demands uniform η-resolution over the full product set
`[ω_lo, ω_hi] × {core transitions}`: `f_max = max|ω − e − a| ≈
ω_max + crossing_edge + a_hi ≈ 2–3·ω_max`, and
`crossing_edge = ω_max + margins` drags every band with `e ≲ ω_max`
into the core.  Yet for each individual evaluation frequency ω only the
thin shell `|ω − e − a| ≲ margins` actually crosses; the rest of the
rectangle is sign-definite and belongs to the logarithmic family.

The same applies to the metallic `sd_core` sliver: its denominator
`x = ω + e + a` is asymmetric — `x ∈ [−(excursion+…), ω_max + …]` — but
the rule is built on the symmetric window `[−f, f]`, `f ≈ ω_max + a_hi`.
Only `(a ≤ excursion + margin) × (ω ≈ 0)` can cross at all.

## 2. Why this bit the semicore campaigns

Widening the Σ evaluation window to reach semicore QP energies (Na
[−56, +7] eV → ω_max ≈ 4.1 Ry → 709 nodes; Fe-class 209 eV transition
span → f_max ~ 15–34 Ry → REFUSED at every production ceiling) scales
the single most expensive object in the Σ stage linearly, because the
executor dispatches one full spatial contraction per tau node
(`sigma.py`: `for t in win.nodes.t: tau_kernel(...)`; cost is
mask-independent).  Total Σ cost ∝ Σ over windows of n_tau.

Two facts make the fix possible with zero executor changes:

1. `accumulator.begin_window` takes `omega_indices/omega_values` per
   window row — windows may serve ω-subsets.
2. The ω grid (`sigma_omega_min/max/step_ev`, uniform) is interpolated
   at QP energies afterwards (`interp_along_omega` is `searchsorted`
   piecewise-linear — no uniformity assumption).  A semicore run's QP
   energies live in a few clusters; the gap between the valence window
   and the semicore cluster needs NO grid points and therefore no rule
   bandwidth.

## 3. The decomposition

With a patched ω grid (dense near the valence window, dense near each
semicore cluster, empty in between — `sigma_omega_patches_ev`), the
planner clusters each branch's `omega_abs` at gaps >
`mpa_sigma_omega_cluster_gap_ry` (default 1.0 Ry; a uniform production
grid is always ONE cluster and reproduces the incumbent plan
bit-for-bit).  Per crossing branch with ≥ 2 clusters, per cluster
`[w_lo, w_hi]`, the core bands split three ways at margin
`m = edge_factor·η + excursion` against the shallow-pole bracket
`[a_lo, a_hi]`:

| band range (e)                          | sign of x = ω−e−a | family |
|-----------------------------------------|-------------------|--------|
| `e < w_lo − a_hi − m` (pos bulk)        | ≥ m > 0           | rotated-Laplace fit, conjugate placement `t = +i·n̄` |
| `e ∈ [w_lo − a_hi − m, w_hi − a_lo + m]` (shell) | crosses    | damped positive rule, `f = max|ω−e−a|` over the CLUSTER — range-independent |
| `e > w_hi − a_lo + m` (neg bulk)        | ≤ −m < 0          | rotated-Laplace fit, b_slab orientation |

* The shell rule's bandwidth is `(w_hi − w_lo) + (a_hi − a_lo) + O(m)`:
  set by the cluster span and the pole bracket, **independent of the
  dynamic range**.  Shell rules are cached across clusters/branches on
  a coarse f lattice.
* The pos-bulk window is the one genuinely new orientation: the
  denominator `ω − e − a + i(γ+η)` has positive real part, i.e. the
  fit domain after conjugation.  Node placement `t = i·conj(n)`,
  weights `conj(w)`, `_apply_external_damping` unchanged.  Because
  `Im t > 0`, the kernel's factorised exponentials grow with (E − ref)
  and (a − ref); the window therefore anchors `E_ref_A` at the mask
  MAXIMUM, `E_ref_B` at the shallow-pole maximum, and carries a
  window-local `E_A` clamped to `E_ref_A` outside its mask so that
  masked-out deep bands contribute exact zeros instead of inf·0 (the
  metallic float-selector path multiplies rather than selects).
  In-mask, every factor is then ≤ 1 in magnitude apart from the fit's
  own certified amplification.
* The `sd_core` sliver decomposes on the same pattern: poles with
  `a > excursion + m` and ω-clusters with `w_lo > excursion + m` are
  sign-definite (`x = ω + e + a ≥ m`) and go to "single"-class Laplace
  windows; only `(a ≤ excursion + m) × (lowest ω cluster)` keeps a
  damped rule, at `f ≈ w_hi + sd_edge + excursion + m`.

Certification is extended, not bypassed: every rule keeps its sampled
error / continuum cover / κ evidence and provenance string; the plan
builder asserts exact coverage (each (band, pole-row, ω) cell of the
incumbent core appears in exactly one replacement window) before
returning; a refusal anywhere remains a refusal.

## 4. Cost law, before → after

For an evaluation set consisting of a filled valence window of measure
`W_v` plus `K` semicore clusters of span `σ_k` (the physical situation;
a deliberately dense uniform grid over the whole span remains linear —
that information demand is genuine):

* before: `n ≈ 87·(ω_max + crossing_edge + a_hi)` — linear in ω_max,
  REFUSED past the ceiling.
* after: `n ≈ 87·(W_v + a_spread + O(m)) + Σ_k 87·(σ_k + a_spread)
  + (2K+…)·rank_Laplace` — **independent of the semicore depth**.
  Doubling the dynamic range moves no term.

## 5. What was considered and rejected

* **Graded/dyadic panels** (the conservative fallback): void — the
  incumbent global rule already sits at the floor; panels only add
  nodes.  Measured: the panelled candidate loses to global at every
  scanned f_max.
* **Zolotarev / two-interval rational rules in x**: a rational node
  `1/(x − ζ_j)` does not factorise `x = ω − e − a` and cannot ride the
  separable tau kernel; direct per-(ω,m,p) evaluation is the very cost
  the kernel exists to avoid.
* **Diagonal (e+a)-tiling of the core into many small damped windows**:
  correct mathematically, loses operationally — the executor pays one
  full spatial dispatch per (window, node), so `Σ n_tau` over tiles
  exceeds the monolithic rule at Na and Fe scale.  The winning cut is
  the coarse 3-way split above, whose bulks are logarithmic and whose
  single shell per (branch, cluster) is range-independent.
