# Historical pane-control crossing-rule cost law

This page records the geometry study behind the frozen
`LORRAX_SIGMA_PLAN=panes` comparison route. The production MPA path now uses
direct denominator boxes and exposes no omega-cluster policy; see
`docs/theory/sigma-quadrature-problem.md`.

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
semicore cluster, empty in between — `sigma_omega_patches_ev`), the pane
control clusters each branch's `omega_abs` at gaps greater than 1.5 requested
grid steps (a contiguous grid is one cluster). Per crossing branch with at
least two clusters, per cluster
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

## 5b. Measured on the real sodium store (arm 21)

The synthetic table above has zero pole spread; the sodium store's
shallow poles spread over ~4.8 Ry of position, which is what the first
cut of the decomposition missed (its cluster shells kept the full pole
bracket and LOST to the monolithic rule, 2104 vs 1550 plan nodes).
With the per-cluster crossing-pole cut (`a_cut = w_hi + margin`; poles
above it cannot cross that cluster and ride a sign-definite slab), the
real [−66:−48, −32:−20, −7:7] eV plan measures (bool-occupation
planner replica, no slivers):

| cluster | shell nodes before cut | after cut |
|---|---|---|
| Fermi (+ω half) | 493 | 121 |
| Fermi (−ω half) | 472 | 97 |
| 2p | 493 | 275 |
| 2s | 493 | 493 |
| **plan total** | **2104** | **1209** |

against 1550 for the monolithic contiguous plan.  The 2s cluster's
shell keeps the full spread because its crossing set genuinely reaches
every shallow pole (`a_cut ≈ a_hi` there) — the residual cost is pole
physics, not geometry, and shrinking it is the fit-side channel of §6.

End-to-end GPU (one-shot Na semicore, 4 ranks): the contiguous replay
at the new tip reproduces the 15_ reference to ≤ 0.06 meV on all ten
in-grid bands with 5723 vs 5725 tau dispatches — the bit-parity class.
The patched-grid arm's dispatch count and band physics are recorded in
`runs/Na/02_soc48b_qsgw_mpa/21_patched_omega_grid_20260816/`.

The same law is what unlocks the η = 0.10 eV fix (claims 0243/0246):
γ_min halves-and-more steepen the slope to ~224 nodes/Ry, so the
monolithic condwin rule (~6 Ry) needed ~1300 nodes and REFUSED at the
500 floor — the η=0.10 arm's NaN.  Decomposed cluster shells at
1–3 Ry certify at 224–774 nodes under the production ceiling.

## 6. What this does NOT fix: the fit-side sibling (design only)

Fe F3's failure chain is a FIT pathology, upstream of everything here.
Measured directly from the F3 store
(`runs/Fe/01_metal_mpa_qsgw/tmp/mpa/mpa_fit_sc_0000.h5`, offline census
walk over all 78 q × 600² blocks × 8 poles, 2026-08-16):

* the sampling line was stretched over the full transition span —
  Re z up to 15.346 Ry = **208.8 eV** with only 16 samples (heights
  0.2 / 2.0 Ry), and the static minimax window hit R = 1.53e13
  (UNCERTIFIED runtime solve, the F3 log);
* the width census: **33.16%** of total residue mass |B| sits on poles
  wider than 16 eV (healthy Si ≈ 1.5%, the rung-10 pathology 49%);
  90.9% is wider than 4 eV; the |B|-weighted mean width is 15.2 eV;
* the position census: the |B|-weighted mean pole position is
  **57.6 eV** — far above the ~10–25 eV plasmon region — with 10.5% of
  mass above 100 eV and significant-residue poles (>1e-3 of block max)
  out to **421 eV**, i.e. twice beyond the sampling span; the head fit
  itself spent 2 of 8 poles at ~185 eV with 19–24 eV widths.

The fit finalized (condition_max 4.5e8, backward error 5.7e-14 — both
certificates pass), which is exactly the census's point: conditioning
and residual cannot see a pole budget spent outside the physics.  The
quadrature decomposition makes the *evaluation* of whatever poles exist
cheap at any depth; it does not make a bad pole fit good.

The design that remains (the directive's candidate 4): the
Loewner/MPA sample-line span should NOT be stretched to cover deep
semicore transitions.  The deep-transition block sees W(ω′) far above
the plasmon region, where the screening is essentially static/bare and
smooth; it belongs in a separate few-pole (or COHSEX-static) channel
fitted on its own narrow window, so the main fit's pole budget stays in
the plasmon region and the census stays healthy.  The W-av head already
shows the pattern (a dedicated low-order channel for a structurally
different piece).  Sketch:

1. Split the χ/W sampling plan by transition depth at the same
   crossing-edge scale the Σ planner uses; fit the deep block with 1–2
   poles (or take its static limit) on a sample line matched to its
   span.
2. Feed the Σ planner both pole sets; the deep set's poles are, by
   construction, DEEP (a > crossing_edge) and route through the
   existing b_slab/stripe log-cost family — no new window classes.
3. Extend the census to certify per-channel: main-fit residue mass
   above 16 eV must stay in the healthy band; the deep channel's poles
   must sit inside its own fitted window.

This touches `sampling.py`/`sample_plan.py`/`fit_driver.py` and the
census, none of which this landing modifies.  Until it lands, Fe F3's
NaN is expected to persist regardless of the ω-grid used.
