# ARECIPE — a scale-free, material-independent shared-pole recipe

Lane ARECIPE. Branch `lane/sp-arecipe-2026-09-11`, worktree
`/pscratch/sd/j/jackm/wt_sp_arecipe`, cut from `integ/sp-union-2026-09-11` at
`9160c501`. Run directory
`runs/frequency_integration_sandbox/369_arecipe_20260911/`.

**Status: design fixed, source not yet changed, nothing measured.** Section 6
lists what is claimed and what is not. The support arithmetic below comes from
`369_arecipe_20260911/resolve_preview.py`, a login-node transcription of the
old and proposed resolvers over deck metadata read from ACONJ's completed
control arms (`360_aconj_20260911/02_{si_p4,na_p16}/control`, job 58209192).

---

## 1. What has to be true

Four constants in `shared_real_pole_v1_r3b` are absolute energies in eV
(`src/gw/shared_pole_recipe.py:19-42`). Each one silently encodes a material:

- `line_break_ev = 12.0` puts the fine/coarse break *below* Si's plasmon
  (16.60 eV) and *above* Na's (6.05 eV), so Si samples its own plasmon at the
  coarse step and Na never reaches the coarse branch at all. Measured
  consequence: halving the coarse step cut Si's conjugate-correction Σ
  sensitivity 2.02× for +10.4 % K (ASIMOM §8).
- `plasma_margin_ev = 3.5` is 21 % of Si's ω_p and 58 % of Na's.
- `imaginary_floor_max_ev = 16.0` is the top of Na's imaginary condenser and is
  *inert* on Si, where `max(16, top)` returns `top`.
- `active_depth_ev = 15.0` decides which electrons set ω_p. On the Na
  production deck the 2p semicore sits at depth **25.04 eV** — 0.04 eV outside
  even the `borderline_depth_ev = 25` diagnostic band. A material whose
  semicore lands at 14 eV would silently triple its ω_p.

And one structural gap (ASIMOM §6): `top_ev` is produced at
`shared_pole_recipe.py:686` and read by nothing else in `src/`. The Σ ω-box is
sized with no knowledge of where the fit's pointwise support ends. On Si the box
top (+18.0 eV) sits at 0.895 of `top_ev`, and the model disagreement rises 40–55×
within 4 eV above `top_ev`.

## 2. The physics the new rules are built on

1. **W_c is pinned pointwise only on [0, top] and asymptotically by M₁/M₃.**
   In between, the retained Krylov span decides, and it wanders: Si 3.5e-4 at
   20.0 eV → 1.9e-2 at 24 eV; Na 3.5e-8 at 9.5 → 2.8e-3 at 12 (ASIMOM §4).
   The cliff is at `top_ev` on both decks, to sweep resolution.
2. **Σ's sensitivity follows the model disagreement in the ω window that deck's
   Σ actually samples** — not the asymptotics, not the region above `top`
   (ASIMOM §8, confirmed by intervention on two dials).
3. **The W frequency Σ needs at evaluation energy E is |E − E_n'|**, over the
   intermediate bands n'. The ones carrying weight are the screening-active
   occupied manifold, so the top of that window is
   `max(|ω_min|, ω_max) + (µ − min E over active occupied bands)`.
   Si: 18.0 + 12.32 = **30.3 eV**, against `top` = 20.1 — the whole story.
   Na: 5.0 + 3.28 = **8.3 eV**, against `top` = 9.5 — inside, by a little.
4. **|W_c(ω)| falls as (ω_p/ω)² above the plasmon**, so the Σ error a region
   can contribute is its relative model error times that factor. This is the
   quantitative reason Na tolerates a 1.3e-2 model error at 60 eV (where
   |W|/|W(0)| ≈ 0.01) while Si does not tolerate 5e-2 at 30 eV (≈ 0.31): 100×
   in the product, against the measured 150× in Σ.
5. **Structure lives at and below the plasmon** — the e–h continuum from E_g up
   and the collective pole at ω_p. Above it W_c is a smooth decaying tail whose
   only scale is ω itself, so the sampling there should be *relative*, which is
   the same condenser logic the imaginary axis already uses.
6. **An electron screens at a frequency only if it is bound by less than that
   frequency.** This is what `active_depth_ev` is trying to say with a number.

## 3. The dial table

Every new rule is a multiple of η, of ω_p, of the band energies, or of the Σ
box, or a dimensionless fraction. Nothing is in absolute eV.

| dial (old) | old value | new rule | why |
|---|---|---|---|
| `active_depth_ev` | 15.0 eV | **removed.** The active (screening) set is the fixed point of *a band joins if its depth below µ is at most `active_plasma_factor` × ω_p of the set including it*; `active_plasma_factor = 1.0`. Seeded on the shallowest occupied band, grown greedily, monotone, terminating. | §2.6. The dial-free statement of "binding energy below the collective frequency". c = 1 needs no calibration; it reproduces both reference decks exactly (§4) and excludes Na's 2p (depth 25.04 eV, ω_p with 2p = 16.0 eV) and a hypothetical Cu 3p (depth 75, ω_p with 3p = 44.5) for the right reason. |
| `borderline_depth_ev` | 25.0 eV | **removed.** Diagnostic band = depth ≤ `2 ×` the active threshold. | Census reporting only; no numerical consumer. |
| `line_break_ev` | 12.0 eV | **removed.** Fine spacing holds on `[0, ω_fine]` with **ω_fine = max(ω_p, E_g + Δ_act)**, Δ_act = µ − min E over the active occupied manifold. | §2.5. ω_p alone is wrong for a dilute system (molecule in a box: ω_p ≈ 2 eV, transitions at 5–20 eV); `E_g + Δ_act` is the top of the first valence→conduction continuum and rescues that case. On both reference decks ω_p wins. |
| `line_high_step_ev` | 1.0 eV (= 4η) | **removed.** Beyond ω_fine the step grows geometrically: `Δ_k = 2η (1+ξ)^k`, i.e. `ω_k = ω_fine + (2η/ξ)((1+ξ)^k − 1)`, with **`line_growth_fraction` ξ = 0.25**. | §2.5. Count beyond the plasmon grows only logarithmically in `top`. The step is continuous at ω_fine (first tail step is exactly 2η), so the plasmon's upper shoulder is still resolved at ≤ h for ~2 eV. ξ = 0.25 puts the leading Hermite error of a 1/ω² tail at 0.31 ξ⁴ ≈ 1.2e-3 relative — at the direction cutoff, below the measured Si held-W defect (3.4e-3). |
| `line_low_step_ev` + `reference_eta_ev` | 0.5 / 0.25 eV | **collapsed** to `line_step_eta_factor = 2.0` (step = 2η). | Identical arithmetic, stated dimensionlessly. The η = 0.10 test (0.11 meV scaled vs 2.0 meV unscaled) is what fixes 2η against h = 4η. |
| `plasma_margin_ev` | 3.5 eV | **removed.** `top = max(2.25 ω_fine, 1.25 (E_Σ + Δ_act))`, where `E_Σ = max |edge|` over the Σ ω-box or, if set, the `sigma_omega_patches_ev` union. | Two floors, both relative. **2.25 ω_fine**: pointwise support runs until \|W_c\| has fallen to ≈ 1/2.25² ≈ 0.2 of its static scale, past which M₁/M₃ carry it (§2.4); the measured Si top ladder 12/17/20/24 eV → 0.72/0.46/0.40/0.37 meV was still improving at 1.45 ω_p, and 2.25 ω_p extrapolates in that direction. **1.25 (E_Σ + Δ_act)**: this is the support–Σ-box consistency ASIMOM's gap demands (§2.3); the 1.25 is headroom for the pre-cliff rise and for SC band drift, not an accuracy lever — the cliff itself is entirely above `top`. |
| `imaginary_floor_max_ev` | 16.0 eV | **removed.** `u_max = 2.0 × ω_fine` (`imaginary_top_factor`). | −W_c(iu) has reached its 1/u² tail by ≈ 2 ω_p, and beyond that the moments are exact; there is nothing for a node to buy. Reproduces **both** decks' measured imaginary counts (Na 3, Si 4) with no absolute constant, where 2.64 ω_p (Na's historical 16 eV) would cost Na a fourth imaginary support = +224 columns. Note the measured row "u_max at the line top is 2.9× worse" was Na at u_max = 1.58 ω_p; 2.0 sits above it. |
| κ (imaginary count) | `top / u_min` | **`u_max / u_min`** | The Zolotarev count `m = max(2, round[ln(16κ²) ln(4/ε)/2π²])` is an estimate for the interval the nodes actually span. Gives the same (3, 4) on both decks, and decouples the imaginary condenser from how far the real line runs — the condenser is now a property of the material's screening scale, the gap and η alone. |
| `held_line_fractions` | (0.25, 0.65) of `top` | **midpoint nearest `0.5 ω_fine`, and midpoint nearest `√(ω_fine · top)`** | One held point per spacing law, each at that law's own natural midpoint — arithmetic in the linear region, geometric in the geometric region, matching the existing held-imaginary construction. Fractions of `top` would have put both Si held points in the tail, where Σ does not weight the model. |
| line endpoint | `top` always appended | `top` **replaces** the last ladder point when the remaining gap is under half the local step | Removes a near-duplicate Hermite block the old recipe produced on both decks (Si 20.00 / 20.101, Na 9.50 / 9.547 — two samples ~30–100 meV apart at height h = 1.0 eV). Costs nothing, saves one bank evaluation per q. |
| `relaxed` tier `line_count = 8` | `linspace(0, top, 8)` | same geometry with `line_step_eta_factor = 4`, `line_growth_fraction = 0.5` | A tier should be coarser dials, not a different shape. With `top` now relative, a linspace over it would step 7.4 eV across Si's plasmon. |
| `height_eta_factor`, `direction_cutoff`, `imaginary_width_fraction`, `infinity_width_fraction`, `imaginary_count_epsilon`, `multiplet_relative_tolerance`, `bank_rule_tolerance`, `sigma_tolerance`, `u_min = max(h, E_g)` | — | **unchanged** | Already scale-free or already certified. |

**New dials introduced: five dimensionless factors** (`active_plasma_factor`,
`line_growth_fraction`, `plasma_top_factor`, `sigma_window_top_factor`,
`imaginary_top_factor`) replacing **four absolute energies** and two
step constants. **No new user-visible deck key.**

## 4. What the two reference decks resolve to

`369_arecipe_20260911/resolve_preview.py`. Deck inputs: Si η = 0.25,
ω_p = 16.601, E_g = 0.693, µ = 6.627 (midgap), Δ_act = 12.325, box −15…+18;
Na η = 0.25, ω_p = 6.047, E_g = 0 (metal), µ = 1.615 (FD), Δ_act = 3.282,
box ±5.

| | Si old | Si new | Na old | Na new |
|---|---|---|---|---|
| ω_fine (eV) | — | 16.601 | — | 6.047 |
| `top` (eV) | 20.101 | **37.906** | 9.547 | **13.606** |
| binding term for `top` | ω_p + 3.5 | 1.25 (18.0 + 12.32) = 37.91 (2.25 ω_fine = 37.35) | ω_p + 3.5 | 2.25 ω_fine = 13.61 (Σ-box term 10.35) |
| line supports | 34 | **45** | 21 | **20** |
| spacing | 0.5 to 12, then 1.0 to 20.1 | 0.5 to 16.6, then ×1.25 per step | 0.5 throughout | 0.5 to 6.05, then ×1.25 per step |
| u range (eV) | 1.0 – 20.101 | 1.0 – 33.202 | 1.0 – 16.0 | 1.0 – 12.094 |
| imaginary supports | 4 | **4** | 3 | **3** |
| held line (eV) | 5.25, 13.50 | 8.25, 25.33 | 2.25, 6.25 | 3.25, 9.54 |
| bank `freq_max` (eV) | 72.56 | 90.37 (**×1.245**) | 159.31 | 163.37 (**×1.025**) |
| relaxed line supports | 8 | 23 | 8 | 11 |

**The active-set fixed point is a no-op on both decks**, which is what makes the
rest attributable: Si grows 2 → 4 → 6 → 8 electrons (ω_p 8.30 → 11.74 → 14.37 →
16.60 eV, every band's depth under the running ω_p) and stops with all four
valence bands; Na seeds on 3s (ω_p 6.047) and rejects 2p (depth 25.04 vs ω_p 16.0
for the 7-electron set). Both reproduce today's ω_p exactly.

**Cost.** Na is free: same imaginary count, one *fewer* line support, +2.5 % on
the stream's `freq_max` (its bandwidth is dominated by the 149.8 eV band span,
not by `top`). Si pays +11 line supports and +24.5 % `freq_max`, i.e. roughly
41 s → 51 s of bank, for the sampling that the ASIMOM intervention measured at
2.02× on Σ sensitivity, plus a `top` that finally clears its own Σ box.
Si K is expected to rise; the ASIMOM spacing rung measured +8 supports → +10.4 %
K, and the extra supports here are at high ω, where the direction blocks are
close to collinear with M₁ and the Gram cut (1e-8) should collapse them. That
is the number this lane must actually measure.

## 5. Alternatives considered and rejected

- **`top` = the largest transition energy Ω_max (E_max − E_min over delivered
  bands).** The principled choice — above Ω_max the exact W_c has *no spectral
  weight*, so the moments are exact there and the unconstrained gap closes
  completely. Rejected on cost: the bank's time rule is sized on
  `max|ω| + Δ_max` (`src/gw/mpa/evaluator.py:135-160`), so the stream node count
  is **linear in `top`**. Ω_max is 52.5 eV on Si (+45 % stream) and 149.8 eV on
  Na (+84 % stream, ~138 s → 254 s) for a region where |W|/|W(0)| ≲ 1e-2.
- **Δ_occ (all occupied bands) instead of Δ_act in the Σ-window term.** On Na
  that is 53.8 eV of semicore, giving `top` = 73.5 eV and +59 % stream on the
  deck that is already at 0.22 meV. Rejected by §2.4: W at 55–75 eV is 1 % of
  its static scale there. The semicore case is served correctly and *on demand*
  — a `sigma_omega_patches_ev` window at −54 eV raises E_Σ, which raises `top`
  through the same rule, and the user pays only when they ask.
- **Raising the moment widths.** Measured: 39.1× better full-M₁ defect bought
  1.23× in Σ for +11.8 % K (ASIMOM §8). Not the lever.
- **Extending the fine region to ω_fine + h** so the plasmon peak is resolved
  symmetrically. Costs 2 supports per deck; rejected because the geometric
  ladder's first steps are already 0.50, 0.62, 0.78 eV — at or below h = 1.0 eV
  for ~2 eV above ω_p. *Flagged for the owner* as the first dial to turn if Na
  regresses.
- **A coarser ladder below E_g**, where an insulator's W_c has no spectral
  weight at all. Real savings for a wide-gap material (E_g = 10 eV → 20 fewer
  supports), exactly zero for Si (0.69) and Na (metal). Deferred, not
  implemented: it adds a third spacing law for no effect on any deck we can
  measure.
- **Anchoring the held line points on the Σ window** (0.25/0.65 of E_Σ + Δ_act)
  rather than on ω_fine. Defensible — it implements ASIMOM's rule directly — but
  it makes a model-quality diagnostic move when the user changes the Σ box.
  *Marked for owner review.*

## 6. Scope

Nothing here is measured yet. The support geometry in §4 is arithmetic over
deck metadata, not a driver run; ω_p, µ, E_g, Δ_act and the band extrema are
read from ACONJ's completed control arms and from `gwjax.out`, and the Σ boxes
are the decks'. No claim is made about K, about Σ accuracy against CD, or about
any deck other than these two.
