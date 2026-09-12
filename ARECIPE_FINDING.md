# ARECIPE — a scale-free, material-independent shared-pole recipe

Lane ARECIPE. Branch `lane/sp-arecipe-2026-09-11`, worktree
`/pscratch/sd/j/jackm/wt_sp_arecipe`, cut from `integ/sp-union-2026-09-11` at
`9160c501`. Run directory
`runs/frequency_integration_sandbox/369_arecipe_20260911/`.

**Status: COMPLETE. Nine legs landed, all EXCLUSIVE, both decks measured.**
The design argument below stands as written; the measured outcome, the one
falsified prediction, the source defect this lane found and the ξ decision are
in `reports/shared_pole_push_2026-09-07/arecipe/report.md` §§8–13, which is the
owner of the results. Predictions were fixed before each run in
`369_arecipe_20260911/PREREGISTERED.md` and `PREREGISTERED_CAP.md`; neither was
edited afterwards.

**Headline.** Na 21 → 20 line supports, +10.4 % K, Σ unmoved (139 µeV, zero
rows over 2 meV). Si 34 → 44, +29.8 % K, and the Σ movement is confined to the
region the incumbent could not support pointwise — where both arms cover they
agree to 1.08 meV with zero rows over the bar. One pre-registered cost law was
falsified (K tracks support *spread*, not count) and replaced with a measured
one; one source defect was found and closed on the recipe side. CD is routed to
lane AIRKA and is the outstanding verdict.

Two corrections to the design text below, both made before the runs and both
recorded where they happened rather than silently applied: `imaginary_top_factor`
moved 2.0 → 2.5 on a recomputed Zolotarev turnover, and a fourth rule — the
bank's remote-domain cap — was added after Si was refused without it.

---

## 1. What has to be true

Four constants in the recipe table were absolute energies in eV. Each one
silently encoded a material:

- `line_break_ev = 12.0` put the fine/coarse break *below* Si's plasmon
  (16.60 eV) and *above* Na's (6.05 eV), so Si sampled its own plasmon at the
  coarse step and Na never reached the coarse branch at all. Measured
  consequence: halving the coarse step cut Si's conjugate-correction Σ
  sensitivity 2.02× for +10.4 % K (ASIMOM §8).
- `plasma_margin_ev = 3.5` is 21 % of Si's ω_p and 58 % of Na's.
- `imaginary_floor_max_ev = 16.0` was the top of Na's imaginary condenser and
  *inert* on Si, where `max(16, top)` returns `top`.
- `active_depth_ev = 15.0` decided which electrons set ω_p. On the Na
  production deck the 2p semicore sits at depth **25.04 eV** — 0.04 eV outside
  even the `borderline_depth_ev = 25` diagnostic band. A material whose
  semicore landed at 14 eV would silently have tripled its ω_p.

And one structural gap (ASIMOM §6, `KNOWN_LORRAX_ISSUES.md`): `top_ev` was
produced by the resolver and read by nothing else in `src/`. The Σ ω-box was
sized with no knowledge of where the fit's pointwise support ends. On Si the
box top (+18.0 eV) sat at 0.895 of `top_ev`, and the model disagreement rises
40–55× within 4 eV above `top_ev`.

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
   Si: 18.0 + 12.325 = **30.3 eV**, against the old `top` = 20.1 — the whole
   story. Na: 5.0 + 3.282 = **8.3 eV**, against 9.5 — inside, by a little.
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
   frequency.** This is what `active_depth_ev` was trying to say with a number.

## 3. The dial table

Every new rule is a multiple of η, of ω_p, of the band energies, or of the Σ
grid, or a dimensionless fraction. Nothing is in absolute eV.
Table `shared_real_pole_v2_r1`, `src/gw/shared_pole_recipe.py:29`.

| dial (old) | old value | new rule | why |
|---|---|---|---|
| `active_depth_ev` | 15.0 eV | **removed.** Active (screening) set = the fixed point of *a band joins if its depth below µ is at most `active_plasma_factor` × ω_p of the set including it*, factor **1.0**; seeded on the shallowest occupied band, grown greedily, monotone, terminating (`shared_pole_recipe.py:699`). | §2.6, the dial-free statement. c = 1 needs no calibration; it reproduces both reference decks exactly (§4) and excludes Na's 2p (depth 25.04 eV against ω_p 16.0 for the 7-electron set) and a hypothetical Cu 3p (75 against 44.5) for the right reason. |
| `borderline_depth_ev` | 25.0 eV | **removed.** Diagnostic band = depth ≤ `2 ×` the active threshold. | Census reporting only; no numerical consumer. |
| `line_break_ev` | 12.0 eV | **removed.** Fine spacing holds on `[0, ω_fine]`, **ω_fine = max(ω_p, E_g + Δ_act)**, Δ_act = µ − min E over the active occupied manifold (`:826`). | §2.5. ω_p alone is wrong for a dilute system (a molecule in a large cell: ω_p ≈ 2 eV, transitions at 5–20 eV); `E_g + Δ_act` is the top of the first valence→conduction continuum and rescues that case. On both reference decks ω_p wins. |
| `line_high_step_ev` | 1.0 eV (= 4η) | **removed.** Beyond ω_fine the step grows geometrically, `Δ_k = 2η (1+ξ)^k`, with **`line_growth_fraction` ξ = 0.25** (`:877`). | §2.5. The count beyond the plasmon is logarithmic in `top`. The step is continuous at ω_fine — the first geometric step *is* 2η — so the plasmon's upper shoulder is still resolved at ≤ h for ~2 eV. ξ = 0.25 puts the leading Hermite error of a 1/ω² tail at 0.31 ξ⁴ ≈ 1.2e-3 relative, at the direction cutoff and below Si's measured held-W defect (3.4e-3). |
| `line_low_step_ev` + `reference_eta_ev` | 0.5 / 0.25 eV | **collapsed** to `line_step_eta_factor = 2.0` (step = 2η). | Identical arithmetic, stated dimensionlessly. The η = 0.10 test (0.11 meV scaled, 2.0 meV unscaled) is what fixes 2η against h = 4η. |
| `plasma_margin_ev` | 3.5 eV | **removed.** `top = max(2.25 ω_fine, 1.25 (E_Σ + Δ_act))`, E_Σ = `max |edge|` over the Σ ω-box or, when set, the `sigma_omega_patches_ev` union (`:832`). | Two floors, both relative. **2.25 ω_fine**: pointwise support runs until |W_c| has fallen to ≈ 1/2.25² ≈ 0.2 of its static scale, past which M₁/M₃ carry it (§2.4); the measured Si top ladder 12/17/20/24 eV → 0.72/0.46/0.40/0.37 meV was still improving at 1.45 ω_p. **1.25 (E_Σ + Δ_act)**: the support–Σ-box consistency ASIMOM's gap demands (§2.3); the 1.25 is headroom for the pre-cliff rise and for SC band drift, not an accuracy lever — the cliff itself is entirely above `top`. |
| `imaginary_floor_max_ev` | 16.0 eV | **removed.** `u_max = 2.5 × ω_fine` (`imaginary_top_factor`, `:838`). | −W_c(iu) reaches its 1/u² tail by ≈ 2 ω_p, so the last node belongs just past that, not at it. The Zolotarev count turns over at κ = 16.096, so 2.5 ω_fine reproduces **both** decks' measured counts (Na 3 at κ = 15.12, Si 4 at κ = 41.5) at no cost, and puts Na's u_max at 15.117 eV — within 6 % of its only measured-good value (16.0 eV = 2.65 ω_p; u_max at the line top, 1.58 ω_p, was 2.9× worse). **Corrected during implementation:** this was first set to 2.0 on the belief that 2.5 would buy Na a fourth imaginary support (+224 columns). That was arithmetic error — the turnover is at κ 16.096, not near 12 — so 2.0 was needlessly conservative and 2.5 is free. Corrected before any leg ran; `PREREGISTERED.md` §4. |
| κ (imaginary count) | `top / u_min` | **`u_max / u_min`** (`:892`) | The Zolotarev count `m = max(2, round[ln(16κ²) ln(4/ε)/2π²])` estimates the interval the nodes actually span. Same (3, 4) on both decks, and it decouples the condenser from how far the real line runs: the condenser is now a property of the material's screening scale, the gap and η alone. |
| `held_line_fractions` | (0.25, 0.65) of `top` | **midpoint nearest `0.5 ω_fine`, and midpoint nearest `√(ω_fine · top)`** (`:900`) | One held point per spacing law, each at that law's own natural midpoint — arithmetic in the linear region, geometric in the geometric one, matching the existing held-imaginary construction. Fractions of `top` would have put both Si held points in the tail, where Σ does not weight the model. |
| line endpoint | `top` always appended | `top` **replaces** the last ladder point when the remaining gap is under half the local step; `ω_fine` likewise absorbs a stub interval (`:874`, `:889`) | Removes a near-duplicate Hermite block the old recipe produced on both decks (Si 20.00 / 20.101, Na 9.50 / 9.547 — two samples 30–100 meV apart at height h = 1.0 eV). Costs nothing, saves one bank evaluation per q. |
| `relaxed` tier `line_count = 8` | `linspace(0, top, 8)` | same geometry, `line_step_eta_factor = 4`, `line_growth_fraction = 0.5` | A tier should be coarser dials, not a different shape. With `top` now relative, a linspace over it would step 7.4 eV across Si's plasmon. |
| `height_eta_factor`, `direction_cutoff`, `imaginary_width_fraction`, `infinity_width_fraction`, `imaginary_count_epsilon`, `multiplet_relative_tolerance`, `bank_rule_tolerance`, `sigma_tolerance`, `u_min = max(h, E_g)` | — | **unchanged** | Already scale-free or already certified. |

**Five dimensionless factors in, four absolute energies and two step constants
out. No new user-visible deck key.**

## 4. What the two reference decks resolve to

`369_arecipe_20260911/deck_report_preview.py` → `deck_report.txt` runs the
**real** `bind_shared_pole_census` and `resolve_shared_pole_recipe` against each
deck's measured metadata (ω_p, µ, E_g, Δ_act, volume, Σ box, η, n read from the
ACONJ control arms, job 58209192). `resolve_preview.py` holds the old recipe,
whose table no longer exists in the source.

| | Si old | Si new | Na old | Na new |
|---|---|---|---|---|
| ω_fine (eV) | — | 16.601 | — | 6.047 |
| `top` (eV) | 20.101 | **37.906** | 9.547 | **13.606** |
| term that bound `top` | ω_p + 3.5 | **Σ window** 37.906 (plasmon 37.352) | ω_p + 3.5 | **plasmon** 13.606 (Σ window 10.352) |
| line supports | 34 | **45** | 21 | **20** |
| spacing | 0.5 to 12, then 1.0 to 20.1 | 0.5 to 16.601, then ×1.25 per step | 0.5 throughout | 0.5 to 6.047, then ×1.25 per step |
| u range (eV) | 1.0 – 20.101 | 1.0 – 41.502 | 1.0 – 16.0 | 1.0 – 15.117 |
| imaginary supports | 4 | **4** | 3 | **3** |
| held line (eV) | 5.25, 13.50 | 8.25, 25.33 | 2.25, 6.25 | 3.25, 9.54 |
| unique bank evaluations | 41 | **52** | — | **26** |
| bank `freq_max` = top + band span (eV) | 72.56 | 90.37 (**×1.245**) | 159.31 | 163.37 (**×1.025**) |
| relaxed line supports | 8 | 23 | 8 | 11 |

**The active-set fixed point is a no-op on both decks**, which is what makes the
rest attributable: Si grows 2 → 4 → 6 → 8 electrons (ω_p 8.30 → 11.74 → 14.37 →
16.60 eV, every band's depth under the running ω_p) and stops with all four
valence bands; Na seeds on 3s (ω_p 6.047) and rejects 2p (depth 25.04 against
ω_p 16.0 for the 7-electron set). Both reproduce today's ω_p exactly.

**Cost.** Na is free: same imaginary count, one *fewer* line support, +2.5 % on
the stream's `freq_max` (its bandwidth is dominated by the 149.8 eV band span,
not by `top`). Si pays +11 line supports and +24.5 % `freq_max`, roughly
41 s → 51 s of bank, for the fine sampling across its plasmon that the ASIMOM
intervention measured at 2.02× on Σ sensitivity, plus a `top` that finally
clears its own Σ box. The K prediction is a **number fixed in advance**:
`PREREGISTERED.md` §2 — Si sum K ratio 1.08–1.12 and strictly below the
no-collapse extrapolation 1.143; Na 0.96–1.00.

## 5. The deck report, exactly as a user sees it

This is the user-facing surface of the change. Both blocks below are verbatim
output of the code on this branch (`deck_report.txt`, and the parse notice from
`LorraxConfig.from_input_file`).

**At deck-parse time** (`gw_config.py:_resolve_shared_pole_inputs`, beside the
`write_w` debug notice). It cannot print resolved numbers — ω_p needs the
wavefunctions — so it states the coupling and points at the block that does:

```
  ==========================================================
  NOTE: sigma_w_model = shared_pole sizes its W frequency
  support from the Sigma grid (-15 .. 18 eV) plus the depth of the
  screening electrons, so a wider grid -- or a
  sigma_omega_patches_ev window over a semicore state -- means a
  longer bank, not a clamped one.  The resolved supports, the
  plasmon and which term set the top are printed in the
  SHARED-POLE SUPPORT block (shared_real_pole_v2_r1) at run time.
  ==========================================================
```

The grid string is the deck's own: `-5 .. 5 eV` on Na, and a patch deck reads
`(-56:-50, -5:5)`.

**At recipe resolution, Si P4:**

```
  ==========================================================
  SHARED-POLE SUPPORT (shared_real_pole_v2_r1, production)
  Screening electrons : 8.000 in 4 active band(s), omega_p = 16.601 eV
                        active set = depth <= 16.601 eV (fixed point); deepest active 12.325 eV
  Structure scale     : omega_fine = 16.601 eV = max(omega_p 16.601, E_g 0.693 + depth 12.325)
  Sigma asks W up to  : 30.325 eV = grid extent 18.000 + active depth 12.325
  Top support         : 37.906 eV, set by the SIGMA WINDOW
                        max(2.25*omega_fine = 37.352, 1.25*Sigma window = 37.906)
  Line supports       : 45 -- step 0.500 eV (2*eta) to 16.601 eV, then x1.25 per step to 37.906 eV
  Imaginary supports  : 4 -- 1.000 to 41.502 eV, log-spaced (kappa = 41.5)
  Sample height       : 1.000 eV = 4*eta
  Bank cost grows with the top support and as 1/height: widening
  sigma_omega_min_ev/max_ev, or adding a sigma_omega_patches_ev
  window over a semicore state, raises the top support with it.
  ==========================================================
```

**At recipe resolution, Na P16:**

```
  ==========================================================
  SHARED-POLE SUPPORT (shared_real_pole_v2_r1, production)
  Screening electrons : 1.000 in 1 active band(s), omega_p = 6.047 eV
                        active set = depth <= 6.047 eV (fixed point); deepest active 3.282 eV
  Structure scale     : omega_fine = 6.047 eV = max(omega_p 6.047, E_g 0.000 + depth 3.282)
  Sigma asks W up to  : 8.282 eV = grid extent 5.000 + active depth 3.282
  Top support         : 13.606 eV, set by the PLASMON floor
                        max(2.25*omega_fine = 13.606, 1.25*Sigma window = 10.352)
  Line supports       : 20 -- step 0.500 eV (2*eta) to 6.047 eV, then x1.25 per step to 13.606 eV
  Imaginary supports  : 3 -- 1.000 to 15.117 eV, log-spaced (kappa = 15.1)
  Sample height       : 1.000 eV = 4*eta
  Bank cost grows with the top support and as 1/height: widening
  sigma_omega_min_ev/max_ev, or adding a sigma_omega_patches_ev
  window over a semicore state, raises the top support with it.
  ==========================================================
```

Under an SC enclosure a further line reads
`SC enclosure <status> at epoch <n>: the retained bounds above may exceed this
map's own.` The full key/rule dump the resolver already printed follows both
blocks unchanged, with every rule string rewritten to the new law.

**The refusal.** `GATE shared_pole_support_window` fires if `top` is ever below
`1.25 × (Σ grid extent + active depth)` and names both numbers. It is
**unreachable by construction** — `top` is a max that includes that term, and
the SC envelope only ever raises it — and is kept as a self-consistency
assertion because the failure it guards is silent in every other receipt. It is
an inline `GATE shared_pole_*` refusal, not a `_GATE_ROWS` entry, so `GATE_HASH`
and every stored gate identity are unchanged.

## 6. Migration consequence

`RECIPE_HASH` changes (the table is new: `shared_real_pole_v2_r1`). Stored banks
and models carry the old hash in `identity/recipe_hash`, so **every stored
shared-pole bank and model is refused and rebuilt** — that is the intended
behaviour of the restart guard, not a regression, and it is the whole reason the
hash exists. Rebuild cost is one bank per deck (Si ~51 s, Na ~140 s at P16).

Unchanged: `GATE_VERSION`/`GATE_HASH` (no gate row added or edited), the on-disk
dataset names (`factor`/`C`), the normalization metadata string, and the
`operator_realization` token. The SC support enclosure version moves to
`sc_interacting_support_enclosure_20260911` because `omega_fine_ev` joins the
three retained scalars — it is now a support *boundary*, so an enclosure that
did not retain it would regenerate a different ladder whenever ω_p moved
between SC maps. All 12 support-session tests pass with their assertions
unchanged.

## 7. Alternatives considered and rejected

- **`top` = the largest transition energy Ω_max (E_max − E_min over delivered
  bands).** The principled choice — above Ω_max the exact W_c has *no spectral
  weight*, so the moments are exact there and the unconstrained gap closes
  completely. Rejected on cost: the bank's time rule is sized on
  `max|ω| + Δ_max` (`src/gw/mpa/evaluator.py:135-160`), so the stream node count
  is **linear in `top`**. Ω_max is 52.5 eV on Si (+45 % stream) and 149.8 eV on
  Na (+84 %, ~138 s → 254 s) to pin a region where |W|/|W(0)| ≲ 1e-2.
- **Δ_occ (all occupied bands) instead of Δ_act in the Σ-window term.** On Na
  that is 53.8 eV of 2s semicore, giving `top` = 73.5 eV and +59 % stream on the
  deck already at 0.22 meV. Rejected by §2.4: W at 55–75 eV is 1 % of its static
  scale there. The semicore case is served correctly and *on demand* — a
  `sigma_omega_patches_ev` window at −54 eV raises E_Σ, which raises `top`
  through the same rule, and the user pays only when they ask.
- **Raising the moment widths.** Measured: 39.1× better full-M₁ defect bought
  1.23× in Σ for +11.8 % K (ASIMOM §8). Not the lever.
- **Extending the fine region to ω_fine + h** so the plasmon peak is resolved
  symmetrically. Costs 2 supports per deck; rejected because the geometric
  ladder's first steps are 0.50, 0.62, 0.78 eV — at or below h = 1.0 eV for
  ~2 eV above ω_p. *Flagged for the owner* as the second dial to turn if Na
  regresses (`PREREGISTERED.md` §4 orders it after `u_max`).
- **A coarser ladder below E_g**, where an insulator's W_c has no spectral
  weight at all. Real savings for a wide-gap material (E_g = 10 eV → 20 fewer
  supports), exactly zero for Si (0.69) and Na (metal). Deferred, not
  implemented: a third spacing law for no effect on any deck we can measure.
- **Anchoring the held line points on the Σ window** (0.25/0.65 of E_Σ + Δ_act)
  rather than on ω_fine. Defensible — it implements ASIMOM's rule directly — but
  it makes a model-quality diagnostic move when the user changes the Σ box.
  *Marked for owner review.*

## 8. Verification and scope

**Verified (login node, CPU, metadata only — no driver, no GPU, no HDF5):**

- `tests/test_shared_pole_inputs.py` + `tests/test_shared_pole_support_session.py`:
  **92 passed**. Includes two new tests that exercise the active-set fixed point
  in both directions (`test_active_set_fixed_point_admits_a_deep_band`) and the
  Σ-window term taking over the top from a semicore patch
  (`test_sigma_window_can_set_the_top`, which also asserts the extra cost of
  reaching 51 eV is ≤ 5 supports — logarithmic, not linear).
- Wider bounded sweep (`test_shared_pole_{store,restart_recipe,restart_member,
  sc_plan,moment_diagnostics,capacity,outputs,local_checks}`,
  `test_afixes_review`, `test_gw_production_report`): **173 passed, 1 failed** —
  `test_shared_pole_capacity.py::test_sigma_inherited_and_second_w_are_separate`,
  which **fails identically on the untouched base tip** `wt_sp_union @ 9160c501`
  (a login-node artifact: the device budget reads 0 B/rank with no GPU). Not
  this lane's.

**Not verified.** No compute-node gate, no driver run, no Σ, no CD comparison,
no K measurement, no timing. The support geometry in §4 is the real resolver on
deck metadata, not a bank. Every number in §4 except the resolved geometry —
notably the ×1.245 bank estimate — is arithmetic from the documented cost law,
not a measurement. The two reference decks are the only materials this has been
resolved for; the rules are argued for arbitrary materials but measured on none.
Nothing is on `origin/main`.

## 9. What runs when a pool lands

Four legs, one four-rank leg per node, launched concurrently with serialized
handshakes, per `PREREGISTERED.md` §6: **Na P16 old and new recipe first** (the
deck that decides the design), then Si P4 old and new, each with a control
repeat for the noise floor. `placement_audit.py` on every band-bearing leg,
refuse on CO-LOCATED. Scored with `tools/compare_bgw_gwjax.py` on the campaign
metric, every figure reported twice — unrestricted and on the delivered subset.
