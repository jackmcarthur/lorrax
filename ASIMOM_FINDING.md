# ASIMOM — why ACONJ moves Si 150x more than Na

Branch `lane/sp-asimom-2026-09-11`, based on ACONJ's numerical tip `3af53203`.
**No driver was run.** Every number is read from ACONJ's completed arms
(`runs/frequency_integration_sandbox/360_aconj_20260911`, jobs 58209192.6/.13/.17
Si and .20/.21/.24 Na, all EXCLUSIVE). Analysis scripts and outputs are in
`runs/frequency_integration_sandbox/366_asimom_20260911/`. Full evidence:
`reports/shared_pole_push_2026-09-07/asimom/report.md` in the sandbox repo.

## Verdict

**Neither hypothesis.** Moment enforcement did not degrade — it improved at
every Si parent. The nonsymmorphic symmetry is exact. The mechanism is the
**frequency coverage of the fit**: ACONJ's span change moves `W` only where the
recipe has no support, and Si's Σ samples that region while Na's does not.

## 1. The moment hypothesis is refuted, from stored receipts

`full_m1_defect` / `full_m3_defect` per parent, candidate/control ratio
(`construction_receipt.json`, written at `src/gw/shared_pole_constructor.py:1085-1086`):

| deck | M1 ratio, all parents | M3 ratio, all parents |
|---|---|---|
| Si P4 (8) | 0.968 – 0.991 | 0.948 – 0.991 |
| Na P16 (29) | 0.999 – 1.003 | 0.988 – 1.004 |

Every Si parent reproduces `M_1` and `M_3` **better** under the candidate. Na is
unchanged. `retained_subspace_moments` is 1e-13 and PASS in all arms; the
∞-state identity defects are 2.4e-11..2.3e-10 (Si) and 2.3e-13..1.1e-11 (Na) and
do not move. Both control repeats are bit-identical on M1, M3 and K, so the
ratios are attributable.

A second, independent refutation: the two arms' models **agree to 5e-7..2e-6 at
|z| ≥ 200 eV** on Si, which is the asymptotic region `M_1`/`M_3` govern, and Si
agrees to 2e-8..1.8e-6 at **every** point of the imaginary axis from 0 to 500 eV.
The moments are doing their job in both arms.

## 2. The nonsymmorphic hypothesis is refuted, from stored receipts

`sigma_diag.dat` carries `star_spread_ev = 0.000000000e+00` and
`star_spread_multiplet_ev = 0.000000000e+00` in **both** Si arms, over all 34
bands and all 64 full-BZ k. `src/gw/gw_output.py:1683-1691` measures this on the
full-BZ Σ arrays against the symmetry service's own star labels and explicitly
emits nothing rather than a gather-derived fake zero. Si is Fd-3m with 48
operations, 36 of them carrying τ = ±1/2 (all listed active in `gwjax.out`).
An unfolding defect could not produce an exact zero.
*Scope: the real diagonal of Σ_XC; the header states it is blind to the TRS
conjugation class.*

The effect is also **uniform across k** on both decks: Si per-k median |ΔQP| is
312–566 µeV at all 8 k with Γ mid-range; Na is 0.29–0.35 µeV at all 29 k. No
correlation with k symmetry class.

## 3. What it actually is

Evaluating the two stored models directly —
`W_c(q,z) = b diag(1/(z²−λ)) bᴴ`, the same expression as
`shared_pole_constructor.py:1036-1038` — on the production line contour:

| | at Re z = 0 | at its `top_ev` | 4 eV / 2.5 eV above it | at 200 eV |
|---|---|---|---|---|
| **Si** (`top_ev` 20.101) | 1.9e-8 – 1.3e-6 | 2.9e-5 – 4.7e-4 | **9.5e-3 – 2.4e-2** | 5.7e-7 – 1.9e-6 |
| **Na** (`top_ev` 9.547) | 4.7e-10 – 6.7e-9 | 3.5e-8 – 1.5e-7 | **7.9e-4 – 3.0e-3** | 1.3e-6 – 2.9e-5 |

Both decks have a sharp, unmarked accuracy cliff **exactly at the top line
support**: Si rises 40–55x over the next 4 eV, Na 7e3–8e4x over the next 2.5 eV.
Between the top support and the asymptotic regime nothing pins the model
pointwise, so the retained span — the only thing ACONJ changed — controls it.
Above ≈ 100 eV (Si) and ≈ 200 eV (Na) the arms re-converge as `M_1`/`M_3` take
over.

Si feels this in Σ and Na does not, for two compounding measured reasons.

**(a) Si's model is ~10⁴ less accurate than Na's inside its own support range.**
Held-W relative defects: Si `Wc` 3.1e-6..3.4e-4, `dWc/ds` 7.4e-7..3.7e-3; Na
`Wc` 2.4e-8..8.4e-7, `dWc/ds` 6.3e-8..1.1e-5. At each deck's own plasmon the two
arms differ by 2.9e-4 (Si, 16.6 eV) against 3.7e-8 (Na, 6.05 eV). Si must span
20.1 eV of gapped, van-Hove-structured `W` with n = 368; Na spans 9.5 eV with
n = 896. The recipe's fixed n-fractions therefore buy Si 4×92 + 46 = **414**
moment columns, 20.6–38.8 % of retained K, against Na's 3×224 + 112 = **784**,
49.3–73.9 % of K. (K/n is 2.90–5.46 on Si and 1.18–1.77 on Na.)

**(b) Si's Σ ω box reaches the bad end of its own support and Na's does not.**
Si `omega_max_ry` = 1.3229751176908335 = **18.000 eV**, 2.1 eV below `top_ev`,
where the relative model disagreement has already climbed four decades. Na's box
is ±5 eV, deep in its benign middle. Measured on the stored Σ cubes,
max|ΔΣ_c(ω)| over (k, band):

- Si: **3 µeV at ω ≈ 0–3 eV → 50519 µeV at ω = +18 eV**, 14966 µeV at −15 eV.
- Na: flat 36–334 µeV across ±5 eV, maximum in the **interior** (ω = −2.25 eV).
- In the overlapping window |ω| ≤ 5 eV, **Si moves 8–16 µeV and Na moves
  150–190 µeV** — Na moves more.

The band ≥ 13 concentration follows from (b), not from the moments. Relative
ΔΣ_c per band runs 3.9e-7 (band 4) → 1.3e-3 (band 32) on Si, a factor 3300,
while |Σ_c| itself is flat at 1.25–6.05 eV. Na runs 7e-7 → 2.7e-6, a factor 25,
peaking at the **frontier** and decaying upward — the opposite band dependence.

## 4. Recommendation

*Written before the experiment; section 5 tested it and every item held.*

1. **Do not raise the Si moment widths on this evidence.** The defect is not in
   the region `M_1`/`M_3` control; the arms already agree to 1e-6 there. The
   guide's "Si prefers ≥ 1.2x" (`algorithm_guide.md` §3, §7) is a sub-0.3 meV
   statement against a direct CD, a different question from this arm-to-arm
   sensitivity, and the current bar is 2–5 meV.
2. **The dials that address the measured defect are the high-frequency end of
   the fit**, and the campaign already prices both: Si line spacing 1 → 0.5 eV
   gives 1.36 → 0.46 meV, and the top ladder 12/17/20/24 eV gives
   0.72/0.46/0.40/0.37 meV (`algorithm_guide.md` §7). Si is the only reference
   deck that ever uses the coarse `high_step_ev = 1.0` branch, and it uses it
   from 12 eV to 20.1 eV — across its own plasmon.
3. **Nothing ties the Σ ω box to the model's `top_ev`.** `top_ev` is produced at
   `src/gw/shared_pole_recipe.py:686` and read nowhere else in `src/`; the Σ
   window planners never see it. Registered in `KNOWN_LORRAX_ISSUES.md`;
   diagnosis only — the owner owns the Σ quadrature.
4. **Judge a recipe by the model disagreement in the ω window that deck's Σ
   actually samples.** Added after the experiment, which measured it: that
   window forecasts the Σ sensitivity for both dials (widths 0.82–0.92 model
   against 0.814 measured; spacing 0.45–0.75 against 0.494), while the
   asymptotics — the thing `full_m1/m3_defect` measure, and the thing the
   widths dial improves 15–25x — forecast nothing. This is item 3's gap seen
   from the other side: it says what to check in place of what is missing.

## 5. The discriminating experiment — RUN, and both predictions held

Pool 58209192, four Si P4 legs, one per node, all EXCLUSIVE
(`placement_audit.py 58209192 36 37 38 39`). Decision rule fixed in advance in
`PREREGISTERED.md`, pushed to `lane/sp-asimom-prereg-2026-09-11` (`d7b7ad84`)
before any arm had an `eqp1.dat`. Each rung compares its own control
(`810c260b`) against its own candidate (`3af53203`) at a different recipe; one
dial each; the deck is byte-identical across all arms.

| rung | dSigma_c meV | vs stock | full M1 defect | M1 gain | held-W | held-W gain | K cost |
|---|---|---|---|---|---|---|---|
| stock | **50.519** | 1.000 | 8.047e-05 | 1.0x | 3.735e-03 | 1.00x | — |
| widths x1.5 | **41.117** | 0.814 | 2.057e-06 | **39.1x** | 3.643e-03 | 1.03x | +11.8 % |
| spacing 0.5 | **24.981** | **0.494** | 7.640e-05 | 1.1x | 2.297e-03 | **1.63x** | +10.4 % |

- Widths x1.5: 41.117 meV, above the pre-registered 35 meV line — **prediction
  holds**, moment width is not the lever.
- Spacing 0.5: 24.981 meV, at the pre-registered 25.3 meV line — **prediction
  holds**, but only just: 2.02x, right at the bar.

**The decisive number: raising the moment widths improved the full `M_1` defect
39.1x and moved Sigma 1.23x.** An intervention, not an observation.

**Why the spacing dial worked.** The model disagreement ratio in the 16-18 eV
window — the top of Si's Sigma box — predicts the Sigma change for both dials:
widths 0.82-0.92 model, 0.814 measured; spacing 0.45-0.75 model, 0.494 measured.
Neither the asymptotics (widths improve them 15x) nor the region above `top_ev`
predicts it. In Sigma itself at omega = +18 eV: 50519 -> 41118 (widths) ->
21359 ueV (spacing).

**Noise floor: exactly zero at both new recipes.** Each rung's own control
repeat is bit-identical to its control -- 0.000 ueV on all 272 QP rows,
identical K, identical full M1/M3 -- as stock already was. Every rung-to-rung
difference is attributable to the dial, with nothing to subtract. That is what
makes the spacing rung's 2.02x safe to read despite sitting on the bar.

**What did not improve:** max |dQP| rose slightly on both rungs (6408, 6468 vs
5862 ueV). It is dominated by states clamped to the box endpoint, and neither
dial moves the box. Median fell for widths (312 vs 487 ueV), rows over 2 meV
fell for spacing (12 vs 20).

**Two corrections to section 3, from the run logs.** (i) Both decks clamp, and
Na clamps *more*: Si has 1195/2176 (54.9 %) of `Sigma(E_DFT)` cells out of grid,
Na 42948/44032 (97.5 %). "Na's Sigma grid never samples that region" was too
strong — what differs is where the endpoint lands relative to each model's
support (Si +18.0 eV = 0.895 of `top_ev`, in the steep region; Na +5.0 eV =
0.524, still exact). (ii) A sharper clamp prediction was tested and **failed**:
clamped states at one k do not share one delta (11-24 distinct among 20-24). The
clamp fixes the evaluation frequency, not the value.

## 6. The original proposal, for the record

Si P4, one node, 4 GPUs, ~35 min total. Three recipe variants at both sources
(`810c260b` control and `3af53203` candidate), plus a control repeat:
stock; moment widths x1.5; `high_step_ev` 1.0 → 0.5.
**Predictions stated in advance:** widths x1.5 leaves ΔΣ_c near 50 meV;
`high_step_ev` 0.5 cuts it materially. A result the other way refutes this
finding.

## What is NOT claimed

No claim that either arm's Σ is the more accurate one — only that they differ by
the stated amounts. No claim about the Si SC failure, q averaging, BSE, or
exciton bands. The star-spread certificate covers the real diagonal of Σ_XC on
these two decks only, not the TRS conjugation class and not off-diagonals. The
direct model sweep is the line contour at Im z = 1 eV and the imaginary axis, at
8 Si parents and 4 Na parents, not every frequency or every parent. The
column-budget and support-range statements are properties of these two decks'
resolved recipes, not general laws. Nothing is on `origin/main`.
