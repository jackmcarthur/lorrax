# The exchange head across the mini-BZ face — measurement, 2026-08-10

This is the measurement `THEORY_LT_HEAD_TENSOR.md` §7 named as missing when
the moment-tensor head landed at e06fc7de: a matched-|Q| ladder that leaves
the Γ mini-BZ cell, run on both the cell-averaged arm and the pointwise-exact
default. The 2026-08-09 validation could not do it because every rung of that
ladder sat inside the first cell, so the predicted convergence of the two arms
was untested rather than falsified.

The standing write-ups are `THEORY_LT_HEAD_TENSOR.md` §6.4 (physics) and
`HEAD_TENSOR_IMPL.md` §5.7–5.8 (tables and the prediction scoring). This file
is the repo-visible record of what was run and what came out.

## What was run, and what it cost

No new deck was staged. On a 3×3 grid the mini-BZ face sits at t\* = 1/6 in
crystal units — the measured inscribed radius is 0.202219 bohr⁻¹ — while Γ→M
runs to t = 0.5, and `--extra-q` takes arbitrary off-grid Q. So the MoS₂
3×3×1 640-centroid slab deck that the 2026-08-09 legs already used reaches
three times the face on its own wavefunctions.

Both arms ran from one tree at `origin/main` a65a5326, where the tensor head
is landed, with `--band-degeneracy off` pinned on every leg so a concurrently
flipping default could not move the numbers. On this deck that flag selects
the same 2v2c window the default already chose, which is why every previously
published row reproduces exactly.

| leg | what | wall |
|---|---|---|
| A0 / A2 | both arms, 26 Q, `--n-eig 6` | 45 s / 37 s (4 GPU) |
| F0 / F2 | both arms, 26 Q, `--n-eig 36` (full spectrum) | 32 s / 38 s (4 GPU) |
| cell probe | 48 kernel-level Q, no BSE, no ISDF | 23 s (1 GPU) |
| grid probe | the same on 3×3, 6×6 and 12×12 mini-BZ geometry | 28 s (1 GPU) |

A denser 6×6 deck exists and is complete
(`lorrax_sandbox/runs/MoS2/07_mos2_ref_80Ry_12x12_400b_2026-07-21/
03_lorrax_gw_6x6_80Ry_bse`), but it carries a 7.2 GB `zeta_q.h5` and a 2.65 GB
ISDF tensor against this deck's 185 MB and 145 MB. It was priced and declined,
because the crossover is a property of the mini-BZ geometry and the Coulomb
kernel rather than of the electronic structure, so the 6×6 and 12×12 answers
come out of the 3×3 deck's own cell in 28 s.

## What came out

Write R for the cell-averaged head divided by the pointwise head it is
supposed to converge to, `R = d*Md / v(Q)|Q·d|²`, for a longitudinal dipole.

**The two heads do meet, and not at the face.** R passes through one at
t/t\* = 0.201, which is |Q| = 0.0407 bohr⁻¹ and one fifth of the way to the
face — deep inside the first cell, where §7 expected no agreement at all. It
keeps falling to a worst disagreement of R = 0.584 at t/t\* = 0.60, turns, and
climbs back toward one from below. Outside the face the approach is a clean
power law with a measured log–log exponent of **−2.05**, which is the
O(Δ²) quadrature error of §3.3 seen directly. At the zone boundary the two
heads still differ by 3.4 %.

**The averaged head sits below the point value, not above it.** The
small-|Q| 2D law would make `v q_a q_b` convex and put the cell average above
the centre value; the measurement says otherwise beyond t/t\* = 0.2, because
the `slab_lr` kernel is not C/q at these |Q|. The Ismail-Beigi factor
saturates once z_c q ≳ 1, and z_c = 11.34 bohr here, so that happens at
t = 0.073 — well inside a mini-BZ whose face is at t = 0.167. Over almost the
whole ladder v(q)q² is concave, and a cell average of a concave function falls
below its centre value.

**The tensor head's direction-blindness is a Γ-cell property.** The in-plane
anisotropy M_TT/M_LL runs 0.997 → 0.369 → 0.033 across t/t\* = 0.06 → 1 → 3.
Isotropic in plane while the cell contains Γ, essentially pure longitudinal by
the zone boundary. M_zz is zero to machine precision at every rung.

**Refining the grid changes the numbers and not the picture.** With the k-grid
alone changed, R = 1 lands at t/t\* = 0.201, 0.306, 0.369 and the worst
disagreement is R = 0.584, 0.702, 0.786 on 3×3, 6×6 and 12×12. Denser grids
average over less, as they must.

**On the exciton spectrum the two arms do not converge.** Reading the arms
apart on energies needs a permutation-proof observable, because the
rank-three head lifts bright states out of any truncated window; the full
36-state spectrum supplies one, and its lowest six reproduce the `--n-eig 6`
legs to 0.000000 meV. The spectral trace difference runs 1.09 to 2.65 eV —
30 to 74 meV per state — across the whole ladder with no decay outside the
face, where the averaging error is already down to 6.7 %. That difference is
the head's *representation*: the ON arm contracts exact Cartesian dipoles, the
OFF arm routes the same head through the ISDF μ-tile. The scope limit
`HEAD_TENSOR_IMPL.md` §6 states as "head and body no longer cancel each
other's ISDF error" is therefore now a number on this deck, and outside the
cell it is the larger of the two effects. The §3.3 convergence is a property
of the head, and it is confirmed; it is not a property of the two code arms.

**Nothing already measured moved.** Every row the 2026-08-09 matched ladder
published reproduces to 0.000000 eV on both arms, and Γ is 0.000000 meV on
both arms at both window sizes.

## One correction the ladder forces

`THEORY_LT_HEAD_TENSOR.md` §6.1 and `HEAD_TENSOR_IMPL.md` §5.2 adopt
(0,t,0) versus (t,−t,0) as the matched-|Q| pair, and matched it is. It is also
a pair of **symmetry-equivalent** points: the angle between those two
directions on this deck's own reciprocal vectors measures 120.0001°, so both
are Γ→M and they are related by the C₃ that MoS₂'s D₃ₕ contains. The
1.2–11.8 meV those sections read as genuine trigonal warping is therefore not
a direction effect; it is the calculation's own symmetry residual. Along the
new ladder that residual runs 1.2, 11.8, 14.4, 29.6, 83.1 and 109.2 meV at
t = 0.02, 0.08, 0.12, 0.18, 0.25 and 0.42, it is present with the head OFF, and
it grows steeply with |Q|.

The genuine matched direction pair is (0,t,0) against (t/√3, t/√3, 0) —
Γ→M against Γ→K at equal |Q|. It spreads 42.6 meV at t = 0.12 against the
equivalent pair's 14.4, and 79.5 meV at t = 0.30 against 48.0. The genuine
effect is the larger one, but only by about a factor of two, and by t ≈ 0.4
the symmetry residual has caught up with it. This deck separates trigonal
warping from its own numerical asymmetry only on the near-Γ rungs.

This is recorded rather than acted on: it is a reading of two already-landed
sections, not a code defect, and the owner should decide whether §6.1 is
amended or annotated.

## Predictions, scored

Written before any leg was submitted, kept verbatim as
`LT_LADDER_PREDICTIONS_0810.md`; scored in `HEAD_TENSOR_IMPL.md` §5.8.

| # | outcome |
|---|---|
| P1 rate — \|Q\|⁻² outside the cell, slope −2.0 ± 0.4 | **RIGHT**: −2.05 |
| P1 sign — averaged head above the point value everywhere | **WRONG**: it is below beyond t/t\* = 0.20; v(q)q² is concave at these \|Q\| |
| P1 knee — at ≈ 0.70 r_ins | **RIGHT within 8 %**: R minimum at 0.60 r_ins |
| P1 arms — ON−OFF decays to an O(1–10) meV/state ISDF floor | **WRONG in magnitude**: it does not decay, and it is 30–74 meV/state |
| P2 — the inside rungs reproduce the 2026-08-09 ladder exactly | **RIGHT**: 0.000000 eV on all six published rows |
| P3 as the work order states it — spread is trigonal warping | **WRONG**: the pair is C₃-equivalent |
| P3 as recorded in advance — equivalent pair is the floor | **RIGHT**: 42.6 vs 14.4 meV, 79.5 vs 48.0 meV |
| P4 — Γ unmoved on both arms | **RIGHT**, exactly |
| P5 cost — under 6 min per leg | **RIGHT**: 32–45 s |

## Reproducing

Run directory `/pscratch/sd/j/jackm/ltladder_0810` (tree at a65a5326,
fixtures symlinked from `/pscratch/sd/j/jackm/headtensor_0809/run`).
The ladder is

    A = 0,t,0        t = 0.02 0.04 0.08 0.12 0.16 | 0.18 0.21 0.25 0.30 0.36 0.42
    B = t,-t,0       same t
    control          0.069282,0.069282,0  and  0.173205,0.173205,0

with `--n-val 2 --n-cond 2 --vq-mode interp --px 2 --py 2
--band-degeneracy off`, once as the default arm and once with
`--head-minibz-average`. The kernel-level probes call
`vq_interp.minibz_head_vlr(..., moment=True)` directly with a minimal
`prep = {GS: [[0],[0],[0]], alpha}`, which is the correct Miller superset for
any q_z = 0 Q inside the first BZ, and need `JAX_ENABLE_X64=1` to match the
driver's draws.
