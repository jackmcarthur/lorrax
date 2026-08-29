# Bispinor GN-PPM regression fixture (MoS2 3×3, nspinor=2)

The one e2e gate on the bispinor path.  As of 2026-07-09 the gate mode is
**bispinor GN-PPM** (dynamic Σ_c on the screened charge W + bare Breit Σ^B
folded into `sigX`) — upgraded from the original screened-charge COHSEX
gate: same 4 ζ-channel / 7 V_q-tile / transverse-γ̃ machinery, plus the
dynamic PPM pipeline on bispinor wavefunctions, which no other gate runs.
Static Σ_SX/Σ_COH kernels on nspinor=2 wavefunctions remain covered by the
`cohsex_debug` gate (WFNsmall is nspinor=2); the scalar static kernels by
`si_cohsex_debug`.

Bispinor GN-PPM runs at HEAD with one input addition: the dynamic head
needs `whead_imfreq` alongside `vhead`/`whead_0freq` (all 0.0 here — the
same explicit head bypass the COHSEX gate used; without it the run dies
in `persist_w0_and_head` at the imaginary probe frequency).

## 2026-07-09 shrink (640/668 → 256/209 centroids) + mode change

Same-code re-freeze on `agent/driver-transparency`.  What changed and why
the frozen values moved (this is a NEW reference — mode changed from
static COHSEX to GN-PPM, labels changed sigSX/sigCOH/sigTOT →
sigX/sigC/sigXC):

| knob | old | new |
|------|-----|-----|
| compute mode | screened-charge COHSEX | GN-PPM (ω ∈ [−4, 4] eV, step 1.0) |
| charge centroids | 640 (non-orbit-closed) | 256 (`kmeans_cli 256 --seed 42 --no-orbit`) |
| transverse centroids | 668 (orbit-closed, `_current`) | 209 (`kmeans_cli 200 --seed 42 --density-mode current`, orbit-aware) |
| nval / ncond / nband | 4 / 4 / 32 | unchanged (nband floor: 26 occupied states ⇒ b3 = 30; nband=16 fails `BandSlices` validation) |
| WFN.h5 | 34-band truncation of the 82-band source | unchanged |

Orbit-closure properties preserved deliberately: charge tiles run
full-BZ-direct (closure check fails → fallback, as before); transverse TT
tiles take the IBZ cascade (`unfold=IBZ→full`).  Note the transverse
IBZ-unfold is exact on MoS2 (see reports/bispinor_tt_conditioning_2026-06-16);
the CrI3 in-plane TT-unfold issue does not apply to this fixture.

Below 256/209 the runtime does not improve (128/104 measured within 1 s —
the fixed cost is the per-channel r-chunk streaming over the 30×30×120
grid at 60 Ry / ngkmax 5545), so the better-conditioned 256/209 set is
kept.  Warm wall ≈ 40 s recorded (was ≈ 100 s at 640/668 static COHSEX).

## 2026-08-09 re-freeze — the k-star completion (owner-authorized)

`sigma_diag_bispinor_ref.dat` was re-cut, and **not** for a fixture change:
every input in this directory is untouched.  `dd727216` landed the k-star
`(Z − Z†)/2i` completion in `src/gw/ppm_accumulators.py`, which had been
closing a one-sided τ grid with an elementwise `Im` — the pair adjoint only
at k = −k.  The correction moves sigC by max **7.7240e-03 eV** over 268 of
270 rows and leaves sigX and VH bit-identical.  Γ moves by 8.530e-04 eV
rather than by zero, which is this deck behaving correctly: MoS2 has no
inversion, TRS carries the spinor rotation, and σ^τ at Γ is only
approximately complex-symmetric, so Γ inherits a correction of its own.

The values were cut at `dd727216` and **re-verified bit-exact** against
`main` @ `5b135f8e` before the freeze (max |Δ| = 0.000e+00, 0 of 1620 cells
differing at all).  The `9a730da8` dipole re-cut cannot reach this deck at
all — `bispinor_test.in` takes the explicit head bypass (`vhead` =
`whead_0freq` = `whead_imfreq` = 0) and the fixture ships no `dipole.h5`.
Full provenance, **including the one thing this reference cannot see**, is
in the file's own header; read it there rather than here.

## Shrink validation (all on 1 GPU, A100)

* Fresh run twice → `sigma_diag`, `eqp0/1.dat` **bit-identical**.
* `LORRAX_EXTRA_MU_PAD=4` pad flip (the historically catastrophic
  transverse 668→672 extent class) → **bit-identical** including Σ_C.
* GN-PPM census: invalid modes 18494/589824 (3.14%).

## Files

- `bispinor_test.in` — GN-PPM bispinor input (Tier-1 gate; the Tier-2
  pad-flip gate reruns it fresh with `LORRAX_EXTRA_MU_PAD=4`; bispinor
  restart round-trips in both layouts since 2026-08-23, see gw_init.py).
- `centroids_frac_256.txt` / `centroids_frac_209_current.txt` — charge /
  transverse ISDF centroid sets (seed 42).
- `sigma_diag_bispinor_ref.dat` — frozen reference (sigX/sigC/sigXC).
- `WFN.h5` (34 bands), `kin_ion.h5`.
