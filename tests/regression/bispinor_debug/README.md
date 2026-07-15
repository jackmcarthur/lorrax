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

## 2026-07-15 regen at 25 Ry (suite-speedup: FFT grid 30×30×120 → 20×20×75)

Full QE regen — NOT a re-freeze of the same physics.  The 2026-07-09 shrink
found the fixture wall was r-chunk streaming over the FFT grid, so this regen
attacks the grid itself: `ecutwfc` 60 → 25 Ry (3.6× fewer grid points).
Fixture-only physics (unconverged cutoff, self-consistency freeze — the gate
was never BGW-anchored).  Source run + provenance:
`runs/MoS2/D_25Ry_bispinor_fixture_2026-07-15/` (QE SCF/NSCF at 25 Ry,
nbnd=40 → `truncate_bands.py` → 34-band WFN.h5 14.8 MB (was 52.6);
`gw.kin_ion_io`; kmeans reruns, same seeds/flags — transverse orbit set
closed at 208 (was 209) on the new grid).

- Orbit-closure properties preserved: charge non-closed (full-BZ-direct,
  log-asserted), transverse closed (IBZ cascade, log-asserted).
- **Chunk coverage now log-asserted**: all 4 ζ-fit passes run **3 r-chunks**
  at `memory_per_device_gb = 30`; the gate fails if any channel drops below
  2 chunks (the streaming seam must stay exercised — lower the memory budget
  if a future shrink collapses it).
- Validation: 3 consecutive fresh runs bit-identical (2160 sigma_diag values,
  max|Δ| = 0.0; timestamp header only diff); `LORRAX_EXTRA_MU_PAD=4` pad twin
  bit-identical (see Tier-2 gate).  Recorded driver wall 29 s warm
  (was ~40 s at 60 Ry).

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

## Shrink validation (all on 1 GPU, A100)

* Fresh run twice → `sigma_diag`, `eqp0/1.dat` **bit-identical**.
* `LORRAX_EXTRA_MU_PAD=4` pad flip (the historically catastrophic
  transverse 668→672 extent class) → **bit-identical** including Σ_C.
* GN-PPM census: invalid modes 18494/589824 (3.14%).

## Files

- `bispinor_test.in` — GN-PPM bispinor input (Tier-1 gate; the Tier-2
  pad-flip gate reruns it fresh with `LORRAX_EXTRA_MU_PAD=4` — bispinor
  restart is not yet supported, see gw_init.py).
- `centroids_frac_256.txt` / `centroids_frac_208_current.txt` — charge /
  transverse ISDF centroid sets (seed 42, on the 25 Ry grid).
- `sigma_diag_bispinor_ref.dat` — frozen reference (sigX/sigC/sigXC).
- `WFN.h5` (34 bands), `kin_ion.h5`.
