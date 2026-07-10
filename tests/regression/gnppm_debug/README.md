# GN-PPM regression fixture (MoS2 3×3, nspinor=2) — the Tier-1/Tier-2 workhorse

MoS2 3×3 full-BZ WFN (82 bands on disk), dynamic GN-PPM Σ_c pipeline.
This fixture backs the Tier-1 `gnppm` frozen gate AND (via a pytest
session fixture that keeps its `tmp/` restart state) every Tier-2
from-restart invariance gate: restart≡fresh, μ-pad flip, kij↔kij_stream,
SC-iter-1≡one-shot, fixed-point frozen rotations, IBZ≡full-BZ.

## 2026-07-09 shrink (642→399 centroids, ncond 54→20, nband 80→46)

Same-code re-freeze on `agent/driver-transparency` (post d03c857).
What changed vs the old fixture and why the frozen values moved:

| knob | old | new |
|------|-----|-----|
| centroids (orbit-closed) | 642 | 399 (`kmeans_cli 400 --seed 42`, orbit-aware default) |
| ncond | 54 | 20 |
| nband | 80 | 46 |
| WFN.h5 / kin_ion.h5 / dipole.h5 | — | unchanged (kin_ion/dipole are band-superset files; WFN not truncated — runtime reads only the requested window, truncation would only shrink the repo at the cost of a new 27 MB blob in history) |

The k-grid stays 3×3 — the IBZ cascade keeps its 5/9 symmetry structure
(`V_q g-flat [CC]: n_q_ibz=5 … unfold=IBZ→full` in the run log).

Frozen references re-frozen from the shrunk fixture (same code):
`sigma_diag_gnppm_ref.dat` (one-shot fresh run) and
`eqp_rotations_fixedpoint_ref.npy` (`qp_solver = fixed_point` from-restart
run — the procedure the Tier-2 gate uses; restart was separately verified
bit-identical to fresh).  The fixed-point ref preserves the pre-toggle
(620b501) eigh-family pin through the re-freeze: it was produced by the
same code whose fixed_point path reproduced the pre-toggle behavior on
the old fixture.

## Shrink validation (all on 1 GPU, A100)

* Fresh run twice → `sigma_diag`, `eqp0/1.dat` **bit-identical**
  (timestamp header only diff).
* `LORRAX_EXTRA_MU_PAD=12` pad flip → Σ_X, Σ_C, Σ_XC **bit-identical**
  (the old 642-centroid fixture drifted 6.3e-5 eV through the
  near-singular PPM fit; the shrunk mode census has no such mode near
  the validity threshold).  eqp0/1 max |Δ| = 3e-9 eV (last printed digit).
* IBZ cascade vs `LORRAX_FORCE_FULL_BZ=1`, static COHSEX
  (`cohsex_ibz_test.in`): all Σ columns **exactly equal** at printed
  precision; cascade active (n_q_ibz=5).
* restart=true from the fresh run's `tmp/` → **bit-identical** to fresh.
* kij vs kij_stream from restart: max|Δ sigma_c_kij_ev| = 4.5e-13 eV.
* SC-iter-1 vs one-shot from restart: **bit-identical**.
* GN-PPM census: invalid modes 8378/1432809 (0.58%), unfulfilled 0.58%.

## Files

- `gnppm_test.in` — dynamic GN-PPM input (Tier-1 gate + session fixture).
- `cohsex_ibz_test.in` — static COHSEX input (Tier-2 IBZ≡full-BZ gate).
- `centroids_frac_399.txt` — orbit-closed ISDF centroids (seed 42).
- `sigma_diag_gnppm_ref.dat` — frozen Tier-1 reference.
- `eqp_rotations_fixedpoint_ref.npy` — frozen `E_qp_nk_rydberg` (9, 46)
  for the fixed-point Tier-2 gate.
- `WFN.h5`, `kin_ion.h5`, `dipole.h5` — unchanged mean-field inputs.
