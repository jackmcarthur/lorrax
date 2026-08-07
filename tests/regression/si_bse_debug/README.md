Si BSE regression fixture (BerkeleyGW-anchored)
===============================================

The one cross-code BSE gate. Every other BSE test is a LORRAX-vs-LORRAX
freeze (interpolation vs native, distributed vs serial, matvec A/B);
this one is checked against BerkeleyGW.

System: bulk Si, 4x4x4 grid, spin-orbit (nspinor=2), 8 IBZ k-points in
the WFN, nband=60, nval=8. The BSE is solved on 4 valence x 4 conduction
Kramers pairs over the full 64-point grid (1024 transitions), TDA, at
DFT mean-field energies.

Input and required data:
- `bse_si_test.in`               (native 3D v(q) body + BGW q->0 head scalars)
- `WFN.h5`                       (9 MB; 8 IBZ k-points, 62 bands, nspinor=2)
- `centroids_frac_480.txt`
- `kin_ion.h5`

References:
- `bse_eigenvalues_ref.dat`      frozen LORRAX lowest-20 (eV)
- `bgw_eigenvalues_dft_ref.dat`  BerkeleyGW `eigenvalues.dat`, verbatim

Provenance
----------
The wavefunction is rebuilt from the QE inputs of the original BSE study
deck (scf + nscf + pw2bgw, ecutwfc 25 Ry, `noncolin`/`lspinorb`). The
rebuild was checked against an independently generated deck of the same
inputs and agrees on every DFT eigenvalue to 0.000 meV. The BerkeleyGW
reference is the verbatim `eigenvalues.dat` from the original study's
absorption run (full diagonalization, `use_momentum`, `skip_interpolation`,
4 valence + 4 conduction spinor bands); its own inputs are archived
alongside it in the study directory.

BGW anchor
----------
MEASURED 2026-08-07, lowest 20 eigenvalues, LORRAX vs BerkeleyGW:

    MAE  3.465 meV     max |delta|  8.877 meV     lowest state  -2.334 meV

Sanity of the comparison beyond the eigenvalues themselves: the spread of
the lowest 20 is 147.5 meV (BGW 136.3) and the count below 2.40 eV is 15
(BGW 14) — the two codes resolve the same states, not merely the same
onset.

The gate bands are MAE <= 10 meV and max <= 25 meV, roughly 3x the
measured values. They are deliberately loose: LORRAX does not use BGW's
head-and-wing treatment for the BSE kernel, so a residual offset of this
size is expected physics, not a defect. The band exists to catch a real
regression (tens of meV and up), not to certify agreement.

The frozen LORRAX reference carries the tight tolerance instead
(`atol = 1e-6 eV`): two independent GPU runs of this fixture reproduce
the eigenvalues bit-for-bit (max |delta| = 0), so anything above last-ULP
noise is a genuine change.

Pinned settings and why
-----------------------
The deck pins `mc_average_vcoul_body` and `zeta_rcond` explicitly rather
than riding the defaults; both have moved, and both are worth more than
the gate tolerance. `mc_average_vcoul_body` is a MATCHING CONVENTION
against the BGW run being compared to: BGW default (`avgcut` 1e12) pairs
with `true`, BGW `cell_average_cutoff 1d-12` pairs with `false`. This
fixture is anchored to a BGW run that leaves `cell_average_cutoff` at its
default, so the deck pins `true`. MEASURED effect of the pairing on the
lowest 20 of this system, against the same BGW reference:

    mc_average_vcoul_body = true    MAE 3.465 meV   max 8.877   lowest -2.334
    mc_average_vcoul_body = false   MAE 3.348 meV   max 9.729   lowest -1.669

Both sit inside the band; the point of pinning is that the default has
moved before and the shift is larger than the frozen-reference tolerance.

The Lanczos iteration count is part of the pinned configuration, not a
free parameter. The lowest-20 spectrum of this system is converged over a
window: at 200 iterations the lowest 20 are resolved, and at much larger
counts spurious repeated Ritz values proliferate and compress the
spectrum. MEASURED on the same system in the wider 8v8c quasiparticle
configuration, where the effect is easiest to see: the spread of the
lowest 20 falls from 68 meV at 400 iterations to 3.6 meV at 1600 and
1.7 meV at 2400, and the state count below 3.0 eV rises from the correct
27 to 113. The test therefore fixes the count; do not raise it expecting
a better answer.

What this gate does NOT cover
-----------------------------
- Quasiparticle-corrected (`--eqp`) BSE. That path is measurably less
  converged on this system at the same settings and is not pinned here.
- Oscillator strengths / absorption spectra. Only eigenvalues are
  compared; the dipole route needs the bare-p (`--skip-vnl`) dipole file
  and has its own convention list.
- The 8 valence x 8 conduction configuration, which is more sensitive to
  the Lanczos convergence window than the 4x4 one pinned here.

## 2026-08-07 — the mini-BZ head-draw fix moves this frozen reference (expected)

Commit 358bb0b fixed `build_v_head_miniBZ_avg_3d`'s uniform draw
(`randvals @ bvec.T` → `randvals @ bvec`; see that commit's message and
`tests/test_vcoul_minibz_head_draw.py`).  This deck pins
`mc_average_vcoul_body = true`, so the head table enters the kernel.  On Si
the fix is a pure RESEED of the MC average (bvec.T = P·bvec, P cyclic), and
the reseed moves the lowest-20 eigenvalues by max 7.067e-5 eV / MAE
7.42e-6 eV (8/20 cells over the 1e-6 pin; eigenvalues 8 and 16 carry it) —
70.7× ATOL_FROZEN_EV but physics-insignificant: the BGW band arm stays
green at MAE 3.4707 / max 8.8728 meV (the frozen ref itself sits at
3.4650/8.8774; the move is noise-level).  The Si result proves Si is BLIND
to the draw bug (cyclic-permutation degeneracy), not that the bug was
small: on non-cubic 3D cells the transposed draw is a BIAS worth ~50 % of
the whole mc-average correction (hexagonal z = 376).  The guard is the
committed hexagonal test arm, not any Si gate.

`bse_eigenvalues_ref.dat` is therefore stale at any HEAD ≥ 358bb0b.  The
owner has authorized adopting the candidate generated at 358bb0b on the
BUILD_NOTES pins —
`/pscratch/sd/j/jackm/svc_vcoul/_gates_after/bse_eigenvalues_candidate.dat`
(md5 cab1dd48…, byte-exact ref format; sidecar README records pins,
jobid/steps and both delta tables) — as the new reference at the
integration head.  ATOL_FROZEN_EV stays 1e-6: it is a bit-reproducibility
pin, and loosening it instead of refreezing was considered and refused.
