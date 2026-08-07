Si 3D COHSEX regression fixture (BGW-anchored)
==============================================

First `sys_dim = 3` end-to-end COHSEX-JAX regression case. Every other e2e
gate is a 2D MoS2 self-consistency freeze; this one exercises the 3D
Coulomb / analytic-head path, and its frozen value is anchored to
BerkeleyGW (not just a re-frozen LORRAX number).

System: bulk Si, 4x4x4 grid, no spin-orbit, 8 IBZ k-points in the WFN
(Sigma is evaluated full-BZ-direct on all 64 k-points). nband=60, nval=8.

Input and required data:
- `cohsex_si_test.in`   (native 3D v(q) body + BGW q->0 head scalars)
- `WFN.h5`              (9 MB; 8 IBZ k-points, 60 bands, nosoc)
- `centroids_frac_960.txt`
- `kin_ion.h5`
- `dipole.h5`

Reference output:
- `eqp_si_ref.dat`

BGW anchor
----------
This is the Si 4x4x4 COHSEX system proven to agree with BerkeleyGW at
MAE = 0.12 meV (max |Δ| = 0.48 meV) — see
`reports/cohsex_si_444_gamma_agreement_2026-05-02/`. That headline number
was obtained with a full BGW `vcoul` body overlay (a 185 MB dump, not
git-committable). This fixture instead uses LORRAX's **native** finite-q
Coulomb body plus BGW's q->0 head injected as two scalars (`vhead`,
`whead_0freq`).

CORRECTION 2026-08-06.  This paragraph used to claim the native body
"matches BGW's 4*pi/|q+G|^2 body for `cell_average_cutoff 1d-12`".  That is
FALSE for the deck as written, in exactly one slot per q, and it cost
136 meV.  `mc_average_vcoul_body` defaults to TRUE, which MC mini-BZ
cell-averages v(q, G=0) at EVERY q != 0.  BGW gates the same operation on
`avgcut`; Common/vcoul_generator.f90:101-103 says it outright:

    If |q+G|^2 < avgcut, calculate <1/(q+G)^2>.  Otherwise, calculate
    1/(q+G)^2.  The default value for avgcut is TOL_SMALL, i.e., average
    only done if G=0.

So under `cell_average_cutoff 1d-12` BGW averages ONLY the literal q+G=0
element and uses the point value everywhere else.  LORRAX averaged one
q-shell too many.

The deck therefore sets `mc_average_vcoul_body = false` explicitly.
MEASURED against `06_si_4x4x4_nosoc/D_bgw_cohsex_noavg`, bare Sigma_X over
128 (k,band) pairs:

    mc_average_vcoul_body = true    MAE 136.202 meV   max 282.961
    mc_average_vcoul_body = false   MAE   0.351 meV   max   1.122

The pairing is a MATCHING CONVENTION, not a bug: BGW default (avgcut=1e12)
<-> true; BGW noavg (avgcut=1d-12) <-> false.  This fixture is anchored to
noavg, so it must set false.

Why the old verification missed it: the check below is stated as a SPREAD,
and a rigid per-occupancy offset cancels in a spread.  Sigma_X was never a
compared column, and the pytest compares against this repo's own frozen
eqp_si_ref.dat -- BerkeleyGW is not in that loop at all. Verified against BGW `sigma_hp.log`
(`06_si_4x4x4_nosoc/D_bgw_cohsex_noavg`) at Gamma: sigCOH tracks BGW to
0.22 meV spread, sigTOT to 3.2 meV spread — matching the validated
185 MB-overlay run. (The residual rigid offsets in sigSX/sigCOH are the
known BGW self-energy-column conventions the report's compare handles;
the physical band-to-band / k-to-k tracking is what is anchored here.)

The pytest regression test copies this directory to a tmp dir, runs

```bash
python -m gw.gw_jax -i cohsex_si_test.in
```

and compares generated `eqp_si_test.dat` against `eqp_si_ref.dat`.
Two independent GPU runs reproduce the reference bit-for-bit (max |Δ| = 0);
the gate `atol = 1e-3 eV` (1 meV) is a physical bound: comfortably above
the 0.12 meV BGW agreement, tight enough to catch a real 3D-path regression.
