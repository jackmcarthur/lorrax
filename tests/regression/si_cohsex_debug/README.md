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
Coulomb body (which matches BGW's 4*pi/|q+G|^2 body for `cell_average_cutoff
1d-12`, i.e. no fixwings) plus BGW's q->0 head injected as two scalars
(`vhead`, `whead_0freq`). Verified against BGW `sigma_hp.log`
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
