# Exact bulk q=0 migration measured, fixture refreeze blocked upstream

**Date:** 2026-09-03  
**Source:** `lane/bisp-bulk-gamma-completion-2026-09-02`  
**Disposition:** no frozen reference, tolerance, or fixture input changed.

Step 4 of the bulk Gamma-completion lane replaces the production scalar bulk
head's scrambled-Sobol/Baldereschi--Tosatti estimator with the exact
Wigner--Seitz-polyhedron receipt already owned by the packed photon
completion.  The requested honest-refreeze stop condition cannot be evaluated
on either canonical fixture because both refuse before q=0 evaluation on the
same pre-existing band-window gates recorded in the 2026-09-02 integration
accounting:

- `si_cohsex_debug`: zeta window `[0,60)` cuts a degenerate multiplet at band
  60, k=3, with a 0.000 meV gap against the 1.000 meV rule.
- `hbn_cohsex_debug`: zeta upper boundary 80 equals the 80-band WFN extent, so
  closure cannot be measured without a spare band.

The lane therefore did **not** re-freeze either reference.  The scalar-owner
change is isolated in its own commit so integration can drop it independently
of the already-gated packed bulk completion.

The Si deck additionally pins `vhead=3303.748102` and
`whead_0freq=150.395600`.  Measurement-only exact/debug runs with those
overrides intact are identical to the printed digit across all 480 Sigma and
QP rows and never call either quadrature: its explained fixture move is
exactly zero.  With the overrides disabled, the exact native owner measures
`<v>=3304.0599019255033`, `<W>=150.38648650223473` a.u.; the incumbent
production-size Sobol diagnostic then fails trying to allocate 10.44 GiB, so
there is no honest latent native-rule A/B to quote.

For diagnosis only, an explicitly labelled harness bypassed those upstream
refusals and the fixtures' separately stale dipole/kinetic provenance checks
without changing payload values.  On hBN, exact and historical-debug runs of
the same head function measured

```
exact:       <v> = 1255.0701670167293, <W> = 292.9480574792446 a.u.
sobol_debug: <v> = 1253.1663652570626, <W> = 292.9759753291651 a.u.
```

With `Omega=244.08179778267163 bohr^3`, `Nk=18`, and 16 occupied
spinor bands, the quadrature-only decomposition predicts

```
Delta Sigma^X              = -5.895688054 meV  (occupied)
Delta (SX - X)             = +5.982143975 meV  (occupied)
Delta sigSX = -Delta<W>/(Omega Nk) = +0.086455921 meV  (occupied)
Delta sigCOH               = -2.991071987 meV  (all bands)
```

The controlled output agrees with that decomposition to
`9.280125595e-7 eV`, within the hBN fixture's `1e-5 eV` tolerance.  This proves
the quadrature algebra but is deliberately **not** called the actual fixture
move: the production fixture never reaches the owner, and a bypassed run is
not refreeze authority.

Evidence:

- canonical refusals:
  `runs/DEV/102_bispinor_orchestrator_2026-09-02/lane_BULK/logs/step4_si_canonical_exact.log`
  and `step4_hbn_exact.log`;
- controlled exact/debug hBN arms:
  `runs/DEV/102_bispinor_orchestrator_2026-09-02/lane_BULK/evidence/step4/hbn_exact_bypass5/`
  and `hbn_sobol_debug_bypass4/`;
- parser-backed calculation:
  `runs/DEV/102_bispinor_orchestrator_2026-09-02/lane_BULK/evidence/step4/hbn_quadrature_decomposition.txt`.
- Si override control and latent native diagnostic:
  `si_override_eqp_ab.txt`, `si_override_sigma_ab.txt`,
  `logs/step4_si_native_exact.log`, and
  `logs/step4_si_native_sobol_debug.log` in the same lane evidence tree.

To close this row, repair or replace the two fixture band windows with
certifiably closed inputs, rerun exact versus the frozen reference without any
bypass, confirm the same decomposition within each fixture's tolerance, and
only then re-freeze.
