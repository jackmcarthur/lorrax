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

Reference re-pin, 2026-08-05
----------------------------
`eqp_si_ref.dat` was regenerated on 2026-08-05, when the mini-BZ Monte-Carlo
sampler behind this deck's finite-q `v(q, G=0)` body-head table was replaced.
This deck is the only thing in the tree that exercises that path.

What changed and what it means HERE. The old sampler drew mini-BZ points as
`randvals @ bvec.T`, using the columns of `bvec` as a lattice period when the
period is its rows. In general that is a real bias — up to 64% per q on a
skewed cell. THIS CELL IS NOT ONE OF THOSE. Measured (Frontera job 7890650)
against a rejection-sampled ground truth: `bvec.T` here is a signed row-
PERMUTATION of `bvec` (permutation (2,0,1)), so the old draw and the new one
are different point sets from the SAME distribution, and both sit at the MC
noise floor (3.7e-3 and 4.4e-3 max relative over the 63 nonzero q, against a
2.2e-3 self-noise). `allclose(bvec, bvec.T)` is False for this cell and is
the wrong predicate; the row-permutation one is the right one.

So the reference moves for two reasons, neither of which is "the old number
was biased":
  1. RESEEDING — a different MC realisation of an unbiased estimator.
  2. A genuine accuracy improvement: `nmax` 1 -> 3 (BGW `ncell`), scrambled
     Sobol instead of `RandomState(42)`, and BGW's adaptive per-q sample
     count. Head-table error against the same ground truth drops ~8x, from
     4.0e-3 max / 9.2e-4 mean to 4.7e-4 / 1.6e-4.

Measured effect on this deck of the sampling-line change alone, everything
else byte-frozen (Frontera job 7890626, two source arms differing by one
line):

    max |d sigTOT| = 1.63 meV     MAE = 0.147 meV
    max |d sigCOH| = 1.63 meV     max |d sigSX| = 0.12 meV
    VH and Eo bit-identical

That is above the gate's own `atol = 1e-3 eV`, which is why the reference had
to move rather than the tolerance. For scale: the same comparison of an
unmodified `origin/main` arm against the previous (GPU-generated) reference
gives max |d sigTOT| = 0.054 meV — the platform noise floor — so the move is
30x it.

The BGW anchor above is UNCHANGED in status: it was established with a full
BGW `vcoul` body overlay, which bypasses LORRAX's mini-BZ average entirely,
and the move sits inside the 3.2 meV sigTOT spread that comparison already
reported. Re-verifying the anchor against BGW with the new sampler is NOT
done here and remains open.

See `docs/architecture/decisions.md`, 2026-08-05.
