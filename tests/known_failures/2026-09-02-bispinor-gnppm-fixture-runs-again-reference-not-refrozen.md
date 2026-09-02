# The bispinor GN-PPM fixture runs end to end again; its 2026-08-09 reference is stale by 1.43 eV and was NOT re-frozen

**Date:** 2026-09-02
**Branch:** `lane/bisp-n-dynamic-packed-2026-09-01`, cut from
`integ/bispinor-static-cleanup-2026-09-01@837ed531`.
**Evidence:** `runs/DEV/307_bisp_n_fixture_refreeze_20260902/` and
`runs/DEV/308_bisp_n_pytest_union_20260902/` in the sandbox, Perlmutter
JID 57850966, one A100-SXM4-40GB per step.

## What was wrong, and what is fixed

`tests/test_gw_jax_regression.py::test_bispinor_gnppm_matches_reference` —
the ONLY end-to-end gate on the bispinor path — had been red since
2026-08-26 (`f80a5f70`, "refuse kin-ion representation mismatches"). It was
an **ERROR**, not a failure: the session fixture died before Sigma with
`ValueError: kin_ion.h5 has no bispinor provenance`, so the gate produced no
number at all. Three blockers, all now removed:

1. **`kin_ion.h5` had no bispinor provenance.** The shipped file dates from
   2026-07-03 and predates the stamp. Regenerated from the deck
   (`python3 -m gw.kin_ion_io -i bispinor_test.in -o kin_ion.h5 -n 32`); the
   new file carries `bispinor=True`, `soc=True` with its measured
   `soc_provenance`, `wfn_fingerprint`, `k_storage=ibz` (5 irreducible k of
   9) and the pseudopotential list.

2. **The fixture shipped no pseudopotentials**, so `kin_ion_io` refused
   outright ("No pseudopotentials loaded"), and TWO incompatible Mo ONCV
   fully-relativistic PBE pseudos exist in the sandbox — v3.3.0 (2017),
   md5 `4e1c3579…`, and v2.1.1 (2014), md5 `703a6da1…` — with the same
   `z_valence` and projector count and a different `V_NL`. The fixture's own
   lineage decides it: its README names the 640/668 centroid sets of
   `runs/MoS2_FROM_OLDER_SANDBOX/D_60Ry_bispinor/`, whose
   `qe/nscf/nscf.in` is the 60 Ry / `noncolin` / `lspinorb` / 82-band MoS2
   3×3 calculation this WFN is a 34-band truncation of, and whose pseudos
   are the v3.3.0 pair. `Mo.upf` and `S.upf` are now shipped beside the
   deck, so the fixture is self-contained. Full write-up:
   `runs/DEV/307_bisp_n_fixture_refreeze_20260902/PSEUDO_PROVENANCE.md`.

3. **The harness could not parse a bispinor `sigma_diag.dat`.**
   `tests/harness.parse_eqp_rows` required a `VH=` column; a bispinor deck
   writes `Hdir=` (the aggregate `Hdir = V_H + H_T` of the
   transverse-Hartree split). So even past blockers 1 and 2 the gate could
   only ever have passed on `_assert_matches_reference`'s byte-identity fast
   path and would have raised `No Sigma data rows were parsed` the moment a
   last-ULP drift sent it to the atol comparison. Fixed by one regex
   alternation (`tests/harness.py:681`).

## What is still red, and why it was NOT re-frozen

The gate now RUNS and reports a number instead of exploding:

```
[xmachine] bispinor: max |Δ| = 1.434e+00 vs atol 1e-05
           (1080 cells over, 1080 of 1620 cells differ at all)
```

Per column, over all 270 rows, new versus the 2026-08-09 reference (eV):

| column | mean | std | min | max |
|---|---|---|---|---|
| `sigX` | +0.003954 | 2.26e-03 | +0.001296 | +0.007798 |
| `sigC` | +0.510978 | 2.10e-01 | +0.038809 | +0.803370 |
| `sigXC` | +0.514932 | 2.10e-01 | +0.044099 | +0.804859 |
| `Hdir` vs the old `VH` | +0.277667 | 4.47e-01 | −0.448791 | +1.434381 |

**The move is entirely pre-existing.** Running the identical deck and the
identical regenerated `kin_ion.h5` at `837ed531` (the phase-1 integration
tip, before this lane's diff) and at this lane's tip gives **byte-identical**
`sigma_diag_bispinor_test.dat` and `eqp1.dat` data rows — the deck stays on
the incumbent route on both, and lane N's change cannot reach it. So the
drift belongs to the code movement between 2026-08-09 and `837ed531`, plus
the `kin_ion.h5` regeneration, and this lane did not separate those two.

**It is therefore NOT re-frozen.** A reference is a certificate; re-cutting
it against an undiagnosed 1.43 eV move would make the fixture certify a
state nobody has explained, which is the opposite of what the gate is for
(TASTE.md: a gate firing says two things disagree, not which one is wrong).
Lane E's `cohsex_debug` re-freeze on 2026-09-01 was different in exactly
this respect — it decomposed its 337 eV `VH` move (the ISDF → live G-space
Hartree switch) and 77 % of its sigma move (the VNL velocity sign) BEFORE
re-cutting.

**The re-freeze is the owner's ruling.** If he wants it, it is one copy:

```
cp runs/DEV/307_bisp_n_fixture_refreeze_20260902/01_regen_tip/sigma_diag_bispinor_test.dat \
   tests/regression/bispinor_debug/sigma_diag_bispinor_ref.dat
```

and the numbers above are what it would be blessing. The `Hdir` half of it
has a candidate explanation (the reference's `VH` column predates the
transverse-Hartree split, so it is a different quantity), but the move is
state-dependent with both signs, so that is a hypothesis and not a
decomposition. The `sigC` half has none.

## Status of the row this replaces

`KNOWN_LORRAX_ISSUES.md`'s 2026-09-01 row (lane J) said the gate was red and
registered nowhere, and offered two exits: refuse the route until phase 3,
or regenerate and re-freeze. Phase 3 landed (this lane), the regeneration is
done, and the re-freeze is deliberately not. The row is AMENDED, not closed:
the gate is red for a **different, now-measured** reason.
