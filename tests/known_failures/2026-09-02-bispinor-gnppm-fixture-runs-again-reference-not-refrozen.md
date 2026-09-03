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

**2026-09-02 (integ reconciliation).** The deck now names `w_dyson_solver = local` explicitly: an unnamed solver on a slab bare_transverse deck is derived to `distributed` at parse time (heads always on), which `distrib_la` refuses on this fixture's 1x1 mesh. With `local` the fixture stays on the incumbent route the reference was cut from and keeps failing at the pre-existing drift documented above.

## 2026-09-03 matched-historical-FFI attribution closure

The earlier ABI limitation is now removed.  Six historical `src/ffi` trees
were built on Perlmutter compute nodes in bounded directories under
`/pscratch/sd/j/jackm/wt_codex_fixture_hist/ffi_<hash>/`; every probe canary
prints the exact loaded `.so`.  The 2026-08-09 freeze tree reproduces the
reference exactly: zero changed cells out of 1620.

The drift is fully assigned.  Each numeric cell below is mean / population
standard deviation / maximum absolute move in eV over 270 rows:

| cause (exact parent -> commit or artifact B - A) | `sigX` | `sigC` | `sigXC` | direct field |
|---|---:|---:|---:|---:|
| `e83ff7d5 -> 07451900`, apply WFN reciprocal scale once | -0.000290 / 0.000183 / 0.000591 | -0.000045 / 0.000108 / 0.000402 | -0.000335 / 0.000251 / 0.000981 | +0.003042 / 0.001112 / 0.004631 |
| `9a3409ca -> 4534dc79`, normalize response by source-WFN states | 0 / 0 / 0 | +0.512756 / 0.210173 / 0.803032 | +0.512756 / 0.210173 / 0.803032 | 0 / 0 / 0 |
| `56627937 -> 8f46b0de`, Coulomb-gauge TT metric sign | +0.004243 / 0.002445 / 0.008389 | 0 / 0 / 0 | +0.004243 / 0.002445 / 0.008389 | 0 / 0 / 0 |
| `d235f100 -> 27f1e5c9`, orbit-safe 0.2% GN pole-tail policy | 0 / 0 / 0 | -0.001734 / 0.002545 / 0.013729 | -0.001734 / 0.002545 / 0.013729 | 0 / 0 / 0 |
| `0b43d0a6`, regenerated minus archived `kin_ion.h5` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | +0.274625 / 0.446359 / 1.429902 |

The remaining endpoint residual after summing these rows is at most 2e-6 eV,
below the gate's 1e-5 eV tolerance.  The two anticipated candidates are
negative controls on this TRS fixture: `07071457` moves all printed columns by
exactly zero, and `b0212b6b` (`-2F` -> the two ordered orientations) moves
`sigX`, `sigC`, and the direct field by zero, with only three `sigXC` values
rounding by 1e-6 eV.  Its own P4 ordered-pair gate is accurate to
`4.936e-16`.  The VNL velocity-sign change remains excluded by the reference
header's exact re-verification, and head/wing changes cannot contribute to the
explicit head-off, zero-head deck.

The large `sigC` step is therefore neither q-grid reconstruction nor the
`-2F` removal.  It is the legitimate `4534dc79` correction at
`src/gw/w_isdf.py:1311-1330` in that tree: a four-component kinetic-balance
representation does not double the number of physical source-WFN states, so
the Dyson response prefactor must use `nspinor_wfnfile`, not `nspinor=4`.
Its commit records an alpha-to-zero P4 gate whose four-state red twin is
exactly one half (JID 57527799, step
`lx-Xg4-081151-102145-8903`).  `07451900`, `8f46b0de`, and `27f1e5c9` are
likewise intentional physics corrections: respectively the missing `blat`
conversion at the WFN boundary, the spatial Lorentz-metric minus in
`D_TT=-vP_T`, and the user-ruled, explicitly lossy but Wc(0)-preserving 0.2%
GN tail policy.  No unregistered physics defect is needed to explain the
fixture.

The direct-field result also replaces the earlier hypothesis.  At the exact
pre-provenance tree, changing only `kin_ion.h5` gives the 1.429902 eV maximum
and zero Sigma movement.  `f80a5f70` is the immediate child and correctly
refuses the archived file: “kin_ion.h5 carries no bispinor provenance stamp”.
The remaining 4.631 meV maximum is exactly `07451900`'s WFN reciprocal-scale
correction.  `H_T` is only `2.110e-35 eV`, and the fixed-artifact
`89db5652 -> 62f1fa22` live-G-space Hartree switch is zero.  Thus the old
`VH`/new `Hdir` label change is not the numerical cause; the value remains
scalar `V_H` to displayed precision.

On the merge topology, `520ad4f5 = 60cf0273^1` retains the full direct-field
artifact/scale move but is exactly frozen in all Sigma columns.
`88711198 = 60cf0273^2` is byte-identical to `60cf0273`; the merge brings the
four localized physics fixes onto the first parent and adds no merge-only
numerical change.  The separate `837ed531 -> 8576f9f9` box-rule result is
unchanged: `sigC` -0.001282 / 0.001591 / 0.004073 eV, with a maximum printed
`eqp1` move of 4.049637 meV.

Full build, bisect, refusal, canary, and artifact receipts are in
`runs/DEV/320_bisp_fixture_drift_20260902_histffi/`.  The older
`runs/DEV/320_bisp_fixture_drift_20260902/` remains immutable.

This follow-up still does **not** authorize a re-freeze.  A literal incumbent
tip copy would be:

```
cp runs/DEV/320_bisp_fixture_drift_20260902/probes/8576f9f9edc40ef5e0174e8665ca854d5375a713/sigma_diag_bispinor_test.dat \
   tests/regression/bispinor_debug/sigma_diag_bispinor_ref.dat
```

It would bless, relative to the present reference, `sigX` +0.003954 / 0.002263
/ 0.007798 eV, `sigC` +0.509697 / 0.209653 / 0.801972 eV, `sigXC` +0.513650 /
0.209679 / 0.803461 eV, and direct field +0.277667 / 0.447171 / 1.434381 eV
(mean / standard deviation / maximum absolute).  Do not run that copy.  The
replacement reference should be generated once by the DELETE lane on its
heads-on, one-GPU packed route and copied from that named run instead.
