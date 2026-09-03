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

## 2026-09-02 historical attribution follow-up

The requested numerical bisect could not cross the historical native-library
boundary without violating the measurement contract.  The mandatory runtime
contains the `8576f9f9` `src/ffi` tree (`8d34f444...`).  The last commit before
the kin-ion provenance refusal, `0b43d0a6`, has `f380b216...`; the frozen
reference ancestry is older again.  The first compatible first-parent commit
is `60cf0273`.  Both the `sigC` (`max |delta| > 0.05 eV`) and direct-field
bisects therefore terminate with the interval
`1e64d83a..60cf0273^` skipped, rather than naming a first bad commit.  The
reference header identifies `dd727216` as the generating tree and `1e64d83a`
as the freeze commit.  Bisect records and every probe are under
`runs/DEV/320_bisp_fixture_drift_20260902/`.

The executable half of the history is decisive:

| comparison (B - A) | `sigX` | `sigC` | `sigXC` | direct field |
|---|---:|---:|---:|---:|
| frozen reference -> `60cf0273` | +0.003954 / 0.002263 / 0.007798 | +0.510978 / 0.210189 / 0.803370 | +0.514932 / 0.210204 / 0.804859 | +0.277667 / 0.447171 / 1.434381 |
| `60cf0273` -> `837ed531` | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| `837ed531` -> `8576f9f9` | 0 / 0 / 0 | -0.001282 / 0.001591 / 0.004073 | -0.001281 / 0.001591 / 0.004073 | 0 / 0 / 0 |

Each cell is mean / population standard deviation / maximum absolute move in
eV over 270 rows.  The post-`837ed531` move first appears between the last
unchanged box-plan-only commit `2fcf81ab` and the first runnable certified
shared-MPA route `0140d997`; intermediate `a751d10b` fails with an unexpected
`B_odd_p` keyword, while `5a0326a7` and `5908434f` refuse because the measured
runtime-noise bound `6.28194e-07` exceeds `5e-07`.  `0140d997` and `8576f9f9`
are identical.  The extra move is therefore the separately requested box-rule
physics change, not part of the old-reference drift; it moves the largest
printed `eqp1` row by 4.049637 meV and the reported effective-H gap from
3.41116 to 3.41286 eV.

The direct-field decomposition also corrects the earlier hypothesis.  The
component-aware receipt at `837ed531` authenticates `Hdir = V_H + H_T`, but
this nonmagnetic fixture has `max |H_T| = 2.110e-35 eV`.  Thus the 1.434381 eV
maximum is entirely the scalar `V_H` representation/artifact move, not the
addition of transverse Hartree.  Holding the exact pre-live artifact fixed
across `89db5652 -> 62f1fa22` measures zero in every printed column, so the
stored-exact -> live-exact G-space code switch itself is also zero here.  The
archived July artifact contains only `kin_ion`; the regenerated pre-live
artifact also contains exact `v_hartree` and `v_hartree_transverse`, so
`hartree_source=auto` changes from ISDF to stored-exact on the old code.  Its
share at `0b43d0a6` remains unmeasured because that exact tree is on the
incompatible FFI side of the boundary.

The same controlled artifact swap at the compatible live-only source
`62f1fa22` moves no printed Sigma/direct-field value, although off-diagonal
kinetic/ionic changes move the largest `eqp1` row by 0.272168 meV.  It is a
surrogate control, not the prohibited historical A/B.

Candidate rulings are correspondingly bounded.  The VNL velocity-sign fix
`6b3ffc1f` postdates the generating SHA but is already present at the
reference header's exact `5b135f8e` re-verification (0/1620 cells changed),
so it is excluded.  Heads cannot contribute because the deck sets
`head_correction=off` and all three head scalars to zero.  The ordered GN
probe and `Re Omega^2` changes postdate `837ed531`, and this run reports
`Global TRS: enabled`.  The `-2F` replacement landed in this ancestry as
`b0212b6b` (patch-equivalent to `8cd30c79`).  Its old and new expressions are
algebraically equal when the ordered orientation is related by TRS, but its
exact fixture contribution remains inside the unprobed interval.  The
unresolved `sigC` move is confined to that ABI-blocked scalar-charge
q-grid/screening interval, which includes the legitimate TRS-coherent q-grid
reconstruction `07071457`.  The low-memory face-layout zeta changes are not
active because this deck leaves `low_mem_bands` at its false default; the
face-wing fixes are separately excluded by the head-off/zero-head inputs.
No per-commit number is assigned to a remaining candidate in the interval:
doing so would be a guess.

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
