# AMENDMENT — SYMMETRY LANDING, on `main` (2026-08-08)

Three verified symmetry branches — the q_irr restart machinery, the
generator-contract/pstrf work, and kin_ion store-compressed — plus four
commits this landing owns: a deck pin, a default restoration, a reader
hardening, and one repaired assertion.  The amortized census stopped the
first attempt with nine reds; all nine are accounted for below, eight by a
fix and one by being a different bug wearing the same census row.

| | |
|---|---|
| machine | Perlmutter, lx pool (JID 56501040), 1 node, 4×A100, Shifter, `lx test` |
| module | `LX_BASE_MODULE=lorrax_J070`, jax `0.7.0.dev20260808` |
| trees | `/pscratch/sd/j/jackm/symland_0808/wt_merged` and `.../wt_base` (`995f9e9d`).  `[lx] source tree:` READ on every leg and correct on every leg |
| `.so` pins | the merge_ckpt pair, unchanged: device md5 `c680c229…`, host md5 `91f330c3…` |
| `LORRAX_FFTW3_SO` | PINNED on every leg |
| artifacts | `_reports/suite_merged.xml` (2271 cells, 796429 B), `_reports/suite_base.xml` (2053 cells, 574384 B), `_reports/setdiff.txt`; re-verification in `_verify/` |
| runs | census: merged 702.88 s, base 539.53 s.  Re-verification: seven further legs |

## The census that stopped the first attempt

2271 cells at the merged head against 2053 at `main` @ `995f9e9d`.  Zero
newly green, zero skip movement, **+222 collected all green with no red
among them**, 4 lost from collection (rename twins, one-for-one), 28 carried
red identical by name on both sides — and nine newly red.

### The +222, by file

| cells | file | branch |
|---|---|---|
| 32 | `services/symmetry_maps/tests/test_symmetry_maps_qirr_store.py` | followup |
| 28 | `tests/test_restart_q_storage_key.py` | followup |
| 27 | `services/symmetry_maps/tests/test_symmetry_maps_rename_compat.py` | followup |
| 20 | `tests/test_write_restart_tensors_key.py` | followup |
| 18 | `tests/test_kin_ion_star_broadcast.py` | task4 |
| 18 | `services/symmetry_maps/tests/test_symmetry_maps_qgrid_resolution.py` | followup |
| 15 | `tests/test_restart_qirr_producer.py` | followup |
| 14 | `tests/test_restart_qirr_consumers.py` | followup |
| 14 | `tests/test_bse_w0_ready_gate.py` | followup |
| 14 | `services/symmetry_maps/tests/test_symmetry_maps_closure.py` | followup |
| 7 | `tests/test_qgrid_symmetry_resolution.py` | followup |
| 7 | `tests/test_centroid_distribution.py` | stamp |
| 4 | `tests/test_symmetry_unfold.py` | followup (rename twins) |
| 4 | `services/symmetry_maps/tests/test_symmetry_maps_multiproc.py` | followup |

The 4 lost are `compute_centroid_sym_perm` → `centroid_source_map_and_wrap`
(2) and `unfold_v_q` → `unfold_isdf_operator` (2), each with its
identically-bodied twin above.  Nothing lost coverage.

## THE NINE — what they were, and what they are now

**Eight were one mechanism, and the mechanism was a default that contradicted
its own design doc.**  `restart_q_storage` shipped defaulting to `auto`;
`auto` resolves to `ibz` on any orbit-closed centroid set; `gnppm_debug`'s
399-centroid set is orbit-closed (this landing's own
`test_a_closed_set_resolves_to_ibz_with_bit_identical_tables[gnppm_debug-399]`
asserts it and passes).  So every gnppm restart file in the suite was written
on the five-q wedge, and neither restart reader could take it back: the BSE
side refused it (correctly, and at every process count, not only P>1 as the
registered row claimed), and the GW side asked nothing at all and died at
`gw/cohsex_sigma.py`'s `W_q - V_q` on `(9,399,399)` against `(5,399,399)`.

`DESIGN_symmetry_restart_followup.md` had already ruled the other way — "the
deck key keeps full-BZ storage as the default until the owner rules on
centroid regeneration, and the q_irr path is opt-in per deck" — and
`SPEC_qirr_restart_tensors.md` agrees.  The gate that pinned `auto` justified
it by citing "the design doc (phase-3 deliverable 1: 'Default auto')", a line
that does not appear in the design doc.  The default was a drift, not a
decision, and restoring it is what fixed these eight.

| cell | at the merged head | at the fixed tip |
|---|---|---|
| `test_invariance_gates::test_ibz_equals_full_bz` | `TypeError: sub got incompatible shapes for broadcasting: (9, 399, 399), (5, 399, 399)` | **GREEN** |
| `test_invariance_gates::test_restart_equals_fresh` | ERROR at setup (`gnppm_restart_baseline`) | **GREEN** |
| `test_invariance_gates::test_mu_pad_flip_invariance_gnppm` | " | **GREEN** |
| `test_invariance_gates::test_sc_iteration1_equals_one_shot` | " | **GREEN** |
| `test_invariance_gates::test_fixed_point_frozen_qp_rotations` | " | **GREEN** |
| `test_bse_w0_resolvent::test_w0_resolvent_matches_restart` | `_MunuSlabPlan` refuses `(5,399,399)` vs `kgrid=(3,3,1)` at P=1 | **GREEN** |
| `test_bse_w_omega_chain::test_w_omega_chain_matches_oracle_q0` | " | **GREEN** |
| `test_bse_w_omega_chain::test_w_omega_chain_matches_oracle_finite_q` | " | **GREEN** |

Evidence: `_verify/nine.xml` / `nine.log`, 214.45 s at the fixed tip, same
node, same pins, source-tree line read.

**THE NINTH WAS NOT THE WEDGE, and the census could not tell.**
`test_gw_jax_regression::test_bispinor_gnppm_matches_reference` went red in
the same census and was reported with the other eight.  Re-running it at the
fixed tip separated them: the physics is fine and was always fine —

    [xmachine] bispinor: max |Δ| = 1.000e-06 vs atol 1e-05
               (10.0% of budget, 0 cells over, 50 of 1620 cells differ at all)

— and what failed is `assert "charge-centroid orbit closure failed" in
bispinor_session.stdout`, a string that this branch's own `53908088`
("q-grid closure: one resolution point, and the fallback stops being
silent") replaced with a service-composed announcement.  The property is
unchanged and still true: the bispinor deck's 256-centroid CHARGE set is not
orbit-closed (1 of 2 ops, worst residual 1.436e-01), so its tiles fall back
to the full BZ and say so.  Repaired to assert the announcement that exists,
in three durable parts rather than one literal sentence.

It survived the branch's own verification the same way
`test_loo_accuracy_vs_reference_thresholds` did, which this file already
records: a GPU regression cell that skips on WSL.  **Two instances now, one
mechanism** — a branch reporting itself verified on WSL alone cannot see its
own GPU-cell assertions.

## What retires the flip — and the knob with it

**OWNER RULING, 2026-08-08 ~13:20: `restart_q_storage` should not exist.**
The flip above is a restoration and it lands, but it is TRANSITIONAL and is
documented that way in `gw_config.py` and `docs/input_reference.md`. `full`
is where the key rests until it is deleted; it is not a setting anybody
should tune, and nothing should be built on it.

The ruling is that the mode switch was the wrong shape from the start.  In
the owner's words: symmetries "should not need an auto mode — if symmetries
are not to be used, the wavefunction file should've been generated with no
symmetries."  The WFN file already carries the answer this key asks the deck
for, so a deck-level tri-state was always a second, weaker source of truth
about the same fact — and this landing's nine reds are what a second source
of truth costs when it disagrees with the first.

**THE REGISTERED WORK, which replaces "teach the readers to unfold":**
consolidate the GW and BSE restart-from-GW-tensors paths into ONE
implementation of a few dozen lines in the core drivers, in which

- restart storage FOLLOWS the WFN's own symmetry — the q wedge whenever the
  deck carries symmetries at all, the full BZ when it does not, decided from
  the file rather than from a key;
- both readers ALWAYS unfold, so there is no file shape a reader can be
  handed and refuse.  The refusal this landing adds to the GW reader, and
  `bse_io._MunuSlabPlan`'s existing one, are scaffolding for the interim and
  both come out with the knob;
- `restart_q_storage` is RETIRED entirely — deleted from `_DEFAULTS`, from
  the parse site, from `docs/input_reference.md`, and from the decks that
  currently pin it, including `si_bse_debug`.  One fewer feature to track,
  which is the point of the ruling rather than a side effect of it.

The owner's target size — "a dozen or a few dozen lines in core drivers" —
is a design constraint on that work, not an estimate.  The q_irr format
itself is sound and is not what is being retired: the producer persists the
pre-unfold block so `unfold(stored)` is an identity, the round trip is
bit-exact on four real ranks, and the wedge is 8x on the tensors and 4.155x
on the file.  What retires is the knob, the tri-state, and the two
independent reader paths that made the knob necessary.

Until that lands: the wedge is for runs that DISCARD the restart artifact or
read it through the serial h5py path, `auto`/`ibz` remain per-deck opt-ins,
and `si_bse_debug` keeps its explicit pin as a record of intent.
## THE UNLISTED PRE-EXISTING REDS — 21 cells, finally named

This file's constitution is that a release ships LISTED known-fails, never
unknown ones.  A whole class has been failing that test.  Twenty-one cells
of the import-isolation / library-open-order family are red on `main` and
have been for some time; the phase ledger
(`POST_WAVE_CLEANUP.md` item 2) has described them since 2026-08-07, but
this file — the one the constitution is written in — never named them.
They are named here, with what each one actually is, measured.

**TWO COUNTS WERE IN CIRCULATION, 21 and 18, AND BOTH WERE RIGHT.**  They
answer different questions and the difference is not a measurement
disagreement: this landing's base census and the perf lane's independent
control census of the same commit agree on all 28 reds CELL FOR CELL, with
zero disagreement in either direction (`_reports/suite_base.xml` against
`perf_bse_0808/_reports_reorth/census_base995.xml`).  21 is the number of
cells IN THE CLASS.  18 is the number whose name appears nowhere in this
file at all.  The three in between: `test_vcoul_imports_and_computes_with_no_scipy`
is genuinely listed as a red in the BSE-perf amendment above, while
`test_a_cpu_platform_process_never_opens_the_cuda_library` and
`test_the_cpu_platform_cell_can_fail` appear only in PROSE describing how
the open-order gates were built two-armed — a mention, not a listing.  So
the honest figure for "red and not listed as red" is **20**, and the
listing below covers all 21 so the question cannot be asked again.

### What they are, by measurement — three arms at one tip, one node, one pin set

The class was assumed to be one thing.  It is three, and the arms separate
them cleanly.  Arm A is the full-suite census (xdist `-n 4`).  Arm B is the
same four files collected ALONE (xdist `-n 4`).  Arm C is those four files
single-process (`-n 0`).

| cells | arm A full suite | arm B 4 files, xdist | arm C 4 files, serial | verdict |
|---|---|---|---|---|
| `tests/test_gpu_pinning` ×5 | red | red | red | **real** |
| `services/distrib_la ... test_distrib_la_contract` ×5 | red | red | red | **real** |
| `services/vcoul ... ::test_the_whole_public_surface_answers_with_no_lorrax` | red | red | red | **real** |
| `services/vcoul ... ::test_vcoul_imports_and_computes_with_no_scipy` | red | red | **GREEN** | **xdist artifact** |
| `services/symmetry_maps ... test_symmetry_maps_import_isolation` ×9 | red | **GREEN** | **GREEN** | **collection-scope artifact** |

Evidence: `_verify/iso_xdist.{xml,log}` (arm B, 12 failed / 113 passed),
`_verify/iso_n0.{xml,log}` (arm C, 11 failed / 114 passed), against the
census `_reports/suite_merged.xml` (arm A).  A fourth scope —
`_verify/branch.{xml,log}`, the branch cells plus all of
`services/symmetry_maps` — puts the symmetry_maps count at **4**, which is
the same finding from a third angle: 9, 4, 0 across three collection scopes
is not a property of the code under test.

**The eleven real ones must be LISTED and are, here.**  Diagnosis, from
`POST_WAVE_CLEANUP.md` item 2 and unchanged by this landing: Perlmutter-only,
because there jax lives at `/opt/jax` and the CUDA plugin does not, so
`dep_dirs` cannot hand the stripped `python -S` isolation child a plugin for
the CUDA it inherits a request for.  On WSL both sit in one site-packages
and the child gets both — re-measured at this landing's merged head, where
the whole `services/symmetry_maps` suite runs 259 passed / 1 xfailed.  The
state-discipline fix is DESIGNED AND NOT IMPLEMENTED (item 2, "Class B"):
each service declares the process-env keys it sets and `import_isolation`
builds the child environment from a scrubbed baseline rather than from
`os.environ` wholesale.

| cell | class |
|---|---|
| `tests.test_gpu_pinning::test_lorraxs_loader_opens_the_cuda_library_before_the_host_one` | real, Class B |
| `tests.test_gpu_pinning::test_the_lorrax_loader_open_order_cell_can_fail` | real, Class B |
| `tests.test_gpu_pinning::test_a_cpu_platform_process_opens_only_the_host_library` | real, Class B |
| `tests.test_gpu_pinning::test_the_lorrax_loader_cpu_platform_cell_can_fail` | real, Class B |
| `tests.test_gpu_pinning::test_a_cpu_only_tree_pays_nothing_for_the_rule` | real, Class B |
| `services.distrib_la...::test_the_host_library_is_never_opened_before_the_cuda_one` | real, Class B |
| `services.distrib_la...::test_the_open_order_cell_can_fail` | real, Class B |
| `services.distrib_la...::test_a_missing_cuda_library_is_not_an_error_for_the_host_path` | real, Class B |
| `services.distrib_la...::test_a_cpu_platform_process_never_opens_the_cuda_library` | real, Class B (prose-mentioned above, never listed) |
| `services.distrib_la...::test_the_cpu_platform_cell_can_fail` | real, Class B (prose-mentioned above, never listed) |
| `services.vcoul...::test_the_whole_public_surface_answers_with_no_lorrax` | real, Class B |

| cell | class |
|---|---|
| `services.vcoul...::test_vcoul_imports_and_computes_with_no_scipy` | **xdist artifact** — green single-process at the same scope where it is red at `-n 4`.  Listed above as a "documented pre-existing flake"; it now has a mechanism instead of a shrug |
| `services.symmetry_maps...test_symmetry_maps_import_isolation` ×9 (`test_symmetry_maps_imports_with_the_monorepo_absent`, `test_the_three_absorbed_modules_all_import_clean`, `test_the_injected_quadrature_keeps_psp_out_of_the_import_graph`, `test_the_package_answers_table_questions_without_importing_h5py`, `test_symmetry_maps_still_imports_clean_with_lorrax_on_the_path`, `test_the_isolation_check_can_fail`, `test_the_dead_file_io_import_would_be_caught`, `test_a_psp_import_would_be_caught_even_though_psp_is_a_namespace_package`, `test_the_wrong_copy_of_the_package_is_a_failure`) | **collection-scope artifact** — 9 red / 4 red / 0 red across three collection scopes at one tip |

**AN ATTRIBUTION THAT WAS OFFERED AND IS FALSIFIED, recorded because it was
nearly believed.**  It was proposed that these nine are the q-wedge default
defect, live on bare `995f9e9d` before either landing, and that this
landing's default flip would cure them.  It cannot be, and the check costs
one command: bare `995f9e9d` has **no** `src/gw/restart_q_storage.py` and
**no** `services/symmetry_maps/src/symmetry_maps/qirr_store.py`
(`git ls-tree` returns empty), all four wiring commits — `89f5e297`,
`968548ee`, `4e8cfd70`, `3e9cea10` — are absent from its history, and
`git show 995f9e9d:src/gw/gw_config.py | grep -c restart_q_storage` returns
**0**.  A default that does not exist in a tree cannot redden that tree.
The nine are a collection-scope artifact, which the three arms show
directly, and the flip is irrelevant to them in both directions.

`test_the_isolation_check_can_fail` — the auditor's own negative control —
being red among them is noted and NOT diagnosed here.  It is consistent with
the scope story (the control fails when the check it controls cannot run),
but nothing in these arms measures that, and inventing a mechanism for it is
how the wedge attribution happened.

## The re-verification, leg by leg

Every leg at the fixed tip, same node, same `.so` pair, same
`LORRAX_FFTW3_SO`, `[lx] source tree:` read on each.

| leg | what | result | artifact |
|---|---|---|---|
| 1 | the nine, xdist as the census ran them | 8 of 9 GREEN; the 9th separated (below) | `_verify/nine.{xml,log}`, 214.45 s |
| 2 | the isolation 4 files, **single process** (`-n 0`) | 11 failed / 114 passed | `_verify/iso_n0.{xml,log}` |
| 3 | the same 4 files, xdist `-n 4` | 12 failed / 113 passed | `_verify/iso_xdist.{xml,log}` |
| 4 | the +222 branch cells + all of `services/symmetry_maps` | 415 passed / 6 failed / 2 skipped / 1 xfailed — **none of the 6 among the +222** | `_verify/branch.{xml,log}` |
| 5 | `test_bispinor_gnppm_matches_reference`, repaired | **PASSED**, 74.89 s | `_verify/bispinor2.{xml,log}` |
| 6 | INTEGRATION red twin: a REAL gnppm wedge file | **PASSED** | `_verify/wedge_twin.log` |

### The integration red twin, verbatim

Arm 1 wrote a genuine wedge restart file by setting the key explicitly, and
the resolution announced itself exactly as the census's failing runs did:

    [restart_write] restart_q_storage=auto -> ibz (centroid set is
    orbit-closed (worst residual 5.551e-17 at tol 1.0e-05) and the q path
    reduced)
    [restart_write] V_qmunu (5, 399, 399) 0.01 GB QUEUED
    V_qmunu on disk: (5, 399, 399)     q_storage attr: 'ibz'

Arm 2 handed that file to the GW restart reader.  Before this landing it
produced `TypeError: sub got incompatible shapes for broadcasting` two
hundred lines later.  Now:

    Restart file …/tmp/isdf_tensors_399.h5: V_qmunu, W0_qmunu are stored on
    the IBZ q WEDGE (restart_q_storage=auto or =ibz wrote it), and the GW
    restart reader does not unfold — it would hand a wedge-shaped V_q to a
    full-BZ W_q and subtract mismatched q blocks.
      Re-run the GW leg with restart_q_storage=full (the DEFAULT; the wedge
      is opt-in per deck), or read this file through the serial h5py path in
      bse_io, which unfolds.

The refusal is asserted on the message, not merely on the raising, because
the failure it replaces was a raise with the wrong message.

## Riding under this landing: CGS2

`origin/main` reached `1dbcaeff` (the batched-CGS2 reorthogonalisation
default) while this branch was being fixed, and it is merged in here rather
than landed around.  The lanes are disjoint in substance: CGS2 touches
`solvers/lanczos.py`, `bse_lanczos.py`, `bse_jax.py` (CLI help), its own new
test, and one `docs/dev/env_vars.md` row.  The ONLY file both lanes edited is
that env registry, where each added a different row — `LORRAX_LANCZOS_REORTH`
theirs, `LORRAX_CENTROID_PC_TOL` ours — which git merged cleanly and both of
which are verified present.  `tests/KNOWN_FAILURES.md` was untouched by their
landing, so there was one amendment to write here rather than two to
reconcile.  Legs 1–6 above were run at the POST-merge tip, so the numbers are
the combination's and not an inference about it.
