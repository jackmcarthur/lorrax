# Known test failures — full-suite census

Two censuses live in this file.  The **Perlmutter** one is authoritative for
this tree; the **Frontera** one below it is the historical record from
2026-08-01 and is kept because several of today's reds are only legible
against it.

A release ships LISTED known-fails, never unknown ones: every non-passing
test in every leg is accounted for below, and every "it is the environment"
claim carries the arm in which it comes out FALSE.

---

# AMENDMENT — VCOUL HEAD-SLOT LANDING, on `main` (2026-08-08, owner-approved after the hBN anchor)

The mini-BZ head injection moved to argmin|q+G| with tied-slot mean (the
landing merge has the full case; the adjudication is
`~/lorrax_service_phase/HBN_HEAD_ANCHOR_2026-08-08.md`).  Two frozen arms
go red because the fix is right:

| gate | measured | status |
|---|---|---|
| `tests/test_gw_jax_regression.py::test_hbn_matches_frozen_reference` | 0.128098 eV, measured TOWARD BerkeleyGW (BGW averages every under-cutoff slot; the old rule's bare tied-partners have no BGW counterpart; new rule 1.42x closer on sigTOT, 3.71x on sigCOH, 18/18 k closer at converged ISDF) | red until re-frozen with the delta documented — the re-cut wave's manifest |
| si_bse_debug frozen BSE arm | a further 0.101 meV atop the exchange landing's movement, same deck, same single re-cut | already listed in the exchange amendment; ONE cut covers both landings |

Registered from the anchor, not acted on: BGW averages the Coulomb at ALL
~9000 under-cutoff slots, not only the head slots — LORRAX averages only at
head slots under either rule.  A separate, larger convergence question
(0.17% residual even at 10-25 Ry), now measurable cheaply with the anchor
artifacts left in place.

# AMENDMENT — BSE EXCHANGE-CONJUGATION LANDING, on `main` (2026-08-08, owner-approved)

**Physics moved by design in this landing, and this amendment is the list of
what honestly went red because of it.**  The exchange term of the BSE
Hamiltonian carried the wrong complex conjugation relative to the direct/W
term at ten sites (see the landing merge's message); fixing it restores the
exciton multiplet degeneracies to the ISDF floor (Si dense: triplet spread
1318.5 → 36.2 µeV against BerkeleyGW's 32.4; lowest-20 MAE 3.1610 → 3.1529
meV; control arm reproduces the pre-fix baseline to 0.000000 meV over all
1024 states) and moves eigenvalues up to 144.17 meV across the dense
spectrum.  Frozen references that pin the OLD spectrum are therefore wrong,
not the code.  Full prose manifest:
`~/lorrax_service_phase/NOTE_bse_exchange_fix_refreeze_manifest.md`.

## Newly RED at this landing, by design — awaiting the re-cut wave

| gate | measured | status |
|---|---|---|
| `tests/test_bse_bgw_regression.py::test_bse_matches_frozen_and_bgw` (si_bse_debug frozen arm) | MAE 3.8203 meV / max 9.9236 meV against a 1e-6 eV pin; its BGW *band* arm IMPROVES | red until the re-cut; re-cut owned by the perf lane |
| w-omega `CHAIN_REL_TOL` gate (`tests/test_bse_w_omega_chain_scan.py`) | frozen-reference based; W-chain quantities move with the fix | expected red; re-cut owned by the perf lane |
| exciton-bands warm-cache eigenvalue check | frozen-reference based | expected red; re-cut owned by the perf lane |

Expected to HOLD (structural, not frozen-spectrum): the non-TDA SHAO gates.

**CORRECTION (same day, found by the perf lane's htransform worker and
A/B-attributed here):** `tests/test_bse_vq_interp.py::
test_loo_accuracy_vs_reference_thresholds` was listed above as expected to
hold and is in fact RED — green at `22d99b5e`, red at `724e5bcb`, B-block
LOO median 0.0796 against a 0.022 threshold (both arms re-measured on one
worktree, one pin set).  The cell skips on WSL, which is how "expected to
hold" went unverified.  Mechanism: the landing flipped `vq_interp`'s
b_block conjugation, so the B tensor changed BY DESIGN and the thresholds
were tuned empirically against the old, wrong object.  The re-cut wave
re-derives them on the fixed tree; if the re-derived floor lands far above
the old one, that is a finding about the correct B block's
interpolability, not a tuning artifact — the re-derivation itself
adjudicates.  Evidence: `~/lorrax_bse_perf_2026-08-08/HTRANSFORM_FFT.md`.

**The re-cut prescription is three pins, all mandatory** (evidence:
`~/lorrax_bse_perf_2026-08-08/CONVERGENCE_CENSUS.md`): cut at the gate's own
geometry (1 GPU, px=py=2 — the current reference was cut at P=4 and has
never been seen green by its own gate, a 4.4887 meV provenance artifact);
cut with THIS fix underneath (or the re-cut freezes the wrong spectrum a
second time); and cut at a 400-iteration or rtol-converged Lanczos budget —
the shipped 200 is unconverged on the record deck (4.27 meV off its exact
1024-dim dense reference and MISSING 3 of the true lowest-20 states; 400
sits at 3.9 µeV with zero misses).

## Registered defects (found by the perf lane's convergence census, in passing)

- `davidson --write-eigs` dies at P>1 (`device_get` on non-addressable
  arrays in `write_eigenvectors_stream`).  Flag-path; the suite never
  exercises it, which is how it stayed hidden.
- `bse_feast --feast-ritz` cannot run multi-process at all
  (`_get_feast_runner` closes over non-addressable arrays).  Same class.
- `bse_w_exact.py:634`'s `max_gmres` column is a LOGGING defect, not a
  solver defect: the column is filled with a residual while `:248` discards
  the real iteration count — a two-line driver fix.  It is why the census
  had to lift the cap to measure convergence at all.

Evidence for all three: `~/lorrax_bse_perf_2026-08-08/CONVERGENCE_CENSUS.md`.

# AMENDMENT — BSE-PERF MERGE, `integration/bse-perf-merge-2026-08-08` @ `e69a867f` (2026-08-08)

**This supersedes the merge-checkpoint amendment below for the counts at THIS
head and configuration, and closes nothing.**  Five BSE performance-campaign
branches merged onto `main` @ `602e1d8b` — the feast-runner cache key, the
warm-cache alpha-gate persistability fix, the `bse_setup` scan, the
`w_omega_chain` conversion and the htransform/exciton instrument — plus the
owner-approved exciton-bands rerun-check default flip.  No FFI signatures
changed; the restage-candidate `.so` pair is unchanged and was re-verified by an
in-process loader probe before the census.

| | |
|---|---|
| machine | Perlmutter, JID 56499811, 1 node, 4xA100, Shifter, `lx test` (default xdist geometry) |
| module | `LX_BASE_MODULE=lorrax_J070`, jax `0.7.0.dev20260808` |
| tree | `/pscratch/sd/j/jackm/perf_bse_0808/wt_merge`; the `[lx] source tree:` line was read on every leg |
| `.so` pins | the restage candidate: device md5 `c680c229...`, host md5 `91f330c3...` |
| **`LORRAX_FFTW3_SO`** | **PINNED** (`lorrax_fftw_cray/.../libfftw3.so.mpi31.3.6.10`).  The checkpoint census below pinned none, which is why its FFT-engine block was red on both its legs and is green here |
| artifacts | `/pscratch/sd/j/jackm/perf_bse_0808/_reports_merge/census_e69a867f.xml` — **1996 testcases**, 341287 B; set-diff by `setdiff.py` in the same directory |
| run | one `lx test` invocation, **303.39 s** |

## The census at this head

| leg | pass | fail | skip | error | total |
|---|---|---|---|---|---|
| `tests/` | 1242 | 2 | 61 | 0 | 1305 |
| `services/distrib_la` | 137 | 0 | 32 | 0 | 169 |
| `services/lxkit` | 120 | 0 | 0 | 0 | 120 |
| `services/symmetry_maps` | 150 | 1 | 14 | 0 | 165 |
| `services/vcoul` | 33 | 1 | 0 | 0 | 34 |
| `services/wfn_loader` | 77 | 0 | 15 | 0 | 92 |
| `services/zeta_loader` | 110 | 0 | 1 | 0 | 111 |
| **ALL** | **1869** | **4** | **123** | **0** | **1996** |

The junitxml counts the single xfail among the skips; pytest printed
`4 failed, 1869 passed, 122 skipped, 1 xfailed`.

## SET-DIFF vs this document

| direction | result |
|---|---|
| **newly RED** | **ZERO.**  Every non-passing cell in every leg is listed in this file by name |
| newly GREEN | the red set is a strict SUBSET of the checkpoint amendment's 39.  **Not attributed cell by cell, and deliberately NOT closed anywhere in this file.**  The difference is pins and geometry — the `LORRAX_FFTW3_SO` row above, one node, this collection order — not code.  A census run in the checkpoint configuration would see them red again, and a row marked closed here would lie to it |
| collection delta | **+30 cells against 1966, all green** — 12 `test_bse_w_omega_chain_scan`, 8 `test_exciton_bands_rerun_default`, 7 `test_bse_feast_runner_cache`, 3 `test_bse_nontda` persistability gates |

## The four reds

| cell | class | fingerprint at this head |
|---|---|---|
| `test_bse_setup_qchunk::test_values_are_invariant_to_the_chunk_width` | **P2** | `_maxdiff = 1.3743988419548263` against `< 1e-10` |
| `test_bse_setup_qchunk::test_chunk_width_ulp_spread_is_reported` | **P2** | 5 spreads, first `(2, 2.220446049250313e-15, ...)` |
| `services/symmetry_maps ... test_the_lorentz_mixing_matches_a_dense_numpy_reference[1-1]` | cross-service conftest collision | `ValueError: Memory kinds passed to jax.jit does not match ...` |
| `services/vcoul ... test_vcoul_imports_and_computes_with_no_scipy` | cross-service conftest collision | `RuntimeError: Backend cuda is not in the list of known backends: [cpu, tpu]` |

**The P2 pair got its own A/B**, because `perf/bse-setup-scan` rewrote exactly
the chunking machinery those two cells gauge (`81891dbd`, the FFI eigh arm
walking q in one program per chunk).  Same node, same pins, one detached
worktree at pre-merge `602e1d8b` against the merge head: **2 failed / 22 passed
on both sides, with `_maxdiff` and the full spread table identical to every
printed digit.**  Inherited, not caused — which is independently what
`FIX_bsesetup.md` §4(7) found at the lane itself.

---

# AMENDMENT — MERGE CHECKPOINT, `integration/merge-checkpoint-2026-08-08` @ `6a4f73da` (2026-08-08)

**This supersedes the hBN amendment below for the counts, and nothing else.**
The checkpoint merges `feat/batched-canonical-2026-08-08`,
`fix/ffi-odr-2026-08-08`, and `chore/post-wave-cleanup-2026-08-08` onto
`main` @ `a16a241c`, with both `.so` legs rebuilt from the merged tree
(the ODR fix changed symbol visibility and the phdf5 type tags while the
kchunk conversion, already in `main`, changed the read-dispatch signatures —
no pre-merge pair is valid for this head).

| | |
|---|---|
| machine | Perlmutter, 4-node lx pool (JID 56485516), 4×A100 per node, Shifter, `lx test` |
| module | `LX_BASE_MODULE=lorrax_J070`, jax 0.7.0 |
| trees | `/pscratch/sd/j/jackm/merge_ckpt_2026-08-08/lorrax` (`047f2929`; the one-cell layering fix `6a4f73da` re-verified on its own file, 78/78) and `/pscratch/sd/j/jackm/wt_main_pristine` (`a16a241c`) |
| `.so` pins | the MERGED pair, built from this tree: device md5 `c680c229…`, host md5 `91f330c3…` (`merge_ckpt_2026-08-08/build_{dev,host}/`); baseline leg ran the kchunk_conv pair `main` requires. Neither leg pinned `LORRAX_FFTW3_SO`, so the FFT-engine block is red on BOTH sides and invisible to the set-diff |
| artifacts | `bwrun/suite_base.xml` — **1935 testcases**; `bwrun/suite_merged.xml` — **1966 testcases**; set-diff by `bwrun/setdiff_mc.py` |
| runs | one `lx test` each: baseline 304 s, merged head 572 s |

## The census at this head

| leg | pass | fail | skip | error | total |
|---|---|---|---|---|---|
| `tests/` (lorrax monorepo) | 1178 | 36 | 61 | 0 | 1275 |
| `services/distrib_la` | 136 | 1 | 32 | 0 | 169 |
| `services/lxkit` | 120 | 0 | 0 | 0 | 120 |
| `services/symmetry_maps` | 150 | 1 | 14 | 0 | 165 |
| `services/vcoul` | 33 | 1 | 0 | 0 | 34 |
| `services/wfn_loader` | 77 | 0 | 15 | 0 | 92 |
| `services/zeta_loader` | 110 | 0 | 1 | 0 | 111 |
| **ALL** | **1804** | **39** | **123** | **0** | **1966** |

## SET-DIFF vs `main` @ `a16a241c`

| direction | result |
|---|---|
| newly RED | **1 at `047f2929`, 0 at this tip** — `tests.test_layering::test_only_the_substrate_constructs_a_mesh` flagged the new GATE 10 file building its own cpu Mesh inside a CUDA process; sanctioned as a mesh owner at `6a4f73da` (the one construction `single_device_mesh`/`resolve_mesh` cannot express), file re-run green 78/78 |
| newly GREEN | **9** — the `services.symmetry_maps` import-isolation cells, red at `a16a241c` in full-suite order only; the per-scope skip-honesty fix (`9455e1d8`, "services stop disarming each other") is the mechanism |
| newly SKIPPED / no longer skipped | **0 / 0** |
| collection delta | **+32 new cells, all green** (17 distrib_la batched-scan + shape-algebra, 3 so_acceptance ODR-surface, 2 wfn_loader per-scope skip-honesty, 3 compile-cache agreement, 2 env registry, 5 sigma_output columns); **1 renamed away** — `test_this_gate_did_not_disarm_distrib_las` (red at base) became the two per-scope cells |
| carried red | **38**, identical by name on both sides |

The mixed-process ODR proof was re-run at the merged head against the merged
pair: gate-off arm B 46 P / 1 skip in 12 s; the deployed-pair falsification
arm C died (hung to its 900 s timeout after the cells preceding the kchunk
read path — the arity mismatch BUILD_NOTES predicts), and the kchunk probe
passes on the merged pair while its deployed-pair twin fails.  GATE 10 ALL
PASS; acceptance tier 12/12 (`merge_ckpt_2026-08-08/_reports/`).

**A trap this census recorded on the way** (it cost one invalid leg): a
census suite launched with a stale `LORRAX_CHECKOUT` ran an unrelated
worktree and produced a plausible-looking false red on the BSE anchor —
the eigenvalues matched that worktree's expanded band window exactly.  The
`[lx] source tree:` line in the log names the tree that actually ran;
READ IT before believing any leg.

# AMENDMENT — hBN FIXTURE FREEZE, on `main` @ `6feaa713` (2026-08-07)

**This supersedes the merged-head census below for the counts, and nothing
else.**  The owner authorized freezing the hBN non-cubic COHSEX reference
(2026-08-07); `tests/regression/hbn_cohsex_debug/eqp_hbn_ref.dat` is live and
two new cells gate it.  Every other row of the census below is unchanged, and
that is measured, not assumed — both sides were run in the same session, on
the same node class, with the same pins.

| | |
|---|---|
| machine | Perlmutter, 1 node, 4×A100-SXM4-40GB, Shifter, `lx test` |
| module | `LX_BASE_MODULE=lorrax_J070`, python 3.12 |
| trees | `/pscratch/sd/j/jackm/hbn_freeze/lorrax` (this head) and `.../main_lorrax` (`21d68e06`); source-tree line read on both legs |
| `.so` pins | BUILD_NOTES 2026-08-07, unchanged |
| artifacts | `_reports/hbn_head.xml` — 460979 B, **1935 testcases**; `_reports/main.xml` — 477280 B, **1931 testcases** (neither a 38-byte stub) |
| runs | one `lx test` invocation each: this head 278 s, `main` 334 s |

## The census at this head

| leg | pass | fail | skip | error | total |
|---|---|---|---|---|---|
| `tests/` (lorrax monorepo) | 1202 | 2 | 61 | 0 | 1265 |
| `services/distrib_la` | 127 | 0 | 22 | 0 | 149 |
| `services/lxkit` | 120 | 0 | 0 | 0 | 120 |
| `services/symmetry_maps` | 141 | 10 | 14 | 0 | 165 |
| `services/vcoul` | 34 | 0 | 0 | 0 | 34 |
| `services/wfn_loader` | 75 | 1 | 15 | 0 | 91 |
| `services/zeta_loader` | 110 | 0 | 1 | 0 | 111 |
| **ALL** | **1809** | **13** | **113** | **0** | **1935** |

## SET-DIFF vs `main` @ `21d68e06`

| direction | result |
|---|---|
| newly RED | **0** — none |
| newly SKIPPED | **0** — none |
| no longer skipped | **0** — none |
| newly GREEN | **1** — `services.vcoul.tests.test_vcoul_import_isolation::test_vcoul_imports_and_computes_with_no_scipy` |
| collection delta | **+4, named below**; nothing lost |

The red set at this head is 13 ids, every one of them also red at `main`
(14).  **Nothing went red.**  Nothing changed status from run to skip or
skip to run, so the `skipif` the two new
cells carry never fires — which is the point: the hBN fixture binaries are
tracked, so the guard is a partial-checkout tripwire, not an optional gate.

THE ONE RED THAT WENT GREEN is not a fix and must not be read as one:
`services.vcoul.tests.test_vcoul_import_isolation::test_vcoul_imports_and_computes_with_no_scipy` is a **documented pre-existing flake** of the
cross-service conftest-collision class already characterized below ("services'
conftests fighting over process-global jax/test state"), A/B'd red at the
vcoul branch tip `23f83780` where nothing of this branch exists.  This head
touches no `services/` file at all.  The mechanism is visible in the numbers:
adding 4 collected ids reshuffles the xdist worker assignment, and this cell's
verdict depends on which sibling service's conftest ran first in its worker.
Expect it to flip back.

COLLECTION DELTA, all four PASS:

| new id | why |
|---|---|
| `test_hbn_matches_frozen_reference` | authored — the freeze |
| `test_hbn_mc_average_vcoul_body_moves_sigma` | authored — the liveness control |
| `test_gvec_padded_layout::test_no_fixture_has_a_physical_G_on_the_sentinel_cell[hbn_cohsex_debug]` | **not authored.** `test_gvec_padded_layout` globs `tests/regression/*/WFN*.h5`, so a new fixture WFN enrols itself in the pad-sentinel gates. Free coverage: the hBN ψ table now carries the band-pad sentinel checks too |
| `test_gvec_padded_layout::test_wfn_loader_pads_with_the_shared_sentinel[hbn_cohsex_debug]` | same |

## The two new headline gates

| gate | result | what it means |
|---|---|---|
| `test_hbn_matches_frozen_reference` | **PASS** | **NEWLY LIVE.**  The tree's only NON-CUBIC 3D e2e deck.  `atol = 1e-5 eV`; measured `max \|Δ\| = 0.000e+00` over all 8640 compared cells, 0.0% of budget.  THREE independent runs agree byte for byte on the data lines (2×2/4-process ×2 at generation, 1×1/1-process at the freeze — the mesh the pytest harness actually pins) |
| `test_hbn_mc_average_vcoul_body_moves_sigma` | **PASS** | **NEWLY LIVE, and it is not a freeze.**  A LIVENESS control: flip `mc_average_vcoul_body` and Σ must MOVE.  Measured sigTOT MAE **13.995 meV** / max 49.732 against a floor of 5.0 and an MC seed width of 0.396 meV.  This is the cell that makes the fixture guard the mini-BZ transpose-bug class — Si FCC is structurally blind to it (`bvec.T = P·bvec`, z = 3.0 = noise; hBN z = 293.7) |

RED TWIN, both directions constructed:

* the freeze gate — the reference was perturbed by exactly `2e-5` eV (2× the
  atol) on one `sigTOT` entry and the gate FAILED, reporting `max |Δ| =
  2.000e-05 vs atol 1e-05 (200.0% of budget, 1 cells over, 1 of 8640 cells
  differ at all)`; the reference was then restored from git and its md5
  re-verified at `14035d12ca40a45e392b54528ee3c76c`.
* the control gate — its predicate was run on the real artifacts both ways:
  reference vs the `mc_average_vcoul_body=false` arm gives 13.9948 meV MAE
  (PASS), reference vs itself — the dead-knob case the cell exists to catch —
  gives 0.0000 (FAIL).  It is not a test that cannot fail.

---

# MERGED-HEAD CENSUS — `integration/2026-08-08-services` @ `029da82` (2026-08-07)

**This is the authoritative census for the tree that lands on main.**  All
five wave-1 service branches are merged, the four owner rulings of
2026-08-07 are applied, and the wave-1 shims are deleted.  The
`svc/distrib_la` census below it is kept because every row here is read
against it.

| | |
|---|---|
| machine | Perlmutter, 1 node, 4×A100-SXM4-40GB, Shifter, `lx test` |
| module | `LX_BASE_MODULE=lorrax_J070`, python 3.12 |
| tree | `/pscratch/sd/j/jackm/landing/lorrax`, source-tree line read on every leg |
| `.so` pins | BUILD_NOTES 2026-08-07, unchanged: deployed device `.so`, h200 host `.so` (md5 `4c4422b8…`), `LORRAX_FFTW3_SO` |
| artifact | `_landing_reports/full.xml` — 395 612 B, **1931 testcases** (not a 38-byte stub) |
| run | one invocation, `lx test`, step `lx-Xg4-180142-150155-6026`, 242 s |

## The census

| leg | pass | fail | skip | error | total |
|---|---|---|---|---|---|
| `tests/` (lorrax monorepo) | 1198 | 2 | 61 | 0 | 1261 |
| `services/distrib_la` | 127 | 0 | 22 | 0 | 149 |
| `services/lxkit` | 120 | 0 | 0 | 0 | 120 |
| `services/symmetry_maps` | 141 | 10 | 14 | 0 | 165 |
| `services/vcoul` | 33 | 1 | 0 | 0 | 34 |
| `services/wfn_loader` | 75 | 1 | 15 | 0 | 91 |
| `services/zeta_loader` | 110 | 0 | 1 | 0 | 111 |
| **ALL** | **1804** | **14** | **113** | **0** | **1931** |

Each service leg was ALSO run standalone by marker (`lx test -- -m <svc>`)
and returns the same numbers, so the markers select what they claim:
distrib_la 149/0F/22S, wfn_loader 91/1F/15S, zeta_loader 111/0F/1S,
symmetry_maps 165/10F/14S, vcoul 34/1F/0S.  distrib_la's **149** is the
corrected count — the flagship's `:98` undercounted `-m distrib_la` by 9.

## The headline gates, all GREEN at the merged head

| gate | result | what it means here |
|---|---|---|
| `test_si_production_matches_frozen_reference` | **PASS** | the phase-wide blocking gate: Si COHSEX survived all five services |
| `test_si_production_matches_berkeleygw` | **PASS** | the suite's ONLY external check, still inside |
| `test_si_fast_matches_frozen_reference` | **PASS** | **NEWLY LIVE** — was a skip until the 2a freeze; it now runs |
| `test_bse_matches_frozen_and_bgw` | **PASS** | the phase's one EXPECTED red, closed by the 2b adoption |
| `test_g2_branch_window_tiles_are_frozen` | **PASS** | closed by the 2c Perlmutter re-freeze |
| `test_gnppm_matches_reference` | **PASS** | on the 2d `zeta_rcond = 1e-10` pin |
| `test_bispinor_gnppm_matches_reference` | **PASS** | on the 2d pin |
| `test_ibz_equals_full_bz` | **PASS** | |
| `test_restart_equals_fresh` | **PASS** | |

## FIXED BY RULING (owner-authorized 2026-08-07, verified at this head)

| was | ruling | verified |
|---|---|---|
| `test_bse_matches_frozen_and_bgw` frozen arm — max 7.067e-5 eV, 8/20 cells over the 1e-6 pin | **2b**: adopt the refreeze candidate (md5 `cab1dd48…`) as `bse_eigenvalues_ref.dat`. The vcoul fix 358bb0b reseeds the mini-BZ head MC; on Si the transposed draw is an exact column permutation, so this is a reseed, not a bias. BGW band unchanged within noise (MAE 3.4707 vs 3.4650 meV). `ATOL_FROZEN_EV` was NOT loosened — it is a bit-reproducibility pin | **PASS** in the census above |
| `test_g2_branch_window_tiles_are_frozen` — shape mismatch, crossing-core ladder (100,) vs (98,) | **P1b/2c**: the reference follows the PERLMUTTER grid; re-frozen there. The Frontera difference is STRUCTURAL (4 shape-mismatched tiles, 2 meta rows differing by exactly 2.0 = the node count), τ-node positions disagree from the first element — two quadratures, not one sampled twice. No tolerance applied, deliberately | **PASS**; red-twin probe confirms the gate can still fail |
| the five `zeta_rcond`-unpinned frozen decks | **2d**: pin `1e-10` on gnppm / cohsex_ibz / bispinor (byte-identical to the default today, margins 5.2× / 5.2× / **2.05×**); measured-sweep headers on cohsex + minimax_selfcheck (~3983× margin, left exercising the shipping default). `ZETA_RCOND_DEFAULT` unchanged | gnppm + bispinor gates **PASS** |
| the 20-band fast gate, skipped since it was written | **2a**: freeze `eqp_si_fast_ref.dat` from the named candidate (md5 `d63cb322…`). A skip whose reason said "copy it in to enable this gate" | **PASS**, and it is a self-freeze, never a BGW anchor |
| P1 cross-machine micro-eV pins | already policy-green at 1e-5 (2026-08-07 ruling); unchanged by this landing | in the 1804 |

## The 14 reds, every one accounted for, NONE new

| tests | class | evidence it is not this landing |
|---|---|---|
| `test_bse_setup_qchunk::test_values_are_invariant_to_the_chunk_width` + `::test_chunk_width_ulp_spread_is_reported` | (b) pre-existing, class **P2** below | already characterized and A/B'd at `5bb4368`; the second cell is the first one's instrument and goes red with it |
| 9 × `services/symmetry_maps/tests/test_symmetry_maps_import_isolation.py` + 1–2 × `test_symmetry_maps_emulated_mesh::test_the_lorentz_mixing_matches_a_dense_numpy_reference` | (b) pre-existing, **cross-service conftest collision** | **A/B'd at the symmetry_maps branch tip `4b35c19`** (pre-merge, pre-shim-deletion, 3 services on disk): the SAME set fails there, `services/` tier, `-n 0`. Set-diff vs this head is EMPTY |
| `services/vcoul/tests/test_vcoul_import_isolation::test_vcoul_imports_and_computes_with_no_scipy` | (b) pre-existing | **A/B'd at the vcoul branch tip `23f83780`**: fails there too, alone in its tier |
| `services/wfn_loader/tests/test_wfn_loader_skip_honesty::test_this_gate_did_not_disarm_distrib_las` | (b) pre-existing, ALREADY REGISTERED | the `lxkit._ARMED` process-global collision — POST_WAVE_CLEANUP item 2 ("FIX `_ARMED` to a list — two services arming skip-honesty in one process currently disarm each other"). Also red on WSL at both sides of the shim deletion |

**THE MECHANISM OF THE ISOLATION REDS, since "it is the environment" needs
the arm where it comes out false.**  Run the file ALONE on the same node
and all 9 cells PASS (step `lx-Xg4-180628-165870-4541`, 9 passed).  Run the
`services/` tier and they fail, with or without xdist (`-n 0` reproduces
it).  The subprocess `import_isolation` spawns dies on
`RuntimeError: Unable to initialize backend 'cuda': Backend 'cuda' is not
in the list of known backends: ['cpu', 'tpu']` — a sibling service's
conftest has put CUDA-flavoured jax state in the process, and the stripped
isolation subprocess inherits the request without the plugin.  It is the
same class as the `_ARMED` row above: services' conftests fighting over
process-global jax/test state.  It is NOT a defect in the isolation claim
these cells make, and it is NOT caused by the merge — the A/B at
`4b35c19` is the falsifying arm.

## Shim deletion — the O2 completion criterion

The five wave-1 shims are gone (`file_io/wfn_loader.py`,
`file_io/zeta_loader.py`, `common/symmetry_maps.py`,
`centroid/orbit_syms.py`, `common/density_symmetry_check.py`; 278 lines).
`tests/test_service_path_bootstrap.py` now carries
`test_the_retired_shim_files_are_gone`,
`test_src_no_longer_imports_any_retired_shim_path` (scanning `src/` **and**
`services/`, at every scope) and the red twin
`test_the_retired_path_scan_can_fail`.  `_SERVICE_DOOR_EXCEPTIONS` lost its
three symmetry_maps rows and `_SHIM_CONSUMERS` is now `()`.

---

# Perlmutter census — `svc/distrib_la-2026-08-07` @ `d5cac09` (2026-08-07)

**First Perlmutter full-suite census in this tree.**  The Frontera census
below was the only one that existed, and three of its "fixed in this pass"
re-freezes turn out to be *platform-local* — see class **P1**.

| | |
|---|---|
| machine | Perlmutter, 1 node, 4×A100-SXM4-40GB, Shifter `ghcr.io/nvidia/jax:jax-2025-07-21` |
| module | `LX_BASE_MODULE=lorrax_J070`, jax `0.7.0.dev20260807`, python 3.12 |
| tree | `/pscratch/sd/j/jackm/svc_distrib_la/lorrax`, `LORRAX_CHECKOUT` (source-tree line read on every leg) |
| `.so` pins | `LORRAX_FFI_SO=~/software/lorrax_ffi_2026-08-07/liblorrax_ffi.so`, `LORRAX_FFI_HOST_SO=/pscratch/sd/j/jackm/svc_distrib_la/build_host_h200/liblorrax_ffi_host.so` (md5 `4c4422b8…`), `LORRAX_FFTW3_SO=~/software/lorrax_fftw_cray/stage/lib/libfftw3.so.mpi31.3.6.10` — BUILD_NOTES 2026-08-07 |
| jobids | **56447670** (gpu pool, all legs except F), **56446562** (Milan cpu pool) |
| artifacts | `/pscratch/sd/j/jackm/svc_distrib_la/_reports_step6/` — one `.log` + one `.xml` per leg, sizes quoted below |

> **Which commit the legs ran at.**  Every leg below ran at `e9340d1`.  The
> two commits since — `d5cac09` (bench baselines) and `efdbf9a` (this file)
> — touch four `.json` under `services/distrib_la/bench/baselines/` and one
> `.md`.  `pyproject`'s `norecursedirs` excludes `bench`, so neither is
> importable or collectable, and `pytest --collect-only` at `e9340d1` and at
> `efdbf9a` returns the **same 1441 ids, diff empty**.  Legs B, E and E2
> were additionally RE-RUN at `efdbf9a` and came back byte-identical
> (130/108P/22S, 120/120P, 130/127P/3S).

> **AND WHICH COMMIT THE FIXES WERE VERIFIED AT.**  `7a1d64f` (B1) and
> `f7c1b17` (B2) land after this census.  Every leg in *§ FIXED AFTER THE
> CENSUS* below was re-run at `f7c1b17` on the same pins and the same node
> class, artifacts in `_reports_fix/`; the leg table above is left as the
> census measured it, at `e9340d1`.

## Verdicts by leg

| leg | invocation | collected | passed | failed | error | skipped |
|---|---|---|---|---|---|---|
| **A** full suite, services deselected | `lx test -- tests/ --no-services -q -rs -p no:randomly` | 1191 | 1120 | 8 | 1 | 62 |
| **A2** full suite WITH services | `lx test -- -q -rs -p no:randomly` (testpaths = tests + services) | 1441 | 1326 | 9 | 22 | 84 |
| **B** full-suite marker leg | `lx test -- -m distrib_la -q -rs` | 130 | 108 | 0 | 0 | 22 |
| **E** lxkit by path | `lx test -- services/lxkit/tests -q -rs` | 120 | 120 | 0 | 0 | 0 |
| **E2** distrib_la by path | `lx test -- services/distrib_la/tests -q -rs` | 130 | 127 | 0 | 0 | 3 |
| **C** device-hungry, 4 emulated host devices | `lx run env XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu python3 -m pytest <9 files>` | 146 | 143 | 1 | 0 | 2 |
| **C2** non-square refusal, 3 emulated devices | `… device_count=3 … -k nonsquare` | 1 | 1 | 0 | 0 | 0 |
| **C3** two-device cells | `… device_count=2 … tests/test_charge_zeta_route.py` | 15 | 7 | 0 | 0 | 8 |
| **D** `extra` tier | `lx test -- -m extra -q -rs` | 26 | 23 | 1 | 0 | 2 |
| **F** L-c REAL 4-process CPU 2×2 | `lx run -N 1 -G 4 -n 4 env JAX_PLATFORMS=cpu python3 …test_distrib_la_multiproc.py --mesh 2x2` | 14 cells | 12 | **0** | 0 | 2 |
| **G** L-c REAL 4-process GPU 2×2 | `lx run -N 1 -G 4 -n 4 … --mesh 2x2` (serialized) | 13 cells | 10 | **0** | 0 | 3 |

Legs F and G are the hostile-geometry / real-multi-process tier and are the
only legs that exercise ScaLAPACK, SLATE and cuSOLVERMp on four real ranks.
Both are clean; their residuals are quoted in
`docs/services/distrib_la.md` § Performance.

### Isolation and falsification legs (not census legs — evidence)

| leg | invocation | result | what it settles |
|---|---|---|---|
| `iso_reds` | every leg-A red, ONE process, `lx run … -m pytest` | 58 / **54 P / 4 F** | 4 of the 9 leg-A reds survive isolation; 5 do not |
| `iso_bse` | the five BSE session files, ONE process | 35 / **35 P** | the A2 21-error cascade is not per-test |
| `xdist_arm` | the SAME six files under `lx test` (xdist, 4 workers) | 24 / 7 P / **17 E** | the cascade is the LAUNCHER |
| `base_xdist` | `xdist_arm` at the branch base `96a6399` | 24 / **24 P** (2 runs) | the cascade is a REGRESSION ON THIS BRANCH |
| `falsify_wfnloader` | `test_no_ffi_at_P_gt_1…` with the FFI pins UNSET | **1 P** | that red is the pin |
| `falsify_aot` | `device.memory_stats()` under `XLA_PYTHON_CLIENT_ALLOCATOR=platform` vs `=bfc` | `None` vs a 10-key dict | that red is the allocator |
| `bisect_fileio` | `tests/test_file_io.py` on the CPU platform at `96a6399` / `b3f3675` / `32e61fe` / HEAD | 42P·1S / 42P·1S / **ABORT** / **ABORT** | names the commit |
| `loadorder` | HEAD, CPU platform, `LORRAX_FFI_SO` unloadable vs pinned | **42P·1S** vs ABORT | names the LINE |
| `bisect_xdist` | `xdist_arm` at `78ddcee` / `6920171` | 24/24 P vs 11P·2F·11E | names the second commit |
| `cvd_probe` | each xdist worker prints its own `CUDA_VISIBLE_DEVICES`, at `78ddcee` and at HEAD | `0,1,2,3` vs `0,0,0,0` | names the mechanism |

---

## FIXED AFTER THE CENSUS — the two reds this branch made

Both are fixed on this branch and re-verified; the diagnosis below is the
census's, unchanged, and each row now carries the arm that closes it.  The
rows stay here rather than disappearing: a census that deletes what it
found cannot be audited against the next one.

**Re-verification, branch tip `f7c1b17`, same node class, same BUILD_NOTES
pins, artifacts in `/pscratch/sd/j/jackm/svc_distrib_la/_reports_fix/`
(one `.log` + one `.xml` per leg):**

| leg | census @ `e9340d1` | fix tip @ `f7c1b17` | reference it must match |
|---|---|---|---|
| `test_file_io.py`, CPU platform, 4 emulated devices | **ABORT** | **42 P / 1 S** | base `96a6399`: 42 P / 1 S |
| cvd probe, 4 xdist workers | `'0','0','0','0'` | **`'0','1','2','3'`** | `78ddcee`: `'0','1','2','3'` |
| xdist arm, 6 gnppm-session files | 7 P / 17 E | **24 / 24 P** | `78ddcee`: 24 / 24 P |
| **B** full-suite `-m distrib_la` | 130 / 108 P / 0 F / 22 S | **140 / 118 P / 0 F / 22 S** | +10 new cells, 0 F, skips unchanged |
| **E** lxkit by path | 120 / 120 P | **120 / 120 P** | unchanged |
| **E2** distrib_la by path | 130 / 127 P / 3 S | **140 / 137 P / 0 F / 3 S** | +10 new cells, 0 F, skips unchanged |
| **A** full suite, services deselected | 1191 / 1120 P / 8 F / 1 E / 62 S | **1211 / 1143 P / 6 F / 0 E / 62 S** | 0 newly red, 3 newly green |
| **A2** full suite with services | 1441 / 1326 P / 9 F / 22 E / 84 S | **1471 / 1381 P / 6 F / 0 E / 84 S** | 0 newly red, **25 newly green** |
| WSL full suite (jax 0.9.1, no FFI) | 1441 / 95 red | **1471 / 95 red** | set-diff EMPTY both directions |
| `python3 tests/test_layering.py` | 75 / 75 | **75 / 75** | unchanged |

The six reds left in A/A2 are P1 (3), P2 (2) and P3 (1) below — every one
pre-existing or an owner row, none from this branch.  Leg B is the tension
point and it holds: a CUDA-capable process still opens CUDA first (the
`-m distrib_la` leg's SLATE cells are green, and
`test_the_blaspp_the_cuda_slate_calls_can_see_the_device` passes there),
while the CPU-platform leg opens the host library and nothing else.

### **B1 — FIXED by `7a1d64f`: `_open_cuda_before_host` broke the host phdf5 path**

| | |
|---|---|
| tests | `tests/test_file_io.py` — `test_read_slab_without_shape_rounds_up_to_the_mesh`, `test_slabio_implicit_pad_write_and_zero_padded_read` and every SlabIO write cell after them, on any leg whose jax platform resolves to **cpu** |
| class | (a) real defect, introduced by this branch — **FIXED, `7a1d64f`** |
| covering leg (census) | none — the GPU legs (A/A2) did not reach it, the CPU leg died in it |
| covering leg (now) | the CPU-platform `test_file_io` leg itself, **42 P / 1 S** at `f7c1b17`, plus the two-armed loader cells on every machine |

`src/ffi/common/ffi_loader.py:576-613` (commit **`32e61fe`**, "the two FFI
libraries share their SLATE, and the first one opened wins") makes
`get_lib("cpu")` call `_open_cuda_before_host()`, which `dlopen`s the CUDA
FFI library `RTLD_GLOBAL` first.  That is correct for SLATE — and it is
measured — but in a process whose jax backend is **cpu** the host phdf5 slab
handlers are then answered across the SONAME boundary.  Symptom at the
handler: the read refuses with

    phdf5 read: logical slab out of bounds
      extent=[2,4,1,6]
      offset_base=[0,0,0,4596944070643295330]
      valid_shape=[3,6,6,4609783128842618077]

Those two integers are IEEE-754 float64 bit patterns (≈0.19 and ≈1.87) read
as `int64` — a different handler's argument layout, which is what "the wrong
library answered" looks like.  Where it does not refuse, it aborts inside
the async writer thread (`common/async_io.py:135` → `_slab_io_ffi.py:1749` →
pjit → `Fatal Python error: Aborted`).

**BISECT, one fast deterministic leg (`tests/test_file_io.py`, CPU platform,
4 emulated host devices), all four arms on the same pins:**

| commit | | result |
|---|---|---|
| `96a6399` | branch base | 42 passed / 1 skipped |
| `b3f3675` | the commit *before* the load-order rule | 42 passed / 1 skipped |
| `32e61fe` | **the load-order rule** | 3 failed, then `Fatal Python error: Aborted` |
| `d5cac09` | HEAD | hang → 300 s wall |

**FALSIFICATION (the arm where the hypothesis comes out false), at HEAD:**
point `LORRAX_FFI_SO` at a path that cannot be `dlopen`ed, so
`_open_cuda_before_host`'s best-effort `get_lib("CUDA")` raises and is
swallowed and the process stays host-only — **42 passed / 1 skipped**, byte
for byte the base result.  Same leg with the CUDA `.so` pinned: 300 s wall.

**FIXED, `7a1d64f`** — "the CUDA-first pre-open is for processes that can
use CUDA, and only those".  Not a revert: the SLATE SONAME collision is
real and `32e61fe`'s evidence stands, so a CUDA-capable process still opens
CUDA first.  The pre-open is now gated on `_process_can_use_cuda()` —
`JAX_PLATFORMS` resolved first-entry-wins plus a visible NVIDIA device
(`CUDA_VISIBLE_DEVICES=""` explicitly masked, else a `/dev/nvidia*` node),
the same two signals in the same order as `runtime._gpu_is_present`.  It is
truthful AT LOAD TIME, which is why it is not `jax.default_backend()`: that
INITIALIZES the XLA backend, so asking it inside a loader call would make
the loader decide the process's platform.  Applied in BOTH loaders
(`src/ffi/common/ffi_loader.py`, `services/distrib_la/src/distrib_la/loader.py`).

VALIDATION: `tests/test_file_io.py`, CPU platform, 4 emulated devices, the
CUDA `.so` **pinned** (not the census's unloadable-`.so` falsification arm)
— **42 passed / 1 skipped**, the base `96a6399` result, at
`_reports_fix/fix_fileio.xml`.  The CUDA arm is unharmed: leg B is 140
cells / 0 failed / 22 skipped and
`test_the_blaspp_the_cuda_slate_calls_can_see_the_device` passes in it.

The four order cells are two-armed now, both sides with a red twin
(`test_a_cpu_platform_process_never_opens_the_cuda_library` +
`test_the_cpu_platform_cell_can_fail`, and the same pair for lorrax's
loader in `tests/test_gpu_pinning.py`), plus an 8-row table per loader
constructing every input of the capability gate.  The CPU-platform arm's
red twin is the load-bearing one: without it that cell stays green on any
machine with no CUDA library to find, which is every WSL leg.

**The ABORT itself was a SECOND defect.  It is now FIXED — see L1, and
note that the mixed state it feared is now survivable: the CPU-platform
leg is green with this capability gate DISABLED, against an abort with
the pre-fix `.so` pair.**

### **B2 — FIXED by `f7c1b17`: the xdist CONTROLLER narrowed the workers' preset**

| | |
|---|---|
| tests | leg A: `test_bse_bgw_regression::test_bse_matches_frozen_and_bgw`, `test_bse_w0_resolvent::test_wq_resolvent_matches_restart_finite_q`, `test_restart_pad_roundtrip::test_restart_mu_pad_roundtrip` (worker crash).  Leg A2: those plus `test_bse_kgrid` (2) and a 21-error cascade over `test_bse_dense_reference` (12), `test_bse_stack_matvec` (3), `test_bse_w_omega_chain` (2), `test_bse_matvec_opts` (2), `test_bse_w0_resolvent` (2) |
| class | (a) real defect, new on this branch — **root-caused to `6920171`**, second symptom layered on at `32e61fe`; **FIXED, `f7c1b17`** (and `32e61fe`'s share by `7a1d64f`) |
| covering leg (census) | the SINGLE-PROCESS leg: `iso_bse` **35/35 pass**, `iso_reds` passes all of these |
| covering leg (now) | the xdist leg itself — **24 / 24 P**, and leg A2's 22-error cascade is 0 |

Every one of these cells is green in one process and red under `lx test`
(1 node, all GPUs, 4 xdist workers, one GPU pinned per worker).  The
session fixture dies with no traceback immediately after

    [restart_write] W0_qmunu (9, 399, 399) 0.02 GB QUEUED in 0.0 s
    [SlabIO.close] draining 1 pending writes for isdf_tensors_399.h5 …

and, separately, reads come back
`ValueError: INVALID_ARGUMENT: phdf5 read: ctx_handle is null`.

MEASURED, same six files, same launcher, same pins, two runs per arm:

| commit | result |
|---|---|
| `96a6399` (base) | 24 / **24 passed** |
| `b3f3675` | 8 passed / 16 error — but `RESOURCE_EXHAUSTED: Failed to allocate 19.9 GiB on device ordinal 0` |
| `d5cac09` (HEAD) | 7 passed / 17 error — the SlabIO drain death |

TWO regressions are stacked here.  Both are now root-caused.

**The memory one is `6920171` ("tests/conftest: the GPU pin was conditional
on xdist, and that hid a leg"), and it puts EVERY xdist worker on GPU 0.**
Bisected on the same arm, then measured directly with a throwaway probe
test that prints its own `CUDA_VISIBLE_DEVICES` (written into the clone for
one run, deleted; both trees verified clean afterwards):

| commit | gw0 | gw1 | gw2 | gw3 | xdist arm |
|---|---|---|---|---|---|
| `78ddcee` (before) | `'0'` | `'1'` | `'2'` | `'3'` | 24 / **24 passed** |
| `6920171` (after) | `'0'` | `'0'` | `'0'` | `'0'` | 11 P / 2 F / 11 E, `RESOURCE_EXHAUSTED … 19.20GiB on device ordinal 0` |
| `d5cac09` (HEAD) | `'0'` | `'0'` | `'0'` | `'0'` | 7 P / 17 E |

MECHANISM.  The pin used to be written only when `PYTEST_XDIST_WORKER`
started with `gw`.  Unconditional, the xdist **controller** — which has no
worker id, so `pin_one_gpu` returns `devs[0]` — now writes
`CUDA_VISIBLE_DEVICES="0"` into its OWN environ at `tests/conftest.py`
module scope.  The workers inherit that environ, so each of them sees
`preset="0"`, `_visible_gpus` returns a ONE-element list, and
`int(worker_id[2:]) % 1 == 0` for all four.  The fan-out
`tests/harness.py:54-78` documents is dead, and four gnppm sessions land on
one 40 GiB A100 instead of four.

`pin_one_gpu` itself is correct and its unit tests pass — they construct
the preset by hand and never see the controller's write.  The gap is the
caller, and `tests/test_gpu_pinning.py` is where the twin belongs: *four
worker ids must map to four distinct devices even when the controller has
already pinned one*.

**FIXED, `f7c1b17`** — "tests/conftest: the xdist CONTROLLER must not
narrow what its workers inherit".  Not a revert: `6920171`'s three reasons
and its 11 cells are untouched, and a plain non-xdist run is still pinned
(that is the leg `6920171` existed to un-hide).  The controller — and only
the controller — takes no device, detected with pytest-xdist's own
predicate: no `PYTEST_XDIST_WORKER` **and** `config.option.dist != "no"`
(`harness.is_xdist_controller`).  `-n 0` and a missing xdist plugin both
leave `dist == "no"`, so both stay pinned.  The pin moves from conftest
module scope to `pytest_configure`, which is where `config` exists and is
still before collection — hence before the first test-module import, hence
before jax.

VALIDATION, all at `_reports_fix/`:

| arm | result |
|---|---|
| cvd probe, 4 xdist workers | gw0..gw3 = **`'0','1','2','3'`** (`fix_cvd.log`) |
| the same probe against the PRE-FIX conftest (`git show 35f3e06:tests/conftest.py`) | **`'0','0','0','0'`, 1 distinct device of 4 — RED ARM FIRES** (`fix_redarm.log`) |
| xdist arm, the 6 gnppm-session files | **24 / 24 passed** (`fix_xdist.xml`) |
| leg A2 | the 22-error cascade is **0**, 25 cells newly green, 0 newly red |

`tests/test_gpu_pinning.py` grows the controller-does-not-pollute twin —
the census probe frozen into the suite: it copies this conftest + harness
into a tmp rootdir (never writing into the checkout), spawns the real
4-worker arm and reads each worker's own `CUDA_VISIBLE_DEVICES` back out of
a file (a worker's stdout does not reach the controller under xdist, `-s`
included — measured).  It scrubs `PYTEST_XDIST_*` from the child env,
because under `lx test` the cell runs INSIDE a worker and the child's
controller would otherwise inherit `PYTEST_XDIST_WORKER=gw2` and pin
`devs[2]` for all four — measured, and the same defect shape from the other
direction.  Beside it: `test_the_controller_takes_no_device_at_all`, its
red twin, and a 7-row `is_xdist_controller` table whose `("", "no")` row is
the non-xdist leg that must still be pinned.

> The compile cache is NOT the cause of the `ctx_handle is null` reads.
> Arm run: same files with `ISDF_JAX_CACHE_DIR=""` → 4 passed; but the
> control (cache ON, same isolation) is *also* green, so the arm is not
> discriminating and the cache hypothesis is unsupported.  Recorded so
> nobody re-derives it.


## SHIP-LISTED FAILURES

### **L-3 — cuSOLVERMp `eigh` with `compute_evecs=False` fails at every `n`** (library defect, now REFUSED at resolve time)

| | |
|---|---|
| tests | none go red: the combination is refused before it can run.  Contract cells `test_resolve_cusolvermp_eigh_refuses_compute_evecs_false`, `test_resolve_cusolvermp_eigh_compute_evecs_true_is_unchanged`, `test_cusolvermp_wrapper_refuses_compute_evecs_false_without_a_gpu` |
| class | (e) third-party library defect — cuSOLVERMp 0.7.2, not LORRAX code |
| evidence | `XlaRuntimeError: INTERNAL: cusolverMpSyevd failed: status=7` (`CUSOLVER_STATUS_INTERNAL_ERROR`) at **n = 64, 256, 1024, 4096**, real 4-process 2×2 CUDA mesh, jobid **56447670**.  `/pscratch/sd/j/jackm/svc_distrib_la_perf/_reports/perf_gpu2x2_decomp.json` rows `"g2. distributed_eigh jobz='N'"`, reproduced independently in `perf_gpu2x2prof_decomp.json` (the `LORRAX_FFI_PROFILE=1` leg) |
| covering leg | the step-6 eigh investigation's `gpu_decomp` and `gpu_decomp_prof` legs |

`compute_evecs=False` (jobz='N') is a **documented parameter** of
`_cusolvermp.distributed_eigh` and it has never worked.  It is not a
wrapper bug: `cusolverMpSyevd_bufferSize` **succeeds** for jobz='N', and
`src/ffi/cpp/cusolvermp/eigh_ffi.cc:106` is a one-line pass-through
(`const char jobz = compute_evecs ? 'V' : 'N'`), so the library sizes the
eigenvalues-only solve and then fails to run it.

**NOT CHASED**, deliberately: no workaround flag was identified in 0.7.2
and chasing a vendor library bug is out of scope for this branch.

**REFUSED, PERMANENTLY, NOT AS A STOPGAP.**  The owner confirms
(2026-08-07) that LORRAX wants `compute_evecs=True` in every case they
can think of, so this parameter's only remaining value on this backend
was a route to an unexplained `INTERNAL_ERROR` three call frames deep.
`resolve.resolve_backend` guard **2c** refuses it, and
`_cusolvermp.distributed_eigh` refuses it again as its first statement —
both, because `Plan.__call__` forwards `**kwargs` straight to the wrapper
(`plan.py:318`) and a resolve-time-only rule would have a hole in it.
A caller who wants eigenvalues only should pass `compute_evecs=True` and
ignore `Q`, or use `jnp.linalg.eigvalsh`.

### **L-4 — SLATE `eigh` SIGSEGVs at n ≥ 4096 on a multi-rank CUDA mesh** (library defect, now REFUSED at resolve time)

| | |
|---|---|
| tests | none go red: the combination is refused before it can run.  Contract cells `test_resolve_slate_cuda_eigh_refuses_at_4096`, `test_resolve_slate_cuda_eigh_still_resolves_at_2048`, `test_resolve_slate_cpu_eigh_at_4096_still_says_the_L2_thing` |
| class | (e) third-party library defect — the **CUDA sibling of L-2** (SLATE host `heev`), same library, same routine family |
| evidence | `srun: error: nid001088: task 0: Segmentation fault` → step exit **139**, all four ranks down.  jobid **56457930**, `/pscratch/sd/j/jackm/svc_distrib_la_perf/_reports/gpu_slate4096_segv.log` — a leg run for the sole purpose of producing this artifact |
| covering leg | the step-6 `gpu_slate4096` leg.  **NOT** `gpu_cross_size.log:51`, which the brief flagged as overwritten: that file now shows the *skip* line, not the crash |

The size sweep **skips** this cell (`gpu_cross_size.log:35`, "slate nq=0
n=4096 -> SKIPPED (known SIGSEGV at n>=4096)") because a crash there takes
the other 20 rows down with it.  The dedicated leg exists so the skip is
backed by an artifact rather than by a memory.

**SIZE-SCOPED, not a removal.**  SLATE eigh RETURNS on the same 2×2 CUDA
mesh at every smaller size measured — 0.401 / 0.546 / 1.387 / 5.444 s at
n = 64 / 256 / 1024 / 2048 (jobid 56447670).  Refusing the whole backend
would delete a working route, and `distributed` eigh on **ROCm** maps to
slate, so "delete slate" is not on the table either.  Nothing between 2048
and 4096 was tried, so the true threshold is somewhere in (2048, 4096];
4096 is the smallest size measured to crash, and erring toward refusal is
correct when the failure mode is a SIGSEGV with no Python traceback.

**A 1×1 CUDA mesh at n ≥ 4096 is UNMEASURED and is NOT refused.**  The
crash was only ever produced multi-rank.  Guard 2d says so, and a contract
cell pins it — this package refuses what someone has watched fail, not
what seems likely.

### **L1 — FIXED by `fix/ffi-odr-2026-08-08`: the two platform `.so`s cross-wired their phdf5 through RTLD_GLOBAL**

| | |
|---|---|
| tests | `services/distrib_la/tests/test_so_acceptance.py` check 6 + `test_each_library_exports_only_its_sanctioned_surface` + `test_the_shared_symbol_check_can_fail` (red twin), and `src/ffi/cpp/gate_one_odr.py` (GATE 10, a live CUDA process doing host phdf5 work) |
| class | (b) pre-existing, structural — a cross-`.so` ODR violation in the C++ |
| covering leg | THREE, all on Perlmutter, artifacts under `/pscratch/sd/j/jackm/svc_ffi_odr/_reports/`: the acceptance tier (12/12), the four-arm `tests/test_file_io.py` mixed-process A/B, and GATE 10 |

**WHAT IT WAS.**  Both libraries are dlopened `RTLD_GLOBAL`, and ld.so
answers a name from the FIRST object that defined it — for the whole
process, INCLUDING for the second library's own internal calls.  MEASURED
2026-08-07, `nm -D --defined-only` on the two BUILD_NOTES-pinned builds
(deployed device lib; `build_host_h200`, md5 `4c4422b8…`): **259 names
defined by both, 25 of them LORRAX's own** — the nine C-linkage
`lrx_phdf5_*` / `lrx_slate_*` entry points and SIXTEEN mangled
`lorrax_ffi::phdf5::*`.  The row as originally written said seven mangled;
the measurement says sixteen — `open_ctx`, `close_ctx`, `ensure_dataset`,
`open_dataset_ro`, `ensure_pinned`, `ensure_read_buf`,
`ensure_mpi_initialized`, plus `env_flag`, **`~PhdfCtx` (D1 and D2)** and
the `dt::` HDF5-type singletons with their guard variables.  The destructor
being on that list is the worst of them: it is a `std::thread` join and a
`std::mutex` destruction at whichever build's field offsets answered.

`src/ffi/cpp/phdf5/ctx.h` compiles `PhdfCtx` with the CUDA stream / event /
pinned-buffer members under `#ifndef LORRAX_FFI_NO_CUDA`.  One C++ type
name, two struct layouts, both libraries exporting one
`open_ctx(...) -> PhdfCtx*`.

**THE FIX — three mechanisms, because the defect had three vectors.**

1. **Linker version scripts**, `src/ffi/cpp/exports_{cuda,host}.map`, wired
   in `CMakeLists.txt` with `LINK_DEPENDS`: `local: *lorrax_ffi*` makes
   every definition LORRAX compiled private to the library that compiled it,
   so an intra-library call binds at static-link time and can neither
   interpose nor be interposed.
2. **A per-leg C ABI**, `src/ffi/cpp/common/c_abi.h`.  The nine `lrx_*`
   entry points cannot be hidden — Python `dlsym`s them — so the host leg's
   carry a `_host` suffix, the same per-library renaming the `*HostFfi`
   handlers already used.  `ffi_loader._bind_c_abi` (and distrib_la's) binds
   the suffixed symbol under the plain Python name at load, so no call site
   changed and a pre-fix `.so` still works.
3. **Split type identity**: `PhdfCtx`'s struct TAG is now `PhdfCtxCudaV1` /
   `PhdfCtxHostV1` with `using PhdfCtx = …` keeping every call site as
   written, so `open_ctx` and friends mangle differently per leg and the
   cross-layout aliasing is unconstructible even without (1).

**NOT `-fvisibility=hidden`, and NOT `local: *`.**  Both were tried.  Hidden
visibility additionally makes gcc mark a type's typeinfo NAME private with a
leading `*`, after which libstdc++ compares typeinfo by POINTER only —
and libslate.so throws `slate::Exception` ACROSS the `.so` boundary into our
handlers, where it matches today by strcmp on that name.  `local: *` was
built and MEASURED: it gives a 26-symbol host table and a zero intersection,
and it **segfaults five SLATE host cells**, because it also unmerges the weak
COMDAT copies of SLATE's template code that the C++ ABI intends the dynamic
linker to merge with libslate.so's.  Five arms, same tree, same command, only
the version script moving (`-k "slate and cpu"`):

| version script | slate host cells |
|---|---|
| none at all | 8 P / 2 skip |
| `local: *lorrax_ffi*` (shipped) | 8 P / 2 skip |
| `local: *`, typeinfo/vtables global | 3 P / **5 CRASH** |
| `local: *`, slate/blas/std templates global | 8 P / 2 skip |
| `local: *` | 3 P / **5 CRASH** |

**EVIDENCE, before and after** (`nm -D --defined-only`, both pairs built
against the HDF5-200 stage; `nm_evidence.log`):

| | shared names | LORRAX's own | C-linkage |
|---|---|---|---|
| pre-fix pair (`96a6399`) | 259 | **25** | 9 |
| rebuilt pair (`b61e028a`) | 234 | **0** | **0** |
| rebuilt device + pre-fix host | 243 | 9 | 9 |
| pre-fix device + rebuilt host | 234 | **0** | **0** |

The third row is the one to read before pinning: a MIXED pin does not get
the fix — an old host lib still exports the unsuffixed `lrx_*`.  The fourth
says the host rebuild alone is sufficient, so the deployed device `.so` does
not have to move.

**THE MIXED-PROCESS PROOF.**  `tests/test_file_io.py`, CPU platform, four
emulated devices, four arms, same tree, same launcher — the probe DISABLES
B1's capability gate, i.e. restores `32e61fe`'s unconditional pre-open, so
the process is put back INTO the mixed state on purpose:

| arm | pins | gate | result |
|---|---|---|---|
| A | rebuilt | on | 46 passed / 1 skipped |
| D | pre-fix | on | 46 passed / 1 skipped |
| B | rebuilt | **off** | **46 passed / 1 skipped**, 0 abort signatures |
| C | pre-fix | **off** | **`Fatal Python error: Aborted`**, srun exit 134, no junitxml |

Arm C is what makes arm B evidence.  The gate edit was local, reverted with
`git checkout`, and `git status --porcelain` printed empty afterwards.

And GATE 10 (`src/ffi/cpp/gate_one_odr.py`), a real CUDA process
(`jax.default_backend() == 'gpu'`, `loaded_platforms_in_order() ==
['CUDA','cpu']`) doing a host phdf5 write + full read + offset-slab read on
a CPU mesh: **every check PASS on the rebuilt pair**; on the deployed Aug-7
pair the process dies at

    [SlabIO.close] draining 1 pending writes for gate_one_odr.h5 …
    Fatal glibc error: tpp.c:83 (__pthread_tpp_change_priority): assertion failed

— `close_ctx` joining a thread and destroying a mutex at the other build's
offsets.  Note that this is, line for line, the death **B2** recorded for
the xdist arm; L1 was a mechanism behind that symptom too.

**WHAT IS NOT FIXED, and where it lives.**  234 third-party vague-linkage
names (`slate::`, `blas::`, `xla::ffi::`, libstdc++ internals) are still
defined by both files and the first loaded still answers them for both.
They are ODR-CORRECT duplicates — same headers, same source — and
localising them is the measured crash above.  Their sharing is the SAME
defect as `libslate.so.2` / `libblaspp.so.2` resolving out of two different
builds under one SONAME.  **`_open_cuda_before_host` therefore STAYS** in
both loaders: it was written for that race, the ODR fix does not touch it,
and retiring it would re-open the eight `blas::get_device_count()=0` cells.
`test_so_acceptance.py`'s check 5 remains the ratchet that will say when it
can go; its docstring is rescoped to say it covers only that.

**BUILD-TIME RATCHETS.**  `config/perlmutter/build_ffi_host.sh` GATE 9
refuses a host `.so` that exports any `lorrax_ffi` internal or any
unsuffixed `lrx_*`; `src/ffi/cpp/build.sh` GATE 9 is the device twin.  Both
passed on the rebuilt pair; `config/frontera/build_ffi_host.sh`'s WANT list
now names the suffixed symbols, so a pre-fix host library fails it by name.

### **P1 — three Tier-1 pins are Frontera-frozen and cannot be green on both machines**

> **RULED, 2026-08-07 (owner): "the micro-eV level is fine for comparisons
> between machines."**  Two of the three rows below are RESOLVED BY POLICY
> and move to *FIXED BY POLICY* immediately after this table.  The third is
> **not covered by the ruling** and stays ship-listed, for a reason given
> where it is listed — it is not a micro-eV problem.

| tests | class | evidence | disposition |
|---|---|---|---|
| `test_gw_jax_regression::test_gnppm_matches_reference` | (c) stale/relocated pin | 20 / 2484 rows, **max abs diff exactly 1.000e-6 eV** against `atol=1e-6` | **FIXED BY POLICY** — `_XMACHINE_ATOL_EV = 1e-5` |
| `test_gw_jax_regression::test_bispinor_gnppm_matches_reference` | " | 24 / 1620 rows, **max abs diff exactly 1.000e-6 eV** | **FIXED BY POLICY** — same constant |
| `test_sigma_ppm_gates::test_g2_branch_window_tiles_are_frozen` | (b) real behavioural difference, mis-filed under P1 | crossing-core node ladder is **100** here, the frozen array is **98** | **STILL SHIP-LISTED** — see P1b |

The Frontera census (`f485b5a`, 2026-08-01, job 7885154) re-froze all three
from Frontera CLX output.  Its own KNOWN_FAILURES row said the drift was
"EXACTLY one unit in the 6th printed decimal … 20/2484 resp. 24/1620 rows".
This census measures the same rows, the same 1-ULP size, in the other
direction — because the reference `f485b5a` *replaced* is what Perlmutter
produces:

    sigma_diag_gnppm_ref.dat     n=12  sigC=  4.771156   (pre-f485b5a)   == today's ACTUAL
                                       sigC=  4.771155   (f485b5a)       == today's DESIRED
    sigma_diag_bispinor_ref.dat  n=10  sigXC=-16.978381  (pre-f485b5a)   == today's ACTUAL
                                       sigXC=-16.978382  (f485b5a)       == today's DESIRED

Nothing on this branch moved them: the WSL full-suite red-set diff over the
whole branch (`96a6399` → `d5cac09`) is empty in both directions.

#### FIXED BY POLICY — the two float pins

A 6-decimal `.dat` at `atol=1e-6` has no room for a cross-platform ULP, so
"re-freeze on whichever machine ran last" was a permanent ping-pong in
which each re-freeze silently turned the other machine red.  The owner's
ruling ends it: the comparison tolerance for these two cross-machine-frozen
pins is now **`_XMACHINE_ATOL_EV = 1e-5` eV** (`tests/test_gw_jax_regression.py`),
10× the observed drift and five orders below anything physical.  The
constant carries the ruling, the date, and the scope; it is named on two
cells and nowhere else.

**What still anchors these tightly.**  Loosening a *cross-machine* pin does
not loosen the tree.  Same-machine drift is caught by the Si COHSEX
byte-identity gate (`test_si_production_matches_frozen_reference`, exact
text match, early return) and by the external BerkeleyGW anchor
(`test_si_production_matches_berkeleygw`, `_BGW_TOL`, sub-meV MAE against
another code).  Those are the gates that would see a physics change; these
two answer "does the frozen MoS2 output reproduce on a different machine",
and that answer should not turn on the 6th decimal of a text file.

`_assert_matches_reference` now REPORTS the observed max |Δ| and what
fraction of the atol budget it used, on every run, pass or fail — a
tolerance whose headroom is invisible cannot be audited, and this ruling
is exactly the kind that needs auditing later.

**VERIFIED ON PERLMUTTER**, `svc/distrib_la-2026-08-07` @ `52f5024`,
jobid **56457930**, `lx test -N 1 -G 1 -n 1`, serial (`-n 0`), BUILD_NOTES
pins, artifacts `/pscratch/sd/j/jackm/svc_distrib_la/_reports_p1/p1b.{log,xml}`:

| cell | max abs Δ | atol | budget used | cells over | cells differing |
|---|---|---|---|---|---|
| `test_gnppm_matches_reference` | **1.000e-06 eV** | 1e-05 | **10.0 %** | 0 | 36 / 2484 |
| `test_bispinor_gnppm_matches_reference` | **1.000e-06 eV** | 1e-05 | **10.0 %** | 0 | 50 / 1620 |

`2 passed in 69.61 s`.  Both were previously RED.  The drift is exactly
the 1-ULP-of-the-6th-decimal the census measured — no larger — so the
band has a full decade of headroom and is not absorbing anything else.

> **Unit note, because the two tables disagree and should not be read as
> contradicting.**  The census row above says "20 / 2484 rows"; the
> measured row here says "36 / 2484 cells".  2484 is the CELL count
> (414 rows × 6 compared columns), so the census's label was wrong even
> though its denominator was right, and the counts differ (20 vs 36)
> because they were taken from different runs of a GPU-nondeterministic
> last-ULP effect.  The quantity that matters — the max |Δ| — is
> **1.000e-06 eV in both**, and that is the one the band is set against.

**The tolerance was actually exercised.**  Both cells took the
`assert_allclose` path, not the byte-identity early return (the report
distinguishes the two by name).  A green here therefore means "the drift
is inside the band", not "the files happened to match", which is the
difference between a verified ruling and a vacuous one.

**Neither reference was re-frozen.**  Re-freezing is the move that created
this row; the fix is the comparison, not the data.

#### P1b — `test_g2_branch_window_tiles_are_frozen` is NOT a micro-eV row — **RULED**

> **RULED AND CLOSED, 2026-08-07.**  The owner chose PERLMUTTER as the
> blessed grid for this reference and it was re-frozen there at the
> integration head; the gate is **GREEN** in the merged-head census at the
> top of this file.  The analysis below stands unchanged and is why no
> tolerance was applied — the disagreement is an integer count of
> quadrature nodes, not a rounding difference.  What the landing added is
> the measurement of HOW the two grids differ (4 shape-mismatched tiles,
> 2 meta rows off by exactly 2.0) and a red-twin probe showing the
> re-frozen gate can still fail.  A Frontera build will now fail this
> cell loudly and BY SHAPE, which is the intended outcome.

Filed under P1 by resemblance and it does not belong there.  The
Perlmutter/Frontera disagreement in this cell is the **crossing-core node
ladder: 100 nodes here, 98 in the frozen array** — an integer count of
quadrature τ points, riding in a `float64` `meta` row.  It is not a
rounding difference, it is not in eV, and a tolerance would hide a real
change in how many points the window integrates over.  The 2026-08-07
ruling therefore does not reach it, and applying it here would be
laundering.

**RE-MEASURED in the same leg** (jobid 56457930, `_reports_p1/p1.log`) and
it is WORSE than the row above recorded — not a 100-vs-98 count with
otherwise-matching values, but a **shape mismatch with different
contents**:

    ω≥E_F cond|0|core|t not bit-identical
    (shapes (100,), (98,) mismatch)
     ACTUAL:  [ 2.666561,  6.499882,  6.936903,  8.974977, ...]
     DESIRED: [ 5.442279e-09, 7.894766, 10.45215, 10.63999, ...]

The τ-node *positions* disagree from the first element, so this is two
different quadratures, not one quadrature sampled twice.  Whatever the
answer is, it is not a tolerance.  Stays ship-listed, class (b), still an
**OWNER DECISION**:
either the ladder legitimately differs between the two machines' minimax
tables (in which case the reference is platform-dependent data and needs a
different mechanism than an atol), or one of the two is wrong.  Nobody has
determined which.  `tests/test_sigma_ppm_gates.py` carries this note at the
comparison itself.

### P2 — the chunk-width gauge cells (pre-existing)

| tests | class | evidence |
|---|---|---|
| `test_bse_setup_qchunk::test_values_are_invariant_to_the_chunk_width` | (b) pre-existing | `_maxdiff = 1.374` against `< 1e-10`; red in isolation, red on WSL, red at the branch base |
| `test_bse_setup_qchunk::test_chunk_width_ulp_spread_is_reported` | (b) pre-existing | reports 5 non-zero spreads, first **`(2, 2.220446049250313e-15, 2.220446049250313e-15)`**, where the pin expects `[]`.  *(Fingerprint corrected 2026-08-08 by the BSE-perf merge checkpoint: this row read `2.22e-16`, one decimal place off.  A/B'd at pre-merge `main` 602e1d8b and at the merge head e69a867f, same node, same pins -- identical to every printed digit on both sides, so the `e-15` is what this tree produces and always did.  A fingerprint off by a decade is worse than none: the next census would read a correct value as drift.)* |

Already characterized and A/B'd at `5bb4368`; cited, not re-derived.  The
second cell is the first one's instrument and goes red with it.  On WSL the
whole file is red (12 cells) for want of an FFI; on Perlmutter exactly these
two are.

### P3 — `test_wfn_loader_eager::test_no_ffi_at_P_gt_1_refuses_and_names_both_libraries`

Class (d) environment.  The cell's own docstring says it "runs the resolver
… on a tree with no `.so` (which is this checkout)".  Under the BUILD_NOTES
pins the `.so` IS present, so the terminal refusal arm is unreachable and
the cell reports `DID NOT RAISE`.  **FALSIFIED**: the same cell with
`LORRAX_FFI_SO`/`LORRAX_FFI_HOST_SO` unset is **1 passed**.  Not a code
defect; the cell needs to neutralise the pins itself (monkeypatch
`_locate_so`, or `delenv` both) instead of assuming its checkout is bare.

### P4 — `test_staged_reshard::test_red_twin_the_unstaged_chain_emits_the_spmd_warning` (leg C)

Class (d) environment, and it is the *instrument's own* red twin, which is
why it matters more than a normal skip.  The cell asserts that the UNSTAGED
chain emits `Involuntary full rematerialization`; on this XLA it emits
nothing, so the twin fails with its own message: *"the instrument cannot go
red here, so a zero count from the staged path means nothing."*  Every other
remat gate in leg C is green — but that green is now **unfalsifiable on this
stack**.  Covering leg: none on Perlmutter; the Frontera census's leg C is
where the instrument last demonstrably worked.

### P5 — `test_aot_memory::test_predicted_peak_matches_runtime_3d_fft` (leg D, `extra`)

Class (d) environment.  `device.memory_stats()` returns `None`, so
`["peak_bytes_in_use"]` raises `TypeError`.  The site's shifter string
carries `--env=XLA_PYTHON_CLIENT_ALLOCATOR=platform`, and the platform
allocator keeps no arena to report — the runtime banner says so in the same
run ("The live client reports no arena accounting at all").  **FALSIFIED**:
`env XLA_PYTHON_CLIENT_ALLOCATOR=bfc` on the same device returns the full
dict (`bytes_in_use, bytes_limit, …, peak_bytes_in_use, …`).  The cell
should skip-with-reason when `memory_stats()` is `None` rather than
`TypeError`.

### P6 — `test_contract_bands::test_ffi_gemm_plan` crashes the interpreter at 4 emulated devices

Class (b) pre-existing.  `SIGSEGV` inside the TEST's own numpy reference
(`test_contract_bands.py:129 _ref` → `numpy/_core/einsumfunc.py:1194
bmm_einsum`), or `SIGABRT` at `:135 _relerr` → `jax…array.__array__` when
threads are pinned.  Reproducible alone; `OMP_NUM_THREADS=1` /
`MKL_NUM_THREADS=1` does not change it; deselecting only that cell leaves
the file at **8 passed**.  **PRE-EXISTING: the same cell segfaults at the
branch base `96a6399` on the same leg.**

This one costs coverage twice over: at 1 device the file's 9 device-gated
cells SKIP, and at 4 emulated devices this cell kills the process — so it
has no leg at all.  It is excluded from leg C by name, which is why leg C
has a junitxml.

---

## Environment-limited (skips, each with its covering leg)

Leg A, 62 skips.  Leg A2 = these 62 plus leg B's 22.

| n | reason | covering leg |
|---|---|---|
| 32 | `needs >=4 devices: XLA_FLAGS=--xla_force_host_platform_device_count=4` (`test_staged_reshard`, `test_staged_reshard_routes`) | **C** (green) |
| 11 | `needs 4 (emulated) devices, got 1` (`test_contract_bands` 9, `test_projection_lgemm` 2) | **C** for `projection_lgemm`; `test_contract_bands` → **P6, uncovered** |
| 7 + 1 | `needs 4 devices, have 1` / `needs 2 devices, have 1` (`test_charge_zeta_route`) | **C** and **C3** (green) |
| 4 | `needs >=4 devices to build a 2x2 mesh` (`test_sharding_fit`) | **C** (green) |
| 1 | `needs >= 4 devices for a 2x2 mesh; have 1` (`test_file_io`) | the 4-device CPU leg — UNCOVERED at census time (B1 aborted it), **covered since `7a1d64f`: 42 P / 1 S** |
| 1 | `device count 1 is a perfect square; the refusal arm needs a non-square count` | **C2** (1 passed at 3 devices) |
| 1 | `needs 4 (emulated) devices` (`test_sanity_gates_jax::test_check_hermitian_sharded`) | **C** (green) |
| 1 | `P=1: jax.devices()[0] IS this process's device, so the negative control cannot fire` | true P>1 srun leg; **not** the emulated legs |
| 1 | `` `lfs` unavailable here, so the stripe count cannot be read `` | none — the skip says outright it verified nothing |
| 1 | `fit_scissor still accepts the energy arrays positionally` | self-documenting deprecation skip |
| 1 | **`eqp_si_fast_ref.dat` is not frozen yet — freezing a reference is the owner's call.  A candidate generated 2026-08-07 at 04b8bba lives in `/pscratch/sd/j/jackm/si_consolidation_2026-08-07/run_fast_final/` (eqp_si_fast.dat); copy it in to enable this gate.** | **INTENTIONAL, VISIBLE.**  The 20-band fast gate.  Owner's call; NOT frozen by this census |

Leg B / E2, 22 and 3 skips:

| n | reason | covering leg |
|---|---|---|
| 19 (leg B only) | `needs >= 4 devices on platform 'cpu', have 1. Set XLA_FLAGS=… BEFORE the first jax import` | legs **F**/**G** — the real 4-process 2×2, which is the coverage these emulate |
| 2 | `slate host heev SIGSEGVs — bug L-2, see docs/dev/linalg_ffi.md` | none; pinned by the skip itself, carried by design |
| 1 | `cholesky/cusolvermp is not usable on a 1x1 gpu mesh` (needs a true-2D mesh) | leg **G** (`cusolvermp_factor_solve[2x2]` green, residual 6.0e-16) |

Leg D, 2 skips: no CrI3 6×6 30Ry SOC `WFN.h5` reachable (out-of-repo
fixture); `jax.jit` decorator-factory form unsupported on jax 0.7.0.

Legs F/G skips are the cross-platform halves — `scalapack_*` and
`batched_eigh_dispatch` are host-only and skip on G; `cusolvermp_*` are
CUDA-only and skip on F.  Between them every cell runs on one of the two.

---

## Instrument notes (how to run this suite on Perlmutter)

1. **A shifter `--env` beats your exported environment, and `XLA_FLAGS` is
   one.**  The NVIDIA jax image ships its own `XLA_FLAGS`, so
   `XLA_FLAGS=--xla_force_host_platform_device_count=4 lx test …` arrives in
   the container as `XLA_FLAGS=  --xla_gpu_enable_latency_hiding_scheduler=true`
   and `jax.device_count()` is **1**.  The leg then SKIPS its way to green
   and looks identical to a leg that ran.  MEASURED both ways; the form that
   takes is `lx run … env XLA_FLAGS=… python3 -m pytest …`, because `env`
   runs *inside* the container.  Every emulated-multi-device leg here uses
   that form, and the first attempt (recorded, superseded) did not.
2. **`lx test` is xdist.**  One node, all GPUs, four workers, one GPU pinned
   per worker by `tests/conftest.py` — and NOT by its controller, which is
   what **B2** was.  It is why the single-process `lx run … -m pytest` leg
   is the control for every session-fixture red.
3. **Judge by artifacts.**  Leg A exits non-zero (`srun: task 0: Exited with
   exit code 1`) and still wrote a 246 kB junitxml with 1191 cells; leg C's
   first attempt exited 139 and wrote NO xml at all.  The exit code
   distinguished neither.
4. **`pytest-timeout` is not installed in this image** — `--timeout=` is an
   `unrecognized argument` and kills the leg with exit 4.  Wrap the payload
   in `timeout N …` instead.
5. **`lx run --cpu` has no container and therefore no jax.**  The CPU L-c
   leg is `lx run -N 1 -G 4 -n 4 env JAX_PLATFORMS=cpu …`, not `--cpu`.
6. Per-leg env, verbatim, is in
   `/pscratch/sd/j/jackm/svc_distrib_la/{census,census2,recheck*,bisect*}.sh`.

## Cross-machine set-diff (WSL, jax 0.9.1, no FFI)

`96a6399` → `d5cac09`, `python -m pytest -q -p no:randomly`, whole branch:
**1420 → 1441 collected, 95 → 95 red, identical ids, 0 removed, 0 newly
red.**  The WSL leg cannot see any of B1/B2 (no FFI, no CUDA) — that is
exactly why the Perlmutter census had to exist.

Re-measured across the two fixes, `35f3e06` → `f7c1b17`: **1441 → 1471
collected, 95 → 95 red, identical ids, 0 removed, 0 newly red, +30 new
cells all green.**  The 30 are the two-armed load-order cells and the two
capability tables (B1) and the controller cells (B2); the xdist twin skips
on WSL for want of the plugin and runs on Perlmutter.

Re-measured again across the step-6 follow-up, `b425291` → `d880e67`:
**1471 → 1480 collected, 95 → 95 red, 185 → 185 skipped, 0 removed,
0 newly red, 0 newly green, +9 new cells ALL GREEN.**  The 9 are the
cost-notice trio, the L-3 trio and the L-4 trio, all resolve-level and all
runnable with no GPU and no `.so`.  `64 failed + 31 errors = 95` in
2348 s.

> **Diff against `wsl_fix/wsl_fix_pre.xml` (1471), NOT `wsl/wsl_efdbf9a.xml`
> (1441)**, or the +30 cells from the B1/B2 fixes are re-counted as new.
> `b425291` is `f7c1b17` plus one `.md`-only commit, so the `f7c1b17`
> artifact is the right baseline for anything measured after it.  This
> trap cost nothing only because it was noticed before the diff was run.

---
---

# Frontera CLX census, 2026-08-01 — HISTORICAL RECORD

**Kept, not deleted.**  Superseded as the tree's census by the Perlmutter
one above; still the authority for what Frontera CPU measured, and the
document class **P1** is measured against.

Complete `python -m pytest tests/` census on Frontera CLX (in-container,
required-FFI defaults, host .so `build_host_MRG`), 2026-08-01, tree
`bbe6e56` + the fixes committed with that census.  Authoritative run:
**job 7885154** (junit XMLs + run dirs under
`/scratch2/08271/jackmc/pytest_p11/`).  Job 7885150 was the first attempt
and is superseded — its 97 failures were dominated by an instrument error
(see "Instrument notes"), kept only as evidence.

## Verdicts by leg (job 7885154)

| leg | invocation | result |
|---|---|---|
| A2: full suite, bare, 1 device | `pytest tests/ --ignore=tests/test_ffi_linalg_contract.py` | 24 failed / 735 passed / 56 skipped / 26 deselected |
| B2: FFI linalg contract | `srun --mpi=pmi2 -n1` + `config/frontera/mpi_transport_env.sh`, `pytest tests/test_ffi_linalg_contract.py` | **0 failed** / 27 passed / 25 skipped |
| C: 4-device leg | `XLA_FLAGS=--xla_force_host_platform_device_count=4`, the 9 device-hungry files | 1 failed / 144 passed / 2 skipped |
| C2: nonsquare refusal | `...device_count=2`, `-k nonsquare` | 1 passed |
| D: extra tier | `-m extra`, 4 devices | 0 failed / 21 passed / 5 skipped |

> **The A2/B2 invocations above are RECORDED AS RUN (job 7885154) and are no
> longer runnable as written.** `tests/test_ffi_linalg_contract.py` moved to
> `services/distrib_la/tests/test_distrib_la_contract.py` and carries the
> `distrib_la` marker, so naming a path is no longer the way to include or
> exclude it. Today the same two legs are
> `pytest tests/ --no-services` and `srun --mpi=pmi2 -n1` +
> `config/frontera/mpi_transport_env.sh` + `pytest -m distrib_la`.
> `tests/conftest.py` owns those hooks and `tests/test_service_selection.py`
> measures that they select what they claim.

Every leg-A2 failure below was triaged; after the fixes in that commit
the only remaining red was the ring-vma class.

## KNOWN FAILURES (ship-listed, as of 2026-08-01)

| tests | class | evidence | status |
|---|---|---|---|
| 10 ring-transport tests: `test_bse_dense_reference` `{w_positive_control,full_H,DV}[ring]` + `test_nontda_matvec_matches_dense_shao` + `test_nontda_solver_reproduces_dense`, `test_bse_stack_matvec::test_stack_memory_flat_in_n_trials`, `test_bse_w0_resolvent` (2), `test_bse_w_omega_chain` (2) | (b) pre-existing — the old handoff's "bse_ring_comm vma", verified present at that HEAD and now precisely diagnosed | `TypeError: scan body ... carry ... {V:(x,y)} varying manual axes` at `src/bse/bse_ring_comm.py:382` (`_apply_V_ring_only` fori_loop carry `A0` unannotated; jax's error prescribes `lax.pvary` on the initial carry). junitA2_7885154. Serial + simple matvec arms PASS, so the dense-reference physics is still covered; only the ring transport arm is dark | **CLOSED on Perlmutter/jax 0.7.0**: all 10 pass in the single-process `iso_bse` leg of the 2026-08-07 census. The Frontera/jax-version scope of the original diagnosis is unretracted |
| `test_ffi_linalg_contract` under a BARE (no-srun) launch with the host .so loadable: silent interpreter death at import | (d) environment — MPI init without PMI2 glue | CLAIMS 30; reproduced 7885125 step1 (srun WITHOUT transport env also dies). **Upgrade that census added: with `mpi_transport_env.sh` sourced under `srun --mpi=pmi2 -n1` the pytest form is fully GREEN (leg B2, 27 passed)** — the CLI matrix is no longer the only instrument | Not a code bug; invocation contract. The bare leg must deselect the service (`--no-services`); the srun+transport leg (`-m distrib_la`) covers it. On Perlmutter neither death occurs: leg B is 130 cells / 0 failed under plain `lx test` |
| `test_centroid_distribution::test_orbit_path_with_trivial_group_matches_plain_path` | (d) environment — reproduces ONLY on an UNSUPPORTED jax. Corrected 2026-08-05: an earlier revision of this row claimed the defect was version-independent; direct measurement disproved that, see evidence | `jax.errors.UnexpectedTracerError`: an `int64[]` tracer whose creating frame is `src/centroid/orbit_syms.py:241` at **`<module>` scope** escapes the jit of `kmeans_pp_init` (`src/centroid/kmeans_isdf.py:585`) and is raised out of the Lloyd loop at `src/centroid/kmeans_isdf.py:738`. Only the orbit-on (`R=`/`Rinv=`/`tau=`) branch trips. **Seen only under the Shifter image's bundled jax 0.5.3** (Perlmutter GPU census, jobid 56385965, `pytest_gpu.log`: 7 failed / 920 passed). Under the SUPPORTED jax the same test **passes** | Still a **tripwire**, and it did not fire: the 2026-08-07 Perlmutter census runs `lorrax_J070` (jax 0.7.0.dev20260807) and `test_centroid_distribution` is green in legs A, A2 and C |

## Environment-limited on Frontera (skips, each with its covering leg)

| tests | reason | coverage |
|---|---|---|
| 45 device-count skips (leg A2): `test_staged_reshard` (14), `test_staged_reshard_routes` (18), `test_charge_zeta_route` (7), `test_sharding_fit` (4), `test_collectives_distribution`, `test_centroid_distribution`, `test_sanity_gates_jax::test_check_hermitian_sharded` | need >=2/4 emulated devices | leg C (all green after the fix below); nonsquare-refusal cell needs a NON-square count → leg C2 (green) |
| `test_centroid_distribution::test_process_local_mesh_is_addressable` negative control | needs true multi-PROCESS (P>1), not emulated devices | P>1 srun leg (P1 scaling legs); `tests/multi_device/` is likewise srun-driven, never pytest-collected |
| `test_bse_kgrid` (7), `test_wfn_transforms::test_to_box_{ibz,full_bz}_mos2` (2), `test_R_proper_cri3` (1, extra tier) | fixtures pinned to `/pscratch/...` — Perlmutter, machine gone | RESOLVED for `test_bse_kgrid` by running on Perlmutter: it is collected and (single-process) green in the 2026-08-07 census. The census's second name, `test_wfn_loader_eager[mos2]` (3), no longer resolves at this HEAD: that cell moved to `services/wfn_loader/tests/test_wfn_loader_contract.py` on 2026-08-07 and its dead `/pscratch` arm was repointed at the in-repo `gnppm_debug/WFN.h5` twin (survey w1_wfn_loader §6.4 — byte-size identical), so it RUNS on both machines now and the Frontera skip count dropped 3 → 2; the census's green-on-Perlmutter finding for it stands and is no longer machine-dependent. Still OWNER's for the 2 remaining `to_box` cells: restage the MoS2 3×3 640c fixture + WFN.h5 on Frontera (or re-point); until then those self-skip |
| 23 CUDA cells in `test_ffi_linalg_contract`, `-m gpu`-dependent extras (3 cufft + 1 CUDA backend in leg D) | need a CUDA jax backend | P1 GPU leg (rtx); on Perlmutter these are the leg-B cells that run |
| `test_slate_cholesky_trsm_cpu` heev cells (2 skips in leg B2) | slate host heev SIGSEGV — documented bug L-2, `docs/dev/linalg_ffi.md` | pre-existing, pinned by the skip itself; SAME 2 skips on Perlmutter (legs B, E2) |
| 26 deselected (`-m extra` tier) | deselected by repo `addopts` default | leg D ran them: 21 passed / 5 skipped |

## Fixed in the Frontera pass (committed with `f485b5a`)

| tests | root cause | fix | validation |
|---|---|---|---|
| `test_file_io` (12) + `test_compute_all_V_q_g_flat::test_..._rejects_r_space_loader` | (c) stale builders: synthetic `zeta_q.h5` helpers never stamped `zeta_is_done`, and `ZetaLoader` now refuses partial files at open (completeness gate) | builders stamp done (complete synthetic payloads); the flag-behaviour tests pass `zeta_is_done=False` explicitly | GREEN in 7885154 leg A2, and GREEN on Perlmutter |
| `test_zq_from_psi_sm_bit_identity` (6) | (c) `_MockPsiGStore` missing `_bpd_per_bc` | mock mirrors `psi_G_store.py:147` | GREEN in 7885154 leg A2, and GREEN on Perlmutter |
| `test_sigma_ppm_gates::test_g2_branch_window_tiles_are_frozen` | (c) stale pin: G2 npz frozen 2026-07-07; `d011a36` reconditioned the Σc HGL crossing quadrature — crossing-core node ladder changed 103→98 | regenerated via `_regenerate_g2_reference()` (job 7885154 step 0, CPU/f64) | GREEN on Frontera — **and RED on Perlmutter, where the ladder is 100. See class P1** |
| `test_gw_jax_regression::test_gnppm_matches_reference`, `::test_bispinor_gnppm_matches_reference` | platform-migrated pins: refs frozen on Perlmutter GPU (b7654ee); on Frontera CPU/FFI the drift is EXACTLY one unit in the 6th printed decimal (max delta 1.000e-6 eV, sigC/sigXC only; 20/2484 resp. 24/1620 rows) against `atol=1e-6` | re-froze both `sigma_diag_*_ref.dat` from the job-7885154 session outputs | **This is the move class P1 documents. The re-freeze relocated the red rather than removing it: on Perlmutter the same 20/2484 and 24/1620 rows are now off by the same 1.000e-6, and the pre-`f485b5a` reference is bit-equal to what Perlmutter produces today** |
| `test_runtime_distributed::test_set_default_env_defaults` | (a) real gap in `skip_gpu_plugin_discovery` | both branches re-apply the demotion pinning | GREEN on Perlmutter (legs A, A2) |
| `test_charge_zeta_route::test_rank_truncate_refuses_rather_than_downgrading` (leg C) | (c) stale pin vs the R15.1 widening | refusal case moved to `_OVER_FACTOR_CAP` (μ=17000) | GREEN on Perlmutter legs C and C3 |
| `test_contract_bands` (9) + `test_projection_lgemm` (2) failing at 1 device | test defect: `assert n_dev >= 4` instead of the suite-wide `pytest.skip` | `_mesh()` skips below 4 devices | GREEN under Frontera leg C; on Perlmutter `test_projection_lgemm` is green and `test_contract_bands` hits class P6 |

## Old-handoff known-fail list, as verified 2026-08-01

| handoff item | verdict |
|---|---|
| file-IO fixtures | root-caused (zeta_is_done completeness gate vs stale test builders) and FIXED |
| bse_ring_comm vma | present on Frontera/jax at that tree; **not reproducible on Perlmutter / jax 0.7.0 (2026-08-07)** |
| kmeans multi-rank segfault | not reproducible in pytest scope; P>1 thread-main refusal FIXED repo-side (24e4dc3, subsumed e97e8ed); true multi-process kmeans belongs to the P>1 srun leg |
| GN-PPM pred remat | remat gates ALL GREEN in Frontera leg C (32 tests). **On Perlmutter the gates are green but their RED TWIN is dead — see class P4** |

## Frontera instrument notes (the 7885150 lesson)

1. Export the environment INSIDE the container: apptainer does not forward
   the host `LD_LIBRARY_PATH`, and without it the required-FFI gate
   refuses (`libhdf5.so.310` unresolvable) — that single mistake produced
   68 failed + 29 errors in job 7885150.  Pattern: job script
   `/scratch2/08271/jackmc/pytest_p11/run_pytest2.sbatch`.
2. The `distrib_la` service suite must be DESELECTED in the bare leg
   (`pytest tests/ --no-services`; import-time death without PMI2 glue)
   and run under `srun --mpi=pmi2 -n1` with
   `config/frontera/mpi_transport_env.sh` sourced (`pytest -m distrib_la`)
   — green there.  Deselect through the hook, never through a second
   `-m`: `pyproject` sets `addopts = "-m 'not extra'"` and an explicit
   `-m` REPLACES it, silently re-enabling the whole `extra` tier.
3. Per-test timeout: `pytest-timeout` staged on `PYTHONPATH`
   (`--timeout=2400 --timeout-method=signal`); do NOT also pass
   `-p pytest_timeout` (double registration).  It is NOT installed in the
   Perlmutter image — see Perlmutter instrument note 4.
4. e2e regression fixtures run the drivers on the CPU node via
   `ISDF_COHSEX_TEST_PLATFORM=auto` (jax native pick); compile cache under
   `$SCRATCH` keeps the whole suite ~21 min.

---

# eqp0.dat / eqp1.dat mix two bases on the self-consistent path

OPEN, UNMEASURED, 2026-08-05.  Carried forward unchanged; neither census
touches it.  `gw_jax.py:652-654` reads `sigma_c_omega_kij_ry` off the object
`run_sc_driver` returns and emits its diagonal as `sigma_c_omega_diag_ev`.
That field is in the QP basis and is correct there — Σ is Hermitised as
½(Σ(E_n)+Σ(E_m)), which only means anything in the basis whose eigenvalues
are those E_n, so it must not be rotated.  The finalize rotates the five
static Σ fields to the DFT basis and carries the cube through unrotated,
which is deliberate.

The defect is downstream: `compute_eqp_diag` forms
`Δ = kin_ion + V_H + Σ_x + Σ_c(E_DFT) − E_DFT` from three DFT-basis
diagonals plus that one QP-basis diagonal.  The sum is basis-consistent
only at U = identity.  `write_results` is unguarded, so both files are
written on the SC path (unlike `eqp_g0w0.dat`, guarded at
`gw_output.py:846`).  The same mixing reaches `sigma_xc_at_dft_ev`
(`gw_jax.py:600-603`).

The error scales with ‖U − 1‖ and NO ONE HAS MEASURED IT.  To measure:
read `U_mnk` from an SC run's `qp_wfn_rotations.h5`, report the largest
off-diagonal element, and bound the eqp error by the off-diagonal Σ weight
it mixes in.  One-shot runs are unaffected — `solve_qp` is reached only in
the non-SC branch (`gw_jax.py:543`), where the whole object is DFT basis.

`tests/test_sigma_result_basis.py` pins which field is in which basis, so
a new Σ channel cannot join the wrong group silently.  It does not and
cannot catch this: the mixing is in the consumer, not the declaration.

## `KStarMap.spread_rel` loses NaN on an emulated partitioned reduction

OPEN, MEASURED, 2026-08-07, owner decision pending.  Carried in the suite
as the ONE strict xfail in `services/symmetry_maps/tests`:
`test_symmetry_maps_multiproc.py::test_a_nan_survives_the_sharded_reduction`.
Strict on purpose — the day the reduction starts propagating, that cell
turns RED and somebody deletes the marker instead of the suite quietly
keeping a stale claim.

**What was measured.**  jax 0.9.1, `JAX_PLATFORMS=cpu`,
`--xla_force_host_platform_device_count=4`, on `np.ones((4,8,8))` with row
3 set to NaN:

| sharding | `jnp.max` | `jnp.min` | `jnp.sum` |
|---|---|---|---|
| `P(None,'x','y')` | **-inf** | **+inf** | nan |
| `P()` / all-replicated | nan | nan | nan |
| `P('x',None,None)` | **1.0** (a wrong FINITE value) | — | — |
| single device | nan | nan | nan |
| numpy host | nan | nan | nan |

`KStarMap.spread_rel`: host `nan`, single-device `nan`, sharded `-inf`.
The per-shard local maxima ARE NaN when read back individually, so the
loss is in the CROSS-SHARD reduction — the max/min identity (-inf/+inf)
coming back from a collective that compared NaN and lost.  `jnp.sum` is
unaffected.

**And what the real four-rank legs measured afterwards, which is not the
same answer.**  Perlmutter, jax 0.7.0, `JAX_PLATFORMS=cpu`, `srun -n 4` at
BOTH 2×2 and 4×1, the same `check_spread_rel_is_one_replicated_scalar`
body: `sharded-NaN -> nan`.  A real cross-PROCESS all-reduce PROPAGATES.
So the loss is so far an EMULATED-mesh result only.  Two variables move
between the legs — process topology (one process holding four devices vs
four processes) and jax version (0.9.1 vs 0.7.0) — and which one causes it
is UNRESOLVED; the deconfound probe belongs in the land-readiness report.

**Why it matters.**  `sc_iteration._check_kstar_spread` refuses on
`not (spread <= tol)`, spelled that way DELIBERATELY (pinned by
`tests/test_sc_kstar_spread.py`) so a poisoned Σ cannot pass by comparing
False.  That only works if `spread_rel` hands NaN back.  Wherever the loss
holds, `-inf <= 1e-6` is True and the k-star spread gate PASSES a poisoned
iteration in silence.  The real-rank measurement above is why that is not
yet a statement about production — it is a statement about every mesh that
reduces the way the emulated one does, and about any jax upgrade that
makes a production mesh reduce that way.

**Not fixed on the symmetry_maps branch, on purpose.**  `_star_stats` is a
diagnostic on a register-don't-touch module, and the fix is a decision
about which reduction to trust: a `jnp.sum`-based NaN sentinel beside the
max, or the refusal moving into `sc_iteration`.  OWNER.

## cohsex_debug's committed Σ has broken SPATIAL star relations

OPEN, MEASURED, 2026-08-07, reported not gated.  §8.2 class.  The
committed `tests/regression/cohsex_debug/sigma_mnk.h5` was produced with
`centroids_frac_60.txt`, which is NOT orbit-closed under the deck's 12-op
spatial group, so the ISDF quadrature does not respect that group and every
within-star pair with a genuine rotation between it disagrees at
O(0.1–0.4) whichever conjugation is applied:

| pair | relation | `hartree_kij_ev` | `sigma_sx_kij_ev` |
|---|---|---|---|
| (1,2), (3,6), (5,7) | TRS-conjugation (XOR=1) | 7.5e-07 – 2.7e-04 | 5.7e-04 – 7.0e-04 |
| (1,3) | spatial | 1.848e-01 | 2.209e-01 |
| (4,8) | spatial | 2.735e-01 | 3.890e-01 |

The TRS-conjugation relations hold at ≤ 7e-4 — three to six orders below
the un-conjugated comparison (3.4–4.0e-01) — which is what makes
`tests/test_star_offdiag_gate.py` a usable off-diagonal symmetry gate.  The
spatial pairs are asserted there as a FACT (`> 0.1`) rather than left out:
a silently ungated pair is indistinguishable from a forgotten one, and if
the production centroids are ever regenerated orbit-closed that cell fails
and tells whoever did it that the spatial arm can be gated too.

A naive whole-star `star_spread` on this fixture reads 8.89 (sigma_sx) /
113.08 (hartree), dominated by the broken spatial relations — it is not a
usable gate on this deck, which is exactly why the gate is built from
`_star_conj_flags`' XOR=1 pairs instead.

**OWNER, and it is not a code fix.**  Regenerating the centroid sets means
re-freezing the BerkeleyGW anchor.  The same class is on the production
side: `centroids_frac_960.txt` carries a 2.611 meV star spread at the BGW
Σ gate.  No `centroids_frac_*.txt` is regenerated by any wave-1 branch, and
tests that need a non-closed set build one synthetically by dropping a
centroid from a closed one.

## vcoul physics fix 358bb0b — one EXPECTED red (2026-08-07) — **CLOSED**

> **CLOSED AT THE LANDING, 2026-08-07.**  The status column below already
> said RESOLVES AT MERGE, and it did: the owner-authorized refreeze
> candidate was adopted as `tests/regression/si_bse_debug/bse_eigenvalues_ref.dat`
> (md5 `09204d36…` → `cab1dd48…`) on `integration/2026-08-08-services`, and
> `test_bse_matches_frozen_and_bgw` is **GREEN** in the merged-head census
> at the top of this file.  The row is kept because it is the record of
> what moved and why a refreeze was the right response rather than a
> loosened tolerance.  See *FIXED BY RULING*.

| tests | class | evidence | status |
|---|---|---|---|
| `test_bse_bgw_regression.py::test_bse_matches_frozen_and_bgw` (frozen arm only) | (a) EXPECTED red from the flagged physics fix 358bb0b (mini-BZ body-head draw `randvals @ bvec.T` → `randvals @ bvec`) — `bse_si_test.in` pins `mc_average_vcoul_body=true`, so its head table reseeds. On Si the fix is EXACTLY a reseed (bvec.T = P·bvec, P cyclic) | MEASURED at 358bb0b on the BUILD_NOTES pins (`/pscratch/sd/j/jackm/svc_vcoul/_gates_after/`, siA/bseB junitxml + bse_after_analysis.txt): frozen arm fails at max 7.067e-5 eV / MAE 7.42e-6 eV against `ATOL_FROZEN_EV=1e-6` (70.7× the pin; 8/20 cells over; eigenvalues 8 (+7.07e-5) and 16 (−5.59e-5) carry it). The BGW band arm REMAINS GREEN: MAE 3.4707 / max 8.8728 meV vs band 10/25 (the frozen ref itself sits 3.4650/8.8774 — the move is noise-level). Both Si COHSEX gates BYTE-IDENTICAL before/after (eqp data md5 `139265eadb0fd1e96483e13d18e45fe8`); the BGW Σ stats identical at full float precision. NOTE: the Si evidence proves Si is BLIND to the draw bug (cyclic-permutation degeneracy), NOT that the bug is small — on non-cubic cells it is a bias worth ~50 % of the mc-average correction (z=376); the guard is `tests/test_vcoul_minibz_head_draw.py`, not any Si gate. Prediction honesty: the fix commit predicted this red in a 1e-4..1e-2 eV window; the measured 7.1e-5 eV is a decade below — the red, the byte-identity and the band survival were all predicted correctly, the magnitude estimate was conservative | RESOLVES AT MERGE — the owner has authorized adopting the refreeze candidate (`/pscratch/sd/j/jackm/svc_vcoul/_gates_after/bse_eigenvalues_candidate.dat`, md5 `cab1dd48…`, generated at 358bb0b on the BUILD_NOTES pins, sidecar README has jobids/deltas) as `tests/regression/si_bse_debug/bse_eigenvalues_ref.dat` at the integration head, turning this arm green there. Until that adoption lands, this is the suite's ONLY expected red from the vcoul fix. Loosening `ATOL_FROZEN_EV` was refused — it is a bit-reproducibility pin, not a physics band |
