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
| ~~`tests/test_bse_bgw_regression.py::test_bse_matches_frozen_and_bgw` (si_bse_debug frozen arm)~~ | MAE 3.8203 meV / max 9.9236 meV against a 1e-6 eV pin; its BGW *band* arm IMPROVES | **STRUCK — GREEN.** Re-cut 2026-08-08 at the gate's own geometry on a 400-iteration budget, validated against a freshly regenerated dense reference (0 missed levels, 4.8 neV worst).  See THE RE-CUT WAVE amendment at the top of this file |
| ~~w-omega `CHAIN_REL_TOL` gate (`tests/test_bse_w_omega_chain_scan.py`)~~ | **NEVER RED — the prediction was never run.** 12/12 green on the re-cut tree | **STRUCK.** The gate's operator is synthetic and defined inside the test, and its “frozen reference” is the old implementation evaluated live in the same process — no exchange site is reachable from it and nothing in it can go stale.  See THE RE-CUT WAVE amendment |
| ~~exciton-bands warm-cache eigenvalue check~~ | **NO SUCH GATE EXISTS.** The cells the warm-cache landing actually added are three Krylov-jit persistability cells in `tests/test_bse_nontda.py`, all green, none of which compares an eigenvalue to anything stored | **STRUCK.** The only stored eigenvalue reference in the test tree is `bse_eigenvalues_ref.dat`.  See THE RE-CUT WAVE amendment |

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

**⚠ THOSE TWO FIGURES ARE PRE-EXCHANGE.** 4.27 meV and miss-3 were measured
before this landing.  Post-fix the same budget is 66.8 meV off
(`DAVIDSON_COMPETITIVE.md` §1.3), reproduced at 66.126 meV with 2 of the
true lowest twenty missing by the re-cut wave, which also measures the 400
budget at 4.8 neV rather than 3.9 µeV once the payload is regenerated
rather than reused.  The prescription is unchanged; only the size of what
it prevents is.  See THE RE-CUT WAVE amendment at the top of this file.

## Registered defects (found by the perf lane's convergence census, in passing)

> **ALL THREE STRUCK 2026-08-09.**  The same campaign that registered them
> also fixed them, and this block spent the interval describing fixed defects
> as open — the exact failure mode the SlabIO lander named when it closed its
> own rows: *a ledger listing a fixed defect as open burns the next reader's
> budget the same way the reverse does.*  Struck in place, per this file's
> convention.  Audit that caught it:
> `~/lorrax_bse_perf_2026-08-08/ASIDES_AUDIT.md` §B1.

- ~~`davidson --write-eigs` dies at P>1 (`device_get` on non-addressable
  arrays in `write_eigenvectors_stream`).  Flag-path; the suite never
  exercises it, which is how it stayed hidden.~~  **FIXED by `df361cd9`**
  ("the eigenvector writer stops trusting one solver's layout"), which is in
  `main`; `src/bse/bse_io.py:298-320` documents the fix at length — the
  writer fetches through `common.collectives.gather_to_host` instead of
  assuming a solver's layout, and `bse_lanczos` pins the Davidson branch to
  the same replicated convention.  **A SEPARATE, UNCONFIRMED `--write-eigs`
  HANG AT P=4 SHARDED PREDATES THIS FIX** — observed at `995f9e9d`, which is
  not a descendant of `df361cd9` — so a reader whose run **hangs** (killed in
  the eigensolve, no `eigenvectors.h5`) rather than dying with a
  non-addressable-array error is **not** looking at this fixed defect; that
  observation owes one confirmation leg and carries its own row in
  `~/lorrax_service_phase/SMALL_ISSUES.md` (row 14).
- ~~`bse_feast --feast-ritz` cannot run multi-process at all
  (`_get_feast_runner` closes over non-addressable arrays).  Same class.~~
  **FIXED by `feb69f70`** ("the FEAST runner takes its operands instead of
  baking them in"), which is in `main`; `src/bse/bse_feast.py:106-132` now
  threads the ten operands through as RUNTIME ARGUMENTS (`matvec_operands`)
  instead of closing over the `data` dict.
- ~~`bse_w_exact.py:634`'s `max_gmres` column is a LOGGING defect, not a
  solver defect: the column is filled with a residual while `:248` discards
  the real iteration count — a two-line driver fix.  It is why the census
  had to lift the cap to measure convergence at all.~~  **FIXED**;
  `src/bse/bse_w_exact.py:716` carries the *"``max_resid`` was headed
  ``max_gmres``"* note and the column is now headed for what it holds.

Evidence for all three as originally registered:
`~/lorrax_bse_perf_2026-08-08/CONVERGENCE_CENSUS.md`.
