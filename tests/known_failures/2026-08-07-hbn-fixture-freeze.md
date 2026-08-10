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
