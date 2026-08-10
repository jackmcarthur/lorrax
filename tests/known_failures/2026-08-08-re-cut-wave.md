# AMENDMENT — THE RE-CUT WAVE, on `main` (2026-08-08, owner-ratified one-cut re-freeze)

The exchange-conjugation and mini-BZ-head landings moved the BSE spectrum by
design, and three frozen arms were listed below as red pending a single
re-cut.  That re-cut is taken here.  One of the three needed cutting; the
other two were never red, and saying so is most of what this amendment is
for.

| | |
|---|---|
| machine | Perlmutter, lx pool (JID 56506934), 4 nodes, 4×A100, Shifter, `lx test` |
| module | `LX_BASE_MODULE=lorrax_J070`, jax `0.7.0.dev20260808` |
| tree | `/pscratch/sd/j/jackm/perf_bse_0808/wt_recut`, branch `chore/recut-wave-2026-08-08` off `main` @ `81a285af`.  The `[lx] source tree:` line was read on every leg |
| `.so` pins | the restage candidate, unchanged: device md5 `c680c229…`, host md5 `91f330c3…` |
| `LORRAX_FFTW3_SO` | PINNED on every leg |
| artifacts | `/pscratch/sd/j/jackm/perf_bse_0808/recut/` — `_logs/` (gw, bse_400, bse_200, dense, gate_green, gate_red, census), `deck_cut/dense_eigs_ev_recut.npy` (**read the convention warning below before scoring anything against it**), `deck_cut/H_dense_recut.npy` |
| prose record | `~/lorrax_bse_perf_2026-08-08/RECUT_WAVE.md` |

**A warning about `dense_eigs_ev_recut.npy`, added 2026-08-09.**  That file was
cut at the deck's DEFAULT `mc_average_vcoul_body = true`, and its filename says
nothing about it.  It is therefore payload-matched only to cells that share
that flag, and it is **not** a valid reference for the convention-matched cells
that run `mc_average_vcoul_body = false`: scoring one against the other returns
the mc-average difference, not a solver error, and nothing in the file or its
name will tell you which of the two you just measured.

A later lane walked straight into this.  Scoring a 480-centroid,
`rcond = 1e-10`, convention-matched cell against this file it read 0.717041 meV
MAE and 1.303 meV max — six orders above the 4.8 neV that the 400-iteration
budget is certified to below — and on tracing the discrepancy found the number
was not solver error at all but an independent six-digit reproduction of the
attribution's separately measured 0.7170 meV flag difference.  Solver error
proper stays certified exactly where this amendment puts it: 400 iterations sit
mid-plateau, 4.8 neV from the exact dense spectrum **on a matched payload**.
Recorded in `~/lorrax_bse_perf_2026-08-08/W_HEAD_ISDF_FLOOR.md` §3.5.

## What the geometry prescription actually names

The prescription said to cut at the gate's own configuration, "1 GPU,
px=py=2".  Those are not two independent settings: `bse_jax` builds its mesh
with `create_mesh_2d()`, which takes every visible device, so `--px/--py` are
**inert on the solve path** and only the device count decides the mesh.  Under
`lx test` the gate's own `conftest` pins the pytest process to one GPU, the BSE
subprocess inherits that, and the run is single-process on a **1×1 mesh** — the
driver says so in its startup block.  The re-cut was taken at exactly that,
with `--px 2 --py 2` on the command line because the gate passes them.  This is
also why the old reference, cut at P=4, could never be seen green by its own
gate.

## The dense validation, regenerated on this tree

Never take a re-cut against a stored dense file.  The 1024-dimension exact
spectrum was regenerated here by materialising the production matvec column by
column and diagonalising it, on the **same GW-regenerated payload** and the
**same 1×1 mesh** the candidate run used, so "did the two solves span the same
transition space" is true by construction.  Probe integrity:
`max|H−Hᴴ| = 1.806e−12` (rel `2.916e−12`), `‖H−Hᴴ‖_F/‖H‖_F = 9.456e−12`,
linearity/ordering `1.499e−15`, `tr(H) − Σλ = 9.095e−13` eV.

| budget | worst-level error vs the fresh dense reference | true lowest-20 levels MISSED |
|---|---|---|
| 200 (shipped) | `1.6293e−03` eV nearest-match; **66.126 meV** in-order | **2** — true levels 2 and 20, with the returned list closing up behind them |
| **400 (cut here)** | **`4.8159e−09` eV (4.8 neV)**, mean `1.9687e−09` eV | **0** — 20 distinct true levels, no duplicates |

The Lanczos spectrum at 400 reproduces the exact spectrum to **all eight
decimals the fixture stores**.  The in-process sweep on the same operator reads
0.0000 meV at 300, 400 and 500 alike with all 13 levels below 2.40 eV found, so
400 sits mid-plateau rather than on an edge.

**A correction the peer fleet needs.**  `DAVIDSON_COMPETITIVE.md` §1.3 reports a
hard ~3 µeV accuracy floor on this deck and attributes it to an antihermitian
residue of `3.051e−05` relative.  That residue is a property of **the payload,
not the tree**: that lane ran against a cached ISDF/vcoul restart in which the
head fix is inert (the trap `RITZ_ORTHO_PROBE.md` §5 names).  Against a
GW-regenerated payload the residue is `9.456e−12`, seven orders smaller, and
there is no 3 µeV floor — the solver reaches 4.8 neV.  Any lane scoring against
a dense reference should regenerate its payload, not only its dense file.

## The one arm that needed cutting

`tests/regression/si_bse_debug/bse_eigenvalues_ref.dat`, rewritten through the
sanctioned path (`chmod u+w` → write → `chmod a-w`; the file is `a-w` at rest
under `harness.protect_fixtures`).  Format unchanged: header, 4-wide index,
8-decimal eV.

| | lowest state | max ｜Δ｜ vs the old file | mean | MAE |
|---|---|---|---|---|
| old → new | `2.34886489` → `2.35372258` eV | **11.4246 meV** (state 16) | +2.7769 meV | 3.6292 meV |

The old file was red against the new run at `1.1425e−02` eV against an
`ATOL_FROZEN_EV` of `1e−6`, which is four orders of headroom — it was never
going to survive, and the pin stays at `1e−6` because it is a
bit-reproducibility pin rather than a physics band.

The gate's **BerkeleyGW band arm stays green** and is now the exact operator's
own number rather than a partly-converged one: MAE **6.5216 meV** against a
10 meV band, worst **9.8398 meV** against a 25 meV band.  For the record, the
old frozen values sat at MAE 4.1192 / worst 11.4453 meV; the MAE rises and the
worst falls, and both are inside the band.  The shipped 200-iteration spectrum
would have read MAE 13.9014 / worst 73.9339 meV — outside **both** bands, which
is an independent statement that 200 was not merely imprecise.

The gate's Lanczos budget moved 200 → 400 in
`tests/test_bse_bgw_regression.py`, with the measurement above written into the
constant, and the fixture README's paragraph telling the reader **not** to raise
the count was corrected — it was measured in the 8v8c quasiparticle
configuration this gate does not run, and is false in the 4v4c DFT one it does.

| leg | result |
|---|---|
| gate at its own geometry, after the cut | **1 passed in 39.77 s** |
| red twin — perturb state 9 by 10× `ATOL_FROZEN_EV`, re-run | **1 failed**, `Mismatched elements: 1 / 20`, max violation `1.e-05` |
| restore | md5 `0f6ed113eda609eebecad5e8657caabc` before and after, byte-identical |

## The two arms that were never red — measured, not argued

Both were listed below as "expected red", and both were listed **without a
measurement**.  On this tree, at their own geometry, they are green, and the
reason is structural rather than lucky.

| listed gate | measured here | why the prediction was wrong |
|---|---|---|
| w-omega `CHAIN_REL_TOL` gate, `tests/test_bse_w_omega_chain_scan.py` | **12 cells, all green** | Its operator is **synthetic and defined inside the test** — a random Hermitian (c,c) coupling with `jnp.zeros(1)` standing in for every physical tensor — so no exchange site is reachable from it.  And its "frozen reference" is not a stored artifact at all: `_eager_reference_chain` is the OLD implementation kept verbatim in the test file and evaluated live in the same process, so it moves with the code by construction.  There is nothing here that a physics change can put out of date. |
| "exciton-bands warm-cache eigenvalue check" | **no such gate exists**; the cells that landed with that work are green | The warm-cache landing (`1c1da604`) added three Krylov-jit persistability cells to `tests/test_bse_nontda.py`, none of which compares an eigenvalue to anything stored.  The only stored eigenvalue reference in the whole test tree is `bse_eigenvalues_ref.dat`, cut above.  `exciton_bands`' own warm re-run compares a solve against a second solve **in the same process** and is now off by default (`e69a867f`); it has no frozen side to go stale. |

Neither row is evidence of anything having been fixed.  They are two rows that
should never have been written as reds without first being run, and they are
struck as **never-red** rather than as **now-green** so that the distinction
survives in the record.

## Left standing, deliberately

`tests/test_gw_jax_regression.py::test_hbn_matches_frozen_reference` is **not**
re-cut here and its row below stands.  This wave's prescription — the gate's own
geometry, the fix underneath, a converged budget validated against a freshly
regenerated dense reference — is a `si_bse_debug` prescription, and none of its
three pins transfers to hBN unexamined: there is no dense instrument for that
deck, the adjudication that would say the new head is *closer to the truth* is
`HBN_HEAD_ANCHOR_2026-08-08.md`'s and not this worker's, and re-cutting a
0.128 eV reference on somebody else's verdict is exactly the move the vcoul
lane declined to make for the same reason.  It stays red-by-design with its
row.

`tests/test_bse_vq_interp.py::test_loo_accuracy_vs_reference_thresholds` also
stands.  The exchange amendment's own CORRECTION block hands the threshold
re-derivation to "the re-cut wave"; re-deriving an empirical accuracy threshold
against the corrected B block is a physics-tuning decision with an
adjudication attached ("if the re-derived floor lands far above the old one,
that is a finding"), not a reference re-cut, and it is registered here rather
than taken in passing.

## The census, and the set-diff

Two legs on the re-cut tree. The full-suite leg was run in the geometry the
symmetry-landing amendment above used (default `lx test` xdist); the BSE leg was
run single-process, which is the geometry the BSE cells are meaningful in, since
the xdist cascade over these files is a documented collection artifact.

| leg | result | artifact |
|---|---|---|
| full suite — `tests/` + all six services, xdist | **20 failed, 2223 passed, 66 skipped, 2 xfailed** (2311 cells, 683.19 s) | `_logs/census_recut.{xml,log}` |
| BSE subset, single process (9 files) | **2 failed, 55 passed, 1 deselected** (384.73 s) | `_logs/census_bse_iso.{xml,log}` |

**The struck rows are green.** `test_bse_matches_frozen_and_bgw` passes in both
legs; `test_bse_w_omega_chain_scan` is 12/12 in both; the `test_bse_nontda`
persistability cells are green in both. The two reds in the single-process BSE
leg are exactly the two rows this amendment deliberately leaves standing —
`test_loo_accuracy_vs_reference_thresholds` and
`test_hbn_matches_frozen_reference`.

**Set-diff on the 20, by name.** Eighteen are already in this file:

| cells | what | where listed |
|---|---|---|
| 11 | the Class B loader-order reds — `test_gpu_pinning` ×5, `distrib_la ... test_distrib_la_contract` ×5, `vcoul ... test_the_whole_public_surface_answers_with_no_lorrax` | the symmetry-landing amendment's "THE UNLISTED PRE-EXISTING REDS" |
| 1 | `vcoul ... test_vcoul_imports_and_computes_with_no_scipy` | same, as the xdist artifact it is |
| 2 | `test_bse_setup_qchunk` ×2 | the BSE-perf merge amendment, class P2 |
| 2 | `symmetry_maps ... test_the_lorentz_mixing_matches_a_dense_numpy_reference` `[1-1]` **and `[2-2]`** | the merge amendment lists `[1-1]`; `[2-2]` is the same cell at a second parametrisation and the same cross-service conftest collision, not a new failure |
| 1 | `test_loo_accuracy_vs_reference_thresholds` | the exchange amendment's CORRECTION block; left standing above |
| 1 | `test_hbn_matches_frozen_reference` | the vcoul head amendment; left standing above |

The remaining two were **A/B-attributed against a pristine baseline worktree
detached at `81a285af`** (`/pscratch/sd/j/jackm/perf_bse_0808/wt_recut_base`),
because neither could be waved through:

| cell | recut, 1 proc | baseline `81a285af`, 1 proc | baseline, xdist | verdict |
|---|---|---|---|---|
| `test_bse_w_omega_chain::test_w_omega_chain_matches_oracle_q0` | **green** | **green** | **green** | red only in the FULL-SUITE collection — the documented xdist/collection cascade over these files, reproduced |
| `test_bse_w0_resolvent::test_wq_resolvent_matches_restart_finite_q` | **red** | **red** | **red** | **pre-existing at `81a285af`, geometry-independent, and NOT this landing's** — **STRUCK 2026-08-08**, diagnosed and fixed at `b5c0cf15`; see the amendment at the head of this file |

`1 failed, 4 passed` on all three legs, the same cell every time. **Zero reds in
this census are attributable to the re-cut**, which is what the set-diff had to
establish.

**One of those two was not listed anywhere, and now is.**
`test_bse_w0_resolvent::test_wq_resolvent_matches_restart_finite_q` is red on
plain `main` in both geometries and appears in no amendment in this file. It is
easy to see how it was missed: the symmetry-landing amendment's "THE NINE" table
closes `test_bse_w0_resolvent::test_w0_resolvent_matches_restart` as GREEN, and
that is a **different cell** — `w0` and static against `wq` and finite-q. A
reader scanning for "w0_resolvent" finds a row saying it was fixed. Registered
here as a real, unlisted, pre-existing red; not diagnosed, because it is not
this wave's and guessing at it would be worth less than naming it.

## CORRECTION — the convergence figures quoted below are PRE-exchange

The re-cut prescription in the exchange-conjugation amendment quotes the
shipped 200-iteration budget as **4.27 meV** off the exact dense reference and
**missing 3** of the true lowest twenty.  Those figures are from
`CONVERGENCE_CENSUS.md`, which measured **before** the exchange fix landed, and
they understate the problem on the post-fix operator.  Post-fix the same budget
is off by **66.8 meV** (`DAVIDSON_COMPETITIVE.md` §1.3: 66.7593 meV worst over
the lowest 20, with 13 of the true 14 levels below 2.40 eV found), independently
reproduced in this wave at **66.126 meV** in-order on a GW-regenerated payload
at the gate's own geometry, with 12 of 13 levels below 2.40 eV found and **2**
of the true lowest twenty missing.  The conclusion the figures were quoted for
is unchanged and strengthened: 200 does not return the true lowest twenty.
