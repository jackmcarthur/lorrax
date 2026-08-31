# Known test failures — full-suite census

> **WHICH RUN THIS FILE ACCOUNTS FOR (unchanged, restated 2026-08-09).**
> Every row below is about the **CENSUS** — `pytest --census`, equivalently
> `pytest -m census`. Since 2026-08-09 a bare `pytest` is the fast default
> gate (the Si end-to-end calculation for the drivers a branch touched, plus
> the services' suites); it is a strict SUBSET of the census and it is not
> what this file counts. **Nothing about this file's meaning, scope or
> accounting changed with that split** — `--census` collects exactly the set
> a bare `pytest` collected before it, plus this branch's own two new
> selection cells and nothing else. Re-measured after the 2026-08-10 rebase
> onto `chore/anchor-window-pin-2026-08-09`: a bare `pytest` at the base
> collects 3369 cells over 203 files, `--census` here collects 3371 over the
> same 203, and the whole difference is
> `tests/test_service_selection.py` going 14 → 16 — the two cells this branch
> adds to prove its own claim. `-m census` collects the identical 3371, and
> `--no-services` and `LX_SKIP_SERVICES=1` collect the identical 2441.
> Do not reconcile a KNOWN_FAILURES row against a default-gate run; it will be
> short by ~2400 cells for reasons that have nothing to do with the row.
> Evidence: `/pscratch/sd/j/jackm/fastgate_rebase_0810/collect_*.raw`.
>
> **WHAT A DEFAULT-GATE RUN LOOKS LIKE AT THIS HEAD (2026-08-10).** A bare
> `lx test` collects 935 — 930 service cells and the five end-to-end driver
> cells — and comes back **0 failed across `tests/`**, which includes the Si
> BSE anchor now that `chore/anchor-window-pin-2026-08-09` pins its band
> window. The nine reds it does report are all in `services/`, and they are
> **pre-existing, not this branch's**: the same three suites run at the base
> commit, under the same module and the same `.so` pair, return nine reds too,
> seven of them the identical node ids and the other two swapping between rows
> this file already lists. Neither this branch nor the pin branch changes a
> single file under `src/` or `services/`, so neither could have caused them.
> Evidence: `log_default_gate.txt` and `log_svc_base.txt` in the directory
> above. Note also that the run needs the site `.so` pair supplied — with no
> `LORRAX_FFI_SO` the FFI gate refuses before any driver starts, and every
> end-to-end cell fails on that refusal rather than on its numbers
> (`log_default_gate_nofficonf.txt`, kept as the negative control).

Two censuses live in this file.  The **Perlmutter** one is authoritative for
this tree; the **Frontera** one below it is the historical record from
2026-08-01 and is kept because several of today's reds are only legible
against it.

A release ships LISTED known-fails, never unknown ones: every non-passing
test in every leg is accounted for below, and every "it is the environment"
claim carries the arm in which it comes out FALSE.

---

# Amendments — one dated file each, under `tests/known_failures/`

Everything this file learned after its censuses were taken now lives beside
it rather than on top of it.  Each amendment is a **separate dated file** in
`tests/known_failures/`, moved here verbatim, and the table below is the only
place they are listed.

**THE CONVENTION, for the next branch that has something to record.**  Add a
new file `tests/known_failures/YYYY-MM-DD-<slug>.md` and add exactly one row
to the table below.  Do not write amendment prose into this file.  The point
is mechanical: two branches amending the ledger on the same night touch two
different files and one table row each, so they no longer collide in the same
region of the same file.  Three such conflicts were hand-resolved in one night
before this split, which is what it is here to end.

Rows are newest first; within one date they keep the order they had before the
split.  A row's status is the amendment's own word for itself — read the file
for the mechanism, the evidence and the disposition, all of which are unchanged.

| date | what it records | status |
|---|---|---|
| 2026-08-11 | [The fifth wall named: `refit_vq` asks the htransform for the top of its own band window, and that is exactly where `f(ε)` is zero](known_failures/2026-08-11-fifth-wall-is-the-f-transform-shoulder.md) | **DIAGNOSTIC ONLY — mechanism convicted on both probed parents, NO fix taken because the fix is not bounded; no source file changed, no tolerance moved.** `build_fH_R` sets `shift := max_k ε[nb−1]` and `f(ε) ≡ 0` above it, so the top of `ctilde`'s window carries no weight in `fH`; `refit_vq` is the one caller asking `compute_wfns_fi` for `band_window_fi=(0, nb)` — zero guard bands — because `refit_prepare` pins the fH window to the ζ-fit window. The per-band overlap is **not** block-unitary: the bands whose `|f|` reaches zero are ABSENT from the returned set (`min_k ‖O[m,:]‖` 0.23 on `dp2628n20`, 0.084 on `p2628n52`) and no other band moves at all. The Gram rank shift is broadband, not a cut artefact, so `spectral_closure` is not the lever. **`refit_ongrid_null` must NOT be re-pointed at `m_leg="stored"`** — see §6 |
| 2026-08-11 | [The narrowed ζ-fit window DOES clear `build_fH_R`, and the route is blocked one step further on by the refit's own ζ solve](known_failures/2026-08-11-narrowed-zeta-window-clears-fh-and-the-tile-null-still-refuses.md) | **MEASURED, merged at `fa86c6b8`. The third wall is down and a fourth is named.** `nb = 52` is the only legal window edge at or below 57 on this lineage (a Kramers pair sits across 53–56), and at that width `build_fH_R` accepts for the first time. The route then refuses at the refit's own ζ solve, which is neither the window nor `n_μ`; `nband` is doing double duty between the ζ fit and the BSE window, and that mismatch is what the next lane owns. No `.dat`, no `.png`, no tolerance moved, no code changed |
| 2026-08-11 | [The 2× centroid GW re-run does NOT open the ζ-window refit, because the bound everyone was quoting is the wrong bound](known_failures/2026-08-11-zeta-window-refit-needs-psi-rank-not-mu-count.md) | **MEASURED, merged at `2638cee7`.** The counting bound `n_μ,parent · n_s ≥ nk · nb_ζ` is necessary and **not** sufficient — what actually gates the refit is the ψ **rank**, not the μ count, so buying centroids does not buy the window. Also: the on-grid tile null samples only small \|Q\|, so passing it is not evidence about the interior of a path |
| 2026-08-11 | [The (μ, ν) restart-layout fact is stated twice, and the second statement cannot adopt the first](known_failures/2026-08-11-munu-layout-fact-stated-twice.md) | **OWNER ROW, not a failure; merged at `6d977832`.** Nothing is wrong today. The `cleanup/bse-loading-io` lane hunted for star/unfold/orbit/wedge logic in `bse_loading.py` that `symmetry_maps` already owns and came back empty and correct; behind the one near-duplicate it did find (`_MunuSlabPlan`) sits a real contract mismatch, so adopting the service call would be the wedge-transport decision wearing a refactor's clothes |
| 2026-08-11 | [Two BSE rank cuts sit outside `common/spectral_closure`, and the dedupe lane could not wire them](known_failures/2026-08-11-bse-rank-cuts-outside-spectral-closure.md) | **FILED, not wired; merged at `468c9118`.** Both are real and neither is fixable under a behavior-preserving dedupe charter — one has a contract mismatch with the closure surface, the other is unreachable from inside a `jit`. Neither is a red test today |
| 2026-08-11 | [The BSE head-override deck reader is a second parser, and it stays one](known_failures/2026-08-11-bse-head-deck-reader-duplication.md) | **FILED, deliberately not swapped; merged at `47040278`.** Not a defect and not a failure: `bse_head._parse_head_overrides` hand-parses the same two keys `gw_config.read_lorrax_input` reads, and the service reader was measured to have worse diagnostics plus an import tangle. Recorded so the next cleanup lane does not spend a leg re-deriving the same answer |
| 2026-08-11 | [`refit_vq` at P>1, and a second named certification grade](known_failures/2026-08-11-refit-vq-sharded-fetch-and-cert-grades.md) | **FIXED and merged at `72945497`.** `vq_interp`'s per-Q ζ'(G) host fetch is `gather_to_host`, not a bare `device_get` (`SMALL_ISSUES` row 39, closed with a red twin). `--cert-grade` gains a second and final grade: `reference` 0.01 meV, `visualization` 1.0 meV, both module constants, no third number reachable. **And the 0.858 meV "route representability floor" is REFUTED** — it was a four-corner sample; the interior on-grid Q of a real `--q-per-segment 16` path reads 22.952 meV, where the driver REFUSES |
| 2026-08-11 | [The two verdicts the downfold q-sign fix voided, re-taken on the fixed tree](known_failures/2026-08-11-qsign-recut-verdicts.md) | **MEASUREMENT COMPLETE, merged at `9c70b5a3`. No code landed, no tolerance moved.** Pre-registered and fully scored, wrong prediction included. The child certification is now 2.593 meV and still REFUSED at the 0.01 meV gate, but the residue is L-truncation — a floor on the *route*, not on the child. The predictor separation is confirmed at 109×. The orbit-floor verdict, re-cut on clean physics, is unchanged in direction and stronger in margin: point-picked 185 beats orbit-floored 168 on every instrument, and the FEAST non-convergence is gone. One new defect found on the way and registered in its §4 rather than fixed — **and that §4 is already superseded**: the bare `device_get` it names at `vq_interp.py:2778` is the one `93f8b572` replaced with `gather_to_host`, so `refit_vq` does run at P>1 on an even n_μ today. Its P=1 parent control stands as reported |
| 2026-08-11 | [FIXED — the downfold built its transfer at −q and applied it at +q](known_failures/2026-08-11-downfold-gram-q-sign.md) | **ROOT CAUSE FOUND AND FIXED, merged at `0578bc89`** — the wedge child unfolds. `downfold.pair_density_gram` took the ISDF fit kernel's own q convention for granted and so built the Gram at the conjugate momentum. This closes the "mechanism is not identified" state the orbit-economics owner row below left open, and it convicts a two-day-old diff rather than any established machinery |
| 2026-08-11 | [AMENDMENT — the third wall is gone and the curve is still not drawn, because the object behind it failed its own certification by 772×](known_failures/2026-08-11-exciton-bands-bse-window-refit-certification.md) | **AMENDS the off-grid-Q row below; merged at `e1fd0fea`.** `--refit-window=bse` fits ζ' on the deck's window instead of the producer's, which turns this lineage's Galerkin rank bound from 3840 into 1280 against a 1920 basis and so reaches the refit for the first time. The contracted dual-solve certification then reads a worst 7.719 meV against the 0.01 meV `reference` gate — 772× over — and the driver refuses. No exciton curve produced |
| 2026-08-11 | [The spectral-closure guard cut the WRONG WAY: the owner's ruling is that a cut landing mid-block DROPS the whole block, and the landed default kept it](known_failures/2026-08-10-spectral-cut-closure.md) | **CORRECTED and merged at `01d462e4`.** `DEFAULT_DIRECTION = "drop_block"` at every wired site; `keep_block` survives as a source-level opt-out that **no site uses** (two ratchets). κ_eff now moves the other way and can only improve, so the call sites' cap assertions lost their slack term. armF is the control: its cut falls in a gap, so ranks {1095, 1098} must be UNCHANGED |
| 2026-08-10 | [OWNER ROW: closing the downfold's star symmetry by COMPLETION is uneconomical on a deck with few orbits, and the alternative is to select in orbit blocks](known_failures/2026-08-10-downfold-orbit-economics-owner-row.md) | **SPEC ONLY, owner ruling owed** — completion at the production μ_S costs the whole parent basis (185 → 480); the orbit-mode kernel that would avoid it already ships on the centroid-generation path. **Superseded in part:** the points-in/floor-to-orbits interface landed at `7a7fe7a8`, and the 2026-08-11 q-sign re-cut above then measured the resulting orbit-closed 168 basis as *beaten* by the point-picked 185 on every instrument, so this row's open question is now which basis to buy, not whether one can be built |
| 2026-08-10 | [THE JAX CACHE CONTRACT — found state, fixed state, and what is red-listed](known_failures/2026-08-10-jax-cache-contract.md) | **FIXED and merged**; four of the five class-B siblings canonicalized, the fifth red-listed with its reason. Companion gate `tests/test_jax_cache_contract.py` and the `LORRAX_JAX_CACHE_KEYDUMP` dump; a `xla_compiles=0 vetoed=0` report is not evidence until the cache **key set** is compared across ranks |
| 2026-08-10 | [`distrib_la` grew a `rank_cut` surface to carry the degeneracy-closure guard, and it was REVERTED whole on the owner's ruling](known_failures/2026-08-10-distrib-la-rank-closure.md) | **REVERTED (owner ruling 2026-08-11), net deletion.** The rule belongs at the truncation sites and the service has none — `distrib_la` factors and solves, it never decides a rank — so `rank_cut`, `cholesky_pivot_spectrum`, `lu_rank_spectrum`, `closure.py`, the `closure=` kwargs, the `LORRAX_DISTRIB_LA_CLOSURE` dial and the consistency cell are all gone. The guard lives once, in `common/spectral_closure`, at the monorepo call sites |
| 2026-08-10 | [The phdf5 WFN reader loses its dataset handle — `read_kchunk_union: ds_id is invalid` — but only when densification is active](known_failures/2026-08-10-wfn-phdf5-kchunk-union-ds-id.md) | **OPEN, mechanism unknown; workaround `LORRAX_WFN_BACKEND=eager`** |
| 2026-08-10 | [The downfold on `q_irr`: the ζ refusal removed, and the star-stability condition named — plus the owed P=4 GPU leg, paid 2026-08-10, and the two defects that leg found, repaired 2026-08-10](known_failures/2026-08-10-downfold-qirr-star-stability.md) | ζ transport **FIXED, merged at `47657990`**; **P>1 defect FIXED, pushed on `fix/owedlegs-p4-repairs-2026-08-10`** — `lorrax-downfold` now runs at P=4 and the `g0` cross-check has been read there; the three 1×1-shaped gate cells are mesh-aware; wedge-storable child still **BLOCKED** — real-deck orbit-closure rate **0 of 185** by accident, the "greedy pivot order" consolation refuted and now corrected in `describe()`, economics in the owner row above. **Read with the 2026-08-11 rows:** `0578bc89` fixed the q-sign that poisoned every child measurement taken here, and `7a7fe7a8` added the floor-to-orbits interface that builds an orbit-closed basis on purpose |
| 2026-08-10 | [The IBZ cascade and the forced full-BZ path disagree by 110 meV on Σ at Si 6×6×6 — adjudicated on the 4×4×4 anchor, where they agree to 7.6 µeV; at 6×6×6 BOTH arms break the Σ k-star identity and the breakage is already in Σ_x, upstream of W](known_failures/2026-08-10-ibz-cascade-vs-full-bz-sigma-6x6x6.md) | **ADJUDICATED — neither code path implicated; the Si 6×6×6 bundle is NOT valid as a reference, suspect is its over-complete ζ basis, one cheap leg named** |
| 2026-08-10 | [The two staged-reshard instruments were never true on a GPU — the HLO pin counted a spelling CUDA does not use, and the remat red twin cannot go red on any platform (retires P4)](known_failures/2026-08-10-staged-reshard-instruments-were-cpu-shaped.md) | FIXED, pushed |
| 2026-08-10 | [SPEC: an unknown deck key should REFUSE, not log-and-proceed — the corollary of `AGENT_PREAMBLE.md` measurement-discipline rule 1](known_failures/2026-08-10-unknown-deck-key-refusal-spec.md) | **PROPOSAL, owner ruling owed** |
| 2026-08-10 | [`mu_small = auto` sizes a downfold by rank, not by accuracy — and the docs recommended it, on a deck that came out 2.087 eV wrong](known_failures/2026-08-10-mu-small-auto-rank-sized.md) | **OPEN** |
| 2026-08-10 | [An exciton bandstructure on a BULK crystal could not be drawn as a bandstructure: off-grid Q needs `vq_interp`, which is slab-only — and the deck's 8-point path was provably the maximum its 4×4×4 exchange tensor can express](known_failures/2026-08-10-exciton-bands-offgrid-Q-is-slab-only.md) | **THREE STACKED WALLS, ALL NAMED. (1) interp is slab-only — no arbitrary-Q exchange on a bulk deck at all. (2) FIXED on `feat/xbands-dense-path-2026-08-10`: the refit's kernel, 3.8e-2 → 3.3e-14, 64/64 on the on-grid null. (3) BLOCKING, structural: the refit needs the ζ-fit window, whose Galerkin rank bound `nk·nb`=3840 exceeds this parent's basis `n_μ·n_s`=1920 by 2×, so the refit is unreachable on this lineage and needs a GW re-run with a bigger centroid set. NO exciton table produced through the refit path.** **Wall (3) is SUPERSEDED — read the four 2026-08-11 rows above instead.** `--refit-window=bse` reaches the refit without a re-run (`e1fd0fea`), the certification there refuses at 772× over gate, a 2× centroid buy does NOT open the bound because the bound is ψ-rank and not μ-count (`2638cee7`), and a narrowed ζ window does clear `build_fH_R` only for the route to refuse one step further on (`fa86c6b8`). Still no exciton table through the refit path |
| 2026-08-10 | [`bse.exciton_bands` could not read a downfolded bundle, three ways, and not one of them was ever a red](known_failures/2026-08-10-exciton-bands-downfolded-bundle.md) | FIXED, pushed |
| 2026-08-10 | [The sharded BSE loader injected a screened q=0 head onto the bare-Coulomb fallback tile](known_failures/2026-08-10-sharded-whead-on-bare-v-fallback.md) | FIXED, pushed |
| 2026-08-10 | [`--vq-mode interp` is a slab path meeting a 3-D cell; punch row 23 was never an off-grid defect](known_failures/2026-08-10-vq-interp-3d-bulk-row-23.md) | MEASURED, REFUSED |
| 2026-08-10 | [The coarse→fine W densifier trigonometrically interpolated a divergent head, so the head channel rang in sign across the fine zone](known_failures/2026-08-10-w-densifier-head-interpolation.md) | FIXED |
| 2026-08-09 | [The Si BSE anchor was red on `main`; the anchor deck now pins its band window](known_failures/2026-08-09-si-bse-anchor-window-pin.md) | CLOSED 2026-08-10 |
| 2026-08-09 | [The dipole producer had no pseudopotential pre-flight — the fixed successor to the `dZ is None` row, correcting three of its claims](known_failures/2026-08-09-dipole-producer-pseudopotential-preflight.md) | FIXED |
| 2026-08-09 | [`exciton_bands` at P=4 never exits](known_failures/2026-08-09-exciton-bands-p4-never-exits.md) | FIXED |
| 2026-08-09 | [The two absorption drivers disagree on a conjugation — the fifth member of the crossed-conventions class](known_failures/2026-08-09-absorption-drivers-conjugation.md) | FIXED |
| 2026-08-09 | [The opt-in mini-BZ exchange-head average is the wrong moment; the default finite-Q path is already exact and must not be "fixed"](known_failures/2026-08-09-minibz-exchange-head-moment.md) | **FIXED, both halves LANDED** — `fix/bsekgrid-head-key-2026-08-09` (the unconditional-enable half) and `feat/head-moment-tensor-2026-08-09` (the wrong-moment half) are both ancestors of `main`; the amendment file still reads "pending land" because it was written before they merged |
| 2026-08-09 | [Four rows transcribed out of the asides audit — true at `main` already, and in no ledger anywhere](known_failures/2026-08-09-asides-audit-four-rows.md) | mixed, per row |
| 2026-08-09 | [The three stacked table campaigns land as one merge: 84 certified entries at 100% certification](known_failures/2026-08-09-certified-tables-lineage.md) | LANDED |
| 2026-08-08 | [`kirr_fullids`: the wedge row map, fixed, and the one fixture it leaves stale](known_failures/2026-08-08-kirr-fullids-wedge-row-map.md) | FIXED |
| 2026-08-08 | [`K^d_B` under ζ sharding: registered and struck](known_failures/2026-08-08-kdb-zeta-sharding.md) | STRUCK |
| 2026-08-08 | [`wq_resolvent` diagnosed and struck](known_failures/2026-08-08-wq-resolvent-struck.md) | STRUCK |
| 2026-08-08 | [The owner-ratified one-cut re-freeze of the arms the BSE landings moved by design](known_failures/2026-08-08-re-cut-wave.md) | LANDED |
| 2026-08-08 | [The symmetry landing on `main`, its census, and the 21 unlisted pre-existing reds it finally named](known_failures/2026-08-08-symmetry-landing.md) | LANDED |
| 2026-08-08 | [The vcoul head-slot landing on `main`, owner-approved after the hBN anchor](known_failures/2026-08-08-vcoul-head-slot-landing.md) | LANDED |
| 2026-08-08 | [The BSE exchange-conjugation landing on `main`, and what it turned red by design](known_failures/2026-08-08-bse-exchange-conjugation-landing.md) | LANDED |
| 2026-08-08 | [Census at the `integration/bse-perf-merge-2026-08-08` head @ `e69a867f`, and its set-diff](known_failures/2026-08-08-bse-perf-merge-census.md) | census |
| 2026-08-08 | [Census at the `integration/merge-checkpoint-2026-08-08` head @ `6a4f73da`, and its set-diff](known_failures/2026-08-08-merge-checkpoint-census.md) | census |
| 2026-08-07 | [The hBN fixture freeze on `main` @ `6feaa713` and its two new headline gates](known_failures/2026-08-07-hbn-fixture-freeze.md) | LANDED |

The census below is the original record and is unchanged by the split.

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

# eqp0.dat / eqp1.dat mixed two bases on the self-consistent path

FIXED IN CODE, END-TO-END MEASUREMENT PENDING, 2026-08-31.  The former
failure was real: the self-consistent finalize returned static fields in the
DFT basis but left `sigma_c_omega_kij_ry` and its at-DFT diagonal in the last
map's QP basis.  Output then added their diagonals to DFT-basis terms, a sum
that is valid only at `U = identity`.  Three guards made that dynamic
configuration fail closed from 2026-08-25 onward.

The finalize now rotates the full correlation operator exactly once with
`C_DFT(omega) = U C_QP(omega) U†`.  It scans omega rows on device, so a
band-sharded cube remains `P(None,None,'x','y')` and no full cube is gathered
to host.  The at-DFT diagonal cache is then rebuilt from the rotated cube;
a diagonal is never rotated element-wise.  The raw `sigma_mnk.h5` cube is
written before this output transform and therefore still records the last
map's QP compute basis.  The returned `SigmaResult` is DFT-basis-consistent
for EQP/output assembly.  The three guards and their documentation
restrictions were removed in the same change.

Focused evidence is green: 171 passed, 1 skipped across the rotation,
basis-contract, config, metallic-SC, zero-head, delivered-window, and
layering tests; a four-logical-device test checks the explicit `U C U†`
value and retained sharding.  CUDA P=4 JID 57754440, step 71, checked the
same row-scanned kernel on four ranks/four devices:
max absolute tile error `9.1551e-16`, output
`P(None,None,'x','y')`.  The requested Na end-to-end measurement remains
blocked before Sigma by two independent deck/planner preconditions: the
strict 24/48-band boundaries are not multiplet-clean, and the delivered
product-window planner refuses the exact `-10..+10 eV` arm at P=4 with
achieved residual `0.00510025` and amplification p99 `7183.46` (JID
57754440 step 91).  Neither refusal may be bypassed to manufacture a passing
gate.  The two newly reachable head-off metallic seams are fixed and tested,
but no SC iteration completed.  The measured SC `U`/fixed-point diagonal
comparison therefore remains owed, and the branch result must not describe
the capability as landing-verified until that artifact exists.

`tests/test_sigma_result_basis.py` pins the returned basis partition and the
single sanctioned cube-rotation helper.  One-shot and fixed-point paths keep
their existing basis contracts.

## Bespoke IBZ→full-BZ unfolding outside the symmetry service — census

REGISTER, 2026-08-15, `refactor/eqp-ibz-2026-08-15`.  The rule being
enforced: `services/symmetry_maps` owns symmetry unfolding, and a driver
that needs a full-BZ quantity from a wedge one calls it rather than
open-coding a star expansion, an index map or a k-matching loop.  A
by-shape sweep of `src/`, `tools/`, `scripts/` and the non-symmetry
services found eleven sites.  **Two were in scope and are FIXED on that
branch; the other nine are recorded here and NOT touched.**

Ordered by "could silently produce a wrong pairing".

### FIXED on this branch (five of eleven)

1. `src/bse/bse_window.py` `apply_eqp_corrections` — matched each full-BZ
   k to a wedge block of `eqp1.dat` by comparing MEAN-FIELD ENERGIES with
   `tol_ev = 0.01`, reachable through `input_file=None`, live in
   `src/bse/exciton_bands.py`'s `--eqp` path.  Right by accident (E_DFT is
   constant over a star) and silently wrong once two stars agree to 10 meV
   across the compared window — `best_ibz` then takes the QP shift from
   the wrong star, and the `matched.all()` gate catches only NO match,
   never a WRONG one.  Now one path, unfolding through the service
   adapter; `input_file` is required.
2. `src/bandstructure/htransform.py` `read_eqp_energies` — required a
   PRE-UNFOLDED full-BZ text file (`nk == sym.nk_tot`) and paired its
   `k-point N:` blocks to full-BZ k BY POSITION, checking only the count,
   never a coordinate.  The unfold itself happened one hop upstream in an
   out-of-tree `make_eqp_htformat.py`.  Now reads the wedge `eqp1.dat`
   directly, verifies the block coordinates against the deck's own wedge,
   and unfolds through the service.

4. `src/postprocess/rotate_wfn_to_qp.py` `find_kpoint_mapping` — a
   coordinate `argmin` at `tol=1e-6` with no uniqueness check, rebuilding
   the table `sym.kirr_fullids` already is.  **The highest-value fix in
   the list**: its output selected `U_mnk[ik_full]` and the QP energies
   for each reduced k, and `gw.eqp_bgw` reads the SAME `kirr_to_kfull`
   dataset out of `qp_wfn_rotations.h5` — so the two disagreeing about a
   k meant the rotated WFN and `eqp{0,1}.dat` disagreed too.  The fix
   turned out to be a deletion: the rotation file ALREADY carries
   `kirr_to_kfull`, written from `sym.kirr_fullids` by both producers
   (`gw_jax`/`write_results` and `sc_iteration.py:2365`), so the module
   now READS it and checks the service's own contract
   (`kpoints_crys[kirr_to_kfull] == wfn.kpoints`) rather than
   re-deriving it.  `--add-mapping` and
   `add_kpoint_mapping_to_rotation_file` are DELETED: that path
   recomputed the map by nearest-coordinate search and *overwrote* the
   service's dataset with an approximation of itself.  No jax import was
   needed, so the module stays jax-free.
5. `src/file_io/qe_save_reader.py` `_reduce_mp_to_ibz` — a 70-line
   hand-rolled orbit reduction (`_EPS = 1e-5` grid snap, `equiv` parent
   array, accumulated `wkk` weights), duplicating
   `symmetry_maps.find_irreducible_bz_points`, which does it in INTEGER
   kgrid coordinates with no tolerance at all and returns the orbit map
   this one discarded.  Now delegates; weights come from
   `np.bincount(irr_idx)`.
   **THE TRANSPOSE CONVENTION WAS MEASURED, NOT ASSUMED.**  The old code
   applied `S @ k` on the raw QE matrices while `SymMaps` builds its
   k-table as `sym_matrices.transpose(0,2,1)` — different operations, and
   no docstring settles which this grid wants.  Checked against the old
   implementation on the committed fixtures: on `si_cohsex_debug`
   (4x4x4, 48 ops, 8 IBZ points) the AS-STORED matrices reproduce it
   exactly and the transposed ones move k by 2.5e-01; on the 2-op
   `gnppm_debug` deck BOTH agree, which is why only a high-symmetry deck
   decides it.  There is no test coverage of this function in tree and no
   QE reference fixture, so that measurement is the only evidence — it is
   pinned in `tests/test_unfold_through_the_service.py`.

### STOPPED, and why — `vcoul/bgw_parity.py`

Item 3 below is NOT fixed, deliberately, on two independent grounds.

**The service does not expose what the rewrite needs.**  The call site
needs `(iq, S_k, kg0)`: an index, the symmetry row, AND the integer
umklapp vector, because `kg0` then shifts the whole G-list at
`bgw_parity.py:189` (`G_input = S_k @ G_miller - kg0`).
`find_irreducible_bz_points` returns `(irr_idx, sym_idx, irr_out)` and no
`kg0`; recovering it means recomputing `q - S_k @ q_table` locally, i.e.
putting back a piece of the hand-rolled matching the rule exists to
remove.  It also wants a q-GRID to integerise against, which
`bgw_parity` never receives — it holds fractions only.  Per the standing
instruction, "the service can't do this yet" is the answer rather than a
local workaround.

**And the surface is already under an owner ruling.**  `use_bgw_vcoul`
defaults False (`gw_config.py:1286`), no in-tree deck or fixture sets it
true, and `file_io/read_bgw_vcoul.py:23` states outright that the
surface is "data-dead in-tree" and under a repair-or-delete owner
question.  Rewriting a data-dead path that may be deleted is work
against a decision that has not been taken.

### REGISTERED, not fixed

3. `services/vcoul/src/vcoul/bgw_parity.py:44-92` `find_q_index` —
   reconstructs the star `q = S_k·q̄ + G` in a double Python loop with
   `tol = 1e-4` on fractional coordinates, FIRST MATCH WINS.  Two hazards:
   the tolerance is loose enough to alias on a fine grid, and when the
   little group is non-trivial the arbitrary `S_k` chosen is then used at
   `bgw_parity.py:180` to rotate the whole G-list, so an equally-valid
   different `S_k` gives a different G permutation.  Consumer:
   `src/gw/compute_vcoul.py:245-261`, which can additionally source
   `sym_mats_k` from a *different* WFN.  The module already carries this
   as "a registered owner question (repair-or-delete on the whole
   `use_bgw_vcoul` surface)".
4. *(FIXED — see item 4 under "FIXED" above.)*
5. *(FIXED — see item 5 under "FIXED" above.)*
6. `src/gw/downfold_run.py:1445-1490` — integer-keyed dict map, well
   guarded (membership refusal at `:1474`, bijectivity at `:1483`), EXCEPT
   the `np.arange` identity fallback at `:1470`, which the code's own
   comment says would "attach grid point i's transfer to wedge slot i and
   write a ζ of plausible, wrong numbers".
7. `src/bse/vq_interp.py:565-605` — rebuilds the full-BZ q list from a
   `meshgrid` when `rk` is short, re-deriving the service's own wrap/C-order
   convention.  Numerically gated and it refuses a genuine IBZ ζ, so the
   risk is ORDERING drift if the writer's convention ever changes.
8. `src/file_io/sigma_output.py:498-527` `compact_star_tables` — a Python
   re-implementation of `star_select`'s first-occurrence row order, whose
   own docstring says it must match that convention; it then feeds
   `symmetry_maps.star_broadcast` at `:637-640`, so a drift produces a
   silently wrong star assignment.
9. `tools/sigma_star_spread_decompose.py:245-260` (and `:417`) — a THIRD
   answer to the same compaction, using SORTED labels where the other two
   use first occurrence.  Diagnostic tool.
10. `tools/bgw_sigma_hp_to_fixture.py:265-272` and `:367-372` — maps BGW
    IBZ k onto LORRAX k by comparing `Eo`/`E_dft` vectors at `2e-3`.
    Self-referential: the fabricated star then feeds the `star_spread`
    statistic at `:288`, i.e. the metric meant to DETECT a broken unfold is
    computed over a star the tolerance invented.  Offline tool.
11. `src/psp/orbital_magnetization.py:384-401` — a hand-built
    "which k is this" dict for finite-difference neighbours.  Full-BZ →
    full-BZ, exact integer keys, so no aliasing; listed only because it
    duplicates the grid lookup the service owns.  Diagnostic path.
12. ~~`src/file_io/epsreader.py:136` `unfold_eps_comps`~~ — **DELETED
    2026-08-16.**  Registered 2026-08-15, removed the next day; a tombstone
    at the site records why and points at the canonical rotation.  Kept in
    this list because the SHAPE recurs: a τ-blind G-rotation, admitted as
    such in its own comment, sitting on a re-exported class with no live
    caller for long enough that nothing knew it was there.  Original entry:  A fifth independent
    `G' = S·G − G_umklapp` in `src/`, found by re-counting the register's
    "four implementations" claim by reading.  Two things make it worth a
    row rather than a fix:
    (a) it is **dead but shipped** — `EPSReader` is re-exported from
    `src/file_io/__init__.py:41`, and its three in-tree instantiators
    (`gw/head_correction.py:286-289` and two `scripts/checks/`) never call
    this method; the only callers are under `misc/archived_tests/`;
    (b) its own comment at `:126` says **"NO SUPPORT FOR TAU (FRAC
    TRANS) CURRENTLY"**, i.e. it is known-wrong on every non-symmorphic
    deck and nothing outside the file said so.
    NOT FIXED HERE: repairing a method with no caller is speculative, and
    deleting a re-exported public method is a surface change.  The
    decision owed is delete-or-fix, not repair-in-place.

**Correction to (3)'s recorded reason.**  It says "the service returns no
`kg0`".  A reader who greps finds `SymMaps.get_umklapp_vector`
(`maps.py:1944`) immediately and concludes this row is stale.  The method
exists; the real blocker is that it is **index-keyed to a
`(SymMaps, WFNReader)` pair**, and `bgw_parity.fill_v_grid_for_q`
(`:147-153`) receives raw fractions and a bare matrix stack, never those
objects.  Separately, `find_irreducible_bz_points` computes the rotation
and discards the umklapp at `maps.py:102` — the `% kg` **is** the thrown-
away `kg0`, and a `return_umklapp=True` there is about two lines.  The
row stands; only its justification needed correcting.

Test-side hand-rolls, lower priority and noted for completeness:
`tests/test_scissor_weights.py:169-175, 296-302, 333-338` (a `_FakeKStar`
oracle that re-implements the thing under test),
`tests/test_restart_qirr_consumers.py:78-83`, and
`tests/harness.py:803-830` `compare_to_bgw`, which assigns each LORRAX k
to a BGW IBZ k "on the whole `Eo` vector, not on k coordinates, because
the two codes do not order k the same way" — the same fingerprint class as
(10), and with the same self-reference, since the star it builds is what
`_star_spread` is then computed over.  The stated reason no longer holds:
`sigma_diag.dat` now carries `# kcrys` on every block and BerkeleyGW's own
`eqp.dat` carries coordinates in its headers, so a coordinate join is
available on both sides.

### Two rotation-math defects found by the 2026-08-15 symmetry inventory

REGISTERED, not fixed — both are behaviour changes in gate/bench code and
belong to the consolidation pass, not to the audit that found them.

1. **FIXED 2026-08-15, and VERIFIED BY EXECUTION at P=4.**
   `tests/multi_device/star_invariance_gate.py` used the WRONG TRS
   predicate.

   A/B on `cohsex_debug` (9 k, 3 stars, ntran=12, 3 TRS rows, and star
   label 2's FIRST member is a TRS row — the only configuration in which
   the two predicates differ), `srun -n 4`:

   | predicate | check 0 classification | residual |
   |---|---|---|
   | fixed (XOR) | 3 spatial + 3 TRS pairs | **5.599e-10** |
   | old (member's own flag) | 4 spatial + 2 TRS pairs | **1.400e-02** |

   The old predicate puts ONE PAIR IN THE WRONG BUCKET and the residual
   rises seven orders of magnitude — the false failure this row predicted,
   now measured rather than reasoned.

   **TWO THINGS THE EXECUTION ALSO ESTABLISHED, both worth knowing:**

   (a) **The gate passes on a deck that does not exercise the branch.** On
   `si_cohsex_debug` (64 k, 8 stars, ntran=48) it reports `TRS pairs=0`
   and VERDICT PASS at 1.169e-15. There are no time-reversed rows in that
   deck's selection, so a wrong TRS predicate is invisible there. Anyone
   running this gate for TRS coverage must use a deck with TRS rows;
   `cohsex_debug` is the one in tree.

   (b) **...and it cannot pass on the deck that does.** `RTOL = 1e-10`
   (`star_invariance_gate.py:77`) is tighter than `cohsex_debug`'s own
   noise floor of 5.599e-10, so checks 1 and 3 report FAIL there **on
   both legs**, before and after the fix, and the overall VERDICT is FAIL
   either way. Those two checks route through the service's
   `star_spread`/`star_broadcast`, which already used the correct XOR, so
   they are unaffected by the predicate — it is purely a tolerance
   calibrated for Si (1.169e-15) being applied to a deck three orders
   noisier. NOT changed here: the gate is superseded (its four checks were
   ported to `services/symmetry_maps/tests` on 2026-08-07) and retiring it
   is on the owner's cleanup checklist. Recorded so the next person to run
   it is not misled by a red that predates them.

   Note the file's own docstring is stale on one point: it says it imports
   `from common.symmetry_maps import ...` at :48, and it does not — it
   imports `from symmetry_maps import ...` through the service door, which
   is why it still runs at all.  It compares each star member against `T[mem[0]]` —
   the star's first FULL-BZ row, i.e. `star_broadcast`'s `"star_row"`
   operand — but decides conjugation with `if int(sidx[j]) >= n_spatial`,
   which is the `"ibz_slab"` predicate.  That is exactly the mix-up
   `star_broadcast`'s own docstring prices at **183.61 eV**
   (`maps.py:2294-2303`: the two predicates disagree on 6 of 9 k-points on
   `cohsex_debug`, with the real diagonal left exactly intact).  The right
   predicate for `star_row` operands is the XOR, `_star_conj_flags`
   (`maps.py:2167`, XOR at `:2194`).  Consequence: this gate will report a
   FALSE FAILURE on any deck whose star begins on a time-reversed row.
   It is a gate, not production, so nothing physical is wrong today.

2. **`tests/bench/charge_density.py:135` `_symmetrise_density` is a broken
   duplicate** of `src/gw/qsgw_density.py:270` `symmetrise_density`.
   Three defects against the canonical: (a) NO τ phase at all, so it is
   silently wrong on every non-symmorphic deck; (b) it rotates G with
   `sym.R_grid` (= `mtrx`) where the live convention is `sym_mats_k`
   (= `mtrx.T`) — the transposed convention, which is the one the stale
   `maps.py` comment corrected in this commit used to state; (c) it works
   in G-space with an FFT round trip instead of the r-grid permutation.
   Called unconditionally at `tests/bench/charge_density.py:130`.
   `src/psp/scf_potential.py:19-20` already records it as known broken.
   Fix is deletion in favour of the canonical, not repair.

Also noted, no action: `SymMaps.R_cart_forward` (`maps.py:1727`) exists
solely to NAME the transpose a rank>=1 Cartesian index needs, and has
**zero production consumers** — while the one live Cartesian-index
rotation (`src/psp/orbital_magnetization.py:172-177`) uses `R_cart`
untransposed.  Its defence (`:185-187`) is that the group is closed under
inverse so the PROJECTOR is transpose-invariant; that argument covers the
sum at `:189` but not obviously the per-op `keep` test at `:176`, which
tests individual ops.

### ~~`sc_on_ibz = true` HAS ROTTED~~ — FIXED 2026-08-15, and the default is now True

**CLOSED.**  Kept because the shape of the failure is the reusable part.

MEASURED 2026-08-15 on `gnppm_debug` with `qp_solver = self_consistent`.  The
flag defaulted False, **no deck in the tree set it, and no regression deck ran
the SC path at all** — every committed deck was `qp_solver = one_shot_dft`.  So
nothing had exercised this since it was written.

    sc_iteration.py:1397  _write_sc_eqp_snapshot
      -> eqp_bgw.py:145   write_bgw_eqp
    ValueError: e_qp shape (5, 46) does not match e_dft (9, 46)

**5 and 9 are the two different IBZs.**  With `sc_on_ibz` on, the loop reduces
H/E/U to the STAR wedge — 5 orbits on this deck — while
`_write_sc_eqp_snapshot` handed the result to a writer expecting the FILE
wedge, `wfn.kpoints`, which is 9 here.  It would NOT have crashed on
`si_cohsex_debug`, where the two wedges coincide at 8 — the same "right where
you test, wrong where you don't" shape.

FIXED at both boundaries by routing star wedge → full BZ → file wedge through
the service (`reduce_full_bz_to_file_wedge ∘ unfold_star_wedge_to_full_bz`);
`dump_qp_wfn_artifacts`'s length-matching `placements` dict went with it.
Default flipped to True on a measured A/B: **1e-6 meV per iterate under linear
mixing**, with the rCROP trajectory k-set dependent by construction (24.45 meV
by map call 5).  See `docs/architecture/symmetry_register.md`, "The SC loop
crosses both wedges".

**The lesson that outlives the bug**: a flag whose default is off, that no deck
sets, on a path no deck runs, is not covered by any number of green cells.
`tests/regression/gnppm_debug/gnppm_sc.in` now runs the SC path, and
`tests/test_sc_on_ibz_wedges.py` pins the boundary.

### TWO MORE pre-existing reds in `test_invariance_gates.py` — one FIXED, one that must NOT be re-frozen

Found 2026-08-15 by sweeping the 22 test files this branch could plausibly
touch. **Both reproduce IDENTICALLY at `0241118f`, the parent commit**, in a
clean worktree, so neither is the `sc_on_ibz` work — whose whole source diff
is two files (`gw_config.py`: the `sc_on_ibz` default; `sc_iteration.py`: the
SC boundary), both reachable only under `qp_solver = self_consistent`, which
neither of these gates uses.

**1. `test_fixed_point_frozen_qp_rotations` — a VALUE difference, ~25.7 meV.**

    Mismatched elements: 382 / 414 (92.3%)
    Max absolute difference among violations: 0.00189262 Ry  (25.75 meV)
    Max relative difference among violations: 0.1063308

against `atol = 1e-6` Ry. Shapes agree — `(9, 46)`, the full BZ — so this is
**not** the wedge move. It is a 25.7 meV move in `E_qp_nk_rydberg` on 92 % of
elements of `qp_wfn_rotations.h5` from a `fixed_point` run, against
`gnppm_debug/eqp_rotations_fixedpoint_ref.npy`, last touched at `b7654ee9`
(the `zeta_rcond` 1e-8 + band-range-centroid default change).

**DO NOT RE-FREEZE THIS ONE.** It is categorically unlike the three `.dat`
references migrated in this branch: those were a k-SET change with values
*proved* bit-identical (0.000000e+00 on both Si decks). This is a physics-level
value change with no explanation attached, on the QP eigenvalues, and the
first two columns still agree exactly (−5.651374, −1.595261, −1.363405) while
the ones between them do not — a pattern that wants diagnosis, not a new
reference. Candidate causes worth separating before anything is re-blessed:
the `zeta_rcond`/centroid-weight defaults that moved at `b7654ee9`, versus
anything in this branch's `4f26ecc5`/`74024fd1`/`954ba9c8` writer work.

**2. `test_mu_pad_flip_invariance_bispinor` — DIAGNOSED AND FIXED. It was a
test defect, and the physics invariant it guards holds exactly.**

Ran the deck at `LORRAX_EXTRA_MU_PAD` 4 and 0 and diffed the three files the
gate compares. Result:

| file | differing DATA lines | differing HEADER lines |
|---|---|---|
| `sigma_diag_bispinor_test.dat` | **0** | 2 |
| `eqp0.dat` | **0** | 0 |
| `eqp1.dat` | **0** | 0 |

Σ is **bit-identical** under the pad flip — the catastrophic transverse-pad
class this gate exists to catch (MoS₂ 668→672 moved Σ^B tile(2,2) from −0.15
to −117.9 eV) is not happening. The whole failure was:

    # star_spread_ev 1.265016891e-08     (pad 0)
    # star_spread_ev 1.265016181e-08     (pad 4)

a **7.1e-15 eV** difference in the 8th significant figure of a 1.3e-8 eV
quantity. `star_spread_ev` is a reduction over the full-BZ Σ printed to 9
significant figures, added to the wedge writers by `4f26ecc5`/`74024fd1`, and
on a clean deck its value IS roundoff. A reduction order over roundoff is not
bit-reproducible and cannot be made so.

FIXED at `tests/harness.py` `normalize_dat`, which now strips
`# star_spread_ev` alongside `# Generated by LORRAX`. **The data is still
compared byte for byte**, which is the whole content of the gate.

The general point, worth more than this instance: **adding a header line to a
file that a bit-identity gate compares can turn a real invariant into an
unachievable one**, and it does so silently — the gate goes red for a reason
that has nothing to do with what it tests. Any new header on
`sigma_diag`/`eqp*` must be checked against `normalize_dat`.

(1) is recorded rather than repaired: it is a real 25.7 meV diagnosis and not
in the symmetry-consolidation scope this branch is doing.

### BAND-SLICING AUDIT 2026-08-15 — which cuts are guarded, which are not

Prompted by the owner's degeneracy question and searched BY SHAPE, not by
name.  Context for why it matters: on `si_cohsex_debug` the deck's own
`nband = 60` edge has a **min gap over k of 0.000000 meV** on the 62-band mean
field, and moving it to a clean edge (40: 818 meV, 36: 157 meV) takes every Σ
channel's within-star spread to **exactly 0.0000**.  A sliced band edge is not
a theoretical hazard on this tree; it is the measured cause of the Si star
spread.

**`snap_cut_to_clean_boundary` DOES NOT EXIST** — no definition, no caller, no
mention, in any branch's history (`git log --all -S` empty).
`src/common/band_degeneracy.py` provides `boundary_min_gaps`,
`resolve_band_window` (modes `strict|snap|off`) and `check_band_window`; the
"widen to a clean boundary" behaviour is an inlined `while` loop inside
`resolve_band_window`, not a separate helper.  Do not plan against it.

**THE TRAP, measured:** `boundary_min_gaps` returns `+inf` at `b = nb` by
construction, so **given an already-truncated window it cannot see the
truncation that produced it**.  Handed the 60-band Σ window it calls edge 60
clean; handed the 62-band mean field it reports 0.000000 meV.  Always pass the
FULL mean field.

**GUARDED** (all in the BSE path):

| site | helper |
|---|---|
| `src/bse/bse_loading.py:861`, `:1202` | `resolve_band_window` |
| `src/bse/bse_window.py:644` | `check_band_window` |
| `src/bse/exciton_bands.py:1606` | `check_band_window` |
| `src/bse/vq_interp.py:2493` | `check_band_window` |
| `src/gw/gw_init.py:861` (ζ closure) | `check_band_window`, **warn-only** unless `zeta_nband` names the edge |

**UNGUARDED and symmetry-sensitive** — the deliverable of this audit:

| # | site | what is cut | note |
|---|---|---|---|
| 1 | `src/gw/gw_init.py:856-866` | the ζ fit window `b3`/`b4` | **measures the zero gap and PRINTS it without refusing.** A guard that measures the defect and proceeds. *Being fixed on another lane — do not touch.* |
| 2 | `src/gw/sc_iteration.py:1978-1987` + `src/gw/scissor.py:238-240` | `protected_mask` — a NON-CONTIGUOUS cut deciding off-diagonal Σ | own entry below |
| 3 | `src/gw/sigma_dispatch.py:308-310` | `v_h_np[:, b0:b3, b0:b3]` | V_H is on the Σ star-spread surface |
| 4 | `src/bse/bse_window.py:597-602` | `min(nb_eqp, nb_full)` eqp overwrite of `enk_full` | QP energies |
| 5 | `src/bse/exciton_bands.py:934-938` | `e_ht[:, :nc]` vs `e_st[:, :nc]` | the gate `sort`s both sides, so a truncated multiplet makes it compare different SETS — the one failure `sort` cannot detect |
| 6 | `src/common/chi_from_dipole.py:156-157` | `arange(nelec, nb)` → S(ω) | k-summed |
| 7 | `tools/bgw_sigma_hp_to_fixture.py:252,287-289` | `nb` then a star-spread max-min | offline tool; also still matches k by `Emf` at 2e-3 eV |
| 8 | `tools/sigma_star_spread_decompose.py:334-335` | `kin[:, :nb, :nb]` → `diag_star_spread` | offline tool |

**FIXED in this branch:** `tests/harness.py:998` `per_band[:nb].max()` — the
star-spread consumer — sliced at whatever the BerkeleyGW fixture carried with
no boundary check.  It now also reports `_cut_clean` (from the `Eo` column
already in the same file, through `boundary_min_gaps`) and
`_star_spread_multiplet`.  Reported, not refused: the fixture's band count is
not ours to move, and the Si cut at 16 measures clean anyway.

Not symmetry-sensitive, listed so nobody re-audits them: mesh-pad drops
(`bse_feast.py:913`, `bse_densify.py:597`, `qsgw_density.py:717`,
`sc_iteration.py:1741`, `mtxel_sweep.py:1180`) and print/plot paths
(`run_nscf.py:436`, `htransform.py:1729`, `bse_feast.py:836`).

### The QSGW band partition is an unguarded, NON-CONTIGUOUS band cut

Found 2026-08-15 by a shape-based audit of every band-axis truncation in the
tree, prompted by the owner's degeneracy question.  **Reported, not fixed:
the fix is a behaviour change to the QSGW Hamiltonian, not a diagnostic.**

`src/gw/scissor.py:238-240` `classify_bands_in_grid` classifies a band as
in-grid by an **all-k** predicate:

    in_window_kn = (E >= omega_min_ev) & (E <= omega_max_ev)
    band_in_grid = np.all(in_window_kn, axis=0)

`src/gw/sc_iteration.py:1978-1987` makes that BOTH the `protected_mask` and
the `in_range_mask`, and `src/gw/band_partition.py:148-151` then zeroes every
off-diagonal outside `protected × protected`.

**Bands degenerate at one k need not be degenerate at another**, and the
predicate is all-k, so band `n` can be in-grid while its multiplet partner
`n+1` is not.  Half a multiplet then carries full off-diagonal Σ and half
takes a scalar scissor — "not a subspace of anything" — and the result is
`eigh`'d and reported as QP energies.  This is exactly the hazard the
degeneracy safeguards exist for, and `band_partition.py` neither imports nor
mentions `common.band_degeneracy`.

Two things make it invisible today:

- `BandPartition.warn_if_protected_outside_grid` (`band_partition.py:82-101`)
  checks only `protected & ~in_range`, and the ONE construction in the tree
  passes the same mask twice (`sc_iteration.py:1986-1987`), so the set is
  identically empty and **the warning can never fire as wired**.
- `_check_kstar_spread` (`sc_iteration.py:1180`) runs on `delta_h_qp`
  **before** the partition is applied at `:1222`, so the star-spread
  enforcement does not cover the partition boundary at all.

**RESOLVED 2026-08-16 — owner ruling: "I want degenerate spaces degenerate in
LORRAX."**  Both halves landed.  `BandPartition.report_multiplet_splits` names
every splitting boundary and the gap it cuts; `promoted_to_multiplets` grows
the mask outward so no manifold is split, and `run_sc_driver` calls them in
that order.  `_check_kstar_spread` moved to AFTER `apply_band_partition`, so
it now gates the object that ships.

**MEASURED on `gnppm_debug`, the only committed deck running the SC path:
`28/46 protected; no boundary splits a multiplet`** — the promotion is a no-op
there, `eqp0`/`eqp1`/`sigma_diag` are byte-identical, and the reordered
star-spread gate reads `0.000e+00`.  So no number moved on any in-tree deck,
the reorder did not turn anything red, and **no deck exercises the promotion**
— `tests/test_band_partition_multiplets.py` carries the whole burden of
proving it works, and its mutation is verified red.

On a deck whose mask does split, every QSGW number moves; that is the accepted
consequence and anything computed there beforehand is superseded.

Retained below: the cost analysis that preceded the ruling.

**IS IT CHEAP?  The diagnostic is; the fix is not, and they are different
changes.**

- **Reporting it is cheap — ~20 lines, no behaviour change.**  Everything
  needed is already in scope at `sc_iteration.py:1978`: `e_dft_ev` is the
  array the mask was derived from, and `boundary_min_gaps` is one import.
  The mask's INTERIOR transitions (`np.diff(mask)`) are exactly the boundaries
  to test; its outer edges are the active window's own and come back `nan`
  from `boundary_min_gaps(..., is_full_spectrum=False)`, which is the correct
  answer — they cannot be judged from the window.  This is the honest minimum
  and nothing about the calculation moves.
- **Fixing it is NOT cheap.**  Promoting the mask to whole multiplets widens
  the protected set, so more bands carry full off-diagonal Σ, so `H_qp`
  changes and every QSGW number moves.  That is a physics decision with a
  convergence story attached, not a guard.

**The related one-line reorder is also not free.**  Moving
`_check_kstar_spread` from `:1180` (before the partition) to after `:1222`
would make it check the object that actually ships, and it is one line — but
the partition is exactly the operation that could make that residual non-zero,
so the reorder can turn the gate red on decks that pass today.  It is a
strengthening, not a cleanup, and it wants its own measurement first: run it
both ways on `gnppm_debug` and see whether the post-partition residual is
still at the 1e-10 floor.

Whoever picks this up: the ordering is (1) land the report, (2) measure the
post-partition spread, (3) then decide about promoting the mask.

### `tools/gen_input_reference.py` REFUSES to run, so `docs/input_reference.md` is hand-maintained

REPRODUCED 2026-08-18. The stale selector entries were removed from the
generator, but generation still stops before its key-drift check because
`band_extrapolation_estimator` has a non-literal `_DEFAULTS` expression that
the AST evaluator cannot resolve. It writes nothing and reports:

    gen_input_reference: cannot evaluate the default for
    'band_extrapolation_estimator'

The refusal is safer than silently generating a false default, but the current
consequence is that `docs/input_reference.md` remains hand-maintained. The
sandbox defect register owns the exact current file:line; fixing the generator
is outside this HDF5 cleanup.

### THREE frozen references were stale from the moment the writers moved to the wedge

MEASURED 2026-08-15, and this is the row the pre-merge anchor run existed to
produce. When the text writers moved from the full BZ to the file wedge
(commit `954ba9c8`, "writers take sym and reduce through the service"), **no
frozen reference was migrated and no regression gate was run.** Three gates
had been red ever since:

| gate | output | reference | verdict |
|---|---|---|---|
| `test_si_production_matches_frozen_reference` | 480 rows (8 k) | 3840 (64 k) | RED |
| `test_si_fast_matches_frozen_reference` | 160 (8 k) | 1280 (64 k) | RED |
| `test_gw_jax_matches_reference[cohsex]` | 120 (4 k) | 270 (9 k) | RED |

`gnppm_debug` (9 = 9), `bispinor_debug` (9 = 9) and `hbn_cohsex_debug`
(18 = 18) stayed green **for the wrong reason** — on those decks the file
wedge IS the full BZ, so the writers' k-set change was invisible. The three
that broke are exactly the three where `nk_red < nk_tot`.

**CONFIRMED PRE-EXISTING, not caused by the `sc_on_ibz` work**: the same
`Row-count mismatch: output (480, 7), reference (3840, 7)` reproduces at
`0241118f`, the parent commit, in a clean worktree.

**Migrated, not regenerated — and the migration was proved before it was
applied.** Each deck was re-run and the new wedge output compared, block by
block, against the frozen full-BZ reference at the full-BZ indices
`kirr_fullids`:

| deck | `kirr_fullids` | max abs difference |
|---|---|---|
| `si_cohsex_debug` production | `[0,1,2,5,6,7,10,27]` | **0.000000e+00 eV** |
| `si_cohsex_debug` fast | `[0,1,2,5,6,7,10,27]` | **0.000000e+00 eV** |
| `cohsex_debug` | `[0,1,2,4]` | **1e-6 eV** (one VH digit, twice) |

and every new block's `# kcrys` line was checked to equal `wfn.kpoints` in
order. So the references still carry the frozen numbers; what changed is
which k are present and the block framing. The two Si decks are bit-exact.
`cohsex_debug`'s one deviation is a last-printed-digit VH move
(`189.059483 → 189.059482`) at its 1e-6 gate; the other textual diffs on that
deck are format-only — the OLD file printed VH real on some rows and complex
on others, which was already internally inconsistent.

**Re-freezing is normally the owner's call** (the `si_fast` gate's own skip
message says so). It is done here because leaving three red regression gates
across a merge is worse, and because the values were *proved* unchanged rather
than re-blessed. It is a SEPARATE COMMIT so it can be reverted on its own.

### `cohsex_debug` CANNOT run `sc_on_ibz`, and the reason couples two open questions

MEASURED 2026-08-15.  `cohsex_debug` is the deck with the sharpest two-wedge
divergence (file wedge 4, star wedge 3) and would be the ideal SC regression
deck.  It refuses:

    ValueError: k-star spread of Σ+V_H is 2.763726e-01 relative, above the
    refusal threshold 1.0e-06.

That is **not** a wedge bug — it is `_check_kstar_spread` working correctly on
a genuinely non-star-invariant Σ.  `centroids_frac_60.txt` is one of the four
deliberately non-orbit-closed sets, and its recorded closure residual is
**2.762e-01** — the same number.  So the deck's Σ really does differ between
members of a star, and selecting a representative would keep an arbitrary one.

Consequence worth stating: **the centroid-closure question and the SC wedge
question are coupled through this fixture.**  Anyone who regenerates
`centroids_frac_60.txt` as orbit-closed changes which decks can exercise the SC
path — and per the register, that set is KEEP-with-justification because
`test_star_offdiag_gate.py` asserts its consequence as a fact.

### MEASURED: independent per-k `eigh` is NOT a source of star-inequivalent E_qp

Same run, `sc_on_ibz` OFF, so `H_qp_dft` is assembled and diagonalised
independently at every one of the 9 full-BZ k:

| quantity | star spread over 5 orbits |
|---|---|
| `E_DFT` | **0.0000 meV** |
| `E_QP`  | **0.0000 meV** |

(`eqp1.dat` prints `%15.9f` eV, so this bounds the spread below ~1e-6 meV, not
merely below a meV.)

So per-k diagonalisation reproduces star-equal eigenvalues **exactly**.  The
8x redundant `eigh` is wasted work, but it is not a source of symmetry
breaking, and **none** of the 2.6 -> 41 meV Sigma star spread can be
attributed to it.

CAVEAT ON SCOPE, and it matters: `gnppm_debug`'s centroid set
(`centroids_frac_399.txt`) IS orbit-closed, so its Sigma star spread is
already ~0 and this deck cannot discriminate the two causes.  The deck that
shows 41 meV is `si_cohsex_debug`, which is `one_shot_dft`; testing it would
need an SC variant, which `sc_on_ibz`'s rot currently blocks.  This result is
CONSISTENT with the conditioning attribution and does not prove it.

### MEASURED, AND IT OVERTURNS THE RECEIVED EXPLANATION: centroid non-closure is NOT what drives the Si production deck's star spread

2026-08-15.  The tree has stated in several places — `si_cohsex_debug/README.md`,
`test_gw_jax_regression.py`'s tolerance comment, and (until this entry)
`docs/architecture/symmetry_register.md` — that the production deck's within-star
Sigma spread is caused by `centroids_frac_960.txt` being a literal,
non-orbit-closed point set.  **That was never measured against the alternative.
It has now been, and it is wrong.**

An orbit-closed 960-point set was generated for the same deck by the same
procedure the orbit-closed 144 set follows (`kmeans_cli 960 --orbit --seed 42
--prune-n-val 8 --prune-n-cond 52`), and verified `closed=True`, worst residual
**1.000e-06 on 48 ops** — identical to the 144 set, both in pure fractional
coordinates and on the deck's (24,24,24) FFT grid.  It is shipped beside the
original as `centroids_frac_960_orbitclosed.txt`.  The deck was pointed at it and
the anchor re-run.

| quantity | non-closed 960 (shipping) | orbit-closed 960 |
|---|---|---|
| closure, on-grid | `False`, worst 1.318e-01, 47/48 ops | `True`, worst 1.000e-06 |
| star spread `[:16]` | 2.6111 meV | **1.9642 meV** |
| star spread `[:60]` | 41.3376 meV | **39.8758 meV** |
| sigSX MAE / max vs BGW | 0.1509 / 0.3030 | **3.3336 / 11.8904** |
| sigCOH MAE / max vs BGW | 0.3513 / 1.2141 | **11.9480 / 39.0121** |
| sigTOT MAE / max vs BGW | 0.4329 / 1.2525 | **14.9426 / 41.1405** |

**Closure removed essentially none of the spread** — 4% at the full 60-band
window, 25% at the 16 bands the anchor compares.  If non-closure were the
mechanism, a closed set would have gone to ~0.000, which is what the 144-point
set measures.  It did not.

**And it cost a factor of ~35 in agreement with BerkeleyGW**, taking every
column far outside the gate (limits 1.5 MAE / 5.0 max).  So the orbit-closed
anchor is worse on the axis the anchor exists to measure and no better on the
axis it was supposed to fix.  The deck is therefore left on the shipping set and
the gate is green; this is recorded rather than acted on.

HYPOTHESIS, NOT ESTABLISHED, for what actually drives it: the spread tracks
centroid COUNT, not geometry.  The 144-point set measures exactly 0.000; both
960-point sets measure ~40 meV over the full window whether closed or not.  The
zeta fit at 960 runs with condition numbers around 1e7-1e8 and an rcond of
1e-10 (`[zeta rank_truncate]` on any run of this deck), so numerical rank
truncation breaking the symmetry is the natural suspect.  Testing that means a
count sweep at fixed closure, which has not been done.

CONSEQUENCE FOR THE ORBIT-CLOSURE WORK GENERALLY: orbit closure remains correct
and worth having — `centroid_source_map_and_wrap` REFUSES without it, and a
non-closed set forces the q-axis to the full BZ.  What is now measured is that
on this deck it is not the source of the star spread, so "regenerate on a closed
set" is not a fix for that symptom.

### MEASURED: how non-orbit-closed the Si production centroid set actually is

2026-08-15, `refactor/eqp-ibz-2026-08-15`, Si production deck
(`cohsex_si_test.in`, 960 literal centroids, 4x4x4, 64 k / 8 wedge).
Star spread — the worst per-band max-min of Re diag Sigma_tot between
members of one star — as a function of how many leading bands are
included:

| bands | star spread |
|-------|-------------|
| `[:8]`  |  0.9796 meV |
| `[:16]` |  **2.6111 meV** |
| `[:24]` |  7.2668 meV |
| `[:32]` |  7.5926 meV |
| `[:40]` | 10.0203 meV |
| `[:60]` | **41.3376 meV** |

`bands[:16] = 2.6111` reproduces the historical figure exactly — that is
the whole scope the BerkeleyGW anchor fixture covers, and therefore the
only part anyone had ever measured.  **The violation keeps growing with
band index, to 41.3 meV over the deck's full 60-band sigma window**, i.e.
sixteen times what the old scope could see.

This is NOT a defect entry: the deck is documented as using a literal,
non-orbit-closed 960-point centroid set, so the ISDF quadrature genuinely
breaks the 48-op point group and a nonzero spread is expected
(`tests/regression/si_cohsex_debug/README.md`, "Known defects"; the
orbit-closed 144-point fast deck measures exactly 0.000).  What is new is
the SIZE and the band dependence, which nobody had, because the metric
was scoped to the fixture's 16 bands.  Anyone reopening the
orbit-closure question wants this table: the cost of the non-closed set
is concentrated in the conduction bands, not spread evenly.

Reproduce: the writer now emits `# star_spread_ev_per_band` into
`sigma_diag.dat`, so the ladder is readable off any run's output without
instrumenting anything.

### Aside: `_assert_matches_reference`'s "BYTE-IDENTICAL" branch is unreachable

Noticed while re-freezing.  `tests/test_gw_jax_regression.py:140` reports
`BYTE-IDENTICAL to the reference (atol not exercised)` when
`output_file.read_text() == reference_file.read_text()`.  That comparison
can never be true: `common.provenance.provenance_header` (`provenance.py:27`)
stamps `datetime.now(UTC)` into the first line of every `.dat` LORRAX
writes, so the output and any committed reference differ in byte 30 of
line 1 no matter what the physics did.  Every frozen-gate pass has
therefore gone through the tolerance path, and the headroom warning is the
only report anyone has ever seen.  Harmless — the atol path IS the gate,
and it prints the margin — but the branch is dead and its message implies
a stronger check than exists.  The fix is one line (compare through
`harness.normalize_dat`, which already strips exactly that line); not taken
here because it changes what a green gate prints on every deck.

The gate for the two fixed sites is
`tests/test_unfold_through_the_service.py`: an AST layer that fails if
either function stops calling the adapter or grows a nested k-loop back,
plus a behavioural layer that fails if an unfold drops, duplicates or
mis-parents a k.  Verified red against the pre-change source of both.

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


## CONSOLIDATION CHECKPOINT — the BSE campaign's ten branches (2026-08-08)

Final consolidation of the 2026-08-08 BSE performance campaign: ten branches
merged onto `main` @ `09db2939`, plus five in-consolidation commits (remove
`krep`; make `yhoist` permanent; delete two dead bare-`shard_map` sites and fix
the R2 docstring; fix two reds this consolidation introduced; and this
amendment).  Censused head **`e495fc45`**, one full suite in the default xdist
geometry — the same geometry as the RE-CUT WAVE census above, so the two are
directly comparable.

```
27 failed, 2204 passed, 131 skipped, 1 xfailed, 16465 warnings in 588.01s
```

**UNEXPLAINED NEW REDS: ZERO.**  Every non-passing cell was looked up by name
(`/pscratch/sd/j/jackm/bse_consol_0808/setdiff.py`, artifact
`_reports/census2_e495fc45.xml`).

| n | cells | disposition |
|---:|---|---|
| 5 | `test_gpu_pinning` ×5 | Class B loader-order, listed |
| 5 | `distrib_la::test_distrib_la_contract` ×5 | Class B loader-order, listed |
| 1 | `vcoul::test_the_whole_public_surface_answers_with_no_lorrax` | Class B, listed |
| 1 | `vcoul::test_vcoul_imports_and_computes_with_no_scipy` | listed xdist artifact |
| 2 | `test_bse_setup_qchunk` ×2 | class P2, listed; `_maxdiff = 1.3743988419548263` and the `2.220446049250313e-15` first spread, both unmoved |
| 1 | `symmetry_maps::test_the_lorentz_mixing_matches_a_dense_numpy_reference[1-1]` | listed cross-service conftest collision |
| 1 | `test_bse_vq_interp::test_loo_accuracy_vs_reference_thresholds` | listed, left standing by the re-cut |
| 1 | `test_gw_jax_regression::test_hbn_matches_frozen_reference` | listed, left standing by the re-cut |
| 1 | `test_bse_w0_resolvent::test_wq_resolvent_matches_restart_finite_q` | listed by the re-cut; **fingerprint added below**. **STRUCK 2026-08-08** at `b5c0cf15` — conjugate pair-density vertex; see the amendment at the head of this file |
| 9 | `symmetry_maps::test_symmetry_maps_import_isolation` ×9 | **not in the re-cut's 20** — A/B-proven pre-existing, below |
| **27** | | **0 unexplained** |

Two of the re-cut's twenty are GREEN here and are NOT closed on that evidence:
`test_the_lorentz_mixing…[2-2]` and
`test_bse_w_omega_chain::test_w_omega_chain_matches_oracle_q0`.  The re-cut
already A/B-attributed the latter as a full-suite collection artifact rather
than a real red, so a geometry that does not trigger it proves nothing about
the code.  Collection also differs (2363 cells here against 2311, and 131 skips
against 66), because the ten merged branches add gates.

### The nine `symmetry_maps` import-isolation reds are PRE-EXISTING

They are not in the re-cut's twenty, so they were A/B'd rather than assumed —
this consolidation contains **zero `services/` changes** (`git diff --stat
09db2939..HEAD -- services/` is empty), but "I did not touch it" is an argument,
not a measurement.

| leg | result |
|---|---|
| base `09db2939`, the isolation file ALONE | **9 passed** |
| head `e495fc45`, the isolation file ALONE | **9 passed** |
| base `09db2939`, whole `services/symmetry_maps` | **11 failed, 247 passed** |
| head `e495fc45`, whole `services/symmetry_maps` | **11 failed, 247 passed** |

Identical on both sides, same eleven cells by name (the nine isolation cells
plus both `lorentz_mixing` parametrisations).  The cells pass alone and fail
once the rest of their own service has run in the process — the same
collection-order family as the Class B reds above, with the failure spelled
`isolated import of 'symmetry_maps' produced no LXKIT_ISOLATION line (rc=1)`.
They are red at base, red at head, and nothing to do with this landing.  They
were absent from the re-cut's twenty because that census's xdist distribution
happened to give those cells different companions.

### The `wq_resolvent` red now has a FINGERPRINT

The RE-CUT WAVE amendment registered
`test_bse_w0_resolvent::test_wq_resolvent_matches_restart_finite_q` as a real,
unlisted, pre-existing red and deliberately did not diagnose it ("guessing at it
would be worth less than naming it").  That was the right call, but it left a
named red with no fingerprint — which is exactly what a future census cannot use
to tell drift from stability.  One exists, from an independent lane:

```
AssertionError: q=(0, 1, 0) col 179: W_q resolvent closure rel_err=6.87e-01 (>1e-6)
assert 0.6874207585174131 < 1e-06
```

`FIX_solver_robustness.md` §7 measured it at base **`013aad92`** and found it
identical on both of that branch's arms — same q, same column, same number to
all sixteen digits — and this census reproduces it again, digit for digit, at
`e495fc45`.  So the cell is now A/B-proven pre-existing **three times, from
three bases**, by lanes that did not coordinate:

| lane | base | geometry | result |
|---|---|---|---|
| solver-robustness | `013aad92` | 1 proc | red on the before arm AND the after arm, `rel_err=6.87e-01` |
| re-cut wave | `81a285af` | 1 proc **and** xdist | red on both, and on the re-cut head |
| this checkpoint | `e495fc45` | xdist | red, same q, same column, same sixteen digits |

It remains **undiagnosed** and is still nobody's in this campaign.  What is
closed is the bookkeeping: it is not new, it is not geometry, and it is not any
of the ten merged branches.

> **CLOSED 2026-08-08.**  The fingerprint above was the right thing to record:
> it is what made the diagnosis checkable.  The defect is a conjugation in
> `build_finite_q_data`'s pair-density vertex, invisible at q=0 because χ₀(0) is
> real, and a dense numpy model reproduces `0.6874207585174131` from it.  Fixed
> at `b5c0cf15`; see the amendment at the head of this file.

**Why it was missed, restated because the trap is still live.**  The
symmetry-landing amendment closes
`test_bse_w0_resolvent::test_w0_resolvent_matches_restart` as GREEN.  That is a
DIFFERENT CELL in the same file — `w0`/static against `wq`/finite-q — so a
reader scanning for "w0_resolvent" finds a row saying it was fixed.  Anyone
adding a row for either cell should spell the full node id.


### The `wq_resolvent` red is NOT the stale-artifact family — ELIMINATED

`test_bse_w0_resolvent::test_wq_resolvent_matches_restart_finite_q` stays **red
and undiagnosed**; this is not a strike.  What is closed is one branch of the
search space, because the obvious hypothesis — "a W test scored against a
pre-head-fix restart", i.e. the same defect `FIX_driver_blockers.md` §3 found
under `bse_nontda` — is **impossible**, on two independent grounds.  Recorded so
the next lane does not spend the hours confirming it.

**1. The fixture regenerates its own restart, every run, on the tree under
test.**  The cell takes `gnppm_session`, and `tests/conftest.py` defines it as a
*fresh* run:

```python
@pytest.fixture(scope="session")
def gnppm_session(tmp_path_factory):
    """Fresh (restart=false) run of the gnppm fixture; Tier-1 state."""
    return _run_session_case(tmp_path_factory, "gnppm_debug", "gnppm_test.in", ...)
```

`_run_session_case` copies the fixture into a fresh tmp dir and calls
`harness.run_gw_jax` — a full GW run, `restart = false`, in the same process
tree as the assertion.  The restart the cell reads is minted minutes earlier by
the code being tested.  There is no artifact old enough to be stale, which is
also why the red reproduces at three unrelated bases.

**2. The quantity the hypothesis blames is 7.15e-12 on this deck, and always
was.**  Measured off the raw `isdf_tensors_399.h5` datasets of three independent
gnppm restarts (`agent_defects_0807/run_gnppm_{before,after2}`,
`zeta_sweep/runs/gnppm/rc_1em8`), written on three different days by three
unrelated lanes — **two of them before the head fix landed**:

| tile | `max｜A(q) − conj(A(−q))｜ / max｜A｜` | `(μ,ν)`-Hermitian |
|---|---:|---:|
| `W0_qmunu` | **7.150631e-12** | 1.047778e-13 |
| `V_qmunu` | **9.006628e-17** | 8.905173e-17 |

Identical to seven digits across all three.  For contrast, the artifact that
*did* carry the defect — `perf_bse_0808/deck_si444`, and an independent GW
re-run of the same deck on the pre-fix tree — measures **8.635142e-04 /
4.274267e-03**, and the same deck regenerated on `66d2a02c` measures
**1.816708e-11 / 2.975721e-15**.  The mini-BZ head-slot defect is a property of
the Si 4×4×4 deck's q=0 head slot; **the MoS2 3×3×1 fixture never carried it**.
So even if this cell did reuse an old artifact, the old artifact would be clean.

Instrument: `perf_bse_0808/nontda/probe_recip2.py` (login node, plain
`h5py`/`numpy`, no container).  Evidence and the rest of the wave in
`FIX_nontda_feature.md`.

**Where a diagnosis should start instead.**  Two leads, neither measured, both
recorded so they can be scored rather than retrofitted:

* `bse_w_exact.build_finite_q_data`'s own docstring records that applying the
  design-doc umklapp phase "breaks the match (**rel_err 0.6–3.2** vs 1e-8)".
  The observed **6.87e-01** sits inside that band.
* The cell's q comes from `_symmetry_reduced_q_list` →
  `SymMaps.q_irr_kgrid_int`, and the one commit touching *both* this test file
  and `bse_w_exact.py` is `e9e0acd3`, the symmetry_maps door replumb — whose
  census was **WSL/CPU only**, a suite in which this `@pytest.mark.gpu` cell
  does not run.  A labelling change behind that door would be invisible to it.

`perf_bse_0808/nontda/probe_wq.py` decides between "the resolvent is right and
is being scored against the wrong tile" and "the finite-q operator is wrong" in
a single solve, by scanning the resolved columns against every tile in the grid.


## FIXES + MPA LANDING — four re-earned fixes and the MPA stack (2026-08-08)

The 2026-08-08 fix queue and the multipole-W infrastructure, landed together
on `main` @ `f23baab2`: five branches in a pinned order, plus two commits the
landing owns.  Censused head **`12097f78`**, against base **`f23baab2`**, one
full suite per leg.

```
head 12097f78 : 86 failed, 2214 passed, 257 skipped, 26 deselected, 1 xfailed, 35 errors in 627.10s
base f23baab2 : 86 failed, 1993 passed, 248 skipped, 26 deselected, 1 xfailed, 35 errors in 291.42s
```

**UNEXPLAINED NEW REDS: ZERO.**  Set-diff by node id
(`bwrun/setdiff_mc.py`, both junitxmls kept beside it — see *Where the
evidence lives*).

| n | class | disposition |
|---:|---|---|
| 0 | NEWLY RED | — |
| 0 | newly green | see *the door red could not appear here*, below |
| 0 | newly skipped / no longer skipped | — |
| 0 | collected only at head, RED | — |
| 232 | collected only at head, not red | the five branches' own gates, all green |
| 2 | lost from collection | replaced by design, below |
| 121 | red in BOTH | identical by name on both legs |

The 121 carried reds are what says the two legs are comparable at all: same
count, same names, both sides.  Collection moves 2363 → 2593 (+232 −2 = +230),
and 230 = the +221 passed plus the +9 skipped, so every new cell is accounted
for arithmetically as well as by name.

### THE ENVIRONMENT IS A CAVEAT, AND A PERLMUTTER CENSUS IS OWED

Both legs ran on **WSL CPU** (`JAX_PLATFORMS=cpu`, sequential, no xdist,
identical invocation, only `PYTHONPATH`/cwd differing), NOT on Perlmutter.
This was forced, not chosen: the shifter **image gateway went down at ~17:36
PDT** (`shifterimg lookup` → "failed to contact the image gateway", reproduced
on three attempts several minutes apart, and again ~20 minutes later), so no
container step can start on any Perlmutter node — the `lx test` legs died at
`FAILED to lookup docker image ghcr.io/nvidia/jax:jax-2025-07-21` in 8 s.  The
ssh ControlPersist master survived the 18:18 cert expiry and the cluster was
otherwise reachable; the blocker is the image service alone.

What this costs, stated plainly: this box has no FFI build, so 86 F + 35 E of
each leg are the FFI-absent family and every `@needs_host_ffi` /
device-requiring cell is red or skipped on **both** sides.  The set-diff is
therefore trustworthy — identical counts and identical names on both legs are
what it is measuring — but it is blind to anything that only shows up with a
real device, a real interconnect or a built `.so`.  Nothing in this landing
touches the FFI, and the restart-consolidation branch's own sharded-BSE work
is explicitly held pending a Perlmutter timing leg, so the exposure is
bounded — but it is not zero.

**REGISTERED AS OWED: a full-suite Perlmutter census of this head at the next
checkpoint**, when the gateway is back.  It is set up and one command away —
`/pscratch/sd/j/jackm/fixmpa_0808/` already carries both worktrees
(`wt_head` @ `12097f78`, `wt_base` @ `f23baab2`), the pinned `env_common.sh`
(restage_candidate `.so` pair, `LORRAX_FFTW3_SO`, `LX_BASE_MODULE=lorrax_J070`)
and `census.sh <head|base>`.  Both legs' `[lx] source tree:` lines were
verified correct before the image lookup failed, so the stale-`LORRAX_CHECKOUT`
trap is already cleared for that run.

### The door red could NOT appear as "newly green", and here is its measurement

The landing's stated gate was that
`tests/test_layering.py::test_lorrax_reaches_a_service_only_through_its_door`
goes green.  It is green at the censused head — but it is green at the BASE
too, and it was never red on `main`, so a base-vs-head set-diff cannot show
it and reporting "newly green: 0" as if the gate were unmet would be wrong in
the other direction.  The red is a property of an INTERMEDIATE state: it does
not exist until merge 5 brings `file_io/mpa_store.py` into the tree, and it is
gone by the head.  So it was measured directly instead, A/B on the two commits
that bracket it:

| tree | door test | `tests/test_mpa_store.py` |
|---|---|---|
| `34bf489d` (merge 5 tip, before the adaptation) | **FAILED** — `{'file_io.mpa_store': [('symmetry_maps.qirr_store', 190)]}` | **36 failed, 5 passed** |
| `12097f78` (the adaptation commit) | **passed** | **41 passed** |

The mechanism, named: `mpa_store` was written while the q_irr checkpoint was
still landing and reached into `symmetry_maps.qirr_store` for thirty-four uses
of five private symbols.  The rank-refusal branch published all five on the
door (`_VERSION_ATTR` → `QIRR_VERSION_ATTR`, `_Dest` → `QirrDest`, `_attr_str`
→ `qirr_attr_str`, `_generator_commit` → `qirr_generator_commit`, `_validate`
→ `validate_qirr_tables`, with `QIRR_TABLE_SUFFIX` already there) and DELETED
the old spellings rather than aliasing them.  So the door rule and the runtime
agreed: those thirty-four sites were live `AttributeError`s, which is why the
adaptation is worth thirty-six cells and not one.  `12097f78` switches every
reach to the door and drops `mpa_store`'s copy of the version-1 rank refusal,
which now lives in `qirr_store.read_tensor` where the unsuspecting consumer
already calls it; `mpa_store` keeps version 2's rank and the `mpa_freq_axis`
cross-check, which are the two things only it can know.

### The two cells that left collection were REPLACED, not lost

```
tests.test_restart_qirr_consumers::test_the_gw_restart_reader_refuses_a_wedge_and_names_the_way_out
tests.test_restart_qirr_consumers::test_the_gw_restart_reader_is_silent_on_every_file_that_exists_today
```

Both are retired by `536cbac9` ("GW restart reader: the wedge is UNFOLDED, not
refused"), which changes the behaviour they pinned: the GW restart reader no
longer refuses a wedge, it unfolds it.  A test asserting a refusal cannot
survive the refusal being deleted on purpose.  Four cells replace them in the
same file — `test_the_gw_restart_reader_unfolds_a_wedge_end_to_end`,
`test_the_gw_reader_unfolds_a_wedge_bit_identically`,
`test_the_gw_restart_reader_is_byte_identical_on_every_file_today` and
`test_the_arms_between_them_exercise_every_unfold_branch` — and all four are
green at the head.  This is the designed replacement of the symmetry lane's
registered wedge-reader row, not an erosion of coverage.

### OPEN ROW: the complex-Laplace catalog is SHIPPED BUT NOT WIRED

`src/common/minimax_assets/catalog_complex_laplace.json` and its eighteen
`complex_laplace/*.npz` tables land here, and **nothing selects them at run
time**.  That is deliberate and it is the tables branch's own instruction —
`55e9edae` says wiring "belongs to the landing commit, and deliberately so —
today's selection rule has no beta axis and would otherwise serve a table
fitted to a different function".  This landing declines to do it, because the
precondition the branch named has not been met: `gw/minimax_screening.py`'s
`_load_shipped_minimax_catalog` reads `catalog.json` and only `catalog.json`,
and its selection has no β axis to match on.  The new catalog's own
`selection_rule` block demands `"beta": "EXACT MATCH REQUIRED for
rule='btv_minimax'.  The target is a different function at every beta, so a
table at beta' != beta is not conservative, it is wrong."`  Wiring a β-indexed
family into a β-blind selector would serve whichever entry happened to sort
first.

Production behaviour is verified UNCHANGED rather than assumed:
`git diff f23baab2..HEAD` is empty for `catalog.json`, for both shipped table
directories (`crossing/`, `noncrossing/`) and for `gw/minimax_screening.py`;
the landing's 19 files under `minimax_assets/` are all additions; and
`tests/test_minimax_quadrature.py` — the shipped-table suite — is **14 passed**
at the head.  The 55 cells of `tests/test_minimax_imag_tables.py` certify the
new tables from their own bytes without any runtime consumer.

**The row belongs to the minimax service extraction**, which is where a
selection rule that can carry a β axis will be designed.  Until then the
tables are inert payload: they cost bytes in the tree and nothing else.

**CLOSED 2026-08-08 by `feat/minimax-beta-selector-2026-08-08`.**  The β axis
now exists as its own module, `src/common/minimax_beta_selector.py`, and
`solve_laplace_minimax_imag_interval` consults it.  The three things the row
above said had to be true before wiring are true: β is matched against each
entry's own stamped `beta_tolerance` band and never rounded for
`rule='btv_minimax'`; the one entry that does round — the positive composite,
whose β dependence is an exact phase on β-independent nodes — rounds downward
only and is *re-phased* at the request rather than served as shipped; and the
envelope's two clauses are checked before the β match, so a Σ-stage width
request cannot be answered from a sampling-height sweep even where the two
overlap numerically.  Everything the selector cannot certify still falls
through to the same uncertified runtime solve on the same cache key, so R1's
stage-2 refusal remains staged and unarmed.  `tests/test_minimax_quadrature.py`
is still **14 passed** — the static and crossing selector is untouched — and
`tests/test_minimax_beta_selector.py` adds 25 cells, of which the red twins are
the neighbouring β, the wrong clause, the upward-rounded composite, the
mis-stamped band, the unknown β axis and the unreadable payload.

**Registered while closing it, and NOT caused by it:**
`tests/test_minimax_imag_tables.py::test_the_fit_stage_floor_is_no_longer_a_quadrature`
fails on this WSL host at `origin/main` itself (`f0435e9a`, measured with the
branch stashed): it asserts `worst_eval < 1e-8` and reaches `9.98e-08`, but its
own *synthesis* baseline — computed with no evaluator, no quadrature and no
table anywhere in the path — is `8.65e-08` here against the `3.81e-09` the
docstring records.  So what moved is the Padé fixture's conditioning on this
host and not the quadrature the cell is about; the cell's third assertion,
`worst_eval < 2 * worst_synth`, still holds.  The fix belongs with whoever owns
that fixture: the absolute `1e-8` needs to become a statement relative to the
synthesis baseline, which is what the docstring already says the cell means.

### Where the evidence lives

Local (WSL): `_reports/census_head.xml`, `_reports/census_base.xml`,
`_reports/setdiff.txt` and `_reports/wsl_census.log` under this session's
scratchpad, with `wsl_census.sh` — the single script that ran both legs — beside
them.  Perlmutter: `/pscratch/sd/j/jackm/fixmpa_0808/` (both worktrees, the
pinned env, `census.sh`, and `_reports/leg_{head,base}.log` showing the image
gateway failure and the correct `[lx] source tree:` lines), staged for the owed
census.  In-tree: this amendment.


## THE OWED PERLMUTTER CENSUS, RECONCILED — and the import edge it found (2026-08-08)

The section above registered a full-suite Perlmutter census of the fixes+MPA
head as OWED, because the shifter image gateway went down at ~17:36 PDT and
both legs had to run on WSL CPU instead.  The gateway came back that evening
and the census RAN — fired from the BSE-perf session, through the landing
lane's own staged, read-only `/pscratch/sd/j/jackm/fixmpa_0808/census.sh`,
their worktrees and their `_reports/`, unmodified.  It is FUNCTIONAL only and
makes no timing claim.

```
head 12097f78 : 2593 tests, 31 failed, 0 errors, 133 skipped, 344 s
base f23baab2 : 2363 tests, 18 failed, 0 errors, 132 skipped, 291 s
```

That is +230 collected and +13 failed, which resolves by node id into 15 newly
red, 2 newly green.  This amendment closes all seventeen.  **The headline is
that the WSL census's central claim does not survive contact with the cluster,
and not in the way the +13 suggests**: nine of the fifteen are not a regression
at all, and four of the remaining six were red on WSL too, on the very machine
that reported them green.

### The nine `symmetry_maps` isolation cells are NOT a regression

The whole of
`services/symmetry_maps/tests/test_symmetry_maps_import_isolation.py` — all
nine cells — is red at the head and green at the base, which is what a
set-diff calls a regression and is what the perf lane recorded, correctly, as
"reproduces on head and not on base, so the landing changed something".  The
landing changed nothing here.  What the cluster found is a defect that has
been in the tree since the service was extracted on **2026-08-07**
(`6238f471`), sitting behind an environment that had never asked it a
question.

**The edge, named.**  `import symmetry_maps` runs, at module scope:

```
symmetry_maps/__init__.py:158   from symmetry_maps.qirr_store import (...)
symmetry_maps/qirr_store.py:117   from symmetry_maps.orbit_syms import (...)
symmetry_maps/orbit_syms.py:246     _CANON_INV = jnp.int64(10**12)
```

`jnp.int64(...)` is not a constant.  It is `asarray` → `device_put` → "which
device?", so that line **initialises a jax backend while the package is being
imported**.  In `lxkit`'s isolation child — `python -S`, `PYTHONPATH` cut down
to the service's `src` plus its declared deps — the session's `JAX_PLATFORMS`
is inherited but the CUDA plugin is not reachable, and the line raises before
the probe can print:

```
RuntimeError: Unable to initialize backend 'cuda': Backend 'cuda' is not in
the list of known backends: ['cpu', 'tpu'].
AssertionError: isolated import of 'symmetry_maps' produced no LXKIT_ISOLATION
line (rc=1)
```

Every one of the nine dies there, which is why the file goes red as a block
and why the three red-twin cells report a regex mismatch rather than their own
sentence: the harness's missing-payload assertion fires before the assertion
they were written to catch.

**Why it is not the landing.**  `orbit_syms.py` is byte-identical between
`f23baab2` and `12097f78` (`git log f23baab2..12097f78 -- orbit_syms.py` is
empty), and the door has imported `qirr_store`, which has imported
`orbit_syms`, since `3e9cea10`.  The rank-refusal branch added seven names to
an import statement that was already there; it added no edge.  Two independent
measurements say the same thing:

* **The two newly GREEN cells are the same defect wearing the other hat.**
  `services.vcoul…test_the_whole_public_surface_answers_with_no_lorrax` and
  `…test_vcoul_imports_and_computes_with_no_scipy` are red on the BASE leg and
  green on the head, with the identical `Unable to initialize backend 'cuda'`
  raised in the identical `python -S` child — at `vcoul/minibz.py:465` and
  `:234` instead, because those two cells' preambles COMPUTE.  `services/vcoul`
  is byte-identical between the two legs (`git diff` over it is empty), so a
  cell that flips red→green across trees that do not differ cannot be
  measuring a tree.  It is measuring which xdist worker the cell landed on and
  what that worker's `JAX_PLATFORMS` had become — and adding 230 tests
  reshuffles that.  The nine and the two are one family of eleven, not a
  regression and two bonus greens.
* **A/B under a portable pin puts both trees on the same side.**  jax refuses
  a platform named in `JAX_PLATFORMS` that it cannot initialise, and refuses
  it at the first device touch rather than at `import jax`.  Pinning
  `JAX_PLATFORMS` to a backend no machine provides therefore asks exactly the
  question the Perlmutter child asks, on any box.  Under that pin, on WSL:
  **9 failed at `f23baab2`, 9 failed at `ed11a955`** — the same nine, the same
  sentence, on the base the census called clean.

**The fix is the edge, not the symptom.**  `_CANON_INV` is a `np.int64`
scalar now.  A numpy scalar and not a Python `int` because the two do not
promote alike — a numpy scalar is strongly typed in jax's lattice exactly as
the jax scalar was — so the keys `_orbit_lex_winner` builds keep their dtype
and their bits: `canonicalize_orbit` and `_orbit_lex_winner` are
**sha256-identical** before and after on a 48-op / 512-rep case.  Nothing was
pinned to CPU and no cell was skipped.  The isolation cells exist to catch a
package that cannot stand up on its own, they caught one, and what they caught
is real: a table library must be IMPORTABLE without a device and needs one only
to COMPUTE.

Two cells are added beside the fix.
`test_importing_the_package_initialises_no_jax_backend` binds the whole public
surface under a `JAX_PLATFORMS` no backend can satisfy, so any module body
that materialises an array fails it by name; and
`test_a_module_scope_device_array_would_be_caught` is its red twin, running
the harness against a temporary COPY of the service tree with
`jnp.int64(10**12)` appended to `orbit_syms` — the exact line, put back — and
asserting the refusal.  The pin is a backend name nothing provides rather than
`cuda` **on purpose**: jax skips `cuda` quietly on a host with no visible
NVIDIA GPU, so a `cuda` pin is green on WSL for the wrong reason, which is the
whole trap this section is about.

**Measured, both environments.**

| leg | tree | result |
|---|---|---|
| WSL, no pin | branch | `test_symmetry_maps_import_isolation.py` **11 passed** |
| WSL, unsatisfiable-backend pin | branch | **11 passed** |
| WSL, no pin | branch with the eager line restored | **1 failed, 10 passed** — the new positive cell, alone, since it carries its own pin |
| Perlmutter GPU node, `JAX_PLATFORMS=cuda,cpu` | `12097f78` + the new cells | **10 failed, 1 passed** (the census's nine, plus the new cell; the red twin green) |
| Perlmutter GPU node, `JAX_PLATFORMS=cuda,cpu` | the same tree, one line changed | **11 passed in 16 s** |

The two Perlmutter arms are copies of the census's own `wt_head` differing in
exactly one line, run on the same node with the same pin.

### THIS IS THE THIRD GPU-ONLY CELL INVALIDATED BY A WSL VERIFICATION TODAY

The WSL census reported **zero newly red** and it was not lying about what it
measured; it could not see any of this.  `jax` skips an unavailable `cuda`
without complaint on a box with no NVIDIA device, so the exact condition that
kills the isolation child — a `JAX_PLATFORMS` naming a platform the stripped
child cannot reach — cannot arise there at all.  That is the third time on
2026-08-08 that a verification run on WSL returned a green that a GPU node
did not honour, and the pattern is worth stating as a rule rather than as
three anecdotes: **a WSL leg is evidence about code, and it is not evidence
about a cell whose premise is a device.**  When a census is forced onto WSL,
the honest report is the set-diff PLUS the list of cell families it is
structurally blind to — which the section above did state, and which is
exactly the part that came due.

### The six new cells shipping red, adjudicated

Six of the fifteen are cells that do not exist at the base, so no set-diff
could call them regressions; they shipped red.  They are two different
stories and must not be quoted as one number.

| # | cell | class | verdict |
|---|---|---|---|
| 1 | `test_mpa_fit_kernel::test_vmap_batch_equals_loop_bit_identical[5]` | runs everywhere | **the cell's claim is false** — owner row |
| 2 | `test_mpa_fit_kernel::test_vmap_batch_equals_loop_bit_identical[32]` | runs everywhere | same, same row |
| 3 | `test_mpa_fit_driver::test_end_to_end_at_the_si_pole_schedule` | runs everywhere | **bar not met, ~10×** — owner row |
| 4 | `test_minimax_imag_tables::test_the_fit_stage_floor_is_no_longer_a_quadrature` | runs everywhere | **bar is below the cell's own baseline** — owner row |
| ~~5~~ | `test_sigma_kirr_extraction::test_the_spread_stat_is_measured_before_the_drop` | WSL-SKIP / Perlmutter-RUN | ~~**genuine production defect**, mechanism pinned below~~ **CLOSED `96472c2e`** |
| ~~6~~ | `test_sigma_kirr_extraction::test_the_stamps_are_kin_ions_and_the_tables_are_filed_with_them` | WSL-SKIP / Perlmutter-RUN | ~~same defect, same row~~ **CLOSED `96472c2e`** |

**Rows 1–4 were red on WSL too, at the censused head.**  This is the second
place the WSL census's arithmetic does not hold: it books 232 cells as
"collected only at head, not red", and four of them were red on the machine it
ran on.  Measured directly, in a targeted run on a worktree at `12097f78`
itself: `4 failed, 1 passed`.  All four sources are byte-identical between
`12097f78` and today's `main` except one docstring line in `fit_driver`
(`e4e4aa3a`, zero executable lines), so this is the same code the census
graded.  None of the four is a device story:

* **`test_vmap_batch_equals_loop_bit_identical`** asserts
  `np.testing.assert_array_equal` between `fit_mpa_poles_batched` under `vmap`
  and the per-element loop.  The `[1]` parameter passes and `[5]` and `[32]`
  fail, in BOTH environments — which is the signature, not a flake: a batch of
  one lowers to the same kernel as the loop, and a batch of many lowers to a
  batched linear-algebra kernel with a different reduction order.  The
  magnitudes are 4.5e-11 to 7.7e-11 on WSL and 1.0e-11 on Perlmutter.  Whether
  the right answer is to relax the claim to a tolerance or to make the batched
  kernel genuinely reproduce the scalar one is a design question for the
  MPA fit-kernel lane, and **it is theirs**: bit-identity may well have been a
  deliberate contract, in which case the failing cell is reporting an
  implementation that drifted from it, and quietly widening the tolerance
  would delete the signal.  Registered, not silenced.
* **`test_end_to_end_at_the_si_pole_schedule`** measures
  `max|ΔΩ| = 9.98e-06` against a `1.0e-6` bar (WSL and Perlmutter agree to
  three digits; the Perlmutter run reads `9.978e-06`) at a worst Padé
  condition of `7.8e+09`.  A bar missed by one order of magnitude at a
  condition number near 1e10 is a statement about the schedule, not about the
  machine.
* **`test_the_fit_stage_floor_is_no_longer_a_quadrature`** measures
  `worst_eval = 9.98e-08` (WSL) / `9.14e-08` (Perlmutter) against a `1.0e-8`
  bar — while the cell's OWN synthesis baseline, the same recovery run on
  exact rather than sampled values, prints `8.65e-08`.  The assertion is
  therefore unsatisfiable as written: it asks the sampled fit to beat, by
  nearly an order of magnitude, a floor the exact fit does not reach.  The
  sampling half of the same cell is fine (`sample_error = 2.1e-12` against a
  `1e-11` bar).  This one is close to self-evident and still belongs to the
  minimax lane, because choosing the bar means deciding what the fit stage is
  promising.

**Rows 5–6 are the ones the WSL census could not have caught, and they are a
production defect, not a test defect.**  **CLOSED 2026-08-08 by `96472c2e`
(`fix/slabio-attrs-2026-08-08`); the diagnosis below stands as written and the
fix is the one it names.  Both cells run 10-passed green on a Perlmutter GPU
node at the landed main.  Everything below is history — read it for the
mechanism, not for the state of the tree.**  Both cells sit behind
`_need_slab_io()`, which skips when there is no phdf5 write symbol; on WSL
they skip ("no phdf5 write symbol on this platform; SlabIO has one transport
and does not fall back") and on Perlmutter they run.  When they run they say
the file carries no attributes at all — `dict(f["hartree_kij_ev"].attrs)` is
`{}` and `ds.attrs["k_storage"]` is a `KeyError`.  The mechanism is exact and
static:

* `file_io/sigma_output.py:920-939` hands every k_irr stamp to SlabIO the only
  way it can, as `io.create_dataset(name, ..., attrs=_attrs(name))`.
* `file_io/_slab_io_ffi.py:1770-1774` — the FFI/phdf5 backend, which is the
  ONLY transport SlabIO has — **discards `attrs=` and emits a
  `warnings.warn`**: "FFI backend: chunks/attrs on create_dataset currently
  no-op; pre-create with h5py if you need explicit chunking or attrs."

So on the one platform where the writer can run at all, `k_storage`,
`k_storage_version`, `n_sym_spatial`, `nk_full` and all five `star_spread_*`
numbers are silently dropped.  **The consequence reaches past the two cells.**
The star TABLES do reach the file, because they go through `write_attr`, which
defers to a rank-0 h5py write at close and works — so a cluster-written wedge
cube is a file with tables and no discriminant, which is precisely the
partial-stamp state `qirr_store` refuses on by design.  And
`file_io/kin_ion.py:219` reads `ds.attrs.get(K_STORAGE_ATTR, K_STORAGE_FULL)`,
so `read_star_map` returns `None` — "stored on the full BZ, read verbatim" —
for a file that is stored on the wedge.  Every k_irr-side consumer then takes
the full-BZ branch: `gw.eqp_bgw`, and the sanity gate this very landing
rewired at `tests/test_sanity_gates_jax.py:600`
(`krows = kmap if sig_star is None else k_irr_rows_for(...)`), which will index
`sigma_c_kij_ev[:, kmap]` into an array that only has the `nrk` wedge rows.
That is wrong Σ rows or an `IndexError`, depending on the deck.

This is NOT fixed here, deliberately.  The fix is in the collective write path
— the shape of it is to defer dataset attributes to close and write them with
rank-0 h5py, exactly as `write_attr` already does with `_deferred_attrs` — and
that is the `file_io` owner's call, it cannot be verified anywhere but the
cluster, and it does not belong on a branch named for a different fix.  What
this amendment does is take it out of the "unadjudicated new red" bucket and
put it in the open-row bucket with its mechanism, its file and line, and its
production consequence written down.

**And that is what closed it, the same day, in `96472c2e`** — the shape named
above is the shape it took: `create_dataset` records the attrs and `close()`
stamps them inside the rank-0 h5py reopen `write_attr` already owned, with the
values passed to h5py untouched so the transport and the host-side writers put
the same bytes in the file.  The `warnings.warn` is gone for `attrs`; `chunks`
keeps one, once per file, because HDF5 fixes layout at H5Dcreate and nothing
written later can move it.  Verified where it had to be: on one Perlmutter GPU
node, two worktrees carrying the same tests and differing only in `src/`, cells
5–6 plus five new transport cells go 9 failed / 1 passed at the base and 10
passed at the head, and the same wedge Σ cube written on a compute node reads
back through `read_star_map` as `None` ("full BZ") at the base and as a wedge
with its unfold tables at the head.

**OPEN ROWS this amendment leaves, all with a named owner:** the MPA
fit-kernel bit-identity contract (cells 1–2), the Si pole-schedule end-to-end
bar (cell 3), and the imaginary-axis fit-floor bar (cell 4).  The fourth — the
SlabIO FFI backend's dropped `create_dataset(attrs=)`, cells 5–6 and the wedge
Σ files it mis-stamped — is CLOSED by `96472c2e`.

### Where the evidence lives

Perlmutter: the census junitxmls the perf lane produced are
`/pscratch/sd/j/jackm/fixmpa_0808/_reports/census_{head_12097f78,base_f23baab2}.xml`,
and this amendment's own A/B is `/pscratch/sd/j/jackm/isoedge_0808/` —
`wt_before` and `wt_after` (copies of the census's `wt_head` differing in one
line), `leg.sh`, and `leg_{before,after}.log`.  The `fixmpa_0808/` tree was
read only.  The `RUNS_INFLIGHT` row for the two arms is struck with its
outcome.  In-tree: this amendment, and the two cells in
`services/symmetry_maps/tests/test_symmetry_maps_import_isolation.py`.

> **PATHS UPDATED AT THE MPA-BATCH LANDING.**  This row was written against
> `src/common/`, and the minimax service extraction merged one commit earlier
> in the same batch moved the bundle, the axis record and both selectors into
> `services/minimax/src/minimax/`.  The paths below are the post-move ones.
> Nothing else in the row changed: the counts, the certification and the open
> consumer question are as the branch measured them.

### OPEN ROW: the `damped_line` catalog is SHIPPED AND WIRED, and the fit
### stage is not yet calling it

`services/minimax/src/minimax/minimax_assets/catalog_damped_line.json` and its 29
`damped_line/*.npz` tables land here together with the door that serves them,
`services/minimax/src/minimax/damped_line_selector.py`.  Unlike the complex-Laplace row
above, this catalog is **not inert**: the selector is complete, it reads the
family's axis record, and it refuses by name above the top of the `A` ladder.
What has not happened yet is the consumer change — `gw.mpa.evaluator
.evaluate_samples` still calls `damped_line_rule` once per line under
`batching='per-line'` and runs two sweeps, where the catalog exists precisely
so it can make one lookup and run one.

That is deliberate and it is a scope boundary, not an oversight.  Wiring it
changes the fit stage's cost report shape (`batching` gains a `'global'` mode
and the per-line rows collapse to one shared row carrying the entry's identity,
node count and worst-case `kappa_0`), and that belongs with the MPA fit driver
rather than with the tables.  Until it happens the tables cost bytes in the
tree and nothing else, and `src/gw/screening.py`'s
`complex-axis omega ... not supported` refusal — the door the whole MPA fit
stage is standing outside — is still there.

**What IS true at this head**, so the row is not read as weaker than it is:

- 29 of 29 shipped entries certify, on a held-out `Delta` grid disjoint from
  every grid the solver saw and four times finer, at every one of the 55-to-58
  shipped weight rows on **both** sampling lines.  17 ship sparse (1.70-2.49x
  against one composite rule per line) and 12 ship composite (1.11-1.19x, from
  the sharing alone).
- ~~**`tests/test_damped_line_tables.py` is 120 passed, 1 failed at this head**~~
  **— SUPERSEDED, see "The far-line contradiction, settled" below.  This exact
  120/1 count is the one that section records as NOT REPRODUCIBLE on these
  bytes.  The measured figure is 16 failed / 105 passed** (re-measured again
  2026-08-09 by the landing-completeness audit, in a clean worktree at
  `bc37b4d3` with PYTHONPATH pinned: `16 failed, 105 passed` in 79.45 s, the
  same sixteen cells by name).  The paragraph is kept unedited below because
  this file strikes in place rather than deleting, but read it knowing that the
  `--spans 200 --merge` command it names closes **one** of the sixteen, not the
  whole red — the other fifteen are the far-line stamp defect, which is a
  generator bug and not a coverage gap.
  ~~and the one red is `test_the_catalog_covers_the_span_and_tier_ladder`.~~  It is
  the test doing its job: the ladder declares eight spans and the sweep landed
  seven of them plus one tier of the eighth, so three cells (`A = 200` at
  1e-8, 1e-10 and 1e-12) are absent.  They are absent for COST, not
  feasibility — the sparse attempt at the top of the ladder spends its whole
  wall budget in the prune before falling back — and
  `tools/generate_damped_line_assets.py --spans 200 --merge` fills them without
  recomputing anything already here.  **This red closes when that command
  runs**; nothing else in the suite depends on it.
- The wider minimax census at this head is `test_minimax_quadrature` **14
  passed**, `test_minimax_beta_selector` **25 passed** (green after the axis
  record was wired into it, which is the only edit this branch makes to that
  door), and `test_minimax_imag_tables` **93 passed, 1 failed**.  That one
  failure is the carried
  `test_the_fit_stage_floor_is_no_longer_a_quadrature` row registered a few
  sections above, and it is **not** this branch's: re-measured here in a clean
  worktree at the branch point `8f2a651d` it fails identically, `recovery
  9.984e-08 against a synthesis baseline of 8.652e-08`, which is the same
  host-conditioning of the Pade fixture that row already diagnoses.  Named here
  only so that a reader of the damped_line census is not surprised by a second
  red elsewhere in the same suite.
- Every entry's `kappa_0` is re-measured from its own shipped bytes and is at
  or under the `<= 2` shipping rule.  This family is the first where that cap
  BINDS: the uncapped selector produced live specimens at `kappa_0` of 406, 620
  and 1267 that meet their sup-norm tolerance, and one of them is in the
  harness as a red twin (`test_red_twin_from_the_gate_zero_specimen`).
- `pyproject.toml`'s `[tool.setuptools.package-data]` gains
  `minimax_assets/damped_line/*.npz` **and** `minimax_assets/complex_laplace/*.npz`.
  The second is a pre-existing packaging defect this branch fixes in passing:
  the complex-Laplace tables have been absent from every wheel install since
  they landed, because `MANIFEST.in` covers the sdist and package-data covers
  the wheel, and only the first listed them.

**Registered, not taken:** at `A >= 100` and `10^-10`, and at `A = 20` and
`10^-8`, the sparse route does not clear the composite and the composite ships.
Those cells are correct and certified — the composite is a rule of the same
family at the same cell, and one shared node set still beats one rule per line
by 1.11-1.19x — but gate zero found a sparse rule at `A = 100`, `10^-10` on the
near line alone, so a better search probably exists there.  The shipping rule's
clause (iii) is what makes the gap harmless rather than silent: an entry ships
sparse only if it uses strictly fewer nodes than the two composite rules it
replaces, and `composite_node_count` is recorded on every entry so the
comparison is a shipped number.
---

## Amendment — the BSE night's second consolidation checkpoint (2026-08-08)

Written by the consolidation-2 worker. Four branches merged onto `ff8631fd`,
two in-consolidation commits, one full census. `main` at **`581dc3cc`**.

### The census

`lx test --wait=1800` with no paths (⇒ `testpaths = ["tests", "services"]`,
default xdist geometry — the same geometry the previous three checkpoints
used), on Perlmutter, JID 56522011, Shifter, `LX_BASE_MODULE=lorrax_J070`,
jax `0.7.0.dev20260808`, `LORRAX_FFTW3_SO` pinned, `LORRAX_CHECKOUT` verified
in every leg's `[lx] source tree:` line.

```
21 failed, 2547 passed, 132 skipped, 1 xfailed in 543.37s   (head 581dc3cc)
```

**Zero unlisted reds.** All 21 are named in this document at the base tip
`ff8631fd`. The accounting was cross-checked per `<testcase>` element rather
than from the summary line — 21 `<failure>`, 0 `<error>`, 133 `<skipped>` out
of 2701 testcases, and the per-cell list length equals the counters. (The
previous amendment's WSL census mis-booked red cells into a collection bucket;
counting from the per-cell list is what makes that visible.)

The red set is SMALLER than the previous checkpoint's 27 by the nine
`symmetry_maps` import-isolation cells, which `fef002e9` closed, and the
`wq_resolvent` cell, which `28ff477f` closed. It is LARGER by the four MPA /
minimax rows this document already carries as open (cells 1–4). Nothing in
this landing is attributed to either direction.

### The one red this consolidation created, and closed

`tests/test_fft_shardmap_context::test_no_eager_local_fft_outside_shardmap_context`
was red on the merge head `2a62e75e` and green on `581dc3cc`. It is a FALSE
POSITIVE of that gate's lexical proxy, not a scaling regression: the non-TDA
port factored the stack matvec's shared conv+decode out of
`build_bse_stack_matvec._w_stack._body` into module-level `_conv_decode`, which
is still called only from inside `shard_map` bodies. Recorded as a ratchet
entry with its reason and with the durable fix named (teach the scanner the
call graph, then delete the entry). It never appeared on
`feat/sdy-nontda-solver-2026-08-08` because that branch's subset did not
include this gate — the red belongs to that branch, not to the merge.

### The K^d_B fix reached one of its two consumers, and now reaches both

`fix/kdb-zeta-sharding-2026-08-08` repaired the coupling encode in
`bse_ring_comm.py`. The SDY matrix-free route does not consume it: that lane
had PORTED the same encode into `bse_stack_matvec.py`, defect included. On the
merge head the two paths therefore disagreed at 2×2 by 47 % on the B block, the
SDY ladder's own cross-path gate (rung a) FAILED, and the driver route still
refused at `metric_sym_err = 9.834e-04` — SDY_SOLVER.md §7's pre-fix number,
unmoved. Commit `3a8223e4` transplants the kdb lane's fix. After it, at 2×2:
rung (a) B block `6.820e-15` PASS, `metric_sym_err` `1.968e-13` (its P=1 value
is `1.970e-13`), the dense-exactness gate `0.0000 µeV`, and the coupling
correction `−0.6980 meV` against the dense route's `−0.6980 meV`. P=1 is
bit-identical across the port on all four blocks.

**SDY_SOLVER.md §7's prediction is now satisfied**: the sharded-mesh refusal
has gone quiet, and it went quiet because the operator was fixed, not because
the check was relaxed. The detector is untouched — `src/bse/bse_nontda.py` and
`src/solvers/bse_sp_lanczos.py` are byte-identical to the SDY branch tip — and
its red twin still fires: reverting `_encode_T_B` to the pre-port form through
the real driver at 2×2 returns `9.834e-04` and refuses.

### Where the evidence lives

Perlmutter, `/pscratch/sd/j/jackm/bse_consol2_0808/`: `_reports/` carries both
census junitxmls (`census_2a62e75e.xml`, `census2_581dc3cc.xml`), the
accounting/set-diff script (`setdiff.py`), the census delta (`delta.py`) and
the pre-/post-port block dumps (`stk_{preport,postport,redtwin}_*.npy`);
`work/_logs/` carries the deck legs (`c2_p1.log`, `c2_p4.log`,
`c2_p1_post.log`, `c2_p4_post.log`, `rc_p4.log`, `rc_p4_post.log`,
`rc_p4_redtwin.log`). WSL: `~/lorrax_bse_perf_2026-08-08/CONSOLIDATION2_REPORT.md`.

---

## Amendment — the MPA batch integration landing (2026-08-08 / 09)

Six branches integrated and landed as one checkpoint, plus three registered bar
questions decided as deliberate commits. Base `70c0472a`; the census pair is
base `a3368c5b` against this batch's head, with the peer's one-file mesh gate
merged after and measured separately (below). This is the night's final landing
on this side.

### What merged, and the one branch that had to be integrated

Five of the six merged as merges. `svc/minimax-2026-08-08` (the service
extraction), `feat/mpa-chi0-resolvent-2026-08-08` (independent files),
`feat/ff-compute-mode-2026-08-08` (a mode that ships refusing) and the
damped-line/fullprec stack all composed without argument once the sixth was
resolved.

The sixth is `feat/minimax-beta-selector-2026-08-08`, and it is worth stating
why it could not simply merge. Both it and the extraction were written against
the same file in the same hour. The extraction moved `common/minimax.py` and
the shipped table bundle into `services/minimax/`; the beta axis was written as
`common.minimax_beta_selector`, resolving its catalog through
`importlib.resources.files("common")`. Merged as written, the selection rule
would sit one import edge away from the bytes it selects from, reading a
package that no longer has them. So the axis moved with the bundle, and when
the damped-line branch arrived one commit later carrying a shared axis record
and a second selector with the same `files("common")` in them, the whole family
layer moved: `beta_selector`, `family_axes` and `damped_line_selector` are all
in the service now, all resolving through `_catalog`'s own `ASSET_PACKAGE`, all
on the door. Two things that git could not have known to fix came with it — the
damped-line generator and its certification harness were still writing to and
reading from `src/common/minimax_assets`, which would have produced tables
nothing could find and a harness certifying an empty ladder.

`complex_laplace` is WIRED, which is the flip that branch was written to earn,
and the flip came with the routing that makes it safe rather than as a flag
change. The family's row said `wired=False` because the generic rule matches on
three axes that all round safely and beta rounds neither way — the target is a
different function at every beta — so a beta-blind match answers a different
question. `lookup(family='complex_laplace', ...)` therefore does not reach
`select_entry` at all; it reaches the beta axis, and it refuses a request that
names no beta or no clause as a vocabulary error rather than as a miss. Two
service cells asserting the old state were rewritten to assert the new one,
each keeping a red twin.

### The census

One script, both legs, run against two worktrees so a set-diff cannot be an
artifact of two different invocations. WSL, PYTHONPATH pinned per worktree per
the build note's trap. This box has one CUDA device, so both legs are
GPU legs.

| | collected | red |
|---|---:|---:|
| base `a3368c5b` | 2701 | 136 |
| head (this batch) | 3046 | 148 |

In both: 2681. Collected only at head: 365. Collected only at base: 20.

**Newly red in the intersection: ONE, and it is not a regression.**
`test_mpa_fit_kernel::test_exact_recovery_metal_grid_alpha2`. This landing does
not touch that cell — the diff against the base for that cell is empty — and it
is FLAKY AT THE BASE: eight runs at `a3368c5b` give six passes and two
failures, always at the same value, `1.0837827302938041e-06` against a `1e-06`
bar. On CPU it passes four times out of four. It is the same mechanism bar
decision (a) below documents, caught in a third cell: XLA selects a different
GEMM implementation between processes, so the answer moves in its last bits and
a bar 8% away is sometimes on the wrong side of it. The census caught it on the
head leg and not the base leg because that is what a 25%-per-run flake does to
a single-sample census. **Registered as a pre-existing flake with a measured
rate, not attributed to this landing.**

**Newly green in the intersection: five**, and they are the three registered bar
questions: `test_the_fit_stage_floor_is_no_longer_a_quadrature`,
`test_end_to_end_at_the_si_pole_schedule`, and the three
`test_vmap_batch_equals_loop_bit_identical` parametrizations.

**The +collected bucket, checked cell by cell rather than trusted.** The
fixes+MPA landing shipped four reds through this exact hole — cells that do not
exist at the base cannot be regressions by construction, so a set-diff drops
them silently. The set-diff script used here grades every member of that bucket
and prints the red ones by name. Of 365 newly collected cells, **16 are red and
all 16 are `test_damped_line_tables`**, adjudicated in the next section. The
other 349 are green.

**The 20 de-collected cells are renames, and none was red.** Eighteen are
`test_minimax_imag_tables` parametrizations whose ids carry the catalog's beta
values, which moved from the request census's three-decimal display
(`b16.006`) to the decks' own full precision when the fullprec campaign
regenerated; the head collects 83 and 54 of those two cells against the base's
18 and 18. Two are `test_minimax_quadrature` cells the service extraction
retired into the service's own suite. All twenty were green at base, so nothing
red left the tree unexamined.

### The far-line contradiction, settled

Two workers reported `test_damped_line_tables` at the damped-line tip
`74423bc7` on the same day and disagreed: 16 failed / 105 passed, and 120
passed / 1 failed. Settled by measurement, in a clean worktree at that tip with
PYTHONPATH pinned: **16 failed / 105 passed**, matching the fullprec worker's
count, and the same sixteen cells by name at this landing's head. The failures
are not a merge artifact and not an environment artifact of the runs taken
here: this file resolves everything from `Path(__file__)` and reads the shipped
bytes off disk, it imports no jax, and `heldout_delta` draws no random numbers,
so the measurement has no device and no seed to vary. The 120/1 count is not
reproducible on these bytes; the catalog and its payloads landed in a single
commit (`04ba10ad`), so it cannot be explained by a half-written bundle either,
and it is recorded here as unreproduced rather than explained.

**And the load-bearing claim is NOT what is red, which is the part that
matters.** The far-line cells assert two different things and only one of them
fails. The design claim — that a composite entry's far line rides the NEAR
line's node set inside its own tighter budget, which is what buys the
one-global-sweep economy — is measured intact: the far rows score between
`1.5e-06` and `2.6e-03` of their budget, three to six orders under it. What
fails is the cell's other assertion, that recertifying from the shipped bytes
reproduces the entry's STAMPED statistic to `rel=1e-9`.

Measured across all 29 entries, comparing the stamp against a recertify on the
full row set and on the far rows alone:

* the two recertifies agree with each other to 1e-12 on every entry, so the
  subset method the cell uses is sound and the far-line measurement is not
  row-set dependent;
* seven entries' stamps are missed by both, by relative amounts from `1.4e-05`
  (A=120, 1e-8) to `5.5e-03` (A=40, 1e-10);
* **all seven are `btv_minimax` and none is `positive_composite`**, which
  points at a final step on that route — a polish or a re-solve applied to the
  payload after the certificate was computed and written.

So this is a generator bookkeeping defect, not a physics defect: the shipped
tables are as good as claimed and the numbers recorded beside them are stale in
their fifth to sixth significant figure. **OPEN ROW, owner: the damped_line
generator lane.** The fix is to re-certify after the last step that touches the
payload and rewrite the stamps, and the check that it worked is this suite going
16 → 0. The far-line documentation lands as written, because its claim is
measured true; what does not land is any assertion that the recorded digits are
reproducible, and that is exactly what these sixteen cells are still saying.

### The three registered bar questions, decided

Each is its own commit with its reasoning, and each was measured before it was
decided. In two of the three the measurement changed the answer.

**(a) The vmap bit-identity cells.** Registered as "the cell's claim is false;
document the contract as reduction-order-tolerant with the measured bound, not
a silent widening". Done — and the registered DIAGNOSIS did not survive
measurement. It read that `[1]` passes because a batch of one lowers to the
same kernel as the loop while a batch of many lowers to a batched kernel. Run
the cell six times on this host's GPU changing nothing and `[1]` fails three
times and passes three times, always by the same `7.99e-11`. The gap is not a
function of batch size; it is XLA choosing between GEMM implementations across
processes, and bit-identity was never a contract this kernel could keep. The
CPU lowering is bit-identical at every batch size, which is the control that
makes this a statement about the backend. Bound sized from measurement over
batch sizes 1, 2, 5, 17, 32, 64: worst relative gap `2.2e-10`, not growing with
batch size. Bar `rtol = atol = 1e-8`, ~45x that, covering the decade of
cross-host spread already on record. **`valid` — the discriminant that decides
which poles are kept — stays exactly equal**, because a 1e-10 gap in a
reduction must never move a keep/drop decision, and relaxing that alongside the
numerics is the silent widening the row existed to prevent.

**(b) The driver's end-to-end bar.** Registered as "bar missed ~10x at cond
7.8e9 — a statement about the schedule, not about the machine". It is about the
machine. Same fixture, same bytes, same condition number `7.835e9`:
`JAX_PLATFORMS=cpu` gives `max|dOmega| = 6.33e-08` and passes the old `1e-6`
bar; this host's GPU gives `9.99e-06` and fails. A factor of 158, and not the
disk (a complex128 round trip is exact) and not the fixture (identical). That
also explains a puzzle in the original row: the cell's own docstring recorded
`6.3e-8`, well UNDER the bar it was failing. Those are the CPU numbers — the
cell was written on one device and censused on another, and both census legs
that reported it were GPU legs, which is why they agreed with each other and
not with the docstring. The theory guide's §3.6 table carries the same `6.3e-8`
row and inherits the caveat. The bar is now computed from the run's own worst
conditioning, which is §3.6's own law: `floor = cond * eps_mach = 1.74e-6`,
with the measurements at 5.7x floor (Omega, GPU), 0.036x (Omega, CPU) and 72x
(B, GPU), and bars at 30x and 300x. It reports an algorithm regression on
either device and lets the fixture's conditioning move without calling weather
a defect.

**(c) The fit-stage floor.** Registered with the beta-selector landing's
suggestion: make it relative to `worst_synth`, which is what the docstring
already claims it means. The relative bar was already there one line below; the
absolute `1e-8` is deleted, nothing is widened. The fullprec stack was expected
to resolve this row outright on a `3.33e-9` recovery against a `3.81e-9`
baseline. It does not resolve it here: `9.98e-08` against a baseline of
`8.65e-08`. That is not two workers disagreeing, it is one mechanism measured
twice — the synthesis baseline is the Padé fixture's conditioning and therefore
the host's LAPACK. The two baselines are 23x apart, the two RATIOS agree to
within 30% (1.15 and 0.87), and **the absolute `1e-8` falls between the two
baselines**, which is precisely why identical code passes it on one machine and
fails it on the other. Taking the greener measurement as the resolution would
have shipped a bar that reports whose machine ran it.

### The fullprec branch: PUSHED, and it subsumed its predecessor

Registered in the brief as possibly absent. It exists at `863a8f01`, stacked on
the damped-line tip, and is merged here: 18 → 54 certified `complex_laplace`
entries. The first campaign swept a beta grid taken from the request census's
three-decimal display, so every entry was fitted near — not at — the beta a
deck asks for, which on an axis that does not round is the difference between a
served table and a near-miss refusal. This one sweeps the decks' own
omega-hats and adds the 1e-12 fit-stage tier. hBN stops being a refusal.

### The peer's mesh gate, measured separately

`70c0472a` adds `tests/test_bse_coupling_routes_mesh_invariance.py` and no
source, so it is outside the census pair by construction. Run at this landing's
head it is **13 failed / 2 passed** — and it is **13 failed / 2 passed at its
own base `70c0472a`** in a clean worktree, identical. Pre-existing on this box
and untouched by this batch. Not attributed here, and named so the next WSL
census is not surprised by thirteen reds with no owner.

> **AMENDED 2026-08-09 (landing-completeness audit) — the count is right and
> the DIAGNOSIS was wrong.** ~~the gate wants a mesh this single-device machine
> cannot provide~~ It is not the mesh. It is `LORRAX_BANDS_GEMM_FFI`. Measured
> on this same single-device box, same tree, same commit:
>
>     pytest tests/test_bse_coupling_routes_mesh_invariance.py   ->  13 failed, 2 passed
>     LORRAX_BANDS_GEMM_FFI=0 pytest <same file>                 ->  15 passed
>
> The gate makes no FFI requirement of its own; what it CALLS does —
> `build_bse_ring_matvec_full` -> `contract_bands_block_reshard` ->
> `gate.require` refuses with "MKL batched-GEMM host backend unavailable" unless
> the dial announces the debug opt-out. The file is green in a DEFAULT census
> only because collecting `tests/test_contract_bands.py` runs a module-scope
> `os.environ.setdefault("LORRAX_BANDS_GEMM_FFI", "0")` that leaks to the whole
> session — so this gate's verdict is **collection-scope dependent** on any
> FFI-less box, and thirteen reds from running it ALONE are an artefact, not a
> mesh-invariance failure and not evidence that either K^d_B fix regressed.
> The caveat now lives in the gate's own module docstring (`db919ee3`); the
> durable fix is to move that `setdefault` into a fixture, and it belongs to
> `test_contract_bands.py`. Recorded because "wants a mesh" would send the next
> census to buy hardware for a one-line env problem.

### An unlisted red on the jax-0.9 leg: a tracing instrument, not a physics cell

**REGISTERED 2026-08-09 by the landing-completeness audit.**
`tests/test_bse_w_omega_chain_scan.py::test_chain_step_traces_once_for_the_whole_chain`
is **RED on jax 0.9.1** (`assert 2 == 1` on the trace count; the file is
1 failed / 11 passed). It has never been named here, and it should have been.

**It is not a campaign regression, and the A/B says so.** It is red at its own
BIRTH commit `ca1ca52a` — checked out via `git archive`, no git mutation — on
CPU and GPU alike under jax 0.9.1, while the 12/12-green evidence behind it was
taken on **jax 0.7.0 on Perlmutter**. So this is a version-leg red on an
instrument that counts traces, the same class as the P4 remat row: an instrument
written against one stack and read on another. Nothing about the chain's
numbers is implicated — the other eleven cells, which are the physics, are green.

**What it is NOT evidence of, checked explicitly.** `fix/kirr-fullids-2026-08-08`
(`7a2ac1db`, merged `b50aa641`) corrects k-row mislabeling on affected decks, and
this file's `CHAIN_REL_TOL` gate holds a frozen reference. Re-run on a tree that
CONTAINS that fix: **11 passed, 1 deselected** — the frozen reference still
holds, so the chain quantities do not traverse the corrected map and there is
nothing to re-anchor here. Recorded so that nobody re-cuts this reference on the
strength of the trace-count red sitting next to it.

**The open question is real and should not be suppressed:** does jax 0.9
legitimately trace that chain twice, or has the single-compile property this
instrument exists to protect actually been lost? A green from pinning the
counter to 2 would answer the wrong question. Owner call; the cell stays red and
named until someone reads the traces.

### THE OWED PERLMUTTER CENSUS

**OWED, and this is the third landing in a row to owe one.** Both legs of this
census ran on WSL, on a box with one CUDA device. The rule the isolation-edge
amendment wrote down applies without modification: a WSL leg is evidence about
code and is not evidence about a cell whose premise is a device or a built
`.so`. This census is structurally blind to the FFI-backed families, to
anything needing a multi-device mesh — which is exactly what the peer's thirteen
reds above are — and to the SlabIO transport cells.

It is staged one command away, on the `fixmpa_0808` precedent: two worktrees at
the base and the landed head, one script that runs the same invocation against
both, `LX_BASE_MODULE=lorrax_J070` exported (unset costs ~52F/32E instead of
14F), `--junitxml=` first, `-m=<mark>` spelled with the equals sign, and the
`[lx] source tree:` line read in every log before any leg is believed.

### The day's defect classes

The conjugation class stays at **four** instances. The K^d_B defect founded a
**second** class — rotating a partial contraction along the shard's own axis —
now cured in both known instances, with a portable gate guarding it. This
landing adds no instance to either. It does surface a third pattern that is not
a defect class but is worth naming, because it cost three separate
adjudications tonight and will cost more: **a numerical bar written on one
device and censused on another**. All three of this landing's bar questions, and
the one newly-red census cell, are that pattern. Two forms: run-to-run GEMM
selection inside one device (bar a, and the metal-grid flake), and CPU-vs-GPU
accuracy at high condition number (bar b, 158x at cond 7.8e9). The defence is
the one bar (b) now uses — derive the bar from the run's own measured
conditioning rather than freezing a constant measured somewhere else.

### Where the evidence lives

WSL: `~/.../scratchpad/_mpabatch_reports/` — `census_base.xml`,
`census_mpahead.xml`, both `.log`s, and `setdiff_mpabatch.txt` (the set-diff
with the +collected red check), beside `census.sh`, the single script both legs
ran, and `setdiff.py`. Worktrees `wt_census_base` (a3368c5b), `wt_peerbase`
(70c0472a) and `wt_mpabatch` (the landed head). In-tree: this amendment, the
three bar-decision commits and their docstrings, which carry the derivations
where the numbers are rather than in a commit message.

## 2026-08-09 amendment — the cluster census ran, both legs, and it found one thing

This census is the one three landings in a row owed: both legs on Perlmutter
(JID 56532188, warm-attach only — the worker created no allocation), the
restage-candidate FFI pair pinned, `LX_BASE_MODULE=lorrax_J070` exported, and
the parser validated BEFORE any number arrived — replayed against the last
registered census (`census2_581dc3cc.xml`) it reproduces that amendment to the
cell, including booking the xfail out of the `<testsuite>` element's
`skipped=133`, which is the one-cell disagreement that makes the summary line
unusable as an instrument. It refuses any junitxml under 50 kB (the
`$HOME`-full 38-byte shape).

### The two legs

BASE `origin/main` @ `e37c6a6e`: 3112 collected — 2946 passed / 33 failed /
0 errors / 132 skipped / 1 xfailed; 2946+33+0+132+1 = 3112, exact. TIP
`bf5604cd` (`integration/mpa-table-2026-08-09`): 3242 collected — 3075 / 34 /
0 / 132 / 1; exact. Each leg ran in its own fresh worktree and the
`[lx] source tree:` line was read in both logs before either was believed.

### The verdict: one new red, and it is a stale test, not a behaviour change

`tests/test_qsgw_dataset_writer.py::test_the_cube_is_written_at_both_seams_and_only_there`
went red on the tip because the driver-seam refactor factored the tail of
`compute_sigma_xc` into a private helper `_finish_dynamic_sigma`, taking the
`write_qsgw_sigma_cube` call with it — and the cell's `_calls_of` proxy
returns the ENCLOSING FunctionDef name, not the call graph. The invariant the
cell guards is intact, proven four independent ways: the helper's sole callers
are inside `compute_sigma_xc` (`sigma_dispatch.py:561`, `:613`); the sibling
gate `test_the_one_shot_cube_write_is_behind_the_file_write_flag` is green on
the tip; the other two seams are structurally identical on both trees; the
file's other 37 cells are green. Red-twin A/B, single process, file alone,
fresh worktree each: BASE 38 passed, TIP 1 failed / 37 passed — deterministic
and geometry-independent. This is the SECOND instance of the lexical-proxy
false-positive class (first: `test_no_eager_local_fft_outside_shardmap_context`
at the second consolidation checkpoint). The fix — teach the cell the call
graph, or assert the helper's sole caller — belongs to the integration branch
and blocks its landing; this row retires when that one-line fix is measured
green on the file-alone A/B.

Beyond that one cell: zero reds went green unexplained, zero reds among the
137 newly-collected cells (the bucket that once shipped four reds unnoticed —
every member was graded by name), zero skip movement, and the 7 de-collected
cells are exactly the ff-mode refusal cells retired by design, all green at
base, replaced by 16 green cells on the tip.

### What the base leg settles

All 33 base reds reconcile against this register BY NAME — zero unlisted.
On the device instrument, at the current head: the nine `symmetry_maps`
import-isolation cells are GREEN (`fef002e9` confirmed where it was owed), the
four MPA/minimax rows red at `581dc3cc` are green, and both registered flakes
(`test_exact_recovery_metal_grid_alpha2`,
`test_wq_resolvent_matches_restart_finite_q`) passed. The owed-census rows
above asked whether the WSL-blind families hide unlisted reds; at head, on
device, with the FFI pair pinned, the answer is measured: they do not. The
historical base/head pairs those rows name were not re-run and stay as
history.

### The damped-line row becomes cell-checkable

The register said "16 red and all 16 are `test_damped_line_tables`" but named
only one cell, so a future census could check the count and the file but never
WHICH sixteen. Measured at `e37c6a6e` on the device leg, they are:
`test_the_catalog_covers_the_span_and_tier_ladder`, plus
`test_shipped_entry_recertifies_from_its_own_bytes` at
`[A100_eps1e-08_btv_minimax]`, `[A100_eps1e-12_positive_composite]`,
`[A120_eps1e-08_btv_minimax]`, `[A120_eps1e-12_positive_composite]`,
`[A160_eps1e-10_positive_composite]`, `[A20_eps1e-12_positive_composite]`,
`[A40_eps1e-12_positive_composite]`, `[A60_eps1e-08_btv_minimax]`, plus
`test_the_far_line_rides_the_same_node_set` at
`[A100_eps1e-06_btv_minimax]`, `[A100_eps1e-08_btv_minimax]`,
`[A120_eps1e-08_btv_minimax]`, `[A40_eps1e-10_btv_minimax]`,
`[A60_eps1e-08_btv_minimax]`, `[A80_eps1e-06_btv_minimax]`,
`[A80_eps1e-08_btv_minimax]`.

### Where the evidence lives

Perlmutter, read-only: `/pscratch/sd/j/jackm/mpa_census_0809/` —
`_reports/suite_base.xml` (606,260 B), `_reports/suite_tip.xml` (624,556 B),
`setdiff.txt`, `reds_base.txt`, `_verify/qsgw_{base,tip}.{xml,log}`, `_logs/`.
The RUNS_INFLIGHT row was appended before first submission and struck with
results.

## 2026-08-09 amendment — two runtime defects from the Σ scaling lane, neither of them a test red

The sigma GN-PPM scaling lane (`perf/sigma-scaling-2026-08-09`, workspace
`/pscratch/sd/j/jackm/sigma_scaling_0809/`, prose record
`~/lorrax_bse_perf_2026-08-08/SIGMA_SCALING.md`) landed one instrumentation
commit and, on the way to it, hit two defects that no cell in this suite
currently catches.  Both are registered here rather than as reds, because
neither turns a test red and both will silently waste an allocation for
whoever meets them next.  The lane's instrument commit is included in the same
landing as this amendment; the two defects below are **not** fixed by it.

### 1. `distributed_eigh` hangs at a 3×3 mesh for every n ≥ 3072

A 3×3 cuSOLVERMp `syevd` completes n = 2049 in 6.9 s and then never returns at
n = 3072 or anything larger.  It does not fail — it hangs silently, with the
cuSOLVERMp banner printed (`library 0.7.2, NCCL 2.27.3, comm path: NCCL,
grid: 3x3 (col-major)`) and nothing after it, on a card reporting 36.4 GiB
free at the moment of the call.  So this is neither memory starvation nor the
`status=7` failure the older record remembers, and the break is bracketed
between n = 2049 and n = 3072.

| mesh | grid reported by cuSOLVERMp | n | allocator arm | result |
|---|---|---|---|---|
| 2×2 | 2x2 (col-major) | 8192 | recommended (`default`, MEM_FRACTION 0.85) | **PASS**, 9.347 s, max eigenvalue error 1.31e−10 |
| 3×3 | 3x3 (col-major) | 2049 | recommended | **PASS**, 6.935 s, max eigenvalue error 1.50e−11 |
| 3×3 | 3x3 (col-major) | 3072 | recommended | **HANG** — killed at 420 s |
| 3×3 | 3x3 (col-major) | 4098 | recommended | **HANG** — killed at 420 s |
| 3×3 | 3x3 (col-major) | 6144 | recommended | **HANG** — killed at 420 s |
| 3×3 | 3x3 (col-major) | 8190 | recommended | **HANG** — killed at 900 s |
| 3×3 | 3x3 (col-major) | 8190 | `platform` (fleet default) | **HANG** — killed at 420 s |

**The allocator is exonerated and the defect is pre-existing.**  The last row
is the load-bearing one: the hang reproduces identically under `platform`,
which is what the fleet runs today, so whatever is wrong at 3×3 was wrong
before the allocator question was ever asked and will still be wrong if the
allocator recommendation is rejected.  This is a `distrib_la` solver defect,
it belongs to whoever owns `distrib_la`, and until it is owned, **no LORRAX
stage that calls `eigh` should be run at a 3×3 mesh on a large matrix** — which
is what makes it a blocker for large decks rather than a curiosity.  The Σ
deck the lane measured never calls `eigh` at that size, so none of the lane's
timing conclusions depend on it.

One lead, offered as a lead and not as a finding: cuSOLVERMp distributes 2-D
block-cyclic, and at a 3×3 grid n = 2049 gives a local block of 683 — below
the usual 1024 tile — while n = 3072 gives exactly 1024 per rank and therefore
more than one block column per rank for the first time.  A hang appearing
exactly where the block-cyclic distribution stops being trivial on a
non-power-of-two grid is a plausible library-side story, but it was not
confirmed.  The control that would separate "3×3 is odd" from "any grid past
2×2" is a 4×4 leg at n = 8192; it never got a placement before the lane's
window closed, and its log carries no cuSOLVERMp line at all, so its non-zero
exit is a queue artifact and must not be read as a hang.

Evidence: `/pscratch/sd/j/jackm/sigma_scaling_0809/_reports/` legs
`la_p4_bfc85`, `la_p9_bfc85`, `la_p9_platform`, `la_p9_small_bfc85`,
`la_p9_3072`, `la_p9_mid` (n = 4098) and `la_p9_6144`; the probe is
`sigma_scaling_0809/probe_pressure_sq.py`, which is the allocator lane's
`probe_pressure.py` with one change — its `build_mesh` picked `(n, 1)` for any
device count other than 1 or 4, which would have measured a 9×1 cuSOLVERMp
grid rather than the 3×3 grid LORRAX actually runs.  Prose:
`SIGMA_SCALING.md` §8.  The `la_p16_8192` leg is the one that did not run.

### 2. A profiler session live across the phdf5 collective close segfaults rank 0

Setting `ISDF_JAX_PROFILE_DIR` opens a profiler session at *every*
`trace_section` call site in a run, and the first of those is `zeta_fit`.  On
a multi-node mesh that session is live across `zeta_q.h5`'s phdf5 collective
close, and rank 0 segfaults there.  Reproduced three times at a 3×3 mesh over
three nodes, always immediately after `[SlabIO.close] H5Fclose returned in
0.0 s` and always about twenty seconds in; a 2×2 single-node mesh traces
cleanly, so single-node tracing is unaffected.  What the geometry evidence
strictly pins is "3×3 over three nodes segfaults, 2×2 on one node does not" —
mesh size and node count were not separated by a control, and a lane that
needs that distinction should measure it rather than quote this row.

The practical consequence was total: the section anybody actually wants, the
sigma tau kernel, is nine sections later and was never reached, so tracing the
production GN-PPM Σ deck above a 2×2 mesh was not slow or noisy but
**impossible**.

**Worked around, not fixed.**  The instrument commit in this landing adds
`ISDF_JAX_PROFILE_SECTIONS`, a comma-separated allowlist of substrings
consulted by `_trace_path` alongside the directory, which makes the tau kernel
reachable at P > 4 by not opening a session at `zeta_fit` at all.  Unset —
which is what every existing caller passes — every section still traces, so no
call site changes behaviour.  **The root cause is open**: why a profiler
session interacting with the phdf5 FFI's collective close should segfault rank
0 is unanswered, and the allowlist only routes around it.  Anyone who sets
`ISDF_JAX_PROFILE_DIR` on a multi-node mesh without also setting
`ISDF_JAX_PROFILE_SECTIONS` will still meet this.

Evidence: `/pscratch/sd/j/jackm/sigma_scaling_0809/_reports/` — the red twin's
gates `gate_allowlist.log` / `.xml` (branch, 6 passed) and
`gate_allowlist_PRE.log` / `.xml` (pre-fix base `3e5e98ba`, 4 failed / 2
passed).  The absence is evidence too: there is no `_traces/tr_p9` or
`_traces/tr_p16` under that workspace, only `_traces/tr_p4/`, and the
`tr_p16` / `tr_p9b` logs record `LX-POOLFULL` with no step rather than a
misleading exit code.  Prose: `SIGMA_SCALING.md` §9.  The guarding cells are
`tests/test_jax_profile_section_allowlist.py`, six of them; they pin the
allowlist's behaviour, including that an empty or whitespace value means
"unset" rather than "trace nothing", and they say nothing about the segfault.

---

## 2026-08-16 — two cells in `tests/test_w_bse_wiring_closure.py` are RED ON PURPOSE

Branch `feat/screening-diagrams-wbse-2026-08-15` (`screening_diagrams =
w_rpa | w_bse`). These are NOT infrastructure failures and NOT flakes; each is
a measurement that came out against the feature, and each is left failing
deliberately rather than `xfail`-ed, because an `xfail` would make the suite
green while `w_bse` is wrong at finite q.

```
tests/test_w_bse_wiring_closure.py::test_the_ladder_w_passes_the_production_w_gate_at_finite_q
tests/test_w_bse_wiring_closure.py::test_the_ladder_operator_obeys_q_conjugate_reciprocity_without_the_unfold
```

**What they measure.** The ladder screening operator violates
`W(-q) = conj(W(q))` — the property the BSE kernel's own hermiticity reduces to
(`common/sanity.py:511-533`), and the one `gw/screening._gate_w` checks over the
whole flat-q axis. Measured on a 2x2 in-process mesh, gnppm_debug fixture,
`n_rmu = 399`, full-basis probe, `lx` JID 57064957:

| observable | value | bar |
|---|---:|---:|
| assembled full-BZ ladder tile, `max abs(W_q - conj(W_-q)) / max abs W` | **7.251e-05** | 1e-05 |
| the SAME assembly with `include_w=False` (RPA operator, same plumbing) | 7.153e-12 | — |
| the production `W0_qmunu` off disk | 7.151e-12 | — |
| operator only: solve at `q` and at `-q` separately, NO symmetry table | RPA **4.081e-11**, LADDER **3.579e-04** | — |
| per-q hermiticity of the ladder W, all five q_irr | 1.316e-11 … 3.408e-11 | 1e-06 |

So the assembly is exonerated to three significant figures and hermiticity is
clean everywhere; the break is in the operator, and it is the KNOWN_FAILURES
line 1248 class (finite-q vertex conjugation, invisible at q=0) reappearing in
the ladder's direct rung.

**Do not** loosen `_gate_w`, and do not mark these `xfail` to get a green
board. `KNOWN_LORRAX_ISSUES.md` (sandbox) carries the defect row; the fix is
theory and is the owner's. The rest of the file — the wiring-closure gate
itself — is GREEN at 8.507e-09 (q=0) and 9.273e-09 (finite q_irr) against a
1e-08 ceiling.

Evidence:
`/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/reports/screening_diagrams_wbse/evidence/`
— `operator_reciprocity_mesh4.*`, `ladder_finiteq_gate_mesh4.*`,
`closure_gate_mesh4.*`, `combined_mesh4.*`. Sandbox claims 0214 / 0215.

## test_w_head_densify's equal-grid cell fails after another suite runs

OPEN, 2026-08-27, environment interaction, not a branch regression.
`tests/test_w_head_densify.py::test_the_loader_does_not_defer_when_the_grids_are_equal`
passes when its file runs alone — 70 passed, with no FFI library, with the
host leg, and with both legs — and fails inside a multi-file run with
`IndexError: index 0 is out of bounds for axis 0 with size 0` at
`src/bse/bse_loading.py:202`, where `_get_local_mesh_coords` looks up
`jax.local_devices()` in the 1x1 mesh the cell just built. Some other
module in the run changed the process's device set first; this is the same
collection-time `os.environ` mutation that makes a bare `pytest tests/`
exit 4.

Measured on both sides of the receipt fix with one command and the same two
FFI libraries: 1 failed / 301 passed / 8 skipped at `6fafd126`, and 1 failed
/ 304 passed / 8 skipped after it, the same cell each time. Pairing
`test_sanity_gates_jax.py` before it reproduces the failure; pairing
`test_jax_cache_contract.py` before it does not.
