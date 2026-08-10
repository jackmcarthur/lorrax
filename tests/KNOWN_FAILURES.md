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

# AMENDMENT — `mu_small = auto` IS RANK-SIZED, NOT ACCURACY-SIZED (2026-08-10) — **OPEN**

**One row, and it is not a red — it is a wrong physical number produced by
following this tree's own documented guidance, with nothing anywhere
refusing.**  It is filed here rather than as a small-issues row because it
implicates physics: the setting `docs/downfold.md` recommended until this
branch sized a downfold whose lowest BSE eigenvalue came out 2.087 eV wrong,
and the owner's bar for production BSE is under 1 meV.  Nothing in the suite
could have caught it, and that is the durable part.  `tests/test_downfold.py`
is deck-free by design — it gates the algebra (T = I on a full keep, the
congruence, the Pythagorean identity and their red twins) and the rank
refusal, none of which is an accuracy statement about a real deck.  The one
cell that would have caught this, "downfold `si_bse_debug`, run the BSE
driver on the small bundle unchanged, compare against the parent", needs a
GPU and a finished GW run and lives in the campaign report rather than in
`tests/`.  So the tree has no gate anywhere that compares a downfolded
observable against its parent's, and until the sizing mode below lands, the
only instrument is a user running that comparison by hand.

| item | mechanism, at this tree | disposition |
|---|---|---|
| **`mu_small = auto` sizes μ_S by the eigenvalue rank ceiling, which is not an accuracy criterion, and the docs recommended it** | `auto` sets μ_S to the number of independent pair-density directions the retained window holds at `downfold_rcond` — the largest value the driver will accept, and a statement about what the parent can still *represent*.  Rank completeness at that ceiling implies observable accuracy only on a parent that is OVER-COMPLETE for the window; nothing in the driver measures whether the parent is, and on a parent that is merely adequate the "compression" is a truncation with a compression's reporting.  Measured on the standard si pipeline walk (`~/lorrax_bse_perf_2026-08-08/PIPELINE_HEALTH.md`, 2026-08-10, `si_bse_debug` 4×4×4, a 936-centroid parent on a `0:20` window): `auto` → 189 at `downfold_rcond = 1.1e-6` moved the lowest BSE eigenvalue from the parent's **2.3449 eV to 0.2579 eV**, an error of **2.087 eV**, and the run exited 0 with a clean log.  `eps_W` read **1.33e-2** — about one per cent — which is the tripwire behaving exactly as documented and is the third measured case in which a ~1 % `eps_W` sat beside an error of 37 meV, 1.7 eV and now 2.09 eV; it is not an accuracy gate and was never claimed to be one.  Tightening to `rcond = 1e-8` (`auto` → 624, a compression of only **1.5×**) still left **1.08 eV**.  The error falls with μ_S, so the transfer solve is doing its job and this is a sizing defect rather than a numerical one — but on this deck the accuracy a BSE needs and the compression the downfold offers did not overlap at any μ_S the driver would accept.  Caveat on the record from the walk: that parent was pruned on the deck's Σ window (`8 / 52`) because it is the only invocation the tree documents, not on the 20-band window the downfold was fitted to; whether a differently-pruned parent behaves better is exactly the question the docs do not answer | **OPEN**, and it stays open until a **target-accuracy sizing mode** exists — the planned stage-4 item on the downfold roadmap, in which the user states a meV bar and the driver sizes μ_S by sweeping it against the parent observable.  Nothing in this tree sizes μ_S for accuracy until that lands.  What `fix/downfold-auto-guard-2026-08-10` does in the meantime is safety only, and refuses nothing: (1) `auto` prints a LOUD warning at selection time, before the expensive stages, carrying the measured hazard verbatim (`gw.downfold.AUTO_HAZARD`), the numbers above, the over-complete-parent precondition and the fact that the accuracy-sized mode does not exist; it repeats at the end of the run, where the reader actually is.  (2) `docs/downfold.md`, `docs/drivers.md` and `docs/input_reference.md` **stop recommending `auto`** — the recommendation is now an explicit integer validated by the observable comparison, the page's own example deck carries an explicit number rather than `auto`, and the over-complete-parent requirement is restated at the recommendation site.  (3) Every run now ends by printing the one-line, copy-pasteable observable comparison (parent and small bundle, same deck, same flags, compare the lowest eigenvalue), with that run's own paths substituted in, because the three numbers the driver prints are all rank or projection statements and none of them is an accuracy gate.  Gate: `tests/test_downfold.py::test_auto_prints_the_loud_accuracy_warning` with its **red twin** `::test_auto_warning_red_twin_an_explicit_mu_small_says_nothing` — the same rank-deficient pool, μ_S spelled as an integer instead of `auto`, must print nothing at all, so the warning cannot decay into a banner that fires on every run.  32 cells in that file, **32 passed** on WSL CPU.  Evidence: `PIPELINE_HEALTH.md` (the walk and its punch table), `DOWNFOLD_S1.md` §3(c) and `DOWNFOLD_RANK_PROBE.md` (campaign reports, not carried in this repository) |

---

# AMENDMENT — `bse.exciton_bands` COULD NOT READ A DOWNFOLDED BUNDLE, THREE WAYS (2026-08-10) — **FOUND AND FIXED ON THIS BRANCH**

**One row, and not one of the three was ever a red.**  `PIPELINE_HEALTH.md`
step 5 records the whole defect: on a 936 → 189 downfold of the shipped
`si_bse_debug` parent, `bse.exciton_bands` refused three separate ways while
`bse.bse_jax` read the same bundle fine.  Nothing in the suite could have
caught any of them — the downfold's own cells are deck-free algebra, and the
one exciton-bands driver cell is gated on an MoS2 fixture that is not a
downfold — so the whole defect lived in the gap between two suites that each
looked complete.  It broke the drop-in promise for exactly the driver the
compression exists to serve.

| item | mechanism, at this tree | disposition |
|---|---|---|
| **`bse.exciton_bands` refused a downfolded bundle three ways: no centroid table, a Galerkin basis that cannot span `nk·nb`, and no `zeta_q.h5`** | The three share one cause. `bse_jax` only READS the stored tensors; `exciton_bands` REBUILDS objects in the same ISDF basis — ψ at finite Q via htransform, and the exchange tile off the grid via `vq_interp` — and rebuilding needs two things the bundle format does not carry and never did. (1) COORDINATES: the htransform leg fits ψ against a centroid table, and a downfolded bundle carried one only when the run had been given `parent_centroids_file`; without it `file_io/centroids.py` raised a bare `FileNotFoundError` on a path the deck named and the child directory did not contain, with nothing to say it was a downfold consequence. (2) THE BASIS THAT FIT RUNS IN: that Galerkin fit needs a basis spanning `nk·nb` — 1280 on this deck — which is a completely different sizing criterion from the retained window's PAIR-DENSITY rank that μ_S = 189 was chosen against, so the fit in the small basis failed `build_fH_R`'s orthonormality gate outright. (3) ζ: `--vq-mode interp` interpolates a stored `zeta_q.h5` beside the restart, and the downfold wrote none; the parent's is the wrong basis, μ_L wide against a μ_S bundle, so copying it would have been worse than leaving it out | **FIXED on `fix/exciton-bands-downfold-2026-08-10`, PUSHED, NOT MERGED.**  Two of the three are completed on the WRITER, which is where the design says the restart contract lives. (1) `gw.downfold_run.resolve_parent_centroids` takes the parent's coordinates from the parent's OWN `zeta_q.h5` `isdf_header` — `r_mu_crystal` is that table by construction — content-verifies them against the bundle's `centroids_charge_md5` (which hashes the int64 FFT-INDEX table, not the text file; comparing a file md5 against it never matches, and `_centroid_table_md5` exists so that mistake is unavailable), and writes them beside the small bundle. `parent_centroids_file` becomes an override rather than a requirement. (3) `write_downfolded_zeta` transports ζ as `ζ_S = conj(T[q]) ζ_L` — `transform_head_vector`'s map at every G rather than only at G = 0 — which substituted into `V = ζ† v ζ` gives back `V_S = T V_L T†`, the congruence the bundle already stores, so the two descriptions cannot drift; the writer cross-checks its q=0/G=0 column against the independently transported `g0_S` on every run (**AGREE, 1.4e-15**, real deck). On a parent whose ζ is q-IBZ-only the transport is refused and says so — nothing is lost, since `vq_interp` refuses an IBZ ζ on the parent too. (2) is necessarily reader-side, because the reader is what runs the fit: `exciton_bands.resolve_isdf_basis` reads the bundle's `downfold_provenance`, fits the htransform leg in the PARENT basis, and `build_conduction_stacks` slices the result to `keep_idx` inside the same jit — exact, because that column slice is the downfold's own definition of the small basis's ψ (`downfold.md`, `mode = cur`). The restart's μ is the authority on the result and a mismatch refuses rather than padding its way out (post-snap window-authority pattern, `7449ece0`). **GATES, all on Perlmutter, shared pool.** (a) END-TO-END on the walk's exact recipe (GW → downfold → `exciton_bands`), **rc=0**, and the Q=0 point of the downfolded exciton bands is `0.257866 0.258312 0.261277 0.597959` eV against `bse_jax --tda --bse` on the SAME bundle at `0.25786551 0.25831244 0.26127694 0.59795887` — agreeing to **5e-7 eV**, i.e. identical at the `.dat`'s own print precision. Same Hamiltonian, same answer. (b) SPLASH PROOF: the parent-basis run's `.dat` AND `.png` are **byte-identical** between base `d1e375ad` and this branch. (c) IDENTITY DOWNFOLD (μ_S = μ_L = 936, T = I) through `exciton_bands` reproduces the parent driver on **all 10 Q rows × 4 eigenvalues**, identical to print precision; the same holds for a synthetic identity `keep_idx` stamped on the parent bundle itself. ζ's transport is verified independently of any gate battery: the transfer recovered from the two on-disk ζ files ALONE (`conj(T) = ζ_S pinv(ζ_L)`, no knowledge of the downfold's `T`) congruences the parent's stored `V_qmunu` into the child's to **5.1e-11 relative at every one of 64 q**. (d) `tests/test_exciton_bands_downfold_dropin.py`, 16 cells, one per mechanism plus a red twin each — **46 passed** on Perlmutter with `tests/test_downfold.py` beside it, 6 of them also green on a WSL CPU box (the other 10 import the driver and need the FFI). (e) **DEFAULT FAST GATE: ZERO DELTA.** 16 unique FAILED node ids on the branch and the IDENTICAL 16 at base `d1e375ad`, `diff` empty; **none of them is under `tests/`** — all sixteen are the pre-existing `distrib_la` / `vcoul` / `wfn_loader` / `zeta_loader` / `symmetry_maps` service reds this file already lists. Note the selection is NOT symmetric and cannot be: the branch touches `src/file_io/` and `tests/`, which map to ALL, so its gate runs all five end-to-end driver cells while base (no diff against main) runs only the services. Every driver cell the branch's gate selects is green. A FIRST branch run showed `test_bse_matches_frozen_and_bgw` red; that is xdist contention on a shared node, not this branch — the cell passes ALONE on the branch AND alone at base, and the repeat full-gate run at the same head is green on it. `mkdocs build --strict`: 25 warnings, the documented baseline, unchanged. **WHAT THIS DOES NOT FIX, stated because the number is alarming and belongs to a different row:** the downfolded exciton bands at μ_S = 189 sit 2.09 eV below the parent's. That is `PIPELINE_HEALTH.md` punch row 3, the open `mu_small` sizing guard, not this defect — measured here as a μ_S sweep, 0.2579 eV at 189 → 1.2605 eV at 624 → 2.3451 eV at 936, which is the SAME curve `bse_jax` traces on the same three bundles, to six figures. The driver now reports the bundle it was given, faithfully; sizing the bundle is still the user's problem. **One pre-existing failure was found and is NOT this branch's:** `--vq-mode interp` on this deck fails `vq_interp`'s `makeVq_vs_disk_Vqmunu_allq_max` and `slab_axes_offdiag` gates — **identically on the PARENT** (3.218e-01 / 1.000e+00) and on the child (4.799e-01 / 1.000e+00) — so the off-grid path is blocked on `si_bse_debug` for reasons that predate the downfold, and the end-to-end gate above runs `--vq-mode ongrid`, which is exact at every Q on the BSE grid. Evidence: `/pscratch/sd/j/jackm/xw_parent`, `xw_small`, `xw_ident`, `xd_parent`, `xd_small`, `xd_small8` |

---

# AMENDMENT — THE SHARDED BSE LOADER INJECTED A SCREENED HEAD ONTO THE BARE-V FALLBACK (2026-08-10) — **FOUND AND FIXED ON THIS BRANCH**

**One row, and it was never a red — it was a silent wrong number on a path
that already knew it was in trouble.**  The Schur design pass over the BSE's
q=0 head (`~/lorrax_bse_perf_2026-08-08/SCHUR_BSE_DESIGN.md` §5(a)) read the
two restart loaders side by side and found that only one of them asks whether
the `W` it is about to decorate is screening at all.  Nothing in the suite
could have caught it: every behavioural cell that exercises
`load_bse_data_from_restart_sharded` runs on a fixture whose `W0_qmunu` is
ready, which is the one state in which the missing gate is invisible.

| item | mechanism, at this tree | disposition |
|---|---|---|
| **`load_bse_data_from_restart_sharded` added the screened head `whead` to a `W` tile that is bare Coulomb `V`** | Both loaders fall back to bare `V` for `W` when the restart carries no ready `W0_qmunu` — the April all-zero-screening gate, `W0_ready`, whose reading side `tests/test_bse_w0_ready_gate.py` pins.  Both then re-attach the q=0 Coulomb head that `compute_vcoul` removes before the Dyson solve, as the rank-one update `(head/V_cell)·conj(g0)⊗g0`.  `vhead` belongs on the exchange tile either way, because that tile is bare Coulomb by construction; `whead` is the head of the SCREENED interaction and belongs on `W` only when a screened `W` was loaded.  The single-device reader `_load_ring_subset` asked exactly that (`w0_ready = W0_qmunu is not None`, and it passed `None` for `W` when the answer was no).  The sharded reader — the P>1 path, and the one production runs at scale — passed `W_q` into the injector unconditionally, so a fallback run got a screened head of the same geometric size as the Si deck's 2600 meV and the MoS2 slab's 3564 meV sitting on an unscreened tile.  The fallback itself warns loudly; the head on top of it was silent, and it is not a smaller error than the fallback it rides on but a **different** one, on the q=0 tile exciton binding is most sensitive to.  Not attributable to a commit: the sharded loader has carried the ungated call since it was written, and the single-device gate was added to its twin without being ported | **FIXED on `fix/sharded-whead-gate-2026-08-10`, PUSHED, NOT MERGED.**  Gate parity, in one authoritative spelling: the new `bse_io._inject_q0_head` owns both the injection and its `w0_ready` condition, and both loaders now call it — so the two cannot drift apart again, and an AST cell refuses either of them reaching past it to `apply_q0_head_rank1_sharded`.  The sharded side records `w0_ready = wq_key is not None` at the dataset-selection block, before the fallback aliases that key to `V_qmunu` and destroys the evidence.  Gate: `tests/test_sharded_whead_gate.py`, 11 cells, **11 passed** on WSL CPU — behavioural, fixture-free, one CPU worker subprocess per device count (`--xla_force_host_platform_device_count`), running the real loader on a synthetic restart in both `W0_ready` states at 1x1 and 2x2.  **Red twin, executed at P=4 through the shipped loader** with the gate monkeypatched back open: the returned q=0 tile is not the bare `V` on disk and is bit-for-bit `V` plus the rank-one head, which is the defect reproduced rather than described.  Post-fix the same arm returns the disk bytes **exactly**, and matches what `_load_ring_subset` composes on the same file, at both meshes.  **Blast radius, measured rather than argued:** of the 44 arrays in the returned bundle, hashed pre-fix and post-fix at 1x1 and at 2x2, **exactly one moves** — `W_q` on the fallback arm.  Every array on the ready arm, and `V_q0` on both arms, is bit-identical, so `vhead` still reaches the exchange tile on a fallback run and the normal path does not move at any mesh.  The only other visible change is the log, which now says `whead=skipped` where it used to print a value — the single-device loader's own spelling.  **Default gate, run on both arms of the same box** (WSL, no site `.so` pair, so this is the header's `LORRAX_FFI_SO`-absent negative control rather than a physics leg): 935 collected, **8 failed / 787 passed / 3 errors on the branch and the IDENTICAL eleven node ids at base `e266fbce`** — the six `distrib_la` contract cells, the `vcoul` import-isolation cell, the Si BSE anchor and the three `gw_jax` regression errors, all of which need the built FFI.  Nothing this branch touches appears in that set.  Evidence: `SCHUR_BSE_DESIGN.md` §5(a), which also records §5(b) (bind the Coulomb policy stamp to the head scalars) as still open |

---

# AMENDMENT — THE SI BSE ANCHOR WAS RED ON `main`; THE ANCHOR DECK NOW PINS ITS WINDOW (2026-08-09, **CLOSED 2026-08-10** on `chore/anchor-window-pin-2026-08-09`)

**One row, and it was a live red on `main` rather than a branch's own red.**
The fast-gate lane reported `test_bse_matches_frozen_and_bgw` failing at
0.0906 eV against a 1e-6 eV pin and correctly declined to attribute it, since
its branch changes no `src/` file; it flagged the number as worth a look on a
properly site-built tree.  It was looked at.  The red reproduced to the digit
on a clean checkout of `main` @ `01e1e609` with the canonical
`restage_candidate_2026-08-08` `.so` pair, so it was neither the fast-gate
branch nor that lane's borrowed 2026-08-07 libraries.  It arrived at
`824032b7`, the `feat/exciton-bands-2026-08-09` merge, and the mechanism is
that merge's new band-window guard running in its default `snap` mode.

The part that mattered most is that **this red was not re-cuttable.**  The hbn
row two amendments down is the shape people will reach for — correct behaviour
outrunning a stale reference, closed by an owner-authorized re-freeze.  This was
not that.  The moved spectrum fails the gate's *external* BerkeleyGW band as
well as its frozen pin, so adopting the new numbers as the reference would have
left the cell red on its second arm.  The gate is the tree's only cross-code
BSE anchor, and for the eighteen hours between `824032b7` and the pin below it
was anchored to nothing.

**HOW IT IS CLOSED.**  The anchor deck's invocation now pins
`--band-degeneracy off`, and the pin carries its reason at the call site.  The
owner's standing rule decides this one: a reference deck is a measurement
standard, so it has to run the calculation its references were cut from.
BerkeleyGW produced `bgw_eigenvalues_dft_ref.dat` at 4v4c and the frozen
`bse_eigenvalues_ref.dat` is that same 1024-dimension problem, so a guard that
silently rewrites the window to 8c is not making the gate safer — it is
substituting a different calculation for the one being measured, which is
exactly the confounding `97735e01` refused for the q=0 head override on this
very deck.  Nothing in `src/` changed; the whole fix is one flag and one
comment in `tests/test_bse_bgw_regression.py`.  **Measured green** at
`29723072` on the shared pool with the same conditions the attribution used
(`LX_BASE_MODULE=lorrax_J070`, the canonical `.so` pair,
`LORRAX_CHECKOUT=/pscratch/sd/j/jackm/anchorpin_0810/wt`): the cell passes both
arms with **zero `[band-window]` lines in the log**, which is the same
signature the A/B's `log_degen_off.txt` showed, and the other four default-gate
cells pass unchanged in the same tree
(`/pscratch/sd/j/jackm/anchorpin_0810/log_anchor_pin.txt`,
`log_other4_pin.txt`).

**THE OTHER REFERENCE DECKS WERE SWEPT FOR THE SAME EXPOSURE, AND NONE OF THEM
HAS IT.**  The exposure needs two things at once — a cell that reaches the
guard, and a requested window whose boundary cuts a multiplet — and the sweep
checked both rather than assuming either.  On reachability, the guard's only
call sites in the tree are `bse_io.resolve_band_window` (twice) and
`bse_io.check_band_window` / `exciton_bands.check_band_window`, all of them
inside `src/bse/`; `gw.gw_jax` does not import `bse_io`, so the four COHSEX and
Σ decks whose frozen references the suite pins — `si_cohsex_debug` (all three
cells, including the BerkeleyGW anchor), `cohsex_debug`, `hbn_cohsex_debug` and
`bispinor_debug` — cannot reach it at all, whatever their spectra look like.
On the one other deck that does reach it, the answer is measured rather than
argued: `gnppm_debug` is loaded at 2v2c by `conftest.bse_dense_state`,
`test_bse_kgrid` and `test_bse_nontda_restart_preflight`, and running the
guard's own `boundary_min_gaps` over that deck's `WFN.h5` puts its valence
boundary (band 24) at **305.171 meV** and its conduction boundary (band 28) at
**75.758 meV**, both five decades clear of the 1 meV tolerance — so `snap` runs
there on every invocation and has never fired.  For contrast the same script
reproduces the anchor's defect exactly: `si_bse_debug` at 4v4c is clean at the
valence boundary (band 4, 47.961 meV) and **0.000 meV at the conduction
boundary, band 12**.  `si_bse_debug` is therefore the only deck pinned, and the
rest are left untouched deliberately.  Sweep evidence:
`/pscratch/sd/j/jackm/anchorpin_0810/sweep_windows.py` and its output
`log_deck_sweep.txt`, which reads each fixture's `mf_header/kpoints/el` and
`ifmax` and calls the shipped `boundary_min_gaps` rather than reimplementing it.

**THE SECOND OWNER ROW IS NOW CLOSED TOO.**  It asked whether `snap` was the
right *default* for a driver flag at all, or whether `strict` was, given that
the guard's whole argument is that a cut multiplet is not a thing to fix
quietly.  Pinning the anchor deck removed the tree's one measured victim of the
default but did not answer the question — the next deck someone wrote with a
boundary inside a multiplet would have hit the same silent doubling without a
frozen reference to notice it with.  **DECISION (2) TAKEN, 2026-08-10 (owner,
verbatim: "do strict"): the default is `strict`.**  A window that cuts a
degenerate multiplet now REFUSES, naming the counts that would work, and never
silently widens the calculation.  `snap` remains available as an explicit
opt-in and `off` is unchanged.  The reason on record is this row and the parity
deck's false 28.6 meV "regression": `snap` silently re-windowed two BGW-parity
decks in one day.  The flip is one name — `common.band_degeneracy.DEFAULT_MODE`
— read by both drivers' `--band-degeneracy` and by every keyword default at the
choke point, with `tests/test_band_degeneracy.py` pinning both the value and
the absence of any second literal that could shadow it.  The anchor deck's own
`--band-degeneracy off` pin from DECISION (1) is untouched and still the right
pin: `off` is what a measurement standard wants, because it runs the window its
references were cut from without the guard having an opinion either way.

**MEASURED at `912f6c79`** on the shared pool under the conditions the
attribution and anchor-pin lanes used (`LX_BASE_MODULE=lorrax_J070`, the
canonical `.so` pair, `LORRAX_CHECKOUT`), evidence in
`/pscratch/sd/j/jackm/degenstrict_0810/`:

* **the deck now refuses with no flag** (`log_probe_ab.txt`, `probe_out.txt`) —
  `si_bse_debug` at 4v4c exits **rc=1** with
  `BandWindowDegeneracyError: … the conduction boundary at band 12 cuts a
  multiplet (gap 0.000 meV at k=0, tol 1.000 meV) … Fix: use --n-val 4
  --n-cond 8, or pass --band-degeneracy snap to widen to those counts
  automatically, or --band-degeneracy off …`;
* **the opt-in is byte-identical to the pre-flip default.**  Explicit
  `--band-degeneracy snap` on the same deck snaps 4c → 8c, runs to rc=0, and
  returns the same twenty eigenvalues the pre-flip `snap` arm returned at
  `01e1e609` — `2.34696443 ×2, 2.34802375 ×3, 2.34977621 ×3, …, 2.40415288`,
  max abs vs the frozen pin **0.09064299 eV**, matching
  `bsegate_attrib_0809/log_degen_snap.txt` digit for digit;
* **the anchor deck is unchanged** (`log_c_anchor.txt`) — the `off`-pinned cell
  passes both arms with **zero `[band-window]` lines in the log**, the same
  signature the pin was measured with at `29723072`;
* **the default fast gate is set-identical to `main`** (`log_d_fastgate.txt`
  vs `log_d_baseline_main.txt` at `a65a5326`): the same nine `services/` reds
  on both sides and nothing else, so the flip adds no red; the guard's own
  census cells pass (`log_e_census.txt`).

| item | mechanism, at this tree | disposition |
|---|---|---|
| **`tests/test_bse_bgw_regression.py::test_bse_matches_frozen_and_bgw` — was RED on `main`, both arms; **FIXED** by pinning the deck's window at `29723072`** | **The default `--band-degeneracy snap` silently doubles the deck's BSE problem.**  `99d73f95` (in the `824032b7` merge) installs `common.band_degeneracy.resolve_band_window` at the band-window choke point with `mode="snap"` as the default, so a window boundary landing inside a degenerate multiplet is widened OUTWARD.  On `si_bse_debug` it fires, and the run says so in its own words: *"[band-window] `_load_ring_subset`: requested n_val=4 n_cond=4 (bands [4, 12) of 60) is not multiplet-safe … the conduction boundary at band 12 cuts a multiplet (gap 0.000 meV at k=0, tol 1.000 meV); the multiplet ends at band 16, so n_cond 4 → 8 … SNAPPED OUTWARD to n_val=4 n_cond=8 (bands [4, 16) of 60).  The BSE problem is now 32 pairs per k instead of 16."*  The gate's Hamiltonian therefore goes from 1024 to 2048 dimensions, the extra transitions fill the bottom of the spectrum, and every one of the lowest twenty moves down: 20/20 cells over the pin, max abs **0.09064299 eV** against `ATOL_FROZEN_EV = 1e-6` (index 0: 2.34696443 actual vs 2.35372258 frozen).  The signature is diagnostic on its own — the returned list acquires multiplicities the frozen list does not have (2.346964 ×2, 2.348024 ×3, 2.349776 ×3) and the four BerkeleyGW states at 2.470–2.484 eV are pushed out of the lowest-20 window entirely | **NOT ATTRIBUTABLE TO ANY BRANCH — it is `main`'s.**  **Attributed to `824032b7`** by walking the day's mainline with the `.so` pair, the deck and the test file all held fixed and only `src/` varying (`/pscratch/sd/j/jackm/bsegate_attrib_0809/`, `probe.sh` + per-commit `log_<sha>.txt`): GREEN at `e37c6a6e`, `7ac49f5e` and `4bcc5b40` (`= 824032b7^1`), RED with byte-identical numbers at `824032b7`, `9024deee`, `2ad51ef9`, `01e1e609` and — re-checked after `origin/main` moved during this lane — `d9d418db`.  **Mechanism proven by A/B on the cell itself** at `01e1e609`, the knob the only variable: `--band-degeneracy off` → **PASSES both arms**, zero guard lines in the log; `--band-degeneracy snap` → the failure above.  So with the window the deck asks for, current `main` still reproduces the frozen pin to better than 1e-6 eV — nothing else in `src/` has drifted.  **The re-cut trap, measured**: against the external reference, the snapped spectrum sits at MAE **21.729 meV** / max **80.803 meV** versus bands of 10 / 25, where the frozen reference itself sits at 6.522 / 9.840 — and the `_assert_aligned` guard does *not* divert it (best-drop ratio 0.913 against a 0.5 threshold), so the BerkeleyGW assert is genuinely reached and genuinely fails.  Freezing the new numbers would trade a red frozen arm for a red band arm.  **DECISION (1) TAKEN, 2026-08-10: the anchor deck pins `--band-degeneracy off`** — BerkeleyGW produced `bgw_eigenvalues_dft_ref.dat` at 4v4c and a silently widened window compares two different problems, which is the same confounding that `97735e01` refused for the q=0 head override on this very deck.  **DECISION (2) TAKEN, 2026-08-10** (owner, verbatim: "do strict"): **`strict` is the default** — a window that cuts a degenerate multiplet refuses with an actionable message naming the counts that would work, and never silently widens the calculation; `snap` stays as an explicit opt-in, `off` unchanged.  The question it closes is the one this row opened, and the evidence for it is this row plus the parity deck's false 28.6 meV "regression": `snap` silently re-windowed two BGW-parity decks in one day.  Landed as one name, `common.band_degeneracy.DEFAULT_MODE`, read by both drivers' `--band-degeneracy` and by every choke-point keyword default.  The `--band-degeneracy off` pin on this deck is unaffected and stays.  **The landing's neutrality control could not have caught this**: `EXCITON_BANDS_FEATURES.md` §1.6 compares base @ `f1e07bb6` at `--n-val 4 --n-cond 4` against the branch snapping *into* 4v4c on the MoS₂ deck and gets bit-identity, which proves the guard changes only *which* window is selected — true, and blind by construction to a deck where the selected window changes.  The five default-gate e2e cells were not run on that branch.  Evidence: `/pscratch/sd/j/jackm/bsegate_attrib_0809/` (`probe.sh`, `mech.sh`, `log_<sha>.txt`, `log_degen_off.txt` / `log_degen_snap.txt`, `dump_off.txt` / `dump_snap.txt`) |

**The other four default-gate cells are GREEN on clean `main` @ `01e1e609`**,
measured in the same lane and the same pool: the three Si COHSEX cells
(`test_si_fast_matches_frozen_reference`,
`test_si_production_matches_frozen_reference`, and the BerkeleyGW anchor
`test_si_production_matches_berkeleygw`) and the dipole sweep smoke
(`test_the_default_analytic_sweep_writes_a_valid_dipole_h5`) all pass —
`/pscratch/sd/j/jackm/bsegate_attrib_0809/log_other4.txt`.  So one of the five
was red, it was this one, and the fast gate's own accounting is otherwise
sound.  Those four were re-measured at `29723072` alongside the pinned cell and
are still green (`/pscratch/sd/j/jackm/anchorpin_0810/log_other4_pin.txt`), so
**all five default-gate cells are now green** and this row leaves no live red
behind it.

---

# AMENDMENT — THE DIPOLE PRODUCER HAD NO PSEUDOPOTENTIAL PRE-FLIGHT (2026-08-09)

**One row, and it is the FIXED successor to the `dZ is None` row the sign-flip
lane opened on `fix/dipole-vnl-sign-2026-08-09` (`3cfcafe2`) — not a second
row for the same defect.**  That lane found the failure by measurement and
handed it over correctly; what changed on investigation is the diagnosis, and
three claims in the original row come out FALSE and are corrected below.  The
short version is that nothing regressed and nothing about tracing is at fault:
`psp.get_dipole_mtxels` is the only one of the tree's three operator drivers
that never called the shared pseudopotential pre-flight, so a deck directory
with no `*.upf` in it reached the sweep instead of being refused, and three of
the four regression decks do not carry their UPFs.

The reason this matters beyond one traceback is that the loud arm was the safe
one.  A missing pseudopotential set crashed the default analytic sweep, but it
made `--vnl-mode numeric` finite-difference an empty projector set and write a
`dipole.h5` with V_NL identically zero, stamped `prov_skip_vnl=False`.  On
si_cohsex_debug that artifact agrees with the `--skip-vnl` artifact to 5.8e-15:
it *is* the p̂-only run wearing the other arm's provenance.

| item | mechanism, at this tree | disposition |
|---|---|---|
| **`psp.get_dipole_mtxels` ran no pseudopotential pre-flight, so a deck with no `*.upf` died inside the sweep at `kdata.dZ is None`** | `psp.operator_checks.validate_operator_inputs` exists for exactly this, and its own module docstring names this caller — *"before computing kin+ion, DIPOLE matrix elements, or any other quantity that depends on pseudopotentials"*.  `gw.kin_ion_io` calls it; `psp.get_DFT_mtxels` calls it; the dipole driver never has, and it is also the only one of the three with no `--pseudo_dir` flag.  With no pseudos, `build_vnl_setup` returns `channels=[]`, `_build_vnl_kdata_core` has no block to concatenate and returns `dZ=None`, and ~30 s later `apply_vnl_velocity_to_ket` conjugates it: *TypeError: conjugate requires ndarray or scalar arguments, got NoneType*, six frames inside a jitted einsum that names neither the deck nor the missing file.  **Deck-dependence explained**: `cohsex_debug` is the only regression deck that commits its UPFs (`Mo_ONCV_PBE_FR-1.0.upf`, `S_ONCV_PBE_FR-1.1.upf`, added with the fixture in `906dd31b`), which is exactly the deck that kept working; si_cohsex_debug / gnppm_debug / hbn_cohsex_debug have never carried theirs, and the cluster log says so in as many words — *"No pseudopotentials (\*.upf) found in …, …/../qe/scf, …/../qe/nscf; Pseudopotentials: none found"*.  **NO COMMIT BROKE THIS.**  `git log -S validate_operator_inputs -- src/psp/get_dipole_mtxels.py` is empty over the whole history: the wiring was never there, the committed fixtures were cut in staging directories that happened to hold the UPFs (hbn's provenance still names one, `/pscratch/…/hbn_fixture_prep`), and from a clean checkout the regeneration path has never worked | **FIXED on `fix/dz-none-dipole-2026-08-09`, PUSHED, NOT MERGED.**  Two levels: the driver now runs `validate_operator_inputs` unless `--skip-vnl` — the one arm entitled to hold no projectors — and gains `--pseudo-dir`, which is what actually unblocks the four `dipole.h5` re-cuts; and `_build_vnl_kdata_core` with `compute_dZ=True` now returns a `(3, total_R, nG)` zero array instead of `None`, so the kernel's documented contract holds for every caller (`Z` already degraded this way as the `(0, nG)` empty projector matrix).  Gate: `tests/test_dipole_regeneration_gate.py`, 8 cells, **8 passed on Perlmutter**; FALSE arm against the pre-fix sources on WSL CPU is **5 failed, 1 passed, 2 skipped**, the single pass being the live-channel control that must be green on both arms.  Cluster legs, `/pscratch/sd/j/jackm/dznone_0809/`: the sign-flip lane's exact command on si_cohsex_debug gives **rc=1 with the TypeError at clean `origin/main`**, **rc=1 with a named refusal post-fix**, and **rc=0 writing a valid `dipole.h5` post-fix with `--pseudo-dir`** — the first time the default analytic sweep has produced one on that deck.  V_NL demonstrably enters it: max|analytic − skip-vnl| is **12.3 % relative**, against 5.8e-15 for the numeric arm before the fix.  **THREE CORRECTIONS to the row this supersedes** (`3cfcafe2`): *(1)* it is **not** a traced-path defect — the eager `build_vnl_kdata_from_kvec` returns `dZ=None` identically on an empty channel set, and the gate carries both cells to say so; *(2)* `--vnl-mode numeric` did not "succeed with V_NL", it succeeded with V_NL **identically zero**, which makes it the more dangerous arm, not the working one; *(3)* "no test catches it because every in-tree fixture builds a non-empty `channels`" is false — `tests/test_psp_padded_gvectors.py` has built a `channels=[]` setup all along and simply never asked it for `dZ`.  Evidence: `~/lorrax_bse_perf_2026-08-08/FIX_dz_none_dipole.md` |

---

# AMENDMENT — `exciton_bands` AT P=4 NEVER EXITS (2026-08-09) — **DIAGNOSED AND FIXED ON THIS BRANCH**

**One row, from the owed-legs batch (`OWED_LEGS_BATCH.md`), and it explains an
entire evening of phantom "pool contention."**  The exit-path lane took it,
reproduced it twice on the base tree, captured per-rank stacks, and closed it;
the row below now records what it actually was.  The suspect the batch named —
`jax.distributed` / coordination-service teardown — was **wrong**, and so was
the "XLA:CPU pool shutdown" deadlock `runtime.finalize_process` was written
for: nothing spins in a worker pool here, only the four MAIN threads do.

| item | mechanism, at this tree | disposition |
|---|---|---|
| **`bse.exciton_bands` at P=4 completes its payload and then never exits** | **A COLLECTIVE RUN FROM `__del__`.**  `htransform.initialize_wfns` returns a MESH-AWARE `WfnLoader` (`setup_wfn_and_sym` → `WfnLoader(wfn_file, mesh=mesh_xy)`), so at P>1 it takes the phdf5 backend and owns a `SlabIO` whose `close()` runs an **unconditional collective barrier** (`file_io/_slab_io_ffi.py`, `_barrier("slab_io_ffi_close_attrs")`).  The driver never closed it, so the collective fell to `WfnLoader.__del__` — i.e. it fired whenever the garbage collector happened to drop the object during interpreter shutdown, a moment no two ranks agree on.  Captured live at the hang (JID 56550230 steps `.9`/`.10`): **three ranks** in `exciton_bands.py:<module>` → `loader.py:318 __del__` → `loader.py:306 close` → `slab_io.py:116 close` → `_slab_io_ffi.py:2385 close` → `collectives.py:172 barrier` → `multihost_utils.py:83 broadcast_one_to_all`, spinning in `cuStreamSynchronize` on a NCCL collective; **the fourth rank** already past that and in `ffi/io.py:207 _atexit_close_all` → `ffi_loader.py:1119 phdf5_close` → `H5Fclose` → `H5FD__mpio_close` → `PMPI_File_close` → `MPI_Barrier` → **`sched_yield`**, which is the 100 % CPU.  Two disjoint collective domains, neither ever satisfied.  Teardown/exit path, not physics: outputs are complete and correct.  **Pre-existing**: reproduced at base `f1e07bb6`, so it is not the 2026-08-09 feature branch; two of the three wild instances had no `LORRAX_JAX_CACHE_EXPLAIN`, so it is not the cache-explain path either | **FIXED on `fix/exciton-exit-hang-2026-08-09`** (`b3813d8f`).  Two lines at the driver's tail: an explicit `wfn.close()` **after** the existing outputs barrier, so the `SlabIO` collective runs at a point every rank reaches in lockstep and `__del__` becomes a no-op; and `runtime.finalize_process(main())` in place of the bare `SystemExit`, so the SECOND unordered collective (`_atexit_close_all`'s `H5Fclose` on the restart and zeta contexts) runs in one stated order on every rank and GC-driven `__del__`s at shutdown never run at all — which is what covers the `--refit-points` path, where `vq_interp.refit_prepare` builds a second mesh-aware loader nothing closes.  Gates, all on `f1e07bb6`: P=4 fixed **exits rc=0 in 31 s**; P=4 unfixed **times out at 240 s** (red twin); P=1 rc=0 on both trees; all six `.dat` outputs bit-identical (md5 `b84fc42e7c2931b7ace0801e764e5b4d`).  **Sibling verdict**: only two production sites build a mesh-aware `WfnLoader` — `htransform.setup_wfn_and_sym` and `gw/gw_jax.py:269`.  `gw.gw_jax` has never hung because it already ends in `finalize_process` → `os._exit`.  `bse.bse_jax` has never hung because its loader path (`bse_io.load_bse_data_from_restart_sharded`) builds `WfnLoader(wfn_path)` with **no mesh** → eager backend → no `SlabIO` → a collective-free `close()`.  **STILL EXPOSED, NOT FIXED HERE**: `bandstructure.htransform`'s own `main()` (it calls `initialize_wfns` at line 1800 and ends `raise SystemExit(main())`) has the identical shape and should hang the same way at P>1.  Evidence: `~/lorrax_bse_perf_2026-08-08/FIX_exciton_exit_hang.md`, `OWED_LEGS_BATCH.md` §3 |

---

# AMENDMENT — THE TWO ABSORPTION DRIVERS DISAGREE ON A CONJUGATION (2026-08-09)

**One row, from the oscillator-strength gate build (`FIX_dipole_vnl_sign.md`).**
This is the fifth member of the campaign's most expensive defect class
(crossed conventions), found because the new gate pins two quantities
precisely so that conjugation blindness in `|d|²` cannot hide it.

| item | mechanism, at this tree | disposition |
|---|---|---|
| **`absorption_eigvecs` contracts `A` without a conjugate; `davidson_absorption:214` uses `.conj()`** | The two absorption drivers assemble oscillator strengths from the eigenvector contraction with OPPOSITE conjugation conventions — genuinely different numbers on the same eigenvectors, not a representation difference.  Which one is right is NOT adjudicated yet; the new gate `tests/test_bse_oscillator_strengths.py` (on the `fix/dipole-vnl-sign-2026-08-09` branch) pins BOTH so the disagreement is measured rather than silent | **FIXED on `fix/absorption-conjugation-2026-08-09`.**  **`absorption_eigvecs` was the wrong driver.**  The correct contraction is `⟨0|r̂_α|S⟩ = Σ_t A^S_t · conj(d^α_t)`: the exciton state is `|S⟩ = Σ_t A^S_t â†_c â_v|0⟩`, so `⟨0|r̂|S⟩` picks up `⟨vk|r̂|ck⟩ = conj(d_t)` where `d_t = ⟨ck|v̂|vk⟩/ΔE` is what `slice_dipole_to_bse_window` returns.  That `A` really is the amplitude in that basis (and not its conjugate) is read off LORRAX's own kernel, not assumed: the exchange term is assembled as `K^x = M V M†` with `M_t = conj(ψ_c)ψ_v` (`bse_simple.py`) and the direct term as `K^d_{tt'} = −Σ conj(ψ_c[k])ψ_c'[k'] W ψ_v[k] conj(ψ_v'[k'])` (`bse_nontda.py:169`), both the standard `⟨t|H|t'⟩` with the conduction index on the bra.  Three independent witnesses agree — BerkeleyGW's `BSE/diag.f90:711` (`Σ u_r·MYCONJG(s1)`, with `s1 = ⟨ck|…|vk⟩/ΔE` from `Common/mtxel_optical.f90`), the convention-free resolvent identity `⟨d|(z−H)⁻¹|d⟩ = Σ_S |⟨S|d⟩|²/(z−E_S)`, and the Haydock route which evaluates that resolvent and needed no change.  **`davidson_absorption` had the right MODULUS** (`Σ conj(A) d` is the complex conjugate of the correct expression, and ε₂ reads only `|·|²`), so its spectra do not move; only the sign of the `Im` column in its BGW-format `eigenvalues_b*.dat` moves, onto BGW's spelling.  Both drivers now route through the one site `absorption_common.exciton_dipole_projections`, which carries the derivation; an AST cell refuses `davidson_absorption` growing a private einsum again.  Crossed red twin: `tests/test_absorption_conjugation.py` runs the FALSE arm (`Σ A d`) against the resolvent identity and shows it fails by ~10⁰, and the mutation check confirms three cells go red when the shared site is flipped.  **Pins that moved, and only these:** the twelve `proj` numbers per fixture in `tests/test_bse_oscillator_strengths.py` were cut with the wrong driver's spelling and are re-cut (movement up to 6.8× per element); `sum_f`, `max_f` and `de_bounds_Ry` are bit-unchanged, and no `dipole.h5` was regenerated.  **Adjacent gap closed in the same pass:** `load_eigenvectors_h5` silently returned the resonant `X` alone from a non-TDA file, giving the TDA answer for a full-BSE solve; it now refuses such a file and names the `(X, Y)` contraction it would need.  Evidence: `~/lorrax_bse_perf_2026-08-08/FIX_absorption_conjugation.md` |

---

# AMENDMENT — THE MINI-BZ EXCHANGE-HEAD AVERAGE IS THE WRONG MOMENT (2026-08-09)

**One row, from the exciton-bands lane's LT-splitting derivation
(`~/lorrax_bse_perf_2026-08-08/EXCITON_BANDS_FEATURES.md` §2).**  The
derivation's headline is a clean acquittal that this row must protect: the
DEFAULT finite-Q exchange path is **already exact** — the driver evaluates
`v(Q)` and the pair-density factor at the true finite Q with G=0 kept, so the
product it forms is exactly `|Q|² v(Q) |q̂·d|²`, the full nonanalytic
(LT-splitting) head.  **Do not "fix" the default; a flagged head there would
be a no-op or a double count.**

| item | mechanism, at this tree | disposition |
|---|---|---|
| **`--head-minibz-average` averaged the wrong moment on the exchange head** — BOTH halves now FIXED (pending land) | With the flag ON, the mini-BZ cell average applied to the exchange head is the scalar `⟨v⟩`, where the object the nonanalytic structure requires is the 3×3 second moment `M_ab = ⟨v q_a q_b⟩` — the scalar is wrong in both direction (single sampled q̂ vs the cell's distribution) and magnitude (sampled vs v-weighted).  **The correct implementation needs NO `∂_qζ`**: the head's q-linear coefficient is the dipole already shipped in `dipole.h5`, so the averaged head is `K^head_{tt'} = (1/N_k)·d*_a(t) M_ab d_b(t')` — rank-three over transitions, never entering the μ basis (`LT_HEAD_PROBLEM.md` §6, which supersedes the ∂_qζ framing this row first carried).  Magnitude on the Si 4×4×4 fixture, computed locally: 512 meV brightest transition point-value, 253 meV cell-averaged — not small against ~50 meV binding energies | **FIXED (pending land), both halves.**  The unconditional-enable half landed first (`fix/bsekgrid-head-key-2026-08-09`); the wrong-moment half is `feat/head-moment-tensor-2026-08-09`.  *Fixed:* `bse_io._interpolate_bse_data_to_grid` no longer forces `head_minibz_average=True` on the `bse_k_grid` coarse→fine path — it reads the deck key (default off, like every other reader) and, with the key off, carries the coarse q=0 tile through untouched instead of replacing it, so the deck's `vhead` survives.  Branch `fix/bsekgrid-head-key-2026-08-09`, three red twins (a fixture-free AST gate plus bit-identity gates on both arms); the opt-in arm is preserved BIT-FOR-BIT because it is the object still under repair.  **The history verdict on why the forcing existed:** `964c682b` (2026-07-20) wanted the FINE mini-BZ head scalar, and said so — a real need, but the q=0 exchange BODY is built from centroids and the G-sphere alone and is k-grid-INVARIANT, so nothing needed densifying, and at Q=0 the head is annihilated by orthogonality anyway, so its SCALE is inert.  What the override actually cost was the BODY: it substituted a b26p/stencil model reconstruction for the exact disk tile.  Deck A/B (MoS2 3x3x1 -> 6x6x1, exact dense diagonalisation): the exciton spectrum moves by at most **1.20 meV** (≈0.1 meV on the low-lying states), NOT the hundreds of meV the LT_HEAD_PROBLEM §7.2 finite-Q figures would suggest — because those figures are the head at a SAMPLED finite Q, and this path evaluates exactly Q=0 where the head is annihilated; the ~1 meV that survives is the model body's reproduction error, and the head channel itself is worth ~0.01 meV.  It also made `bse_k_grid` unusable on IBZ-only-ζ restarts, since `vq_interp` refuses those; the default arm no longer touches ζ.  *Fixed (2026-08-09, branch `feat/head-moment-tensor-2026-08-09`):* the wrong-moment half.  With the key ON the head channel now leaves the μ basis entirely (`eval_vq` gets `head_val=0` at `gstar`, zeroing the LR G* column) and comes back as `K^head_{t,t'} = (1/N_k)·conj(d_a(t)) M_ab d_b(t')`, rank three over TRANSITIONS, assembled from the `dipole.h5` dipoles and the new `vcoul.minibz_moment_tensor` — six components on the SAME mini-BZ Voronoi draws, same two BGW branches, `minibz_average` itself untouched so no existing head number moves.  Gated on identities, not reference files: `tr M = 8π/Ω` reproduces 0.09304729699 against 0.09304729699 on Si 4x4x4 plain MC and to 1.5e-5 relative on the Baldereschi branch (whose closed-form tensor twin `δ_ab (4/9π) q0^3 Ω N_k` had to be written); the hBN slab gives tr M = 6.669e-2 / 4.318e-2 / 2.492e-2 Ry/bohr² at 3x3 / 6x6 / 12x12 with M_zz identically 0, i.e. rank two and vanishing linearly with the cell.  Four crossed-convention red twins on the contraction (transposed encode, unconjugated encode, conjugated decode, transposed moment), plus the offset-vs-momentum twin the trace identity exists to catch.  **Sequencing gate cleared first:** `SMALL_ISSUES.md` row 22 is settled in its own commit — the Cartesian q²-coefficient is canonical (it is the one with readers) and `run_sternheimer` converts its crystal Hessian at the write site; the convention is written down in `docs/theory/s-tensor-convention.md`.  **Deck validation** (MoS2 3x3x1 640-centroid slab, matched-|Q| ladder along Γ→M and Γ→K, dipoles at velocity_sign=+1 / skip_vnl=False): the OFF arm is bit-identical across trees (max|Δ| = 0.000000 eV, every Q, all 6 states) and Γ does not move on either ON arm (0.000 meV) — the Q=0 annihilation residual cannot have grown, since this branch does not touch the Γ tile.  α-Hermiticity 2.8e-12.  tr M is constant to four digits (2.5419 / 2.5415 / 2.5428 e-2 Ry/bohr²) across the near-Γ cells while the point-value head scales as |Q|, which is why the two arms cannot agree inside the Γ cell.  At the smallest ladder point the two lowest states move +0.006 and +0.003 meV while two others shift by tens of meV — the rank-≤3 PSD structure measured directly (dark states untouched, bright states shifted).  The OLD scalar arm turns out to be nearly INERT at finite Q too: ≤0.2 meV on every state at every Q but one, where it moves −8.4 meV (negative, because inside the Γ cell the point value exceeds the cell average) — the finite-Q counterpart of the Q=0 annihilation, and the reason a wrong-moment average survived unnoticed.  Cost: solve 1.07 s with the head vs 1.06 s without.  **Two limits, stated:** the opt-in arm runs through `vq_interp`, which is the truncated-SLAB evaluator, so bulk 3D decks never reach it (the tracked hBN deck is bulk 3D and cannot); and the dipole route linearises the pair amplitude across the cell, an O(Δ²)+O(sΔ) statement that is controlled near Γ and degrades as the cell centre moves out — which is where the cell average stops being needed anyway.  **Also found and corrected:** `EXCITON_BANDS_FEATURES.md` §2.4 and `LT_HEAD_PROBLEM.md` §7.4(3) prescribe the matched-|Q| pair as (0,t,0) vs (t,t,0); on a cell whose b1·b2 angle is 60° that pair differs by √3 in |Q|.  The matched pair is (0,t,0) vs (t,−t,0), and the correction moves the apparent direction spread from 21–220 meV to 1.2–11.8 meV (genuine trigonal warping).  Evidence: `HEAD_TENSOR_IMPL.md`, `THEORY_LT_HEAD_TENSOR.md`.  *Retired by measurement:* the Q=0 cancellation residual is no longer unmeasured.  On the MoS2 3x3x1 640-centroid deck (Ω=702.20 bohr³, N_k=9, `vhead`=1655.33 Ry·bohr³ ⇒ prefactor `vhead/(ΩN_k)` = 0.26193 Ry = **3563.7 meV**, the same geometric constant as the Si deck's 2600 meV), the orthogonality residual `absA = abs(Σ_μ C_μ conj(g0_μ))` is **max 4.88e-3 / rms 2.50e-3** on the BSE window and 1.14e-2 over the whole 26occ×20unocc block, so the spurious rank-one contamination the Γ tile carries is **0.085 meV worst-transition and 0.022 meV rms** (0.46 meV worst over the wide block) — the "nothing here" branch of `LT_HEAD_PROBLEM.md` §7.4(1).  Free normalisation check: the same contraction on the diagonal returns the norm `⟨u_b,u_b⟩ = 1` to max 1.4e-2 across the deck's whole `nband=40` window.  Sequencing unchanged.  Evidence: `EXCITON_BANDS_FEATURES.md` §2; `LT_HEAD_PROBLEM.md`; `FIX_bsekgrid_head_key.md` |

---

# AMENDMENT — FOUR ROWS TRANSCRIBED OUT OF THE ASIDES AUDIT (2026-08-09)

**No code moved for this amendment; these four were already true at `main` and
had no row anywhere.**  An archaeology pass over the two-fleet campaign
transcripts (`~/lorrax_bse_perf_2026-08-08/ASIDES_AUDIT.md`) found thirteen
things that were said out loud in a worker report — under headings its own
authors wrote as *"listed, not chased"*, *"registered defect"*, *"worth
flagging upward"* — and then never reached a ledger.  Two of the thirteen are
physics or deadlock class and do not belong in a small-issues list; two more
already have a decision or a write-up somewhere a census reader does not look.
Those four are transcribed here.  The rest went to
`~/lorrax_service_phase/SMALL_ISSUES.md`, the punch list and `BUILD_NOTES.md`;
the audit's own *→ Routed* section says which went where.

Verified against the tree at `f1e07bb6` (every line number below was re-read,
not copied out of the report).

| item | mechanism, at this tree | disposition |
|---|---|---|
| **Σ_c breaks the k-star relation at the five non-TRIM k** | `src/gw/ppm_accumulators.py:104-105` completed a one-sided τ grid with an ELEMENTWISE `Im` — `contrib = coeff_re * sigma_im + coeff_im * sigma_re` under a comment reading *"Crossing window — keep Im[coeff·σ]"* — where the sine sum needs `(Z − Z†)/2i`.  **The two coincide only at k = −k.**  That discriminator rides this row on purpose: a check taken at the TRIM k alone comes back green and says NOTHING about the other five, so a TRIM-only green must never be read as coverage.  Found by the MPA three-way-table lane 2026-08-09 by running the thing | **CLOSED 2026-08-09 — the fix is landed, mesh-invariance is proven, the two references are frozen, and NO OWNER ACTION IS OUTSTANDING.**  *Fixed and landed* on `main` @ `dd727216` (fix commit `c80601b8`; derived independently; the algebra agrees with `integration/mpa-table-2026-08-09`'s `27bd0984`, whose PLACEMENT it corrects — that one takes the adjoint per-τ inside the per-shard projector, which is the band adjoint only when the mesh does not cut the QP band window, i.e. only at the `-G=1` leg it was measured on).  Gates: `tests/test_ppm_crossing_completion.py` (12, incl. the sharded pair).  Measured on the Si 4×4×4 arm-b deck, `/pscratch/sd/j/jackm/sigma_kstar_0809/`: Σ_c star spread **46.7623 raw / 43.8463 diag eV → 0.0000 / 0.0000**; exact-Emf eqp0 degeneracy spread 67.64 / 10.50 eV → 0.0000; the band-9-below-band-8 inversion present at exactly the five non-TRIM k, gone at all eight; TRIM rows held to **2.59e-07 eV** while non-TRIM eqp0 moves up to **20.63 eV** (bands 1–16).  *Mesh-invariance proven* at a 2×2 mesh by the completion pass's `fix_g4b` leg: the fix reproduces its own `-G=1` `eqp0` to **1.10e-06 eV** against the pre-fix control's 9.97e-07, and sits at the η floor (1.26e-06–1.80e-06 eV) on all eight k rather than on the three TRIM k alone.  *References frozen* (owner-authorized 2026-08-09, ``1e64d83a`` on `chore/gnppm-freeze-2026-08-09`): `test_gnppm_matches_reference` and `test_bispinor_gnppm_matches_reference` were frozen FROM the elementwise form and had been red-by-correction since `dd727216`; they now carry the corrected values and are green.  The values frozen are the ones cut at `dd727216` and preserved at `/pscratch/sd/j/jackm/sigma_kstar_0809/_pytest_recut2/`, **re-verified against current `main` @ `5b135f8e` before the freeze and bit-exact** — max|Δ| = 0.000e+00 with 0 of 2484 (gnppm) and 0 of 1620 (bispinor) compared cells differing at all — which settles the one live question the freeze had: the intervening `9a730da8` dipole re-cut does NOT reach either deck.  It re-cut si and hbn only, `gnppm_debug/dipole.h5` is unchanged there (recorded as re-cut-blocked-no-deck) and `mtxel_sweep.dipole_operator` is called by `psp.get_dipole_mtxels` and by nothing on the Σ path, while `bispinor_test.in` takes the explicit head bypass and ships no `dipole.h5` at all.  The movement the two references record is gnppm **2.9794e-02 eV** and bispinor **7.7240e-03 eV** against `_XMACHINE_ATOL_EV = 1e-5`, with **sigX and VH bit-identical (max|Δ| = 0.000e+00)** on both — the control that the static path is untouched — and the per-k movement matching the peer's independent prediction cell for cell.  **THE PEER-PLACEMENT WARNING IS KEPT AS PERMANENT RECORD, and it is why these two green gates are not coverage.**  The per-shard placement's damage is INVISIBLE to `eqp0`: `peer_g4b`'s `eqp0.dat` is **byte-identical** to `fix_g4b`'s, because `swapaxes(-1,-2)` is the correct band adjoint exactly on the DIAGONAL shard blocks and a `one_shot` `eqp0` reads only Σ_c(i,i).  It shows up on Hermiticity: `max|Σ_c − Σ_c†|` at ω = 0 is **1.3e-06–1.8e-06 eV at all eight k under the fix** against **0.132–0.257 eV at all eight under the per-shard form** — which breaks the three TRIM k that the PRE-FIX code got exactly right — and `max|Σ_c(fix) − Σ_c(peer)|` over the whole cube is **5.2333 eV**.  **A diagonal-Σ deck observable can never discriminate the two placements; the 2×2 unit gate is the only thing that can.**  The two references frozen above are themselves `sigma_diag` files, so they stay green under BOTH placements: they pin the corrected VALUES and they are not, and must never be read as, coverage of the off-diagonal defect class — that is `tests/test_ppm_crossing_completion.py`'s sharded Hermiticity pair, and each reference carries the caveat in its own provenance header so the point survives without this row.  Evidence: `~/lorrax_bse_perf_2026-08-08/FIX_sigma_kstar.md` (§COMPLETION), `ASIDES_AUDIT.md` §A1 |
| **`jit__multi_slice` has a rank-dependent persistent-cache key** | **ROOT-CAUSED AND FIXED 2026-08-09 (`fix/multislice-cachekey-2026-08-09`), row kept for the mechanism and for the two corrections it forced.**  `jit__multi_slice` is **not LORRAX code**: it is jax's own `ArrayImpl._multi_slice` (`jax/_src/numpy/array_methods.py`), declared `@jit(static_argnums=(1,2,3))`, and `jax/_src/array.py::shard_device_array` calls it with `sharding.addressable_devices_indices_map(x.shape)` — **this rank's** shard bounds.  So the shard offsets are static arguments, rank *r* compiles `slice(x, [r·n/P, 0], [(r+1)·n/P, m])`, and the ranks hold different keys for different programs.  Writes are process-0-only, hence rank 0 hits while the peers miss.  It is reached whenever a **single-device, fully addressable** `jax.Array` meets a **partitioned** multi-device sharding; a fully-replicated target short-circuits before the jit, and a **numpy** source takes `pxla._shard_np_array`, which does not jit at all | **FIXED on the branch, PUSHED, NOT MERGED.**  The canonicalization keeps the shard SIZES static and passes the OFFSETS as dynamic `lax.dynamic_slice` operands, so every rank compiles one program (`jit__lorrax_canonical_shard_slice`); `common/jax_compile_cache.py::_install_shard_slice_patch`, armed at P>1 only, red twin `LORRAX_JAX_CACHE_SHARD_SLICE=0`, gates in `tests/test_compile_cache_shard_slice.py`.  **MEASURED A/B, one env var apart, same tree, same deck** (`si444`, P=4, warm): red `rank 0: xla_compiles=0 hits=36` vs `ranks 1,2,3: xla_compiles=1 hits=35 vetoed=1` with three distinct `jit__multi_slice-*` keys — the registered defect, reproduced; green **`xla_compiles=0 hits=38 vetoed=0` on all four ranks** and zero `jit__multi_slice` keys.  Eigenvalues identical across all four legs.  **CORRECTION 1 — the wall cost in the old row was wrong in the safe direction and the framing was wrong in the unsafe one.**  This row said *permanent silent hang*.  It is the documented deadlock **precondition**, but the hang has **not** been observed for this module and structurally would not be: the compiled module is `parameter → kLoop fusion(slice)` (dumped 2026-08-09) with **no autotune candidates**, so `AutotunerPass` has no work to shard and nothing blocks on `MultiProcessKeyValueStore`.  Both P=4 legs run with the divergence deliberately present completed `rc=0`.  What justifies the fix is that it is exactly the divergence shape `jax_compile_cache.py` exists to prevent, nothing pins the module's contents against a future jax, and it is now free.  **CORRECTION 2 — the reproducer is not BSE-specific.**  `runtime/__init__.py::nccl_warmup` puts `jnp.ones(...)` onto a partitioned `NamedSharding` at mesh bootstrap, so **any** P>1 GPU run reaches this path before any physics.  Evidence: `~/lorrax_bse_perf_2026-08-08/FIX_multislice_cachekey.md`; `FIX_warmcache.md` §1.2; `ASIDES_AUDIT.md` §A4 |
| ~~**POINTER — LORRAX cannot read an `nspinor = 1` WFN.h5**~~ | ~~`services/symmetry_maps/src/symmetry_maps/maps.py:912` carries the comment *"For ns=1 (non-SOC), U_eff is the 1×1 identity and this einsum is a no-op"*.  That claim is false: `spinor_rotation_for_sym_row` (`:692`) returns `(2, 2)` or `(nk, 2, 2)` **unconditionally**, so the full-BZ unfold rotates a one-component slab with a two-component matrix.~~  **Every in-tree fixture is nspinor = 2, so the suite structurally cannot see this** — which is still the whole argument for a row here rather than a fixture README.  The comment was true of nothing: numpy's *and* JAX's `einsum` BROADCAST a size-1 labelled axis instead of raising, so a scalar ψ came back 2-component holding `(U[j,0]+U[j,1])·ψ`, and on a TRS row the summing was done by `iσ_y·conj(U)`'s off-diagonals | **FIXED AT UNIT LEVEL, 2026-08-09, `fix/nspinor1-loader-2026-08-09` — DECK LEG OWED.**  `spinor_rotation_for_sym_row` now takes `nspinor` and returns a true 1×1 identity for scalar ψ on **both** row kinds (spinless time reversal is Θ = K, not iσ_y K — a different representation, not a truncation), which fixes the host path (`unfold_psi`) and the device path (`WfnLoader._ensure_phdf5_static` → the phdf5 unfold kernel's `einsum("kac,bckg->bakg", …)`, which carried the same defect and is **not** named in the original write-ups) from one place; `unfold_psi` now raises on a spinor/ψ width mismatch rather than letting einsum broadcast it.  **Gates:** ns=1 correctness is synthetic by necessity — 8 unit cells in `services/symmetry_maps/tests/test_symmetry_maps_algebra.py` (incl. a red twin that reproduces the broadcast and its wrong values) plus 2 end-to-end cells in `services/wfn_loader/tests/test_wfn_loader_contract.py` that build a real `nspinor = 1` HDF5 in `tmp_path` and read it through `WfnLoader`; ns=2 splash is proven by **bit-identical** full-BZ ψ pre/post on all six checked-in decks.  **What is still owed, and why it is not a claim this row makes:** no deck-level validation on a genuine scalar QE run — no such fixture was built (deliberately, per the fix worker's brief), so *"LORRAX reads a real QE scalar WFN and gets the right Σ"* remains UNMEASURED.  The evidence area for that leg is still `/pscratch/sd/j/jackm/svc_vcoul/hbn_fixture_prep/qe_scalar_nspinor1_ATTEMPT/`.  Full write-up: `~/lorrax_bse_perf_2026-08-08/FIX_nspinor1_loader.md`.  Prior write-ups at `tests/regression/hbn_cohsex_debug/README.md:274` and `README_PLAN.md:115` (both cite the pre-drift `maps.py:907`) are now stale on the mechanism and are left as provenance.  `ASIDES_AUDIT.md` §B2 |
| **the velocity-commutator sign at `src/common/mtxel_sweep.py:676`** | The dipole operator's non-local term.  Four sign sites across two producer routes; the `vnl_velocity_sign` knob now reaches all of them and **the default is the `+1` arm** — `p + ∂V_NL/∂K`, which `velocity_matrix_k` and `orbital_magnetization` already called canonical and which reproduces BerkeleyGW's q→0 head (ε₀₀ 24.2208 against BGW's 24.2205, versus 31.8204 on the old arm).  A thing the pointer row never said, found by the pre-sweep: `--vnl-mode numeric` was the arithmetic negative of `--vnl-mode analytic`, so the two modes sat on opposite arms; normalised in the same change | **FLIPPED on `fix/dipole-vnl-sign-2026-08-09`, PENDING THE FIXTURE RE-CUT.**  The four protected `dipole.h5` fixtures and the gnppm/cohsex/hbn frozen references have NOT been regenerated, so this tree's default operator and its committed fixtures are different operators — visible rather than silent via `prov_vnl_velocity_sign` on the h5 and `tests/test_bse_oscillator_strengths.py`, which holds the before-picture and goes red when they are re-cut.  The `-1` arm stays reachable from a deck key and stays bit-identical to the pre-knob expression.  `~/lorrax_bse_perf_2026-08-08/FIX_dipole_vnl_sign.md`.  The re-cut was blocked by the missing pseudopotential pre-flight (row above, FIXED at `a5c68576`); with `--pseudo-dir` it is unblocked, and **only hbn and si are re-cuttable at all** — `gnppm_debug` and `cohsex_debug` committed their fixtures at band counts no deck in the tree still requests, so those two are **re-cut-blocked-no-deck** and are an owner row, not a worker improvisation. |

---

# AMENDMENT — THE CERTIFIED TABLES LINEAGE LANDS (2026-08-09)

The three stacked table campaigns (damped-line global sets, full-precision
imaginary-axis entries, and the width clause) land as one merge.  The
catalog now holds **84 certified entries — 54 height + 15 width + the
damped-line family — at 100% certification**, serving every measured
production request, every deck at its own ω̂ with zero band spend, and
**100.000000% of the 81,432,576-pole Σ routing demand** (the audit's
histogram, served bin by bin at both R extremes).  The width clause ships
all-composite by a measured argument: BTV is 4.3× cheaper in nodes and
covers 1.9×10⁶× less of a continuous demand.  β_max for the width clause
is **1.0, derived** (x_min ≥ Re Ω plus the fitter's own width_ratio_max),
with a drift test reading DEFAULT_GUARDS directly.  The two clause
catalogs stay separate files (their ranges overlap near 0.6; a merged
catalog would let arithmetic serve the wrong physics), and the
zero-width-tables state the Σ audit found survives as a red twin of a
condition that can no longer occur by accident.

Landing notes: one textual conflict (the service move × the clause split
in `beta_selector._catalog_path`) resolved as the union; five test
imports repointed from the pre-move spelling; the `test_damped_line_tables`
16-red census row was already corrected by the completeness audit and
stands as written there.  Suites at this head: beta-selector + imag-tables
159/159; the minimax service suite green.  The Σ routing's mirror constant
(`SHIPPED_WIDTH_BETA_MAX = 2/3` on feat/mpa-sigma) goes red against this
landing BY DESIGN — the mirror test catching the handoff — and flips to
1.0 in that branch's re-anchor.

---

# AMENDMENT — `kirr_fullids`: THE WEDGE ROW MAP, FIXED, AND THE ONE FIXTURE IT LEAVES STALE (2026-08-08)

`fix/kirr-fullids-2026-08-08`, off `main` @ `bc37b4d3`.  Nothing on this list
turns red.  The amendment is here for the two things a green suite does not
say: which committed fixture now disagrees with the live code, and which live
code path stops answering and starts refusing.

**THE DEFECT.**  `SymMaps.kirr_fullids[i]` is supposed to be the full-BZ row
that IS the WFN file's irreducible k-point `i`, and every wedge-shaped output
in the tree gathers with it — `gw_output.write_results` builds every column of
`eqp0.dat` / `eqp1.dat` that way, and `sc_iteration.dump_qp_wfn_artifacts`
reads `U` at those rows.  It was built instead from the STAR LABELS: the first
full-BZ row carrying label `i`, taken out of `irr_idx_k`, with a silent
identity fallback `kirr_fullids[i] = i` for labels no row carries.  Labels do
go uncarried: `find_symmetry_ops_simple`'s op-selection policy has no `break`,
so a full-BZ row reachable from more than one stored IBZ point is labelled
with the HIGHEST of them and the lower ones are orphaned, which happens on any
deck whose stored wedge has two entries in one orbit.

**MEASURED at `bc37b4d3`, all four in-tree decks:**

| deck | shipped | correct | damage |
|---|---|---|---|
| `gnppm_debug` | `[0,1,1,3,4,5,3,5,4]` | `[0..8]` | 4 rows name a k 1/3 of a **b** away; 3 row pairs collide; IBZ k2, k6, k7, k8 never emitted |
| `bispinor_debug` | `[0,1,1,3,4,5,3,5,4]` | `[0..8]` | identical to the above (same mesh, same group) |
| `cohsex_debug` | `[0,1,1,4]` | `[0,1,2,4]` | row 2 duplicates row 1; IBZ k2 never emitted |
| `si_cohsex_debug` | `[0,1,2,5,6,7,10,27]` | same | correct **by luck** — 48 ops, eight disjoint stars, no orphaned label |
| `hbn_cohsex_debug` | `arange(18)` | same | `ntran = 1`, so the trivial branch, which is right by construction |

The fix matches on k itself — the row whose `unfolded_kpts` entry equals
`wfn.kpoints[i]` modulo a reciprocal lattice vector, to
`find_symmetry_ops_simple`'s own 1e-6 — and raises instead of falling back
when a stored k is not on the grid its file's `kgrid`/`shift` generate.  New
gate: `services/symmetry_maps/tests/test_symmetry_maps_kirr_fullids.py`,
14 cells, of which 10 are red on the pre-fix construction.

**FIXTURE ADJUDICATION — one file, registered, NOT re-frozen.**

`tests/regression/cohsex_debug/qp_wfn_rotations.h5` carries a
`kirr_to_kfull` dataset whose stored value is `[0, 1, 1, 4]` — the broken map,
frozen on the day the file was written.  The live class now produces
`[0, 1, 2, 4]`, so the blob and the code disagree.  **It is not re-frozen
here**: re-cutting a committed deck artifact is the re-cut wave's row, not a
fix branch's, and the manifest discipline is that a stale blob is registered
with its measured value rather than quietly replaced.  Whoever next re-cuts
`cohsex_debug` should regenerate the file and strike this paragraph.

Everything else in `tests/regression/` was checked and is CLEAN.  The reason
is worth stating because it is what kept the defect out of the gates: every
committed `eqp_*.dat` / `sigma_diag_*.dat` in the tree is the SIGMA-DIAGNOSTIC
format (`k-point N:` blocks) written on the FULL BZ un-subset — 9 blocks on
the 3×3×1 decks, 18 on hBN, 8 only on Si — and the sigma-diagnostic writer
never touches `kirr_to_kfull`.  No committed fixture is a BGW-format wedge
`eqp{0,1}.dat`.  `gnppm_debug/eqp_rotations_fixedpoint_ref.npy` is `(9, 46)`,
i.e. full-BZ, and `cohsex_debug/sigma_mnk.h5` carries no wedge map at all.

**ONE LIVE PATH CHANGES ANSWER TO REFUSAL, and it is the designed one.**
`eqp_bgw`'s post-hoc CLI pairs `kirr_to_kfull` with a wedge-stored
`sigma_mnk.h5` through `file_io.sigma_output.k_irr_rows_for`, which refuses
any requested row that is not itself a stored row.  On `cohsex_debug` the k_irr
store keeps rows `[0, 1, 4]`; the broken map asked for `[0, 1, 1, 4]`, which
all land on stored rows, so the CLI silently handed back **k1's matrix under
k2's label**.  The corrected map asks for `[0, 1, 2, 4]`, row 2 is not stored,
and the call now raises by name.  That is `k_irr_rows_for`'s whole purpose —
"the refusal is the whole point", `sigma_output.py:775` — reaching a real deck
for the first time.  It is a silent wrong answer becoming a loud one, not a
regression, and no collected test exercises that pairing on that deck.

**§7.7.1 OF `BGW_CD_COMPARISON_DESIGN.md` IS CORRECTED BY THIS BRANCH.**  That
section attributes its mislabelled k rows to `kirr_fullids` evaluating to
`[0..7]` on the Σ_x probe's tree.  It did not: that tree (`si_mpa_0808/wt` @
`59fa874b`) carries the same construction as `main`, its deck's stored wedge is
the same eight k in the same order as `si_cohsex_debug`, and the run's own
`eqp0.dat` lists all eight correct IBZ k.  What the probe read was
`eqp_r1128x.dat`, the **sigma-diagnostic dump, 64 `k-point` blocks on the full
BZ**, whose first eight rows are full-BZ rows 0-7 with IBZ parents
1, 2, 3, 2, 2, 4, 5, 6 — exactly the table §7.7.1 prints.  The probe's remap
and every measurement built on it stand; the attribution does not.  The defect
above is real and was found independently, on the other three decks.

---

# AMENDMENT — `K^d_B` UNDER ζ SHARDING: REGISTERED AND STRUCK (2026-08-08)

**`tests/test_bse_coupling_zeta_sharding.py` is registered here and struck in
the same commit.**  Registering a defect nobody had ever seen fail needs a
word of explanation: the non-TDA coupling block's screened-direct term
`K^d_B` was **wrong at every P > 1** and had been since it was written, and it
was not on this list because **no test in the tree applied the coupling block
on a mesh with more than one device**.  The existing non-TDA gates all run on
the 1×1 mesh `lx test`'s conftest pins, which is exactly the one configuration
where the defect is invisible.  A row that only appears once someone writes
the missing check is still a row: it is registered with its measured red, and
struck with the fix, so the census carries the fingerprint rather than losing
it to a green file.

| | |
|---|---|
| machine | Perlmutter, lx pool (JID 56522011), 4×A100, Shifter, `lx test` |
| module | `LX_BASE_MODULE=lorrax_J070`, jax 0.7.0 |
| tree | `/pscratch/sd/j/jackm/kdb_0808/wt`, branch `fix/kdb-zeta-sharding-2026-08-08` off `main` @ `28ff477f` |
| files | `src/bse/bse_ring_comm.py` (the coupling encode), `tests/test_bse_coupling_zeta_sharding.py` (new) |
| prose record | `~/lorrax_bse_perf_2026-08-08/FIX_kdb_sharding.md` |

**THE DEFECT — a ζ shard carried away by a `ppermute`, not a conjugation.**
The coupling encode carries Henneke's `j_c ↔ j_v` swap, so the ζ index it
builds from the conduction axis (sharded on `'x'`) is `ν` (sharded on `'y'`)
and the one it builds from the valence axis (on `'y'`) is `μ` (on `'x'`) — the
pairing is CROSSED relative to the resonant block's, where the ζ index a stage
produces lands on the same mesh axis as the orbital axis it consumed.  The
shipped chain therefore ring-rotated a partially-contracted intermediate along
`'y'`, the very axis its `ν` shard lived on.  A `ppermute` moves **every** axis
of the buffer it is handed, so each `'y'` rank accumulated its neighbours' ζ
tiles against its own ζ shard.  At P=1 a `ppermute` is the identity, so the
term was bit-exact there and nowhere else.

**MEASURED, on the record deck (Si 4×4×4 SOC, `nontda/deck_clean`, N=1024),
same payload, same process, 1×1 vs 2×2:**

| quantity | pre-fix | post-fix |
|---|---:|---:|
| `K^d_B` ‖P1−P4‖/‖P1‖ | **5.525e-01** | **1.749e-15** |
| `K^d_B` ‖K−Kᵀ‖/‖K‖ at 2×2 | **6.911e-01** | **2.864e-11** (= its P=1 value) |
| coupling correction, state 1 | **−0.223277 meV** | **−0.697956 meV** |
| max\|Im λ\| of the 2048-dim operator | 2.647e-06 Ry | 1.025e-13 Ry |
| `A` (resonant), `K^x_B` (coupling exchange) | clean | **bit-identical to pre-fix** |

The physics cost was two thirds of the coupling correction, lost silently, on
exactly the configuration multi-process non-TDA exists to run.

**THE FALSE CASE SHIPS WITH THE CHECK.**  `test_the_pre_fix_coupling_encode_is_caught`
monkeypatches the 2026-08-08 chain back in and requires both gates to fire —
cross-mesh `‖P1−P4‖/‖P1‖` ≥ 1e-3 and `‖K−Kᵀ‖/‖K‖` ≥ 1e-3 — **and** requires
the twin to leave `A` and `K^x_B` bit-clean, so the red proves the gate is
pointed at the coupling screened-direct term and not at the mesh in general.
The gate is portable: synthetic payload, CPU host devices, no GPU, no restart
file, no deck.

**Struck on:** the new file green (6 passed) with its red twin red; the deck
detector green at 2×2 on both encode routes (`low_mem` ring and `all_gather`);
P=1 **bit-identical** to pre-fix on all four blocks (`0.000e+00`, exact array
equality), which is the acceptance that says the fix touched only the P>1
path; and the coupling correction at 2×2 reproducing the P=1 value to
**0.0000 µeV** over all twenty states.

---

# AMENDMENT — `wq_resolvent` DIAGNOSED AND STRUCK (2026-08-08)

**`test_bse_w0_resolvent::test_wq_resolvent_matches_restart_finite_q` is STRUCK
from this file.**  The last undiagnosed red of its family — registered by the
RE-CUT WAVE amendment below, fingerprinted at `rel_err=6.87e-01` and A/B-proven
pre-existing from four bases (`013aad92`, `81a285af`, `e495fc45`, `f0435e9a`) —
was a real code defect.  It is diagnosed and fixed.

| | |
|---|---|
| machine | Perlmutter, lx pool (JID 56522011), 4×A100, Shifter, `lx test`, 1 GPU, `--workers=1` |
| module | `LX_BASE_MODULE=lorrax_J070`, jax 0.7.0 |
| tree | `/pscratch/sd/j/jackm/wq_phase_0808/wt`, branch `fix/wq-resolvent-phase-2026-08-08` off `main` @ `fef002e9` |
| commit | `b5c0cf15` — `src/bse/bse_w_exact.py`, one function (`build_finite_q_data`) |
| prose record | `~/lorrax_bse_perf_2026-08-08/FIX_wq_resolvent.md` |

**THE DEFECT — a CONJUGATION, and it is exactly invisible at q=0.**  The W_q
resolvent's four pair-density vertices carry one fixed conjugation convention
(`K^x = M V M†`, conjugate on the ENCODE leg), pinned by the optical BSE
exchange term.  Composed, that chain assembles `conj(χ₀) = χ₀ᵀ`, while the GW
producer's χ₀(q) — the object whose Dyson solve wrote the `W0_qmunu[q]` tile the
cell is scored against — is the other one.  At q=0 the k-sum runs over ±k pairs
whose pair densities are complex conjugates under TRS, so **χ₀(0) is REAL**
(‖χ₀−χ₀ᵀ‖/‖χ₀‖ = 4.7e-11 measured on the gnppm fixture) and the two conventions
are the same matrix.  At q≠0 they are not (2.9e-01 on the same fixture), and the
chain resums χ₀(−q) against the +q Coulomb tile — a hybrid that is no stored tile
in any conjugation, which is why an earlier grid-wide argmin scan over every q′
in `T(q′)`, `conj(T(q′))`, `conj(T(−q′))` found no match and returned "operator,
not label".  **Fix: conjugate ψ on both legs in `build_finite_q_data`**; each
vertex is bilinear in (ψ_c, ψ_v) with exactly one conj, so this flips all four at
once.  Exact — no TRS assumption.

**THE DIAGNOSIS IS AN A/B, NOT AN ARGUMENT.**  A dense numpy model of the same
chain, run on the restart on a login node with no jax and no allocation,
reproduces the observed per-column failure **digit for digit** under the shipped
convention and closes under the GW one:

| col | rel(GW convention) | rel(shipped convention) | observed on GPU, pre-fix |
|---:|---:|---:|---:|
| 179 | 2.4575e-08 | **6.8742e-01** | **6.8742e-01** |
| 375 | 2.4575e-08 | **6.8742e-01** | **6.8742e-01** |
| 337 | 2.4608e-08 | **3.5347e-01** | **3.5347e-01** |
| 253 | 2.4570e-08 | **7.0595e-01** | **7.0595e-01** |

**GATES.**

| leg | selection | result |
|---|---|---|
| the cell's own file + its chain sibling | `test_bse_w0_resolvent.py test_bse_w_omega_chain.py` | **5 passed** (34.03 s) |
| BSE subset, 9 files | + `test_bse_dense_reference`, `test_bse_w_donation`, `test_bse_matvec_opts`, `test_bse_stack_matvec`, `test_fft_shardmap_context`, `test_bse_feast_runner_cache`, `test_bse_nontda_restart_preflight` | **67 passed, 1 deselected** (71.05 s) — **zero reds** |
| red twin A — revert the flip | same file | **1 failed, 2 passed**, and the failure is `q=(0,1,0) col 179: rel_err=6.87e-01`, the historical fingerprint reproduced |
| red twin B — flip the conduction leg only | same file | **1 failed, 2 passed**, `rel_err=9.28e-01` — a half-flip is a third, also-wrong operator |
| restore | | md5 `bb9584c3d182c98f52fe5eba74a25147` on both sides, `git status` clean — **RESTORE EXACT** |

The cell now closes at **2.459e-08** against its own unchanged 1e-6 gate, and the
**q=0 sibling is unmoved at 2.157e-09** — it does not go through this function.
Both twins leave the q=0 sibling and `test_kgrid_shift_map_matches_roll` green, so
the twin measures the finite-q vertex and nothing global.

**The whole symmetry-reduced q grid**, via `bse_w_exact --compare-wq` on the gnppm
fixture — every q≠0 was `6.87e-01`-class before:

```
 iq   q (kgrid)  max_rel_err      median   max_resid
  0   (0, 0, 0)    3.203e-09   2.253e-09   4.219e-10
  1   (0, 1, 0)    2.854e-08   2.444e-08   6.031e-10
  2   (1, 0, 0)    2.905e-08   2.626e-08   3.481e-10
  3   (1, 1, 0)    7.895e-08   4.871e-08   1.456e-10
  4   (1, 2, 0)    2.820e-08   2.464e-08   3.766e-10
```

That driver already printed "Closure at the GW minimax-quadrature floor confirms
`W_q = v_q(0-H_RPA^q)^-1 v_q + v_q` at every symmetry-reduced q".  The sentence is
now true.

**BLAST RADIUS: one function, three callers, none on a solve path.**
`build_finite_q_data` is called only by `bse_w_exact`'s own `--compare-wq` and
`--w-omega-chain` arms and by the two test files that gate them
(`git grep build_finite_q_data`).  Nothing in `bse_nontda`, the Lanczos/FEAST
solvers or `bse_jax` reaches it, so the non-TDA coupling cross-check is untouched
by construction, and was re-measured on both sides of the branch point to
confirm it: base `fef002e9` and this branch return the Si non-TDA and TDA arms
**identical to every printed digit** (`FIX_wq_resolvent.md` §6.3), so the
non-TDA coupling correction is 0.698 meV on both.

Every leg above was re-taken after the rebase onto `fef002e9`; the branch was cut
at `ed11a955`, gated there in full, and the rebase was conflict-free with an
identical diff.

**Two hypotheses this closes, both previously live.**  The stale-artifact family
was already dead (`FIX_nontda_feature.md` §4); this adds that the artifact was
never the question — the fresh restart and an Aug-7 one give **identical** dense
closure to all printed digits.  And the umklapp-phase lead its own docstring
ranked first is **wrong**: 6.87e-01 does sit inside the 0.6–3.2 band that
docstring quotes, but the band was coincidence.  The `no umklapp Bloch phase`
bullet stands, verified — the phase structure was exact at every q all along
(|ratio| = 1.00000, arg = 0.003° between the two constructions of χ₀).

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

---

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

# AMENDMENT — VCOUL HEAD-SLOT LANDING, on `main` (2026-08-08, owner-approved after the hBN anchor)

The mini-BZ head injection moved to argmin|q+G| with tied-slot mean (the
landing merge has the full case; the adjudication is
`~/lorrax_service_phase/HBN_HEAD_ANCHOR_2026-08-08.md`).  Two frozen arms
go red because the fix is right:

| gate | measured | status |
|---|---|---|
| `tests/test_gw_jax_regression.py::test_hbn_matches_frozen_reference` | 0.128098 eV, measured TOWARD BerkeleyGW (BGW averages every under-cutoff slot; the old rule's bare tied-partners have no BGW counterpart; new rule 1.42x closer on sigTOT, 3.71x on sigCOH, 18/18 k closer at converged ISDF) | red until re-frozen with the delta documented — the re-cut wave's manifest |
| ~~si_bse_debug frozen BSE arm~~ | a further 0.101 meV atop the exchange landing's movement, same deck, same single re-cut | **STRUCK — GREEN.** The one cut was taken 2026-08-08 and covers both landings, as predicted.  See THE RE-CUT WAVE amendment |

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

# AMENDMENT — THE COARSE→FINE W DENSIFIER TRIGONOMETRICALLY INTERPOLATED A DIVERGENT HEAD (2026-08-10) — **FIXED ON THIS BRANCH**

**This was never a red either, and it could not have been one.** Both paths
that densify W — the `bse_k_grid` deck key and `exciton_bands --w-coarse-grid`
— are opt-in, neither is exercised by any deck in the tree or on Perlmutter
(`bse_k_grid` appears only inside comment headers; no deck contains
`w_coarse`), and the suite's densification cells check shapes, solvability and
the on-grid no-op rather than the value of W between the coarse samples. The
defect was found by reading BerkeleyGW's BSE side against ours during the
Schur design pass (`~/lorrax_bse_perf_2026-08-08/SCHUR_BSE_DESIGN.md` §1.4),
not by a failing test.

The mechanism is short. `kernel.x` writes a divergence-**stripped** head, wing
and body to `bsemat.h5` and `intkernel` restores the singular factors at
assembly, per fine q, and the source says why twice in identical words:
"what we actually interpolate is only the head matrix elements (i.e. excluding
the 1/q² factor)…". LORRAX's densifier inverse-FFTs `W_q` to the coarse
R-lattice, zero-pads and FFTs back — exact band-limited trigonometric
interpolation — and until this branch its operand was the **post-injection**
tile `W_body0(q) + Δ·δ_{q,0}`, with `Δ = (whead/Ω)·conj(g₀)⊗g₀`. A Kronecker
delta is the one function shape a band-limited interpolant cannot represent.
Its interpolant is a Dirichlet kernel, so a fraction of a prefactor worth
~2600 meV on the Si anchor deck (3564 meV on the MoS₂ slab) was deposited, with
alternating sign, at fine q that should carry none of it; and the 1/q² rise
that fine q *inside* the coarse Γ cell genuinely should carry was missing
altogether, because no smooth interpolant between Γ's cell-averaged value and
its neighbours produces one. The second half is the one that matters
physically: the densified kernel under-binds excitons near the zone centre, and
the error grows with the densification factor instead of shrinking.

| item | mechanism, at this tree | disposition |
|---|---|---|
| **The coarse→fine W densifier interpolated the Γ head as a Kronecker delta, so the head channel rang in sign across the fine zone and supplied none of the 1/q² rise inside the coarse Γ cell** | `bse_io.make_w_densifier` is exact trigonometric interpolation, and its operand carried `Δ·δ_{q,0}` because the loader injects the rank-one head at `bse_io:1843` and `_interpolate_bse_data_to_grid` runs afterwards; `decimate_W_q_to_subgrid`'s docstring states the same contract from the other side ("the q=0 tile, incl. its rank-1 head, is preserved"). Measured on the shipped densifier at m = 2: the interpolated head **changes sign** — impossible for a screened Coulomb head, since `S(q) = v/(1 − 8π q̂ᵀSq̂)` is strictly positive — puts **more than half** its weight outside the coarse Γ cell, and at fine Γ stays within 5% of the coarse value where the cell average should grow like the inverse square of the cell. Not attributable to a commit: the densifier has interpolated the injected tile since the ordering was established, and both consumers are opt-in | **FIXED on `feat/schur-c1-densify-2026-08-10`, PUSHED, NOT MERGED.** The head is split off BEFORE the densifier and re-attached per fine q as `S_fine(q)·conj(g₀)⊗g₀/Ω` — the fine mini-BZ cell average at Γ, the pointwise integrand at the other fine q inside the coarse Γ cell, zero outside it (where the coarse tiles already carry their own heads through the full solve, so re-attaching would double count). Both expressions come from the ONE ratified q=0 integrand, which is **called and never modified**. New module `src/gw/head_densify.py`; one composer `bse_io.build_w_head_channel` serves both densification paths; the loader defers the injection through a new `defer_whead` on `_inject_q0_head`, kept explicitly distinct from the `w0_ready` gate so the two skips cannot be confused in the log. Default is the repaired path on both consumers; `w_head_densify = legacy` (deck key, or `--w-head-densify` on `exciton_bands`) restores the old behaviour and exists only as the A/B control. **GATES.** (a) ON-GRID IDENTITY, bitwise: with fine == coarse the re-attached array is `[whead at Γ, 0 elsewhere]` under `np.array_equal` on fcc/simple-cubic/hexagonal/triclinic cells at three grids — bitwise because the anchor is applied as `whead·(S/gamma_ref)` and the Γ entry IS `gamma_ref`, so the ratio is a float over itself; through the real loader, `bse_k_grid == coarse` returns a byte-identical bundle, and an AST cell pins that the fine grid is resolved before the head is injected. (b) THE HEAD SUM RULE: the zone average is exact at m = 1 (bitwise), converges monotonically at **16.1 / 5.1 / 4.4 / 2.8 %** for m = 2/3/4/6, and the design's RED TWIN (`gamma_cell='coarse'` — re-attach at the coarse mini-BZ scale) is invisible at m = 1 exactly as predicted and **3.3× to 6.8× worse** at every m > 1. The synthetic 16.1% at m = 2 is reproduced to three figures by the real Si deck's own log line. (c) THE A/B on the `--w-coarse-grid` harness, Si 4×4×4 anchor deck decimated to 2×2×2 and densified back, against the natively fine reference — **split verdict, and the split is the finding.** On the OBJECT, relative `‖W_dense − W_native‖`: legacy 0.74793, **C1 0.68989**, twin 0.69773 — C1 is 7.8% closer over the whole zone, **16.3% closer inside the coarse Γ cell** (0.82764 → 0.69237), and 3.8% closer OUTSIDE it, which is pure removal of the ringing since C1 adds nothing there; both arms are exact at the Γ tile and the twin is 57% wrong there. On the EXCITON SPECTRUM, C1 **loses**: MAE against native is legacy 17.6 meV, C1 28.4 meV, twin 117.2 meV. The same run says why, and it is not the head. At m = 2 — the only ratio this deck supports, since the finest bulk W grid anywhere on the system is 4×4×4 and the one 6×6×1 restart is a slab — the residual is overwhelmingly the BODY: `‖ΔW‖/‖W‖ ≈ 0.7` for *both* arms. Every arm under-binds, and they order monotonically by total deposited head weight (zone averages 2.18 twin < 3.94 C1 < 18.80 legacy, against the native 2.35; Q=0 energies 2.4952 > 2.4065 > 2.3906 > 2.3624 eV). Legacy's head is **eight times** the native zone weight, because decimation leaves the FINE cell's head average sitting on a coarse grid, and that excess attraction partially cancels a much larger body deficit — a cancellation, not a correctness signal, and demonstrably so, since legacy is farther from the native W in *every* region including the one where its eigenvalues look better. The harness on this system cannot adjudicate the head treatment on the spectrum; settling it there needs a 6×6×6 or 8×8×8 bulk restart, which does not exist today. (d) HERMITICITY: machine zero (≤1e-13 relative) by construction — `S_fine(q)` is float64 and the update is a real multiple of `conj(g₀)⊗g₀` — with a deliberately complex scalar as the red twin, breaking it by exactly `2|Im S|·|g₀|²` to 1e-10 relative; the shipped path cannot reach that twin, because `head_scalar_pointwise` refuses a complex head rather than casting it. (e) THE 0.41 meV PARITY CONFIG is untouched, structurally and by measurement: it is an on-grid configuration, C1's code runs only when a densification is pending, and no deck in the tree or on Perlmutter sets either consumer. Measured on Perlmutter, the same default-path `exciton_bands` run on this branch and at base `c3e8bda6` is **byte-identical** once the solver is converged (`--max-iter 200`). At the default 40 iterations the two differ by 3e-5 eV at X and agree exactly at Q = 0 — that X point is a six-fold near-degenerate cluster the block-Lanczos has not resolved, and it moves by 9e-4 eV between a 1-GPU and a 4-GPU run of the SAME code, i.e. thirty times the branch-vs-base difference. Convergence noise, not a delta, and the converged run settles it. (f) DEFAULT FAST GATE: **ZERO DELTA** — 11 unique failed node ids on the branch and the IDENTICAL 11 at base `c3e8bda6` (8 failed + 3 errors, 787 passed on both), every one of them the pre-existing FFI-dependent set this file already accounts for on a box with no built `.so`: the six `distrib_la` contract cells, the `vcoul` import-isolation cell, the Si BSE anchor and the three `gw_jax` regression errors. Nothing this branch touches appears in that set. The new cells are census-class and correctly deselected from the default gate (2608 deselected on the branch against 2547 at base, the difference being exactly the 61 new ones). `tests/test_w_head_densify.py`, 61 cells, fixture-free and CPU-only — **61 passed** on WSL. **TWO IMPLEMENTATION FINDINGS the design did not anticipate, both caught by their own gates.** The re-attachment domain cannot be a geometric `\|q\| ≤ \|q − K\|` predicate: it needs a tie rule, ties are generic (an even densification factor puts fine q exactly on the coarse cell boundary at every face centre), and the first cut kept **9** fine q where 8 were due on fcc and **18** where 8 were due on a hexagonal cell — an over-count no norm-based check would have seen, since the extra points carry small heads. Membership is now coset arithmetic on the indices, where the count is exactly `[Λ_f : Λ_c]` with no tolerance, and the builder asserts it. And boundary q must SHARE their weight `1/k` rather than one representative winning: `q → −q` maps coset `c` to coset `−c` preserving `\|q\|`, so a lowest-index tie-break keeps `q` and drops `−q` and the head channel stops being even in q — which is what carries reciprocity `W_q = conj(W_{−q})` through the re-attachment. A 3×3×2 → 9×9×4 hexagonal case failed outright before the weights went in. **ONE CORRECTION TO THE DESIGN'S OWN CLAIM:** §1.4 says trigonometric interpolation does not preserve the head channel's zone sum. It does — the densifier is linear with a fixed R = 0 component, so `(1/N_q)Σ_q W(q)` is conserved identically. What it fails to do is REFINE the quadrature: it re-deposits the frozen coarse answer smeared as a Dirichlet kernel, so a finer grid buys nothing. The gate is written against the refinement statement, not the conservation one. Evidence: `SCHUR_BSE_DESIGN.md` §8; `/pscratch/sd/j/jackm/c1_ab` |
