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
