# FIXED — the downfold built its transfer at −q and applied it at +q (2026-08-11)

**STATUS: ROOT CAUSE FOUND AND FIXED**, branch
`diag/wedge-child-composition-2026-08-10`, pushed, **NOT merged**.  This closes
the "mechanism is not identified" state that
`tests/known_failures/2026-08-10-downfold-orbit-economics-owner-row.md` left
open, and it convicts a two-day-old diff rather than any established machinery.

## The defect, in one paragraph

`gw/downfold.py::pair_density_gram` builds the pair-density Gram by calling the
ISDF fit's own kernel, `isdf.core.c_q_from_psi_sm`. That kernel returns the
Gram labelled by **−q**. Its own caller, the ISDF fit, is self-consistent in
that labelling and is correct; the downfold is the caller that has to agree
with something else — the restart tensors' q axis — because it multiplies this
Gram's transfer into `V_qmunu[q]`. Nobody checked the two against each other.
The transfer the downfold built was therefore `conj(T)`, the transfer belonging
to `−q`, and the congruence applied it to the tensor at `+q`, at every q with
`−q ≢ q`. On a 4×4×4 grid that is **56 of 64 q**. The fix is a q-axis relabel
in `pair_density_gram` — `negate_q_index`, a pure permutation, exact, and
correct for symmetric and asymmetric windows alike. `c_q_from_psi_sm` is NOT
touched, and no other caller of it exists.

Introduced by `0dd61f20` (2026-08-09), the downfold's own birth commit.
`pair_density_gram` has no caller outside `gw/downfold*.py` and its tests, so
the blast radius is the downfold and everything downstream of a downfolded
child.

## The convicting numbers

Deck `si_bse_debug`, μ_L = 480, μ_S = 168, window 0:20 symmetric,
`downfold_rcond = 1.1e-6`. Read against the child the orbit-floor lane actually
shipped, `/pscratch/sd/j/jackm/orbitfloor_0810/child_floor`. Evidence:
`/pscratch/sd/j/jackm/wedgechild_0811/`.

| measurement | value |
|---|---|
| the shipped child, reproduced by `conj(T) V Tᵀ` (the −q transfer) | **7.541e-05** on all 64 q (`W0`: 1.682e-04) |
| the shipped child, reproduced by `T V T^H` (the +q transfer) | **9.947e-01** — order one |
| wedge-composition covariance of the shipped child | **1.170e+00** / `W0` **1.241e+00** — the driver's own gate, reproduced to the digit by an independent host implementation |
| wedge-composition covariance of `T V T^H`, same deck/keep/rcond/tables | **3.721e-08** / `W0` **3.001e-08** |
| q unaffected | the **8** with `−q ≡ q`: stars `q_irr` = (0,0,0), (0,0,½), (0,½,½) |

The host chain reproduces the driver's transfer solve on every number the
driver prints — retained rank per q **elementwise identical** (120/123/124),
`eigen_rank_kept` 122 of 168 at q=0, `kappa_eff` 8.94e+05 against the driver's
8.945e+05 — so it is the same solve, and the only thing that differs is which
q the Gram belongs to.

Confirmed a second, independent way, with no symmetry machinery involved at
all: on an **asymmetric** window (the only shape that can see a leg mapping)
`c_q_from_psi_sm`'s output matches a dense double loop over the definition
**only after the q axis is reversed**.

## What this REFUTES, and each was measured, not argued

Every hypothesis that was live when this lane opened is dead, and the
measurements are cheap enough that they should be re-run rather than trusted.

* **The pair-density Gram is NOT non-covariant.** `S_LL[q]` against
  `U S_LL[i(q)] U†`, host-side on the parent's own stored tables: **3.109e-10**,
  and `S_SS` and `S_cross` the same. The band projector `P(k)` is covariant at
  **2.069e-15**. This was the decisive measurement the previous lane named, and
  it came back the opposite way to the steer.
* **The band window is NOT cutting a multiplet.** `[0,20)` has a minimum
  boundary gap of **1.678e-02 Ry** over all 64 k — 0 of 64 cut. (It is `[0,24)`
  that would cut, 7 of 64.)
* **The spectral-closure guard is INERT on this run**, so neither it nor its
  drop-block flip can be the mechanism: `df_floor.log:195` reads "the cut falls
  in a gap on all 64 q", minimum relative gap **1.374e-01**.
* **The unfold monomials are unitary.** All 96 op rows of `sym_perm` are
  bijections, on the parent (480) and on the child (168).
* **The child's unfold tables are NOT a bare permutation and are NOT rebuilt
  per q.** The tables stored beside the child are **bit-identical** to the
  parent's `sym_perm` and `L_table` restricted to `keep`, phases included.
* **The conditioning is not the mechanism either**, though it is close enough
  to be worth stating: the pinv's amplification of the Gram's 3.1e-10
  covariance residual accounts for the 3.7e-08 floor and nothing beyond it,
  and it scales with `rcond` exactly as a pinv perturbation should
  (`kappa_eff` 8.9e5 → 3.7e-08; 9.9e1 → 1.2e-09).

## Why a suite with a "matches the definition" cell stayed green

`tests/test_downfold.py::_dense_gram` is described in its own docstring as
"the slow honest way, straight off the definition". It carried **the same
q-sign error as the code it checks**: its prose said the left (m) window sits
at `k−q`, its code put it at `k+q`. Three cells compared the implementation
against it and agreed, because both were at `−q`.

This is the failure mode `AGENT_PREAMBLE`'s measurement discipline exists to
refuse, one level up from where it is usually caught: a reference written from
the same hand as the code is not a reference. The repair is a cell that could
not have been written that way —
`test_the_gram_is_labelled_by_PLUS_q_analytically` — which uses a **closed
form**: with one band and `psi(k, r_mu) = exp(2 pi i k R_mu)` the pair density
at momentum transfer q is `exp(2 pi i q R_mu)`, so
`S(q)[mu,nu] = nk * exp(2 pi i q (R_mu - R_nu))` exactly, and that is not
invariant under `q -> -q`. `_dense_gram` is corrected too, and the closed form
now arbitrates between it and the kernel rather than the two arbitrating each
other.

The red twin is the shipped call — the kernel's output with no relabel — and it
is held to the *shape* of the failure and not merely its size: it must agree
EXACTLY at the self-inverse q and miss by order one everywhere else. That
signature is what named the mechanism in the first place, and it is what
distinguishes it from a missing unfold phase, which would have tracked the
umklapp wrap count instead.

## What this poisons, and what it does NOT settle

**Every downfolded child produced between 2026-08-09 and this fix is wrong at
56 of 64 q**, so everything measured off one is void. In particular the
orbit-floored-vs-point-picked accuracy comparison recorded in the orbit
economics row — `eps_W` medians 4.319e-02 vs 1.056e-02, the point-picked arm's
**−348.6 meV** exciton error, the orbit-floored arm's FEAST non-convergence and
its +47 % Lanczos `E_max` — was taken with **both arms** carrying this defect.
It measured two flavours of the same bug and cannot be read as a statement
about orbit flooring. **That comparison was NOT re-run by this lane** (it is a
follow-on decision, and it is now cheap to re-take correctly).

It does explain the orbit-floored arm's non-convergence at least as well as the
direction count did, and this row does not claim to have separated them: a
transfer built at the wrong q is not a small perturbation, and an arm whose
`eps_W` is 4× the other's is the arm with less headroom to absorb it.

What this does **not** touch: the orbit floor itself, the ruling behind it, the
spectral-closure guard, `child_unfold_tables`, or `c_q_from_psi_sm`. The
economics finding — that an orbit-closed basis on this deck buys 0.73
independent directions per point against 0.92 — is a statement about the
selection Gram at q = 0, where `−q ≡ q`, so it is untouched by this defect and
stands.

## Gates

**WSL CPU, 1×1, both sides run, worktree pin proven by `__file__` before
measuring.** Base `origin/main` `7a7fe7a8`; branch
`diag/wedge-child-composition-2026-08-10`.

*(filled in below by the A/B and the P=4 leg)*
Suites: `tests/test_downfold.py` + `tests/test_exciton_bands_downfold_dropin.py`
+ `tests/test_spectral_closure.py` + `tests/test_layering.py`.

| arm | HEAD | result | collected |
|---|---|---|---|
| base | `7a7fe7a8` | **184 passed, 10 skipped** | 194 |
| branch | this tree | **187 passed, 10 skipped** | 197 |

Delta is exactly the three new cells — the closed-form q-labelling gate, its
red twin, and the relabel's involution/fixed-point check — and **zero
regressions**. The 10 skips are one reason on both sides, the documented WSL
`liblorrax_ffi_host.so` condition. The count is checked rather than read off a
green bar, and it caught a real defect in this lane's own method: the first
base arm was taken in a worktree that turned out to sit at `ad8d342f`, ten
commits behind, and reported a +17 delta. A stale arm is not a control
(`AGENT_PREAMBLE`, measurement discipline rule 4); the numbers above are from a
worktree checked out at `7a7fe7a8` with `git rev-parse HEAD` printed.

**FOUR REAL GPUs, FOUR PROCESSES, ON THE DECK — and the number was predicted
before the run.** Workspace `/pscratch/sd/j/jackm/wedgechild_0811/`, tree at
`0f44bc16`, whose `HEAD^{tree}` is `b877a2dd15e4e0ae415faa6c21118ac162a6c949`
— **identical to the pushed branch's tree**, so the patch-applied cluster tree
and the branch are the same bytes. `git dirty-count: 0` on all four ranks;
`mesh 2x2 over 4 device(s), 4 process(es)`; the deck is the orbit-floor lane's
own `df_floor.in` with only the output path changed, reading
`owedlegs_0810/parent_auto` READ-ONLY. `_logs/df_fixed3.log`, rc=0 in 52 s.

| quantity | before (orbit-floor lane) | after (this fix) | host-chain PREDICTION |
|---|---|---|---|
| composition gate, `V_qmunu` | **1.170e+00** | **3.729e-08** | 3.721e-08 |
| composition gate, `W0_qmunu` | **1.241e+00** | **3.004e-08** | 3.001e-08 |

The prediction was recorded from the host chain before the leg ran and is
matched to three digits on both tensors. Everything the run is supposed to hold
fixed did: μ_S requested 185 → realized **168** (4 orbits, 48+48+48+24),
ORBIT-CLOSED under all 96 ops, retained rank per q **120/123/124**, spectral
closure "cut falls in a gap on all 64 q", control **0.000e+00**.

**THE GATE STILL READS `REFUTED`, AND THE TOLERANCE IS NOT MOVED.** 3.729e-08
is above `CHILD_COVARIANCE_TOL = 1e-9`, so the verdict line is unchanged even
though the defect it was reporting is gone. That tolerance was "chosen in the
empty decades between the synthetic floor 1.7e-15 and the red twin 8.6e-01" —
but the synthetic gate runs at a condition number of order 20, and this deck
runs at `kappa_eff = 8.9e+05`. The pinv's perturbation is second order in the
condition number, so the honest floor here is the one measured in the sweep
above (8.9e5 → 3.7e-08, 9.9e1 → 1.2e-09, 2.0e1 → 3.9e-10), not 1e-15.
**1e-9 is unreachable on this deck at this rcond, by arithmetic and not by any
defect.** Loosening it to make the gate green is exactly what
`AGENT_PREAMBLE`'s standing rule forbids, so it is left where it is and the
question is handed over as an OWNER ROW: the tolerance should be a function of
the run's own `kappa_eff` rather than a constant, and until it is, this gate
cannot pass on a production-conditioned deck. **That is a separate defect from
the one this row fixes, and it is now quantified rather than guessed.**

> **DECISION TAKEN, 2026-08-11 — the owner ruled that it should scale.** In
> his words: *"make it scale if you think that is the most likely thing to be
> more robust to say 100x more atoms and centroids."* Implemented on branch
> `fix/child-covariance-tol-kappa-2026-08-11`; see the amendment at the foot
> of this row for what shipped and what it was measured against. The paragraph
> above stands as written — it is the record of the state the ruling was taken
> from, and nothing in it was retracted. **Note that this row's own sentence
> "the pinv's perturbation is second order in the condition number" is
> REFUTED by this row's own sweep**: the three points fit a log-log slope of
> **0.409**, not 2. The exponent that shipped came from the measurements.

**AND THE ERROR BAR MOVED, UPWARD, BECAUSE IT IS NOW MEASURING WHAT IT CLAIMS.**
`eps_W(W0)` min/median/max went from 9.983e-02 / **1.273e-01** / 2.391e-01
against the orbit-floor lane's 3.205e-02 / 4.319e-02 / 1.328e-01. That is not a
regression: `epsilon_w` contracts the tensor at q against the Gram at q, and
before this fix the Gram it was handed was at −q, so the pre-fix `eps_W` was not
the Pythagorean residual of anything. **Every `eps_W` recorded for any
downfolded child before 2026-08-11 is void**, including both arms of the
comparison in the orbit economics row. The real accuracy of this deck at
μ_S = 168 is ~13 %, not ~4 %.

**THE SUITES AT FOUR REAL GPUs.** `tests/test_downfold.py` +
`tests/test_exciton_bands_downfold_dropin.py` + `tests/test_spectral_closure.py`
in ONE process holding four `CudaDevice`s (`[inleg] jax device_count: 4`, so
`resolve_mesh()` builds a genuine 2×2 over real cards):
**117 passed, 0 failed, 0 skipped**, rc=0 in 81 s, `_logs/gates_gpu4b.log`.
Collected is checked against the expected set rather than read off a green bar:
WSL reports 107 passed + 10 skipped = **117 collected** on the same three
suites, and the 10 WSL skips are the documented `liblorrax_ffi_host.so`
driver-import cells, which run here. Same 117 both sides, so the green is over
the same set and not a smaller one.

Note the shape rule that applies in both directions on this branch, and it is
the one `orbitfloor_0810` paid a round for: for **pytest** the real P=4 shape is
`-G 4 -n 1` (one process, four devices), and for the **driver** it is
`-G 4 -n 4` (four processes) — `lorrax-downfold` refuses a four-device single
process. Both legs above are at their own correct shape and both assert it
in-log.

## Owed after this lane

1. ~~**The tolerance.**~~ **DECISION TAKEN AND IMPLEMENTED, 2026-08-11 —
   see the amendment below.** `CHILD_COVARIANCE_TOL = 1e-9` was unreachable at
   production conditioning; the owner ruled *"make it scale if you think that
   is the most likely thing to be more robust to say 100x more atoms and
   centroids"*, and it now does. Branch
   `fix/child-covariance-tol-kappa-2026-08-11`, pushed, NOT merged.
2. **The comparison this poisons.** The orbit-floored-vs-point-picked accuracy
   comparison must be re-taken; it is now cheap and it was NOT re-run here.
3. **A second deck.** `xbwin_0811` reports a BSE-window refit certification
   failing 7.719 meV with ~89 % attributed to a downfolded child (μ 960→191)
   against 0.858 meV for the un-downfolded parent on the identical route, and a
   per-q error ordering that REARRANGES between child and parent. That is the
   signature of this defect and not of a wrap phase; the predictor to correlate
   is **`−q ≡ q`**, not the umklapp wrap count. On their grid the q that should
   be clean are exactly the ones fixed by inversion (Γ and the zone-boundary
   half-lattice points); Σ is not one of them, which is consistent with Σ being
   that child's worst point. Checkable against their existing per-q numbers
   without re-running anything.

---

## Amendment, 2026-08-11 — the tolerance now scales with `kappa_eff`

This closes owed item 1 above. Branch
`fix/child-covariance-tol-kappa-2026-08-11` at `2c7a7417`, off `origin/main`
`0578bc89`, pushed and **not merged**. Evidence
`/pscratch/sd/j/jackm/covtol_0811/`.

### What the gate does now

The child-covariance gate compares the run's own measured covariance residual
against

    tol(kappa) = max(1e-9, 4.0 * 8.3e5 * kappa**0.41 * eps_mach)

which is a module-level function, `gw.downfold_run.child_covariance_tol`.
There is no CLI flag, no environment variable and no deck key that reaches it,
which is the same discipline the refit-window certification tolerance keeps.
`CHILD_COVARIANCE_TOL` survives as the **floor**, so a well-conditioned
synthetic is gated no more loosely than it was before this ruling, and the
floor is reached below about `kappa = 2`, which no real solve sees.

### Where the exponent came from, which is the part worth arguing about

Not from theory. The three-point rcond sweep recorded higher up this page —
`kappa_eff` 2.0e1 to 3.9e-10, 9.9e1 to 1.2e-09, 8.9e5 to 3.7e-08 — has a
least-squares slope in log-log of **0.4087**, and 0.41 is what ships. That
number contradicts this row's own prose, which reasons that the pseudo-inverse
perturbation is "second order in the condition number". The prose is a
hypothesis and the sweep is a measurement, and where they disagree the
measurement wins; a tolerance fitted to an exponent of 2 would have been three
decades too loose at production conditioning and would have cost the gate most
of its discrimination for no reason anyone had measured.

The extrapolation error also runs in the safe direction, which matters because
the owner's question was explicitly about a hundredfold larger system. The
measured local slope falls as `kappa` rises — 0.70 across the first decade of
the sweep and 0.38 across the last four — so a single global 0.41 opens the
gate slightly faster than the data actually does out at large `kappa`.

The coefficient `8.3e5` (in units of machine epsilon) is simply the smallest
constant for which the fitted power law covers all three filed points; the
three implied constants are 5.14e5, 8.21e5 and 6.06e5. The **4x safety factor
is a separate named constant** rather than being folded into that coefficient,
so a reader can audit the fit and the margin apart from one another. Four is
modest on purpose: all three points come from one deck at three values of
`rcond`, so deck-to-deck spread is genuinely unmeasured, and 4x covers the
1.6x spread of the fit itself with room over.

### `kappa_eff` is a measurement and never a knob

The gate is handed the achieved amplification the transfer solve already
computes — the max over q of `sigma_max / sigma_min_kept` from the same
`common.rank_criterion` reports that the `[downfold/solve]` banner prints — so
the number in the verdict line is the number in the solve's own banner. It is
not estimated inside the gate and it cannot be supplied by a deck. A caller
that fails to supply it gets the old absolute floor **and a verdict line that
says out loud that the tolerance was not scaled**, because an absence is not a
measurement and a gate that invented a `kappa` in order to open itself would
be precisely the loosening this row refuses.

Because the verdict is now a comparison against a run-dependent number, the
residual alone is no longer readable after the fact, so the bundle records the
pair. `downfold_provenance` gains `kappa_eff_per_q`, and the child's table
group gains `covariance_kappa_eff`, `covariance_tol` and
`covariance_verdict` beside the `covariance_worst_rel` it already carried.
Provenance version goes 2 to 3; the change is purely additive.

One thing deliberately did **not** scale: the gate's control. The control
slices the parent's own tensor and unfolds it — a permutation and a
unit-modulus phase, with no pseudo-inverse anywhere on that route — so its
honest floor is a few ulp at any conditioning. Scaling it with a `kappa` it
does not depend on would blunt the one check that separates "this harness is
wrong" from "the child is not covariant".

### What this was verified against

On the deck this row is about, the arithmetic is now decided: at
`kappa_eff = 8.945e+05` the tolerance is **2.031e-07**, so the fixed deck's
3.729e-08 and 3.004e-08 both read **PASS**, with 5.4x of margin. The same
deck's pre-fix numbers, 1.170e+00 and 1.241e+00, read **REFUTED** at that same
tolerance and miss it by **5.8e+06**. That pair is the whole claim: the gate
stopped reporting `kappa_eff` and did not stop reporting breakage.

The red twin the ruling demanded is not an arithmetic one. It is a live
synthetic run through the driver's own gate function at production-scale
conditioning: an ill-conditioned wedge parent whose keep-block solve reaches a
**measured** `kappa_eff` of **7.054e+05**, with the transfer built from the
Gram belonging to `-q` and applied at `+q` — the shipped defect this row
convicts, the bare `conj(T)` construction. It lands at a covariance of
**4.877e-01** and prints REFUTED, clearing the scaled tolerance by
**2.6e+06**. The correct arm of the same synthetic, at the same conditioning,
sits at 1.135e-11 and passes.

That synthetic cannot demonstrate the scaling's *necessity*, and the cell says
so rather than letting a green imply it: its parent is covariant to machine
precision by construction, so its correct arm would have passed at 1e-9 too.
The necessity is a property of a real pair-density Gram, which carries a
3.1e-10 covariance residual of its own for the pinv to amplify, and that
evidence is this deck's and is filed above.

### Gates

**WSL CPU A/B, both arms run, worktree pin proven by `__file__` before
measuring.** Suites `tests/test_downfold.py` +
`tests/test_exciton_bands_downfold_dropin.py` + `tests/test_spectral_closure.py`
+ `tests/test_layering.py`.

| arm | HEAD | result | collected |
|---|---|---|---|
| base | `0578bc89` | **187 passed, 10 skipped** | 197 |
| branch | `2c7a7417` | **194 passed, 10 skipped** | 204 |

The delta is exactly the seven new cells and there are zero regressions. The
10 skips are the same documented WSL `liblorrax_ffi_host.so` condition on both
sides. The base arm was taken in a worktree checked out at `0578bc89` with
`git rev-parse HEAD` printed, because a stale arm is not a control and this
row already paid for that lesson once.

**THE SUITES AT FOUR REAL GPUs, ONE COMBINED LEG.** Same four suites in ONE
process holding four `CudaDevice`s — `DEVICE_COUNT 4`, `PROCESS_COUNT 1`,
`resolve_mesh()` to `{'x': 2, 'y': 2}`, all asserted in-leg before pytest was
allowed to start — on `nid001005`, tree at `41dde3e2` with `git dirty-count:
0` and `gw.downfold_run.__file__` printed from the worktree:
**204 passed, 0 failed, 0 skipped**, 126.6 s, `_logs/gates_p4b.log`. Collected
is checked against the expected set rather than read off a green bar: WSL
reports 194 passed + 10 skipped = **204 collected** on the same four suites,
and the 10 WSL skips are the documented `liblorrax_ffi_host.so` driver-import
cells, which run here. Same 204 both sides, so the green is over the same set.

**AND THE FIRST ATTEMPT AT THAT LEG WAS THROWN AWAY, because it measured
nothing.** It reported 3 failed / 191 passed / 10 skipped, and the failures
were an XLA HLO-verifier `RET_CHECK` inside `slice_psi_to_centroids` in the
mesh-invariance subprocess cells. The cause was in the harness, not the tree:
the leg ran under **jax 0.5.3.dev from `/opt/jax`**, outside the supported
window, because `LX_BASE_MODULE=lorrax_J070` was exported inside the step's
own `env.sh` — which runs after `lx` has already chosen the container image.
The variable has to be set on the LOGIN NODE, before `lx run`. Note the tell
and how close it came to being banked as a real red: the ten skips carried a
`jax-support.version` REFUSED banner rather than the usual
`liblorrax_ffi_host.so` reason, and a lane that had only counted the skips
would not have seen the difference. With the module pinned host-side the same
tree, same node and same command give 204/204.

**THE DECK ITSELF, AT FOUR REAL GPUs AND FOUR PROCESSES.** The claim "the
fixed production deck now prints PASS" is not left as arithmetic on filed
numbers — the deck was re-run, because the driver's wiring of `kappa_eff` into
the gate is three lines that no unit cell reaches. `lorrax-downfold` on the
orbit-floor lane's own deck, reading `owedlegs_0810/parent_auto` READ-ONLY,
`mesh {'x': 2, 'y': 2} on 4 device(s), 4 process(es)`, provenance version 3
printed in-leg. It reproduces this row's numbers to the digit — μ_S 480 to
168, ORBIT-CLOSED under 96 ops, `kappa_eff` max **8.945e+05**, `V_qmunu`
**3.729e-08**, `W0_qmunu` **3.004e-08**, control **0.000e+00** — and the
verdict line now reads

    [downfold/star] VERDICT: PASS (worst 3.729e-08 against tol 2.031e-07).

The bundle it wrote carries the pair the verdict needs to stay readable:
`kappa_eff_per_q` over 64 q spanning 7.292e+05 to 8.945e+05, and beside the
child's tables `covariance_kappa_eff = 894486.8`, `covariance_tol =
2.0311e-07`, `covariance_verdict = PASS`. Evidence
`/pscratch/sd/j/jackm/covtol_0811/_logs/driver_p4.log`, bundle
`child_covtol/tmp/isdf_tensors_168.h5`.
---

## CLOSED 2026-08-11 — rows 2 and 3 discharged, and the predictor is confirmed at 109x

Owed rows 2 and 3 above were both taken by the re-measurement lane; the numbers
are in `tests/known_failures/2026-08-11-qsign-recut-verdicts.md` and the
evidence is `/pscratch/sd/j/jackm/qsign_recut_0811/`. Row 1, the `kappa_eff`
tolerance, was closed by the amendment above this section (the
`kappa**0.41` scaling), which merged separately; the two amendments
crossed in flight, which is why this one still calls it open.

**Row 2 — the comparison this poisons.** Re-taken. The orbit-floored-168 vs
point-picked-185 verdict is unchanged in direction and stronger in margin, but
the −348.6 meV became −42.6 meV and the FEAST non-convergence cleared entirely.
The amendment is on
`tests/known_failures/2026-08-10-downfold-orbit-economics-owner-row.md`.

**Row 3 — the second deck, and the predictor this row named.** Confirmed, and
more sharply than the row dared claim. Correlating `xbwin_0811`'s existing
per-Q errors against **−q ≡ q** costs nothing and was done first: on the
LEVEL of the errors the predictor does **not** separate the four Q cleanly —
the two self-inverse Q are elevated 8–26x over the parent, which no q-sign
story explains — but on the **ORDERING** it separates completely. The mean rank
shift between the parent's per-Q ordering and the child's was **0.50 for the
self-inverse Q against 2.50 for the non-self-inverse Q**: Σ went parent-best to
child-worst (4→1) and W went parent-worst to child-3rd (1→3), while L stayed at
2 and X moved only 3→4. **The rearrangement this row flagged as the signature
was carried entirely by the q the defect could act on**, and the residual
elevation at the self-inverse Q is the child's own basis truncation
(μ 191 against 960), not a wrap phase.

Re-running the certification on a fixed child then arbitrated it outright:

| Q | −q ≡ q | before (P=4) | after (P=4) | factor |
|---|---|---|---|---|
| X (0, ½, ½) | **yes** | 1.07403 | 0.40475 | 2.7x |
| (¼, ¾, ½) | no | 1.27721 | 0.22834 | 5.6x |
| L (½, ½, ½) | **yes** | 3.05016 | 2.59335 | 1.2x |
| Σ (¼, ½, ¼) | no | **6.81943** | 0.01654 | **412x** |

Mean improvement **1.9x at the self-inverse Q against 209x at the rest — a
109-fold separation on the predictor** — and the worst point moved from Σ to L,
which is self-inverse and was therefore never corrupted. Σ, this row's named
suspect, fell to within a factor of two of the 0.01 meV gate itself. **The
predictor was right and the umklapp wrap count was not needed.**

The certification still fails, at 2.593 meV against 0.01 meV, and that residue
is the windowed-ζ' representability error of the refit route — visible on the
un-downfolded parent at 0.858 meV, where there is no downfold to blame. That is
a different, already-registered question
(`2026-08-10-exciton-bands-offgrid-Q-is-slab-only.md`), and the dense exciton
band plots stay OWED behind it.
