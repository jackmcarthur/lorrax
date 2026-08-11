# OWNER ROW — closing the downfold's star symmetry by COMPLETION is uneconomical on a deck with few orbits, and the alternative is to select in orbit blocks (2026-08-10) — **DECISION TAKEN, IMPLEMENTED**

> **THE RULING, 2026-08-10, VERBATIM.**  "everything the user has input on
> they should be specifying in units of points, and we should be choosing the
> quantity of orbits that comes closest to that number of points without
> exceeding it."

**STATUS: CLOSED.**  The four questions this row put to the owner are answered
below under "The ruling, applied", each beside the part of the tree that now
carries it.  Everything above that heading is the row as it was written, kept
verbatim because it is the measurement the ruling was made on and a
measurement is worth more standing than edited.  Implementation:
`feat/downfold-orbit-floor-2026-08-10`, pushed, **NOT merged**.

---


This is a decision to be made by whoever owns the downfold's centroid selection. It
is not a defect report and it is not a change: the branch that carries this file
implements none of it, deliberately. What it does carry is the measurement that
makes the decision necessary, and enough of a design sketch that the ruling can be
made on numbers rather than on taste.

## The finding, in one paragraph

The downfold's kept centroid set has to be closed under the parent's symmetry
orbits before the child can be stored on the q wedge — that is the covariance
condition `gw/downfold.py` derives and `star_stability` measures. The repair the
tree already ships for a set that is not closed is **completion**: round the kept
set outward to whole orbits (`orbit_complete_keep`), then re-take the rank
certificate on the enlarged set. That repair was sized on a synthetic and looked
cheap. On a production centroid set it is not cheap; it is unavailable. Measured on
`si_bse_debug` on 2026-08-10, completing the selection that `mu_small = auto`
actually makes takes μ_S from 185 to 480 — **the entire parent basis**, which is
the same thing as not downfolding at all. The reason is arithmetic rather than
numerical: the parent's 480 centroids form only **11 orbits**, so an orbit-closed
subset has very few sizes available to it, and the pivot order lands on none of
them.

## The numbers

All from the owed-legs lane of 2026-08-10, workspace
`/pscratch/sd/j/jackm/owedlegs_0810/`. The deck is `si_bse_debug` with
μ_L = 480, 96 symmetry ops (48 spatial plus time reversal),
`downfold_rcond = 1.1e-6`, and a rank ceiling of 185 independent pair-density
directions. Every admissible μ_S from 1 to 185 was closure-tested against the
parent's own stored `sym_perm`, using the shipping pivoted-Cholesky kernel's actual
pivot order rather than a model of it.

| quantity | measured value | evidence |
|---|---|---|
| orbit-closure rate of the CUR selection | **0 of 185 = 0.0000** | `_logs/closure_p4.log`, `closure_auto.json` |
| ops violating closure at `mu_small = auto → μ_S = 185` | **94 of 96** | same |
| completion cost at that μ_S | **+295 centroids, μ_S 185 → 480** | same |
| completion cost over all open μ_S | min **35**, median **323**, max **387** | same |
| orbit census of the parent set | **11 orbits: 2 × 24 + 9 × 48 = 480** | `_logs/orbits.log`, `orbits.json` |

Two things follow from the census by arithmetic alone, and they are labelled as
derived rather than measured because that is what they are. First, an orbit-closed
subset is a union of orbits, so its size is `24a + 48b` with `a ≤ 2` and `b ≤ 9`;
the only such sizes at or below the ceiling of 185 are **24, 48, 72, 96, 120, 144
and 168** — seven values, and the measured closure rate of zero says the real pivot
order reaches none of them. Second, the largest orbit-closed μ_S this deck admits
under its own ceiling is therefore **168**, which is 91 % of the 185 directions the
rank criterion certifies and 35 % of the parent basis.

## What the shipping code does with this today

It refuses, correctly, and that is worth stating plainly because it means nothing
is silently wrong right now. `select_cur_centroids` completes outward and then
re-certifies against the ceiling, and on this deck 480 > 185, so it raises the
"the SYMMETRY-LEGAL mu_S nearest it is over the ceiling" error and stops. Its
suggested fix is to lower `mu_small` until completion lands at or under the
ceiling. That advice is right, and it is also a request that the user find the
orbit ladder above by bisection, one downfold at a time, without ever being told
that a ladder is what they are searching. Note also that this path is not reachable
on the driver today at all: `downfold_run` does not build the parent's centroid
source map, so `select_cur_centroids` is called with `sym_perm=None` and prints the
"orbit closure NOT CHECKED" absence instead. Nothing here is wrong; it is
unverified, and every child is written on the full BZ, which is correct and merely
larger than it needs to be.

## The design alternative: ORBIT-BLOCK-GREEDY selection

Instead of picking points and repairing the set afterwards, pick **orbits**. The
pivot loop takes one pivot per orbit and marks that pivot's whole orbit inactive, so
the delivered set is a union of orbits and is orbit-closed by construction; μ_S then
arrives in orbit quanta and no completion, no re-certification and no refusal is
needed. Completion is a repair applied to a selection that has already gone wrong;
this is the selection not going wrong.

**The important thing about this proposal is that the kernel already exists.**
`centroid/pivoted_cholesky.py` has had orbit mode all along: pass `orbit_id` (one
integer per candidate, equal for symmetry-equivalent candidates) to
`pivoted_cholesky_select` or to `prune_candidates_by_pivoted_cholesky` and it does
exactly the above — its own docstring promises the returned `keep_idx` is
"guaranteed orbit-closed under the sym group used to assign `orbit_id`". It is the
path the centroid *generator* uses, and it is why `si_bse_debug`'s 480-point parent
set is orbit-closed in the first place. The downfold does not use it:
`gw/downfold.py`'s `select_cur_centroids` calls the sharded select as
`select_step(G_rows, None, active_init)`, and that `None` is the `orbit_id`
argument. So the change being specced is one argument wide at the call site, and
the orbit labels themselves are a cheap derivation — the orbits are the connected
components of the action of `sym_perm`'s rows, which is the same table
`star_stability` already takes.

Do not confuse this with the *other* block-greedy already registered a few hundred
lines above in the same file ("take the top-b entries of one snapshot of d per
round"). That one is a latency optimisation, it is deliberately not built, and it is
about batching pivot picks to cut collective round trips. Orbit mode is about which
points are picked, it is built, and it is shipping on another path.

## What the owner has to rule on

**1. Which rank certificate governs μ_S in orbit mode.** This is the substantial
question and it has bitten this project before. In orbit mode the greedy select
deflates the Schur complement by one direction per orbit while removing all members
of that orbit from contention, so the `rank` it reports counts **orbits**, not
points. The downfold validates μ_S against a *point* ceiling — the eigenvalue rank
of the pool Gram, 185 here. Those two numbers are not comparable, and
`centroid/pivoted_cholesky.point_granularity_rank` exists in this tree precisely
because the confusion already shipped once: a gate passed at "42 of 42 directions
certified" on a centroid file holding 1908 points, and the ζ back-solve then
truncated to about 1450 modes per q, logged eight times a leg and read by nobody.
Whatever is decided, the downfold must not repeat that.

**2. Whether μ_S is allowed to be quantised at all.** On this deck the ladder has
seven rungs and its step is 24 centroids at the bottom and 48 at the top. A user who
asks for 185 gets 168. That is a defensible answer — degeneracy-closed band windows
work the same way, and `AGENT_PREAMBLE`'s standing ruling on band degeneracy is that
`strict` is the default and you do not loosen a criterion to make a gate pass — but
it changes what the `mu_small` deck key means, and it should change deliberately.
The natural spelling is that `mu_small` stays a request in centroids and the
resolver reports the rung it landed on, the same way the band-window resolver
reports the window it snapped to.

**3. Whether the quantisation costs anything physical.** Unknown, and it is the
cheapest thing on this list to find out. The measurement is two downfolds of the
same parent — one at the point-picked μ_S = 185 the driver chooses today, one
orbit-blocked at μ_S = 168 — compared on `eps_w`, the Pythagorean error bar the
downfold already prints per q. If the error bar barely moves, the quantisation is
free and this is an easy ruling. If it moves a lot, the small-orbit-count decks are
telling us they cannot be both compressed and wedge-stored, and that is worth
knowing before any code is written. **This leg has not been run and this row does
not predict its outcome.**

**4. Regeneration.** Orbit-blocked selection picks a different centroid subset than
the current point-picked one, so every existing downfolded child is in a different
basis than a re-run would produce. That is the same objection the registered
block-greedy carries, and it wants the same answer: a story for existing artifacts,
not just a flag.

## Status and scope

**OPEN, awaiting an owner ruling.** Nothing in this row is implemented on the branch
that carries it, and the branch changes no selection behaviour. The related repairs
that lane *did* make — `lorrax-downfold` running at P>1 at all, and
`star_stability.describe()` no longer printing the refuted "completion costs one
orbit's tail" consolation — are separate and are in the same commit series. The
measurement this row rests on lives with the amendment it settles,
`tests/known_failures/2026-08-10-downfold-qirr-star-stability.md`; the raw evidence
is `/pscratch/sd/j/jackm/owedlegs_0810/` (`closure_auto.json`, `orbits.json` and the
logs named in the table above), registered in
`/pscratch/sd/j/jackm/EVIDENCE_MANIFEST.md`.

---

# The ruling, applied (2026-08-10)

Branch `feat/downfold-orbit-floor-2026-08-10`, off `origin/main` `9709ff0b`,
pushed and **NOT merged**.  What follows answers the four questions in "What
the owner has to rule on", in that order, and then records the gates.

## The shape of the change

`select_cur_centroids` passed `None` where the sharded select wanted
`orbit_id`, exactly as this row said, and the change is that argument plus the
floor semantics around it.  The orbit labels are the connected components of
the action of `sym_perm`'s rows (`gw.downfold.centroid_orbit_id`, union-find,
the same table `star_stability` already takes), the kernel picks one pivot per
orbit and retires that orbit's whole membership, and the delivered set is the
union of the orbits of the longest PREFIX of the pivot order whose POINT total
does not exceed `mu_small`.

A prefix and not a knapsack, and the reason is the certificate rather than
convenience: the pivot order is a quality ranking produced by deflating the
Schur complement in that order, so skipping orbit *k* to fit a smaller orbit
*k+1* would use a residual computed under the assumption that *k* was taken.
The ruling asks for "the quantity of orbits" — a COUNT, in the order the
selection ranks them — and prefix totals are strictly increasing, so the
largest admissible count answers both readings at once.

**Orbit COMPLETION is off the selection path.**  `orbit_complete_keep` and
`star_stability` both remain: the first as an offline instrument for a keep set
that arrived from somewhere else, the second as the verdict
`select_cur_centroids` still takes on what it delivered — a guarantee that
lives in another module's loop is measured here rather than inherited.

## 1. Which rank certificate governs μ_S in orbit mode

**The eigenvalue rank of the pool Gram, in POINTS, at the REALIZED count** —
the knob-trap discipline this row warned about, unchanged.  Three numbers now
print with their units attached and the orbit one is labelled as not
comparable:

* `eigen_rank_pool` — POINTS.  The ceiling, resolved through
  `common/spectral_closure`, and the number the user's request is validated
  against.  The refusal above it fires on the number the user typed and can
  fire on nothing else, because `realized <= requested <= ceiling` holds by
  construction once the floor is inward.
* `select_rank` — **ORBITS** in orbit mode, and the report says so in the same
  breath, naming `centroid.pivoted_cholesky.point_granularity_rank` and the
  incident that instrument exists for (a gate passing at "42 of 42 directions
  certified" on a file holding 1908 points).
* `eigen_rank_kept` — POINTS, on the REALIZED set: `S_SS[keep, keep]` at q = 0.
  This is the one comparable to μ_S and it is the one the solve will act on.

The floor's own point-granularity validation is asserted rather than assumed
(`realized <= ceiling`), because "cannot fail by construction" is exactly what
the previous design believed about completion's cost before it was measured.

## 2. Whether μ_S is allowed to be quantised at all

**Yes, and the spelling is the one this row proposed**: `mu_small` stays a
request in centroids — the deck key and its type are unchanged — and the
resolver reports the rung it landed on, loudly, on every run:

```
  *** [downfold/select] ORBIT-FLOORED: mu_S requested 185 points -> REALIZED 168 (4 orbits: 48+48+48+24) ***
  [downfold/select] mu_small is a request in POINTS and the realized basis is the largest union of WHOLE
      symmetry orbits that does not exceed it — 17 of the 185 points you budgeted were not spent.  The next
      orbit in the pivot order holds 48, and 168 + 48 = 216 would exceed 185.
  [downfold/select] parent census: 11 orbits: 2 x 24 + 9 x 48 = 480.  Legal point counts at or below the
      ceiling 185: 24, 48, 72, 96, 120, 144, 168.
```

The last line is the direct answer to this row's complaint that the shipping
refusal "is a request that the user find the orbit ladder by bisection, one
downfold at a time, without ever being told that a ladder is what they are
searching".  The ladder is printed.  Both numbers are also stamped in the
child's `downfold_provenance` group (`mu_small_requested`, `mu_small`,
`n_orbits_kept`, `n_orbits_pool`, `parent_orbit_sizes`,
`selection_granularity`), so a reader who finds 168 in a deck that says 185 can
learn why without re-running anything.

**The direction is INWARD and it is stated in only one place.**  An earlier
draft of this work argued the floor was "deliberately opposite" to
`common/spectral_closure`'s snap-outward on the rank cut.  That contrast was
retired within the hour — a parallel lane is aligning `spectral_closure` to
floor as well, on this same ruling — and the code now states its own direction
and takes the ceiling AS RESOLVED, whatever resolves it.  A direction asserted
in two modules is a direction that will disagree with itself; there is a
comment at the site saying so.

## 3. Whether the quantisation costs anything physical

THE DECIDING LEG.  See "The deciding leg" below.

## 4. Regeneration

Unchanged in kind and now smaller in blast radius, and it is worth being exact
about who is affected.

* **Existing downfolded CHILDREN are in a different basis than a re-run would
  produce.**  That is true and unavoidable: a different selection rule picks a
  different subset.  The mitigation is that a child says so — every bundle
  carries `keep_idx` and now `selection_granularity` in its provenance, so
  "was this made under the point rule or the orbit rule" is a question an
  artifact can answer about itself rather than one answered from a date.
* **Existing PARENT centroid files are untouched.**  The generator's floor
  (below) changes what a NEW `--orbit` run delivers; it does not touch a file
  already written, and nothing in this branch regenerates one.
* **Nothing was regenerated by this lane.**  The gates below run against the
  shipped `si_bse_debug` 480-centroid set as it stands.

## The sweep: other user-facing counts that orbit-quantise

One other site, and it had the same defect pointing the other way.

| site | user-facing count | before | after |
|---|---|---|---|
| `gw/downfold.py::select_cur_centroids` | `mu_small` (deck key) | points, then completed OUTWARD to whole orbits — 185 → 480 on `si_bse_debug`, then refused | **FLOORED**: largest whole-orbit union ≤ the request, both numbers printed and stamped |
| `centroid/kmeans_cli.py::_prune` → `prune_candidates_by_pivoted_cholesky` | `N_c` (positional CLI arg) | orbit target `ceil(N_c · n_orbits / n_unique)`, then the delivered POINT count is Σ orbit_size over whatever the greedy picked — **can overrun `N_c`** | **FLOORED** via the new `n_point_budget=` argument: the pivot prefix is truncated to `N_c` points |
| `gw/downfold.py` `downfold_rcond`, `downfold_select_tol` | tolerances, not counts | — | not applicable: they are thresholds on a spectrum, not quantities bought |
| `mu_S` device padding (`padded_mu_extent`) | not user-facing | — | out of scope: the pad is exact-zero rows that never reach disk |

On the generator the orbit TARGET is deliberately left exactly as it was, so
that `refuse_unless_select_certified` and the rank gate in `main()` are still
stated against the same number; the floor can then only ever TRUNCATE, which
means the generator now delivers at most `N_c` points and never more, with
every existing refusal reading exactly as before.  `n_point_budget=None` is the
historical behaviour, so no other caller of that kernel changes.

## Gates

**WSL CPU, 1×1, both sides run, worktree pin proven by `__file__` before
measuring.**  Base `origin/main` `9709ff0b`: `tests/test_downfold.py` +
`tests/test_exciton_bands_downfold_dropin.py` + `tests/test_spectral_closure.py`
= **3 failed, 87 passed, 10 skipped** (100 collected); branch: **95 passed, 10
skipped** (105 collected).  The three base failures are the three cells that
asserted the retired completion behaviour, and they are rewritten rather than
deleted — the retirement is gated, not merely described:

| cell | what it now measures |
|---|---|
| `test_TRUE_a_mid_orbit_request_is_FLOORED_INWARD` | realized ≤ requested at every μ_S, realized is the LARGEST legal rung ≤ the request, the delivered set is orbit-closed, and the report carries BOTH numbers |
| `test_RED_TWIN_a_selection_that_would_exceed_the_budget_floors_LOUDLY` | **the red twin the ruling names.**  At μ_S = 11 the point-granular selection is asserted NOT orbit-closed first (so the cell is exercising the case the floor exists for), then orbit mode must come back smaller, closed, and must PRINT `ORBIT-FLOORED`, both numbers and the unspent budget |
| `test_the_floor_can_NEVER_exceed_the_rank_ceiling` | the knob trap, over a deliberately rank-deficient window: realized ≤ `eigen_rank_pool` at every admissible μ_S, any refusal is on the REQUEST and never on a number the floor invented, and `describe()` says ORBITS where it means orbits |
| `test_qirr_THE_COMPOSITION_...` (×3, §6) | **the composition.**  The covariance gate with the keep set coming from the SHIPPING selection at a request between rungs, instead of the hand-picked `QIRR_KEEP_CLOSED` every other cell in that section uses |
| `test_qirr_RED_TWIN_the_point_granular_selection_still_cuts_orbits` | the control arm: same Gram, same μ_S, `sym_perm=None` — the delivered set breaks closure and `child_unfold_tables` refuses it, so the floor is shown to have bought something |
| `tests/test_layering.py` | 80 passed — the new import edges cross no layer |

**FOUR REAL GPUs, and the shape asserted rather than assumed.**  Workspace
`/pscratch/sd/j/jackm/orbitfloor_0810/`, own allocation 56616165.

| leg | shape | result |
|---|---|---|
| `gates_gpu4` | the three suites, ONE process holding four `CudaDevice`s | **105 passed, 0 skipped, 0 failed** at `752a3d43`; `MESH_SHAPE probe: local_devices=4 global_devices=4 processes=1`, `resolve_mesh(): {'x': 2, 'y': 2}` |
| `gates_merged` | the same, on the tree REBASED onto `origin/main` 01d462e4 | **114 passed, 0 skipped, 0 failed** — +9 are main's own new drop-block cells |
| `df_floor` / `df_point` | `lorrax-downfold`, **4 ranks x 4 GPUs** | rc=0; `mesh {'x': 2, 'y': 2} on 4 device(s), 4 process(es)` |
| `df_floor_merged` | the same deck on the rebased tree | rc=0 in 30.2 s, **every number identical** — the floor, the census, the ladder, the verdict |

The GPU run has **zero skips** against WSL's ten: those ten are the
`liblorrax_ffi_host.so` driver-import cells, which run here.  Collected is
105 on both, so the green is over the same set and not a smaller one.

**A P=4 SHAPE CORRECTION THAT COST THIS LANE A ROUND, and it cuts both ways.**
`lx run -G 4 -n 4` on a PYTEST file places four independent single-GPU
sessions, each a 1x1 mesh, and reports a green that is four copies of P=1 — the
gates leg above is `-G 4 -n 1` for exactly that reason, with an in-leg probe
that exits 95 if it sees fewer than four global devices.  But the opposite is
true of these DRIVERS: `lorrax-downfold` and `bse.bse_jax` both REFUSE a
four-device single process (`resolve_mesh: mesh 2x2=4 != process_count()=1`;
`bse_loading._get_local_mesh_coords` raises `IndexError: index 0 is out of
bounds for axis 0 with size 0`), so their real P=4 shape is four PROCESSES at
`-G 4 -n 4`.  One rule does not cover both; assert the mesh in-leg either way.

The count is verified against the expected set rather than read off a green
bar: 100 collected on base, 105 on branch, +5 = three parametrised composition
cells, one composition red twin, and a net +1 in `test_spectral_closure.py`
(two cells replaced by three).

## THE DECIDING LEG — question 3, answered, and the answer is expensive

This is the measurement this row asked for and refused to predict: "two
downfolds of the same parent — one at the point-picked μ_S = 185 the driver
chooses today, one orbit-blocked at μ_S = 168 — compared on `eps_w`".  It has
now been run, at **four GPUs in four processes**, and the row's own alternative
outcome is the one that happened: *it moves a lot.*

Workspace `/pscratch/sd/j/jackm/orbitfloor_0810/`, logs `_logs3/`, tree
detached at `e8fe346a` with `git dirty-count: 0` printed by every rank.  Both
arms are `si_bse_debug`, μ_L = 480, window 0:20 symmetric, `mu_small = auto`,
`downfold_rcond = 1.1e-6`, mesh `{'x': 2, 'y': 2} on 4 device(s), 4
process(es)` — the production geometry, and the shape asserted in-log rather
than assumed.

**The two arms differ in one thing only: whether the parent carries a centroid
source map.**  Arm A reads the owed-legs lane's wedge-stored `parent_auto`,
which carries its own `sym_perm`, so the selection runs in orbits and floors.
Arm B reads `parent_full`, the same deck stored on the full BZ, which carries
no table, so the selection is point-granular — the historical behaviour, on the
same code, at the same commit.  Those two parents were measured BIT-IDENTICAL
through the downfold by the owed-legs lane, so the parent choice is a switch
for the selection rule and not a second variable.

| | arm A — ORBIT-FLOORED | arm B — point-picked (control) |
|---|---|---|
| μ_S requested (points) | 185 | 185 |
| μ_S realized (points) | **168** — 4 orbits, 48+48+48+24 | **185** |
| orbit closure of the delivered set | **CLOSED under all 96 ops** | UNMEASURED (no table) — an absence |
| eigenvalue rank of `S_SS[keep,keep]` at q=0 | **122 of 168** | **171 of 185** |
| retained rank of the transfer per q, min/med/max over 64 q | **120 / 123 / 124** | **171 / 173 / 175** |
| `eps_W(W0_qmunu)` min/median/max | 3.205e-02 / **4.319e-02** / 1.328e-01 | 7.232e-03 / **1.056e-02** / 1.341e-02 |
| `eps_W(V_qmunu)` min/median/max | 3.024e-02 / **5.179e-02** / 1.527e-01 | 7.533e-03 / **1.096e-02** / 1.691e-02 |
| leg wall | 409.6 s | 237.1 s |

**The symmetric-but-smaller basis is beaten, not matched: about 4x worse at the
median q and about 10x worse at the worst q.**  Both numbers are the printed
Pythagorean error bar, which the driver itself labels a tripwire rather than an
accuracy gate — the observable arm is reported separately below — but a
tripwire that moves by a decade is not a subtle effect.

**And the mechanism is measured rather than inferred, which is the part worth
carrying forward.**  It is not that 168 < 185; a 9 % cut in points does not buy
a 4x error bar.  It is that **an orbit-closed basis is inefficient in
directions per point on this deck**:

* 168 orbit-closed points carry **122** independent directions — 0.73 per point.
* 185 point-picked points carry **171** — 0.92 per point.

The reason is the same q = 0 group invariance that makes orbit mode possible in
the first place.  The selection Gram commutes with the whole group, so it block-
diagonalises by irrep, and a single orbit of 48 centroids contributes only as
many independent directions as that orbit's permutation representation contains
irreps of the Gram's support — nowhere near 48.  Taking whole orbits therefore
buys mostly REDUNDANCY.  The row's arithmetic said the ladder is coarse; the
measurement says the rungs are also *cheap in points and expensive in
directions*, and that second fact is the one that costs the error bar.

This retires, on numbers, the hope in this row's §3 that "if the error bar
barely moves, the quantisation is free and this is an easy ruling".  It does not
retire the ruling: the owner ruled on the INTERFACE — what a user-facing count
means and which way it rounds — and that ruling is right independently of what
any one deck's error bar does with it.  What the measurement settles is the
DECK question the row put beside it: *"the small-orbit-count decks are telling
us they cannot be both compressed and wedge-stored, and that is worth knowing
before any code is written."*  On `si_bse_debug`, with 11 orbits over 480
centroids, that is now the measured answer.

## AND THE WEDGE CHILD DOES NOT MATERIALISE ON THIS DECK EITHER

The composition this row's alternative was FOR — an orbit-closed selection makes
the child wedge-storable — holds in the algebra and fails on this deck, and the
driver now measures that per run instead of assuming it.

```
  [downfold/star] kept centroid set is ORBIT-CLOSED under 96 ops (168 centroids) — the child is wedge-storable.
  [downfold/star] CHILD WEDGE-STORABILITY GATE, on this run's own tensors: ...
  [downfold/star]   V_qmunu:  max rel 1.170e+00
  [downfold/star]   W0_qmunu: max rel 1.241e+00
```

Order one — the red twin's magnitude, not the 1.7e-15 the synthetic covariance
gate reads.  **Orbit closure is not the missing piece: it HOLDS** (the run says
so two lines above, under all 96 ops).  Closure is what gives the child an
unfold TABLE; it is not what makes the child's TENSORS unfold.  That needs
`T[q] = U^S T[i(q)] (U^L)†`, and `T` is built by a RANK-TRUNCATED solve.

**THE FIRST HYPOTHESIS WAS REFUTED BY THE INSTRUMENT WRITTEN TO TEST IT, and
that is recorded rather than quietly replaced.**  The retained rank runs 120 to
124 across the 64 q, which looks exactly like a truncation that moves within a
star — and a covariant `S_SS` has an identical spectrum at every star member, so
that would have made the pair-density Gram non-covariant and the retained band
window the culprit (the mechanism the band-degeneracy work of 2026-08-10 found
at 6x6x6).  `_star_rank_constancy` groups the per-q rank BY STAR and says:

> `[downfold/star] star-rank constancy: the retained rank IS constant on every
> star, so the truncation is not the mechanism and the disagreement is upstream
> of it`

The 120..124 spread is ACROSS stars, not within one.  So the band window is not
implicated and the solve is not implicated, and the honest state of this lane is
that **the mechanism is not identified.**

**Nor is it yet established that there is a mechanism to find.**  The route has
three parts that could each be wrong alone — the wedge-slot derivation, the
table restriction, and the unfold call — and an order-one number cannot tell
"the child is not covariant" from "this harness is wrong".  A CONTROL that can
is now in the gate and its result is reported below: the PARENT's own tensor was
stored on this wedge with these tables and unfolded by the reader before the
driver saw it, so slicing it back and unfolding it MUST return it, and both arms
run the same `_unfold_roundtrip_rel` so they cannot differ by a line of
plumbing.  **The control has now been read and it is the strongest possible pass:**

```
  [downfold/star]   CONTROL, the PARENT's own tensor through the SAME route
      (it was stored on this wedge with these tables, so it MUST return): max rel 0.000e+00
  [downfold/star] VERDICT: REFUTED (worst 1.241e+00 against tol 1e-09).
```

Bit-exact zero, not the reassociation floor — because on the control arm the two
routes really are the same arithmetic on the same bytes.  So the wedge-slot
derivation, the table restriction and the unfold call are all exonerated, and
**the child's order-one disagreement is a real covariance failure of the child,
not a defect in the instrument that found it.**

**The mechanism remains unidentified, and this lane says so rather than
choosing a story.**  What is excluded: the harness (control, 0.000e+00) and any
truncation that moves within a star (`star_rank_constancy`, constant on every
star).  What is NOT excluded, and is the strongest remaining candidate: the
covariance of the pair-density Gram itself.  Equal ranks across a star are
NECESSARY for a covariant `S_SS` and nowhere near sufficient — `S_SS[q]` and
`U S_SS[i] U†` can share a spectrum without being equal — so the constancy
result clears the rank SHADOW of covariance and leaves covariance open.  The
decisive next measurement is one gather and one unfold, and it is named in the
driver's own log: compare `S_LL[q]` against `U S_LL[i(q)] U†` directly.  **This
lane did not run it.**

Two consequences for whoever picks this up:

1. The gate is now a VERDICT (`CHILD_COVARIANCE_TOL = 1e-9`, chosen in the empty
   decades between the synthetic floor 1.7e-15 and the red twin 8.6e-01) and it
   prints the star-rank constancy measurement on a failure, so the next lane
   starts from a mechanism rather than from a number.  The first version of this
   gate narrated whatever it measured as "the reassociation floor"; it would
   have called 1.170e+00 a floor, and that is exactly the class of green this
   project's measurement discipline exists to refuse.
2. The child's tables are still stored beside the child with the verdict next to
   them (`downfold_provenance/child_unfold_tables/covariance_worst_rel`), and the
   log says in as many words that they must not be used to write a wedge child
   until the verdict passes.

## THE OBSERVABLE, which is the honest accuracy instrument and not eps_W

The driver says on every run that `eps_W` is a tripwire and that "the honest
accuracy instrument is the observable comparison against the parent".  So it was
taken, three arms, same deck, same flags, `_logs6/`.

**The band-degeneracy guard refused the first attempt on all three arms
identically** — `--n-val 4 --n-cond 4` cuts a multiplet on this deck — and named
its own fix.  The window was widened to the degeneracy-closed `4v8c` the guard
asked for, on all three arms together.  `--band-degeneracy snap` and `off` were
both available and neither was used: `AGENT_PREAMBLE`'s standing ruling is that
you never loosen that criterion to make a leg run, and a window chosen to make
a comparison possible is a window that is choosing the comparison's answer.

| arm | μ_S | lowest exciton (eV) | vs parent |
|---|---|---|---|
| parent | 480 | **3.575096** | — |
| point-picked | 185 | **3.226476** | **−348.6 meV** |
| orbit-floored | **168** | **none — the eigensolver did not converge** | no answer at all |

The 168 arm dies in FEAST's Ritz step with `numpy.linalg.LinAlgError:
Eigenvalues did not converge` (`bse_feast.py:1483`), on the same deck, the same
flags and the same four-process mesh on which both other arms return.  That is
reported as what it is — **a failure to produce an observable, not a measured
observable** — and it is not converted into a number.  It is consistent with the
error bar (that arm's `eps_W` is 4-10x the other's) and with the direction count
(122 independent directions against 171), and consistency is not proof: a FEAST
non-convergence has other possible causes and this lane did not separate them.

One more number that IS available on all three arms, because it comes from the
Lanczos bound rather than from the solve, and it says the same thing about
spectral distortion:

| arm | E_max (Lanczos) | vs parent |
|---|---|---|
| parent | 13.001 eV | — |
| point-picked 185 | 14.471 eV | +11 % |
| orbit-floored 168 | 19.139 eV | **+47 %** |

`E_min` is 0.677 eV on all three and discriminates nothing.

**The answer to the question the owner row asked in its §3 — "does the
symmetric-but-smaller basis beat or match the asymmetric-larger one?" — is
NEITHER.  It is beaten, on every instrument that returned a number, and on the
observable it did not return one.**

## What this lane is NOT claiming

* Not that orbit flooring is wrong.  The ruling is about what a user-facing
  count MEANS and which way it rounds, and that is right independently of what
  any deck's error bar does with it.  A user who asks for 185 points and is
  silently given 480 has been failed; a user given 168 with both numbers printed
  has been told the truth about a deck that cannot do better.
* Not that this generalises past `si_bse_debug`.  The economics are driven by
  the orbit census — 11 orbits over 480 centroids — and a deck with many small
  orbits has a finer ladder and a better directions-per-point ratio.  The
  measurement to make on a new deck is the one above and it is cheap.
* Not that the wedge child is unreachable.  The composition is exact in the
  algebra and is gated green on the synthetic; what failed is this deck, with
  the harness exonerated at 0.000e+00 and the mechanism open.
