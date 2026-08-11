# OWNER ROW — closing the downfold's star symmetry by COMPLETION is uneconomical on a deck with few orbits, and the alternative is to select in orbit blocks (2026-08-10) — **SPEC ONLY, NOTHING IMPLEMENTED**

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
