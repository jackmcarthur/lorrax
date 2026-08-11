# AMENDMENT — THE DEGENERACY-CLOSURE GUARD BECOMES A `distrib_la` FEATURE, OPT-IN AND OFF (2026-08-10)

**The owner's line was "i guess this could even be a feature wired directly
into distrib_la for both cholesky and lu?". It could, and this is it — but
the shape of the answer is not the shape of the question, and that is the
first thing to read.**

`2026-08-10-spectral-cut-closure.md` gave the monorepo a guard on spectral
cuts: a rank truncation may not stop inside a degenerate block, because the
retained set is a subspace and half a degenerate block is a
symmetry-arbitrary slice of an eigenspace. That amendment swept `src/` and
`services/` and wired six sites. This one takes the same criterion to the
`distrib_la` service's rank-revealing operations.

## The finding, which changes the shape of the work

**`distrib_la` had no rank-revealing operation.** Its three ops — `eigh`,
`cholesky`, `solve_lu` — factor and solve. Not one of them returns a rank,
and nothing in the package cut a spectrum. `factor`/`solve` hands back an
opaque `FactorToken`; `plan(...).batched(A)` hands back a factor or a
handle. So the guard could not be "wired into" an existing decision,
because there was no decision there.

That is worth stating rather than quietly working around, because it is
also why the previous amendment's sweep found nothing here: it was looking
for cuts, and there were none. The hazard in this package is not a cut that
lands mid-block today — it is that every consumer that wanted a rank off
one of these factors had to make the cut itself, in its own module, with no
guard and no shared idea of what the rank-revealing spectrum even was.

So the surface is new, and it is the smallest one that is honestly
implementable.

## What was built

`distrib_la.rank_cut(op, values, *, rcond | n_keep, closure=None)` is the
one place in the package where a rank cut is made, and it returns a
`RankCut` — the proposal, the delivered rank, whether the guard fired, the
mode it ran under, and the full cluster dict. Reachable three ways, all the
same function: the free function, `Plan.rank_cut` and
`FactorToken.rank_cut`, the last two binding the op and the mode so a call
site cannot disagree with its own plan.

Two helpers say what a factorization's rank-revealing spectrum *is*, which
is the part every consumer was previously re-deriving:

* `cholesky_pivot_spectrum(L)` → `|diag(L)|²`. With the pivot order fixed,
  that is exactly the Schur-complement diagonal a pivoted Cholesky compares
  against its tolerance, so it is the pivot spectrum of every Cholesky
  route in the package under one expression — `native`, `native2d`,
  cuSOLVERMp through `cholesky_handle_to_natural_L`, SLATE through
  `SlateLowerL.to_jax_lower`. **Squared, and it is load-bearing.** A
  relative-gap criterion is not invariant under a square root: two pivots
  agreeing to 1e-6 have diagonals agreeing to ~5e-7, so reading the
  diagonal instead of the pivot moves every block boundary by a factor of
  two in tolerance and invents degeneracies. There is a red twin built at
  exactly the tolerance where the two answers differ.
* `lu_rank_spectrum(LU)` → `|diag(U)|`, unsquared, because `A = P L U`
  already lives on the operator's scale. `getrf` packs `U` on and above the
  diagonal, so no unpacking is needed.

Both are pure `jnp` on the last two axes, so one call serves a tile and a
stack, and either can sit inside a caller's own `jit`.

## The copy, and the cell that holds it

`services/distrib_la/src/distrib_la/closure.py` is a **copy** of
`common/spectral_closure`'s host face, not an import. Services are
import-isolated from `src/` by charter and by gate
(`test_distrib_la_import_isolation.py` asserts through `sys.modules` AND
`sys.path`), and the whole worth of that isolation is that it has no
exceptions. The kernel is ~90 lines of stdlib arithmetic; a copy is cheaper
than an import edge.

A copy with nothing holding it is a fork waiting to happen. **The hold is a
consistency cell in `tests/test_spectral_closure.py`** — the lorrax side,
because it is the only side that can see both. It runs thirteen shared
synthetic spectra (a featureless power law, planted blocks at four
positions, exact ties, a null tail, an indefinite spectrum, a block open at
the bottom, the armF gap, a singleton, an empty spectrum, an all-equal run)
through **both** `cluster_at_cut` implementations at **every** cut and
asserts the dicts agree **field for field** — not merely on the rank,
because the messages and the owner-facing numbers are built out of
`gap_rel`, `members`, `span_rel` and the two `kappa`s, and a copy that
agreed only on the rank would still print different evidence for the same
event. A second cell does the same for `resolve_spectral_cut` under each
explicit mode, comparing raise-or-not by exception class NAME. A third
asserts the copy imports nothing from `common`, so the cells above cannot
pass vacuously.

## The one deliberate divergence: the default is `off` here and `snap` there

`common/spectral_closure` defaults to `snap`, and the argument is
arithmetic: a snapped spectral cut admits directions within `rtol` of ones
already retained, so `κ_eff` moves by under one part in 10⁴ and refusing
that by default would be refusing the repair. **That argument transfers
here unchanged.** What does not transfer is who may change a route.

This service's resolution and route semantics are **certified surface**: a
resolved backend name is a promise that every guard passed, and the worst
measured defect in this tree was a silent route change that ran to
completion with `rc=0` and a QP gap of **−161 eV**. A guard that arrived
switched on would change the rank a shipped operation hands back without
its caller asking, which is the same shape of event. So this round the
guard is **opt-in and OFF at every entry point**, one kwarg or one
environment variable away.

The dial is `LORRAX_DISTRIB_LA_CLOSURE` and it is deliberately **not**
`LORRAX_SPECTRAL_CLOSURE`. Sharing a name would mean a run that armed the ζ
fit's guard silently armed this one — a caller getting a different rank out
of `cholesky` because of a variable set for an unrelated seam. Two guards
with different defaults must not share one dial, and there is a cell that
fails if they ever do.

**OWNER ROW: whether the default here should follow its sibling to `snap`
is the owner's call, exactly as the strict-vs-snap row already open on the
monorepo guard is. It is a single constant,
`distrib_la.closure.DEFAULT_MODE`, and
`test_the_default_mode_is_off_and_that_is_the_opt_in_claim` is where the
change gets registered.**

## The certified-surface claim is a check, not a promise

`test_arming_the_closure_changes_no_resolution_fact` builds the same plan
under all three modes and compares `op`, `requested`, `backend`, `n`, both
shardings, `donates` and `batched_route` — parametrized over `native` (the
`off` arm) and `native2d` (the arm that has real shardings and a real
route, so the claim is not half-measured). The scan cache key is asserted
not to see the mode at all, two ways: `scan_signature` has no such
parameter, and two plans differing only in closure produce the same
signature for the same operands. On a real 2×2 emulated mesh,
`test_the_closure_does_not_change_what_native2d_computes` compares the
factor itself across the three modes with `np.array_equal` — bit-identity,
not a tolerance, because the guard runs on host after the call returns and
anything else would mean it had reached inside.

## Gates

Every check ships with the case where it returns FALSE. No exceptions.

**L-a, `services/distrib_la/tests/test_distrib_la_closure.py`, 125 cells,
0 skipped** — no mesh, no devices, no `.so`.

* **the criterion**, TRUE (a cut at each of the three interior positions of
  a planted 4-member block snaps to the block edge, and the cut it moves to
  is itself clean — the difference between fixing the problem and
  relocating it) against FALSE (a featureless power law fires at **every
  one** of its 23 interior cuts being silent, not a sample).
* **the three modes**: `snap` names the block, the move and `κ_eff` either
  side; `strict` names the rank that works and both escape hatches; `off`
  is silent, changes nothing, and still returns the full info dict. The
  FALSE arm is silent in all three.
* **DISARMED ≠ clean.** `RankCut.describe()` says DISARMED under `off` and
  reports the gap under an armed mode. Measurement rule 10: for an opt-in
  guard, "no news" reading like "a good number" is the single most likely
  way to be misled by it.
* **monotonicity swept** over 30 `(block position, cut)` pairs — the guard
  never returns a smaller rank — paired against `strict` refusing on
  **exactly** the same 30, so neither mode is a constant function of its
  input.
* **the null pad is never swallowed.** Exactly-equal values are trivially
  degenerate, so a walk that did not stop at a null pair would swallow a
  distributed factorization's mesh pad and make the retained rank a
  **function of the device count** — a different answer on a 2×2 than on a
  4×4 for the same matrix.
* **two ratchets, one with a red twin**: no entry point spells a mode
  literal as a default (AST, not a line regex — the first draft was a
  regex and it fired on the prose), and every wired entry point still
  calls the guard by name.
* the closure module is asserted to import nothing but `__future__`,
  `dataclasses` and `typing`, so the vocabulary reads with no jax and no
  `.so` — the property `BACKEND_CHOICES` has, for the same reason.

**L-b, `test_distrib_la_emulated_mesh.py`, 6 new cells** on a real 2×2 of
emulated host devices: `native` and `native2d` — two genuinely different
algorithms with different reduction orders — agree on the pivot spectrum to
1e-12 AND both fire AND snap to the same rank (all three, because agreement
on the rank alone is also satisfied by two backends that both failed to
fire); the same pair agree that a generic HPD's cuts are all clean under
`strict`; arming changes not one bit of the factor.

**L-c, `test_distrib_la_multiproc.py`** — `check_closure_agrees_across_backends`
plus a `_CLI_CELLS` row. This is the claim that makes the guard a service
feature rather than a utility function: **every backend must snap the cut
to the same place**, or the retained subspace becomes a function of which
library was compiled. It plants a known block in a diagonal HPD (whose
Cholesky is exactly `diag(√d)`, so the spectrum every backend must
reproduce is analytic), and defers by name any backend this machine cannot
resolve — an absence, never a pass.

## Suite A/B, both sides run

WSL CPU, worktree pin proven before measuring.
`services/distrib_la/tests` + `tests/test_spectral_closure.py`:

| arm | passed | skipped | failed |
|---|---|---|---|
| branch | 240 + 64 | 62 | 6 |
| base `ad8d342f` | 108 + 46 | 62 | 6 |

Delta is **exactly the 132 + 18 new cells**, zero regressions, skips
unchanged. **The 6 failures are PRE-EXISTING and identical on both arms** —
`test_distrib_la_contract.py` loader cells that need a pinned `.so`, which
WSL has none of; they are not this branch's and they are not new.

## The cluster leg

EVIDENCE_PATH_PLACEHOLDER

## Limits, stated rather than buried

* **`FactorToken` has no `rank_spectrum()`, and this round it could not.** A
  token's factor is opaque by design — ScaLAPACK's `ipiv` and LU,
  cuSOLVERMp's raw buffer, SLATE's `SlateLowerL` are block-cyclic on a
  specific grid — and pulling `diag` out of one in place is a distributed
  gather with a per-backend layout rule each. Real work, none of it
  measured. A caller materialises the factor through that backend's own
  documented route and hands the spectrum in. **OWED, and named here so
  nobody has to rediscover that it is missing.**
* **No device face.** `common/spectral_closure` carries a pure-`jnp`
  `snap_keep_outward` because the ζ cut lives inside a jitted kernel that
  never brings its eigenvalues to host. No rank decision in this service
  does: `plan()` is eager by construction and a rank decision here is made
  after a factorization has returned. So there is one face, it is host, and
  `strict` raises where it is called — no deferred refusal, no host
  callback, nothing to keep in sync. If a device-side rank cut ever appears
  here, `snap_keep_outward` is what gets copied next and the consistency
  cell is where it gets pinned.
* **`|diag(U)|` is a weaker rank revealer** than a rank-revealing QR or an
  SVD, and can miss a near-null space a column-pivoted factorization would
  expose (Kahan's matrix is the classical counterexample). The guard
  bounds where a cut may land; it does not claim the spectrum being cut is
  the right one to read. Where that distinction matters, cut an `eigh`
  spectrum instead. Said in `lu_rank_spectrum`'s own docstring, at the
  place it is used.
* **Nothing in LORRAX calls this yet.** The feature is the surface plus its
  guard; migrating existing consumers onto it is a separate lane, and this
  one deliberately did not touch `src/` beyond the consistency cell.
