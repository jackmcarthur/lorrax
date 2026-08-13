"""Farming a Σ pass at WINDOW-GROUP granularity, and refusing a farm that lost a leg.

WHY THIS FILE EXISTS
--------------------
The multipole Σ pass had exactly one split axis — ``mpa_pole_subset`` —
so an ``n_p = 8`` pass had eight schedulable pieces.  The 2026-08-10 farm
measurement put numbers on what that costs: sixteen GPUs run a pass at
**26.5 % utilisation** because half of them have nothing to do and the
eight that work are only 53 % busy (the dearest pole is 3.95x the
cheapest), and the wall does not move at all from P=8 to P=16 —
1 583 s either way, BFC@0.85, sha ``66bc3fb1``.

The pass loop already contains a finer axis and has since it was written.
``sigma_pass.run_pass_branch`` walks the branch's WINDOW GROUPS, calling
``ppm_sigma._integrate_tau_windows_for_branch`` once per group; each group
carries its own ``idx_B`` selection and its own ``Omega_q`` operand, the
tau accumulator sums groups additively, and ``plan_branch_groups`` already
computes every group's tau-node count before a single node is dispatched.
A (pole, branch, group-range) leg is therefore the same kind of object as
a pole leg, one level down — and it needs NOTHING from inside the
integrator, which is the file the handoff marks as unchanged and which
this lane does not touch.  On the production deck that is 918 groups
across the eight poles instead of eight poles, which is enough pieces to
balance sixteen legs by dispatch count.

WHAT IS AND IS NOT CHANGED BY SPLITTING HERE
--------------------------------------------
Not changed: which groups exist, which modes are in them, which certified
rule serves each, how many tau nodes each runs, and in which order a
process walks the ones it owns.  ``plan_branch_groups`` is called with the
identical arguments and its output is FILTERED, never re-planned — a leg
integrates a subsequence of the pinned walk, exactly as ``pole_subset``
gives a leg a subsequence of the pinned pole order.

Changed: where the floating-point sum is cut.  A branch's Σ is a left fold
over its windows (``_MemoryTileSink.consume_window`` does
``total += window``), so cutting the fold into legs and adding the leg
cubes is a RE-ASSOCIATION of that sum.  This is the same statement the
pole farm already carries — ``fit_driver.accumulate_over_pole_passes`` is
exact in exact arithmetic and not bit-exact in floating point — and it is
answered the same way: **the pinned ascending recombination is the
canonical result**, and the combiner measures what the other orders would
have cost rather than assuming it.  What this module adds is that the
ascending order is now over (pole, branch, group) addresses rather than
over poles, and that a leg's address range is stamped in the cube it
writes so the combiner can check coverage at the granularity the split
actually happened at.

THE INTEGRITY HALF, WHICH IS NOT A PERFORMANCE FEATURE
-------------------------------------------------------
The sixteen-leg fit farm of 2026-08-10 lost leg 12 at placement
(``lx_pool: timeout running scontrol show step 56575336``, sixteen legs
having polled slurm's control plane inside a 48-second window).  Its q
window [48, 52) was never fitted.  Nothing downstream could tell: the farm
wall printed, fifteen cost reports printed, and **a fit store missing four
q looks exactly like a fit store**.  The same hole exists on the pass side
and is worse there, because a partial Σ cube has the shape, dtype and
units of a finished one and differs from it by a smooth few tens of meV —
which is the size of the effect the campaign is measuring.

So a farm run by this module declares its legs BEFORE it launches them.
:func:`write_manifest` writes every leg's spec and its expected output
path; :func:`refuse_incomplete` is what the merge calls first, and it
names the missing legs — leg id, the range it owned, and the path that is
not there — rather than summing what happened to arrive.  The check is
kind-agnostic on purpose: a ``pass`` leg owns a group range and a ``fit``
leg owns a q range, and the failure they guard is one failure.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re

__all__ = [
    "BRANCH_KEYS",
    "CENSUS_FORMAT",
    "MANIFEST_FORMAT",
    "FarmIncomplete",
    "balance_legs",
    "branch_key",
    "census_rows_from_groups",
    "check_partition",
    "format_group_subset",
    "full_plan_digest",
    "group_plan_digest",
    "group_plan_digest_from_rows",
    "group_range",
    "merge_census_files",
    "plan_digest_rows",
    "parse_group_subset",
    "plan_fit_legs",
    "read_census",
    "read_manifest",
    "refuse_incomplete",
    "select_branch_groups",
    "universe_from_census",
    "write_census",
    "write_manifest",
]


#: The four branch keys, in the order ``ppm_windows._iter_branches`` yields
#: them.  A KEY AND NOT AN ORDINAL: a deck whose ω grid is one-signed
#: yields two branches rather than four, and an ordinal would then name a
#: different branch in the census than in the leg.  ``(space,
#: neg_omega_half)`` is the branch's physical identity and it is what the
#: window builder itself reads, so it is what a leg address is written in.
BRANCH_KEYS = ("pos_cond", "pos_val", "neg_cond", "neg_val")

CENSUS_FORMAT = "lorrax-mpa-pass-census/1"
MANIFEST_FORMAT = "lorrax-farm-manifest/1"

_SUBSET_ITEM = re.compile(
    r"^\s*(?P<pole>\d+)\.(?P<branch>[a-z_]+)\s*:\s*"
    r"(?P<lo>\d+)\s*-\s*(?P<hi>\d+)\s*/\s*(?P<total>\d+)\s*$")


class FarmIncomplete(RuntimeError):
    """A farm whose declared legs did not all land.

    Its own class because the caller that has to act on it is a merge
    step, and "some legs are missing" is the one failure that must never
    be caught alongside "this file is malformed" and turned into a
    smaller-but-plausible answer.
    """


def branch_key(space, neg_omega_half):
    """``("cond", False) -> "pos_cond"`` — a branch's stable address."""
    key = f"{'neg' if bool(neg_omega_half) else 'pos'}_{str(space)}"
    if key not in BRANCH_KEYS:
        raise ValueError(
            f"branch_key: (space={space!r}, neg_omega_half="
            f"{neg_omega_half!r}) is not one of the four Σ branches "
            f"{BRANCH_KEYS}.  A leg addressed to a branch that does not "
            f"exist would integrate nothing and report success.")
    return key


def plan_digest_rows(groups):
    """``(name, n_modes, n_tau)`` per group — what the partition digest is of.

    Split out from :func:`group_plan_digest` so a plan READ BACK FROM A
    FILE can be fingerprinted from its header, without materializing the
    hundreds of megabytes of index sets the groups themselves carry.  Both
    callers hash the identical byte stream, which is the point: a loaded
    plan and a freshly computed one are comparable because there is one
    digest and not two.
    """
    return [(str(g.name), int(g.n_modes),
             int(sum(int(w.n_tau) for w in g.windows))) for g in groups]


def group_plan_digest_from_rows(rows):
    """The partition digest of ``(name, n_modes, n_tau)`` rows."""
    h = hashlib.sha256()
    for i, (name, n_modes, n_tau) in enumerate(rows):
        h.update(f"{i}|{name}|{int(n_modes)}|{int(n_tau)}\n".encode())
    return h.hexdigest()[:32]


def group_plan_digest(groups):
    """A fingerprint of one branch's planned groups, in planner order.

    THE ANTI-STALE-CENSUS GATE.  A leg is told "integrate groups [12, 25)
    of pole 3's ``pos_val`` branch" by a manifest that was balanced from a
    census taken earlier.  If anything moved the planner's output between
    the two — a different store, a different ω grid, a different sha — the
    leg would happily integrate a range of groups that is no longer the
    range the manifest meant, every leg would succeed, and the merged Σ
    would be a smooth, finite, wrong number.  So the census records what it
    saw and the leg checks what it sees against it.

    The digest covers the group's name, its live-mode count and its tau
    node count — the three things that change when the partition changes.
    It deliberately does NOT cover the pole values themselves; the store
    identity is checked by the partial cube's own ``mpa_partial_fit_store``
    stamp, and a digest over eighty million poles would cost more than the
    plan it is guarding.
    """
    return group_plan_digest_from_rows(plan_digest_rows(groups))


def full_plan_digest(groups):
    """A fingerprint of EVERY number the integrator reads out of a plan.

    :func:`group_plan_digest` is the partition digest and is deliberately
    cheap: it answers "are these the same groups?", which is the question a
    manifest's ranges depend on.  This one answers a different and stricter
    question — "is this the same plan, to the last bit?" — and it exists
    because the plan is now an ARTIFACT that can be written once and loaded
    by sixteen legs.  The claim that a loaded plan is the plan the planner
    would have computed is a claim about the τ nodes, the weights, the
    masks, the reference energies, the prefactors and the selections, not
    about the group count; so the gate that establishes it hashes those.

    It is not a replacement for the partition digest and does not become
    one: it costs a pass over the index sets, which is why the per-leg
    check remains the cheap digest and this is what the verification leg
    runs.
    """
    import numpy as np

    h = hashlib.sha256()
    for i, g in enumerate(groups):
        selector = getattr(g, "selector_bounds_B", None)
        membership = np.ascontiguousarray(
            selector if selector is not None else g.idx_B)
        suffix = (f"bounds|{membership.dtype.str}|{membership.size}"
                  if selector is not None else
                  f"{membership.dtype.str}|{membership.size}")
        h.update(f"G{i}|{g.name}|{int(g.n_modes)}|{float(g.b_mass)!r}|"
                 f"{tuple(int(x) for x in g.field_shape)}|{g.provenance}|"
                 f"{suffix}\n".encode())
        h.update(memoryview(membership).cast("B"))
        op = np.asarray(g.omega_operand)
        h.update(f"|operand:{op.dtype.str}{op.shape}\n".encode())
        for j, w in enumerate(g.windows):
            h.update(
                f"W{j}|{w.name}|{float(w.E_ref_A)!r}|{float(w.E_ref_B)!r}|"
                f"{int(w.omega_sign)}|{w.project}|{float(w.prefactor)!r}|"
                f"{w.mask_B_mode}|{w.mask_B_threshold!r}|{w.crossing_kind!r}|"
                f"{w.max_error!r}|{w.provenance!r}\n".encode())
            for name in ("t", "alpha"):
                arr = np.ascontiguousarray(
                    np.asarray(getattr(w.nodes, name), dtype=np.complex128))
                h.update(f"|{name}{arr.shape}".encode())
                h.update(memoryview(arr).cast("B"))
            mask = np.ascontiguousarray(np.asarray(w.mask_A, dtype=bool))
            h.update(f"|mask_A{mask.shape}".encode())
            h.update(memoryview(mask).cast("B"))
    return h.hexdigest()[:32]


# ---------------------------------------------------------------------------
#  The deck key: which (pole, branch, group range) triples one leg owns
# ---------------------------------------------------------------------------

def parse_group_subset(raw):
    """``"0.pos_cond:0-37/112"`` -> ``{(0, "pos_cond"): (0, 37, 112)}``.

    ``None``/empty means every group of every branch this process walks,
    which is the pole-farm behaviour and the single-process production
    route.

    THE TOTAL IS PART OF THE ADDRESS AND IS NOT DECORATION.  ``0-37`` on
    its own is a range that silently clips against whatever the planner
    happens to return; ``0-37/112`` is a claim about the partition the
    manifest was balanced from, and :func:`select_branch_groups` refuses
    when the planner disagrees with it.  That is the only cheap place to
    catch a manifest that has gone stale, and the failure it catches is a
    coverage error — the one class of defect on this path that produces a
    finite, smooth, entirely plausible self-energy.
    """
    text = str(raw or "").strip()
    if not text:
        return None
    spec = {}
    for tok in text.replace(";", ",").split(","):
        if not tok.strip():
            continue
        m = _SUBSET_ITEM.match(tok)
        if m is None:
            raise ValueError(
                f"mpa_group_subset: {tok!r} is not a group range.  The "
                f"spelling is '<pole>.<branch>:<lo>-<hi>/<total>' with "
                f"<branch> one of {BRANCH_KEYS} and [lo, hi) a half-open "
                f"range into that branch's planned groups, e.g. "
                f"'0.pos_cond:0-37/112'.  Items are comma separated.")
        pole = int(m.group("pole"))
        bkey = m.group("branch")
        lo, hi, total = (int(m.group("lo")), int(m.group("hi")),
                         int(m.group("total")))
        if bkey not in BRANCH_KEYS:
            raise ValueError(
                f"mpa_group_subset: {bkey!r} is not a Σ branch; the four "
                f"are {BRANCH_KEYS}.")
        if not (0 <= lo < hi <= total):
            raise ValueError(
                f"mpa_group_subset: {tok!r} names the range [{lo}, {hi}) of "
                f"{total} groups, which is not a non-empty range inside "
                f"that branch.  An empty leg is a leg that reports success "
                f"having integrated nothing.")
        if (pole, bkey) in spec:
            raise ValueError(
                f"mpa_group_subset: pole {pole} branch {bkey} appears twice "
                f"in one leg's spec.  A group summed twice is a finite, "
                f"smooth, plausible Σ that no Hermiticity or shape check "
                f"can see, so it is refused rather than merged.")
        spec[(pole, bkey)] = (lo, hi, total)
    if not spec:
        return None
    return spec


def format_group_subset(spec):
    """The canonical spelling of a parsed spec — sorted, so it round-trips."""
    items = sorted(spec.items(),
                   key=lambda kv: (kv[0][0], BRANCH_KEYS.index(kv[0][1])))
    return ",".join(f"{p}.{b}:{lo}-{hi}/{tot}" for (p, b), (lo, hi, tot) in items)


def group_range(spec, *, pole, bkey, n_groups):
    """``(lo, hi, total)`` for one (pole, branch), or ``None`` if unowned.

    ``spec is None`` — the unsplit walk — owns everything, so it returns
    the whole range.  Split out so the plan-loading route can learn what a
    leg owns from the header of an artifact, before it reads any of it,
    and still be checked by exactly the code below.
    """
    if spec is None:
        return (0, int(n_groups), int(n_groups))
    got = spec.get((int(pole), str(bkey)))
    if got is None:
        return None
    return (int(got[0]), int(got[1]), int(got[2]))


def check_partition(*, n_groups, digest_got, pole, bkey, total, lo, hi,
                    digest=None, source="the planner"):
    """The two anti-stale refusals, in one place because there are two callers.

    A leg either PLANS its groups or LOADS them from a plan artifact, and
    the failure both routes have to refuse is identical: the partition this
    leg is looking at is not the partition its manifest's ranges were
    written against.  One copy of the refusal, so the two routes cannot
    drift into checking different things — the load route is newer and
    would have been the one to drift.
    """
    if int(n_groups) != int(total):
        raise ValueError(
            f"mpa_group_subset: this leg was told pole {pole} branch {bkey} "
            f"has {total} window groups and would integrate [{lo}, {hi}) of "
            f"them, but {source} returned {int(n_groups)}.  The manifest "
            f"this leg is running under was balanced from a DIFFERENT "
            f"partition, so its ranges no longer name the groups it meant — "
            f"and every leg would still succeed, leaving a merged Σ that is "
            f"smooth, finite and wrong by the size of whatever moved.  "
            f"Re-take the census against this store and this sha, re-balance, "
            f"and re-run the farm; do not widen this check.")
    if digest is not None and str(digest_got) != str(digest):
        raise ValueError(
            f"mpa_group_subset: pole {pole} branch {bkey} came back from "
            f"{source} with {int(n_groups)} groups as the manifest said, but "
            f"their names, mode counts and tau-node counts fingerprint to "
            f"{digest_got} against the census's {digest}.  Same count, "
            f"different partition — which is the stale-census failure "
            f"that a count alone cannot see.")


def select_branch_groups(groups, *, pole, bkey, spec, digest=None):
    """The groups of one (pole, branch) this leg owns, in planner order.

    Returns ``(selected, lo)`` — the subsequence and the index it starts
    at, which is what the partial cube stamps so the combiner can check
    coverage.  ``spec is None`` returns everything, which is the unsplit
    walk and is bit-identical to it because it IS it.

    A (pole, branch) the spec does not name returns ``([], 0)`` and the
    caller skips the branch entirely — it does not plan it, it does not
    integrate it, and by linearity its contribution to this leg is zero.
    """
    rng = group_range(spec, pole=pole, bkey=bkey, n_groups=len(groups))
    if rng is None:
        return [], 0
    lo, hi, total = rng
    if spec is not None:
        check_partition(
            n_groups=len(groups),
            digest_got=(group_plan_digest(groups) if digest is not None
                        else None),
            pole=pole, bkey=bkey, total=total, lo=lo, hi=hi, digest=digest)
    return list(groups)[lo:hi], lo


# ---------------------------------------------------------------------------
#  The census: what the planner would do, measured before anything is run
# ---------------------------------------------------------------------------

def census_rows_from_groups(groups, *, pole, bkey, branch_tag):
    """One census row per planned group, carrying its dispatch count.

    ``n_tau`` is the sum of the group's windows' node counts, which is
    exactly the number of tau dispatches integrating that group will
    cost.  It is the balancing weight because it is the only per-group
    quantity the measured cost is proportional to: the 2026-08-09 profile
    established that a tau node costs 0.130 - 0.165 s INDEPENDENTLY of the
    group's mask fraction, so a group's cost is its node count and not its
    mode count.
    """
    rows = []
    for i, g in enumerate(groups):
        rows.append({
            "pole": int(pole),
            "branch": str(bkey),
            "branch_tag": str(branch_tag),
            "index": int(i),
            "name": str(g.name),
            "n_modes": int(g.n_modes),
            "n_tau": int(sum(int(w.n_tau) for w in g.windows)),
            "n_windows": int(len(g.windows)),
        })
    return rows


def write_census(path, rows, *, fit_store, n_p, sha="", digests=None,
                 extra=None):
    """Write one census file — the planner's own account of a pass's work."""
    doc = {
        "format": CENSUS_FORMAT,
        "fit_store": str(fit_store),
        "n_p": int(n_p),
        "sha": str(sha),
        "written_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "digests": dict(digests or {}),
        "rows": list(rows),
    }
    if extra:
        doc.update(dict(extra))
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return str(path)


def read_census(path):
    with open(path) as f:
        doc = json.load(f)
    if str(doc.get("format")) != CENSUS_FORMAT:
        raise ValueError(
            f"read_census: {path} declares format {doc.get('format')!r}, "
            f"not {CENSUS_FORMAT!r}.")
    return doc


def merge_census_files(paths):
    """Fold per-pole census files into one, refusing a disagreeing set.

    The census is farmed the same way the pass is — one leg per pole — so
    the whole-deck census is eight files.  They must agree on the store and
    on ``n_p``, and no two of them may claim the same pole: a census that
    covered a pole twice would balance legs against a work estimate that
    does not exist, and one that covered it zero times would leave that
    pole out of the manifest entirely, which is the silent-subset failure
    this module exists to close.
    """
    rows, digests, stores, n_ps, shas, seen = [], {}, set(), set(), set(), {}
    plans = {}
    for p in sorted(str(x) for x in paths):
        doc = read_census(p)
        stores.add(str(doc["fit_store"]))
        n_ps.add(int(doc["n_p"]))
        shas.add(str(doc.get("sha", "")))
        for key, dig in (doc.get("digests") or {}).items():
            digests[key] = dig
        for key, info in (doc.get("plans") or {}).items():
            prev = plans.setdefault(key, info)
            if str(prev.get("address")) != str(info.get("address")):
                raise ValueError(
                    f"merge_census_files: {key} has two plan artifacts with "
                    f"different addresses ({prev.get('address')} and "
                    f"{info.get('address')}).  An address is a digest over "
                    f"every planner input, so two of them for one "
                    f"(pole, branch) means these census files were taken "
                    f"against different inputs and the balance struck from "
                    f"them would name group ranges of two different "
                    f"partitions.")
        for r in doc["rows"]:
            owner = seen.setdefault((int(r["pole"]), str(r["branch"]),
                                     int(r["index"])), p)
            if owner != p:
                raise ValueError(
                    f"merge_census_files: pole {r['pole']} branch "
                    f"{r['branch']} group {r['index']} is claimed by both "
                    f"{owner} and {p}.  Two censuses of the same pole "
                    f"balance a farm against work that will be done once, "
                    f"which is how a leg comes to be missing from the "
                    f"manifest in the first place.")
            rows.append(r)
    if len(stores) != 1 or len(n_ps) != 1:
        raise ValueError(
            f"merge_census_files: these census files do not describe one "
            f"calculation — stores {sorted(stores)}, n_p {sorted(n_ps)}.")
    if len(shas) != 1:
        raise ValueError(
            f"merge_census_files: these census files were taken at "
            f"different shas {sorted(shas)}.  The planner is part of the "
            f"source tree, so a census and the legs it balances have to "
            f"come from one tree.")
    return {
        "format": CENSUS_FORMAT,
        "fit_store": stores.pop(),
        "n_p": n_ps.pop(),
        "sha": shas.pop(),
        "written_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "digests": digests,
        "plans": plans,
        "rows": sorted(rows, key=_row_sort_key),
    }


def _row_sort_key(r):
    """THE PINNED WALK, written as a sort key.

    Pole ascending (the store's own axis and the accumulation direction
    ``fit_driver.pole_pass_order`` pins), then branch in the order
    ``_iter_branches`` yields, then the planner's own group order.  Every
    ordering in this module — the census, the balance, the leg order, the
    recombination — is this one, because a farm whose merge order is not
    the walk order is a farm whose answer depends on which leg finished
    first.
    """
    return (int(r["pole"]), BRANCH_KEYS.index(str(r["branch"])),
            int(r["index"]))


def universe_from_census(census):
    """``{(pole, branch): n_groups}`` — what a complete farm has to cover."""
    universe = {}
    for r in census["rows"]:
        key = (int(r["pole"]), str(r["branch"]))
        universe[key] = max(universe.get(key, 0), int(r["index"]) + 1)
    for (pole, bkey), n in universe.items():
        have = {int(r["index"]) for r in census["rows"]
                if int(r["pole"]) == pole and str(r["branch"]) == bkey}
        if have != set(range(n)):
            raise ValueError(
                f"universe_from_census: pole {pole} branch {bkey} has "
                f"{n} groups by its highest index but the census carries "
                f"{sorted(have)}.  A census with a hole in it balances a "
                f"farm that cannot be complete.")
    return universe


# ---------------------------------------------------------------------------
#  The balance: 918 groups into N dispatch-balanced legs
# ---------------------------------------------------------------------------

def _greedy_parts(weights, cap):
    """Contiguous greedy fill at capacity ``cap``; returns the cut points."""
    cuts, acc = [], 0
    for i, w in enumerate(weights):
        if acc and acc + w > cap:
            cuts.append(i)
            acc = 0
        acc += w
    return cuts


def balance_legs(census, n_legs, *, min_leg_tau=1):
    """Partition the pinned walk into ``n_legs`` dispatch-balanced legs.

    THE PARTITION IS CONTIGUOUS IN THE PINNED WALK, and that is a
    decision rather than a convenience.  A contiguous leg owns a run of
    consecutive addresses, so its merge position is unambiguous, its spec
    is three integers per branch instead of a list, and — the reason that
    matters — the recombination order is the walk order, which is what
    keeps the split's re-association as small as a split's re-association
    can be.  It also means a leg touches as few poles as the balance
    allows, and a pole it touches is a 2.6 GB slab it has to read.

    The weight is ``n_tau``, the group's tau-dispatch count, because the
    measured per-node cost does not depend on the group's mask fraction.
    The objective is the minimum achievable maximum leg weight over
    contiguous partitions, found by binary search on the capacity — exact,
    not a heuristic, and deterministic for a given census.
    """
    rows = sorted(census["rows"], key=_row_sort_key)
    if not rows:
        raise ValueError("balance_legs: the census has no groups in it.")
    n_legs = int(n_legs)
    if not (1 <= n_legs <= len(rows)):
        raise ValueError(
            f"balance_legs: {n_legs} legs asked of a walk with "
            f"{len(rows)} groups.  A farm cannot have more legs than the "
            f"pass has schedulable pieces — that is the whole finding this "
            f"module answers, one level down.")
    w = [max(int(r["n_tau"]), int(min_leg_tau)) for r in rows]
    lo, hi = max(w), sum(w)
    while lo < hi:                       # minimal feasible capacity
        mid = (lo + hi) // 2
        if len(_greedy_parts(w, mid)) + 1 <= n_legs:
            hi = mid
        else:
            lo = mid + 1
    cuts = _greedy_parts(w, lo)
    bounds = [0] + cuts + [len(rows)]
    # The greedy may finish under budget; spend the remaining legs on the
    # heaviest parts so the farm uses every device it was given.
    while len(bounds) - 1 < n_legs:
        widths = [(sum(w[bounds[i]:bounds[i + 1]]), i)
                  for i in range(len(bounds) - 1)
                  if bounds[i + 1] - bounds[i] > 1]
        if not widths:
            break
        _, i = max(widths)
        half, acc = sum(w[bounds[i]:bounds[i + 1]]) / 2.0, 0
        cut = bounds[i] + 1
        for j in range(bounds[i], bounds[i + 1] - 1):
            acc += w[j]
            cut = j + 1
            if acc >= half:
                break
        bounds = sorted(set(bounds + [cut]))
    legs = []
    for i in range(len(bounds) - 1):
        chunk = rows[bounds[i]:bounds[i + 1]]
        spec, labels = {}, []
        for r in chunk:
            key = (int(r["pole"]), str(r["branch"]))
            cur = spec.get(key)
            idx = int(r["index"])
            spec[key] = (min(cur[0], idx) if cur else idx,
                         max(cur[1], idx + 1) if cur else idx + 1,
                         0)
        universe = universe_from_census(census)
        for key in sorted(spec, key=lambda k: (k[0], BRANCH_KEYS.index(k[1]))):
            lo_i, hi_i, _ = spec[key]
            spec[key] = (lo_i, hi_i, int(universe[key]))
            labels.append(f"pole {key[0]} {key[1]}[{lo_i},{hi_i}) of "
                          f"{universe[key]}")
        legs.append({
            "id": f"leg{i:02d}",
            "kind": "pass",
            "group_subset": format_group_subset(spec),
            "poles": sorted({int(r["pole"]) for r in chunk}),
            "n_groups": len(chunk),
            "n_tau": int(sum(int(r["n_tau"]) for r in chunk)),
            "range_label": "; ".join(labels),
        })
    return legs


def plan_fit_legs(n_q, n_legs):
    """The fit farm's legs — q windows, declared the same way.

    THE SAME MECHANISM, AND ON PURPOSE.  The fit farm is what lost leg 12
    and its q window [48, 52) on 2026-08-10, and the reason nothing noticed
    is not that the fit is different from the pass; it is that neither of
    them wrote down what it was going to produce before it produced it.
    A fit leg is a (kind, id, range, expected output) row exactly as a pass
    leg is, so :func:`refuse_incomplete` serves both and there is one
    completeness check on this path rather than two that drift.
    """
    n_q, n_legs = int(n_q), int(n_legs)
    if not (1 <= n_legs <= n_q):
        raise ValueError(
            f"plan_fit_legs: {n_legs} legs over {n_q} q is not a partition.")
    legs = []
    for i in range(n_legs):
        q_lo = i * n_q // n_legs
        q_hi = (i + 1) * n_q // n_legs
        legs.append({
            "id": f"fit{i:02d}",
            "kind": "fit",
            "q_lo": q_lo,
            "q_hi": q_hi,
            "n_q": q_hi - q_lo,
            "range_label": f"q [{q_lo}, {q_hi})",
        })
    covered = sorted(q for leg in legs for q in range(leg["q_lo"], leg["q_hi"]))
    if covered != list(range(n_q)):
        raise ValueError(                       # pragma: no cover - arithmetic
            "plan_fit_legs: the q windows do not cover [0, n_q) exactly once.")
    return legs


# ---------------------------------------------------------------------------
#  The manifest, and the refusal that is the point of it
# ---------------------------------------------------------------------------

def write_manifest(path, legs, *, kind, fit_store, n_p=None, sha="",
                   out_dir=None, out_suffix=".h5", census=None, inputs=None,
                   extra=None):
    """Declare every leg and its expected output BEFORE the farm launches.

    The manifest is written first and read by the merge.  That ordering is
    the whole mechanism: a leg that never started leaves a row with no file
    beside it, which is a refusal; a farm that was never declared leaves
    nothing to check against, which is how [48, 52) went missing.

    ``inputs`` DECLARES WHAT THE FARM READS, on the same terms.  Since
    2026-08-10 a leg does not plan its own window groups: it loads them
    from a plan artifact addressed by the planner's inputs, so the farm's
    result depends on files that are not its own outputs.  A declared
    input is a row with an id, a label and a path, probed by
    :func:`refuse_incomplete` exactly as a leg's output is, because "the
    plan this farm ran against is gone" and "a leg did not land" are the
    same failure seen from the two ends — in both cases what is on disk no
    longer says what was computed.
    """
    rows = []
    for leg in legs:
        row = dict(leg)
        row.setdefault("kind", kind)
        if "output" not in row:
            if out_dir is None:
                raise ValueError(
                    "write_manifest: a leg has no 'output' and no out_dir "
                    "was given, so the manifest cannot say what to look "
                    "for — which is the only thing it is for.")
            row["output"] = os.path.join(str(out_dir),
                                         f"{row['id']}{out_suffix}")
        rows.append(row)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(
            f"write_manifest: leg ids are not unique ({ids}).  Two legs "
            f"sharing an id share an output path, so one silently "
            f"overwrites the other and the farm still looks complete.")
    input_rows = [dict(r) for r in (inputs or [])]
    for r in input_rows:
        if not r.get("id") or not r.get("output"):
            raise ValueError(
                f"write_manifest: declared input {r!r} has no id or no path.  "
                f"An input that cannot be named cannot be refused by name, "
                f"which is the only thing declaring it achieves.")
    doc = {
        "format": MANIFEST_FORMAT,
        "kind": str(kind),
        "fit_store": str(fit_store),
        "n_p": None if n_p is None else int(n_p),
        "sha": str(sha),
        "n_legs": len(rows),
        "total_n_tau": int(sum(int(r.get("n_tau", 0)) for r in rows)),
        "written_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "inputs": input_rows,
        "legs": rows,
    }
    if census is not None:
        doc["universe"] = {f"{p}.{b}": int(n) for (p, b), n
                           in sorted(universe_from_census(census).items())}
        doc["digests"] = dict(census.get("digests") or {})
    if extra:
        doc.update(dict(extra))
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return str(path)


def read_manifest(path):
    with open(path) as f:
        doc = json.load(f)
    if str(doc.get("format")) != MANIFEST_FORMAT:
        raise ValueError(
            f"read_manifest: {path} declares format {doc.get('format')!r}, "
            f"not {MANIFEST_FORMAT!r}.")
    return doc


def _default_probe(path):
    """"Is this leg's output there, and is it more than a stub?"

    Size is checked as well as existence because ``$HOME`` filling is a
    measured failure on this fleet and it produces files that exist, are
    nonzero, and contain nothing — the 38-byte junitxml that parsed as
    zero tests.  A 4 kB floor is below any real cube or store and above
    every stub that class of failure has produced.
    """
    if not os.path.exists(path):
        return False, "no such file"
    size = os.path.getsize(path)
    if size < 4096:
        return False, f"{size} bytes, which is a stub and not an output"
    return True, f"{size} bytes"


def refuse_incomplete(manifest, *, probe=None, print_fn=print):
    """Refuse a farm that did not land, naming the legs and their ranges.

    Returns the manifest's leg rows, in manifest order, when every one of
    them landed.  Raises :class:`FarmIncomplete` otherwise — and the
    message names each missing leg's id, the range it owned and the path
    that is not there, because "the merge refused" is only actionable if
    it says which four q, or which groups of which pole, are the hole.

    NEVER MERGES A SUBSET.  There is no ``allow_partial`` on this
    function and there should not be one: every output of this pipeline
    that is missing a term has the shape, dtype and units of one that is
    not, so a subset that is allowed through is indistinguishable from a
    complete answer for the rest of its life.
    """
    probe = probe or _default_probe
    legs = list(manifest["legs"])
    inputs = list(manifest.get("inputs") or [])
    gone = [(r, why) for r, why in
            ((r, probe(str(r["output"]))) for r in inputs) if not why[0]]
    if gone:
        lines = [
            f"{manifest.get('kind', '?')} farm inputs missing: "
            f"{len(gone)} of {len(inputs)} declared inputs are not there.  "
            f"These are the artifacts the legs READ — the window plans "
            f"their group ranges are ranges into — so without them the "
            f"cubes on disk cannot be shown to be the partition this "
            f"manifest declares, and nothing about the cubes themselves "
            f"would say otherwise.",
        ]
        for r, (_ok, why) in gone:
            lines.append(
                f"  MISSING input {r['id']}: {r.get('range_label', '?')} "
                f"-> {r['output']} ({why})")
        lines.append(
            "  Re-run the planning step against this deck and sha; do not "
            "merge cubes whose plan is gone.")
        raise FarmIncomplete("\n".join(lines))
    missing = []
    for leg in legs:
        ok, why = probe(str(leg["output"]))
        if not ok:
            missing.append((leg, why))
    if missing:
        lines = [
            f"{manifest.get('kind', '?')} farm incomplete: "
            f"{len(missing)} of {len(legs)} declared legs did not land.  "
            f"The manifest that declared them is the only reason this is "
            f"visible at all — a missing leg's output has the shape and "
            f"units of a present one, so a merge that summed what arrived "
            f"would return a smooth, finite, wrong answer with nothing "
            f"downstream able to tell.",
        ]
        for leg, why in missing:
            lines.append(
                f"  MISSING leg {leg['id']}: {leg.get('range_label', '?')} "
                f"-> {leg['output']} ({why})")
        lines.append(
            "  Re-run the named legs against the same manifest and merge "
            "again.  Do not merge the rest.")
        raise FarmIncomplete("\n".join(lines))
    print_fn(
        f"  farm completeness: all {len(legs)} declared "
        f"{manifest.get('kind', '?')} legs landed "
        f"(manifest {manifest.get('written_utc', '?')}, "
        f"{manifest.get('total_n_tau', 0)} tau dispatches declared)"
        + (f", and all {len(inputs)} declared inputs are present."
           if inputs else "."))
    return legs
