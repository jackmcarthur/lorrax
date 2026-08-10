"""The window plan, computed once per pole and read by every leg that needs it.

WHY THIS FILE EXISTS
--------------------
The window-group farm of 2026-08-10 (``window_farm.py``, §9 of the 16-GPU
plan) split the Σ pass at the window group and took the pass from 26.4 to
9.2 minutes on sixteen GPUs.  Its own measurement then named what stood
between it and the projected 7.4: **the planner does not divide with the
farm, it multiplies.**  A leg calls ``plan_branch_groups`` for every
(pole, branch) it owns, at ~16 s per branch and ~65 s per pole; a
sixteen-leg farm touches 23 poles instead of 8, because a balanced
contiguous partition puts leg boundaries inside poles, and each leg pays
for its own.  That is a per-leg fixed term of ~90-110 s against a ~450 s
leg — 20 % of the wall spent computing, sixteen times over, an object that
is the same object every time.

It is the same object every time because the planner is a pure function of
its inputs and none of those inputs is the split.  ``plan_branch_groups``
reads one pole's ``(Re Omega, Gamma)`` field, the branch's ``E_A`` and its
mask, the ω grid and a handful of quadrature scalars, and returns the
window groups.  Nothing in it knows or can know which groups a leg will go
on to integrate.  So the plan can be computed once, written down, and
loaded — and the farm stops paying for it once per leg.

WHAT MAKES A LOADED PLAN SAFE, WHICH IS THE WHOLE DESIGN
--------------------------------------------------------
A stale plan is the worst failure this pipeline can have.  A leg that
integrates the groups of a plan built from a different store, a different ω
grid or a different sha would succeed, write a cube of the right shape and
units, and leave a merged Σ that is smooth, finite and wrong by whatever
moved.  ``window_farm``'s census carries one guard against that shape of
failure already — the per-branch plan digest, checked in every leg — and
this module does not weaken it.  It adds a second, stronger one:

**The plan artifact is addressed by its own inputs.**  The file name
carries a digest over EVERY argument ``plan_branch_groups`` reads: the pole
slab's ``Re Omega`` and ``Gamma``, the live mask, the branch's ``E_A`` and
mask, the ω half-grid, and every scalar knob, together with the source sha,
the store path and the pole index.  A leg computes that address from the
inputs IT holds and asks for that file by name.  So a plan that was built
from anything else is not a plan this leg can find: staleness is not
detected, it is structurally impossible to express.  What remains possible
is ABSENCE, and absence is refused by name — which is the same discipline
``refuse_incomplete`` applies to a farm that lost a leg.

WHAT IS AND IS NOT IN THE ARTIFACT
-----------------------------------
Everything the integrator reads out of a ``WindowGroup``, and nothing that
can be reconstructed from the pole slab the leg has already read.

Stored: each group's name, its ``idx_B`` selection, its live-mode count,
its ``b_mass``, its provenance, and every field of every ``_SigmaWindow``
it carries — the τ nodes and weights, the A-side mask, both reference
energies, the sign, the projection, the prefactor, the B-side mask mode and
threshold, the crossing kind, the achieved error and the rule's provenance
string.  The window dataclass is introspected at write time and a field it
gains that this module does not know about is a REFUSAL, not a silent
omission: a dropped field is exactly the class of defect that produces a
plausible wrong self-energy.

Not stored: ``omega_operand``.  It is the pole slab itself — ``Re Omega``
for a legacy-routed group and ``Re Omega - i Gamma`` for an MPA-routed one
— and the leg has already read that slab, because the residues ``B_p`` on
the same slab are what it integrates against.  Writing it down would add
2.6 GB per pole to an artifact that exists to save time, so the group
records WHICH operand it takes and the loader rebuilds it from the leg's
own arrays.  The rebuild is bit-identical by construction: it is the same
two host arrays and the same expression.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import os

import numpy as np

from . import window_farm as WF

__all__ = [
    "PLAN_FORMAT",
    "PlanMissing",
    "branch_address",
    "slab_digest",
    "declared_plan_inputs",
    "plan_path",
    "read_branch_plan",
    "read_plan_header",
    "refuse_missing_plan",
    "write_branch_plan",
]

PLAN_FORMAT = "lorrax-mpa-branch-plan/1"

#: Every field of ``ppm_windows._SigmaWindow``, as this module knows it.
#: Checked against the dataclass at write time -- see :func:`_window_fields`.
_WINDOW_FIELDS = (
    "name", "nodes", "mask_A", "E_ref_A", "E_ref_B", "omega_sign", "project",
    "prefactor", "mask_B_mode", "mask_B_threshold", "crossing_kind",
    "max_error", "provenance",
)

#: Every field of ``sigma_pass.WindowGroup``.  Same check, same reason.
_GROUP_FIELDS = (
    "name", "windows", "idx_B", "field_shape", "omega_operand", "n_modes",
    "b_mass", "provenance",
)

#: How a group's ``omega_operand`` is rebuilt from the leg's own pole slab.
#: ``"real"`` is ``Re Omega`` (the legacy-routed two-point evaluation) and
#: ``"complex"`` is ``Re Omega - i Gamma`` (the MPA route).  The tag is the
#: physics identity of the group, so a group whose operand this module
#: cannot name is refused rather than guessed.
_OPERAND_REAL = "real"
_OPERAND_COMPLEX = "complex"


class PlanMissing(RuntimeError):
    """The plan this leg needs is not in the plan store.

    Its own class for the same reason :class:`window_farm.FarmIncomplete`
    is: the caller has to act on it, and "the plan is absent" must never be
    caught beside "the plan is malformed" and answered by planning one
    quietly.  A leg that silently re-planned would be correct and slow,
    which sounds harmless until it is the one leg in sixteen doing it and
    the farm's wall is a measurement of nothing.
    """


# ---------------------------------------------------------------------------
#  The address: a plan is named by the inputs it was built from
# ---------------------------------------------------------------------------

def _feed(h, label, value):
    """Feed one addressed input into the digest, labelled and length-framed."""
    h.update(f"|{label}:".encode())
    if isinstance(value, np.ndarray):
        arr = np.ascontiguousarray(value)
        h.update(f"{arr.dtype.str}{arr.shape}#".encode())
        h.update(memoryview(arr).cast("B"))
    elif isinstance(value, (bool, np.bool_)):
        h.update(f"bool={bool(value)}".encode())
    elif isinstance(value, (int, np.integer)):
        h.update(f"int={int(value)}".encode())
    elif isinstance(value, float):
        # repr, not a rounded format: the address must move when the knob
        # moves in its last bit, because the plan does.
        h.update(f"float={float(value)!r}".encode())
    elif value is None:
        h.update(b"none")
    else:
        h.update(f"str={value!s}".encode())


def slab_digest(**arrays):
    """The digest of the POLE-LEVEL planner inputs, taken once per pole.

    ``Re Omega``, ``Gamma``, the live mask and ``|B|`` are the same four
    arrays for all four branches of a pole and they are the large ones --
    2.0 GB on the production deck, against a few kilobytes for everything
    the branch contributes.  Hashing them per branch cost 12.56 s a pole
    where hashing them once costs 3.14 s (measured 2026-08-10, census leg
    of pole 0), so the address is built in two stages: this digest, and
    then :func:`branch_address` over it plus what the branch adds.

    The address still covers every byte of every planner input.  Nothing
    is summarized away — the staging is an order of operations, not a
    weaker guarantee.
    """
    h = hashlib.blake2b(digest_size=32)
    h.update(f"{PLAN_FORMAT}/slab\n".encode())
    for key in sorted(arrays):
        _feed(h, f"array.{key}", np.asarray(arrays[key]))
    return h.hexdigest()


#: Attribute naming the stats keys whose value was ``None``.  Kept out
#: of the address on purpose: it is a property of how the stats are
#: SPELLED on disk, not of the plan the address identifies.
_NONE_KEYS_ATTR = "_none_valued_keys"


def branch_address(*, source_sha, fit_store, n_p, pole, bkey, slab,
                   arrays, scalars):
    """The content address of one (pole, branch) plan, over ALL its inputs.

    ``slab`` is :func:`slab_digest` of the pole-level arrays; ``arrays``
    and ``scalars`` are what the branch adds.  Between them they are every
    argument ``plan_branch_groups`` reads, and the digest covers the array
    BYTES rather than any summary of them.  That is the expensive-looking
    choice and it is the correct one: a summary (a shape, a min and a max,
    a store path and an mtime) is exactly what a stale plan can match while
    being a different plan, and the failure it would let through is a
    self-energy that is wrong by the size of whatever moved and looks like
    every other self-energy.

    The cost is one streaming hash of arrays the leg has ALREADY read --
    the pole slab it is about to integrate against -- so it adds a pass
    over memory, not a read.  It is measured and reported beside the
    planning time it replaces rather than assumed to be small.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(f"{PLAN_FORMAT}\n".encode())
    _feed(h, "slab", str(slab))
    _feed(h, "source_sha", str(source_sha))
    _feed(h, "fit_store", str(fit_store))
    _feed(h, "n_p", int(n_p))
    _feed(h, "pole", int(pole))
    _feed(h, "branch", str(bkey))
    for key in sorted(scalars):
        _feed(h, f"scalar.{key}", scalars[key])
    for key in sorted(arrays):
        _feed(h, f"array.{key}", np.asarray(arrays[key]))
    return h.hexdigest()


def plan_path(plan_dir, *, pole, bkey, address):
    """``<dir>/plan_p3.pos_val.<address>.h5`` -- the name IS the guarantee."""
    return os.path.join(str(plan_dir),
                        f"plan_p{int(pole)}.{str(bkey)}.{str(address)}.h5")


def refuse_missing_plan(plan_dir, *, pole, bkey, address):
    """Refuse a leg whose plan is not there, and say which case it is.

    Two cases, and telling them apart is the whole value of the message.
    NO plan for this (pole, branch) at all means the planning step never
    ran or wrote somewhere else.  A plan for this (pole, branch) at a
    DIFFERENT address means the planning step ran against different
    inputs -- a different store, a different ω grid, a different sha, a
    different quadrature knob -- and the leg is being asked to integrate a
    partition that no longer describes this calculation.  Under the old
    re-plan-per-leg route the second case did not exist because there was
    nothing to be stale; it exists now, and it is answered by naming it
    rather than by falling back to planning, which would hide it.
    """
    import glob

    want = plan_path(plan_dir, pole=pole, bkey=bkey, address=address)
    others = sorted(glob.glob(os.path.join(
        str(plan_dir), f"plan_p{int(pole)}.{str(bkey)}.*.h5")))
    lines = [
        f"mpa_plan_store: pole {int(pole)} branch {bkey} has no plan at "
        f"{want}.",
    ]
    if others:
        lines += [
            f"  The store DOES carry {len(others)} plan(s) for this "
            f"(pole, branch) at other addresses:",
        ] + [f"    {os.path.basename(o)}" for o in others] + [
            "  An address is a digest over every argument the planner "
            "reads -- the pole slab, the branch's E_A and mask, the ω "
            "half-grid, every quadrature scalar, the store path and the "
            "source sha.  A different address therefore means this leg's "
            "inputs are not the inputs those plans were built from, so "
            "their window groups are not this calculation's window groups. "
            "Re-run the planning step against THIS deck and sha.",
        ]
    else:
        lines += [
            "  The store carries no plan for this (pole, branch) at any "
            "address.  The planning step (a census run with "
            "mpa_plan_store set) has not run for this pole against this "
            "deck, or it wrote to a different directory.",
        ]
    lines.append(
        "  This leg will NOT plan one for itself: a leg that quietly "
        "re-planned would be correct and slow, and a farm in which one leg "
        "in sixteen does that has a wall that measures nothing.")
    raise PlanMissing("\n".join(lines))


# ---------------------------------------------------------------------------
#  Writing
# ---------------------------------------------------------------------------

def _window_fields():
    """The window dataclass's fields, refusing any this module does not know.

    A ``_SigmaWindow`` that gains a field which the plan store does not
    write is a plan that silently loses it, and a window missing one field
    integrates to a finite, smooth, plausible number.  So the field set is
    asserted against the dataclass here, at the write, where the fix is
    cheap -- rather than discovered later as a discrepancy in a Σ.
    """
    from ..ppm_windows import _SigmaWindow

    have = tuple(f.name for f in dataclasses.fields(_SigmaWindow))
    if set(have) != set(_WINDOW_FIELDS):
        raise ValueError(
            f"plan_store: ppm_windows._SigmaWindow carries fields {have}, "
            f"but this module serializes {_WINDOW_FIELDS}.  A field the "
            f"plan store does not write is a field a loaded plan does not "
            f"have, and a window missing one integrates to a smooth, "
            f"finite, wrong Σ rather than to an error.  Teach this module "
            f"the new field (and the digest that covers it) rather than "
            f"widening this check.")
    return have


def _group_fields():
    """The same assertion for ``WindowGroup``."""
    from .sigma_pass import WindowGroup

    have = tuple(f.name for f in dataclasses.fields(WindowGroup))
    if set(have) != set(_GROUP_FIELDS):
        raise ValueError(
            f"plan_store: sigma_pass.WindowGroup carries fields {have}, but "
            f"this module serializes {_GROUP_FIELDS}.  See _window_fields "
            f"for why this is a refusal and not a warning.")
    return have


def _operand_tag(group, *, a_ry, omega_complex):
    """Which of the two slab operands this group takes, by identity.

    Compared by VALUE against the two operands the planner can hand a
    group, not inferred from the group's name: the name is a label a future
    bucketing change is free to alter, and the operand is the difference
    between a two-point plasmon-pole evaluation and an MPA one.
    """
    op = np.asarray(group.omega_operand)
    if op.dtype.kind == "f" and op.shape == np.shape(a_ry):
        if np.array_equal(op, a_ry):
            return _OPERAND_REAL
    elif op.dtype.kind == "c" and op.shape == np.shape(omega_complex):
        if np.array_equal(op, omega_complex):
            return _OPERAND_COMPLEX
    raise ValueError(
        f"plan_store: group {group.name!r} carries an Omega_q operand that "
        f"is neither this pole's Re Omega nor its Re Omega - i Gamma "
        f"(dtype {op.dtype}, shape {op.shape}).  The plan store rebuilds "
        f"the operand from the leg's own slab rather than storing 2.6 GB of "
        f"it per pole, so an operand it cannot name is one it cannot "
        f"rebuild -- and guessing would put the wrong pole field under a "
        f"certified rule.")


def write_branch_plan(path, groups, *, address, pole, bkey, source_sha,
                      fit_store, n_p, stats, a_ry, omega_complex,
                      print_fn=print):
    """Write one (pole, branch) plan: every group, every window, verbatim.

    ``stats`` travels with the plan because the pass record is built from
    it -- the narrow/wide split and its ``b`` masses are part of what the
    planner computed, and a leg that loads a plan must report the same
    numbers as one that computed it.
    """
    import h5py

    _group_fields()
    win_fields = _window_fields()
    tmp = f"{path}.tmp"
    rows = WF.plan_digest_rows(groups)
    with h5py.File(tmp, "w") as f:
        f.attrs["format"] = PLAN_FORMAT
        f.attrs["address"] = str(address)
        f.attrs["pole"] = int(pole)
        f.attrs["branch"] = str(bkey)
        f.attrs["source_sha"] = str(source_sha)
        f.attrs["fit_store"] = str(fit_store)
        f.attrs["n_p"] = int(n_p)
        f.attrs["n_groups"] = int(len(groups))
        f.attrs["group_plan_digest"] = WF.group_plan_digest(groups)
        f.attrs["full_plan_digest"] = WF.full_plan_digest(groups)
        f.attrs["window_fields"] = ",".join(win_fields)
        f.attrs["written_utc"] = datetime.datetime.now(
            datetime.timezone.utc).isoformat()
        st = f.create_group("stats")
        # A STAT MAY BE ABSENT, AND ABSENT IS A VALUE.  The planner's stats
        # are whatever the planner reports, and a knob that is OFF reports
        # ``None`` -- ``binned_width_clause`` is the live example, pinned
        # ``is None`` by its own lane's cell.  h5py has no native object
        # dtype, so writing it raw raises "Object dtype has no native HDF5
        # equivalent" and the plan store becomes the thing that decides
        # which stats a planner is allowed to have.  It is recorded as a
        # sentinel name instead and read back as ``None``, so the stats a
        # leg loads are the stats the planner produced.
        none_keys = sorted(k for k, v in dict(stats or {}).items()
                           if v is None)
        for k, v in dict(stats or {}).items():
            if v is not None:
                st.attrs[k] = v
        st.attrs[_NONE_KEYS_ATTR] = ",".join(none_keys)
        gg = f.create_group("groups")
        for i, g in enumerate(groups):
            node = gg.create_group(str(i))
            node.attrs["name"] = str(g.name)
            node.attrs["n_modes"] = int(g.n_modes)
            node.attrs["n_tau"] = int(rows[i][2])
            node.attrs["b_mass"] = float(g.b_mass)
            node.attrs["provenance"] = str(g.provenance)
            node.attrs["field_shape"] = np.asarray(
                [int(x) for x in g.field_shape], dtype=np.int64)
            node.attrs["operand"] = _operand_tag(
                g, a_ry=a_ry, omega_complex=omega_complex)
            node.attrs["n_windows"] = int(len(g.windows))
            node.create_dataset("idx_B", data=np.asarray(g.idx_B))
            wg = node.create_group("windows")
            for j, w in enumerate(g.windows):
                wn = wg.create_group(str(j))
                wn.attrs["name"] = str(w.name)
                wn.attrs["E_ref_A"] = float(w.E_ref_A)
                wn.attrs["E_ref_B"] = float(w.E_ref_B)
                wn.attrs["omega_sign"] = int(w.omega_sign)
                wn.attrs["project"] = str(w.project)
                wn.attrs["prefactor"] = float(w.prefactor)
                wn.attrs["mask_B_mode"] = str(w.mask_B_mode)
                wn.attrs["has_mask_B_threshold"] = bool(
                    w.mask_B_threshold is not None)
                wn.attrs["mask_B_threshold"] = float(
                    0.0 if w.mask_B_threshold is None else w.mask_B_threshold)
                wn.attrs["has_crossing_kind"] = bool(
                    w.crossing_kind is not None)
                wn.attrs["crossing_kind"] = str(w.crossing_kind or "")
                wn.attrs["has_max_error"] = bool(w.max_error is not None)
                wn.attrs["max_error"] = float(
                    0.0 if w.max_error is None else w.max_error)
                wn.attrs["has_provenance"] = bool(w.provenance is not None)
                wn.attrs["provenance"] = str(w.provenance or "")
                wn.create_dataset("t", data=np.asarray(
                    w.nodes.t, dtype=np.complex128))
                wn.create_dataset("alpha", data=np.asarray(
                    w.nodes.alpha, dtype=np.complex128))
                wn.create_dataset("mask_A", data=np.asarray(
                    w.mask_A, dtype=bool))
    os.replace(tmp, path)
    size = os.path.getsize(path)
    print_fn(
        f"    plan written: p{int(pole)} {bkey} -> {os.path.basename(path)} "
        f"({len(groups)} groups, {size / 1e6:.1f} MB, digest "
        f"{WF.group_plan_digest(groups)})")
    return str(path)


# ---------------------------------------------------------------------------
#  Reading
# ---------------------------------------------------------------------------

def read_plan_header(path):
    """The plan's metadata and per-group rows, WITHOUT the index sets.

    The header is what the anti-stale checks run against — the group count
    and the census digest — and it is a few kilobytes, while the index sets
    are hundreds of megabytes.  Reading them separately is what lets a leg
    check the whole partition and then load only the groups it owns.
    """
    import h5py

    with h5py.File(str(path), "r") as f:
        if str(f.attrs.get("format")) != PLAN_FORMAT:
            raise ValueError(
                f"read_plan_header: {path} declares format "
                f"{f.attrs.get('format')!r}, not {PLAN_FORMAT!r}.")
        n_groups = int(f.attrs["n_groups"])
        gg = f["groups"]
        rows = [(str(gg[str(i)].attrs["name"]),
                 int(gg[str(i)].attrs["n_modes"]),
                 int(gg[str(i)].attrs["n_tau"])) for i in range(n_groups)]
        stats = {k: _unwrap(v) for k, v in f["stats"].attrs.items()
                 if k != _NONE_KEYS_ATTR}
        for k in str(_unwrap(
                f["stats"].attrs.get(_NONE_KEYS_ATTR, ""))).split(","):
            if k:
                stats[k] = None
        head = {
            "path": str(path),
            "address": str(f.attrs["address"]),
            "pole": int(f.attrs["pole"]),
            "branch": str(f.attrs["branch"]),
            "source_sha": str(f.attrs["source_sha"]),
            "fit_store": str(f.attrs["fit_store"]),
            "n_p": int(f.attrs["n_p"]),
            "n_groups": n_groups,
            "group_plan_digest": str(f.attrs["group_plan_digest"]),
            "full_plan_digest": str(f.attrs["full_plan_digest"]),
            "written_utc": str(f.attrs["written_utc"]),
            "rows": rows,
            "stats": stats,
        }
    return head


def _unwrap(v):
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, bytes):
        return v.decode()
    return v


def read_branch_plan(path, *, lo, hi, a_ry, omega_complex):
    """The groups ``[lo, hi)`` of a written plan, as ``WindowGroup`` objects.

    ``a_ry`` and ``omega_complex`` are THIS leg's own slab arrays and are
    what the group's ``omega_operand`` is rebuilt from.  They are not
    checked against the plan here because they cannot disagree with it: the
    address the caller resolved this file by is a digest over those very
    arrays, so a leg holding a different slab is a leg that did not find
    this file.
    """
    import h5py

    from .sigma_pass import WindowGroup
    from ..ppm_windows import _SigmaWindow
    from ..minimax_screening import MinimaxNodes

    groups = []
    with h5py.File(str(path), "r") as f:
        if str(f.attrs.get("format")) != PLAN_FORMAT:
            raise ValueError(
                f"read_branch_plan: {path} declares format "
                f"{f.attrs.get('format')!r}, not {PLAN_FORMAT!r}.")
        gg = f["groups"]
        for i in range(int(lo), int(hi)):
            node = gg[str(i)]
            tag = str(node.attrs["operand"])
            if tag == _OPERAND_REAL:
                operand = a_ry
            elif tag == _OPERAND_COMPLEX:
                operand = omega_complex
            else:
                raise ValueError(
                    f"read_branch_plan: group {i} of {path} names operand "
                    f"{tag!r}, which is neither {_OPERAND_REAL!r} nor "
                    f"{_OPERAND_COMPLEX!r}.")
            wins = []
            wg = node["windows"]
            for j in range(int(node.attrs["n_windows"])):
                wn = wg[str(j)]
                wins.append(_SigmaWindow(
                    name=str(wn.attrs["name"]),
                    nodes=MinimaxNodes(
                        t=np.asarray(wn["t"][()], dtype=np.complex128),
                        alpha=np.asarray(wn["alpha"][()],
                                         dtype=np.complex128)),
                    mask_A=np.asarray(wn["mask_A"][()], dtype=bool),
                    E_ref_A=float(wn.attrs["E_ref_A"]),
                    E_ref_B=float(wn.attrs["E_ref_B"]),
                    omega_sign=int(wn.attrs["omega_sign"]),
                    project=str(wn.attrs["project"]),
                    prefactor=float(wn.attrs["prefactor"]),
                    mask_B_mode=str(wn.attrs["mask_B_mode"]),
                    mask_B_threshold=(float(wn.attrs["mask_B_threshold"])
                                      if bool(wn.attrs["has_mask_B_threshold"])
                                      else None),
                    crossing_kind=(str(wn.attrs["crossing_kind"])
                                   if bool(wn.attrs["has_crossing_kind"])
                                   else None),
                    max_error=(float(wn.attrs["max_error"])
                               if bool(wn.attrs["has_max_error"]) else None),
                    provenance=(str(wn.attrs["provenance"])
                                if bool(wn.attrs["has_provenance"]) else None),
                ))
            groups.append(WindowGroup(
                name=str(node.attrs["name"]),
                windows=wins,
                idx_B=np.asarray(node["idx_B"][()]),
                field_shape=tuple(int(x) for x in
                                  np.asarray(node.attrs["field_shape"])),
                omega_operand=operand,
                n_modes=int(node.attrs["n_modes"]),
                b_mass=float(node.attrs["b_mass"]),
                provenance=str(node.attrs["provenance"]),
            ))
    return groups


def declared_plan_inputs(plans):
    """The manifest rows that declare a farm's plan artifacts.

    A farm's legs are declared before they run so a missing one is
    visible; its INPUTS are declared for the same reason and in the same
    shape.  A plan artifact that has been deleted between the balance and
    the merge leaves cubes whose partition cannot be re-derived from
    anything written down, which is the provenance half of the same
    failure -- so it is a row with an ``id``, a ``range_label`` and an
    expected path, and :func:`window_farm.refuse_incomplete` probes it
    exactly as it probes a leg.
    """
    rows = []
    for key in sorted(plans, key=lambda k: (int(k.split(".")[0]),
                                            WF.BRANCH_KEYS.index(
                                                k.split(".")[1]))):
        info = plans[key]
        rows.append({
            "id": f"plan {key}",
            "kind": "plan",
            "range_label": (f"pole {key.split('.')[0]} "
                            f"{key.split('.')[1]} window plan, address "
                            f"{info['address']}"),
            "address": str(info["address"]),
            "output": str(info["path"]),
        })
    return rows
