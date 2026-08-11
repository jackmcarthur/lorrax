"""The downfold pipeline: read one restart bundle, write a smaller one.

Five stages, in order, each timed under a ``downfold.*`` section so it shows
up in the standard timing report beside ``zeta_fit`` and ``sigma.exec``:

    downfold.load        the parent bundle, sharded, wedge already unfolded
    downfold.grams       S_LL over the retained window (one kernel call)
    downfold.select      CUR selection + the rank refusal
    downfold.solve       T = S_SS^-1 S_cross, rank-truncated
    downfold.congruence  V, W, the _nohead twins and g0 through T
    downfold.residual    the per-q Pythagorean error bar
    downfold.write       the small bundle, in the SAME format at smaller mu
    downfold.zeta        zeta transported to the small basis, beside it

THE HEADLINE PROPERTY, and the thing to protect if anyone proposes changing
this file: the output is a restart bundle in the **unchanged format** at a
smaller mu.  Not a new bespoke layout, not a sidecar.  That is what makes
every existing BSE consumer a zero-change drop-in —
``bse_io.load_bse_data_from_restart_sharded`` reads it, ``_MunuSlabPlan``
plans it, the ring and stack matvecs eat it, and none of them learns that a
downfold happened.  If a reviewer proposes a bespoke ``.h5`` for downfolded
objects, the answer is no.

AND WHERE THAT PROPERTY IS NOT ENOUGH, which is the 2026-08-10 lesson.  It
covers every consumer that only READS the stored tensors.  It does not cover
one that REBUILDS an object in the same ISDF basis, because rebuilding needs
two things that are not in the bundle format at all and never were: the
CENTROID COORDINATES (``bse.exciton_bands`` refits psi at finite Q against a
coordinate table, and its Galerkin fit is sized by nk*nb, so it must run in
the PARENT basis and slice) and ZETA (the off-grid exchange interpolates it).
So stages D6 and the centroid writer below produce those two beside the
bundle.  They are SIBLINGS, not bundle datasets — the format is still
unchanged, and the answer above is still no — but without them the drop-in
promise held for ``bse_jax`` and failed for the driver the compression was
built to serve.

WHAT THE BAND AXIS DOES *NOT* DO, in stage 1.  The retained band window is
the FIT window — it decides which pair densities the compression is faithful
to.  It is NOT a truncation of the stored band axis: ``psi_full_y`` and
``enk_full`` are written with their band axis unchanged, and only mu is
compressed.  That is deliberate.  Truncating bands would renumber every band
index in the bundle, move the ``band_window`` stamp that
``assert_restart_window_matches`` refuses on, and break the drop-in property
outright — a consumer asking for ``--n-occ 8`` would silently get different
states.  Band-axis truncation is a separate, later stage with its own
renumbering contract.
"""
from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass, field

import h5py
import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from common import timing
from common.collectives import barrier, process_rank
from file_io import (
    load_restart_state_from_h5,
    parse_coulomb_policy,
    read_coulomb_policy_from_h5,
    read_munu_tensor_from_h5,
    write_head_scalars_to_h5,
    write_restart_state_to_h5,
)
from gw.downfold import (
    BandWindow,
    R19_ANCHOR,
    build_transfer,
    child_unfold_tables,
    congruence,
    epsilon_w,
    pair_density_gram,
    select_cur_centroids,
    slice_psi_to_centroids,
    transform_head_vector,
)

__all__ = ["DownfoldResult", "resolve_restart_file", "run_downfold"]

#: Bumped when the provenance group's contents change meaning.
#: 2 (2026-08-10) adds the ISDF-basis paths — ``parent_centroids_file``,
#: ``parent_centroids_md5``, ``centroids_file``, ``zeta_file`` — which is
#: what a consumer needs to rebuild something NEW in this basis rather than
#: only read the tensors already in it.  Purely additive: every version-1
#: field keeps its meaning, so a reader that only wants ``keep_idx`` need not
#: look at the version at all.
DOWNFOLD_PROVENANCE_VERSION = 2

#: The two-point tensors a downfold transports.  Each is a LINEAR object in
#: the centroid basis and therefore transforms by congruence.  A plasmon-pole
#: reduction does NOT belong on this list: ``B_q`` would transform but
#: ``Omega_q`` is a pole POSITION per matrix element and no congruence maps a
#: table of pole frequencies between bases.  Downfold the linear objects and
#: re-fit the PPM/MPA model in the small basis.
_MUNU_TENSORS = ("V_qmunu", "W0_qmunu", "V_qmunu_nohead", "W0_qmunu_nohead")


@dataclass
class DownfoldResult:
    """Everything the run learned, for the report and for the provenance."""

    source_file: str
    output_file: str
    mu_large: int
    mu_small: int
    n_q: int
    n_bands: int
    keep_idx: np.ndarray
    selection: object
    rank_per_q: np.ndarray
    eps_w: dict = field(default_factory=dict)
    norms: dict = field(default_factory=dict)
    wrote_nohead: tuple = ()
    centroid_file: str = ""
    parent_centroid_file: str = ""
    zeta_file: str = ""


# ---------------------------------------------------------------------------
# locating the parent bundle
# ---------------------------------------------------------------------------

def resolve_restart_file(path: str) -> str:
    """A run directory or an ``.h5`` → THE parent restart file.

    ``isdf_tensors_*.h5`` is namespaced by CENTROID COUNT, not by run, so a
    directory can legitimately hold several and picking one silently is a
    wrong answer that passes every shape check downstream:
    ``isdf_tensors_1194.h5`` sorts before ``isdf_tensors_276.h5`` and neither
    order is meaningful.  ``bse_io._find_restart_file`` resolves the same
    ambiguity by newest mtime with a loud warning, which is the right call
    for a driver whose input file already named the run.  This one REFUSES
    instead, because the downfold's whole job is to produce a second bundle
    at a different mu in a nearby directory — it is the tool most likely to
    create the ambiguity, so it is the one that must not resolve it by luck.
    Name the file when there is more than one.
    """
    if os.path.isfile(path):
        return os.path.abspath(path)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"downfold: source_restart={path!r} is neither a file nor a "
            f"directory.  Point it at the finished GW run's directory, or at "
            f"its tmp/isdf_tensors_<mu>.h5 directly.")
    cands = sorted(glob.glob(os.path.join(path, "tmp", "isdf_tensors_*.h5")))
    cands += sorted(glob.glob(os.path.join(path, "isdf_tensors_*.h5")))
    if not cands:
        raise FileNotFoundError(
            f"downfold: no isdf_tensors_*.h5 under {path} (looked in "
            f"'tmp/' and in the directory itself).  A downfold needs a "
            f"FINISHED GW calculation's restart bundle; if the GW run set "
            f"write_restart_tensors = false there is nothing on disk to "
            f"compress and the run has to be repeated with it true.")
    if len(cands) > 1:
        raise ValueError(
            f"downfold: {len(cands)} restart bundles under {path}: "
            f"{[os.path.basename(c) for c in cands]}.  They hold DIFFERENT "
            f"centroid counts and nothing about the file names says which "
            f"one this downfold means.  Set source_restart to the file "
            f"itself.")
    return os.path.abspath(cands[0])


def _read_geometry(filename: str) -> dict:
    """The small, replicated facts, read serially before any tensor moves."""
    with h5py.File(filename, "r") as f:
        if "psi_full_y" not in f:
            raise ValueError(
                f"downfold: {filename} has no psi_full_y.  Without "
                f"psi-at-centroids there are no pair densities to fit "
                f"against, so there is no downfold to do — this is not a "
                f"LORRAX restart bundle, or it is one from before the "
                f"canonical writer.")
        if "V_qmunu" not in f:
            raise ValueError(
                f"downfold: {filename} has no V_qmunu.")
        if "W0_qmunu" not in f:
            raise ValueError(
                f"downfold: {filename} has no W0_qmunu — the parent GW run "
                f"wrote its V and psi but never got as far as persisting the "
                f"screened interaction.  A downfold of V alone is a "
                f"legitimate thing to want and is not what this driver does; "
                f"finish the parent run first.")
        if not bool(f["W0_qmunu"].attrs.get("W0_ready", False)):
            raise ValueError(
                f"downfold: {filename} carries W0_qmunu but its W0_ready "
                f"flag is FALSE — the dataset is the all-zeros PLACEHOLDER "
                f"the writer pre-allocates, not screening.  Downfolding it "
                f"would produce a small bundle full of zeros that every "
                f"shape check passes; the flag exists because that happened "
                f"once already.  Re-run the parent GW to completion.")
        if not bool(f["V_qmunu"].attrs.get("V_ready", True)):
            raise ValueError(
                f"downfold: {filename} says V_ready = False.")
        geom = {
            "n_rmu_logical": (int(np.asarray(f["n_rmu_logical"])[()])
                              if "n_rmu_logical" in f
                              else int(f["V_qmunu"].shape[-1])),
            "kgrid": (tuple(int(v) for v in np.asarray(f["kgrid"])[:])
                      if "kgrid" in f else None),
            "band_window": (np.asarray(f["band_window"])[:].astype(np.int64)
                            if "band_window" in f else None),
            "nb": int(f["psi_full_y"].shape[1]),
            "nk": int(f["psi_full_y"].shape[0]),
            "nspinor": int(f["psi_full_y"].shape[2]),
            "vhead": (np.asarray(f["vhead"])[()] if "vhead" in f else None),
            "whead": (np.asarray(f["whead"][:]) if "whead" in f else None),
            "omega_grid": (np.asarray(f["whead"].attrs["omega_grid"])
                           if "whead" in f and "omega_grid" in f["whead"].attrs
                           else None),
            # The head INTEGRAND's S tensor.  It rides through a downfold
            # untouched for the same reason vhead/whead do: the head channel
            # is a property of the cell and the screening, not of the ISDF
            # basis the tensors were compressed into.  Dropping it would make
            # a downfolded bundle silently unable to densify W (the child
            # would have to rebuild S from dipole.h5, which a bundle-only
            # consumer has no path to).
            "S_cart": (np.asarray(f["S_cart_head"][:])
                       if "S_cart_head" in f else None),
            "centroids_charge_md5": f.attrs.get("centroids_charge_md5"),
            "present": tuple(n for n in _MUNU_TENSORS if n in f),
        }
    if geom["kgrid"] is None:
        raise ValueError(
            f"downfold: {filename} carries no kgrid, so the q axis of "
            f"V/W cannot be split into (nkx, nky, nkz).  The Gram build is a "
            f"convolution over k and needs that split; there is no way to "
            f"guess it from the flat q extent.  The bundle predates the "
            f"kgrid stamp — regenerate it with a current gw_jax, or read the "
            f"grid off the WFN the parent run used and note that this "
            f"driver deliberately does not take a WFN (its whole premise is "
            f"that a finished restart is self-describing).")
    return geom


def _read_parent_unfold_tables(src: str, mu_L: int, print_fn):
    """The parent's ``QirrTables``, or ``None`` when it stores the full BZ.

    THE TABLE IS READ, NOT RE-DERIVED, and that is the whole point.  Building
    ``(α, L)`` from geometry needs the parent's symmetry ops, which live on a
    WFN this driver deliberately does not open — and even if it did, a
    second derivation of the permutation would be a second answer to a
    question the parent's own file already answers.  ``qirr_store`` persists
    the tables beside the tensor they deconstructed, so the selection is
    closed under exactly the permutation the parent's wedge storage rests on.

    Returns ``None`` — an ABSENCE, announced as one — when the parent's
    ``V_qmunu`` is stored on the full BZ.  Nothing is wrong in that case; the
    selection is point-granular, closure is unmeasured, and the child cannot
    be wedge-stored because there is no wedge in its lineage to store it on.
    """
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import dataset_q_storage, read_tables

    try:
        with h5py.File(src, "r") as f:
            storage = dataset_q_storage(f["V_qmunu"])
        if storage != "ibz":
            print_fn(
                "  [downfold/star] the parent stores V_qmunu on the FULL BZ, "
                "so it carries no centroid source map and this run's "
                "selection is POINT-GRANULAR: orbit closure is UNMEASURED, "
                "which is an absence and not a pass, and the child cannot be "
                "wedge-stored (there is no wedge in its lineage).  Set the "
                "PARENT run's restart_q_storage = auto to get both.")
            return None
        tables = read_tables(src, "V_qmunu")
    except (KeyError, ValueError, OSError) as exc:
        print_fn(
            f"  [downfold/star] the parent's q_irr tables could not be read "
            f"({type(exc).__name__}: {exc}).  Falling back to a "
            f"POINT-GRANULAR selection with closure UNMEASURED — an absence, "
            f"not a pass.")
        return None

    perm = np.asarray(tables.sym_perm)
    if int(perm.shape[1]) < int(mu_L):
        raise ValueError(
            f"downfold: the parent's stored sym_perm describes "
            f"{int(perm.shape[1])} centroids but the bundle declares "
            f"mu_L={mu_L}.  The table and the tensor are not about the same "
            f"centroid set, and selecting orbits against the wrong table "
            f"would produce a child whose 'orbits' are arbitrary index "
            f"classes.")
    print_fn(
        f"  [downfold/star] parent centroid source map read from its own "
        f"q_irr tables: {int(perm.shape[0])} op rows "
        f"({int(tables.n_sym_spatial)} spatial + TRS) over "
        f"{int(perm.shape[1])} centroids, {int(np.asarray(tables.q_irr_frac).shape[0])} "
        f"of {int(np.asarray(tables.irr_idx_q).shape[0])} q on the wedge.")
    return tables


def _wedge_q_slots(tables) -> np.ndarray:
    """Full-BZ index of each wedge slot: ``irr_idx == i`` and ``sym_idx == 0``.

    REFUSES rather than guessing.  ``qirr_store``'s own module docstring
    records that slicing an unfolded tensor back to the wedge is bit-exact
    "only because ``sym_idx_q[q_irr_full_idx] == 0``, which is a consequence
    of a policy nobody has agreed to freeze for this purpose".  A downfold has
    no pre-unfold capture to fall back on — the reader unfolds the parent
    before this driver sees it — so the policy is CHECKED here on every run
    instead of relied upon, and the round trip below measures the result.
    """
    irr = np.asarray(tables.irr_idx_q).astype(np.int64).ravel()
    sym = np.asarray(tables.sym_idx_q).astype(np.int64).ravel()
    n_ibz = int(np.asarray(tables.q_irr_frac).shape[0])
    slots = np.empty(n_ibz, dtype=np.int64)
    for i in range(n_ibz):
        hit = np.flatnonzero((irr == i) & (sym == 0))
        if hit.size != 1:
            raise ValueError(
                f"downfold: wedge slot {i} is carried by {hit.size} full-BZ q "
                f"with sym_idx == 0 (expected exactly one).  The child's "
                f"wedge block would have to be chosen rather than restricted, "
                f"and 'chosen' is the word for the failure this refusal "
                f"exists to prevent.")
        slots[i] = int(hit[0])
    return slots


#: Above this relative disagreement the child's wedge block does NOT unfold to
#: the child, and the gate says REFUTED rather than quoting the number and
#: calling it a floor.  The two routes apply the same unitary at different
#: points of one chain, so the honest floor is a few ulp (the synthetic gate in
#: tests/test_downfold.py reads 1.7e-15); a DIFFERENT algebra misses by order
#: one (its red twin reads 8.6e-01).  1e-9 sits in the empty decades between
#: them, so no plausible reassociation reaches it and no real disagreement
#: hides under it.
CHILD_COVARIANCE_TOL = 1e-9


def _star_rank_constancy(rank_per_q, tables):
    """Is the transfer's retained rank CONSTANT on each symmetry star?

    THE PRECONDITION UNDER THE PRECONDITION.  Orbit closure of ``keep`` is what
    gives the child an unfold table; it is not what makes the child's tensors
    unfold.  That needs ``T[q] = U^S T[i(q)] (U^L)†``, and ``T`` is built by a
    RANK-TRUNCATED solve.  If ``S_SS[q]`` is covariant its spectrum is
    identical across a star, so the retained rank must be constant on each
    star; a star that spends different ranks at different members has been
    handed non-covariant Grams, and no selection rule can repair that.

    Cheap — a group-by over 64 integers — and it is the first thing to read
    when the covariance gate fails, because it separates "the tables are wrong"
    from "the operands were never covariant".

    Returns ``(constant, worst_star, worst_spread, n_bad)``.
    """
    ranks = np.asarray(rank_per_q, dtype=np.int64).ravel()
    irr = np.asarray(tables.irr_idx_q, dtype=np.int64).ravel()
    if ranks.size != irr.size:
        return None, -1, -1, -1
    bad, worst, worst_star = 0, 0, -1
    for i in np.unique(irr):
        r = ranks[irr == i]
        spread = int(r.max() - r.min())
        if spread:
            bad += 1
            if spread > worst:
                worst, worst_star = spread, int(i)
    return bad == 0, worst_star, worst, bad


def _unfold_roundtrip_rel(arr_full, tables, sym_perm, L_table, slots,
                          mesh_xy, n_mu_logical):
    """max rel |unfold(arr_full[slots]) - arr_full| — one number, host-side.

    The common half of the storability gate and its control, written once so
    that the two are the SAME route by construction: a control that differed
    from the thing it controls by a line of plumbing would not be a control.
    """
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import unfold_isdf_operator
    from common.collectives import gather_to_host

    full = np.asarray(gather_to_host(arr_full))
    n_pad = int(full.shape[-1]) - int(n_mu_logical)
    perm, L = np.asarray(sym_perm), np.asarray(L_table)
    if n_pad:
        tail = np.arange(n_mu_logical, n_mu_logical + n_pad, dtype=perm.dtype)
        perm = np.concatenate(
            [perm[:, :n_mu_logical],
             np.broadcast_to(tail, (perm.shape[0], n_pad))], axis=1)
        L = np.concatenate(
            [L[:, :n_mu_logical],
             np.zeros((L.shape[0], n_pad, 3), L.dtype)], axis=1)
    got = np.asarray(gather_to_host(unfold_isdf_operator(
        jnp.asarray(full[slots]),
        irr_idx=jnp.asarray(np.asarray(tables.irr_idx_q)),
        sym_idx=jnp.asarray(np.asarray(tables.sym_idx_q)),
        sym_perm=jnp.asarray(perm), L_table=jnp.asarray(L),
        q_irr_frac=jnp.asarray(np.asarray(tables.q_irr_frac)),
        mesh_xy=mesh_xy, n_sym_spatial=int(tables.n_sym_spatial))))
    scale = float(np.max(np.abs(full))) or 1.0
    return float(np.max(np.abs(got - full))) / scale


def _gate_child_wedge_storability(small, keep_idx, tables, mesh_xy, mu_S,
                                  mu_S_pad, *, rank_per_q, parent_tensor=None,
                                  mu_L=None, print_fn=print):
    """THE COMPOSITION, MEASURED ON THE RUN'S OWN TENSORS.

    An orbit-floored selection is orbit-closed by construction, so the child
    HAS unfold tables — ``child_unfold_tables`` builds them as restrictions of
    the parent's.  Whether the child is genuinely wedge-storable is a
    different claim, and it is the one the q_irr amendment gated on a
    synthetic: unfolding the child's wedge block with the child's own tables
    must reproduce the child on the full BZ.

    This takes that gate on the REAL tensors this run just produced, per run,
    at whatever mesh it is on.  Cost is one unfold of an
    ``(n_q_ibz, μ_S, μ_S)`` block, which is the smallest tensor in the
    pipeline.  Returns ``(child_perm, child_L, slots, worst_rel)``.
    """
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import unfold_isdf_operator
    from common.collectives import gather_to_host

    child_perm, child_L = child_unfold_tables(
        keep_idx, np.asarray(tables.sym_perm)[:, :],
        np.asarray(tables.L_table))
    slots = _wedge_q_slots(tables)

    # The tables carry the CHILD's logical μ; the tensors carry its device
    # pad.  Pad the tables with the identity tail / zero wrap the format
    # itself uses, so the unfold runs at the tensor's own extent.
    n_pad = int(mu_S_pad) - int(mu_S)
    if n_pad:
        tail = np.arange(mu_S, mu_S_pad, dtype=child_perm.dtype)
        child_perm_p = np.concatenate(
            [child_perm, np.broadcast_to(tail, (child_perm.shape[0], n_pad))],
            axis=1)
        child_L_p = np.concatenate(
            [child_L, np.zeros((child_L.shape[0], n_pad, 3), child_L.dtype)],
            axis=1)
    else:
        child_perm_p, child_L_p = child_perm, child_L

    worst = {name: _unfold_roundtrip_rel(
                 arr, tables, child_perm, child_L, slots, mesh_xy, mu_S)
             for name, arr in small.items()}
    worst_rel = max(worst.values()) if worst else float("nan")

    # THE CONTROL, AND IT IS NOT OPTIONAL.  The route above has three parts
    # that could each be wrong on their own — the wedge-slot derivation, the
    # table restriction, and the unfold call — and a failure tells them apart
    # only against a case whose answer is known.  The PARENT's own tensor is
    # that case: it was STORED on this wedge, with these tables, and the
    # reader unfolded it before this driver saw it, so slicing it back and
    # unfolding it must return it.  If the control fails, the finding is in
    # this harness and not in the physics, and no claim about covariance may
    # be made from the number above.
    control_rel = None
    if parent_tensor is not None and mu_L is not None:
        control_rel = _unfold_roundtrip_rel(
            parent_tensor, tables, np.asarray(tables.sym_perm),
            np.asarray(tables.L_table), slots, mesh_xy, int(mu_L))

    passed = worst_rel < CHILD_COVARIANCE_TOL
    control_ok = (control_rel is not None
                  and control_rel < CHILD_COVARIANCE_TOL)
    lines = [
        f"  [downfold/star] CHILD WEDGE-STORABILITY GATE, on this run's own "
        f"tensors: the child's {len(slots)}-q wedge block unfolded with the "
        f"child's own tables against the full-BZ child --",
    ] + [f"  [downfold/star]   {n}: max rel {v:.3e}"
         for n, v in sorted(worst.items())] + [
        (f"  [downfold/star]   CONTROL, the PARENT's own tensor through the "
         f"SAME route (it was stored on this wedge with these tables, so it "
         f"MUST return): max rel "
         + ("NOT TAKEN" if control_rel is None else f"{control_rel:.3e}")),
        f"  [downfold/star] VERDICT: {'PASS' if passed else 'REFUTED'} "
        f"(worst {worst_rel:.3e} against tol {CHILD_COVARIANCE_TOL:.0e}).",
    ]
    if control_rel is not None and not control_ok:
        lines += [
            f"  [downfold/star] *** THE CONTROL FAILED ({control_rel:.3e}). "
            f"The parent's own tensor does not survive slice-then-unfold "
            f"through this route, and the parent's storage is KNOWN GOOD — so "
            f"the disagreement above is in THIS HARNESS (the wedge-slot "
            f"derivation, the table restriction, or the unfold call), NOT in "
            f"the child's covariance.  NO CLAIM ABOUT WEDGE STORABILITY MAY "
            f"BE MADE FROM THE NUMBER ABOVE. ***",
        ]
        print_fn("\n".join(lines))
        return child_perm, child_L, slots, worst_rel
    if passed:
        lines.append(
            "  [downfold/star] the child's unfold tables are the parent's "
            "restricted to the kept rows (keep[alpha_S(j)] = alpha(keep[j])), "
            "so the phase the child applies is the phase the parent would "
            "have applied to those centroids.  A number at this scale is the "
            "reassociation floor and not an agreement band: the two routes "
            "apply the same unitary at different points of one chain.")
    else:
        # DO NOT NARRATE AN ORDER-ONE NUMBER AS A FLOOR.  The mechanism is
        # measured here rather than guessed at, because the two candidates
        # want opposite fixes and the log is the only place the reader has.
        const, worst_star, spread, n_bad = _star_rank_constancy(
            rank_per_q, tables)
        lines += [
            "  [downfold/star] THE CHILD IS NOT WEDGE-STORABLE ON THIS DECK, "
            "and orbit closure of the selection is not the missing piece — it "
            "HOLDS (the verdict two blocks up).  Closure is what gives the "
            "child an unfold TABLE; it is not what makes the child's TENSORS "
            "unfold.  That needs T[q] = U^S T[i(q)] (U^L)+, and T is built by "
            "a RANK-TRUNCATED solve.",
        ]
        if control_rel is None:
            lines.append(
                "  [downfold/star] the control was NOT TAKEN, so 'the child "
                "is not covariant' and 'this harness is wrong' are not yet "
                "distinguished.  Read the mechanism below as a candidate.")
        if const is None:
            lines.append(
                "  [downfold/star] star-rank constancy: NOT MEASURED (the "
                "per-q rank vector and the q tables have different lengths).")
        elif const:
            lines.append(
                "  [downfold/star] star-rank constancy: the retained rank IS "
                "constant on every star, so a truncation that moves WITHIN a "
                "star is not the mechanism.  Note what that does and does not "
                "exclude: equal ranks across a star are NECESSARY for a "
                "covariant S_SS and nowhere near sufficient, so this clears "
                "the rank-motion signature and leaves the covariance of the "
                "pair-density Gram itself open."
                + ("  With the control at the reassociation floor the "
                   "wedge-slot derivation, the table restriction and the "
                   "unfold call are all exonerated too, so the disagreement "
                   "is in the CHILD's covariance and not in this route.  THE "
                   "MECHANISM IS NOT IDENTIFIED BY THIS RUN.  The decisive "
                   "next measurement is one gather and one unfold: compare "
                   "S_LL[q] against U S_LL[i(q)] U-dagger directly, which "
                   "tests covariance rather than its rank shadow."
                   if control_ok else ""))
        else:
            lines.append(
                f"  [downfold/star] star-rank constancy: **REFUTED** — "
                f"{n_bad} star(s) spend different retained ranks at different "
                f"members, worst spread {spread} at wedge slot {worst_star}.  "
                f"S_SS[q] is therefore NOT covariant across the star: a "
                f"covariant Gram has an identical spectrum at every member, "
                f"so the rank cut cannot move.  The pair-density Gram is "
                f"built over the RETAINED BAND WINDOW, and a window that "
                f"slices a degenerate manifold is exactly how a star-invariant "
                f"quantity stops being one (see the band-degeneracy closure "
                f"work of 2026-08-10).  NO selection rule can repair this; it "
                f"is a statement about the window, not about mu_small.")
        lines.append(
            "  [downfold/star] the child's tables are stored anyway, WITH "
            "this verdict beside them (covariance_worst_rel), because the "
            "next lane needs the number and not a rebuild.  They must not be "
            "used to write a wedge child until the verdict passes.")
    lines.append(
        "  [downfold/star] the child's TENSORS are written on the FULL BZ "
        "either way, and the remaining blocker is named rather than left to "
        "be rediscovered: symmetry_maps.qirr_store refuses to stamp a wedge "
        "without a CentroidClosureVerdict, which is a GEOMETRIC measurement "
        "over the parent's symmetry operations, and those live on a WFN this "
        "driver does not open.  The permutation-level statement above is a "
        "different object and must not be stamped as if it were that one.")
    print_fn("\n".join(lines))
    return child_perm, child_L, slots, worst_rel


class _BandSlices:
    """The five-integer band-window stamp, carried through verbatim.

    ``write_restart_state_to_h5`` wants an object with ``b0..b4`` and stamps
    those five numbers so a later run under a CHANGED window refuses instead
    of silently misindexing.  The downfold does not touch the band axis, so
    the right stamp is the PARENT's, unchanged — a small bundle that claimed
    a different window than the psi and enk it carries would be exactly the
    lie the stamp exists to catch.
    """

    def __init__(self, five):
        self.b0, self.b1, self.b2, self.b3, self.b4 = (int(v) for v in five)


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------

def run_downfold(cfg, mesh_xy, *, print_fn=print) -> DownfoldResult:
    """Read ``cfg.source_restart``, write ``cfg.output_restart``, report."""
    from runtime.padding import padded_mu_extent

    src = resolve_restart_file(cfg.source_restart)
    print_fn(f"  [downfold] parent bundle: {src}")

    with timing.section("downfold.load", announce=True,
                        label="parent restart bundle"):
        geom = _read_geometry(src)
        mu_L = int(geom["n_rmu_logical"])
        window = BandWindow(left=tuple(cfg.band_range_left),
                            right=tuple(cfg.band_range_right))
        window.validate(geom["nb"])
        rs = load_restart_state_from_h5(src, mesh_xy)
        tensors = {}
        for name in geom["present"]:
            if name == "V_qmunu":
                tensors[name] = rs.V_qmunu
            else:
                tensors[name] = read_munu_tensor_from_h5(src, name, mesh_xy)
        n_q = int(tensors["V_qmunu"].shape[0])
        mu_L_pad = int(tensors["V_qmunu"].shape[-1])

    print_fn(
        f"  [downfold] parent: mu_L={mu_L} (in memory {mu_L_pad}), "
        f"n_q={n_q}, nk={geom['nk']}, nb={geom['nb']}, "
        f"nspinor={geom['nspinor']}, kgrid={geom['kgrid']}")
    print_fn(f"  [downfold] retained window: {window.describe()}")
    if not window.symmetric:
        print_fn(
            "  [downfold] NOTE: the window is ASYMMETRIC.  That is the "
            "Sigma-serving shape — Sigma's internal band sum runs over the "
            "full window while its outer projection does not — and the rank "
            "probe measures it at about 2x the mu_S of the symmetric case, "
            "not the order of magnitude the design feared.  The algebra "
            "below is identical, but NO END-TO-END Sigma GATE HAS BEEN RUN "
            "on it: stage 1 was scoped BSE-first.  Treat the result as "
            "unvalidated.")
    if len(geom["present"]) > 2:
        print_fn(f"  [downfold] parent also carries "
                 f"{[n for n in geom['present'] if n not in ('V_qmunu', 'W0_qmunu')]}"
                 f" — they ride the same congruence.")

    # ---- D1: the parent Gram over the retained window -------------------
    with timing.section("downfold.grams", announce=True,
                        label="pair-density Grams"):
        S_LL = pair_density_gram(rs.psi_rmuT_X, rs.psi_rmu_Y, window,
                                 kgrid=geom["kgrid"], mesh_xy=mesh_xy)
        S_LL.block_until_ready()

    # ---- D2: CUR selection, with the rank refusal ------------------------
    with timing.section("downfold.select", announce=True,
                        label="CUR centroid selection"):
        # q = 0 is index 0 of the flat-q axis (the axis is the FFT's own
        # (kx,ky,kz) flattening), and the probe measured the retained rank to
        # be flat in q, so the q = 0 Gram is the right selection Gram.
        S_q0 = jax.lax.with_sharding_constraint(
            S_LL[0], NamedSharding(mesh_xy, P("x", "y")))
        # THE PARENT'S CENTROID SOURCE MAP, when the parent carries one.
        # It is not re-derived from geometry — the parent's own wedge tensors
        # were stored against exactly this table and it travels in the file
        # beside them, so reading it is the only way to be sure the selection
        # is being closed under the SAME permutation the parent's storage
        # rests on.  With it, the selection runs in whole orbits and floors to
        # the user's point budget (owner ruling, 2026-08-10); without it, the
        # selection is point-granular and closure is an ABSENCE.
        parent_tables = _read_parent_unfold_tables(src, mu_L, print_fn)
        keep_idx, sel = select_cur_centroids(
            S_q0, cfg.mu_small, rcond=cfg.downfold_rcond,
            select_tol=cfg.downfold_select_tol, mesh_xy=mesh_xy,
            mu_large_logical=mu_L, print_fn=print_fn,
            sym_perm=(None if parent_tables is None
                      else np.asarray(parent_tables.sym_perm)[:, :mu_L]))
        mu_S = int(sel.mu_small)
        mu_S_pad = int(padded_mu_extent(mu_S, int(jax.device_count())))
    print_fn(sel.describe())
    if sel.eigen_rank_kept < mu_S:
        print_fn(
            f"  [downfold/select] NOTE: {mu_S - sel.eigen_rank_kept} of the "
            f"{mu_S} kept centroids add no independent direction at "
            f"rcond={cfg.downfold_rcond:g}.  The transfer solve will "
            f"truncate about that many modes per q; the small basis is "
            f"smaller than it looks, and mu_small = auto would have reported "
            f"that ceiling up front — a RANK statement, not an accuracy one.")
    if mu_S_pad != mu_S:
        print_fn(
            f"  [downfold] mu_S {mu_S} -> {mu_S_pad} in memory (+"
            f"{mu_S_pad - mu_S} ZERO centroid rows so the extent divides "
            f"both mesh axes).  A zero psi row gives a zero pair density, "
            f"hence a zero row and column of every Gram and a zero "
            f"eigenvalue the truncation drops — the pad is inert, and disk "
            f"stores the logical {mu_S}.")

    # ---- the small basis's coefficients: a column slice ------------------
    psi_S_X, psi_S_Y = slice_psi_to_centroids(
        rs.psi_rmuT_X, rs.psi_rmu_Y, keep_idx, mu_S_pad, mesh_xy)

    with timing.section("downfold.grams_small", announce=False):
        S_cross = pair_density_gram(psi_S_X, rs.psi_rmu_Y, window,
                                    kgrid=geom["kgrid"], mesh_xy=mesh_xy)
        S_SS = pair_density_gram(psi_S_X, psi_S_Y, window,
                                 kgrid=geom["kgrid"], mesh_xy=mesh_xy)
        S_SS.block_until_ready()

    # ---- D3: the transfer solve ------------------------------------------
    with timing.section("downfold.solve", announce=True,
                        label="rank-truncated transfer solve"):
        T_x, T_y, reports = build_transfer(
            S_SS, S_cross, mesh_xy, rcond=cfg.downfold_rcond,
            print_fn=print_fn)
        T_x.block_until_ready()
    rank_per_q = np.array([r.rank_criterion for r in reports], dtype=np.int64)

    # ---- D4: the congruence ----------------------------------------------
    with timing.section("downfold.congruence", announce=True,
                        label="W_S = T W_L T-dagger"):
        project = congruence(mesh_xy, T_x, T_y)
        small = {name: project(arr) for name, arr in tensors.items()}
        g0_S = (transform_head_vector(rs.G0_mu_nu, T_x, 0, mesh_xy)
                if rs.G0_mu_nu is not None else None)
        small["V_qmunu"].block_until_ready()
    if g0_S is not None:
        print_fn(
            "  [downfold] g0_mu (zeta(G=0)) transported as g0_S = "
            "conj(T[q=0]) g0_L.  It is a one-index object, so it does not "
            "take the congruence — and the CONJUGATE is load-bearing: the "
            "q->0 head is the rank-1 s*conj(g0_mu)*g0_nu, and only "
            "conj(T) g0 makes the congruence of that matrix the same "
            "rank-1 form in the small basis.  This inherits the open "
            "'g0_mu placement' owner row on the zeta_loader ledger rather "
            "than answering it.")

    # ---- D5: the error bar ------------------------------------------------
    eps = {}
    norms = {}
    if cfg.report_residual:
        with timing.section("downfold.residual", announce=True,
                            label="Pythagorean error bar"):
            for name in ("W0_qmunu", "V_qmunu"):
                e, n_L, n_S = epsilon_w(tensors[name], S_LL, small[name],
                                        S_SS, mesh_xy)
                eps[name] = e
                norms[name] = (n_L, n_S)
        _announce_residual(eps, print_fn)
        worst = max(float(np.nanmax(v)) for v in eps.values())
        if (cfg.residual_refuse_above is not None
                and worst > float(cfg.residual_refuse_above)):
            raise ValueError(
                f"downfold: worst-q eps_W = {worst:.3e} exceeds the deck's "
                f"residual_refuse_above = {cfg.residual_refuse_above:g}, so "
                f"the small bundle was NOT written.  The retained window "
                f"holds directions the kept centroids do not span; raise "
                f"mu_small (the ceiling at this rcond is "
                f"{sel.eigen_rank_pool}) or widen the window.")
    else:
        print_fn(
            "  [downfold] report_residual = false: the per-q Pythagorean "
            "error bar was NOT computed, and this run therefore has no "
            "answer to 'did the downfold work' that does not require a "
            "reference calculation.  It costs two GEMMs at mu_L per q.")

    # ---- D5b: is the CHILD wedge-storable? --------------------------------
    # Only askable now: it is a statement about the child's TENSORS, and they
    # exist one statement ago.  Only ANSWERABLE at all because the selection
    # ran in whole orbits — a point-granular keep has no child tables, which
    # is what ``child_unfold_tables`` refuses by name.
    child_tables = None
    if parent_tables is not None and sel.star is not None and sel.star.closed:
        with timing.section("downfold.star", announce=False):
            child_tables = _gate_child_wedge_storability(
                small, keep_idx, parent_tables, mesh_xy, mu_S, mu_S_pad,
                rank_per_q=rank_per_q, parent_tensor=tensors.get("V_qmunu"),
                mu_L=mu_L, print_fn=print_fn)
    elif parent_tables is not None:
        print_fn(
            "  [downfold/star] the selection is NOT orbit-closed, so the "
            "child has no unfold tables and the wedge-storability gate did "
            "not run.  In orbit mode this cannot happen; reaching it means "
            "the selection fell back to point granularity.")

    # ---- the write --------------------------------------------------------
    with timing.section("downfold.write", announce=True,
                        label="small restart bundle"):
        out_file = _write_small_bundle(
            cfg, geom, small, g0_S, rs.enk_full, psi_S_Y, keep_idx, mu_S,
            mesh_xy, sel=sel, rank_per_q=rank_per_q, eps=eps,
            child_tables=child_tables, print_fn=print_fn)

    parent_table, fft_grid = resolve_parent_centroids(
        cfg, src, mu_L, geom, os.path.dirname(out_file), print_fn=print_fn)
    cent_file, parent_file = _write_centroid_subset(
        cfg, keep_idx, out_file, parent_table, fft_grid, print_fn=print_fn)

    # ---- D6: the finite-Q leg's operand ----------------------------------
    with timing.section("downfold.zeta", announce=True,
                        label="zeta transported to the small basis"):
        zeta_file = write_downfolded_zeta(
            src, out_file, T_x, keep_idx, mu_S, n_q, geom["kgrid"], g0_S,
            print_fn=print_fn)

    _stamp_isdf_basis(out_file, parent_file, cent_file, zeta_file,
                      print_fn=print_fn)

    return DownfoldResult(
        source_file=src, output_file=out_file, mu_large=mu_L, mu_small=mu_S,
        n_q=n_q, n_bands=geom["nb"], keep_idx=keep_idx, selection=sel,
        rank_per_q=rank_per_q, eps_w=eps, norms=norms,
        wrote_nohead=tuple(n for n in small if n.endswith("_nohead")),
        centroid_file=cent_file, parent_centroid_file=parent_file,
        zeta_file=zeta_file)


def _announce_residual(eps: dict, print_fn) -> None:
    print_fn(
        "  [downfold/residual] eps_W(q) = sqrt(1 - ||W_S||^2/||W||^2), the "
        "relative error of the DOWNFOLDED OBSERVABLE on the retained "
        "window.  It is exact, not an estimate: the fit is an orthogonal "
        "projection, so Pythagoras holds and the residual needs no "
        "reference calculation and never forms the N x N observable.")
    for name, e in eps.items():
        finite = e[np.isfinite(e)]
        if finite.size == 0:
            print_fn(f"  [downfold/residual] {name}: no finite value "
                     f"(||W||^2 was zero at every q)")
            continue
        print_fn(
            f"  [downfold/residual] {name}: eps_W min/median/max = "
            f"{finite.min():.3e} / {np.median(finite):.3e} / "
            f"{finite.max():.3e} over {finite.size} q  "
            f"(worst at q={int(np.nanargmax(e))})")
    print_fn(
        "  [downfold/residual] q=0's absolute norms are head-dominated and "
        "should not be compared across q; the RATIO is still meaningful "
        "there because the head contaminates both norms identically.")


# ---------------------------------------------------------------------------
# the writer
# ---------------------------------------------------------------------------

def _write_small_bundle(cfg, geom, small, g0_S, enk_full, psi_S, keep_idx,
                        mu_S, mesh_xy, *, sel, rank_per_q, eps,
                        child_tables=None, print_fn=print):
    out_dir = cfg.output_restart
    tmp_dir = os.path.join(out_dir, "tmp")
    if process_rank() == 0:
        os.makedirs(tmp_dir, exist_ok=True)
    barrier("downfold.mkdir")
    out_file = os.path.join(tmp_dir, f"isdf_tensors_{mu_S}.h5")

    band_slices = (_BandSlices(geom["band_window"])
                   if geom["band_window"] is not None else None)
    policy = read_coulomb_policy_from_h5(
        resolve_restart_file(cfg.source_restart))

    # THE COULOMB POLICY IS INHERITED VERBATIM, and stamped.  The congruence
    # is LINEAR and applied AFTER the Dyson solve, so the head convention
    # (vhead, whead, mc_average_placement) rides through untouched — but two
    # bundles with different head conventions and no stamp is a
    # silent-disagreement bug of exactly the class the stamp exists to
    # prevent, so the small bundle carries the parent's policy string rather
    # than an empty one.
    write_restart_state_to_h5(
        out_file, n_rmu_logical=int(mu_S),
        V_qmunu=small["V_qmunu"], W0_qmunu=small["W0_qmunu"],
        G0_mu_nu=g0_S, enk_full=enk_full,
        mesh=mesh_xy, mode="w",
        kgrid=tuple(int(v) for v in geom["kgrid"]),
        band_slices=band_slices, coulomb_policy=policy)
    write_restart_state_to_h5(
        out_file, n_rmu_logical=int(mu_S), psi_full_y=psi_S,
        mesh=mesh_xy, mode="a")

    # The _nohead twins, when the parent had them.  Same congruence, same
    # writer; they are read-only opt-ins that nothing in-tree writes, so a
    # downfold that dropped them would quietly make an opt-in unavailable.
    for name in ("V_qmunu_nohead", "W0_qmunu_nohead"):
        if name in small:
            _append_munu(out_file, name, small[name], mu_S, mesh_xy)

    if (geom["vhead"] is not None or geom["whead"] is not None
            or geom["S_cart"] is not None):
        write_head_scalars_to_h5(
            out_file, vhead=geom["vhead"], whead=geom["whead"],
            omega_grid=geom["omega_grid"], S_cart=geom["S_cart"])

    _stamp_downfold_provenance(out_file, cfg, geom, keep_idx, mu_S,
                               sel=sel, rank_per_q=rank_per_q, eps=eps,
                               child_tables=child_tables)
    barrier("downfold.bundle_written")
    print_fn(f"  [downfold] wrote {out_file}  "
             f"(mu {sel.mu_large} -> {mu_S}, "
             f"{(sel.mu_large / max(mu_S, 1)) ** 2:.0f}x smaller per "
             f"(mu,mu) tensor)")
    return out_file


def _append_munu(filename, name, arr, mu_S, mesh_xy):
    """Append one extra (nq, mu, mu) tensor under its own dataset name."""
    from file_io.slab_io import SlabIO
    shape = tuple(list(arr.shape[:-2]) + [int(mu_S), int(mu_S)])
    with SlabIO(filename, mode="a", mesh=mesh_xy) as io:
        io.create_dataset(name, shape=shape, dtype=arr.dtype)
        io.write_slab(name, arr)


def _stamp_downfold_provenance(filename, cfg, geom, keep_idx, mu_S, *,
                               sel, rank_per_q, eps, child_tables=None):
    """Rank-0 group recording what this bundle IS and how it was made.

    A downfolded bundle is indistinguishable from a natively-fitted one by
    shape, and that is the point — but it is NOT the same object, and a
    reader that wants to know can ask.  Everything needed to reproduce the
    compression, or to map an index back to the parent, is here.
    """
    if process_rank() != 0:
        barrier("downfold.provenance")
        return
    with h5py.File(filename, "a") as f:
        if "downfold_provenance" in f:
            del f["downfold_provenance"]
        g = f.create_group("downfold_provenance")
        g.attrs["downfold_provenance_version"] = np.int64(
            DOWNFOLD_PROVENANCE_VERSION)
        g.attrs["mode"] = cfg.mode
        g.attrs["plan"] = cfg.plan
        g.attrs["parent_file"] = str(resolve_restart_file(cfg.source_restart))
        g.attrs["parent_mu"] = np.int64(sel.mu_large)
        g.attrs["mu_small"] = np.int64(mu_S)
        g.attrs["downfold_rcond"] = np.float64(cfg.downfold_rcond)
        g.attrs["downfold_select_tol"] = np.float64(sel.select_tol)
        g.attrs["window"] = np.asarray(
            [cfg.band_range_left[0], cfg.band_range_left[1],
             cfg.band_range_right[0], cfg.band_range_right[1]],
            dtype=np.int64)
        g.attrs["eigen_rank_pool"] = np.int64(sel.eigen_rank_pool)
        g.attrs["select_rank"] = np.int64(sel.select_rank)
        g.attrs["eigen_rank_kept"] = np.int64(sel.eigen_rank_kept)
        g.attrs["r19_anchor"] = R19_ANCHOR
        # THE REQUEST AND THE REALIZATION, BOTH, and in POINTS.  A reader
        # that finds mu_small = 168 in a deck that says 185 must be able to
        # learn here that the difference is the orbit floor and not a typo.
        g.attrs["mu_small_requested"] = np.int64(sel.mu_small_requested)
        g.attrs["selection_granularity"] = (
            "orbit" if sel.orbit_mode else "point")
        if sel.orbit_mode:
            g.attrs["n_orbits_kept"] = np.int64(sel.n_orbits_kept)
            g.attrs["n_orbits_pool"] = np.int64(sel.n_orbits_pool)
            g.attrs["orbit_select_rank"] = np.int64(sel.orbit_rank)
            g.create_dataset("parent_orbit_sizes",
                             data=np.asarray(sel.orbit_sizes, dtype=np.int64))
        if geom.get("centroids_charge_md5") is not None:
            g.attrs["parent_centroids_charge_md5"] = geom[
                "centroids_charge_md5"]
        g.create_dataset("keep_idx", data=np.asarray(keep_idx, dtype=np.int64))
        g.create_dataset("retained_rank_per_q",
                         data=np.asarray(rank_per_q, dtype=np.int64))
        for name, e in eps.items():
            g.create_dataset(f"eps_w_{name}",
                             data=np.asarray(e, dtype=np.float64))
        # THE CHILD'S OWN UNFOLD TABLES, when the selection is orbit-closed.
        # They are the parent's restricted to the kept rows, they were
        # VERIFIED against this run's own tensors before they were written
        # (``_gate_child_wedge_storability``), and they are stored so that a
        # wedge writer for the child does not have to re-derive them — a
        # second derivation of a permutation is a second answer to a question
        # this run already answered.
        if child_tables is not None:
            child_perm, child_L, slots, worst_rel = child_tables
            cg = g.create_group("child_unfold_tables")
            cg.attrs["covariance_worst_rel"] = np.float64(worst_rel)
            cg.attrs["note"] = (
                "child sym_perm/L_table = the parent's restricted to keep_idx;"
                " q-side tables are the parent's, unchanged (the downfold is"
                " q-diagonal).  The child's TENSORS in this file are on the"
                " FULL BZ.")
            cg.create_dataset("sym_perm",
                              data=np.asarray(child_perm, dtype=np.int32))
            cg.create_dataset("L_table",
                              data=np.asarray(child_L, dtype=np.int8))
            cg.create_dataset("q_irr_full_idx",
                              data=np.asarray(slots, dtype=np.int64))
    barrier("downfold.provenance")


def _stamp_isdf_basis(filename, parent_table, kept_table, zeta_file, *,
                      print_fn=print):
    """Where the small bundle's ISDF basis CAME FROM, on the same group.

    ``keep_idx`` says which of the parent's centroid rows survived; these
    three paths say which table those indices index, which table the kept
    rows were written to, and where the transported ζ went.  Together they
    are what lets a consumer rebuild something new in this basis instead of
    only reading the tensors already in it — which is the whole difference
    between ``bse.bse_jax`` (reads) and ``bse.exciton_bands`` (rebuilds ψ at
    finite Q, in the PARENT basis, then slices).

    Absolute paths, deliberately.  They are provenance — a record of what
    this bundle was made from — and a consumer that finds them stale is
    better off being told which file has moved than being handed a relative
    path that silently resolves somewhere else.  Every reader treats a
    missing file as "not available" and says so; none of them requires it.
    """
    if process_rank() != 0:
        barrier("downfold.isdf_basis")
        return
    with h5py.File(filename, "a") as f:
        g = f["downfold_provenance"]
        if parent_table:
            g.attrs["parent_centroids_file"] = str(os.path.abspath(parent_table))
            g.attrs["parent_centroids_md5"] = _md5(parent_table)
        if kept_table:
            g.attrs["centroids_file"] = str(os.path.abspath(kept_table))
        if zeta_file:
            g.attrs["zeta_file"] = str(os.path.abspath(zeta_file))
    have = [n for n, v in (("parent centroid table", parent_table),
                           ("kept centroid table", kept_table),
                           ("transported zeta_q.h5", zeta_file)) if v]
    miss = [n for n, v in (("parent centroid table", parent_table),
                           ("kept centroid table", kept_table),
                           ("transported zeta_q.h5", zeta_file)) if not v]
    print_fn(f"  [downfold] ISDF-basis provenance stamped: "
             f"{', '.join(have) if have else 'nothing'}"
             + (f"; MISSING {', '.join(miss)}" if miss else ""))
    barrier("downfold.isdf_basis")


def _md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


# ---------------------------------------------------------------------------
# D6 — ζ, the one object the finite-Q consumer reads that is not in the bundle
# ---------------------------------------------------------------------------

def write_downfolded_zeta(src_restart, out_file, T_x, keep_idx, mu_S, n_q,
                          kgrid, g0_S, *, print_fn=print) -> str:
    """``zeta_q.h5`` in the small basis, beside the small bundle.

    THE TRANSFORM IS THE HEAD VECTOR'S, AT EVERY G.  ``G0_mu_nu`` is the
    single G = 0 column of ζ̃ and :func:`gw.downfold.transform_head_vector`
    already establishes that it rides ``g0_S = conj(T[q]) g0_L``.  Nothing in
    that derivation used G = 0, so the whole sphere goes the same way:

        zeta_S[q, sigma, G] = conj(T[q])[sigma, mu] * zeta_L[q, mu, G]

    and the consistency is not a hope.  ``V_{mu nu}(q) = sum_G conj(zeta_mu)
    v(q+G) zeta_nu`` gives, on substitution,
    ``V_S = T V_L T-dagger`` — EXACTLY the congruence stage D4 applied to the
    stored tile.  So ``vq_interp``'s own per-q makeVq-vs-disk gate (which
    rebuilds V from ζ and compares against the stored ``V_qmunu`` at 5e-6)
    is a check on this function, and it is the reason no separate numerical
    tolerance is invented here.  The q = 0 / G = 0 column is compared against
    the independently-transported ``g0_S`` before the file is closed, which
    is the cheap end of the same statement.

    WHY THE DOWNFOLD HAS TO DO THIS AT ALL.  ζ is where the finite-Q exchange
    comes from: ``bse.exciton_bands --vq-mode interp`` builds V(Q) off the
    grid by interpolating the stored ζ, and it looks for ``zeta_q.h5`` beside
    the restart file.  A downfolded bundle that carries none is not a drop-in
    for that driver, and the parent's ζ is the WRONG BASIS — μ_L wide against
    a μ_S bundle — so copying it would be worse than leaving it out.
    Transporting it is the only answer that keeps the small bundle honest.

    A q-IBZ ζ TRANSPORTS, AND NO UNFOLD IS NEEDED TO DO IT.  This function
    used to refuse whenever the parent's ζ held fewer q than the bundle,
    calling the gap "the deferred IBZ-zeta unfold".  That was over-broad, and
    the reason is the owner's ruling of 2026-08-10: **the downfold is
    q-DIAGONAL**, so it should be done on ``q_irr`` in the first place.  Every
    stage after the selection acts on one q at a time — ``T[q]`` is built from
    ``S_SS[q]`` and ``S_cross[q]`` alone, and the congruence is per-q — so the
    transfer at a wedge q is the SAME MATRIX whether the surrounding run
    enumerated the wedge or the whole star.  A wedge ζ therefore needs no
    unfold: it needs the transfer AT ITS OWN q, which is one row of the
    transfer this run already holds.  :func:`_zeta_q_to_restart_q` was already
    the function that finds that row — an exact wrapped-integer match on the
    q labels, refusing on anything it cannot place — and the count test in
    front of it was rejecting inputs the matcher handles.  What the count
    genuinely rules out is a ζ on a DIFFERENT grid, and that the matcher
    refuses by name rather than by arithmetic.

    WHAT THIS DOES AND DOES NOT BUY.  The child's ζ comes out on the parent's
    own q set, so a wedge parent yields a wedge child and the child is exactly
    as capable as its parent — which is the drop-in promise, held rather than
    asserted.  It does NOT make ``--vq-mode interp`` work on a wedge lineage:
    ``vq_interp`` still requires ``nq == nk`` and that refusal is untouched
    here, because it is a statement about ITS reader and not about this
    transport.  The difference from before is that the child no longer loses a
    capability the parent had — previously a wedge parent produced a child
    with NO ζ at all, which is strictly worse than the parent and was
    described in the log as taking nothing away.

    Returns the written path, or ``""`` when no ζ was transported.
    """
    from file_io.isdf_header import (IsdfHeader, mark_zeta_done,
                                     read_isdf_header, write_isdf_header)
    from file_io.mf_header import copy_mf_header

    src_zeta = os.path.join(os.path.dirname(os.path.abspath(src_restart)),
                            "zeta_q.h5")
    if not os.path.isfile(src_zeta):
        print_fn(
            f"  [downfold/zeta] the parent run has no {src_zeta} — nothing "
            f"to transport, and the small bundle carries no zeta_q.h5.  "
            f"bse.bse_jax does not read one; bse.exciton_bands --vq-mode "
            f"interp does, and it was equally unavailable on the parent.")
        return ""

    hdr = read_isdf_header(src_zeta)
    if hdr.zeta_layout != "G_flat":
        print_fn(
            f"  [downfold/zeta] parent zeta_q.h5 has zeta_layout="
            f"{hdr.zeta_layout!r}, not 'G_flat'.  No reader in this tree "
            f"consumes an r-space ζ, so there is nothing a transported copy "
            f"would serve; skipped.")
        return ""
    if not bool(hdr.zeta_is_done):
        print_fn(
            "  [downfold/zeta] parent zeta_q.h5 has zeta_is_done = False — "
            "the fit did not finish, so its contents are not ζ.  Skipped; "
            "re-run the parent's ζ fit to completion.")
        return ""

    keep = np.asarray(keep_idx, dtype=np.int64)
    mu_L_file = int(hdr.n_rmu)
    if int(keep.max()) >= mu_L_file:
        raise ValueError(
            f"downfold: the parent's zeta_q.h5 holds {mu_L_file} centroids "
            f"but the kept index set reaches {int(keep.max())}.  That ζ was "
            f"fitted on a DIFFERENT centroid table than the tensors this "
            f"downfold read; transporting it would attach the wrong "
            f"interpolation vectors to the right numbers.")

    from zeta_loader import ZetaLoader
    zl = ZetaLoader(src_zeta, mesh=None)
    try:
        nq_disk = int(np.asarray(zl.ngk_per_q).shape[0])
        if nq_disk > int(n_q):
            # NOT the wedge case.  More ζ tiles than the bundle has q is a
            # grid disagreement, and the one direction the per-q matcher
            # below cannot resolve: every ζ q would still find a bundle q to
            # match, but two of them would have to share one transfer row.
            # Refused here rather than as a duplicate-permutation message
            # three lines later, because the diagnosis is different.
            print_fn(
                f"  [downfold/zeta] parent zeta_q.h5 stores {nq_disk} q "
                f"against the bundle's {n_q} — MORE ζ than there are q in "
                f"the restart.  The two files do not describe the same "
                f"q set, so no ζ is transported.")
            return ""
        if nq_disk < int(n_q):
            print_fn(
                f"  [downfold/zeta] parent zeta_q.h5 stores {nq_disk} q "
                f"against the bundle's {n_q} — the q-IBZ WEDGE, and it "
                f"transports.  The downfold is q-diagonal: T[q] is built "
                f"from S_SS[q] and S_cross[q] alone, so the transfer at a "
                f"wedge q is the same matrix whichever q set the run "
                f"enumerated.  Each stored ζ is matched to its own row of T "
                f"by its q label (exact wrapped-integer match; a q that is "
                f"not on the grid refuses).  The child's ζ lands on the "
                f"PARENT's q set, so the child is exactly as capable as the "
                f"parent — note that vq_interp still requires nq == nk, so "
                f"--vq-mode interp remains unavailable on this lineage, as "
                f"it was on the parent.")

        # T[q] at the LOGICAL extents.  The transfer carries the device-legal
        # μ pads on both axes; ζ on disk is logical on both, and the pad rows
        # are exactly zero, so the slice is a restriction and not a choice.
        #
        # ``gather_to_host``, NOT ``jax.device_get``.  ``T_x`` is μ_L-SHARDED
        # on 'x', so at P>1 its shards live on other processes and
        # ``device_get`` RAISES on it — the trap ``exciton_bands._gather_host``
        # documents at length.  The service branches on
        # ``is_fully_addressable`` and issues the collective only when it is
        # needed, so every rank calls this and they stay in lockstep; the
        # per-rank cost is one (nq, μ_S, μ_L) host buffer, bounded by the
        # dimension this whole exercise makes small.
        from common.collectives import gather_to_host
        T_np = np.asarray(gather_to_host(T_x))[:, :int(mu_S), :mu_L_file]

        # THE SAME SWAP FOR ``g0_S``, AND HOISTED OUT OF THE RANK-0 BRANCH —
        # the two halves are one fix and neither works alone.  ``g0_S`` comes
        # off ``transform_head_vector`` constrained to ``P('y')``, so at P>1
        # it is exactly as non-addressable as ``T_x`` and the gate's bare
        # ``jax.device_get`` refused it with "Fetching value for `jax.Array`
        # that spans non-addressable (non process local) devices" — which is
        # why ``lorrax-downfold`` could not run at P>1 AT ALL, on every rank,
        # after the tensors had already reached disk
        # (``tests/known_failures/2026-08-10-downfold-qirr-star-stability.md``
        # defect 1).  Swapping in ``gather_to_host`` at the old site would
        # have replaced a refusal with a HANG: its third arm is a
        # ``process_allgather``, a collective, and the call sat inside
        # ``if process_rank() == 0``, so ranks 1..P-1 would never enter it.
        # The gather therefore happens HERE, beside ``T_np``, where every
        # rank is; only the comparison and the print stay on the writer.
        # Cost is one μ_S host vector.
        g0_host = (None if g0_S is None
                   else np.asarray(gather_to_host(g0_S)).ravel())

        perm = _zeta_q_to_restart_q(zl, kgrid, nq_disk, n_q=int(n_q),
                                    print_fn=print_fn)

        ngkmax = int(zl.ngkmax_zeta)
        out_zeta = os.path.join(os.path.dirname(out_file), "zeta_q.h5")
        small_hdr = IsdfHeader.build(
            r_mu_fft_idx=np.asarray(hdr.r_mu_fft_idx)[keep],
            fft_grid=_fft_grid_of(hdr),
            density=hdr.density, vertex_mu_L=int(hdr.vertex_mu_L),
            zeta_layout="G_flat",
            gvec_components=np.asarray(hdr.gvec_components),
            ngk_per_q=np.asarray(hdr.ngk_per_q),
            zeta_cutoff_ry=float(hdr.zeta_cutoff_ry))

        if process_rank() == 0:
            if os.path.exists(out_zeta):
                os.remove(out_zeta)
            copy_mf_header(src_zeta, out_zeta, dst_mode="w")
            write_isdf_header(out_zeta, small_hdr, mode="a")
            with h5py.File(out_zeta, "a") as f:
                ds = f.create_dataset(
                    "zeta_q_G", shape=(nq_disk, int(mu_S), ngkmax),
                    dtype=np.complex128, chunks=(1, int(mu_S), ngkmax))
                g0_chk = None
                for q in range(nq_disk):
                    z_L = np.asarray(zl.read_zeta_G_local(q))   # (mu_L, ngk)
                    z_S = np.conj(T_np[perm[q]]) @ z_L[:mu_L_file]
                    ds[q] = z_S
                    if perm[q] == 0:
                        g0_chk = z_S[:, 0].copy()
            mark_zeta_done(out_zeta)
            _gate_g0_against_zeta(g0_chk, g0_host, print_fn)
            print_fn(
                f"  [downfold/zeta] wrote {out_zeta}  (zeta_S = conj(T[q]) "
                f"zeta_L over {nq_disk} q, mu {mu_L_file} -> {mu_S}, "
                f"ngkmax {ngkmax}).  V rebuilt from it is T V_L T-dagger by "
                f"construction — vq_interp's per-q makeVq-vs-disk gate is "
                f"the end-to-end check.")
    finally:
        zl.close()
    barrier("downfold.zeta_written")
    return out_zeta


def _zeta_q_to_restart_q(zl, kgrid, nq_disk, *, n_q=None, print_fn=print):
    """``perm[i]`` = the restart's flat-q index of ζ's q slot ``i``.

    THE TWO AXES ARE NOT THE SAME OBJECT AND ONE OF THEM IS RECONSTRUCTED.
    The restart's q axis is the FFT's own C-order ``(kx, ky, kz)`` flattening
    of ``kgrid`` — that is the axis ``T`` is indexed by.  ζ's q axis is the
    fit writer's, labelled in ``mf_header/kpoints/rk``, which is copied
    VERBATIM from the WFN and therefore holds the IBZ list whenever the
    mean-field run used symmetry, even though the ζ beside it is full-BZ.

    So: when ``rk`` is long enough to label every stored q, MATCH on it (an
    exact wrapped-integer match, no tolerance games).  When it is short, the
    writer's own full-BZ order is the wrapped C-order grid — the same
    reconstruction ``vq_interp.load_zeta_coarse`` makes on the read side, and
    the same one its per-q makeVq-vs-disk gate verifies — so the identity is
    the answer and this says which branch it took.

    Refusing beats guessing here: a permuted q axis writes each q's ζ against
    a different q's transfer, which is a bundle full of plausible numbers.

    THE IDENTITY BRANCH IS FOR A FULL-BZ ζ ONLY, and ``n_q`` is what says so.
    Its premise is that the writer's own order is the wrapped C-order grid,
    which is true of the full-BZ writer and false of the wedge writer — the
    wedge is ``q_irr_frac``'s order, an arbitrary subsequence of the grid, so
    ``arange`` would hand slot ``i`` the transfer of grid point ``i`` and
    write a bundle of plausible numbers.  When a wedge ζ reaches the short-
    ``rk`` branch there is nothing left to match on and this REFUSES; a
    wedge ζ is transportable exactly when its q are labelled, which on a
    symmetry-reduced WFN they are, because ``rk`` IS the irreducible list.
    """
    kg = np.asarray(kgrid, dtype=np.int64)
    idx = np.stack(np.meshgrid(np.arange(kg[0]), np.arange(kg[1]),
                               np.arange(kg[2]), indexing="ij"),
                   axis=-1).reshape(-1, 3)
    keyed = {tuple(int(v) for v in row): i for i, row in enumerate(idx)}
    qraw = np.asarray(zl.kpoints, dtype=np.float64)
    if qraw.shape[0] < nq_disk:
        if n_q is not None and int(nq_disk) < int(n_q):
            raise ValueError(
                f"downfold: the parent's zeta_q.h5 holds {nq_disk} q against "
                f"the bundle's {n_q} — a q-IBZ WEDGE — and "
                f"mf_header/kpoints/rk labels only {qraw.shape[0]} of them, "
                f"so there is nothing to match its q on.  The identity "
                f"fallback below is the FULL-BZ writer's wrapped C-order "
                f"grid; a wedge is q_irr_frac's order, an arbitrary "
                f"subsequence of that grid, so taking the identity would "
                f"attach grid point i's transfer to wedge slot i and write a "
                f"ζ of plausible, wrong numbers.  Re-fit the parent's ζ from "
                f"a WFN whose rk labels every stored q.")
        print_fn(
            f"  [downfold/zeta] mf_header/kpoints/rk labels only "
            f"{qraw.shape[0]} of the {nq_disk} stored q (the WFN is "
            f"symmetry-reduced).  Taking the writer's full-BZ order as the "
            f"wrapped C-order {tuple(int(v) for v in kg)} grid — the same "
            f"reconstruction vq_interp makes on the read side.")
        return np.arange(nq_disk, dtype=np.int64)
    perm = np.empty(nq_disk, dtype=np.int64)
    for i in range(nq_disk):
        key = tuple(int(v) % int(kg[a])
                    for a, v in enumerate(np.rint(qraw[i] * kg)))
        if key not in keyed:
            raise ValueError(
                f"downfold: zeta_q.h5's q #{i} = {qraw[i]} is not on the "
                f"bundle's {tuple(int(v) for v in kg)} grid, so its ζ cannot "
                f"be matched to a column of the transfer T.  The two files "
                f"do not describe the same k-grid.")
        perm[i] = keyed[key]
    if len(set(perm.tolist())) != nq_disk:
        raise ValueError(
            "downfold: zeta_q.h5's q labels are not a permutation of the "
            "bundle's q grid (duplicates after wrapping), so no unique "
            "assignment of T to ζ exists.")
    return perm


def _isdf_fft_grid(src_restart):
    """The FFT grid off the parent's ISDF header, or ``None`` if there is none.

    Only needed on the ``parent_centroids_file`` arm, where the table came
    from the user rather than from the header, and the grid is still wanted so
    the kept subset can be stamped in the hash algebra the tree's own reader
    uses.
    """
    zeta = os.path.join(os.path.dirname(os.path.abspath(src_restart)),
                        "zeta_q.h5")
    if not os.path.isfile(zeta):
        return None
    try:
        from file_io.isdf_header import read_isdf_header
        return _fft_grid_of(read_isdf_header(zeta))
    except (OSError, KeyError, ValueError):
        return None


def _fft_grid_of(hdr):
    """The FFT grid an ``IsdfHeader`` was built against.

    ``r_mu_crystal = r_mu_fft_idx / fft_grid`` is the only place the grid
    survives in the header, so recover it by division rather than by asking
    the caller for a number it would have to get from somewhere else and
    could get wrong.  Integer-exact: both operands come off the same file.
    """
    idx = np.asarray(hdr.r_mu_fft_idx, dtype=np.float64)
    cry = np.asarray(hdr.r_mu_crystal, dtype=np.float64)
    nz = idx != 0.0
    grid = np.zeros(3, dtype=np.float64)
    for a in range(3):
        m = nz[:, a]
        if not np.any(m):
            raise ValueError(
                "downfold: the parent zeta_q.h5 isdf_header has an all-zero "
                "centroid column, so its FFT grid cannot be recovered from "
                "r_mu_fft_idx / r_mu_crystal.")
        grid[a] = np.round(np.median(idx[m, a] / cry[m, a]))
    return grid


def _gate_g0_against_zeta(g0_from_zeta, g0_S_host, print_fn) -> None:
    """ζ_S(q=0, G=0) against the independently-transported ``g0_S``.

    Two routes to the same vector: this file's per-q ``conj(T) ζ_L`` and
    stage D4's ``transform_head_vector``.  They are the same algebra, which
    is exactly why comparing them is worth the two lines — it catches a
    transposed T, a missing conjugate, or a q-axis that is not the one T is
    indexed by, at the moment of writing rather than in an exciton spectrum.
    Reported, not asserted: ``G0_mu_nu`` is absent from bundles whose parent
    never wrote one, and the open 'g0_mu placement' owner row means its
    storage convention is not yet settled enough to hard-fail on.

    ``g0_S_host`` IS ALREADY ON HOST, and that is a precondition rather than
    a convenience.  This function runs on the writer rank only, so it must
    not be the place a collective is issued; the caller gathers ``g0_S``
    through ``common.collectives.gather_to_host`` on every rank before the
    ``process_rank() == 0`` branch is taken.  Fetching here — which is what
    this line used to do, with a bare ``jax.device_get`` — is a refusal at
    P>1 on the sharded input and a deadlock if the fetch is made collective
    in place.
    """
    if g0_S_host is None or g0_from_zeta is None:
        return
    g0 = np.asarray(g0_S_host).ravel()[:g0_from_zeta.shape[0]]
    denom = max(float(np.max(np.abs(g0))), 1e-300)
    rel = float(np.max(np.abs(g0_from_zeta - g0))) / denom
    verdict = "AGREE" if rel < 1e-10 else "DISAGREE"
    print_fn(f"  [downfold/zeta] g0 cross-check: zeta_S(q=0, G=0) vs the "
             f"transported g0_S -> {verdict} (max rel {rel:.3e}).  Both are "
             f"conj(T[0]) applied to the same parent column.")


def _frac_to_fft_idx(rows, fft_grid):
    """Fractional centroid coordinates -> FFT-grid indices.

    ``file_io.centroids.load_centroids``'s arithmetic, restated on host numpy
    so this module can reproduce the index table a consumer will derive from a
    coordinate file — including its wrap of an index that rounds up ONTO the
    grid extent.  Restated rather than imported for the same reason
    :func:`_centroid_table_md5` is: what has to agree is the FORMULA, and
    spelling it here is what makes a disagreement visible in review.
    """
    fg = np.asarray(fft_grid, dtype=np.int64).reshape(3)
    idx = np.rint(np.asarray(rows, dtype=np.float64)[:, :3]
                  * fg[None, :]).astype(np.int64)
    for a in range(3):
        idx[idx[:, a] == fg[a], a] = 0
    return idx


def _centroid_table_md5(fft_idx) -> str:
    """The parent bundle's ``centroids_charge_md5``, recomputed.

    ONE-LINE COPY OF ``gw_init._centroid_table_md5`` AND THAT IS DELIBERATE
    — but it is a copy of the FORMULA, not of a number: int64, C-order
    bytes, md5.  It is restated here rather than imported because
    ``gw.gw_init`` is the GW driver's setup module and pulls the whole
    fitting stack in behind it, and because this file needs the hash for the
    opposite purpose: gw_init STAMPS it, this VERIFIES it.

    The thing worth writing down is what the hash is OF.  It is not the
    content of ``centroids_frac_*.txt``: that file holds FRACTIONAL
    coordinates and the stamp hashes the FFT-GRID INDICES they round to.
    Comparing a text file's own md5 against the stamp therefore never
    matches, which is a mistake this function exists to make unavailable.
    """
    import hashlib
    arr = np.ascontiguousarray(np.asarray(fft_idx, dtype=np.int64))
    return hashlib.md5(arr.tobytes()).hexdigest()


def resolve_parent_centroids(cfg, src_restart, mu_large, geom, out_dir, *,
                             print_fn=print) -> tuple:
    """The PARENT basis's centroid coordinates, made available beside the child.

    WHY THE PARENT'S AND NOT THE KEPT SUBSET'S.  ``bse.exciton_bands`` refits
    psi(k+Q) against a centroid table rather than reading the stored
    coefficients, and that Galerkin fit needs a basis spanning ``nk*nb`` —
    a completely different (and much larger) criterion than the retained
    window's pair-density rank that mu_S was chosen against.  So the fit runs
    at the PARENT's mu and its output is sliced to the kept rows afterwards.
    Without the parent's coordinates that driver cannot open the small bundle
    at all: measured 2026-08-10, a bare ``FileNotFoundError`` out of
    ``np.loadtxt`` naming a file the deck mentioned and the directory did not
    contain, with nothing to say it was a downfold consequence.

    THE COORDINATES ARE NOT HUNTED FOR, THEY ARE DERIVED.  The parent's
    ``zeta_q.h5`` carries ``isdf_header/centroids/r_mu_crystal``, which IS
    the fractional table — ``r_mu_fft_idx / fft_grid``, and
    ``file_io.centroids.load_centroids`` rounds that straight back to the
    same indices — so the table is written out from the parent's own ISDF
    header and the child directory becomes SELF-CONTAINED.  That matters more
    than it sounds: "point the driver at a different directory and everything
    else is as it was" is the downfold's whole promise, and it cannot hold if
    the child needs a file that only exists next to the parent.

    ``parent_centroids_file`` still wins when it is set, because a user who
    has arranged their own table should not be overruled.  When neither is
    available — a parent with no zeta_q.h5 and no named table — the run says
    so and names the consequence rather than leaving it to be discovered.

    Every route is CHECKED against the parent bundle's ``centroids_charge_md5``
    when it carries one: the hash is over the int64 FFT-index table (see
    :func:`_centroid_table_md5`), so a table with the right row count and the
    wrong points is rejected rather than silently attached to the right
    numbers.

    Returns ``(path, fft_grid)``; the path is ``""`` when nothing could be
    resolved, and the grid is ``None`` when the parent carried no ISDF header
    to read one from (it is what turns fractional coordinates into the index
    table ``centroids_charge_md5`` hashes, so a caller without it must not
    stamp).
    """
    stamped = geom.get("centroids_charge_md5")
    if isinstance(stamped, bytes):
        stamped = stamped.decode("utf-8")

    def _verdict(fft_idx):
        """(ok, how) for a candidate's FFT-index table against the stamp."""
        if stamped is None:
            return True, (f"row count {mu_large} only — the parent bundle "
                          f"carries no centroids_charge_md5 to check against")
        if _centroid_table_md5(fft_idx) == stamped:
            return True, "content-verified against centroids_charge_md5"
        return False, "content does NOT match centroids_charge_md5"

    named = str(cfg.parent_centroids_file or "").strip()
    if named:
        if not os.path.isfile(named):
            raise FileNotFoundError(
                f"downfold: parent_centroids_file = {named} does not exist.  "
                f"It names the PARENT run's centroid coordinate table (the "
                f"one its cohsex.in points centroids_file at), not the small "
                f"basis this run is about to produce.")
        print_fn(f"  [downfold] parent centroid table: {named} "
                 f"(named by parent_centroids_file).")
        return named, _isdf_fft_grid(src_restart)

    # The parent's own ISDF header, which is the table by construction.
    src_zeta = os.path.join(os.path.dirname(os.path.abspath(src_restart)),
                            "zeta_q.h5")
    if os.path.isfile(src_zeta):
        try:
            from file_io.isdf_header import read_isdf_header
            hdr = read_isdf_header(src_zeta)
        except (OSError, KeyError, ValueError) as exc:      # noqa: BLE001
            hdr = None
            print_fn(f"  [downfold] parent zeta_q.h5 has no readable "
                     f"isdf_header ({type(exc).__name__}: {exc}); falling "
                     f"back to a search beside the parent bundle.")
        if hdr is not None:
            idx = np.asarray(hdr.r_mu_fft_idx, dtype=np.int64)
            if idx.shape[0] != int(mu_large):
                print_fn(
                    f"  [downfold] parent zeta_q.h5 describes "
                    f"{idx.shape[0]} centroids but the bundle holds "
                    f"{mu_large}; the two are not the same ISDF basis, so "
                    f"its centroid table is not written out.")
            else:
                ok, how = _verdict(idx)
                if not ok:
                    print_fn(
                        f"  [downfold] parent zeta_q.h5's centroid table "
                        f"{how} — the zeta beside the parent bundle was "
                        f"fitted on different points than its tensors.  Not "
                        f"written out; name the right table with "
                        f"parent_centroids_file.")
                else:
                    out = os.path.join(out_dir,
                                       f"centroids_frac_{mu_large}_parent.txt")
                    if process_rank() == 0:
                        np.savetxt(out, np.asarray(hdr.r_mu_crystal,
                                                   dtype=np.float64),
                                   fmt="%.12f")
                    barrier("downfold.parent_centroids")
                    print_fn(
                        f"  [downfold] wrote {out} — the PARENT basis's "
                        f"{mu_large} centroid coordinates, taken from its own "
                        f"zeta_q.h5 isdf_header ({how}).  bse.exciton_bands "
                        f"fits its htransform leg in this basis and slices "
                        f"the result to the kept rows, so the child directory "
                        f"is now self-contained for it.")
                    return out, _fft_grid_of(hdr)

    # Last resort: the table the parent GW run was given, if it is still
    # beside its bundle.  Needs the FFT grid to reproduce the stamp, and
    # without a zeta header there is none — so this arm can only check the
    # row count, and says so.
    bundle_dir = os.path.dirname(os.path.abspath(src_restart))
    search_dirs = [bundle_dir, os.path.dirname(bundle_dir)]
    cands: list = []
    for d in search_dirs:
        cands += sorted(glob.glob(
            os.path.join(d, f"centroids_frac_{mu_large}*.txt")))
    seen = set()
    cands = [c for c in cands if not (c in seen or seen.add(c))]
    for c in cands:
        try:
            rows = np.loadtxt(c, ndmin=2)
        except (OSError, ValueError):
            continue
        if int(rows.shape[0]) != int(mu_large):
            continue
        print_fn(
            f"  [downfold] parent centroid table found beside the parent "
            f"bundle: {c}.  CHECKED ON ROW COUNT ONLY — the parent has no "
            f"zeta_q.h5, so there is no FFT grid here to turn these "
            f"fractional coordinates into the index table centroids_charge_md5 "
            f"hashes.  A regenerated table with the same count and different "
            f"points would pass this check.")
        return c, None

    print_fn(
        f"  [downfold] no parent centroid table: parent_centroids_file is "
        f"unset, the parent has no readable zeta_q.h5 isdf_header, and no "
        f"matching centroids_frac_{mu_large}*.txt was found in "
        f"{' or '.join(search_dirs)}.  The small bundle will serve "
        f"bse.bse_jax exactly as before, but bse.exciton_bands cannot open "
        f"it — its htransform leg needs the PARENT basis's coordinates.  Set "
        f"parent_centroids_file to the parent deck's centroids_file.")
    return "", None


def _write_centroid_subset(cfg, keep_idx, out_file, parent_table, fft_grid, *,
                           print_fn):
    """The kept rows of the parent's centroid table, and the parent's own path.

    The bundle format carries no centroid COORDINATES — only a content hash
    of the table they came from — so a downfolded bundle is complete for
    ``bse.bse_jax`` without this.  It is NOT complete for a fresh GW run on
    the small basis, which needs the kept points, nor for
    ``bse.exciton_bands``, which needs the PARENT's points.  Writing the
    kept rows costs a slice, and stamping the new table's content hash is
    what lets the small bundle claim a centroid table honestly.  Stamping the
    PARENT's hash would be a lie about which points these tensors describe, so
    when no parent table can be resolved the stamp is simply absent and the
    run says so.

    THE STAMP IS THE FFT-INDEX HASH, NOT THE TEXT FILE'S.  This function used
    to write ``md5(bytes of the .txt)``, which is a perfectly good hash of a
    different object: ``centroids_charge_md5`` is defined by
    ``gw_init._centroid_table_md5`` as md5 over the int64 FFT-GRID INDICES,
    and ``gw_init`` compares against it in exactly that algebra.  A bundle
    stamped the other way can never match, so the stamp was not merely
    unverifiable but actively false — a downfolded bundle handed to a fresh
    GW run would have been told its own centroid table was a different one.
    Found 2026-08-10 while teaching the driver to VERIFY a parent table with
    the same hash; the two sides now use one spelling (:func:`_centroid_table_md5`).

    ``fft_grid`` is what turns fractional coordinates into those indices.  It
    comes from the parent's ISDF header; without it there is no stamp, and the
    run says that rather than stamping something in the wrong algebra.

    Returns ``(kept_table_path, parent_table_path)``; either may be ``""``.
    """
    if not parent_table:
        return "", ""
    # Validated on EVERY rank — a refusal raised only on rank 0 would hang
    # the others on the barrier below.  np.loadtxt of a centroid table is
    # host-local and cheap; only the write is rank-gated.
    src = parent_table
    rows = np.loadtxt(src)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[0] <= int(np.max(keep_idx)):
        raise ValueError(
            f"downfold: parent_centroids_file {src} has {rows.shape[0]} rows "
            f"but the kept index set reaches {int(np.max(keep_idx))}.  That "
            f"table is not the one the parent bundle's tensors were built "
            f"on, and slicing it would silently attach the wrong "
            f"coordinates to the right numbers.")
    out = os.path.join(os.path.dirname(out_file),
                       f"centroids_frac_{len(keep_idx)}_downfold.txt")
    kept = rows[np.asarray(keep_idx)]
    if process_rank() == 0:
        np.savetxt(out, kept, fmt="%.12f")
        if fft_grid is None:
            print_fn(
                f"  [downfold] wrote {out} ({len(keep_idx)} kept centroids) "
                f"but stamped NO centroids_charge_md5: that hash is over the "
                f"FFT-grid INDEX table and no FFT grid was available here "
                f"(the parent has no readable zeta_q.h5 isdf_header).  A "
                f"fresh GW run on this basis will re-derive the indices "
                f"itself; it simply has no stamp to check them against.")
        else:
            md5 = _centroid_table_md5(_frac_to_fft_idx(kept, fft_grid))
            with h5py.File(out_file, "a") as f:
                f.attrs["centroids_charge_md5"] = md5
            print_fn(f"  [downfold] wrote {out} ({len(keep_idx)} kept "
                     f"centroids, centroids_charge_md5 {md5[:12]}...) and "
                     f"stamped it on the bundle")
    barrier("downfold.centroids")
    return out, os.path.abspath(src)
