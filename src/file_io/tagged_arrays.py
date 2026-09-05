"""Canonical restart-state I/O for GW/BSE workflows.

This module reads/writes HDF5 restart files in the v2 format used by gw_jax.
"""
from __future__ import annotations

import time

import numpy as np
import jax
import jax.numpy as jnp
import h5py
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import rank0_transaction
from .commit_state import set_commit_state
import common.timing as timing


RESTART_LOGICAL_SHAPE_ATTR = "restart_logical_shape"
RESTART_CARRIER_SHAPE_ATTR = "restart_carrier_shape"
BAND_WINDOW_SCHEMA_DATASET = "band_window_schema"
BAND_WINDOW_SCHEMA_VERSION = 2
BAND_WINDOW_CARRIER_DATASET = "band_window_carrier"
CHARGE_ZETA_IDENTITY_DATASET = "charge_zeta_identity"


def _encode_charge_zeta_identity(receipt):
    """Validate and encode the opaque two-string charge-zeta receipt."""
    if receipt is None:
        return None
    if not isinstance(receipt, dict) or set(receipt) != {"scheme", "digest"}:
        raise ValueError(
            "charge_zeta_identity must contain exactly the two strings "
            "'scheme' and 'digest'")
    values = tuple(receipt[key] for key in ("scheme", "digest"))
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError(
            "charge_zeta_identity scheme and digest must be nonempty strings")
    return np.asarray(values, dtype="S")


def _decode_charge_zeta_identity(value, *, where):
    """Decode a stored receipt without assigning semantics to its strings."""
    raw = np.asarray(value)
    if raw.shape != (2,) or raw.dtype.kind not in ("S", "U", "O"):
        raise ValueError(
            f"{where}: charge-zeta receipt "
            f"{CHARGE_ZETA_IDENTITY_DATASET!r} must be a two-string "
            f"dataset; got shape={raw.shape}, dtype={raw.dtype}")
    out = []
    for item in raw.tolist():
        if isinstance(item, (bytes, np.bytes_)):
            try:
                item = bytes(item).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"{where}: charge-zeta receipt is not UTF-8") from exc
        if not isinstance(item, str) or not item:
            raise ValueError(
                f"{where}: charge-zeta receipt fields must be nonempty strings")
        out.append(item)
    return {"scheme": out[0], "digest": out[1]}


def _logical_storage_shape(shape, logical_axes, logical_extent, *, where):
    """Return a logical disk shape, refusing a short source carrier.

    ``logical_axes`` are the axes whose on-disk extent is exactly
    ``logical_extent``.  A carrier may be larger because a process mesh padded
    it; it may never be smaller.  The old ``min(source, logical)`` spelling
    silently blessed a short carrier and let SlabIO zero-fill data that the
    producer never supplied.
    """
    source = tuple(int(s) for s in shape)
    extent = int(logical_extent)
    if extent < 0:
        raise ValueError(
            f"{where}: logical extent must be nonnegative; got {extent}.")
    out = list(source)
    normalized = []
    for raw_axis in logical_axes:
        axis = int(raw_axis)
        if axis < 0:
            axis += len(source)
        if not 0 <= axis < len(source):
            raise ValueError(
                f"{where}: logical axis {raw_axis} is outside source shape "
                f"{source}.")
        if axis in normalized:
            raise ValueError(
                f"{where}: logical axis {raw_axis} was declared twice for "
                f"source shape {source}.")
        normalized.append(axis)
        if source[axis] < extent:
            raise ValueError(
                f"{where}: source carrier shape {source} is SHORT on axis "
                f"{axis}: extent {source[axis]} < declared logical extent "
                f"{extent}. Refusing before opening or mutating the restart "
                "file; missing physical rows must never be stored as padding.")
        out[axis] = extent
    return tuple(out)


def _mu_logical_shape(shape, mu_axes, n_rmu_logical, *, where="restart tensor"):
    """On-disk (logical) shape for a μ-padded in-memory array.

    Disk contract (SHARDING_RULES §2): files store the LOGICAL μ extent
    so a restart written at any device count re-reads on any other; the
    in-memory pad (``Meta.n_rmu_padded``, zero rows by construction) is
    re-applied on read via ``runtime.padding.padded_mu_extent``.

    Refuses when any declared μ axis is shorter than the logical extent.
    This check belongs before SlabIO opens: a short source is a producer defect,
    not permission for the reader to invent zero rows.
    """
    return _logical_storage_shape(
        shape, mu_axes, n_rmu_logical, where=where)


def _shape_receipt_attrs(carrier_shape, logical_shape):
    """Dataset attrs that keep producer carrier and disk shape distinct."""
    return {
        RESTART_CARRIER_SHAPE_ATTR: np.asarray(
            tuple(int(v) for v in carrier_shape), dtype=np.int64),
        RESTART_LOGICAL_SHAPE_ATTR: np.asarray(
            tuple(int(v) for v in logical_shape), dtype=np.int64),
    }


def _band_window_receipts(band_slices):
    """Return logical identity plus the producer's padded band carrier.

    The first four edges are already physical indices. ``b4`` is a storage
    edge and may be mesh padded; ``b4_logical`` is the physical loaded top.
    The chi/sigma tops use the same convention as ``Meta``: clipping either
    against the logical loaded top removes only the zero-band carrier tail.
    """
    carrier = tuple(int(getattr(band_slices, f"b{i}")) for i in range(5))
    logical_top = int(getattr(band_slices, "b4_logical", 0) or carrier[4])
    if not carrier[0] <= carrier[1] <= carrier[2] <= carrier[3] <= logical_top:
        raise ValueError(
            "write_restart_state_to_h5: invalid logical band window "
            f"{carrier[:4] + (logical_top,)} derived from carrier {carrier}.")
    if logical_top > carrier[4]:
        raise ValueError(
            "write_restart_state_to_h5: logical loaded-band top "
            f"{logical_top} exceeds carrier top {carrier[4]}.")
    logical = carrier[:4] + (logical_top,)
    split = (
        min(int(getattr(band_slices, "b4_chi", carrier[4])), logical_top),
        min(int(getattr(band_slices, "b4_sigma", carrier[4])), logical_top),
    )
    if min(split) < carrier[2] or max(split) != logical_top:
        raise ValueError(
            "write_restart_state_to_h5: logical chi/Sigma tops "
            f"{split} are inconsistent with band window {logical}.")
    return logical, carrier, split


def _validate_shape_receipt(name, ds) -> None:
    """Refuse an authenticated dataset whose stored shape changed."""
    if RESTART_LOGICAL_SHAPE_ATTR not in ds.attrs:
        return
    stamped = tuple(int(v) for v in np.asarray(
        ds.attrs[RESTART_LOGICAL_SHAPE_ATTR]).reshape(-1))
    actual = tuple(int(v) for v in ds.shape)
    if stamped != actual:
        raise ValueError(
            f"Restart dataset {name!r}: stamped logical storage shape "
            f"{stamped} does not match actual dataset shape {actual}. The file "
            "is torn or hand-edited; regenerate it with restart=false.")
    if RESTART_CARRIER_SHAPE_ATTR not in ds.attrs:
        raise ValueError(
            f"Restart dataset {name!r} has a logical storage receipt but no "
            f"{RESTART_CARRIER_SHAPE_ATTR!r}. The authenticated receipt is "
            "partial; regenerate the file with restart=false.")
    carrier = tuple(int(v) for v in np.asarray(
        ds.attrs[RESTART_CARRIER_SHAPE_ATTR]).reshape(-1))
    if len(carrier) != len(actual) or any(
            c < s for c, s in zip(carrier, actual)):
        raise ValueError(
            f"Restart dataset {name!r}: producer carrier receipt "
            f"{carrier} cannot cover logical storage shape {actual}. The "
            "file is internally inconsistent.")


def _restart_write_log_on() -> bool:
    """Rank-0 owner for debug-only per-dataset storage telemetry.

    Large writes can take long enough that this detail is valuable while
    diagnosing a run.  It is nevertheless storage-library chatter rather
    than a physics result, so production mode stays quiet and the driver's
    one debug switch restores it.
    """
    from runtime import debug_print_enabled
    return debug_print_enabled() and jax.process_index() == 0


def _log_restart_write(name, shape, dtype, dt) -> None:
    """One ``[restart_write]`` line for a dataset handed to the writer.

    Single implementation on purpose: the AF.4c line format is
    load-bearing for log diagnosis, and this module used to carry four
    hand-synced copies of it (audit 2026-07-28; QUALITY_PATTERNS #3).
    Emitted by the rank-zero owner selected by
    :func:`_restart_write_log_on` when driver debug printing is enabled.

    ``dt`` is the CALLER's elapsed time, and SlabIO's write path returns
    as soon as the tile is queued on its writer thread — so ``dt`` is the
    enqueue, not the transfer, and this line must not present it as one.
    It used to print ``nb/dt`` as a bandwidth, which reports thousands of
    MB/s for a dispatch that has moved nothing.  Worse, the flush it hid
    did not vanish: ``create_dataset`` drains the writer before H5Dcreate
    (the MPI datatype-cache interleave), so each dataset's ``dt`` is
    mostly the PREVIOUS dataset's transfer, and a multi-GB tensor's write
    surfaced against the scalar or header array logged after it, at an
    apparent 0 MB/s.  Both halves of that read as a writer pathology and
    neither is one.  So the size and the rate come from the
    ``SlabIO.close`` drain line, which is the first moment any of the
    bytes are on disk, and ``dt`` is reported here named for what it is.
    """
    if not _restart_write_log_on():
        return
    nb = int(np.prod(shape)) * int(np.dtype(dtype).itemsize)
    print(f"  [restart_write] {name} {tuple(int(v) for v in shape)}"
          f" {nb / 1e9:.2f} GB QUEUED in {dt:.1f} s"
          f" (transfer time and rate: see the SlabIO.close drain line)",
          flush=True)


# ---------------------------------------------------------------------------
# Coulomb-kernel policy stamp
# ---------------------------------------------------------------------------
#
# THE DEFECT THIS CLOSES.  The restart file records ``V_ready``, the band
# window, ``n_rmu``, the k-grid, the q-set symmetry tables and the centroid
# md5s — and, until this stamp, NOTHING about the Coulomb kernel.  A
# ``restart = true`` run reuses ``V_qmunu`` verbatim and never re-runs
# ``compute_V_q`` (``gw_init.py``'s restart branch), so ANY change of
# averaging policy — ``mc_average_vcoul_body``, the mini-BZ placement, the
# bare-Coulomb cutoff, the BGW vcoul overlay — is inherited silently by
# every existing restart, with every current guard passing.  This is the
# same defect class as the band-window bug the file's own comments cite
# (job 7874375: window-70 tensors reused at window 80 gave a QP gap of
# -135 eV while every stage reported success).
#
# It is a WARNING, not a refusal, and the asymmetry is deliberate.  The
# band window changes what the tensors are INDEXED by, so reusing them is
# wrong with no way to be right.  A Coulomb-policy change makes the stored
# V a legitimate tensor built under a different convention — sometimes
# exactly what the operator wants (re-scoring an old restart against a new
# Sigma path), sometimes a silent physics change.  The failure to remove is
# the SILENCE, so the stamp is loud and the decision stays the operator's.
#
# Files written before the stamp read as legacy and get one line saying so,
# because "no stamp" and "a stamp that matches" are different facts and a
# reader that conflates them has re-created the original defect one level up.

COULOMB_POLICY_DATASET = "coulomb_policy"
COULOMB_POLICY_VERSION = 1

#: The keys stamped, in order.  Anything that changes ``v(q+G)`` or where
#: its mini-BZ average lands belongs here; anything that does not, does not.
COULOMB_POLICY_KEYS = (
    "mc_average_vcoul_body",
    "mc_average_placement",
    "mc_average_placement_vcoul",
    "head_minibz_average",
    "bare_coulomb_cutoff",
    "use_bgw_vcoul",
    "bgw_vcoul_file",
    "sys_dim",
)


def coulomb_policy_from_config(cfg, meta=None) -> dict:
    """The Coulomb-kernel policy of a running config, as a flat str dict.

    Reads ``cfg.head`` (and ``meta.sys_dim``) rather than being handed the
    values, so a key added to ``HeadConfig`` and to
    :data:`COULOMB_POLICY_KEYS` is stamped without a third edit at every
    call site.
    """
    head = getattr(cfg, "head", None)
    out = {}
    for k in COULOMB_POLICY_KEYS:
        if k == "sys_dim":
            v = getattr(meta, "sys_dim", None) if meta is not None else None
        else:
            v = getattr(head, k, None)
        out[k] = _policy_scalar(v)
    return out


def _policy_scalar(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(float(v))
    return str(v)


def format_coulomb_policy(policy: dict) -> str:
    """``v1;key=value;...`` — one line, sorted by :data:`COULOMB_POLICY_KEYS`.

    Deliberately a readable string rather than JSON or a pickle: the whole
    point of a provenance stamp is that ``h5dump`` answers the question
    without LORRAX in the loop.
    """
    body = ";".join(
        f"{k}={_policy_scalar(policy.get(k))}" for k in COULOMB_POLICY_KEYS)
    return f"v{COULOMB_POLICY_VERSION};{body}"


def parse_coulomb_policy(raw) -> dict | None:
    """Inverse of :func:`format_coulomb_policy`; ``None`` for an unstamped file.

    Tolerates unknown keys (a file written by a newer LORRAX) and missing
    keys (an older one) — both come back in the dict as they are, and the
    comparison below reports them as differences rather than crashing.  A
    stamp is provenance; failing to READ one must never be worse than not
    having it.
    """
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    elif isinstance(raw, np.ndarray):
        raw = bytes(raw.tobytes()).decode("utf-8", "replace").rstrip("\x00")
    raw = str(raw).strip()
    if not raw:
        return None
    parts = raw.split(";")
    if parts and parts[0].startswith("v"):
        parts = parts[1:]
    out = {}
    for p in parts:
        if "=" not in p:
            continue
        k, _, v = p.partition("=")
        out[k.strip()] = v.strip()
    return out or None


def compare_coulomb_policy(stamped: dict | None, running: dict) -> list:
    """Return ``[(key, stamped, running), ...]`` for every disagreement.

    Empty list means the file was built under this run's Coulomb policy.
    ``stamped is None`` (legacy file) is NOT a disagreement — it is the
    absence of evidence, and the caller says so in its own words.
    """
    if stamped is None:
        return []
    keys = list(dict.fromkeys(list(stamped.keys()) + list(running.keys())))
    return [(k, stamped.get(k, "<absent>"), running.get(k, "<absent>"))
            for k in keys if stamped.get(k, "<absent>") != running.get(k, "<absent>")]


def read_coulomb_policy_from_h5(filename) -> dict | None:
    """Read the stamp off a restart file with serial h5py; ``None`` if absent.

    Scalar-class metadata, read the same way ``assert_restart_window_matches``
    reads the band window — no SlabIO handle, no collective, safe to call on
    any rank before the tensors move.
    """
    try:
        with h5py.File(filename, "r") as f:
            if COULOMB_POLICY_DATASET not in f:
                return None
            return parse_coulomb_policy(f[COULOMB_POLICY_DATASET][()])
    except (OSError, KeyError):
        return None


#: The group ``gw.downfold_run`` stamps on a compressed bundle.  Named here,
#: beside the Coulomb-policy stamp, because it is part of the RESTART FORMAT
#: and has two sides: the downfold writes it and the BSE drivers read it.  A
#: reader that needed its own copy of the group name would be a second owner
#: of a format detail, which is the drift this module exists to prevent.
DOWNFOLD_PROVENANCE_GROUP = "downfold_provenance"


def read_downfold_provenance(filename) -> dict | None:
    """The ``downfold_provenance`` group of a restart bundle; ``None`` if absent.

    ``None`` means "natively fitted, as far as this file says" — a bundle
    written by ``gw.gw_jax`` carries no such group, and so does one written
    by a downfold predating the stamp.  Both are read as not-downfolded,
    which is the safe direction: every consumer's existing behaviour is what
    it gets.

    WHY A READER LIVES HERE AT ALL.  A downfolded bundle is deliberately
    indistinguishable from a natively fitted one BY SHAPE — that is what
    makes it a drop-in for ``bse.bse_jax``.  But two facts about it are not
    derivable from shape and are load-bearing for any consumer that has to
    build something NEW in the same ISDF basis rather than only read the
    stored tensors: which centroid table the parent basis came from, and
    which of the parent's centroid rows survived.  ``bse.exciton_bands``
    needs both (its htransform leg fits ψ in the PARENT basis and slices the
    result to the kept rows), and this is the one place either is recorded.

    Serial h5py, no SlabIO handle, no collective — the same contract as
    :func:`read_coulomb_policy_from_h5`, so it is safe to call on any rank
    before the tensors move.

    Returns the group's attributes as a plain dict (bytes decoded to str),
    plus ``keep_idx`` / ``retained_rank_per_q`` as numpy arrays when present.
    """
    try:
        with h5py.File(filename, "r") as f:
            if DOWNFOLD_PROVENANCE_GROUP not in f:
                return None
            g = f[DOWNFOLD_PROVENANCE_GROUP]
            out = {}
            for k, v in g.attrs.items():
                out[k] = v.decode("utf-8") if isinstance(v, bytes) else v
            for name in ("keep_idx", "retained_rank_per_q"):
                if name in g:
                    out[name] = np.asarray(g[name][:])
            return out
    except (OSError, KeyError):
        return None


def describe_coulomb_policy_stamp(filename) -> str:
    """One line naming the Coulomb policy a restart file's tensors carry.

    For readers that CONSUME W rather than rebuild V — the BSE, which by
    its own note "does NOT compute W, it READS it off the GW restart".
    They have no Coulomb config of their own to compare against, so the
    honest disclosure is the stored policy itself, not a match verdict.
    Without this line a BSE run's log has no record of which averaging
    convention its screening was built under, which is exactly the gap
    that made the cross-code residual arguable in the first place.
    """
    stamped = read_coulomb_policy_from_h5(filename)
    if stamped is None:
        return ("  [restart stamp] Coulomb-kernel policy: NOT STAMPED "
                "(GW restart predates the stamp) - the screening in this "
                "file was built under an unrecorded averaging convention.")
    return ("  [restart stamp] screening built under Coulomb policy: "
            + ";".join(f"{k}={v}" for k, v in stamped.items()))


def describe_coulomb_policy_match(filename, cfg, meta=None) -> str:
    """One line for a restart log: matched, mismatched, or legacy-unstamped.

    Returns the text; the caller prints it, so this stays importable from
    the BSE side (which reads the same file and owes the same disclosure)
    without either side owning the other's print function.
    """
    running = coulomb_policy_from_config(cfg, meta)
    stamped = read_coulomb_policy_from_h5(filename)
    if stamped is None:
        return ("  [restart stamp] Coulomb-kernel policy: NOT STAMPED "
                "(file predates the stamp). Read as legacy — the stored V/W "
                "were built under whatever averaging policy that run used, "
                "and this run cannot tell which. Running policy is "
                f"{format_coulomb_policy(running)}")
    diffs = compare_coulomb_policy(stamped, running)
    if not diffs:
        return (f"  [restart stamp] Coulomb-kernel policy matches: "
                f"{format_coulomb_policy(stamped)}")
    detail = "; ".join(f"{k}: file={a!r} run={b!r}" for k, a, b in diffs)
    return (
        "  [restart stamp] WARNING - Coulomb-kernel policy MISMATCH between "
        "this restart file and the running config. The restart reuses "
        "V_qmunu verbatim and never re-runs compute_V_q, so the stored "
        "tensors carry the FILE's policy and every other guard will pass. "
        f"Differences -> {detail}. Rerun with restart = false if the "
        "running policy is the one you meant.")


def write_restart_state_to_h5(
    filename,
    *,
    n_rmu_logical: int,
    V_qmunu=None,
    psi_full_y=None,
    psi_full_y_mun=None,
    psi_full_y_transverse=None,
    psi_full_y_transverse_mun=None,
    n_rmu_transverse_logical: int | None = None,
    enk_full=None,
    S_qmunu=None,
    V0_noG0_munu=None,
    G0_mu_nu=None,
    W0_qmunu=None,
    init_W0: bool = False,
    mesh=None,
    mode: str = "w",
    kgrid: tuple[int, int, int] | None = None,
    band_slices=None,
    qirr=None,
    coulomb_policy=None,
    qp_state_source_record: dict | None = None,
    charge_zeta_identity: dict | None = None,
):
    """Write (subset of) canonical restart state via SlabIO.

    All array arguments are optional — only the provided ones are
    written, so this function can be called multiple times to flush
    pieces of the restart state as they become available.  With
    ``mode="w"`` the file is truncated first (and the format-version
    attribute written); with ``mode="a"`` the file is opened for
    append / overwrite of the named datasets.

    ``n_rmu_logical`` (= ``meta.n_rmu``) is the μ extent stated on
    disk; SlabIO clips the in-memory pad rows against it.  In-memory arrays
    carry the P-dependent padded extent ``meta.n_rmu_padded`` whose pad
    rows are exact zeros, and persisting them verbatim would make the
    restart file unreadable at a different device count (the
    ROOT_CAUSE.md defect class, one hop downstream).
    ``load_restart_state_from_h5`` re-pads on read.

    ``init_W0=True`` pre-allocates an all-zeros W0_qmunu dataset sized
    from ``V_qmunu``; the ``W0_ready`` attr on that dataset is set to
    False so downstream readers (bse_io) know to treat it as a
    placeholder.  Passing ``W0_qmunu`` directly flips ``W0_ready`` to
    True.

    ``psi_full_y_mun`` (``low_mem_bands = true`` only) is the SECOND face
    of the two-face carrier (``gw.wavefunction_bundle`` ``psi_mun``,
    ``(nk, s, μ, n)``) — ADDITIVE to ``psi_full_y``, which under
    ``low_mem_bands`` is sourced from the FIRST face (``psi_nmu``) rather
    than the legacy ``psi_yr``.  Both share ``psi_full_y``'s on-disk μ
    clip (``n_rmu_logical``); nothing else about the ``psi_full_y``
    schema changes, so BSE/downfold — which read only ``psi_full_y`` at
    the legacy y-only spec — are unaffected regardless of which layout
    wrote it.  Writing two datasets instead of one buys the reader a
    direct hyperslab for EACH face and therefore zero reshard collectives
    on restart read (see ``file_io.load_restart_state_from_h5``,
    ``low_mem_bands=True``); the cost is doubled ψ bytes on disk.

    ``psi_full_y_transverse`` (bispinor only) is the σ^B-side ψ sampled
    at the TRANSVERSE centroid set — the per-channel second ψ dataset
    the bispinor restart round-trip needs.  Its μ axis is clipped by
    ``n_rmu_transverse_logical`` (the transverse centroid count, which
    differs from the charge ``n_rmu_logical``).  The count is also
    stamped as the ``n_rmu_transverse_logical`` dataset:
    ``read_restart_state_from_h5`` cross-checks it against the stored
    dataset's μ extent at load (torn/hand-edited-file guard); the
    loader itself re-pads from the dataset shape
    (``load_restart_state_from_h5`` → ``n_rmu_transverse_disk``).

    ``qirr`` IS THE ONE RESOLUTION FOR BOTH TENSORS.  A
    ``gw.restart_q_storage.RestartQStorage`` whose ``.store_wedge`` is True
    means V is written on the IBZ q wedge (from ``qirr.capture.X_ibz``, the
    PRE-UNFOLD block the producer held) and the W0 placeholder is sized from
    THAT — which is the coupling this function's comment has been asking the
    next writer to close since dbe3b4ec.  ``None`` (the default, and what
    every existing caller passes) is today's behaviour exactly: full-BZ V,
    full-BZ placeholder, no stamp, no table group.

    ``qp_state_source_record`` identifies the WFN whose matched
    ``psi_full_y`` / ``enk_full`` state this restart stores.  Its format and
    serialization belong only to :mod:`file_io.qp_wfn`; this writer transports
    the opaque bytes through the incumbent SlabIO metadata path on ``mode=w``.
    """
    from .slab_io import SlabIO

    encoded_charge_zeta_identity = _encode_charge_zeta_identity(
        charge_zeta_identity)
    if encoded_charge_zeta_identity is not None and mode != "w":
        raise ValueError(
            "charge_zeta_identity is immutable restart provenance and may "
            "only be stamped by the mode='w' transaction")

    # ---- THE ONE RESOLUTION, APPLIED ONCE, BEFORE ANY WRITE -----------
    # Both the tensor and the placeholder are decided here, together, from
    # one object.  The old shape-inheritance is what made V's decision
    # silently become W0's; the substitution below is the whole of the
    # change, and the placeholder block further down reads ``V_on_disk``
    # rather than ``V_qmunu`` for exactly that reason.
    V_on_disk = V_qmunu
    if qirr is not None and qirr.store_wedge and V_qmunu is not None:
        if qirr.capture is None:
            raise ValueError(
                "write_restart_state_to_h5: restart_q_storage resolved to "
                "'ibz' but no pre-unfold capture reached the writer.  The "
                "wedge exists for one statement inside the V_q producer and "
                "is offered to an open capture scope there; a resolution "
                "that says 'ibz' with nothing captured would otherwise "
                "SLICE the unfolded tensor, which is a different array whose "
                "equality to the wedge depends on an op-selection policy "
                "nobody froze for this purpose.")
        V_on_disk = qirr.capture.X_ibz

    # Resolve EVERY logical storage shape before SlabIO opens.  SlabIO's
    # ``mode='w'`` replaces the inode during construction; discovering a short
    # producer inside ``_write`` would therefore already have destroyed the
    # previous file.  The plan also keeps the producer carrier shape beside the
    # logical disk shape instead of making readers reverse-engineer one from
    # the other.
    band_receipts = None
    n_band_logical = None
    if band_slices is not None:
        band_receipts = _band_window_receipts(band_slices)
        n_band_logical = int(band_receipts[0][4] - band_receipts[0][0])
    elif mode != "w" and any(
            arr is not None for arr in (
                psi_full_y, psi_full_y_mun, psi_full_y_transverse,
                psi_full_y_transverse_mun, enk_full)):
        # Append calls intentionally do not repeat band_slices.  A schema-2
        # file's band_window is logical, so it is sufficient to clip the later
        # psi faces to the same portable disk extent.  A legacy file has no
        # way to distinguish physical rows from its mesh pad and stays on its
        # historical full-carrier storage path.
        with h5py.File(filename, "r") as f:
            schema = (int(np.asarray(f[BAND_WINDOW_SCHEMA_DATASET])[()])
                      if BAND_WINDOW_SCHEMA_DATASET in f else None)
            if schema == BAND_WINDOW_SCHEMA_VERSION and "band_window" in f:
                stored_logical = tuple(
                    int(v) for v in np.asarray(f["band_window"]).reshape(-1))
                if len(stored_logical) != 5:
                    raise ValueError(
                        f"Restart file {filename}: schema-{schema} band_window "
                        f"has {len(stored_logical)} entries, expected 5.")
                n_band_logical = stored_logical[4] - stored_logical[0]

    if ((psi_full_y_transverse is not None
         or psi_full_y_transverse_mun is not None)
            and n_rmu_transverse_logical is None):
        raise ValueError(
            "write_restart_state_to_h5: transverse psi requires "
            "n_rmu_transverse_logical (the transverse centroid count).")
    n_T = (int(n_rmu_transverse_logical)
           if n_rmu_transverse_logical is not None else None)

    write_plan = {}

    def _plan(name, arr, *, mu_axes=(), n_logical=None, band_axes=()):
        if arr is None:
            return
        n_log = n_rmu_logical if n_logical is None else n_logical
        shape = _mu_logical_shape(
            arr.shape, mu_axes, n_log,
            where=f"write_restart_state_to_h5 dataset {name!r}")
        if band_axes and n_band_logical is not None:
            shape = _logical_storage_shape(
                shape, band_axes, n_band_logical,
                where=f"write_restart_state_to_h5 dataset {name!r} band axis")
        write_plan[name] = (
            shape, _shape_receipt_attrs(arr.shape, shape))

    _plan("V_qmunu", V_on_disk, mu_axes=(-2, -1))
    _plan("S_qmunu", S_qmunu, mu_axes=(-2, -1))
    _plan("V0_noG0_munu", V0_noG0_munu, mu_axes=(-2, -1))
    _plan("G0_mu_nu", G0_mu_nu, mu_axes=(-1,))
    _plan("psi_full_y", psi_full_y, mu_axes=(-1,), band_axes=(1,))
    _plan("psi_full_y_mun", psi_full_y_mun, mu_axes=(-2,), band_axes=(-1,))
    _plan("enk_full", enk_full, band_axes=(-1,))
    _plan("psi_full_y_transverse", psi_full_y_transverse,
          mu_axes=(-1,), n_logical=n_T, band_axes=(1,))
    _plan("psi_full_y_transverse_mun", psi_full_y_transverse_mun,
          mu_axes=(-2,), n_logical=n_T, band_axes=(-1,))
    _plan("W0_qmunu", W0_qmunu, mu_axes=(-2, -1))

    if init_W0 and W0_qmunu is None:
        if V_qmunu is None:
            raise ValueError("init_W0=True requires V_qmunu to size the placeholder")
        # V_on_disk, not the possibly full-BZ V_qmunu, is the resolved q
        # storage.  Its preflight above covers the placeholder too.
        v_shape, v_attrs = write_plan["V_qmunu"]
        write_plan["W0_qmunu"] = (v_shape, dict(v_attrs))

    with SlabIO(filename, mode=mode, mesh=mesh,
) as io:
        if mode == "w":
            io.write_attr("restart_format_version", np.int64(2))
            if qp_state_source_record is not None:
                from .qp_wfn import (
                    QP_STATE_SOURCE_DATASET,
                    encode_qp_state_source_provenance,
                )
                io.write_attr(
                    QP_STATE_SOURCE_DATASET,
                    encode_qp_state_source_provenance(
                        qp_state_source_record))
            if encoded_charge_zeta_identity is not None:
                io.write_attr(
                    CHARGE_ZETA_IDENTITY_DATASET,
                    encoded_charge_zeta_identity)
        # kgrid attr lets BSE recover the (nkx,nky,nkz) split from
        # flat-q V_qmunu / W0_qmunu without re-opening the WFN.  Stored
        # as a length-3 int64 dataset (the SlabIO ``write_attr`` path
        # accepts list/tuple).  Optional: callers that don't pass it
        # leave the attr unset; BSE falls back to reading WFN.
        if kgrid is not None and mode == "w":
            io.write_attr("kgrid", np.asarray(kgrid, dtype=np.int64))
        # BAND-WINDOW PROVENANCE.  V_qmunu / psi_full_y / enk_full are all
        # indexed by the band window they were BUILT under; a restart that
        # changes nval/ncond/nband re-reads them under a different window and
        # silently misindexes Sigma -- no crash, just wrong physics (job
        # 7874375: window 70 tensors reused at window 80 gave a QP gap of
        # -135 eV while every stage reported success).  Stamp the window here
        # so :func:`assert_restart_window_matches` can refuse that on load.
        if band_receipts is not None and mode == "w":
            logical_window, carrier_window, logical_split = band_receipts
            # Schema 2 changes ``band_window`` from a P-dependent carrier
            # receipt to the physical identity.  The producer carrier remains
            # available under its own explicit name; readers never compare it
            # as physics.  Presence of the schema dataset makes legacy files
            # unambiguous.
            if enk_full is not None:
                actual_top = logical_window[0] + int(enk_full.shape[-1])
                carrier_window = carrier_window[:4] + (actual_top,)
            io.write_attr(BAND_WINDOW_SCHEMA_DATASET,
                          np.int64(BAND_WINDOW_SCHEMA_VERSION))
            io.write_attr("band_window", np.asarray(
                logical_window, dtype=np.int64))
            io.write_attr(BAND_WINDOW_CARRIER_DATASET, np.asarray(
                carrier_window, dtype=np.int64))
            # THE χ / Σ SPLIT, IN A SEPARATE ATTR ON PURPOSE (2026-08-16).
            # Widening ``band_window`` from 5 entries to 7 would have made
            # every restart file already on disk compare unequal and strand
            # it.  A new attr instead: absent == "written by an unsplit run",
            # which resolves to (b4, b4) and matches an unsplit run exactly.
            io.write_attr("band_window_split", np.asarray(
                logical_split, dtype=np.int64))
        if mode == "w":
            io.write_attr("n_rmu_logical", np.int64(int(n_rmu_logical)))
        # COULOMB-KERNEL PROVENANCE.  Unconditional on the ``w`` pass: a
        # restart written without it is exactly the file this stamp exists
        # to stop producing, so there is no opt-in.  Callers that pass
        # nothing get the stamp with empty values, which still records
        # "this writer knew about the policy and was handed none" — a
        # different and more useful fact than an absent dataset.
        if mode == "w":
            io.write_attr(
                COULOMB_POLICY_DATASET,
                np.asarray(format_coulomb_policy(coulomb_policy or {})
                           .encode("utf-8"), dtype="S"))

        # DEBUG PER-DATASET LIVENESS (scorecard AF.4c): in driver debug
        # mode every dataset below emits one [restart_write] line naming
        # its size as it is handed to the writer thread.  The transfer is
        # asynchronous and timed where it completes, on the debug-only
        # SlabIO.close drain line.  The one-switch/rank-0 policy lives at
        # :func:`_restart_write_log_on` / :func:`_log_restart_write`.

        def _write(name, arr):
            """create+write one dataset, μ axes clipped to ``n_logical``
            (default: the charge ``n_rmu_logical``), with the AF.4c
            debug telemetry line.  Single write path for every dataset in
            this file, including the transverse ψ and the real W0
            (audit 2026-07-28 — the transverse block used to be an
            inline copy of this helper)."""
            if arr is None:
                return
            shape, attrs = write_plan[name]
            _t0 = time.time()
            # The LOGICAL shape is stated once, to create_dataset; the
            # write clips ``arr``'s μ pad rows against it on its own
            # (decisions.md 2026-08-04).
            io.create_dataset(name, shape=shape, dtype=arr.dtype, attrs=attrs)
            io.write_slab(name, arr)
            _log_restart_write(name, shape, arr.dtype, time.time() - _t0)

        _write("V_qmunu", V_on_disk)
        _write("S_qmunu", S_qmunu)
        _write("V0_noG0_munu", V0_noG0_munu)
        _write("G0_mu_nu", G0_mu_nu)
        _write("psi_full_y", psi_full_y)
        # (nk, s, μ, n): μ is axis -2, not -1 — the mun face's axis order
        # differs from every other dataset this writer knows about.
        _write("psi_full_y_mun", psi_full_y_mun)
        _write("enk_full", enk_full)

        # Bispinor per-channel ψ: μ axis clipped to the TRANSVERSE
        # logical extent (its own centroid count, not n_rmu_logical).
        if psi_full_y_transverse is not None:
            _write("psi_full_y_transverse", psi_full_y_transverse)
            # Face layout (low_mem_bands): the ADDITIVE second face of
            # the transverse carrier, mirroring psi_full_y_mun exactly
            # ((nk, s, μ_T, n) — μ at axis -2).  Written only when the
            # caller holds a face-layout transverse bundle; a legacy
            # caller passes None and the file stays byte-identical to
            # the pre-face schema.
            _write("psi_full_y_transverse_mun", psi_full_y_transverse_mun)
            # Stamped for the load-time extent cross-check in
            # read_restart_state_from_h5.
            io.write_attr("n_rmu_transverse_logical", np.int64(n_T))

        # W0_qmunu: either write the real data or pre-allocate an
        # all-zeros placeholder.
        w0_touched = W0_qmunu is not None or init_W0
        w0_ready = False
        if W0_qmunu is not None:
            _write("W0_qmunu", W0_qmunu)
            w0_ready = True
        elif init_W0:
            # THE PLACEHOLDER'S SHAPE IS V'S SHAPE.  W0 is allocated here
            # from ``V_qmunu.shape``, so V's storage decision silently
            # becomes W0's — including its q extent.  That coupling is
            # fine and deliberate while both tensors are full-BZ, and it
            # is a TRAP the moment they need not be: a run that stored V
            # on the q wedge and W0 on the full BZ (or the reverse) would
            # get a placeholder of the wrong length here, and
            # ``write_w0_qmunu_to_h5`` re-creates the dataset later, so
            # the mismatch would surface as a shape error deep in the W
            # write rather than as a decision anyone took.
            # THE RULE: V and W0 resolve their q storage ONCE, together.
            # Whoever teaches this writer about wedge storage must pass
            # the resolved mode in rather than let it be inherited from
            # an argument's shape.
            # ``V_on_disk``, NOT ``V_qmunu``: when the resolution says
            # wedge the two differ on the q axis, and taking the
            # placeholder from the in-memory full-BZ tensor is precisely
            # the inheritance the rule above forbids.  ONE resolution
            # decided both, which is what dbe3b4ec asked for.
            v_shape, v_attrs = write_plan["W0_qmunu"]
            v_dtype = V_on_disk.dtype
            _t0 = time.time()
            io.create_dataset("W0_qmunu", shape=v_shape, dtype=v_dtype,
                              attrs=v_attrs)
            if _restart_write_log_on():
                # Allocation ONLY -- no data is written here, so this
                # deliberately does NOT use the _log_restart_write
                # completed-write format.  Naming that explicitly
                # matters: under parallel HDF5 this call makes the file
                # jump by the full tensor size, which reads exactly like
                # progress and is not (AC.3b).
                _nb = int(np.prod(v_shape)) * int(np.dtype(v_dtype).itemsize)
                print(f"  [restart_write] W0_qmunu placeholder ALLOCATED "
                      f"{tuple(int(v) for v in v_shape)} {_nb / 1e9:.2f} GB "
                      f"in {time.time() - _t0:.1f} s (no data written)",
                      flush=True)

    # bse_io.py reads W0_ready as an HDF5 attr on the W0_qmunu dataset.
    # Set it rank-0-only after SlabIO has released the file, to stay
    # compatible with that reader.
    #
    # ``V_ready`` IS THE SAME PROMISE FOR V, AND IT IS NEW.  W0 has carried
    # a persisted flag since the April all-zero-screening incident, and
    # every W0 consumer gates on it; V_qmunu carried nothing, and
    # ``bse_io._load_ring_subset`` read it unconditionally on the same line
    # that gated W0.  Today that asymmetry is harmless — V is never
    # allocated as a placeholder, so present implies written — but "the
    # invariant happens to hold" and "the file says so" are different
    # states, and only the second survives a writer that grows a
    # placeholder path.  Stamped True here because reaching this line means
    # the data went in; readers treat ABSENT as True so every restart file
    # written before this attr existed keeps loading byte-for-byte.
    v_touched = V_qmunu is not None
    def _publish_readiness():
        if not (w0_touched or v_touched):
            return
        with h5py.File(filename, "a") as f:
            set_commit_state(f, False)
            if qirr is not None and qirr.store_wedge:
                _stamp_qirr(f, qirr, n_rmu_logical,
                            v_touched=v_touched,
                            w0_placeholder=(w0_touched and not w0_ready),
                            w0_data=(w0_touched and w0_ready))
            if w0_touched:
                f["W0_qmunu"].attrs["W0_ready"] = w0_ready
            if v_touched:
                f["V_qmunu"].attrs["V_ready"] = True
            set_commit_state(f, True)
    rank0_transaction(filename, stage="restart.readiness", write=_publish_readiness)



def _stamp_qirr(f, qirr, n_rmu_logical, *, v_touched, w0_placeholder,
                w0_data):
    """Stamp the q_irr tables/attrs onto datasets SlabIO has already written.

    Rank-0 only, called from inside the one h5py block that owns the
    persisted flags.  Never raises past the caller with a half-stamped
    file: :func:`symmetry_maps.stamp_qirr_tensor` writes the table group
    and the version attr together, and the reader's partial-stamp refusal
    is what catches an interrupted write — a file with tables and no
    version is refused rather than read as legacy.
    """
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import stamp_qirr_tensor
    from gw.restart_q_storage import assert_capture_matches

    # THE CROSS-CHECK, BEFORE ANY ATTR IS WRITTEN.  The capture came out of
    # the compute path and the resolution was taken again at the writer; if
    # they describe different centroid sets the file would carry tables that
    # do not reconstruct its own tensor, silently.
    assert_capture_matches(qirr.capture, qirr.resolution,
                           context="write_restart_state_to_h5")
    tables = qirr.capture.tables()
    verdict = qirr.resolution.verdict
    n_log = int(qirr.capture.n_rmu_logical)
    if n_log != int(n_rmu_logical):
        raise ValueError(
            f"write_restart_state_to_h5: the captured wedge declares "
            f"n_rmu_logical={n_log} but the writer was told "
            f"{int(n_rmu_logical)}.  The tables would be stripped to one "
            f"extent and the tensor clipped to the other, and the file "
            f"would describe two different centroid sets.")
    if v_touched:
        stamp_qirr_tensor(f, "V_qmunu", tables=tables,
                          closure_verdict=verdict, n_rmu_logical=n_log,
                          data_ready=True)
    if w0_data or w0_placeholder:
        stamp_qirr_tensor(f, "W0_qmunu", tables=tables,
                          closure_verdict=verdict, n_rmu_logical=n_log,
                          data_ready=bool(w0_data))


def write_w0_qmunu_to_h5(
    filename, W0_qmunu, *, n_rmu_logical: int, mesh=None, qirr=None,
):
    """Overwrite or append the W0_qmunu dataset in an existing restart file.

    ``n_rmu_logical`` clips the trailing (μ, μ) axes to the logical
    on-disk extent — same contract as ``write_restart_state_to_h5``.

    ``qirr`` IS THE SAME OBJECT V RESOLVED WITH, carrying W's OWN capture.
    This writer re-creates the dataset the placeholder allocated, so it is
    the second half of the coupling: a W0 written on the full BZ into a file
    whose V is a wedge (or the reverse) would be a file no reader can make
    sense of, and the two are kept together by passing one decision to both
    rather than by hoping.  ``None`` is today's behaviour exactly.
    """
    from .slab_io import SlabIO

    if qirr is not None and qirr.store_wedge:
        if qirr.capture is None:
            raise ValueError(
                "write_w0_qmunu_to_h5: restart_q_storage resolved to 'ibz' "
                "but no pre-unfold W capture reached the writer.  W's wedge "
                "is the array the Dyson solve produced, one statement before "
                "screening unfolds it; slicing the unfolded W instead would "
                "make the stored block depend on an op-selection policy "
                "nobody froze for this purpose.")
        # PRE-FLIGHT BEFORE MUTATION.  ``_stamp_qirr`` repeats these checks at
        # the metadata seam, but waiting until then would recreate/overwrite
        # W0 first and only afterwards discover that its tables describe a
        # different centroid set or logical extent.
        from gw.restart_q_storage import assert_capture_matches
        assert_capture_matches(
            qirr.capture, qirr.resolution,
            context="write_w0_qmunu_to_h5 preflight")
        capture_n_rmu = int(qirr.capture.n_rmu_logical)
        if capture_n_rmu != int(n_rmu_logical):
            raise ValueError(
                "write_w0_qmunu_to_h5 preflight: captured wedge declares "
                f"n_rmu_logical={capture_n_rmu}, writer was told "
                f"{int(n_rmu_logical)}; refusing before W0 mutation.")
        W0_qmunu = qirr.capture.X_ibz

    # Preflight before SlabIO opens the existing file: a short producer must
    # not recreate/mutate W0 and only then report its bad geometry.
    shape = _mu_logical_shape(
        W0_qmunu.shape, (-2, -1), n_rmu_logical,
        where="write_w0_qmunu_to_h5 dataset 'W0_qmunu'")
    shape_attrs = _shape_receipt_attrs(W0_qmunu.shape, shape)
    with SlabIO(filename, mode="a", mesh=mesh,
) as io:
        _t0 = time.time()
        io.create_dataset("W0_qmunu", shape=shape, dtype=W0_qmunu.dtype,
                          attrs=shape_attrs)
        io.write_slab("W0_qmunu", W0_qmunu)
        # Same instrument as ``write_restart_state_to_h5`` (AF.4c).  This
        # is the SECOND (nq, mu, mu) tensor the run writes -- another
        # 13.34 GB at c2406 -- and it had no telemetry at all, so a repeat
        # of the writer pathology would have been invisible here even
        # after AF instrumented its sibling.
        _log_restart_write("W0_qmunu", shape, W0_qmunu.dtype,
                           time.time() - _t0)

    # W0_ready flag is a per-dataset attr read by bse_io.py.  The q_irr
    # stamp rides in the same rank-0 block, for the same reason as in
    # ``write_restart_state_to_h5``: SlabIO has released the file and no
    # other writer may open it between these two statements.
    def _publish_w0():
        with h5py.File(filename, "a") as f:
            set_commit_state(f, False)
            if qirr is not None and qirr.store_wedge:
                _stamp_qirr(f, qirr, n_rmu_logical, v_touched=False,
                            w0_placeholder=False, w0_data=True)
            f["W0_qmunu"].attrs["W0_ready"] = True
            set_commit_state(f, True)
    rank0_transaction(filename, stage="restart.W0_readiness", write=_publish_w0)



def write_head_scalars_to_h5(
    filename: str,
    *,
    vhead: complex | None = None,
    whead: np.ndarray | jnp.ndarray | None = None,
    omega_grid: np.ndarray | jnp.ndarray | None = None,
    S_cart: np.ndarray | jnp.ndarray | None = None,
    head_correction: str | None = None,
    response_kind: str | None = None,
    head_source: str | None = None,
):
    """Persist q=0 Coulomb head scalars to the restart file.

    Stored alongside ``G0_mu_nu``; consumed by ``bse_io._load_ring_subset``
    (and any future Σ-builder) via ``head_correction.apply_q0_head_rank1``.

    - ``vhead``: scalar v(q→0, G=G'=0) in Ry, BGW convention.
    - ``whead``: shape ``(n_omega,)``. Length 1 for static COHSEX,
      length 2 for GN-PPM (static, iω_p).
    - ``omega_grid``: optional ``(n_omega,)`` array of the ω values
      (in Ry) corresponding to ``whead`` — written as an attribute on
      the ``whead`` dataset for consumer interpretation.
    - ``S_cart``: optional ``(3, 3)`` complex — the Cartesian q²-coefficient
      tensor that PRODUCED ``whead[0]``, in the canonical convention of
      ``docs/theory/s-tensor-convention.md``.  ``None`` on the ``epshead``
      head branch, which fits an isotropic γ and has no tensor.
    - ``head_correction``, ``response_kind``, and ``head_source``: optional
      provenance attrs on ``whead``.  Together they distinguish a direct
      epsilon head from a once-folded or already micro-reducible W head, so a
      restart consumer cannot safely infer reduction state from array shape.

      WHY A TENSOR JOINS TWO SCALARS HERE.  ``whead`` is the cell average
      ``⟨v/(1 − v qᵀSq)⟩`` over ONE mini-BZ, so it is bound to the grid it was
      computed on.  Any consumer that changes the grid — and the BSE's
      coarse→fine W densifier does exactly that — needs the INTEGRAND, not the
      average, and rebuilding it means re-reading ``dipole.h5`` and redoing
      the ``S(ω)`` sum.  Nine numbers on a multi-GB restart make that
      unnecessary and, more importantly, make the re-attached head provably
      the same screening the run solved with rather than a re-derivation that
      merely ought to agree.  Absent on restarts written before this existed;
      ``head_correction.resolve_head_S_cart`` falls back to the rebuild.

    Rank-0-only write (these are tiny; no MPI-IO needed).
    """
    # JAX metadata conversion is replicated, before the serial writer.
    if vhead is not None:
        vhead = np.complex128(vhead)
    if whead is not None:
        whead = np.asarray(whead, dtype=np.complex128).reshape(-1)
    if omega_grid is not None:
        omega_grid = np.asarray(omega_grid, dtype=np.float64).reshape(-1)
    if S_cart is not None:
        S_cart = np.asarray(S_cart, dtype=np.complex128).reshape(3, 3)

    def _write_heads():
        with h5py.File(filename, "a") as f:
            set_commit_state(f, False)
            if vhead is not None:
                if "vhead" in f:
                    del f["vhead"]
                f.create_dataset("vhead", data=np.complex128(vhead))
            if whead is not None:
                if "whead" in f:
                    del f["whead"]
                arr = np.asarray(whead, dtype=np.complex128).reshape(-1)
                ds = f.create_dataset("whead", data=arr)
                if omega_grid is not None:
                    ds.attrs["omega_grid"] = np.asarray(omega_grid, dtype=np.float64).reshape(-1)
                if head_correction is not None:
                    ds.attrs["head_correction"] = str(head_correction)
                if response_kind is not None:
                    ds.attrs["response_kind"] = str(response_kind)
                if head_source is not None:
                    ds.attrs["head_source"] = str(head_source)
            if S_cart is not None:
                if "S_cart_head" in f:
                    del f["S_cart_head"]
                S = np.asarray(S_cart, dtype=np.complex128).reshape(3, 3)
                sd = f.create_dataset("S_cart_head", data=S)
                sd.attrs["convention"] = "cartesian_q2_coefficient"
                sd.attrs["omega_ry"] = 0.0
            set_commit_state(f, True)
    rank0_transaction(filename, stage="restart.head_scalars", write=_write_heads)


def assert_restart_window_matches(filename, band_slices=None,
                                 n_rmu_logical=None) -> None:
    """Refuse a restart whose tensors were built under a DIFFERENT band
    window or centroid count.

    ``V_qmunu``, ``psi_full_y`` and ``enk_full`` are all indexed by the band
    window in force when they were written.  Reusing them under a changed
    ``nval``/``ncond``/``nband`` misindexes Sigma with no shape error and no
    crash — job 7874375 reused window-70 tensors at window 80 and produced a
    QP gap of -135 eV while every stage reported success.  This turns that
    into a loud, actionable failure naming BOTH windows.

    Files written before this attr existed carry no ``band_window``; those
    are passed through with no check (back-compat), since refusing them would
    strand existing restart files.
    """
    with h5py.File(filename, "r") as f:
        from .commit_state import assert_committed
        assert_committed(f, path=filename)
        stored_w = np.asarray(f["band_window"]).tolist() if "band_window" in f else None
        stored_split = (np.asarray(f["band_window_split"]).tolist()
                        if "band_window_split" in f else None)
        stored_schema = (
            int(np.asarray(f[BAND_WINDOW_SCHEMA_DATASET])[()])
            if BAND_WINDOW_SCHEMA_DATASET in f else None)
        stored_carrier = (
            np.asarray(f[BAND_WINDOW_CARRIER_DATASET]).tolist()
            if BAND_WINDOW_CARRIER_DATASET in f else None)
        stored_mu = (int(np.asarray(f["n_rmu_logical"])[()])
                     if "n_rmu_logical" in f else None)

    if stored_schema is not None:
        if stored_schema != BAND_WINDOW_SCHEMA_VERSION:
            raise ValueError(
                f"Restart file {filename} has unsupported band-window schema "
                f"{stored_schema}; this reader supports "
                f"{BAND_WINDOW_SCHEMA_VERSION}.")
        if stored_w is None or stored_carrier is None or stored_split is None:
            raise ValueError(
                f"Restart file {filename} declares band-window schema "
                f"{stored_schema} but is missing band_window, "
                f"band_window_split, or {BAND_WINDOW_CARRIER_DATASET}. The "
                "geometry receipt is torn.")
        if len(stored_w) != 5 or len(stored_carrier) != 5:
            raise ValueError(
                f"Restart file {filename} has malformed band-window receipts: "
                f"logical={stored_w}, carrier={stored_carrier}; expected two "
                "five-edge windows.")
        logical = tuple(int(v) for v in stored_w)
        carrier = tuple(int(v) for v in stored_carrier)
        if logical[:4] != carrier[:4] or carrier[4] < logical[4]:
            raise ValueError(
                f"Restart file {filename} has inconsistent logical/carrier "
                f"band receipts: logical={logical}, carrier={carrier}. The "
                "carrier may add only a zero-pad tail above the logical b4.")

    # THE χ COUNT MUST MATCH; THE Σ COUNT NEED NOT, and the asymmetry is the
    # point.  Every tensor in this file is a function of the SCREENING side or
    # of the loaded extent: ``V_qmunu`` / ``W0_qmunu`` are built from the χ0
    # band sum, ``psi_full_y`` / ``enk_full`` / ζ span [b0, b4) =
    # max(chi, sigma) — which ``band_window``'s b4 already pins.  NOTHING on
    # disk is a function of ``number_bands_sigma``: Σ slices [0, b4_sigma) out
    # of tensors that already exist.
    #
    # So a Σ-count sweep at fixed χ reuses this file legitimately — which is
    # the case the split exists to make cheap (χ at full bands, Σ short and
    # extrapolated), and it is why this is a targeted check rather than a
    # blanket "no restart under a split".  Changing χ is refused, for exactly
    # the reason the 5-tuple check above exists.
    if band_slices is not None:
        want_b4_logical = int(
            getattr(band_slices, "b4_logical", 0) or band_slices.b4)
        want_split = (
            min(int(band_slices.b4_chi), want_b4_logical),
            min(int(band_slices.b4_sigma), want_b4_logical),
        )
        # Legacy split stamps carried the padded larger edge.  Keep their
        # historical comparison exact: without schema 2 the file cannot prove
        # whether rows between logical and carrier b4 were zeros or physical.
        if stored_schema is None:
            want_split = (int(band_slices.b4_chi),
                          int(band_slices.b4_sigma))
        if stored_split is None and stored_w is not None:
            stored_split = [int(stored_w[4]), int(stored_w[4])]
        if stored_split is not None and int(stored_split[0]) != want_split[0]:
            raise ValueError(
                f"Restart file {filename} was written with a chi0/W band sum "
                f"topping out at band {int(stored_split[0])}, but this run "
                f"has number_bands_chi -> band {want_split[0]}.  V_qmunu and "
                f"W0_qmunu ARE the screening, so reusing them would run this "
                f"deck's Sigma against the OTHER deck's W and report rc=0 "
                f"(the same silent-misindex class as the band-window check "
                f"below; see job 7874375).  Either restore the original "
                f"number_bands_chi, or set restart=false.  Note that "
                f"number_bands_SIGMA may be changed freely on a restart: no "
                f"tensor in this file depends on it.")

    if stored_w is not None and band_slices is not None:
        want_b4 = int(band_slices.b4)
        if stored_schema == BAND_WINDOW_SCHEMA_VERSION:
            want_b4 = int(
                getattr(band_slices, "b4_logical", 0) or band_slices.b4)
        want = [int(band_slices.b0), int(band_slices.b1), int(band_slices.b2),
                int(band_slices.b3), want_b4]
        stored_stable = [int(stored_w[index]) for index in (0, 1, 2, 4)]
        want_stable = [want[index] for index in (0, 1, 2, 4)]
        if stored_stable != want_stable:
            raise ValueError(
                f"Restart file {filename} was written under stable restart "
                f"window (b0,b1,b2,b4)={tuple(stored_stable)} but this run "
                f"has {tuple(want_stable)}. V_qmunu / psi_full_y / enk_full "
                f"are indexed by that window, so reusing them would MISINDEX "
                f"Sigma silently (no crash, wrong QP energies -- see job "
                f"7874375). Either restore the original nval and loaded/chi "
                f"band extent, or set restart=false to rebuild the tensors. "
                f"The Sigma-only b3 edge may change because no restart "
                f"tensor depends on number_bands_sigma."
            )
    if stored_mu is not None and n_rmu_logical is not None:
        if int(stored_mu) != int(n_rmu_logical):
            raise ValueError(
                f"Restart file {filename} was written with n_rmu={stored_mu} "
                f"but this run has n_rmu={int(n_rmu_logical)}.  The ISDF basis "
                f"differs, so V_qmunu / psi_full_y are not reusable.  Set "
                f"restart=false (or point at the matching centroid file)."
            )


def _munu_slab_request(ds_shape, n_rmu_pad):
    """``(offset, shape, spec)`` for a (…, μ, ν) restart tensor.

    The μ/ν axes are ALWAYS the trailing two — that is the disk contract
    for ``V_qmunu`` / ``S_qmunu`` / ``V0_noG0_munu`` / ``W0_qmunu``, and
    it holds across all three historical layouts because the leading axes
    only ever grew in front:

      * 3-D flat-q ``(nq, μ, ν)``            — what the current pipeline writes
      * 6-D transitional ``(1, npol, npol, nq, μ, ν)``
      * 8-D legacy ``(1, npol, npol, nkx, nky, nkz, μ, ν)``

    So one rule covers all three: shard the last two axes on ('x', 'y'),
    replicate everything in front, and read the leading ``(1, npol, npol)``
    block at index 0 with extent 1 — which is exactly the ``[0, 0, 0]``
    the whole-file reader used to take AFTER materialising the array.

    ``bse_io._resolve_munu_reader`` / ``_MunuSlabPlan`` state the same
    three layouts for the BSE consumer, which additionally needs to
    select a single q; this one never does, so it does not need the
    kgrid.  If a third consumer appears, the layout fact should move here
    (L3) rather than be stated a third time.
    """
    ndim = len(ds_shape)
    if ndim < 2:
        raise ValueError(
            f"restart (μ, ν) tensor has rank {ndim} (shape "
            f"{tuple(ds_shape)}); it must have at least the two trailing "
            f"μ/ν axes.")
    # ONLY the two legacy layouts carry the ``(1, npol, npol)`` prefix that
    # has to be read at index 0.  Everything else -- 2-D ``V0_noG0_munu``,
    # 3-D flat-q ``V_qmunu``, 5-D ``S_qmunu`` -- keeps every leading axis
    # whole, so the rank does not need enumerating and a new leading axis
    # does not need a code change.  (Enumerating it DID cost a bug: the
    # first version of this listed 3/5/6/8 and refused the 2-D
    # ``V0_noG0_munu``, caught by restart_sharded_parity at P=4.)
    lead = 3 if ndim in (6, 8) else 0
    offset = [0] * ndim
    shape = [1] * lead + [int(v) for v in ds_shape[lead:-2]]
    shape += [int(n_rmu_pad), int(n_rmu_pad)]
    spec = P(*([None] * (ndim - 2) + ["x", "y"]))
    return tuple(offset), tuple(shape), spec


def _collapse_leading(A, ds_shape, mesh_xy):
    """Drop the legacy ``(1, npol, npol)`` prefix, flattening q if present.

    Local: the leading axes are extent-1 or replicated and μ/ν keep their
    ('x', 'y') sharding across the reshape, so GSPMD moves nothing.  The
    output sharding is PINNED rather than inferred — an inferred
    resharding of an N_mu²-class object would be a silent all-to-all,
    which is the one thing this reader must never do.  Same reasoning,
    and the same spelling, as ``bse_io._slabio_read_munu``.

    Only the 6-D/8-D legacy layouts reach the reshape at all; the 3-D
    flat-q form the current pipeline writes passes straight through.
    """
    ndim = len(ds_shape)
    if ndim not in (6, 8):            # 3-D flat-q, or S_qmunu's own 5-D
        return A
    mu, nu = int(A.shape[-2]), int(A.shape[-1])
    return jax.jit(
        lambda a: jnp.reshape(a, (-1, mu, nu)),
        out_shardings=NamedSharding(mesh_xy, P(None, "x", "y")))(A)


def _check_nspinor(nspinor: int, where: str) -> int:
    """Gate the ψ spinor axis.  1 (scalar), 2 (spinor), 4 (bispinor).

    THE GATE THE PAD AUDIT ASKED FOR.  The μ axis of every restart tensor
    is padded to a mesh-divisible extent and the pad rows are exact zeros
    by construction; the SPINOR axis is not padded by anything here and
    must not be.  It is read at its on-disk extent and carried through
    replicated, so a 2-component spinor restart and a 4-component
    bispinor restart differ only in that extent.

    The failure this refuses is a file whose spinor axis is neither —
    which, unchecked, would sail through as a perfectly shardable
    replicated axis and misindex every downstream ψ contraction with no
    shape error.  ``nspinor`` is small and replicated, so there is no
    cost to checking it.
    """
    ns = int(nspinor)
    if ns not in (1, 2, 4):
        raise ValueError(
            f"{where}: ψ spinor axis has extent {ns}; expected 1 (scalar), "
            f"2 (spinor) or 4 (bispinor).  The restart file is not one this "
            f"pipeline wrote — regenerate it (restart = false).")
    return ns


def _qirr_wedge_tables(f):
    """The unfold tables for each restart tensor stored on the IBZ q WEDGE.

    Returns ``{dataset_name: QirrTables}``, empty for every restart file that
    stores its tensors on the full BZ — which is every file written before
    the q_irr format existed and every file a full-BZ run writes today.  An
    empty answer means the reader below does nothing at all, so the byte path
    for those files is the one they have always had.

    THIS REPLACED A REFUSAL, and the refusal is worth remembering because it
    is what the unfold has to be better than.  Until 2026-08-08 this reader
    did not ask the question; a wedge file came back with a q axis of
    ``n_q_ibz``, flowed through ``gw_init``'s restart branch untouched, and
    met a ``W_q`` that screening HAD unfolded at ``gw/cohsex_sigma.py``'s
    ``W_q - V_q``, dying as ``TypeError: sub got incompatible shapes for
    broadcasting: (9, 399, 399), (5, 399, 399)``.  ``50db6299`` turned that
    into a named refusal, deliberately NOT an unfold, because the sharded
    reader was believed unable to redistribute.  Per the owner's ruling of
    2026-08-08 ~13:20 the readers ALWAYS unfold, and the belief was measured
    and found wrong — see :func:`_unfold_wedge`.

    ASKED IN PASS 1, WHILE THE SERIAL HANDLE IS OPEN, and it costs kilobytes:
    the tables are a permutation and a wrap table, not a tensor.  Doing it
    here rather than inside the SlabIO block keeps the two handles from
    overlapping, which is the same rule the geometry read above follows.

    THE PROBE IS ``symmetry_maps.dataset_q_storage`` — the same one
    ``bse_io.is_q_wedge`` wraps, called directly here because ``file_io``
    must not import ``bse`` (that is uphill, and the layering ratchet says
    so).  Two callers, in two layers, and
    ``test_restart_qirr_consumers.py::test_the_probe_has_exactly_two_named_callers``
    names both rather than letting a third appear unnoticed.
    """
    from symmetry_maps import read_tables, dataset_q_storage
    return {name: read_tables(f, name)
            for name in ("V_qmunu", "S_qmunu", "V0_noG0_munu", "W0_qmunu")
            if name in f and dataset_q_storage(f[name]) == "ibz"}


def _unfold_wedge(A, tables, n_rmu_pad, mesh_xy):
    """IBZ wedge -> full BZ, with the tables the FILE itself carries.

    ``tables is None`` (the full-BZ and legacy case) returns ``A`` untouched,
    so this is a no-op on every restart file that is not a wedge.

    THE SAME CALL THE PRODUCER MADE, ON THE SAME TABLES.  The format stores
    the PRE-UNFOLD block, so ``unfold(stored)`` is the identity — the same
    function on the same inputs the producing run itself used — rather than a
    property that depends on the bit-frozen op-selection policy.  That is why
    the tables come out of the file rather than being re-derived from this
    run's ``sym``: a table that reconstructs the tensor must be the table
    that deconstructed it.

    A SHARDED UNFOLD IS AVAILABLE, and this is the point the tree used to say
    the other way.  ``bse_io._MunuSlabPlan``'s refusal argues that a per-rank
    (μ, ν) hyperslab cannot unfold because the unfold gathers across the very
    axes it shards on.  The premise is true of SlabIO and the conclusion does
    not follow: ``unfold_isdf_operator`` is a ``shard_map`` over four
    ``lax.all_to_all`` collectives that redistribute those axes
    volume-preservingly, never exceeding one tile per rank, and it takes and
    returns ``P(None,'x','y')`` — exactly the spec ``_munu_slab_request``
    produces.  The producer runs it on the real distributed mesh twice per
    run.  MEASURED bit-identical against the single-device unfold at 2x2, 4x1
    and 1x4 (DESIGN_restart_consolidation.md §1, element-wise on the
    off-diagonals).  What SlabIO cannot do is unfold as a hyperslab OFFSET,
    and nothing here asks it to: it reads the wedge exactly as it reads a
    full-BZ tensor, and the collective happens afterwards, in jax.

    THE PAD IS THIS READER'S, NEVER THE WRITER'S.  The file stores the
    LOGICAL μ extent (SHARDING_RULES §2), so the tables are re-padded against
    THIS process's device count with ``QirrTables.padded`` — identity tail on
    the permutation, zero tail on the wrap, the same pure function the writer
    inverted.  That is what makes a file written on four ranks read on eight,
    and it is why the tables move to the tensor's extent rather than the
    tensor being clipped to theirs.  ``unfold_isdf_operator`` additionally
    REFUSES a μ extent not divisible by Px·Py; ``n_rmu_pad`` came from
    ``padded_mu_extent`` two frames up and already satisfies it, which is
    stated here so nobody later optimises the pad away.
    """
    if tables is None:
        return A
    from symmetry_maps import unfold_isdf_operator
    t = tables.padded(int(n_rmu_pad))
    return unfold_isdf_operator(
        A, irr_idx=t.irr_idx_q, sym_idx=t.sym_idx_q, sym_perm=t.sym_perm,
        L_table=t.L_table, q_irr_frac=t.q_irr_frac, mesh_xy=mesh_xy,
        n_sym_spatial=int(t.n_sym_spatial))


def read_restart_state_from_h5(filename, mesh_xy, *, low_mem_bands=False,
                               n_band_carrier=None):
    """Read canonical restart state as PER-RANK TILES (restart format v2).

    Returns arrays that are ALREADY sharded on ``mesh_xy`` and ALREADY at
    the padded μ extent.  Nothing larger than one rank's tile is
    materialised at any point, so this works at any process count.

    WHAT THIS REPLACED, AND WHY IT MATTERS
    --------------------------------------
    Until 2026-08-06 this was a full-file reader — every dataset read with
    ``[:]`` into one array on the calling process, ``V_qmunu`` and
    ``S_qmunu`` at ``(nq, μ, μ)`` and ``V0_noG0_munu`` at ``(μ, μ)``, all
    N_mu²-class, whole, on every rank — and it was GUARDED OFF above one
    process with no replacement, which removed a capability that had
    worked at deck scale.  The guard was correct about the cost and wrong
    to be the end of the story: MEASURED at P=4 (job 56389339, MoS2 6x6,
    N_mu=1496, nq=36) the old path cost **+1.53 GiB of VmHWM on every
    rank** (0.95 → 2.47 GiB) and nothing warned; at the design envelope
    (N_mu=20000, nq=64) the same read is **381.47 GiB per rank**
    (CLAIMS 69).

    The port follows ``bse_io.load_bse_data_from_restart_sharded``: read
    the SHAPES and the small replicated metadata with serial h5py, close
    that handle, then move the big tensors through SlabIO under
    collective MPI-IO.  Two live handles on one file is a hazard nobody
    needs, which is why the h5py block closes before SlabIO opens.

    WHAT STAYS ON SERIAL h5py, AND WHY IT IS NOT A LOOPHOLE
    -------------------------------------------------------
    ``enk_full`` ``(nk, nb)``, ``G0_mu_nu`` ``(μ,)``, and the scalar
    stamps.  These are μ-class or smaller — ``G0`` is 320 KB at the
    envelope — and every rank needs all of them.  The doctrine forbids
    materialising an N_mu²-class object, not reading a vector; sharding
    ``G0`` would buy nothing and cost a reshard at every use, and
    ``bse_io`` reads it exactly the same way.

    PADDING.  Disk stores the LOGICAL μ extent so a restart written at
    any device count re-reads at any other (SHARDING_RULES §2).  The
    in-memory convention is ``padded_mu_extent(n_rmu, device_count)``,
    and the pad rows are exact zeros.  Both facts are now enforced by the
    same mechanism: SlabIO is asked for the PADDED shape and zero-fills
    everything past the dataset, so the pad is the read, and there is no
    ``jnp.pad`` and no ``with_sharding_constraint`` applied to an
    already-resident global array.

    THE q WEDGE IS UNFOLDED, ALWAYS, AND THE CALLER NEVER LEARNS OF IT.
    A restart file whose V/W sit on the IBZ q wedge comes back on the FULL
    BZ, at the same shape and the same sharding as a full-BZ file's, so
    ``gw_init``'s restart branch reads ``rs.V_qmunu`` without asking which
    q-set the bytes were on.  That is the owner's ruling of 2026-08-08
    ~13:20 — storage follows the WFN's own symmetry and readers always
    unfold — and it is what retires this reader's former refusal.  The
    tables come out of the FILE (:func:`_qirr_wedge_tables`) and the unfold
    is the producer's own call (:func:`_unfold_wedge`), so what comes back
    is the same function of the same inputs the producing run used.  On
    every full-BZ and legacy file nothing happens at all: the probe returns
    an empty map and the read is the byte path it has always been.

    SPINOR AND BISPINOR.  ``nspinor`` is read from the ψ dataset and
    carried through as a replicated axis at its on-disk extent — 2 for a
    spinor restart, 4 for a bispinor one — and gated by
    :func:`_check_nspinor`.  It is never padded.  The bispinor
    ``psi_full_y_transverse`` is read at its OWN μ extent (the transverse
    centroid count differs from the charge one) with its own pad.

    ``low_mem_bands=True`` reads the TWO 2-D-sharded faces
    (``gw.wavefunction_bundle`` ``psi_nmu``/``psi_mun``) instead of the
    legacy single-axis ``psi_full_y``: ``psi_nmu`` is a direct hyperslab
    of the SAME "psi_full_y" dataset at the face partition spec (its axis
    order (nk, n, s, μ) already matches, so this is not a different
    dataset, only a different sharding of the same one); ``psi_mun`` is a
    direct hyperslab of the ADDITIVE "psi_full_y_mun" dataset (axis order
    (nk, s, μ, n)).  Neither derivation performs a reshard: each face is
    exactly what its own hyperslab holds.  This is the "request both face
    specs from SlabIO" branch of the restart audit (report §"Restart
    write/read") rather than a one-face-plus-transpose branch — chosen
    because it needs no new cross-mesh-axis collective to write or
    verify, at the cost of the doubled on-disk ψ bytes.  Bispinor under
    ``low_mem_bands`` (2026-08-23): the transverse pair rides the
    identical two-hyperslab pattern — ``psi_full_y_transverse`` at the
    nmu face spec plus the ADDITIVE ``psi_full_y_transverse_mun`` — at
    the transverse mu extent; a file carrying only the legacy-written
    nmu-order dataset refuses by name.
    """
    from .slab_io import SlabIO
    from runtime.padding import padded_mu_extent
    from common.collectives import device_put_process_local

    divisor = int(jax.device_count())

    # ---- pass 1: geometry + the small replicated arrays, serial h5py ----
    with h5py.File(filename, "r") as f:
        if "psi_full_y" not in f:
            raise ValueError(
                f"Restart file {filename} is missing canonical psi_full_y "
                "dataset. Regenerate restart tensors with current gw_jax.")
        # THE UNFOLD TABLES, while this handle is open and before any tensor
        # bytes move.  Empty on every full-BZ and legacy file, which is what
        # keeps those reads on the byte path they have always had.
        wedge_tables = _qirr_wedge_tables(f)
        shapes = {k: tuple(int(s) for s in f[k].shape)
                  for k in ("V_qmunu", "S_qmunu", "V0_noG0_munu",
                            "psi_full_y", "psi_full_y_mun",
                            "psi_full_y_transverse",
                            "psi_full_y_transverse_mun")
                  if k in f}
        dtypes = {k: f[k].dtype for k in shapes}
        for name in shapes:
            _validate_shape_receipt(name, f[name])
        if (low_mem_bands and "psi_full_y_transverse" in shapes
                and "psi_full_y_transverse_mun" not in shapes):
            raise ValueError(
                f"Restart file {filename} has 'psi_full_y_transverse' but "
                f"no 'psi_full_y_transverse_mun' dataset: it was written "
                f"by a legacy-layout run.  Read it with low_mem_bands = "
                f"false, or rerun with restart = false so the transverse "
                f"face pair is written.")
        if low_mem_bands and "psi_full_y_mun" not in shapes:
            raise ValueError(
                f"Restart file {filename} has no 'psi_full_y_mun' dataset "
                "but low_mem_bands = true was requested.  Either this file "
                "predates the two-face restart format (regenerate with "
                "restart = false, low_mem_bands = true), or it was written "
                "with low_mem_bands = false — restart under a DIFFERENT "
                "low_mem_bands than the write is not supported.")
        enk_full = (np.asarray(f["enk_full"][:]) if "enk_full" in f else None)
        G0_mu_nu = (np.asarray(f["G0_mu_nu"][:]) if "G0_mu_nu" in f else None)
        stored_T = (int(np.asarray(f["n_rmu_transverse_logical"])[()])
                    if "n_rmu_transverse_logical" in f else None)
        charge_zeta_identity = (
            _decode_charge_zeta_identity(
                f[CHARGE_ZETA_IDENTITY_DATASET][()],
                where=f"Restart file {filename}")
            if CHARGE_ZETA_IDENTITY_DATASET in f else None)

    n_rmu_disk = int(shapes["V_qmunu"][-1])
    n_rmu_pad = padded_mu_extent(n_rmu_disk, divisor)
    nspinor = _check_nspinor(shapes["psi_full_y"][2],
                             f"read_restart_state_from_h5({filename})")

    # Integrity cross-check of the stamped transverse extent against the
    # dataset it describes (audit 2026-07-28: the stamp used to be
    # write-only shadow metadata, QUALITY_PATTERNS #3 — a mismatch means a
    # torn or hand-edited file and must refuse loudly rather than feed
    # downstream re-padding, #7).  Checked on the SHAPE, before any bytes
    # move, so a bad file costs nothing.
    if "psi_full_y_transverse" in shapes and stored_T is not None:
        disk_T = int(shapes["psi_full_y_transverse"][-1])
        if stored_T != disk_T:
            raise ValueError(
                f"Restart file {filename}: stamped "
                f"n_rmu_transverse_logical={stored_T} does not match the "
                f"psi_full_y_transverse μ extent on disk ({disk_T}).  The "
                f"file is internally inconsistent (torn write or "
                f"hand-edited) — regenerate the restart tensors "
                f"(restart=false).")

    # ---- pass 2: the N_mu²-class and ψ tensors, one tile per rank -------
    psi_spec = P(None, None, None, "y")          # legacy: (nk, n, s, μ_Y)
    psi_nmu_spec = P(None, "x", None, "y")        # face:   (nk, n_X, s, μ_Y)
    psi_mun_spec = P(None, None, "x", "y")        # face:   (nk, s, μ_X, n_Y)

    def _read_munu(io, name):
        if name not in shapes:
            return None
        off, shape, spec = _munu_slab_request(shapes[name], n_rmu_pad)
        with timing.section(
                f"gw_jax.restart.read.{name}", announce=True,
                label=f"restart SlabIO read {name}"):
            arr = io.read_slab(name, shape=shape, dtype=dtypes[name],
                               offset=off, mesh=mesh_xy,
                               partition_spec=spec)
            jax.block_until_ready(arr)
        with timing.section(
                f"gw_jax.restart.wedge_transform.{name}", announce=True,
                label=f"restart wedge transform {name}"):
            arr = _collapse_leading(arr, shapes[name], mesh_xy)
        # THE UNFOLD, AFTER THE READ AND BEFORE THE CALLER SEES IT.  The
        # request above derives its q extent from the dataset shape, so a
        # wedge simply arrives as (n_q_ibz, mu_pad, nu_pad) on the same spec
        # the unfold takes and returns.  A no-op on every non-wedge file.
            arr = _unfold_wedge(
                arr, wedge_tables.get(name), n_rmu_pad, mesh_xy)
            jax.block_until_ready(arr)
        return arr

    def _read_psi(io, name, n_mu_logical, *, spec, mu_axis=-1,
                  spinor_axis=2, band_axis=1):
        """One direct hyperslab of a ψ dataset, μ padded, at ``spec``.

        ``mu_axis``/``spinor_axis`` default to the legacy/nmu axis order
        (nk, n, s, μ); the mun face (nk, s, μ, n) passes both explicitly
        — its μ is axis -2 and its spinor is axis 1, not axis -1/2.  NO
        RESHARD happens here regardless of ``spec``: this is a straight
        SlabIO hyperslab read, so a face spec costs exactly what the
        legacy spec costs (one direct read), never a transpose collective.
        """
        if name not in shapes:
            return None
        ds = shapes[name]
        _check_nspinor(ds[spinor_axis], f"{name} in {filename}")
        pad = padded_mu_extent(int(n_mu_logical), divisor)
        shape = list(int(s) for s in ds)
        shape[mu_axis] = int(pad)
        if n_band_carrier is not None:
            b_axis = int(band_axis)
            if b_axis < 0:
                b_axis += len(shape)
            if shape[b_axis] > int(n_band_carrier):
                raise ValueError(
                    f"Restart dataset {name!r} stores {shape[b_axis]} logical "
                    f"bands but this run's carrier has only "
                    f"{int(n_band_carrier)} slots. The band identity preflight "
                    "should have refused this mismatch before tensor I/O.")
            shape[b_axis] = int(n_band_carrier)
        with timing.section(
                f"gw_jax.restart.read.{name}", announce=True,
                label=f"restart SlabIO read {name}"):
            arr = io.read_slab(
                name, shape=tuple(shape), dtype=dtypes[name],
                mesh=mesh_xy, partition_spec=spec)
            jax.block_until_ready(arr)
        return arr

    n_rmu_T_disk = (int(shapes["psi_full_y_transverse"][-1])
                    if "psi_full_y_transverse" in shapes else None)

    with SlabIO(filename, mode="r", mesh=mesh_xy) as io:
        V_qmunu = _read_munu(io, "V_qmunu")
        S_qmunu = _read_munu(io, "S_qmunu")
        V0_noG0_munu = _read_munu(io, "V0_noG0_munu")
        if low_mem_bands:
            psi_full_y = None
            psi_nmu = _read_psi(io, "psi_full_y", n_rmu_disk,
                                spec=psi_nmu_spec)
            psi_mun = _read_psi(io, "psi_full_y_mun", n_rmu_disk,
                                spec=psi_mun_spec, mu_axis=-2, spinor_axis=1,
                                band_axis=-1)
            # Transverse (bispinor) faces: same two-hyperslab pattern at
            # the transverse μ extent.  A file holding the nmu-order
            # dataset WITHOUT the additive mun face was written by a
            # legacy-layout run — refuse rather than derive the second
            # face with an unowned x<->y transpose (the same
            # request-both-specs ruling as the charge pair).
            psi_full_y_transverse = None
            if n_rmu_T_disk is not None:
                # (missing-mun refusal fired in pass 1, before SlabIO)
                psi_nmu_T = _read_psi(
                    io, "psi_full_y_transverse", n_rmu_T_disk,
                    spec=psi_nmu_spec)
                psi_mun_T = _read_psi(
                    io, "psi_full_y_transverse_mun", n_rmu_T_disk,
                    spec=psi_mun_spec, mu_axis=-2, spinor_axis=1,
                    band_axis=-1)
            else:
                psi_nmu_T = None
                psi_mun_T = None
        else:
            psi_full_y = _read_psi(io, "psi_full_y", n_rmu_disk,
                                   spec=psi_spec)
            psi_nmu = None
            psi_mun = None
            psi_nmu_T = None
            psi_mun_T = None
            psi_full_y_transverse = (
                _read_psi(io, "psi_full_y_transverse", n_rmu_T_disk,
                          spec=psi_spec)
                if n_rmu_T_disk is not None else None)

    # G0: μ-class, read whole above.  Collapse a legacy 2-D (nqz, μ) store
    # to its q=0 row, pad to the same in-memory μ extent as everything
    # else, and pin it to the ν axis.  Done HERE so the reader's contract
    # is uniform — every array it returns is padded and sharded — rather
    # than leaving one straggler for the caller to remember.
    if G0_mu_nu is not None:
        if G0_mu_nu.ndim > 1:
            G0_mu_nu = G0_mu_nu[0]
        g0_pad = int(n_rmu_pad) - int(G0_mu_nu.shape[-1])
        if g0_pad > 0:
            G0_mu_nu = np.pad(G0_mu_nu, (0, g0_pad))
        # ``device_put_process_local``, NOT ``jax.device_put`` (AA.1):
        # G0 is host numpy read identically on every rank, and a plain
        # device_put onto a multi-process NamedSharding fires a hidden
        # assert_equal all-gather to prove exactly that.  The old reader
        # used ``with_sharding_constraint`` here, which was only ever
        # exercised at P=1 because the reader was refused above it -- so
        # there was no proven multi-process spelling to inherit.
        G0_mu_nu = device_put_process_local(
            np.ascontiguousarray(G0_mu_nu),
            NamedSharding(mesh_xy, P("y")))
    if enk_full is not None:
        if n_band_carrier is not None:
            n_disk_band = int(enk_full.shape[-1])
            n_target_band = int(n_band_carrier)
            if n_disk_band > n_target_band:
                raise ValueError(
                    f"Restart enk_full stores {n_disk_band} logical bands but "
                    f"this run's carrier has only {n_target_band} slots. The "
                    "band identity preflight should have refused this mismatch.")
            if n_disk_band < n_target_band:
                if enk_full.size == 0:
                    raise ValueError(
                        "Restart enk_full is empty and cannot define the finite "
                        "energy sentinel needed for band-carrier padding.")
                sentinel = float(np.max(enk_full)) + 1.0
                pad = np.full(
                    (enk_full.shape[0], n_target_band - n_disk_band),
                    sentinel, dtype=enk_full.dtype)
                enk_full = np.concatenate((enk_full, pad), axis=-1)
        enk_full = device_put_process_local(
            np.ascontiguousarray(enk_full),
            NamedSharding(mesh_xy, P(None, None)))

    del nspinor  # gated above; the extent itself rides on the arrays
    return (V_qmunu, S_qmunu, psi_full_y, enk_full, V0_noG0_munu, G0_mu_nu,
            psi_full_y_transverse, n_rmu_T_disk, psi_nmu, psi_mun,
            psi_nmu_T, psi_mun_T, charge_zeta_identity)


def read_munu_tensor_from_h5(filename, name, mesh_xy, *, n_rmu_logical=None):
    """Read ONE ``(…, μ, ν)`` restart tensor, sharded, wedge unfolded.

    ``read_restart_state_from_h5`` reads the fixed set of tensors ``gw_init``
    writes on its ``mode="w"`` pass and deliberately does not know about
    ``W0_qmunu`` — W is written later, by a different function, once the
    Dyson solve has produced it.  Every consumer that wants W back has
    therefore had to re-derive the slab request, the legacy-layout collapse
    and the wedge unfold for itself (``bse_io`` does, at its own scale).
    This is that read, named once, on the same three private helpers the
    canonical reader uses, so a fourth consumer does not spell it a fourth
    way.

    Returns ``None`` when the dataset is absent — which is the normal case
    for ``V_qmunu_nohead`` / ``W0_qmunu_nohead``, an opt-in pair nothing
    in-tree writes.  Callers that REQUIRE the tensor say so themselves; a
    reader that raised here could not serve the optional ones.

    Padding, sharding and the wedge follow the canonical reader exactly:
    disk holds the LOGICAL μ extent, memory holds
    ``padded_mu_extent(μ, device_count)`` with zero pad rows, output is
    ``P(None,'x','y')``, and an IBZ-wedge dataset comes back on the FULL BZ
    with the caller none the wiser.

    ``n_rmu_logical`` overrides the μ extent read off the dataset — pass it
    only when the dataset itself is the thing under suspicion.
    """
    from .slab_io import SlabIO
    from runtime.padding import padded_mu_extent

    with h5py.File(filename, "r") as f:
        if name not in f:
            return None
        ds_shape = tuple(int(s) for s in f[name].shape)
        ds_dtype = f[name].dtype
        wedge_tables = _qirr_wedge_tables(f)

    n_rmu_disk = int(ds_shape[-1] if n_rmu_logical is None else n_rmu_logical)
    n_rmu_pad = padded_mu_extent(n_rmu_disk, int(jax.device_count()))
    off, shape, spec = _munu_slab_request(ds_shape, n_rmu_pad)
    with SlabIO(filename, mode="r", mesh=mesh_xy) as io:
        arr = io.read_slab(name, shape=shape, dtype=ds_dtype, offset=off,
                           mesh=mesh_xy, partition_spec=spec)
    arr = _collapse_leading(arr, ds_shape, mesh_xy)
    return _unfold_wedge(arr, wedge_tables.get(name), n_rmu_pad, mesh_xy)


def load_restart_state_from_h5(filename, mesh_xy, band_slices=None,
                              n_rmu_logical=None, low_mem_bands=False):
    """Load canonical restart state, in ONE of two mutually exclusive ψ
    shapes selected by ``low_mem_bands`` (mirrors
    ``gw.wavefunction_bundle.Wavefunctions``'s ``layout`` tag).

    ``low_mem_bands=False`` (default) returns a ``SimpleNamespace`` with:

      V_qmunu, S_qmunu, V0_noG0_munu, G0_mu_nu, enk_full
      psi_rmu_Y   (nk, nb, ns, n_rmu)   P(None, None, None, 'y')
                  un-conjugated ψ, for :func:`gw.wavefunction_bundle.
                  build_wavefunctions`.
      psi_rmuT_X  (nk, n_rmu, nb, ns)   P(None, 'x', None, None)
                  conjugated ψ* (matches the pair-density convention
                  ``load_centroids_band_chunked`` uses).  Derived from
                  ``psi_rmu_Y`` with a single y→x all-to-all on the μ
                  axis — this remains the ONLY reshard on the legacy
                  restart path.

    ``low_mem_bands=True`` skips that derivation entirely and instead
    returns the two FACE arrays, each read as its own direct SlabIO
    hyperslab (see :func:`read_restart_state_from_h5`'s docstring — this
    is the "request both face specs" branch, not a one-face-plus-transpose
    branch, so there is NO reshard collective on this path either):

      psi_nmu  (nk, n, s, μ)   P(None, 'x', None, 'y')
      psi_mun  (nk, s, μ, n)   P(None, None, 'x', 'y')

      ``psi_rmu_Y``/``psi_rmuT_X`` are ``None`` in this mode; bispinor
      transverse fields are always ``None`` (not supported under
      ``low_mem_bands`` — refused by the caller before this is reached).
    """
    from types import SimpleNamespace
    # Loud-fail BEFORE any tensor is trusted (see the function's docstring).
    assert_restart_window_matches(filename, band_slices=band_slices,
                                  n_rmu_logical=n_rmu_logical)
    # Everything below arrives ALREADY sharded on mesh_xy and ALREADY at
    # the padded μ extent.  What used to be here — an 8-D/6-D collapse, a
    # ``jnp.pad`` on both μ axes of four tensors, and a
    # ``with_sharding_constraint`` on each — all operated on arrays that
    # were already resident whole on every rank, which is precisely why
    # the reader had to be guarded off above one process.  The pad is now
    # the read (SlabIO zero-fills past the dataset) and the sharding is
    # the read (SlabIO returns the tile), so none of it survives here.
    (V_qmunu, S_qmunu, psi_rmu_Y, enk_full, V0_noG0_munu, G0_mu_nu,
     psi_rmu_Y_T, n_rmu_T_disk, psi_nmu, psi_mun,
     psi_nmu_T, psi_mun_T, charge_zeta_identity) = read_restart_state_from_h5(
        filename, mesh_xy, low_mem_bands=bool(low_mem_bands),
        n_band_carrier=(
            int(band_slices.b4) - int(band_slices.b0)
            if band_slices is not None else None))

    if low_mem_bands:
        # No derivation, no reshard: both faces already arrived at their
        # own spec.  Legacy psi_rmu_Y/psi_rmuT_X are not built at all.
        # The transverse (bispinor) pair follows the identical pattern at
        # its own mu extent; None on a scalar/spinor file.
        return SimpleNamespace(
            V_qmunu=V_qmunu, S_qmunu=S_qmunu, V0_noG0_munu=V0_noG0_munu,
            G0_mu_nu=G0_mu_nu, enk_full=enk_full,
            psi_rmu_Y=None, psi_rmuT_X=None,
            psi_nmu=psi_nmu, psi_mun=psi_mun,
            psi_rmu_Y_transverse=None, psi_rmuT_X_transverse=None,
            psi_nmu_transverse=psi_nmu_T, psi_mun_transverse=psi_mun_T,
            n_rmu_transverse_disk=n_rmu_T_disk,
            charge_zeta_identity=charge_zeta_identity,
        )

    x1_psi_X = NamedSharding(mesh_xy, P(None, "x", None, None))

    # psi_rmuT_X: conj + transpose(nb↔μ) then y→x reshard on μ.  This is
    # the ONLY reshard on the legacy restart path, and it is deliberate:
    # the two ψ copies are what the pair-density contraction needs.
    with timing.section(
            "gw_jax.restart.final_reshard.charge", announce=True,
            label="restart final charge-wavefunction reshard"):
        psi_rmuT_X = jax.lax.with_sharding_constraint(
            jnp.conj(psi_rmu_Y).transpose(0, 3, 1, 2), x1_psi_X)
        jax.block_until_ready(psi_rmuT_X)

    # Bispinor transverse ψ: same two-copy derivation as the charge ψ, at
    # the TRANSVERSE μ extent (its own centroid count, its own pad, both
    # already applied by the reader).
    psi_rmuT_X_T = None
    if psi_rmu_Y_T is not None:
        with timing.section(
                "gw_jax.restart.final_reshard.transverse", announce=True,
                label="restart final transverse-wavefunction reshard"):
            psi_rmuT_X_T = jax.lax.with_sharding_constraint(
                jnp.conj(psi_rmu_Y_T).transpose(0, 3, 1, 2), x1_psi_X)
            jax.block_until_ready(psi_rmuT_X_T)

    return SimpleNamespace(
        V_qmunu=V_qmunu, S_qmunu=S_qmunu, V0_noG0_munu=V0_noG0_munu,
        G0_mu_nu=G0_mu_nu, enk_full=enk_full,
        psi_rmu_Y=psi_rmu_Y, psi_rmuT_X=psi_rmuT_X,
        psi_nmu=None, psi_mun=None,
        psi_rmu_Y_transverse=psi_rmu_Y_T,
        psi_rmuT_X_transverse=psi_rmuT_X_T,
        psi_nmu_transverse=None, psi_mun_transverse=None,
        n_rmu_transverse_disk=n_rmu_T_disk,
        charge_zeta_identity=charge_zeta_identity,
    )
