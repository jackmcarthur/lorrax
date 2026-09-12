"""Canonical restart-state I/O for GW/BSE workflows.

This module reads/writes HDF5 restart files in the v2 format used by gw_jax.
"""
from __future__ import annotations

import hashlib
import time
import os
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import h5py
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import (
    all_gather_processes, barrier, rank0_transaction,
)
from .commit_state import set_commit_state
import common.timing as timing
from runtime.padding import (
    authenticate_axis,
    authenticate_padded_axis,
    mesh_divisor,
    pad_to_axis,
    padded_axis,
    padded_mu_axis,
)


RESTART_LOGICAL_SHAPE_ATTR = "restart_logical_shape"
RESTART_CARRIER_SHAPE_ATTR = "restart_carrier_shape"
RESTART_PADDED_AXES_ATTR = "restart_padded_axes"
BAND_WINDOW_SCHEMA_DATASET = "band_window_schema"
BAND_WINDOW_SCHEMA_VERSION = 2
BAND_WINDOW_CARRIER_DATASET = "band_window_carrier"
CHARGE_ZETA_IDENTITY_DATASET = "charge_zeta_identity"
SHARED_POLE_MEMBER_DATASET = "shared_pole_member"
_SHARED_POLE_MEMBER_FIELDS = ("path", "schema", "digest", "iteration_id")


class SharedPoleMemberMissing(ValueError):
    """No member in a committed bundle; the caller may construct it once."""

    status = "missing"

    def __init__(self, reason):
        self.reason = reason
        super().__init__(reason)


class SharedPoleMemberRefused(ValueError):
    """Existing member is incompatible or corrupt; never rebuild in place."""

    status = "refused"

    def __init__(self, reason):
        self.reason = reason
        super().__init__(f"{reason}; never overwrite this member; run against "
                         "a copy of the bundle (restart=true)")


def _shared_pole_member_record(values):
    """Authenticate the small, ordered restart membership receipt."""
    raw = np.asarray(values)
    if raw.shape != (4,) or raw.dtype.kind not in ("S", "U", "O"):
        raise ValueError("GATE shared_pole_member: expected four UTF-8 strings")
    decoded = []
    for value in raw.tolist():
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        if not isinstance(value, str) or not value or "\x00" in value:
            raise ValueError("GATE shared_pole_member: empty or invalid field")
        decoded.append(value)
    record = dict(zip(_SHARED_POLE_MEMBER_FIELDS, decoded))
    if Path(record["path"]).is_absolute() or record["path"] == ".":
        raise ValueError("GATE shared_pole_member: member path must be relative")
    digest = record["digest"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("GATE shared_pole_member: digest must be a SHA256 hex string")
    return record


def register_shared_pole_restart_member(
        restart_path, model_path, *, expected_identity, mesh_xy, capacity=None):
    """Register an authenticated immutable model after its construction.

    Parameters
    ----------
    restart_path, model_path : path-like
        Existing restart bundle and completed model. Only a relative path,
        schema, SHA256 digest and iteration ID enter the restart bundle.
    expected_identity : dict
        Current SC identity passed unchanged to the model's validator.
    mesh_xy : jax.sharding.Mesh
        Current named mesh; every rank participates in validation/transaction.
    capacity : CapacityLedger, optional
        Same map ledger with bound caller lifetimes; required for payload
        authentication. Metadata-only validation cannot register a member.

    Returns
    -------
    dict
        Immutable member receipt. Registering the same receipt is idempotent;
        changing it requires a new restart bundle. No model data are appended.
    """
    from .commit_state import assert_committed
    from .shared_pole_store import validate_shared_pole_model

    restart_path = Path(restart_path).resolve()
    model_path = Path(model_path).resolve()
    member = {}

    def _validate():
        if model_path == restart_path:
            raise ValueError("GATE shared_pole_member: model must be a separate file")
        header = validate_shared_pole_model(
            model_path, expected_identity=expected_identity, mesh_xy=mesh_xy,
            **({"capacity":capacity} if capacity is not None else {}))
        if header.get("validation_receipt", {}).get("status") == "NOT_MEASURED":
            raise ValueError("GATE shared_pole_member: payload authentication requires capacity")
        member.update(_shared_pole_member_record([
            os.path.relpath(model_path, restart_path.parent), header["schema"],
            header["digest"], header["identity"]["iteration_id"]]))

    def _write():
        # r+ cannot accidentally create a missing restart. Model validation
        # has released its handle before this metadata-only transaction.
        with h5py.File(restart_path, "r+") as h5:
            assert_committed(h5, path=restart_path)
            if SHARED_POLE_MEMBER_DATASET in h5:
                existing = _shared_pole_member_record(
                    h5[SHARED_POLE_MEMBER_DATASET][()])
                if existing != member:
                    raise ValueError(
                        "GATE shared_pole_member: immutable member replacement "
                        "refused; use a new restart bundle for this SC map")
                return
            set_commit_state(h5, False)
            h5.create_dataset(
                SHARED_POLE_MEMBER_DATASET,
                data=np.asarray([member[k].encode("utf-8")
                                 for k in _SHARED_POLE_MEMBER_FIELDS], dtype="S"))
            set_commit_state(h5, True)

    rank0_transaction(restart_path, stage="restart.shared_pole_member",
                      validate=_validate, write=_write)
    return member


def read_shared_pole_restart_member(
        restart_path, *, expected_identity, mesh_xy, capacity=None,
        return_header=False):
    """Authenticate current-map membership without expanding factors into W.

    Parameters
    ----------
    restart_path : path-like
        Restart bundle containing the immutable member receipt.
    expected_identity : dict
        Current SC identity, authenticated by the model format owner.
    mesh_xy : jax.sharding.Mesh
        Current named mesh passed to the model validator on every rank.
    capacity : CapacityLedger, optional
        Same map ledger with bound caller lifetimes; required to authenticate
        payload bytes before accepting the linked digest.
    return_header : bool, optional
        Return (member, header) from the SAME validation when True. The
        default remains the four-string member dictionary.

    Returns
    -------
    dict or tuple of (dict, dict)
        Present: relative path, schema, digest and iteration ID, optionally
        paired with the already-authenticated header. No second validation.

    Raises
    ------
    SharedPoleMemberMissing
        status="missing": no member in an existing committed bundle. The
        caller may construct/register once in that bundle.
    SharedPoleMemberRefused
        status="refused": incomplete, malformed, missing linked payload,
        identity/recipe/gate mismatch or corrupt payload. ``reason`` preserves
        the exact validator diagnostic. Never classified as missing, and
        never permission to replace an immutable member.
    """
    from .commit_state import agree_io_refusal, assert_committed
    from .shared_pole_store import validate_shared_pole_model

    try:
        # Agree after serial metadata I/O and before the validator enters
        # its collective payload digest. Missing is rebuildable only when
        # every reader sees the same committed bundle with no member.
        restart_path = Path(restart_path).absolute()
        member, error = None, None
        try:
            restart_path = restart_path.resolve()
            with h5py.File(restart_path, "r") as h5:
                assert_committed(h5, path=restart_path)
                if SHARED_POLE_MEMBER_DATASET in h5:
                    member = _shared_pole_member_record(h5[SHARED_POLE_MEMBER_DATASET][()])
        except Exception as exc:
            error = ValueError(exc.reason) if isinstance(exc, SharedPoleMemberRefused) else exc
        agree_io_refusal(error, path=restart_path,
                         stage="restart.shared_pole_member/read")
        receipt = hashlib.sha256(repr(member).encode("utf-8")).digest()
        receipts = np.asarray(all_gather_processes(np.frombuffer(receipt, np.uint8)))
        if not np.all(receipts == receipts.reshape(-1, 32)[0]):
            raise SharedPoleMemberRefused("GATE shared_pole_member: ranks read different membership receipts")
        if member is None:
            raise SharedPoleMemberMissing("GATE shared_pole_member: restart has no model member")
        header = validate_shared_pole_model(
            restart_path.parent / member["path"],
            expected_identity=expected_identity, mesh_xy=mesh_xy,
            **({"capacity":capacity} if capacity is not None else {}))
        if header.get("validation_receipt", {}).get("status") == "NOT_MEASURED":
            raise ValueError("GATE shared_pole_member: payload authentication requires capacity")
        linked = {"schema": header["schema"], "digest": header["digest"],
                  "iteration_id": header["identity"]["iteration_id"]}
        for key, value in linked.items():
            if member[key] != value:
                raise ValueError(
                    f"GATE shared_pole_member: linked {key} changed; "
                    f"got {value!r}, want {member[key]!r}; rebuild the SC bundle")
        return (member, header) if return_header else member
    except (SharedPoleMemberMissing, SharedPoleMemberRefused):
        raise
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        raise SharedPoleMemberRefused(str(exc)) from exc


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


def _shape_receipt_attrs(
        carrier_shape, logical_shape, *, axis_receipts=()):
    """Dataset attrs that keep producer carrier and disk shape distinct."""
    attrs = {
        RESTART_CARRIER_SHAPE_ATTR: np.asarray(
            tuple(int(v) for v in carrier_shape), dtype=np.int64),
        RESTART_LOGICAL_SHAPE_ATTR: np.asarray(
            tuple(int(v) for v in logical_shape), dtype=np.int64),
    }
    if axis_receipts:
        attrs[RESTART_PADDED_AXES_ATTR] = np.asarray(
            tuple((int(axis), int(tag.logical), int(tag.carrier),
                   int(tag.divisor))
                  for axis, tag in axis_receipts),
            dtype=np.int64)
    return attrs


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


def _loaded_band_axis(logical: int, mesh_or_divisor):
    """Canonical receipt shared by every persistent restart band dataset."""
    shape = getattr(mesh_or_divisor, "shape", {})
    if "x" in shape and "y" in shape:
        from common.wfn_layout import PSI_MUN_SPEC, PSI_NMU_SPEC
        return padded_axis(
            int(logical), mesh_or_divisor, name="restart loaded-band carrier",
            specs=((PSI_NMU_SPEC, 1), (PSI_MUN_SPEC, 3)))
    return padded_axis(
        int(logical), mesh_divisor(mesh_or_divisor),
        name="restart loaded-band carrier")


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
    if RESTART_PADDED_AXES_ATTR not in ds.attrs:
        return
    receipts = np.asarray(ds.attrs[RESTART_PADDED_AXES_ATTR], dtype=np.int64)
    if receipts.ndim != 2 or receipts.shape[1] != 4:
        raise ValueError(
            f"Restart dataset {name!r}: {RESTART_PADDED_AXES_ATTR!r} must "
            f"have shape (n,4); got {receipts.shape}.")
    seen = set()
    for raw_axis, logical, carried, divisor in receipts.tolist():
        axis = int(raw_axis)
        if not 0 <= axis < len(actual) or axis in seen:
            raise ValueError(
                f"Restart dataset {name!r}: invalid or repeated padded axis "
                f"{axis} in {RESTART_PADDED_AXES_ATTR!r}.")
        seen.add(axis)
        tag = authenticate_padded_axis(
            int(logical), int(carried), int(divisor),
            name=f"Restart dataset {name!r} axis {axis}")
        if actual[axis] != tag.logical or carrier[axis] != tag.carrier:
            raise ValueError(
                f"Restart dataset {name!r}: padded-axis receipt for axis "
                f"{axis} declares logical/carrier "
                f"{tag.logical}/{tag.carrier}, but dataset/carrier shapes "
                f"declare {actual[axis]}/{carrier[axis]}.")


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




#: The group ``gw.downfold_run`` stamps on a compressed bundle.  Named here,
#: beside the Coulomb-policy stamp, because it is part of the RESTART FORMAT
#: and has two sides: the downfold writes it and the BSE drivers read it.  A
#: reader that needed its own copy of the group name would be a second owner
#: of a format detail, which is the drift this module exists to prevent.
DOWNFOLD_PROVENANCE_GROUP = "downfold_provenance"








def write_restart_state_to_h5(
    filename,
    *,
    n_rmu_logical: int,
    V_qmunu=None,
    psi_parent_y=None,
    psi_parent_y_mun=None,
    psi_parent_y_transverse=None,
    psi_parent_y_transverse_mun=None,
    parent_k_rows=None,
    psi_layout="face",
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

    ``psi_parent_y`` and ``psi_parent_y_mun`` store the two raw-parent
    orientations, (parent_k, band, spin, centroid) and
    (parent_k, spin, centroid, band). Both memory layouts write these same
    logical arrays. Current-family companions use their own centroid extent.
    ``file_io.restart_bundle`` restores the requested sharding on read.

    ``qirr`` IS THE ONE RESOLUTION FOR BOTH TENSORS.  A
    ``gw.restart_q_storage.RestartQStorage`` whose ``.store_wedge`` is True
    means V is written on the IBZ q wedge (from ``qirr.capture.X_ibz``, the
    PRE-UNFOLD block the producer held) and the W0 placeholder is sized from
    THAT — which is the coupling this function's comment has been asking the
    next writer to close since dbe3b4ec.  ``None`` (the default, and what
    every existing caller passes) is today's behaviour exactly: full-BZ V,
    full-BZ placeholder, no stamp, no table group.

    ``qp_state_source_record`` identifies the WFN whose matched
    ``psi_parent_y`` / ``enk_full`` state this restart stores.  Its format and
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
    carrier_divisor = mesh_divisor(
        mesh if mesh is not None else int(jax.device_count()))
    band_receipts = None
    n_band_logical = None
    loaded_band_tag = None
    if band_slices is not None:
        band_receipts = _band_window_receipts(band_slices)
        n_band_logical = int(band_receipts[0][4] - band_receipts[0][0])
        loaded_band_tag = _loaded_band_axis(
            n_band_logical,
            mesh if mesh is not None else carrier_divisor)
        authenticate_padded_axis(
            n_band_logical,
            int(band_receipts[1][4] - band_receipts[1][0]),
            loaded_band_tag.divisor, name=loaded_band_tag.name)
    elif mode != "w" and any(
            arr is not None for arr in (
                psi_parent_y, psi_parent_y_mun,
                psi_parent_y_transverse, psi_parent_y_transverse_mun, enk_full)):
        # Append calls intentionally do not repeat band_slices.  A schema-2
        # file's band_window is logical, so it is sufficient to clip the later
        # psi faces to the same portable disk extent.  A legacy file has no
        # way to distinguish physical rows from its mesh pad and stays on its
        # historical full-carrier storage path.
        # Finish the preceding publication on every rank before opening any
        # reader. barrier delegates to multihost_utils.sync_global_devices.
        barrier("restart.band_window.before_read")
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
                loaded_band_tag = _loaded_band_axis(
                    n_band_logical,
                    mesh if mesh is not None else carrier_divisor)
        # A fast rank must not open SlabIO for append while a peer still has
        # this metadata reader open (or has not reached its open yet).
        barrier("restart.band_window.readers_closed")

    if ((psi_parent_y_transverse is not None
         or psi_parent_y_transverse_mun is not None)
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
        ndim = len(tuple(arr.shape))
        axis_receipts = []
        mu_tag = padded_mu_axis(int(n_log), carrier_divisor)
        shape = _mu_logical_shape(
            arr.shape, mu_axes, n_log,
            where=f"write_restart_state_to_h5 dataset {name!r}")
        for raw_axis in mu_axes:
            axis = int(raw_axis) % ndim
            authenticate_axis(
                arr, mu_tag, axis=axis,
                where=f"write_restart_state_to_h5 dataset {name!r}")
            axis_receipts.append((axis, mu_tag))
        if band_axes and n_band_logical is not None:
            band_tag = loaded_band_tag
            if band_tag is None:
                raise RuntimeError(
                    f"restart dataset {name!r}: missing loaded-band receipt")
            shape = _logical_storage_shape(
                shape, band_axes, n_band_logical,
                where=f"write_restart_state_to_h5 dataset {name!r} band axis")
            for raw_axis in band_axes:
                axis = int(raw_axis) % ndim
                authenticate_axis(
                    arr, band_tag, axis=axis,
                    where=f"write_restart_state_to_h5 dataset {name!r}")
                axis_receipts.append((axis, band_tag))
        write_plan[name] = (
            shape, _shape_receipt_attrs(
                arr.shape, shape, axis_receipts=axis_receipts))

    _plan("V_qmunu", V_on_disk, mu_axes=(-2, -1))
    _plan("S_qmunu", S_qmunu, mu_axes=(-2, -1))
    _plan("V0_noG0_munu", V0_noG0_munu, mu_axes=(-2, -1))
    _plan("G0_mu_nu", G0_mu_nu, mu_axes=(-1,))
    # Parents-only storage (gw_init): the raw-parent faces in CANONICAL
    # centroid order, n_parent = k_irr rows, and nothing on the full BZ.
    # ``psi_parent_k_rows`` names the full-k row each parent IS
    # (SymMaps.kirr_fullids), so a restart can check its plan against the
    # file before trusting the rows.
    _plan("psi_parent_y", psi_parent_y, mu_axes=(-1,), band_axes=(1,))
    _plan("psi_parent_y_mun", psi_parent_y_mun, mu_axes=(-2,),
          band_axes=(-1,))
    if (psi_parent_y is None) != (psi_parent_y_mun is None) or (
            (psi_parent_y is not None) != (parent_k_rows is not None)):
        raise ValueError(
            "write_restart_state_to_h5: psi_parent_y, psi_parent_y_mun and "
            "parent_k_rows travel together (all or none).")
    if (psi_parent_y_transverse is None) != (psi_parent_y_transverse_mun is None):
        raise ValueError("write_restart_state_to_h5: transverse parent faces travel together.")
    if psi_parent_y_transverse is not None and (
            psi_parent_y is None or psi_parent_y_transverse.shape[0] != psi_parent_y.shape[0]):
        raise ValueError("write_restart_state_to_h5: both families must share the raw-parent rows.")
    _plan("psi_parent_y_transverse", psi_parent_y_transverse,
          mu_axes=(-1,), n_logical=n_T, band_axes=(1,))
    _plan("psi_parent_y_transverse_mun", psi_parent_y_transverse_mun,
          mu_axes=(-2,), n_logical=n_T, band_axes=(-1,))
    _plan("enk_full", enk_full, band_axes=(-1,))
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
        if parent_k_rows is not None:
            io.write_attr("psi_parent_k_rows",
                          np.asarray(parent_k_rows, dtype=np.int64))
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
        # BAND-WINDOW PROVENANCE.  V_qmunu / psi_parent_y / enk_full are all
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
            if name.startswith("psi_parent_"):
                from common.wfn_layout import psi_specs
                psi_specs(psi_layout)
                attrs = dict(attrs, psi_layout=psi_layout)
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
        # (nk, s, μ, n): μ is axis -2, not -1 — the mun face's axis order
        # differs from every other dataset this writer knows about.
        # Parents-only storage: the raw-parent faces (k_irr rows), same
        # two axis orders as the full-k pair above.
        _write("psi_parent_y", psi_parent_y)
        _write("psi_parent_y_mun", psi_parent_y_mun)
        _write("enk_full", enk_full)

        # Bispinor per-channel ψ: μ axis clipped to the TRANSVERSE
        # logical extent (its own centroid count, not n_rmu_logical).
        _write("psi_parent_y_transverse", psi_parent_y_transverse)
        _write("psi_parent_y_transverse_mun", psi_parent_y_transverse_mun)
        if psi_parent_y_transverse is not None:
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
    mu_tag = padded_mu_axis(
        int(n_rmu_logical),
        mesh if mesh is not None else int(jax.device_count()))
    authenticate_axis(
        W0_qmunu, mu_tag, axis=-2,
        where="write_w0_qmunu_to_h5 dataset 'W0_qmunu'")
    authenticate_axis(
        W0_qmunu, mu_tag, axis=-1,
        where="write_w0_qmunu_to_h5 dataset 'W0_qmunu'")
    ndim = int(W0_qmunu.ndim)
    shape_attrs = _shape_receipt_attrs(
        W0_qmunu.shape, shape,
        axis_receipts=((ndim - 2, mu_tag), (ndim - 1, mu_tag)))
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




















def write_parent_wavefunctions(filename, psi, *, n_rmu_logical, mesh):
    """Persist an unreduced parent set in both canonical face orientations.

    ``psi`` is (parent_k, band, spin, centroid) in canonical centroid order.
    An unreduced downfold has every k as a parent and needs no new k action.
    """
    from common.wfn_layout import psi_specs
    nmu, mun = psi_specs("axis")
    y = jax.jit(lambda a: a, out_shardings=NamedSharding(mesh, nmu))(psi)
    x = jax.jit(lambda a: a.transpose(0, 2, 3, 1),
                out_shardings=NamedSharding(mesh, mun))(psi)
    write_restart_state_to_h5(filename, n_rmu_logical=n_rmu_logical,
        psi_parent_y=y, psi_parent_y_mun=x,
        parent_k_rows=np.arange(psi.shape[0]), psi_layout="axis", mesh=mesh, mode="a")
