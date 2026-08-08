"""q_irr restart tensors: write the wedge, unfold on read, refuse on drift.

WHAT IS STORED, AND WHY IT IS THE PRE-UNFOLD BLOCK.  A GW restart tensor
(``W0_qmunu``, ``V_qmunu``) is ``(n_q_full, N_μ, N_μ)`` complex128 — 0.944
GB each at the Si production shape — and on a deck whose centroid set is
orbit-closed every full-BZ row is a symmetry image of an IBZ row.  The
producer HOLDS the IBZ-shaped array one statement before it unfolds it
(``screening.py`` and ``v_q_g_flat.py`` both call
``unfold_isdf_operator`` on an ``(n_q_ibz, N_μ, N_μ)`` array they
already have).  This module persists
THAT array.  The round trip ``unfold(stored) == what the run used`` is then
an IDENTITY — the same function on the same inputs — rather than a property
that depends on the bit-frozen op-selection policy.  Slicing the unfolded
tensor instead is also bit-exact (MEASURED 0.000e+00), but only because
``sym_idx_q[q_irr_full_idx] == 0``, which is a consequence of a policy
nobody has agreed to freeze for this purpose.

THE PREREQUISITE IS ENFORCED AT WRITE TIME AND IT REFUSES.  Reconstructing
``W(Sq)`` from ``W(q)`` is a permutation of the (μ, ν) indices and that
permutation exists only if the centroid set is closed under the group.  A
q_irr file written against a non-closed set is silently unrecoverable —
there is no α to invert with — so :func:`write_qirr_tensor` calls
:func:`symmetry_maps.verify_centroid_orbit_closure` and REFUSES, naming
every offending op and its residual.  It never warns and continues.  On
today's production 960-centroid Si deck that refusal is the NORMAL path
(47 of 48 ops violating, worst 1.318e-01), so it is not a dead branch: it
is the branch production hits, and it has a first-class test.

THE TABLES LIVE IN THE FILE.  ``irr_idx_q``, ``sym_idx_q``, ``q_irr_frac``,
``sym_perm``, ``L_table`` and ``n_sym_spatial`` are written into a sibling
group of the dataset, not referenced from elsewhere: a tensor whose
reconstruction tables live in another artifact is a tensor that silently
decays the moment anything upstream is regenerated.  A digest over those
tables is stamped and re-checked on read.

SHAPE IS THE PRIMARY DISCRIMINANT; THE ATTR IS THE CROSS-CHECK.  ζ has
inferred IBZ-versus-full from the stored q extent since before this format
existed (``zeta_loader.py``: ``q_layout = "full_bz" if n_q_on_disk ==
n_q_full else "ibz"``) and has worked that way in production, so the shape
decides and ``q_storage`` is asserted against it.  Disagreement is a
REFUSAL, not a preference — an attr that is stamped and wrong is exactly
the failure a cross-check exists to catch.  A file carrying NO q_irr attrs
at all is read as ``q_storage="full"`` and returned unchanged, so every
restart file written before this format keeps working.

THE μ PAD NEVER REACHES DISK (SHARDING_RULES §2).  The producer bakes a
μ pad into ``sym_perm``/``L_table`` once at construction — identity tail
on the permutation, zero tail on the wrap — and its width is
``padded_mu_extent(n_rmu, device_count())``.  That is a
DEVICE-COUNT-DEPENDENT quantity, and a restart artifact may not carry
one, for the reason restart artifacts exist: a file written on four ranks
must read on eight.  Every tensor in the restart format already obeys
that rule; the unfold tables are new and obey it here.  ``write_qirr_tensor``
takes ``n_rmu_logical``, ASSERTS the tail really is a pad — identity
permutation, zero wrap, zero tensor rows, each refusing separately —
strips it, and stamps the logical extent.  ``read_tensor`` takes
``n_mu_padded`` and re-applies the READER's own.  The pad is
reconstructed, never stored, which is what makes the round trip an
identity ACROSS device counts and not merely within one.

WHAT THIS MODULE IS NOT.  It is file format only.  It does not know where
``W0_qmunu`` is produced or consumed and it touches no gw/bse call site;
wiring it into the producers is a separate change.  Everything here is
testable against a temporary HDF5 file with no deck, no mesh larger than
the caller's, and no lorrax on the path.

h5py IS IMPORTED LAZILY, ON PURPOSE.  ``import symmetry_maps`` must not
pull h5py in — the package answers table questions from arrays and
``test_symmetry_maps_import_isolation.py`` measures that it does — so the
import happens inside the two functions that open a file, and h5py is
declared as the ``store`` optional dependency rather than a hard one.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import os

import numpy as np

from symmetry_maps.maps import unfold_isdf_operator
from symmetry_maps.orbit_syms import (CLOSURE_TOL_DEFAULT,
                                      CentroidClosureVerdict,
                                      verify_centroid_orbit_closure)

#: Bump when the on-disk layout changes in a way a previous reader would
#: misread.  The reader refuses on any version it was not built for rather
#: than guessing — a format that reads an unknown version "best effort" is
#: a format that returns wrong numbers on the day it changes.
QIRR_FORMAT_VERSION = 1

#: Suffix of the sibling group holding the unfold tables for ``<name>``.
#: Per tensor, not per file: ``V_qmunu`` and ``W0_qmunu`` normally share
#: one set of tables, and sharing them on disk would make each tensor
#: depend on the other's rewrite history for its own recoverability.  The
#: duplication is a few kilobytes against a gigabyte of tensor.
QIRR_TABLE_SUFFIX = "__qirr"

#: The attr that says "this is a q_irr-format dataset".  Its PRESENCE is
#: what distinguishes a stamped file from a legacy one; see
#: :func:`read_tensor` for why a partially-stamped file refuses instead of
#: falling through to the legacy path.
_VERSION_ATTR = "qirr_format_version"

#: Every attr this format owns.  Used by the partial-stamp refusal.
_OWNED_ATTRS = (
    _VERSION_ATTR, "q_storage", "qirr_table_hash", "qirr_centroid_hash",
    "qirr_closure_verdict", "qirr_closure_worst_residual",
    "qirr_closure_tol", "qirr_n_q_full", "qirr_data_ready",
    "qirr_n_rmu_logical",
)

#: Table name -> the dtype it is canonicalised to on write AND hashed as.
#: Fixing these is what makes the digest reproducible from a read-back:
#: ``L_table`` arrives int8 from ``centroid_source_map_and_wrap`` and would
#: hash differently if a caller ever handed int64.
_TABLE_DTYPES = {
    "irr_idx_q": np.int32,
    "sym_idx_q": np.int32,
    "q_irr_frac": np.float64,
    "sym_perm": np.int32,
    "L_table": np.int64,
}

_TABLE_ORDER = ("irr_idx_q", "sym_idx_q", "q_irr_frac", "sym_perm",
                "L_table")

_COMMIT_CACHE: list = []


# ---------------------------------------------------------------------------
# The two records
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class QirrTables:
    """Everything needed to unfold a stored wedge back to the full BZ.

    The same five arrays and one scalar
    :func:`symmetry_maps.unfold_isdf_operator` takes, bundled so that a writer
    cannot supply four of them.  Built from ``SymMaps.irr_idx_q`` /
    ``sym_idx_q`` / ``q_irr_frac`` and
    ``centroid_source_map_and_wrap(..., extend_trs=True)``.
    """

    irr_idx_q: np.ndarray
    sym_idx_q: np.ndarray
    q_irr_frac: np.ndarray
    sym_perm: np.ndarray
    L_table: np.ndarray
    n_sym_spatial: int

    @property
    def n_q_full(self) -> int:
        return int(np.asarray(self.irr_idx_q).shape[0])

    @property
    def n_q_ibz(self) -> int:
        return int(np.asarray(self.q_irr_frac).shape[0])

    @property
    def n_mu(self) -> int:
        return int(np.asarray(self.sym_perm).shape[-1])

    def canonical(self) -> "QirrTables":
        """A copy with every array at its stored dtype."""
        return QirrTables(
            irr_idx_q=np.ascontiguousarray(self.irr_idx_q,
                                           dtype=_TABLE_DTYPES["irr_idx_q"]),
            sym_idx_q=np.ascontiguousarray(self.sym_idx_q,
                                           dtype=_TABLE_DTYPES["sym_idx_q"]),
            q_irr_frac=np.ascontiguousarray(
                self.q_irr_frac, dtype=_TABLE_DTYPES["q_irr_frac"]),
            sym_perm=np.ascontiguousarray(self.sym_perm,
                                          dtype=_TABLE_DTYPES["sym_perm"]),
            L_table=np.ascontiguousarray(self.L_table,
                                         dtype=_TABLE_DTYPES["L_table"]),
            n_sym_spatial=int(self.n_sym_spatial),
        )

    def logical(self, n_rmu_logical: int) -> "QirrTables":
        """Strip the μ PAD, asserting first that the tail really is one.

        WHY THE PAD MUST NOT REACH DISK.  The producer bakes the μ pad
        into ``sym_perm``/``L_table`` once at construction — an identity
        tail on the permutation, a zero tail on the wrap — and the pad
        width is ``padded_mu_extent(n_rmu, device_count())``.  It is a
        DEVICE-COUNT-DEPENDENT quantity, and SHARDING_RULES §2 forbids
        those in a restart artifact for the obvious reason: a file
        written on four ranks must read on eight.  Everything else in the
        restart format already obeys that rule (``_mu_logical_shape``
        clips every tensor on the way out); the unfold tables are new and
        must obey it too.

        THE ASSERTION IS FREE CORRUPTION DETECTION, which is why it is
        here and not left to the caller.  Three things must hold, and
        each fails differently:

        * the permutation tail is the identity — if it is not, this is
          not a pad, it is a table that genuinely addresses more
          centroids than the caller claims, and stripping it would throw
          away real structure;
        * the wrap tail is zero — same argument for ``L_table``;
        * the logical block maps into ITSELF.  This follows from the
          first point for a bijection, and is asserted anyway because it
          is the property that makes stripping meaning-preserving, and a
          property nobody states is a property nobody notices losing.
        """
        n_log = int(n_rmu_logical)
        perm = np.asarray(self.sym_perm)
        L = np.asarray(self.L_table)
        n_pad = int(perm.shape[-1])
        if n_log > n_pad:
            raise ValueError(
                f"qirr_store: n_rmu_logical={n_log} exceeds the table "
                f"extent {n_pad}; the logical set cannot be larger than "
                f"the padded one.")
        if n_log == n_pad:
            return self
        tail = perm[:, n_log:]
        want = np.broadcast_to(np.arange(n_log, n_pad, dtype=tail.dtype),
                               tail.shape)
        if not np.array_equal(tail, want):
            bad = int(np.argmax(np.any(tail != want, axis=0)))
            raise ValueError(
                f"qirr_store: sym_perm's μ tail is not the identity pad — "
                f"column {n_log + bad} of the padded extent {n_pad} reads "
                f"{tail[:, bad].tolist()[:4]}..., expected all "
                f"{n_log + bad}.  Either n_rmu_logical={n_log} is wrong or "
                f"these tables address real centroids past it; stripping "
                f"would silently discard a permutation this format then "
                f"could not invert.")
        if L.shape[1] > n_log and np.any(L[:, n_log:] != 0):
            raise ValueError(
                f"qirr_store: L_table's μ tail past {n_log} is not zero, so "
                f"it is not the pad the identity tail claimed it was.  The "
                f"umklapp phase those wraps carry would be dropped.")
        head = perm[:, :n_log]
        if head.size and int(head.max()) >= n_log:
            raise ValueError(
                f"qirr_store: sym_perm maps a logical centroid to index "
                f"{int(head.max())}, outside the logical block [0, "
                f"{n_log}).  The logical set is not closed under its own "
                f"permutation, so it cannot be stored without the pad.")
        return dataclasses.replace(
            self, sym_perm=head, L_table=L[:, :n_log])

    def padded(self, n_rmu_padded: int) -> "QirrTables":
        """Re-apply a μ pad of the READER's width — the inverse of above.

        The reader's device count is not the writer's, so the pad it
        needs is its own: identity tail on the permutation, zero tail on
        the wrap, exactly the shape the producer bakes in.  Reconstructed
        rather than stored, because a stored pad is the device-dependent
        thing :meth:`logical` exists to keep off disk.
        """
        n_pad = int(n_rmu_padded)
        perm = np.asarray(self.sym_perm)
        n_log = int(perm.shape[-1])
        if n_pad == n_log:
            return self
        if n_pad < n_log:
            raise ValueError(
                f"qirr_store: cannot pad tables of extent {n_log} down to "
                f"{n_pad}.")
        L = np.asarray(self.L_table)
        tail = np.broadcast_to(
            np.arange(n_log, n_pad, dtype=perm.dtype),
            (perm.shape[0], n_pad - n_log))
        return dataclasses.replace(
            self,
            sym_perm=np.concatenate([perm, tail], axis=1),
            L_table=np.concatenate(
                [L, np.zeros((L.shape[0], n_pad - n_log, L.shape[2]),
                             dtype=L.dtype)], axis=1))

    def digest(self) -> str:
        """``'sha256:<hex>'`` over the canonicalised tables, in a fixed order.

        Names go into the hash beside the bytes so that two tables of the
        same shape and dtype cannot be swapped for each other without the
        digest moving.
        """
        can = self.canonical()
        h = hashlib.sha256()
        for key in _TABLE_ORDER:
            arr = getattr(can, key)
            h.update(key.encode("utf-8"))
            h.update(str(arr.shape).encode("utf-8"))
            h.update(arr.tobytes())
        h.update(b"n_sym_spatial")
        h.update(np.int64(can.n_sym_spatial).tobytes())
        return "sha256:" + h.hexdigest()


@dataclasses.dataclass(frozen=True)
class QirrHeader:
    """What :func:`read_tensor` learned about the file it just read.

    Returned BESIDE the array rather than folded into it, because every
    field here is something a caller has a reason to assert on and a
    function that returned only the tensor would make each of those
    assertions a second file open.

    ``data_ready`` is the one that is not bookkeeping.  ``gw_init`` writes
    a full-size ZERO ``W0`` placeholder unconditionally, so
    ``os.path.exists`` on a restart file is always true and a run whose
    ``persist_w0`` never fired finds a file of exactly the right shape full
    of zeros — the mechanism behind the April BSE incident where a
    plausible excitonic spectrum came out of an all-zero screening tensor.
    A q_irr format that inherited that property would just make the zero
    file eight times smaller.
    """

    q_storage: str
    format_version: int | None
    n_q_on_disk: int
    n_q_full: int | None
    n_mu: int | None
    #: The LOGICAL μ extent stated on disk.  Equal to ``n_mu`` for a file
    #: written without a pad; smaller than the producer's in-memory extent
    #: whenever it padded for its device count.  ``None`` on a legacy
    #: file, which had no tables and therefore no pad question.
    n_rmu_logical: int | None
    centroid_hash: str | None
    table_hash: str | None
    closure_verdict: str | None
    closure_worst_residual: float | None
    closure_tol: float | None
    data_ready: bool | None
    provenance: dict

    @property
    def is_legacy(self) -> bool:
        """True for a file written before this format existed."""
        return self.format_version is None

    @property
    def was_unfolded(self) -> bool:
        return self.q_storage == "ibz"


# ---------------------------------------------------------------------------
# Provenance, in the kin_ion.h5 style
# ---------------------------------------------------------------------------

def _generator_commit() -> str:
    """Commit of the SOURCE TREE THIS MODULE RAN FROM — not the cwd's.

    Same rule and same fallback shape as ``kin_ion_io._generator_commit``:
    ``git -C <src>`` anchors the answer to the file that is executing,
    because the caller's cwd is routinely a scratch directory that is not a
    checkout.  A named ``unknown:<reason>`` beats a blank attr, which is
    the failure mode the stamp exists to remove, and the ``-dirty`` suffix
    is not decoration — a stamp that reads clean on an edited tree is worse
    than no stamp, because it is the exact claim the reader wanted.

    Memoised: the pair of git calls is ~1 s and a writer that stamps one
    tensor per restart should not pay it twice, let alone once per cell of
    a test suite.
    """
    if _COMMIT_CACHE:
        return _COMMIT_CACHE[0]
    import subprocess
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = "unknown:unset"
    try:
        rev = subprocess.run(
            ["git", "-C", src, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=60)
        if rev.returncode != 0:
            out = "unknown:not-a-git-checkout"
        else:
            out = rev.stdout.strip()
            dirty = subprocess.run(
                ["git", "-C", src, "status", "--porcelain",
                 "--untracked-files=no"],
                capture_output=True, text=True, timeout=60)
            if dirty.returncode == 0 and dirty.stdout.strip():
                out += "-dirty"
    except Exception as exc:                                  # noqa: BLE001
        out = f"unknown:{type(exc).__name__}"
    _COMMIT_CACHE.append(out)
    return out


# ---------------------------------------------------------------------------
# File-handle plumbing
# ---------------------------------------------------------------------------

def _require_h5py():
    try:
        import h5py
    except ImportError as exc:                           # pragma: no cover
        raise ImportError(
            "symmetry_maps.qirr_store needs h5py, which is the service's "
            "``store`` optional dependency and NOT one of its runtime "
            "dependencies: the package answers every table question from "
            "arrays and only this module opens a file.  Install "
            "``symmetry_maps[store]`` or add h5py to the environment."
        ) from exc
    return h5py


class _Dest:
    """Context manager over "an h5 group, or a path to open one"."""

    def __init__(self, target, mode):
        self._target = target
        self._mode = mode
        self._owned = None

    def __enter__(self):
        h5py = _require_h5py()
        if isinstance(self._target, (h5py.File, h5py.Group)):
            return self._target
        if not isinstance(self._target, (str, bytes, os.PathLike)):
            raise TypeError(
                f"qirr_store: expected an h5py File/Group or a path, got "
                f"{type(self._target).__name__}")
        self._owned = h5py.File(self._target, self._mode)
        return self._owned

    def __exit__(self, *exc):
        if self._owned is not None:
            self._owned.close()
        return False


def _attr_str(node, key):
    if key not in node.attrs:
        return None
    v = node.attrs[key]
    if isinstance(v, bytes):
        return v.decode("utf-8")
    return str(v)


# ---------------------------------------------------------------------------
# Table validation — every way the tables can fail to describe the tensor
# ---------------------------------------------------------------------------

def _validate(tables: QirrTables, n_q_on_disk: int, n_mu: int) -> str:
    """Check the tables against the wedge, and return the shape verdict.

    Everything here is a way a stored tensor becomes unrecoverable while
    looking fine, so each check names what it found rather than asserting.
    Returns ``"ibz"`` or ``"full"`` — the SHAPE verdict, which is the
    primary discriminant the attr is later cross-checked against.
    """
    t = tables.canonical()
    if t.irr_idx_q.ndim != 1 or t.sym_idx_q.ndim != 1:
        raise ValueError(
            f"qirr_store: irr_idx_q and sym_idx_q must be 1-D; got "
            f"{t.irr_idx_q.shape} and {t.sym_idx_q.shape}")
    n_q_full = int(t.irr_idx_q.shape[0])
    if int(t.sym_idx_q.shape[0]) != n_q_full:
        raise ValueError(
            f"qirr_store: irr_idx_q has {n_q_full} rows but sym_idx_q has "
            f"{int(t.sym_idx_q.shape[0])}; they index the same q axis.")
    if t.q_irr_frac.ndim != 2 or t.q_irr_frac.shape[1] != 3:
        raise ValueError(
            f"qirr_store: q_irr_frac must be (n_q_ibz, 3); got "
            f"{t.q_irr_frac.shape}")
    n_q_ibz = int(t.q_irr_frac.shape[0])
    if n_q_on_disk != n_q_ibz:
        raise ValueError(
            f"qirr_store: the stored tensor has {n_q_on_disk} q rows but "
            f"q_irr_frac describes {n_q_ibz} IBZ q-points.  The tables do "
            f"not describe this tensor.")
    if n_q_ibz > n_q_full:
        raise ValueError(
            f"qirr_store: {n_q_ibz} IBZ q-points is more than the "
            f"{n_q_full} full-BZ rows irr_idx_q covers.")
    seen = set(int(v) for v in t.irr_idx_q.tolist())
    if seen != set(range(n_q_ibz)):
        missing = sorted(set(range(n_q_ibz)) - seen)
        extra = sorted(v for v in seen if not 0 <= v < n_q_ibz)
        raise ValueError(
            f"qirr_store: irr_idx_q must reference every stored IBZ row "
            f"exactly once as a set and nothing else.  Unreferenced rows: "
            f"{missing[:8]}; out-of-range indices: {extra[:8]}.  An "
            f"out-of-range parent gathers silently under "
            f"promise_in_bounds; an unreferenced row is a wedge row no q "
            f"can ever recover.")
    if t.sym_perm.ndim != 2:
        raise ValueError(
            f"qirr_store: sym_perm must be (n_sym_rows, n_mu); got "
            f"{t.sym_perm.shape}")
    n_sym_rows = int(t.sym_perm.shape[0])
    if int(t.sym_perm.shape[1]) != n_mu:
        raise ValueError(
            f"qirr_store: sym_perm covers {int(t.sym_perm.shape[1])} "
            f"centroids but the tensor's μ extent is {n_mu}; tables and "
            f"tensor must share one (padded) extent.")
    if t.L_table.shape != (n_sym_rows, n_mu, 3):
        raise ValueError(
            f"qirr_store: L_table must be {(n_sym_rows, n_mu, 3)}; got "
            f"{t.L_table.shape}")
    max_sym = int(t.sym_idx_q.max()) if t.sym_idx_q.size else -1
    if max_sym >= n_sym_rows:
        raise ValueError(
            f"qirr_store: sym_idx_q reaches row {max_sym} but sym_perm has "
            f"only {n_sym_rows}.  Build sym_perm via "
            f"``centroid_source_map_and_wrap(..., extend_trs=True)`` so it "
            f"covers the TRS-augmented half.")
    n_spatial = int(t.n_sym_spatial)
    if n_spatial <= 0:
        raise ValueError(
            f"qirr_store: n_sym_spatial must be positive; got {n_spatial}")
    if max_sym >= n_spatial and n_sym_rows != 2 * n_spatial:
        raise ValueError(
            f"qirr_store: sym_idx_q uses TRS-augmented rows (max={max_sym} "
            f"≥ n_sym_spatial={n_spatial}) but sym_perm has {n_sym_rows} "
            f"rows ≠ 2·n_sym_spatial.  Time reversal keeps r fixed, so the "
            f"augmented half must duplicate the spatial half.")
    # THE SHAPE VERDICT.  Same rule zeta_loader has shipped: the wedge is
    # the whole BZ exactly when there is nothing to unfold, and in that
    # degenerate case (ntran=1, or a q-grid the group does not reduce) the
    # honest label is "full" for reader and writer alike.
    return "ibz" if n_q_ibz < n_q_full else "full"


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_qirr_tensor(
    dest,
    name,
    X_ibz,
    *,
    tables: QirrTables,
    closure_verdict: CentroidClosureVerdict | None = None,
    centroids_frac=None,
    sym_matrices=None,
    tnp=None,
    tau=None,
    fft_grid=None,
    tol: float = CLOSURE_TOL_DEFAULT,
    n_rmu_logical: int | None = None,
    provenance: dict | None = None,
    data_ready: bool = True,
    mode: str = "a",
):
    """Write ``X_ibz`` on the q wedge, with the tables that unfold it.

    REFUSES on a non-closed centroid set.  Either hand it a verdict from
    :func:`symmetry_maps.verify_centroid_orbit_closure` or hand it the
    centroid set and let it take one; there is no third path and no way to
    write without the question having been asked.

    Parameters
    ----------
    dest
        An open ``h5py.File``/``Group``, or a path.  A path is opened in
        ``mode`` (default ``"a"``) and closed again.
    name
        Dataset name, e.g. ``"W0_qmunu"``.  The tables land in the sibling
        group ``name + "__qirr"``.
    X_ibz
        ``(n_q_ibz, n_mu, n_mu)`` — the PRE-UNFOLD block, the array the
        producer holds one statement before it calls ``unfold_isdf_operator``.
        Any array with ``__array__``; it is materialised on the host here.
    tables
        :class:`QirrTables`.  Written into the file, hashed, and re-checked
        on read.
    closure_verdict
        A :class:`CentroidClosureVerdict`.  Mutually exclusive with
        ``centroids_frac``; exactly one is required.
    centroids_frac, sym_matrices, tnp, tau, fft_grid, tol
        Passed straight to :func:`verify_centroid_orbit_closure` when
        ``centroids_frac`` is given.  ``tnp``/``tau`` keep that function's
        exclusive-keyword contract, so the 2π cannot be applied twice or
        not at all on this path either.
    n_rmu_logical
        The LOGICAL centroid count, when ``X_ibz`` and ``tables`` carry
        the producer's μ pad.  The pad width is
        ``padded_mu_extent(n_rmu, device_count())`` — a
        DEVICE-COUNT-DEPENDENT quantity, which SHARDING_RULES §2 forbids
        in a restart artifact because a file written on four ranks must
        read on eight.  Given this, the writer ASSERTS the tail really is
        a pad (identity on the permutation, zero on the wrap, zero on the
        tensor) and then strips it, stamping the logical extent so a
        reader can re-apply its OWN.  Omit it when nothing is padded.
    provenance
        Extra attrs to stamp, kin_ion.h5 style (the WFN checksum, the deck
        name, whatever the caller can prove).  Keys are prefixed
        ``prov_`` so they can never collide with the format's own.
    data_ready
        Stamped as ``qirr_data_ready``.  ``False`` writes the metadata for
        a tensor whose data is a placeholder — see
        :func:`allocate_qirr_placeholder`, which is the intended way to do
        that.

    Returns
    -------
    QirrHeader
        What was stamped, without a second open.
    """
    if (closure_verdict is None) == (centroids_frac is None):
        raise ValueError(
            "write_qirr_tensor: pass exactly one of closure_verdict= (from "
            "verify_centroid_orbit_closure) or centroids_frac= (with "
            "sym_matrices= and tnp=/tau=, to take the verdict here).  "
            "Writing a q_irr tensor without the closure question having "
            "been asked is the one thing this writer exists to prevent: a "
            "wedge stored against a non-closed set has no permutation α "
            "and is unrecoverable, silently.")
    if closure_verdict is None:
        if sym_matrices is None:
            raise ValueError(
                "write_qirr_tensor: centroids_frac= needs sym_matrices= "
                "beside it.")
        closure_verdict = verify_centroid_orbit_closure(
            centroids_frac, sym_matrices, tnp=tnp, tau=tau,
            fft_grid=fft_grid, tol=tol)
    elif any(v is not None for v in (sym_matrices, tnp, tau, fft_grid)):
        raise ValueError(
            "write_qirr_tensor: closure_verdict= already carries the "
            "answer; passing sym_matrices/tnp/tau/fft_grid beside it would "
            "describe a measurement that is not the one being stamped.")
    if not isinstance(closure_verdict, CentroidClosureVerdict):
        raise TypeError(
            f"write_qirr_tensor: closure_verdict must be a "
            f"CentroidClosureVerdict, got "
            f"{type(closure_verdict).__name__}")

    # THE REFUSAL.  Loud, with every offending op and its residual, and
    # never a warning: see the module docstring.
    closure_verdict.raise_if_not_closed(
        f"write_qirr_tensor({name!r}) refuses q_irr storage")

    X = np.asarray(X_ibz)
    if X.ndim != 3 or X.shape[1] != X.shape[2]:
        raise ValueError(
            f"qirr_store: expected a (n_q_ibz, n_mu, n_mu) tensor; got "
            f"{X.shape}")
    # THE μ PAD COMES OFF HERE, TENSOR AND TABLES TOGETHER.  Stripping one
    # and not the other is the failure this ordering forbids: the two
    # extents are cross-checked by ``_validate`` two statements down, so a
    # half-strip REFUSES rather than writing a file whose tables address
    # more centroids than its tensor has.
    if n_rmu_logical is not None:
        n_log = int(n_rmu_logical)
        n_pad = int(X.shape[-1])
        if n_log > n_pad:
            raise ValueError(
                f"qirr_store: n_rmu_logical={n_log} exceeds the tensor's "
                f"μ extent {n_pad}.")
        if n_log < n_pad:
            worst = max(
                float(np.abs(X[:, n_log:, :]).max(initial=0.0)),
                float(np.abs(X[:, :, n_log:]).max(initial=0.0)))
            if worst != 0.0:
                raise ValueError(
                    f"qirr_store: the tensor's μ rows/columns past "
                    f"{n_log} are NOT exactly zero (worst |value| "
                    f"{worst:.3e}), so they are not the pad they were "
                    f"declared to be.  Pad rows are zeros by "
                    f"construction; anything else here is real data that "
                    f"stripping would delete.")
            X = np.ascontiguousarray(X[:, :n_log, :n_log])
        tables = tables.logical(n_log)

    # CREATE, THEN STAMP — and the stamp is :func:`stamp_qirr_tensor`, not a
    # copy of it.  The producer writes its wedge through SlabIO and stamps
    # separately; if that stamp and this one were two code paths they would
    # be two claims about what the file says, differing on the day one of
    # them gained an attr.  The tables are already logical here, so the
    # ``n_rmu_logical=None`` below is not a lost check — the strip and its
    # three invariants ran above, against the TENSOR that only this path
    # holds.
    with _Dest(dest, mode) as grp:
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=X)
        return stamp_qirr_tensor(
            grp, name, tables=tables, closure_verdict=closure_verdict,
            provenance=provenance, data_ready=data_ready)


def stamp_qirr_tensor(
    dest,
    name,
    *,
    tables: QirrTables,
    closure_verdict: CentroidClosureVerdict,
    n_rmu_logical: int | None = None,
    provenance: dict | None = None,
    data_ready: bool = True,
    mode: str = "a",
):
    """Make an ALREADY-WRITTEN dataset a q_irr tensor: tables + attrs only.

    WHY THIS EXISTS BESIDE :func:`write_qirr_tensor`.  The GW producer does
    not write its restart tensors with h5py — it writes them through
    ``file_io.slab_io.SlabIO``, where every rank contributes its own (μ, ν)
    hyperslab and no rank ever holds the whole array.  A format function
    that insists on creating the dataset itself would force that write back
    through one process, which is a bigger regression than the file size it
    was saving.  So the producer writes the wedge with the machinery it
    already has, and this stamps it.

    THE SPLIT IS EXACTLY ONE THING: who calls ``create_dataset``.
    Everything else — the closure refusal, the μ-pad strip and its three
    invariants, the table validation, the digest, the attrs, the provenance
    — is this function, and :func:`write_qirr_tensor` calls it.  One
    implementation of the stamp, because a second one is a second answer to
    "what does this file claim".

    THE TENSOR'S PAD IS ALREADY OFF, and that is the one asymmetry with
    :func:`write_qirr_tensor`.  ``SlabIO`` clips the μ axes to the logical
    extent on the way out (``tagged_arrays._mu_logical_shape``), so the
    dataset on disk is logical before this runs and there is no tensor tail
    left to check.  The TABLES still carry the producer's pad, so
    ``n_rmu_logical`` strips them — with the identity-tail / zero-wrap
    assertions intact, since those are the ones that would catch a table
    that addresses real centroids past the declared extent.  The dataset's
    own μ extent is asserted against ``n_rmu_logical`` instead: that is the
    check that replaces the tensor-tail one, and it is what makes a
    half-clipped write refuse rather than produce a file whose tables and
    tensor describe different centroid sets.
    """
    if not isinstance(closure_verdict, CentroidClosureVerdict):
        raise TypeError(
            f"stamp_qirr_tensor: closure_verdict must be a "
            f"CentroidClosureVerdict, got {type(closure_verdict).__name__}")
    closure_verdict.raise_if_not_closed(
        f"stamp_qirr_tensor({name!r}) refuses q_irr storage")
    if n_rmu_logical is not None:
        tables = tables.logical(int(n_rmu_logical))
    can = tables.canonical()

    with _Dest(dest, mode) as grp:
        if name not in grp:
            raise KeyError(
                f"qirr_store: {name!r} is not in this file.  "
                f"stamp_qirr_tensor stamps a dataset the caller has already "
                f"written; use write_qirr_tensor to create one.")
        ds = grp[name]
        if ds.ndim != 3 or int(ds.shape[1]) != int(ds.shape[2]):
            raise ValueError(
                f"qirr_store: {name!r} is {ds.shape}; expected "
                f"(n_q_ibz, n_mu, n_mu).")
        n_mu_on_disk = int(ds.shape[-1])
        if n_rmu_logical is not None and n_mu_on_disk != int(n_rmu_logical):
            raise ValueError(
                f"qirr_store: {name!r} has μ extent {n_mu_on_disk} on disk "
                f"but n_rmu_logical={int(n_rmu_logical)}.  The tables are "
                f"stripped to the logical extent and the tensor must "
                f"already be at it; a disagreement here means the tensor "
                f"and its tables would describe different centroid sets.")
        q_storage = _validate(can, int(ds.shape[0]), n_mu_on_disk)
        table_hash = can.digest()

        tgrp_name = name + QIRR_TABLE_SUFFIX
        if tgrp_name in grp:
            del grp[tgrp_name]
        tgrp = grp.create_group(tgrp_name)
        for key in _TABLE_ORDER:
            tgrp.create_dataset(key, data=getattr(can, key))
        tgrp.attrs["n_sym_spatial"] = np.int64(can.n_sym_spatial)
        tgrp.attrs["table_hash"] = table_hash

        ds.attrs[_VERSION_ATTR] = np.int64(QIRR_FORMAT_VERSION)
        ds.attrs["q_storage"] = q_storage
        ds.attrs["qirr_n_q_full"] = np.int64(can.n_q_full)
        ds.attrs["qirr_n_rmu_logical"] = np.int64(can.n_mu)
        ds.attrs["qirr_table_hash"] = table_hash
        ds.attrs["qirr_centroid_hash"] = closure_verdict.centroid_hash
        ds.attrs["qirr_closure_verdict"] = closure_verdict.as_attr()
        ds.attrs["qirr_closure_worst_residual"] = np.float64(
            closure_verdict.worst_residual)
        ds.attrs["qirr_closure_tol"] = np.float64(closure_verdict.tol)
        ds.attrs["qirr_data_ready"] = bool(data_ready)
        ds.attrs["qirr_generator_commit"] = _generator_commit()
        ds.attrs["qirr_written_utc"] = datetime.datetime.now(
            datetime.timezone.utc).replace(microsecond=0).isoformat()
        ds.attrs["qirr_writer"] = "symmetry_maps.qirr_store"
        for key, val in (provenance or {}).items():
            ds.attrs["prov_" + str(key)] = val
        return _header_from(ds, tgrp, can)


def allocate_qirr_placeholder(dest, name, shape, *, tables, dtype=None,
                              mode="a", **kw):
    """Create a q_irr dataset with NO DATA and ``qirr_data_ready=False``.

    ``gw_init`` allocates a full-size zero ``W0`` before the screening that
    fills it exists, so a restart file of exactly the right shape and full
    of zeros is a state the pipeline genuinely reaches.  This is that state
    in the q_irr format, and it exists here so the reader's refusal has
    something real to refuse rather than a hand-forged file: the April BSE
    incident produced a plausible excitonic spectrum out of exactly this.

    Takes the same closure arguments as :func:`write_qirr_tensor` and
    refuses the same way — a placeholder for a tensor that could never be
    unfolded is still unrecoverable.
    """
    dtype = np.complex128 if dtype is None else dtype
    shape = tuple(int(s) for s in shape)
    if len(shape) != 3:
        raise ValueError(
            f"allocate_qirr_placeholder: expected (n_q_ibz, n_mu, n_mu); "
            f"got {shape}")
    # h5py allocates lazily, so this costs metadata rather than the tensor;
    # the value read back is zeros, which is precisely the hazard.
    return write_qirr_tensor(dest, name, np.zeros(shape, dtype=dtype),
                             tables=tables, data_ready=False, mode=mode,
                             **kw)


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def _header_from(ds, tgrp, tables):
    prov = {k[len("prov_"):]: v for k, v in ds.attrs.items()
            if str(k).startswith("prov_")}
    for k in ("qirr_generator_commit", "qirr_written_utc", "qirr_writer"):
        if k in ds.attrs:
            prov[k] = _attr_str(ds, k)
    ready = ds.attrs.get("qirr_data_ready", None)
    return QirrHeader(
        q_storage=_attr_str(ds, "q_storage"),
        format_version=int(ds.attrs[_VERSION_ATTR]),
        n_q_on_disk=int(ds.shape[0]),
        n_q_full=int(ds.attrs["qirr_n_q_full"]),
        n_mu=int(ds.shape[-1]),
        n_rmu_logical=int(ds.attrs.get("qirr_n_rmu_logical",
                                       int(ds.shape[-1]))),
        centroid_hash=_attr_str(ds, "qirr_centroid_hash"),
        table_hash=_attr_str(ds, "qirr_table_hash"),
        closure_verdict=_attr_str(ds, "qirr_closure_verdict"),
        closure_worst_residual=float(
            ds.attrs["qirr_closure_worst_residual"]),
        closure_tol=float(ds.attrs["qirr_closure_tol"]),
        data_ready=None if ready is None else bool(ready),
        provenance=prov,
    )


def dataset_q_storage(ds) -> str:
    """``"ibz"`` or ``"full"`` for an OPEN h5py dataset, reading no data.

    THE PROBE A CONSUMER ASKS BEFORE IT DECIDES HOW TO READ.  Every restart
    reader in the tree already holds an open dataset when it has to choose a
    path, and the choice must cost attrs rather than a gigabyte: a consumer
    that had to call :func:`read_tensor` to find out whether it wanted
    :func:`read_tensor` would materialise the tensor to answer.

    THE SAME THREE-WAY RULE AS :func:`read_tensor`, and it is here rather
    than copied into each caller because a second implementation of "is this
    a q_irr file" is how a reader ends up disagreeing with the format about
    what it is holding:

    * no ``qirr_*`` attrs at all -> ``"full"``.  Every restart file written
      before this format existed, unchanged;
    * a full stamp -> whatever ``q_storage`` says;
    * a PARTIAL stamp -> refuse.  The missing half is exactly the half that
      would say whether the shape means what it looks like.

    This reads the ATTR and not the shape, deliberately and unlike
    :func:`read_tensor`, which cross-checks the two.  The cross-check needs
    the tables; this is the cheap question asked before the tables are
    opened, and the expensive answer still refuses on a disagreement.
    """
    present = [a for a in _OWNED_ATTRS if a in ds.attrs]
    if _VERSION_ATTR not in ds.attrs:
        if present:
            raise ValueError(
                f"qirr_store: {ds.name!r} carries q_irr attrs {present} but "
                f"no {_VERSION_ATTR!r}.  'No attrs' is read as "
                f"q_storage='full' for backward compatibility; a PARTIAL "
                f"stamp is not.")
        return "full"
    stamped = _attr_str(ds, "q_storage")
    if stamped not in ("ibz", "full"):
        raise ValueError(
            f"qirr_store: {ds.name!r} stamps q_storage={stamped!r}, which is "
            f"neither 'ibz' nor 'full'.")
    return stamped


def read_tables(src, name, *, mode="r") -> QirrTables:
    """The stored unfold tables for ``name``, as written.

    Separate from :func:`read_tensor` because a caller that wants to
    compare two files' tables, or to unfold on its own schedule, should not
    have to materialise a gigabyte to do it.
    """
    with _Dest(src, mode) as grp:
        tgrp_name = name + QIRR_TABLE_SUFFIX
        if tgrp_name not in grp:
            raise ValueError(
                f"qirr_store: {name!r} carries no {tgrp_name!r} table "
                f"group.  A q_irr tensor whose reconstruction tables live "
                f"anywhere but beside it is a tensor that silently decays "
                f"when anything upstream is regenerated.")
        tgrp = grp[tgrp_name]
        missing = [k for k in _TABLE_ORDER if k not in tgrp]
        if missing:
            raise ValueError(
                f"qirr_store: {tgrp_name!r} is missing {missing}; the "
                f"tensor cannot be unfolded.")
        return QirrTables(
            irr_idx_q=tgrp["irr_idx_q"][()],
            sym_idx_q=tgrp["sym_idx_q"][()],
            q_irr_frac=tgrp["q_irr_frac"][()],
            sym_perm=tgrp["sym_perm"][()],
            L_table=tgrp["L_table"][()],
            n_sym_spatial=int(tgrp.attrs["n_sym_spatial"]),
        )


def read_tensor(
    src,
    name,
    *,
    mesh_xy=None,
    unfold: bool = True,
    n_mu_padded: int | None = None,
    require_persisted: bool = True,
    expect_centroid_hash: str | None = None,
    expect_table_hash: str | None = None,
    mode: str = "r",
):
    """Read ``name``, unfolding to the full BZ when it is stored on a wedge.

    Returns ``(array, header)``.  The header is not optional decoration:
    ``header.data_ready`` is the persisted flag, ``header.q_storage`` says
    whether an unfold ran, and ``header.closure_verdict`` carries the
    write-time measurement the tensor's recoverability rests on.

    THE DISCRIMINANT IS THE SHAPE.  The stored q extent is compared against
    the ``n_q_full`` the tables carry; that decides ibz-versus-full.  The
    ``q_storage`` attr is then asserted against that verdict and a
    disagreement REFUSES, because a stamped-and-wrong attr is exactly what
    a cross-check is for.

    NO ATTRS AT ALL MEANS FULL.  A dataset carrying no ``qirr_*`` attrs is
    a restart file written before this format existed; it is returned
    unchanged and none of the checks below apply.  A dataset carrying SOME
    of them but no version attr is refused rather than read that way — a
    file that was half-stamped is a file whose q_storage claim nobody
    should act on.

    Parameters
    ----------
    mesh_xy
        Required only when an unfold actually has to run.
    unfold
        ``False`` returns the stored wedge itself, with the same header.
        For a caller that wants to hold the small array.
    n_mu_padded
        Re-apply a μ pad of THIS width — the reader's own, from
        ``padded_mu_extent(n_rmu, device_count())`` — before unfolding.
        The file stores the LOGICAL extent (SHARDING_RULES §2: a restart
        artifact may not carry a device-count-dependent quantity), so a
        consumer that wants the padded in-memory layout asks for it here
        rather than finding the writer's pad and hoping it matches.  The
        pad is reconstructed, never stored: identity tail on the
        permutation, zero tail on the wrap and on the tensor.
    require_persisted
        Refuse when ``qirr_data_ready`` is False.  Default True: a
        zero-filled placeholder must never read as data.  Legacy files
        carry no flag and are unaffected.
    expect_centroid_hash, expect_table_hash
        Refuse unless the file's stamp matches.  This is how a consumer
        says "the centroid set I am about to build ζ from is the one this
        tensor was written against".
    """
    with _Dest(src, mode) as grp:
        if name not in grp:
            raise KeyError(f"qirr_store: {name!r} is not in this file")
        ds = grp[name]
        present = [a for a in _OWNED_ATTRS if a in ds.attrs]

        # ---- the backward-compatible path: no attrs means full BZ -------
        if _VERSION_ATTR not in ds.attrs:
            if present:
                raise ValueError(
                    f"qirr_store: {name!r} carries q_irr attrs {present} "
                    f"but no {_VERSION_ATTR!r}.  'No attrs' is read as "
                    f"q_storage='full' for backward compatibility; a "
                    f"PARTIAL stamp is not, because the missing half is "
                    f"exactly what would say whether the shape means what "
                    f"it looks like.")
            return ds[()], QirrHeader(
                q_storage="full", format_version=None,
                n_q_on_disk=int(ds.shape[0]), n_q_full=None,
                n_mu=int(ds.shape[-1]) if ds.ndim >= 1 else None,
                n_rmu_logical=None,
                centroid_hash=None, table_hash=None, closure_verdict=None,
                closure_worst_residual=None, closure_tol=None,
                data_ready=None, provenance={})

        version = int(ds.attrs[_VERSION_ATTR])
        if version != QIRR_FORMAT_VERSION:
            raise ValueError(
                f"qirr_store: {name!r} is format version {version}; this "
                f"reader is version {QIRR_FORMAT_VERSION}.  Refusing "
                f"rather than guessing: a reader that reads an unknown "
                f"version best-effort returns wrong numbers on the day the "
                f"layout changes.")

        tables = read_tables(grp, name)
        can = tables.canonical()
        recomputed = can.digest()
        stamped = _attr_str(ds, "qirr_table_hash")
        if stamped != recomputed:
            raise ValueError(
                f"qirr_store: {name!r} table hash mismatch.  Stamped "
                f"{stamped}, tables on disk hash to {recomputed}.  The "
                f"unfold tables are not the ones this tensor was written "
                f"against, so every q it reconstructs would be a "
                f"permutation of the wrong centroids.")
        if expect_table_hash is not None and expect_table_hash != recomputed:
            raise ValueError(
                f"qirr_store: {name!r} table hash {recomputed} does not "
                f"match the expected {expect_table_hash}.")
        cent = _attr_str(ds, "qirr_centroid_hash")
        if expect_centroid_hash is not None and expect_centroid_hash != cent:
            raise ValueError(
                f"qirr_store: {name!r} was written against centroid set "
                f"{cent}, the caller expected {expect_centroid_hash}.  "
                f"sym_perm addresses centroids by row position, so a "
                f"different set is a different permutation.")

        n_q_on_disk = int(ds.shape[0])
        n_mu = int(ds.shape[-1])
        shape_says = _validate(can, n_q_on_disk, n_mu)
        attr_says = _attr_str(ds, "q_storage")
        if attr_says != shape_says:
            raise ValueError(
                f"qirr_store: {name!r} shape says q_storage={shape_says!r} "
                f"({n_q_on_disk} q rows on disk against {can.n_q_full} "
                f"full-BZ rows in the tables) but the attr says "
                f"{attr_says!r}.  The SHAPE is the primary discriminant — "
                f"it has been ζ's since before this format existed — and "
                f"the attr is its cross-check, so a disagreement is a "
                f"refusal and not a preference.")
        stored_full = int(ds.attrs["qirr_n_q_full"])
        if stored_full != can.n_q_full:
            raise ValueError(
                f"qirr_store: {name!r} stamps n_q_full={stored_full} but "
                f"its irr_idx_q table has {can.n_q_full} rows.")

        header = _header_from(ds, grp[name + QIRR_TABLE_SUFFIX], can)
        if require_persisted and header.data_ready is False:
            raise ValueError(
                f"qirr_store: {name!r} is present and correctly shaped but "
                f"qirr_data_ready is False — it is a PLACEHOLDER, not "
                f"data.  Reading it would hand a consumer a tensor of "
                f"zeros that passes every shape check, which is the "
                f"mechanism behind the all-zero-screening incident: a "
                f"plausible excitonic spectrum out of a W that was never "
                f"written.  Pass require_persisted=False only to inspect "
                f"the placeholder deliberately.")

        raw = ds[()]

    if not unfold or shape_says == "full":
        return raw, header

    if n_mu_padded is not None and int(n_mu_padded) != int(can.n_mu):
        pad = int(n_mu_padded) - int(can.n_mu)
        if pad < 0:
            raise ValueError(
                f"qirr_store: {name!r} stores {can.n_mu} logical centroids "
                f"and the caller asked to pad DOWN to {n_mu_padded}.  The "
                f"pad only ever grows the extent; a smaller request means "
                f"the caller and the file disagree about the centroid set.")
        raw = np.pad(np.asarray(raw), ((0, 0), (0, pad), (0, pad)))
        can = can.padded(int(n_mu_padded))

    if mesh_xy is None:
        raise ValueError(
            f"qirr_store: {name!r} is stored on the q wedge "
            f"({n_q_on_disk} of {can.n_q_full} q) and unfolding it needs a "
            f"mesh; pass mesh_xy= or unfold=False to take the wedge.")
    import jax.numpy as jnp
    full = unfold_isdf_operator(
        jnp.asarray(raw),
        irr_idx=can.irr_idx_q,
        sym_idx=can.sym_idx_q,
        sym_perm=can.sym_perm,
        L_table=can.L_table,
        q_irr_frac=can.q_irr_frac,
        mesh_xy=mesh_xy,
        n_sym_spatial=int(can.n_sym_spatial),
    )
    return full, header
