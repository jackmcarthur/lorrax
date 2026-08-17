"""The HDF5 operation journal — every open, write, read and close, on disk.

WHY THIS MODULE EXISTS.  Three HDF5 failure signatures on the metallic
MPA-QSGW driver (segfault after ``SlabIO.close``; a phantom "already open
for write" on a fresh store; a garbage ``offset_base`` on a no-offset
read) are all *native* deaths or *native* refusals: by the time anything
Python can see them, the state that would explain them is gone.  The
static audit that named their mechanisms
(``reports/metal_mpa_plan_2026-08-15/SLAB_IO_ROOT_CAUSE_AUDIT.md`` §C)
asked for the one instrument that survives a SIGSEGV — a line-buffered
per-rank text file, written at the Python file_io seam, current to the
last line when the process dies.

WHAT IS JOURNALED, AND WHERE FROM.  Three existing choke points, no new
seams:

* :mod:`file_io.hdf5_owner` — ``note_open`` / ``note_close``.  EVERY
  HDF5 open in this process is declared there, through either stack, so
  this is the one place that sees h5py and the FFI in one stream.
* :class:`file_io.slab_io.SlabIO` — ``__init__`` / ``close`` /
  ``create_dataset`` / ``write_slab`` / ``read_slab`` / ``read_slabs`` /
  ``write_attr`` / ``stamp_dataset_attrs``: the caller-level request.
* ``file_io._slab_io_ffi`` — the lifecycle calls themselves
  (``open_file``, ``phdf5_ensure_dataset``, ``phdf5_open_dataset_ro``,
  ``close_file``) and the two serial-h5py touches on an FFI-driven path
  (``_introspect_dataset``'s metadata read, ``close``'s deferred-attr
  reopen).

So ONE slab read or write is one line, and one SlabIO file open is three
(the registry's claim, the FFI's issue of ``H5Fopen``, and SlabIO's
completion line carrying the ctx handle).  That is deliberate: opens are
where all three signatures live, and the three lines are the three
different facts about one open.

WHEN THE LINE IS WRITTEN.  **At issue, before the call**, with everything
known at that moment — which is what makes the file a black box: a
process that dies inside HDF5 leaves its last line naming the op that
killed it.  The one field that cannot be known at issue is the handle of
an ``open`` (the call has not returned it yet), so an ``open`` line
carries ``handle=-`` and the handle appears on the completion line the
SlabIO seam writes next.  An op that RAISES appends a second line with
``rc=refused:…`` and dumps the crash ring.

LINE FORMAT (frozen; fixed key order, one line per op)::

    t=<monotonic.6f> rank=<R> stack={h5py|ffi} op={open|close|create|read|
    write|attr_r|attr_w} path=<abs> handle=<id-or-ctxptr> ds=<name|->
    off=<t> cnt=<t> mode=<r|w|a|-> owner=<registry-verdict>
    rc={ok|refused:<first 40 chars>}

``owner`` is :func:`file_io.hdf5_owner.live_verdict` — which stacks hold
a live handle on this path at this instant (``free``, ``ffi:1w``,
``h5py+ffi:2r``) — and NOT the outcome of the registry's check; the
outcome is ``rc``.  Every value is whitespace-free except ``rc``'s
message tail, which is last on the line.

Two ``handle`` spellings are not the FFI ctx pointer and say so by their
shape: ``tokN`` on a registry line is the registry's own open token (it
pairs that open with its close), and on a ``stack=h5py`` line written
from ``_slab_io_ffi`` the handle is the FFI **ctx that is live on the
same path while h5py touches it** — which is the concurrency the line
exists to record, and is corroborated by ``owner=h5py+ffi:2r``.

TOGGLE.  ``LORRAX_H5_JOURNAL``: ``1`` (default, ON), ``0`` off, ``sync``
= fsync after every line, for segfault-grade capture where even the
kernel's page cache is suspect.  ``LORRAX_H5_JOURNAL_DIR`` moves the
output; default is the process's working directory, which is the run
directory for every LORRAX driver launch.  Files:
``h5_journal.rank<R>.log`` and ``h5_journal_crash.rank<R>.txt``.

COST.  ~1-2 µs per formatted, line-buffered line; the measured op rate is
~1030 HDF5 ops per SC iteration and this journal writes ~1 line per slab
op and 3 per file open, i.e. a few thousand lines against a ~280 s
iteration — order 1e-5.  That is why the default is ON.  ``sync`` costs
~1 ms/line and is opt-in only.

THE JOURNAL NEVER TAKES A RUN DOWN.  An I/O error on the journal file
disables the journal with one warning and lets the caller proceed: a log
that kills a 40-node job because its directory went read-only is worse
than no log.  A bad ``op``/``stack`` spelling DOES refuse — those are
in-tree constants, covered by ``tests/test_h5_journal.py``, and a
mis-spelled op in a log is a fact nobody can recover later.
"""
from __future__ import annotations

import atexit
import contextlib
import os
import threading
import time
from collections import deque

__all__ = [
    "DIR_ENV",
    "MODE_ENV",
    "MODE_OFF",
    "MODE_ON",
    "MODE_SYNC",
    "OPS",
    "STACKS",
    "crash_dump",
    "crash_path",
    "enabled",
    "fail",
    "format_line",
    "journal_path",
    "mode",
    "mode_is_sync",
    "op_scope",
    "record",
    "reset_for_test",
    "ring",
]

#: The toggle.  ``1``/on (default), ``0``/off, ``sync``.
MODE_ENV = "LORRAX_H5_JOURNAL"
#: Where the per-rank files land.  Default: the working directory.
DIR_ENV = "LORRAX_H5_JOURNAL_DIR"
# THE READS BELOW SPELL THESE NAMES AS LITERALS, and that duplication is
# deliberate.  ``tools/env_audit.py`` and ``tests/test_env_registry.py``
# both resolve only literal arguments, so a read through a module
# constant is INVISIBLE to the gate that requires a docs/dev/env_vars.md
# row — which is exactly how the sibling ``LORRAX_HDF5_ONE_OWNER`` came
# to have no row for a week.  The constants stay because tests and error
# messages need the name; the suite pins them to the literals by setting
# the environment through ``J.MODE_ENV`` / ``J.DIR_ENV`` and asserting
# the behaviour changes.

MODE_OFF = "off"
MODE_ON = "on"
MODE_SYNC = "sync"

_ON_TOKENS = frozenset({"1", "true", "yes", "on", ""})
_OFF_TOKENS = frozenset({"0", "false", "no", "off"})
_SYNC_TOKENS = frozenset({"sync", "fsync"})

#: The op vocabulary.  Frozen with the line format.
OPS = frozenset({"open", "close", "create", "read", "write",
                 "attr_r", "attr_w"})
#: The two HDF5 library instances this process can map.  Spelled the same
#: as :data:`file_io.hdf5_owner.STACK_H5PY` / ``STACK_FFI`` — the registry
#: owns the concept, this module only prints it.
STACKS = frozenset({"h5py", "ffi"})

#: The crash ring depth (audit §C: "last 256 lines").
RING_LINES = 256
#: How much of a refusal message a line carries (audit §C).
REFUSAL_CHARS = 40

_LOCK = threading.RLock()
_RING: "deque[str]" = deque(maxlen=RING_LINES)
_STATE = {
    "stream": None,       # the open, line-buffered log file
    "log_path": None,     # its path, for the crash-dump header
    "rank_of_file": None,  # the rank that path was named for
    "rank": None,         # resolved lazily; see _rank()
    "world": 1,
    "tries": 0,           # rank resolutions so far; see RANK_RESOLVE_TRIES
    "abnormal": False,    # a refusal or exception has been journaled
    "dumped": 0,          # crash dumps written so far
    "broken": False,      # journal I/O failed; disabled for this process
    "atexit": False,
}
# NEVER JOURNAL THE JOURNAL (audit §C implementation note).  One module
# flag, not a lock: the reentrant caller is always this thread.
_INSIDE = threading.local()


# ---------------------------------------------------------------------------
# The toggle
# ---------------------------------------------------------------------------

def mode() -> str:
    """``off`` / ``on`` / ``sync``, from :data:`MODE_ENV`.

    Read on every call rather than cached: the variable is a debug
    switch, and a cached parse is a switch that stops working the moment
    a test or a driver flips it.
    """
    got = str(os.environ.get("LORRAX_H5_JOURNAL", "1")).strip().lower()
    if got in _OFF_TOKENS:
        return MODE_OFF
    if got in _SYNC_TOKENS:
        return MODE_SYNC
    if got in _ON_TOKENS:
        return MODE_ON
    raise ValueError(
        f"{MODE_ENV}={got!r} is not one of 1/true/yes/on (journal on, the "
        f"default), 0/false/no/off (journal off), or sync (fsync every "
        f"line — segfault-grade capture, ~1 ms/line).  The journal is the "
        f"black box for the native HDF5 failures in "
        f"SLAB_IO_ROOT_CAUSE_AUDIT.md; refusing an unrecognised token is "
        f"how you find out you spelled it wrong, instead of losing the "
        f"evidence for the run you were trying to instrument.")


#: :func:`record` takes a ``mode=`` field — the HDF5 OPEN mode, a
#: different thing that English calls the same word — which shadows
#: :func:`mode` inside its body.  One alias beats renaming a field of the
#: frozen line format.
_toggle = mode


def enabled() -> bool:
    """True when the journal is on (either ``on`` or ``sync``)."""
    return (not _STATE["broken"]) and mode() != MODE_OFF


# ---------------------------------------------------------------------------
# Rank and paths
# ---------------------------------------------------------------------------

#: How many journaled ops may re-ask for the rank while the world still
#: looks like 1.  See :func:`_rank`.
RANK_RESOLVE_TRIES = 256


def _rank() -> int:
    """This process's index, lazily, then pinned.

    ``common.collectives.process_rank`` answers 0 when JAX is absent OR
    not distributed yet, and neither is distinguishable from a genuine
    rank 0.  A journal that resolved once, before ``jax.distributed``
    came up, would name EVERY rank's file ``rank0`` and let 144 ranks
    append to one file — so while the world still looks like 1 the
    answer is re-asked, for the first :data:`RANK_RESOLVE_TRIES` ops.

    The cap is what keeps this off the hot path: ``process_rank()`` goes
    through ``jax.process_index()``, measured ~5 µs, and paying that on
    every one of ~1030 ops per iteration would make the journal cost
    more than the audit budgeted for it.  In every real launch the
    runtime is distributed long before the first HDF5 op, so the first
    call pins it; the cap only bounds the pathological case.
    """
    st = _STATE
    if st["rank"] is not None and (st["world"] > 1
                                   or st["tries"] >= RANK_RESOLVE_TRIES):
        return st["rank"]
    st["tries"] += 1
    try:
        from common.collectives import process_count, process_rank
        st["rank"] = int(process_rank())
        st["world"] = int(process_count())
    except Exception:                                          # noqa: BLE001
        st["rank"] = 0 if st["rank"] is None else st["rank"]
        st["world"] = 1
    return st["rank"]


def _dir() -> str:
    return os.environ.get("LORRAX_H5_JOURNAL_DIR") or os.getcwd()


def journal_path() -> str:
    """``<run_dir>/h5_journal.rank<R>.log`` for this process."""
    return os.path.join(_dir(), f"h5_journal.rank{_rank()}.log")


def crash_path() -> str:
    """``<run_dir>/h5_journal_crash.rank<R>.txt`` for this process."""
    return os.path.join(_dir(), f"h5_journal_crash.rank{_rank()}.txt")


def _break(exc: BaseException, what: str) -> None:
    """Disable the journal after an I/O failure, once, loudly."""
    _STATE["broken"] = True
    stream = _STATE["stream"]
    _STATE["stream"] = None
    _STATE["log_path"] = None
    with contextlib.suppress(Exception):
        if stream is not None:
            stream.close()
    import warnings
    warnings.warn(
        f"LORRAX HDF5 journal disabled for this process: {what} failed with "
        f"{type(exc).__name__}: {exc}.  Set {DIR_ENV} to a writable "
        f"directory to get it back, or {MODE_ENV}=0 to silence this.  The "
        f"run continues — the journal is an instrument, not a gate.")


def _stream():
    """The line-buffered per-rank log file, opened on first use.

    The path is NOT recomputed per line: once the stream is open the
    only thing that can invalidate it is the rank becoming known (see
    :func:`_rank`), which is one integer comparison.  ``LORRAX_H5_JOURNAL_DIR``
    is therefore read once per process — a knob that moves the output
    mid-run would split one rank's journal in two, which is worse than
    ignoring it.  ``reset_for_test`` re-reads it.
    """
    stream = _STATE["stream"]
    if stream is not None and _STATE["rank_of_file"] == _rank():
        return stream
    want = journal_path()
    if stream is not None:                        # the rank became known
        with contextlib.suppress(Exception):
            stream.close()
        _STATE["stream"] = None
    try:
        os.makedirs(os.path.dirname(want) or ".", exist_ok=True)
        # buffering=1 is line buffering: every line reaches the OS as it
        # is written, which is the property that makes this file survive
        # a SIGSEGV.
        _STATE["stream"] = open(want, "a", buffering=1)
        _STATE["log_path"] = want
        _STATE["rank_of_file"] = _rank()
    except Exception as exc:                                   # noqa: BLE001
        _break(exc, f"opening {want}")
        return None
    if not _STATE["atexit"]:
        _STATE["atexit"] = True
        atexit.register(_atexit_dump)
    return _STATE["stream"]


# ---------------------------------------------------------------------------
# The line
# ---------------------------------------------------------------------------

def _tok(value) -> str:
    """One whitespace-free field value; ``-`` for absent.

    The ``str`` and ``None`` fast paths are first because they are the
    common cases on the hot line and this function is called ~12 times
    per op — it was measured as the single largest term in the line's
    cost before the fast paths were added.
    """
    if value is None:
        return "-"
    if type(value) is str:
        if not value:
            return "-"
        return value if value.isprintable() and " " not in value else (
            "_".join(value.split()) or "-")
    if isinstance(value, (tuple, list)):
        return "(" + ",".join(_tok(v) for v in value) + ")"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    text = "_".join(str(value).split())
    return text or "-"


def _handle_tok(value) -> str:
    """Handles print as hex — they are ``PhdfCtx*`` and dataset ids.

    The C++ side prints the same pointer, so a journal line and an
    ``announce_error`` line can be matched by eye.
    """
    if value is None:
        return "-"
    if isinstance(value, bool):
        return _tok(value)
    if isinstance(value, int):
        return hex(value)
    if isinstance(value, (tuple, list)):
        return "(" + ",".join(_handle_tok(v) for v in value) + ")"
    return _tok(value)


def _rc_tok(rc) -> str:
    """``ok`` or ``refused:<first 40 chars>``."""
    if rc is None or rc == "ok":
        return "ok"
    if isinstance(rc, BaseException):
        msg = " ".join(str(rc).split()) or type(rc).__name__
    else:
        msg = " ".join(str(rc).split())
        if msg.startswith("refused:"):
            msg = msg[len("refused:"):]
    return "refused:" + msg[:REFUSAL_CHARS]


def _owner_tok(path, owner) -> str:
    """The registry's live-ownership verdict for ``path``.

    Imported lazily, and inside the guard, because
    :mod:`file_io.hdf5_owner` journals THROUGH this module — the import
    is a cycle at module scope and a dict lookup here.
    """
    if owner is not None:
        return _tok(owner)
    try:
        from . import hdf5_owner
        return _tok(hdf5_owner.live_verdict(path))
    except Exception:                                          # noqa: BLE001
        return "-"


def format_line(*, t, rank, stack, op, path, handle, ds, off, cnt, mode,
                owner, rc) -> str:
    """The frozen line, fixed key order.  Public so tests can pin it."""
    return (f"t={t:.6f} rank={rank} stack={stack} op={op} path={path} "
            f"handle={handle} ds={ds} off={off} cnt={cnt} mode={mode} "
            f"owner={owner} rc={rc}")


def record(op: str, path, *, stack: str, handle=None, ds=None, off=None,
           cnt=None, mode=None, owner=None, rc="ok") -> str | None:
    """Journal one HDF5 op.  Returns the line, or ``None`` when off.

    ``op`` and ``stack`` must be in :data:`OPS` / :data:`STACKS` — a
    mis-spelled op is a fact nobody can recover from the log later, and
    every call site is an in-tree constant.  Everything else is
    best-effort and prints as ``-`` when absent.
    """
    if op not in OPS:
        raise ValueError(
            f"h5_journal: unknown op {op!r}; the vocabulary is frozen with "
            f"the line format: {sorted(OPS)}")
    if stack not in STACKS:
        raise ValueError(
            f"h5_journal: unknown stack {stack!r}; there are two HDF5 "
            f"library instances and they are spelled {sorted(STACKS)} "
            f"(file_io.hdf5_owner owns the names)")
    if getattr(_INSIDE, "flag", False) or _STATE["broken"]:
        return None
    # ONE toggle parse per line, not two: this is the hot path (~1030
    # HDF5 ops per SC iteration) and ``mode()`` is a getenv + strip +
    # lower every time it is asked.
    live = _toggle()
    if live == MODE_OFF:
        return None
    _INSIDE.flag = True
    try:
        abs_path = os.path.abspath(os.fspath(path)) if path else None
        line = format_line(
            t=time.monotonic(), rank=_rank(), stack=stack, op=op,
            path=_tok(abs_path),
            handle=_handle_tok(handle), ds=_tok(ds), off=_tok(off),
            cnt=_tok(cnt), mode=_tok(mode),
            owner=_owner_tok(abs_path, owner), rc=_rc_tok(rc))
        with _LOCK:
            _RING.append(line)
            if not line.endswith("rc=ok"):
                _STATE["abnormal"] = True
            stream = _stream()
            if stream is None:
                return line
            try:
                stream.write(line + "\n")
                if live == MODE_SYNC:
                    stream.flush()
                    os.fsync(stream.fileno())
            except Exception as exc:                           # noqa: BLE001
                _break(exc, f"writing {_STATE['log_path']}")
        return line
    finally:
        _INSIDE.flag = False


def mode_is_sync() -> bool:
    """``LORRAX_H5_JOURNAL=sync`` — fsync every line."""
    return mode() == MODE_SYNC


# ---------------------------------------------------------------------------
# The crash ring
# ---------------------------------------------------------------------------

def ring() -> list:
    """The last :data:`RING_LINES` journal lines, oldest first."""
    with _LOCK:
        return list(_RING)


def crash_dump(reason: str, *, exc: BaseException | None = None) -> str | None:
    """Dump the ring into ``h5_journal_crash.rank<R>.txt``.  Never raises.

    APPENDS, with a header per dump: a second refusal is evidence about
    the first, and a dump that overwrites its predecessor loses exactly
    the sequence that explains it.
    """
    if getattr(_INSIDE, "flag", False) or not enabled():
        return None
    _INSIDE.flag = True
    try:
        path = crash_path()
        lines = ring()
        head = (
            f"# LORRAX HDF5 journal crash ring — {reason}\n"
            f"#   rank={_rank()} dumped={_STATE['dumped'] + 1} "
            f"t={time.monotonic():.6f} lines={len(lines)} "
            f"(ring depth {RING_LINES})\n"
            f"#   journal: {_STATE['log_path'] or journal_path()}\n")
        if exc is not None:
            detail = " ".join(str(exc).split())
            head += f"#   exception: {type(exc).__name__}: {detail}\n"
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a") as fh:
                fh.write(head)
                fh.write("\n".join(lines))
                fh.write("\n\n")
                fh.flush()
                os.fsync(fh.fileno())
        except Exception as write_exc:                         # noqa: BLE001
            _break(write_exc, f"writing {path}")
            return None
        with _LOCK:
            _STATE["dumped"] += 1
        return path
    finally:
        _INSIDE.flag = False


def _atexit_dump() -> None:
    """Dump the ring at exit if anything abnormal was journaled.

    A refusal that the caller caught and continued past still deserves
    its ring: the pipeline may have survived it and died of something
    else 20 minutes later.
    """
    with contextlib.suppress(Exception):
        if _STATE["abnormal"] and _STATE["dumped"] == 0:
            crash_dump("atexit: a refusal or exception was journaled "
                       "during this run")
        stream = _STATE["stream"]
        if stream is not None:
            stream.flush()
            stream.close()
            _STATE["stream"] = None


# ---------------------------------------------------------------------------
# The call-site helpers
# ---------------------------------------------------------------------------

def fail(op: str, path, exc: BaseException, *, stack: str, **kw) -> None:
    """Journal ``op`` as refused and dump the ring.  Never raises."""
    with contextlib.suppress(Exception):
        kw.pop("rc", None)
        record(op, path, stack=stack, rc=exc, **kw)
        crash_dump(f"{type(exc).__name__} crossing {stack} {op} "
                   f"{os.path.basename(str(path))}", exc=exc)


@contextlib.contextmanager
def op_scope(op: str, path, *, stack: str, **kw):
    """Journal ``op`` at ISSUE, and again as refused if the body raises.

    The issue-time line is the point of the instrument: a process that
    dies inside HDF5 leaves a journal whose last line names the op that
    killed it.  See the module docstring for why an ``open`` line cannot
    carry its handle.
    """
    record(op, path, stack=stack, **kw)
    try:
        yield
    except BaseException as exc:
        fail(op, path, exc, stack=stack, **kw)
        raise


def reset_for_test() -> None:
    """Close the stream and drop all state.  Tests only."""
    with _LOCK:
        stream = _STATE["stream"]
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.close()
        _RING.clear()
        _STATE.update({"stream": None, "log_path": None, "rank_of_file": None,
                       "rank": None, "world": 1, "tries": 0,
                       "abnormal": False, "dumped": 0, "broken": False})
        _INSIDE.flag = False
