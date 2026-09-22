"""Remove exact, known-benign JAX runtime notices from process stderr.

JAX persistent-cache hits deserialize an XLA executable through PjRt-IFRT.
In the XLA revision bundled with jaxlib 0.9.1,
``PjRtCompiler::IsExecutableVersionCompatible`` returns ``Unimplemented``
and ``pjrt_executable.cc`` logs a warning before deliberately assuming
compatibility.  A warm GW run therefore prints the same line once per cache
hit.  OpenXLA commit 77e9933e7d3a009aab643c9cf759203c5377d532 removed that
one log call as verbose and unactionable while preserving the compatibility
and error-propagation logic.

The message is an absl C++ ``LOG(WARNING)`` written directly to file
descriptor 2.  ``warnings.filterwarnings`` and replacing ``sys.stderr``
cannot see it; raising ``TF_CPP_MIN_LOG_LEVEL`` would hide every C++ warning.
This module instead forwards fd 2 through a small line filter and discards
only the exact upstream-removed notice.  After every process has passed the
explicit finalization barrier, it may additionally discard JAX 0.9.1's
three-line ``WatchJobStateAsync`` shutdown receipt: rank zero emits
``CANCELLED`` or an ``UNAVAILABLE`` connection-close variant when it
deliberately closes the coordination service it owns.  The shutdown gate
matters—an identical message during calculation is still forwarded.
Every other byte is passed to the original descriptor unchanged.
"""
from __future__ import annotations

import atexit
import os
import re
import threading


_SPURIOUS_PJRT_LINE = re.compile(
    rb"W\d{4} \d{2}:\d{2}:\d{2}\.\d+ +\d+ "
    rb"pjrt_executable\.cc:\d+\] "
    rb"Assume version compatibility\. PjRt-IFRT does not track XLA "
    rb"executable versions\.\r?\n?\Z"
)

_CONNECTION_REFUSED = (
    rb"failed to connect to all addresses; last error: UNKNOWN: "
    rb"ipv4:[0-9.]+:\d+: Failed to connect to remote host: Connection refused"
)
_CANCELLING_ALL_CALLS = rb"Cancelling all calls"
_CLEAN_SHUTDOWN_HEAD_LINE = re.compile(
    rb"W\d{4} \d{2}:\d{2}:\d{2}\.\d+ +\d+ "
    rb"pjrt_client\.cc:\d+\] WatchJobStateAsync failed for task \d+: "
    rb"(?:CANCELLED: CANCELLED|UNAVAILABLE: (?:" + _CONNECTION_REFUSED +
    rb"|" + _CANCELLING_ALL_CALLS + rb"))\r?\n?\Z"
)
_CLEAN_SHUTDOWN_CONTEXT_LINE = (
    b"Additional GRPC error information from remote target "
    b"coordination_service while calling "
    b"/tensorflow.CoordinationService/WatchJobState:"
)


def is_spurious_pjrt_version_notice(line: bytes) -> bool:
    """Return whether *line* is exactly the upstream-removed PJRT notice."""
    return _SPURIOUS_PJRT_LINE.fullmatch(line) is not None


def is_clean_shutdown_notice(line: bytes) -> bool:
    """Match one line of JAX's expected coordinator-close receipts."""
    stripped = line.rstrip(b"\r\n")
    if (_CLEAN_SHUTDOWN_HEAD_LINE.fullmatch(line)
            or stripped == _CLEAN_SHUTDOWN_CONTEXT_LINE):
        return True
    if not stripped.startswith(b":UNKNOWN:Error received from peer  {"):
        return False
    cancelled = (b"grpc_status:1" in stripped
                 and b'grpc_message:"CANCELLED"' in stripped)
    unavailable = (b"grpc_status:14" in stripped and (
        re.search(_CONNECTION_REFUSED, stripped) is not None
        or b'grpc_message:"Cancelling all calls"' in stripped))
    return cancelled or unavailable


_REMATERIALIZATION_LINE = re.compile(
    rb"W\d{4} \d{2}:\d{2}:\d{2}\.\d+ +\d+ "
    rb"hlo_rematerialization\.cc:\d+\] "
    rb"Can't reduce memory use below [\d.]+\w+ \((\d+) bytes\) "
    rb"by rematerialization; only reduced to [\d.]+\w+ \((\d+) bytes\), "
    rb"down from [\d.]+\w+ \((\d+) bytes\) originally"
)

#: Incremented for every rematerialization-failure line seen on this process.
#: A driver gate reads it through :func:`rematerialization_events` and refuses;
#: the filter itself only reports, because it runs on a forwarding thread where
#: raising would be swallowed rather than surfaced.
_REMAT_EVENTS = 0
_REMAT_LOCK = threading.Lock()


def parse_rematerialization_notice(line: bytes):
    """Return ``(limit, achieved, original)`` bytes for XLA's remat give-up line.

    ``None`` when *line* is not that message.  XLA emits it once per process
    per module, so a P36 run prints 36 identical copies of a condition the
    operator must act on exactly once.
    """
    hit = _REMATERIALIZATION_LINE.search(line)
    if hit is None:
        return None
    return tuple(int(v) for v in hit.groups())


def rematerialization_events() -> int:
    """How many rematerialization give-up lines this process has seen."""
    return _REMAT_EVENTS


def rematerialization_banner(limit: int, achieved: int, original: int) -> bytes:
    """The one actionable message that replaces XLA's repeated warning."""
    freed = original - achieved
    verdict = (f"freed {freed} bytes" if freed else
               "freed NOTHING -- it ran, failed, and changed no allocation")
    return (
        "\n"
        "+---------------------------------------------------------------+\n"
        "| LORRAX GATE xla_rematerialization: a module does not fit      |\n"
        "+---------------------------------------------------------------+\n"
        f"| got    peak {original} B against an XLA budget of\n"
        f"|        {limit} B; rematerialization {verdict}.\n"
        "| cost   the pass ADDS buffers when it fails: a duplicated Green\n"
        "|        slot of 19.94 GB/rank was measured in the temp arena of\n"
        "|        a TaAs 8x8x8 P36 module (module_0208, 2026-09-21).\n"
        "| why    XLA's budget is a FRACTION of the card, and LORRAX's\n"
        "|        FFI (cuFFT arena, cuSOLVERMp/libcal) allocates OUTSIDE\n"
        "|        it -- so raising the fraction starves them instead.\n"
        "| fix    1. XLA_FLAGS=--xla_disable_hlo_passes=rematerialization\n"
        "|           when it frees nothing (measured cost: +256 B).\n"
        "|        2. Reduce the module's peak -- more ranks (peak goes as\n"
        "|           1/P) or fewer centroids.  The pass is NOT what makes\n"
        "|           an over-budget module fit.\n"
        "| doc    docs/environment/overview.md (allocator + FFI budget)\n"
        "+---------------------------------------------------------------+\n"
    ).encode()


def _write_all(fd: int, payload: bytes) -> None:
    """Write all of *payload*, including across an interrupted/partial write."""
    view = memoryview(payload)
    while view:
        try:
            written = os.write(fd, view)
        except InterruptedError:
            continue
        if written == 0:                                  # pragma: no cover
            raise OSError("zero-byte write while forwarding process stderr")
        view = view[written:]


class ExactPjrtNoticeFilter:
    """A start/stop fd filter for exact known-benign PJRT notices.

    ``target_fd`` is injectable so unit tests do not disturb pytest's own
    stderr capture.  Production always uses fd 2 through the module-level
    singleton below.
    """

    def __init__(self, target_fd: int = 2):
        self._target_fd = int(target_fd)
        self._lock = threading.Lock()
        self._read_fd: int | None = None
        self._saved_fd: int | None = None
        self._thread: threading.Thread | None = None
        self._started = False
        self._suppressed = 0
        self._clean_shutdown = False

    @property
    def suppressed(self) -> int:
        return self._suppressed

    def start(self) -> None:
        """Install the filter.  Repeated calls are no-ops."""
        with self._lock:
            if self._started:
                return
            read_fd, write_fd = os.pipe()
            try:
                saved_fd = os.dup(self._target_fd)
            except Exception:
                os.close(read_fd)
                os.close(write_fd)
                raise

            thread = threading.Thread(
                target=self._forward,
                args=(read_fd, saved_fd),
                name="lorrax-pjrt-log-filter",
                daemon=True,
            )
            self._read_fd = read_fd
            self._saved_fd = saved_fd
            self._thread = thread
            thread.start()
            try:
                os.dup2(write_fd, self._target_fd)
            except Exception:
                os.close(write_fd)
                thread.join(timeout=1.0)
                os.close(saved_fd)
                self._read_fd = None
                self._saved_fd = None
                self._thread = None
                raise
            os.close(write_fd)
            self._started = True

    def begin_clean_shutdown(self) -> None:
        """Permit exact coordinator-cancel receipts after the final barrier."""
        with self._lock:
            if self._started:
                self._clean_shutdown = True

    def _note_rematerialization(self, line: bytes, saved_fd: int) -> bool:
        """Replace XLA's repeated remat give-up line with ONE actionable gate.

        Returns whether *line* was that message.  The first occurrence writes
        the banner; later copies (one per rank, per module) are dropped, so an
        operator reads the condition once instead of 36 times.
        """
        global _REMAT_EVENTS
        parsed = parse_rematerialization_notice(line)
        if parsed is None:
            return False
        with _REMAT_LOCK:
            _REMAT_EVENTS += 1
            first = _REMAT_EVENTS == 1
        if first:
            _write_all(saved_fd, rematerialization_banner(*parsed))
        return True

    def _is_suppressed(self, line: bytes) -> bool:
        return (is_spurious_pjrt_version_notice(line)
                or (self._clean_shutdown
                    and is_clean_shutdown_notice(line)))

    def _forward(self, read_fd: int, saved_fd: int) -> None:
        pending = bytearray()
        try:
            while True:
                try:
                    chunk = os.read(read_fd, 65536)
                except InterruptedError:
                    continue
                if not chunk:
                    break
                pending.extend(chunk)
                while True:
                    newline = pending.find(b"\n")
                    if newline < 0:
                        break
                    line = bytes(pending[:newline + 1])
                    del pending[:newline + 1]
                    if self._note_rematerialization(line, saved_fd):
                        self._suppressed += 1
                    elif self._is_suppressed(line):
                        self._suppressed += 1
                    else:
                        _write_all(saved_fd, line)
                # stderr is a text log, but do not let one malformed writer
                # grow this buffer without bound.  A payload this long cannot
                # be the 159-byte notice, so forwarding it is exact and safe.
                if len(pending) > 4096:
                    _write_all(saved_fd, bytes(pending))
                    pending.clear()
            if pending:
                line = bytes(pending)
                if self._note_rematerialization(line, saved_fd):
                    self._suppressed += 1
                elif self._is_suppressed(line):
                    self._suppressed += 1
                else:
                    _write_all(saved_fd, line)
        finally:
            try:
                os.close(read_fd)
            except OSError:
                pass

    def stop(self) -> int:
        """Restore the target descriptor, drain the pipe, and return count."""
        with self._lock:
            if not self._started:
                return self._suppressed
            assert self._saved_fd is not None
            assert self._thread is not None

            # dup2 first closes the pipe writer occupying target_fd.  The
            # reader then sees EOF after draining all C++/Python writes.
            os.dup2(self._saved_fd, self._target_fd)
            self._thread.join(timeout=5.0)
            if not self._thread.is_alive():
                os.close(self._saved_fd)
                self._saved_fd = None
            # If an external sink blocked the forwarding thread, retain its
            # descriptor rather than invalidating an in-flight os.write.
            # The thread is daemonized and runtime.finalize_process ends in
            # os._exit, so this cannot hold process teardown hostage.
            self._read_fd = None
            self._thread = None
            self._started = False
            return self._suppressed


_PROCESS_FILTER = ExactPjrtNoticeFilter()
_ATEXIT_REGISTERED = False


def install_pjrt_log_filter() -> None:
    """Install the process-wide exact-line filter once."""
    global _ATEXIT_REGISTERED
    _PROCESS_FILTER.start()
    if not _ATEXIT_REGISTERED:
        atexit.register(stop_pjrt_log_filter)
        _ATEXIT_REGISTERED = True


def stop_pjrt_log_filter() -> int:
    """Stop the process-wide filter, returning its cumulative drop count."""
    return _PROCESS_FILTER.stop()


def begin_clean_shutdown_log_filter() -> None:
    """Enable exact clean-shutdown cancellation filtering process-wide."""
    _PROCESS_FILTER.begin_clean_shutdown()
