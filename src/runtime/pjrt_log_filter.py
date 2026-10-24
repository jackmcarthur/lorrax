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
explicit finalization barrier, it may additionally discard the PJRT client's
``WatchTasksAsync`` (JAX 0.12) or ``WatchJobStateAsync`` (JAX 0.9.1) shutdown
receipts.  Every rank's backend client
watches the coordination service's task states; when rank zero deliberately
closes the service it owns, each watcher reports ``CANCELLED`` once and then
``UNAVAILABLE ... Connection refused`` on every retry, about once a second,
until the process reaches ``os._exit`` (measured on Perlmutter, P = 4: 10-13
receipts per process).  The watcher belongs to the backend client, which
:func:`runtime.finalize_process` deliberately never destroys, so its receipts
cannot be prevented, only recognized.  Each receipt is a block: the head line,
the GRPC context line and the status detail; JAX 0.12 adds a source-location
trace and a blank line.  Those extra lines are dropped only inside a block that opened
with an exact head line.  The shutdown gate matters: an identical message
during calculation is still forwarded.  Every other byte is passed to the
original descriptor unchanged.

The filter owns fd 2 only while fd 2 still IS its pipe.  A harness may move fd
2 under it (the hsuite chain points fds 1/2 at each stage's log and back while
one pytest process runs every driver): once the pipe's last writer is gone
the forwarding thread ends, and the clean-shutdown receipts at
:func:`runtime.finalize_process` would reach whatever fd 2 is by then,
unfiltered (Frontier/A100 2026-10-16: one ``WatchJobStateAsync ... CANCELLED``
receipt per rank after pytest's summary).  So :meth:`begin_clean_shutdown`
re-attaches the filter to the CURRENT fd 2 when it is no longer the pipe, and
:meth:`stop` restores fd 2 only when fd 2 is still the pipe -- it never writes
a stale saved descriptor over a descriptor somebody else installed.
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
_CLEAN_SHUTDOWN_STATUS = (rb"(?:CANCELLED: ?CANCELLED|UNAVAILABLE: ?(?:" + _CONNECTION_REFUSED +
                          rb"|" + _CANCELLING_ALL_CALLS + rb"))")
_CLEAN_SHUTDOWN_HEAD_LINE = re.compile(
    rb"W\d{4} \d{2}:\d{2}:\d{2}\.\d+ +\d+ "
    rb"pjrt_client\.cc:\d+\] Watch(?:Tasks|JobState)Async failed for task \d+: "
    + _CLEAN_SHUTDOWN_STATUS
)
_CLEAN_SHUTDOWN_CONTEXT_LINES = (
    b"Additional GRPC error information from remote target "
    b"coordination_service while calling "
    b"/xla.coordination.CoordinationService/WatchTasks:",
    b"Additional GRPC error information from remote target "
    b"coordination_service while calling "
    b"/tensorflow.CoordinationService/WatchJobState:",
)
_CLEAN_SHUTDOWN_DETAIL_LINE = re.compile(
    rb":" + _CLEAN_SHUTDOWN_STATUS +
    rb"(?: \[type\.googleapis\.com/tensorflow\.DerivedStatus=''\])?"
)
_LEGACY_SHUTDOWN_DETAIL_LINE = re.compile(
    rb':UNKNOWN:Error received from peer  \{(?:'
    rb'grpc_status:(\d+), grpc_message:"([^"\r\n]+)"|'
    rb'grpc_message:"([^"\r\n]+)", grpc_status:(\d+))\}'
)
_TRACE_HEADER_LINE = b"=== Source Location Trace: ==="
_TRACE_FRAME_LINE = re.compile(rb"[\w./+-]+\.(?:h|cc):\d+")


def is_spurious_pjrt_version_notice(line: bytes) -> bool:
    """Return whether *line* is exactly the upstream-removed PJRT notice."""
    return _SPURIOUS_PJRT_LINE.fullmatch(line) is not None


def clean_shutdown_step(line: bytes, in_receipt: bool) -> tuple[bool, bool]:
    """One line of JAX's coordinator-close receipts: ``(suppress, in_receipt)``.

    A receipt opens only on an exact head line; inside one, its context,
    detail and source-location lines are dropped. The legacy detail or the
    newer format's blank line closes it. Any other line closes it and is forwarded.
    """
    stripped = line.rstrip(b"\r\n")
    if _CLEAN_SHUTDOWN_HEAD_LINE.fullmatch(stripped):
        return True, True
    if not in_receipt:
        return False, False
    legacy = _LEGACY_SHUTDOWN_DETAIL_LINE.fullmatch(stripped)
    if legacy:
        status = legacy[1] or legacy[4]
        message = legacy[2] or legacy[3]
        label = {b"1": b"CANCELLED", b"14": b"UNAVAILABLE"}.get(status, b"")
        # Reuse the same status whitelist; the legacy three-line block ends here.
        suppress = _CLEAN_SHUTDOWN_DETAIL_LINE.fullmatch(b":" + label + b":" + message)
        return suppress is not None, False
    if not stripped:
        return True, False
    if (stripped in _CLEAN_SHUTDOWN_CONTEXT_LINES
            or _CLEAN_SHUTDOWN_DETAIL_LINE.fullmatch(stripped)
            or stripped.rstrip() == _TRACE_HEADER_LINE
            or _TRACE_FRAME_LINE.fullmatch(stripped)):
        return True, True
    return False, False


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
        self._in_receipt = False       # inside a clean-shutdown receipt block
        self._pipe_id: tuple[int, int] | None = None

    def _attached(self) -> bool:
        """Whether ``target_fd`` is still this filter's pipe (caller holds the lock)."""
        if not self._started or self._pipe_id is None:
            return False
        try:
            st = os.fstat(self._target_fd)
        except OSError:
            return False
        return (st.st_dev, st.st_ino) == self._pipe_id

    def _retire(self) -> None:
        """Forget a filter whose pipe no longer is ``target_fd`` (caller holds the lock).

        Its thread ended when the pipe lost its last writer; then its saved
        descriptor is closed.  If something else still holds a copy of the
        pipe's write end, the thread keeps forwarding that copy's output to its
        saved descriptor and owns both until EOF.
        """
        thread, saved = self._thread, self._saved_fd
        if saved is not None and (thread is None or not thread.is_alive()):
            try:
                os.close(saved)
            except OSError:
                pass
        self._read_fd = self._saved_fd = self._thread = None
        self._pipe_id = None
        self._started = False

    @property
    def suppressed(self) -> int:
        return self._suppressed

    def start(self) -> None:
        """Install the filter.  Repeated calls are no-ops."""
        with self._lock:
            self._start_locked()

    def _start_locked(self) -> None:
        if self._started:
            if self._attached():
                return
            self._retire()
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
        st = os.fstat(write_fd)
        os.close(write_fd)
        self._pipe_id = (st.st_dev, st.st_ino)
        self._started = True

    def begin_clean_shutdown(self) -> None:
        """Permit exact coordinator-cancel receipts after the final barrier."""
        with self._lock:
            if not self._started:
                return
            if not self._attached():
                # fd 2 was moved under the filter (see the module docstring):
                # filter the descriptor that will actually receive the receipts.
                try:
                    self._start_locked()
                except OSError:
                    return
            self._clean_shutdown = True

    def _is_suppressed(self, line: bytes) -> bool:
        if is_spurious_pjrt_version_notice(line):
            return True
        if not self._clean_shutdown:
            return False
        suppress, self._in_receipt = clean_shutdown_step(line, self._in_receipt)
        return suppress

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
                    if self._is_suppressed(line):
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
                if self._is_suppressed(line):
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
            if not self._attached():
                # fd 2 is no longer this pipe: restoring the saved descriptor
                # would overwrite whatever the harness put there.
                self._retire()
                return self._suppressed

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
            self._pipe_id = None
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
