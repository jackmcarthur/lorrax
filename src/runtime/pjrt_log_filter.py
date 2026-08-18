"""Remove one upstream-spurious PjRt-IFRT notice from process stderr.

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
only the exact upstream-removed notice.  Every other byte is passed to the
original descriptor unchanged.
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


def is_spurious_pjrt_version_notice(line: bytes) -> bool:
    """Return whether *line* is exactly the upstream-removed PJRT notice."""
    return _SPURIOUS_PJRT_LINE.fullmatch(line) is not None


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
    """A start/stop fd filter for the one known-spurious PJRT notice.

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
                    if is_spurious_pjrt_version_notice(line):
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
                if is_spurious_pjrt_version_notice(line):
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
