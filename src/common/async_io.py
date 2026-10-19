"""Single-worker daemon dispatcher with bounded back-pressure.

Generalises the ``_dispatch_loop`` pattern from
:mod:`file_io._slab_io_ffi` so the same threading discipline can drive
any single-resource async work.  Its one production consumer today is
the SlabIO write side (H5Dwrite tasks queued by ``SlabIO``).  A read-side
consumer (a prefetching wrapper over ``WfnLoader.load``) also existed
but was deleted 2026-07-25: measured H2D/compute overlap was 0.000 and
it had no callers (see ``common/psi_G_store.py``).

Why a single-worker thread and not a pool
-----------------------------------------
HDF5's MPI-IO file handles are not safe to drive from more than one
thread concurrently — the MPI datatype cache on the file handle
interleaves on out-of-order calls and trips
``MPI_File_set_view: Invalid datatype``.  A single worker preserves
FIFO order on whatever resource the caller owns (file handle, FFI
context), so callers can rely on submit-order = completion-order.

Why bounded queue
-----------------
Every queued task pins resources (a ``jax.Array`` buffer for writes;
a ψ-tile buffer for reads).  ``maxsize`` provides back-pressure so a
slow consumer can't accumulate unbounded in-flight work; tuned at the
SlabIO writer to K=2 — same throughput as K=4 while saving
2 × chunk-sized buffers (see ``_slab_io_ffi.py:330-339`` for the
measurement).
"""
from __future__ import annotations

import queue
import threading
from typing import Callable, Optional


class AsyncDispatcher:
    """Single daemon thread draining a bounded queue of zero-arg tasks.

    Public API
    ----------
    submit(task)
        Enqueue ``task: () -> None``.  Blocks if queue is at maxsize.
        NEVER raises a worker error (see below).
    drain()
        Wait until every in-flight task has finished.  Never raises.
    error
        The first worker exception, or ``None``.  STICKY: never cleared.
    close()
        Drain, send a poison-pill to the worker, join the thread.
        Idempotent; never raises.

    Error semantics
    ---------------
    A worker exception is kept in ``error`` and the worker goes on running
    every later task.  Nothing on the caller's thread raises it: the caller
    reads ``error`` at a point where every rank can agree on it
    (``_slab_io_ffi._FfiBackend.close`` -> ``collectives.agree_io_error``).

    WHY NOT RE-RAISE AT THE NEXT submit/drain (the pre-2026-09-24
    behaviour): the tasks are collective H5Dwrites.  A rank-local error
    re-raised there made that rank leave the write sequence and enter the
    collective H5Fclose while its peers were still in H5Dwrite: mismatched
    collectives, which decisions.md 2026-08-04 forbids.  Reproduced at P4
    (runs/runtime/slabio_concurrency_20260924, step
    lx-Xg4-115734-1647932-3912): MPICH aborted on "message sizes do not
    match across processes" inside H5Fclose; elsewhere it is a silent hang.
    A task that fails BEFORE entering its own collective still leaves its
    peers waiting; this class cannot recover that, only never add to it.
    """

    def __init__(self, name: str, maxsize: int = 2):
        self._queue: queue.Queue = queue.Queue(maxsize=maxsize)
        self._pending: int = 0
        self._mu = threading.Lock()
        self._cv = threading.Condition(self._mu)
        self._error: Optional[BaseException] = None
        self._closed = False
        self._worker = threading.Thread(
            target=self._loop, name=name, daemon=True)
        self._worker.start()

    def submit(self, task: Callable[[], None]) -> None:
        if self._closed:
            raise RuntimeError(
                f"AsyncDispatcher({self._worker.name}) already closed")
        with self._cv:
            self._pending += 1
        self._queue.put(task)

    def drain(self) -> None:
        with self._cv:
            while self._pending > 0:
                self._cv.wait()

    @property
    def pending(self) -> int:
        with self._mu:
            return self._pending

    @property
    def error(self) -> Optional[BaseException]:
        """The first worker exception (sticky), or ``None``."""
        with self._mu:
            return self._error

    def close(self) -> None:
        # The worker is joined unconditionally: the caller (SlabIO) is
        # inside a collective teardown, and a rank that abandoned its
        # writer thread here would leave it alive with a live MPI file
        # handle while its peers close theirs.
        if self._closed:
            return
        self.drain()
        self._closed = True
        self._queue.put(None)  # poison pill
        self._worker.join()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _loop(self) -> None:
        while True:
            task = self._queue.get()
            if task is None:
                return
            try:
                task()
            except BaseException as exc:  # noqa: BLE001
                with self._cv:
                    if self._error is None:
                        self._error = exc
            finally:
                with self._cv:
                    self._pending -= 1
                    self._cv.notify_all()


__all__ = ["AsyncDispatcher"]
