"""SlabIO read-after-write agreement (``_FfiBackend._drain_for_read``).

A writer error is sticky and never raised by a drain (``common.async_io``),
so no rank skips a collective write.  A read on the SAME handle must then
agree on that error before it returns bytes: otherwise a rank whose write
failed reads stale or unwritten data silently until ``close()``, and a value
derived from it can land in another file that commits first.

These cells drive the gating logic on one process with a stub dispatcher and
a recording ``agree_io_error``.  The real four-rank transport, with a write
that fails on one rank only, is ``tests/core/slab_io_read_after_write_p4.py``.
"""
from __future__ import annotations

import inspect

import pytest

import common.collectives as collectives
from file_io._slab_io_ffi import _FfiBackend


class _Dispatcher:
    def __init__(self, error=None):
        self.error = error
        self.drains = 0

    def drain(self):
        self.drains += 1


def _backend(*, mode="a", error=None, submitted=0, agreed=0):
    backend = _FfiBackend.__new__(_FfiBackend)
    backend.mode = mode
    backend.path = "/nonexistent/read_after_write.h5"
    backend._dispatcher = _Dispatcher(error)
    backend._queued_bytes = 0
    backend._writes_submitted = submitted
    backend._writes_agreed = agreed
    return backend


@pytest.fixture
def agreements(monkeypatch):
    seen = []

    def agree(error, *, path, stage):
        seen.append((error, path, stage))
        if error is not None:
            raise RuntimeError(
                f"GATE io_global_commit: path={path}; stage={stage}; {error}")

    monkeypatch.setattr(collectives, "agree_io_error", agree)
    return seen


def test_a_read_after_a_failed_write_raises_the_writer_error(agreements):
    backend = _backend(error=OSError("injected write failure"), submitted=2)
    with pytest.raises(RuntimeError, match="stage=SlabIO.read_after_write;"
                                           " injected write failure"):
        backend._drain_for_read()
    assert backend._dispatcher.drains == 1
    assert [stage for *_, stage in agreements] == ["SlabIO.read_after_write"]


def test_the_agreement_is_taken_once_per_batch_of_writes(agreements):
    backend = _backend(submitted=3)
    backend._drain_for_read()
    backend._drain_for_read()
    assert len(agreements) == 1, "a read with no new write must not agree"
    backend._writes_submitted += 1
    backend._drain_for_read()
    assert len(agreements) == 2
    assert backend._dispatcher.drains == 3


def test_a_read_only_handle_never_agrees(agreements):
    backend = _backend(mode="r", submitted=1)
    backend._drain_for_read()
    assert agreements == []
    assert backend._dispatcher.drains == 1


def test_a_handle_with_no_writes_since_open_does_not_agree(agreements):
    backend = _backend(submitted=1, agreed=1)   # the COMMIT_STATE write
    backend._drain_for_read()
    assert agreements == []


@pytest.mark.parametrize("door", [
    "read_slab", "read_slabs", "read_whole", "padded_shape_for"])
def test_every_read_door_drains_through_the_agreement(door):
    # Red twin for a future edit that restores a bare ``_drain_pending()``
    # at the top of a read door: that door would read unagreed bytes again.
    source = inspect.getsource(getattr(_FfiBackend, door))
    assert "self._drain_for_read()" in source
    first = source.index("self._drain_for_read()")
    assert "self._drain_pending()" not in source[:first]
