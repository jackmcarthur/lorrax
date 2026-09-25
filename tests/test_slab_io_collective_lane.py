"""An async union read must finish before the same handle re-enters HDF5."""

import jax
import pytest

from file_io._slab_io_ffi import _CollectiveLane, _LaneSlot


class _PendingRead:
    def __init__(self):
        self.ready = False


@pytest.mark.parametrize("next_door", ["metadata_or_read", "write"])
def test_same_handle_waits_for_its_union_read(next_door, monkeypatch):
    lane = _CollectiveLane()
    slot = _LaneSlot()
    marker = _PendingRead()
    lane._reads[id(slot)] = [marker]
    events = []

    def block(value):
        assert value == [marker]
        marker.ready = True
        events.append("read complete")

    class _Worker:
        pending = 0

        def submit(self, task):
            assert marker.ready, "write entered the lane before the read finished"
            events.append("write queued")

    monkeypatch.setattr(jax, "block_until_ready", block)
    monkeypatch.setattr(lane, "_worker", lambda: _Worker())

    if next_door == "write":
        lane.submit(slot, lambda: None)
        assert events == ["read complete", "write queued"]
    else:
        lane.quiesce(slot)
        assert events == ["read complete"]
    assert lane._reads == {}
