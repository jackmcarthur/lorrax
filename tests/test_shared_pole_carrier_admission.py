"""Optional shared-pole reuse must pass admission before any wider allocation."""

from types import SimpleNamespace

import numpy as np
import pytest


class _Panel:
    def __init__(self, width):
        self.shape = (1, 2, width)
        self.sharding = "fake-sharding"


def _round(width):
    return [(1j, _Panel(width), _Panel(width), _Panel(width))], (
        _Panel(width), _Panel(width), _Panel(width))


def test_high_water_is_model_local_and_falls_back_before_padding(monkeypatch):
    from gw import shared_pole_local as local

    events = []
    monkeypatch.setattr(
        local, "_pad_columns",
        lambda _sharding, _shape, width: (
            lambda _panel: events.append(("pad", width)) or _Panel(width)))
    first = local.carrier_history(SimpleNamespace())
    other = local.carrier_history(SimpleNamespace())
    assert first is not other

    def run(history, width, budget, *, preview_error=False):
        states, infinity = _round(width)

        def preview(widths, infinity_width, _reuse):
            events.append(("preview", widths[0], infinity_width))
            if preview_error:
                raise ValueError("optional native plan unavailable")
            return max(*widths, infinity_width) <= budget

        def admit(widths, infinity_width, _reuse):
            events.append(("admit", widths[0], infinity_width))
            if max(*widths, infinity_width) > budget:
                raise MemoryError("physical round does not fit")
            return widths[0]

        return local.grow_round(
            ("scalar", 2), states, infinity, history=history,
            preview=preview, admit=admit)

    run(first, 8, 8)
    events.clear()
    states, infinity, admitted = run(first, 4, 4)
    assert admitted == 4
    assert states[0][1].shape[-1] == infinity[0].shape[-1] == 4
    assert events == [("preview", 8, 8), ("admit", 4, 4)]
    assert first[(("scalar", 2), "states", 1)] == (8,)

    events.clear()
    assert run(first, 4, 4, preview_error=True)[2] == 4
    assert events == [("preview", 8, 8), ("admit", 4, 4)]

    events.clear()
    states, infinity, admitted = run(first, 4, 8)
    assert admitted == 8
    assert states[0][1].shape[-1] == infinity[0].shape[-1] == 8
    assert events[0:2] == [("preview", 8, 8), ("admit", 8, 8)]
    assert all(event[0] == "pad" for event in events[2:])

    events.clear()
    assert run(other, 4, 4)[2] == 4
    assert events == [("preview", 4, 4), ("admit", 4, 4)]

    events.clear()
    with pytest.raises(MemoryError, match="physical round"):
        run(other, 6, 4)
    assert all(event[0] != "pad" for event in events)


def test_capacity_preview_does_not_charge_a_rejected_candidate():
    from gw.shared_pole_capacity import ConstructorCapacity

    budget = ConstructorCapacity.__new__(ConstructorCapacity)
    budget._native_maxima = {"eigh": 5}
    budget._workspace = 5
    budget._ledger = SimpleNamespace(preview=lambda **_kwargs: {"device_budget_status": "FAIL"})
    budget._upstream = ()

    def quote(_side, **_kwargs):
        budget._native_maxima["eigh"] = 100
        budget._workspace = 100
        return {"resident_bytes_per_rank": 10}, {"eigh": 100}

    budget.quote = quote
    assert budget.preview(20, phase="reduction")["device_budget_status"] == "FAIL"
    assert budget._native_maxima == {"eigh": 5}
    assert budget._workspace == 5


def test_wider_panels_change_admission_at_the_same_pencil_side():
    from gw.shared_pole_capacity import ConstructorCapacity, round_padding_output_bytes

    class Sharding:
        def shard_shape(self, shape):
            return shape

    class Panel:
        def __init__(self, width):
            self.shape = (4, 8, width)
            self.sharding = Sharding()
            self.dtype = np.dtype("complex128")

    states = [(1j, Panel(2), Panel(2), Panel(2))]
    infinity = (Panel(2), Panel(2), Panel(2), Panel(2))
    padded = round_padding_output_bytes(states, infinity, (32,), 32)
    assert padded == 7 * 4 * 8 * 32 * 16
    assert round_padding_output_bytes(states, infinity, (2,), 2) == 0

    budget = ConstructorCapacity.__new__(ConstructorCapacity)
    budget.execution = "local"
    budget._meta = SimpleNamespace(n_rmu_padded=8)
    budget._mesh_xy = SimpleNamespace(shape={"x": 2, "y": 2})
    budget._resolution = SimpleNamespace(layout="local")
    budget._n = 8
    budget._native_maxima = {"eigh": 0}
    budget._workspace = 0
    budget._upstream = ()
    budget.retained_panels = ()
    budget.batch_width = 4
    budget.eigenplan = lambda _side: None
    budget.query_workspace = lambda *_args: 0
    side = 8
    base, _ = budget.quote(side, phase="reduction")
    limit = base["resident_bytes_per_rank"] + padded // 2
    budget._ledger = SimpleNamespace(preview=lambda **price: {
        "device_budget_status": "PASS" if price["resident_bytes_per_rank"]
        + price["workspace_bytes_per_rank"] <= limit else "FAIL"})

    assert budget.preview(side, phase="reduction")["device_budget_status"] == "PASS"
    assert budget.preview(side, phase="reduction",
        padding_output_bytes_per_rank=padded)["device_budget_status"] == "FAIL"
    charged, _ = budget.quote(side, phase="reduction",
        padding_output_bytes_per_rank=padded)
    assert charged["terms_bytes_per_rank"]["round_padding_outputs"] == padded
