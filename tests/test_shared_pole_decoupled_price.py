"""The decoupled sector ledger against the measured CrI3 24x24 P64 peaks, and its runtime guard.

Replay of runs/CrI3/512_fm_24x24_750b_20261002/04_sectfast2_20261006/leg01 (claims 3545, 3555):
rank-0 pool high-water per stage, GB: TT.all 64.91, CT 61.34, CC.all 35.77 (budget 72). That run
priced TT.all at 47.07 and CT at 33.81: the held all-parent outputs had no row and the eighs
ran in their rooms with no row. The decoupled stage now reserves its stacks row (stacks, panels,
the stage program) and, while each eigh stack runs, a row of its boundary and its whole room.
Shapes from the leg's compile log and receipts (P64 8x8, 61 parents, sub-batches of 3):
TT pencil side 25856 (H'_vv 12928), kept span 2c 18432, 5184 rows, state panels 24576 columns;
CC side 19264, 2c 12288, 3328 rows, 18432 columns; face round program at width 3 13.1 / 5.6 GB.
"""
import numpy as np
import pytest

GB = 1e9
RANKS, NQ = 64, 61
AVAILABLE = 72 * GB          # memory_per_device_gb (the report's budget)
AMBIENT = 7.44 * GB          # sc.w_response's ledger before the sector stage


def _panels(rows, columns, infinity=640):
    per_rank = lambda e: e * 16 * NQ // RANKS
    held = per_rank(2 * rows * columns) + per_rank(5 * rows * infinity)
    return held, per_rank(rows * columns)


def _room(boundary, ambient):
    room = int(AVAILABLE - ambient - boundary)
    return (room >> 30) << 30 if room >= 1 << 30 else 0


def _price(side, carrier, rows, columns, program, ambient):
    from gw.shared_pole_execution import decoupled_stage_bytes
    held, dw = _panels(rows, columns)
    resident, boundaries = decoupled_stage_bytes(nq=NQ, ranks=RANKS, side=side, carrier=carrier, packed=rows,
                                                 held=held, dw_panels=dw, program=program)
    eighs = [ambient + b + _room(b, ambient) for b in boundaries]
    return ambient + resident, max(eighs), boundaries


@pytest.mark.parametrize("name,side,carrier,rows,columns,program,peak", [
    ("TT.all", 25856, 9216, 5184, 24576, 13.1 * GB, 64.91 * GB),
    ("CC.all", 19264, 6144, 3328, 18432, 5.6 * GB, 35.77 * GB),
])
def test_decoupled_price_bounds_the_measured_peak(name, side, carrier, rows, columns, program, peak):
    stacks_row, eigh_row, boundaries = _price(side, carrier, rows, columns, program, AMBIENT)
    assert stacks_row <= AVAILABLE, (name, stacks_row / GB)           # the stage is admitted
    assert max(stacks_row, eigh_row) >= peak, (name, stacks_row / GB, eigh_row / GB)


def test_tt_reduced_stack_room_fits_route_c_when_tt_runs_first():
    """The 61 x 18432 Y^H G_r Y stack compiled to 43.29 GB on route (c) against a 37.58 GB room
    (one room beside the largest boundary, CC's outputs live). Its own boundary, the dW Q panels
    released and TT before CC give it the room."""
    _, _, boundaries = _price(25856, 9216, 5184, 24576, 13.1 * GB, AMBIENT)
    assert _room(boundaries[2], AMBIENT) >= 43.29 * GB, _room(boundaries[2], AMBIENT) / GB


def test_ct_price_bounds_the_measured_peak():
    """CT ran in rounds there; now it runs decoupled beside both sectors' held outputs (rows):
    the pencil stack and keep stage, one round's program (15.4 GB/rank at width 3), and each eigh
    under its boundary and whole room (K 13824 = K_C 5632 + K_T 8192)."""
    from gw.shared_pole_execution import decoupled_cross_bytes
    held_tt, _ = _panels(5184, 24576)
    held_cc, _ = _panels(3328, 18432)
    span = lambda side, two, rows: (side * two + 2 * rows * two) * 16 * NQ // RANKS   # coefficients, models
    ambient = AMBIENT + held_tt + held_cc + span(25856, 18432, 5184) + span(19264, 12288, 3328)
    stacks, boundaries = decoupled_cross_bytes(nq=NQ, ranks=RANKS, side=13824, rows=(3328, 5184))
    stacks_row = ambient + stacks + 15.4 * GB
    eigh_rows = [ambient + b + _room(b, ambient) for b in boundaries]
    assert stacks_row <= AVAILABLE, stacks_row / GB
    assert max(stacks_row, *eigh_rows) >= 61.34 * GB, (stacks_row / GB, [e / GB for e in eigh_rows])


def test_guard_runs_a_stack_on_the_mesh_when_the_pool_is_short(monkeypatch):
    import jax.numpy as jnp
    import gw.shared_pole_execution as ex
    from distrib_la.plan import StackRoute
    calls = []

    class Plan:
        def __init__(self, tag, route):
            self.tag, self.route = tag, route

        def stack_route(self, shape, dtype, traced=True):
            return self.route

        def batched(self, a):
            calls.append(self.tag)
            return a

    stack = jnp.zeros((2, 4, 4))
    monkeypatch.setattr(ex, "face_eigh", lambda mesh, n, room=None: Plan("mesh", StackRoute("scan")))
    route_c = StackRoute("batch_reshard", 1, 1, program_bytes=10 * GB, room=20 * GB)
    for free, want in ((None, "c"), (30 * GB, "c"), (5 * GB, "mesh")):
        monkeypatch.setattr(ex, "_device_free_bytes", lambda free=free: free)
        calls.clear()
        with pytest.warns(RuntimeWarning) if want == "mesh" else _nothing():
            ex.guarded_eigh(Plan("c", route_c), stack, mesh=None, label="test")
        assert calls == [want], (free, calls)
    monkeypatch.setattr(ex, "_device_free_bytes", lambda: 0)
    calls.clear()
    ex.guarded_eigh(Plan("p", StackRoute("scan")), stack, mesh=None, label="test")   # never widens
    assert calls == ["p"]


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
