"""The decoupled sector ledger against the measured CrI3 24x24 P64 peaks, and its runtime check.

Replay of runs/CrI3/512_fm_24x24_750b_20261002/04_sectfast2_20261006/leg01 (claims 3545, 3555):
rank-0 pool high-water per stage, GB: TT.all 64.91, CC.all 35.77 (budget 72). That leg priced
TT.all at 47.07 (CC's held outputs, live through TT.all, had no row) and ran each decoupled eigh
in its room with no row. Now the stacks row carries the stacks, the panels and the stage program,
and each eigh runs under a row of its boundary and its own program (the service's route-(c)
program, or on the whole mesh its vectors stack and workspace), so the replay asserts against
priced rows, not the room. Shapes from the leg's compile log and receipts (P64 8x8, 61 parents,
sub-batches of 3): TT pencil side 25856, 2c 18432, 5184 rows, panels 24576 columns; CC side 19264,
2c 12288, 3328 rows, 18432 columns; face round program at width 3 13.1 / 5.6 GB. CT ran in rounds
there; the decoupled CT has no P64 measurement yet, so it has no replay here.
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


def _held_outputs(side, two, rows, columns, infinity=640):
    """A sector's held all-parent outputs per rank: coefficient span [side, 2c], signed and
    positive factors [rows, 2c], (Q, O) panels and the infinity panels."""
    held, _ = _panels(rows, columns, infinity)
    return held + (side * two + 2 * rows * two) * 16 * NQ // RANKS


# The leg's route-(c) eigh programs (receipts: compiled GB/rank per stack); TT's 61 x 18432
# stack ran on the whole mesh there: its vectors stack (the solve's vendor workspace is not
# queryable on CPU and is left out, which only makes the bound stricter).
PROGRAMS = {"TT": (21.31 * GB, 10.84 * GB, 61 * 18432 ** 2 * 16 / RANKS),
            "CC": (11.84 * GB, 4.82 * GB, 19.25 * GB)}
SHAPES = {"TT": (25856, 9216, 5184, 24576, 13.1 * GB), "CC": (19264, 6144, 3328, 18432, 5.6 * GB)}


def _price(name, ambient):
    from gw.shared_pole_execution import decoupled_stage_bytes
    side, carrier, rows, columns, program = SHAPES[name]
    held, dw = _panels(rows, columns)
    # That leg kept the dW Q panels to the end: price the boundaries with them.
    resident, boundaries = decoupled_stage_bytes(nq=NQ, ranks=RANKS, side=side, carrier=carrier, packed=rows,
                                                 held=held + dw, dw_panels=0, program=program)
    eighs = [ambient + b + p for b, p in zip(boundaries, PROGRAMS[name])]
    return ambient + resident, eighs


def test_decoupled_rows_bound_the_measured_peaks():
    """Replay of the leg (CC reduced first, so CC's held outputs were live through TT.all): the
    stacks rows and eigh rows, each with its own program bytes, against the measured peaks."""
    cc_stacks, cc_eighs = _price("CC", AMBIENT)
    assert cc_stacks <= AVAILABLE
    assert max(cc_stacks, *cc_eighs) >= 35.77 * GB, (cc_stacks / GB, [e / GB for e in cc_eighs])
    held_cc = _held_outputs(19264, 12288, 3328, 18432)
    tt_stacks, tt_eighs = _price("TT", AMBIENT + held_cc)
    assert tt_stacks <= AVAILABLE
    assert max(tt_stacks, *tt_eighs) >= 64.91 * GB, (tt_stacks / GB, [e / GB for e in tt_eighs])


def test_tt_reduced_stack_room_fits_route_c_when_tt_runs_first():
    """The 61 x 18432 Y^H G_r Y stack compiled to 43.29 GB on route (c) against a 37.58 GB room
    (one room beside the largest boundary, CC's outputs live). Its own boundary, the dW Q panels
    released and TT before CC give it the room."""
    from gw.shared_pole_execution import decoupled_stage_bytes
    side, carrier, rows, columns, program = SHAPES["TT"]
    held, dw = _panels(rows, columns)
    _, boundaries = decoupled_stage_bytes(nq=NQ, ranks=RANKS, side=side, carrier=carrier, packed=rows,
                                          held=held, dw_panels=dw, program=program)
    room = int(AVAILABLE - AMBIENT - boundaries[2])
    assert ((room >> 30) << 30) >= 43.29 * GB, room / GB


def test_guard_warns_and_never_changes_the_route(monkeypatch):
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
    route_c = StackRoute("batch_reshard", 1, 1, program_bytes=10 * GB, room=20 * GB)
    for free, warns in ((None, False), (30 * GB, False), (5 * GB, True)):
        monkeypatch.setattr(ex, "_device_free_bytes", lambda free=free: free)
        calls.clear()
        with pytest.warns(RuntimeWarning, match="short by 5.0 GB") if warns else _nothing():
            ex.guarded_eigh(Plan("c", route_c), stack, mesh=None, label="test")
        assert calls == ["c"], (free, calls)          # the ledger's route, whatever the pool holds


class _nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Ledger:
    """The map ledger's preview at the leg's budget: aggregate = resident + the live set."""
    def __init__(self, live):
        self.live = live

    def preview(self, *, resident_bytes_per_rank, workspace_bytes_per_rank, concurrent_with):
        return dict(available_device_bytes_per_rank=AVAILABLE,
                    aggregate_bytes_per_rank=resident_bytes_per_rank + workspace_bytes_per_rank + self.live)


def test_stage_width_fits_beside_the_stacks_at_a_wider_face_batch():
    """The whole-price face step admits 10 of 61 at P64 (test_shared_pole_face_price), where main ran
    3; the stage programs (13.1 / 5.6 GB at 3, linear in the width) then run at the largest width
    whose program fits beside each sector's stacks row, TT first: TT 5, CC 10."""
    from gw.shared_pole_execution import decoupled_stage_bytes, decoupled_width
    widths = {}
    live = AMBIENT
    for name in ("TT", "CC"):
        side, carrier, rows, columns, program = SHAPES[name]
        held, dw = _panels(rows, columns)
        resident, _ = decoupled_stage_bytes(nq=NQ, ranks=RANKS, side=side, carrier=carrier, packed=rows,
                                            held=held, dw_panels=dw, program=0)
        widths[name] = w = decoupled_width(_Ledger(live), resident, program / 3, 10, concurrent_with=())
        assert live + resident + w * program / 3 <= AVAILABLE < live + resident + (w + 1) * program / 3 or w == 10
        live += _held_outputs(side, 2 * carrier, rows, columns)
    assert widths == {"TT": 5, "CC": 10}, widths
