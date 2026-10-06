"""The face batch admission replayed from production receipts (CPU only, seconds).

Each receipt is a constructor_receipt.json written by main while it still
compiled every face program to size it: the route rows (packed extents,
conservative sides, pole budgets, signed bounds), the three compiled program
sizes per rank at the admitted width, and the reduction admission row
(resident, native workspace, upstream live stages, budget, limit). The shape
price must bound every compiled figure and the admission must land on the
same batch, so the programs that run are the ones main ran.

Receipts (all 2x2 or 8x8 meshes, bispinor sectors, retained spans):
* ``p64``: CrI3 24x24 P64 cold SC map 0, main 0a393dd72, 2026-10-05
  (runs/CrI3/512_fm_24x24_750b_20261002/02_ccb_lever1_20261005/leg02): batch 3 of 61.
* ``fe_774``: Fe 4^3 bispinor face deck (2.0 GB), P4, main 0a393dd72, map 2
  (runs/DEV/774_coldcompile_20261006/legs/fef_main): batch 1 of 64, over budget at 1.
* ``fe_773``: the same deck on main 2f8bfd55a (runs/DEV/773_ccb_nafix_20261005/legs/fef_main3).
* ``cri3_773``: CrI3 6x6 forced-face SC (6 GB), P4, main 2f8bfd55a
  (runs/DEV/773_ccb_nafix_20261005/legs/cri3f_main3): batch 1 of 36, over budget at 1.
"""
from types import SimpleNamespace

import pytest


def route(sector, packed, side, budget, signed):
    return dict(sector=sector, packed_extent=packed, conservative_pencil_side=side, pole_budget=budget,
                signed_side_bound=signed, line_width=1, infinity_width=1)


RECEIPTS = {
    # name: (mesh x, y, nq, joint packed, routes, compiled GB per rank at the batch, batch, row)
    # row: aggregate, resident, workspace, available, limit (bytes per rank), upstream = aggregate - resident - workspace
    "p64": (8, 8, 61, 8512, [route("CC", 3328, 20800, 5940, 12288), route("TT", 5184, 32000, 8910, 18432)],
            dict(CC=4.357360783, TT=10.243909135, CT=12.546433059), 3,
            dict(aggregate=27537436963, resident=14648882211, workspace=6918995200, available=71999000000, limit=75271680000.0)),
    "fe_774": (2, 2, 64, 864, [route("CC", 432, 3472, 778, 1664), route("TT", 432, 3472, 778, 1664)],
               dict(CC=0.350244219, TT=0.350244219, CT=0.747110991), 1,
               dict(aggregate=2479234639, resident=902302287, workspace=1576932352, available=1999000000, limit=2293235712.0)),
    "fe_773": (2, 2, 64, 864, [route("CC", 432, 3472, 778, 1664), route("TT", 432, 3472, 778, 1664)],
               dict(CC=0.350244219, TT=0.350244219, CT=0.541699663), 1,
               dict(aggregate=2273823311, resident=696890959, workspace=1576932352, available=1999000000, limit=2293235712.0)),
    "cri3_773": (2, 2, 36, 2352, [route("CC", 984, 7168, 1761, 3584), route("TT", 1368, 9856, 2430, 5120)],
                 dict(CC=1.712494827, TT=2.814849419, CT=2.770165511), 1,
                 dict(aggregate=15575352715, resident=3823185803, workspace=10888652288, available=5999000000, limit=6611217408.0)),
}


class Ledger:
    """The map ledger's preview on a receipt's budget: aggregate = resident + workspace + upstream."""
    live_stages = ('upstream',)

    def __init__(self, row):
        self.upstream = row['aggregate'] - row['resident'] - row['workspace']
        self.row = row

    def preview(self, *, resident_bytes_per_rank, workspace_bytes_per_rank, concurrent_with):
        aggregate = resident_bytes_per_rank + workspace_bytes_per_rank + self.upstream
        return dict(aggregate_bytes_per_rank=aggregate, available_device_bytes_per_rank=self.row['available'],
                    device_budget_status='PASS' if aggregate <= self.row['limit'] else 'FAIL')


def replay(monkeypatch, name):
    import file_io  # noqa: F401  (service path bootstrap)
    import gw.shared_pole_capacity as cap
    import gw.shared_pole_execution as ex
    px, py, nq, joint_packed, routes, compiled, batch, row = RECEIPTS[name]
    mesh = SimpleNamespace(shape={'x': px, 'y': py}, size=px * py)

    def quote(self, side, *, phase, sample_batch=1, selection_faces=None, eigen_side=None,
              cross_original_sides=None, padding_output_bytes_per_rank=0):
        # The native eigh and matmul workspace is a vendor query (0 off CUDA): the receipt's figure stands in.
        price = self.resident_quote(side, phase=phase, sample_batch=sample_batch, selection_faces=selection_faces,
                                    cross_original_sides=cross_original_sides,
                                    padding_output_bytes_per_rank=padding_output_bytes_per_rank)
        return price, {'eigh': row['workspace']}
    monkeypatch.setattr(cap.ConstructorCapacity, 'quote', quote)
    monkeypatch.setattr(ex, 'line_panel_count', lambda recipe: 2)
    width, receipt = ex.sector_batch_width(SimpleNamespace(n_rmu_padded=joint_packed), None, {'fit_ids': [0, 1, 2, 3, 4]},
                                           routes, mesh=mesh, ledger=Ledger(row), nq=nq)
    return width, receipt, compiled, batch


@pytest.mark.parametrize("name", sorted(RECEIPTS))
def test_price_bounds_every_compiled_program_and_lands_on_its_batch(monkeypatch, name):
    width, receipt, compiled, batch = replay(monkeypatch, name)
    prices = receipt['sector_program_bytes_per_rank']
    for sector, gb in compiled.items():
        assert prices[sector] >= gb * 1e9, (name, sector, prices[sector] / 1e9, gb)
    assert width == batch, (name, width, batch)


def test_p64_receipt_reproduces_the_non_program_resident(monkeypatch):
    width, receipt, compiled, batch = replay(monkeypatch, "p64")
    row = RECEIPTS["p64"][-1]
    nonprogram = (receipt['reduction']['aggregate_bytes_per_rank'] - (row['aggregate'] - row['resident'] - row['workspace'])
                  - row['workspace'] - receipt['program_bytes_per_rank'])
    assert abs(nonprogram - (row['resident'] - compiled['CT'] * 1e9)) < 0.15e9


def test_price_is_linear_in_the_batch_and_zero_free():
    import file_io  # noqa: F401
    import gw.shared_pole_execution as ex
    mesh = SimpleNamespace(shape={'x': 2, 'y': 2}, size=4)
    one = ex.face_reduction_bytes(mesh, 1, rows=256, side=1024, carrier=128, retain_span=True)
    assert one > 0 and ex.face_reduction_bytes(mesh, 3, rows=256, side=1024, carrier=128, retain_span=True) == 3 * one
    assert ex.face_cross_bytes(mesh, 2, 512, (1024, 2048), (256, 512)) == 2 * ex.face_cross_bytes(mesh, 1, 512, (1024, 2048), (256, 512))
    assert ex.face_check_bytes(mesh, 256, 1024) > 0
