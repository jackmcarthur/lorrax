"""The face batch admission replayed from a production receipt (CPU only, seconds).

CrI3 24x24 bispinor QSGW at P64 (8x8 mesh), cold SC map 0, 2026-10-05
(``runs/CrI3/512_fm_24x24_750b_20261002/02_ccb_lever1_20261005/leg02``,
``tmp/mpa/sc_0000_shared_pole/constructor_receipt.json``): the compiled
sizing admitted 3 of 61 parents per round, with CC 4.357 GB, TT 10.244 GB and
CT 12.546 GB per rank compiled at width 3, a reduction row of 14.649 GB
resident, 6.919 GB native workspace and 5.969 GB of upstream live stages
against a 71.999 GB budget (limit 75.272 GB). The shape price must bound every
compiled figure and land on the same batch, so the programs that run are the
ones main ran.
"""
from types import SimpleNamespace

import numpy as np


COMPILED_GB = dict(CC=4.357360783, TT=10.243909135, CT=12.546433059)
UPSTREAM, WORKSPACE = 27537436963 - 14648882211 - 6918995200, 6918995200
AVAILABLE, LIMIT = 71999000000, 75271680000.0
ROUTES = [dict(sector='CC', packed_extent=3328, conservative_pencil_side=20800, pole_budget=5940,
               signed_side_bound=12288, line_width=208, infinity_width=416),
          dict(sector='TT', packed_extent=5184, conservative_pencil_side=32000, pole_budget=8910,
               signed_side_bound=18432, line_width=320, infinity_width=640)]


class Ledger:
    """The map ledger's preview on the receipt's budget: aggregate = resident + workspace + upstream."""
    live_stages = ('sector_models.sc_0000',)

    def preview(self, *, resident_bytes_per_rank, workspace_bytes_per_rank, concurrent_with):
        aggregate = resident_bytes_per_rank + workspace_bytes_per_rank + UPSTREAM
        return dict(aggregate_bytes_per_rank=aggregate, available_device_bytes_per_rank=AVAILABLE,
                    device_budget_status='PASS' if aggregate <= LIMIT else 'FAIL')


def test_p64_receipt_lands_on_the_compiled_batch(monkeypatch):
    import file_io  # noqa: F401  (service path bootstrap)
    import gw.shared_pole_capacity as cap
    import gw.shared_pole_execution as ex

    mesh = SimpleNamespace(shape={'x': 8, 'y': 8}, size=64)

    def quote(self, side, *, phase, sample_batch=1, selection_faces=None, eigen_side=None,
              cross_original_sides=None, padding_output_bytes_per_rank=0):
        # The native eigh and matmul workspace is a vendor query (0 off CUDA): the receipt's figure stands in.
        price = self.resident_quote(side, phase=phase, sample_batch=sample_batch, selection_faces=selection_faces,
                                    cross_original_sides=cross_original_sides,
                                    padding_output_bytes_per_rank=padding_output_bytes_per_rank)
        return price, {'eigh': WORKSPACE}
    monkeypatch.setattr(cap.ConstructorCapacity, 'quote', quote)
    monkeypatch.setattr(ex, 'line_panel_count', lambda recipe: 2)
    joint = SimpleNamespace(n_rmu_padded=8512)
    width, receipt = ex.sector_batch_width(joint, None, {'fit_ids': [0, 1, 2, 3, 4]}, ROUTES,
                                           mesh=mesh, ledger=Ledger(), nq=61)
    assert width == 3
    prices = receipt['sector_program_bytes_per_rank']
    for name, compiled in COMPILED_GB.items():
        assert prices[name] >= compiled * 1e9, (name, prices[name] / 1e9, compiled)
    # The receipt's non-program resident at width 3 (panels, scalars, dense samples) is reproduced.
    nonprogram = receipt['reduction']['aggregate_bytes_per_rank'] - UPSTREAM - WORKSPACE - receipt['program_bytes_per_rank']
    assert abs(nonprogram - (14648882211 - 12546433059)) < 0.15e9


def test_price_is_linear_in_the_batch_and_zero_free():
    import file_io  # noqa: F401
    import gw.shared_pole_execution as ex
    mesh = SimpleNamespace(shape={'x': 2, 'y': 2}, size=4)
    one = ex.face_reduction_bytes(mesh, 1, rows=256, side=1024, carrier=128, retain_span=True)
    assert one > 0 and ex.face_reduction_bytes(mesh, 3, rows=256, side=1024, carrier=128, retain_span=True) == 3 * one
    assert ex.face_cross_bytes(mesh, 2, 512, (1024, 2048), (256, 512)) == 2 * ex.face_cross_bytes(mesh, 1, 512, (1024, 2048), (256, 512))
    assert ex.face_check_bytes(mesh, 256, 1024) > 0
