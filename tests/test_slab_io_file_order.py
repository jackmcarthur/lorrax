"""SlabIO's file-order write plan: which writes leave as independent row blocks.

``file_io._slab_io_ffi._file_order_plan`` is a pure function of the operand's
shape, valid extent, sharding and the dataset's extent.  Each cell is a
production layout with the verdict the P16 bench (A2, 2026-09-25) calls for:
long per-rank file runs are written independently, strided tiles and anything
that could only be cut by slicing a sharded axis stay collective, and no piece
exceeds the per-rank staging budget.
"""
import pytest

from file_io._slab_io_ffi import (_FILE_ORDER_MIN_RUN, _FILE_ORDER_PIECE_BYTES,
                                  _file_order_plan, _file_run_bytes)

XY = ("x", "y")


def plan(shape, spec, ds, p, itemsize=16, vshape=None):
    return _file_order_plan(shape, vshape or shape, spec, itemsize, p, XY, ds)


def test_the_face_bank_sample_splits_its_rows_per_q():
    # Wc at one sample, all 59 q, face (q, s, mu_X, nu_Y) in a 22-sample bank.
    k, rows, lead = plan((59, 1, 3164, 3164), (None, None, ("x",), ("y",)),
                         (59, 22, 3164, 3164), 16)
    assert (k, rows) == (2, 3168)            # rows of mu, whole nu
    assert lead == 6                          # six q per piece: 6 x 10 MB per rank
    run = _file_run_bytes((1, 1, rows // 16, 3164), (59, 22, 3164, 3164), 16)
    assert run >= _FILE_ORDER_MIN_RUN and run <= _FILE_ORDER_PIECE_BYTES


def test_a_g_windowed_wavefunction_splits_bands_within_the_budget():
    # WFN_qp window: (bands, spinors, G window, re/im) f64, G-sharded.
    k, rows, lead = plan((900, 2, 76544, 2), (None, None, XY, None),
                         (900, 2, 2295337, 2), 16, itemsize=8)
    assert k == 0 and rows % 16 == 0 and lead == 1
    assert (rows // 16) * 2 * 76544 * 2 * 8 <= _FILE_ORDER_PIECE_BYTES


def test_an_operand_already_in_row_blocks_is_written_as_is():
    # zeta-like (q, mu over all ranks, G): long runs, no copy.
    assert plan((8, 4096, 50000), (None, XY, None), (8, 4096, 50000), 16) == "as-is"


def test_strided_or_uncuttable_layouts_stay_collective():
    # A face (q, mu_X, nu_Y) V tile too large to take whole on its sharded axis
    # (the coordinator's P4 case): cutting mu would gather it.
    assert plan((10, 8000, 8000), (None, ("x",), ("y",)), (10, 8000, 8000), 4) is None
    # Short runs: a small G window would give 80 kB independent runs.
    assert plan((900, 2, 5008, 2), (None, None, XY, None),
                (900, 2, 150000, 2), 16, itemsize=8) is None
    # Tiny: the per-file commit receipt.
    assert plan((16,), (XY,), (16,), 16, itemsize=4) is None


def test_one_rank_writes_as_is():
    assert plan((64, 3200, 3200), (None, None, None), (64, 3200, 3200), 1) is None


@pytest.mark.parametrize("p", [4, 16, 64])
def test_every_piece_fits_the_budget(p):
    for shape, spec, ds in (
            ((59, 1, 3164, 3164), (None, None, ("x",), ("y",)), (59, 22, 3164, 3164)),
            ((900, 2, 76544, 2), (None, None, XY, None), (900, 2, 2295337, 2)),
            ((64, 3200, 3200), (None, None, None), (64, 3200, 3200))):
        got = plan(shape, spec, ds, p)
        if isinstance(got, tuple):
            k, rows, lead = got
            row = 16
            for d in shape[k + 1:]:
                row *= d
            assert rows % p == 0 and lead * (rows // p) * row <= _FILE_ORDER_PIECE_BYTES
