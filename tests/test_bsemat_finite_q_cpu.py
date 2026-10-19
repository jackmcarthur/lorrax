"""Finite-Q bsemat bookkeeping (``bse.bsemat_export.finite_q_legs``, the header), host only.

CPU clause of the four-GPU rule: index arithmetic and file layout, no mesh. The
finite-Q kernel itself is verified end to end by BerkeleyGW absorption on the file
against ``bse.exciton_bands`` at the same Q (docs/theory/bgw-bsemat-export.md#finite-q).
"""
import os

import h5py
import numpy as np
import pytest

GRID = np.array([3, 3, 1])
K = np.array([[i / 3, j / 3, 0.0] for i in range(3) for j in range(3)])
FIXTURE_WFN = os.path.join(os.path.dirname(__file__), "hsuite", "fixture", "WFN.h5")


def _gi(frac):
    from bse.bsemat_export import grid_index
    return grid_index(np.asarray(frac, float), GRID)


def test_q_zero_and_integer_q_are_qflag_1():
    from bse.bsemat_export import finite_q_legs
    for Q in ((0, 0, 0), (1, 0, 0), (0, -1, 0)):
        qflag, c, v = finite_q_legs(K, GRID, Q)
        assert qflag == 1 and np.array_equal(c, v) and np.array_equal(c, _gi(K))


def test_on_grid_q_pairs_are_exciton_bands_pairs():
    """qflag 2: row j holds (hole k_j − Q, electron k_j); as a set this is |v k, c k+Q⟩."""
    from bse.bsemat_export import finite_q_legs
    Q = np.array([1 / 3, 0.0, 0.0])
    qflag, c, v = finite_q_legs(K, GRID, Q)
    assert qflag == 2
    assert np.array_equal(v, _gi(K - Q)) and np.array_equal(c, _gi(K))
    file_pairs = set(zip(v.tolist(), c.tolist()))
    exb_pairs = set(zip(_gi(K).tolist(), _gi(K + Q).tolist()))       # (v at k, c at k+Q)
    assert file_pairs == exb_pairs
    # Red twin: the opposite sign (hole at k + Q) is the exciton at −Q, a different pair set.
    wrong = set(zip(_gi(K + Q).tolist(), _gi(K).tolist()))
    assert wrong != exb_pairs


def test_off_grid_q_needs_the_shifted_wfn_co():
    from bse.bsemat_export import finite_q_legs
    Q = np.array([1 / 6, 0.0, 0.0])
    qflag, c, v = finite_q_legs(K + Q, GRID, Q)                    # WFN_co = grid + Q
    assert qflag == 0 and c is None and np.array_equal(v, _gi(K))
    with pytest.raises(ValueError, match="GATE bsemat_finite_q_grid"):
        finite_q_legs(K, GRID, Q)                                  # the unshifted WFN_co


@pytest.mark.skipif(not os.path.exists(FIXTURE_WFN), reason="hsuite fixture WFN absent")
def test_header_carries_qflag_and_minus_q(tmp_path):
    from file_io import bgw_bsemat as F
    kw = dict(wfn_file=FIXTURE_WFN, kpts=K, kgrid=GRID, nvb=1, ncb=1, extended=False, icutv=0,
              ecuts=5.0, ecutg=5.0, efermi=0.0, nspinor=1)
    F.create_bsemat_file(str(tmp_path / "q0.h5"), **kw)
    F.create_bsemat_file(str(tmp_path / "q.h5"), qflag=2, exciton_Q_shift=(-1 / 3, 0, 0), **kw)
    h0, h = F.read_bsemat_header(str(tmp_path / "q0.h5")), F.read_bsemat_header(str(tmp_path / "q.h5"))
    assert int(h0["kpoints/qflag"]) == 1 and np.all(h0["kpoints/exciton_Q_shift"] == 0)
    assert int(h["kpoints/qflag"]) == 2
    assert np.allclose(h["kpoints/exciton_Q_shift"], [-1 / 3, 0, 0], rtol=0, atol=1e-15)
    with h5py.File(str(tmp_path / "q.h5"), "r") as f:
        assert f["bse_header/kpoints/exciton_Q_shift"].dtype == np.float64
