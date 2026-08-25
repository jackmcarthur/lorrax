"""Small contracts for the shared finite-q representative plumbing."""
from pathlib import Path

import numpy as np
import pytest

from common import kq_mapping


_EXPECTED_SRC = Path(__file__).resolve().parents[1] / "src"
if _EXPECTED_SRC not in Path(kq_mapping.__file__).resolve().parents:
    raise RuntimeError(
        f"split-source gate: common.kq_mapping imported from "
        f"{kq_mapping.__file__}, expected {_EXPECTED_SRC}")


def test_bgw_signed_q_representative_keeps_positive_half_grid_tie():
    q = np.asarray([
        [0.0, 0.25, 0.5],
        [0.75, 1.0 - np.finfo(np.float64).eps, 0.5],
    ])
    got = kq_mapping.bgw_signed_q_representative(q)
    np.testing.assert_array_equal(
        got,
        np.asarray([[0.0, 0.25, 0.5], [-0.25, -np.finfo(np.float64).eps,
                                        0.5]]))


@pytest.mark.parametrize("bad", [
    np.zeros((2, 2)),
    np.asarray([np.nan, 0.0, 0.0]),
    np.asarray([-0.6, 0.0, 0.0]),
    np.asarray([1.1, 0.0, 0.0]),
])
def test_bgw_signed_q_representative_rejects_nonstored_rows(bad):
    with pytest.raises(ValueError, match="bgw_signed_q_representative"):
        kq_mapping.bgw_signed_q_representative(bad)
