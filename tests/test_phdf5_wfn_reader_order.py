import numpy as np
import pytest

from common.phdf5_wfn_reader import PhdfWfnReader


def _reader_stub(*, nk_full, ibz_per_full_k, kpt_starts):
    reader = object.__new__(PhdfWfnReader)
    reader.nk_full = int(nk_full)
    reader._ibz_per_full_k = np.asarray(ibz_per_full_k, dtype=np.int32)
    reader.kpt_starts = np.asarray(kpt_starts, dtype=np.int64)
    return reader


def test_k_read_order_reorders_nosym_file_order_to_requested_order():
    reader = _reader_stub(
        nk_full=6,
        ibz_per_full_k=np.arange(6, dtype=np.int32),
        kpt_starts=np.arange(6, dtype=np.int64) * 10,
    )

    k_ids, ibz_file_sorted, position = reader._k_read_order([5, 2, 5, 0])

    np.testing.assert_array_equal(k_ids, np.array([5, 2, 5, 0], dtype=np.int32))
    np.testing.assert_array_equal(ibz_file_sorted, np.array([0, 2, 5], dtype=np.int32))
    np.testing.assert_array_equal(position, np.array([2, 1, 2, 0], dtype=np.int32))

    union_read_file_order = np.array([100, 200, 500])
    np.testing.assert_array_equal(
        union_read_file_order[position],
        np.array([500, 200, 500, 100]),
    )


def test_k_read_order_dedupes_symmetry_related_full_kpoints():
    reader = _reader_stub(
        nk_full=5,
        ibz_per_full_k=np.array([3, 1, 3, 0, 1], dtype=np.int32),
        kpt_starts=np.array([300, 100, 200, 0], dtype=np.int64),
    )

    _k_ids, ibz_file_sorted, position = reader._k_read_order([0, 1, 2, 4])

    np.testing.assert_array_equal(ibz_file_sorted, np.array([3, 1], dtype=np.int32))
    np.testing.assert_array_equal(position, np.array([0, 1, 0, 1], dtype=np.int32))


def test_k_read_order_rejects_invalid_k_ids():
    reader = _reader_stub(
        nk_full=3,
        ibz_per_full_k=np.arange(3, dtype=np.int32),
        kpt_starts=np.arange(3, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="at least one"):
        reader._k_read_order([])
    with pytest.raises(ValueError, match=r"\[0, 3\)"):
        reader._k_read_order([0, 3])
    with pytest.raises(ValueError, match="1-D"):
        reader._k_read_order([[0, 1]])
