import pytest
import numpy as np

from file_io._slab_io_ffi import (
    _normalize_slab_request,
    _normalize_valid_shape,
    _validate_block_divisible,
)
from file_io.slab_io import SlabIO


def test_normalize_slab_request_rejects_rank_mismatch():
    with pytest.raises(ValueError, match="rank mismatch"):
        _normalize_slab_request(
            op="write_slab",
            name="zeta_q",
            offset=(0, 0),
            slab_shape=(36, 70000, 1500),
            global_shape=(36, 1125000, 1500),
        )


def test_normalize_slab_request_rejects_out_of_bounds():
    with pytest.raises(ValueError, match=r"dim 1: 1120000\+70000>1125000"):
        _normalize_slab_request(
            op="write_slab",
            name="zeta_q",
            offset=(0, 1120000, 0),
            slab_shape=(36, 70000, 1500),
            global_shape=(36, 1125000, 1500),
        )


def test_normalize_slab_request_allows_nonzero_read_offset_without_extent():
    off, shape, gshape = _normalize_slab_request(
        op="read_slab",
        name="zeta_q",
        offset=(35, 0, 1300),
        slab_shape=(1, 1125000, 200),
        global_shape=None,
        check_bounds=False,
    )
    assert off == (35, 0, 1300)
    assert shape == (1, 1125000, 200)
    assert gshape == shape


def test_validate_block_divisible_rejects_ragged_write_axis():
    with pytest.raises(ValueError, match="dimension 1 size 70001"):
        _validate_block_divisible(
            op="write_slab",
            name="zeta_q",
            shape=(36, 70001, 1500),
            axis_count_per_dim=(0, 2, 0),
            axis_flat=(0, 1),
            mesh_shape=(4, 4),
        )


def test_validate_block_divisible_accepts_cri3_zeta_shape():
    _validate_block_divisible(
        op="write_slab",
        name="zeta_q",
        shape=(36, 70000, 1500),
        axis_count_per_dim=(0, 2, 0),
        axis_flat=(0, 1),
        mesh_shape=(4, 4),
    )


def test_normalize_valid_shape_rejects_prefix_larger_than_physical_slab():
    with pytest.raises(ValueError, match="valid_shape exceeds slab shape"):
        _normalize_valid_shape(
            op="write_slab",
            name="zeta_q",
            valid_shape=(36, 70001, 1500),
            slab_shape=(36, 70000, 1500),
            offset=(0, 0, 0),
            global_shape=(36, 1125000, 1500),
        )


def test_normalize_valid_shape_allows_ragged_last_file_chunk():
    vshape = _normalize_valid_shape(
        op="write_slab",
        name="zeta_q",
        valid_shape=(36, 65432, 1500),
        slab_shape=(36, 70000, 1500),
        offset=(0, 1050000, 0),
        global_shape=(36, 1125432, 1500),
    )
    assert vshape == (36, 65432, 1500)


def test_allgather_valid_shape_write_and_zero_padded_read(tmp_path):
    path = tmp_path / "slab.h5"
    physical = np.arange(12, dtype=np.float64).reshape(3, 4)
    with SlabIO(str(path), mode="w") as io:
        io.write_slab(
            "A",
            physical,
            global_shape=(3, 3),
            valid_shape=(3, 3),
        )

    with SlabIO(str(path), mode="r") as io:
        host = io.read_slab(
            "A",
            shape=(3, 4),
            valid_shape=(3, 3),
            as_numpy=True,
        )

    expected = np.zeros((3, 4), dtype=np.float64)
    expected[:, :3] = physical[:, :3]
    np.testing.assert_array_equal(host, expected)
