import pytest

from file_io._slab_io_ffi import (
    _normalize_slab_request,
    _validate_block_divisible,
)


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
