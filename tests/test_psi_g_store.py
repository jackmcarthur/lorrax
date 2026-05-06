import numpy as np
import pytest

from common.psi_G_store import _zero_user_band_pad_in_shard


def test_zero_user_band_pad_in_shard_zeros_only_requested_pad_slots():
    data = np.ones((2, 4, 1, 1, 1, 1), dtype=np.complex128)
    data[:, :, 0, 0, 0, 0] *= np.arange(4)[None, :] + 1

    out = _zero_user_band_pad_in_shard(
        data,
        bc_range=(128, 192),
        shard_band_slice=slice(20, 24),
        user_band_stop=150,
    )

    assert out is not data
    np.testing.assert_array_equal(out[:, :2], data[:, :2])
    assert np.all(out[:, 2:] == 0.0)
    np.testing.assert_array_equal(data[:, :, 0, 0, 0, 0], [[1, 2, 3, 4], [1, 2, 3, 4]])


def test_zero_user_band_pad_in_shard_returns_original_when_no_pad_owned():
    data = np.ones((1, 3, 1, 1, 1, 1), dtype=np.complex128)

    out = _zero_user_band_pad_in_shard(
        data,
        bc_range=(64, 128),
        shard_band_slice=slice(0, 3),
        user_band_stop=150,
    )

    assert out is data


def test_zero_user_band_pad_in_shard_rejects_strided_band_slice():
    data = np.ones((1, 3, 1, 1, 1, 1), dtype=np.complex128)

    with pytest.raises(ValueError, match="contiguous"):
        _zero_user_band_pad_in_shard(
            data,
            bc_range=(0, 64),
            shard_band_slice=slice(0, 6, 2),
            user_band_stop=4,
        )
