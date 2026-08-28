"""Focused SlabIO roundtrip for the bounded photon-head completion receipt."""

from types import SimpleNamespace

import h5py
import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from file_io.static_gauge_head import (
    STATIC_PHOTON_HEAD_COMPLETION_SCHEMA_VERSION,
    read_static_photon_head_completion_receipt_h5,
    write_static_photon_head_completion_receipt_h5,
)


def _mesh():
    devices = jax.devices()
    side = 2 if len(devices) >= 4 else 1
    return Mesh(
        np.asarray(devices[:side * side]).reshape(side, side),
        axis_names=("x", "y"),
    )


@pytest.fixture(autouse=True)
def _provider_digest_door(monkeypatch):
    monkeypatch.setattr(
        "vcoul.slab_minibz_photon_receipt_digest",
        lambda receipt: receipt._provider_digest)


def _completion():
    bare = (
        np.arange(16, dtype=np.float64).reshape(4, 4)
        + 1j * np.arange(16, 32, dtype=np.float64).reshape(4, 4)
    ).astype(np.complex128)
    moments = (
        np.arange(3 * 3 * 4 * 4, dtype=np.float64).reshape(3, 3, 4, 4)
        - 0.25j
    ).astype(np.complex128)
    cubature = SimpleNamespace(
        method="true_ws_polygon_duffy_gauss_legendre_v1",
        orders=(16, 24, 32),
        physical_counts=(512, 1152, 2048),
        padded_counts=(2048, 2048, 2048),
        polygon_area=0.03125,
        cell_volume=128.0,
        weight_sum_defects=(1.0e-16, 2.0e-16, 3.0e-16),
        weighted_q_centroids=(
            (1.0e-17, -2.0e-17, 0.0),
            (2.0e-17, -3.0e-17, 0.0),
            (3.0e-17, -4.0e-17, 0.0),
        ),
        # A production provider carries its q/D/weight chunks here.  The
        # format writer must never inspect or persist them.
        chunks=object(),
        _provider_digest="a" * 64,
    )
    cubature_moments = SimpleNamespace(
        bare_D_mean=bare,
        screened_moments=moments,
        cubature_receipt=cubature,
        observed_physical_counts=cubature.physical_counts,
        observed_padded_solve_counts=cubature.padded_counts,
        max_backward_residual=1.25e-13,
        min_dyson_singular_value=0.125,
        max_dyson_condition_number=8.0,
        max_dyson_forward_error_bound=1.0e-12,
        mixed_scale_qstar=0.03125,
        mixed_convergence_error_ratios=(0.25, 0.125),
    )
    return SimpleNamespace(
        cubature=cubature_moments,
        ward_residual=2.0e-13,
        hermiticity_residual=3.0e-13,
        # Deliberately unrepresentable sentinels: neither the runtime carrier
        # nor identity framework fields belong to this derived evidence file.
        q0_factors=object(),
        hamiltonian_config_operator_fingerprint=object(),
    )


def test_static_photon_head_completion_receipt_roundtrip_is_bounded(tmp_path):
    completion = _completion()
    path = tmp_path / "static_slab_photon_head_completion.h5"
    metadata = write_static_photon_head_completion_receipt_h5(
        path, completion, mesh=_mesh())
    assert path.is_file()
    assert not (tmp_path / "static_slab_photon_head_completion.h5.partial").exists()
    loaded = read_static_photon_head_completion_receipt_h5(
        path, mesh=_mesh())

    assert loaded["schema_version"] == (
        STATIC_PHOTON_HEAD_COMPLETION_SCHEMA_VERSION)
    assert loaded["cubature_method"] == (
        completion.cubature.cubature_receipt.method)
    assert loaded["cubature_provider_digest"] == (
        completion.cubature.cubature_receipt._provider_digest)
    assert loaded["cubature_polygon_area"] == (
        completion.cubature.cubature_receipt.polygon_area)
    np.testing.assert_array_equal(
        loaded["cubature_weight_sum_defects"],
        completion.cubature.cubature_receipt.weight_sum_defects)
    np.testing.assert_array_equal(
        loaded["bare_D_mean"], completion.cubature.bare_D_mean)
    np.testing.assert_array_equal(
        loaded["screened_moments"], completion.cubature.screened_moments)
    np.testing.assert_array_equal(
        loaded["observed_physical_counts"],
        completion.cubature.observed_physical_counts)
    assert loaded["max_backward_residual"] == (
        completion.cubature.max_backward_residual)
    assert loaded["ward_residual"] == completion.ward_residual
    assert metadata["photon_head_completion_receipt_path"] == str(
        path.resolve())
    assert metadata["photon_head_completion_schema_version"] == "1"
    assert not any(
        token in key for key in metadata for token in ("bare", "moment"))

    with h5py.File(path, "r") as h5:
        assert tuple(h5["screened_moments"].shape) == (3, 3, 4, 4)
        forbidden = (
            "q0_factors", "chunks", "fingerprint", "bounded_response")
        assert not any(
            token in name for name in h5.keys() for token in forbidden)
    with pytest.raises(FileExistsError, match="immutable"):
        write_static_photon_head_completion_receipt_h5(
            path, completion, mesh=_mesh())


def test_static_photon_head_completion_receipt_refuses_wrong_tensor_shape(
        tmp_path):
    completion = _completion()
    completion.cubature.screened_moments = np.zeros(
        (9, 4, 4), dtype=np.complex128)
    path = tmp_path / "malformed.h5"
    with pytest.raises(ValueError, match="screened_moments.*shape"):
        write_static_photon_head_completion_receipt_h5(
            path, completion, mesh=_mesh())
    assert not path.exists()
