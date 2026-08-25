"""Focused local gates for the static gauge artifact and Lorentz adapters."""
from dataclasses import replace
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from file_io.static_gauge_head import (
    LoadedStaticGaugeHeadResponse,
    STATIC_GAUGE_HEAD_CONVENTION_ID,
    load_static_gauge_head_artifact,
    write_static_gauge_head_artifact,
)
from gw.gw_config import HeadCorrection
from gw.head_correction import StaticGaugeHeadResponse
from gw.photon_layout import (
    PhotonBasisLayout,
    pack_photon_response_tiles,
    unpack_photon_response_tiles,
)
from gw.w_isdf import compute_static_photon_response


_FINGERPRINT = "sha256:" + "6" * 64


def _mesh():
    devices = jax.devices()
    side = 2 if len(devices) >= 4 else 1
    return Mesh(
        np.asarray(devices[:side * side]).reshape(side, side),
        axis_names=("x", "y"))


def _response(mesh):
    layout = PhotonBasisLayout.from_centroid_extents(4, 4, mesh)
    n_body = layout.packed_extent
    S = np.zeros((2, 2, 4, 4), dtype=np.complex128)
    S[0, 0, 0, 0] = 0.31 + 0.0j
    S[0, 1, 0, 0] = S[1, 0, 0, 0] = -0.07 + 0.0j
    S[1, 1, 0, 0] = 0.43 + 0.0j
    rng = np.random.default_rng(925)
    Y = (rng.normal(size=(2, 4, n_body))
         + 1j * rng.normal(size=(2, 4, n_body))).astype(np.complex128)
    Z = (rng.normal(size=(2, n_body, 4))
         + 1j * rng.normal(size=(2, n_body, 4))).astype(np.complex128)
    return StaticGaugeHeadResponse(
        layout=layout,
        S_direct=jax.device_put(S, NamedSharding(mesh, P())),
        sigma_H=np.asarray((0.0, 0.0, 0.19), dtype=np.float64),
        Y_x=jax.device_put(
            Y, NamedSharding(mesh, P(None, None, "x"))),
        Z_y=jax.device_put(
            Z, NamedSharding(mesh, P(None, "y", None))),
        hamiltonian_config_operator_fingerprint=_FINGERPRINT,
        operator_current_equivalent=True,
        contact_is_exact=True,
        ward_residual=0.0,
        hermiticity_residual=0.0,
    )


@pytest.mark.parametrize(
    ("write_ibz_only", "low_mem_bands"), ((False, False), (True, True)))
def test_static_gauge_slabio_roundtrip_preserves_dtype_sharding_and_policy(
    tmp_path, write_ibz_only, low_mem_bands,
):
    mesh = _mesh()
    source = _response(mesh)
    path = tmp_path / (
        f"static_gauge_ibz{int(write_ibz_only)}_low{int(low_mem_bands)}.h5")
    write_static_gauge_head_artifact(
        path, source, mesh_xy=mesh,
        source_write_ibz_only=write_ibz_only,
        source_low_mem_bands=low_mem_bands)

    assert path.is_file()
    assert not (tmp_path / (path.name + ".partial")).exists()
    loaded = load_static_gauge_head_artifact(
        path, mesh_xy=mesh,
        expected_hamiltonian_config_operator_fingerprint=_FINGERPRINT)

    assert type(loaded) is LoadedStaticGaugeHeadResponse
    assert loaded.convention_id == STATIC_GAUGE_HEAD_CONVENTION_ID
    assert loaded.source_write_ibz_only is write_ibz_only
    assert loaded.source_low_mem_bands is low_mem_bands
    assert np.dtype(loaded.S_direct.dtype) == np.dtype(np.complex128)
    assert np.dtype(loaded.Y_x.dtype) == np.dtype(np.complex128)
    assert np.dtype(loaded.Z_y.dtype) == np.dtype(np.complex128)
    assert np.dtype(loaded.sigma_H.dtype) == np.dtype(np.float64)
    assert loaded.Y_x.sharding.is_equivalent_to(
        NamedSharding(mesh, P(None, None, "x")), 3)
    assert loaded.Z_y.sharding.is_equivalent_to(
        NamedSharding(mesh, P(None, "y", None)), 3)
    np.testing.assert_array_equal(np.asarray(loaded.S_direct),
                                  np.asarray(source.S_direct))
    np.testing.assert_array_equal(np.asarray(loaded.sigma_H), source.sigma_H)
    np.testing.assert_array_equal(np.asarray(loaded.Y_x), np.asarray(source.Y_x))
    np.testing.assert_array_equal(np.asarray(loaded.Z_y), np.asarray(source.Z_y))

    with pytest.raises(TypeError, match="issued only by"):
        replace(loaded, _loader_token=object())
    with pytest.raises(FileExistsError, match="immutable"):
        write_static_gauge_head_artifact(
            path, source, mesh_xy=mesh,
            source_write_ibz_only=write_ibz_only,
            source_low_mem_bands=low_mem_bands)


def test_static_gauge_loader_rejects_stale_identity_and_cannot_open_full(
    tmp_path,
):
    mesh = _mesh()
    source = _response(mesh)
    path = tmp_path / "static_gauge.h5"
    write_static_gauge_head_artifact(
        path, source, mesh_xy=mesh,
        source_write_ibz_only=True, source_low_mem_bands=True)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_static_gauge_head_artifact(
            path, mesh_xy=mesh,
            expected_hamiltonian_config_operator_fingerprint=(
                "sha256:" + "7" * 64))

    loaded = load_static_gauge_head_artifact(
        path, mesh_xy=mesh,
        expected_hamiltonian_config_operator_fingerprint=_FINGERPRINT)
    config = SimpleNamespace(
        head=SimpleNamespace(correction=HeadCorrection.FULL))
    with pytest.raises(
            ValueError, match="static_gauge_head_producer_unavailable"):
        compute_static_photon_response(
            None, None, None, None, None, mesh, config=config,
            gauge_head_response=loaded)


def test_response_tile_adapters_reuse_packer_and_zero_absent_channels():
    mesh = _mesh()
    layout = PhotonBasisLayout.from_centroid_extents(4, 4, mesh)
    nq = 3
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    tile_01 = jax.device_put(
        np.arange(nq * 16, dtype=np.float64).reshape(nq, 4, 4).astype(
            np.complex128), sharding)
    tile_32 = jax.device_put(
        (2j * np.ones((nq, 4, 4))).astype(np.complex128), sharding)

    packed = pack_photon_response_tiles(
        {(0, 1): tile_01, (3, 2): tile_32, (2, 2): None},
        nq, layout, mesh)
    tiles = unpack_photon_response_tiles(packed, layout, mesh)

    assert len(tiles) == 4 and all(len(row) == 4 for row in tiles)
    np.testing.assert_array_equal(np.asarray(tiles[0][1]),
                                  np.asarray(tile_01))
    np.testing.assert_array_equal(np.asarray(tiles[3][2]),
                                  np.asarray(tile_32))
    for A in range(4):
        for B in range(4):
            if (A, B) not in ((0, 1), (3, 2)):
                np.testing.assert_array_equal(np.asarray(tiles[A][B]), 0.0)
