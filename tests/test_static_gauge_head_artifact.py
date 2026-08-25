"""Focused SlabIO schema and photon-tile adapter gates."""

from dataclasses import replace
from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from file_io.slab_io import SlabIO
from file_io.static_gauge_head import (
    LoadedStaticGaugeHeadResponse,
    STATIC_GAUGE_HEAD_SCHEMA_VERSION,
    load_static_gauge_head_artifact,
    write_static_gauge_head_artifact,
)
from gw.head_correction import StaticGaugeHeadResponse
from gw.photon_layout import (
    PhotonBasisLayout,
    pack_photon_response_tiles,
    unpack_photon_response_tiles,
)


_OPERATOR_FINGERPRINT = "sha256:" + "a" * 64
_BODY_FINGERPRINT = "sha256:" + "b" * 64


def _mesh():
    devices = jax.devices()
    side = 2 if len(devices) >= 4 else 1
    return Mesh(
        np.asarray(devices[:side * side]).reshape(side, side),
        axis_names=("x", "y"),
    )


def _wfn(*, energy_shift=0.0):
    return SimpleNamespace(
        energies=np.asarray([[energy_shift, 0.2, 0.7, 1.1]],
                            dtype=np.float64),
        kpoints=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        nelec=2,
        nspinor=2,
        nbands=4,
        path=None,
    )


def _ward_closed_S():
    S = np.zeros((2, 2, 4, 4), dtype=np.complex128)
    S[0, 0, 0, 0] = 0.4
    S[0, 1, 0, 0] = S[1, 0, 0, 0] = 0.1
    S[1, 1, 0, 0] = 0.7
    beta = 1.3
    S[1, 1, 1, 1] = beta
    S[0, 0, 2, 2] = beta
    for a, b in ((0, 1), (1, 0)):
        S[a, b, 1, 2] = -0.5 * beta
        S[a, b, 2, 1] = -0.5 * beta
    return S


def _response(mesh):
    layout = PhotonBasisLayout.from_centroid_extents(1, 1, mesh)
    n_body = layout.packed_extent
    Y = (np.arange(2 * 4 * n_body).reshape(2, 4, n_body)
         + 0.25j).astype(np.complex128)
    Z = (np.arange(2 * n_body * 4).reshape(2, n_body, 4)
         - 0.5j).astype(np.complex128)
    return StaticGaugeHeadResponse(
        layout=layout,
        S_direct=jax.device_put(
            _ward_closed_S(), NamedSharding(mesh, P())),
        sigma_H=np.asarray((0.0, 0.0, 0.37), dtype=np.float64),
        Y_x=jax.device_put(Y, NamedSharding(mesh, P(None, None, "x"))),
        Z_y=jax.device_put(Z, NamedSharding(mesh, P(None, "y", None))),
        hamiltonian_config_operator_fingerprint=_OPERATOR_FINGERPRINT,
        operator_current_equivalent=True,
        contact_is_exact=True,
        ward_residual=0.0,
        hermiticity_residual=0.0,
    )


def _write(path, response, mesh, wfn):
    write_static_gauge_head_artifact(
        path,
        response,
        mesh_xy=mesh,
        wfn=wfn,
        band_start=0,
        band_stop=4,
        body_response_fingerprint=_BODY_FINGERPRINT,
        source_write_ibz_only=True,
        source_low_mem_bands=True,
    )


def _load(path, response, mesh, default_wfn, **changes):
    kwargs = dict(
        mesh_xy=mesh,
        wfn=default_wfn,
        expected_band_start=0,
        expected_band_stop=4,
        expected_layout=response.layout,
        expected_body_response_fingerprint=_BODY_FINGERPRINT,
        expected_hamiltonian_config_operator_fingerprint=(
            _OPERATOR_FINGERPRINT),
    )
    kwargs.update(changes)
    return load_static_gauge_head_artifact(path, **kwargs)


def test_static_gauge_head_roundtrip_is_sharded_sealed_and_immutable(tmp_path):
    mesh = _mesh()
    wfn = _wfn()
    response = _response(mesh)
    path = tmp_path / "static_gauge_head.h5"

    _write(path, response, mesh, wfn)
    assert path.is_file()
    assert not (tmp_path / "static_gauge_head.h5.partial").exists()

    loaded = _load(path, response, mesh, wfn)
    assert isinstance(loaded, LoadedStaticGaugeHeadResponse)
    assert not isinstance(response, LoadedStaticGaugeHeadResponse)
    assert loaded.source_write_ibz_only
    assert loaded.source_low_mem_bands
    assert loaded.band_start == 0 and loaded.band_stop == 4
    assert loaded.body_response_fingerprint == _BODY_FINGERPRINT
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(loaded.S_direct)),
        np.asarray(jax.device_get(response.S_direct)))
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(loaded.Y_x)),
        np.asarray(jax.device_get(response.Y_x)))
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(loaded.Z_y)),
        np.asarray(jax.device_get(response.Z_y)))
    np.testing.assert_array_equal(loaded.sigma_H, response.sigma_H)
    assert loaded.Y_x.sharding.is_equivalent_to(
        NamedSharding(mesh, P(None, None, "x")), 3)
    assert loaded.Z_y.sharding.is_equivalent_to(
        NamedSharding(mesh, P(None, "y", None)), 3)

    with pytest.raises(FileExistsError, match="immutable StaticGaugeHead"):
        _write(path, response, mesh, wfn)
    with pytest.raises(TypeError, match="issued only"):
        replace(loaded, _loader_token=object())


@pytest.mark.parametrize(
    "changes",
    (
        {"wfn": _wfn(energy_shift=0.01)},
        {"expected_band_stop": 3},
        {"expected_body_response_fingerprint": "sha256:" + "c" * 64},
        {"expected_hamiltonian_config_operator_fingerprint":
         "sha256:" + "d" * 64},
    ),
)
def test_static_gauge_head_refuses_mismatched_runtime_identity(
        tmp_path, changes):
    mesh = _mesh()
    wfn = _wfn()
    response = _response(mesh)
    path = tmp_path / "static_gauge_head.h5"
    _write(path, response, mesh, wfn)
    with pytest.raises(ValueError, match="static_gauge_head_artifact_mismatch"):
        _load(path, response, mesh, wfn, **changes)


def test_static_gauge_head_refuses_absent_partial_and_incomplete(tmp_path):
    mesh = _mesh()
    wfn = _wfn()
    response = _response(mesh)
    with pytest.raises(FileNotFoundError, match="artifact_absent"):
        _load(tmp_path / "absent.h5", response, mesh, wfn)
    with pytest.raises(ValueError, match="static_gauge_head_partial"):
        _load(tmp_path / "candidate.h5.partial", response, mesh, wfn)

    incomplete = tmp_path / "incomplete.h5"
    with SlabIO(incomplete, mode="w", mesh=mesh) as io:
        io.write_attr("complete", np.int32(0))
    with pytest.raises(ValueError, match="static_gauge_head_incomplete"):
        _load(incomplete, response, mesh, wfn)


def test_static_gauge_head_schema_version_has_a_red_twin(tmp_path):
    mesh = _mesh()
    wfn = _wfn()
    response = _response(mesh)
    path = tmp_path / "static_gauge_head.h5"
    _write(path, response, mesh, wfn)
    with SlabIO(path, mode="a", mesh=mesh) as io:
        io.write_attr(
            "schema_version",
            np.int32(STATIC_GAUGE_HEAD_SCHEMA_VERSION + 1))
    with pytest.raises(
            ValueError, match="static_gauge_head_artifact_mismatch"):
        _load(path, response, mesh, wfn)


def test_response_tile_adapters_delegate_canonical_pack_and_views():
    mesh = _mesh()
    layout = PhotonBasisLayout.from_centroid_extents(1, 1, mesh)
    nq = 2
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    block = np.full(
        layout.block_shape(nq, 0, 0), 3.0 + 2.0j, dtype=np.complex128)
    tile = jax.device_put(block, sharding)

    packed = pack_photon_response_tiles(
        {(0, 0): tile}, nq, layout, mesh)
    views = unpack_photon_response_tiles(packed, layout, mesh)

    expected = np.zeros_like(block)
    expected[:, :layout.logical_extent(0), :layout.logical_extent(0)] = (
        3.0 + 2.0j)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(views[0][0])), expected)
    for A in range(4):
        for B in range(4):
            if (A, B) != (0, 0):
                np.testing.assert_array_equal(
                    np.asarray(jax.device_get(views[A][B])), 0.0)
