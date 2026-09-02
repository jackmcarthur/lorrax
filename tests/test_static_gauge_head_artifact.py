"""SlabIO gates for the immutable Hall artifact and the photon-tile adapters."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.parallel_transport import wfn_fingerprint
from file_io.slab_io import SlabIO
from file_io.static_gauge_head import (
    STATIC_GAUGE_HALL_SCHEMA_VERSION,
    load_static_gauge_hall_artifact,
    write_static_gauge_hall_artifact,
)
from gw.photon_layout import (
    PhotonBasisLayout,
    pack_photon_response_tiles,
    unpack_photon_response_tiles,
)
from gw.qsgw_head import (
    StaticGaugeHallTransaction,
    _static_gauge_hall_transaction_from_artifact,
)


_OPERATOR_FINGERPRINT = "sha256:" + "a" * 64
_SIGMA_H = (1.9528890297769742e-11, 4.1597624292025984e-11,
            -4.223240852693869e-08)


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


def _transaction(mesh, wfn):
    return _static_gauge_hall_transaction_from_artifact(
        sigma_H=np.asarray(_SIGMA_H, dtype=np.float64),
        hamiltonian_config_operator_fingerprint=_OPERATOR_FINGERPRINT,
        wfn_fingerprint=wfn_fingerprint(wfn),
        band_start=0, band_stop=4, nk_tot=6, mesh=mesh)


def _load(path, mesh, wfn=None, **changes):
    """Load with the writer's identity unless ``changes`` overrides one."""
    kwargs = dict(
        mesh_xy=mesh, wfn=_wfn() if wfn is None else wfn,
        expected_band_start=0, expected_band_stop=4, expected_nk_tot=6)
    kwargs.update(changes)
    return load_static_gauge_hall_artifact(path, **kwargs)


def test_hall_artifact_roundtrip_is_sealed_and_immutable(tmp_path):
    mesh = _mesh()
    wfn = _wfn()
    transaction = _transaction(mesh, wfn)
    path = tmp_path / "static_gauge_hall.h5"

    write_static_gauge_hall_artifact(path, transaction, mesh_xy=mesh)
    assert path.is_file()
    assert not (tmp_path / "static_gauge_hall.h5.partial").exists()

    loaded = _load(path, mesh, wfn)
    assert isinstance(loaded, StaticGaugeHallTransaction)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(loaded.sigma_H)),
        np.asarray(_SIGMA_H, dtype=np.float64))
    assert loaded.sigma_H.sharding.is_equivalent_to(
        NamedSharding(mesh, P()), 1)
    assert (loaded.band_start, loaded.band_stop, loaded.nk_tot) == (0, 4, 6)
    assert loaded.wfn_fingerprint == wfn_fingerprint(wfn)
    assert (loaded.hamiltonian_config_operator_fingerprint
            == _OPERATOR_FINGERPRINT)

    with pytest.raises(FileExistsError, match="immutable StaticGaugeHall"):
        write_static_gauge_hall_artifact(path, transaction, mesh_xy=mesh)
    with pytest.raises(TypeError, match="sealed canonical Hall"):
        write_static_gauge_hall_artifact(
            tmp_path / "other.h5", SimpleNamespace(sigma_H=np.zeros(3)),
            mesh_xy=mesh)


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        ({"wfn": _wfn(energy_shift=0.01)}, "WFN identity differs"),
        ({"expected_band_stop": 3}, "band_stop=4, expected 3"),
        ({"expected_nk_tot": 5}, "nk_tot=6, expected 5"),
    ),
)
def test_hall_artifact_refuses_mismatched_runtime_identity(
        tmp_path, changes, message):
    mesh = _mesh()
    wfn = _wfn()
    path = tmp_path / "static_gauge_hall.h5"
    write_static_gauge_hall_artifact(path, _transaction(mesh, wfn), mesh_xy=mesh)
    with pytest.raises(ValueError, match=message):
        _load(path, mesh, **changes)


def test_hall_artifact_refuses_absent_partial_and_incomplete(tmp_path):
    mesh = _mesh()
    wfn = _wfn()
    with pytest.raises(FileNotFoundError, match="artifact_absent"):
        _load(tmp_path / "absent.h5", mesh, wfn)
    with pytest.raises(ValueError, match="static_gauge_hall_partial"):
        _load(tmp_path / "candidate.h5.partial", mesh, wfn)

    path = tmp_path / "static_gauge_hall.h5"
    write_static_gauge_hall_artifact(path, _transaction(mesh, wfn), mesh_xy=mesh)
    with SlabIO(path, mode="a", mesh=mesh) as io:
        io.write_attr("complete", np.int32(0))
    with pytest.raises(ValueError, match="incomplete"):
        _load(path, mesh, wfn)

    bare = tmp_path / "bare.h5"
    with SlabIO(bare, mode="w", mesh=mesh) as io:
        io.write_attr("complete", np.int32(1))
    with pytest.raises(ValueError, match="static_gauge_hall_schema"):
        _load(bare, mesh, wfn)


def test_hall_artifact_schema_version_has_a_red_twin(tmp_path):
    mesh = _mesh()
    wfn = _wfn()
    path = tmp_path / "static_gauge_hall.h5"
    write_static_gauge_hall_artifact(path, _transaction(mesh, wfn), mesh_xy=mesh)
    with SlabIO(path, mode="a", mesh=mesh) as io:
        io.write_attr(
            "schema_version",
            np.int32(STATIC_GAUGE_HALL_SCHEMA_VERSION + 1))
    with pytest.raises(ValueError, match="schema_version"):
        _load(path, mesh, wfn)


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
