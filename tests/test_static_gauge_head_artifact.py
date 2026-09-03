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
from gw.mpa.sample_plan import faraday_imaginary_plan, plan_z


_OPERATOR_FINGERPRINT = "sha256:" + "a" * 64
_SIGMA_H = (1.9528890297769742e-11, 4.1597624292025984e-11,
            -4.223240852693869e-08)
_HALL_FREQUENCIES = np.asarray([0.0 + 0.0j, 0.0 + 2.0j])


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


def _transaction(mesh, wfn, *, sigma_H=_SIGMA_H):
    static = np.asarray(sigma_H, dtype=np.float64)
    return _static_gauge_hall_transaction_from_artifact(
        frequencies_ry=_HALL_FREQUENCIES,
        sigma_H_frequency=np.stack((static, 0.75 * static)).astype(
            np.complex128),
        hamiltonian_config_operator_fingerprint=_OPERATOR_FINGERPRINT,
        wfn_fingerprint=wfn_fingerprint(wfn),
        band_start=0, band_stop=4, nk_tot=6, mesh=mesh)


def _transaction_v3(mesh, wfn):
    plan = faraday_imaginary_plan(3, 2.0)
    frequencies = plan_z(plan)
    sigma = np.arange(frequencies.size * 3, dtype=np.float64).reshape(-1, 3)
    transaction = _static_gauge_hall_transaction_from_artifact(
        frequencies_ry=frequencies,
        sigma_H_frequency=sigma.astype(np.complex128),
        hamiltonian_config_operator_fingerprint=_OPERATOR_FINGERPRINT,
        wfn_fingerprint=wfn_fingerprint(wfn),
        band_start=0, band_stop=4, nk_tot=6, mesh=mesh,
        artifact_schema_version=3,
        sample_plan_label=plan["label"],
        sample_plan_n_poles=plan["n_poles"],
        sample_plan_alpha=plan["sampling_alpha"],
        sample_plan_schedule=plan["sampling_schedule"],
        sample_plan_omega_max_ry=plan["omega_max_ry"],
    )
    return transaction, plan


def _load(path, mesh, wfn=None, **changes):
    """Load with the writer's identity unless ``changes`` overrides one."""
    kwargs = dict(
        mesh_xy=mesh, wfn=_wfn() if wfn is None else wfn,
        expected_band_start=0, expected_band_stop=4, expected_nk_tot=6)
    kwargs.update(changes)
    return load_static_gauge_hall_artifact(path, **kwargs)


def _static_bare_config(tmp_path, hall_path):
    """Return the real resolved config whose Hall admission is under test."""
    from gw.gw_config import LorraxConfig

    deck = tmp_path / "static_bare.in"
    deck.write_text(
        "[cohsex]\n"
        "nval = 2\n"
        "ncond = 2\n"
        "nband = 10\n"
        "memory_per_device_gb = 4.0\n"
        "bispinor = true\n"
        "bispinor_gw = bare_transverse\n"
        "sys_dim = 2\n"
        "qp_solver = one_shot_dft\n"
        "head_correction = full\n"
        "compute_mode = cohsex\n"
        f"static_gauge_hall_file = {hall_path}\n",
        encoding="utf-8",
    )
    return LorraxConfig.from_input_file(
        str(deck), print_fn=lambda *args, **kwargs: None)


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
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(loaded.frequencies_ry)),
        _HALL_FREQUENCIES)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(loaded.sigma_H_at(2.0j))),
        0.75 * np.asarray(_SIGMA_H))
    assert loaded.sigma_H.sharding.is_equivalent_to(
        NamedSharding(mesh, P()), 1)
    assert loaded.sigma_H_frequency.sharding.is_equivalent_to(
        NamedSharding(mesh, P(None, None)), 2)
    assert (loaded.band_start, loaded.band_stop, loaded.nk_tot) == (0, 4, 6)
    assert loaded.wfn_fingerprint == wfn_fingerprint(wfn)
    assert (loaded.hamiltonian_config_operator_fingerprint
            == _OPERATOR_FINGERPRINT)
    with pytest.raises(ValueError, match="static_gauge_hall_frequency_missing"):
        loaded.sigma_H_at(1.0j)

    with pytest.raises(FileExistsError, match="immutable StaticGaugeHall"):
        write_static_gauge_hall_artifact(path, transaction, mesh_xy=mesh)
    with pytest.raises(TypeError, match="sealed canonical Hall"):
        write_static_gauge_hall_artifact(
            tmp_path / "other.h5", SimpleNamespace(sigma_H=np.zeros(3)),
            mesh_xy=mesh)


def test_hall_artifact_v3_roundtrip_stamps_nested_mpa_plan(tmp_path):
    mesh = _mesh()
    wfn = _wfn()
    transaction, plan = _transaction_v3(mesh, wfn)
    path = tmp_path / "static_gauge_hall_v3.h5"

    write_static_gauge_hall_artifact(path, transaction, mesh_xy=mesh)
    loaded = _load(path, mesh, wfn)
    assert loaded.artifact_schema_version == STATIC_GAUGE_HALL_SCHEMA_VERSION
    assert loaded.sample_plan_label == plan["label"]
    assert loaded.sample_plan_n_poles == 3
    assert loaded.sample_plan_alpha == 1
    assert loaded.sample_plan_schedule == "nested"
    assert loaded.sample_plan_omega_max_ry == 2.0
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(loaded.frequencies_ry)),
        np.asarray([0.0j, 0.25j, 0.5j, 1.0j, 1.5j, 2.0j]))


def test_hall_artifact_v3_refuses_tampered_sample_plan_stamp(tmp_path):
    mesh = _mesh()
    wfn = _wfn()
    transaction, _ = _transaction_v3(mesh, wfn)
    path = tmp_path / "static_gauge_hall_v3.h5"
    write_static_gauge_hall_artifact(path, transaction, mesh_xy=mesh)
    with SlabIO(path, mode="a", mesh=mesh) as io:
        io.write_attr("sample_plan_alpha", np.int32(2))
    with pytest.raises(
            ValueError, match="static_gauge_hall_sample_plan_provenance"):
        _load(path, mesh, wfn)


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
    with pytest.raises(FileNotFoundError, match="static_gauge_hall_file_missing") as exc:
        _load(tmp_path / "absent.h5", mesh, wfn)
    for part in ("got:", "want:", "why:"):
        assert part in str(exc.value)
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


def test_bare_route_accepts_authenticated_exact_zero_hall(tmp_path):
    """Positive control for the narrowed bare-route Hall refusal.

    The exact consumer tensor is identically zero, so this admitted named
    artifact produces the same Hall contribution as the unnamed default.
    """
    from gw.head_correction import static_hall_linear_response
    from gw.w_isdf import _load_static_photon_hall

    mesh = _mesh()
    wfn = _wfn()
    path = tmp_path / "zero_hall.h5"
    write_static_gauge_hall_artifact(
        path, _transaction(mesh, wfn, sigma_H=np.zeros(3)), mesh_xy=mesh)
    config = _static_bare_config(tmp_path, path)
    meta = SimpleNamespace(b_id_0=0, b_id_4_chi_user=4, nk_tot=6)

    hall = _load_static_photon_hall(
        config, meta, mesh, wfn, None, screen_current=False,
        print_fn=lambda *args, **kwargs: None)
    np.testing.assert_array_equal(np.asarray(jax.device_get(hall.sigma_H)), 0.0)
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(static_hall_linear_response(
            hall.sigma_H, dimension=2))),
        0.0)


def test_bare_route_still_refuses_authenticated_nonzero_hall(tmp_path):
    """Negative control: a nonzero Gamma-only CT/TC block still refuses."""
    from gw.w_isdf import _load_static_photon_hall

    mesh = _mesh()
    wfn = _wfn()
    path = tmp_path / "nonzero_hall.h5"
    write_static_gauge_hall_artifact(
        path, _transaction(mesh, wfn), mesh_xy=mesh)
    config = _static_bare_config(tmp_path, path)
    meta = SimpleNamespace(b_id_0=0, b_id_4_chi_user=4, nk_tot=6)

    with pytest.raises(
            ValueError,
            match="(?s)packed_bare_transverse_hall_unavailable.*got:.*want:.*why:"):
        _load_static_photon_hall(
            config, meta, mesh, wfn, None, screen_current=False,
            print_fn=lambda *args, **kwargs: None)


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
