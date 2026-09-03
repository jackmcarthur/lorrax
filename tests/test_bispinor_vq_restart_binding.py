"""Host-side gates for packed-photon restart composition and readiness."""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import h5py
import jax
import numpy as np
import pytest
from jax.sharding import Mesh

from common.bispinor_init import KINETIC_BALANCE_LIFT_PROVENANCE
from common.parallel_transport import WFN_FINGERPRINT_SCHEME
from common.wfn_transforms import FULL_BLOCH_TRANSFORM_SCHEME
from file_io.bispinor_vq_restart import (
    BISPINOR_VQ_RESTART_BINDING_DATASET,
    BispinorVqRestartBinding,
    authenticate_or_recover_bispinor_vq_restart_binding,
    assert_bispinor_vq_restart_binding,
    read_bispinor_vq_restart_binding,
)
from file_io.isdf_header import IsdfHeader, write_isdf_header
from file_io.mf_header import copy_mf_header
from file_io.qp_wfn import encode_qp_state_source_provenance
from file_io.tagged_arrays import (
    format_coulomb_policy,
    read_photon_g0_vectors_from_h5,
    read_ready_w0_qmunu_from_h5,
    write_restart_state_to_h5,
)
from file_io.wfn_basis import (
    CENTROID_TABLE_FINGERPRINT_SCHEME,
    WavefunctionBasisReceipt,
    centroid_table_md5,
)
from tests.test_file_io import _make_fake_wfn


def _basis(role: str, *, pad: int = 8, centroid: str = "1" * 32):
    return WavefunctionBasisReceipt(
        role=role,
        wfn_fingerprint_scheme=WFN_FINGERPRINT_SCHEME,
        wfn_fingerprint="a" * 64,
        band_interval=(2, 12),
        fft_grid=(8, 8, 4),
        centroid_fingerprint_scheme=CENTROID_TABLE_FINGERPRINT_SCHEME,
        centroid_table_md5=centroid,
        n_rmu_logical=7,
        n_rmu_padded=pad,
        source_identity=FULL_BLOCH_TRANSFORM_SCHEME,
        nspinor_sampled=4,
        bispinor_lift_provenance=KINETIC_BALANCE_LIFT_PROVENANCE,
    )


def _binding(*, pad=8, provenance=None, policy="v1;sys_dim=2"):
    return BispinorVqRestartBinding.from_sources(
        v_qmunu_format="bispinor_lorentz_v2",
        zeta_fit_provenance=(
            provenance or ("charge", "current-1", "current-2", "current-3")),
        charge_basis_receipt=_basis("charge", pad=pad),
        transverse_basis_receipt=_basis(
            "transverse", pad=pad, centroid="2" * 32),
        coulomb_policy=policy,
    )


def _write_binding(path, binding=None):
    with h5py.File(path, "w") as h5:
        h5["payload"] = np.zeros((1,), dtype=np.float64)
        if binding is not None:
            h5[BISPINOR_VQ_RESTART_BINDING_DATASET] = binding.encode()


def _write_pre_schema_zeta(
    path, *, wfn_path, centroid_idx, channel, payload, provenance,
):
    """Write a complete tiny G-flat zeta with no packed-restart additions."""
    copy_mf_header(wfn_path, path, dst_mode="w")
    nq, nmu, ngk = payload.shape
    gvec = np.zeros((nq, 3, ngk), dtype=np.int32)
    ngk_per_q = np.full((nq,), ngk, dtype=np.int32)
    header = IsdfHeader.build(
        r_mu_fft_idx=centroid_idx, fft_grid=(16, 16, 16),
        density="scalar", vertex_mu_L=channel, zeta_is_done=True,
        zeta_layout="G_flat", gvec_components=gvec,
        ngk_per_q=ngk_per_q, zeta_cutoff_ry=10.0,
        fit_provenance=provenance,
    )
    write_isdf_header(path, header, mode="a")
    with h5py.File(path, "a") as h5:
        h5.create_dataset("zeta_q_G", data=payload)


def _pre_schema_fixture(tmp_path):
    """Return a legacy artifact family and its fresh-path G=0 oracle."""
    kgrid = (2, 3, 4)
    nq = int(np.prod(kgrid))
    charge_idx = np.arange(21, dtype=np.int32).reshape(7, 3) % 16
    current_idx = (np.arange(21, dtype=np.int32).reshape(7, 3) + 3) % 16
    charge_basis = replace(
        _basis("charge"), fft_grid=(16, 16, 16),
        centroid_table_md5=centroid_table_md5(charge_idx))
    current_basis = replace(
        _basis("transverse", centroid="2" * 32),
        fft_grid=(16, 16, 16),
        centroid_table_md5=centroid_table_md5(current_idx))
    policy = format_coulomb_policy({"sys_dim": "2"})

    restart = tmp_path / "isdf_tensors_7.h5"
    source_record = {
        "schema": 1,
        "wfn_fingerprint_scheme": charge_basis.wfn_fingerprint_scheme,
        "wfn_fingerprint": charge_basis.wfn_fingerprint,
        "qp_wfn_stamp": None,
    }
    with h5py.File(restart, "w") as h5:
        h5["payload"] = np.zeros((1,), dtype=np.float64)
        h5["qp_state_source_provenance"] = (
            encode_qp_state_source_provenance(source_record))
        h5["coulomb_policy"] = np.asarray(policy.encode(), dtype="S")
        h5.attrs["centroids_charge_md5"] = charge_basis.centroid_table_md5
        h5.attrs["centroids_transverse_md5"] = (
            current_basis.centroid_table_md5)

    tile = tmp_path / "v_q_bispinor.h5"
    with h5py.File(tile, "w") as h5:
        h5["v_qmunu_format"] = np.bytes_("bispinor_lorentz_v2")
        h5["kgrid"] = np.asarray(kgrid, dtype=np.int64)
        h5["n_rmu_C"] = np.int64(7)
        h5["n_rmu_T"] = np.int64(7)
        h5["n_q_total"] = np.int64(nq)

    wfn_path = tmp_path / "WFN.h5"
    _make_fake_wfn(str(wfn_path))
    rng = np.random.default_rng(913)
    zeta_paths = []
    payloads = []
    provenance = tuple(f"fit-{channel}" for channel in range(4))
    for channel in range(4):
        payload = (
            rng.standard_normal((nq, 7, 3))
            + 1j * rng.standard_normal((nq, 7, 3))).astype(np.complex128)
        path = tmp_path / ("zeta_q.h5" if channel == 0
                           else f"zeta_q_mu{channel}.h5")
        _write_pre_schema_zeta(
            path, wfn_path=wfn_path,
            centroid_idx=(charge_idx if channel == 0 else current_idx),
            channel=channel, payload=payload,
            provenance=provenance[channel])
        zeta_paths.append(path)
        payloads.append(payload)
    expected = BispinorVqRestartBinding.from_sources(
        v_qmunu_format="bispinor_lorentz_v2",
        zeta_fit_provenance=provenance,
        charge_basis_receipt=charge_basis,
        transverse_basis_receipt=current_basis,
        coulomb_policy=policy,
    )
    return {
        "restart": restart, "tile": tile, "zeta_paths": tuple(zeta_paths),
        "payloads": tuple(payloads), "charge_basis": charge_basis,
        "current_basis": current_basis, "policy": policy,
        "kgrid": kgrid, "expected": expected,
    }


def test_binding_roundtrip_excludes_runtime_padding(tmp_path):
    p4 = _binding(pad=8)
    p16 = _binding(pad=16)
    assert p4 == p16
    assert b"n_rmu_padded" not in bytes(p4.encode())
    path = tmp_path / "binding.h5"
    _write_binding(path, p4)
    assert read_bispinor_vq_restart_binding(path) == p4


def test_matching_artifacts_authenticate(tmp_path):
    binding = _binding()
    restart = tmp_path / "restart.h5"
    photon = tmp_path / "v_q_bispinor.h5"
    _write_binding(restart, binding)
    _write_binding(photon, binding)
    assert assert_bispinor_vq_restart_binding(
        restart_path=restart, v_q_path=photon) == binding


def test_removed_tile_refuses_by_exact_gate_name(tmp_path):
    restart = tmp_path / "restart.h5"
    _write_binding(restart, _binding())
    with pytest.raises(ValueError) as exc:
        assert_bispinor_vq_restart_binding(
            restart_path=restart,
            v_q_path=tmp_path / "removed_v_q_bispinor.h5")
    assert "GATE bispinor_packed_restart_binding_missing" in str(exc.value)


def test_foreign_same_shape_tile_refuses_by_exact_gate_name(tmp_path):
    restart = tmp_path / "restart.h5"
    photon = tmp_path / "v_q_bispinor.h5"
    _write_binding(restart, _binding())
    _write_binding(photon, replace(
        _binding(), zeta_fit_provenance=("foreign", "b", "c", "d")))
    with pytest.raises(ValueError) as exc:
        assert_bispinor_vq_restart_binding(
            restart_path=restart, v_q_path=photon)
    assert "GATE bispinor_packed_restart_binding_mismatch" in str(exc.value)
    assert "zeta_fit_provenance" in str(exc.value)


def test_gamma_and_w0_readiness_refuse_before_distributed_io(tmp_path):
    path = tmp_path / "restart.h5"
    with h5py.File(path, "w"):
        pass
    with pytest.raises(ValueError) as gamma:
        read_photon_g0_vectors_from_h5(
            path, object(), n_rmu_charge_logical=7,
            n_rmu_transverse_logical=7)
    assert "GATE bispinor_packed_restart_gamma_missing" in str(gamma.value)
    with pytest.raises(ValueError) as w0:
        read_ready_w0_qmunu_from_h5(
            path, object(), n_rmu_logical=7)
    assert "GATE bispinor_packed_restart_w0_missing" in str(w0.value)


def test_pre_schema_restart_recovers_fresh_literal_gamma_vectors(tmp_path):
    fixture = _pre_schema_fixture(tmp_path)
    binding, recovered = authenticate_or_recover_bispinor_vq_restart_binding(
        restart_path=fixture["restart"], v_q_path=fixture["tile"],
        zeta_paths=fixture["zeta_paths"],
        charge_basis_receipt=fixture["charge_basis"],
        transverse_basis_receipt=fixture["current_basis"],
        coulomb_policy=fixture["policy"],
        expected_kgrid=fixture["kgrid"],
        expected_v_qmunu_format="bispinor_lorentz_v2")
    assert recovered is True
    assert binding == fixture["expected"]

    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    vectors = read_photon_g0_vectors_from_h5(
        fixture["restart"], mesh,
        n_rmu_charge_logical=7, n_rmu_transverse_logical=7,
        pre_schema_zeta_paths=fixture["zeta_paths"])
    for got, payload in zip(vectors, fixture["payloads"]):
        np.testing.assert_allclose(
            np.asarray(jax.device_get(got)), payload[:, :, 0],
            rtol=0.0, atol=1.0e-14)


def test_pre_schema_restart_mismatched_centroid_table_refuses(tmp_path):
    fixture = _pre_schema_fixture(tmp_path)
    with h5py.File(fixture["zeta_paths"][2], "a") as h5:
        h5["isdf_header/centroids/r_mu_fft_idx"][0, 0] += 1
    with pytest.raises(ValueError) as exc:
        authenticate_or_recover_bispinor_vq_restart_binding(
            restart_path=fixture["restart"], v_q_path=fixture["tile"],
            zeta_paths=fixture["zeta_paths"],
            charge_basis_receipt=fixture["charge_basis"],
            transverse_basis_receipt=fixture["current_basis"],
            coulomb_policy=fixture["policy"],
            expected_kgrid=fixture["kgrid"],
            expected_v_qmunu_format="bispinor_lorentz_v2")
    assert "GATE bispinor_pre_schema_restart_provenance_mismatch" in str(
        exc.value)
    assert "zeta centroid table" in str(exc.value)


def test_restart_writer_transports_the_exact_binding(monkeypatch, tmp_path):
    queued = {}

    class FakeSlabIO:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write_attr(self, name, value):
            queued[name] = value

    monkeypatch.setattr("file_io.slab_io.SlabIO", FakeSlabIO)
    binding = _binding()
    write_restart_state_to_h5(
        tmp_path / "restart.h5", n_rmu_logical=7, mode="w",
        bispinor_vq_restart_binding=binding)
    raw = queued[BISPINOR_VQ_RESTART_BINDING_DATASET]
    assert bytes(raw) == bytes(binding.encode())


def test_cross_file_binding_gate_precedes_distributed_restart_load():
    source = Path(__file__).parents[1] / "src" / "gw" / "gw_init.py"
    module = ast.parse(source.read_text(encoding="utf-8"))
    prepare = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "prepare_isdf_and_wavefunctions")

    def calls(name):
        return [
            node.lineno for node in ast.walk(prepare)
            if isinstance(node, ast.Call)
            and ((isinstance(node.func, ast.Name) and node.func.id == name)
                 or (isinstance(node.func, ast.Attribute)
                     and node.func.attr == name))]

    assert max(calls(
        "authenticate_or_recover_bispinor_vq_restart_binding")) < min(
        calls("load_restart_state_from_h5"))
