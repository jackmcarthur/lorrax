"""Host-side gates for packed-photon restart composition and readiness."""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from common.bispinor_init import KINETIC_BALANCE_LIFT_PROVENANCE
from common.parallel_transport import WFN_FINGERPRINT_SCHEME
from common.wfn_transforms import FULL_BLOCH_TRANSFORM_SCHEME
from file_io.bispinor_vq_restart import (
    BISPINOR_VQ_RESTART_BINDING_DATASET,
    BispinorVqRestartBinding,
    assert_bispinor_vq_restart_binding,
    read_bispinor_vq_restart_binding,
)
from file_io.tagged_arrays import (
    read_photon_g0_vectors_from_h5,
    read_ready_w0_qmunu_from_h5,
    write_restart_state_to_h5,
)
from file_io.wfn_basis import (
    CENTROID_TABLE_FINGERPRINT_SCHEME,
    WavefunctionBasisReceipt,
)


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

    assert max(calls("assert_bispinor_vq_restart_binding")) < min(
        calls("load_restart_state_from_h5"))
