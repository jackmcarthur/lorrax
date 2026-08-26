"""Focused host gates for the bispinor V/restart composition receipt."""
from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
    coulomb_policy_from_config,
    format_coulomb_policy,
    write_restart_state_to_h5,
)
from file_io.wfn_basis import (
    CENTROID_TABLE_FINGERPRINT_SCHEME,
    WavefunctionBasisReceipt,
)


def _basis(role: str, *, pad: int = 8, centroid="1" * 32):
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


def _policy(*, tt_head=False):
    head = SimpleNamespace(
        mc_average_vcoul_body=True,
        mc_average_placement="off",
        mc_average_placement_vcoul=None,
        head_minibz_average=False,
        bare_coulomb_cutoff=None,
        use_bgw_vcoul=False,
        bgw_vcoul_file=None,
        bispinor_tt_head_correction=tt_head,
    )
    policy = coulomb_policy_from_config(
        SimpleNamespace(head=head), SimpleNamespace(sys_dim=2))
    return format_coulomb_policy(policy)


def _binding(*, pad=8, provenance=None, tt_head=False):
    return BispinorVqRestartBinding.from_sources(
        v_qmunu_format="bispinor_lorentz_v2",
        zeta_fit_provenance=(
            provenance or ("charge", "transverse-1", "transverse-2",
                           "transverse-3")),
        charge_basis_receipt=_basis("charge", pad=pad),
        transverse_basis_receipt=_basis(
            "transverse", pad=pad, centroid="2" * 32),
        coulomb_policy=_policy(tt_head=tt_head),
    )


def _write(path, binding=None):
    with h5py.File(path, "w") as h5:
        h5.create_dataset("payload", data=np.zeros((1,), dtype=np.float64))
        if binding is not None:
            h5.create_dataset(
                BISPINOR_VQ_RESTART_BINDING_DATASET,
                data=binding.encode())


def test_binding_roundtrip_reuses_exact_sources_without_runtime_pad(tmp_path):
    p4 = _binding(pad=8)
    p16 = _binding(pad=16)
    assert p4 == p16
    assert b"n_rmu_padded" not in bytes(p4.encode())

    path = tmp_path / "binding.h5"
    _write(path, p4)
    assert read_bispinor_vq_restart_binding(path) == p4


def test_composition_does_not_call_wfn_fingerprint(monkeypatch):
    import common.parallel_transport as transport

    def unexpected_scan(_wfn):
        raise AssertionError("composition receipt rescanned WFN")

    monkeypatch.setattr(transport, "wfn_fingerprint", unexpected_scan)
    assert _binding().charge_basis_source["wfn_fingerprint"] == "a" * 64


def test_matching_artifacts_authenticate_and_return_record(tmp_path):
    binding = _binding()
    restart = tmp_path / "restart.h5"
    photon = tmp_path / "v_q_bispinor.h5"
    _write(restart, binding)
    _write(photon, binding)
    assert assert_bispinor_vq_restart_binding(
        restart_path=restart, v_q_path=photon) == binding


@pytest.mark.parametrize(
    "changed,field",
    [
        (_binding(provenance=("charge-new", "transverse-1",
                              "transverse-2", "transverse-3")),
         "zeta_fit_provenance"),
        (_binding(tt_head=True), "coulomb_policy"),
        (replace(_binding(), v_qmunu_format="bispinor_lorentz_v3"),
         "v_qmunu_format"),
    ],
)
def test_same_shape_stale_photon_artifact_refuses(tmp_path, changed, field):
    restart = tmp_path / "restart.h5"
    photon = tmp_path / "v_q_bispinor.h5"
    _write(restart, _binding())
    _write(photon, changed)
    with pytest.raises(ValueError, match=field):
        assert_bispinor_vq_restart_binding(
            restart_path=restart, v_q_path=photon, where="restart gate")


def test_legacy_unstamped_artifact_refuses(tmp_path):
    restart = tmp_path / "restart.h5"
    photon = tmp_path / "v_q_bispinor.h5"
    _write(restart, _binding())
    _write(photon)
    with pytest.raises(ValueError, match="Legacy artifacts cannot authenticate"):
        assert_bispinor_vq_restart_binding(
            restart_path=restart, v_q_path=photon)


def test_restart_writer_stamps_the_exact_binding_object(monkeypatch, tmp_path):
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
    assert BispinorVqRestartBinding.from_record(
        __import__("json").loads(bytes(raw).decode())) == binding


def test_gw_restart_binding_gate_precedes_distributed_payload_load():
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

    assert calls("assert_bispinor_vq_restart_binding")
    assert calls("load_restart_state_from_h5")
    assert max(calls("assert_bispinor_vq_restart_binding")) < min(
        calls("load_restart_state_from_h5"))
