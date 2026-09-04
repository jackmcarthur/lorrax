"""Pure login-node tests for the deck-doctor launcher seam."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from lxkit.deck_doctor import (
    _backend_lines,
    _device_lines,
    required_input_paths,
)


def _config(tmp_path: Path, **overrides):
    paths = SimpleNamespace(
        wfn_file=str(tmp_path / "WFN.h5"),
        centroids_file=str(tmp_path / "centroids_frac.txt"),
        centroids_file_current=None,
        kin_ion_file=str(tmp_path / "kin_ion.h5"),
        parallel_transport_file="parallel_transport.h5",
        static_gauge_hall_file="",
    )
    head = SimpleNamespace(
        correction="off",
        wcoul0_source="s_tensor",
        use_bgw_vcoul=False,
        bgw_vcoul_file=None,
        bgw_vcoul_sym_wfn=None,
    )
    config = SimpleNamespace(
        paths=paths,
        head=head,
        sc=SimpleNamespace(head_update="off"),
        qp_solver="one_shot_dft",
        mpa=SimpleNamespace(fit_reuse_file=None),
        restart=False,
    )
    for owner, values in overrides.items():
        target = getattr(config, owner)
        for key, value in values.items():
            setattr(target, key, value)
    return config


def test_one_shot_head_off_requires_only_three_core_inputs(tmp_path):
    config = _config(tmp_path)
    rows = required_input_paths(config, tmp_path / "cohsex.in", n_rmu=12)
    assert [row.role for row in rows] == [
        "DFT wavefunctions", "ISDF centroids", "mean-field Hamiltonian"]


def test_selected_optional_inputs_and_restart_are_all_checked(tmp_path):
    config = _config(
        tmp_path,
        paths={
            "centroids_file_current": "centroids_frac_current.txt",
            "static_gauge_hall_file": "hall.h5",
        },
        head={
            "correction": "full",
            "use_bgw_vcoul": True,
            "bgw_vcoul_file": "vcoul.h5",
            "bgw_vcoul_sym_wfn": "WFN_sym.h5",
        },
        sc={"head_update": "parallel_transport"},
        mpa={"fit_reuse_file": "fit.h5"},
    )
    config.qp_solver = "self_consistent"
    config.restart = True
    rows = required_input_paths(config, tmp_path / "cohsex.in", n_rmu=836)
    assert {row.role for row in rows} == {
        "DFT wavefunctions",
        "ISDF centroids",
        "mean-field Hamiltonian",
        "current centroids",
        "long-wave dipoles",
        "parallel transport",
        "static Hall response",
        "BerkeleyGW Coulomb matrix",
        "BerkeleyGW Coulomb symmetry WFN",
        "authenticated MPA fit reuse",
        "ISDF restart tensors",
    }
    restart = next(row.path for row in rows
                   if row.role == "ISDF restart tensors")
    assert restart == tmp_path / "tmp" / "isdf_tensors_836.h5"


def test_epshead_replaces_dipole_requirement(tmp_path):
    config = _config(
        tmp_path,
        head={"correction": "full", "wcoul0_source": "epshead"},
    )
    roles = {row.role for row in required_input_paths(
        config, tmp_path / "cohsex.in", n_rmu=12)}
    assert "long-wave epsilon head" in roles
    assert "long-wave dipoles" not in roles


def test_zero_gpu_report_does_not_import_jax():
    lines = _device_lines(gpu=False)
    assert "use --gpu to measure" in lines[0]
    assert _backend_lines(None, n_rmu=12, gpu=False) == [
        "BACKEND_PROBE skipped (zero-GPU doctor; add --gpu)"]
