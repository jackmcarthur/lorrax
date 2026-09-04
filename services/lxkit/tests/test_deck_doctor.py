"""Pure login-node tests for the deck-doctor launcher seam."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

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
    assert _backend_lines(
        None, n_rmu=12, gpu=False, science_mesh=(2, 2)) == [
        "BACKEND_PROBE skipped (zero-GPU doctor; add --gpu)"]


def test_backend_probe_defers_only_true2d_collective(monkeypatch):
    config = SimpleNamespace(backend=SimpleNamespace(
        distributed_cholesky="auto",
        distributed_lu="cusolvermp",
        eigh_backend="cusolvermp",
    ))

    class FakeMesh:
        def __init__(self, *_args, **_kwargs):
            pass

    calls = []

    def resolve(op, requested, _mesh, *, n):
        calls.append((op, requested, n))
        if op == "solve_lu":
            raise ValueError(
                "solve_lu backend 'cusolvermp' needs a true-2D mesh "
                "(px >= 2 and py >= 2)")
        return "native" if requested == "auto" else requested

    monkeypatch.setitem(
        sys.modules, "jax", SimpleNamespace(local_devices=lambda: [object()]))
    monkeypatch.setitem(sys.modules, "jax.sharding", SimpleNamespace(Mesh=FakeMesh))
    monkeypatch.setitem(
        sys.modules, "distrib_la", SimpleNamespace(resolve_backend=resolve))

    lines = _backend_lines(
        config, n_rmu=836, gpu=True, science_mesh=(2, 2))
    assert any("op=cholesky" in line and "resolved=native" in line
               for line in lines)
    assert any("op=solve_lu" in line
               and "resolved=deferred-to-science-launch" in line
               and "live_provider=usable" in line for line in lines)
    assert any("op=eigh" in line and "resolved=cusolvermp" in line
               for line in lines)
    assert calls == [
        ("cholesky", "auto", 836),
        ("solve_lu", "cusolvermp", 836),
        ("eigh", "cusolvermp", 836),
    ]


def test_backend_probe_does_not_hide_bad_science_geometry(monkeypatch):
    config = SimpleNamespace(backend=SimpleNamespace(
        distributed_cholesky="cusolvermp",
        distributed_lu="auto",
        eigh_backend="auto",
    ))

    class FakeMesh:
        def __init__(self, *_args, **_kwargs):
            pass

    def resolve(op, requested, _mesh, *, n):
        del op, requested, n
        raise ValueError("backend 'cusolvermp' needs a true-2D mesh")

    monkeypatch.setitem(
        sys.modules, "jax", SimpleNamespace(local_devices=lambda: [object()]))
    monkeypatch.setitem(sys.modules, "jax.sharding", SimpleNamespace(Mesh=FakeMesh))
    monkeypatch.setitem(
        sys.modules, "distrib_la", SimpleNamespace(resolve_backend=resolve))

    import pytest
    with pytest.raises(ValueError, match="needs a true-2D mesh"):
        _backend_lines(
            config, n_rmu=64, gpu=True, science_mesh=(1, 1))
