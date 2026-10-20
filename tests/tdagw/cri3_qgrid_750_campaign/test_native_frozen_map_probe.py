"""External probe metadata/contract tests, NOT native GW physics verification."""
import ast
import dataclasses
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "native_frozen_map_probe", Path(__file__).with_name("native_frozen_map_probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_unaccepted_or_one_map_reference_refuses_before_array_load(tmp_path):
    (tmp_path / "driver_complete.json").write_text(json.dumps(dict(
        reference_ready=False, driver_returncode=0, solver={})))
    (tmp_path / "native_solver_return.json").write_text(json.dumps(dict(
        status="ONE_MAP_DIAGNOSTIC", frame="missing")))
    with pytest.raises(ValueError, match="accepted native"):
        probe.load_accepted_frame(tmp_path)


def test_tampered_frame_array_digest_refuses_before_shape_claim(tmp_path):
    h = probe.helpers()
    solver = dict(status="CONVERGED_ACCEPTED_INPUT", accepted_input_join=True,
                  proper_sc_requested=True, native_converged=True, frame="frame0")
    (tmp_path / "native_solver_return.json").write_text(json.dumps(solver))
    (tmp_path / "driver_complete.json").write_text(json.dumps(dict(
        solver=solver, reference_ready=True, driver_returncode=0)))
    (tmp_path / "frame0.json").write_text(json.dumps(dict(
        schema="native_sc_input_frame_v1", frame="frame0", U_sha256="wrong")))
    h.exclusive_write(tmp_path / "frame0.npz", arrays={"U_active": np.eye(2)})
    with pytest.raises(ValueError, match="digest mismatch: U_active"):
        probe.load_accepted_frame(tmp_path)


def test_complex_band_pair_report_keeps_row_column_and_offdiagonal_errors():
    from common.units import RYD_TO_EV
    fine = np.ones((2, 3, 2, 2), dtype=complex)
    coarse = fine.copy()
    coarse[1, 2, 0, 1] += 2.e-4 + 3.e-4j
    result = probe.band_pair_comparison(fine, coarse)
    assert result["shape"] == [2, 3, 2, 2]
    assert result["imaginary"]["fine_frobenius_ry"] == 0
    assert result["imaginary"]["relative"] is None
    assert result["imaginary"]["difference_frobenius_ry"] == pytest.approx(3.e-4)
    assert result["offdiagonal"]["difference_frobenius_ry"] == pytest.approx(abs(2.e-4+3.e-4j))
    assert result["per_k_max_abs_mev"][:2] == [0., 0.]
    assert result["max_abs_mev"] == pytest.approx(abs(2.e-4+3.e-4j)*RYD_TO_EV*1000)


def test_native_configuration_clone_preserves_physics_and_exposes_debug_twin(tmp_path):
    from gw.gw_config import LorraxConfig
    path = tmp_path/'metadata_only.in'
    path.write_text('[cohsex]\nsys_dim=2\nnumber_bands=750\nnumber_bands_protected=160\n'
        'qgrid=4 4 1\nhead_correction=full\nsc_head_update=dft_velocity\n'
        'qp_solver=self_consistent\ncompute_mode=mpa\nsigma_w_model=shared_pole\n'
        'sc_tol_ev=.001\nrestart=false\nwrite_restart_tensors=false\n'
        'write_wfn_h5=false\nqp_rotations_k_storage=full\n')
    config = LorraxConfig.from_input_file(str(path), resolve_hardware=False)
    fine = dataclasses.replace(config, qgrid=(8, 8, 1), debug=dataclasses.replace(
        config.debug, sigma_freq_debug_output=True))
    assert fine.nband == config.nband == 750
    assert fine.number_bands_protected == config.number_bands_protected == 160
    assert fine.sigma.regularization_ev == config.sigma.regularization_ev
    assert fine.sc.tol_ev == config.sc.tol_ev == .001
    assert fine.head is config.head and fine.sigma is config.sigma
    assert fine.debug.sigma_freq_debug_output and not config.debug.sigma_freq_debug_output


def test_actual_native_owner_hook_signatures_and_no_alternate_solver():
    """AST-only owner contract; never executes a map or a distributed operation."""
    source = Path(__file__).resolve().parents[3]
    modules = {name: ast.parse((source / name).read_text()) for name in (
        "src/gw/sc_iteration.py", "src/gw/sigma_dispatch.py", "src/gw/gw_init.py")}
    expected = {"src/gw/sc_iteration.py": ("gw_iteration_map", "run_self_consistency", "_sc_eigh_bands"),
                "src/gw/sigma_dispatch.py": ("finalize_dynamic_sigma",),
                "src/gw/gw_init.py": ("_finalize_vq_views", "screening_coulomb_view")}
    for filename, names in expected.items():
        found = {node.name for node in modules[filename].body if isinstance(node, ast.FunctionDef)}
        assert set(names) <= found
    functions = {node.name: node for node in modules["src/gw/sc_iteration.py"].body
                 if isinstance(node, ast.FunctionDef)}
    assert [arg.arg for arg in functions["gw_iteration_map"].args.args] == ["state", "inputs"]
    text = Path(probe.__file__).read_text()
    assert "mapped = owner.gw_iteration_map(state, leg)" in text
    assert "EXTERNAL_FROZEN_PROBE_COMPLETE" in text
    assert "full_W_comparison=\"NOT_MEASURED\"" in text
