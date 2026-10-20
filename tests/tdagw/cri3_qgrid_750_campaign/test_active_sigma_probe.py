"""Host metadata/dispatch contract only, not a native Sigma timing."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("active_sigma_probe", Path(__file__).with_name("active_sigma_probe.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture(autouse=True)
def fp64_metadata():
    import jax
    with jax.enable_x64(True):
        yield


def tables():
    energy = np.broadcast_to((np.arange(750)-129.5)/10, (64, 750)).copy()
    occupations = np.broadcast_to((np.arange(750) < 130).astype(float), energy.shape).copy()
    return energy, occupations


def test_request_contains_only_active_energies_and_preserves_full_virtual_tables():
    from common.units import RYD_TO_EV
    energy, occupations = tables()
    original = energy.copy()
    config = SimpleNamespace(sigma=SimpleNamespace(fermi_reference="midgap", out_of_grid="cover"),
                             omega_grid_ev=np.array([-200., 200.]))
    request, mu, policy = probe.active_request(energy, occupations, config)
    assert mu == pytest.approx(0.)
    assert np.max(abs(request.frequencies_ry)) <= abs(energy[:, 100:144]).max()+1.e-12
    np.testing.assert_allclose(request.table_ev("pair"), energy[:, 100:144]*RYD_TO_EV,
                               rtol=1.e-15, atol=1.e-15)
    np.testing.assert_array_equal(energy, original)
    assert occupations.shape == energy.shape == (64, 750)


def test_noninsulating_or_fractional_metadata_does_not_become_an_equilibrium_benchmark():
    energy, occ = tables()
    config = SimpleNamespace(sigma=SimpleNamespace(fermi_reference="midgap", out_of_grid="cover"),
                             omega_grid_ev=np.array([-200., 200.]))
    occ[0, 0] = .5
    with pytest.raises(ValueError, match="insulating energy/occupation"):
        probe.active_request(energy, occ, config)
    energy, occ = tables()
    energy[0, 130] = -20.
    with pytest.raises(ValueError, match="insulating gap"):
        probe.active_request(energy, occ, config)


def test_dispatch_has_compact_destinations_and_no_hidden_fullprefix_or_optical_solve():
    import ast
    tree = ast.parse(Path(probe.__file__).read_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "dict"]
    arguments = next(n for n in calls if any(k.arg == "projection_bands" for k in n.keywords))
    projection = next(k.value for k in arguments.keywords if k.arg == "projection_bands")
    assert ast.literal_eval(projection) == (100, 144)
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_pre_sigma_probe")
    loop = next(n for n in ast.walk(run) if isinstance(n, ast.For)
                and isinstance(n.target, ast.Name) and n.target.id == "repeat")
    phase = next(n for n in ast.walk(loop) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and n.func.attr == "at_phase")
    compute = next(n for n in ast.walk(loop) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == "compute_sigma_c_mpa_omega_grid")
    assert phase.lineno < compute.lineno
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ("run_bse_probe", "gw_iteration_map", "run_propagation")
                   for n in ast.walk(run))
