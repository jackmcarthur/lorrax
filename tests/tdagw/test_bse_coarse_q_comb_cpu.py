"""Tiny owner/oracle tests; CPU reference FFT is not a native GPU receipt."""
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ROOT = Path(__file__).parent
DIAG = ROOT/'cri3_qgrid_750_campaign/diagnostics/verify_bse_coarse_q_p4.py'
HELPER = ROOT/'cri3_full_microscopic/native_bse_probe.py'


def test_equal_grid_q_comb_is_bitwise_identity_and_bad_grids_refuse():
    from common.fft_helpers import make_bosonic_q_comb
    value = np.array([complex(-0., 1.), complex(0., -0.)])
    assert make_bosonic_q_comb(None, (8, 8, 1), (8, 8, 1))(value) is value
    for fine, coarse in [((8, 8, 1), (3, 4, 1)), ((8, 8, 1), (4, 0, 1)),
                         ((8, 8), (4, 4, 1)), ((8., 8., 1.), (4, 4, 1))]:
        with pytest.raises(ValueError, match='bosonic_q_comb'):
            make_bosonic_q_comb(None, fine, coarse)


def test_actual_bse_matvec_coarse_literal_sum_distinct_heads_and_fourier_dual():
    import jax
    from jax.sharding import Mesh
    from ffi import fft
    # This pytest invocation deliberately uses one CPU device, not a large
    # emulated multirank material fixture. Same function is the P4 diagnostic.
    devices = np.asarray(jax.devices('cpu'))
    if len(devices) != 1:
        pytest.skip('use single CPU device here; native MPI proof is the separate diagnostic')
    mesh = Mesh(devices.reshape(1, 1), ('x', 'y'))
    oracle = load('bse_coarse_oracle', DIAG)
    with jax.enable_x64(True), fft.xla_reference():
        result = oracle.run_oracle(mesh)
    assert result['max_complex_matvec_error'] < 3.e-12
    assert result['heads']['bare_applications'] == result['heads']['screened_applications'] == 1
    assert result['heads']['fine_bare'] != result['heads']['coarse_screened']


def test_bse_payload_uses_matching_bosonic_body_but_fine_bare_head_before_embedding():
    tree = ast.parse(HELPER.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_bse_probe')
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)]
    static = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'build_static_screened_w')
    assert static.args[1].id == 'interaction'
    inject = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == '_inject_q0_head')
    assert ast.unparse(inject.args[4]) == 'bare_sample.vc0'
    assert 'hs.wcoul0' in ast.unparse(inject.args[5])
    comb = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'make_bosonic_q_comb')
    assert comb.lineno < static.lineno < inject.lineno
    compiled = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == 'compiled')
    price = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == 'reserve')
    assert inject.lineno < price.lineno < compiled.lineno


def test_prepared_gpu_diagnostic_has_native_guards_and_no_reference_override():
    tree = ast.parse(DIAG.read_text())
    text = DIAG.read_text()
    assert 'xla_reference' not in text and 'monkeypatch' not in text
    assert "jax.process_count() != 4" in text and "max_wall_seconds <= 120" in text
    assert "!= 'mathdx'" in text and 'runtime.finalize_process(0)' in text
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == 'fft' for n in ast.walk(tree))
