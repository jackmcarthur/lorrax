"""One-site native dielectric diagnostic contracts, not a material/GPU proof."""
import ast
import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest


def test_request_is_explicit_q0_upper_half_plane_and_default_is_unchanged():
    from gw.shared_pole_screening import _direct_dielectric_request, screen_general_shared_poles
    assert _direct_dielectric_request({}) is None
    p = [-.16154070956060437, 1.0066523909325316]
    assert _direct_dielectric_request({'direct_dielectric_probe': {'q': 0, 'z_ry': p}}) == {
        'q': 0, 'z_ry': complex(*p)}
    for bad in ({}, {'q': 1, 'z_ry': 1j}, {'q': True, 'z_ry': 1j},
                {'q': 0, 'z_ry': 0j}, {'q': 0, 'z_ry': complex(np.nan, 1)},
                {'q': 0, 'z_ry': ['bad', 1]}, {'q': 0, 'z_ry': 1j, 'eta': .1}):
        # Actual entrypoint must refuse BEFORE consulting any source arrays.
        with pytest.raises(ValueError, match='shared_general_dielectric_request'):
            screen_general_shared_poles(None, None, None, None, mesh_xy=None, sym=None,
                snapshot={'direct_dielectric_probe': bad}, run_dir=None, label='unused')


def test_scalar_witness_owner_preserves_current_complex_epsilon_padding_and_basic_before_lu(
        tmp_path, monkeypatch):
    import jax
    import common.collectives as cc
    from gw.shared_pole_screening import _save_general_dielectric_probe
    jax.config.update('jax_enable_x64', True)
    monkeypatch.setattr(jax, 'jit', lambda fn, **kwargs: fn)
    transactions, calls = [], []
    rank = [0]
    def transaction(path, *, stage, write):
        transactions.append((rank[0], stage))
        if rank[0] == 0:
            write()
    monkeypatch.setattr(cc, 'rank0_transaction', transaction)
    mask = np.array([True, False, True])
    physical = np.array([[.7+.1j, 3.-.2j], [.03j, 1.2-.1j]])
    epsilon = np.eye(3, dtype=complex)
    epsilon[np.ix_(mask, mask)] = physical
    root = np.diag([2., 0., 3.])[None]
    raw = np.arange(9).reshape(1, 3, 3).astype(complex)
    def matrix(h, r):
        assert h is root and r is raw
        calls.append((rank[0], 'matrix'))
        return epsilon[None]
    def inverse(h, r):
        assert h is root and r is raw
        assert (tmp_path/'q_000_dielectric_00_basic.json').exists()
        calls.append((rank[0], 'inverse'))
        return epsilon[None], np.linalg.inv(epsilon)[None]
    operators = dict(dyson=SimpleNamespace(dielectric_matrix=matrix, dielectric=inverse),
        H=root, matmul=lambda a, b: a@b, face=None, rep=None, mask=mask)
    record = dict(q=0, z_ry=[-.16, 1.], gamma_ry=.003, prefactor=.125)
    for value in (0, 1):
        rank[0] = value
        result = _save_general_dielectric_probe(raw, operators, record, root=tmp_path, index=0)
        assert result['dielectric_inverse_file'] == 'q_000_dielectric_00_inverse.json'
    assert calls == [(0, 'matrix'), (0, 'inverse'), (1, 'matrix'), (1, 'inverse')]
    assert transactions == [(r, s) for r in (0, 1) for s in
        ('shared_general.dielectric_basic', 'shared_general.dielectric_inverse')]
    result = json.loads((tmp_path/'q_000_dielectric_00_inverse.json').read_text())
    assert not result['admitted'] and not result['sigma_computed']
    assert result['logical_dimension'] == 2 and result['packed_dimension'] == 3
    bounds = result['inverse_bounds']
    smallest = np.linalg.svd(physical, compute_uv=False)[-1]
    assert bounds['resolved'] and bounds['sigma_min_lower'] <= smallest <= bounds['sigma_min_upper']
    assert result['additional_collision_eta_ry'] == 0


def test_diagnostic_selection_and_stop_precede_all_dyson_values_directions_and_fitting():
    from gw.shared_pole_screening import screen_general_shared_poles, _general_current_site_samples
    source = inspect.getsource(screen_general_shared_poles)
    tree = ast.parse(source)
    stops = [n for n in ast.walk(tree) if isinstance(n, ast.Raise)
             and 'shared_general_dielectric_diagnostic_stop' in ast.unparse(n)]
    assert len(stops) == 1
    stop = stops[0]
    # The ACTUAL owner stream receives the one-row query table and keeps the
    # same physical runtime operands/executor; no separate contraction path.
    stage_tree = ast.parse(inspect.getsource(_general_current_site_samples))
    streams = [n for n in ast.walk(stage_tree) if isinstance(n, ast.Call)
               and isinstance(n.func, ast.Name) and n.func.id == 'full_density_response_stream']
    primary = min(streams, key=lambda n: n.lineno)
    keywords = {k.arg: ast.unparse(k.value) for k in primary.keywords}
    assert keywords['q_ids'] == 'query_q_ids'
    assert keywords['k_unfold_plan'] == 'plan' and keywords['execute'] == 'execute_response'
    assert 'np.arange(nq) if direct_request is None else np.asarray([0])' in source
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == '_general_current_site_samples')
    assert next(ast.unparse(k.value) for k in call.keywords if k.arg == 'evaluate_dyson') == 'direct_request is None'
    assert call.lineno < stop.lineno
    assert stop.lineno < next(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'values' for t in n.targets))
    assert source.index('shared_general_dielectric_diagnostic_stop') < source.index('begin("directions")')
    assert 'dielectric_bytes' in source and 'query_count = len(query_q_ids)' in source
