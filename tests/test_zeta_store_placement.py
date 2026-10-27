"""Actual orbit slots and current host headroom determine store residence."""
import ast
from dataclasses import replace
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from gw.gflat_memory_model import MuBatchPlan, _ZSTORE_HOST_FRAC, zstore_layout


def test_resolved_orbit_slots_reprice_the_store_at_the_same_quota():
    # Captured Fe P64: the proposal had 1152 slots; two executed batches
    # allocate 2*1024. Q=59 and the padded G carrier=832 are unchanged.
    quota = 20_000_000
    kwargs = dict(n_vertex=1, n_ranks=64, host_budget_bytes=quota)
    assert zstore_layout(59, 1152, 832, **kwargs) == (14_137_344.0, 'host')
    actual, placement = zstore_layout(59, 2048, 832, **kwargs)
    assert (actual, placement) == (25_133_056.0, 'disk')
    for available, expected in ((actual, 'host'), (actual - 1, 'disk')):
        assert zstore_layout(
            59, 2048, 832, n_vertex=1, n_ranks=64,
            host_budget_bytes=available) == (actual, expected)
    # Each current channel owns a separate store at the same padded shape.
    assert zstore_layout(
        59, 2048, 832, n_vertex=3, n_ranks=64,
        host_budget_bytes=3 * actual) == (3 * actual, 'host')


@pytest.mark.parametrize('channels,proposed,fresh,expected', [
    (1, 30_000_000, 20_000_000, 'disk'),
    (1, 20_000_000, 30_000_000, 'disk'),
    (1, 30_000_000, 25_133_056, 'host'),
    (3, 80_000_000, 60_000_000, 'disk'),
    (3, 80_000_000, 75_399_168, 'host'),
])
def test_store_allocation_refreshes_headroom_after_compilation(
        monkeypatch, channels, proposed, fresh, expected):
    # Execute the real allocation boundary, without fitting fields or opening
    # files. Constructor observers refuse a stale decision before allocation.
    from common import gpu_utils
    events, reports = [], []
    def available(fraction):
        assert fraction == _ZSTORE_HOST_FRAC
        events.append('fresh quota')
        return fresh
    def store(**kwargs):
        assert kwargs['placement'] == expected
        assert events and events[0] == 'fresh quota'
        assert kwargs['b'] == 1024 and kwargs['n_batch'] == 2
        events.append('allocate')
        return kwargs
    monkeypatch.setattr(gpu_utils, 'host_bytes_per_process', available)
    plan = MuBatchPlan(route='G', source='resident', band_chunk=64, k_chunk=1,
        b=1024, n_batch=2, r_sub=1, row_chunk=0, g_tile=64, placement='host',
        hwm_bytes=0., budget_bytes=0., target_bytes=0., breakdown={}, transfer={},
        host_budget_bytes=proposed)
    source = Path(__file__).parents[1]/'src/gw/isdf_fitting.py'
    fit = next(n for n in ast.parse(source.read_text()).body
               if isinstance(n, ast.FunctionDef) and n.name == '_fit_mubatch')
    # This boundary begins after executable construction and its room check.
    start = next(i for i, n in enumerate(fit.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'split_kernels' for t in n.targets))
    stop = next(i for i, n in enumerate(fit.body) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'stores' for t in n.targets))
    namespace = dict(debug_print_enabled=lambda: False, plan=plan,
        _ZSTORE_HOST_FRAC=_ZSTORE_HOST_FRAC, zstore_layout=zstore_layout,
        Q=59, mb=SimpleNamespace(n_batch=2, slot_of_packed=None), b=1024,
        g_axis=SimpleNamespace(carrier=832), vertices=tuple(range(channels)),
        P_=64, c_out=1, n_blk=1, replace=replace,
        log_mubatch_plan=lambda p, **_k: reports.append(p), bispinor=channels > 1,
        zmb=SimpleNamespace(ZStore=store), mesh_xy=None, q_axis=None,
        mu_pad=1152, scratch_dir='unused', os=os)
    exec(compile(ast.Module(body=fit.body[start:stop+1], type_ignores=[]),
                 str(source), 'exec'), namespace)
    assert events == ['fresh quota'] + ['allocate'] * channels
    assert reports[0].host_budget_bytes == min(proposed, fresh)
    assert reports[0].store_bytes == channels * 25_133_056
    assert plan.host_budget_bytes == proposed
