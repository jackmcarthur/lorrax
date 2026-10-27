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


@pytest.mark.parametrize('cpu', [False, True])
@pytest.mark.parametrize('channels,proposed,fresh,expected', [
    (1, 30_000_000, 20_000_000, 'disk'),
    (1, 20_000_000, 30_000_000, 'disk'),
    (1, 30_000_000, 25_133_056, 'host'),
    (3, 80_000_000, 60_000_000, 'disk'),
    (3, 80_000_000, 75_399_168, 'host'),
])
def test_store_allocation_refreshes_headroom_after_compilation(
        monkeypatch, channels, proposed, fresh, expected, cpu):
    # Execute the real allocation boundary, without fitting fields or opening
    # files. Constructor observers refuse a stale decision before allocation.
    from common import gpu_utils
    reserve, fit_new_bytes = 0, 0
    if cpu:
        import jax
        import jax.numpy as jnp
        from runtime.aot_memory import compiled_new_bytes
        jax.config.update('jax_enable_x64', True)
        # Use an actual batch allocation quote and its lookahead output, not
        # a guessed slope; the quota must retain this same future workspace.
        batch = jax.jit(lambda: jnp.zeros((144, 144), jnp.complex128)).lower().compile()
        fit_new_bytes = compiled_new_bytes(batch, extra=batch.memory_analysis().output_size_in_bytes)
        reserve = fit_new_bytes * jax.local_device_count()
        proposed, fresh = proposed + reserve, fresh + reserve
    else:
        jax = SimpleNamespace(default_backend=lambda: 'gpu')
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
        P_=64, c_out=1, n_blk=1, replace=replace, jax=jax,
        fit_new_bytes=fit_new_bytes, memory=SimpleNamespace(room=lambda **_k: 1 << 40),
        log_mubatch_plan=lambda p, **_k: reports.append(p), bispinor=channels > 1,
        zmb=SimpleNamespace(ZStore=store), mesh_xy=None, q_axis=None,
        mu_pad=1152, scratch_dir='unused', os=os)
    exec(compile(ast.Module(body=fit.body[start:stop+1], type_ignores=[]),
                 str(source), 'exec'), namespace)
    assert events == ['fresh quota'] + ['allocate'] * channels
    assert reports[0].host_budget_bytes == min(proposed, fresh) - reserve
    assert reports[0].store_bytes == channels * 25_133_056
    assert plan.host_budget_bytes == proposed
    if cpu:
        # A smaller JAX-room ceiling must also reserve the same next batch,
        # even when the original and fresh host quotas both permit the store.
        room = int((reports[0].store_bytes + reserve - 1) // jax.local_device_count())
        namespace['memory'].room = lambda **_k: room
        expected = 'disk'
        events.clear()
        reports.clear()
        exec(compile(ast.Module(body=fit.body[start:stop+1], type_ignores=[]),
                     str(source), 'exec'), namespace)
        assert reports[0].host_budget_bytes < reports[0].store_bytes
        assert events == ['fresh quota'] + ['allocate'] * channels
