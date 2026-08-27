"""Same-process JIT-cache contract across the production zeta-fit cleanup."""
from __future__ import annotations

import ast
from pathlib import Path

import jax
import jax.numpy as jnp

from gw.isdf_fitting import _collect_fit_setup_garbage


_SOURCE = Path(__file__).resolve().parents[1] / "src/gw/isdf_fitting.py"


def _trace_counted_kernel():
    traces = {"count": 0}

    @jax.jit
    def kernel(x):
        traces["count"] += 1
        return x + 1

    return kernel, traces


def test_fit_cleanup_preserves_an_identical_jit_executable():
    """The production cleanup boundary must not retrace a stable JIT."""
    kernel, traces = _trace_counted_kernel()
    value = jnp.arange(8, dtype=jnp.float32)

    kernel(value).block_until_ready()
    _collect_fit_setup_garbage()
    kernel(value).block_until_ready()

    assert traces["count"] == 1


def test_red_control_global_clear_retraces_the_identical_jit():
    """RED control: the removed global clear reproduces the second trace."""
    kernel, traces = _trace_counted_kernel()
    value = jnp.arange(8, dtype=jnp.float32)

    kernel(value).block_until_ready()
    jax.clear_caches()
    kernel(value).block_until_ready()

    assert traces["count"] == 2


def test_fit_zeta_to_h5_uses_only_the_cache_preserving_cleanup():
    """Pin the tested helper into the production boundary, with no flush."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"), filename=str(_SOURCE))
    fit = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "fit_zeta_to_h5"
    )
    calls = [node.func for node in ast.walk(fit) if isinstance(node, ast.Call)]

    assert any(
        isinstance(call, ast.Name)
        and call.id == "_collect_fit_setup_garbage"
        for call in calls
    )
    assert not any(
        isinstance(call, ast.Attribute)
        and isinstance(call.value, ast.Name)
        and call.value.id == "jax"
        and call.attr == "clear_caches"
        for call in calls
    )
