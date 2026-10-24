"""Constant row schedules compile without losing overlap/padding semantics.

The closed geometry lookup after the protected start reproduced the GPU
HloEvaluator failure. CPU runs provide the same literal arithmetic control;
qualified native runs cover the compiler-specific refusal regression.
"""
from runtime.source_closure import ensure_source_closure
ensure_source_closure()

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gw.subtile_stream import scan_passes

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("count", [2, 4, 5, 16])
@pytest.mark.parametrize("ns", [1, 2])
def test_constant_schedule_with_geometry_lookup_and_overlapping_tail(count, ns):
    rows, width = 3 * count - 1, 7 * ns
    windows = tuple((3 * p, 0, 3) for p in range(count - 1)) + ((rows - 3, 1, 3),)
    geometry = np.arange(rows - 1, -1, -1, dtype=np.int32)

    @jax.jit
    def accumulate(a):
        def step(start, lo, hi, total):
            indices = jax.lax.dynamic_slice_in_dim(jnp.asarray(geometry), start, 3)
            block = a[indices]
            live = (jnp.arange(3) >= lo) & (jnp.arange(3) < hi)
            block = jnp.where(live[:, None], block, 0)
            return total + block.T @ block
        return scan_passes(windows, step, jnp.zeros((width, width), jnp.float64))

    data = np.arange(rows * width, dtype=np.float64).reshape(rows, width) / 100
    got = np.asarray(accumulate(data))
    np.testing.assert_allclose(got, data.T @ data, rtol=2e-14, atol=2e-13)
