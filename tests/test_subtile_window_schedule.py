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


@pytest.mark.parametrize('only', [None, 2])
def test_streamed_node_chunks_share_opaque_schedule_and_exact_dead_rows(only):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from gw.subtile_stream import Block, PassPlan, stream_passes
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))
    rows, width, block, cap, count = 11, 5, 3, 5, 3
    windows = ((0, 0, 3), (3, 0, 3), (6, 0, 3), (8, 1, 3))
    plan = PassPlan(passes=tuple((s+lo, hi-lo) for s,lo,hi in windows),
                    chunk=2, rows=block, windows=windows, blocks=(Block(),))
    rng = np.random.default_rng(241024)
    values = rng.normal(size=(cap, 2, rows, width)) + 1j*rng.normal(size=(cap, 2, rows, width))
    weights = rng.normal(size=(1, 3, cap)) + 1j*rng.normal(size=(1, 3, cap))
    weights[:, -1] = 0  # inactive output prefix; unused nodes deliberately nonzero
    spec = NamedSharding(mesh, P(None, None, 'x', 'y'))

    @jax.jit
    def run(values, weights):
        def prepare(window):
            start, lo, hi = window
            chunk = jax.lax.dynamic_slice_in_dim(values, start, block, axis=2)
            live = (jnp.arange(block) >= lo) & (jnp.arange(block) < hi)
            return jnp.where(live[None, None, :, None], chunk, 0)
        def node_rows(prepared, index):
            return prepared[index][None, None]
        output_rows = rows if only is None else block
        carry = jax.lax.with_sharding_constraint(
            jnp.zeros((3, 2, output_rows, width), jnp.complex128), spec)
        return stream_passes(carry, mesh=mesh, plan=plan, weights=weights,
                             count=jnp.int32(count), node_rows=node_rows,
                             prepare=prepare, only=only)

    got = np.asarray(run(jax.device_put(values, spec), jnp.asarray(weights)))
    reference = np.einsum('on,nqrc->oqrc', weights[0, :, :count], values[:count])
    if only is not None:
        start, lo, hi = windows[only]
        reference = reference[:, :, start:start+block].copy()
        reference[:, :, :lo] = 0
        reference[:, :, hi:] = 0
    np.testing.assert_allclose(got, reference, rtol=3e-14, atol=2e-13)
