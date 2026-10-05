"""The scanned row-pass loop on a host mesh under x64 (CPU, seconds).

``gw.subtile_stream.stream_passes`` adds each window at a traced int32 row
offset; the host contour block accumulate must take it beside the x64 Python
ints, and two windows must add the bytes of one whole-tile window.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh

jax.config.update("jax_enable_x64", True)


def test_two_windows_match_one_window():
    from gw.subtile_stream import Block, PassPlan, stream_passes

    mesh = Mesh(np.array(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    L, N = 6, 2
    weights = jnp.asarray([[[1.0, 0.5 + 0.25j]]], jnp.complex128)   # [sets, out, cap]

    def run(plan):
        R = int(plan.rows)

        def node_rows(ops, n):
            row = jnp.arange(R, dtype=jnp.float64)[:, None] + (ops if ops is not None else 0)
            return ((n + 1) * (row + 1j * jnp.arange(N))).astype(jnp.complex128)[None, None, None]

        prepare = lambda window: None if window is None else window[0].astype(jnp.float64)
        f = jax.jit(lambda c, cnt: stream_passes(c, mesh=mesh, plan=plan, weights=weights,
                                                 count=cnt, node_rows=node_rows, prepare=prepare))
        return np.asarray(f(jnp.zeros((1, 1, L, N), jnp.complex128), jnp.int32(2)))

    one = run(PassPlan(passes=((0, L),), chunk=1, rows=L, windows=((0, 0, L),),
                       blocks=(Block(),)))
    two = run(PassPlan(passes=((0, 3), (3, 3)), chunk=1, rows=3,
                       windows=((0, 0, 3), (3, 0, 3)), blocks=(Block(),)))
    assert np.any(one != 0)
    np.testing.assert_array_equal(two, one)
