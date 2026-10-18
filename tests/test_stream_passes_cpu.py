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


def test_live_outputs_and_nodes_match_full_accumulate():
    """Padded member slots (zero weights) and a chunk's nodes past the count are skipped
    exactly: the carry equals the dense every-term, every-output accumulate."""
    from gw.subtile_stream import Block, PassPlan, stream_passes

    mesh = Mesh(np.array(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    L, N, n_out, cap, live_out, count = 6, 5, 9, 7, 4, 5
    rng = np.random.default_rng(1)
    w = rng.standard_normal((2, n_out, cap)) + 1j * rng.standard_normal((2, n_out, cap))
    w[:, live_out:] = 0
    w[:, :, count:] = 0
    weights = jnp.asarray(w)
    planes = jnp.asarray(rng.standard_normal((cap, 2, L, N)) + 1j * rng.standard_normal((cap, 2, L, N)))

    def node_rows(ops, n):
        return planes[n][None, :, None]

    plan = PassPlan(passes=((0, 3), (3, 3)), chunk=3, rows=3,
                    windows=((0, 0, 3), (3, 0, 3)), blocks=(Block(),))

    def node_rows_w(ops, n):
        return jax.lax.dynamic_slice_in_dim(planes[n], ops, 3, axis=1)[None, :, None]

    f = jax.jit(lambda c: stream_passes(c, mesh=mesh, plan=plan, weights=weights,
                                        count=jnp.int32(count), node_rows=node_rows_w,
                                        prepare=lambda window: window[0]))
    out = np.asarray(f(jnp.zeros((n_out, 1, L, N), jnp.complex128)))
    ref = np.zeros((n_out, 1, L, N), np.complex128)
    for n in range(count):
        for s in range(2):
            ref = ref + w[s, :, n][:, None, None, None] * np.asarray(planes[n, s])[None, None]
    np.testing.assert_allclose(out, ref, rtol=1e-13, atol=1e-13)
    assert np.all(out[live_out:] == 0)
