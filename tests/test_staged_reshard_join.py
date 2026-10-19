"""``common.staged_reshard.concatenate_sharded_axis``: one exchange for every block (CPU 2x2 mesh).

Run with ``JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4``.
The blocks' local tiles are joined before the first ``all_to_all`` and read back
block-major, so a join equals ``np.concatenate`` on face and slab layouts, and its
compiled program holds two all-to-alls for 2 blocks and for 12.
"""
import numpy as np
import pytest
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)


def _mesh():
    if len(jax.devices()) < 4:
        pytest.skip("needs 4 host devices (XLA_FLAGS=--xla_force_host_platform_device_count=4)")
    return Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ("x", "y"))


def _join(mesh, spec, axis, blocks):
    from common.staged_reshard import concatenate_sharded_axis
    sh = NamedSharding(mesh, spec)
    args = [jax.device_put(b, sh) for b in blocks]
    fn = jax.jit(lambda *xs: concatenate_sharded_axis(xs, axis, mesh, spec))
    return fn(*args), fn.lower(*args).compile().as_text()


def test_join_matches_concatenate_with_two_exchanges():
    mesh = _mesh()
    rng = np.random.default_rng(2)
    face = P(None, "x", "y")
    for axis in (-1, -2):
        for widths in ((2, 6), (2, 4, 2, 6, 8, 2, 4, 2, 10, 2, 6, 4)):
            shape = lambda w: (3, 6, w) if axis == -1 else (3, w, 6)
            blocks = [rng.normal(size=shape(w)) + 1j * rng.normal(size=shape(w)) for w in widths]
            got, text = _join(mesh, face, axis, blocks)
            assert got.sharding.spec == face
            assert np.array_equal(np.asarray(got), np.concatenate(blocks, axis=axis)), (axis, widths)
            assert text.count(" all-to-all(") == 2, (axis, len(widths), text.count(" all-to-all("))
    slab = P(("x", "y"), None, None)
    blocks = [rng.normal(size=(4 * b, 5, 3)) for b in (1, 3, 2)]
    got, _ = _join(mesh, slab, 0, blocks)
    assert np.array_equal(np.asarray(got), np.concatenate(blocks, axis=0))
