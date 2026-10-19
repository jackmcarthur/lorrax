"""The shared mini-BZ head average gives every rank one share shape (CPU).

The compile agreement refuses when ranks compile different shapes, and a
2^18-point draw splits unevenly over P = 9, 25, 36, 49 or 100 ranks. Three
ranks are simulated on a 10-point draw (the uneven split is 3, 3, 4); the
shares must have one shape and sum to the one-process average.
"""
import numpy as np

from ffi import _services

_services.ensure_on_path()

import jax.numpy as jnp  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402
from vcoul import bulk_3d  # noqa: E402


def test_every_rank_takes_one_share_shape(monkeypatch):
    rng = np.random.default_rng(0)
    batches = [jnp.asarray(rng.normal(size=(10, 3)) + 2.0) for _ in range(2)]
    S = [0.1 * np.eye(3), 0.2 * np.eye(3)]
    whole = np.asarray(bulk_3d.Bulk3D()._screened_means(batches, S, None))
    shapes, parts = set(), []

    def extra(q):
        shapes.add(tuple(q.shape))
        return jnp.zeros((2, q.shape[0]), jnp.complex128)

    monkeypatch.setattr(multihost_utils, "process_allgather",
                        lambda x, tiled: np.asarray(x)[None])
    monkeypatch.setattr(bulk_3d.jax, "process_count", lambda: 3)
    for rank in range(3):
        monkeypatch.setattr(bulk_3d.jax, "process_index", lambda r=rank: r)
        parts.append(np.asarray(bulk_3d.Bulk3D()._screened_means(
            batches, S, extra, shared=True)))
    assert shapes == {(4, 3)}
    np.testing.assert_allclose(sum(parts), whole, rtol=1e-13, atol=0.0)
