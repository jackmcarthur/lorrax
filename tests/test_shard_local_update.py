"""Independent rectangle insertion contracts, including wrapped/drop tails."""
import itertools

import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common.staged_reshard import shard_local_update


@pytest.fixture
def mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def rectangle_oracle(dst, tile, starts):
    """Enumerate coordinates, normalize negatives once, drop out-of-range."""
    expected = dst.copy()
    starts = np.asarray(starts).astype(np.int32)
    for source in itertools.product(*(range(size) for size in tile.shape)):
        target = [int(starts[d]) + source[d] for d in range(dst.ndim)]
        target = [value + dst.shape[d] if value < 0 else value
                  for d, value in enumerate(target)]
        if all(0 <= value < dst.shape[d] for d, value in enumerate(target)):
            expected[tuple(target)] = tile[source]
    return expected


@pytest.mark.parametrize("tile_shape,starts,constant", [
    ((2, 2, 3), (0, 0, 0), False),
    ((2, 2, 3), (2, 3, 3), False),
    ((4, 5, 6), (0, 0, 0), False),
    ((2, 2, 3), (3, 4, 4), False),
    ((2, 2, 3), (-1, -1, -1), False),
    ((2, 2, 3), (-5, -6, -7), False),
    ((2, 2, 3), (4, 5, 6), False),
    ((2, 2, 3), (-10, -10, -10), False),
    ((2, 2, 8), (0, 0, 0), False),
    # Wrapped and positive coordinates collide here. Equal values make
    # the public unordered scatter's result deterministic without claiming
    # an order for conflicting writes.
    ((2, 2, 8), (0, 0, -2), True),
    ((2, 2, 3), (1.9, 2.9, 2.9), False),
    ((0, 2, 3), (1, 1, 1), False),
])
def test_rectangle_update_matches_independent_drop_oracle(
        mesh, tile_shape, starts, constant):
    dst = np.arange(4 * 5 * 6, dtype=np.float64).reshape(4, 5, 6) + 1j
    tile = (np.full(tile_shape, 7. + 3j) if constant else
            np.arange(np.prod(tile_shape), dtype=np.float64).reshape(tile_shape)
            + 1000. + 2j)
    expected = rectangle_oracle(dst, tile, starts)
    sharding = NamedSharding(mesh, P("x", "y", None))
    actual = shard_local_update(mesh, spec=sharding.spec)(
        jax.device_put(dst, sharding), jax.device_put(tile, sharding),
        jax.device_put(np.asarray(starts), NamedSharding(mesh, P())))
    np.testing.assert_array_equal(np.asarray(actual), expected)


def test_rectangle_update_rejects_wrong_starts_rank(mesh):
    sharding = NamedSharding(mesh, P("x", "y"))
    dst = jax.device_put(np.zeros((4, 4)), sharding)
    tile = jax.device_put(np.ones((2, 2)), sharding)
    with pytest.raises(ValueError, match="starts.shape"):
        shard_local_update(mesh, spec=sharding.spec)(
            dst, tile, jax.device_put(np.asarray([0]), NamedSharding(mesh, P())))
