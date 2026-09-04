import os

_force_host = "--xla_force_host_platform_device_count=4"
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = f"{_xla_flags} {_force_host}".strip()

from jax import config

config.update("jax_enable_x64", True)

import numpy as np
import jax
import jax.numpy as jnp


def main() -> None:
    devices = jax.devices()[:4]

    key = jax.random.key(0)
    mat = jax.random.normal(key, (128, 128), dtype=jnp.complex128)
    vec = jax.random.normal(key, (128,), dtype=jnp.complex128)

    # Split matrix into 2x2 blocks and place each on a device
    blocks = [
        mat[i * 64 : (i + 1) * 64, j * 64 : (j + 1) * 64]
        for i in range(2)
        for j in range(2)
    ]
    mat_sharded = jax.device_put_sharded(blocks, devices)

    print("Shard layout:")
    for idx, shard in enumerate(mat_sharded.addressable_shards):
        row, col = divmod(idx, 2)
        print(f"block ({row}, {col}) -> {shard.device} {shard.data.shape}")

    # Each device gets the vector segment matching its column block
    vec_parts = [vec[:64], vec[64:]]
    vec_sharded = jnp.stack([vec_parts[0], vec_parts[1], vec_parts[0], vec_parts[1]])

    from functools import partial

    @partial(jax.pmap, axis_name="d")
    def mv_reduce(a, x):
        y = a @ x
        return jax.lax.psum(y, "d", axis_index_groups=[[0, 1], [2, 3]])

    row_sums = mv_reduce(mat_sharded, vec_sharded)
    result_rows = jax.device_get(row_sums[jnp.array([0, 2])])
    result_dist = result_rows.reshape(-1)

    result_local = mat @ vec

    print("Results close:", np.allclose(np.array(result_dist), np.array(result_local)))

if __name__ == '__main__':
    main()
