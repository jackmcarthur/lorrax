"""Replicated int32 device scalars made on the device (compiled constants), not placed.

A program that takes an index as a traced argument (a dynamic slice start, a sample id)
reads it from here: the scalars of one block are the outputs of ONE compiled program per
(mesh, block), made once per process, so no index crosses from the host per call.
"""
from functools import lru_cache

__all__ = ["device_index"]

#: Device scalar indices are made in blocks of this many (one program per block size).
_INDEX_BLOCK = 64


@lru_cache(maxsize=None)
def _index_block(mesh_xy, block):
    """Replicated int32 scalars ``block * B .. (block + 1) * B - 1``, made on the device by one
    compiled program (constants in the executable: nothing crosses from the host)."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    rep = NamedSharding(mesh_xy, P())
    first = block * _INDEX_BLOCK
    return jax.jit(lambda: tuple(jnp.asarray(first + i, jnp.int32) for i in range(_INDEX_BLOCK)),
                   out_shardings=(rep,) * _INDEX_BLOCK)()


def device_index(mesh_xy, i):
    """The replicated int32 scalar ``i`` on ``mesh_xy``, made on the device once per process."""
    i = int(i)
    if i < 0:
        raise ValueError("device_index: a non-negative index")
    return _index_block(mesh_xy, i // _INDEX_BLOCK)[i % _INDEX_BLOCK]
