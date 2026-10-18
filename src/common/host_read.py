"""One device-to-host read of a pytree of device arrays, bit for bit.

A host decision that needs several small device results (gate flags, spectra, counts) reads
them in ONE transfer: the leaves are reinterpreted as bytes on the device, joined by one
program per tree signature, read once and cut back into numpy leaves of their own dtype and
shape.  Host leaves pass through.  Each leaf must be fully addressable on this process
(replicated), as ``np.asarray`` requires.
"""
from functools import lru_cache

__all__ = ["host_tree"]


def host_tree(tree):
    """Every leaf of a pytree of device arrays on the host in ONE device-to-host read, bit for bit.

    The leaves (any shape; bool, integer, float or complex) are reinterpreted as bytes on the
    device and joined (:func:`_host_bytes_program`, one program per tree signature), read once,
    and cut back into numpy leaves of their own dtype and shape; host leaves pass through. Each
    leaf must be fully addressable on this process (replicated), as ``np.asarray`` requires.
    """
    import jax
    import numpy as np
    leaves, treedef = jax.tree.flatten(tree)
    device = [i for i, a in enumerate(leaves) if isinstance(a, jax.Array)]
    if not device:
        return tree
    signature = tuple((tuple(leaves[i].shape), np.dtype(leaves[i].dtype).str) for i in device)
    raw = np.asarray(_host_bytes_program(signature)(*(leaves[i] for i in device)))
    out, at = list(leaves), 0
    for i, (shape, dt) in zip(device, signature):
        dt = np.dtype(dt)
        size = int(np.prod(shape, dtype=np.int64)) * dt.itemsize
        chunk = raw[at:at + size]
        at += size
        out[i] = (chunk.view(np.uint8).astype(bool) if dt == np.bool_ else chunk.view(dt)).reshape(shape)
    return jax.tree.unflatten(treedef, out)


@lru_cache(maxsize=None)
def _host_bytes_program(signature):
    import jax
    import jax.numpy as jnp
    import numpy as np

    def as_bytes(a):
        dt = np.dtype(a.dtype)
        if dt == np.bool_:
            return a.astype(jnp.uint8).reshape(-1)
        if np.issubdtype(dt, np.complexfloating):
            a = jnp.stack((jnp.real(a), jnp.imag(a)), axis=-1)
        return jax.lax.bitcast_convert_type(a, jnp.uint8).reshape(-1)
    return jax.jit(lambda *leaves: jnp.concatenate([as_bytes(a) for a in leaves]))


