"""Process-local placement of globally described JAX arrays.

This module owns the anti-collective used when every process already holds
the same host table.  A raw ``jax.device_put`` to a non-fully-addressable
sharding first asserts cross-process equality with an all-gather; declaring
each process's addressable shards avoids that P-linear transient.

JAX and NumPy imports stay inside the function so importing :mod:`lxkit`
remains stdlib-only and cannot initialize a backend before distributed JAX.
"""

from __future__ import annotations

__all__ = ["device_put_process_local"]


def device_put_process_local(host_array, sharding, *, check: bool | None = None):
    """Place a process-replicated host array without a hidden collective.

    This is the sanctioned replacement for ``jax.device_put(host_array,
    sharding)`` when every process already holds the same global host value
    and ``sharding`` spans more than one process.  Each process slices only
    the shards owned by its addressable devices and declares the global array
    with ``jax.make_array_from_single_device_arrays``.

    ``host_array`` must be bit-identical on every process.  Set
    ``LORRAX_CHECK_REPLICA=1`` (or pass ``check=True``) to opt back into
    JAX's equality assertion and its associated all-gather.
    """
    import os

    import jax
    import numpy as np

    # An already globally sharded array is a real reshard, not host staging.
    # Never np.asarray it: that would gather the value this helper exists to
    # keep distributed.
    if isinstance(host_array, jax.Array) and not host_array.is_fully_addressable:
        return jax.device_put(host_array, sharding)

    arr = np.asarray(host_array)
    if jax.process_count() <= 1 or bool(
            getattr(sharding, "is_fully_addressable", False)):
        return jax.device_put(arr, sharding)

    if check is None:
        check = os.environ.get("LORRAX_CHECK_REPLICA", "0").strip().lower() \
            not in ("", "0", "false", "no", "off")
    if check:
        return jax.device_put(arr, sharding)

    shape = tuple(int(s) for s in arr.shape)
    idx_map = sharding.addressable_devices_indices_map(shape)
    if not idx_map:
        # JAX also skips its equality assertion when this process owns no
        # device in the target sharding.
        return jax.device_put(arr, sharding)
    shards = []
    for dev, idx in idx_map.items():
        piece = np.asarray(arr[idx])
        # ``np.ascontiguousarray`` promotes a 0-D scalar to shape ``(1,)``.
        # Preserve the scalar shard shape required by a replicated P() value;
        # higher-rank shards still take the contiguous host-staging route.
        if piece.ndim:
            piece = np.ascontiguousarray(piece)
        shards.append(jax.device_put(piece, dev))
    return jax.make_array_from_single_device_arrays(shape, sharding, shards)
