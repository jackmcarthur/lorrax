"""Process-local placement of globally described JAX arrays.

This module owns the anti-collective used when every process already holds
the same host table.  A raw ``jax.device_put`` to a non-fully-addressable
sharding first asserts cross-process equality with an all-gather; declaring
each process's addressable shards avoids that P-linear transient.

JAX and NumPy imports stay inside the function so importing :mod:`lxkit`
remains stdlib-only and cannot initialize a backend before distributed JAX.
"""

from __future__ import annotations

__all__ = ["device_put_process_local", "device_put_process_tiles",
           "device_put_process_local_cached", "placement_cache_stats"]


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

    # A traced or already globally sharded array needs a real reshard.
    # Never np.asarray it: that would gather the value this helper exists to
    # keep distributed.
    is_device = isinstance(host_array, jax.Array)
    if isinstance(host_array, jax.core.Tracer) or (is_device and (
            not host_array.is_fully_addressable or
            bool(getattr(sharding, "is_fully_addressable", False)))):
        return jax.device_put(host_array, sharding)

    # A process-local device array also follows the anti-collective contract:
    # slice only this process's target shards on device, without either a host
    # replica or JAX's cross-process equality assertion on the global value.
    arr = host_array if is_device else np.asarray(host_array)
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
        piece = arr[idx] if is_device else np.asarray(arr[idx])
        # ``np.ascontiguousarray`` promotes a 0-D scalar to shape ``(1,)``.
        # Preserve the scalar shard shape required by a replicated P() value;
        # higher-rank shards still take the contiguous host-staging route.
        if piece.ndim and not is_device:
            piece = np.ascontiguousarray(piece)
        shards.append(jax.device_put(piece, dev))
    return jax.make_array_from_single_device_arrays(shape, sharding, shards)


def device_put_process_tiles(shape, sharding, tile):
    """Place a global array from its addressable shards, built on demand.

    ``tile(index)`` returns the host value of the global array at ``index``
    (a tuple of slices from ``sharding.addressable_devices_indices_map``).
    The same placement as :func:`device_put_process_local` of the whole host
    array, without the whole host array on any process: each process builds
    only its own shards.  ``tile`` must be a pure function of rank-invariant
    inputs.
    """
    import jax
    import numpy as np

    shape = tuple(int(s) for s in shape)
    shards = []
    for dev, idx in sharding.addressable_devices_indices_map(shape).items():
        piece = np.asarray(tile(idx))
        if piece.ndim:
            piece = np.ascontiguousarray(piece)
        shards.append(jax.device_put(piece, dev))
    return jax.make_array_from_single_device_arrays(shape, sharding, shards)


# Content-keyed device constants (``device_put_process_local_cached``).
# Bounded LRU: an entry is at most _CACHE_ENTRY_BYTES, the whole cache at most
# _CACHE_TOTAL_BYTES of device memory per device (control vectors and counts,
# never operators).  Stats feed the run's transfer census.
_CACHE_ENTRY_BYTES = 64 * 1024
_CACHE_TOTAL_BYTES = 8 * 1024 * 1024
_cache = None
_cache_lock = None
_stats = {"hits": 0, "misses": 0, "bypass": 0, "bytes": 0, "peak_bytes": 0}


def device_put_process_local_cached(host_array, sharding):
    """``device_put_process_local`` of a small control value, placed once per distinct value.

    Loop control operands that are pure functions of host plan data (slab
    offsets and handles, per-row retained counts, real-row counts, window
    counts) take few distinct values per run but were placed on every call:
    one host-to-device transfer per read, eigh or window.  This returns the
    device array placed the first time the same (sharding, dtype, shape,
    bytes) was asked for, so a repeated value costs no transfer; the result
    is bit-identical to an uncached placement.  Values above
    ``_CACHE_ENTRY_BYTES`` take the uncached route.

    Contract: the returned array is shared and immutable -- never donate it
    (JAX refuses a deleted buffer by name, so a violation is loud).  Every
    process must pass the same value (``device_put_process_local``).
    """
    import collections
    import threading

    import jax
    import numpy as np

    global _cache, _cache_lock
    if isinstance(host_array, jax.Array) or isinstance(host_array, jax.core.Tracer):
        return device_put_process_local(host_array, sharding)
    arr = np.asarray(host_array)
    if arr.dtype == object or arr.nbytes > _CACHE_ENTRY_BYTES:
        _stats["bypass"] += 1
        return device_put_process_local(arr, sharding)
    if _cache is None:
        _cache, _cache_lock = collections.OrderedDict(), threading.Lock()
    key = (sharding, arr.dtype.str, arr.shape, np.ascontiguousarray(arr).tobytes())
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            _stats["hits"] += 1
            return hit
    out = device_put_process_local(arr, sharding)
    with _cache_lock:
        _stats["misses"] += 1
        _cache[key] = out
        _stats["bytes"] += int(arr.nbytes)
        while _stats["bytes"] > _CACHE_TOTAL_BYTES and len(_cache) > 1:
            old_key, _ = _cache.popitem(last=False)
            _stats["bytes"] -= len(old_key[3])
        _stats["peak_bytes"] = max(_stats["peak_bytes"], _stats["bytes"])
    return out


def placement_cache_stats():
    """``{hits, misses, bypass, bytes, peak_bytes}`` of the cached placement (bytes per device)."""
    return dict(_stats)
