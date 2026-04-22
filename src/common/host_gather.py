"""Small host-gather helpers shared across GW drivers.

These wrap ``jax.experimental.multihost_utils.process_allgather`` with a
fallback to ``jax.device_get`` so single-process / replicated paths just
work.  Useful wherever a module needs a shard-aware D2H gather but doesn't
want to depend on the multihost import succeeding.

All three helpers are pure host-side utilities — no jit tracing, no
device arrays as return values.  They belong here rather than inside any
particular GW kernel module so chi0 / cohsex_sigma / ppm_sigma / future
callers can all reach for the same implementation.

(Note: ``minimax_screening._to_host_np`` is a close sibling with a
slightly different ``tiled=True`` default, kept local there because the
PPM-fit path gathers per-q arrays with a different tiling convention.
If the two converge further, unify then.)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def to_host_np(a, dtype=np.complex128, *, tiled: bool = False):
    """Gather a possibly sharded array to host as a numpy array.

    Parameters
    ----------
    a
        JAX array or anything ``np.asarray`` accepts.
    dtype
        NumPy dtype to cast the result to (default complex128).
    tiled
        Passed through to ``process_allgather``.  ``tiled=False``
        (default) returns the globally-replicated shape; ``tiled=True``
        returns a per-process tile concatenation.
    """
    try:
        return np.asarray(
            jax.experimental.multihost_utils.process_allgather(a, tiled=tiled),
            dtype=dtype,
        )
    except Exception:
        return np.asarray(jax.device_get(a), dtype=dtype)


def to_host_scalar(a, dtype=float):
    """Fetch a scalar value from a possibly-sharded device array.

    ``dtype`` may be a Python scalar type (``float``, ``int``) or a numpy
    dtype; the wrapped value comes back via ``dtype(...)`` at the end so
    ``int(...)`` stays an int, ``float(...)`` stays a float.
    """
    np_dtype = np.dtype(dtype)
    gathered = to_host_np(jnp.asarray(a), dtype=np_dtype, tiled=False)
    return dtype(np.asarray(gathered).reshape(-1)[0])


def masked_stats_device(values, mask):
    """Return (total, count, min, max) of ``values[mask]`` via device reductions.

    ``total`` is the full array size (Python int), ``count`` is the number
    of True entries, and ``min``/``max`` are ``None`` when ``count == 0``.
    The min/max reductions run on-device with ``jnp.where`` masking so the
    full array never lands on host; only the three scalar results do.
    """
    total = int(np.prod(values.shape))
    count = int(to_host_scalar(jnp.sum(mask, dtype=jnp.int64), int))
    if count == 0:
        return total, 0, None, None
    min_val = float(to_host_scalar(jnp.min(jnp.where(mask, values, jnp.inf)), float))
    max_val = float(to_host_scalar(jnp.max(jnp.where(mask, values, -jnp.inf)), float))
    return total, count, min_val, max_val
