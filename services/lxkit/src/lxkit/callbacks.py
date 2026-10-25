"""Host callbacks whose host functions see NumPy arrays, never ``jax.Array``.

JAX hands a ``pure_callback`` / ``io_callback`` / ``debug.callback`` host
function its arguments as ``jax.Array``.  A slice, reshape or arithmetic on one
dispatches a new computation to the device that is still running the caller's
program, and on the CPU client the two can wait on each other for ever (the CPU
Davidson stall, 4aba67e0d: every thread in a futex at 0-3% CPU, an XLA thread
inside ``lax.reshape`` in ``subspace.project``'s callback).  ``np.asarray`` of a
callback argument is a host view and dispatches nothing, so every wrapper here
converts each argument leaf before the host function runs.

Every host callback in the tree goes through these three; the lint
``tools/lint/test_host_callbacks.py`` refuses a raw ``jax.pure_callback``,
``io_callback`` or ``jax.debug.callback`` anywhere else.  jax and numpy are
imported in the function bodies (``import lxkit`` stays stdlib-only).
"""

from __future__ import annotations


def _numpy_views(fn):
    """``fn`` with every argument leaf replaced by its ``np.asarray`` host view."""
    import jax
    import numpy as np

    def host(*args, **kwargs):
        args, kwargs = jax.tree_util.tree_map(np.asarray, (args, kwargs))
        return fn(*args, **kwargs)
    return host


def host_pure_callback(fn, result_shape, *args, **kwargs):
    """``jax.pure_callback(fn, result_shape, *args)`` with ``fn`` given NumPy views."""
    import jax
    return jax.pure_callback(_numpy_views(fn), result_shape, *args, **kwargs)


def host_io_callback(fn, result_shape, *args, **kwargs):
    """``jax.experimental.io_callback(fn, result_shape, *args)`` with ``fn`` given NumPy views."""
    from jax.experimental import io_callback
    return io_callback(_numpy_views(fn), result_shape, *args, **kwargs)


def host_debug_callback(fn, *args, **kwargs):
    """``jax.debug.callback(fn, *args)`` with ``fn`` given NumPy views."""
    import jax
    return jax.debug.callback(_numpy_views(fn), *args, **kwargs)
