"""Face-pinned block glue for stacked operators on the ``('x','y')`` mesh.

Eager slicing, concatenation and ``a + a^H`` of face-sharded operands return
replicated arrays: a ``[b, R, R]`` block then occupies ``16 R^2`` bytes on
every rank instead of ``16 R^2/(Px Py)`` (measured on a 2x2 host mesh,
``runs/frequency_integration_sandbox/425_trint_20260915/logs/eager_sharding_probe.log``;
on CrI3 q=1 the ordered pencil reduction peak fell 21.3 -> 2.907 GiB/rank once
the same arithmetic was compiled with face output shardings, claim 2359).

Each helper here runs one elementwise program with the operand's own face as
its output sharding and keeps one executable per (function, layout, statics).
The arithmetic is unchanged, so results are bitwise equal to the eager form.
Traced operands (inside ``jit``/``shard_map``) and unsharded host arrays take
the plain function; the helpers are therefore safe at every call site.

Shapes: operators are ``[b, R, R]`` complex128 at ``P(None,'x','y')``;
column panels are ``[b, n, R]`` in the same layout. No helper gathers, pads or
reshards; a caller that needs a different layout converts before calling.
"""
from __future__ import annotations

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding

__all__ = ["face_sharding", "on_face", "hermitian_part", "hermitian_block",
           "join_columns", "diagonal_like"]


def face_sharding(array):
    """Return the NamedSharding of a concrete array sharded on every axis, else None."""
    if isinstance(array, jax.core.Tracer):
        return None
    sharding = getattr(array, "sharding", None)
    if isinstance(sharding, NamedSharding) and len(sharding.spec) == array.ndim:
        return sharding
    return None


@lru_cache(maxsize=None)
def _pinned(fn, out, static):
    target = partial(fn, **dict(static)) if static else fn
    return jax.jit(target, out_shardings=out)


def on_face(fn, out, *operands, **static):
    """Evaluate ``fn(*operands, **static)`` with its outputs placed on ``out``.

    ``out`` is a NamedSharding, or a tuple of them matching ``fn``'s outputs.
    ``None``, or any traced operand, calls ``fn`` directly. ``fn`` must be a
    module-level function and ``static`` hashable Python values, so repeated
    calls with the same layout reuse one compiled program.
    """
    if out is None or any(isinstance(leaf, jax.core.Tracer)
                          for leaf in jax.tree.leaves(operands)):
        return fn(*operands, **static)
    return _pinned(fn, out, tuple(sorted(static.items())))(*operands)


def _adjoint(a):
    return jnp.conj(jnp.swapaxes(a, -1, -2))


def _hermitian(a):
    return (a + _adjoint(a)) * 0.5


def _hermitian_block(block, off, corner):
    return jnp.concatenate((jnp.concatenate((block, _adjoint(off)), axis=-1),
                            jnp.concatenate((off, corner), axis=-1)), axis=-2)


def _join(a, b):
    return jnp.concatenate((a, b), axis=-1)


def _diagonal(values, *, n, dtype):
    return jnp.eye(n, dtype=dtype)[None] * values[:, None, :]


def hermitian_part(a):
    """``(a + a^H)/2`` for ``[b, R, R]``, on ``a``'s face."""
    return on_face(_hermitian, face_sharding(a), a)


def hermitian_block(block, off, corner):
    """``[[block, off^H], [off, corner]]`` on ``block``'s face.

    ``block`` is ``[b, R, R]``, ``off`` is ``[b, r, R]`` and ``corner`` is
    ``[b, r, r]``; the result is ``[b, R + r, R + r]``.
    """
    return on_face(_hermitian_block, face_sharding(block), block, off, corner)


def join_columns(a, b):
    """Concatenate column panels ``[b, n, R]`` and ``[b, n, r]`` on ``a``'s face."""
    return on_face(_join, face_sharding(a), a, b)


def diagonal_like(values, like):
    """``diag(values)`` as ``[b, R, R]`` in ``like``'s dtype and face; ``values`` is ``[b, R]``."""
    return on_face(_diagonal, face_sharding(like), values,
                   n=int(like.shape[-1]), dtype=jnp.dtype(like.dtype))
