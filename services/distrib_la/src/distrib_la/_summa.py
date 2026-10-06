"""``summa`` — batched 2-D SUMMA on the x/y face, in pure JAX.

The products of a stack of parents, every parent's matrix tiled 2-D over all
P ranks at ``P(None,'x','y')``, as one explicit ``shard_map`` program: at
each of the ``p = Px = Py`` steps the ranks of a row gather A's panel of
k-columns and the ranks of a column gather B's panel of k-rows, one
``all_gather`` message each for the whole batch, and every rank adds the
batched local product. Each gather carries ``(p-1)/p`` of one tile per
operand, so a product moves ``2 n^2 / p`` elements per rank, the 2-D
optimum; the steps are unrolled, so XLA's asynchronous collectives overlap
step ``s+1``'s gathers with step ``s``'s GEMM.

The panel of step ``s`` is the ``s``-th fine block of every rank's tile:
rank ``y`` of a row holds the k-columns ``[y k/p, (y+1) k/p)``; the panel
is ``{y k/p + s kb + i}`` over all ``y`` (``kb = k/p^2``, the tile padded
with zero columns to a multiple of ``p``), and B's row panel over ``x`` is
the same index set in the same order, so the two panels contract on matching
k. No skew, no transpose of the grid inside the loop; an adjoint operand is
transposed once on the tile grid before the loop (one tile per rank).

Any backend: there is no vendor call. ``split`` runs ``split`` sub-steps per
fine block to halve the live panels when a stack is wide.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la._shard_map import shard_map
from distrib_la.resolve import mesh_key

__all__ = ["summa", "summa_program"]


def _adjoint_on_face(a, face):
    """conj(a)^T with its tile grid transposed back onto the face (one tile per rank)."""
    return jax.lax.with_sharding_constraint(jnp.conj(jnp.swapaxes(a, -1, -2)), face)


@lru_cache(maxsize=None)
def summa_program(mesh: Mesh, split: int = 1):
    """The jitted batched SUMMA of ``[B, m, k] x [B, k, n] -> [B, m, n]`` on ``mesh``."""
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    if px != py:
        raise ValueError(f"summa needs a square mesh, got {px}x{py}")
    p = px
    face = NamedSharding(mesh, P(None, "x", "y"))

    def local(a, b):
        nb, ml, kl = a.shape
        nl = b.shape[-1]
        kb = -(-kl // (p * split))                  # fine block per rank and sub-step
        pad = kb * p * split - kl
        if pad:
            a = jnp.pad(a, ((0, 0), (0, 0), (0, pad)))
            b = jnp.pad(b, ((0, 0), (0, pad), (0, 0)))

        def panels(s):
            a_s = jax.lax.all_gather(a[:, :, s * kb:(s + 1) * kb], "y", axis=2, tiled=True)
            b_s = jax.lax.all_gather(b[:, s * kb:(s + 1) * kb, :], "x", axis=1, tiled=True)
            return a_s, b_s
        c = jnp.zeros((nb, ml, nl), jnp.result_type(a.dtype, b.dtype))
        a_cur, b_cur = panels(0)
        for s in range(p * split):
            if s + 1 < p * split:
                a_nxt, b_nxt = panels(s + 1)
            c = c + jnp.matmul(a_cur, b_cur)
            if s + 1 < p * split:
                a_cur, b_cur = a_nxt, b_nxt
        return c
    mapped = shard_map(local, mesh=mesh, in_specs=(P(None, "x", "y"), P(None, "x", "y")),
                       out_specs=P(None, "x", "y"), check_vma=False)
    return jax.jit(mapped), face


def summa(a, b, *, mesh: Mesh, transa: str = "N", transb: str = "N", split: int = 1):
    """``op(a) @ op(b)`` for face stacks ``[B, ., .]``; ``'C'`` takes the adjoint, ``'T'`` the transpose."""
    program, face = summa_program(mesh, int(split))
    if transa != "N":
        a = _adjoint_on_face(a, face) if transa == "C" else jnp.conj(_adjoint_on_face(a, face))
    if transb != "N":
        b = _adjoint_on_face(b, face) if transb == "C" else jnp.conj(_adjoint_on_face(b, face))
    return program(a, b)
