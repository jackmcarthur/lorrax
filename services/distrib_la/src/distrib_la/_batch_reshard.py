"""Staged face ↔ batch movement for :class:`distrib_la.plan.Plan`.

This is the independently-installable service sibling of LORRAX's
``common.staged_reshard.face_to_batch_reshard``.  ``distrib_la`` cannot
import ``common``: the service is installable with only JAX and ``lxkit``.
The movement contract is nevertheless the same, including its reason for
being explicit rather than a pair of sharding constraints::

    (B, M, N) P(None, 'x', 'y')
      -- all_to_all x: split B, join M
      -- all_to_all y: split B, join N
    (B, M, N) P(('x','y'), None, None)

The reverse is the literal inverse schedule (``y`` then ``x``).  Both
directions and the local dense operation live inside ONE ``shard_map``.
Consequently GSPMD never sees a direct face→batch or batch→face reshard it
could lower as replicate-then-partition, and no full matrix crosses the host.

The leading batch need not divide the mesh.  It is padded locally before
the first exchange and dropped after the inverse exchange.  Synthetic local
rows never enter a dense kernel: a scalar ``lax.cond`` around each local slot
runs the operation only when its global q index is real.  Matrix dimensions
still have to tile the incoming
``P(None,'x','y')`` face exactly.  Padding those dimensions would change the
linear-algebra problem, so a consumer that needs it must pad before calling
the plan and slice the result afterward; this route pads only the leading
batch.
"""
from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from distrib_la._shard_map import shard_map
from distrib_la.resolve import mesh_key

__all__ = ["batch_reshard_call", "validate_batch_reshard_operands"]


_JIT_CACHE: dict = {}


def validate_batch_reshard_operands(
    op: str, mesh: Mesh, ops: Sequence,
) -> tuple[int, int]:
    """Validate route-(c) operands and return ``(nbatch, batch_pad)``.

    This is eager shape algebra only: every refusal happens before a
    collective is entered.  The route accepts a ragged leading batch and
    pads it itself, but the matrix face must already tile the mesh.
    """
    if op not in ("eigh", "cholesky", "solve_lu"):
        raise ValueError(
            f"batch_reshard: unsupported op {op!r}; expected "
            "eigh|cholesky|solve_lu")
    expected = 2 if op == "solve_lu" else 1
    if len(ops) != expected:
        raise ValueError(
            f"batch_reshard {op}: expected {expected} operand(s), got "
            f"{len(ops)}")

    axes = tuple(mesh.axis_names)
    if "x" not in axes or "y" not in axes or axes[-1] != "y":
        raise ValueError(
            f"batch_reshard: expected a mesh containing ('x','y') with "
            f"'y' minor, got axes={axes!r}")
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    ndev = px * py

    A = ops[0]
    if A.ndim != 3 or int(A.shape[1]) != int(A.shape[2]):
        raise ValueError(
            f"batch_reshard {op}: expected A of shape (B,N,N), got "
            f"{tuple(A.shape)}")
    nb, n = int(A.shape[0]), int(A.shape[1])
    if nb < 1:
        raise ValueError(
            f"batch_reshard {op}: the batch must be nonempty, got B={nb}")
    if n % px or n % py:
        raise ValueError(
            f"batch_reshard {op}: the matrix face must tile the {px}x{py} "
            f"mesh exactly, but N={n} has remainders ({n % px},{n % py}). "
            f"Pad the matrix extent before calling and slice the result "
            f"afterward; only the leading batch is padded by this route.")

    if op == "solve_lu":
        B = ops[1]
        if (B.ndim != 3 or int(B.shape[0]) != nb
                or int(B.shape[1]) != n):
            raise ValueError(
                f"batch_reshard solve_lu: B must be (B,N,NRHS) with B,N "
                f"matching A; got A={tuple(A.shape)}, B={tuple(B.shape)}")
        if A.dtype != B.dtype:
            raise ValueError(
                f"batch_reshard solve_lu: A.dtype {A.dtype} != "
                f"B.dtype {B.dtype}")
        nrhs = int(B.shape[2])
        if nrhs % py:
            raise ValueError(
                f"batch_reshard solve_lu: the RHS face is sharded over "
                f"'y', but NRHS={nrhs} is not divisible by Py={py}.  Pad "
                f"the RHS columns before calling and slice the result "
                f"afterward.")

    nb_pad = -(-nb // ndev) * ndev
    return nb, nb_pad - nb


def _face_to_batch(a, *, px: int, py: int):
    """Two volume-preserving exchanges: face tile → whole matrices."""
    if px > 1:
        a = jax.lax.all_to_all(
            a, "x", split_axis=0, concat_axis=1, tiled=True)
    if py > 1:
        a = jax.lax.all_to_all(
            a, "y", split_axis=0, concat_axis=2, tiled=True)
    return a


def _batch_to_face(a, *, px: int, py: int):
    """Literal inverse of :func:`_face_to_batch`: ``y`` then ``x``."""
    if py > 1:
        a = jax.lax.all_to_all(
            a, "y", split_axis=2, concat_axis=0, tiled=True)
    if px > 1:
        a = jax.lax.all_to_all(
            a, "x", split_axis=1, concat_axis=0, tiled=True)
    return a


def _pad_leading(a, amount: int):
    if amount == 0:
        return a
    return jnp.pad(a, ((0, amount), (0, 0), (0, 0)))


def _dense_real_rows(
    op: str, A, B=None, *, nbatch: int, py: int,
):
    """Apply one dense operation only to real local batch rows.

    After face-to-batch movement each device owns ``ceil(Q/P)`` whole
    matrices.  The last local slots may be synthetic padding.  A vmapped
    conditional is not sufficient here because a batched predicate may run
    both branches; the scalar ``fori_loop`` + ``lax.cond`` is the same
    schedule used by ``isdf.core._factor_c_q_replicated_qparallel`` and makes
    the expensive branch unreachable for a synthetic global q index.
    """
    local_nb, n, _ = (int(v) for v in A.shape)
    device = (jax.lax.axis_index("x") * py + jax.lax.axis_index("y"))
    first_q = device * local_nb

    if op == "eigh":
        w0 = jnp.zeros((local_nb, n), dtype=jnp.real(A).dtype)
        z0 = jnp.zeros_like(A)

        def _one(i, acc):
            W, Z = acc
            A1 = jax.lax.dynamic_slice_in_dim(A, i, 1, axis=0)

            def _work(a):
                result = jnp.linalg.eigh(a)
                # EighResult is a named tuple while the neutral branch is a
                # plain tuple; lax.cond requires identical pytree node types.
                return result[0], result[1]

            def _skip(a):
                return (jnp.zeros((1, n), dtype=jnp.real(a).dtype),
                        jnp.zeros_like(a))

            W1, Z1 = jax.lax.cond(first_q + i < nbatch, _work, _skip, A1)
            return (jax.lax.dynamic_update_slice(W, W1, (i, 0)),
                    jax.lax.dynamic_update_slice(Z, Z1, (i, 0, 0)))

        return jax.lax.fori_loop(0, local_nb, _one, (w0, z0))

    out0 = (jnp.zeros_like(A) if op == "cholesky"
            else jnp.zeros_like(B))

    def _one(i, out):
        A1 = jax.lax.dynamic_slice_in_dim(A, i, 1, axis=0)
        if op == "cholesky":
            operand = A1

            def _work(a):
                return jnp.linalg.cholesky(a)

            def _skip(a):
                return jnp.zeros_like(a)
        else:
            B1 = jax.lax.dynamic_slice_in_dim(B, i, 1, axis=0)
            operand = (A1, B1)

            def _work(ab):
                return jnp.linalg.solve(ab[0], ab[1])

            def _skip(ab):
                return jnp.zeros_like(ab[1])

        out1 = jax.lax.cond(
            first_q + i < nbatch, _work, _skip, operand)
        return jax.lax.dynamic_update_slice(out, out1, (i, 0, 0))

    return jax.lax.fori_loop(0, local_nb, _one, out0)


def _replicate_batch_vector(v, *, px: int, py: int):
    """Restore the service's replicated eigenvalue-vector contract."""
    if px * py == 1:
        return v
    return jax.lax.all_gather(
        v, ("x", "y"), axis=0, tiled=True)


def batch_reshard_call(
    op: str,
    mesh: Mesh,
    ops: Sequence,
):
    """Run route (c): staged face→batch, local dense op, staged inverse.

    The returned arrays obey the ordinary :meth:`Plan.batched` layout:
    matrix outputs at ``P(None,'x','y')`` and eigh eigenvalues replicated.
    """
    ops = tuple(ops)
    nbatch, batch_pad = validate_batch_reshard_operands(op, mesh, ops)
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])

    key = (
        op,
        mesh_key(mesh),
        tuple((tuple(int(s) for s in x.shape), str(x.dtype)) for x in ops),
    )
    fn = _JIT_CACHE.get(key)
    if fn is None:
        in_specs = tuple(P(None, "x", "y") for _ in ops)
        out_specs = ((P(), P(None, "x", "y")) if op == "eigh"
                     else P(None, "x", "y"))

        def _body(*local_faces):
            local = tuple(
                _face_to_batch(_pad_leading(a, batch_pad), px=px, py=py)
                for a in local_faces)
            A = local[0]

            if op == "eigh":
                if batch_pad:
                    W, Z = _dense_real_rows(
                        op, A, nbatch=nbatch, py=py)
                else:
                    W, Z = jnp.linalg.eigh(A)
                W = _replicate_batch_vector(W, px=px, py=py)[:nbatch]
                Z = _batch_to_face(Z, px=px, py=py)[:nbatch]
                return W, Z
            if op == "cholesky":
                L = (_dense_real_rows(op, A, nbatch=nbatch, py=py)
                     if batch_pad else jnp.linalg.cholesky(A))
                return _batch_to_face(L, px=px, py=py)[:nbatch]

            X = (_dense_real_rows(
                    op, A, local[1], nbatch=nbatch, py=py)
                 if batch_pad else jnp.linalg.solve(A, local[1]))
            return _batch_to_face(X, px=px, py=py)[:nbatch]

        mapped = shard_map(
            _body, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
            check_vma=False)
        # Do not request donation here.  The two explicit exchanges on each
        # side prevent XLA from aliasing a face input to the final face
        # output; asking anyway emits "donated buffers were not usable" on
        # every call.  Plan.donates remains the conservative caller contract
        # (a caller never relies on survival), while this route stays quiet.
        fn = jax.jit(mapped)
        _JIT_CACHE[key] = fn
    return fn(*ops)
