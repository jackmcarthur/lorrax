"""Face ↔ batch movement for :class:`distrib_la.plan.Plan`.

This is the independently-installable service sibling of LORRAX's
``common.staged_reshard.face_to_batch_reshard``.  ``distrib_la`` cannot
import ``common``: the service is installable with only JAX and ``lxkit``.
The movement is explicit rather than a pair of sharding constraints, for the
same reason::

    (B, M, N) P(None, 'x', 'y')
      -- one all_to_all over ('x', 'y'): split B, join the (M, N) tiles
    (B, M, N) P(('x','y'), None, None)

Every byte crosses the network once.  The staged form (``x`` then ``y``)
moved each tile twice, and at Ni 20^3 P64 the W(τ) exchange of the
whole-parent synthesis paid for it at every τ node.  The reverse is the
literal inverse.  Both directions and the local dense operation live inside
ONE ``shard_map``.
Consequently GSPMD never sees a direct face→batch or batch→face reshard it
could lower as replicate-then-partition, and no full matrix crosses the host.

The leading batch need not divide the mesh.  It is padded locally before
the first exchange and dropped after the inverse exchange.  Synthetic local
rows never enter a factorization or a solve: a scalar ``lax.cond`` around each
local slot runs the operation only when its global q index is real. An eigh
solves the whole local stack in one batched call instead
(:func:`_local_stack_eigh`): the synthetic rows enter as exact zeros and leave
as exact zeros.  Matrix dimensions
still have to tile the incoming
``P(None,'x','y')`` face exactly.  Padding those dimensions would change the
linear-algebra problem, so a consumer that needs it must pad before calling
the plan and slice the result afterward; this route pads only the leading
batch.
"""
from __future__ import annotations

from typing import Sequence
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la._result_check import native_eigh
from distrib_la._shard_map import shard_map
from distrib_la.resolve import mesh_key

__all__ = ["batch_layout", "batch_layout_eigh_call", "batch_reshard_call",
           "is_batch_layout", "local_batch", "reshard_program", "reshard_rounds_call",
           "validate_batch_reshard_operands"]


_JIT_CACHE: dict = {}


def validate_batch_reshard_operands(
    op: str, mesh: Mesh, ops: Sequence,
) -> tuple[int, int]:
    """Validate route-(c) operands and return ``(nbatch, batch_pad)``.

    This is eager shape algebra only: every refusal happens before a
    collective is entered.  The route accepts a ragged leading batch and
    pads it itself, but the matrix face must already tile the mesh.
    """
    if op not in ("eigh", "checked_eigh", "normal_eigh", "polar", "cholesky", "solve_lu"):
        raise ValueError(
            f"batch_reshard: unsupported op {op!r}; expected "
            "eigh|checked_eigh|normal_eigh|polar|cholesky|solve_lu")
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
    """One volume-preserving exchange over both axes: face tile -> whole matrices.

    Rank ``r = x*py + y`` receives batch rows ``[r*B/P, (r+1)*B/P)`` of every
    rank's ``(B, M/px, N/py)`` tile and places tile ``(x', y')`` at rows
    ``x'*M/px`` and columns ``y'*N/py``.
    """
    p = px * py
    if p == 1:
        return a
    b, m, n = a.shape
    a = jax.lax.all_to_all(a, ("x", "y"), split_axis=0, concat_axis=0, tiled=True)
    a = a.reshape(px, py, b // p, m, n)
    return jnp.transpose(a, (2, 0, 3, 1, 4)).reshape(b // p, px * m, py * n)


def _batch_to_face(a, *, px: int, py: int):
    """Literal inverse of :func:`_face_to_batch`: whole matrices -> face tile, one exchange."""
    p = px * py
    if p == 1:
        return a
    b, m, n = a.shape
    a = a.reshape(b, px, m // px, py, n // py)
    a = jnp.transpose(a, (1, 3, 0, 2, 4)).reshape(p * b, m // px, n // py)
    return jax.lax.all_to_all(a, ("x", "y"), split_axis=0, concat_axis=0, tiled=True)


def _pad_leading(a, amount: int):
    if amount == 0:
        return a
    return jnp.pad(a, ((0, amount), (0, 0), (0, 0)))


def _checked_eigh(a):
    """Check Hermiticity where eigh already owns local complete matrices.

    The reduction fuses the local transpose/difference, with no face
    exchange. Invalid local batches skip the solver and return a NaN spectrum;
    the existing replicated-spectrum readback refuses them on every rank.
    """
    defect = jnp.max(jnp.abs(a - jnp.conj(jnp.swapaxes(a, -1, -2))), axis=(-2, -1))
    scale = jnp.max(jnp.abs(a), axis=(-2, -1))
    valid = jnp.all(jnp.isfinite(scale) & (defect <= 1e-12 * scale))
    def solve(value):
        return native_eigh(value)
    def refuse(value):
        return (jnp.full(value.shape[:-1], jnp.nan, dtype=value.real.dtype),
                jnp.zeros_like(value))
    return jax.lax.cond(valid, solve, refuse, a)


def _local_stack_eigh(op: str, A, *, nbatch: int, py: int):
    """Eigh of a rank's whole local stack in ONE batched call.

    ``op`` is ``eigh``, ``checked_eigh`` or ``normal_eigh`` (ascending
    singular values and right singular vectors from the eigh of A.H A).
    A row whose global index is ``>= nbatch`` is synthetic: it enters the
    solver as an exact zero matrix and its outputs are exact zeros, and the
    Hermiticity check passes them. The batched solver amortizes its launches
    over the stack: against one call per row it is 1.36x faster at 8 rows and
    2.2x at 32 rows of complex n = 912 (one A100, 2026-09-30). A synthetic row
    costs no wall time, because rank 0 always holds a full stack of real rows.
    """
    from distrib_la.polar import _normal_svd
    local_nb = int(A.shape[0])
    first = (jax.lax.axis_index("x") * py + jax.lax.axis_index("y")) * local_nb
    real = first + jnp.arange(local_nb) < nbatch
    A = jnp.where(real[:, None, None], A, jnp.zeros((), A.dtype))
    if op == "normal_eigh":
        w, z = _normal_svd(A, jnp.linalg.eigh)
    elif op == "checked_eigh":
        w, z = _checked_eigh(A)
    else:
        w, z = native_eigh(A)
    return (jnp.where(real[:, None], w, jnp.zeros((), w.dtype)),
            jnp.where(real[:, None, None], z, jnp.zeros((), z.dtype)))


def _dense_real_rows(
    op: str, A, B=None, *, nbatch: int, py: int,
):
    """Apply one Cholesky or LU solve only to real local batch rows.

    After face-to-batch movement each device owns ``ceil(Q/P)`` whole
    matrices.  The last local slots may be synthetic padding.  A vmapped
    conditional is not sufficient here because a batched predicate may run
    both branches; the scalar ``fori_loop`` + ``lax.cond`` is the same
    schedule used by ``isdf.core._factor_c_q_replicated_qparallel`` and makes
    the expensive branch unreachable for a synthetic global q index.
    """
    local_nb = int(A.shape[0])
    device = (jax.lax.axis_index("x") * py + jax.lax.axis_index("y"))
    first_q = device * local_nb

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


def _real_rows(kernel, operands, *, nbatch: int, py: int):
    """Apply ``kernel`` row by row to the real local batch rows only.

    ``operands`` are whole local matrices ``(local_nb, ., .)`` (``None``
    entries pass through); ``kernel`` maps one-row slices to one output row.
    The output of a synthetic row is exact zeros. This is the schedule of
    :func:`_dense_real_rows` for an arbitrary local kernel, used where the
    output shape differs from every operand (GEMM).
    """
    live = [x for x in operands if x is not None]
    local_nb = int(live[0].shape[0])
    device = (jax.lax.axis_index("x") * py + jax.lax.axis_index("y"))
    first_q = device * local_nb
    row = lambda i: tuple(None if x is None else jax.lax.dynamic_slice_in_dim(x, i, 1, axis=0)
                          for x in operands)
    template = jax.eval_shape(kernel, *row(0))
    out0 = jax.tree.map(lambda x: jnp.zeros((local_nb,) + tuple(x.shape[1:]), x.dtype), template)

    def _one(i, out):
        value = jax.lax.cond(first_q + i < nbatch,
                             lambda ops: kernel(*ops),
                             lambda ops: jax.tree.map(lambda x: jnp.zeros(x.shape, x.dtype), template),
                             row(i))
        return jax.tree.map(lambda a, v: jax.lax.dynamic_update_slice(
            a, v, (i,) + (0,) * (a.ndim - 1)), out, value)

    return jax.lax.fori_loop(0, local_nb, _one, out0)


def _batch_spec(ndim: int):
    """The batch layout of a rank-``ndim`` stack: rows over ``('x','y')``."""
    return P(("x", "y"), *([None] * (ndim - 1)))


def is_batch_layout(a, mesh) -> bool:
    """True when ``a`` is a concrete stack already in the batch layout on ``mesh``.

    That is ``(Bp, ...)`` at ``P(('x','y'), None, ...)`` with ``Bp`` a
    multiple of ``Px*Py``: rank ``x*Py + y`` owns rows
    ``[rank*Bp/P, (rank+1)*Bp/P)`` as whole trailing blocks. A tracer, a host
    array or any other sharding answers ``False``.
    """
    sharding = getattr(a, "sharding", None)
    if not isinstance(sharding, NamedSharding) or sharding.mesh != mesh:
        return False
    ndev = int(mesh.shape["x"]) * int(mesh.shape["y"])
    return (a.ndim >= 1 and int(a.shape[0]) % ndev == 0
            and tuple(sharding.spec) + (None,) * (a.ndim - len(sharding.spec))
            == tuple(_batch_spec(a.ndim)))


def batch_layout(a, mesh):
    """Place a batch in the batch layout ONCE, to stay resident across calls.

    ``a`` is either a face stack ``(B, M, N)`` at ``P(None,'x','y')``, which
    moves by the same two exchanges as every route-(c) call, or a fully
    replicated ``(B, ...)`` array (a pivot table, say), which each rank slices
    locally. The result is ``(Bp, ...)`` at ``P(('x','y'), None, ...)`` with
    ``Bp = ceil(B/(Px*Py))*Px*Py``; the ``Bp - B`` padded rows are zeros and
    never reach a :func:`local_batch` kernel. Per rank this holds
    ``ceil(B/P)`` whole blocks, the same bytes the route-(c) exchange puts
    there transiently on every call. Any other input layout refuses: an
    implicit reshard of a sharded stack is where GSPMD replicates.
    """
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    ndev = px * py
    nb = int(a.shape[0])
    pad = (-nb) % ndev
    sharding = getattr(a, "sharding", None)
    face_in = (a.ndim == 3 and isinstance(sharding, NamedSharding)
               and tuple(sharding.spec) == (None, "x", "y"))
    if not face_in and not getattr(sharding, "is_fully_replicated", False):
        raise ValueError(
            f"batch_layout: expected a (B,M,N) face stack at P(None,'x','y') or a "
            f"fully replicated array; got shape {tuple(a.shape)} with sharding "
            f"{sharding!r}. Place the operand in one of those layouts first.")
    if face_in and (int(a.shape[1]) % px or int(a.shape[2]) % py):
        raise ValueError(
            f"batch_layout: the face {tuple(a.shape[1:])} must tile the {px}x{py} "
            f"mesh exactly; pad the matrix extent before calling.")
    key = ("batch_layout", mesh_key(mesh), tuple(int(s) for s in a.shape),
           str(a.dtype), face_in)
    fn = _JIT_CACHE.get(key)
    if fn is None:
        if face_in:
            fn = jax.jit(shard_map(
                lambda t: _face_to_batch(_pad_leading(t, pad), px=px, py=py),
                mesh=mesh, in_specs=(P(None, "x", "y"),),
                out_specs=_batch_spec(3), check_vma=False))
        else:
            widths = ((0, pad),) + ((0, 0),) * (a.ndim - 1)
            fn = jax.jit(lambda t: jnp.pad(t, widths),
                         out_shardings=NamedSharding(mesh, _batch_spec(a.ndim)))
        _JIT_CACHE[key] = fn
    return fn(a)


def local_batch(kernel, mesh, *, resident=(), out_layout="face", nbatch=None):
    """Run a composition of dense equations q-locally with one exchange each way.

    Inputs/outputs are face-sharded matrix batches (outputs may be a pytree).
    A singleton input batch is broadcast; padded q rows never enter the kernel.
    Reuses the same movement and real-row schedule as the individual plans.
    ``out_layout="batch"`` returns the outputs in the batch layout instead
    (``Bp`` rows, the padded rows exact zeros), for a consumer that works on
    whole matrices per rank: no exchange back to the face.

    ``resident`` names operand positions that are already in the batch layout
    (:func:`batch_layout`, padded to ``Bp`` rows): those are not exchanged.
    A factor laid out once therefore serves every later right-hand side while
    only the right-hand side moves (face -> batch -> face), i.e. per rank
    ``2*ceil(B/P)`` RHS blocks cross the network per call instead of whole
    matrices. A concrete operand declared resident that is not in that
    layout refuses before tracing, because the implicit reshard it would
    otherwise get is exactly the per-call matrix movement this exists to
    remove.  When every operand is resident, ``nbatch`` names the real rows
    (the batch the face output carries); only the output moves.
    """
    px, py = int(mesh.shape['x']), int(mesh.shape['y'])
    face = P(None, 'x', 'y')
    resident = frozenset(int(i) for i in resident)
    if out_layout not in ("face", "batch"):
        raise ValueError(f"local_batch: out_layout {out_layout!r}; expected 'face' or 'batch'")
    on_face = out_layout == "face"

    @jax.jit
    def _run(*operands):
        moving = [a for i, a in enumerate(operands) if i not in resident]
        if not moving and nbatch is None:
            raise ValueError('local_batch needs a face operand or nbatch to set the batch')
        nb = int(nbatch) if not moving else max(a.shape[0] for a in moving)
        if any(a.ndim != 3 or a.shape[0] not in (1, nb)
               or a.shape[1] % px or a.shape[2] % py for a in moving):
            raise ValueError('local_batch requires matching matrix batches tiling the mesh')
        pad = (-nb) % (px * py)
        for i in resident:
            if int(operands[i].shape[0]) != nb + pad:
                raise ValueError(
                    f'local_batch: resident operand {i} has {operands[i].shape[0]} rows; '
                    f'the batch layout of a {nb}-row batch carries {nb + pad}')
        inputs = tuple(
            a if i in resident
            else _pad_leading(jnp.broadcast_to(a, (nb, *a.shape[1:])), pad)
            for i, a in enumerate(operands))
        in_specs = tuple(_batch_spec(a.ndim) if i in resident else face
                         for i, a in enumerate(inputs))

        @partial(shard_map, mesh=mesh, in_specs=in_specs,
                   out_specs=face if on_face else _batch_spec(3), check_vma=False)
        def work(*tiles):
            local = tuple(t if i in resident else _face_to_batch(t, px=px, py=py)
                          for i, t in enumerate(tiles))
            result = _real_rows(kernel, local, nbatch=nb, py=py)
            if not on_face:
                return result
            return jax.tree.map(lambda a: _batch_to_face(a, px=px, py=py), result)

        return jax.tree.map(lambda a: a[:nb], work(*inputs)) if on_face else work(*inputs)

    def run(*operands):
        for i in resident:
            # A traced operand (a caller's own program) gets the batch spec at
            # the shard_map boundary; a concrete one must already carry it.
            if not isinstance(operands[i], jax.core.Tracer) and not is_batch_layout(
                    operands[i], mesh):
                raise ValueError(
                    f'local_batch: operand {i} was declared resident but is not in the '
                    f'batch layout P((x,y),None,...) on this mesh; place it once with '
                    f'distrib_la.batch_layout')
        return _run(*operands)
    run.lower = _run.lower          # HLO/collective census of the same executable
    return run


def _replicate_batch_vector(v, *, px: int, py: int):
    """Restore the service's replicated eigenvalue-vector contract."""
    if px * py == 1:
        return v
    return jax.lax.all_gather(
        v, ("x", "y"), axis=0, tiled=True)


def batch_layout_eigh_call(op: str, mesh: Mesh, A, *, real_rows: int | None = None):
    """Route (c) without movement, for an operand already in batch layout.

    ``A`` is ``(B, N, N)`` at ``P(('x','y'), None, None)`` with ``B`` a
    multiple of ``Px*Py``: rank ``x*Py + y`` owns rows
    ``[rank*B/P, (rank+1)*B/P)`` as whole matrices. Each rank solves its
    whole local stack in one batched call (:func:`_local_stack_eigh`; the
    stack, its vectors and the batched kernel workspace live at once); rows
    whose global index is ``>= real_rows`` enter as zeros and their outputs
    are exact zeros. Eigenvalues return replicated
    through one ``all_gather``; vectors stay in batch layout. ``op`` is
    ``eigh``, ``checked_eigh`` or ``normal_eigh`` (the right singular
    vectors of ``A`` from the N x N eigh of ``A.H @ A``; the spectrum is the
    N singular values, ascending), the same equations as
    :func:`batch_reshard_call`, and both routes run the same local solve.
    """
    if op not in ("eigh", "checked_eigh", "normal_eigh"):
        raise ValueError(
            f"batch layout: unsupported op {op!r}; expected eigh|checked_eigh|normal_eigh")
    axes = tuple(mesh.axis_names)
    if "x" not in axes or "y" not in axes:
        raise ValueError(f"batch layout: expected a mesh with ('x','y'), got {axes!r}")
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    if A.ndim != 3 or int(A.shape[1]) != int(A.shape[2]):
        raise ValueError(f"batch layout {op}: expected A of shape (B,N,N), got {tuple(A.shape)}")
    nb, n = int(A.shape[0]), int(A.shape[1])
    if nb < 1 or nb % (px * py):
        raise ValueError(
            f"batch layout {op}: the leading batch B={nb} must be a positive multiple of "
            f"Px*Py={px * py}; synthetic slots are the caller's (pass real_rows)")
    nreal = nb if real_rows is None else int(real_rows)
    if not 1 <= nreal <= nb:
        raise ValueError(f"batch layout {op}: real_rows={real_rows} outside [1, {nb}]")
    key = ("batch_layout", op, mesh_key(mesh), nb, n, str(A.dtype), nreal)
    fn = _JIT_CACHE.get(key)
    if fn is None:
        spec = P(("x", "y"), None, None)

        def _body(local):
            W, Z = _local_stack_eigh(op, local, nbatch=nreal, py=py)
            return _replicate_batch_vector(W, px=px, py=py), Z

        fn = jax.jit(shard_map(_body, mesh=mesh, in_specs=(spec,),
                               out_specs=(P(), spec), check_vma=False))
        _JIT_CACHE[key] = fn
    return fn(A)


def batch_reshard_call(
    op: str,
    mesh: Mesh,
    ops: Sequence,
    *, rcond=None,
):
    """Run route (c): face→batch, local dense op, inverse exchange.

    The returned arrays obey the ordinary :meth:`Plan.batched` layout:
    matrix outputs at ``P(None,'x','y')`` and eigh eigenvalues replicated.
    The service-internal ``normal_eigh`` operation transports the original
    square response and returns its right singular vectors and singular
    values from the local eigh of ``A.H @ A`` (the polar owner's
    ``_normal_svd``); the length-n spectrum keeps the ordinary vector gather.
    """
    ops = tuple(ops)
    validate_batch_reshard_operands(op, mesh, ops)
    return reshard_program(op, mesh, tuple(
        (tuple(int(s) for s in x.shape), str(x.dtype)) for x in ops), rcond)(*ops)


def reshard_program(op: str, mesh: Mesh, signature, rcond=None):
    """The jitted route-(c) program of :func:`batch_reshard_call` for operand
    ``signature`` (``((shape, dtype), ...)``), built once. Its compiled size
    is what :meth:`distrib_la.plan.Plan.stack_route` admits."""
    nbatch = int(signature[0][0][0])
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    batch_pad = -(-nbatch // (px * py)) * px * py - nbatch
    key = (op, mesh_key(mesh), tuple(signature), rcond)
    fn = _JIT_CACHE.get(key)
    if fn is None:
        in_specs = tuple(P(None, "x", "y") for _ in signature)
        out_specs = ((P(), P(None, "x", "y")) if op in ("eigh", "checked_eigh", "normal_eigh", "polar")
                     else P(None, "x", "y"))

        def _body(*local_faces):
            local = tuple(
                _face_to_batch(_pad_leading(a, batch_pad), px=px, py=py)
                for a in local_faces)
            A = local[0]

            if op == "polar":
                from distrib_la.polar import _polar_from_matrix
                def kernel(a):
                    link, s = _polar_from_matrix(a, jnp.linalg.eigh, rcond)
                    return s, link
                W, Z = (_real_rows(kernel, (A,), nbatch=nbatch, py=py)
                        if batch_pad else kernel(A))
                return (_replicate_batch_vector(W, px=px, py=py)[:nbatch],
                        _batch_to_face(Z, px=px, py=py)[:nbatch])

            if op == "normal_eigh":
                W, Z = _local_stack_eigh(op, A, nbatch=nbatch, py=py)
                W = _replicate_batch_vector(W, px=px, py=py)[:nbatch]
                return W, _batch_to_face(Z, px=px, py=py)[:nbatch]
            if op in ("eigh", "checked_eigh"):
                W, Z = _local_stack_eigh(op, A, nbatch=nbatch, py=py)
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
    return fn


def reshard_rounds_call(op: str, mesh: Mesh, A, *, rounds: int):
    """Route (c) on a face stack in ``rounds`` slices of ``m = ceil(B/rounds)`` matrices.

    One ``lax.scan`` over the slice starts: each slice runs
    :func:`batch_reshard_call` (exchange, local eigh, inverse exchange) and
    is written in place into the face outputs, so a rank holds one slice's
    whole matrices at a time beside the outputs. The last slice starts at
    ``B - m``; the matrices it shares with the previous slice are solved
    again and written with the same values. ``op`` is ``eigh``,
    ``checked_eigh`` or ``normal_eigh``; outputs follow
    :func:`batch_reshard_call`.
    """
    rounds = int(rounds)
    if rounds == 1:
        return batch_reshard_call(op, mesh, (A,))
    nb, n = int(A.shape[0]), int(A.shape[-1])
    m = -(-nb // rounds)
    validate_batch_reshard_operands(op, mesh, (jax.ShapeDtypeStruct((m, n, n), A.dtype),))
    key = ("rounds", op, mesh_key(mesh), nb, n, str(A.dtype), rounds)
    fn = _JIT_CACHE.get(key)
    if fn is None:
        program = reshard_program(op, mesh, (((m, n, n), str(A.dtype)),))
        face, replicated = NamedSharding(mesh, P(None, "x", "y")), NamedSharding(mesh, P())
        starts = jnp.asarray([min(r * m, nb - m) for r in range(rounds)], jnp.int32)
        real = jnp.zeros((), A.dtype).real.dtype

        @partial(jax.jit, out_shardings=(replicated, face))
        def fn(a):
            def body(carry, start):
                w_all, z_all = carry
                x = jax.lax.with_sharding_constraint(
                    jax.lax.dynamic_slice_in_dim(a, start, m, axis=0), face)
                w, z = program(x)
                return (jax.lax.dynamic_update_slice_in_dim(w_all, w.astype(w_all.dtype), start, axis=0),
                        jax.lax.with_sharding_constraint(
                            jax.lax.dynamic_update_slice_in_dim(z_all, z, start, axis=0), face)), None
            init = (jnp.zeros((nb, n), real),
                    jax.lax.with_sharding_constraint(jnp.zeros_like(a), face))
            (w, z), _ = jax.lax.scan(body, init, starts)
            return w, z
        _JIT_CACHE[key] = fn
    return fn(A)
