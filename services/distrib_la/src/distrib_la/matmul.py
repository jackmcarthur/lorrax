"""Top-level distributed GEMM.

The explicit ``batched_route='auto'`` route runs the face GEMM,
:func:`distrib_la.panel_matmul`'s batched SUMMA, on every platform
for every backend name except ``'off'`` (the retired provider names
``'cublasmp'``, ``'cusolvermp'``, ``'scalapack'`` and ``'slate'`` included:
no native batched GEMM is called).

The default ``batched_route='batch_reshard'`` route moves A, B and C from
faces to whole matrices in one all_to_all over (x, y) each, runs local
``jnp.matmul``, then applies the literal inverse exchange to D.  Use ``backend='off'``
with that route for a provider-free call; any other name still needs the
square mesh the face GEMM needs.
"""
from __future__ import annotations

from functools import lru_cache, partial
from typing import Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax import shard_map
from distrib_la.plan import (BATCHED_ROUTE_CHOICES, BATCHED_ROUTE_DEFAULT,
                             ROUTE_BATCH_RESHARD, ensure_sharding)
from distrib_la.resolve import mesh_key

__all__ = ["MATMUL_BACKEND_CHOICES", "matmul", "resolve_matmul_backend",
           "contract_faces"]

MATMUL_BACKEND_CHOICES = (
    "auto", "off", "distributed", "cusolvermp", "cublasmp",
    "scalapack", "slate",
)
"""Public provider vocabulary accepted by :func:`matmul` and its resolver."""

_OP_CODE = {"N": 0, "T": 1, "C": 2}
_RESHARD_CACHE: dict = {}
#: The face GEMM every platform resolves to: :func:`distrib_la.panel_matmul`.
PANEL = "panel_matmul"


def contract_faces(b_X, b_Y, weights, start, stop, *, mesh: Mesh,
                   return_transpose: bool = False):
    """Contract two row faces with replicated column weights, locally.

    Parameters
    ----------
    b_X, b_Y
        Matching arrays [b,m,Kcap] at P(None,'x',None) and
        P(None,'y',None), or [b,mu,spin,Kcap] at
        P(None,'x',None,None) and P(None,'y',None,None). The latter merge
        mu*spin inside this service. No column or batch sharding is allowed.
    weights
        Complex [b,Kcap], replicated. Units are supplied by the caller;
        the service adds no normalization or conjugation to these weights.
    start, stop
        Replicated integer [b] column bounds selecting [start,stop).
        The caller resolves physical intervals and masks inactive columns.
    mesh
        Supplied mesh with axes ('x','y'). Both global linalg policies use
        this same local operation; no provider is resolved or called.
    return_transpose
        Also return (conj(b_X)*weights) @ b_Y.T, at the same weights.

    Returns
    -------
    W : jax.Array or tuple of jax.Array
        [b,m,m] at P(None,'x','y'), (b_X*weights) @ b_Y.H, optionally
        paired with its endpoint-transpose orientation. Each shard_map
        body contains local GEMMs only, with no collective or provider call.
    """
    _mesh_shape(mesh)
    if b_X.ndim not in (3, 4) or b_Y.shape != b_X.shape:
        raise ValueError("contract_faces requires matching rank-3/4 faces")
    if b_X.dtype != b_Y.dtype or b_X.dtype != weights.dtype:
        raise TypeError("contract_faces factors and weights must share dtype")
    b, k = b_X.shape[0], b_X.shape[-1]
    if weights.shape != (b, k) or start.shape != (b,) or stop.shape != (b,):
        raise ValueError("contract_faces weights [b,K] and bounds [b] required")
    if start.dtype.kind not in "iu" or stop.dtype.kind not in "iu":
        raise TypeError("contract_faces interval bounds must be integers")
    explicit_spin = b_X.ndim == 4
    xs = P(None, 'x', None, None) if explicit_spin else P(None, 'x', None)
    ys = P(None, 'y', None, None) if explicit_spin else P(None, 'y', None)
    # Concrete wrong layouts must not hide an input reshard in the hot path.
    for value, spec in ((b_X, xs), (b_Y, ys), (weights, P()),
                        (start, P()), (stop, P())):
        if not isinstance(value, jax.core.Tracer):
            want = NamedSharding(mesh, spec)
            if not value.sharding.is_equivalent_to(want, value.ndim):
                raise ValueError(f"contract_faces requires input layout {spec}")
    return _contract_faces_kernel(mesh, explicit_spin, return_transpose)(
        b_X, b_Y, weights, start, stop)


@lru_cache(maxsize=None)
def _contract_faces_kernel(mesh, explicit_spin, return_transpose):
    """Reuse the local weighted b_X b_Y† contraction for a static layout."""
    xs = P(None, 'x', None, None) if explicit_spin else P(None, 'x', None)
    ys = P(None, 'y', None, None) if explicit_spin else P(None, 'y', None)
    out = P(None, 'x', 'y')

    @partial(shard_map, mesh=mesh, in_specs=(xs, ys, P(), P(), P()),
             out_specs=(out, out) if return_transpose else out,
             check_vma=False)
    def _local(x, y, d, lo, hi):
        b, k = x.shape[0], x.shape[-1]
        if explicit_spin:
            x = x.reshape((b, x.shape[1] * x.shape[2], k))
            y = y.reshape((b, y.shape[1] * y.shape[2], k))
        columns = jnp.arange(k)[None, :]
        d = jnp.where((columns >= lo[:, None]) & (columns < hi[:, None]),
                      d, 0)
        w = (x * d[:, None, :]) @ jnp.swapaxes(jnp.conj(y), -1, -2)
        if return_transpose:
            wt = (jnp.conj(x) * d[:, None, :]) @ jnp.swapaxes(y, -1, -2)
            return w, wt
        return w

    return jax.jit(_local)


def _mesh_shape(mesh: Mesh) -> tuple[int, int]:
    if tuple(mesh.axis_names) != ("x", "y") or len(mesh.devices.shape) != 2:
        raise ValueError(
            "matmul mesh must be exactly 2-D with y-minor axes "
            f"('x','y'); got axes={mesh.axis_names}, "
            f"shape={mesh.devices.shape}")
    return int(mesh.shape["x"]), int(mesh.shape["y"])


def resolve_matmul_backend(requested: str, mesh: Mesh, *,
                           batched_route: str = BATCHED_ROUTE_DEFAULT) -> str:
    """Resolve a public request to an actual GEMM provider.

    Every name except ``off`` selects the face GEMM,
    :func:`distrib_la.panel_matmul` (:data:`PANEL`), on every platform; it
    needs a square mesh.  ``off`` is legal only with the local
    ``batch_reshard`` route, where no provider call is made.
    """
    requested = str(requested).strip().lower()
    route = str(batched_route).strip().lower()
    if requested not in MATMUL_BACKEND_CHOICES:
        raise ValueError(
            f"matmul backend must be one of "
            f"{'|'.join(MATMUL_BACKEND_CHOICES)}, got {requested!r}")
    if route not in BATCHED_ROUTE_CHOICES:
        raise ValueError(
            f"matmul batched_route must be one of "
            f"{'|'.join(BATCHED_ROUTE_CHOICES)}, got {route!r}")
    if requested == "off":
        if route != ROUTE_BATCH_RESHARD:
            raise RuntimeError(
                "matmul backend 'off' has no distributed provider; select "
                "batched_route='batch_reshard' for all-to-all/local GEMM")
        return "off"
    px, py = _mesh_shape(mesh)
    if px != py:
        raise ValueError(
            f"matmul: the face GEMM (distrib_la.panel_matmul) needs a square mesh; "
            f"got {px}x{py}")
    return PANEL


def _op_shape(shape: tuple[int, int], op: str) -> tuple[int, int]:
    return shape if op == "N" else (shape[1], shape[0])


def _validate_operands(A, B, C, transa: str, transb: str):
    if A.ndim not in (2, 3) or B.ndim != A.ndim:
        raise ValueError(
            "distrib_la.matmul expects A and B to be both rank 2 or both "
            f"rank 3; got A={A.shape}, B={B.shape}")
    if C is not None and C.ndim != A.ndim:
        raise ValueError(f"C rank {C.ndim} does not match A/B rank {A.ndim}")
    if A.dtype != B.dtype or (C is not None and A.dtype != C.dtype):
        raise ValueError(
            f"matmul dtypes disagree: A={A.dtype}, B={B.dtype}, "
            f"C={None if C is None else C.dtype}")
    if A.ndim == 3 and int(A.shape[0]) != int(B.shape[0]):
        raise ValueError(
            f"matmul batch dims disagree: A={A.shape[0]}, B={B.shape[0]}")
    if A.ndim == 3 and int(A.shape[0]) < 1:
        raise ValueError("matmul batch must be nonempty")
    a_shape = _op_shape((int(A.shape[-2]), int(A.shape[-1])), transa)
    b_shape = _op_shape((int(B.shape[-2]), int(B.shape[-1])), transb)
    if a_shape[1] != b_shape[0]:
        raise ValueError(
            f"matmul contraction mismatch: op(A)={a_shape}, op(B)={b_shape}")
    out = ((int(A.shape[0]), a_shape[0], b_shape[1])
           if A.ndim == 3 else (a_shape[0], b_shape[1]))
    if C is not None and tuple(C.shape) != out:
        raise ValueError(f"C shape {tuple(C.shape)} != output shape {out}")
    return out


@partial(jax.jit, static_argnums=(0, 1, 2))
def _zeros(shape, dtype, sharding):
    """Allocate a sharded zero tile with one executable per shape and dtype."""
    return jax.lax.with_sharding_constraint(
        jnp.zeros(shape, dtype=dtype), sharding)


def _panel(mesh, A, B, C, *, alpha, beta, transa, transb):
    """The face GEMM: :func:`distrib_la.panel_matmul`, ``alpha·op(A)·op(B) + beta·C``.
    Transposed operands ride the square-mesh panel route (one ``ppermute`` per
    operand); the panel budget is one output tile per rank."""
    from distrib_la._panel_matmul import panel_matmul
    px, py = _mesh_shape(mesh)
    nq, m, n = (int(v) for v in C.shape)
    tile = int(A.dtype.itemsize) * nq * (m // px) * (n // py)
    d = panel_matmul(A, B, mesh=mesh, panel_bytes=tile, transa=transa, transb=transb)
    if alpha != 1:
        d = jnp.asarray(alpha, A.dtype) * d
    return d if beta == 0 else d + jnp.asarray(beta, A.dtype) * C


def _batch_reshard(mesh, A, B, C, *, alpha, beta, transa, transb):
    from distrib_la._batch_reshard import (_batch_to_face, _face_to_batch,
                                           _real_rows)

    px, py = _mesh_shape(mesh)
    ptotal = px * py
    nb = int(A.shape[0])
    nb_pad = ((nb + ptotal - 1) // ptotal) * ptotal
    pad = nb_pad - nb
    has_c = C is not None
    key = (mesh_key(mesh), tuple(A.shape), tuple(B.shape),
           None if C is None else tuple(C.shape), str(A.dtype), alpha, beta,
           transa, transb)
    fn = _RESHARD_CACHE.get(key)
    if fn is None:
        def _gemm(a, b, c):
            if transa != "N":
                a = jnp.swapaxes(a, -1, -2)
                if transa == "C":
                    a = jnp.conj(a)
            if transb != "N":
                b = jnp.swapaxes(b, -1, -2)
                if transb == "C":
                    b = jnp.conj(b)
            d = jnp.asarray(alpha, A.dtype) * jnp.matmul(a, b)
            if c is not None:
                d = d + jnp.asarray(beta, A.dtype) * c
            return d

        def _body(a, b, c):
            if pad:
                a = jnp.pad(a, ((0, pad), (0, 0), (0, 0)))
                b = jnp.pad(b, ((0, pad), (0, 0), (0, 0)))
                if c is not None:
                    c = jnp.pad(c, ((0, pad), (0, 0), (0, 0)))
            a = _face_to_batch(a, px=px, py=py)
            b = _face_to_batch(b, px=px, py=py)
            if c is not None:
                c = _face_to_batch(c, px=px, py=py)
            # Synthetic rows of a ragged batch are exchanged but never multiplied:
            # the same scalar cond-in-loop skip as the eigh/cholesky/solve route.
            d = (_real_rows(_gemm, (a, b, c), nbatch=nb, py=py) if pad
                 else _gemm(a, b, c))
            d = _batch_to_face(d, px=px, py=py)
            return d[:nb]

        if has_c:
            @partial(shard_map, mesh=mesh,
                     in_specs=(P(None, "x", "y"),) * 3,
                     out_specs=P(None, "x", "y"), check_vma=False)
            def _local(a, b, c):
                return _body(a, b, c)
        else:
            @partial(shard_map, mesh=mesh,
                     in_specs=(P(None, "x", "y"),) * 2,
                     out_specs=P(None, "x", "y"), check_vma=False)
            def _local(a, b):
                return _body(a, b, None)

        # Exchanges prevent reliable face-input/output aliasing.  Requesting
        # donation here emits the same unusable-donation warning avoided by
        # the other batch_reshard operations.
        fn = jax.jit(_local)
        _RESHARD_CACHE[key] = fn
    return fn(A, B, C) if has_c else fn(A, B)


def matmul(
    A: jax.Array,
    B: jax.Array,
    C: jax.Array | None = None,
    *,
    mesh: Mesh,
    alpha: Union[float, complex] = 1.0,
    beta: Union[float, complex] = 0.0,
    transa: str = "N",
    transb: str = "N",
    backend: str = "auto",
    batched_route: str = BATCHED_ROUTE_DEFAULT,
    budget_bytes: int | None = None,
) -> jax.Array:
    """Compute ``alpha * op(A) @ op(B) + beta * C`` over ``mesh``.

    Parameters
    ----------
    A, B
        Both rank 2 or both rank 3, with matching dtype.  Rank-2 matrices use
        ``P('x','y')``; rank-3 stacks use ``P(None,'x','y')`` and must have
        the same leading batch. ``op(A)`` and ``op(B)`` must contract.
    C
        Optional output-shaped addend in the same rank and face layout.  It
        is replaced by zero only when ``beta == 0``; otherwise it is required.
        Provider routes may donate/consume this buffer as GEMM's output;
        callers should not reuse it after the call. The staged route does not
        request donation and skips C entirely when ``beta == 0``.
    mesh
        Exact 2-D JAX mesh with y-minor axes ``('x','y')``. Provider routes
        additionally require cell ``(ix,iy)`` to own process ``ix*Py+iy``.
    alpha, beta
        GEMM scalars.
    transa, transb
        ``'N'`` (unchanged), ``'T'`` (transpose), or ``'C'`` (conjugate
        transpose).
    backend
        A name in :data:`MATMUL_BACKEND_CHOICES`. Every name but ``'off'``
        runs the face GEMM (:func:`distrib_la.panel_matmul`) on every platform.
        ``'off'`` is provider-free and requires the staged route.
    batched_route
        ``'batch_reshard'`` (the default) pads a ragged leading batch with
        zero matrices, exchanges each face into whole per-device matrices in
        one all_to_all over (x, y), runs local JAX GEMM, and returns D
        through the inverse exchange. Explicit ``'auto'`` calls the resolved distributed
        provider.
    budget_bytes
        Optional per-rank device budget. With ``batched_route='auto'`` the
        route becomes a capacity decision: when one rank's whole local A, B
        and D matrices (``ceil(batch/P)`` of each) plus the local GEMM
        temporary fit (:func:`distrib_la.workspace.fits_local`), the staged
        route runs; otherwise the provider. Pass a value every rank shares.

    Returns
    -------
    jax.Array
        Rank 2 at ``P('x','y')`` or rank 3 at ``P(None,'x','y')``, matching
        the input rank and the shape of ``op(A) @ op(B)``.

    Notes
    -----
    The face GEMM requires a square mesh, one JAX process per mesh cell in
    y-minor order and exact face tiling; a transposed operand moves its tile
    to the transposed grid position by one ``ppermute``.

    The staged route does not require a provider or square mesh when selected
    with ``backend='off'``, but every physical input face and the output face
    must tile the mesh.  Each device holds complete local A, B, C, and D
    matrices for ``ceil(batch/(Px*Py))`` batch elements plus original face,
    exchange-buffer, and native-GEMM storage; it is therefore only a
    below-single-device-capacity route. Each local batch element has complete
    A, B, and D matrices (plus C only when ``beta != 0``); C is not allocated
    or exchanged when ``beta=0``.
    """
    transa, transb = str(transa).upper(), str(transb).upper()
    if transa not in _OP_CODE or transb not in _OP_CODE:
        raise ValueError(
            f"transa/transb must be N/T/C; got {transa!r}/{transb!r}")
    beta_c = complex(beta)
    if C is None and beta_c != 0:
        raise ValueError("C is required when beta is nonzero")
    out_shape = _validate_operands(A, B, C, transa, transb)
    route = str(batched_route).strip().lower()
    if route == "auto" and budget_bytes is not None:
        from distrib_la.workspace import fits_local
        from types import SimpleNamespace
        ranks = int(mesh.shape["x"]) * int(mesh.shape["y"])
        nb = int(A.shape[0]) if A.ndim == 3 else 1
        local = -(-nb // ranks)
        a = (local,) + _op_shape((int(A.shape[-2]), int(A.shape[-1])), transa)
        b = (local,) + _op_shape((int(B.shape[-2]), int(B.shape[-1])), transb)
        d = (local, a[1], b[2])
        if fits_local(SimpleNamespace(mesh=mesh), "gemm", (a, b, d), A.dtype, budget_bytes):
            route = ROUTE_BATCH_RESHARD
    provider = resolve_matmul_backend(backend, mesh, batched_route=route)
    px, py = _mesh_shape(mesh)
    m_out, n_out = int(out_shape[-2]), int(out_shape[-1])
    if m_out % px or n_out % py:
        raise ValueError(
            f"matmul output face must tile {px}x{py}; got ({m_out}, "
            f"{n_out}) with remainders ({m_out % px},{n_out % py})")
    single = A.ndim == 2
    if single:
        A, B = A[None], B[None]
        C = None if C is None else C[None]
        out_shape = (1,) + tuple(out_shape)
    for name, x in (("A", A), ("B", B), ("C", C)):
        if x is None:
            continue
        if int(x.shape[-2]) % px or int(x.shape[-1]) % py:
            raise ValueError(
                f"matmul {name} face must tile {px}x{py}; got {x.shape[-2:]}")
    if route == ROUTE_BATCH_RESHARD or provider == PANEL:
        # CPU/MPI may only create first-use collective communicators from
        # MPI's main thread, never from the XLA worker that runs shard_map.
        from distrib_la._collectives import warm_mesh_cliques
        warm_mesh_cliques(mesh)
    sharding = NamedSharding(mesh, P(None, "x", "y"))
    A, B = ensure_sharding(A, sharding), ensure_sharding(B, sharding)
    C = None if C is None else ensure_sharding(C, sharding)
    alpha_c = complex(alpha)
    if A.dtype.kind != "c" and (alpha_c.imag or beta_c.imag):
        raise ValueError(
            "matmul alpha/beta must be real for a real operand dtype")
    alpha_arg = alpha_c if A.dtype.kind == "c" else alpha_c.real
    beta_arg = beta_c if A.dtype.kind == "c" else beta_c.real
    if route == ROUTE_BATCH_RESHARD:
        out = _batch_reshard(
            mesh, A, B, C if beta_c != 0 else None,
            alpha=alpha_arg, beta=beta_arg,
            transa=transa, transb=transb)
    else:
        if C is None:
            C = _zeros(out_shape, A.dtype, sharding)
        out = _panel(mesh, A, B, C, alpha=alpha_arg, beta=beta_arg,
                     transa=transa, transb=transb)
    return out[0] if single else out
