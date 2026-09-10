"""Top-level distributed GEMM across the :mod:`distrib_la` providers.

The explicit ``batched_route='auto'`` route calls an actual 2-D provider
operation: cuBLASMp next to
cuSOLVERMp, PBLAS ``pdgemm``/``pzgemm`` next to ScaLAPACK, or
``slate::multiply`` next to SLATE.  Unlike :func:`distrib_la.plan`,
``backend='auto'`` here selects that platform provider; it does not select a
native JAX floor.

The default ``batched_route='batch_reshard'`` route performs x-then-y staged
face-to-batch exchanges for A, B, and C, runs local ``jnp.matmul``, then
applies the literal y-then-x inverse exchanges to D.  Use ``backend='off'``
with that route for a provider-free call.  A non-``off`` request is still
resolved and capability-probed even though the selected route does not call
the provider.
"""
from __future__ import annotations

from functools import lru_cache, partial
from typing import Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la import loader
from distrib_la._shard_map import shard_map
from distrib_la.plan import (BATCHED_ROUTE_CHOICES, BATCHED_ROUTE_DEFAULT,
                             ROUTE_BATCH_RESHARD, ensure_sharding)
from distrib_la.resolve import mesh_key, mesh_platform

__all__ = ["MATMUL_BACKEND_CHOICES", "matmul", "resolve_matmul_backend",
           "contract_faces", "matmul_adjoint_pair"]

MATMUL_BACKEND_CHOICES = (
    "auto", "off", "distributed", "cusolvermp", "cublasmp",
    "scalapack", "slate",
)
"""Public provider vocabulary accepted by :func:`matmul` and its resolver."""

_TARGETS = {
    "cublasmp": "lorrax_cublasmp_batched_gemm",
    "scalapack": "lorrax_scalapack_batched_gemm",
    "slate": "lorrax_slate_batched_gemm",
}
_OP_CODE = {"N": 0, "T": 1, "C": 2}
_CUBLASMP_CACHE: dict = {}
_RESHARD_CACHE: dict = {}


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


def _provider_platform(provider: str, mesh: Mesh) -> str:
    platform = mesh_platform(mesh)
    if provider == "cublasmp" and platform != "CUDA":
        raise RuntimeError(
            f"matmul backend 'cublasmp' is CUDA-only; mesh is {platform!r}")
    if provider == "scalapack" and platform != "cpu":
        raise RuntimeError(
            f"matmul backend 'scalapack' is host-only; mesh is {platform!r}")
    if provider == "slate" and platform not in ("CUDA", "cpu", "rocm"):
        raise RuntimeError(
            f"matmul backend 'slate' has no provider on {platform!r}")
    return platform


def _require_provider(provider: str, mesh: Mesh) -> None:
    px, py = _mesh_shape(mesh)
    if provider in ("cublasmp", "slate") and px != py:
        raise ValueError(
            f"matmul backend {provider!r} needs a square mesh: its "
            f"one-face GEMM layout is invalid on a {px}x{py} grid")
    if px * py != int(jax.process_count()):
        raise RuntimeError(
            f"matmul backend {provider!r} needs one JAX process per mesh "
            f"cell; mesh={px}x{py}, process_count={jax.process_count()}")
    bad = []
    for ix in range(px):
        for iy in range(py):
            got = int(mesh.devices[ix, iy].process_index)
            want = ix * py + iy
            if got != want:
                bad.append(f"({ix},{iy})->process {got}, expected {want}")
    if bad:
        raise RuntimeError(
            f"matmul backend {provider!r} requires a y-minor process grid; "
            + "; ".join(bad[:4]))
    platform = _provider_platform(provider, mesh)
    probe = loader.probe_target(_TARGETS[provider], platform)
    if not probe:
        raise RuntimeError(
            f"matmul backend {provider!r} is unavailable: {probe.reason}")


def resolve_matmul_backend(requested: str, mesh: Mesh, *,
                           batched_route: str = BATCHED_ROUTE_DEFAULT) -> str:
    """Resolve a public request to an actual GEMM provider.

    ``cusolvermp`` maps to its matrix-multiply sibling ``cublasmp``.
    ``auto`` and ``distributed`` select cuBLASMp on CUDA and ScaLAPACK on
    CPU, and SLATE on ROCm.  Explicit requests never demote.  ``off`` is
    legal only with the local ``batch_reshard`` route, where no provider call
    is made. Every other result has already passed platform,
    provider-specific mesh geometry (including cuBLASMp/SLATE square grids),
    one-process-per-cell, shared-library, and handler-symbol guards.

    Route selection is orthogonal to provider selection: an explicit or
    automatic non-``off`` provider is still probed when
    ``batched_route='batch_reshard'``.  This matches :class:`distrib_la.Plan`
    semantics and makes a requested capability a promise rather than a hint.
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
    if requested in ("auto", "distributed"):
        platform = mesh_platform(mesh)
        if platform == "CUDA":
            provider = "cublasmp"
        elif platform == "cpu":
            provider = "scalapack"
        elif platform == "rocm":
            provider = "slate"
        else:
            raise RuntimeError(
                f"matmul has no distributed provider for platform {platform!r}")
    elif requested == "cusolvermp":
        provider = "cublasmp"
    else:
        provider = requested
    # Match Plan semantics: even the local route honours an explicit provider
    # request and proves its capability at construction/call entry.
    _require_provider(provider, mesh)
    return provider


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


@lru_cache(maxsize=None)
def _zeros_kernel(shape, dtype, sharding):
    """Retain the executable, never the allocated (possibly donated) buffer."""
    return jax.jit(
        lambda: jnp.zeros(shape, dtype=dtype), out_shardings=sharding)


def _zeros(shape, dtype, sharding):
    return _zeros_kernel(tuple(shape), jnp.dtype(dtype), sharding)()


@lru_cache(maxsize=None)
def _transpose_kernel(op, tile):
    """Reuse a distributed endpoint transpose for one operation and layout."""
    @jax.jit(out_shardings=tile)
    def move(x):
        t = jnp.swapaxes(x, -1, -2)
        return jnp.conj(t) if op == 'C' else t
    return move


def _cublasmp(mesh, A, B, C, *, alpha: complex, beta: complex,
              transa: str, transb: str):
    from distrib_la._cusolvermp import get_or_init_context

    px, py = _mesh_shape(mesh)
    if transa != "N" or transb != "N":
        # cuBLASMp's multi-rank native transpose descriptors have produced
        # wrong answers / deadlock. Move endpoint tiles on device, then use
        # the certified N,N provider. This is a distributed transpose, not
        # a local tile transpose and not a host/full-matrix gather.
        tile = NamedSharding(mesh, P(None, 'x', 'y'))
        if transa != 'N':
            A = _transpose_kernel(transa, tile)(A)
        if transb != 'N':
            B = _transpose_kernel(transb, tile)(B)
        transa, transb = 'N', 'N'
    if A.dtype not in (jnp.dtype("float64"), jnp.dtype("complex128")):
        raise ValueError(
            f"cuBLASMp matmul supports float64/complex128; got {A.dtype}")
    nq = int(A.shape[0])
    ar, ac = int(A.shape[1]), int(A.shape[2])
    br, bc = int(B.shape[1]), int(B.shape[2])
    m, k = _op_shape((ar, ac), transa)
    _, n = _op_shape((br, bc), transb)
    ctx = get_or_init_context(mesh, col_major=False)
    attrs = dict(
        nq=nq, m=m, n=n, k=k,
        mb_a=ar // px, nb_a=ac // py, mb_b=br // px, nb_b=bc // py,
        mb_c=m // px, nb_c=n // py,
        lld_a=ar // px, lld_b=br // px, lld_c=m // px,
        transa=_OP_CODE[transa], transb=_OP_CODE[transb],
        alpha_re=float(alpha.real), alpha_im=float(alpha.imag),
        beta_re=float(beta.real), beta_im=float(beta.imag),
        ctx_handle=int(ctx),
    )
    key = (mesh_key(mesh), tuple(A.shape), tuple(B.shape), tuple(C.shape),
           str(A.dtype), transa, transb, alpha, beta, int(ctx))
    fn = _CUBLASMP_CACHE.get(key)
    if fn is None:
        out_t = jax.ShapeDtypeStruct((nq, n // py, m // px), C.dtype)

        @partial(shard_map, mesh=mesh, in_specs=(P(None, "x", "y"),) * 3,
                 out_specs=P(None, "x", "y"), check_vma=False)
        def _local(a, b, c):
            at = jnp.transpose(a, (0, 2, 1))
            bt = jnp.transpose(b, (0, 2, 1))
            ct = jnp.transpose(c, (0, 2, 1))
            dt = jax.ffi.ffi_call(
                _TARGETS["cublasmp"], out_t,
                input_output_aliases={2: 0})(at, bt, ct, **attrs)
            return jnp.transpose(dt, (0, 2, 1))

        fn = jax.jit(_local, donate_argnums=(2,))
        _CUBLASMP_CACHE[key] = fn
    return fn(A, B, C)


def _provider_matmul(provider, mesh, A, B, C, *, alpha, beta,
                     transa, transb):
    if provider == "cublasmp":
        return _cublasmp(
            mesh, A, B, C, alpha=alpha, beta=beta,
            transa=transa, transb=transb)
    module = (__import__(f"distrib_la._{provider}", fromlist=["x"]))
    return module.batched_distributed_matmul(
        A, B, C, mesh=mesh, alpha=alpha, beta=beta,
        transa=transa, transb=transb)


def _local_product(a, b, c, *, alpha, beta, transa, transb):
    """Incumbent local GEMM expression, shared by single and paired actions."""
    if transa != "N":
        a = jnp.swapaxes(a, -1, -2)
        if transa == "C":
            a = jnp.conj(a)
    if transb != "N":
        b = jnp.swapaxes(b, -1, -2)
        if transb == "C":
            b = jnp.conj(b)
    d = jnp.asarray(alpha, a.dtype) * jnp.matmul(a, b)
    if c is not None:
        d = d + jnp.asarray(beta, a.dtype) * c
    return d


def _batch_reshard(mesh, A, B, C, *, alpha, beta, transa, transb,
                   adjoint_pair=False):
    from distrib_la._batch_reshard import _batch_to_face, _face_to_batch

    px, py = _mesh_shape(mesh)
    ptotal = px * py
    nb = int(A.shape[0])
    nb_pad = ((nb + ptotal - 1) // ptotal) * ptotal
    pad = nb_pad - nb
    has_c = C is not None
    key = (mesh_key(mesh), tuple(A.shape), tuple(B.shape),
           None if C is None else tuple(C.shape), str(A.dtype), alpha, beta,
           transa, transb)
    if adjoint_pair:
        key += ("adjoint_pair",)
    fn = _RESHARD_CACHE.get(key)
    if fn is None:
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
            d = _local_product(a, b, c, alpha=alpha, beta=beta,
                               transa=transa, transb=transb)
            if adjoint_pair:
                adjoint = _local_product(a, b, None, alpha=alpha, beta=beta,
                                         transa="C", transb="N")
                return (_batch_to_face(d, px=px, py=py)[:nb],
                        _batch_to_face(adjoint, px=px, py=py)[:nb])
            return _batch_to_face(d, px=px, py=py)[:nb]

        if has_c:
            @partial(shard_map, mesh=mesh,
                     in_specs=(P(None, "x", "y"),) * 3,
                     out_specs=P(None, "x", "y"), check_vma=False)
            def _local(a, b, c):
                return _body(a, b, c)
        else:
            @partial(shard_map, mesh=mesh,
                     in_specs=(P(None, "x", "y"),) * 2,
                     out_specs=((P(None, "x", "y"),) * 2 if adjoint_pair
                                else P(None, "x", "y")), check_vma=False)
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
        A name in :data:`MATMUL_BACKEND_CHOICES`. ``'auto'`` and
        ``'distributed'`` choose cuBLASMp on CUDA, ScaLAPACK/PBLAS on CPU,
        and SLATE on ROCm. ``'cusolvermp'`` is an alias for its cuBLASMp
        sibling. ``'off'`` is provider-free and requires the staged route.
    batched_route
        ``'batch_reshard'`` (the default) pads a ragged leading batch with
        zero matrices, exchanges each face x then y into whole per-device
        matrices, runs local JAX GEMM, and returns D through the inverse y
        then x exchanges. Explicit ``'auto'`` calls the resolved distributed
        provider.

    Returns
    -------
    jax.Array
        Rank 2 at ``P('x','y')`` or rank 3 at ``P(None,'x','y')``, matching
        the input rank and the shape of ``op(A) @ op(B)``.

    Notes
    -----
    Provider routes require float64 or complex128, one JAX process per mesh
    cell in y-minor order, exact face tiling, and an available handler.
    cuBLASMp and SLATE additionally require a square mesh; multi-rank
    cuBLASMp implements transpose/adjoint modes by a device face transpose
    followed by its N,N provider call. Its native transpose descriptors are
    never used: transpose-A returned wrong answers and transpose-B could
    deadlock. These explicit endpoint moves carry collective communication
    and one additional operand-sized distributed buffer per moved operand.

    The staged route does not require a provider or square mesh when selected
    with ``backend='off'``, but every physical input face and the output face
    must tile the mesh.  Each device holds complete local A, B, C, and D
    matrices for ``ceil(batch/(Px*Py))`` batch elements plus original face,
    exchange-buffer, and native-GEMM storage; it is therefore only a
    below-single-device-capacity route. Each local batch element has complete
    A, B, and D matrices (plus C only when ``beta != 0``); C is not allocated
    or exchanged when ``beta=0``.
    """
    return _matmul(A, B, C, mesh=mesh, alpha=alpha, beta=beta,
                   transa=transa, transb=transb, backend=backend,
                   batched_route=batched_route)


def matmul_adjoint_pair(A, B, *, mesh: Mesh, backend="auto",
                        batched_route=BATCHED_ROUTE_DEFAULT):
    """Return ``(A @ B, A.conj().T @ B)`` with one staged input movement.

    A and B are [batch,n,n] and [batch,n,r] faces at P(None,'x','y'),
    or their rank-2 counterparts. Each output has B's shape and face layout.
    Units multiply as in matmul. The staged route shares only ingress;
    neither operands nor outputs are stacked. Distributed providers retain
    their existing two calls. All ranks run the same program and shapes.
    """
    if A.shape[-2] != A.shape[-1]:
        raise ValueError("matmul_adjoint_pair requires square A")
    return _matmul(A, B, None, mesh=mesh, alpha=1.0, beta=0.0,
                   transa="N", transb="N", backend=backend,
                   batched_route=batched_route, adjoint_pair=True)


def _matmul(A, B, C, *, mesh, alpha, beta, transa, transb, backend,
            batched_route, adjoint_pair=False):
    transa, transb = str(transa).upper(), str(transb).upper()
    if transa not in _OP_CODE or transb not in _OP_CODE:
        raise ValueError(
            f"transa/transb must be N/T/C; got {transa!r}/{transb!r}")
    beta_c = complex(beta)
    if C is None and beta_c != 0:
        raise ValueError("C is required when beta is nonzero")
    out_shape = _validate_operands(A, B, C, transa, transb)
    route = str(batched_route).strip().lower()
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
    if route == ROUTE_BATCH_RESHARD:
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
            transa=transa, transb=transb, adjoint_pair=adjoint_pair)
    else:
        if adjoint_pair:
            return tuple(matmul(A[0] if single else A, B[0] if single else B,
                                mesh=mesh, transa=mode, backend=backend,
                                batched_route=route) for mode in ("N", "C"))
        if C is None:
            C = _zeros(out_shape, A.dtype, sharding)
        out = _provider_matmul(
            provider, mesh, A, B, C, alpha=alpha_arg, beta=beta_arg,
            transa=transa, transb=transb)
    if adjoint_pair:
        return tuple(value[0] for value in out) if single else out
    return out[0] if single else out
