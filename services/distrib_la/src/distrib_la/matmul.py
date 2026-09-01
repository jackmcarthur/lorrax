"""Top-level distributed GEMM across the :mod:`distrib_la` providers.

The default route calls an actual 2-D provider operation: cuBLASMp next to
cuSOLVERMp, PBLAS ``pdgemm``/``pzgemm`` next to ScaLAPACK, or
``slate::multiply`` next to SLATE.  Unlike :func:`distrib_la.plan`,
``backend='auto'`` here selects that platform provider; it does not select a
native JAX floor.

The explicit ``batched_route='batch_reshard'`` route performs x-then-y staged
face-to-batch exchanges for A, B, and C, runs local ``jnp.matmul``, then
applies the literal y-then-x inverse exchanges to D.  Use ``backend='off'``
with that route for a provider-free call.  A non-``off`` request is still
resolved and capability-probed even though the selected route does not call
the provider.
"""
from __future__ import annotations

from functools import partial
from typing import Union

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la import loader
from distrib_la._shard_map import shard_map
from distrib_la.plan import (BATCHED_ROUTE_CHOICES, ROUTE_BATCH_RESHARD,
                             ensure_sharding)
from distrib_la.resolve import mesh_key, mesh_platform

__all__ = ["MATMUL_BACKEND_CHOICES", "matmul", "resolve_matmul_backend"]

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
    if provider in ("cublasmp", "scalapack", "slate") and px != py:
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
                           batched_route: str = "auto") -> str:
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


def _zeros(shape, dtype, sharding):
    return jax.jit(
        lambda: jnp.zeros(shape, dtype=dtype), out_shardings=sharding)()


def _cublasmp(mesh, A, B, C, *, alpha: complex, beta: complex,
              transa: str, transb: str):
    from distrib_la._cusolvermp import get_or_init_context

    px, py = _mesh_shape(mesh)
    if (transa != "N" or transb != "N") and px * py > 1:
        raise ValueError(
            f"cuBLASMp matmul op={transa}/{transb} on {px}x{py} is "
            "refused: multi-rank transpose modes are not trustworthy "
            "(transa returned a wrong result; transb can deadlock). Use "
            "SLATE/PBLAS, pretranspose into a face-sharded array, or select "
            "the batch_reshard route.")
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


def _batch_reshard(mesh, A, B, C, *, alpha, beta, transa, transb):
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
    batched_route: str = "auto",
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
        ``'auto'`` calls the resolved distributed provider.  Explicit
        ``'batch_reshard'`` pads a ragged leading batch with zero matrices,
        exchanges each face x then y into whole per-device matrices, runs
        local JAX GEMM, and returns D through the inverse y then x exchanges.

    Returns
    -------
    jax.Array
        Rank 2 at ``P('x','y')`` or rank 3 at ``P(None,'x','y')``, matching
        the input rank and the shape of ``op(A) @ op(B)``.

    Notes
    -----
    Provider routes require float64 or complex128, one JAX process per mesh
    cell in y-minor order, exact face tiling, and an available handler.
    cuBLASMp, ScaLAPACK, and SLATE require a square mesh; multi-rank
    cuBLASMp accepts only ``N,N``.
    Its transpose-A mode returned a wrong answer in the real P=4 gate, while
    transpose-B can return rank-divergent ``INVALID_VALUE`` and deadlock; both
    are refused before entering the provider.

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
            transa=transa, transb=transb)
    else:
        if C is None:
            C = _zeros(out_shape, A.dtype, sharding)
        out = _provider_matmul(
            provider, mesh, A, B, C, alpha=alpha_arg, beta=beta_arg,
            transa=transa, transb=transb)
    return out[0] if single else out
