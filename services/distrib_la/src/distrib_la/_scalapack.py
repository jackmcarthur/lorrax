"""ScaLAPACK FFI backend — host-platform distributed linear algebra.

Ported verbatim from LORRAX ``src/ffi/linalg/_scalapack.py`` at 96a6399
(itself the wave-2 merge of the former ``ffi.scalapack`` package:
``eigh.py`` — ``pXheevd``, the permanent CPU backend for distributed eigh —
and ``solve_lu.py`` — ``pXgetrf``/``pXgetrs``, the ``distributed_lu`` axis).
Nothing is renamed; the only edits are the import lines.

**ScaLAPACK is a published API, not a product.**  Which implementation is
linked (MKL / LibSci / netlib / AOCL) is decided in
``src/ffi/cpp/CMakeLists.txt`` and nothing here can observe it — do not
write a vendor name into this family's code as if it were a fact about the
build.  Both ops need a SQUARE or 1-D ``('x','y')`` mesh (square descriptor
blocks, one tile per rank).

The SlateCtx lifecycle (``ensure_registered`` / ``get_or_init_context`` /
``validate_mesh`` / ``_mesh_key``) is shared with the SLATE backend and
imported from :mod:`._slate`: the BLACS grid is built on the slate
context's rank-remapped MPI comm.

Section docstrings from the merged modules follow inline.
"""
from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

from distrib_la._shard_map import shard_map
from distrib_la._slate import (_mesh_key, ensure_registered,
                               get_or_init_context, validate_mesh)

__all__ = [
    "batched_distributed_matmul",
    "batched_distributed_eigh",
    "distributed_eigh",
    "validate_eigh_mesh",
    "batched_distributed_solve_lu",
    "batched_distributed_getrf",
    "batched_distributed_getrs",
]

# =========================================================================
#  eigh  (merged from scalapack/eigh.py; original docstring below)
# =========================================================================
"""``batched_distributed_eigh`` — ScaLAPACK ``pzheevd`` / ``pdsyevd``.

The **permanent CPU backend for distributed Hermitian eigendecomposition**
(``distrib_la.resolve_backend('eigh', 'distributed', <cpu mesh>)``).  SLATE's
host ``heev`` SIGSEGVs deterministically on this stack — bug L-2, reproduced
down to a 1×1 mesh, single rank — while ScaLAPACK's routines on the SAME
library and MPI context are clean.

Layout contract
---------------
::

    A : (Nq, N, N)  P(None, 'x', 'y')     Hermitian (lower triangle read)
    W : (Nq, N)     REPLICATED float64    ascending eigenvalues
    Z : (Nq, N, N)  P(None, 'x', 'y')     eigenvectors as COLUMNS,
                                          A[q] @ Z[q] == Z[q] @ diag(W[q])

``A`` is not donated (``pXheevd`` destroys its input, so the handler stages
each q into a scratch block).

Mesh restriction: ``pXheevd`` requires SQUARE descriptor blocks
(``MB_A == NB_A``), so with the one-tile-per-rank layout only **square and
1-D** meshes are layout-consistent (block ``g = N / max(Px, Py)``) — the
same geometry class as :func:`batched_distributed_solve_lu`, and unlike
SLATE there is no 1×q stride assert, so ``1×q`` is allowed too.  Host-only:
on a GPU mesh use the cuSOLVERMp backend.

GAUGE WARNING for callers: eigenVECTORS are not mesh-invariant.  Any
degenerate (or numerically near-degenerate) eigenvalue leaves an arbitrary
unitary rotation inside its subspace, and the block-cyclic reduction picks
a different representative on a different process grid.  Compare
gauge-INVARIANT combinations (``W``, ``Z f(W) Zᴴ``, a projector, a
pseudo-inverse) across meshes — never ``Z`` itself.
"""

_EIGH_TARGET = "lorrax_scalapack_eigh"
_GEMM_TARGET = "lorrax_scalapack_batched_gemm"

# jit(shard_map) cache per signature — eager shard_map re-traces per call.
_JIT_CACHE: dict = {}


def batched_distributed_matmul(
    A: jax.Array, B: jax.Array, C: jax.Array, *, mesh: Mesh,
    alpha=1.0, beta=0.0, transa: str = "N", transb: str = "N",
) -> jax.Array:
    """PBLAS ``pdgemm``/``pzgemm`` over a face-sharded matrix stack.

    A, B, and C are rank-3 ``P(None,'x','y')`` arrays; the result has C's
    shape and returns at the same sharding.  The public
    :func:`distrib_la.matmul` door supplies rank-2 lifting, placement,
    provider resolution, and the optional zero C.  This backend requires a
    CPU process grid, float64 or complex128, N/T/C operation codes, exact
    one-face-per-rank tiling, and one JAX process per mesh cell.
    """
    if A.ndim != 3 or B.ndim != 3 or C.ndim != 3:
        raise ValueError("scalapack matmul expects three rank-3 operands")
    if A.dtype != B.dtype or A.dtype != C.dtype:
        raise ValueError("scalapack matmul operands must share dtype")
    if A.dtype not in (jnp.dtype("float64"), jnp.dtype("complex128")):
        raise ValueError(
            f"scalapack matmul supports float64/complex128; got {A.dtype}")
    transa, transb = transa.upper(), transb.upper()
    if transa not in "NTC" or transb not in "NTC":
        raise ValueError("scalapack matmul transa/transb must be N/T/C")
    if mesh.devices.flat[0].platform != "cpu":
        raise ValueError("scalapack matmul is host-only")
    px, py = validate_mesh(mesh)
    nq, ar, ac = map(int, A.shape)
    br, bc = int(B.shape[1]), int(B.shape[2])
    m, ka = ((ar, ac) if transa == "N" else (ac, ar))
    kb, n = ((br, bc) if transb == "N" else (bc, br))
    if ka != kb or tuple(C.shape) != (nq, m, n):
        raise ValueError(
            f"scalapack matmul shapes disagree: A={A.shape}, B={B.shape}, "
            f"C={C.shape}, trans={transa}/{transb}")
    for dim, divisor, name in (
            (ar, px, "A rows"), (ac, py, "A cols"),
            (br, px, "B rows"), (bc, py, "B cols"),
            (m, px, "C rows"), (n, py, "C cols")):
        if dim % divisor:
            raise ValueError(
                f"scalapack matmul {name}={dim} not divisible by {divisor}")
    ensure_registered(mesh)
    ctx = get_or_init_context(mesh)
    alpha, beta = complex(alpha), complex(beta)
    attrs = dict(
        nq=nq, m=m, n=n, k=ka,
        a_rows=ar, a_cols=ac, b_rows=br, b_cols=bc,
        mb_a=ar // px, nb_a=ac // py,
        mb_b=br // px, nb_b=bc // py,
        mb_c=m // px, nb_c=n // py,
        transa={"N": 0, "T": 1, "C": 2}[transa],
        transb={"N": 0, "T": 1, "C": 2}[transb],
        alpha_re=float(alpha.real), alpha_im=float(alpha.imag),
        beta_re=float(beta.real), beta_im=float(beta.imag),
        ctx_handle=int(ctx))
    key = ("scalapack_gemm", _mesh_key(mesh), A.dtype, A.shape, B.shape,
           C.shape, transa, transb, alpha, beta, int(ctx))
    fn = _JIT_CACHE.get(key)
    if fn is None:
        local_out_t = jax.ShapeDtypeStruct((nq, n // py, m // px), C.dtype)

        @partial(shard_map, mesh=mesh, in_specs=(P(None, "x", "y"),) * 3,
                 out_specs=P(None, "y", "x"), check_vma=False)
        def _call(a, b, c):
            return jax.ffi.ffi_call(
                _GEMM_TARGET, local_out_t,
                input_output_aliases={2: 0})(
                    jnp.transpose(a, (0, 2, 1)),
                    jnp.transpose(b, (0, 2, 1)),
                    jnp.transpose(c, (0, 2, 1)), **attrs)

        @partial(shard_map, mesh=mesh, in_specs=P(None, "y", "x"),
                 out_specs=P(None, "x", "y"), check_vma=False)
        def _untranspose(d):
            return jnp.transpose(d, (0, 2, 1))

        fn = jax.jit(lambda a, b, c: _untranspose(_call(a, b, c)),
                     donate_argnums=(2,))
        _JIT_CACHE[key] = fn
    return fn(A, B, C)


def validate_eigh_mesh(mesh: Mesh) -> tuple[int, int]:
    """``(Px, Py)`` for a mesh this backend can run on; raises otherwise.

    THE call-time geometry guard, mirrored by
    ``distrib_la.resolve._check_geometry`` so a resolved ``'scalapack'``
    is a promise the call keeps (the lesson of bug L-1).
    """
    Px, Py = validate_mesh(mesh)
    if Px > 1 and Py > 1 and Px != Py:
        raise ValueError(
            f"scalapack eigh: mesh {Px}x{Py} unsupported — pXheevd needs "
            f"square descriptor blocks (MB == NB), which the "
            f"one-tile-per-rank layout only gives on square or 1-D meshes.")
    return Px, Py


def batched_distributed_eigh(
    A: jax.Array, *, mesh: Mesh,
) -> Tuple[jax.Array, jax.Array]:
    """Eigendecompose every ``A[q]`` with ScaLAPACK's divide & conquer.

    One FFI call for the whole stack (the handler loops q internally,
    reusing one descriptor and one workspace), so ``Nq`` matrices cost one
    collective-serialisation round, not ``Nq``.

    See the module docstring for the sharding contract and the gauge
    warning.
    """
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"scalapack eigh: expected A of shape (Nq, N, N); got {A.shape}")

    plat = mesh.devices.flat[0].platform
    if plat != "cpu":
        raise ValueError(
            f"scalapack eigh is host-only but the mesh devices are {plat!r}; "
            f"use the cusolvermp backend on GPU meshes.")

    Px, Py = validate_eigh_mesh(mesh)
    nq = int(A.shape[0])
    n = int(A.shape[1])
    gmax = max(Px, Py)
    if n % gmax != 0:
        raise ValueError(f"N={n} must be divisible by max(Px,Py)={gmax}.")

    ensure_registered(mesh)
    ctx_handle = get_or_init_context(mesh)

    g = n // gmax                      # square block (MB_A == NB_A)

    key = ("scalapack_eigh", _mesh_key(mesh), A.dtype, nq, n, g,
           int(ctx_handle))
    jit_eigh = _JIT_CACHE.get(key)
    if jit_eigh is None:
        W_out = jax.ShapeDtypeStruct((nq, n), jnp.float64)
        # Z comes back as ScaLAPACK col-major local blocks = row-major
        # bytes of the per-rank block of Zᵀ; assembled at P(None,'y','x')
        # then locally untransposed (same trick as the slate eigh).
        Z_local_T = jax.ShapeDtypeStruct((nq, n // Py, n // Px), A.dtype)
        attrs = dict(nq=nq, n=n, g=g, ctx_handle=int(ctx_handle))

        @partial(shard_map, mesh=mesh,
                 in_specs=P(None, "x", "y"),
                 out_specs=(P(), P(None, "y", "x")),
                 check_vma=False)
        def _call(local_A):
            # Inner transpose: row-major shard → col-major local block
            # (same convention as every slate/scalapack wrapper).
            local_A_T = jnp.transpose(local_A, (0, 2, 1))
            return jax.ffi.ffi_call(
                _EIGH_TARGET, (W_out, Z_local_T))(local_A_T, **attrs)

        @partial(shard_map, mesh=mesh,
                 in_specs=P(None, "y", "x"), out_specs=P(None, "x", "y"),
                 check_vma=False)
        def _untranspose(local_Z_T):
            return jnp.transpose(local_Z_T, (0, 2, 1))

        def _eigh(A_):
            W, Z_T = _call(A_)
            return W, _untranspose(Z_T)

        jit_eigh = jax.jit(_eigh)
        _JIT_CACHE[key] = jit_eigh

    return jit_eigh(A)


def distributed_eigh(A: jax.Array, *, mesh: Mesh) -> Tuple[jax.Array, jax.Array]:
    """Single-matrix form of :func:`batched_distributed_eigh`.

    ``A`` is ``(N, N)`` at ``P('x','y')``; returns ``(W, Z)`` with ``W``
    ``(N,)`` replicated and ``Z`` ``(N, N)`` at ``P('x','y')``.  This is the
    shape :meth:`distrib_la.plan.Plan.__call__` uses (one tile spread over
    the mesh); the batched entry point is what the ζ-fit uses.
    """
    if A.ndim != 2:
        raise ValueError(
            f"scalapack distributed_eigh: expected a square matrix, "
            f"got {A.shape}")
    W, Z = batched_distributed_eigh(A[None], mesh=mesh)
    return W[0], Z[0]


# =========================================================================
#  solve_lu  (merged from scalapack/solve_lu.py; the split getrf/getrs
#  pair — hoisted transverse ζ factor stage — merged from the
#  zeta-transverse-hoist branch, 2026-08)
# =========================================================================
"""``batched_distributed_solve_lu`` — per-q ScaLAPACK ``pXgetrf`` + ``pXgetrs``.

Host-platform (JAX CPU backend) twin of the cuSOLVERMp batched solve — the
portable backend for the ``distributed_lu`` axis.  Identical call contract:

    A : (Nq, N, N)     P(None, 'x', 'y')   (donated — factored in place)
    B : (Nq, N, NRHS)  P(None, 'x', 'y')
    X : same shape/sharding as B

The BLACS grid is built on the slate context's rank-remapped MPI comm ("C"
grid order lands JAX shard (mx, my) at grid (mx, my)); the wrapper reuses
:mod:`._slate` for mesh validation, registration, and the per-(p, q) ctx
cache.

Mesh restriction: ``pXgetrf`` requires SQUARE descriptor blocks
(MB_A == NB_A), so with the one-tile-per-rank layout only square and 1-D
meshes are layout-consistent (block g = N / max(Px, Py)) — same geometry
class as the slate single-matrix ops, but 1×q is fine here (no SLATE
stride assert).  Host-only: on a GPU mesh use ``distributed_lu =
cusolvermp`` instead.
"""

_SOLVE_LU_TARGET = "lorrax_scalapack_batched_solve_lu"
_GETRF_TARGET = "lorrax_scalapack_batched_getrf"
_GETRS_TARGET = "lorrax_scalapack_batched_getrs"


def _validate_lu_geometry(A, mesh: Mesh, *, what: str):
    """Shared guard ladder for the LU family (fused solve, getrf, getrs):
    host platform, registered square/1-D mesh, divisibility.  Returns
    ``(Px, Py, nq, n, g)``."""
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"scalapack {what}: expected A of shape (Nq, N, N); got {A.shape}")
    plat = mesh.devices.flat[0].platform
    if plat != "cpu":
        raise ValueError(
            f"scalapack {what} is host-only but the mesh devices are "
            f"{plat!r}; use distributed_lu = cusolvermp on GPU meshes.")
    Px, Py = validate_mesh(mesh)
    if Px > 1 and Py > 1 and Px != Py:
        raise ValueError(
            f"scalapack {what}: mesh {Px}x{Py} unsupported — pXgetrf "
            f"needs square blocks (MB == NB), which the one-tile-per-rank "
            f"layout only gives on square or 1-D meshes.")
    nq = int(A.shape[0])
    n = int(A.shape[1])
    gmax = max(Px, Py)
    if n % gmax != 0:
        raise ValueError(f"N={n} must be divisible by max(Px,Py)={gmax}.")
    return Px, Py, nq, n, n // gmax


def _ipiv_local_len(n: int, Px: int, g: int) -> int:
    """Per-rank ipiv extent: ``LOCr(M_A) + MB_A`` per the ScaLAPACK spec.
    On the square / 1-D meshes the wrapper allows, ``LOCr = n / Px``."""
    return n // max(Px, 1) + g


def batched_distributed_getrf(
    A: jax.Array,
    *,
    mesh: Mesh,
) -> tuple[jax.Array, jax.Array]:
    """Factor ``A[q]`` via per-q ScaLAPACK ``pXgetrf``, ONCE — the hoisted
    transverse ζ factor stage.

    Same sharding contract as :func:`batched_distributed_solve_lu`'s A
    (``(Nq, N, N)`` at ``P(None,'x','y')``, donated — factored in place),
    but the factors are RETURNED instead of consumed:

    Returns
    -------
    LU : (Nq, N, N) at ``P(None, 'x', 'y')``
        The block-cyclic L/U factors; each rank's shard is its local
        block, exactly as ``pXgetrf`` left it.  Feed back VERBATIM into
        :func:`batched_distributed_getrs` — never reshard it (the values
        only mean anything in this layout on this grid).
    ipiv : (Nq, P·ipiv_len) int32 at ``P(None, ('x','y'))``
        Each rank's own ScaLAPACK ipiv rows (``ipiv_len = LOCr + MB``);
        opaque, never gathered, never interpreted host-side.

    ``pXgetrf`` on a given matrix is bit-identical whether or not the
    ``pXgetrs`` follows immediately (same descriptors, same grid), so
    getrf-once + getrs-per-r-chunk reproduces the fused
    :func:`batched_distributed_solve_lu` to the bit — gated in
    ``tests/test_transverse_factor_hoist.py`` (srun leg).

    :func:`distrib_la.factor` is the facade over this pair; it carries the
    ``(LU, ipiv)`` in an opaque token so "feed back verbatim" stops being a
    comment and becomes the type.
    """
    Px, Py, nq, n, g = _validate_lu_geometry(A, mesh, what="getrf")
    ensure_registered(mesh)
    ctx_handle = get_or_init_context(mesh)
    ipiv_len = _ipiv_local_len(n, Px, g)

    key = ("scalapack_getrf", _mesh_key(mesh), A.dtype,
           nq, n, g, int(ctx_handle))
    jit_getrf = _JIT_CACHE.get(key)
    if jit_getrf is None:
        LU_local_T = jax.ShapeDtypeStruct((nq, n // Py, n // Px), A.dtype)
        ipiv_local = jax.ShapeDtypeStruct((nq, ipiv_len), jnp.int32)
        attrs = dict(nq=nq, n=n, g=g, ipiv_len=ipiv_len,
                     ctx_handle=int(ctx_handle))

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "x", "y"),),
                 out_specs=(P(None, "x", "y"), P(None, ("x", "y"))),
                 check_vma=False)
        def _getrf(local_A):
            local_A_T = jnp.transpose(local_A, (0, 2, 1))
            LU_T, ipiv = jax.ffi.ffi_call(
                _GETRF_TARGET, (LU_local_T, ipiv_local),
                input_output_aliases={0: 0},
            )(local_A_T, **attrs)
            return jnp.transpose(LU_T, (0, 2, 1)), ipiv

        jit_getrf = jax.jit(_getrf, donate_argnums=(0,))
        _JIT_CACHE[key] = jit_getrf

    return jit_getrf(A)


def batched_distributed_getrs(
    LU: jax.Array,
    ipiv: jax.Array,
    B: jax.Array,
    *,
    mesh: Mesh,
) -> jax.Array:
    """Solve ``A[q] X[q] = B[q]`` from the factors of
    :func:`batched_distributed_getrf` via per-q ScaLAPACK ``pXgetrs`` —
    the per-r-chunk half of the hoisted transverse factor stage.

    ``LU``/``ipiv`` must arrive EXACTLY as getrf returned them (same
    mesh, same shardings); ``B`` is ``(Nq, N, NRHS)`` at
    ``P(None,'x','y')`` with ``NRHS % Py == 0`` (donated — solved in
    place).  Returns X in B's shape/sharding.
    """
    Px, Py, nq, n, g = _validate_lu_geometry(LU, mesh, what="getrs")
    if (B.ndim != 3 or int(B.shape[0]) != nq or int(B.shape[1]) != n):
        raise ValueError(
            f"scalapack getrs: B must be (Nq, N, NRHS) matching LU; "
            f"got LU={LU.shape}, B={B.shape}")
    if LU.dtype != B.dtype:
        raise ValueError(f"LU.dtype {LU.dtype} != B.dtype {B.dtype}")
    ipiv_len = _ipiv_local_len(n, Px, g)
    P_tot = Px * Py
    if (ipiv.ndim != 2 or int(ipiv.shape[0]) != nq
            or int(ipiv.shape[1]) != P_tot * ipiv_len
            or ipiv.dtype != jnp.int32):
        raise ValueError(
            f"scalapack getrs: ipiv must be (Nq, P*ipiv_len) int32 = "
            f"({nq}, {P_tot * ipiv_len}) as returned by getrf; got "
            f"{ipiv.shape} {ipiv.dtype}")
    nrhs = int(B.shape[2])
    if nrhs % Py != 0:
        raise ValueError(f"NRHS={nrhs} must be divisible by Py={Py}.")

    ensure_registered(mesh)
    ctx_handle = get_or_init_context(mesh)
    nb_b = nrhs // Py if Py > 1 else nrhs

    key = ("scalapack_getrs", _mesh_key(mesh), LU.dtype,
           nq, n, nrhs, g, nb_b, int(ctx_handle))
    jit_getrs = _JIT_CACHE.get(key)
    if jit_getrs is None:
        X_local_T = jax.ShapeDtypeStruct((nq, nrhs // Py, n // Px), B.dtype)
        attrs = dict(nq=nq, n=n, nrhs=nrhs, g=g, nb_b=nb_b,
                     ipiv_len=ipiv_len, ctx_handle=int(ctx_handle))

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "x", "y"), P(None, ("x", "y")),
                           P(None, "x", "y")),
                 out_specs=P(None, "x", "y"),
                 check_vma=False)
        def _getrs(local_LU, local_ipiv, local_B):
            local_LU_T = jnp.transpose(local_LU, (0, 2, 1))
            local_B_T = jnp.transpose(local_B, (0, 2, 1))
            X_T = jax.ffi.ffi_call(
                _GETRS_TARGET, X_local_T,
                input_output_aliases={2: 0},
            )(local_LU_T, local_ipiv, local_B_T, **attrs)
            return jnp.transpose(X_T, (0, 2, 1))

        jit_getrs = jax.jit(_getrs, donate_argnums=(2,))
        _JIT_CACHE[key] = jit_getrs

    return jit_getrs(LU, ipiv, B)


def batched_distributed_solve_lu(
    A: jax.Array,
    B: jax.Array,
    *,
    mesh: Mesh,
) -> jax.Array:
    """Solve ``A[q] X[q] = B[q]`` via per-q ScaLAPACK ``getrf`` + ``getrs``.

    See the module docstring for the sharding contract.  ``A`` is donated
    (the LU factors scribble over its buffer); pivots stay internal.
    """
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"scalapack solve_lu: expected A of shape (Nq, N, N); got {A.shape}")
    if (B.ndim != 3 or int(B.shape[0]) != int(A.shape[0])
            or int(B.shape[1]) != int(A.shape[1])):
        raise ValueError(
            f"scalapack solve_lu: B must be (Nq, N, NRHS) with Nq, N "
            f"matching A; got A={A.shape}, B={B.shape}")
    if A.dtype != B.dtype:
        raise ValueError(f"A.dtype {A.dtype} != B.dtype {B.dtype}")

    plat = mesh.devices.flat[0].platform
    if plat != "cpu":
        raise ValueError(
            f"scalapack solve_lu is host-only but the mesh devices are "
            f"{plat!r}; use distributed_lu = cusolvermp on GPU meshes.")

    Px, Py = validate_mesh(mesh)
    if Px > 1 and Py > 1 and Px != Py:
        raise ValueError(
            f"scalapack solve_lu: mesh {Px}x{Py} unsupported — pXgetrf "
            f"needs square blocks (MB == NB), which the one-tile-per-rank "
            f"layout only gives on square or 1-D meshes.")

    nq = int(A.shape[0])
    n = int(A.shape[1])
    nrhs = int(B.shape[2])
    gmax = max(Px, Py)
    if n % gmax != 0:
        raise ValueError(f"N={n} must be divisible by max(Px,Py)={gmax}.")
    if nrhs % Py != 0:
        raise ValueError(f"NRHS={nrhs} must be divisible by Py={Py}.")

    ensure_registered(mesh)
    ctx_handle = get_or_init_context(mesh)

    g = n // gmax                      # square block (MB_A == NB_A)
    nb_b = nrhs // Py if Py > 1 else nrhs

    key = ("scalapack_lu", _mesh_key(mesh), A.dtype,
           nq, n, nrhs, g, nb_b, int(ctx_handle))
    jit_solve = _JIT_CACHE.get(key)
    if jit_solve is None:
        X_local_T = jax.ShapeDtypeStruct((nq, nrhs // Py, n // Px), B.dtype)
        attrs = dict(nq=nq, n=n, nrhs=nrhs, g=g, nb_b=nb_b,
                     ctx_handle=int(ctx_handle))

        @partial(shard_map, mesh=mesh,
                 in_specs=(P(None, "x", "y"), P(None, "x", "y")),
                 out_specs=P(None, "x", "y"),
                 check_vma=False)
        def _solve(local_A, local_B):
            # Inner transpose: row-major shard → col-major local block
            # (same convention as every slate/cusolvermp wrapper).
            local_A_T = jnp.transpose(local_A, (0, 2, 1))
            local_B_T = jnp.transpose(local_B, (0, 2, 1))
            X_T = jax.ffi.ffi_call(
                _SOLVE_LU_TARGET, X_local_T,
                input_output_aliases={1: 0},
            )(local_A_T, local_B_T, **attrs)
            return jnp.transpose(X_T, (0, 2, 1))

        jit_solve = jax.jit(_solve, donate_argnums=(0, 1))
        _JIT_CACHE[key] = jit_solve

    return jit_solve(A, B)
