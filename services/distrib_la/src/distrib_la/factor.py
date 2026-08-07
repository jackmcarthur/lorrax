"""``factor`` / ``solve`` — the factor-once, back-solve-many surface.

The plan facade (:mod:`distrib_la.plan`) covers ops whose result is an
ARRAY.  Three of this package's routes do not return one: their factor is
a library handle carrying block-cyclic geometry that only means anything
on the grid that produced it, and every consumer of them in LORRAX had
grown the same comment —

    never reshard this, feed it back verbatim

— next to a hand-rebuilt handle, a stacked list, or an opaque ``ipiv``
threaded through three call frames.  :class:`FactorToken` makes that
comment the TYPE.  A token is opaque: it carries the factor, the mesh it
was made on and the extents it was made at, and :func:`solve` consumes it
whole.  There is nothing to reshard because there is nothing to reach.

The four handle paths this replaces (all measured in LORRAX at 96a6399)::

    scalapack solve_lu   batched_distributed_getrf -> (LU, ipiv)
                         ipiv is P(None,('x','y')) i32, never gathered
    cusolvermp cholesky  batched_distributed_cholesky(...).raw, then the
                         CALLER rebuilding CusolverMpBatchedLowerL(raw,
                         mesh, n, mb, nb, nbatch) hundreds of lines later
    slate cholesky       a per-q Python loop over distributed_cholesky
                         plus jnp.stack, because a SlateLowerL is not an
                         array and plan.batched refuses to stack handles

``factor`` collapses the SLATE loop into the token and removes the
rebuild: the token IS the handle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la.plan import ensure_sharding
from distrib_la.resolve import NATIVE, backend_module, resolve_backend

__all__ = ["FactorToken", "factor", "solve"]

#: Ops that have a factor/solve split at all.
FACTOR_OPS = ("cholesky", "solve_lu")


@dataclass(frozen=True)
class FactorToken:
    """An opaque distributed factorization.  Feed it to :func:`solve`.

    Attributes
    ----------
    op, backend
        What was factored and by which library.  Introspection only —
        branching on ``backend`` outside this package is the drift the
        token exists to remove.
    mesh, n, nbatch
        The grid and extents the factor is valid on.  :func:`solve`
        checks the RHS against them, so a mismatched B refuses here
        instead of corrupting a solve or hanging in a collective.

    The factor itself is deliberately NOT a public attribute: its layout
    is block-cyclic on ``mesh``'s specific grid, and "feed it back
    verbatim" is only enforceable while nothing can take it out.
    """

    op: str
    backend: str
    mesh: Mesh
    n: int
    nbatch: int
    _factor: Any = field(repr=False)

    def __repr__(self) -> str:                      # no factor bytes in logs
        px, py = int(self.mesh.shape["x"]), int(self.mesh.shape["y"])
        return (f"FactorToken({self.op}/{self.backend}, n={self.n}, "
                f"nbatch={self.nbatch}, mesh={px}x{py})")


def factor(op: str, A, mesh_xy: Mesh, *, backend: str = "auto",
           n: int | None = None) -> FactorToken:
    """Factor a STACK ``A`` ``(nbatch, n, n)`` once; return an opaque token.

    ``op`` is ``'cholesky'`` (``potrf``) or ``'solve_lu'`` (``getrf``).
    Resolution is :func:`distrib_la.resolve.resolve_backend` — the same
    ladder, the same refusals, the same promise — so a token's backend is
    a backend whose guards all passed.

    ``A`` is moved to ``P(None,'x','y')`` first and is DONATED (see
    :data:`distrib_la.plan.DONATES`): the factors are written over its
    buffer wherever the library supports it.

    The pure-JAX backends have no split: ``native`` cholesky IS the
    caller's channel-policy route and ``native`` solve_lu is a per-q
    ``jnp.linalg.solve``, neither of which produces a reusable
    distributed factor.  Both refuse here rather than pretending, and name
    what to call instead.
    """
    if op not in FACTOR_OPS:
        raise ValueError(
            f"factor() is defined for {'|'.join(FACTOR_OPS)}, got {op!r}.  "
            f"eigh has no factor/solve split — its factor IS its result.")
    if getattr(A, "ndim", None) != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"factor({op!r}): expected A of shape (nbatch, n, n); "
            f"got {getattr(A, 'shape', type(A))}")
    nb, extent = int(A.shape[0]), int(A.shape[1])
    resolved = resolve_backend(op, backend, mesh_xy,
                               n=extent if n is None else int(n))
    if resolved == NATIVE:
        raise NotImplementedError(
            f"factor({op!r}) resolved to the NATIVE backend, which has no "
            f"reusable distributed factor: the native cholesky is the "
            f"caller's own channel-policy route and the native LU is a "
            f"per-q jnp.linalg.solve.  Request a distributed backend, or "
            f"branch on plan(...).is_native and factor it yourself.")

    mod = backend_module(resolved)
    A = ensure_sharding(A, NamedSharding(mesh_xy, P(None, "x", "y")))

    if op == "solve_lu":
        if resolved != "scalapack":
            raise NotImplementedError(
                f"factor('solve_lu') has no split on backend {resolved!r}: "
                f"its FFI entry point is the FUSED getrf+getrs "
                f"(batched_distributed_solve_lu), which allocates its "
                f"pivots per call and never surfaces them.  Call "
                f"plan('solve_lu', mesh, backend={resolved!r}).batched(A, B) "
                f"instead — one factor per solve is what that route costs.")
        held = mod.batched_distributed_getrf(A, mesh=mesh_xy)
    elif resolved == "cusolvermp":
        held = mod.batched_distributed_cholesky(A, mesh=mesh_xy)
    else:                                                       # slate
        # The per-q loop that lived at isdf/core.py:3199, moved inside.
        # SLATE's batched potrf distributes the BATCH over mesh 'x', which
        # does not match this call site's replicated-q layout; a per-q loop
        # over nq <~ tens of matrices is the correct shape here.  The
        # handles are kept as handles — that is the whole point of the
        # token, and it is what lets solve() use slate::trsm instead of
        # materialising a row-major L and solving it replicated.
        held = tuple(mod.distributed_cholesky(A[i], mesh=mesh_xy)
                     for i in range(nb))

    return FactorToken(op=op, backend=resolved, mesh=mesh_xy,
                       n=extent, nbatch=nb, _factor=held)


def solve(token: FactorToken, B) -> jax.Array:
    """Solve ``A[q] X[q] = B[q]`` from ``token``, for ``B`` ``(nbatch, n, nrhs)``.

    ``B`` is moved to ``P(None,'x','y')`` and is DONATED.  ``X`` comes back
    in the same shape and sharding.  The token is consumed VERBATIM: no
    reshard, no gather, no reinterpretation of a pivot vector.
    """
    if not isinstance(token, FactorToken):
        raise TypeError(f"solve() takes a FactorToken from factor(); "
                        f"got {type(token).__name__}")
    if getattr(B, "ndim", None) != 3:
        raise ValueError(
            f"solve: expected B of shape (nbatch, n, nrhs); "
            f"got {getattr(B, 'shape', type(B))}")
    if int(B.shape[0]) != token.nbatch or int(B.shape[1]) != token.n:
        raise ValueError(
            f"solve: B is {tuple(B.shape)} but {token} was factored at "
            f"nbatch={token.nbatch}, n={token.n}.  A token is valid only "
            f"for the system it factored.")
    mesh = token.mesh
    B = ensure_sharding(B, NamedSharding(mesh, P(None, "x", "y")))
    mod = backend_module(token.backend)

    if token.op == "solve_lu":                                  # scalapack
        LU, ipiv = token._factor
        return mod.batched_distributed_getrs(LU, ipiv, B, mesh=mesh)
    if token.backend == "cusolvermp":
        return mod.batched_distributed_potrs(token._factor, B, mesh=mesh)

    # SLATE cholesky: two triangular solves per q against the handle.
    # ``distributed_trsm`` feeds the SlateLowerL buffer straight back to
    # SLATE (op='N' forward, op='C' adjoint), so the col-major factor is
    # never materialised into a row-major L and never replicated.
    #
    # UNMEASURED ON A REAL MESH as of this commit: LORRAX's slate cholesky
    # consumer went through ``to_jax_lower()`` and a replicated triangular
    # solve, with "wiring slate::trsm for the back-solve is a perf
    # follow-up" written next to it.  This is that wiring; its first
    # execution is the service suite's real-4-process leg.  The shape
    # algebra and the refusals below are what is checked today.
    import jax.numpy as jnp
    rows = []
    for i in range(token.nbatch):
        L = token._factor[i]
        y = mod.distributed_trsm(L, B[i], mesh=mesh, op="N")
        rows.append(mod.distributed_trsm(L, y, mesh=mesh, op="C"))
    return ensure_sharding(jnp.stack(rows),
                           NamedSharding(mesh, P(None, "x", "y")))
