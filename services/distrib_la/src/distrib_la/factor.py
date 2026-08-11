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

The batch axis is a ``lax.scan``, here as in :mod:`distrib_la.plan`
--------------------------------------------------------------------
The SLATE route factors one matrix per call and back-solves one matrix per
call, so both ends of it used to be a Python loop over ``nbatch``.  They
are now ``lax.scan``\\ s at :data:`distrib_la.plan.BATCHED_SCAN_UNROLL`,
for the same reason and with the same consequence as the batched surface
in that module: one traced body, one compiled loop, whatever ``nbatch`` is.

It matters more here than anywhere else in the package.
``_slate.distributed_cholesky`` and ``_slate.distributed_trsm`` build their
``shard_map`` at eager top level with no ``jax.jit`` around it and no
per-signature cache — the shape every OTHER wrapper here has and these two
do not — so an ``nbatch``-long Python loop re-traced and re-compiled the
kernel ``nbatch`` times.  A scan traces it once.

What the token holds for SLATE is therefore a STACKED raw factor
``(nbatch, n, n)`` at ``P(None,'y','x')`` plus the tile size, not a tuple
of ``nbatch`` handles.  That is invisible from outside — ``_factor`` is
private and always has been — and :func:`solve` rebuilds the per-``q``
``SlateLowerL`` inside its own scan body from a slice, which is the same
object the loop used to pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la import closure as _closure
from distrib_la.plan import (BATCHED_SCAN_UNROLL, cached_scan, ensure_sharding,
                             scan_signature)
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

    closure
        The degeneracy-closure mode :meth:`rank_cut` runs under, resolved
        once at :func:`factor` time.  ``'off'`` by default — see
        :mod:`distrib_la.closure`.

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
    closure: str = _closure.DEFAULT_MODE

    def __repr__(self) -> str:                      # no factor bytes in logs
        px, py = int(self.mesh.shape["x"]), int(self.mesh.shape["y"])
        cl = "" if self.closure == "off" else f", closure={self.closure}"
        return (f"FactorToken({self.op}/{self.backend}, n={self.n}, "
                f"nbatch={self.nbatch}, mesh={px}x{py}{cl})")

    def rank_cut(self, values, **kwargs):
        """Decide a rank in ``values`` under THIS token's closure mode.

        The factor/solve split is where a rank decision most obviously
        belongs — you have factored, and the next question is how much of
        the factor to trust — so the token carries the guard the same way
        :class:`distrib_la.Plan` does, with the same refusal on ``op=`` /
        ``closure=``.

        **THE SPECTRUM IS THE CALLER'S TO SUPPLY, AND THIS ROUND IT HAS
        TO BE.** A token's factor is opaque by design: ScaLAPACK's
        ``ipiv`` and LU, cuSOLVERMp's raw buffer and SLATE's
        ``SlateLowerL`` are block-cyclic on a specific grid, and there is
        deliberately no accessor to reach one. Pulling ``diag`` out of a
        block-cyclic handle is a distributed gather with a per-backend
        layout rule each — real work, and none of it measured. So a
        caller that wants a closed rank off an FFI factor materialises the
        factor through that backend's own documented route
        (``cholesky_handle_to_natural_L``, ``SlateLowerL.to_jax_lower``)
        and hands the spectrum here. **OWED, named rather than implied:
        a ``FactorToken.rank_spectrum()`` that reads the diagonal in
        place.**
        """
        for banned in ("op", "closure"):
            if banned in kwargs:
                raise ValueError(
                    f"FactorToken.rank_cut: {banned}= is the TOKEN's "
                    f"({getattr(self, banned)!r}) and cannot be overridden "
                    f"per call — a token is valid only for the system it "
                    f"factored, and that includes the guard it was "
                    f"factored under.")
        return _closure.rank_cut(self.op, values, closure=self.closure,
                                 **kwargs)


def factor(op: str, A, mesh_xy: Mesh, *, backend: str = "auto",
           n: int | None = None,
           closure: str | None = None) -> FactorToken:
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

    ``closure`` is the degeneracy-closure mode the returned token's
    :meth:`FactorToken.rank_cut` runs under — ``'snap'`` / ``'strict'`` /
    ``'off'``, or ``None`` for ``LORRAX_DISTRIB_LA_CLOSURE`` and then
    ``off``.  It is OPT-IN and changes NOTHING about the factorization:
    the same backend resolves, the same library runs, the same bytes come
    back.  It decides only what a rank cut taken OFF that factor may do.
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
        # The per-q loop that lived at isdf/core.py:3199, moved inside and
        # then made a scan.  SLATE's own batched potrf distributes the
        # BATCH over mesh 'x', which does not match this call site's
        # replicated-q layout, so factoring one matrix at a time is the
        # correct shape here — but "one at a time" is a loop the COMPILER
        # walks, not one Python walks.  The factor is kept in SLATE's own
        # layout, which is what lets solve() use slate::trsm instead of
        # materialising a row-major L and solving it replicated.
        held = _slate_potrf_stack(mod, A, mesh_xy)

    return FactorToken(op=op, backend=resolved, mesh=mesh_xy,
                       n=extent, nbatch=nb, _factor=held,
                       closure=_closure.resolve_mode(
                           _closure.mode_from_env() if closure is None
                           else closure))


def _slate_potrf_stack(mod, A, mesh_xy: Mesh) -> tuple:
    """Scan ``slate::potrf`` down the batch axis; return ``(raw, nb)``.

    ``raw`` is ``(nbatch, n, n)`` at ``P(None,'y','x')`` — the per-matrix
    ``SlateLowerL.raw`` buffers stacked, still in SLATE's col-major tile
    layout, never materialised as a row-major L and never replicated.
    ``nb`` is the tile size SLATE chose, carried out of the trace because
    it is the one piece of the handle that is not the buffer and
    :func:`solve` has to hand it back — so the cache stores the dict the
    body writes it into ALONGSIDE the compiled scan, because a cache HIT
    never re-enters the body to read it again.
    """
    def _build():
        tile: dict[str, int] = {}

        def _potrf(carry, A_q):
            L = mod.distributed_cholesky(A_q, mesh=mesh_xy)
            # Trace-time constant: the tile size is a function of n and the
            # mesh, so every iteration writes the same value.  Reading it
            # out of the body is how the handle's non-buffer half survives
            # a scan.
            tile["nb"] = int(L.nb)
            return carry, L.raw

        def _scanned(A_):
            _, raw_ = jax.lax.scan(_potrf, None, A_,
                                   unroll=BATCHED_SCAN_UNROLL)
            return raw_

        return jax.jit(_scanned), tile

    key = scan_signature("cholesky", "slate", mesh_xy, (A,), {},
                         extra="slate.potrf")
    fn, tile = cached_scan(key, _build)
    raw = fn(A)                     # traces (and fills ``tile``) on a MISS
    return raw, tile["nb"]


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

    # SLATE cholesky: two triangular solves per q against the handle,
    # scanned down the batch axis.  ``distributed_trsm`` feeds the
    # SlateLowerL buffer straight back to SLATE (op='N' forward, op='C'
    # adjoint), so the col-major factor is never materialised into a
    # row-major L and never replicated.  The handle is rebuilt per
    # iteration from a slice of the stacked raw plus the tile size the
    # factor recorded — the same object the Python loop passed, assembled
    # inside the trace instead of outside it.
    #
    # FIRST EXECUTED at step 2 on a real 4-process mesh, and that is what
    # this route is: LORRAX's slate cholesky consumer went through
    # ``to_jax_lower()`` and a replicated triangular solve, with "wiring
    # slate::trsm for the back-solve is a perf follow-up" written next to
    # it.  Residuals against the native reference, rtol 1e-12 NOT relaxed:
    # 3.8e-16 / 4.5e-16 CPU c128 per-q, 1.0e-15 / 6.9e-16 GPU c128 per-q,
    # 4.3e-16 / 4.4e-16 for the CPU / GPU 2x2 token round trips.
    raw, nb = token._factor
    n, SlateLowerL = token.n, mod.SlateLowerL

    def _build():
        def _two_trsms(carry, xs):
            raw_q, B_q = xs
            L = SlateLowerL(raw=raw_q, mesh=mesh, n=n, nb=nb)
            y = mod.distributed_trsm(L, B_q, mesh=mesh, op="N")
            return carry, mod.distributed_trsm(L, y, mesh=mesh, op="C")

        def _scanned(raw_, B_):
            _, X_ = jax.lax.scan(_two_trsms, None, (raw_, B_),
                                 unroll=BATCHED_SCAN_UNROLL)
            return X_

        return jax.jit(_scanned)

    key = scan_signature("cholesky", "slate", mesh, (raw, B), {},
                         extra=("slate.trsm", n, nb))
    X = cached_scan(key, _build)(raw, B)
    return ensure_sharding(X, NamedSharding(mesh, P(None, "x", "y")))
