"""Resolved linalg PLANS — resolve once, then just call.

A :class:`Plan` is the answer to "what will this op actually run, where
does its input have to live, and where does its output come out?",
computed ONCE::

    p = distrib_la.plan("eigh", mesh_xy, backend=cfg.eigh_backend, n=rank)
    if p.is_native:
        lam, R = jnp.linalg.eigh(A_batch)      # caller owns the fast path
    else:
        lam, R = p(A_tile)                     # or p.batched(A_stack)

Why this exists
---------------
Every FFI-linalg call site grew the same five lines around the one that
matters: ``resolve_backend(...)`` → compare against ``NATIVE`` → build a
``NamedSharding(mesh, P('x','y'))`` → ``device_put`` /
``with_sharding_constraint`` the operand into it → loop the batch axis and
``jnp.stack`` the results because this particular backend has no batched
entry point.  Five copies of that is five places for the FFI-adjacent
resharding to drift, and resharding around an FFI call is where this code
base has lost the most time (silent NaNs, a per-call recompile, a deleted
``_reshard_z``).

So the plan owns them.  :attr:`Plan.in_sharding` is the contract, written
down instead of re-derived; ``p(A)`` puts ``A`` there before calling;
``p.batched(A)`` is the SAME call for every backend whether or not the
library underneath happens to have a batched entry point.

What "batched" MEANS here
-------------------------
The batched surface is a ``lax.scan`` over this package's own single-matrix
operation.  That is the definition, not an implementation detail that
happens to be true today: ``p.batched`` is ``p`` under a scan, and a
backend that owns a stacked FFI entry point gets to substitute it
*underneath* that definition (:attr:`Plan.batched_route`), never beside it.
There is exactly one public batched surface and there will not be a second.

The reason to insist on it is that a scan is a place where a decision can
live.  A Python loop over ``nb`` matrices is ``nb`` separate calls the
compiler never sees together, and there is nowhere in it to put "run this
batch some other way".  A scan is one node, so the choice of HOW a batch
executes collapses to :attr:`Plan.batched_route`: the distributed scan, a
backend's stacked entry, or staged batch-axis movement around a device-local
native kernel.

TWO PHASES, and they stay two
-----------------------------
:func:`plan` is EAGER: it dlopens the library and calls
``jax.process_count()``.  What it returns is TRACE-SAFE.  A single-phase
API would have to lie about when it checked — only the platform and
handler guards can fire at resolve time, while operand dtype/rank/extent
are trace-time facts.  :attr:`Plan.native_fn` is the pure end of that
split: a closure containing no ``process_count``, no ``device_put`` and no
``dlopen``, so it can be built once and called inside somebody else's
``jit``.

What it deliberately does NOT do
--------------------------------
* **It does not change resolution.**  :func:`plan` calls
  :func:`distrib_la.resolve.resolve_backend` with the caller's arguments and
  stores the answer.  Route strings, ``auto`` policy and every guard are
  byte-identical to calling the resolver directly.
* **It does not own the caller's fused math.**  ``native`` cholesky /
  solve_lu are channel-policy routes (replicated dense factor, per-q ridged
  solve) that are not one call at all; ``p.is_native`` says "you own this
  one" and ``p(...)`` raises rather than pretending.  ``eigh`` is the
  exception — its native path IS one call, and the plan runs it.

Layout contract (the same for all three ops, and the reason one class
covers them): the FFI backends take **one tile spread over the whole
mesh**, ``P('x','y')``, and return their matrix-shaped outputs in that
same layout.  Stacked/batched forms carry a leading, unsharded batch
axis: ``P(None,'x','y')``.  Vector-shaped outputs (eigenvalues) are
REPLICATED.  Eigenvectors are COLUMNS.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la._result_check import deflate_zero_rows, native_eigh
from distrib_la.resolve import (NATIVE, NATIVE2D, OPS, backend_module,
                                mesh_key, resolve_backend)

__all__ = ["Plan", "plan", "StackRoute", "ensure_sharding", "BATCHED_SCAN_UNROLL",
           "BATCHED_ROUTES", "BATCHED_ROUTE_CHOICES",
           "BATCHED_ROUTE_DEFAULT", "ROUTE_SCAN",
           "ROUTE_BACKEND_BATCHED", "ROUTE_BATCH_RESHARD"]


#: ``unroll=`` for the batched scan — route ``ROUTE_SCAN`` below.
#:
#: ONE means the compiled module holds exactly ONE copy of the single-matrix
#: op and the XLA while loop runs it ``nb`` times.  That is the point of
#: making the batched surface a scan: a Python loop hands the compiler
#: ``nb`` separate calls (and, on a wrapper that builds its ``shard_map``
#: eagerly with no ``jax.jit`` around it, ``nb`` traces), while the scan
#: compiles once whatever ``nb`` is.
#:
#: It is a named constant and not a literal because raising it is the
#: obvious first thing to try if a backend's per-matrix launch latency ever
#: dominates its arithmetic: ``unroll=k`` buys ``nb/k`` loop trips for ``k``
#: copies of the op in the module.  Nobody has measured a shape where that
#: wins, so it stays 1 — changing it is a measurement, not a preference.
BATCHED_SCAN_UNROLL = 1

#: (a) ``lax.scan`` over this package's own single-matrix op. Selected by
#: explicit ``auto`` when the backend has no stacked entry.
ROUTE_SCAN = "scan"
#: (b) the backend's own stacked FFI entry point, where the library has one.
ROUTE_BACKEND_BATCHED = "backend_batched"
#: (c) staged batch-axis reshard + a device-local native JAX kernel.
ROUTE_BATCH_RESHARD = "batch_reshard"

#: Every route the toggle can name.  A vocabulary, like the backend names:
#: importable with no FFI ``.so``, so configuration code can enumerate the
#: routes without probing a vendor library or constructing a mesh.
BATCHED_ROUTES = (ROUTE_SCAN, ROUTE_BACKEND_BATCHED, ROUTE_BATCH_RESHARD)

#: Public construction-time grammar.  ``scan`` and ``backend_batched`` stay
#: implementation details selected by ``auto``; route (c) is the shipping
#: default and changes the memory/capacity tradeoff deliberately.
BATCHED_ROUTE_CHOICES = ("auto", ROUTE_BATCH_RESHARD)
#: Shipping route for array-returning batched operations. ``auto`` remains
#: a public explicit choice for the face-sharded provider/scan path.
BATCHED_ROUTE_DEFAULT = ROUTE_BATCH_RESHARD

#: ``jit``-compiled batched scans, one per (op, backend, mesh, signature).
#:
#: NOT an optimization to taste — a MEASURED requirement, and the same one
#: every FFI wrapper in this package already answers with a ``_JIT_CACHE``.
#: An eager ``shard_map`` re-traces on every call, and an eager ``lax.scan``
#: around one re-traces AND re-lowers the whole loop on every call.  Without
#: this cache the scan is CORRECT and SLOWER: Perlmutter CPU 2x2, nq=8 n=64,
#: the serial eigh route measured cold 0.106 s / warm 0.080 s eager against
#: the same scan as a compiled executable at 0.011 s, i.e. ~69 ms of pure
#: per-call retrace — worse than the Python loop it replaced (warm 0.024 s),
#: which paid nothing per call because each backend call hit the wrapper's
#: own cache.
#:
#: Keyed on :func:`distrib_la.mesh_key`, which is STRICTLY finer than the
#: ``(Px, Py)`` the backends key their MPI/NCCL contexts on, so this cache
#: cannot hold an executable whose baked-in context handle has moved on.
#: Finer than necessary can only cost an extra compile; coarser is the one
#: failure mode a cache key must not have.
_SCAN_CACHE: dict = {}


def scan_signature(op: str, backend: str, mesh: Mesh, ops, kwargs,
                   extra=()) -> int | None:
    """A hashable identity for a compiled batched scan, or ``None``.

    ``None`` means "this call cannot be keyed" — an unhashable keyword — and
    the caller then builds the scan without caching it.  Refusing to cache
    is always safe; guessing a key is not.
    """
    try:
        return hash((
            op, backend, mesh_key(mesh), BATCHED_SCAN_UNROLL, extra,
            tuple((tuple(x.shape), str(x.dtype)) for x in ops),
            tuple(sorted((k, repr(v)) for k, v in kwargs.items()))))
    except TypeError:
        return None


def cached_scan(key, build):
    """The compiled scan for ``key``, built once.  ``key=None`` skips the
    cache entirely and builds every time."""
    if key is None:
        return build()
    fn = _SCAN_CACHE.get(key)
    if fn is None:
        fn = _SCAN_CACHE[key] = build()
    return fn


class StackRoute(NamedTuple):
    """How :meth:`Plan.batched` runs one eigh stack ``(B, n, n)``.

    ``route`` is route (c) or the provider route; on route (c) every rank
    holds ``per_rank`` whole matrices at a time, over ``rounds`` slices of
    the stack, and the program that runs it needs ``program_bytes`` per rank
    (:func:`_stack_bytes`) against the caller's ``room``.
    """
    route: str
    per_rank: int = 0
    rounds: int = 1
    program_bytes: int | None = None
    room: int | None = None


#: Decided stack routes, one per (mesh, op, B, n, dtype, room), and the keys
#: :func:`new_stack_routes` has already reported.
_STACK_ROUTES: dict = {}
_REPORTED: set = set()


def new_stack_routes() -> list[str]:
    """One line per eigh-stack decision made since the last call, for a driver log.

    The service decides (:meth:`Plan.stack_route`); a driver prints these
    lines through its own reporter, once per (op, B, n, dtype, room).
    """
    lines = []
    for key, route in _STACK_ROUTES.items():
        if key not in _REPORTED:
            _REPORTED.add(key)
            lines.append(f"{key[1]} stack {key[2]} x {key[3]}^2 {key[4]}: {_describe_stack(route)}")
    return lines


@lru_cache(maxsize=None)
def _stack_bytes(op: str, mesh: Mesh, nb: int, n: int, dtype: str, rounds: int, room: int) -> int:
    """Per-rank device bytes route (c) adds for a face stack of ``nb`` matrices in ``rounds``.

    The compiled size of the program that runs (:func:`_reshard_stack_program`:
    the slices' exchanges, local eighs, inverse exchanges and the result
    check; outputs and temporaries, the caller's operand excluded) plus the
    local solver's runtime workspace for one slice, which cuSOLVER reports
    and the compiler does not count.
    """
    import numpy as np
    from distrib_la.resolve import mesh_platform
    from distrib_la.workspace import _vendor_query
    face = NamedSharding(mesh, P(None, "x", "y"))
    program = _reshard_stack_program(op, mesh, (nb, n, n), dtype, rounds, "admission",
                                     _gathered_fits(mesh, (nb, n, n), dtype))
    stats = program.lower(jax.ShapeDtypeStruct((nb, n, n), np.dtype(dtype), sharding=face)
                          ).compile().memory_analysis()
    if stats is None:
        raise RuntimeError("route (c) admission: the compiler returned no memory analysis")
    compiled = stats.output_size_in_bytes + stats.temp_size_in_bytes - stats.alias_size_in_bytes
    vendor = 0
    if mesh_platform(mesh) == "CUDA":
        ranks = int(mesh.shape["x"]) * int(mesh.shape["y"])
        per_rank = -(-(-(-nb // rounds)) // ranks)
        # One cuSOLVER workspace per whole matrix (4 n^2 elements at
        # complex128, the size jaxlib's syevBatched allocates at runtime).
        vendor = per_rank * (_vendor_query(0, "eigh", (n,), np.dtype(dtype).str)[0] + 4)
    return int(compiled + vendor)


def _gathered_fits(mesh: Mesh, shape, dtype) -> bool:
    """Whether an eigh's gathered retry fits every rank within
    GATHERED_EIGH_BYTES: the compiled size of the gathered program (outputs
    and temporaries) plus cuSOLVER's workspace for each whole matrix, since
    every rank solves all of them (7 n^2 elements per matrix, against the
    3 n^2 of the matrix, its vectors and one copy). XLA reserves the retry
    branch's temporaries in every program that holds it, taken or not, so a
    caller's room does not raise the limit."""
    import numpy as np
    limit = GATHERED_EIGH_BYTES
    return _agreed_gathered(mesh, tuple(int(v) for v in shape), np.dtype(dtype).name, limit)


@lru_cache(maxsize=None)
def _agreed_gathered(mesh: Mesh, shape: tuple, dtype: str, limit: int) -> bool:
    """:func:`_gathered_fits`, agreed over ranks (any rank's no wins), so every
    rank builds the same retry chain and its collectives (INVARIANTS 21)."""
    import numpy as np
    from distrib_la._collectives import agreed_minimum
    fits = (3 * np.dtype(dtype).itemsize * int(np.prod(shape)) <= limit
            and _gathered_bytes(mesh, shape, dtype) <= limit)
    return bool(agreed_minimum((int(fits),), tag="gathered eigh retry")[0])


@lru_cache(maxsize=None)
def _gathered_bytes(mesh: Mesh, shape: tuple, dtype: str) -> int:
    """Per-rank device bytes of :func:`_gathered_eigh` on a face operand of ``shape``."""
    import numpy as np
    from distrib_la.resolve import mesh_platform
    from distrib_la.workspace import _vendor_query
    face = NamedSharding(mesh, P(*([None] * (len(shape) - 2)), "x", "y"))
    stats = jax.jit(partial(_gathered_eigh, mesh=mesh)).lower(
        jax.ShapeDtypeStruct(shape, np.dtype(dtype), sharding=face)).compile().memory_analysis()
    compiled = stats.output_size_in_bytes + stats.temp_size_in_bytes - stats.alias_size_in_bytes
    if mesh_platform(mesh) != "CUDA":
        return int(compiled)
    matrices = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
    return int(compiled + matrices * (_vendor_query(0, "eigh", (shape[-1],), np.dtype(dtype).str)[0] + 4))


@lru_cache(maxsize=None)
def _reshard_stack_program(op: str, mesh: Mesh, shape: tuple, dtype: str, rounds: int, site: str,
                           gathered: bool):
    """The jitted route-(c) program for one face stack: ``rounds`` slices
    (:func:`distrib_la._batch_reshard.reshard_rounds_call`), and for an eigh
    the service's probe check on the Hermitian part of the input
    (:func:`_local_eigh_errors`). A failed check solves again shifted, then
    gathered when ``gathered`` (the distributed eigh's retry chain); a result
    no attempt repairs is NaN-poisoned and named (GATE
    distrib_la_result_check). ``normal_eigh`` (polar's right singular
    vectors) is not probe-checked here."""
    from distrib_la._batch_reshard import reshard_rounds_call
    from distrib_la._result_check import checked, shifted
    solve = partial(reshard_rounds_call, op, mesh, rounds=int(rounds))
    out = (NamedSharding(mesh, P()), NamedSharding(mesh, P(None, "x", "y")))
    if op == "normal_eigh":
        return jax.jit(solve, out_shardings=out)
    attempts = [solve, shifted(solve)]
    if gathered:
        attempts.append(partial(_gathered_eigh, mesh=mesh))
    return jax.jit(lambda a: checked("eigh", tuple(attempts), lambda r: _local_eigh_errors(a, *r),
                                     (a,), site=site), out_shardings=out)


def _local_eigh_errors(a, values, vectors):
    """:func:`distrib_la._result_check.eigh_errors` against Herm(a) = (a + a^H)/2,
    the matrix the local solver diagonalizes (``jnp.linalg.eigh`` symmetrizes
    its input). a^H V X is read as (X^H V^H a)^H, so the face-tiled stack is
    never transposed."""
    from distrib_la._result_check import _norm, probes
    x = probes(a.shape[-1], vectors.dtype)
    vx = vectors @ x
    left = jnp.conj(jnp.swapaxes(jnp.conj(jnp.swapaxes(vx, -1, -2)) @ a, -1, -2))
    residual = _norm(0.5 * (a @ vx + left) - vectors @ (values[..., :, None] * x)) / (
        jnp.maximum(_norm(a), jnp.finfo(values.dtype).tiny) * _norm(x))
    orthogonality = _norm(jnp.conj(jnp.swapaxes(vectors, -1, -2)) @ vx - x) / _norm(x)
    finite = jnp.all(jnp.isfinite(values))
    return (jnp.where(finite, jnp.max(residual), jnp.inf),
            jnp.where(finite, jnp.max(orthogonality), jnp.inf))


def ensure_sharding(x, sharding: NamedSharding):
    """Put ``x`` on ``sharding`` — THE one FFI-adjacent reshard helper.

    Traced (inside a ``jit``) → ``with_sharding_constraint``.  Concrete and
    already there → returned untouched, so a caller that already built the
    operand in the contract layout pays nothing.  Concrete and elsewhere →
    ``device_put``.

    Every FFI call site used to spell one of those three by hand and got to
    pick a different one; picking wrong is either a silent extra copy of an
    ``(n, n)`` c128 tile or, at P > 1, an operand-sharding error minutes
    into a run.
    """
    if isinstance(x, jax.core.Tracer):
        return jax.lax.with_sharding_constraint(x, sharding)
    have = getattr(x, "sharding", None)
    # Compare (mesh, spec), not the sharding objects: a jit output and a
    # hand-built NamedSharding of the same layout can differ in
    # ``memory_kind`` and compare unequal, which would turn "already in the
    # contract" into a pointless device_put of an (n, n) c128 tile.
    if (getattr(have, "spec", None) == sharding.spec
            and getattr(have, "mesh", None) == sharding.mesh):
        return x
    # Process-local for host/uncommitted operands (a global jax.Array is
    # handed straight to ``device_put`` inside the helper — a genuine
    # reshard).  Plain ``device_put`` of host numpy onto a multi-process
    # sharding fires JAX's hidden ``assert_equal`` all-gather at
    # P × x.nbytes — for an (n, n) c128 FFI operand, exactly the class of
    # silent cost this helper exists to prevent.
    from distrib_la._collectives import device_put_process_local
    return device_put_process_local(x, sharding)


def _is_batch_layout(A, mesh) -> bool:
    """A rank-3 stack at P(('x','y'), None, None) on a mesh with more than one rank."""
    if getattr(A, "ndim", 0) != 3 or int(mesh.shape["x"]) * int(mesh.shape["y"]) == 1:
        return False
    have = getattr(A, "sharding", None)
    return have is not None and have.is_equivalent_to(
        NamedSharding(mesh, P(("x", "y"), None, None)), 3)


def _eigh_columns(backend: str, lam, Q):
    """Normalise an FFI eigh result to TRUE column eigenvectors.

    ``A @ Q == Q @ diag(lam)``.  cuSOLVERMp's wrapper returns the raw
    device buffer, whose documented layout is the conjugate transpose of
    that; SLATE and ScaLAPACK already return columns.  This is the ONE
    place the difference is known — it used to live inside ``dispatch_eigh``
    only, so anything reaching a backend module directly had to remember it
    independently, and the serial dispatch path silently returned ROWS on
    the only platform that took it.

    RANK-AGNOSTIC: the transpose is on the LAST TWO axes, so one normaliser
    serves an ``(n, n)`` tile and a ``(nq, n, n)`` stack alike.  It has to
    be, because the SAME normaliser is the ``post`` of both routes in
    :attr:`Plan.batched_route`: the scan hands it one tile per iteration
    and a stacked entry hands it the whole stack.  The day cuSOLVERMp
    grows a ``batched_distributed_eigh``, one ``_IMPL`` row flips this
    function's operand from ``(n, n)`` to ``(nq, n, n)`` with no other
    edit, and the old ``.T`` (reverse ALL axes) would have silently
    returned ``(n, n, nq)``.
    """
    if backend == "cusolvermp":
        return lam, jnp.conj(jnp.swapaxes(Q, -1, -2))
    return lam, Q


#: Per-rank bytes a gathered local eigh retry may take, compiled program plus
#: cuSOLVER workspace (:func:`_gathered_fits`; n <= 3096 complex128).
GATHERED_EIGH_BYTES = 1 << 30


def _gathered_eigh(A, *, mesh):
    """Eigh of the whole matrix on every rank: the last retry of a checked distributed eigh."""
    whole = jax.lax.with_sharding_constraint(A, NamedSharding(mesh, P()))
    values, vectors = native_eigh(whole)
    face = NamedSharding(mesh, P(*([None] * (A.ndim - 2)), "x", "y"))
    return values, jax.lax.with_sharding_constraint(vectors, face)


#: (op, backend) → how to call it.
#:
#:   one         attribute implementing the SINGLE-tile form, or None
#:   many        attribute implementing the STACKED form, or None
#:   post        result normaliser, applied to both forms
#:   one_handle  ``one`` returns a library HANDLE rather than arrays
#:               (absent means it returns arrays)
#:
#: A missing ``many`` is filled in by scanning ``one`` — that is what
#: :meth:`Plan.batched` IS.  A missing ``one`` means the library only ever
#: factors a whole stack (one descriptor, one workspace), so there is no
#: single-tile call to scan and the stacked entry is the only route.
#:
#: ``cholesky`` is asymmetric on purpose: cuSOLVERMp's batched potrf returns
#: a HANDLE object carrying the block-cyclic geometry, while SLATE's
#: per-tile potrf returns a ``SlateLowerL`` handle of its own.  The plan
#: does not flatten that difference away — :func:`distrib_la.factor` is
#: where it is handled — it just stops every caller from re-deriving which
#: is which.
#:
#: ``one_handle`` is DECLARED and not discovered, and the difference is the
#: point.  The old loop-and-stack path found out by calling ``one`` once and
#: looking at what came back; a scan cannot, because a handle is not a
#: pytree and the failure would arrive from inside ``lax.scan`` as a type
#: error naming neither the op nor the backend.  Declaring it lets
#: :meth:`Plan.batched` refuse BEFORE it runs anything, with the sentence
#: that says what to call instead.  ``tests/test_distrib_la_shape_algebra``
#: cross-checks the flag against each wrapper's own return annotation, so
#: the declaration cannot quietly disagree with the code.
_IMPL: dict[tuple[str, str], dict[str, Any]] = {
    ("eigh", "cusolvermp"):     dict(one="distributed_eigh",
                                     many=None, post=_eigh_columns),
    ("eigh", "slate"):          dict(one="distributed_eigh",
                                     many=None, post=_eigh_columns),
    ("eigh", "scalapack"):      dict(one="distributed_eigh",
                                     many="batched_distributed_eigh",
                                     post=_eigh_columns),
    ("cholesky", "cusolvermp"): dict(one=None,
                                     many="batched_distributed_cholesky",
                                     post=None),
    ("cholesky", "slate"):      dict(one="distributed_cholesky",
                                     many=None, post=None,
                                     one_handle=True),
    ("solve_lu", "cusolvermp"): dict(one=None,
                                     many="batched_distributed_solve_lu",
                                     post=None),
    ("solve_lu", "scalapack"):  dict(one=None,
                                     many="batched_distributed_solve_lu",
                                     post=None),
    # native2d is pure JAX, but it routes through the same table: its
    # kernel is a single lax.map over the whole q axis, so it has a
    # stacked entry and no single-tile one, exactly like cuSOLVERMp's
    # batched potrf.  Nothing about calling it is special-cased.
    ("cholesky", NATIVE2D):     dict(one=None, many="cholesky", post=None),
}

#: Ops whose ``native`` path is a single JAX call the plan can run itself.
#: ``cholesky``/``solve_lu`` are not: their native routes are the
#: replicated dense factor / per-q ridged solve chosen by a channel policy
#: this module has no business duplicating.
_NATIVE_CALLABLE = {"eigh": lambda A, *a, **k: native_eigh(A)}

#: Which operands an op DONATES.  Declared per OP, not per backend: the
#: caller has to know whether its buffers survive the call before it knows
#: which library will run, and "it depends on the backend" is not an answer
#: a call site can act on.  ``solve_lu`` donates BOTH A and B (the LU
#: factors scribble over A; the solve is in-place on B); the eigh backends
#: donate nothing (``pXheevd`` destroys its input, so the handler stages
#: each q into scratch, and the cuSOLVERMp wrapper has neither
#: ``donate_argnums`` nor ``input_output_aliases`` on that path); cholesky
#: donates its operand on the cuSOLVERMp route and the declaration is the
#: conservative union.
DONATES: dict[str, tuple[int, ...]] = {
    "eigh":     (),
    "cholesky": (0,),
    "solve_lu": (0, 1),
}


@lru_cache(maxsize=None)
def _native2d_trace_fn(impl: Callable, mesh: Mesh) -> Callable:
    """Keep one trace-safe callable for equivalent native2d plans.

    Only the implementation and mesh are captured; extent and numerical
    operands remain traced. Reusing this callable lets outer cached kernels
    retain their identity across newly resolved plans.
    """
    return lambda A, **kw: impl(A, mesh=mesh, **kw)


def _describe_stack(route: StackRoute) -> str:
    """One phrase for a decided stack route (log line and :meth:`Plan.describe`)."""
    gb = lambda v: "n/a" if v is None else f"{v / 1e9:.2f} GB"
    if route.route == ROUTE_BATCH_RESHARD:
        return (f"{route.route}, {route.per_rank} whole matrix(es) per rank, "
                f"{route.rounds} round(s), compiled {gb(route.program_bytes)}/rank of room "
                f"{gb(route.room)}")
    if route.program_bytes is None:
        return f"{route.route} on the whole mesh (room {gb(route.room)})"
    return (f"{route.route} on the whole mesh ({route.per_rank} whole matrix(es) per rank "
            f"compile to {gb(route.program_bytes)} against room {gb(route.room)})")


@dataclass(frozen=True)
class Plan:
    """A resolved linear-algebra call: backend + layout contract + call.

    Construct with :func:`plan`; never instantiate directly (the
    constructor does no guard checking, :func:`plan` does).

    Attributes
    ----------
    op, requested, backend
        The op, what the caller asked for, and what
        :func:`~distrib_la.resolve.resolve_backend` returned.  ``backend``
        is ``'native'`` or a concrete backend name.
    mesh, n
        The ``('x','y')`` mesh, and the matrix extent if the caller pinned
        one (which also runs the divisibility guard at resolve time).
    in_sharding, batch_in_sharding
        WHERE an operand has to live for this plan: ``P('x','y')`` for a
        single tile, ``P(None,'x','y')`` for a stack.  ``None`` on an
        automatic native plan, which accepts any layout; an explicit
        batch-reshard native plan exposes the face shardings because staged
        movement starts from that contract.
    requested_batched_route
        The construction-time public selection, ``'auto'`` or
        ``'batch_reshard'``. :attr:`batched_route` is the resolved route.
    donates
        Which positional operands this OP donates — see :data:`DONATES`.
    budget_bytes
        Device bytes per rank the caller admits for one batched call, or
        ``None``. With ``'auto'`` requested and a budget given, the route is
        decided per stack by capacity (:meth:`route_for`): a stack whose
        per-rank whole matrices fit runs route (c), otherwise the provider.
    """

    op: str
    requested: str
    backend: str
    mesh: Mesh
    n: int | None
    in_sharding: NamedSharding | None
    batch_in_sharding: NamedSharding | None
    requested_batched_route: str = BATCHED_ROUTE_DEFAULT
    budget_bytes: int | None = None

    # ---- introspection -------------------------------------------------
    @property
    def is_native(self) -> bool:
        """True when the resolved backend is the native JAX path."""
        return self.backend == NATIVE

    @property
    def donates(self) -> tuple[int, ...]:
        """Positional operands this call donates (per OP, not per backend)."""
        return DONATES[self.op]

    @property
    def module(self):
        """The backend module.  Raises for a native plan — it has none."""
        return backend_module(self.backend)

    @property
    def batched_route(self) -> str:
        """HOW :meth:`batched` will execute a stack.  THE toggle.

        This property is the one place in the package that decides how a
        batch runs, and the only place a new way of running one may be
        added.  Everything else — :meth:`batched`, the dispatchers, every
        caller — reads the answer or ignores it; nothing re-derives it.

        (a) ``ROUTE_SCAN`` — ``lax.scan`` (:data:`BATCHED_SCAN_UNROLL`) over
            this plan's own single-matrix op. The fallback selected by an
            explicit ``auto`` request when no stacked entry exists.

        (b) ``ROUTE_BACKEND_BATCHED`` — the backend's own stacked FFI entry
            point, taken whenever the library has one.  ScaLAPACK's eigh
            loops ``q`` in C++ around ONE descriptor and ONE workspace, so
            an ``Nq``-matrix stack costs one collective-serialisation round
            instead of ``Nq``; cuSOLVERMp's potrf and both solve_lu entries
            are the same bargain.  That saving lives in C++ and no scan can
            recover it, which is why the route exists — but it is a
            backend-internal optimization BEHIND this interface, never a
            second surface a caller can see or has to know about.

        (c) ``ROUTE_BATCH_RESHARD`` — move the batch axis onto the mesh
            (the service-private staged exchange, kept inside this package
            so an installed service has no upward dependency:
            ``P(None,'x','y') → P(('x','y'),None,None)``, two single-mesh-
            axis ``all_to_all``s, because the one-step move is not a tile
            permutation and GSPMD silently degrades it to
            replicate-then-partition) and run the LOCAL jax kernel on each
            rank's own matrices, then move back through the literal inverse
            pair of ``all_to_all``s.  Eigh eigenvalues use a device
            ``all_gather`` to recover this service's replicated-vector
            contract; no output is gathered through the host.

            The reason it is worth a named route: for every matrix that
            fits on one device, the distributed libraries' cost IS their
            fixed per-call charge, which the native replicated eigh does
            not pay (the per-call rule: § "Distributed is a capacity route" in
            ``docs/services/distrib_la/backends.md``).
            Route (c) serves that whole regime with no distributed-library
            call at all, and it is how small-system linalg happens without
            a second, parallel API.

        ``plan(..., batched_route=...)`` is the one caller-facing selection:
        ``'batch_reshard'`` is the shipping default, while ``'auto'``
        preserves the historical provider/scan route. ``batched()`` also
        keeps a private ``_route``
        override solely for route-comparison gates.
        """
        if self.requested_batched_route not in BATCHED_ROUTE_CHOICES:
            raise ValueError(
                f"unknown requested batched route "
                f"{self.requested_batched_route!r} (known: "
                f"{'|'.join(BATCHED_ROUTE_CHOICES)})")
        if self.requested_batched_route == ROUTE_BATCH_RESHARD:
            return ROUTE_BATCH_RESHARD
        if self.is_native:
            # jnp.linalg.eigh owns a true stacked entry; the native
            # cholesky/solve plans reach Plan.batched only under explicit
            # route (c), because their auto implementation is caller-owned.
            return ROUTE_BACKEND_BATCHED
        if _IMPL[(self.op, self.backend)]["many"] is not None:
            return ROUTE_BACKEND_BATCHED
        return ROUTE_SCAN

    def route_for(self, shape, dtype) -> str:
        """The route :meth:`batched` takes for one operand stack (:meth:`stack_route`)."""
        return self.stack_route(shape, dtype).route

    def stack_route(self, shape, dtype, op: str = "eigh") -> StackRoute:
        """How :meth:`batched` runs one operand stack: the service's decision.

        :attr:`batched_route` is the answer for every stack except one case:
        an eigh plan built with ``budget_bytes`` (the caller's room per rank,
        beside its own live set) and no explicit route. Then capacity decides:
        the stack runs route (c), one or more whole matrices per rank, when
        the compiled program that runs it fits the room (:func:`_stack_bytes`,
        compiled <= room, never a formula); the stack
        is cut into as many equal slices (rounds) as that needs. When not even
        one whole matrix per rank fits, it runs on the whole mesh. The room is
        a caller value every rank shares and the choice is agreed over ranks
        (:meth:`_agreed_stack`), so every rank takes the same route and
        rounds. Every decision is
        listed by :meth:`describe`, and :func:`new_stack_routes` hands each to
        a driver log once.
        """
        static = self.batched_route
        shape = tuple(int(v) for v in shape)
        nb = shape[0] if len(shape) == 3 else 1
        ranks = int(self.mesh.shape["x"]) * int(self.mesh.shape["y"])
        if static == ROUTE_BATCH_RESHARD:
            return StackRoute(static, -(-nb // ranks))
        if (self.requested_batched_route != "auto" or self.budget_bytes is None
                or self.is_native or self.op != "eigh"):
            return StackRoute(static)
        if self.budget_bytes == 0:
            return StackRoute(static, room=0)
        import numpy as np
        n, dtype = shape[-1], np.dtype(dtype).name
        key = (mesh_key(self.mesh), op, nb, n, dtype, self.budget_bytes)
        decided = _STACK_ROUTES.get(key)
        if decided is None:
            decided = _STACK_ROUTES[key] = self._agreed_stack(op, nb, n, dtype)
        return decided

    def _agreed_stack(self, op, nb, n, dtype) -> StackRoute:
        """:meth:`_decide_stack` agreed over ranks: every rank takes the fewest
        whole matrices per rank any rank chose (0, the whole mesh, wins), so
        the route, the rounds and their collectives are the same everywhere
        (INVARIANTS 21). A compiled figure is read on each rank and may differ."""
        from distrib_la._collectives import agreed_minimum
        mine = self._decide_stack(op, nb, n, dtype)
        local = mine.per_rank if mine.route == ROUTE_BATCH_RESHARD else 0
        agreed, = agreed_minimum((local,), tag="eigh stack route")
        if agreed == local:
            return mine
        if agreed == 0:
            return StackRoute(self.batched_route, mine.per_rank, 1, mine.program_bytes, mine.room)
        ranks = int(self.mesh.shape["x"]) * int(self.mesh.shape["y"])
        rounds = -(-(-(-nb // ranks)) // agreed)
        return StackRoute(ROUTE_BATCH_RESHARD, -(-(-(-nb // rounds)) // ranks), rounds,
                          _stack_bytes(op, self.mesh, nb, n, dtype, rounds, int(self.budget_bytes)),
                          mine.room)

    def _decide_stack(self, op, nb, n, dtype) -> StackRoute:
        """Route (c) at the most whole matrices per rank whose program compiles within the room."""
        ranks = int(self.mesh.shape["x"]) * int(self.mesh.shape["y"])
        room, provider = int(self.budget_bytes), self.batched_route
        if n % int(self.mesh.shape["x"]) or n % int(self.mesh.shape["y"]) or room <= 0:
            return StackRoute(provider, room=room)
        per_rank, tried, compiled = -(-nb // ranks), 0, None
        while per_rank >= 1:
            rounds = -(-(-(-nb // ranks)) // per_rank)
            m = -(-nb // rounds)
            tried, compiled = -(-m // ranks), _stack_bytes(op, self.mesh, nb, n, dtype, rounds, room)
            if compiled <= room:
                return StackRoute(ROUTE_BATCH_RESHARD, tried, rounds, compiled, room)
            per_rank = min(per_rank - 1, per_rank * room // compiled)
        # The provider route; per_rank names the smallest slice that was compiled.
        return StackRoute(provider, tried, 1, compiled, room)

    @property
    def native_fn(self) -> Callable:
        """A PURE, TRACE-SAFE closure for this plan's math.

        Available on the pure-JAX backends only, and that restriction is
        the point:
        the FFI wrappers must call ``jax.process_count()`` and dlopen a
        library, so they cannot live inside somebody else's trace.  This
        closure contains none of those, so it can be built once at plan
        time and called inside a ``jit``/``shard_map``/``scan`` body.

        It exists for NEW code and for any existing site that measures
        neutral-or-better.  The eight fusion-critical sites in LORRAX keep
        their inlined ``jnp`` math: pure ``jnp``/``lax`` calls with no
        vendor dependency are not a quarantine violation, and splitting one
        of them once cost 10.1 GiB/device and killed a 16×A100 run.
        """
        if self.is_native:
            fn = _NATIVE_CALLABLE.get(self.op)
            if fn is not None:
                return fn
        elif self.backend == NATIVE2D:
            impl = getattr(self.module, _IMPL[(self.op, self.backend)]["many"])
            return _native2d_trace_fn(impl, self.mesh)
        raise NotImplementedError(
            f"native_fn is defined for the pure-JAX backends only; this "
            f"plan is {self.op}/{self.backend}.  An FFI backend's wrapper "
            f"is eager by construction (it dlopens a library and reads "
            f"jax.process_count()), so there is no trace-safe closure to "
            f"hand out — call the plan itself.")

    def describe(self) -> str:
        """One line for a run banner: what resolved, and to what geometry."""
        px, py = int(self.mesh.shape["x"]), int(self.mesh.shape["y"])
        where = ("native JAX, any layout" if self.is_native else
                 f"ONE tile over the {px}x{py} mesh at P('x','y')")
        n = "" if self.n is None else f", n={self.n}"
        line = (f"{self.op}: {self.requested!r} -> {self.backend} "
                f"({where}{n}); batched "
                f"{self.requested_batched_route!r} -> {self.batched_route}")
        if self.budget_bytes is None:
            return line
        stacks = [f"B={key[2]} n={key[3]}: {_describe_stack(route)}"
                  for key, route in _STACK_ROUTES.items()
                  if key[0] == mesh_key(self.mesh) and key[5] == self.budget_bytes
                  and self.n in (None, key[3])]
        return line + f"; room {self.budget_bytes / 1e9:.2f} GB/rank" + (
            "; stacks " + "; ".join(stacks) if stacks else "")

    # ---- calling -------------------------------------------------------
    def _entry(self, key: str):
        """The backend's ``one`` or ``many`` call and its result normaliser.

        An FFI eigh or LU solve comes back checked (:mod:`distrib_la._result_check`,
        ``post`` None), because the libraries fail silently: eigh with its zero
        rows deflated, retried shifted, in another layout and gathered, and
        refused by name if no attempt passes; an LU solve refused by name (its
        operands are consumed).
        """
        spec = _IMPL[(self.op, self.backend)]
        name = spec[key]
        call = None if name is None else getattr(self.module, name)
        if call is None or self.op not in ("eigh", "solve_lu"):
            return call, spec["post"]
        if self.op == "solve_lu":
            return self._checked_solve(call), None
        return self._checked_eigh(call, spec["post"]), None

    def _refuse_if_poisoned(self, out, A):
        """An eager FFI eigh or LU solve whose checks all failed refuses here, by name."""
        if self.op not in ("eigh", "solve_lu") or self.is_native or isinstance(A, jax.core.Tracer):
            return out
        from distrib_la._result_check import call_site, refuse_if_poisoned
        return refuse_if_poisoned(out, self.op, int(A.shape[-1]), call_site())

    def _checked_eigh(self, call, post):
        from distrib_la._result_check import call_site, checked_eigh, shifted
        backend = self.backend

        def safe(A, *, mesh, **kwargs):
            def solve(a, **extra):
                return post(backend, *call(a, mesh=mesh, **kwargs, **extra))
            # Zero rows leave the solver as distinct sentinels (deflate_zero_rows).
            # A failed check solves again: shifted (a near-zero cluster moved off
            # the origin), then in the other cuSOLVERMp layout with and without
            # the sentinels (themselves a cluster), then gathered.
            attempts = [deflate_zero_rows(solve), deflate_zero_rows(shifted(solve))]
            n = int(A.shape[-1])
            if backend == "cusolvermp":
                from distrib_la._cusolvermp import retry_block
                block = retry_block(n, int(mesh.shape["x"]))
                if block is not None:
                    attempts.append(deflate_zero_rows(partial(solve, block=block)))
                    attempts.append(partial(solve, block=block))
            # A gathered local solve, where its compiled program and the
            # solver's workspace fit GATHERED_EIGH_BYTES on every rank (agreed).
            if _gathered_fits(mesh, A.shape, A.dtype):
                attempts.append(partial(_gathered_eigh, mesh=mesh))
            return checked_eigh(attempts, A, site=call_site())
        return safe

    def _checked_solve(self, call):
        from distrib_la._result_check import call_site, checked, matrix_sketch, rhs_sketch, solve_errors

        def safe(A, B, *, mesh, **kwargs):
            # The sketch is taken before the call, which may consume A and B.
            sketch = (*matrix_sketch(A), *rhs_sketch(B))
            X = call(A, B, mesh=mesh, **kwargs)
            return checked("solve_lu", (lambda x: x,), lambda x: solve_errors(sketch, x),
                           (X,), site=call_site(), n=A.shape[-1])
        return safe

    def __call__(self, A, *args, **kwargs):
        """Run the op on ONE tile (no batch axis).

        ``A`` (and any further matrix operands, e.g. ``solve_lu``'s RHS)
        are moved to :attr:`in_sharding` first — see
        :func:`ensure_sharding`.  A native plan runs the in-tree JAX call
        where one exists (``eigh``) and raises otherwise, naming what owns
        that route.
        """
        if self.is_native:
            fn = _NATIVE_CALLABLE.get(self.op)
            if fn is None:
                raise NotImplementedError(
                    f"{self.op} resolved to the NATIVE backend, whose "
                    f"implementation is the caller's own channel-policy "
                    f"route (in LORRAX: the replicated dense factor / 2-D "
                    f"blocked shard_map / per-q ridged solve in "
                    f"isdf/core), not a single call.  Branch on "
                    f"plan.is_native and run it there.")
            return fn(A, *args, **kwargs)
        one, post = self._entry("one")
        if one is None:
            raise NotImplementedError(
                f"{self.op} backend {self.backend!r} has no single-tile "
                f"entry point — its FFI call factors a whole stack in one "
                f"go (one descriptor, one workspace).  Use plan.batched(); "
                f"a stack of one is a legal stack.")
        ops = [ensure_sharding(x, self.in_sharding) for x in (A, *args)]
        out = one(*ops, mesh=self.mesh, **kwargs)
        return self._refuse_if_poisoned(post(self.backend, *out) if post is not None else out, A)

    def batched(self, A, *args, _route: str | None = None, **kwargs):
        """Run the op on a STACK ``(nb, n, n)`` — uniform across backends.

        The same call for every backend, whatever the library underneath
        can and cannot do with a stack.  Operands are moved to
        :attr:`batch_in_sharding` first; :attr:`batched_route` decides how
        the batch executes and is the only thing that decides.

        The native ``eigh`` plan just forwards to ``jnp.linalg.eigh``,
        which is natively batched (and is why ``auto`` picks it).

        ``_route`` is a PRIVATE override, for the batched-vs-serial gate
        that has to run two routes over one set of operands and compare
        them.  Production code passes nothing.  It is private because a
        caller that picks a route has taken back the decision this whole
        module exists to make once — the same reason there is no
        ``backend=`` on a call and only on :func:`plan`.
        """
        if (self.op == "eigh" and _route is None and not isinstance(A, jax.core.Tracer)
                and _is_batch_layout(A, self.mesh)):
            # Whole matrices already live on their ranks: only the local kernel can
            # serve them, whatever the plan's route (no movement, one batched solve).
            from distrib_la._batch_reshard import batch_layout_eigh_call
            return batch_layout_eigh_call("eigh", self.mesh, A)
        route = self.route_for(A.shape, A.dtype) if _route is None else _route
        if route not in BATCHED_ROUTES:
            raise ValueError(
                f"unknown batched route {route!r} "
                f"(known: {'|'.join(BATCHED_ROUTES)})")

        # Route (c) deliberately runs the pure-JAX operation even when the
        # plan resolved an FFI backend.  Its layout contract still starts at
        # the same face sharding, including on an explicit native plan.
        if route == ROUTE_BATCH_RESHARD:
            if self.op == "eigh":
                self._reshard_options(kwargs)
                rounds = 1 if _route is not None else self.stack_route(A.shape, A.dtype).rounds
                return self._checked_reshard("eigh", A, rounds)
            return self._batch_reshard((A, *args), kwargs)

        if self.is_native:
            if route != ROUTE_BACKEND_BATCHED:
                raise NotImplementedError(
                    f"native {self.op} has no route {route!r}; its automatic "
                    f"batched entry is {ROUTE_BACKEND_BATCHED!r}, and the "
                    f"only public alternative is "
                    f"{ROUTE_BATCH_RESHARD!r}.")
            return self(A, *args, **kwargs)

        ops = tuple(ensure_sharding(x, self.batch_in_sharding)
                    for x in (A, *args))
        if route == ROUTE_BACKEND_BATCHED:
            many, post = self._entry("many")
            if many is None:
                raise NotImplementedError(
                    f"{self.op} backend {self.backend!r} has no stacked FFI "
                    f"entry point, so route {ROUTE_BACKEND_BATCHED!r} does "
                    f"not exist for it.  Its batched route is "
                    f"{self.batched_route!r}.")
            out = many(*ops, mesh=self.mesh, **kwargs)
            return self._refuse_if_poisoned(post(self.backend, *out) if post is not None else out, A)
        if route == ROUTE_SCAN:
            return self._refuse_if_poisoned(self._scan_over_single(ops, kwargs), A)
        raise AssertionError(f"unhandled batched route {route!r}")

    def reshard_stack(self, op: str, A):
        """Route (c) for a face stack: ``op`` is ``eigh``, ``checked_eigh`` or
        ``normal_eigh`` (polar's right singular vectors), in this plan's
        decided rounds (:meth:`stack_route`), checked as :meth:`_checked_reshard`."""
        return self._checked_reshard(op, A, self.stack_route(A.shape, A.dtype, op).rounds)

    def _checked_reshard(self, op: str, A, rounds: int):
        """Route (c) in ``rounds`` slices, checked (:func:`_reshard_stack_program`);
        an eager call whose checks all failed refuses here by name."""
        from distrib_la._batch_reshard import validate_batch_reshard_operands
        from distrib_la._result_check import call_site, refuse_if_poisoned
        validate_batch_reshard_operands(op if op != "normal_eigh" else "eigh", self.mesh, (A,))
        A = ensure_sharding(A, NamedSharding(self.mesh, P(None, "x", "y")))
        shape, site = tuple(int(v) for v in A.shape), call_site()
        out = _reshard_stack_program(op, self.mesh, shape, str(A.dtype), int(rounds), site,
                                     _gathered_fits(self.mesh, shape, A.dtype))(A)
        if op == "normal_eigh" or isinstance(A, jax.core.Tracer):
            return out
        return refuse_if_poisoned(out, "eigh", shape[-1], site)

    @staticmethod
    def _reshard_options(kwargs: dict):
        """Refuse keywords route (c) cannot honour (a block size and the
        vectors hint are distributed-library options; the eigh contract
        returns (W, Z) on every route)."""
        options = {k: v for k, v in kwargs.items() if k not in ("block_size", "compute_evecs")}
        if options:
            raise TypeError(
                f"batch_reshard eigh: keyword(s) {', '.join(sorted(options))} have no "
                f"native-JAX meaning")

    def _batch_reshard(self, ops: tuple, kwargs: dict):
        """Route (c): staged movement, local native op, staged inverse."""
        options = dict(kwargs)
        # A block size configures a distributed library descriptor.  The
        # local JAX kernel has no such descriptor, so retaining the keyword
        # would make a universal route flag fail only at call time.
        options.pop("block_size", None)
        if self.op == "eigh":
            # The public contract returns (W, Z) on every route.  Computing Z
            # even when a backend-specific hint says it may be ignored is the
            # only native spelling that preserves that contract.
            options.pop("compute_evecs", None)
        if options:
            raise TypeError(
                f"batch_reshard {self.op}: keyword(s) "
                f"{', '.join(sorted(options))} have no native-JAX meaning")

        from distrib_la._batch_reshard import (
            batch_reshard_call, validate_batch_reshard_operands,
        )
        # Shape refusals precede even the face placement, and therefore
        # precede every collective in the route.
        validate_batch_reshard_operands(self.op, self.mesh, ops)
        ops = tuple(ensure_sharding(x, self.batch_in_sharding) for x in ops)
        return batch_reshard_call(self.op, self.mesh, ops)

    def _scan_over_single(self, ops: tuple, kwargs: dict):
        """Route (a): ``lax.scan`` over this plan's own single-matrix call.

        The body is exactly what :meth:`__call__` does to one tile —
        :func:`ensure_sharding` onto :attr:`in_sharding`, the backend's
        single-matrix entry, then the per-backend result normaliser — so
        the two surfaces cannot drift apart by construction.  ``lax.scan``
        stacks the per-matrix outputs itself, preserving whatever pytree
        the single-matrix call returns (a bare array, or eigh's
        ``(W, Z)``).

        The scan is ``jax.jit``-ed and cached per signature.  That is not
        decoration: see :data:`_SCAN_CACHE` for the measurement that says
        an uncached eager scan is slower than the Python loop it replaces.
        """
        spec = _IMPL[(self.op, self.backend)]
        one, post = self._entry("one")
        if one is None:
            raise NotImplementedError(
                f"{self.op} backend {self.backend!r} has neither a stacked "
                f"FFI entry point nor a single-tile one to scan, which "
                f"should be unreachable: every row of the impl table has at "
                f"least one of them.")
        if spec.get("one_handle"):
            raise NotImplementedError(
                f"{self.op} backend {self.backend!r} has no batched entry "
                f"point and its single-tile entry returns a library HANDLE "
                f"rather than an array, so there is no stack for a scan to "
                f"build.  Use distrib_la.factor()/solve(), which carries "
                f"the handle in an opaque token instead of flattening it.")

        mesh, backend, tile = self.mesh, self.backend, self.in_sharding

        def _build():
            def _one_matrix(carry, tiles):
                out = one(*(ensure_sharding(t, tile) for t in tiles),
                          mesh=mesh, **kwargs)
                return carry, (post(backend, *out) if post is not None
                               else out)

            def _scanned(*stacks):
                # An EMPTY carry: nothing crosses from matrix q to matrix
                # q+1, and writing that down is the claim that the batch
                # axis is independent.
                _, stacked = jax.lax.scan(_one_matrix, None, stacks,
                                          unroll=BATCHED_SCAN_UNROLL)
                return stacked

            return jax.jit(_scanned)

        key = scan_signature(self.op, self.backend, mesh, ops, kwargs)
        return cached_scan(key, _build)(*ops)


def plan(op: str, mesh_xy: Mesh, *, backend: str = "auto",
         n: int | None = None,
         batched_route: str | None = None,
         budget_bytes: int | None = None) -> Plan:
    """Resolve ``op`` on ``mesh_xy`` ONCE and return the callable plan.

    Every guard (vocabulary, platform, known-broken combinations,
    compiled-capability probe, process coverage, mesh geometry,
    divisibility when ``n`` is given) fires HERE, before any work — this is
    exactly :func:`distrib_la.resolve.resolve_backend`, so the resolution is
    identical to calling it directly and the same ``ValueError`` /
    ``RuntimeError`` messages come out.

    Hoist the plan out of loops.  Resolution loads the FFI library and
    calls ``jax.process_count``; the first FFI call additionally builds a
    BLACS / cuSOLVERMp context and compiles an XLA module (1.4–2.7 s
    measured), and both are per-plan, amortised from call 2.

    Parameters
    ----------
    op
        One of :data:`distrib_la.resolve.OPS`.
    mesh_xy
        The ``('x','y')`` device mesh the op runs on.
    backend
        A name from ``BACKEND_CHOICES[op]``; ``'auto'`` by default.
    n
        Matrix extent, when known.  Passing it turns the divisibility rule
        into a resolve-time error instead of a call-time one.
    batched_route
        ``'batch_reshard'`` (the default without a budget) stages the batch
        over the mesh and runs the device-local native JAX operation.
        ``'auto'`` explicitly requests the backend-batched/scan choice. See
        :attr:`Plan.batched_route`. Leave it out with ``budget_bytes``.
    budget_bytes
        The caller's room per rank beside its own live set, the same on every
        rank (0: no room). Given without a route, the service decides each
        eigh stack by capacity (:meth:`Plan.stack_route`): whole matrices per
        rank (route (c), in rounds) when the compiled program that runs it
        fits the room, else the whole-mesh provider.

    There is deliberately NO ``batched=`` flag.  The design sketch carried
    one; it would have changed nothing about resolution (both shardings are
    on every plan and :meth:`Plan.batched` is always available), so it could
    only be a second way to say what the call already says — which is the
    kind of concept this extraction exists to remove, not add.
    """
    if op not in OPS:
        raise ValueError(f"unknown linalg op {op!r} (known: {'|'.join(OPS)})")
    if batched_route is None:
        batched_route = "auto" if budget_bytes is not None else BATCHED_ROUTE_DEFAULT
    batched_route = str(batched_route).strip().lower()
    if batched_route not in BATCHED_ROUTE_CHOICES:
        raise ValueError(
            f"unknown batched route {batched_route!r} (known: "
            f"{'|'.join(BATCHED_ROUTE_CHOICES)})")
    resolved = resolve_backend(op, backend, mesh_xy, n=n)
    ffi = resolved != NATIVE
    if budget_bytes is not None:
        import operator
        budget_bytes = operator.index(budget_bytes)
        if budget_bytes < 0:
            raise ValueError(f"budget_bytes must be non-negative, got {budget_bytes}")
    face = ffi or batched_route == ROUTE_BATCH_RESHARD
    tile = NamedSharding(mesh_xy, P("x", "y")) if face else None
    stack = NamedSharding(mesh_xy, P(None, "x", "y")) if face else None
    if batched_route == ROUTE_BATCH_RESHARD or budget_bytes is not None:
        # Eager plan construction is the legal place for runtime setup; the
        # returned Plan.batched remains trace-safe.
        from distrib_la._collectives import warm_mesh_cliques
        warm_mesh_cliques(mesh_xy)
    return Plan(op=op, requested=str(backend), backend=resolved,
                mesh=mesh_xy, n=None if n is None else int(n),
                in_sharding=tile, batch_in_sharding=stack,
                requested_batched_route=batched_route,
                budget_bytes=budget_bytes)
