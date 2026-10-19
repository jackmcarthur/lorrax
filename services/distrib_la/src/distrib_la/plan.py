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

from dataclasses import dataclass, field
from functools import lru_cache, partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import math
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la._result_check import (_flag, _notices, deflate_zero_rows, in_checked_retry, native_eigh,
                                      traced_phase)
from distrib_la.resolve import (NATIVE, NATIVE2D, ONE_DEVICE, OPS, backend_module,
                                mesh_key, mesh_platform, resolve_backend,
                                resolve_eigh_capacity, size_route_note)

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


class RoomExhausted(NamedTuple):
    """A room of 0 that a caller KNOWS: its stage was admitted over budget.

    The shared-pole ledger warns and continues (owner ruling 2026-10-01), so an eigh
    inside such a stage has no budgeted room. The whole mesh is then the
    minimum-footprint route: :meth:`Plan.stack_route` takes it and prints one
    warning line, and route (c) stays fail-closed. A plan given room 0 WITHOUT
    this (a missing room) is refused when it is built (``GATE distrib_la_room``).
    Pass it as ``budget_bytes``; it counts as 0 bytes.
    """
    stage: str             #: the admitted stage that left no room
    priced_bytes: int      #: its admitted aggregate per rank
    available_bytes: int   #: the device budget per rank it was admitted against

    def __index__(self):
        return 0


def room_or_exhausted(room, *, stage, priced_bytes, available_bytes):
    """``room`` when positive, else :class:`RoomExhausted` naming ``stage``."""
    room = int(room)
    if room > 0:
        return room
    return RoomExhausted(str(stage), int(priced_bytes), int(available_bytes))


def _plan_caller(depth=3):
    """The first frames outside distrib_la that built a plan: 'file:line function' <- ..."""
    import inspect
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for frame in inspect.stack()[2:]:
        path = os.path.abspath(frame.filename)
        if path.startswith(here) or os.path.basename(path) in ("dataclasses.py", "functools.py", "<string>"):
            continue
        out.append(f"{os.path.relpath(path)}:{frame.lineno} {frame.function}")
        if len(out) == depth:
            break
    return " <- ".join(out) or "?"


class StackRoute(NamedTuple):
    """How :meth:`Plan.batched` runs one eigh stack ``(B, n, n)``.

    ``route`` is route (c) or the provider route; on route (c) every rank
    holds ``per_rank`` whole matrices at a time, over ``rounds`` slices of
    the stack, and the program that runs it needs ``program_bytes`` per rank
    (:func:`_stack_price`, from the shapes) against the caller's ``room``.
    """
    route: str
    per_rank: int = 0
    rounds: int = 1
    program_bytes: int | None = None
    room: int | None = None


#: Decided stack routes, one per (mesh, op, B, n, dtype, room, phase, backend,
#: route), and the keys
#: :func:`new_stack_routes` has already reported.
_STACK_ROUTES: dict = {}
_REPORTED: set = set()


def new_stack_routes() -> list[str]:
    """One line per eigh-stack decision made since the last call, for a driver log.

    The service decides (:meth:`Plan.stack_route`); a driver prints these
    lines through its own reporter, once per (op, B, n, dtype, room).  The
    ``distributed`` size route's decisions (resolve guard 1b) ride the same
    channel, once per (op, leg, mesh, n), as ``"<what>: <backend>, <reason>"``.
    """
    from distrib_la import resolve as _resolve
    lines = []
    for key, line in _resolve._SIZE_DECISIONS.items():
        if ("size",) + key not in _REPORTED:
            _REPORTED.add(("size",) + key)
            lines.append(line)
    for key, route in _STACK_ROUTES.items():
        if key not in _REPORTED:
            _REPORTED.add(key)
            where = "first attempt" if key[6] == "first" else "whole chain"
            lines.append(f"{key[1]} stack {key[2]} x {key[3]}^2 {key[4]} ({where}): "
                         f"{_describe_stack(route)}")
    return lines


#: What a route-(c) program reserves, by caller. An eager call's first program,
#: and a traced one inside distrib_la.checked_program's first program, holds
#: the first attempt and its check ("first"); an eager call's retries run as a
#: separate program ("retry") only when that check failed. Any other traced
#: caller's program holds the whole checked chain ("all": first attempt,
#: retries, refusal); a checked_program's own whole-chain program runs every
#: stack on the whole mesh (Plan.stack_route).
PHASES = ("all", "first", "retry")


def _stack_price(op, mesh, nb, n, dtype, rounds, gathered=False) -> int:
    """Per-rank device bytes of a route-(c) stack program in ``rounds``, priced from the shapes.

    :data:`BATCH_EIGH_TILES` n^2 per whole matrix a rank holds at a time (the
    first attempt, its exchanges, the local solve with its workspace and the
    vectors, as :func:`distrib_la.eigh_stack_bytes` prices it), plus the
    checked chain's temporaries (:data:`CHECKED_EIGH_TILES`) and its gathered
    retry where admitted. The same on every rank: nothing is compiled to be
    measured and no size is exchanged.
    """
    ranks = int(mesh.shape["x"]) * int(mesh.shape["y"])
    per_rank = -(-(-(-nb // rounds)) // ranks)
    item = np.dtype(dtype).itemsize
    whole = BATCH_EIGH_TILES * n * n * item
    checked = CHECKED_EIGH_TILES * (n * n // ranks) * item
    if op != "normal_eigh" and gathered:
        checked += _gathered_bytes((n, n), dtype)
    return per_rank * whole + checked


def _mesh_price(n, dtype, ranks) -> int:
    """Per-rank bytes of one n x n eigh on the whole mesh, from the shape: the checked chain's
    temporaries (:data:`CHECKED_EIGH_TILES`), the vectors and the provider's workspace (one
    n^2/P tile each; cuSOLVERMp 0.9.1's aligned workspace is 1.1 n^2/P at 4x4, n 100000,
    A100; other providers unmeasured)."""
    return (CHECKED_EIGH_TILES + 2) * -(-n * n * np.dtype(dtype).itemsize // ranks)


#: Measured eigh walls by platform, complex128: CUDA = A100-SXM4-80GB, ``dla_sweep`` P4 2x2 (the
#: rows behind resolve.ONE_DEVICE_BY_ROOM): seconds per matrix on one device (jaxlib cuSOLVER,
#: one rank's batch) and per call on the whole mesh (cuSOLVERMp 0.9.1): n 312 x64 0.022 / 1.97 s,
#: 8192 x4 2.05 / 6.53 s, 16384 x4 13.7 / 29.0 s. A platform without a row (ROCm: unmeasured)
#: decides by memory alone (route (c) where it fits). float64 takes the complex128 row.
_EIGH_SECONDS = {"CUDA": {"one": ((312, 0.022 / 16), (8192, 2.05), (16384, 13.7)),
                          "mesh": ((312, 1.97 / 64), (8192, 6.53 / 4), (16384, 29.0 / 4))}}


def _eigh_seconds(kind, n, ranks=4, platform="CUDA"):
    """Expected seconds of one n x n eigh, ``kind`` 'one' (one device) or 'mesh' (``ranks``), or
    None without a measured row: the points log-log interpolated, n^3 past the last, the fixed
    cost below the first. On P <= 4 the mesh's part above its fixed cost scales as 4/P. On P > 4
    the mesh is held at its P4 time up to the last measured n (16384; P16 is unmeasured there,
    open: a P16 sweep at n 2048-8192), and past it only the growth beyond t(16384) scales as 4/P,
    so the curve is continuous (n 100000 at P16: 417 s predicted, 312 s measured, eighmem
    2026-10-18)."""
    pts = _EIGH_SECONDS.get(str(platform), {}).get(kind)
    if pts is None:
        return None
    t = pts[-1][1] * (n / pts[-1][0]) ** 3 if n > pts[-1][0] else pts[0][1]
    for (a, ta), (b, tb) in zip(pts, pts[1:]):
        if a < n <= b:
            t = ta * (tb / ta) ** (math.log(n / a) / math.log(b / a))
    if kind == "one":
        return t
    if ranks > 4:
        return t if n <= pts[-1][0] else pts[-1][1] + (t - pts[-1][1]) * 4 / ranks
    return pts[0][1] + (t - pts[0][1]) * 4 / ranks


def place(op, mesh, nb, n, dtype, room, provider, backend=None) -> StackRoute:
    """THE placement of an eigh stack (nb, n, n) in ``room`` bytes per rank: the faster of the
    routes whose price fits, priced from the shapes only (no vendor query) and timed by
    :func:`_eigh_seconds`, so every rank decides alike with no exchange.

    Route (c) at the most whole matrices per rank whose program (:func:`_stack_price`) fits,
    ceil(nb/P) one-device solves; the whole mesh (:func:`_mesh_price`), nb mesh solves. A
    single large matrix goes to the mesh (route (c) pads it to every rank and solves it on
    one); a small stack stays whole per rank. A ``one_device`` plan's fallback holds one whole
    matrix per rank (resolve's size route), priced so. :meth:`Plan.stack_route` refuses by
    name a fallback over ``room`` (no route fits). Route (c) never costs less than the mesh
    (8 n^2 + 7 n^2/P against 9 n^2/P), so a mesh over the room has no route (c) under it.
    """
    x, y = int(mesh.shape["x"]), int(mesh.shape["y"])
    ranks, gathered = x * y, _gathered_admitted((n, n), dtype, room)
    platform = mesh_platform(mesh)
    whole = StackRoute(provider, 0, 1, (BATCH_EIGH_TILES * n * n * np.dtype(dtype).itemsize
                                        if backend == ONE_DEVICE else _mesh_price(n, dtype, ranks)), room)
    if n % x == 0 and n % y == 0:
        per_rank = -(-nb // ranks)
        while per_rank >= 1:
            rounds = -(-(-(-nb // ranks)) // per_rank)
            tried = -(-(-(-nb // rounds)) // ranks)
            price = _stack_price(op, mesh, nb, n, dtype, rounds, gathered)
            if price <= room:
                one, on_mesh = (_eigh_seconds(k, n, ranks, platform) for k in ("one", "mesh"))
                if one is None or -(-nb // ranks) * one <= nb * on_mesh:
                    return StackRoute(ROUTE_BATCH_RESHARD, tried, rounds, price, room)
                break
            per_rank = min(per_rank - 1, max(1, per_rank * room // price))
    return whole


def _gathered_admitted(shape, dtype, room=None) -> bool:
    """Whether an eigh's gathered retry joins its chain: the matrices, their
    vectors and the solver's copy (GATHERED_EIGH_FACTOR n^2 elements each)
    within GATHERED_EIGH_BYTES, or with the local heevd/syevd workspace within the
    plan's room (its ``budget_bytes``; the programs that hold the retry are priced
    in it: :func:`_stack_price`, ``distrib_la.workspace``). Shapes and a caller's
    agreed room: the same on every rank. cuSOLVERMp 0.9.1 left n 30000 (run30k) with
    no correct attempt short of this one."""
    small = GATHERED_EIGH_FACTOR * np.dtype(dtype).itemsize * int(np.prod(tuple(shape)))
    return small <= GATHERED_EIGH_BYTES or (room is not None and _gathered_bytes(shape, dtype) <= room)


def local_eigh_workspace(n, dtype, platform) -> int:
    """Device bytes of one local n x n eigh's workspace (``lxkit.memory.vendor``: a remembered
    query, else the leg's measured row -- cusolverDnXsyevd = 2.0 x LAPACK heevd on CUDA, LAPACK
    itself on the host).  No vendor query is issued while planning (a query at trace time could
    create a handle mid-run)."""
    from lxkit import leg_for_platform, memory
    cuda = platform == "CUDA"
    return int(memory.vendor(leg_for_platform("CUDA" if cuda else "cpu"), "cusolverdn" if cuda
                             else "lapack", "heevd", int(n), 1, np.dtype(dtype).itemsize).fixed)


def _gathered_bytes(shape, dtype) -> int:
    """Per-rank bytes the gathered retry reserves: GATHERED_EIGH_FACTOR n^2 per matrix and the
    local eigh workspace (:func:`local_eigh_workspace`)."""
    from lxkit import device_vendor
    item, elems, n = np.dtype(dtype).itemsize, int(np.prod(tuple(shape))), int(shape[-1])
    platform = "CUDA" if device_vendor(jax.devices()[0]) == "cuda" else "other"
    return GATHERED_EIGH_FACTOR * item * elems + elems // (n * n) * local_eigh_workspace(n, dtype, platform)


@lru_cache(maxsize=None)
def _reshard_stack_program(op: str, mesh: Mesh, shape: tuple, dtype: str, rounds: int, phase: str, cut=None,
                           gathered=False):
    """The jitted route-(c) program for one face stack and ``phase`` (:data:`PHASES`).

    ``rounds`` slices (:func:`distrib_la._batch_reshard.reshard_rounds_call`),
    and for an eigh the service's probe check on the Hermitian part of the
    input (:func:`_local_eigh_errors`) with the distributed eigh's chain:
    the first attempt, then shifted slice by slice (vectors
    re-orthonormalized by each slice's GEMMs), then gathered when admitted. Returns
    ``((values, vectors), failed)``; a result no attempt repairs is
    NaN-poisoned, ``failed`` set and, outside ``"first"``, named (GATE
    distrib_la_result_check). ``normal_eigh`` (polar's right singular
    vectors) returns ``(values, vectors)`` and is not probe-checked here.
    """
    from distrib_la._batch_reshard import reshard_rounds_call
    from distrib_la._result_check import checked, eigh_layout
    solve = partial(reshard_rounds_call, op, mesh, rounds=int(rounds))
    replicated = NamedSharding(mesh, P())
    out = (replicated, NamedSharding(mesh, P(None, "x", "y")))
    if op == "normal_eigh":
        return jax.jit(solve, out_shardings=out)
    attempts = [solve, partial(reshard_rounds_call, op, mesh, rounds=int(rounds), shifted=True)]
    if gathered:
        attempts.append(partial(_gathered_eigh, mesh=mesh))
    attempts = {"first": attempts[:1], "retry": attempts[1:], "all": attempts}[phase]
    return jax.jit(lambda a: checked("eigh", tuple(attempts), lambda r: _local_eigh_errors(a, *r, mesh=mesh, cut=cut),
                                     (a,), constrain=eigh_layout(mesh, 3),
                                     final=phase != "first"),
                   out_shardings=(out, replicated))


def _local_eigh_errors(a, values, vectors, *, mesh=None, cut=None):
    """:func:`distrib_la._result_check.eigh_errors` against Herm(a) = (a + a^H)/2,
    the matrix the local solver diagonalizes (``jnp.linalg.eigh`` symmetrizes
    its input). a^H V X is read as (X^H V^H a)^H, so the face-tiled stack is
    never transposed."""
    from distrib_la._result_check import _all_ranks, _norm, used_probes
    x, used = used_probes(values, vectors.dtype, cut)
    vx = vectors @ x
    left = jnp.conj(jnp.swapaxes(jnp.conj(jnp.swapaxes(vx, -1, -2)) @ a, -1, -2))
    from distrib_la._result_check import _face, _rows
    nd = a.ndim
    norm_x = jnp.maximum(_norm(x), jnp.finfo(values.dtype).tiny)
    gram = jnp.conj(jnp.swapaxes(vectors, -1, -2)) @ vx - x
    residual = _norm(0.5 * (a @ vx + left) - vectors @ (values[..., :, None] * x), mesh, _rows(nd)) / (
        jnp.maximum(_norm(a, mesh, _face(nd)), jnp.finfo(values.dtype).tiny) * norm_x)
    orthogonality = _norm(gram if used is None else jnp.where(used, gram, 0), mesh, _rows(nd)) / norm_x
    finite = _all_ranks(jnp.all(jnp.isfinite(values)), mesh)
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


#: Per-rank bytes a gathered local eigh retry may take when the plan has no
#: budget, and the n x n element copies it holds per rank. XLA reserves the
#: retry inside every program that contains the eigh, executed or not:
#: 1.75 n^2 per rank measured (memory_analysis, n 3328, P16), so the cap is
#: small (n <= 1448 complex128) and planners price it (workspace.py).
GATHERED_EIGH_BYTES = 64 << 20
GATHERED_EIGH_FACTOR = 2
#: The checked eigh chain's own XLA temporaries per rank, in units of
#: n^2 / P elements (deflation, probes, retries; measured 7.0-7.1 at n 3328
#: and 9152, P16, against 1-2 for a bare deflated eigh).
CHECKED_EIGH_TILES = 7
#: What route (c) adds per rank for each whole n x n matrix the rank holds, in
#: n^2 elements: the staged exchanges, the local solve with its workspace, the
#: vectors and the first attempt's check, the operand excluded. Compiled size
#: plus the vendor query on A100 (bundle B11, one or two matrices per rank):
#: 7.96-7.98 at P64 for n 6144-18432, 7.9-8.0 at P4 for n 984-7936.
BATCH_EIGH_TILES = 8


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
#: The agreed size-route costs of a ``distributed`` plan resolved to ``one_device``, one per
#: room-free call signature (:func:`_measured_rounds`' on-disk key; None where no library
#: resolves).  The room is applied per call: a new room is arithmetic, with no exchange, no
#: disk read and no compile.
_MEASURED: dict = {}
#: Library plans of :meth:`Plan._library_plan`, one per (op, mesh, n, route, library).
_LIBRARY_PLANS: dict = {}


def _program_bytes(fn, abstract) -> int | None:
    """Per-device bytes a compiled program allocates beside its arguments: XLA's
    ``memory_analysis`` temp + output, less the outputs aliased to arguments.  None where the
    backend reports no analysis."""
    stats = jax.jit(fn).lower(*abstract).compile().memory_analysis()
    if stats is None:
        return None
    return (int(stats.temp_size_in_bytes) + int(stats.output_size_in_bytes)
            - int(getattr(stats, "alias_size_in_bytes", 0) or 0))


def _library_bytes(p, abstract, single: bool) -> int | None:
    """The leg library program's compiled per-device bytes on these operands, or None where no
    library resolves on this host."""
    try:
        lib = p._library_plan()
    except (RuntimeError, ValueError):
        return None
    return _program_bytes((lambda *o: lib(*o)) if single else (lambda *o: lib.batched(*o)), abstract)


_LIBRARY_BYTES_DEFAULT = _library_bytes


def _round_candidates(nb: int, ranks: int) -> list:
    """Rounds tried in order: 1, then doubling, up to one matrix per rank per round."""
    last = max(1, -(-int(nb) // int(ranks)))
    out, r = [1], 2
    while r < last:
        out.append(r)
        r *= 2
    if last > 1:
        out.append(last)
    return out


@lru_cache(maxsize=1)
def _measured_code_hash() -> str:
    """The distrib_la sources the measured programs are built from (and the jax build): a cached
    cost is reused only by the same code."""
    import hashlib
    import os
    import jaxlib
    h = hashlib.sha256(f"{jax.__version__}/{jaxlib.__version__}".encode())
    here = os.path.dirname(os.path.abspath(__file__))
    for name in sorted(os.listdir(here)):
        if name.endswith(".py"):
            with open(os.path.join(here, name), "rb") as f:
                h.update(name.encode() + f.read())
    return h.hexdigest()[:16]


def native_closure_identity(platform: str) -> str:
    """The native closure the measured programs run on: the enforced dependency
    closure identity of the platform's FFI leg
    (:func:`lxkit.native_provider.native_closure_identity` -- every library the
    leg maps, by path and content hash, verified at start-up), or ``"none"``
    where no leg of that platform is open.  A cost measured against any other
    library set is another key.  Not cached here: lxkit computes it once, at
    enforcement, and a ``"none"`` asked before the leg opens must not stick."""
    from lxkit import native_provider
    try:
        return native_provider.native_closure_identity(_lxkit_platform(platform))
    except native_provider.ClosureNotEnforced:
        return "none"


def _lxkit_platform(platform: str) -> str:
    """distrib_la's platform name -> the lxkit.legs.LEGS key."""
    from lxkit import LEGS
    if platform in LEGS:
        return platform
    for key, spec in LEGS.items():
        if platform.lower() in (key.lower(), str(getattr(spec, "vendor", "")).lower()):
            return key
    return platform


def _measured_cache_dir() -> str:
    import os
    from lxkit import user_cache_dir
    return os.path.join(str(user_cache_dir("distrib_la")), "measured_cost")


def _measured_cache_file(disk_key: str) -> str:
    import hashlib
    import os
    return os.path.join(_measured_cache_dir(), hashlib.sha256(disk_key.encode()).hexdigest()[:32] + ".json")


def _payload_digest(disk_key: str, value) -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps([disk_key, value], sort_keys=True).encode()).hexdigest()


def _valid_costs(value, measured, need_library: bool = True) -> bool:
    """A cached figure set is used only whole and well-typed: the library's bytes where the
    incumbent is the library, and every measured round's largest and smallest per-rank bytes
    (an int each); anything else is a miss and is re-measured."""
    if not isinstance(value, dict):
        return False
    if need_library and not isinstance(value.get("library"), int):
        return False
    want = {str(r) for r in measured}
    for name in ("rounds", "rounds_min"):
        rounds = value.get(name)
        if not (isinstance(rounds, dict) and set(rounds) == want
                and all(isinstance(v, int) and v >= 0 for v in rounds.values())):
            return False
    return True


def _measured_cache_get(disk_key: str, measured, need_library: bool = True):
    """This signature's agreed costs from its own file, validated (key, digest, types); None on
    a miss, a stale code hash (another key), a partial or corrupt file: never a mis-route."""
    import json
    try:
        with open(_measured_cache_file(disk_key)) as f:
            entry = json.load(f)
    except (OSError, ValueError):
        return None
    if (not isinstance(entry, dict) or entry.get("key") != disk_key
            or entry.get("digest") != _payload_digest(disk_key, entry.get("value"))
            or not _valid_costs(entry.get("value"), measured, need_library)):
        return None
    return entry["value"]


def _measured_cache_put(disk_key: str, value) -> None:
    """One writer per key (process 0, after the agreement), published atomically: written to a
    private temp file, then os.replace'd over the key's file; a reader sees the old file or the
    whole new one."""
    import json
    import os
    if int(jax.process_index()) != 0:
        return
    path = _measured_cache_file(disk_key)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.{os.getpid()}.{os.urandom(4).hex()}.tmp"
        with open(tmp, "w") as f:
            json.dump({"key": disk_key, "value": value,
                       "digest": _payload_digest(disk_key, value)}, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError:
        pass


#: Agreements per signature hash in this process (every rank traces in the same order).
_AGREE_GEN: dict = {}


def _exchange(tag: str, payload: dict) -> list:
    """Every process's ``payload`` (JSON), in process order, through the coordination service's
    key-value store: a host exchange, no device work and no device sync.  One process: itself."""
    import json
    if jax.process_count() == 1:
        return [payload]
    from jax._src.distributed import global_state
    from distrib_la._collectives import _wait_for_bootstrap
    client = global_state.client
    gen = _AGREE_GEN[tag] = _AGREE_GEN.get(tag, -1) + 1
    base = f"distrib_la/measured/{tag}/{gen}"
    me = int(jax.process_index())
    client.key_value_set(f"{base}/{me}", json.dumps(payload))
    return [json.loads(_wait_for_bootstrap(client, f"{base}/{r}", me,
                                           "every rank's measured route costs"))
            for r in range(jax.process_count())]


def _agreed(values: list, need_library: bool = True):
    """The conservative agreement of every rank's figures: the SMALLEST library bytes, and per
    round count the LARGEST one_device bytes (``rounds``: what a candidate is charged) and the
    SMALLEST (``rounds_min``: what a one_device incumbent is credited with), so the route
    decided from them is the same on every rank and never raises any rank's peak above the
    incumbent's.  Where the incumbent is the library: None if no rank has one (all agree on
    that), and a mix is refused by name on every rank alike."""
    libs = [v.get("library") for v in values]
    if need_library:
        if all(l is None for l in libs):
            return None
        if any(l is None for l in libs):
            raise RuntimeError("distrib_la measured size route: the leg library resolves on some "
                               "processes and not others; refusing rather than route divergently")
    rounds, low = {}, {}
    for v in values:
        for r, c in v["rounds"].items():
            rounds[r] = None if c is None or rounds.get(r, 0) is None else max(rounds.get(r, 0), int(c))
        for r, c in (v.get("rounds_min") or v["rounds"]).items():
            low[r] = (None if c is None or (r in low and low[r] is None)
                      else min(low.get(r, int(c)), int(c)))
    return {"library": min(int(l) for l in libs) if need_library else None,
            "rounds": rounds, "rounds_min": low}


def _incumbent(p, shapes, single: bool) -> int:
    """The route today's main takes for this call -- the owner's INCUMBENT (2026-10-17, "never
    regress main"): on a leg with a size row (:data:`resolve.ONE_DEVICE_MAX_N`, ROCM) the size
    route's ``one_device`` in the round rule's count against that row
    (:func:`distrib_la._one_device.rounds_for`; one matrix: 1 round); on a room leg (CUDA, no
    row) the leg library.  Returns that round count, or 0 for the library.  A shape rule: no
    exchange, no sync."""
    from distrib_la import resolve as _resolve
    from distrib_la._one_device import rounds_for
    row = _resolve.size_row(p.op, _resolve.mesh_platform(p.mesh))
    if row is None:
        return 0
    if single:
        return 1
    ranks = int(p.mesh.shape["x"]) * int(p.mesh.shape["y"])
    nb, n = int(shapes[0][0]), int(shapes[0][-1])
    width = int(shapes[1][-1]) if p.op == "solve_lu" and len(shapes) > 1 else None
    return int(rounds_for(nb, n, ranks, _resolve.one_device_round_n(
        p.op, _resolve.mesh_platform(p.mesh), int(row)), width))


_RULE = ("the fewest-round one_device whose per-rank footprint is within the incumbent's"
         "{room}, else the incumbent")


def _measured_rounds(p, A, args, single: bool = False):
    """The size-route decision of a ``distributed`` plan resolved to ``one_device``, for one call
    signature (owner rules: no stage's per-rank device peak may go up, 2026-10-16; never regress
    main, 2026-10-17).

    The INCUMBENT is what today's main runs on this leg and shape (:func:`_incumbent`): the leg
    library on a room leg (CUDA), ``one_device`` in the round rule's count on a size-row leg
    (ROCM).  The candidates are the ``one_device`` round counts (:func:`_round_candidates`) whose
    per-rank footprint -- XLA's ``memory_analysis`` of the compiled program, temp + output per
    device, the caller's operands common to all -- is within the incumbent's and within the
    plan's room; the fewest-round candidate runs (fewer rounds is the faster program on both
    legs: a deterministic preference, no wall-time measurement), else the incumbent.  Only what
    could beat the incumbent is measured: a ``one_device`` incumbent in k rounds compiles the
    rounds below k (and k, for its own figure), and in 1 round nothing at all (no compile, no
    exchange).  Figures are agreed over every rank (largest candidate bytes against the
    smallest incumbent bytes, the room refused by name unless every rank's is the same) and
    cached per signature on disk and in memory, without the room (:data:`_MEASURED`).

    Returns the rounds to run, 0 for the library, None for ``one_device``'s own round rule (the
    incumbent on a size-row leg, or where no library resolves on this host).  Each decision is a
    route-ledger line (:func:`distrib_la.new_stack_routes`) naming the incumbent, its figure and
    the rule.  Vendor workspaces that a library or jaxlib kernel allocates at run time are
    outside both analyses."""
    import json
    from distrib_la import resolve as _resolve
    dtypes = (A.dtype,) + tuple(x.dtype for x in args)
    shapes = (tuple(int(s) for s in A.shape),) + tuple(tuple(int(s) for s in x.shape) for x in args)
    key = (p.op, _resolve.mesh_key(p.mesh), shapes, tuple(str(d) for d in dtypes), p.budget_bytes,
           p.requested_batched_route, bool(single))
    room = None if p.budget_bytes is None else int(p.budget_bytes)
    platform = _resolve.mesh_platform(p.mesh)
    library = _resolve._DISTRIBUTED_DEFAULT.get((p.op, platform))
    ranks = int(p.mesh.shape["x"]) * int(p.mesh.shape["y"])
    candidates = [1] if single else _round_candidates(shapes[0][0], ranks)
    incumbent = _incumbent(p, shapes, single)
    need_library = incumbent == 0
    if need_library:
        measured = list(candidates)
    else:
        fewer = [r for r in candidates if r < incumbent]
        measured = fewer + [incumbent] if fewer else []
    import hashlib
    import time
    started = time.perf_counter()
    costs, inc_cost, agreed, source = {}, None, None, None
    if measured:
        devices = p.mesh.devices.flat
        kind = str(getattr(next(iter(devices)), "device_kind", ""))
        disk_key = json.dumps([p.op, shapes, [str(d) for d in dtypes],
                               [int(p.mesh.shape["x"]), int(p.mesh.shape["y"])], platform, kind,
                               library, p.requested_batched_route, bool(single), incumbent,
                               measured, _measured_code_hash(), native_closure_identity(platform)])
        if disk_key in _MEASURED:
            source, agreed = "memoised", _MEASURED[disk_key]
        else:
            tag = hashlib.sha256(disk_key.encode()).hexdigest()[:24]
            default = _library_bytes is _LIBRARY_BYTES_DEFAULT
            cached = _measured_cache_get(disk_key, measured, need_library) if default else None
            # Round 1: does EVERY rank hold this signature's agreed costs?  Only then is
            # nothing compiled; one miss and every rank measures, so all lower the same
            # programs (a library plan's first lowering may bootstrap its context collectively).
            probe = _exchange(tag + "/probe", {
                "room": room, "hit": cached is not None,
                "library": None if cached is None else cached["library"],
                "rounds": {} if cached is None else cached["rounds"],
                "rounds_min": {} if cached is None else cached["rounds_min"]})
            if len({v["room"] for v in probe}) > 1:
                raise RuntimeError(f"distrib_la measured size route: {p.op} {shapes[0]} has a room "
                                   f"that differs across processes ({[v['room'] for v in probe]}); "
                                   f"refusing rather than route divergently")
            if all(v["hit"] for v in probe):
                source = "cached"
                agreed = _agreed(probe, need_library)
            else:
                source = "compiled"
                spec = P("x", "y") if single else P(None, "x", "y")
                face = NamedSharding(p.mesh, spec)
                abstract = tuple(jax.ShapeDtypeStruct(s, d, sharding=face)
                                 for s, d in zip(shapes, dtypes))
                lib_local = _library_bytes(p, abstract, single) if need_library else None
                rounds_local = {}
                if lib_local is not None or not need_library:
                    for r in measured:
                        fn = ((lambda *o: p(*o, _rounds=1)) if single
                              else (lambda *o, _r=r: p.batched(*o, _rounds=_r)))
                        c_r = _program_bytes(fn, abstract)
                        rounds_local[str(r)] = None if c_r is None else int(c_r)
                # Round 2: every rank's own figures, agreed conservatively.
                mine = {"library": None if lib_local is None else int(lib_local),
                        "rounds": rounds_local}
                agreed = _agreed(_exchange(tag + "/measure", mine), need_library)
                if agreed is not None and default and _valid_costs(agreed, measured, need_library):
                    _measured_cache_put(disk_key, agreed)
            _MEASURED[disk_key] = agreed
        if agreed is not None:
            costs = {int(r): c for r, c in agreed["rounds"].items()}
            low = {int(r): c for r, c in agreed["rounds_min"].items()}
            inc_cost = agreed["library"] if need_library else low.get(incumbent)
    seconds = time.perf_counter() - started
    within = lambda r: (costs.get(r) is not None and inc_cost is not None and costs[r] <= inc_cost
                        and (room is None or costs[r] <= room))
    if need_library:
        if agreed is None:
            chosen = None                     # no library on this host: one_device's own rounds
        else:
            fits = [r for r in measured if within(r)]
            chosen = fits[0] if fits else 0
    else:
        fits = [r for r in measured if r < incumbent and within(r)]
        chosen = fits[0] if fits else None    # None: the incumbent, main's own round rule
    gb = lambda b: "n/a" if b is None else f"{b / 1e9:.4f}"
    inc_name = library if need_library else f"one_device in {incumbent} round(s)"
    rule = _RULE.format(room="" if room is None else f" and the room {room / 1e9:.2f} GB")
    if need_library and agreed is None:
        verdict = "one_device's own rounds (no library on this host)"
    elif chosen == 0:
        verdict = f"{library} (the incumbent)"
    elif chosen is None:
        verdict = f"one_device in {incumbent} round(s) (the incumbent)"
    else:
        verdict = f"one_device in {chosen} round(s)"
    if measured:
        tried = ", ".join(f"{r}: {gb(c)}" for r, c in costs.items())
        how = (f"measured ({source}, agreed over {jax.process_count()} process(es), "
               f"{seconds:.2f} s) per-rank GB: incumbent {inc_name} {gb(inc_cost)}, "
               f"one_device by rounds {{{tried}}}")
    else:
        how = (f"incumbent {inc_name} (main's round rule on the leg row); no candidate has fewer "
               f"rounds, nothing measured")
    line = (f"{p.op} 'distributed' {'x'.join(map(str, shapes[0]))} on the "
            f"{int(p.mesh.shape['x'])}x{int(p.mesh.shape['y'])} {platform} mesh: {how}; "
            f"rule: {rule} -> {verdict}")
    _resolve._SIZE_DECISIONS.setdefault(("measured",) + key, line)
    _resolve.announce_once(("distrib_la.plan", "measured", key),
                           f"  [distrib_la.plan] {line}", scope="rank0")
    return chosen


_IMPL: dict[tuple[str, str], dict[str, Any]] = {
    ("eigh", "cusolvermp"):     dict(one="distributed_eigh",
                                     many=None, post=_eigh_columns),
    ("eigh", "slate"):          dict(one="distributed_eigh",
                                     many=None, post=_eigh_columns),
    ("eigh", "elpa"):           dict(one="distributed_eigh",
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
    ("solve_lu", "slate"):      dict(one=None,
                                     many="batched_distributed_solve_lu",
                                     post=None),
    # one_device (the size route of ``distributed``): pure JAX with the FFI
    # operand contract, so the checked chain and every route apply to it as
    # to a library.  Its stacked entry is route (c) in bounded rounds.
    ("eigh", ONE_DEVICE):       dict(one="distributed_eigh",
                                     many="batched_distributed_eigh",
                                     post=_eigh_columns),
    ("solve_lu", ONE_DEVICE):   dict(one=None,
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
    sized = ", priced from the shapes" if route.program_bytes is not None else ""
    if route.route == ROUTE_BATCH_RESHARD:
        return (f"{route.route}, {route.per_rank} whole matrix(es) per rank, "
                f"{route.rounds} round(s), compiled {gb(route.program_bytes)}/rank of room "
                f"{gb(route.room)}{sized}")
    if route.program_bytes is None:
        return f"{route.route} on the whole mesh (room {gb(route.room)})"
    return f"{route.route} on the whole mesh ({gb(route.program_bytes)}/rank of room {gb(route.room)}{sized})"


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
        The caller's room per rank beside its own live set, or ``None``. With
        ``'auto'`` requested and a budget given, each eigh stack's route is
        the service's capacity decision (:meth:`stack_route`): route (c) when
        the compiled program that runs it fits, otherwise the provider.
    capacity_refusal
        ``None``, or why an ``auto`` eigh plan with a room has no capacity
        route on this mesh (:func:`~distrib_la.resolve.resolve_eigh_capacity`).
        Such a plan is native: it runs route (c) wherever a stack fits the
        room and refuses by name where one does not (:meth:`require_capacity`).
    """

    op: str
    requested: str
    backend: str
    mesh: Mesh
    n: int | None
    in_sharding: NamedSharding | None
    batch_in_sharding: NamedSharding | None
    requested_batched_route: str = BATCHED_ROUTE_DEFAULT
    # A decision input, never program identity: plans that differ only by
    # room are equal, so every program cached on a plan is shared across
    # rooms (the room enters no HLO; the decided route does, per shape).
    budget_bytes: int | None = field(default=None, compare=False)
    capacity_refusal: str | None = None
    #: What the ``distributed`` size route decided (resolve guard 1b), or None.
    size_route: str | None = field(default=None, compare=False)
    #: Set when the room is 0 because an admitted over-budget stage exhausted it
    #: (:class:`RoomExhausted`): the whole mesh with a warning, not a refusal.
    room_exhausted: RoomExhausted | None = field(default=None, compare=False)
    #: A plan for pricing only (``distrib_la.workspace`` queries, ``known_route``):
    #: built with no room, it may never decide or run a stack (:meth:`stack_route`).
    price_only: bool = False

    def __post_init__(self):
        # Plan construction (plan() and every dataclasses.replace) is where a missing room
        # is refused: a room-0 plan whose stacks the room decides would refuse every one,
        # far from the caller that forgot it.
        if self.price_only or not self.room_decides:
            return
        if int(self.budget_bytes) > 0 or self.room_exhausted is not None:
            return
        px, py = int(self.mesh.shape["x"]), int(self.mesh.shape["y"])
        raise RuntimeError(
            f"GATE distrib_la_room: got: an executing {self.op} plan (n={self.n}) on the {px}x{py} "
            f"mesh with room 0 and no stage that exhausted it; want: the caller's room per rank "
            f"(> 0), RoomExhausted(stage, ...) where an admitted over-budget stage leaves none, "
            f"or plan(..., price_only=True) to price; why: room 0 refuses every stack; "
            f"caller: {_plan_caller()}")

    @property
    def room_decides(self) -> bool:
        """True when :meth:`stack_route` decides this plan's eigh stacks from its room."""
        if (self.op != "eigh" or self.requested_batched_route != "auto" or self.budget_bytes is None
                or (self.is_native and self.capacity_refusal is None)
                or self.batched_route == ROUTE_BATCH_RESHARD):
            return False
        return int(self.mesh.shape["x"]) * int(self.mesh.shape["y"]) > 1

    # ---- introspection -------------------------------------------------
    @property
    def is_native(self) -> bool:
        """True when the resolved backend is the native JAX path."""
        return self.backend == NATIVE

    def require_capacity(self, what: str) -> None:
        """Refuse by name when ``what`` needs the whole-mesh eigh and this plan has none.

        A no-op on every plan except an ``auto`` eigh plan whose capacity
        route did not resolve (:attr:`capacity_refusal`).  Called by
        :meth:`stack_route` for each stack it sends to the whole mesh, and by
        a caller that decides the whole mesh itself before any work.
        """
        if self.capacity_refusal is None:
            return
        px, py = int(self.mesh.shape["x"]), int(self.mesh.shape["y"])
        raise RuntimeError(
            f"GATE eigh_capacity_route: {what} needs the whole-mesh (capacity) "
            f"eigh, and the {px}x{py} {mesh_platform(self.mesh)} mesh has none "
            f"({self.capacity_refusal}).  The 'auto' eigh runs whole matrices "
            f"per rank (route (c)) only where they fit the room.  Fix: more "
            f"ranks or a larger memory_per_device_gb so route (c) fits, or a "
            f"leg that carries the platform's distributed eigh.")

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

    def route_for(self, shape, dtype, traced: bool = True) -> str:
        """The route :meth:`batched` takes for one operand stack (:meth:`stack_route`)."""
        return self.stack_route(shape, dtype, traced=traced).route

    def decision(self, A, *args) -> tuple:
        """What :meth:`batched` traces for this call signature, without the room.

        ``(backend, route, rounds)`` (:meth:`_decide`): everything a plan's room decides in
        the program :meth:`batched` traces for ``A`` and ``args``, so a caller's program cache
        keys on this tuple and never on the room (COMPILEAUDIT 2026-10-05 rule 4).  Host-side,
        on arrays or ``jax.ShapeDtypeStruct`` (only shape and dtype are read).  An eigh plan
        with a room decides per stack in :meth:`stack_route` and is not covered.
        """
        if self.op == "eigh" and self.budget_bytes is not None:
            raise NotImplementedError("Plan.decision: an eigh plan with a room decides per "
                                      "stack (stack_route); not covered")
        return self._decide(A, args)

    def _decide(self, A, args, traced: bool = True, route: str | None = None) -> tuple:
        """``(backend, route, rounds)`` :meth:`batched` runs ``A``, ``args`` with: the one place
        it is decided.  ``rounds`` is set on ``one_device``'s stacked entry only: the measured
        size route's (:func:`_measured_rounds`; 0 the leg library), else one_device's own round
        rule (:func:`distrib_la._one_device.rounds_for` against
        :func:`distrib_la.resolve.one_device_round_n` of the bound: the leg row, or on a room leg
        the room this plan was resolved against, capped by the leg's round row).  ``route`` is the gates' override."""
        route = self.route_for(A.shape, A.dtype, traced) if route is None else route
        rounds = None
        if route == ROUTE_BACKEND_BATCHED and self.backend == ONE_DEVICE:
            if self._measured_applies():
                rounds = _measured_rounds(self, A, args)
            if rounds is None:
                from distrib_la import _one_device as od, resolve as _resolve
                n = int(A.shape[-1])
                bound = od._bound(self.op, self.mesh, n, _resolve.one_device_max_n(
                    self.op, _resolve.mesh_platform(self.mesh), self.budget_bytes))
                # one_device's own round rule: against the leg's round row (ONE_DEVICE_ROUND_N,
                # resolve.one_device_round_n), exactly as _one_device computes it without rounds=.
                rounds = od.rounds_for(int(A.shape[0]), n, od._ranks(self.mesh),
                                       _resolve.one_device_round_n(
                                           self.op, _resolve.mesh_platform(self.mesh), bound),
                                       int(args[0].shape[-1]) if self.op == "solve_lu" else None)
        return (self.backend, route, rounds)

    def stack_route(self, shape, dtype, op: str = "eigh", traced: bool = True) -> StackRoute:
        """How :meth:`batched` runs one operand stack: the service's decision.

        :attr:`batched_route` is the answer for every stack except one case:
        an eigh plan built with ``budget_bytes`` (the caller's room per rank,
        beside its own live set) and no explicit route. Then capacity decides:
        the stack runs route (c), one or more whole matrices per rank, when
        the program that runs it, priced from the shapes (:func:`_stack_price`),
        fits the room; the stack is cut into as many equal slices (rounds) as
        that needs. When not even one whole matrix per rank fits, it runs on the
        whole mesh. The room is a caller value every rank shares and the price
        is a shape formula, so every rank takes the same route and rounds with
        no exchange. ``traced`` says what the program reserves
        (:data:`PHASES`). Every decision is listed by
        :meth:`describe`, and :func:`new_stack_routes` hands each to a driver
        log once.
        """
        if self.price_only:
            raise RuntimeError(
                f"GATE distrib_la_price_only: got: a stack {tuple(shape)} decided on a price-only "
                f"{self.op} plan; want: an executing plan with the caller's room; why: a pricing "
                f"plan has no room; caller: {_plan_caller()}")
        static = self.batched_route
        shape = tuple(int(v) for v in shape)
        nb = shape[0] if len(shape) == 3 else 1
        ranks = int(self.mesh.shape["x"]) * int(self.mesh.shape["y"])
        if static == ROUTE_BATCH_RESHARD:
            return StackRoute(static, -(-nb // ranks))
        capacity = self.capacity_refusal is None
        if (self.requested_batched_route != "auto" or self.budget_bytes is None
                or (self.is_native and capacity) or self.op != "eigh"):
            return StackRoute(static)
        if capacity and in_checked_retry():
            # A checked program's retry is the whole-mesh program its caller
            # sized; a plan with no whole mesh retries on its decided route.
            self.require_capacity(f"an eigh stack {shape} with no room")
            return StackRoute(static, room=0)
        key = self._stack_key(op, nb, shape[-1], dtype, traced)
        decided = _STACK_ROUTES.get(key)
        if decided is None:
            decided = self._decide_stack(op, nb, key[3], key[4], key[6], traced)
            if decided.route != ROUTE_BATCH_RESHARD:
                self.require_capacity(
                    f"the eigh stack {shape} ({_describe_stack(decided)})")
                if decided.program_bytes > decided.room and self.room_exhausted is not None:
                    # The stage was admitted over budget: the whole mesh is the smallest route.
                    x = self.room_exhausted
                    print(f"warning: distrib_la {op} stack {shape} {np.dtype(dtype).name}: room "
                          f"exhausted by the admitted over-budget stage {x.stage} (priced "
                          f"{x.priced_bytes / 1e9:.2f} GB/rank of {x.available_bytes / 1e9:.2f}); "
                          f"on the whole mesh at {decided.program_bytes / 1e9:.2f} GB/rank, the "
                          f"minimum-footprint route (an OOM is possible)", flush=True)
                elif decided.program_bytes > decided.room:
                    raise RuntimeError(
                        f"GATE distrib_la_capacity: the {op} stack {shape} {np.dtype(dtype).name} "
                        f"needs {decided.program_bytes / 1e9:.2f} GB/rank on the whole mesh, over its "
                        f"room {decided.room / 1e9:.2f} GB/rank (route (c) holds whole matrices: more).")
            _STACK_ROUTES[key] = decided
        return decided

    def _stack_key(self, op, nb, n, dtype, traced):
        phase = "all" if op == "normal_eigh" else traced_phase() if traced else "first"
        return (mesh_key(self.mesh), op, int(nb), int(n), np.dtype(dtype).name,
                self.budget_bytes, phase, self.backend, self.batched_route)

    def known_route(self, shape, dtype) -> str:
        """The route a stack takes when :meth:`stack_route` already decided it,
        else this plan's provider route: a pure query, with no compile and no
        exchange, for pricing (``distrib_la.workspace``)."""
        static = self.batched_route
        if (static == ROUTE_BATCH_RESHARD or self.budget_bytes is None
                or (self.is_native and self.capacity_refusal is None)):
            return static
        shape = tuple(int(v) for v in shape)
        nb = shape[0] if len(shape) == 3 else 1
        for traced in (True, False):
            decided = _STACK_ROUTES.get(self._stack_key("eigh", nb, shape[-1], dtype, traced))
            if decided is not None:
                return decided.route
        return static

    def _decide_stack(self, op, nb, n, dtype, phase, traced=True) -> StackRoute:
        return place(op, self.mesh, nb, n, dtype, int(self.budget_bytes), self.batched_route, self.backend)

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
        where = ("route (c) only; no whole-mesh eigh on this mesh"
                 if self.capacity_refusal is not None else
                 "native JAX, any layout" if self.is_native else
                 f"each matrix whole on one device, face layout P('x','y') on the {px}x{py} mesh"
                 if self.backend == ONE_DEVICE else
                 f"ONE tile over the {px}x{py} mesh at P('x','y')")
        if self.size_route is not None:
            where += f"; {self.size_route}"
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
    def _measured_applies(self) -> bool:
        """A plan the size route resolved from ``distributed`` to ``one_device``: each call
        shape is decided by measured cost (:func:`_measured_rounds`)."""
        return (self.backend == ONE_DEVICE
                and str(self.requested).strip().lower() == "distributed")

    def _library_plan(self):
        """The leg library plan ``distributed`` names on this mesh (built once)."""
        from distrib_la import resolve as _resolve
        library = _resolve._DISTRIBUTED_DEFAULT[(self.op, _resolve.mesh_platform(self.mesh))]
        key = (self.op, _resolve.mesh_key(self.mesh), self.n, self.requested_batched_route, library)
        lib = _LIBRARY_PLANS.get(key)
        if lib is None:
            lib = _LIBRARY_PLANS[key] = plan(self.op, self.mesh, backend=library, n=self.n,
                                             batched_route=self.requested_batched_route)
        return lib

    def _entry(self, key: str, rounds: int | None = None):
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
        if call is not None and self.backend == ONE_DEVICE:
            # The bound that plans the rounds: the leg row, or on a room leg the room this
            # plan was resolved against (resolve.one_device_max_n), so both agree.
            import functools
            from distrib_la import resolve as _resolve
            call = functools.partial(call, max_n=_resolve.one_device_max_n(
                self.op, _resolve.mesh_platform(self.mesh), self.budget_bytes))
            if rounds is not None and key == "many":
                call = functools.partial(call, rounds=int(rounds))      # Plan._decide's
        if call is None or self.op not in ("eigh", "solve_lu"):
            return call, spec["post"]
        if self.op == "solve_lu":
            return self._checked_solve(call), None
        return self._checked_eigh(call, spec["post"]), None

    def _is_checked(self):
        """An FFI eigh or LU solve: its result comes back checked, with a failure flag."""
        return self.op in ("eigh", "solve_lu") and not self.is_native

    def _finish(self, out, A, site):
        """Split a checked call's replicated failure flag off; an eager call raises on it."""
        from distrib_la._result_check import raise_if_failed
        *result, failed = out
        raise_if_failed(failed, self.op, int(A.shape[-1]), A.dtype, site)
        return tuple(result) if len(result) > 1 else result[0]

    def _program(self, fn, ops, kwargs, tag, site):
        """A checked entry as one cached program for an eager call; a traced caller inlines it.

        Keyed by the call's signature only (the refusal names its site on the
        host), so a repeated eager call compiles once from every call site.
        """
        if any(isinstance(o, jax.core.Tracer) for o in ops):
            return fn(*ops, mesh=self.mesh, _phase=traced_phase(), **kwargs)
        mesh = self.mesh

        def program(phase, donate):
            key = scan_signature(self.op, self.backend, mesh, ops, kwargs,
                                 extra=(tag, phase))
            return cached_scan(key, lambda: jax.jit(
                lambda *o: fn(*o, mesh=mesh, _phase=phase, **kwargs),
                donate_argnums=donate))
        if self.op != "eigh":
            return program("all", self.donates)(*ops)
        # An eager eigh runs its first attempt and check alone; the retries
        # are a separate program, compiled and run only on the reduced
        # failure flag, so their buffers are never reserved beside the first
        # attempt's (EIGHSTACK: n 18304, P4, 80 GB).
        out = program("first", ())(*ops)
        if not _flag(out[-1], int(ops[0].shape[-1]), ops[0].dtype):
            return out
        _notices(out[-1], self.op, int(ops[0].shape[-1]), ops[0].dtype, site, final=False)
        del out
        return program("retry", ())(*ops)

    def _checked_eigh(self, call, post):
        from distrib_la._result_check import checked_eigh, eigh_layout, shifted
        from distrib_la.matmul import matmul
        backend = self.backend

        def safe(A, *, mesh, _phase="all", cut=None, **kwargs):
            # ``cut``: check what you use (checked_eigh), on every attempt of the chain.
            def solve(a, **extra):
                return post(backend, *call(a, mesh=mesh, **kwargs, **extra))
            # Zero rows leave the solver as distinct sentinels (deflate_zero_rows),
            # on every attempt. A failed check solves again: shifted (a near-zero
            # cluster moved off the origin, the vectors re-orthonormalized), then
            # gathered.
            # The face SUMMA (panel_matmul): a single matrix never batch-reshards onto one device.
            gemm = partial(matmul, mesh=mesh, backend="distributed", batched_route="auto")
            pin = eigh_layout(mesh, A.ndim)
            deflate = partial(deflate_zero_rows, constrain=pin, mesh=mesh)
            attempts = [deflate(solve), deflate(shifted(solve, matmul=gemm, mesh=mesh))]
            n = int(A.shape[-1])
            if backend == "cusolvermp":
                # cuSOLVERMp 0.9.1 syevd returns vectors that are not orthonormal
                # inside a near-zero cluster, with status 0 and info 0, at some
                # (matrix, layout, shift) and not others; no one of them is safe
                # (EIGHOOM, n 7908 and 11776-15872, P4). The retries change one
                # at a time: the layout first (retry_block, a smaller block on the
                # same side: no more workspace), then the shift with
                # re-orthonormalization, then the shift at the smaller block.
                from distrib_la._cusolvermp import retry_block
                block = retry_block(n, int(mesh.shape["x"]))
                if block is not None:
                    attempts = [attempts[0], deflate(partial(solve, block=block)), attempts[1],
                                deflate(partial(solve, block=block), shift=True)]
            # A gathered local solve, where the whole matrix, its vectors and the
            # solver's copy fit GATHERED_EIGH_BYTES, or with its workspace the plan's
            # room, on every rank (_gathered_admitted; priced in the same room).
            if _gathered_admitted(A.shape, A.dtype, self.budget_bytes):
                attempts.append(partial(_gathered_eigh, mesh=mesh))
            # An eager call's first program holds the first attempt; its retry
            # program the rest (Plan._program). A traced call holds them all.
            attempts = {"first": attempts[:1], "retry": attempts[1:], "all": attempts}[_phase]
            (values, vectors), failed = checked_eigh(attempts, A, mesh=mesh, final=_phase != "first", cut=cut)
            return values, vectors, failed
        return safe

    def _checked_solve(self, call):
        from distrib_la._result_check import checked, matrix_sketch, rhs_sketch, solve_errors

        def safe(A, B, *, mesh, _phase="all", **kwargs):
            # The sketch is taken before the call, which consumes A and B: the LU
            # factors are an output aliased to A and X one aliased to B.  The
            # barrier keeps XLA from scheduling the sketch after the call.
            sketch = (*matrix_sketch(A, mesh), *rhs_sketch(B, mesh))
            sketch, A, B = jax.lax.optimization_barrier((sketch, A, B))
            X = call(A, B, mesh=mesh, **kwargs)
            return checked("solve_lu", (lambda x: x,), lambda x: solve_errors(sketch, x, mesh),
                           (X,), n=A.shape[-1], dtype=A.dtype)
        return safe

    def __call__(self, A, *args, _rounds: int | None = None, **kwargs):
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
        if (_rounds is None and self.op == "eigh" and self._measured_applies()
                and _measured_rounds(self, A, args, True) == 0):
            return self._library_plan()(A, *args, **kwargs)
        one, post = self._entry("one")
        if one is None:
            raise NotImplementedError(
                f"{self.op} backend {self.backend!r} has no single-tile "
                f"entry point — its FFI call factors a whole stack in one "
                f"go (one descriptor, one workspace).  Use plan.batched(); "
                f"a stack of one is a legal stack.")
        ops = [ensure_sharding(x, self.in_sharding) for x in (A, *args)]
        if self._is_checked():
            from distrib_la._result_check import call_site
            site = call_site()
            return self._finish(self._program(one, ops, kwargs, "one", site), A, site)
        out = one(*ops, mesh=self.mesh, **kwargs)
        return post(self.backend, *out) if post is not None else out

    def batched(self, A, *args, _route: str | None = None, _rounds: int | None = None,
                **kwargs):
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
        traced = isinstance(A, jax.core.Tracer)
        route = self.route_for(A.shape, A.dtype, traced) if _route is None else _route
        if route not in BATCHED_ROUTES:
            raise ValueError(
                f"unknown batched route {route!r} "
                f"(known: {'|'.join(BATCHED_ROUTES)})")

        # Route (c) deliberately runs the pure-JAX operation even when the
        # plan resolved an FFI backend.  Its layout contract still starts at
        # the same face sharding, including on an explicit native plan.
        if route == ROUTE_BATCH_RESHARD:
            if self.op == "eigh":
                cut = kwargs.pop("cut", None)
                self._reshard_options(kwargs)
                rounds = (1 if _route is not None else
                          self.stack_route(A.shape, A.dtype, traced=traced).rounds)
                return self._checked_reshard("eigh", A, rounds, cut)
            return self._batch_reshard((A, *args), kwargs)

        if self.is_native:
            if route != ROUTE_BACKEND_BATCHED:
                raise NotImplementedError(
                    f"native {self.op} has no route {route!r}; its automatic "
                    f"batched entry is {ROUTE_BACKEND_BATCHED!r}, and the "
                    f"only public alternative is "
                    f"{ROUTE_BATCH_RESHARD!r}.")
            return self(A, *args, **kwargs)

        rounds = _rounds if _rounds is not None else self._decide(A, args, traced, route)[2]
        if rounds == 0:
            return self._library_plan().batched(A, *args, **kwargs)
        ops = tuple(ensure_sharding(x, self.batch_in_sharding)
                    for x in (A, *args))
        if route == ROUTE_BACKEND_BATCHED:
            many, post = self._entry("many", rounds=rounds)
            if many is None:
                raise NotImplementedError(
                    f"{self.op} backend {self.backend!r} has no stacked FFI "
                    f"entry point, so route {ROUTE_BACKEND_BATCHED!r} does "
                    f"not exist for it.  Its batched route is "
                    f"{self.batched_route!r}.")
            if self._is_checked():
                from distrib_la._result_check import call_site
                site = call_site()
                return self._finish(self._program(many, ops, kwargs, "many", site), A, site)
            out = many(*ops, mesh=self.mesh, **kwargs)
            return post(self.backend, *out) if post is not None else out
        if route == ROUTE_SCAN:
            if self._is_checked():
                from distrib_la._result_check import call_site
                site = call_site()
                if self.op != "eigh" or isinstance(A, jax.core.Tracer):
                    phase = traced_phase() if isinstance(A, jax.core.Tracer) else "all"
                    return self._finish(self._scan_over_single(ops, kwargs, phase=phase), A, site)
                # Eager: the retries run as a second scan only when a matrix failed.
                out = self._scan_over_single(ops, kwargs, phase="first")
                if _flag(out[-1], int(A.shape[-1]), A.dtype):
                    _notices(out[-1], self.op, int(A.shape[-1]), A.dtype, site, final=False)
                    del out
                    out = self._scan_over_single(ops, kwargs, phase="retry")
                return self._finish(out, A, site)
            return self._scan_over_single(ops, kwargs)
        raise AssertionError(f"unhandled batched route {route!r}")

    def reshard_stack(self, op: str, A):
        """Route (c) for a face stack: ``op`` is ``eigh``, ``checked_eigh`` or
        ``normal_eigh`` (polar's right singular vectors), in this plan's
        decided rounds (:meth:`stack_route`), checked as :meth:`_checked_reshard`."""
        traced = isinstance(A, jax.core.Tracer)
        return self._checked_reshard(op, A, self.stack_route(A.shape, A.dtype, op, traced).rounds)

    def _checked_reshard(self, op: str, A, rounds: int, cut=None):
        """Route (c) in ``rounds`` slices, checked (:func:`_reshard_stack_program`).

        A traced caller inlines its phase of the chain (:data:`PHASES`). An eager call runs
        its first attempt alone and the retries as a second program only when the
        first check failed; a result no attempt repairs refuses by name.
        """
        from distrib_la._batch_reshard import validate_batch_reshard_operands
        from distrib_la._result_check import call_site, raise_if_failed
        validate_batch_reshard_operands(op if op != "normal_eigh" else "eigh", self.mesh, (A,))
        face = NamedSharding(self.mesh, P(None, "x", "y"))
        A = ensure_sharding(A, face)
        shape, dtype, site = tuple(int(v) for v in A.shape), str(A.dtype), call_site()

        def program(phase):
            return _reshard_stack_program(op, self.mesh, shape, dtype, int(rounds), phase,
                                          cut if cut is None or callable(cut) else float(cut),
                                          _gathered_admitted(shape[-2:], dtype, self.budget_bytes))
        if op == "normal_eigh":
            return program("all")(A)
        if isinstance(A, jax.core.Tracer):
            values_vectors, failed = program(traced_phase())(A)
            raise_if_failed(failed, "eigh", shape[-1], A.dtype, site)
            return values_vectors
        values_vectors, failed = program("first")(A)
        if _flag(failed, shape[-1], A.dtype):
            _notices(failed, "eigh", shape[-1], A.dtype, site, final=False)
            del values_vectors
            values_vectors, failed = program("retry")(A)
            raise_if_failed(failed, "eigh", shape[-1], A.dtype, site)
        return values_vectors

    def _reshard_options(self, kwargs: dict):
        """Refuse keywords route (c) cannot honour.

        A block size configures a distributed library descriptor the local
        JAX kernel lacks, so it is dropped rather than failing a universal
        route flag at call time; eigh's vectors hint is moot, because the
        public contract returns (W, Z) on every route. Any other keyword has
        no native-JAX meaning.
        """
        options = {k for k in kwargs
                   if k != "block_size" and not (self.op == "eigh" and k == "compute_evecs")}
        if options:
            raise TypeError(
                f"batch_reshard {self.op}: keyword(s) "
                f"{', '.join(sorted(options))} have no native-JAX meaning")

    def _batch_reshard(self, ops: tuple, kwargs: dict):
        """Route (c) for a factor or solve: staged movement, local native op, staged inverse."""
        self._reshard_options(kwargs)
        from distrib_la._batch_reshard import (
            batch_reshard_call, validate_batch_reshard_operands,
        )
        # Shape refusals precede even the face placement, and therefore
        # precede every collective in the route.
        validate_batch_reshard_operands(self.op, self.mesh, ops)
        ops = tuple(ensure_sharding(x, self.batch_in_sharding) for x in ops)
        return batch_reshard_call(self.op, self.mesh, ops)

    def _scan_over_single(self, ops: tuple, kwargs: dict, *, phase=None):
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
                # A checked entry returns its failure flag too; the scan stacks it.
                extra = {} if phase is None else {"_phase": phase}
                out = one(*(ensure_sharding(t, tile) for t in tiles),
                          mesh=mesh, **extra, **kwargs)
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

        key = scan_signature(self.op, self.backend, mesh, ops, kwargs, extra=(phase,))
        return cached_scan(key, _build)(*ops)


def _size_routed_library(op: str, requested: str, mesh_xy: Mesh, n, resolved: str) -> bool:
    """``distributed`` resolved to the leg's library BY SIZE: ``n`` past the leg's
    one-device row (:data:`distrib_la.resolve.ONE_DEVICE_MAX_N`).  Recorded on the
    route ledger (:func:`new_stack_routes`) when true.  A shape rule: no exchange."""
    from distrib_la import resolve as _resolve
    if (str(requested).strip().lower() != "distributed" or n is None
            or resolved in (NATIVE, ONE_DEVICE)):
        return False
    platform = _resolve.mesh_platform(mesh_xy)
    bound = _resolve.size_row(op, platform)
    if bound is None or int(n) <= int(bound):
        return False
    px, py = int(mesh_xy.shape["x"]), int(mesh_xy.shape["y"])
    _resolve._SIZE_DECISIONS.setdefault(
        ("batched-route", op, platform, px, py, int(n)),
        f"{op} 'distributed' n={int(n)} on the {px}x{py} {platform} mesh: batched route "
        f"'auto' -> {resolved}'s own entry (n > {bound}: the size route chose {resolved}; "
        f"the default route (c) would solve each matrix whole on one device)")
    return True


def plan(op: str, mesh_xy: Mesh, *, backend: str = "auto",
         n: int | None = None,
         batched_route: str | None = None,
         budget_bytes: int | RoomExhausted | None = None,
         price_only: bool = False) -> Plan:
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
        fits the room, else the whole-mesh provider.  With ``backend='auto'``
        (eigh) the whole-mesh provider is the platform's ``distributed``
        backend when it resolves on this mesh, so the plan equals the one an
        explicit ``distributed`` request builds; when it does not, the plan is
        native, runs route (c), and refuses by name at the first stack that
        needs the whole mesh (:func:`~distrib_la.resolve.resolve_eigh_capacity`).
        This is how a ``linalg = local`` deck asks for the capacity route.
        The room also caps an explicit ``distributed`` request's size route
        (resolve guard 1b): ``one_device`` only where one whole matrix fits it.

    ``budget_bytes`` may be :class:`RoomExhausted` (0 bytes, a stage admitted over
    budget: the whole mesh with a warning). An ``auto`` eigh plan that its room
    decides refuses room 0 otherwise (``GATE distrib_la_room``, naming the caller)
    unless ``price_only=True``: a plan for pricing queries that never decides or
    runs a stack (``GATE distrib_la_price_only``).

    There is deliberately NO ``batched=`` flag.  The design sketch carried
    one; it would have changed nothing about resolution (both shardings are
    on every plan and :meth:`Plan.batched` is always available), so it could
    only be a second way to say what the call already says — which is the
    kind of concept this extraction exists to remove, not add.
    """
    # A room exhausted by an admitted over-budget stage counts as 0 bytes (RoomExhausted).
    exhausted = budget_bytes if isinstance(budget_bytes, RoomExhausted) else None
    if exhausted is not None:
        budget_bytes = 0
    if op not in OPS:
        raise ValueError(f"unknown linalg op {op!r} (known: {'|'.join(OPS)})")
    route_defaulted = batched_route is None and budget_bytes is None
    if batched_route is None:
        batched_route = "auto" if budget_bytes is not None else BATCHED_ROUTE_DEFAULT
    batched_route = str(batched_route).strip().lower()
    if batched_route not in BATCHED_ROUTE_CHOICES:
        raise ValueError(
            f"unknown batched route {batched_route!r} (known: "
            f"{'|'.join(BATCHED_ROUTE_CHOICES)})")
    capacity_refusal = None
    if op == "eigh" and budget_bytes is not None and str(backend).strip().lower() == "auto":
        # Local where a stack fits the room, the platform's whole-mesh eigh
        # where it does not (resolve_eigh_capacity; refused by name at that
        # stack when this mesh has none).
        resolved, capacity_refusal = resolve_eigh_capacity(mesh_xy, n=n)
    else:
        # A room caps the size route of ``distributed`` as well (guard 1b): one
        # whole matrix per rank must fit it, or the leg's library runs.
        resolved = resolve_backend(op, backend, mesh_xy, n=n, room=budget_bytes)
    if (route_defaulted and batched_route == ROUTE_BATCH_RESHARD
            and _size_routed_library(op, backend, mesh_xy, n, resolved)):
        # Route (c) solves each matrix WHOLE on one device: exactly what the size
        # route just ruled out for this n.  Left as the default it ran a 20480
        # complex LU on one GCD of a plan that reports slate (MI250X P4,
        # dlascale probe1); the leg library's own route runs instead.  An
        # explicit batched_route is honoured as named.
        batched_route = "auto"
    ffi = resolved != NATIVE
    if budget_bytes is not None:
        import operator
        budget_bytes = operator.index(budget_bytes)
        if budget_bytes < 0:
            raise ValueError(f"budget_bytes must be non-negative, got {budget_bytes}")
    face = ffi or batched_route == ROUTE_BATCH_RESHARD or capacity_refusal is not None
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
                budget_bytes=budget_bytes, capacity_refusal=capacity_refusal,
                size_route=size_route_note(op, backend, mesh_xy, n, resolved, budget_bytes),
                room_exhausted=exhausted, price_only=bool(price_only))
