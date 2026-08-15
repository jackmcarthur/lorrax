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
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from distrib_la.resolve import (NATIVE, NATIVE2D, OPS, backend_module,
                                mesh_key, resolve_backend)

__all__ = ["Plan", "plan", "ensure_sharding", "BATCHED_SCAN_UNROLL",
           "BATCHED_ROUTES", "BATCHED_ROUTE_CHOICES", "ROUTE_SCAN",
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

#: (a) ``lax.scan`` over this package's own single-matrix op.  The default.
ROUTE_SCAN = "scan"
#: (b) the backend's own stacked FFI entry point, where the library has one.
ROUTE_BACKEND_BATCHED = "backend_batched"
#: (c) staged batch-axis reshard + a device-local native JAX kernel.
ROUTE_BATCH_RESHARD = "batch_reshard"

#: Every route the toggle can name.  A vocabulary, like the backend names:
#: importable with no jax and no ``.so``, so a test can enumerate the routes
#: without a mesh.
BATCHED_ROUTES = (ROUTE_SCAN, ROUTE_BACKEND_BATCHED, ROUTE_BATCH_RESHARD)

#: Public construction-time grammar.  ``scan`` and ``backend_batched`` stay
#: implementation details selected by ``auto``; the one user-facing opt-in is
#: route (c), which changes the memory/capacity tradeoff deliberately.
BATCHED_ROUTE_CHOICES = ("auto", ROUTE_BATCH_RESHARD)

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
_NATIVE_CALLABLE = {"eigh": lambda A, *a, **k: jnp.linalg.eigh(A)}

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
    """

    op: str
    requested: str
    backend: str
    mesh: Mesh
    n: int | None
    in_sharding: NamedSharding | None
    batch_in_sharding: NamedSharding | None
    requested_batched_route: str = "auto"

    # ---- introspection -------------------------------------------------
    @property
    def is_native(self) -> bool:
        """True when the resolved backend is the in-tree JAX path."""
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
            this plan's own single-matrix op.  The default, and the
            definition of the batched surface.

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
            (the service-private sibling of
            ``common.staged_reshard.face_to_batch_reshard``:
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
            fixed per-call charge — cuSOLVERMp eigh is a flat ~1.55 s per
            matrix from n=64 to n=1024, where native replicated is 1.5 ms
            to 27 ms (§ Performance in ``docs/services/distrib_la.md``).
            Route (c) serves that whole regime with no distributed-library
            call at all, and it is how small-system linalg happens without
            a second, parallel API.

        ``plan(..., batched_route=...)`` is the one caller-facing selection:
        ``'auto'`` preserves the historical route, while ``'batch_reshard'``
        opts into (c).  ``batched()`` also keeps a private ``_route``
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
            mesh = self.mesh
            return lambda A, **kw: impl(A, mesh=mesh, **kw)
        raise NotImplementedError(
            f"native_fn is defined for the pure-JAX backends only; this "
            f"plan is {self.op}/{self.backend}.  An FFI backend's wrapper "
            f"is eager by construction (it dlopens a library and reads "
            f"jax.process_count()), so there is no trace-safe closure to "
            f"hand out — call the plan itself.")

    def describe(self) -> str:
        """One line for a run banner: what resolved, and to what geometry."""
        px, py = int(self.mesh.shape["x"]), int(self.mesh.shape["y"])
        where = ("in-tree JAX, any layout" if self.is_native else
                 f"ONE tile over the {px}x{py} mesh at P('x','y')")
        n = "" if self.n is None else f", n={self.n}"
        return (f"{self.op}: {self.requested!r} -> {self.backend} "
                f"({where}{n}); batched "
                f"{self.requested_batched_route!r} -> {self.batched_route}")

    # ---- calling -------------------------------------------------------
    def _entry(self, key: str):
        spec = _IMPL[(self.op, self.backend)]
        name = spec[key]
        return (None if name is None
                else getattr(self.module, name)), spec["post"]

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
        return post(self.backend, *out) if post is not None else out

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
        route = self.batched_route if _route is None else _route
        if route not in BATCHED_ROUTES:
            raise ValueError(
                f"unknown batched route {route!r} "
                f"(known: {'|'.join(BATCHED_ROUTES)})")

        # Route (c) deliberately runs the pure-JAX operation even when the
        # plan resolved an FFI backend.  Its layout contract still starts at
        # the same face sharding, including on an explicit native plan.
        if route == ROUTE_BATCH_RESHARD:
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
            return post(self.backend, *out) if post is not None else out
        if route == ROUTE_SCAN:
            return self._scan_over_single(ops, kwargs)
        raise AssertionError(f"unhandled batched route {route!r}")

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
         n: int | None = None, batched_route: str = "auto") -> Plan:
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
        ``'auto'`` preserves the historical backend-batched/scan choice;
        ``'batch_reshard'`` stages the batch over the mesh and runs the
        device-local native JAX operation.  See :attr:`Plan.batched_route`.

    There is deliberately NO ``batched=`` flag.  The design sketch carried
    one; it would have changed nothing about resolution (both shardings are
    on every plan and :meth:`Plan.batched` is always available), so it could
    only be a second way to say what the call already says — which is the
    kind of concept this extraction exists to remove, not add.
    """
    if op not in OPS:
        raise ValueError(f"unknown linalg op {op!r} (known: {'|'.join(OPS)})")
    batched_route = str(batched_route).strip().lower()
    if batched_route not in BATCHED_ROUTE_CHOICES:
        raise ValueError(
            f"unknown batched route {batched_route!r} (known: "
            f"{'|'.join(BATCHED_ROUTE_CHOICES)})")
    resolved = resolve_backend(op, backend, mesh_xy, n=n)
    ffi = resolved != NATIVE
    face = ffi or batched_route == ROUTE_BATCH_RESHARD
    tile = NamedSharding(mesh_xy, P("x", "y")) if face else None
    stack = NamedSharding(mesh_xy, P(None, "x", "y")) if face else None
    if batched_route == ROUTE_BATCH_RESHARD:
        # Eager plan construction is the legal place for runtime setup; the
        # returned Plan.batched remains trace-safe.
        from distrib_la._collectives import warm_mesh_cliques
        warm_mesh_cliques(mesh_xy)
    return Plan(op=op, requested=str(backend), backend=resolved,
                mesh=mesh_xy, n=None if n is None else int(n),
                in_sharding=tile, batch_in_sharding=stack,
                requested_batched_route=batched_route)
