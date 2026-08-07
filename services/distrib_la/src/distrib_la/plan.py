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
                                resolve_backend)

__all__ = ["Plan", "plan", "ensure_sharding"]


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
    be, because :func:`dispatch_batched_eigh` probes for a backend's
    stacked entry with ``getattr`` rather than a name table — the day
    cuSOLVERMp grows a ``batched_distributed_eigh`` this function starts
    receiving stacks with no other edit, and the old ``.T`` (reverse ALL
    axes) would have silently returned ``(n, n, nq)``.
    """
    if backend == "cusolvermp":
        return lam, jnp.conj(jnp.swapaxes(Q, -1, -2))
    return lam, Q


#: (op, backend) → how to call it.
#:
#:   one    attribute implementing the SINGLE-tile form, or None
#:   many   attribute implementing the STACKED form, or None
#:   post   result normaliser, applied to both forms
#:
#: A ``None`` on either side is filled in by looping/slicing the other —
#: that is the whole point of :meth:`Plan.batched`.  ``cholesky`` is
#: asymmetric on purpose: cuSOLVERMp's batched potrf returns a HANDLE
#: object carrying the block-cyclic geometry, while SLATE's per-tile potrf
#: returns a conventional row-major L.  The plan does not flatten that
#: difference away — :func:`distrib_la.factor` is where it is handled — it
#: just stops every caller from re-deriving which is which.
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
                                     many=None, post=None),
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
        single tile, ``P(None,'x','y')`` for a stack.  ``None`` on a native
        plan, which accepts any layout.
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
                f"({where}{n})")

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

    def batched(self, A, *args, **kwargs):
        """Run the op on a STACK ``(nb, n, n)`` — uniform across backends.

        Uses the backend's own batched entry point when it has one (one FFI
        call, one descriptor, one workspace for the whole stack) and
        otherwise loops the leading axis and stacks, which is what every
        call site used to write out by hand.  Operands are moved to
        :attr:`batch_in_sharding` first.

        The native ``eigh`` plan just forwards to ``jnp.linalg.eigh``,
        which is natively batched (and is why ``auto`` picks it).
        """
        if self.is_native:
            return self(A, *args, **kwargs)
        ops = [ensure_sharding(x, self.batch_in_sharding) for x in (A, *args)]
        many, post = self._entry("many")
        if many is not None:
            out = many(*ops, mesh=self.mesh, **kwargs)
            return post(self.backend, *out) if post is not None else out
        one, _ = self._entry("one")
        nb = int(A.shape[0])
        rows = [one(*[x[i] for x in ops], mesh=self.mesh, **kwargs)
                for i in range(nb)]
        rows = [post(self.backend, *r) if post is not None else r
                for r in rows]
        return _stack_results(rows, self.op, self.backend)


def _stack_results(rows, op: str, backend: str):
    """``[r0, r1, …]`` → stacked, tuple-shape-preserving.

    Refuses non-array results explicitly.  SLATE's per-tile ``potrf``
    returns a ``SlateLowerL`` HANDLE, not an array, so a loop-and-stack
    batched cholesky would need a per-row ``.to_jax_lower()`` and a
    sharding constraint on the stack.  Say so rather than raise a
    ``TypeError`` from inside ``jnp.stack`` — and note that
    :func:`distrib_la.factor` is the surface that DOES carry handles.
    """
    def _stackable(x):
        return hasattr(x, "shape") and hasattr(x, "dtype")
    flat = rows[0] if isinstance(rows[0], tuple) else (rows[0],)
    if not all(_stackable(y) for y in flat):
        raise NotImplementedError(
            f"{op} backend {backend!r} has no batched entry point and its "
            f"single-tile result is not an array, so plan.batched() cannot "
            f"stack it.  Use distrib_la.factor()/solve(), which carries the "
            f"handle in an opaque token instead of flattening it.")
    if isinstance(rows[0], tuple):
        return tuple(jnp.stack([r[j] for r in rows])
                     for j in range(len(rows[0])))
    return jnp.stack(rows)


def plan(op: str, mesh_xy: Mesh, *, backend: str = "auto",
         n: int | None = None) -> Plan:
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

    There is deliberately NO ``batched=`` flag.  The design sketch carried
    one; it would have changed nothing about resolution (both shardings are
    on every plan and :meth:`Plan.batched` is always available), so it could
    only be a second way to say what the call already says — which is the
    kind of concept this extraction exists to remove, not add.
    """
    if op not in OPS:
        raise ValueError(f"unknown linalg op {op!r} (known: {'|'.join(OPS)})")
    resolved = resolve_backend(op, backend, mesh_xy, n=n)
    ffi = resolved != NATIVE
    tile = NamedSharding(mesh_xy, P("x", "y")) if ffi else None
    stack = NamedSharding(mesh_xy, P(None, "x", "y")) if ffi else None
    return Plan(op=op, requested=str(backend), backend=resolved,
                mesh=mesh_xy, n=None if n is None else int(n),
                in_sharding=tile, batch_in_sharding=stack)
