"""``distrib_la`` — distributed dense linear algebra over a JAX device mesh.

One door for ``eigh``, ``cholesky`` and ``solve_lu`` on an ``('x','y')``
device mesh, over four backend families: **scalapack** (CPU preferred),
**slate** (CPU fallback where it is not broken; ROCm always,
declared-untested), **cusolvermp** (CUDA preferred) and **native** (pure
JAX, everywhere).  A caller says what it wants computed and on which mesh;
which library runs is a resolved fact it can read but never has to
branch on.

THE PACKAGE IS THE DOOR.  There is no separate facade module: everything
a consumer needs is a top-level name here, and importing
``distrib_la.<submodule>`` from outside is a layering violation the
monorepo's ``tests/test_layering.py`` fails on.  That is what makes
"only distrib_la sees scalapack/slate/cusolvermp" checkable rather than
aspirational — those three appear in exactly one dependency edge in this
package (:mod:`distrib_la.loader`, which dlopens a ``.so`` by path) and in
zero of its declared dependencies.

The surface
-----------
``plan(op, mesh, *, backend='auto', n=None) -> Plan``
    Resolve once, then call.  ``Plan(A)`` for one tile at ``P('x','y')``,
    ``Plan.batched(A_stack)`` for ``P(None,'x','y')``.  Eigenvalues come
    back replicated, eigenvectors as COLUMNS, on every backend.
``Plan.batched_route`` / ``BATCHED_ROUTES`` / ``BATCHED_SCAN_UNROLL``
    The batched surface is a ``lax.scan`` over the single-matrix op, and
    ``batched_route`` is the ONE place that says how a given stack will
    actually run: the scan, or the backend's own stacked entry where the
    library has one (a backend-internal optimization behind the same
    interface, never a second surface).  A third route — move the batch
    axis onto the mesh and run the local kernel — is reserved and named
    there.  Introspection: a caller reads it, never branches on it.
``Plan.native_fn``
    A pure, trace-safe closure — native backends only — for a call site
    that needs the math INSIDE its own ``jit``.
``factor(op, A, mesh, ...) -> FactorToken`` / ``solve(token, B)``
    Factor once, back-solve many.  The token is opaque and carries the
    handle (scalapack's ``ipiv``, cuSOLVERMp's raw buffer, SLATE's
    ``SlateLowerL``), so "never reshard, feed it back verbatim" is the
    type rather than a comment.
``rank_cut(op, values, *, rcond|n_keep, closure=...) -> RankCut``
    The rank-revealing surface: cut a Cholesky's pivot spectrum
    (:func:`cholesky_pivot_spectrum`), an LU's ``|diag(U)|``
    (:func:`lu_rank_spectrum`) or an eigenvalue vector, with the
    DEGENERACY-CLOSURE guard.  A cut that would stop inside a degenerate
    cluster keeps a symmetry-arbitrary slice of an eigenspace; the guard
    moves it OUTWARD past the cluster (``snap``), refuses (``strict``) or
    is not there (``off``).  Also reachable as ``Plan.rank_cut`` /
    ``FactorToken.rank_cut``, which bind the op and the mode.

    **OPT-IN, and OFF by default this round.**  ``closure=`` on
    :func:`plan` / :func:`factor`, or ``LORRAX_DISTRIB_LA_CLOSURE``.
    Nothing about resolution, routing, sharding or the bytes a
    factorization returns depends on it.  :mod:`distrib_la.closure` has
    the criterion, the copy-not-import argument, and the owner row on
    the default.
``resolve_backend(op, requested, mesh, *, n=None) -> str``
    The raising probe.  ``n`` is decoupled from the operands on purpose:
    a caller that will pad can ask before it has built anything.
``list_backends(op, mesh) -> dict``
    The never-raising report, for startup banners.
``BACKEND_CHOICES``
    The vocabulary.  Importable with NO ``.so`` anywhere on the machine —
    a deck parser must not need the FFI layer to read a deck.
``dial_key() -> tuple``
    The factory-time cache-key aggregate.
``mesh_key(mesh) -> tuple``
    A stable, hashable mesh identity (axes, extents, platform, device ids)
    for any cache whose stored value does NOT retain the mesh.

Two phases, and they stay two
-----------------------------
``plan()`` is EAGER (it dlopens and reads ``jax.process_count()``); what it
returns is TRACE-SAFE.  Only platform and handler guards can fire at
resolve time — operand dtype, rank and extent are trace-time facts — so a
single-phase API would have to lie about when it checked.

Promise semantics
-----------------
A returned backend name means EVERY guard passed; the call cannot then
fail for an availability or geometry reason.  Explicit requests refuse
rather than downgrade; only ``auto`` demotes, and it announces on rank 0.
The worst measured defect in this tree was a silent route change that ran
to completion with rc=0 and a QP gap of −161 eV, so this is not
punctilio — it is the failure mode.
"""

from __future__ import annotations

from distrib_la.closure import (
    DEFAULT_MODE as CLOSURE_DEFAULT_MODE,
    DEFAULT_RTOL as CLOSURE_RTOL,
    MODE_ENV as CLOSURE_ENV,
    MODES as CLOSURE_MODES,
    RANK_SPECTRA,
    RankCut,
    SpectralClusterError,
    cholesky_pivot_spectrum,
    lu_rank_spectrum,
    rank_cut,
)
from distrib_la.dispatch import dispatch_batched_eigh
from distrib_la.factor import FactorToken, factor, solve
from distrib_la.loader import dial_key, has_target, probe_target
from distrib_la.plan import (
    BATCHED_ROUTES,
    BATCHED_SCAN_UNROLL,
    DONATES,
    ROUTE_BACKEND_BATCHED,
    ROUTE_BATCH_RESHARD,
    ROUTE_SCAN,
    Plan,
    ensure_sharding,
    plan,
)
from distrib_la.resolve import (
    BACKEND_CHOICES,
    CHOLESKY_BACKENDS,
    EIGH_BACKENDS,
    LU_BACKENDS,
    NATIVE,
    OPS,
    backend_module,
    list_backends,
    mesh_is_cpu,
    mesh_key,
    mesh_platform,
    resolve_backend,
)

__all__ = [
    # plan
    "Plan", "plan", "ensure_sharding", "DONATES",
    # the batched route toggle and its dial
    "BATCHED_ROUTES", "ROUTE_SCAN", "ROUTE_BACKEND_BATCHED",
    "ROUTE_BATCH_RESHARD", "BATCHED_SCAN_UNROLL",
    # factor / solve
    "FactorToken", "factor", "solve",
    # the rank-revealing surface and its degeneracy-closure guard
    "rank_cut", "RankCut", "cholesky_pivot_spectrum", "lu_rank_spectrum",
    "SpectralClusterError", "CLOSURE_MODES", "CLOSURE_DEFAULT_MODE",
    "CLOSURE_RTOL", "CLOSURE_ENV", "RANK_SPECTRA",
    # resolution
    "resolve_backend", "list_backends", "backend_module",
    "BACKEND_CHOICES", "EIGH_BACKENDS", "CHOLESKY_BACKENDS", "LU_BACKENDS",
    "OPS", "NATIVE", "mesh_platform", "mesh_is_cpu", "mesh_key",
    # capability
    "probe_target", "has_target", "dial_key",
    # dispatch
    "dispatch_batched_eigh",
]
