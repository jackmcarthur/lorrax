"""``distrib_la`` — distributed dense linear algebra over a JAX device mesh.

One door for ``polar_factor``, ``eigh``, ``cholesky``, ``solve_lu`` and
``matmul`` on an ``('x','y')`` device mesh, over four backend families:
**scalapack/PBLAS** (CPU preferred),
**slate** (CPU fallback where it is not broken; ROCm always,
declared-untested), **cusolvermp/cuBLASMp** (CUDA preferred) and **native**
(pure JAX, everywhere).  A caller says what it wants computed and on which mesh;
which library runs is a resolved fact it can read but never has to
branch on.

THE PACKAGE IS THE DOOR.  There is no separate facade module: everything
a consumer needs is a top-level name here, and importing
``distrib_la.<submodule>`` from outside is a layering violation the
monorepo's ``tests/test_layering.py`` fails on.  That is what makes
"only distrib_la sees the provider families" checkable rather than
aspirational — ScaLAPACK/PBLAS, SLATE, and cuSOLVERMp/cuBLASMp appear in
exactly one dependency edge in this package (:mod:`distrib_la.loader`, which
dlopens a ``.so`` by path) and in zero of its declared dependencies.

The surface
-----------
polar_factor(A, mesh, ...) -> (L, s)
    Square distributed polar/SVD through a Hermitian dilation and one planned
    Hermitian eigensolve.  The planned form separates eager resolution from
    the trace-safe operation used in streamed k-point loops.
``plan(op, mesh, *, backend='auto', n=None, batched_route='auto') -> Plan``
    Resolve once, then call.  ``Plan(A)`` for one tile at ``P('x','y')``,
    ``Plan.batched(A_stack)`` for ``P(None,'x','y')``.  Eigenvalues come
    back replicated, eigenvectors as COLUMNS, on every backend.
``Plan.batched_route`` / ``BATCHED_ROUTES`` / ``BATCHED_SCAN_UNROLL``
    The batched surface is a ``lax.scan`` over the single-matrix op, and
    ``batched_route`` is the ONE place that says how a given stack will
    actually run: the scan, or the backend's own stacked entry where the
    library has one (a backend-internal optimization behind the same
    interface, never a second surface).  The opt-in third route moves the
    batch axis onto the mesh, runs the device-local native kernel, and moves
    matrix outputs back through explicit staged collectives.  Production
    selection is ``BATCHED_ROUTE_CHOICES = ('auto', 'batch_reshard')``.
``Plan.native_fn``
    A pure, trace-safe closure — native backends only — for a call site
    that needs the math INSIDE its own ``jit``.
``matmul(A, B, C=None, *, mesh, alpha=1, beta=0, transa='N', transb='N',
backend='auto', batched_route='auto')``
    Distributed rank-2 or batched rank-3 GEMM in the same face layout as a
    plan.  The default dispatches to cuBLASMp, PBLAS or SLATE; the explicit
    opt-in performs x/y face-to-batch exchanges, local GEMM, then y/x inverse
    exchanges. ``backend='off'`` makes that route provider-free.
``gemm_plan(mesh, *, m, k, n, nq, dtype, backend='auto', alpha=1, beta=0) ->
GemmPlan``
    Resolve, probe, warm and COMPILE one N,N GEMM shape ONCE, for a caller
    that will call it many times from inside its own ``jax.jit``/
    ``lax.scan`` (G construction, per-tau Sigma projection).  ``GemmPlan(A,
    B, C=None, *, out=None)`` is trace-safe through cuBLASMp or ScaLAPACK,
    with one replicated leading batch (holds k) and no transpose modes.
    ``GemmPlan.local_call`` is the cuBLASMp-only manual-shard-map entry.
``factor(op, A, mesh, ...) -> FactorToken`` / ``solve(token, B)``
    Factor once, back-solve many.  The token is opaque and carries the
    handle (scalapack's ``ipiv``, cuSOLVERMp's raw buffer, SLATE's
    ``SlateLowerL``), so "never reshard, feed it back verbatim" is the
    type rather than a comment.
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

from distrib_la.dispatch import dispatch_batched_eigh
from distrib_la.factor import FactorToken, factor, solve
from distrib_la.loader import dial_key, has_target, probe_target
from distrib_la.matmul import (
    MATMUL_BACKEND_CHOICES,
    matmul,
    resolve_matmul_backend,
)
from distrib_la.matmul_plan import GemmPlan, gemm_plan
from distrib_la.plan import (
    BATCHED_ROUTE_CHOICES,
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
from distrib_la.polar import PolarPlan, plan_polar_factor, polar_factor
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
    # polar / SVD
    "PolarPlan", "plan_polar_factor", "polar_factor",
    # distributed matrix multiplication
    "matmul", "resolve_matmul_backend", "MATMUL_BACKEND_CHOICES",
    # planned N,N GEMM (trace-safe, for hot loops)
    "GemmPlan", "gemm_plan",
    # the batched route toggle and its dial
    "BATCHED_ROUTES", "ROUTE_SCAN", "ROUTE_BACKEND_BATCHED",
    "ROUTE_BATCH_RESHARD", "BATCHED_ROUTE_CHOICES", "BATCHED_SCAN_UNROLL",
    # factor / solve
    "FactorToken", "factor", "solve",
    # resolution
    "resolve_backend", "list_backends", "backend_module",
    "BACKEND_CHOICES", "EIGH_BACKENDS", "CHOLESKY_BACKENDS", "LU_BACKENDS",
    "OPS", "NATIVE", "mesh_platform", "mesh_is_cpu", "mesh_key",
    # capability
    "probe_target", "has_target", "dial_key",
    # dispatch
    "dispatch_batched_eigh",
]
