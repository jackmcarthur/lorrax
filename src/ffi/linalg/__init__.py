"""Distributed dense linear algebra over a JAX device mesh — the facade.

One interface from JAX-side calls to the choice of library, depending on
what is compiled, what the input file / CLI requested, and whether the
mesh is GPU or CPU.  Full docs: ``docs/dev/linalg_ffi.md``.

Four layers:

* **resolve** (``resolve_backend``, ``list_backends``) — turns a
  requested backend name (``auto|off|distributed|cusolvermp|slate|
  scalapack``, per op) into a concrete, guaranteed-callable backend,
  applying every guard (platform, known-broken combinations,
  compiled-capability probe, process coverage, mesh geometry,
  divisibility) in one place at RESOLVE time.
* **plan** (``plan`` → :class:`LinalgPlan`) — THE call-site interface.
  Resolves once and carries the answer together with the layout
  contract, so a call site is ``plan(A)`` / ``plan.batched(A)`` with the
  FFI-adjacent resharding, the batch loop and the per-backend output
  conventions all inside.
* **dispatch** (``dispatch_eigh``) — the older single-call entry point,
  now a thin shim over a plan.  New code should take a plan.
* **backends** — ``ffi.cusolvermp`` (CUDA), ``ffi.slate`` (CUDA + host),
  ``ffi.scalapack`` (host), reached only through ``backend_module``; and
  the in-tree ``native`` implementations (pure JAX — ``jnp.linalg.eigh``,
  the replicated/sharded Cholesky and per-q LU in ``isdf/core``), which
  are first-class backends available everywhere.

Typical use::

    from ffi import linalg

    p = linalg.plan("eigh", mesh, backend=requested, n=n)   # once
    log(p.describe())
    if p.is_native:
        lam, R = jnp.linalg.eigh(A_batch)   # caller owns the fused fast path
    else:
        lam, R = p.batched(A_stack)         # one call, any backend

    print(linalg.list_backends("cholesky", mesh))  # what CAN run here?
"""
from .dispatch import dispatch_batched_eigh, dispatch_eigh
from .plan import LinalgPlan, ensure_sharding, plan
from .resolve import (
    BACKEND_CHOICES,
    CHOLESKY_BACKENDS,
    EIGH_BACKENDS,
    LU_BACKENDS,
    NATIVE,
    OPS,
    backend_module,
    list_backends,
    mesh_is_cpu,
    mesh_platform,
    resolve_backend,
)

__all__ = [
    "BACKEND_CHOICES",
    "CHOLESKY_BACKENDS",
    "EIGH_BACKENDS",
    "LU_BACKENDS",
    "LinalgPlan",
    "NATIVE",
    "OPS",
    "backend_module",
    "dispatch_batched_eigh",
    "dispatch_eigh",
    "ensure_sharding",
    "list_backends",
    "mesh_is_cpu",
    "mesh_platform",
    "plan",
    "resolve_backend",
]
