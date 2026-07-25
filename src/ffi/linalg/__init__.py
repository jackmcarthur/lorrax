"""Distributed dense linear algebra over a JAX device mesh — the facade.

One interface from JAX-side calls to the choice of library, depending on
what is compiled, what the input file / CLI requested, and whether the
mesh is GPU or CPU.  Full docs: ``docs/dev/linalg_ffi.md``.

Three layers:

* **resolve** (``resolve_backend``, ``list_backends``) — turns a
  requested backend name (``auto|off|cusolvermp|slate|scalapack``, per
  op) into a concrete, guaranteed-callable backend, applying every guard
  (platform, compiled-capability probe, process coverage, mesh geometry,
  divisibility) in one place at RESOLVE time.
* **dispatch** (``dispatch_eigh``) — routes one call to the resolved
  backend and normalizes output conventions.
* **backends** — ``ffi.cusolvermp`` (CUDA), ``ffi.slate`` (CUDA + host),
  ``ffi.scalapack`` (host), reached only through ``backend_module``; and
  the in-tree ``native`` implementations (pure JAX — ``jnp.linalg.eigh``,
  the replicated/sharded Cholesky and per-q LU in ``isdf/core``), which
  are first-class backends available everywhere.

Typical use::

    from ffi import linalg

    resolved = linalg.resolve_backend("eigh", requested, mesh, n=n)
    if resolved == linalg.NATIVE:
        lam, R = jnp.linalg.eigh(A_batch)          # q-batched, sharded
    else:
        lam, R = linalg.dispatch_eigh(A_tile, mesh, resolved)

    print(linalg.list_backends("cholesky", mesh))  # what CAN run here?
"""
from .dispatch import dispatch_eigh
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
    "NATIVE",
    "OPS",
    "backend_module",
    "dispatch_eigh",
    "list_backends",
    "mesh_is_cpu",
    "mesh_platform",
    "resolve_backend",
]
