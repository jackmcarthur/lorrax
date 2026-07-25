"""Call-time dispatch for the distributed dense-linalg facade.

One function per operation, shared by every consumer, so the backend
names and the per-backend output conventions are defined exactly once.
Consumers today:

  * ``bse.vq_interp.prepare_coarse``      — the coarse C_q tiles (n_μ²)
  * ``bandstructure.bse_setup.compute_wfns_fi`` — the htransform fH_q
    (rank², rank ≈ nspinor·n_μ)

Do not add a second dispatcher next to this one; add the operation here.
Backend *resolution* (guards, capability probing, auto policy) lives in
``ffi.linalg.resolve`` — this module only routes an already-legal call.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax.sharding import Mesh

from .resolve import EIGH_BACKENDS, NATIVE, backend_module, resolve_backend

__all__ = ["dispatch_eigh", "EIGH_BACKENDS"]


def dispatch_eigh(A, mesh_xy: Mesh, backend: str):
    """One Hermitian eigendecomposition, backend-dispatched.

    ``off``        → native ``jnp.linalg.eigh``.  Accepts a LEADING BATCH
                     axis and is the only backend that does: every device
                     eigh-decomposes the matrices of its own shard of that
                     axis, which beats a distributed single-tile solve
                     whenever a whole matrix fits on one device.  Also
                     what ``auto`` resolves to, at every call site.
    ``cusolvermp`` → cuSOLVERMp FFI, ONE ``(n, n)`` tile spread over the
                     whole mesh.  Square 2-D mesh, ``n`` divisible by both
                     axis sizes.
    ``slate``      → SLATE FFI, same one-tile-per-call geometry; portable
                     backend (also has CPU-platform handlers).

    Both FFI backends run one JAX process per GPU — which is how LORRAX
    runs, so it is not a restriction, it is the architecture.  What DOES
    decide between the backends is measured cost: the native path solves
    ndev matrices at once, the FFI path solves one matrix ndev-ways and
    walks the batch serially.  So ``off`` wins by roughly ndev on a long
    batch of tiles that fit, and the FFI backends win when a single tile
    does not fit on one device at all.  Measured (``common.eigh_benchmark
    --mode dispatch``, complex128, 2×2 mesh of A100-80GB, native batch 32)
    the FFI is 640×/249×/281×/94× slower PER MATRIX at n =
    512/1024/2048/4096 for cusolvermp and 221×/164×/216× at n =
    512/1024/2048 for slate — fixed-cost dominated, so ``off`` is the
    default everywhere.  See
    ``reports/htransform_distributed_eigh_2026-07-21/report.md``.

    Parameters
    ----------
    A
        ``(n, n)`` Hermitian, sharded ``P('x','y')`` for the FFI
        backends; ``(..., n, n)`` in any layout for ``off``.
    mesh_xy
        ('x','y') device mesh.
    backend
        One of ``EIGH_BACKENDS`` (or an already-resolved ``'native'``).
        Explicit FFI names go through ``resolve_backend`` first, so a
        CPU mesh, an uncompiled handler, or a rectangular mesh is
        rejected here with the resolver's message instead of failing
        (or deadlocking) inside the FFI call.

    Returns
    -------
    (lam, R)
        Ascending eigenvalues and TRUE column eigenvectors
        (``A @ R == R @ diag(lam)``): the cusolvermp wrapper's raw Q is
        conj-transposed here (its documented layout), SLATE already
        returns columns.  ``R`` is left sharded ``P('x','y')`` on the FFI
        paths and in whatever layout XLA picks on the native path.
    """
    if backend in ("auto", "off", NATIVE):
        return jnp.linalg.eigh(A)
    resolved = resolve_backend("eigh", backend, mesh_xy)
    if resolved == NATIVE:                 # (unreachable today: eigh has no
        return jnp.linalg.eigh(A)          # silent FFI→native fallback)
    if A.ndim != 2:
        raise ValueError(
            f"eigh backend {backend!r} takes ONE (n, n) matrix — got shape "
            f"{A.shape}.  The FFI backends distribute a single tile over the "
            f"mesh; batch by looping the caller, not by a leading axis.")
    if resolved == "cusolvermp":
        lam, Qraw = backend_module("cusolvermp").distributed_eigh(A, mesh=mesh_xy)
        return lam, jnp.conj(Qraw).T       # raw buffer → column eigenvectors
    lam, Q = backend_module("slate").distributed_eigh(A, mesh=mesh_xy)
    return lam, Q                          # already true columns
