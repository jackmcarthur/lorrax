"""Backend dispatch for the distributed dense-linalg call sites.

One function per operation, shared by every consumer, so the backend
names (``auto|off|cusolvermp|slate``) and the per-backend output
conventions are defined exactly once.  Consumers today:

  * ``bse.vq_interp.prepare_coarse``      — the coarse C_q tiles (n_μ²)
  * ``bandstructure.bse_setup.compute_wfns_fi`` — the htransform fH_q
    (rank², rank ≈ nspinor·n_μ)

Do not add a second dispatcher next to this one; add the operation here.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax.sharding import Mesh

__all__ = ["dispatch_eigh", "EIGH_BACKENDS"]

EIGH_BACKENDS = ("auto", "off", "cusolvermp", "slate")


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
        One of ``EIGH_BACKENDS``.

    Returns
    -------
    (lam, R)
        Ascending eigenvalues and TRUE column eigenvectors
        (``A @ R == R @ diag(lam)``): the cusolvermp wrapper's raw Q is
        conj-transposed here (its documented layout), SLATE already
        returns columns.  ``R`` is left sharded ``P('x','y')`` on the FFI
        paths and in whatever layout XLA picks on the native path.
    """
    if backend in ("auto", "off"):
        return jnp.linalg.eigh(A)
    if A.ndim != 2:
        raise ValueError(
            f"eigh backend {backend!r} takes ONE (n, n) matrix — got shape "
            f"{A.shape}.  The FFI backends distribute a single tile over the "
            f"mesh; batch by looping the caller, not by a leading axis.")
    if backend == "cusolvermp":
        from ..cusolvermp.eigh import distributed_eigh
        lam, Qraw = distributed_eigh(A, mesh=mesh_xy)
        return lam, jnp.conj(Qraw).T       # raw buffer → column eigenvectors
    if backend == "slate":
        from ..slate.eigh import distributed_eigh
        lam, Q = distributed_eigh(A, mesh=mesh_xy)
        return lam, Q                      # already true columns
    raise ValueError(
        f"eigh backend must be one of {'|'.join(EIGH_BACKENDS)}, got {backend!r}")
