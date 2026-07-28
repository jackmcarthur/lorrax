"""Call-time dispatch for the distributed dense-linalg facade.

One function per operation, so the backend names and the per-backend
output conventions are defined exactly once.  ``dispatch_eigh`` is a
retained back-compat entry point (external scripts via the
``ffi.common.dispatch`` shim): the former in-tree consumers —
``bse.vq_interp.prepare_coarse`` and
``bandstructure.bse_setup.compute_wfns_fi`` — both migrated to the plan
facade (``linalg_plan("eigh", ...)``) and no in-tree caller remains; the
only in-repo coverage is ``tests/test_ffi_linalg_contract.py``.

Do not add a second dispatcher next to this one; add the operation here.
Backend *resolution* (guards, capability probing, auto policy) lives in
``ffi.linalg.resolve`` — this module only routes an already-legal call.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax.sharding import Mesh

from .plan import plan as _plan
from .resolve import EIGH_BACKENDS, NATIVE

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
    ``slate``      → SLATE FFI, same one-tile-per-call geometry.  CUDA
                     only in practice: the CPU handler SIGSEGVs and is
                     refused at resolve time (bug L-2).
    ``scalapack``  → ScaLAPACK ``pzheevd`` FFI, host-only, square or 1-D
                     mesh.  The permanent CPU distributed backend; also
                     what ``distributed`` resolves to there.

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
    p = _plan("eigh", mesh_xy, backend=backend)
    if p.is_native:                        # (unreachable: resolve has NO
        return jnp.linalg.eigh(A)          # silent FFI→native fallback for
                                           # any op — explicit requests
                                           # refuse at resolve time, and
                                           # auto/off returned above)
    if A.ndim != 2:
        raise ValueError(
            f"eigh backend {backend!r} takes ONE (n, n) matrix — got shape "
            f"{A.shape}.  The FFI backends distribute a single tile over the "
            f"mesh; batch by looping the caller, not by a leading axis.")
    # ``LinalgPlan`` owns the operand reshard and the per-backend
    # eigenvector-layout normalisation (cuSOLVERMp's raw buffer is the
    # conjugate transpose of the column form; slate/scalapack are columns).
    return p(A)
