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

from .plan import _eigh_columns, ensure_sharding, plan as _plan
from .resolve import EIGH_BACKENDS, NATIVE

__all__ = ["dispatch_eigh", "dispatch_batched_eigh", "EIGH_BACKENDS"]

#: The STACKED-form entry point a backend may or may not expose.  Probed
#: with ``getattr`` in :func:`dispatch_batched_eigh` — see that docstring
#: for why a name probe and not a platform test.
_BATCHED_ENTRY = "batched_distributed_eigh"


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


def dispatch_batched_eigh(A, mesh_xy: Mesh, backend: str = "distributed",
                          *, _force_serial: bool = False):
    """A STACK of Hermitian matrices, backend-dispatched.

    Parameters
    ----------
    A : (Nq, N, N)  P(None, 'x', 'y')   Hermitian (lower triangle read)
    mesh_xy : ('x','y') device mesh
    backend : any ``EIGH_BACKENDS`` name; ``'distributed'`` resolves to the
        platform's distributed eigh (ScaLAPACK on host, cuSOLVERMp on CUDA).
    _force_serial : test-only.  Skip the batched entry even when the
        backend has one, so the fallback is reachable where both paths
        are — see "GATE" below.  Never pass it from production code.

    Returns
    -------
    (W, Z)
        ``W`` ``(Nq, N)`` REPLICATED float64 ascending; ``Z``
        ``(Nq, N, N)`` ``P(None,'x','y')`` eigenvectors as COLUMNS
        (``A[q] @ Z[q] == Z[q] @ diag(W[q])``).  Same layout contract as
        ``_scalapack.batched_distributed_eigh``, whatever backend ran.

    WHY THIS EXISTS.  Batching is not universal: only the ScaLAPACK
    backend exposes ``batched_distributed_eigh``, which costs ONE
    collective-serialisation round for the whole stack instead of ``Nq``.
    cuSOLVERMp and SLATE expose the single-matrix ``distributed_eigh``
    only.  A caller that reaches for the batched name directly therefore
    works on a host mesh and ``AttributeError``s on CUDA.  This is the one
    place that asymmetry is handled.

    The capability test is ``getattr``, not a platform test, so a backend
    that gains a batched entry later is picked up with no edit here.  The
    price of that promise is paid in ``plan._eigh_columns``, which had to
    become rank-agnostic: a future cuSOLVERMp batched entry would hand a
    ``(nq, n, n)`` stack to a normaliser written for one tile.

    ONE RESOLVE, NOT Nq — but the interesting part is what that did NOT
    fix.  Everything independent of ``q`` is a
    :class:`~ffi.linalg.plan.LinalgPlan` built once here: the guard ladder
    (platform, known-broken combination, compiled-handler probe, process
    coverage, mesh geometry, ``N`` divisibility), the ``P('x','y')`` /
    ``P(None,'x','y')`` shardings, the module lookup, and the per-backend
    eigenvector-layout normaliser.  Gate check 5 counts it: ONE resolve
    for a 6-matrix serial dispatch.  The placeholder already called
    ``resolve_backend`` outside its loop, so that half was not the defect.
    The defect was the normaliser, which is also a per-``q`` decision and
    was simply absent: ``mod.distributed_eigh`` returns the backend's RAW
    eigenvector buffer, and on CUDA that is cuSOLVERMp's conjugate
    transpose (``ffi/cusolvermp/eigh.py`` Layout note), so the serial
    path returned ROWS and silently broke the COLUMNS half of the contract
    above — on the only platform where that path is ever taken.  Routing
    the loop body through ``plan(A[q])`` fixes it by construction, because
    the plan is the object that owns ``_eigh_columns``.

    What deliberately still runs per ``q`` is the backend wrapper's OWN
    call-time validation (``_scalapack.validate_eigh_mesh`` + platform +
    divisibility, ``_scalapack.py:121-136``).  That is not an oversight to
    hoist: it is the bug L-1 mirror — a rule enforced only at resolve time
    turns a returned backend name into a broken promise — and it costs a
    few int comparisons plus a cached context lookup, not the resolve
    ladder's symbol probe and ``jax.process_count``.

    DONATION: the serial fallback does not donate, must not, and has
    nothing to preserve.  Neither eigh backend donates its operand in the
    first place — ScaLAPACK's handler stages each q into scratch because
    ``pXheevd`` destroys its input (``_scalapack.py`` module docstring;
    ``jax.jit(_eigh)`` at ``_scalapack.py:175`` takes no ``donate_argnums``)
    and the cuSOLVERMp wrapper has neither ``donate_argnums`` nor
    ``input_output_aliases`` (``ffi/cusolvermp/eigh.py:136-146``).  Donating
    the STACK across a Python loop is incoherent regardless: the q-th call
    would invalidate the buffer the q+1-th call still has to slice.  The
    cost this leaves is a transient, not a leak — ``A`` stays live for the
    caller, and the Nq ``Z[q]`` shards are held until ``jnp.stack``
    copies them, so the serial path peaks at ~2x the Z stack (per rank,
    Nq·N²/P elements) where the batched path peaks at 1x.  That is a
    reason to prefer a backend WITH a stacked entry, not a reason to
    donate.

    GATE: ``tests/multi_device/batched_eigh_dispatch_gate.py`` (srun, any
    square host mesh).  A CPU mesh is the only place both paths are
    reachable at all, since ScaLAPACK is the only backend with a stacked
    entry; the serial path is forced there by the private
    ``_force_serial`` argument rather than by deleting the attribute from
    the module.  Reason: ``backend_module('scalapack')`` deliberately
    hands out the ``ffi.scalapack`` re-export shim, not
    ``ffi.linalg._scalapack`` (``_scalapack.py`` module docstring), so a
    monkeypatch has to know which of the two module objects to hit and
    would silently stop forcing anything the day that transitional shim is
    removed — a gate that quietly tests the same path twice.  The private
    argument also keeps the gate on the real entry point.

    Measured job 7889132, 2x2 host mesh (2 nodes, P=4), Nq=6, N=32,
    complex128, all 7 checks PASS: the two paths are BIT-IDENTICAL in both
    W and Z (0 ulp, not merely 1e-12 — same routine, same descriptors,
    same grid, only the ``nq`` the handler loops over differs);
    ``A[q] Z[q] == Z[q] diag(W[q])`` holds at 1.849e-15 relative on BOTH
    paths with the conj-transposed control at 1.636e+00 (without that
    control the residual check passes on a dispatcher that returns rows,
    which is what the raw fallback did); eigenvalues match
    ``numpy.linalg.eigvalsh`` at 1.763e-15; resolution runs EXACTLY ONCE
    for a 6-matrix serial dispatch; the indivisible-N and off-platform
    guards still refuse; and ``qsgw_density.distributed_eigh_bands`` is
    unchanged.

    LAX.SCAN: it IS scan-safe.  The belief recorded in the placeholder
    ("the FFI call is not scan-safe") is FALSE on this stack and is
    written down here so it stops being repeated.  Job 7889132, probe leg,
    same mesh: a ``lax.scan`` whose body is
    ``_scalapack.distributed_eigh`` TRACES (``jax.eval_shape``), LOWERS,
    COMPILES and RUNS — ``shard_map`` and ``jax.ffi.ffi_call`` both have
    the rules ``scan`` needs, and the collective inside the XLA while body
    did not deadlock at P=4.  Its W is bit-identical to this function's
    batched path, and that path is itself 9.6e-16 away from
    ``jnp.linalg.eigh``, so the agreement is the FFI running, not a silent
    demote to native.  Per-call cost on that leg, warm: scan 12.2 ms,
    Python loop 21.7 ms, batched entry 11.3 ms (reproduced on job
    7889130: 11.3 / 21.6 / 11.1).

    The loop stays Python anyway, and the reason is coverage, not
    capability.  The serial path is DEAD on the one backend the probe
    could reach: ScaLAPACK has a stacked entry, so ``getattr`` always
    finds it and the fallback never runs in production on a host mesh.
    The fallback exists for cuSOLVERMp and SLATE, whose wrappers are a
    different shape (``ffi/cusolvermp/eigh.py:136-146`` calls
    ``shard_map`` at eager top level with no ``jax.jit`` around it) over
    NCCL + cal_comm, and no CUDA mesh was available to probe.  Shipping a
    collective inside an XLA while body on the strength of a CPU-only
    measurement invites the failure class this repo has lost the most time
    to — not this construction, but its shape: ``resolve`` guard 5 exists
    because cusolverMpSyevd HANGS inside a collective on a rectangular
    grid instead of returning an error status, i.e. the CUDA distributed
    eigh's failure mode is a hang with no traceback, not an exception.  A
    scan would also not recover what the batched entry buys — that saving
    is in C++, where the handler takes ``nq`` as a compile-time attribute
    and loops q around ONE descriptor and ONE workspace
    (``_scalapack.py:152``) — so it is worth revisiting only for the
    CUDA fallback, and only with a CUDA measurement.  The 1.8x above is a
    Python-dispatch saving (~1.6 ms/matrix) at a size where 6 32x32 eigh
    solves are plainly not FLOP-bound; it will shrink as N grows.
    """
    if backend in ("auto", "off", NATIVE):
        return jnp.linalg.eigh(A)
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"dispatch_batched_eigh: expected (Nq, N, N); got {A.shape}")
    p = _plan("eigh", mesh_xy, backend=backend, n=int(A.shape[1]))
    if p.is_native:                        # (unreachable, exactly as in
        return jnp.linalg.eigh(A)          # dispatch_eigh above)
    A = ensure_sharding(A, p.batch_in_sharding)
    batched = (None if _force_serial
               else getattr(p.module, _BATCHED_ENTRY, None))
    if batched is not None:
        return _eigh_columns(p.backend, *batched(A, mesh=mesh_xy))
    # Serial fallback.  ``p(A[q])`` reshards the slice to P('x','y') (a
    # no-op given the constraint above) and applies the layout normaliser.
    W, Z = zip(*(p(A[q]) for q in range(int(A.shape[0]))))
    return (jnp.stack(W),
            ensure_sharding(jnp.stack(Z), p.batch_in_sharding))
