"""Call-time dispatch: the one asymmetry a plan cannot hide.

Ported from LORRAX ``src/ffi/linalg/dispatch.py`` at 96a6399, minus
``dispatch_eigh`` (deleted with this extraction — it had no in-tree caller
and no test).  Backend *resolution* lives in :mod:`distrib_la.resolve`;
this module only routes an already-legal call.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax.sharding import Mesh

from distrib_la.plan import ROUTE_SCAN, ensure_sharding, plan as _plan
from distrib_la.resolve import EIGH_BACKENDS, NATIVE

__all__ = ["dispatch_batched_eigh", "EIGH_BACKENDS"]


def dispatch_batched_eigh(A, mesh_xy: Mesh, backend: str = "distributed",
                          *, _force_serial: bool = False):
    """A STACK of Hermitian matrices, backend-dispatched.

    Parameters
    ----------
    A : (Nq, N, N)  P(None, 'x', 'y')   Hermitian (lower triangle read)
    mesh_xy : ('x','y') device mesh
    backend : any ``EIGH_BACKENDS`` name; ``'distributed'`` resolves to the
        platform's distributed eigh (ScaLAPACK on host, cuSOLVERMp on CUDA).
    _force_serial : test-only.  Force the scan route even when the backend
        has a stacked entry, so both routes are reachable on one mesh —
        see "GATE" below.  Never pass it from production code.

    Returns
    -------
    (W, Z)
        ``W`` ``(Nq, N)`` REPLICATED float64 ascending; ``Z``
        ``(Nq, N, N)`` ``P(None,'x','y')`` eigenvectors as COLUMNS
        (``A[q] @ Z[q] == Z[q] @ diag(W[q])``).  Same layout contract as
        ``_scalapack.batched_distributed_eigh``, whatever backend ran.

    WHY THIS STILL EXISTS.  It does not choose a route any more — it is
    ``plan(...).batched(A)`` with the eigh layout contract reasserted on
    the way out, kept because ``gw.qsgw_density`` calls it by this name.
    Batching is not universal (only the ScaLAPACK backend exposes
    ``batched_distributed_eigh``; cuSOLVERMp and SLATE expose the
    single-matrix ``distributed_eigh`` only), but that asymmetry is now
    :attr:`distrib_la.plan.Plan.batched_route`'s business and is stated
    once, there, for every op rather than once here for eigh.

    The capability test is the ``_IMPL`` table, not a ``getattr`` probe on
    the module.  It used to be the probe, on the argument that a backend
    which grew a stacked entry would then be picked up with no edit here —
    but "no edit here" was only true because this function was a second
    route selector, and a second route selector is the thing the design
    forbids.  A new stacked entry is one ``_IMPL`` row, which is already
    what ``docs/dev/linalg_ffi.md`` § "Adding a backend" tells you to
    write.  What survives from that argument is
    :func:`distrib_la.plan._eigh_columns` staying RANK-AGNOSTIC: the day
    cuSOLVERMp grows a batched entry, the normaliser starts receiving
    ``(nq, n, n)`` stacks, and a ``.T`` would silently have returned
    ``(n, n, nq)``.

    ONE RESOLVE, NOT Nq — but the interesting part is what that did NOT
    fix.  Everything independent of ``q`` is a :class:`~distrib_la.plan.Plan`
    built once here: the guard ladder, the ``P('x','y')`` /
    ``P(None,'x','y')`` shardings, the module lookup, and the per-backend
    eigenvector-layout normaliser.  The placeholder already called
    ``resolve_backend`` outside its loop, so that half was not the defect.
    The defect was the normaliser, which is also a per-``q`` decision and
    was simply absent: ``mod.distributed_eigh`` returns the backend's RAW
    eigenvector buffer, and on CUDA that is cuSOLVERMp's conjugate
    transpose, so the serial path returned ROWS and silently broke the
    COLUMNS half of the contract above — on the only platform where that
    path is ever taken.  The scan body applies ``_eigh_columns`` per matrix
    for the same reason the loop body did.

    What deliberately still runs per ``q`` is the backend wrapper's OWN
    call-time validation (``_scalapack.validate_eigh_mesh`` + platform +
    divisibility).  That is not an oversight to hoist: it is the bug L-1
    mirror — a rule enforced only at resolve time turns a returned backend
    name into a broken promise — and under the scan it costs LESS than it
    did under the loop, because the body is traced once: the int
    comparisons and the cached context lookup happen at trace time, once,
    not once per matrix.

    DONATION IS UNCHANGED, and the reason is unchanged: neither eigh
    backend donates its operand (:data:`distrib_la.plan.DONATES`).
    ScaLAPACK's handler stages each q into scratch because ``pXheevd``
    destroys its input, and the cuSOLVERMp wrapper has neither
    ``donate_argnums`` nor ``input_output_aliases``.  What the scan does
    change is the transient the old loop left: the Python loop held all
    ``Nq`` ``Z[q]`` shards alive until ``jnp.stack`` copied them, peaking
    at ~2x the Z stack, and the scan writes each result into the output
    stack as it goes.

    GATE: ``tests/multi_device/batched_eigh_dispatch_gate.py`` and its
    adopted twin ``check_batched_eigh_dispatch`` in the service's L-c
    suite (srun, any square host mesh).  A CPU mesh is the only place both
    routes are reachable at all, since ScaLAPACK is the only backend with
    a stacked entry; the scan route is forced there by the private
    ``_force_serial`` argument, which is the plan's ``_route`` override
    under an older name.

    Measured job 7889132, 2x2 host mesh (2 nodes, P=4), Nq=6, N=32,
    complex128, all 7 checks PASS: the two paths are BIT-IDENTICAL in both
    W and Z (0 ulp, not merely 1e-12 — same routine, same descriptors, same
    grid, only the ``nq`` the handler loops over differs);
    ``A[q] Z[q] == Z[q] diag(W[q])`` holds at 1.849e-15 relative on BOTH
    paths with the conj-transposed control at 1.636e+00 (without that
    control the residual check passes on a dispatcher that returns rows,
    which is what the raw fallback did); eigenvalues match
    ``numpy.linalg.eigvalsh`` at 1.763e-15; resolution runs EXACTLY ONCE
    for a 6-matrix serial dispatch; and the indivisible-N and off-platform
    guards still refuse.

    LAX.SCAN: it IS scan-safe, and this is where that was first measured.
    The belief recorded in the original placeholder ("the FFI call is not
    scan-safe") is FALSE on this stack.  Job 7889132, probe leg, 2x2 host
    mesh: a ``lax.scan`` whose body is ``_scalapack.distributed_eigh``
    TRACES, LOWERS, COMPILES and RUNS, and the collective inside the XLA
    while body did not deadlock at P=4.  Per-call cost, warm, Nq=6 N=32:
    scan 12.2 ms, Python loop 21.7 ms, batched entry 11.3 ms (reproduced
    on job 7889130: 11.3 / 21.6 / 11.1) — the scan is 1.8x the Python loop
    and within 8% of the C++ stacked entry.

    That measurement is why the loop is now the scan.  The earlier
    reservation was coverage: ScaLAPACK is the only backend the CPU probe
    could reach, and the serial route only ever RUNS on cuSOLVERMp and
    SLATE, over NCCL + cal_comm, where the failure mode is a HANG with no
    traceback (guard 5 exists because ``cusolverMpSyevd`` hangs inside a
    collective on a rectangular grid instead of returning an error
    status).  Closing that gap is a MEASUREMENT and not an argument: the
    gate below runs the scan route on a real CUDA mesh, and § Performance
    in ``docs/services/distrib_la.md`` carries what it cost.

    What the scan does NOT do is recover what the stacked entry buys.
    That saving is in C++, where the handler takes ``nq`` as a
    compile-time attribute and loops q around ONE descriptor and ONE
    workspace, so ScaLAPACK eigh keeps
    :data:`~distrib_la.plan.ROUTE_BACKEND_BATCHED`.
    """
    if backend in ("auto", "off", NATIVE):
        return jnp.linalg.eigh(A)
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError(
            f"dispatch_batched_eigh: expected (Nq, N, N); got {A.shape}")
    p = _plan("eigh", mesh_xy, backend=backend, n=int(A.shape[1]))
    if p.is_native:                        # (unreachable: resolve has NO
        return jnp.linalg.eigh(A)          # silent FFI→native fallback for
                                           # any op — explicit requests
                                           # refuse, and auto/off returned
                                           # above)
    W, Z = p.batched(A, _route=ROUTE_SCAN if _force_serial else None)
    # Reassert the eigenvector layout this entry point promises.  On
    # ROUTE_BACKEND_BATCHED it is already the shard_map's ``out_specs`` and
    # ``ensure_sharding`` returns Z untouched; the line is here so the
    # promise does not depend on which route ran.
    return W, ensure_sharding(Z, p.batch_in_sharding)
