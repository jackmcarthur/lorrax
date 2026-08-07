"""Call-time dispatch: the one asymmetry a plan cannot hide.

Ported from LORRAX ``src/ffi/linalg/dispatch.py`` at 96a6399, minus
``dispatch_eigh`` (deleted with this extraction — it had no in-tree caller
and no test).  Backend *resolution* lives in :mod:`distrib_la.resolve`;
this module only routes an already-legal call.
"""
from __future__ import annotations

import jax.numpy as jnp
from jax.sharding import Mesh

from distrib_la.plan import _eigh_columns, ensure_sharding, plan as _plan
from distrib_la.resolve import EIGH_BACKENDS, NATIVE

__all__ = ["dispatch_batched_eigh", "EIGH_BACKENDS"]

#: The STACKED-form entry point a backend may or may not expose.  Probed
#: with ``getattr`` in :func:`dispatch_batched_eigh` — see that docstring
#: for why a name probe and not a platform test.
_BATCHED_ENTRY = "batched_distributed_eigh"


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

    WHY THIS EXISTS.  Batching is not universal: only the ScaLAPACK backend
    exposes ``batched_distributed_eigh``, which costs ONE
    collective-serialisation round for the whole stack instead of ``Nq``.
    cuSOLVERMp and SLATE expose the single-matrix ``distributed_eigh``
    only.  A caller that reaches for the batched name directly therefore
    works on a host mesh and ``AttributeError``s on CUDA.  This is the one
    place that asymmetry is handled.

    The capability test is ``getattr``, not a platform test, so a backend
    that gains a batched entry later is picked up with no edit here.  The
    price of that promise is paid in :func:`distrib_la.plan._eigh_columns`,
    which had to become rank-agnostic: a future cuSOLVERMp batched entry
    would hand a ``(nq, n, n)`` stack to a normaliser written for one tile.

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
    path is ever taken.  Routing the loop body through ``plan(A[q])`` fixes
    it by construction, because the plan is the object that owns
    ``_eigh_columns``.

    What deliberately still runs per ``q`` is the backend wrapper's OWN
    call-time validation (``_scalapack.validate_eigh_mesh`` + platform +
    divisibility).  That is not an oversight to hoist: it is the bug L-1
    mirror — a rule enforced only at resolve time turns a returned backend
    name into a broken promise — and it costs a few int comparisons plus a
    cached context lookup, not the resolve ladder's symbol probe and
    ``jax.process_count``.

    DONATION: the serial fallback does not donate, must not, and has
    nothing to preserve.  Neither eigh backend donates its operand in the
    first place (``:data:`distrib_la.plan.DONATES```): ScaLAPACK's handler
    stages each q into scratch because ``pXheevd`` destroys its input, and
    the cuSOLVERMp wrapper has neither ``donate_argnums`` nor
    ``input_output_aliases``.  Donating the STACK across a Python loop is
    incoherent regardless: the q-th call would invalidate the buffer the
    q+1-th call still has to slice.  The cost this leaves is a transient,
    not a leak — the Nq ``Z[q]`` shards are held until ``jnp.stack`` copies
    them, so the serial path peaks at ~2x the Z stack where the batched
    path peaks at 1x.  That is a reason to prefer a backend WITH a stacked
    entry, not a reason to donate.

    GATE: ``tests/multi_device/batched_eigh_dispatch_gate.py`` (srun, any
    square host mesh).  A CPU mesh is the only place both paths are
    reachable at all, since ScaLAPACK is the only backend with a stacked
    entry; the serial path is forced there by the private ``_force_serial``
    argument rather than by deleting the attribute from the module, which
    keeps the gate on the real entry point.

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

    LAX.SCAN: it IS scan-safe.  The belief recorded in the placeholder
    ("the FFI call is not scan-safe") is FALSE on this stack and is written
    down here so it stops being repeated.  Job 7889132, probe leg, same
    mesh: a ``lax.scan`` whose body is ``_scalapack.distributed_eigh``
    TRACES, LOWERS, COMPILES and RUNS, and the collective inside the XLA
    while body did not deadlock at P=4.  Per-call cost, warm: scan 12.2 ms,
    Python loop 21.7 ms, batched entry 11.3 ms (reproduced on job 7889130:
    11.3 / 21.6 / 11.1).

    The loop stays Python anyway, and the reason is coverage, not
    capability.  The serial path is DEAD on the one backend the probe could
    reach: ScaLAPACK has a stacked entry, so ``getattr`` always finds it.
    The fallback exists for cuSOLVERMp and SLATE, over NCCL + cal_comm, and
    no CUDA mesh was available to probe — and the CUDA distributed eigh's
    failure mode is a HANG with no traceback (guard 5 exists because
    ``cusolverMpSyevd`` hangs inside a collective on a rectangular grid
    instead of returning an error status), which is not a thing to ship on
    the strength of a CPU-only measurement.  A scan would also not recover
    what the batched entry buys — that saving is in C++, where the handler
    takes ``nq`` as a compile-time attribute and loops q around ONE
    descriptor and ONE workspace.
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
