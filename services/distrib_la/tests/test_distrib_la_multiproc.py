"""Layer L-c: four REAL processes, a real 2x2 mesh, the FFI backends.

This is the tier the other two cannot reach.  GUARD 4 (``resolve.py``, the
coverage rung) refuses every FFI backend when ``jax.process_count()`` is
smaller than ``mesh.devices.size``, because cuSOLVERMp, SLATE and
ScaLAPACK each carry a per-PROCESS MPI/NCCL context.  So an emulated 2x2
can exercise ``native`` and ``native2d`` and nothing else, and every claim
about a distributed library on a mesh bigger than 1x1 has to come from
here::

    lx run -N 1 -G 4 -n 4 python3 \\
        services/distrib_la/tests/test_distrib_la_multiproc.py --mesh 2x2
    lx run --cpu -N 1 -n 4 python3 \\
        services/distrib_la/tests/test_distrib_la_multiproc.py --mesh 2x2

ONE SET OF CHECK BODIES, TWO CALLERS — the ``_CLI_CELLS`` pattern copied
from the contract suite rather than reinvented.  Every ``check_*(mesh,
...)`` below is called by a pytest cell (on whatever mesh this process can
build, which is 1x1) AND by ``_cli_main`` under ``srun``.  Duplicating the
logic across the two would mean the multi-rank leg tests something
slightly different from the thing the suite pins, which is how a matrix
leg drifts out of agreement with its own reference.

WHAT THIS FILE COVERS THAT NOTHING ELSE DOES

* every (backend x op) the package has, on a REAL 2x2;
* hostile extents through the FFI paths -- not the arithmetic (L-a) and
  not the pure-JAX kernel (L-b), but the block-cyclic descriptors;
* the factor/solve TOKENS: scalapack's ``ipiv`` round-trip, cuSOLVERMp's
  potrf handle, and --

* THE SLATE trsm BACK-SOLVE, WHOSE FIRST EXECUTION THIS IS.  ``factor.py``
  says so in its own words:

      UNMEASURED ON A REAL MESH as of this commit: LORRAX's slate cholesky
      consumer went through ``to_jax_lower()`` and a replicated triangular
      solve, with "wiring slate::trsm for the back-solve is a perf
      follow-up" written next to it.  This is that wiring; its first
      execution is the service suite's real-4-process leg.

  ``check_slate_factor_solve`` IS that leg.  Its numbers are FINDINGS, not
  noise: the bar is relative 1e-12 (C8 -- the repo's engine-parity bar,
  relative and never bit-exact), and if it misses, the shapes, residuals
  and the q it missed on get reported rather than the tolerance getting
  relaxed.

* ``tests/multi_device/batched_eigh_dispatch_gate.py``'s checks, adopted.
  That file matches no pytest pattern and runs in NO CI (survey 1, S9);
  it is the only check of batched-vs-serial agreement in the tree.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest

# Path first: this file runs as a bare script under srun, where nothing has
# put services/*/src anywhere (`lx` rewrites the container PYTHONPATH to
# exactly <checkout>/src).
_TESTS = os.path.dirname(os.path.abspath(__file__))
_SERVICES = os.path.dirname(os.path.dirname(_TESTS))
_REPO = os.path.dirname(_SERVICES)
for _svc in ("lxkit", "distrib_la"):
    _src = os.path.join(_SERVICES, _svc, "src")
    if os.path.isdir(_src) and _src not in sys.path:
        sys.path.insert(0, _src)

# CLI multi-rank mode uses the same runtime boundary as production drivers.
# This keeps JAX initialization, square-mesh clique warm-up, transport
# enforcement, FFI gates and ordered finalization in one source of truth.
if __name__ == "__main__":
    from lxkit.gate import platform_from_env
    from runtime import initialize_communicator_stack
    _plat = platform_from_env()
    _RUNTIME = initialize_communicator_stack(
        platform="gpu" if _plat == "CUDA" else "cpu")

import distrib_la as D                                        # noqa: E402
from lxkit.testing import hostile_extents                     # noqa: E402

#: The engine-parity bar (C8): RELATIVE 1e-12, never bit-exact.  A blocked
#: distributed factorization and a single-rank LAPACK call are different
#: reductions in a different order; demanding bit-equality across them
#: would be demanding that the library not be distributed.
RTOL = 1e-12

#: Cholesky backends whose factor is a library HANDLE rather than an
#: array, so ``plan.batched`` refuses them by design and the factor/solve
#: token is the only route.  ``native2d`` is dense-in/dense-out and is not
#: one of them.
_HANDLE_CHOLESKY = ("slate", "cusolvermp")


# ---------------------------------------------------------------------------
# Operand builders and measures — host numpy, identical seed on every rank
# ---------------------------------------------------------------------------

def _rng_mat(rng, shape, dtype):
    a = rng.standard_normal(shape)
    if np.dtype(dtype).kind == "c":
        a = a + 1j * rng.standard_normal(shape)
    return a.astype(dtype)


def _hpd(rng, nq, n, dtype):
    z = _rng_mat(rng, (nq, n, n), dtype)
    return (0.5 * (z + np.conj(np.swapaxes(z, -1, -2)))
            + (n + 4) * np.eye(n)[None]).astype(dtype)


def _herm(rng, nq, n, dtype):
    z = _rng_mat(rng, (nq, n, n), dtype)
    return (0.5 * (z + np.conj(np.swapaxes(z, -1, -2)))).astype(dtype)


def _put(np_arr, mesh, spec):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(np.asarray(np_arr), NamedSharding(mesh, P(*spec)))


def _gather(x):
    """Bring a sharded array back to host numpy, multi-process aware."""
    import jax
    if jax.process_count() == 1:
        return np.asarray(x)
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-300)


def _resid(A, X, B):
    """max_q ||A[q] X[q] - B[q]|| / ||B[q]||."""
    return max(float(np.linalg.norm(A[q] @ X[q] - B[q])
                     / max(np.linalg.norm(B[q]), 1.0))
               for q in range(A.shape[0]))


# ---------------------------------------------------------------------------
# Check bodies — shared by the pytest cells and the CLI matrix.
# Each raises AssertionError with the residuals in the message.
# ---------------------------------------------------------------------------

def check_resolution_is_a_promise(mesh, dtype="complex128"):
    """On a real mesh, every backend ``resolve_backend`` NAMES must run.

    That is the promise semantics rung, and a real multi-process mesh is
    the only place it can be falsified end to end: guard 4 makes every
    single-process check of it vacuous for the FFI backends.  For each op,
    ask ``list_backends`` (which never raises) what is available, then
    CALL each available one.  A name that comes back available and then
    refuses at call time is exactly the broken promise the ladder exists
    to prevent.
    """
    import jax.numpy as jnp
    n = 64
    rng = np.random.default_rng(101)
    called = []
    for op in D.OPS:
        for backend, status in D.list_backends(op, mesh).items():
            if not status.startswith("available") or backend in ("auto", "off"):
                continue
            try:
                resolved = D.resolve_backend(op, backend, mesh, n=n)
            except (ValueError, RuntimeError):
                # A geometry/divisibility refusal at THIS n is a legitimate
                # answer; list_backends reports capability, not geometry.
                continue
            if resolved in (D.NATIVE, "native2d"):
                continue
            A = jnp.asarray(_hpd(rng, 2, n, dtype))
            if op == "solve_lu":
                B = jnp.asarray(_rng_mat(rng, (2, n, 32), dtype))
                D.plan(op, mesh, backend=backend, n=n).batched(A, B)
            elif op == "cholesky" and resolved in _HANDLE_CHOLESKY:
                # MEASURED, Perlmutter CPU 2x2, 2026-08-07: plan.batched
                # REFUSES slate cholesky --
                #
                #   cholesky backend 'slate' has no batched entry point and
                #   its single-tile result is not an array, so
                #   plan.batched() cannot stack it.  Use
                #   distrib_la.factor()/solve() ...
                #
                # -- which is the contract, not a defect.  Routing through
                # the token here is what makes this check cover the
                # handle-returning backends instead of reporting a GUARD
                # and moving on, which is what the first draft did.
                D.solve(D.factor(op, A, mesh, backend=backend, n=n),
                        jnp.asarray(_rng_mat(rng, (2, n, 8), dtype)))
            else:
                D.plan(op, mesh, backend=backend, n=n).batched(A)
            called.append(f"{op}/{backend}->{resolved}")
    assert called, (
        "no FFI backend was both available and callable on this mesh, so "
        "this check asserted nothing.  On a machine with the .so pins that "
        "is a defect; the report must say which backends list_backends "
        f"offered: "
        + "; ".join(f"{op}:{D.list_backends(op, mesh)}" for op in D.OPS)[:900])
    return called


def check_scalapack_factor_solve(mesh, dtype="complex128", nq=2, n=64,
                                 nrhs=32):
    """The ``ipiv`` round-trip through the TOKEN.

    ``factor('solve_lu')`` on scalapack is ``pXgetrf``; the pivots come
    back as an opaque ``i32`` array at ``P(None,('x','y'))`` that is never
    gathered and must be fed back verbatim.  Before ``FactorToken`` that
    was a comment threaded through three call frames in ``isdf/core.py``.

    Three claims, and they fail differently:
      1. the solve is a solve (residual against the operands);
      2. a SECOND right-hand side solved from the SAME token agrees with
         a fresh fused solve BIT-FOR-BIT -- that is the factor reuse the
         split exists for, and the thing a stale ``ipiv`` breaks;
      3. a token refuses a B it was not factored for, rather than
         corrupting a solve or hanging inside a collective.
    """
    import jax.numpy as jnp
    rng = np.random.default_rng(11)
    A_np = _herm(rng, nq, n, dtype)              # Hermitian INDEFINITE
    B_np = _rng_mat(rng, (nq, n, nrhs), dtype)
    B2_np = _rng_mat(rng, (nq, n, nrhs), dtype)

    tok = D.factor("solve_lu", _put(A_np, mesh, (None, "x", "y")), mesh,
                   backend="scalapack")
    assert tok.backend == "scalapack" and tok.n == n and tok.nbatch == nq
    X1 = _gather(D.solve(tok, _put(B_np, mesh, (None, "x", "y"))))
    X2 = _gather(D.solve(tok, _put(B2_np, mesh, (None, "x", "y"))))

    r1, r2 = _resid(A_np, X1, B_np), _resid(A_np, X2, B2_np)
    assert r1 < 1e-10 and r2 < 1e-10, (
        f"scalapack factor/solve residuals: rhs1={r1:.3e} rhs2={r2:.3e}")

    fused = _gather(D.plan("solve_lu", mesh, backend="scalapack", n=n)
                    .batched(_put(A_np, mesh, (None, "x", "y")),
                             _put(B2_np, mesh, (None, "x", "y"))))
    assert np.array_equal(X2, fused), (
        f"the token's SECOND solve drifts from a fresh fused solve_lu "
        f"(rel {_rel(X2, fused):.3e}) — the ipiv did not survive reuse")

    bad = jnp.zeros((nq, n + 2, nrhs), dtype)
    try:
        D.solve(tok, bad)
    except ValueError as exc:
        assert "factored at" in str(exc), str(exc)
    else:
        raise AssertionError(
            "solve() accepted a B the token was not factored for; the "
            "token's whole contract is that it cannot be misapplied")
    return dict(rhs1=r1, rhs2=r2)


def check_cusolvermp_factor_solve(mesh, dtype="complex128", nq=2, n=64,
                                  nrhs=32):
    """The cuSOLVERMp potrf TOKEN.

    The handle used to be passed as ``.raw`` and REBUILT by the caller
    hundreds of lines later (``isdf/core.py`` :3180 and :3486).  Here it
    never leaves the token.  Checks the solve residual and that the
    factor really is the Cholesky factor of A.
    """
    rng = np.random.default_rng(7)
    A_np = _hpd(rng, nq, n, dtype)
    B_np = _rng_mat(rng, (nq, n, nrhs), dtype)
    tok = D.factor("cholesky", _put(A_np, mesh, (None, "x", "y")), mesh,
                   backend="cusolvermp")
    assert tok.backend == "cusolvermp"
    X = _gather(D.solve(tok, _put(B_np, mesh, (None, "x", "y"))))
    r = _resid(A_np, X, B_np)
    assert r < 1e-10, f"cusolvermp potrf/potrs residual {r:.3e}"
    return dict(residual=r)


def check_cusolvermp_lu_factor_solve(mesh, dtype="complex128", nq=2, n=64,
                                     nrhs=32):
    """Split CUDA getrf/getrs parity, reuse, and pivot ownership.

    The second RHS uses the same opaque token and is compared against the
    incumbent fused handler.  The internal handle is inspected only here to
    prove every process owns ``LOCr(M_A)+MB_A = 2*n/Px`` pivots per q:
    the carrier remains rank-private rather than replicating a global pivot.
    """
    rng = np.random.default_rng(20260827)
    A_np = _herm(rng, nq, n, dtype)
    A_np += np.eye(n, dtype=A_np.dtype)[None, :, :] * 0.37
    B_np = _rng_mat(rng, (nq, n, nrhs), dtype)
    B2_np = _rng_mat(rng, (nq, n, nrhs), dtype)

    for target in ("lorrax_cusolvermp_batched_getrf",
                   "lorrax_cusolvermp_batched_getrs"):
        usable, why = D.probe_target(target, "CUDA")
        assert usable, f"new CUDA handler {target} is not loadable: {why}"

    tok = D.factor("solve_lu", _put(A_np, mesh, (None, "x", "y")), mesh,
                   backend="cusolvermp")
    X1 = _gather(D.solve(tok, _put(B_np, mesh, (None, "x", "y"))))
    X2 = _gather(D.solve(tok, _put(B2_np, mesh, (None, "x", "y"))))
    r1, r2 = _resid(A_np, X1, B_np), _resid(A_np, X2, B2_np)
    assert r1 < 1e-10 and r2 < 1e-10, (
        f"cusolvermp split residuals rhs1={r1:.3e} rhs2={r2:.3e}")

    fused = _gather(D.plan("solve_lu", mesh, backend="cusolvermp", n=n)
                    .batched(_put(A_np, mesh, (None, "x", "y")),
                             _put(B2_np, mesh, (None, "x", "y"))))
    assert np.array_equal(X2, fused), (
        f"split CUDA second solve drifts from fused getrf+getrs "
        f"(rel {_rel(X2, fused):.3e})")

    held = tok._factor  # service-internal white-box ownership gate
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    ipiv_len = 2 * (n // px)
    want_local = (nq, ipiv_len)
    want_global = (nq, px * py * ipiv_len)
    assert tuple(held.ipiv.shape) == want_global, (
        f"pivot carrier global shape {held.ipiv.shape}, want {want_global}")
    local_shapes = [tuple(s.data.shape) for s in held.ipiv.addressable_shards]
    assert local_shapes and all(s == want_local for s in local_shapes), (
        f"rank-local pivot shapes {local_shapes}, want {want_local}")
    assert not held.ipiv.is_fully_replicated, "pivot carrier is replicated"
    return dict(rhs1=r1, rhs2=r2, fused_rel=_rel(X2, fused),
                pivot_global=want_global, pivot_local=want_local,
                pivot_local_bytes=nq * ipiv_len * 8)


def check_slate_factor_solve(mesh, dtype="complex128", nq=2, n=64, nrhs=32):
    """FIRST EXECUTION of the SLATE trsm back-solve.  Read the docstring.

    ``distrib_la.factor.solve`` says of this path:

        UNMEASURED ON A REAL MESH as of this commit: LORRAX's slate
        cholesky consumer went through ``to_jax_lower()`` and a replicated
        triangular solve, with "wiring slate::trsm for the back-solve is a
        perf follow-up" written next to it.  This is that wiring; its
        first execution is the service suite's real-4-process leg.

    So: two triangular solves per q against the SlateLowerL handle
    (``op='N'`` forward, ``op='C'`` adjoint), never materialising a
    row-major L and never replicating it.  The reference is the NATIVE
    solve of the same system.  Bar: relative 1e-12 (C8).

    The report carries the per-q residual either way.  A miss is a FINDING
    -- shapes, residuals, and which q -- and never a reason to move the
    bar.
    """
    rng = np.random.default_rng(29)
    A_np = _hpd(rng, nq, n, dtype)
    B_np = _rng_mat(rng, (nq, n, nrhs), dtype)
    tok = D.factor("cholesky", _put(A_np, mesh, (None, "x", "y")), mesh,
                   backend="slate")
    assert tok.backend == "slate"
    X = _gather(D.solve(tok, _put(B_np, mesh, (None, "x", "y"))))
    assert X.shape == B_np.shape, f"solve returned {X.shape}, want {B_np.shape}"

    # NATIVE reference: the same A X = B, solved per q on the host.
    X_ref = np.stack([np.linalg.solve(A_np[q], B_np[q]) for q in range(nq)])
    per_q = [_rel(X[q], X_ref[q]) for q in range(nq)]
    resid = _resid(A_np, X, B_np)
    worst_q = int(np.argmax(per_q))
    assert max(per_q) < RTOL, (
        f"SLATE trsm back-solve (FIRST EXECUTION) vs the native reference: "
        f"worst q={worst_q} rel {per_q[worst_q]:.6e} > {RTOL:.0e}; "
        f"per-q {['%.3e' % v for v in per_q]}; A X - B residual "
        f"{resid:.6e}; shapes A{A_np.shape} B{B_np.shape} X{X.shape}; "
        f"mesh {int(mesh.shape['x'])}x{int(mesh.shape['y'])} dtype {dtype}")
    assert resid < 1e-10, f"A X - B residual {resid:.3e}"
    return dict(per_q=per_q, residual=resid, worst_q=worst_q)


def check_factor_refuses_what_it_cannot_split(mesh, dtype="complex128"):
    """The refusals ``factor`` owes its callers, on a real mesh.

    A refusal that only fires on a dev box is a refusal nobody has seen
    fire where it matters.  ``eigh`` has no split; ``native`` has no
    reusable distributed factor; ``solve_lu`` on a backend whose entry
    point is the FUSED getrf+getrs cannot hand back pivots.
    """
    import jax.numpy as jnp
    A = jnp.asarray(_hpd(np.random.default_rng(3), 2, 32, dtype))
    with _raises(ValueError, "no factor/solve split"):
        D.factor("eigh", A, mesh)
    with _raises(NotImplementedError, "NATIVE backend"):
        D.factor("cholesky", A, mesh, backend="off")
    with _raises(ValueError, "expected A of shape"):
        D.factor("cholesky", A[0], mesh, backend="off")
    return True


def check_hostile_extents_through_the_ffi(mesh, dtype="complex128",
                                          backend="scalapack", op="solve_lu"):
    """Non-dividing extents through the block-cyclic DESCRIPTORS.

    L-a checks the arithmetic and L-b checks the pure-JAX kernel; this is
    the only tier where a non-dividing extent reaches a real ScaLAPACK or
    SLATE descriptor on a real grid.  For each hostile family: the LOGICAL
    extent must be refused at RESOLVE time (guard 6 -- before any
    collective is entered, which is the difference between an error and a
    hang), and the PADDED extent must produce a solve whose logical block
    is right and whose pad rows are exact zeros.

    The anti-tautology self-assertion is the same one the FFI padding
    contract carries: if nothing was refused, the loop proved nothing.
    """
    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    rng = np.random.default_rng(41)
    refused, ran = 0, 0
    for case in hostile_extents((px, py)):
        n_log, n_pad = case.logical[0], case.padded[0]
        if n_log % px or n_log % py:
            try:
                D.resolve_backend(op, backend, mesh, n=n_log)
            except (ValueError, RuntimeError):
                refused += 1
            else:
                raise AssertionError(
                    f"{case.name}: n={n_log} does not divide the {px}x{py} "
                    f"mesh and {backend} accepted it at resolve time — a "
                    f"guard that fires inside the collective is a hang, "
                    f"not an error")
        if n_pad < 8:
            continue
        nq, nrhs = 2, 8
        A_log = _herm(rng, nq, n_log, dtype)
        B_log = _rng_mat(rng, (nq, n_log, nrhs), dtype)
        A_pad = np.zeros((nq, n_pad, n_pad), dtype)
        A_pad[:, :n_log, :n_log] = A_log
        for i in range(n_log, n_pad):
            A_pad[:, i, i] = 1.0
        B_pad = np.zeros((nq, n_pad, nrhs), dtype)
        B_pad[:, :n_log, :] = B_log
        assert n_pad > n_log or (n_log % px == 0 and n_log % py == 0)
        X = _gather(D.plan(op, mesh, backend=backend, n=n_pad).batched(
            _put(A_pad, mesh, (None, "x", "y")),
            _put(B_pad, mesh, (None, "x", "y"))))
        ran += 1
        r = _resid(A_log, X[:, :n_log, :], B_log)
        assert r < 1e-10, (
            f"{case.name}: n_log={n_log} n_pad={n_pad} logical-block "
            f"residual {r:.3e}")
        if n_pad > n_log:
            assert not X[:, n_log:, :].any(), (
                f"{case.name}: pad rows are not exact zeros")
    assert refused >= 1, (
        f"no hostile family was refused on a {px}x{py} mesh, so the "
        f"refusal half of this check proved nothing")
    return dict(refused=refused, ran=ran)


# ---------------------------------------------------------------------------
# ADOPTED from tests/multi_device/batched_eigh_dispatch_gate.py — which
# matches no pytest pattern and runs in NO CI today (survey 1, S9).  It is
# the only check of batched-vs-serial agreement in the tree.
# ---------------------------------------------------------------------------

def check_batched_eigh_dispatch(mesh, dtype="complex128", nq=6, n=64):
    """The dispatcher's two paths, and that they agree.

    A HOST mesh is the only place both are reachable: ScaLAPACK is the
    only backend exposing ``batched_distributed_eigh``, so on CUDA the
    serial fallback is the only path there is.

      1. ``off`` is still ``jnp.linalg.eigh``, BIT-IDENTICAL.
      2. batched vs serial agree in W and Z.  Same grid, same descriptors,
         differing only in the ``nq`` the handler loops over, so the
         expectation is bit-identity — reported either way, never assumed.
      3. the LAYOUT contract on BOTH paths, ``A[q] Z[q] == Z[q] diag(W)``,
         WITH the conj-transpose as a NEGATIVE CONTROL: without it the
         residual passes on a dispatcher that returned ROWS, which is
         what the raw fallback did on CUDA before the fix.
      4. eigenvalues against ``numpy.linalg.eigvalsh``.
      5. exactly ONE resolve for an nq-matrix SERIAL dispatch (the guard
         hoist), counted by rebinding the name on ``distrib_la.plan`` —
         the module whose globals ``plan()`` actually reads.  Patching a
         re-export shim would count ZERO and report PASS.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    rng = np.random.default_rng(20260804)
    A = _herm(rng, nq, n, dtype)
    A_j = _put(A, mesh, (None, "x", "y"))
    out = {}

    W_off, Z_off = D.dispatch_batched_eigh(A_j, mesh, "off")
    W_nat, Z_nat = jnp.linalg.eigh(A_j)
    assert float(jnp.max(jnp.abs(W_off - W_nat))) == 0.0, \
        "backend 'off' is no longer bit-identical to jnp.linalg.eigh (W)"
    assert float(jnp.max(jnp.abs(Z_off - Z_nat))) == 0.0, \
        "backend 'off' is no longer bit-identical to jnp.linalg.eigh (Z)"

    W_b, Z_b = D.dispatch_batched_eigh(A_j, mesh, "distributed")
    W_s, Z_s = D.dispatch_batched_eigh(A_j, mesh, "distributed",
                                       _force_serial=True)
    d_W = _rel(_gather(W_s), _gather(W_b))
    d_Z, exact_Z = 0.0, True
    for sh_s, sh_b in zip(Z_s.addressable_shards, Z_b.addressable_shards):
        a, b = np.asarray(sh_s.data), np.asarray(sh_b.data)
        d_Z = max(d_Z, _rel(a, b))
        exact_Z = exact_Z and np.array_equal(a, b)
    out.update(batched_vs_serial_W=d_W, batched_vs_serial_Z=d_Z,
               bit_identical=bool(exact_Z))
    assert d_W <= RTOL and d_Z <= RTOL, \
        f"batched vs serial: W rel {d_W:.3e} Z rel {d_Z:.3e}"

    for tag, W_x, Z_x in (("batched", W_b, Z_b), ("serial", W_s, Z_s)):
        lhs = jnp.einsum("qmn,qnp->qmp", A_j, Z_x, optimize=True)
        r_col = (float(jnp.max(jnp.abs(lhs - Z_x * W_x[:, None, :]
                                       .astype(Z_x.dtype))))
                 / max(float(jnp.max(jnp.abs(A_j))), 1e-300))
        Zt = jnp.conj(jnp.swapaxes(Z_x, -1, -2))
        lhs_t = jnp.einsum("qmn,qnp->qmp", A_j, Zt, optimize=True)
        r_row = (float(jnp.max(jnp.abs(lhs_t - Zt * W_x[:, None, :]
                                       .astype(Zt.dtype))))
                 / max(float(jnp.max(jnp.abs(A_j))), 1e-300))
        out[f"layout_{tag}"] = r_col
        assert r_col <= RTOL, f"{tag}: A Z != Z diag(W), rel {r_col:.3e}"
        assert r_row > 1e-6, (
            f"{tag}: the TRANSPOSED control also passed (rel {r_row:.3e}); "
            f"the layout assertion cannot distinguish columns from rows "
            f"and is measuring nothing")
        assert Z_x.sharding.spec == P(None, "x", "y"), \
            f"{tag}: Z spec {Z_x.sharding.spec}"
        assert W_x.sharding.is_fully_replicated and W_x.dtype == jnp.float64, \
            f"{tag}: W repl={W_x.sharding.is_fully_replicated} {W_x.dtype}"

    d_E = _rel(_gather(W_b), np.linalg.eigvalsh(A))
    out["eigenvalues_vs_numpy"] = d_E
    assert d_E <= 1e-10, f"eigenvalues vs numpy rel {d_E:.3e}"

    planmod = sys.modules["distrib_la.plan"]
    seen, orig = [], planmod.resolve_backend

    def _counting(op, requested, mesh_, **kw):
        seen.append((op, requested))
        return orig(op, requested, mesh_, **kw)

    planmod.resolve_backend = _counting
    try:
        D.dispatch_batched_eigh(A_j, mesh, "distributed", _force_serial=True)
    finally:
        planmod.resolve_backend = orig
    out["resolves_for_serial_dispatch"] = len(seen)
    assert len(seen) == 1, (
        f"a {nq}-matrix SERIAL dispatch resolved {len(seen)} times "
        f"(want 1): the guard hoist regressed")

    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    if px > 1 or py > 1:
        bad = jnp.zeros((2, n + 1, n + 1), dtype=jnp.complex128)
        with _raises((ValueError, RuntimeError), "divisible"):
            D.dispatch_batched_eigh(bad, mesh, "distributed")
    del jax
    return out


def check_batch_reshard_local_ops(mesh, dtype="complex128", nq=5, n=64,
                                  nrhs=32):
    """Route (c), all array-returning ops, including a ragged batch.

    This body is shared by pytest's 1x1 smoke and the real P=4 CLI matrix.
    On the latter ``nq=5`` is deliberately not divisible by four, so the
    op-safe pad/drop path is load-bearing.  Every result is checked at the
    ordinary Plan.batched output layout before the test-only host readback.
    """
    from jax.sharding import PartitionSpec as P

    ndev = int(mesh.devices.size)
    if ndev > 1:
        assert nq % ndev != 0, (
            f"route-c gate batch nq={nq} divides P={ndev}; the ragged "
            f"padding this gate claims to cover is vacuous")
    rng = np.random.default_rng(20260815)
    out = {}

    A_eigh = _herm(rng, nq, n, dtype)
    W, Z = D.plan(
        "eigh", mesh, backend="off", n=n,
        batched_route=D.ROUTE_BATCH_RESHARD,
    ).batched(_put(A_eigh, mesh, (None, "x", "y")))
    assert W.sharding.is_fully_replicated
    assert Z.sharding.spec == P(None, "x", "y")
    W_np, Z_np = _gather(W), _gather(Z)
    out["eigh_values"] = _rel(W_np, np.linalg.eigvalsh(A_eigh))
    out["eigh_residual"] = _rel(
        A_eigh @ Z_np, Z_np * W_np[:, None, :])
    assert out["eigh_values"] < 1e-10
    assert out["eigh_residual"] < RTOL

    A_chol = _hpd(rng, nq, n, dtype)
    L = D.plan(
        "cholesky", mesh, backend="off", n=n,
        batched_route=D.ROUTE_BATCH_RESHARD,
    ).batched(_put(A_chol, mesh, (None, "x", "y")), block_size=17)
    assert L.sharding.spec == P(None, "x", "y")
    L_np = _gather(L)
    out["cholesky"] = _rel(L_np, np.linalg.cholesky(A_chol))
    out["cholesky_residual"] = _rel(
        L_np @ np.conj(np.swapaxes(L_np, -1, -2)), A_chol)
    assert out["cholesky"] < RTOL
    assert out["cholesky_residual"] < RTOL

    A_lu = _hpd(rng, nq, n, dtype)
    B_lu = _rng_mat(rng, (nq, n, nrhs), dtype)
    X = D.plan(
        "solve_lu", mesh, backend="off", n=n,
        batched_route=D.ROUTE_BATCH_RESHARD,
    ).batched(_put(A_lu, mesh, (None, "x", "y")),
              _put(B_lu, mesh, (None, "x", "y")))
    assert X.sharding.spec == P(None, "x", "y")
    X_np = _gather(X)
    out["solve"] = _rel(X_np, np.linalg.solve(A_lu, B_lu))
    out["solve_residual"] = _resid(A_lu, X_np, B_lu)
    assert out["solve"] < RTOL
    assert out["solve_residual"] < 1e-10
    return out


def check_distributed_matmul(mesh, dtype="complex128", *,
                             backend="off", batched_route="batch_reshard",
                             nq=5, m=48, k=64, n=80):
    """Top-level GEMM through a real provider or staged local route.

    The provider cells enter cuBLASMp, PBLAS ``p?gemm`` or
    ``slate::multiply`` on the real process grid.  The local cell uses a
    deliberately ragged leading batch, exercising forward x/y exchanges,
    local GEMM, and inverse y/x exchanges without an all-gather.
    """
    from jax.sharding import PartitionSpec as P

    ndev = int(mesh.devices.size)
    if batched_route == D.ROUTE_BATCH_RESHARD and ndev > 1:
        assert nq % ndev != 0, (
            f"matmul batch nq={nq} divides P={ndev}; ragged padding is "
            "not exercised")
    rng = np.random.default_rng(20260816 + len(backend))
    A_np = _rng_mat(rng, (nq, m, k), dtype)
    B_np = _rng_mat(rng, (nq, k, n), dtype)
    C_np = _rng_mat(rng, (nq, m, n), dtype)
    alpha = (1.25 - 0.375j if np.dtype(dtype).kind == "c" else 1.25)
    beta = (-0.5 + 0.125j if np.dtype(dtype).kind == "c" else -0.5)
    got = D.matmul(
        _put(A_np, mesh, (None, "x", "y")),
        _put(B_np, mesh, (None, "x", "y")),
        _put(C_np, mesh, (None, "x", "y")),
        mesh=mesh, alpha=alpha, beta=beta, backend=backend,
        batched_route=batched_route)
    assert got.sharding.spec == P(None, "x", "y")
    got_np = _gather(got)
    want = alpha * (A_np @ B_np) + beta * C_np
    rel = _rel(got_np, want)
    assert rel < RTOL, (
        f"matmul {backend}/{batched_route} residual {rel:.3e}; "
        f"shapes A={A_np.shape} B={B_np.shape} C={C_np.shape}")

    # A conjugate-transpose catches descriptor/layout mistakes that an
    # N,N square-product smoke cannot see.  The provider C buffer is
    # donated, so this is a fresh call with fresh operands.
    At_np = _rng_mat(rng, (2, k, m), dtype)
    Bt_np = _rng_mat(rng, (2, k, n), dtype)
    if backend == "cublasmp" and ndev > 1:
        with _raises(ValueError, "not trustworthy"):
            D.matmul(
                _put(At_np, mesh, (None, "x", "y")),
                _put(Bt_np, mesh, (None, "x", "y")),
                mesh=mesh, transa="C", backend=backend,
                batched_route=batched_route)
        return {"nn_residual": rel, "conj_transpose": "refused"}
    got_t = D.matmul(
        _put(At_np, mesh, (None, "x", "y")),
        _put(Bt_np, mesh, (None, "x", "y")),
        mesh=mesh, transa="C", backend=backend,
        batched_route=batched_route)
    got_t_np = _gather(got_t)
    want_t = np.conj(np.swapaxes(At_np, -1, -2)) @ Bt_np
    rel_t = _rel(got_t_np, want_t)
    assert rel_t < RTOL, (
        f"matmul {backend}/{batched_route} transa=C residual {rel_t:.3e}")
    return {"nn_residual": rel, "conj_transpose_residual": rel_t}


def check_gemm_plan_cublasmp(mesh, dtype="complex128", *, nq=3,
                             m=None, k=None, n=None,
                             backend="cublasmp", expected_backend=None):
    """``distrib_la.gemm_plan``/``GemmPlan``: numerics, nested ``jit``,
    ``lax.scan`` reuse, the donated ``out=`` path, the ``beta!=0``
    accumulate path, and the internal-zero-``C`` path -- all against an
    ``A @ B`` numpy reference.  This is the ONLY tier that can certify any
    of it: an emulated mesh never reaches the collective provider code this
    checks.  ``backend`` is ``cublasmp`` on CUDA and ``scalapack`` on CPU.

    ``m, k, n`` default from the mesh so the plan's own tiling checks
    (``m % px``, ``k % px``, ``k % py``, ``n % py``) are exercised on
    whatever square mesh this runs on, not hard-coded to 2x2.
    """
    import jax
    from jax.sharding import PartitionSpec as P

    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    assert px == py, f"gemm_plan needs a square mesh; got {px}x{py}"
    m = m if m is not None else 2 * px
    k = k if k is not None else 3 * px
    n = n if n is not None else 2 * px
    rng = np.random.default_rng(20260822)
    A_np = _rng_mat(rng, (nq, m, k), dtype)
    B_np = _rng_mat(rng, (nq, k, n), dtype)
    A = _put(A_np, mesh, (None, "x", "y"))
    B = _put(B_np, mesh, (None, "x", "y"))
    want = A_np @ B_np

    plan = D.gemm_plan(mesh, m=m, k=k, n=n, nq=nq, dtype=dtype,
                       backend=backend)
    expected_backend = expected_backend or backend
    assert plan.backend == expected_backend
    assert plan.describe(), "describe() must produce a non-empty banner line"
    out = {}

    # 1. eager call, no C -- the internal-zero-C kernel folded into one
    # compiled program (module docstring, "Output liveness").
    D_eager_j = plan(A, B)
    assert D_eager_j.sharding.spec == P(None, "x", "y")
    D_eager = _gather(D_eager_j)
    out["eager_no_c"] = _rel(D_eager, want)
    assert out["eager_no_c"] < RTOL, out["eager_no_c"]

    # 2. inside a jax.jit -- proves the closure composes under an outer
    # trace with no eager resolve/probe/dlopen inside it: gemm_plan()
    # already ran and warmed all of that before this function was called.
    @jax.jit
    def _once(a, b):
        return plan(a, b)
    D_jit = _gather(_once(A, B))
    out["jit_no_c"] = _rel(D_jit, want)
    assert out["jit_no_c"] < RTOL, out["jit_no_c"]

    # 3. inside lax.scan -- the actual per-tau/per-k hot-loop shape this
    # plan exists for (ppm_tau_kernel.py:311-317,438-490).  The scan body
    # traces ONCE; if the plan secretly needed eager work per call this
    # would either fail to trace or, on P>1, hang inside a mid-trace NCCL
    # collective instead of raising.
    nsteps = 3
    A_stack_np = _rng_mat(rng, (nsteps, nq, m, k), dtype)
    B_stack_np = _rng_mat(rng, (nsteps, nq, k, n), dtype)
    A_stack = _put(A_stack_np, mesh, (None, None, "x", "y"))
    B_stack = _put(B_stack_np, mesh, (None, None, "x", "y"))

    @jax.jit
    def _scanned(a_stack, b_stack):
        def body(carry, ab):
            a, b = ab
            return carry, plan(a, b)
        _, out_stack = jax.lax.scan(body, None, (a_stack, b_stack), unroll=1)
        return out_stack

    D_scan = _gather(_scanned(A_stack, B_stack))
    want_scan = A_stack_np @ B_stack_np
    out["scan_no_c"] = _rel(D_scan, want_scan)
    assert out["scan_no_c"] < RTOL, out["scan_no_c"]

    # 4. explicit C, beta != 0 -- the donated-C kernel's accumulate form,
    # and its own refusal when a beta!=0 plan is called with no C.
    plan_beta = D.gemm_plan(mesh, m=m, k=k, n=n, nq=nq, dtype=dtype,
                            backend=backend, beta=-0.5)
    C_np = _rng_mat(rng, (nq, m, n), dtype)
    C = _put(C_np, mesh, (None, "x", "y"))
    D_c = _gather(plan_beta(A, B, C))
    want_c = A_np @ B_np + (-0.5) * C_np
    out["beta_accumulate"] = _rel(D_c, want_c)
    assert out["beta_accumulate"] < RTOL, out["beta_accumulate"]
    with _raises(ValueError, "C is required"):
        plan_beta(A, B)

    # 5. out= -- a caller-owned buffer donated purely for its storage
    # (beta=0, content ignored).  Correctness matches case 1; this proves
    # the call succeeds through the donated-C kernel with a real live
    # buffer in that slot, not only ever the plan's internal zeros.
    scratch_np = _rng_mat(rng, (nq, m, n), dtype)
    D_out = _gather(
        plan(A, B, out=_put(scratch_np, mesh, (None, "x", "y"))))
    out["out_donated"] = _rel(D_out, want)
    assert out["out_donated"] < RTOL, out["out_donated"]
    with _raises(ValueError, "not both"):
        plan(A, B, C, out=C)

    # 6. repeated calls on the SAME warmed plan with fresh operands each
    # time -- a real-mesh reuse signal (no per-call resolve/dlopen/probe,
    # no drift in the answer).  A dedicated allocator/HLO memory-telemetry
    # leg is the deeper capacity claim; this is correctness under reuse.
    for i in range(5):
        Ai = _rng_mat(rng, (nq, m, k), dtype)
        Bi = _rng_mat(rng, (nq, k, n), dtype)
        Di = _gather(plan(_put(Ai, mesh, (None, "x", "y")),
                          _put(Bi, mesh, (None, "x", "y"))))
        r = _rel(Di, Ai @ Bi)
        assert r < RTOL, f"repeated call {i}: rel {r:.3e}"
    out["repeated_calls"] = 5
    return out


def check_gemm_plan_scalapack(mesh, dtype="complex128", *,
                              backend="scalapack"):
    """The planned global-call contract through the canonical PBLAS path.

    The first exact-shape use deliberately occurs inside ``jit`` + ``scan``.
    PBLAS plan construction warms only a scalar tile per rank, so this is the
    regression gate for trace-safe cold exact-shape composition without
    production-sized dummy operands.  ``backend='auto'`` is the production
    call shape used by the face-layout GW kernels; its resolved provider must
    still be ScaLAPACK on this CPU mesh.
    """
    import jax

    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    assert px == py
    nq, m, k, n, nsteps = 2, 5 * px, 7 * px, 6 * px, 2
    rng = np.random.default_rng(20260901)
    A_np = _rng_mat(rng, (nsteps, nq, m, k), dtype)
    B_np = _rng_mat(rng, (nsteps, nq, k, n), dtype)
    A = _put(A_np, mesh, (None, None, "x", "y"))
    B = _put(B_np, mesh, (None, None, "x", "y"))
    plan = D.gemm_plan(mesh, m=m, k=k, n=n, nq=nq, dtype=dtype,
                       backend=backend)
    assert plan.backend == "scalapack"

    @jax.jit
    def _cold_scan(A_steps, B_steps):
        def _body(carry, operands):
            return carry, plan(operands[0], operands[1])
        _, values = jax.lax.scan(_body, None, (A_steps, B_steps))
        return values

    cold_rel = _rel(_gather(_cold_scan(A, B)), A_np @ B_np)
    assert cold_rel < RTOL, cold_rel
    out = check_gemm_plan_cublasmp(
        mesh, dtype, backend=backend, expected_backend="scalapack")
    out["cold_first_scan"] = cold_rel
    out["resolved_backend"] = plan.backend
    return out


def check_gemm_plan_manual_shard_map(mesh, dtype="complex128", *, nq=3,
                                     m=None, k=None, n=None, nsteps=3):
    """``GemmPlan.local_call``: the SAME planned N,N GEMM as
    ``check_gemm_plan_cublasmp``'s ``__call__`` coverage, but invoked from
    INSIDE a manual-mode ``shard_map`` + ``lax.scan`` -- the composition
    ``GemmPlan.__call__`` structurally cannot do (``matmul_plan.py``'s own
    module docstring, "Composition inside a MANUAL-mode shard_map";
    ``isdf.core._z_q_face``'s docstring names this as the reason its own
    band-window reconstruction stays on a masked ``psum`` rather than a
    SUMMA GEMM).  This is the real 4-rank CUDA proof that the composition
    itself works -- numerics against a numpy reference, from inside a
    manual ``shard_map`` body, streamed through ``lax.scan`` exactly like
    a real r-chunk-shaped kernel, covering the internal-zero-C path, a
    single donated-C accumulate call, ``out=``, and (case 5) C DONATED AS
    THE SCAN CARRY ITSELF -- an accumulate-over-many-iterations shape no
    other case here exercises, and the one composition closest to a real
    production tau/q-accumulation kernel.
    """
    import jax
    from functools import partial
    from jax.sharding import PartitionSpec as P
    from distrib_la._shard_map import shard_map

    px, py = int(mesh.shape["x"]), int(mesh.shape["y"])
    assert px == py, f"gemm_plan needs a square mesh; got {px}x{py}"
    m = m if m is not None else 2 * px
    k = k if k is not None else 3 * px
    n = n if n is not None else 2 * px
    rng = np.random.default_rng(20260822 + 7)
    out = {}

    plan = D.gemm_plan(mesh, m=m, k=k, n=n, nq=nq, dtype=dtype,
                       backend="cublasmp")

    # 1. One call, no scan, no jit around the shard_map (proves local_call
    # needs neither) -- the minimal case showing the WRAPPER, not the FFI,
    # was the obstacle.
    A_np = _rng_mat(rng, (nq, m, k), dtype)
    B_np = _rng_mat(rng, (nq, k, n), dtype)
    A = _put(A_np, mesh, (None, "x", "y"))
    B = _put(B_np, mesh, (None, "x", "y"))
    want = A_np @ B_np

    @partial(shard_map, mesh=mesh, in_specs=(P(None, "x", "y"),) * 2,
             out_specs=P(None, "x", "y"), check_vma=False)
    def _manual_once(a, b):
        return plan.local_call(a, b)

    D_once = _gather(_manual_once(A, B))
    out["manual_no_c"] = _rel(D_once, want)
    assert out["manual_no_c"] < RTOL, out["manual_no_c"]

    # 2. Inside lax.scan, streamed -- the actual r-chunk-shaped
    # composition (isdf.core._z_q_face's per-band-chunk scan body): one
    # shard_map entry wrapping many local_call invocations inside its
    # scan body, all under one outer jax.jit.
    A_stack_np = _rng_mat(rng, (nsteps, nq, m, k), dtype)
    B_stack_np = _rng_mat(rng, (nsteps, nq, k, n), dtype)
    A_stack = _put(A_stack_np, mesh, (None, None, "x", "y"))
    B_stack = _put(B_stack_np, mesh, (None, None, "x", "y"))
    want_scan = A_stack_np @ B_stack_np

    @partial(shard_map, mesh=mesh, in_specs=(P(None, None, "x", "y"),) * 2,
             out_specs=P(None, None, "x", "y"), check_vma=False)
    def _manual_scan(a_stack, b_stack):
        def body(carry, ab):
            a, b = ab
            return carry, plan.local_call(a, b)
        _, out_stack = jax.lax.scan(body, None, (a_stack, b_stack), unroll=1)
        return out_stack

    D_scan = _gather(jax.jit(_manual_scan)(A_stack, B_stack))
    out["manual_scan_no_c"] = _rel(D_scan, want_scan)
    assert out["manual_scan_no_c"] < RTOL, out["manual_scan_no_c"]

    # 3. beta!=0 accumulate (donated C=), from inside the same manual
    # shard_map -- donate_argnums=(2,) on the outer jit, mirroring
    # _build_kernel's own fn_with_c contract, since C here IS a top-level
    # jit argument (unlike the with_c=False path's internal jnp.zeros,
    # which needs no donation to alias -- see _local_gemm_call).
    plan_beta = D.gemm_plan(mesh, m=m, k=k, n=n, nq=nq, dtype=dtype,
                            backend="cublasmp", beta=-0.5)
    C_np = _rng_mat(rng, (nq, m, n), dtype)
    C = _put(C_np, mesh, (None, "x", "y"))

    @partial(shard_map, mesh=mesh, in_specs=(P(None, "x", "y"),) * 3,
             out_specs=P(None, "x", "y"), check_vma=False)
    def _manual_c(a, b, c):
        return plan_beta.local_call(a, b, C=c)

    D_c = _gather(jax.jit(_manual_c, donate_argnums=(2,))(A, B, C))
    want_c = A_np @ B_np + (-0.5) * C_np
    out["manual_beta_accumulate"] = _rel(D_c, want_c)
    assert out["manual_beta_accumulate"] < RTOL, out["manual_beta_accumulate"]

    # 4. out= donation on a beta==0 plan, from inside the same manual
    # shard_map.
    scratch_np = _rng_mat(rng, (nq, m, n), dtype)

    @partial(shard_map, mesh=mesh, in_specs=(P(None, "x", "y"),) * 3,
             out_specs=P(None, "x", "y"), check_vma=False)
    def _manual_out(a, b, scratch):
        return plan.local_call(a, b, out=scratch)

    D_out = _gather(jax.jit(_manual_out, donate_argnums=(2,))(
        A, B, _put(scratch_np, mesh, (None, "x", "y"))))
    out["manual_out_donated"] = _rel(D_out, want)
    assert out["manual_out_donated"] < RTOL, out["manual_out_donated"]

    # 5. C AS THE lax.scan CARRY ITSELF (not a scan-stacked output like
    # case 2, and not a single call outside scan like case 3) --
    # carry_{i+1} = A_i@B_i + beta*carry_i, accumulated over nsteps scan
    # iterations, the FFI's own input_output_aliases={2:0} donation
    # composing with scan's own carry-buffer reuse.  This is the one
    # untested cell in the task's own "FFI lifetime hazards ... across
    # scan iterations" concern: case 2 proved repeated create/destroy of
    # cuBLASMp's per-call descriptors against the SAME persistent
    # ctx/workspace is safe under scan, but never donated C THROUGH the
    # carry; case 3 proved donated-C accumulate but only as a single
    # call, never repeated.  A production accumulate-over-tau/q pattern
    # is shaped exactly like this case.
    # beta stays REAL (like case 3's -0.5): gemm_plan refuses a complex
    # alpha/beta on a real dtype (float64 is one of this cell's two
    # dtypes -- caught by exactly that guard on the first attempt here,
    # which used a complex beta and silently dropped float64 coverage of
    # this case instead of erroring).
    beta5 = -0.75
    plan_beta2 = D.gemm_plan(mesh, m=m, k=k, n=n, nq=nq, dtype=dtype,
                             backend="cublasmp", beta=beta5)
    C0_np = _rng_mat(rng, (nq, m, n), dtype)
    carry_want = C0_np.copy()
    for i in range(nsteps):
        carry_want = A_stack_np[i] @ B_stack_np[i] + beta5 * carry_want

    @partial(shard_map, mesh=mesh,
             in_specs=(P(None, None, "x", "y"),) * 2 + (P(None, "x", "y"),),
             out_specs=P(None, "x", "y"), check_vma=False)
    def _manual_scan_carry(a_stack, b_stack, c0):
        def body(carry, ab):
            a, b = ab
            return plan_beta2.local_call(a, b, C=carry), None
        final_carry, _ = jax.lax.scan(body, c0, (a_stack, b_stack), unroll=1)
        return final_carry

    D_carry = _gather(jax.jit(_manual_scan_carry, donate_argnums=(2,))(
        A_stack, B_stack, _put(C0_np, mesh, (None, "x", "y"))))
    out["manual_scan_carry_donation"] = _rel(D_carry, carry_want)
    assert (out["manual_scan_carry_donation"] < RTOL
           ), out["manual_scan_carry_donation"]

    return out


class _raises:
    """``pytest.raises`` that also works in the pytest-free CLI mode."""

    def __init__(self, exc, match=""):
        self.exc, self.match = exc, match

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(
                f"expected {self.exc} matching {self.match!r}, nothing raised")
        if not issubclass(et, self.exc if isinstance(self.exc, tuple)
                          else (self.exc,)):
            return False
        if self.match and self.match not in str(ev):
            raise AssertionError(
                f"{et.__name__} raised but {self.match!r} not in {str(ev)!r}")
        return True


# ---------------------------------------------------------------------------
# pytest entry points.  Single process, so the mesh is 1x1 and the FFI
# cells run there; the 2x2 answers come from the CLI matrix below.
# ---------------------------------------------------------------------------

def _mesh_1x1(platform=None):
    import jax
    from jax.sharding import Mesh
    devs = jax.devices() if platform is None else jax.devices(platform)
    return Mesh(np.asarray(devs[:1]).reshape(1, 1), ("x", "y"))


def _needs(op, backend, platform):
    """SKIP unless (op, backend) is usable on a 1x1 mesh of ``platform``.

    ABSENT-vs-BROKEN is upstream of this: ``resolve_backend`` raises with
    the loader's three-way reason, and the reason is quoted into the skip
    so the report says which of the three it was.  The covering leg is
    named because a skip that names no covering run is lost coverage.
    """
    import jax
    try:
        jax.devices(platform)
    except Exception as exc:                                   # noqa: BLE001
        pytest.skip(f"no {platform} backend here ({exc}); covered by leg L-c")
    mesh = _mesh_1x1(platform)
    try:
        D.resolve_backend(op, backend, mesh, n=64)
    except (ValueError, RuntimeError) as exc:
        pytest.skip(f"{op}/{backend} is not usable on a 1x1 {platform} mesh: "
                    f"{exc}  Covered by leg L-c "
                    f"(lx run -n 4 ... --mesh 2x2) where the library is "
                    f"pinned and there are four ranks")
    return mesh


def test_scalapack_factor_solve_token_round_trip():
    check_scalapack_factor_solve(_needs("solve_lu", "scalapack", "cpu"))


def test_cusolvermp_factor_solve_token_round_trip():
    check_cusolvermp_factor_solve(_needs("cholesky", "cusolvermp", "gpu"))


def test_cusolvermp_lu_factor_solve_token_round_trip():
    check_cusolvermp_lu_factor_solve(
        _needs("solve_lu", "cusolvermp", "gpu"))


def test_slate_factor_solve_first_execution():
    """The SLATE trsm back-solve, at 1x1.  Its 2x2 execution is leg L-c."""
    check_slate_factor_solve(_needs("cholesky", "slate", "cpu"))


def test_factor_refuses_what_it_cannot_split():
    """No library needed: every refusal here is resolve-time or shape."""
    pytest.importorskip("jax")
    check_factor_refuses_what_it_cannot_split(_mesh_1x1("cpu"))


def test_batched_eigh_dispatch_gate_native_arm():
    """The adopted gate's check 1, which needs no library: backend ``off``
    stays bit-identical to ``jnp.linalg.eigh``.  The FFI arms need a host
    mesh with ScaLAPACK and are leg L-c."""
    import jax.numpy as jnp
    mesh = _mesh_1x1("cpu")
    A = _put(_herm(np.random.default_rng(20260804), 4, 32, "complex128"),
             mesh, (None, "x", "y"))
    W_off, Z_off = D.dispatch_batched_eigh(A, mesh, "off")
    W_nat, Z_nat = jnp.linalg.eigh(A)
    assert float(jnp.max(jnp.abs(W_off - W_nat))) == 0.0
    assert float(jnp.max(jnp.abs(Z_off - Z_nat))) == 0.0


def test_batch_reshard_local_ops_smoke():
    """The real staged movement is the P=4 CLI cell of the same body."""
    check_batch_reshard_local_ops(_mesh_1x1("cpu"), nq=5, n=16, nrhs=8)


def test_gemm_plan_cublasmp_smoke():
    """Single-GPU smoke of the planned surface; real 2x2 numerics, nested
    jit/scan and the donated paths are leg L-c's ``gemm_plan_cublasmp``
    cell -- ``gemm_plan`` itself refuses a 1x1-only irrelevant geometry
    nowhere, so this cell is real coverage, not a placeholder."""
    import jax
    try:
        jax.devices("gpu")
    except Exception as exc:                                     # noqa: BLE001
        pytest.skip(f"no CUDA backend here ({exc}); covered by leg L-c")
    mesh = _mesh_1x1("gpu")
    try:
        D.resolve_matmul_backend("cublasmp", mesh)
    except (ValueError, RuntimeError) as exc:
        pytest.skip(
            f"cublasmp not usable on a 1x1 CUDA mesh: {exc}  Covered by "
            f"leg L-c (lx run -n 4 ... --mesh 2x2) where the library is "
            f"pinned and there are four ranks")
    check_gemm_plan_cublasmp(mesh, nq=2, m=2, k=3, n=2)


def test_gemm_plan_manual_shard_map_smoke():
    """Single-GPU smoke of ``GemmPlan.local_call`` inside a manual
    shard_map; real 2x2 numerics under lax.scan are leg L-c's
    ``gemm_plan_manual_shard_map`` cell."""
    import jax
    try:
        jax.devices("gpu")
    except Exception as exc:                                     # noqa: BLE001
        pytest.skip(f"no CUDA backend here ({exc}); covered by leg L-c")
    mesh = _mesh_1x1("gpu")
    try:
        D.resolve_matmul_backend("cublasmp", mesh)
    except (ValueError, RuntimeError) as exc:
        pytest.skip(
            f"cublasmp not usable on a 1x1 CUDA mesh: {exc}  Covered by "
            f"leg L-c (lx run -n 4 ... --mesh 2x2) where the library is "
            f"pinned and there are four ranks")
    check_gemm_plan_manual_shard_map(mesh, nq=2, m=2, k=3, n=2)


def test_batch_reshard_matmul_smoke():
    """The real staged movement is the P=4 CLI cell of the same body."""
    check_distributed_matmul(
        _mesh_1x1("cpu"), backend="off",
        batched_route=D.ROUTE_BATCH_RESHARD, nq=5, m=8, k=12, n=16)


def test_the_cli_cells_are_all_reachable():
    """Every ``_CLI_CELLS`` row names a function that exists and every
    check body is in the table.

    Cheap, and it is the failure the CLI mode cannot report: a typo'd or
    dropped row makes the multi-rank leg quietly run a smaller matrix and
    print ``done: 0 failures``.
    """
    names = {name for name, _, _ in _CLI_CELLS}
    assert len(names) == len(_CLI_CELLS), "duplicate _CLI_CELLS name"
    # Read the GLOBALS each row's lambda actually references, not its
    # label.  Matching on the label is the version of this check that
    # silently rots: rename a body and the substring stops matching while
    # both sides still exist.
    called = set()
    for _name, _plat, fn in _CLI_CELLS:
        called |= {g for g in fn.__code__.co_names if g.startswith("check_")}
        assert called & set(globals()), f"{_name}: calls no check body"
    bodies = {k for k in globals() if k.startswith("check_")}
    missing = bodies - called
    assert not missing, (
        f"check bodies with no _CLI_CELLS row (the 2x2 leg would never run "
        f"them): {sorted(missing)}")
    unknown = called - bodies
    assert not unknown, f"_CLI_CELLS names bodies that do not exist: {unknown}"


# ---------------------------------------------------------------------------
# CLI mode — the real multi-rank matrix.
# ---------------------------------------------------------------------------

_CLI_CELLS = [
    # (name, platform: 'cpu' | 'CUDA' | '', fn(mesh, dtype))
    ("resolution_promise", "",
     lambda mesh, dt: check_resolution_is_a_promise(mesh, dt)),
    ("factor_refusals", "",
     lambda mesh, dt: check_factor_refuses_what_it_cannot_split(mesh, dt)),
    ("scalapack_factor_solve", "cpu",
     lambda mesh, dt: check_scalapack_factor_solve(mesh, dt)),
    ("scalapack_hostile_extents", "cpu",
     lambda mesh, dt: check_hostile_extents_through_the_ffi(
         mesh, dt, backend="scalapack", op="solve_lu")),
    ("slate_factor_solve", "",
     lambda mesh, dt: check_slate_factor_solve(mesh, dt)),
    ("cusolvermp_factor_solve", "CUDA",
     lambda mesh, dt: check_cusolvermp_factor_solve(mesh, dt)),
    ("cusolvermp_lu_factor_solve", "CUDA",
     lambda mesh, dt: check_cusolvermp_lu_factor_solve(mesh, dt)),
    ("cusolvermp_hostile_extents", "CUDA",
     lambda mesh, dt: check_hostile_extents_through_the_ffi(
         mesh, dt, backend="cusolvermp", op="solve_lu")),
    ("batched_eigh_dispatch", "cpu",
     lambda mesh, dt: check_batched_eigh_dispatch(mesh, dt)),
    ("batch_reshard_local_ops", "",
     lambda mesh, dt: check_batch_reshard_local_ops(mesh, dt)),
    ("matmul_batch_reshard", "",
     lambda mesh, dt: check_distributed_matmul(
         mesh, dt, backend="off", batched_route=D.ROUTE_BATCH_RESHARD)),
    ("matmul_cublasmp", "CUDA",
     lambda mesh, dt: check_distributed_matmul(
         mesh, dt, backend="cublasmp", batched_route="auto", nq=4)),
    ("matmul_scalapack", "cpu",
     lambda mesh, dt: check_distributed_matmul(
         mesh, dt, backend="scalapack", batched_route="auto", nq=4)),
    ("matmul_slate", "",
     lambda mesh, dt: check_distributed_matmul(
         mesh, dt, backend="slate", batched_route="auto", nq=4)),
    ("gemm_plan_cublasmp", "CUDA",
     lambda mesh, dt: check_gemm_plan_cublasmp(mesh, dt)),
    ("gemm_plan_scalapack", "cpu",
     lambda mesh, dt: check_gemm_plan_scalapack(mesh, dt)),
    ("gemm_plan_scalapack_auto", "cpu",
     lambda mesh, dt: check_gemm_plan_scalapack(
         mesh, dt, backend="auto")),
    ("gemm_plan_manual_shard_map", "CUDA",
     lambda mesh, dt: check_gemm_plan_manual_shard_map(mesh, dt)),
]


def _mesh_from_arg(spec):
    import jax
    from jax.sharding import Mesh
    px, py = (int(v) for v in spec.lower().split("x"))
    return Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))


def _cli_main():
    import jax

    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="PxQ process mesh")
    ap.add_argument("--only", default="", help="substring filter")
    ap.add_argument("--dtypes", default="complex128,float64")
    args = ap.parse_args()
    mesh = _mesh_from_arg(args.mesh)
    is_cpu = jax.default_backend() == "cpu"
    p0 = print if jax.process_index() == 0 else (lambda *a, **k: None)
    p0(f"backend={jax.default_backend()} mesh={args.mesh} "
       f"processes={jax.process_count()} devices={jax.device_count()}",
       flush=True)
    for op in D.OPS:
        p0(f"  list_backends({op}) = "
           f"{ {k: v.split(':')[0] for k, v in D.list_backends(op, mesh).items()} }",
           flush=True)

    failures, ran = 0, 0
    for name, platform, fn in _CLI_CELLS:
        if args.only and args.only not in name:
            continue
        if platform == "cpu" and not is_cpu:
            p0(f"SKIP {name}[{args.mesh}] (host-only backend)", flush=True)
            continue
        if platform == "CUDA" and is_cpu:
            p0(f"SKIP {name}[{args.mesh}] (CUDA-only backend)", flush=True)
            continue
        for dt in args.dtypes.split(","):
            tag = f"{name}[{args.mesh},{dt}]"
            try:
                out = fn(mesh, dt)
                ran += 1
                p0(f"PASS {tag} {out if out is not True else ''}", flush=True)
            except AssertionError as exc:
                failures += 1
                p0(f"FAIL {tag}: {exc}", flush=True)
            except (ValueError, RuntimeError) as exc:
                # A resolve-time refusal IS the contract for an
                # unsupported mesh class; it is reported, not counted.
                p0(f"GUARD {tag}: {' '.join(str(exc).split())[:400]}",
                   flush=True)
            except Exception as exc:                           # noqa: BLE001
                failures += 1
                p0(f"ERROR {tag}: {type(exc).__name__}: "
                   f"{' '.join(str(exc).split())[:400]}", flush=True)
    # RAN, not just failures.  "0 failures" out of 0 cells is the shape of
    # every artifact-free green in this tree's history.
    p0(f"done: {ran} cells ran, {failures} failures", flush=True)
    return 1 if (failures or ran == 0) else 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(_cli_main)
