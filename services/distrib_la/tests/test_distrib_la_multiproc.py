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

# CLI multi-rank mode: jax.distributed.initialize must run before ANY
# XLA-backend touch, so it happens at import time when this module is the
# entry point of a multi-task launch.  Same order as the contract suite's
# CLI mode, for the same measured reason (job 7885123): the jax CPU mpi
# collectives plugin calls MPI_Init_thread unconditionally, so the SLATE
# warm-up that also initializes MPI must stay on CUDA only.
if __name__ == "__main__":
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        from lxkit.gate import platform_from_env
        _plat = platform_from_env()
        if _plat == "CUDA":
            try:
                from distrib_la.loader import get_lib as _get_lib
                _get_lib(_plat).lrx_slate_init_mpi()
            except Exception as _exc:                          # noqa: BLE001
                print(f"slate_init_mpi skipped: {_exc}", flush=True)
        import jax
        if _plat == "CUDA":
            jax.distributed.initialize(local_device_ids=[0])
        else:
            jax.distributed.initialize()

import distrib_la as D                                        # noqa: E402
from lxkit.testing import hostile_extents                     # noqa: E402

#: The engine-parity bar (C8): RELATIVE 1e-12, never bit-exact.  A blocked
#: distributed factorization and a single-rank LAPACK call are different
#: reductions in a different order; demanding bit-equality across them
#: would be demanding that the library not be distributed.
RTOL = 1e-12


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
            if op == "eigh":
                D.plan(op, mesh, backend=backend, n=n).batched(A)
            elif op == "solve_lu":
                B = jnp.asarray(_rng_mat(rng, (2, n, 32), dtype))
                D.plan(op, mesh, backend=backend, n=n).batched(A, B)
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
    ("cusolvermp_hostile_extents", "CUDA",
     lambda mesh, dt: check_hostile_extents_through_the_ffi(
         mesh, dt, backend="cusolvermp", op="solve_lu")),
    ("batched_eigh_dispatch", "cpu",
     lambda mesh, dt: check_batched_eigh_dispatch(mesh, dt)),
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
    sys.exit(_cli_main())
