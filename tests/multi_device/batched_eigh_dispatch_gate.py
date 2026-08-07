"""``ffi.linalg.dispatch_batched_eigh``: the two paths, and the scan question.

A host mesh is the only place BOTH paths of that dispatcher are reachable:
ScaLAPACK is the only backend that exposes ``batched_distributed_eigh``
(``ffi/linalg/_scalapack.py:109``), so on CUDA the serial fallback is the
only path there is and there is nothing to compare it against.  Hence this
gate, and hence the private ``_force_serial`` argument it uses to reach the
fallback here.

``BE_MODE=gate`` (default) — seven checks:

  1. ``off``/``auto``/``native`` is still ``jnp.linalg.eigh``, BIT-IDENTICAL.
     The single-process contract the dispatcher must not disturb.
  2. batched vs serial agree in W and in Z.  Both run pXheevd on the same
     grid with the same descriptors and differ only in the ``nq`` the
     handler loops over, so the expectation is bit-identity; the delta is
     reported either way (repo doctrine: measure, do not assume).
  3. the LAYOUT contract on both paths: ``A[q] Z[q] == Z[q] diag(W[q])``,
     W replicated float64, Z at P(None,'x','y').  With the conj-transpose
     of Z as the negative control — without it the residual passes on a
     dispatcher that returned ROWS, which is exactly what the raw
     ``mod.distributed_eigh`` fallback did on CUDA before this change.
  4. eigenvalues against ``numpy.linalg.eigvalsh``.
  5. ONE resolve for an Nq-matrix serial call (the guard hoist), counted
     by wrapping ``ffi.linalg.plan.resolve_backend``.
  6. an indivisible N and an off-platform backend still REFUSE at resolve
     time — hoisting the guards must not have dropped them.
  7. ``gw.qsgw_density.distributed_eigh_bands``, the only in-tree caller,
     still works unchanged.

Every residual is computed ON DEVICE (jnp over the sharded arrays, reduced
to a replicated scalar).  No gather: a full (Nq, N, N) on one rank is the
object this whole layout contract exists to avoid.

``BE_MODE=scan`` — the ``lax.scan`` probe, in its OWN process because a
collective FFI call inside a while body can hang rather than raise.  Four
stages, reported separately: trace (``jax.eval_shape``), lower, compile,
run.  Never gates the verdict; it is a recorded finding.

Env: BE_NQ (6), BE_N (32).
"""
import os
import sys

from runtime import initialize_communicator_stack, finalize_process

RUNTIME = initialize_communicator_stack()

import numpy as np                                            # noqa: E402
import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
from jax.sharding import NamedSharding, PartitionSpec as P    # noqa: E402

from common.collectives import (process_count, process_rank,  # noqa: E402
                                resolve_mesh)
from ffi import linalg                                        # noqa: E402
from ffi.linalg import dispatch_batched_eigh                  # noqa: E402

NQ = int(os.environ.get("BE_NQ", "6"))
N = int(os.environ.get("BE_N", "32"))
RTOL = 1e-12


def _herm_stack(rng, nq, n):
    """The same Hermitian stack on every rank (identical seed, host numpy)."""
    A = (rng.standard_normal((nq, n, n))
         + 1j * rng.standard_normal((nq, n, n)))
    return 0.5 * (A + np.conj(np.swapaxes(A, -1, -2)))


def _rel(a, b):
    """max|a-b| / max|b| — the repo's relative measure, host arrays."""
    return float(np.abs(a - b).max()) / max(float(np.abs(b).max()), 1e-300)


def _residual(A_j, W, Z):
    """max |A Z - Z diag(W)| / max|A|, computed on device and reduced to a
    replicated scalar (never materialises (Nq, N, N) on a rank)."""
    lhs = jnp.einsum('qmn,qnp->qmp', A_j, Z, optimize=True)
    rhs = Z * W[:, None, :].astype(Z.dtype)
    num = jnp.max(jnp.abs(lhs - rhs))
    den = jnp.max(jnp.abs(A_j))
    return float(num) / max(float(den), 1e-300)


def main():
    rank, world = process_rank(), process_count()
    p0 = print if rank == 0 else (lambda *a, **k: None)
    mesh = resolve_mesh()
    px, py = (int(s) for s in mesh.devices.shape)
    stack_sh = NamedSharding(mesh, P(None, "x", "y"))
    p0(f"[be] world={world} mesh=({px},{py}) nq={NQ} n={N} "
       f"resolved={linalg.resolve_backend('eigh', 'distributed', mesh)}")

    rng = np.random.default_rng(20260804)
    A = _herm_stack(rng, NQ, N)
    A_j = jax.make_array_from_callback(A.shape, stack_sh, lambda idx: A[idx])

    # ---- 1. the native path is untouched --------------------------------
    W_off, Z_off = dispatch_batched_eigh(A_j, mesh, "off")
    W_nat, Z_nat = jnp.linalg.eigh(A_j)
    ok1 = (float(jnp.max(jnp.abs(W_off - W_nat))) == 0.0
           and float(jnp.max(jnp.abs(Z_off - Z_nat))) == 0.0)
    p0(f"[be] 1. backend 'off' is jnp.linalg.eigh, bit-identical  "
       f"{'PASS' if ok1 else 'FAIL'}")

    # ---- 2. batched vs serial -------------------------------------------
    W_b, Z_b = dispatch_batched_eigh(A_j, mesh, "distributed")
    W_s, Z_s = dispatch_batched_eigh(A_j, mesh, "distributed",
                                     _force_serial=True)
    d_W = _rel(np.asarray(W_s), np.asarray(W_b))          # W is replicated
    # Z is sharded and NOT gauge-invariant across MESHES, but these two
    # calls ran on the SAME mesh, so a per-shard comparison is legitimate.
    d_Z, exact_Z = 0.0, True
    for sh_s, sh_b in zip(Z_s.addressable_shards, Z_b.addressable_shards):
        a, b = np.asarray(sh_s.data), np.asarray(sh_b.data)
        d_Z = max(d_Z, _rel(a, b))
        exact_Z = exact_Z and np.array_equal(a, b)
    exact = exact_Z and np.array_equal(np.asarray(W_s), np.asarray(W_b))
    ok2 = d_W <= RTOL and d_Z <= RTOL
    p0(f"[be] 2. batched vs serial  W rel {d_W:.3e}  Z rel {d_Z:.3e}  "
       f"bit-identical={exact}  {'PASS' if ok2 else 'FAIL'}")

    # ---- 3. the layout contract on BOTH paths ---------------------------
    ok3 = True
    for tag, W_x, Z_x in (("a batched", W_b, Z_b), ("b serial ", W_s, Z_s)):
        r_col = _residual(A_j, W_x, Z_x)
        r_row = _residual(A_j, W_x, jnp.conj(jnp.swapaxes(Z_x, -1, -2)))
        spec_ok = (Z_x.sharding.spec == P(None, "x", "y")
                   and Z_x.sharding.mesh == mesh
                   and W_x.sharding.is_fully_replicated
                   and W_x.dtype == jnp.float64)
        ok = r_col <= 1e-12 and r_row > 1e-6 and spec_ok
        ok3 = ok3 and ok
        p0(f"[be] 3{tag}: A Z == Z diag(W) rel {r_col:.3e}  transposed "
           f"control {r_row:.3e}  Z spec={Z_x.sharding.spec} W repl="
           f"{W_x.sharding.is_fully_replicated} W dtype={W_x.dtype}  "
           f"{'PASS' if ok else 'FAIL'}")

    # ---- 4. eigenvalues -------------------------------------------------
    E_ref = np.linalg.eigvalsh(A)
    d_E = _rel(np.asarray(W_b), E_ref)
    ok4 = d_E <= 1e-10
    p0(f"[be] 4. eigenvalues vs numpy eigvalsh  rel {d_E:.3e}  "
       f"{'PASS' if ok4 else 'FAIL'}")

    # ---- 5. ONE resolve for the whole serial stack ----------------------
    # Patch the module whose globals ``plan()`` actually looks ``resolve_
    # backend`` up in.  That is ``distrib_la.plan`` since the facade moved
    # to services/distrib_la/; ``ffi.linalg.plan`` is a re-export shim, so
    # rebinding its attribute would count ZERO calls and this check would
    # pass while measuring nothing.  (``import ffi.linalg.plan`` also gives
    # back the FUNCTION, not the module — the package re-exports ``plan``
    # and that binding shadows the submodule of the same name — which is
    # why this goes through sys.modules either way.)
    _planmod = sys.modules["distrib_la.plan"]
    seen = []
    _orig = _planmod.resolve_backend

    def _counting(op, requested, mesh_, **kw):
        seen.append((op, requested))
        return _orig(op, requested, mesh_, **kw)

    _planmod.resolve_backend = _counting
    try:
        dispatch_batched_eigh(A_j, mesh, "distributed", _force_serial=True)
    finally:
        _planmod.resolve_backend = _orig
    ok5 = len(seen) == 1
    p0(f"[be] 5. resolve calls for a {NQ}-matrix SERIAL dispatch: "
       f"{len(seen)} (want 1)  {'PASS' if ok5 else 'FAIL'}")

    # ---- 6. the hoisted guards still refuse -----------------------------
    ok6a = ok6b = False
    if px > 1 or py > 1:
        bad = jnp.zeros((2, N + 1, N + 1), dtype=jnp.complex128)
        try:
            dispatch_batched_eigh(bad, mesh, "distributed")
        except ValueError as exc:
            ok6a = "divisible" in str(exc)
    else:
        ok6a = True                     # no divisibility rule on a 1x1 mesh
    try:
        dispatch_batched_eigh(A_j, mesh, "cusolvermp")
    except (ValueError, RuntimeError) as exc:
        ok6b = "CUDA-only" in str(exc) or "not usable" in str(exc)
    ok6 = ok6a and ok6b
    p0(f"[be] 6. hoisted guards still refuse: indivisible-N={ok6a} "
       f"off-platform={ok6b}  {'PASS' if ok6 else 'FAIL'}")

    # ---- 7. the only in-tree caller -------------------------------------
    from gw.qsgw_density import distributed_eigh_bands
    E_c, U_c = distributed_eigh_bands(A_j, mesh=mesh)
    padded = tuple(E_c.shape) != (NQ, N)
    d_c = -1.0 if padded else _rel(np.asarray(E_c), E_ref)
    ok7 = (U_c.sharding.spec == P(None, "x", "y")
           and (padded or d_c <= 1e-10))
    p0(f"[be] 7. qsgw_density.distributed_eigh_bands  E shape "
       f"{tuple(E_c.shape)} padded={padded}  rel {d_c:.3e}  "
       f"U spec={U_c.sharding.spec}  {'PASS' if ok7 else 'FAIL'}")

    ok = ok1 and ok2 and ok3 and ok4 and ok5 and ok6 and ok7
    p0(f"[be] VERDICT {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def scan_probe():
    """Can the serial loop be a ``lax.scan`` over the single-matrix FFI call?

    Staged, so the answer names WHICH stage refuses: JAX's transform rules
    (trace), XLA (lower / compile), or the run.  Its own process — a
    collective custom call inside a while body can hang instead of raising.
    """
    p0 = print if process_rank() == 0 else (lambda *a, **k: None)
    mesh = resolve_mesh()
    resolved = linalg.resolve_backend("eigh", "distributed", mesh)
    mod = linalg.backend_module(resolved)
    stack_sh = NamedSharding(mesh, P(None, "x", "y"))
    rng = np.random.default_rng(20260804)
    A = _herm_stack(rng, NQ, N)
    A_j = jax.make_array_from_callback(A.shape, stack_sh, lambda idx: A[idx])

    def body(carry, Aq):
        W, Z = mod.distributed_eigh(Aq, mesh=mesh)
        return carry, (W, Z)

    def scanned(A_):
        _, out = jax.lax.scan(body, jnp.zeros((), jnp.float64), A_)
        return out

    p0(f"[scan] backend={resolved} mesh={tuple(mesh.devices.shape)} "
       f"nq={NQ} n={N}")
    stage = "trace"
    try:
        shp = jax.eval_shape(scanned, A_j)
        p0(f"[scan] 1. TRACE ok -> {[(s.shape, str(s.dtype)) for s in shp]}")
        stage = "lower"
        low = jax.jit(scanned).lower(A_j)
        p0("[scan] 2. LOWER ok")
        stage = "compile"
        exe = low.compile()
        p0("[scan] 3. COMPILE ok")
        stage = "run"
        W_s, _ = exe(A_j)
        W_ref, _ = dispatch_batched_eigh(A_j, mesh, "distributed")
        W_nat, _ = jnp.linalg.eigh(A_j)
        d_ffi = _rel(np.asarray(W_s), np.asarray(W_ref))
        # DISTINCTNESS: the scan result matching the FFI result proves
        # nothing unless the FFI result is itself distinguishable from
        # jnp.linalg.eigh.  pzheevd on a 2-D grid and LAPACK zheevd on one
        # rank are different reductions, so this must be nonzero.
        d_nat = _rel(np.asarray(W_ref), np.asarray(W_nat))
        p0(f"[scan] 4. RUN ok  W(scan) vs W(FFI) rel {d_ffi:.3e}  "
           f"W(FFI) vs W(jnp) rel {d_nat:.3e} (must be > 0, else the "
           f"agreement above is vacuous)")
        # COST: is the scan actually worth anything?  Both are warm here
        # (one cached executable each); this times DISPATCH, not compile.
        import time
        for _ in range(2):                       # warm both
            jax.block_until_ready(exe(A_j))
            jax.block_until_ready(dispatch_batched_eigh(
                A_j, mesh, "distributed", _force_serial=True))
        t0 = time.perf_counter()
        for _ in range(3):
            jax.block_until_ready(exe(A_j))
        t_scan = (time.perf_counter() - t0) / 3
        t0 = time.perf_counter()
        for _ in range(3):
            jax.block_until_ready(dispatch_batched_eigh(
                A_j, mesh, "distributed", _force_serial=True))
        t_loop = (time.perf_counter() - t0) / 3
        t0 = time.perf_counter()
        for _ in range(3):
            jax.block_until_ready(dispatch_batched_eigh(
                A_j, mesh, "distributed"))
        t_batched = (time.perf_counter() - t0) / 3
        p0(f"[scan] 5. COST per call (nq={NQ}, n={N}): scan {t_scan*1e3:.1f} "
           f"ms  python-loop {t_loop*1e3:.1f} ms  batched-entry "
           f"{t_batched*1e3:.1f} ms")
        p0("[scan] VERDICT lax.scan over the FFI eigh IS viable here")
    except BaseException as exc:                                # noqa: BLE001
        msg = " ".join(str(exc).split())[:700]
        p0(f"[scan] REFUSED at stage {stage!r}: {type(exc).__name__}: {msg}")
        p0("[scan] VERDICT lax.scan over the FFI eigh is NOT viable here")
    return 0


if __name__ == "__main__":
    import traceback
    rc = 1
    try:
        rc = scan_probe() if os.environ.get("BE_MODE") == "scan" else main()
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        rc = 1
    finalize_process(rc)
