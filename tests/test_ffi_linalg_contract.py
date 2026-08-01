"""Contract tests for the block-cyclic distributed-linalg FFI wrappers.

One test per (op x backend): residual against a numpy/scipy reference AND
bit-exact rerun determinism, on a 1x1 mesh (single process, one GPU) so
the suite runs on any dev box.  The full multi-rank matrix (2x2 / 4x1 /
1x4 process meshes) runs via the CLI mode of this same file inside a
multi-task allocation — same check functions, no duplicated logic:

    lxrun python3 -m tests.test_ffi_linalg_contract --mesh 2x2

Optional-dependency semantics: every test SKIPS cleanly when
``liblorrax_ffi.so`` (or any native library it links: cuSOLVERMp,
cuBLASMp, SLATE, NCCL, MPI) is absent — ``_ffi_skip_reason`` probes by
actually loading the library.  Nothing here may fail on a machine
without the FFI stack.

Host platform: the slate ops also have CPU-backend handlers
(``liblorrax_ffi_host.so``, registered under platform="cpu" — see
src/ffi/cpp/slate/host_ffi.cc).  The ``*_cpu`` tests run the SAME check
bodies on a 1x1 mesh of CPU devices in this very process (works on GPU
nodes too: ``jax.ffi.ffi_call`` picks the handler by lowering platform)
and skip cleanly when the host library is absent.  Multi-rank CPU
meshes: run the CLI mode under ``JAX_PLATFORMS=cpu`` (non-slate cells
are skipped there; ``--only slate`` narrows the log to the same set).
Note: when BOTH libraries are loaded (GPU node), the shared
``libslate.so.2`` soname means the host handlers run against the
already-loaded cuda-build SLATE (host-side execution) — the
``gpu_backend=none`` binary is exercised on CPU nodes, where the host
library loads alone (see src/ffi/slate/README.md "Dual-lib caveat").

Multi-rank findings this file pins (see
reports/slate_linalg_ffi_2026-07-10/report.md for the full matrix):

* slate trsm with a rectangular RHS (m != n) used square-nb tiles for X
  — uncatchable ``blas::Error`` abort (2x2) or silent corruption (1x4).
  Fixed in trsm_ffi.cc / batched_trsm_ffi.cc; the rectangular case is
  exercised here on 1x1 and by the CLI matrix on multi-rank meshes.
* slate eigh returned stale pre-back-transform eigenvector tiles (MOSI
  misuse) with a missing layout transpose on top.  Fixed; the strict
  ``A @ Q == Q @ diag(W)`` contract is asserted here.
* cuBLASMp comm-ABI generation must match the loaded library, not the
  cuSOLVERMp version (status=6 on every mesh when the stages drift).
* cusolverMp syevd DEADLOCKS on non-square meshes (mb != nb) — no
  status, just a hang.  distributed_eigh is square-mesh-only; there is
  no wrapper-level guard cheap enough to test single-process, so the
  CLI matrix documents it (XFAIL row).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pytest

# CLI multi-rank mode: jax.distributed.initialize must run before ANY
# XLA-backend touch — including the availability probe below — so it
# happens at import time when this module is the entry point of a
# multi-task launch.
if __name__ == "__main__":
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    if int(os.environ.get("SLURM_NTASKS", "1")) > 1:
        # Platform must be EXPLICIT here: get_lib(None) asks
        # jax.default_backend(), which would initialize XLA before
        # jax.distributed.initialize.
        from ffi.common.ffi_loader import platform_from_env
        _plat = platform_from_env()
        # Production init ORDER (host + impl=mpi): the jax CPU mpi
        # collectives plugin calls MPI_Init_thread unconditionally, so it
        # must initialize MPI FIRST and the slate/BLACS context piggyback
        # (its ensure_mpi_initialized is guarded; the plugin's is not).
        # Eagerly calling lrx_slate_init_mpi here — the CUDA-era warm-up —
        # aborts every cpu+impl=mpi run with "Cannot call MPI_INIT ...
        # more than once" (job 7885123).  Keep the warm-up on CUDA only.
        if _plat == "CUDA":
            try:
                from ffi.common.ffi_loader import get_lib as _get_lib
                _get_lib(_plat).lrx_slate_init_mpi()
            except Exception as _exc:
                print(f"slate_init_mpi skipped: {_exc}", flush=True)
        import jax
        if _plat == "CUDA":
            jax.distributed.initialize(local_device_ids=[0])
        else:
            jax.distributed.initialize()

# h5py has to bind its HDF5 BEFORE anything dlopens liblorrax_ffi.so: the
# FFI library links the Cray parallel HDF5 out of /lorrax_phdf5, and h5py
# imported afterwards initialises against those symbols and dies with
# "ValueError: Not a datatype (not a datatype)".  The availability probes
# below dlopen the FFI library at module import, and the production-wiring
# checks import LORRAX modules that pull h5py — so the order is forced here.
try:                                     # noqa: E402
    import h5py                          # noqa: F401
except Exception:
    pass

# ---------------------------------------------------------------------------
# Availability probes (module import must stay cheap + exception-free).
# ---------------------------------------------------------------------------


def _ffi_skip_reason():
    """Return None if liblorrax_ffi.so loads, else the reason to skip.

    Loading the .so pulls every native dependency (cuSOLVERMp, cuBLASMp,
    SLATE, NCCL, HDF5, MPI) via DT_NEEDED, so one probe covers them all.
    """
    try:
        import jax
        if not any(getattr(d, "platform", "") in ("gpu", "cuda")
                   for d in jax.devices()):
            return "no CUDA GPU visible"
    except Exception as exc:  # jax missing / no backend
        return f"jax backend unavailable: {exc}"
    if jax.process_count() != 1:
        return "contract tests are single-process (use the CLI mode)"
    try:
        from ffi.common.ffi_loader import get_lib
        get_lib("CUDA")
    except Exception as exc:
        return f"liblorrax_ffi.so unavailable: {exc}"
    return None


def _host_ffi_skip_reason():
    """Return None if liblorrax_ffi_host.so loads, else the skip reason.

    The host library is CUDA-free (slate + MPI only), so this probe
    passes on CPU-only machines where ``_ffi_skip_reason`` skips.
    """
    try:
        import jax
        jax.devices("cpu")
    except Exception as exc:
        return f"jax cpu backend unavailable: {exc}"
    if jax.process_count() != 1:
        return "contract tests are single-process (use the CLI mode)"
    try:
        from ffi.common.ffi_loader import get_lib
        get_lib("cpu")
    except Exception as exc:
        return f"liblorrax_ffi_host.so unavailable: {exc}"
    return None


_SKIP = _ffi_skip_reason()
needs_ffi = pytest.mark.skipif(_SKIP is not None, reason=_SKIP or "")

_HOST_SKIP = _host_ffi_skip_reason()
needs_host_ffi = pytest.mark.skipif(_HOST_SKIP is not None,
                                    reason=_HOST_SKIP or "")


def _mesh_1x1():
    import jax
    from jax.sharding import Mesh
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))


def _mesh_cpu_1x1():
    import jax
    from jax.sharding import Mesh
    return Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))


def _mesh_from_arg(spec):
    import jax
    from jax.sharding import Mesh
    px, py = (int(v) for v in spec.lower().split("x"))
    return Mesh(np.asarray(jax.devices()).reshape(px, py), ("x", "y"))


def _put(np_arr, mesh, spec):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.device_put(np_arr, NamedSharding(mesh, P(*spec)))


def _gather(x):
    import jax
    if jax.process_count() == 1:
        return np.asarray(x)
    from jax.experimental import multihost_utils
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))


def _rng_mat(rng, shape, dtype):
    a = rng.standard_normal(shape)
    if np.dtype(dtype).kind == "c":
        a = a + 1j * rng.standard_normal(shape)
    return a.astype(dtype)


def _hpd(rng, nq, n, dtype):
    z = _rng_mat(rng, (nq, n, n), dtype)
    return (0.5 * (z + np.conj(np.swapaxes(z, -1, -2)))
            + n * np.eye(n)[None]).astype(dtype)


def _herm(rng, nq, n, dtype):
    z = _rng_mat(rng, (nq, n, n), dtype)
    return (0.5 * (z + np.conj(np.swapaxes(z, -1, -2)))).astype(dtype)


# ---------------------------------------------------------------------------
# Check bodies — shared between the 1x1 pytest cases and the CLI matrix.
# Each returns None on success and raises AssertionError with residuals
# in the message on failure.
# ---------------------------------------------------------------------------


def check_cusolvermp_chol(mesh, dtype, nq=2, n=32, mrhs=48):
    from ffi.cusolvermp import (batched_distributed_cholesky,
                                batched_distributed_potrs,
                                cholesky_handle_to_natural_L)
    rng = np.random.default_rng(7)
    A_np = _hpd(rng, nq, n, dtype)
    B_np = _rng_mat(rng, (nq, n, mrhs), dtype)

    def solve():
        A = _put(A_np, mesh, (None, "x", "y"))
        B = _put(B_np, mesh, (None, "x", "y"))
        L = batched_distributed_cholesky(A, mesh=mesh)
        X = batched_distributed_potrs(L, B, mesh=mesh)
        return _gather(X), _gather(cholesky_handle_to_natural_L(L))

    X1, Ln = solve()
    X2, _ = solve()
    assert np.array_equal(X1, X2), "potrf+potrs rerun not bit-deterministic"
    for q in range(nq):
        res_x = (np.linalg.norm(A_np[q] @ X1[q] - B_np[q])
                 / max(np.linalg.norm(B_np[q]), 1.0))
        res_l = (np.linalg.norm(Ln[q] @ np.conj(Ln[q].T) - A_np[q])
                 / max(np.linalg.norm(A_np[q]), 1.0))
        assert res_x < 1e-11 and res_l < 1e-11, \
            f"q={q}: res_x={res_x:.3e} res_L={res_l:.3e}"


def check_cusolvermp_lu(mesh, dtype, nq=2, n=32, nrhs=16, herm=True):
    from ffi.cusolvermp import batched_distributed_solve_lu
    rng = np.random.default_rng(11)
    if herm:
        A_np = _herm(rng, nq, n, dtype)     # Hermitian INDEFINITE
    else:
        A_np = _rng_mat(rng, (nq, n, n), dtype) + n * np.eye(n)[None]
        A_np = A_np.astype(dtype)
    B_np = _rng_mat(rng, (nq, n, nrhs), dtype)

    def solve():
        A = _put(A_np, mesh, (None, "x", "y"))
        B = _put(B_np, mesh, (None, "x", "y"))
        return _gather(batched_distributed_solve_lu(A, B, mesh=mesh))

    X1 = solve()
    X2 = solve()
    assert np.array_equal(X1, X2), "solve_lu rerun not bit-deterministic"
    for q in range(nq):
        res = (np.linalg.norm(A_np[q] @ X1[q] - B_np[q])
               / max(np.linalg.norm(B_np[q]), 1.0))
        assert res < 1e-10, f"q={q}: res={res:.3e}"


def check_cusolvermp_eigh(mesh, dtype, n=32):
    """Contract: eigenvalues match numpy; the RAW Q buffer satisfies
    ``Q_raw^H = eigenvectors`` (cuSOLVERMp writes col-major tiles that JAX
    reads row-major — the documented conj-transpose convention)."""
    from ffi.cusolvermp import distributed_eigh
    rng = np.random.default_rng(13)
    A_np = _herm(rng, 1, n, dtype)[0]

    def solve():
        A = _put(A_np, mesh, ("x", "y"))
        W, Q = distributed_eigh(A, mesh=mesh)
        return _gather(W)[:n], _gather(Q)

    W1, Q1 = solve()
    W2, Q2 = solve()
    assert np.array_equal(W1, W2) and np.array_equal(Q1, Q2), \
        "eigh rerun not bit-deterministic"
    ev_err = float(np.max(np.abs(W1 - np.linalg.eigvalsh(A_np))))
    assert ev_err < 1e-10, f"eigenvalue error {ev_err:.3e}"
    Qh = np.conj(Q1.T)
    res = (np.linalg.norm(A_np @ Qh - Qh * W1[None, :])
           / max(np.linalg.norm(A_np), 1.0))
    assert res < 1e-11, f"Q^H eigvec residual {res:.3e}"


def check_cublasmp_gemm(mesh, dtype, transa="C", transb="N"):
    from ffi.cublasmp import batched_distributed_gemm
    rng = np.random.default_rng(19)
    nq, m, k, n = 2, 32, 16, 24
    alpha, beta = ((1.3 - 0.7j, 0.4 + 0.2j)
                   if np.dtype(dtype).kind == "c" else (1.5, 0.5))
    A_np = _rng_mat(rng, (nq, m, k) if transa == "N" else (nq, k, m), dtype)
    B_np = _rng_mat(rng, (nq, k, n) if transb == "N" else (nq, n, k), dtype)
    C_np = _rng_mat(rng, (nq, m, n), dtype)

    def opx(X, t):
        if t == "N":
            return X
        Xt = np.swapaxes(X, -1, -2)
        return Xt if t == "T" else np.conj(Xt)

    def solve():
        A = _put(A_np, mesh, (None, "x", "y"))
        B = _put(B_np, mesh, (None, "x", "y"))
        C = _put(C_np, mesh, (None, "x", "y"))
        return _gather(batched_distributed_gemm(
            A, B, C, mesh=mesh, alpha=alpha, beta=beta,
            transa=transa, transb=transb))

    D1 = solve()
    D2 = solve()
    assert np.array_equal(D1, D2), "gemm rerun not bit-deterministic"
    D_ref = (alpha * np.einsum("qij,qjk->qik", opx(A_np, transa),
                               opx(B_np, transb)) + beta * C_np)
    res = np.linalg.norm(D1 - D_ref) / max(np.linalg.norm(D_ref), 1.0)
    assert res < 1e-12, f"gemm residual {res:.3e}"


def check_cublasmp_wsolve(mesh, dtype, nq=2, n=32, pref=0.37):
    from ffi.cublasmp import batched_fused_w_solve
    rng = np.random.default_rng(23)
    V_np = _hpd(rng, nq, n, dtype)
    C = _rng_mat(rng, (nq, n, n), dtype)
    chi_np = (-(C @ np.conj(np.swapaxes(C, -1, -2))) / n).astype(dtype)

    def solve():
        V = _put(V_np, mesh, (None, "x", "y"))
        chi = _put(chi_np, mesh, (None, "x", "y"))
        return _gather(batched_fused_w_solve(V, chi, pref, mesh=mesh))

    W1 = solve()
    W2 = solve()
    assert np.array_equal(W1, W2), "w_solve rerun not bit-deterministic"
    eye = np.eye(n)
    for q in range(nq):
        # W = X (I - X^H pref.chi X)^{-1} X^H  ==  (I - pref V chi)^{-1} V
        W_ref = np.linalg.solve(eye - pref * V_np[q] @ chi_np[q], V_np[q])
        res = (np.linalg.norm(W1[q] - W_ref)
               / max(np.linalg.norm(W_ref), 1.0))
        assert res < 1e-11, f"q={q}: w_solve residual {res:.3e}"


def check_scalapack_lu(mesh, dtype, nq=2, n=32, nrhs=16, herm=True):
    """Host-platform twin of check_cusolvermp_lu — ScaLAPACK pXgetrf+
    pXgetrs (Cray LibSci) through ffi.scalapack, same math contract."""
    from ffi.scalapack import batched_distributed_solve_lu
    rng = np.random.default_rng(11)
    if herm:
        A_np = _herm(rng, nq, n, dtype)     # Hermitian INDEFINITE
    else:
        A_np = _rng_mat(rng, (nq, n, n), dtype) + n * np.eye(n)[None]
        A_np = A_np.astype(dtype)
    B_np = _rng_mat(rng, (nq, n, nrhs), dtype)

    def solve():
        A = _put(A_np, mesh, (None, "x", "y"))
        B = _put(B_np, mesh, (None, "x", "y"))
        return _gather(batched_distributed_solve_lu(A, B, mesh=mesh))

    X1 = solve()
    X2 = solve()
    assert np.array_equal(X1, X2), "scalapack solve_lu rerun not bit-deterministic"
    for q in range(nq):
        res = (np.linalg.norm(A_np[q] @ X1[q] - B_np[q])
               / max(np.linalg.norm(B_np[q]), 1.0))
        assert res < 1e-10, f"q={q}: res={res:.3e}"


def check_scalapack_getrf_getrs(mesh, dtype, nq=2, n=32, nrhs=16):
    """The SPLIT pair (hoisted transverse ζ factor stage): pXgetrf once,
    pXgetrs per RHS, must be BIT-IDENTICAL to the fused
    batched_distributed_solve_lu on the same operands — same descriptors,
    same grid, only WHEN the factor work happens differs.  Also solves a
    second RHS from the SAME factors (the r-chunk reuse the split
    exists for)."""
    from ffi.scalapack import (batched_distributed_getrf,
                               batched_distributed_getrs,
                               batched_distributed_solve_lu)
    rng = np.random.default_rng(11)
    A_np = _herm(rng, nq, n, dtype)         # Hermitian INDEFINITE
    B_np = _rng_mat(rng, (nq, n, nrhs), dtype)
    B2_np = _rng_mat(rng, (nq, n, nrhs), dtype)

    def fused(Bx):
        A = _put(A_np, mesh, (None, "x", "y"))
        B = _put(Bx, mesh, (None, "x", "y"))
        return _gather(batched_distributed_solve_lu(A, B, mesh=mesh))

    def split():
        A = _put(A_np, mesh, (None, "x", "y"))
        LU, ipiv = batched_distributed_getrf(A, mesh=mesh)
        X = _gather(batched_distributed_getrs(
            LU, ipiv, _put(B_np, mesh, (None, "x", "y")), mesh=mesh))
        X2 = _gather(batched_distributed_getrs(
            LU, ipiv, _put(B2_np, mesh, (None, "x", "y")), mesh=mesh))
        return X, X2

    Xf, Xf2 = fused(B_np), fused(B2_np)
    Xs, Xs2 = split()
    assert np.array_equal(Xf, Xs), \
        "split getrf+getrs drifts from fused solve_lu (RHS 1)"
    assert np.array_equal(Xf2, Xs2), \
        "split getrf+getrs drifts from fused solve_lu (RHS 2, factor reuse)"
    for q in range(nq):
        res = (np.linalg.norm(A_np[q] @ Xs[q] - B_np[q])
               / max(np.linalg.norm(B_np[q]), 1.0))
        assert res < 1e-10, f"q={q}: res={res:.3e}"


def check_slate_chol_trsm(mesh, dtype, n=32, m=32):
    """Includes the rectangular-RHS case (m != n) that used to abort the
    whole multi-rank job via an uncatchable blas::Error."""
    from ffi.slate import distributed_cholesky, distributed_trsm
    rng = np.random.default_rng(29)
    A_np = _hpd(rng, 1, n, dtype)[0]
    B_np = _rng_mat(rng, (n, m), dtype)

    def solve():
        A = _put(A_np, mesh, ("x", "y"))
        B = _put(B_np, mesh, ("x", "y"))
        L = distributed_cholesky(A, mesh=mesh)
        Xf = distributed_trsm(L, B, mesh=mesh, op="N")
        Xa = distributed_trsm(L, B, mesh=mesh, op="C")
        return _gather(L.to_jax_lower()), _gather(Xf), _gather(Xa)

    L1, Xf1, Xa1 = solve()
    L2, Xf2, Xa2 = solve()
    assert (np.array_equal(L1, L2) and np.array_equal(Xf1, Xf2)
            and np.array_equal(Xa1, Xa2)), \
        "cholesky+trsm rerun not bit-deterministic"
    nrmA = max(np.linalg.norm(A_np), 1.0)
    nrmB = max(np.linalg.norm(B_np), 1.0)
    res_l = np.linalg.norm(L1 @ np.conj(L1.T) - A_np) / nrmA
    res_f = np.linalg.norm(L1 @ Xf1 - B_np) / nrmB
    res_a = np.linalg.norm(np.conj(L1.T) @ Xa1 - B_np) / nrmB
    assert max(res_l, res_f, res_a) < 1e-12, \
        f"L={res_l:.3e} fwd={res_f:.3e} adj={res_a:.3e}"


def check_slate_batched(mesh, dtype, nbatch=4, n=16, nrhs=8):
    """nrhs != n exercises the rectangular batched-trsm tile fix."""
    from ffi.slate import (batched_distributed_cholesky,
                           batched_distributed_trsm)
    rng = np.random.default_rng(31)
    A_np = _hpd(rng, nbatch, n, dtype)
    B_np = _rng_mat(rng, (nbatch, n, nrhs), dtype)

    def solve():
        A = _put(A_np, mesh, ("x", None, "y"))
        B = _put(B_np, mesh, ("x", None, "y"))
        L = batched_distributed_cholesky(A, mesh=mesh)
        return _gather(batched_distributed_trsm(L, B, mesh=mesh, op="N"))

    X1 = solve()
    X2 = solve()
    assert np.array_equal(X1, X2), "batched rerun not bit-deterministic"
    for b in range(nbatch):
        L_ref = np.linalg.cholesky(A_np[b])
        res = (np.linalg.norm(L_ref @ X1[b] - B_np[b])
               / max(np.linalg.norm(B_np[b]), 1.0))
        assert res < 1e-12, f"b={b}: residual {res:.3e}"


def check_slate_eigh(mesh, dtype, n=32):
    """Strict contract (post-fix): W matches numpy, Q columns are TRUE
    eigenvectors of A (``A @ Q == Q @ diag(W)``), Q unitary."""
    from ffi.slate import distributed_eigh
    rng = np.random.default_rng(37)
    A_np = _herm(rng, 1, n, dtype)[0]

    def solve():
        A = _put(A_np, mesh, ("x", "y"))
        W, Q = distributed_eigh(A, mesh=mesh)
        return _gather(W)[:n], _gather(Q)

    W1, Q1 = solve()
    W2, Q2 = solve()
    assert np.array_equal(W1, W2) and np.array_equal(Q1, Q2), \
        "slate eigh rerun not bit-deterministic"
    ev_err = float(np.max(np.abs(W1 - np.linalg.eigvalsh(A_np))))
    assert ev_err < 1e-10, f"eigenvalue error {ev_err:.3e}"
    nrmA = max(np.linalg.norm(A_np), 1.0)
    res = np.linalg.norm(A_np @ Q1 - Q1 * W1[None, :]) / nrmA
    orth = np.linalg.norm(np.conj(Q1.T) @ Q1 - np.eye(n))
    assert res < 1e-11 and orth < 1e-11, \
        f"eigvec residual {res:.3e}, orthonormality {orth:.3e}"


def check_padding_solve_at_logical(mesh, dtype, nq=2, n_log=13, n_pad=16,
                                   nrhs=8):
    """Pad-extent contract (the 2.5 eV device-invariance bug class):
    identity-padded operands + ``solve_at_logical`` must reproduce the
    unpadded solve on the logical block bit-for-bit, with exact-zero pad
    rows, through the REAL distributed solver."""
    from ffi.cusolvermp import batched_distributed_solve_lu
    from runtime.padding import pad_last_axis_to, solve_at_logical
    rng = np.random.default_rng(41)
    A_log = _herm(rng, nq, n_log, dtype)
    B_log = _rng_mat(rng, (nq, n_log, nrhs), dtype)

    # Padded operands per the Phase-3a contract: zero pad rows/cols,
    # identity on the pad-block diagonal of A.
    A_pad = np.zeros((nq, n_pad, n_pad), dtype)
    A_pad[:, :n_log, :n_log] = A_log
    for i in range(n_log, n_pad):
        A_pad[:, i, i] = 1.0
    B_pad = np.zeros((nq, n_pad, nrhs), dtype)
    B_pad[:, :n_log, :] = B_log

    def dist_solve(A_np, B_np):
        A = _put(A_np, mesh, (None, "x", "y"))
        B = _put(B_np, mesh, (None, "x", "y"))
        return batched_distributed_solve_lu(A, B, mesh=mesh)

    X_log = _gather(dist_solve(A_log, B_log))
    X_pad = _gather(solve_at_logical(
        dist_solve, n_log, (_put(A_pad, mesh, (None, "x", "y")),),
        _put(B_pad, mesh, (None, "x", "y"))))

    assert X_pad.shape == (nq, n_pad, nrhs)
    assert np.array_equal(X_pad[:, :n_log, :], X_log), \
        "padded solve_at_logical != logical solve (pad-extent leak)"
    assert not X_pad[:, n_log:, :].any(), "pad rows not exact zeros"
    # NRHS-pad idiom: zero RHS columns -> zero solution columns, retained
    # columns bit-identical (per-column triangular solves are independent).
    B_wide, n_orig = pad_last_axis_to(
        _put(B_log, mesh, (None, "x", "y")), 5)   # 8 -> 10 columns
    assert n_orig == nrhs
    B_wide_np = np.asarray(_gather(B_wide))
    assert B_wide_np.shape[-1] > nrhs, "pad divisor made the check a no-op"
    X_wide = _gather(dist_solve(A_log, B_wide_np))
    assert np.array_equal(X_wide[..., :nrhs], X_log), \
        "NRHS zero-pad changed the retained solution columns"
    assert not X_wide[..., nrhs:].any(), "zero RHS cols gave nonzero X cols"


def check_tile_layout_validation():
    """Pure-logic contract for the SLATE layout guard (no GPU needed):
    the guard must reject exactly the (n, nb, p, q) combos where JAX's
    block shards diverge from SLATE's block-cyclic tiles, plus the 1xq
    grids whose lld != nb stride mismatch SIGABRTs inside SLATE."""
    from ffi.slate.context import validate_tile_layout
    validate_tile_layout(64, 64, 1, 1, what="t")       # 1x1: any nb
    validate_tile_layout(64, 7, 1, 1, what="t")
    validate_tile_layout(64, 32, 2, 2, what="t")       # one tile per rank
    validate_tile_layout(64, 16, 4, 1, what="t")
    # 1xq: rejected for single-matrix ops (stride assert)...
    with pytest.raises(ValueError):
        validate_tile_layout(64, 16, 1, 4, what="t")
    # ...but allowed for the batched wrappers' (1, Py) sub-grids.
    validate_tile_layout(64, 16, 1, 4, what="t", allow_row_grid=True)
    with pytest.raises(ValueError):                    # p!=q, both >1
        validate_tile_layout(64, 16, 2, 4, what="t")
    with pytest.raises(ValueError):                    # nb != n/p
        validate_tile_layout(64, 16, 2, 2, what="t")
    with pytest.raises(ValueError):                    # nb != n/q
        validate_tile_layout(64, 8, 1, 4, what="t", allow_row_grid=True)


# ---------------------------------------------------------------------------
# pytest entry points — 1x1 mesh, one test per (op x backend).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh11():
    return _mesh_1x1()


@needs_ffi
@pytest.mark.parametrize("dtype", ["complex128", "float64"])
def test_cusolvermp_batched_cholesky_potrs(mesh11, dtype):
    check_cusolvermp_chol(mesh11, dtype)


@needs_ffi
def test_cusolvermp_batched_cholesky_small_nrhs(mesh11):
    # NRHS < N: the historical cuSOLVERMp 0.6.0 silent-wrong regime —
    # pinned correct on the current stack.
    check_cusolvermp_chol(mesh11, "complex128", nq=2, n=32, mrhs=8)


@needs_ffi
@pytest.mark.parametrize("dtype", ["complex128", "float64"])
def test_cusolvermp_solve_lu_hermitian_indefinite(mesh11, dtype):
    check_cusolvermp_lu(mesh11, dtype)


@needs_ffi
def test_cusolvermp_solve_lu_general(mesh11):
    check_cusolvermp_lu(mesh11, "complex128", herm=False)


@needs_ffi
@pytest.mark.parametrize("dtype", ["complex128", "float64"])
def test_cusolvermp_eigh(mesh11, dtype):
    check_cusolvermp_eigh(mesh11, dtype)


@needs_ffi
@pytest.mark.parametrize("transa,transb", [("N", "N"), ("C", "N")])
def test_cublasmp_gemm(mesh11, transa, transb):
    check_cublasmp_gemm(mesh11, "complex128", transa, transb)


@needs_ffi
def test_cublasmp_fused_w_solve(mesh11):
    check_cublasmp_wsolve(mesh11, "complex128")


@needs_ffi
@pytest.mark.parametrize("dtype", ["complex128", "float64"])
def test_slate_cholesky_trsm(mesh11, dtype):
    check_slate_chol_trsm(mesh11, dtype)


@needs_ffi
@pytest.mark.parametrize("m", [16, 48])
def test_slate_trsm_rectangular_rhs(mesh11, m):
    check_slate_chol_trsm(mesh11, "complex128", n=32, m=m)


@needs_ffi
def test_slate_batched_cholesky_trsm(mesh11):
    check_slate_batched(mesh11, "complex128")


@needs_ffi
@pytest.mark.parametrize("dtype", ["complex128", "float64"])
def test_slate_eigh_true_eigenvectors(mesh11, dtype):
    check_slate_eigh(mesh11, dtype)


@needs_ffi
def test_padding_solve_at_logical_through_ffi(mesh11):
    check_padding_solve_at_logical(mesh11, "complex128")


def test_slate_tile_layout_validation():
    # Pure python — runs everywhere, even without the .so.
    pytest.importorskip("jax")
    check_tile_layout_validation()


# ---------------------------------------------------------------------------
# Host platform (JAX CPU backend) — the same slate check bodies on a 1x1
# mesh of CPU devices, exercising liblorrax_ffi_host.so's handlers
# (fromScaLAPACK + Target::HostTask) through the unchanged Python wrappers.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mesh_cpu11():
    return _mesh_cpu_1x1()


@needs_host_ffi
@pytest.mark.parametrize("dtype", ["complex128", "float64"])
def test_slate_cholesky_trsm_cpu(mesh_cpu11, dtype):
    check_slate_chol_trsm(mesh_cpu11, dtype)


@needs_host_ffi
@pytest.mark.parametrize("m", [16, 48])
def test_slate_trsm_rectangular_rhs_cpu(mesh_cpu11, m):
    check_slate_chol_trsm(mesh_cpu11, "complex128", n=32, m=m)


@needs_host_ffi
def test_slate_batched_cholesky_trsm_cpu(mesh_cpu11):
    check_slate_batched(mesh_cpu11, "complex128")


@needs_host_ffi
@pytest.mark.parametrize("dtype", ["complex128", "float64"])
def test_slate_eigh_true_eigenvectors_cpu(mesh_cpu11, dtype):
    # BUG L-2: SLATE's host `heev` SIGSEGVs deterministically (1x1 mesh,
    # n=64, single rank — not MPI, not the layout contract).  A SIGSEGV is
    # not a test failure: it takes the pytest process down mid-run, so
    # every cell after this one in the file was silently never executed.
    # Skip until upstream is fixed; the CPU distributed eigh contract is
    # covered by ``test_scalapack_eigh_true_eigenvectors_cpu`` below, and
    # the resolver refuses this combination
    # (``test_compute_wfns_fi_slate_refused_on_cpu``).
    pytest.skip("slate host heev SIGSEGVs — bug L-2, see docs/dev/linalg_ffi.md")


def check_scalapack_eigh(mesh, dtype, n=32):
    """ScaLAPACK ``pzheevd``/``pdsyevd`` through ffi.scalapack — the same
    STRICT contract ``check_slate_eigh`` states (and SLATE's host handler
    cannot meet): W matches numpy, Z's COLUMNS are true eigenvectors
    (``A @ Z == Z @ diag(W)`` — a wrong layout/transpose passes the
    eigenvalue check and fails this), Z unitary, rerun bit-deterministic
    on a fixed grid."""
    from ffi.scalapack import distributed_eigh
    rng = np.random.default_rng(37)
    A_np = _herm(rng, 1, n, dtype)[0]

    def solve():
        A = _put(A_np, mesh, ("x", "y"))
        W, Z = distributed_eigh(A, mesh=mesh)
        return _gather(W)[:n], _gather(Z)

    W1, Z1 = solve()
    W2, Z2 = solve()
    assert np.array_equal(W1, W2) and np.array_equal(Z1, Z2), \
        "scalapack eigh rerun not bit-deterministic"
    ev_err = float(np.max(np.abs(W1 - np.linalg.eigvalsh(A_np))))
    assert ev_err < 1e-10, f"eigenvalue error {ev_err:.3e}"
    nrmA = max(np.linalg.norm(A_np), 1.0)
    res = np.linalg.norm(A_np @ Z1 - Z1 * W1[None, :]) / nrmA
    orth = np.linalg.norm(np.conj(Z1.T) @ Z1 - np.eye(n))
    assert res < 1e-11 and orth < 1e-11, \
        f"eigvec residual {res:.3e}, orthonormality {orth:.3e}"


@needs_host_ffi
@pytest.mark.parametrize("dtype", ["complex128", "float64"])
def test_scalapack_eigh_true_eigenvectors_cpu(mesh_cpu11, dtype):
    check_scalapack_eigh(mesh_cpu11, dtype)


@needs_host_ffi
@pytest.mark.parametrize("dtype", ["complex128", "float64"])
def test_scalapack_solve_lu_hermitian_indefinite_cpu(mesh_cpu11, dtype):
    check_scalapack_lu(mesh_cpu11, dtype)


@needs_host_ffi
def test_scalapack_solve_lu_general_cpu(mesh_cpu11):
    check_scalapack_lu(mesh_cpu11, "complex128", herm=False)


@needs_host_ffi
@pytest.mark.parametrize("dtype", ["complex128", "float64"])
def test_scalapack_getrf_getrs_split_is_bit_identical_cpu(mesh_cpu11, dtype):
    check_scalapack_getrf_getrs(mesh_cpu11, dtype)


@needs_host_ffi
def test_scalapack_resolver_and_host_only_guard(mesh_cpu11):
    """distributed_lu=scalapack resolves to scalapack_lu (explicit only —
    auto never picks it); a GPU-device mesh is rejected loudly."""
    from isdf.core import _resolve_solver_kind_transverse
    assert _resolve_solver_kind_transverse(
        mesh_cpu11, "scalapack") == "scalapack_lu"
    assert _resolve_solver_kind_transverse(
        mesh_cpu11, "auto") != "scalapack_lu"
    import jax
    from ffi.scalapack import batched_distributed_solve_lu
    gpus = [d for d in jax.devices() if d.platform in ("gpu", "cuda")]
    if gpus:
        from jax.sharding import Mesh
        gpu_mesh = Mesh(np.asarray(gpus[:1]).reshape(1, 1), ("x", "y"))
        A = jax.numpy.zeros((1, 8, 8), dtype="complex128")
        B = jax.numpy.zeros((1, 8, 4), dtype="complex128")
        with pytest.raises(ValueError, match="host-only"):
            batched_distributed_solve_lu(A, B, mesh=gpu_mesh)


# ---------------------------------------------------------------------------
# CLI mode — the multi-rank matrix.  Same checks on a PxQ process mesh.
# ---------------------------------------------------------------------------

_CLI_CELLS = [
    # (name, needs_square_mesh, fn(mesh, dtype)).
    # needs_square=True for: syevd (mb==nb or DEADLOCK), potrf/fused
    # w_solve (mb==nb or INVALID_VALUE), gemm (1-D grids rejected by
    # cuBLASMp), slate heev (SLATE algorithm).  getrf/getrs (lu) and
    # slate potrf/trsm run on 1-D meshes too.
    ("cusolvermp_chol", True,
     lambda mesh, dt: check_cusolvermp_chol(mesh, dt, n=64, mrhs=96)),
    ("cusolvermp_chol_small_nrhs", True,
     lambda mesh, dt: check_cusolvermp_chol(mesh, dt, n=64, mrhs=16)),
    ("cusolvermp_lu", False,
     lambda mesh, dt: check_cusolvermp_lu(mesh, dt, n=64, nrhs=32)),
    ("cusolvermp_eigh", True,
     lambda mesh, dt: check_cusolvermp_eigh(mesh, dt, n=64)),
    ("cublasmp_gemm", True,
     lambda mesh, dt: check_cublasmp_gemm(mesh, dt)),
    ("cublasmp_wsolve", True,
     lambda mesh, dt: check_cublasmp_wsolve(mesh, dt, n=64)),
    ("slate_chol_trsm", False,
     lambda mesh, dt: check_slate_chol_trsm(mesh, dt, n=64, m=64)),
    ("slate_trsm_rect_small", False,
     lambda mesh, dt: check_slate_chol_trsm(mesh, dt, n=64, m=32)),
    ("slate_trsm_rect_large", False,
     lambda mesh, dt: check_slate_chol_trsm(mesh, dt, n=64, m=128)),
    ("slate_batched", False,
     lambda mesh, dt: check_slate_batched(mesh, dt, nbatch=4, n=32,
                                          nrhs=16)),
    ("slate_eigh", True,
     lambda mesh, dt: check_slate_eigh(mesh, dt, n=64)),
    # Host-only (skipped on CUDA backends by the platform gate below).
    # No needs_square: scalapack's square-block requirement is satisfied
    # on square AND 1-D meshes (g = n/max(p,q)); p!=q with both >1 GUARDs.
    ("scalapack_lu", False,
     lambda mesh, dt: check_scalapack_lu(mesh, dt, n=64, nrhs=32)),
    ("scalapack_lu_general", False,
     lambda mesh, dt: check_scalapack_lu(mesh, dt, n=64, nrhs=32,
                                         herm=False)),
    ("scalapack_getrf_getrs", False,
     lambda mesh, dt: check_scalapack_getrf_getrs(mesh, dt, n=64, nrhs=32)),
    # Production wiring: the htransform fH_q eigh routed through the FFI.
    # rank=64 divides 1/2/4, so the same cell runs on every mesh the
    # wrappers accept.  dtype is ignored (fH_q is complex by construction).
    ("bse_setup_eigh_cusolvermp", True,
     lambda mesh, dt: check_compute_wfns_fi_backend(mesh, "cusolvermp")),
    ("bse_setup_eigh_slate", True,
     lambda mesh, dt: check_compute_wfns_fi_backend(mesh, "slate")),
]


def _cli_main():
    # Distributed init already happened at import time (top of module).
    import jax

    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="PxQ process mesh")
    ap.add_argument("--only", default="", help="substring filter")
    args = ap.parse_args()
    mesh = _mesh_from_arg(args.mesh)
    px = int(mesh.shape["x"])
    py = int(mesh.shape["y"])
    is_cpu = jax.default_backend() == "cpu"
    if jax.process_index() == 0:
        print(f"backend={jax.default_backend()} mesh={args.mesh}", flush=True)

    failures = 0
    for name, needs_square, fn in _CLI_CELLS:
        if args.only and args.only not in name:
            continue
        if is_cpu and not name.startswith(("slate", "scalapack")):
            if jax.process_index() == 0:
                print(f"SKIP {name}[{args.mesh}] (CUDA-only backend)",
                      flush=True)
            continue
        if not is_cpu and name.startswith("scalapack"):
            if jax.process_index() == 0:
                print(f"SKIP {name}[{args.mesh}] (host-only backend)",
                      flush=True)
            continue
        for dt in ("complex128", "float64"):
            tag = f"{name}[{args.mesh},{dt}]"
            if needs_square and px != py:
                if jax.process_index() == 0:
                    print(f"SKIP {tag} (square mesh required)", flush=True)
                continue
            try:
                fn(mesh, dt)
                if jax.process_index() == 0:
                    print(f"PASS {tag}", flush=True)
            except AssertionError as exc:
                failures += 1
                if jax.process_index() == 0:
                    print(f"FAIL {tag}: {exc}", flush=True)
            except ValueError as exc:
                # Wrapper-level validation (unsupported mesh class for
                # this op) — the rejection IS the contract there.
                if jax.process_index() == 0:
                    print(f"GUARD {tag}: {exc}", flush=True)
            except Exception as exc:
                failures += 1
                if jax.process_index() == 0:
                    print(f"ERROR {tag}: {type(exc).__name__}: {exc}",
                          flush=True)
    if jax.process_index() == 0:
        print(f"done: {failures} failures", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_cli_main())


# ---------------------------------------------------------------------------
# distributed_cholesky = slate — the input-file-selectable backend
# (factor_c_q wiring; see isdf.core._resolve_solver_kind_charge)
# ---------------------------------------------------------------------------

def check_factor_c_q_slate(mesh):
    """factor_c_q(solver_kind='slate_cholesky') returns a conventional L
    equal to the numpy Cholesky (1×1 mesh), so downstream solve_zeta's
    triangular-solve branch consumes it unchanged."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from isdf.core import factor_c_q, _resolve_solver_kind_charge

    assert _resolve_solver_kind_charge(mesh, "slate") == "slate_cholesky"

    rng = np.random.default_rng(7)
    n, nq = 32, 3
    A = rng.standard_normal((nq, n, n)) + 1j * rng.standard_normal((nq, n, n))
    C = A @ np.conj(np.swapaxes(A, -1, -2)) + n * np.eye(n)[None]
    C_dev = jax.device_put(jnp.asarray(C), NamedSharding(mesh, P(None, "x", "y")))

    L = np.asarray(factor_c_q(C_dev, mesh, solver_kind="slate_cholesky"))
    L_ref = np.linalg.cholesky(C)
    resid = np.max(np.abs(L - L_ref)) / np.max(np.abs(L_ref))
    assert resid < 1e-12, f"slate factor_c_q vs numpy Cholesky: rel {resid:.2e}"

    # Determinism: second call bit-identical.
    L2 = np.asarray(factor_c_q(C_dev, mesh, solver_kind="slate_cholesky"))
    assert np.array_equal(L, L2)


@needs_ffi
def test_factor_c_q_slate_matches_reference():
    check_factor_c_q_slate(_mesh_1x1())


@needs_host_ffi
def test_factor_c_q_slate_matches_reference_cpu():
    # Same production wiring on a CPU-device mesh — the input-file
    # ``distributed_cholesky = slate`` path a CPU-backend run takes.
    check_factor_c_q_slate(_mesh_cpu_1x1())


@needs_ffi
def test_resolver_never_auto_picks_slate():
    from isdf.core import _resolve_solver_kind_charge
    mesh = _mesh_1x1()
    assert _resolve_solver_kind_charge(mesh, "auto") != "slate_cholesky"
    assert _resolve_solver_kind_charge(mesh, "off") == "sharded_cholesky"


# ---------------------------------------------------------------------------
# ffi.linalg.plan — the call-site interface.  These cells PIN the promise
# that a plan resolves EXACTLY like resolve_backend: the plan exists to
# move resharding and batching off the call sites, not to change routes.
# ---------------------------------------------------------------------------

def test_plan_resolution_is_identical_to_resolve_backend():
    """Every (op, requested) resolves the same through both spellings —
    including the ones that RAISE, with the same exception type."""
    from ffi import linalg
    mesh = _mesh_cpu_1x1()
    for op in linalg.OPS:
        for requested in linalg.BACKEND_CHOICES[op]:
            try:
                expected = linalg.resolve_backend(op, requested, mesh)
            except (ValueError, RuntimeError) as exc:
                with pytest.raises(type(exc)):
                    linalg.plan(op, mesh, backend=requested)
                continue
            p = linalg.plan(op, mesh, backend=requested)
            assert p.backend == expected, (op, requested)
            assert p.is_native == (expected == linalg.NATIVE)
            assert p.op == op and p.requested == requested


def test_plan_native_contract():
    """A native plan advertises no sharding contract, runs eigh itself,
    and REFUSES the two ops whose native route lives in isdf/core."""
    import jax.numpy as jnp
    from ffi import linalg
    mesh = _mesh_cpu_1x1()

    p = linalg.plan("eigh", mesh, backend="auto")
    assert p.is_native and p.in_sharding is None
    rng = np.random.default_rng(0)
    A = _herm(rng, 1, 8, "complex128")[0]
    lam, R = p(jnp.asarray(A))
    assert np.max(np.abs(np.asarray(A) @ np.asarray(R)
                         - np.asarray(R) * np.asarray(lam)[None, :])) < 1e-10
    # batched() on a native plan is jnp.linalg.eigh's own batching.
    lam_b, R_b = p.batched(jnp.asarray(_herm(rng, 3, 8, "complex128")))
    assert lam_b.shape == (3, 8) and R_b.shape == (3, 8, 8)

    for op in ("cholesky", "solve_lu"):
        q = linalg.plan(op, mesh, backend="auto")
        assert q.is_native
        with pytest.raises(NotImplementedError, match="isdf/core"):
            q(jnp.asarray(A))


def test_plan_describe_and_module_are_honest():
    from ffi import linalg
    mesh = _mesh_cpu_1x1()
    p = linalg.plan("eigh", mesh, backend="off")
    assert "native" in p.describe() and "any layout" in p.describe()
    with pytest.raises(ValueError, match="no FFI backend module"):
        _ = p.module


# ---------------------------------------------------------------------------
# resolve guard 5 — the 1-D-mesh cusolvermp geometry contract (doctrine 3).
# Pure resolve-level cells: the mesh/capability facts are mocked, so these
# run on any dev box with no GPU and no .so.  They pin the CONTRACT
# (refusal for explicit requests, announced demote for 'auto') by
# substring, not exact text.  The old behavior — explicit cusolvermp
# silently returning NATIVE on a 1-D mesh — forced a compensating
# ``p.is_native`` re-raise in gw/w_isdf; both were removed together
# (audit fix/zq 2026-07-28).
# ---------------------------------------------------------------------------

class _FakeCudaMesh:
    """Shape/platform stand-in: resolve_backend reads only
    ``mesh.shape['x'/'y']``, ``mesh.devices.flat[0].platform`` and
    ``mesh.devices.size``."""

    def __init__(self, px, py):
        from types import SimpleNamespace
        self.shape = {"x": px, "y": py}
        self.devices = SimpleNamespace(
            flat=[SimpleNamespace(platform="gpu")], size=px * py)


def _mock_cuda_capabilities(monkeypatch, nproc):
    from ffi.linalg import resolve as R
    from ffi.common import ffi_loader as FL
    monkeypatch.setattr(FL, "probe_target", lambda t, p: (True, "ok"))
    monkeypatch.setattr(FL, "has_target", lambda t, p: True)
    monkeypatch.setattr(R, "_process_count", lambda: nproc)
    return R


@pytest.mark.parametrize("op", ["cholesky", "solve_lu"])
def test_resolve_explicit_cusolvermp_refuses_on_1d_mesh(monkeypatch, op):
    R = _mock_cuda_capabilities(monkeypatch, nproc=4)
    with pytest.raises(ValueError, match="true-2D"):
        R.resolve_backend(op, "cusolvermp", _FakeCudaMesh(1, 4))
    with pytest.raises(ValueError, match="1x4"):
        R.resolve_backend(op, "cusolvermp", _FakeCudaMesh(1, 4))


def test_resolve_distributed_spelling_refuses_and_names_itself(monkeypatch):
    # 'distributed' maps to cusolvermp on CUDA and must refuse the same
    # way — naming the user's original spelling, so the message points at
    # the key the deck actually set (w_dyson_solver=distributed).
    R = _mock_cuda_capabilities(monkeypatch, nproc=4)
    with pytest.raises(ValueError, match="distributed"):
        R.resolve_backend("solve_lu", "distributed", _FakeCudaMesh(4, 1))


def test_resolve_explicit_cusolvermp_2d_mesh_is_a_promise(monkeypatch):
    R = _mock_cuda_capabilities(monkeypatch, nproc=4)
    assert R.resolve_backend(
        "cholesky", "cusolvermp", _FakeCudaMesh(2, 2)) == "cusolvermp"
    assert R.resolve_backend(
        "solve_lu", "distributed", _FakeCudaMesh(2, 2)) == "cusolvermp"


def test_resolve_auto_1d_mesh_demotes_with_announcement(monkeypatch, capsys):
    R = _mock_cuda_capabilities(monkeypatch, nproc=4)
    R._AUTO_GEOMETRY_DEMOTE_ANNOUNCED.clear()
    mesh = _FakeCudaMesh(1, 4)
    assert R.resolve_backend("solve_lu", "auto", mesh) == R.NATIVE
    out = capsys.readouterr().out
    assert "auto" in out and "native" in out and "1x4" in out
    # Deduped: a second resolve of the same decision stays quiet.
    assert R.resolve_backend("solve_lu", "auto", mesh) == R.NATIVE
    assert "native" not in capsys.readouterr().out


@needs_host_ffi
def test_plan_batched_matches_the_backend_call_cpu():
    """``plan.batched`` on the CPU distributed eigh == the raw wrapper.

    Pins the migration of ``isdf/core._factor_c_q_distributed_rank_truncate``
    (which now says ``backend='distributed'`` and ``plan.batched``) against
    what it used to call directly.  ScaLAPACK returns TRUE column
    eigenvectors, so the plan's normaliser is the identity here — and that
    is exactly what must stay true.
    """
    import jax.numpy as jnp
    from ffi import linalg
    mesh = _mesh_cpu_1x1()
    rng = np.random.default_rng(11)
    A = jnp.asarray(_hpd(rng, 2, 16, "complex128"))
    assert linalg.plan("eigh", mesh, backend="distributed").backend == "scalapack"
    W_ref, Z_ref = linalg.backend_module("scalapack").batched_distributed_eigh(
        A, mesh=mesh)
    W, Z = linalg.plan("eigh", mesh, backend="distributed", n=16).batched(A)
    assert np.array_equal(np.asarray(W), np.asarray(W_ref))
    assert np.array_equal(np.asarray(Z), np.asarray(Z_ref))


# ---------------------------------------------------------------------------
# eigh_backend = cusolvermp | slate — the htransform fH_q wiring
# (bandstructure.bse_setup.compute_wfns_fi; ffi.common.dispatch.dispatch_eigh)
# ---------------------------------------------------------------------------

def _synthetic_htransform(nk_grid=(2, 2, 1), nb=4, rank=64, n_mu=6, ns=2,
                          seed=5):
    """A small but STRUCTURALLY REAL htransform input triple.

    ``ctilde`` must be band-orthonormal per k (that is what
    ``streaming_galerkin_solve`` produces, and what makes the eigenvalues of
    fH_q = Σ_n f(ε_n) c_n c_nᴴ equal f(ε_n) at the coarse k), otherwise the
    recovered energies are meaningless and a backend comparison would be
    comparing two kinds of garbage.  Built by QR.
    """
    import numpy as _np
    rng = _np.random.default_rng(seed)
    nk = nk_grid[0] * nk_grid[1] * nk_grid[2]
    ct = _np.empty((nk, nb, rank), dtype=_np.complex128)
    for k in range(nk):
        z = (rng.standard_normal((rank, nb))
             + 1j * rng.standard_normal((rank, nb)))
        q, _ = _np.linalg.qr(z)                     # (rank, nb), orthonormal
        ct[k] = _np.conj(q.T)
    # Dispersive, ascending, non-degenerate energies (Ry).
    enk = (_np.linspace(-0.6, 0.4, nb)[:, None]
           + 0.05 * _np.cos(2 * _np.pi * _np.arange(nk) / nk)[None, :])
    B = (rng.standard_normal((rank, ns, n_mu))
         + 1j * rng.standard_normal((rank, ns, n_mu)))
    return ct, enk, B, nk_grid


def check_compute_wfns_fi_backend(mesh, backend, dtype="complex128"):
    """``compute_wfns_fi(eigh_backend=<ffi>)`` == the native batched path.

    Gates the two things a backend swap can break: the EIGENVALUES (hence the
    Newton-inverted energies, which are what the on-grid gate measures) and
    the eigenVECTOR convention (cusolvermp returns a raw conj-transposed
    buffer, SLATE true columns — a wrong choice there silently returns a
    transposed ψ).  ψ is compared through the WINDOW DENSITY MATRIX
    Σ_n ψ_n ψ_nᴴ, which is invariant under both the per-band phase and any
    unitary mixing inside a degenerate group — the only comparison that is
    well posed for eigenvectors.  A transposed or mis-conjugated Q changes it.
    """
    import jax
    import jax.numpy as jnp
    from bandstructure.bse_setup import compute_wfns_fi

    ct, enk, B, kgrid = _synthetic_htransform()
    kw = dict(ctilde=jnp.asarray(ct), B_at_mu=jnp.asarray(B),
              enk_sigma=jnp.asarray(enk), kgrid_co=kgrid,
              band_window_fi=(1, 3), mesh_xy=mesh, kgrid_fi=(4, 4, 1),
              batch_size=8)
    with mesh:
        ref = compute_wfns_fi(eigh_backend="off", **kw)
        got = compute_wfns_fi(eigh_backend=backend, **kw)

    lam_r, lam_g = np.asarray(ref.lam_fi), np.asarray(got.lam_fi)
    e_r, e_g = np.asarray(ref.enk_full), np.asarray(got.enk_full)
    dlam = float(np.max(np.abs(lam_r - lam_g)))
    de = float(np.max(np.abs(e_r - e_g)))
    assert dlam < 1e-10, f"{backend}: fH_q eigenvalues differ by {dlam:.3e}"
    assert de < 1e-10, f"{backend}: recovered energies differ by {de:.3e} Ry"

    def _dm(psi):
        p = np.asarray(psi).reshape(psi.shape[0], psi.shape[1], -1)
        return np.einsum("qni,qnj->qij", p, np.conj(p))

    D_r, D_g = _dm(ref.psi_rmu_Y), _dm(got.psi_rmu_Y)
    dpsi = float(np.max(np.abs(D_r - D_g)) / max(np.max(np.abs(D_r)), 1e-300))
    assert dpsi < 1e-9, \
        f"{backend}: window density matrix differs from native by {dpsi:.3e}"


@needs_ffi
def test_compute_wfns_fi_cusolvermp_matches_native(mesh11):
    check_compute_wfns_fi_backend(mesh11, "cusolvermp")


@needs_ffi
def test_compute_wfns_fi_slate_matches_native(mesh11):
    check_compute_wfns_fi_backend(mesh11, "slate")


@needs_host_ffi
def test_compute_wfns_fi_slate_refused_on_cpu(mesh_cpu11):
    """Host platform: ``eigh_backend='slate'`` must be REFUSED, not run.

    Bug L-2 — SLATE's host ``heev`` SIGSEGVs deterministically, down to a
    1x1 mesh at n=64.  This cell used to call it and take the whole pytest
    process down with it (no traceback, no summary, every later test in the
    file unrun).  The handler IS compiled, so no capability probe catches
    it; ``resolve_backend`` therefore refuses the combination outright.
    """
    import jax.numpy as jnp
    from bandstructure.bse_setup import compute_wfns_fi

    ct, enk, B, kgrid = _synthetic_htransform()
    with mesh_cpu11, pytest.raises(RuntimeError, match="REJECTED on CPU"):
        compute_wfns_fi(ctilde=jnp.asarray(ct), B_at_mu=jnp.asarray(B),
                        enk_sigma=jnp.asarray(enk), kgrid_co=kgrid,
                        band_window_fi=(1, 3), mesh_xy=mesh_cpu11,
                        kgrid_fi=(4, 4, 1), batch_size=8,
                        eigh_backend="slate")


@needs_host_ffi
def test_compute_wfns_fi_scalapack_matches_native_cpu(mesh_cpu11):
    # Host platform: the same wiring a CPU-backend run takes, on the
    # backend that actually works there (ScaLAPACK pzheevd).  This is the
    # cell that runs on a machine with no GPU at all.
    check_compute_wfns_fi_backend(mesh_cpu11, "scalapack")


def test_compute_wfns_fi_rejects_bad_backend():
    """Pure python — an unknown backend name must fail loudly, not silently
    fall back to the replicated path."""
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from bandstructure.bse_setup import compute_wfns_fi
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    ct, enk, B, kgrid = _synthetic_htransform()
    # (The message names the OP, not the config key: resolve.py is
    # op-generic and must not hard-code per-consumer key names.)
    with pytest.raises(ValueError, match="backend must be one of"):
        compute_wfns_fi(ctilde=jnp.asarray(ct), B_at_mu=jnp.asarray(B),
                        enk_sigma=jnp.asarray(enk), kgrid_co=kgrid,
                        band_window_fi=(1, 3), mesh_xy=mesh,
                        kgrid_fi=(2, 2, 1), eigh_backend="replicated")
