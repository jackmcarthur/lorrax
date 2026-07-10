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
src/ffi/slate/cpp/host_ffi.cc).  The ``*_cpu`` tests run the SAME check
bodies on a 1x1 mesh of CPU devices in this very process (works on GPU
nodes too: ``jax.ffi.ffi_call`` picks the handler by lowering platform)
and skip cleanly when the host library is absent.  Multi-rank CPU
meshes: run the CLI mode under ``JAX_PLATFORMS=cpu`` (non-slate cells
are skipped there; ``--only slate`` narrows the log to the same set).

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
        _plat = "CUDA" if "cuda" in os.environ["JAX_PLATFORMS"] else "cpu"
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


def check_cusolvermg_eigh(n=32, tile=16):
    """f64-only, single-process multi-GPU.  Contract: eigenvalues match;
    Q^T = eigenvectors (documented transpose convention)."""
    import jax
    from ffi.cusolvermg import eigh_mg
    rng = np.random.default_rng(17)
    A_np = _herm(rng, 1, n, "float64")[0]

    def solve():
        W, Q = eigh_mg(jax.device_put(A_np), tile_size=tile)
        return np.asarray(W), np.asarray(Q)

    W1, Q1 = solve()
    W2, Q2 = solve()
    assert np.array_equal(W1, W2) and np.array_equal(Q1, Q2), \
        "eigh_mg rerun not bit-deterministic"
    ev_err = float(np.max(np.abs(W1 - np.linalg.eigvalsh(A_np))))
    assert ev_err < 1e-10, f"eigenvalue error {ev_err:.3e}"
    Qt = Q1.T
    res = (np.linalg.norm(A_np @ Qt - Qt * W1[None, :])
           / max(np.linalg.norm(A_np), 1.0))
    assert res < 1e-11, f"Q^T eigvec residual {res:.3e}"


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
def test_cusolvermg_eigh():
    check_cusolvermg_eigh()


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
    check_slate_eigh(mesh_cpu11, dtype)


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

    failures = 0
    for name, needs_square, fn in _CLI_CELLS:
        if args.only and args.only not in name:
            continue
        if is_cpu and not name.startswith("slate"):
            if jax.process_index() == 0:
                print(f"SKIP {name}[{args.mesh}] (CUDA-only backend)",
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

@needs_ffi
def test_factor_c_q_slate_matches_reference():
    """factor_c_q(solver_kind='slate_cholesky') returns a conventional L
    equal to the numpy Cholesky (1×1 mesh), so downstream solve_zeta's
    triangular-solve branch consumes it unchanged."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from isdf.core import factor_c_q, _resolve_solver_kind_charge

    mesh = _mesh_1x1()
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
def test_resolver_never_auto_picks_slate():
    from isdf.core import _resolve_solver_kind_charge
    mesh = _mesh_1x1()
    assert _resolve_solver_kind_charge(mesh, "auto") != "slate_cholesky"
    assert _resolve_solver_kind_charge(mesh, "off") == "sharded_cholesky"
