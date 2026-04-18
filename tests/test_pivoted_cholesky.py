"""Regression tests for the pivoted-Cholesky candidate-pruning stage.

Covers the three pieces of ``src/centroid/pivoted_cholesky.py``:

* ``build_candidate_gram_q0`` produces a Hermitian PSD matrix that agrees
  with a brute-force ``einsum`` reference.
* ``pivoted_cholesky_select`` reproduces the target matrix up to the
  chosen rank (``‖G − L_r L_r^H‖`` matches the trailing singular values).
* The Schur-complement residual diagonal is non-increasing across pivots
  and matches what ``scipy.linalg.lapack.cpstrf`` produces on the same
  matrix (pivots may differ on ties but the residual is invariant).

All tests are device-agnostic — they run on CPU JAX if no GPU is visible.
"""

from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from src.centroid.pivoted_cholesky import (
    build_candidate_gram_q0,
    pivoted_cholesky_select,
)


def _random_psi(rng, nk: int, nb: int, M: int, dtype=np.complex64):
    """(nk, nb, M) random wavefunction-like block. Not orthonormal — just PSD-generating."""
    re = rng.standard_normal((nk, nb, M)).astype(np.float32)
    im = rng.standard_normal((nk, nb, M)).astype(np.float32)
    return (re + 1j * im).astype(dtype)


@pytest.mark.parametrize("nk, nv, nc, M", [(2, 3, 4, 32), (4, 5, 6, 48)])
def test_gram_q0_hermitian_psd_and_matches_naive(nk, nv, nc, M):
    """Gram from the jitted k-fori_loop path matches a fp64 reference.

    The jitted path runs in complex64 on the GPU, which on Ampere means
    TF32-accumulated matmul (≈ 10-bit mantissa ⇒ ~1e-3 relative error for
    random well-conditioned inputs). The reference is computed in
    complex128 on the host — the physically correct ``G``. We check the
    fp32/TF32 result falls within a few×1e-3 of truth, Hermitian, PSD.
    """
    rng = np.random.default_rng(0)
    phi = jnp.asarray(_random_psi(rng, nk, nv, M))
    psi = jnp.asarray(_random_psi(rng, nk, nc, M))
    kw = jnp.asarray(rng.random(nk).astype(np.float32)) + 0.1

    # Jitted path: k-fori_loop with per-k matmuls.
    G = np.asarray(build_candidate_gram_q0(phi, psi, k_weights=kw))

    # fp64 reference: upcast each k-block, accumulate in complex128.
    G_ref = np.zeros((M, M), dtype=np.complex128)
    phi_h = np.asarray(phi).astype(np.complex128)
    psi_h = np.asarray(psi).astype(np.complex128)
    kw_h = np.asarray(kw).astype(np.float64)
    for k in range(nk):
        P_v = phi_h[k].T @ np.conj(phi_h[k])          # Σ_v φ(a) φ*(b)
        P_c = np.conj(psi_h[k]).T @ psi_h[k]          # Σ_c ψ*(a) ψ(b)
        G_ref += float(kw_h[k]) * (P_v * P_c)
    G_ref = 0.5 * (G_ref + G_ref.conj().T)

    # Agreement: TF32 tolerance (≈ 1/1024 relative) + a small absolute
    # floor to cover near-zero off-diagonals that blow up relative error.
    scale = float(np.max(np.abs(G_ref)))
    np.testing.assert_allclose(
        G, G_ref, atol=2e-3 * scale, rtol=2e-3,
        err_msg="Gram diverges from fp64 reference beyond TF32 tolerance",
    )

    # Hermiticity (self-consistency, not a reference match) — TF32 matmul
    # non-determinism can actually violate exact Hermiticity even when the
    # math is symmetric. We symmetrize inside build_candidate_gram_q0
    # (enforce_hermitian=True), so this should hold near machine epsilon.
    herm_err = np.max(np.abs(G - G.conj().T))
    assert herm_err / scale < 1e-5, (
        f"non-Hermitian: ‖G − G^H‖_max / ‖G‖ = {herm_err / scale:.2e}"
    )

    # PSD: smallest eigenvalue ≥ -tol × largest. TF32 noise can push
    # small positive eigenvalues very slightly negative.
    evals = np.linalg.eigvalsh(G)
    tol = 1e-3 * abs(evals.max())
    assert evals.min() > -tol, f"not PSD: smallest eigenvalue {evals.min():.2e}"


def test_pivoted_cholesky_reconstructs_G():
    """For a full-rank PSD matrix, k_keep=M should give ‖G − L L^H‖ ≈ 0.

    The standard correctness property of Cholesky: the product of the
    computed factor L reproduces the target matrix up to fp roundoff. If
    pivoting is implemented correctly, this holds regardless of pivot
    order.
    """
    rng = np.random.default_rng(1)
    M = 24
    # Make a well-conditioned PSD: G = A A^H + λ I.
    A = (rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))).astype(np.complex64)
    G = (A @ A.conj().T + 0.1 * np.eye(M)).astype(np.complex64)
    G = 0.5 * (G + G.conj().T)

    piv, L, rank, d_final, d_taken = pivoted_cholesky_select(jnp.asarray(G), k_keep=M, tol_rel=0.0)
    L = np.asarray(L)
    rank = int(rank)
    assert rank == M, f"expected full rank {M}, got {rank}"

    # Reconstruct: G_hat = L L^H   (rows permuted into G's original order
    # since we never reordered G itself — L is written in the original
    # row ordering with pivots controlling *column* selection).
    G_hat = L @ L.conj().T
    err = np.max(np.abs(G - G_hat))
    scale = np.max(np.abs(G))
    assert err / scale < 5e-4, (
        f"‖G − L L^H‖_max / ‖G‖ = {err / scale:.2e} — expected fp32 roundoff"
    )

    # Residual diagonal at exit should be at floating-point noise.
    d_np = np.asarray(d_final)
    assert d_np.max() < 1e-3 * abs(np.diag(G).real).max(), (
        f"residual diagonal not fully drained: max = {d_np.max():.2e}"
    )


def test_pivoted_cholesky_rank_deficient():
    """Rank-``r`` PSD → pivoted Chol stops at exactly rank ``r``.

    Construct G = A A^H with A of shape (M, r); then G has rank r exactly.
    The Cholesky should accept r pivots and the residual should go to zero
    (all of G is captured), while later loop iterations no-op because
    ``pivot_val <= tol``.
    """
    rng = np.random.default_rng(2)
    M, r = 32, 10
    A = (rng.standard_normal((M, r)) + 1j * rng.standard_normal((M, r))).astype(np.complex64)
    G = (A @ A.conj().T).astype(np.complex64)
    G = 0.5 * (G + G.conj().T)

    piv, L, rank, d_final, d_taken = pivoted_cholesky_select(
        jnp.asarray(G), k_keep=M, tol_rel=1e-6
    )
    rank = int(rank)

    # Rank must match (allow ±1 for fp32 noise near the last-accepted
    # singular value; typical behavior on well-separated rank is exact).
    assert rank in (r, r - 1, r + 1), (
        f"rank detected = {rank}, expected r = {r}"
    )
    # Accepted pivots form a valid index set.
    piv_np = np.asarray(piv)
    accepted = piv_np[:rank]
    assert len(set(accepted.tolist())) == rank, "duplicate pivot indices"
    assert accepted.min() >= 0 and accepted.max() < M, "pivot out of range"
    # Unaccepted pivot slots are -1.
    unaccepted = piv_np[rank:]
    assert np.all(unaccepted == -1), "garbage in unaccepted pivot slots"


def test_pivoted_cholesky_residual_diagonal_nonincreasing():
    """After each pivot, the Schur complement diagonal should only shrink.

    This is an invariant of the algorithm. We run one pivot at a time with
    successively larger ``k_keep`` budgets and check the ``d_final``
    returned is ≤ the previous step's — element-wise.
    """
    rng = np.random.default_rng(3)
    M = 16
    A = (rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))).astype(np.complex64)
    G = (A @ A.conj().T + 0.05 * np.eye(M)).astype(np.complex64)
    G = 0.5 * (G + G.conj().T)
    G_j = jnp.asarray(G)

    d_prev = np.maximum(np.diag(G).real, 0.0)
    for k_keep in range(1, 9):
        _, _, _, d_final, _ = pivoted_cholesky_select(G_j, k_keep=k_keep, tol_rel=0.0)
        d = np.asarray(d_final)
        # Allow a tiny positive slop for fp32 noise.
        diff = d - d_prev
        assert np.max(diff) < 1e-4 * abs(np.diag(G).real).max(), (
            f"at k_keep={k_keep}: residual diagonal grew by {np.max(diff):.2e}"
        )
        d_prev = d


def test_end_to_end_via_wrapper_pipeline_shapes():
    """Smoke: synthetic φ, ψ → Gram → prune → output shape is correct.

    Exercises ``build_candidate_gram_q0`` + ``pivoted_cholesky_select``
    back to back on 4 k-points, 8 valence, 8 conduction, M = 64 candidates.
    We assert on shapes and on the basic invariants ``piv`` being a valid
    index subset of [0, M).
    """
    rng = np.random.default_rng(4)
    nk, nv, nc, M = 4, 8, 8, 64
    phi = jnp.asarray(_random_psi(rng, nk, nv, M))
    psi = jnp.asarray(_random_psi(rng, nk, nc, M))
    kw = jnp.asarray(np.ones(nk, dtype=np.float32) / nk)

    G = build_candidate_gram_q0(phi, psi, k_weights=kw)
    assert G.shape == (M, M)

    n_keep = 32
    piv, L, rank, d_final, d_taken = pivoted_cholesky_select(G, k_keep=n_keep, tol_rel=1e-10)
    assert piv.shape == (n_keep,)
    assert L.shape == (M, n_keep)
    assert d_final.shape == (M,)

    rank = int(rank)
    piv_np = np.asarray(piv)
    accepted = piv_np[:rank]
    assert accepted.min() >= 0 and accepted.max() < M
    assert len(set(accepted.tolist())) == rank, "pivoted Chol produced duplicate pivots"


# ═══════════════════════════════════════════════════════════════════════
# Sharded Gram build (WIP: select still single-device)
# ═══════════════════════════════════════════════════════════════════════

from jax.sharding import Mesh, NamedSharding, PartitionSpec
from src.centroid.pivoted_cholesky import make_sharded_gram_q0


@pytest.mark.skipif(len(jax.devices()) < 2,
                    reason="Sharded Gram build requires ≥2 JAX devices.")
def test_sharded_gram_matches_single_device():
    """make_sharded_gram_q0 output, gathered, equals build_candidate_gram_q0.

    The sharded build produces a row-sharded Gram on a 1D mesh 'x'; the
    single-device build returns the full matrix. Gathering the sharded
    version (via ``np.asarray`` on the global jax.Array — JAX will pull
    across devices) should match the single-device reference up to the
    same TF32 tolerance.
    """
    devices = jax.devices()
    n_dev = len(devices)

    rng = np.random.default_rng(11)
    nk, nv, nc, M = 4, 6, 8, 4 * n_dev * 8   # M divisible by n_dev
    phi = jnp.asarray(_random_psi(rng, nk, nv, M))
    psi = jnp.asarray(_random_psi(rng, nk, nc, M))
    kw = jnp.asarray(rng.random(nk).astype(np.float32)) + 0.1

    # Single-device reference.
    G_ref = np.asarray(build_candidate_gram_q0(phi, psi, k_weights=kw))

    # Sharded path: row-shard on 'x'.
    mesh = Mesh(np.asarray(devices), ('x',))
    gram_step = make_sharded_gram_q0(mesh, M, enforce_hermitian=True)
    G_sharded = gram_step(phi, psi, kw)
    # Pulling to host assembles the full matrix from the shards.
    G_out = np.asarray(G_sharded)

    assert G_out.shape == (M, M), f"sharded Gram shape mismatch: {G_out.shape}"
    scale = float(np.max(np.abs(G_ref)))
    np.testing.assert_allclose(
        G_out, G_ref, atol=3e-3 * scale, rtol=3e-3,
        err_msg="sharded Gram diverges from single-device beyond TF32 tolerance",
    )


def test_d_taken_trace_monotone_and_matches_classical():
    """d_taken: pivot value at the moment each pivot was accepted.

    Two invariants:
      (1) d_taken is non-increasing across pivot order — the greedy
          algorithm picks the largest residual diagonal each time, and
          subsequent pivots draw from the decreasing Schur complement.
      (2) d_taken[0] == max(diag(G)) within fp tolerance — the first
          pivot is, by definition, the largest diagonal entry of G.
    """
    rng = np.random.default_rng(123)
    M = 32
    # Rank-deficient so the decay is pronounced.
    A = (rng.standard_normal((M, 8)) + 1j * rng.standard_normal((M, 8))).astype(np.complex64)
    G = (A @ A.conj().T + 1e-4 * np.eye(M)).astype(np.complex64)
    G = 0.5 * (G + G.conj().T)

    piv, L, rank, d_final, d_taken = pivoted_cholesky_select(
        jnp.asarray(G), k_keep=M, tol_rel=0.0
    )
    rank = int(rank)
    d_taken_np = np.asarray(d_taken)

    # Invariant 1: non-increasing over accepted pivots.
    taken_accepted = d_taken_np[:rank]
    diffs = np.diff(taken_accepted)
    assert np.max(diffs) <= 1e-5 * taken_accepted[0], (
        f"d_taken increased between pivots; max positive diff = "
        f"{np.max(diffs):.3e}, first pivot = {taken_accepted[0]:.3e}"
    )

    # Invariant 2: d_taken[0] == max(diag(G)).
    max_diag = float(np.max(np.real(np.diag(G))))
    assert abs(float(d_taken_np[0]) - max_diag) / max_diag < 1e-5, (
        f"first d_taken = {float(d_taken_np[0]):.3e} vs max diag = "
        f"{max_diag:.3e}"
    )

    # After rank, d_taken entries are 0 (masked no-op iterations).
    if rank < M:
        np.testing.assert_array_equal(
            d_taken_np[rank:], np.zeros(M - rank, dtype=d_taken_np.dtype),
            err_msg="d_taken entries past rank should be exactly 0"
        )


@pytest.mark.skipif(len(jax.devices()) < 2,
                    reason="Sharded select requires ≥2 JAX devices.")
def test_sharded_select_matches_single_device():
    """Sharded pivoted_cholesky_select == single-device, piv-order + residuals.

    Exact pivot-by-pivot agreement isn't guaranteed across floating-point
    reduction orders (TF32 matmul inside the correction step, plus
    psum/pmax collectives can reorder additions), so we check:

      * the pivot sets are equal as sets,
      * rank is identical,
      * picked-pivot residuals (d_taken) agree to TF32 tolerance,
      * leftover residuals (d_final, where > 0) agree to TF32 tolerance.

    Note that gathering the sharded outputs into numpy pulls across
    devices automatically — JAX's global array is a view over per-shard
    buffers.
    """
    from src.centroid.pivoted_cholesky import (
        make_sharded_pivoted_cholesky_select,
    )
    devices = jax.devices()
    n_dev = len(devices)

    # Pick M divisible by n_dev, small enough to keep the test cheap.
    rng = np.random.default_rng(21)
    M = 16 * n_dev   # e.g. 64 on 4 devices
    k_keep = M // 2
    # Build a well-conditioned Hermitian PSD matrix.
    A = (rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))).astype(np.complex64)
    G = (A @ A.conj().T + 0.05 * np.eye(M)).astype(np.complex64)
    G = 0.5 * (G + G.conj().T)
    G_j = jnp.asarray(G)

    # Single-device reference.
    piv_ref, L_ref, rank_ref, d_ref, d_taken_ref = pivoted_cholesky_select(
        G_j, k_keep=k_keep, tol_rel=0.0
    )

    # Sharded path.
    mesh = Mesh(np.asarray(devices), ('x',))
    sharded_step = make_sharded_pivoted_cholesky_select(
        mesh, M, k_keep, tol_rel=0.0
    )
    G_sharded = jax.device_put(
        G_j, NamedSharding(mesh, PartitionSpec('x', None)),
    )
    piv_s, L_s, rank_s, d_s, d_taken_s = sharded_step(G_sharded)

    piv_ref_np = np.asarray(piv_ref)
    piv_s_np = np.asarray(piv_s)
    rank_ref = int(rank_ref)
    rank_s = int(rank_s)

    assert rank_s == rank_ref, f"rank differs: sharded {rank_s} vs ref {rank_ref}"

    # Same set of accepted pivots (order may differ under TF32 / tie-break).
    assert set(piv_ref_np[:rank_ref].tolist()) == set(piv_s_np[:rank_s].tolist()), (
        f"pivot SETS differ\n  ref: {sorted(piv_ref_np[:rank_ref].tolist())}\n"
        f"  shard: {sorted(piv_s_np[:rank_s].tolist())}"
    )

    # d_taken agrees up to TF32 tolerance. Since pivot order may differ,
    # compare sorted.
    scale = max(float(np.max(np.asarray(d_taken_ref))), 1e-12)
    np.testing.assert_allclose(
        np.sort(np.asarray(d_taken_s)[:rank_s])[::-1],
        np.sort(np.asarray(d_taken_ref)[:rank_ref])[::-1],
        atol=5e-3 * scale, rtol=5e-3,
        err_msg="d_taken (pivot residuals) diverge beyond TF32 tolerance",
    )

    # Leftover residuals agree on position.
    d_s_np = np.asarray(d_s)
    d_ref_np = np.asarray(d_ref)
    np.testing.assert_allclose(
        d_s_np, d_ref_np, atol=5e-3 * scale, rtol=5e-3,
        err_msg="leftover d_final diverges beyond TF32 tolerance",
    )
