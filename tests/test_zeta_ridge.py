"""Tikhonov ζ-ridge (cohsex.in ``zeta_ridge_eps``) — unit gates.

The opt-in ridge fit solves the SPD normal equations

    (C² + ε_q² I) ζ = C Z,     ε_q = ε_rel · λ̂_max(C_q)

i.e. applies the spectral filter f_ε(λ) = λ/(λ²+ε²) to the stock LSQ
solution — the "cleaned ζ" identity of the F-scheme study
(arbitrary_q_bse.md §12).  Gates (1×1 mesh, single device — no
multi-GPU requirement):

* OFF (eps = 0) is bit-identical to the historical factor+solve path.
* ON matches an eigh-based f_ε(C)Z reference built with the SAME λ̂
  (per-element spectral formula, not just norms).
* λ̂_max power-iteration estimate matches eigvalsh on a gapped spectrum.
* ε → 0 limit recovers the stock solution on a well-conditioned C.
* Padded-μ-extent contract: ζ pad rows exact zeros, logical block
  matches the unpadded solve.
* Transverse channels loud-fail (charge-only for now).
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update("jax_enable_x64", True)

from isdf.core import (
    factor_c_q,
    solve_zeta,
    _zeta_ridge_lambda_max,
)

MESH = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
            axis_names=('x', 'y'))


def _mk_psd(nq, n, seed=0, tail_decay=6.0):
    """Batch of Hermitian PSD matrices with a fast-decaying spectrum
    (ISDF-Gram-like): λ_i ∝ 10^{-tail_decay·i/n}, random unitary frame."""
    rng = np.random.default_rng(seed)
    C = np.empty((nq, n, n), dtype=np.complex128)
    for iq in range(nq):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Q, _ = np.linalg.qr(A)
        lam = 10.0 ** (-tail_decay * np.arange(n) / n) * (1.0 + 0.1 * iq)
        C[iq] = (Q * lam[None, :]) @ Q.conj().T
        C[iq] = 0.5 * (C[iq] + C[iq].conj().T)
    return C


def _mk_rhs(nq, n, nrhs, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((nq, n, nrhs))
            + 1j * rng.standard_normal((nq, n, nrhs))).astype(np.complex128)


def _ridge_reference(C, Z, eps_rel, lam_hat):
    """Per-element eigh reference: ζ = R f_ε(Λ) R^H Z, f_ε(λ)=λ/(λ²+ε²),
    ε = eps_rel·λ̂ (the implementation's λ̂, so the comparison isolates
    the solve path, not the λ estimate)."""
    nq = C.shape[0]
    out = np.empty_like(Z)
    for iq in range(nq):
        lam, R = np.linalg.eigh(C[iq])
        eps = eps_rel * lam_hat[iq]
        f = lam / (lam ** 2 + eps ** 2)
        out[iq] = R @ (f[:, None] * (R.conj().T @ Z[iq]))
    return out


def test_lambda_max_matches_eigvalsh():
    nq, n = 3, 32
    C = _mk_psd(nq, n, seed=3)
    lam_hat = np.asarray(_zeta_ridge_lambda_max(jnp.asarray(C), n))
    lam_ref = np.array([np.linalg.eigvalsh(C[iq])[-1] for iq in range(nq)])
    # Rayleigh quotient is a lower bound; gapped spectrum → tight.
    assert np.all(lam_hat <= lam_ref * (1 + 1e-12))
    np.testing.assert_allclose(lam_hat, lam_ref, rtol=1e-8)


def test_off_is_bit_identical():
    nq, n, nrhs = 2, 24, 5
    C = jnp.asarray(_mk_psd(nq, n))
    Z_np = _mk_rhs(nq, n, nrhs)
    L_stock = factor_c_q(C, MESH, n_rmu_logical=n)
    L_off = factor_c_q(C, MESH, n_rmu_logical=n, zeta_ridge_eps=0.0)
    assert np.array_equal(np.asarray(L_stock), np.asarray(L_off))
    # solve_zeta DONATES Z (buffer aliasing in _reshard_z) — fresh
    # device copies per call.
    z_stock = solve_zeta(L_stock, jnp.asarray(Z_np), MESH,
                         q_chunk_size=nq, n_rmu_logical=n)
    z_off = solve_zeta(L_off, jnp.asarray(Z_np), MESH, q_chunk_size=nq,
                       n_rmu_logical=n, ridge_c_q=None)
    assert np.array_equal(np.asarray(z_stock), np.asarray(z_off))


@pytest.mark.parametrize("eps_rel", [1e-4, 1e-5, 1e-6])
def test_ridge_matches_spectral_filter(eps_rel):
    nq, n, nrhs = 3, 32, 7
    C_np = _mk_psd(nq, n, seed=7)
    Z_np = _mk_rhs(nq, n, nrhs, seed=8)
    C = jnp.asarray(C_np)
    lam_hat = np.asarray(_zeta_ridge_lambda_max(C, n))
    L_B = factor_c_q(C, MESH, n_rmu_logical=n, zeta_ridge_eps=eps_rel)
    zeta = np.asarray(solve_zeta(
        L_B, jnp.asarray(Z_np), MESH, q_chunk_size=nq,
        n_rmu_logical=n, ridge_c_q=C))
    ref = _ridge_reference(C_np, Z_np, eps_rel, lam_hat)
    # The normal-equations solve squares the condition number:
    # cond(B) = (λmax²+ε²)/(λmin²+ε²).  Cholesky-vs-eigh roundoff is
    # amplified by cond(B) (e.g. tail_decay=6, ε_rel=1e-6 → cond(B)
    # ~1e12 → 1e-4-class rel drift in the ε-dominated modes).  Bound
    # the comparison by machine-eps × cond(B) with a ×100 safety.
    lam_min = np.array([np.linalg.eigvalsh(C_np[iq])[0]
                        for iq in range(nq)])
    eps_q = eps_rel * lam_hat
    condB = ((lam_hat ** 2 + eps_q ** 2)
             / (lam_min ** 2 + eps_q ** 2)).max()
    atol = 100 * 2.3e-16 * condB * np.abs(ref).max()
    np.testing.assert_allclose(zeta, ref, atol=atol, rtol=0)


def test_small_eps_recovers_stock():
    nq, n, nrhs = 2, 24, 4
    # Well-conditioned C (shallow spectrum) → f_ε → 1/λ as ε → 0.
    C_np = _mk_psd(nq, n, seed=11, tail_decay=2.0)
    Z_np = _mk_rhs(nq, n, nrhs, seed=12)
    C = jnp.asarray(C_np)
    L = factor_c_q(C, MESH, n_rmu_logical=n)
    z_stock = np.asarray(solve_zeta(L, jnp.asarray(Z_np), MESH,
                                    q_chunk_size=nq, n_rmu_logical=n))
    L_B = factor_c_q(C, MESH, n_rmu_logical=n, zeta_ridge_eps=1e-8)
    z_ridge = np.asarray(solve_zeta(
        L_B, jnp.asarray(Z_np), MESH, q_chunk_size=nq,
        n_rmu_logical=n, ridge_c_q=C))
    # rel error bound (ε·λmax/λmin)² = (1e-8·1e2)² = 1e-12
    rel = np.abs(z_ridge - z_stock).max() / np.abs(z_stock).max()
    assert rel < 1e-9


def test_padded_extent_pad_rows_zero():
    nq, n_log, n_pad, nrhs = 2, 29, 32, 5
    # Mild spectrum (cond(C)=1e3 → cond(B) ≤ 1e6): the padded and
    # unpadded solves differ by summation order alone, but that drift
    # is amplified by cond(B) — keep it small so the comparison is a
    # real contract check, not a conditioning measurement.
    C_log = _mk_psd(nq, n_log, seed=21, tail_decay=3.0)
    Z_log = _mk_rhs(nq, n_log, nrhs, seed=22)
    # Zero-pad rows/cols per the Phase 3a contract (bilinear in
    # zero-padded ψ).
    C_pad = np.zeros((nq, n_pad, n_pad), dtype=np.complex128)
    C_pad[:, :n_log, :n_log] = C_log
    Z_pad = np.zeros((nq, n_pad, nrhs), dtype=np.complex128)
    Z_pad[:, :n_log, :] = Z_log
    eps_rel = 1e-5
    L_pad = factor_c_q(jnp.asarray(C_pad), MESH, n_rmu_logical=n_log,
                       zeta_ridge_eps=eps_rel)
    z_pad = np.asarray(solve_zeta(
        L_pad, jnp.asarray(Z_pad), MESH, q_chunk_size=nq,
        n_rmu_logical=n_log, ridge_c_q=jnp.asarray(C_pad)))
    # ζ pad rows exact zeros (zero RHS pad rows through a block-diag
    # solve).
    assert np.array_equal(z_pad[:, n_log:, :],
                          np.zeros((nq, n_pad - n_log, nrhs)))
    # Logical block matches the unpadded solve (same logical system;
    # summation-order drift only).
    L_log = factor_c_q(jnp.asarray(C_log), MESH, n_rmu_logical=n_log,
                       zeta_ridge_eps=eps_rel)
    z_log = np.asarray(solve_zeta(
        L_log, jnp.asarray(Z_log), MESH, q_chunk_size=nq,
        n_rmu_logical=n_log, ridge_c_q=jnp.asarray(C_log)))
    np.testing.assert_allclose(z_pad[:, :n_log, :], z_log,
                               atol=1e-8 * np.abs(z_log).max(), rtol=0)


def test_transverse_loud_fails():
    nq, n = 2, 16
    C = jnp.asarray(_mk_psd(nq, n))
    with pytest.raises(NotImplementedError, match="transverse"):
        factor_c_q(C, MESH, vertex_mu_L=1, n_rmu_logical=n,
                   zeta_ridge_eps=1e-4)
    # solve-side guard: ridge_c_q with an LU solver kind is rejected.
    Z = jnp.asarray(_mk_rhs(nq, n, 3))
    with pytest.raises(NotImplementedError, match="charge/Cholesky"):
        solve_zeta(C, Z, MESH, q_chunk_size=nq, vertex_mu_L=1,
                   n_rmu_logical=n, ridge_c_q=C)
