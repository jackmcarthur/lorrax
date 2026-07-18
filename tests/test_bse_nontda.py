"""Non-TDA (full BSE) structure-preserving eigensolver + solver-P1 unit gates.

Small synthetic operators (host numpy / CPU jax) — no restart, no GPU — so they
run in the plain suite.  The dense-vs-solver validation on the real spinor
fixture lives in ``test_bse_dense_reference.py`` (``test_nontda_*``).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402
jax.config.update("jax_enable_x64", True)


def _pos(H, k):
    ev = np.linalg.eigvals(H).real
    return np.sort(ev[ev > 1e-9])[:k]


# ---------------------------------------------------------------------------
# Structure-preserving reductions.
# ---------------------------------------------------------------------------
def test_nontda_product_real_bse():
    """Design product form: REAL BSE (A,B real symmetric, A±B PD) — the solver's
    ``omega^2 = eig((A-B)(A+B))`` reduction reproduces the dense [[A,B],[-B,-A]]
    positive spectrum with paired X²-Y²=1 eigenvectors."""
    from bse.bse_nontda import solve_nontda_product
    rng = np.random.default_rng(3); n = 30
    A = rng.standard_normal((n, n)); A = A @ A.T / n + 3.0 * np.eye(n)
    Gb = rng.standard_normal((n, n)) * 0.15; B = 0.5 * (Gb + Gb.T)
    Href = np.block([[A, B], [-B, -A]])
    ref = _pos(Href, 5)
    omega, Z = solve_nontda_product(A, B, 5)
    assert np.allclose(np.sort(omega), ref, atol=1e-8), f"{np.sort(omega)} vs {ref}"
    for j in range(5):
        X, Y = Z[:n, j], Z[n:, j]
        assert abs(float(np.real(X.conj() @ X - Y.conj() @ Y)) - 1.0) < 1e-8
        res = np.linalg.norm(Href @ Z[:, j] - omega[j] * Z[:, j]) / np.linalg.norm(Z[:, j])
        assert res < 1e-8


def test_nontda_definite_pencil_complex():
    """Physical case: complex BSE (A Hermitian, B complex-SYMMETRIC, K PD) — the
    definite-pencil solver reproduces the dense SHAO [[A,B],[-B*,-A*]] spectrum
    with X^H X - Y^H Y = +1."""
    from bse.bse_nontda import solve_nontda_definite_pencil
    rng = np.random.default_rng(5); n = 24
    Ga = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    A = Ga @ Ga.conj().T / n + 4.0 * np.eye(n)                 # Hermitian PD-ish
    Gb = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * 0.1
    B = 0.5 * (Gb + Gb.T)                                      # complex-symmetric
    assert np.linalg.norm(B - B.T) < 1e-12 and np.linalg.norm(B - B.conj().T) > 1e-3
    Href = np.block([[A, B], [-B.conj(), -A.conj()]])
    ref = _pos(Href, 5)
    omega, Z = solve_nontda_definite_pencil(A, B, 5)
    assert np.allclose(np.sort(omega), ref, atol=1e-8), f"{np.sort(omega)} vs {ref}"
    for j in range(5):
        X, Y = Z[:n, j], Z[n:, j]
        assert abs(float(np.real(X.conj() @ X - Y.conj() @ Y)) - 1.0) < 1e-8
        res = np.linalg.norm(Href @ Z[:, j] - omega[j] * Z[:, j]) / np.linalg.norm(Z[:, j])
        assert res < 1e-8


def test_nontda_positive_definiteness_check():
    """(A-B)/K not positive definite (triplet/charge instability) must RAISE
    with a clear message, not silently return imaginary energies."""
    from bse.bse_nontda import solve_nontda_definite_pencil, solve_nontda_product
    n = 10
    A = np.diag(np.linspace(0.1, 1.0, n)).astype(np.complex128)
    B = 2.0 * np.eye(n)                                        # A-B < 0 => K indefinite
    with pytest.raises(ValueError, match="positive definite"):
        solve_nontda_definite_pencil(A, B, 3)
    with pytest.raises(ValueError, match="positive definite"):
        solve_nontda_product(A, B, 3)


# ---------------------------------------------------------------------------
# Solver P1 — block-Lanczos final-slot overwrite regression.
# ---------------------------------------------------------------------------
def test_block_lanczos_eigenvector_residual_p1():
    """P1: with the +1 Krylov slot the last block is retained (not clobbered by
    the final Q_next), so at Krylov = N block Lanczos reconstructs eigenVECTORS
    with tiny residual.  The pre-fix final-slot overwrite corrupted this block."""
    from solvers.lanczos import block_lanczos_eig_jit
    rng = np.random.default_rng(0); n = 40
    G = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = 0.5 * (G + G.conj().T)
    Hj = jnp.asarray(H)

    def matvec(V):                    # (bs, n) -> (bs, n)
        return (Hj @ V.T).T

    bs = 4; m = n // bs               # Krylov = bs*m = n (full space)
    ev, evec = block_lanczos_eig_jit(matvec, n, n_eig=4, block_size=bs,
                                     max_iter=m, n_reorth=m)
    ev = np.asarray(jax.device_get(ev)); evec = np.asarray(jax.device_get(evec))
    ref = np.sort(np.linalg.eigvalsh(H))[:4]
    assert np.allclose(np.sort(ev.real), ref, atol=1e-8), f"{np.sort(ev.real)} vs {ref}"
    for i in range(4):
        v = evec[i]; lam = ev[i]
        res = np.linalg.norm(H @ v - lam * v) / max(np.linalg.norm(v), 1e-30)
        assert res < 1e-6, f"state {i} eigenvector residual {res:.2e} (final-slot P1)"


def test_lanczos_jit_final_slot_shapes_p1():
    """P1: the single-vector and converged block variants also carry the +1 slot
    and return correctly-shaped, correct eigenpairs on a synthetic Hermitian op."""
    from solvers.lanczos import lanczos_eig_jit, block_lanczos_eig_jit_converged
    rng = np.random.default_rng(1); n = 30
    G = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    H = 0.5 * (G + G.conj().T); Hj = jnp.asarray(H)
    ref = np.sort(np.linalg.eigvalsh(H))[:3]

    ev, evec = lanczos_eig_jit(lambda v: Hj @ v, n, n_eig=3, max_iter=n, n_reorth=n)
    ev = np.asarray(jax.device_get(ev))
    assert evec.shape == (3, n)
    assert np.allclose(np.sort(ev.real), ref, atol=1e-8)

    def mvb(V):
        return (Hj @ V.T).T
    evb, evecb, nit = block_lanczos_eig_jit_converged(
        mvb, n, n_eig=3, block_size=3, max_iter=n // 3, rtol=1e-10, n_reorth=n)
    evb = np.asarray(jax.device_get(evb))
    assert np.allclose(np.sort(evb.real), ref, atol=1e-6)
