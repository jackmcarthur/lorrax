"""CPU identity tests of the ISDF bsemat split (docs/theory/bgw-bsemat-export.md, E2–E7).

A dense plane-wave system: random ζ (``Z[G, μ]``), a positive diagonal v with
a dominant head slot, a negative-semidefinite μ-space χ, ``V = Zᴴ v Z``,
``W = (1 − Vχ)⁻¹V`` and the plane-wave ``W_PW = (1 − vZχZᴴ)⁻¹ v``. The
file-only split (E3–E6) and the kernel parts (E7) must equal the
plane-wave head/wing/body of E1 to rounding, with V full rank (N_G > N_μ)
and rank-deficient (N_G < N_μ, the stored-sphere case). Red twins: dropping
E5's head term, conjugating z, and V in place of W − V each fail.
Unit cells: exempt from the four-GPU rule (pure host algebra).
"""
import numpy as np
import pytest

from bse.bsemat_algebra import (add_q0_body_update, direct_parts, exchange_part,
                                head_split, check_split)

OMEGA = 270.0


def _system(n_g, n_mu, seed):
    rng = np.random.default_rng(seed)
    Z = (rng.standard_normal((n_g, n_mu)) + 1j * rng.standard_normal((n_g, n_mu))) / np.sqrt(n_mu)
    v = np.concatenate([[60.0], rng.uniform(0.2, 6.0, n_g - 1)])
    A = (rng.standard_normal((n_mu, n_mu)) + 1j * rng.standard_normal((n_mu, n_mu))) / n_mu
    chi = -0.5 * A @ A.conj().T
    V = Z.conj().T @ (v[:, None] * Z)
    W = np.linalg.solve(np.eye(n_mu) - V @ chi, V)
    Wpw = np.linalg.solve(np.eye(n_g) - (v[:, None] * Z) @ chi @ Z.conj().T, np.diag(v))
    return rng, Z, v, V, W, Wpw


def _eigh(M):
    return np.linalg.eigh(0.5 * (M + M.conj().T))


def _dense_parts(Ccc, Cvv, Z, Wpw, lam):
    """E1a–c with M(G) = √Ω Σ_μ conj(Z[G, μ]) C(μ) and W̄ = W_PW/8π."""
    Mc = np.sqrt(OMEGA) * np.einsum("gm,bcdm->bgcd", Z.conj(), Ccc)
    Mv = np.sqrt(OMEGA) * np.einsum("gm,bwum->bgwu", Z.conj(), Cvv)
    Wb = Wpw / (8 * np.pi)
    head = np.einsum("bcd,bwu->bcdwu", Mc[:, 0], Mv[:, 0].conj())
    wing = (np.einsum("bgcd,g,bwu->bcdwu", Mc[:, 1:], Wb[1:, 0], Mv[:, 0].conj())
            + np.einsum("bcd,h,bhwu->bcdwu", Mc[:, 0], Wb[0, 1:], Mv[:, 1:].conj()))
    body = np.einsum("bgcd,gh,bhwu->bcdwu", Mc[:, 1:], Wb[1:, 1:], Mv[:, 1:].conj())
    return head, lam[:, None, None, None, None] * wing, body


@pytest.mark.parametrize("n_g,n_mu", [(60, 24), (16, 24)])
def test_split_equals_plane_wave_partition(n_g, n_mu):
    rng, Z, v, V, W, Wpw = _system(n_g, n_mu, seed=n_g)
    assert np.allclose(Z.conj().T @ Wpw @ Z, W, atol=1e-10)      # E2
    sp = head_split(V, W, Z[0], v[0], eigh=_eigh)
    check_split(sp, where="synthetic")
    assert abs(sp.W11 - Wpw[0, 0].real) < 1e-9 * abs(Wpw[0, 0])  # E4
    y_pw = Wpw[0, 1:] @ Z[1:]
    assert np.linalg.norm(sp.y - y_pw) < 1e-9 * np.linalg.norm(y_pw)   # E5
    H = sp.W11 * np.outer(Z[0].conj(), Z[0])
    Y = np.outer(Z[0].conj(), sp.y) + np.outer(sp.y.conj(), Z[0])
    P1 = np.zeros((n_g, n_g)); P1[0, 0] = 1.0
    Pb = np.eye(n_g) - P1
    assert np.allclose(H, Z.conj().T @ P1 @ Wpw @ P1 @ Z, atol=1e-9)            # E6
    assert np.allclose(Y, Z.conj().T @ (P1 @ Wpw @ Pb + Pb @ Wpw @ P1) @ Z, atol=1e-9)
    assert np.allclose(W - H - Y, Z.conj().T @ Pb @ Wpw @ Pb @ Z, atol=1e-9)

    b, nc, nv = 3, 2, 3
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    Ccc, Cvv = cplx(b, nc, nc, n_mu), cplx(b, nv, nv, n_mu)
    lam = rng.uniform(0.2, 1.0, b)
    got = direct_parts(Ccc, Cvv, np.broadcast_to(W, (b, n_mu, n_mu)),
                       np.broadcast_to(Z[0], (b, n_mu)), np.broadcast_to(sp.y, (b, n_mu)),
                       np.full(b, sp.W11), omega=OMEGA, lam_q=lam)
    want = _dense_parts(Ccc, Cvv, Z, Wpw, lam)
    for g, w, name in zip(got, want, ("head", "wing", "body")):
        assert np.linalg.norm(g - w) < 1e-9 * np.linalg.norm(w), name    # E7 vs E1


def test_red_twins_fail():
    rng, Z, v, V, W, Wpw = _system(60, 24, seed=7)
    y_pw = Wpw[0, 1:] @ Z[1:]
    sp = head_split(V, W, Z[0], v[0], eigh=_eigh)
    # (1) E5 without its head term: y would carry W's head along the G1 row.
    y_bad = sp.y + v[0] ** 2 * sp.figures["s"] * Z[0]
    assert np.linalg.norm(y_bad - y_pw) > 1e-2 * np.linalg.norm(y_pw)
    # (2) the conjugated column is not V's head row: the gate refuses it.
    with pytest.raises(ValueError, match="bsemat_head_row|bsemat_head_screening"):
        check_split(head_split(V, W, Z[0].conj(), v[0], eigh=_eigh), where="red twin")
    # (3) V in place of W − V: the head comes out unscreened-plus, not W11.
    a = np.linalg.solve(V, Z[0].conj())
    s_bad = float((a.conj() @ V @ a).real)
    assert abs(v[0] + v[0] ** 2 * s_bad - Wpw[0, 0].real) > 1e-2 * abs(Wpw[0, 0])


def test_exchange_matches_bgw_literal_form_for_real_zeta():
    """E7 exchange vs BerkeleyGW's Σ_G M*_vc v̄ M_v′c′ when ζ(r) is real (Z(−G) = Z(G)*)."""
    rng = np.random.default_rng(3)
    n_half, n_mu, nk, nc, nv = 10, 12, 2, 2, 2
    Zh = rng.standard_normal((n_half, n_mu)) + 1j * rng.standard_normal((n_half, n_mu))
    Z = np.concatenate([rng.standard_normal((1, n_mu)), Zh, Zh.conj()])   # G = 0, +G, -G
    v = np.concatenate([[0.0], np.tile(rng.uniform(0.5, 3.0, n_half), 2)])
    Vbar = Z.conj().T @ (v[:, None] * Z)
    assert np.allclose(Vbar.imag, 0, atol=1e-12)
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    Ccv = cplx(nk, nc, nv, n_mu)                         # C_cv = Σ conj(ψ_c) ψ_v
    X = exchange_part(Ccv, Ccv, Vbar, omega=OMEGA)
    Mvc = np.sqrt(OMEGA) * np.einsum("gm,kcvm->kgvc", Z.conj(), Ccv.conj())   # M_vc(G), C_vc = conj(C_cv)
    lit = np.einsum("kgvc,g,pgud->pkdcuv", Mvc.conj(), v / (8 * np.pi), Mvc)
    assert np.linalg.norm(X - lit) < 1e-10 * np.linalg.norm(lit)


def test_q0_hook_moves_only_the_body():
    """At q = 0 (W11 = 0, y = 0) a rank-n update of W(Γ) changes the body by C AᴴDA C* and nothing else."""
    rng = np.random.default_rng(11)
    n_mu, b, nc, nv = 10, 2, 2, 2
    cplx = lambda *s: rng.standard_normal(s) + 1j * rng.standard_normal(s)
    H = cplx(n_mu, n_mu); W0 = H @ H.conj().T
    A = cplx(1, n_mu); D = np.array([[0.7]])
    Ccc, Cvv, z = cplx(b, nc, nc, n_mu), cplx(b, nv, nv, n_mu), cplx(n_mu)
    args = dict(omega=OMEGA, lam_q=np.zeros(b))
    zero = np.zeros((b, n_mu), complex)
    h0, w0, b0 = direct_parts(Ccc, Cvv, np.broadcast_to(W0, (b, n_mu, n_mu)), np.broadcast_to(z, (b, n_mu)),
                              zero, np.zeros(b), **args)
    W1 = add_q0_body_update(W0, A, D)
    h1, w1, b1 = direct_parts(Ccc, Cvv, np.broadcast_to(W1, (b, n_mu, n_mu)), np.broadcast_to(z, (b, n_mu)),
                              zero, np.zeros(b), **args)
    a_c = np.einsum("bcdm,m->bcd", Ccc, A[0].conj()); a_v = np.einsum("bwum,m->bwu", Cvv, A[0].conj())
    want = OMEGA / (8 * np.pi) * 0.7 * np.einsum("bcd,bwu->bcdwu", a_c, a_v.conj())
    assert np.allclose(h1, h0) and np.allclose(w1, w0) and not np.any(w1)
    assert np.linalg.norm((b1 - b0) - want) < 1e-10 * np.linalg.norm(want)
