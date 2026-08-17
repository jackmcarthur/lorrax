"""Sign-sensitivity gates for the ladder screening operator.

WHY THIS FILE EXISTS.  Every other ladder gate — the RPA limit (W -> 0),
per-q hermiticity, W(-q) = conj(W(q)) reciprocity, and the dense oracle
(which is built from the same derived block spec) — passes IDENTICALLY under
a flipped direct rung.  The first physics A/B (scalar Si, claim 0228) opened
the gap, the opposite of the e-h-ladder-vertex literature's full-W
expectation, and the sign question could not be answered by any gate then in
the tree.  Measured 2026-08-16 (evidence/sign_probe/): the operator's sign IS
correct — the exchange-free ladder poles sit below the free e-h continuum and
the ladder ENHANCES screening on the real scalar-Si restart at q=0 AND finite
q (Loewner-negative Wc delta, 4-8%); the gap opening is an observable of the
v1 body-only scope (the q=0 head stays RPA by design), not of the rung sign.

These cells pin that verdict permanently, densely and in milliseconds, with
kernels that carry the PHYSICAL sign: V and W are both positive-definite
Coulomb-like tiles.  The suite's random-Hermitian synthetic W cannot answer a
sign question — a sign gate needs signed inputs.

Two observables, each with its red twin (the FLIPPED rung must fail):

  1. POLES: min eig(D - Kd) < min(D)  (attraction pulls the exchange-free
     ladder poles below the continuum); flipped rung pushes them above.
  2. SCREENING DIRECTION: the dense 2N resolvent's Wc(0) tile is MORE
     negative with the production-sign rung than RPA (trace and Loewner
     direction), and LESS negative with the flipped rung.

Dense blocks follow the certified oracle spec (test_bse_w_ladder_dense.py's
_dense_ladder_blocks / _dense_wc_columns) with the rung sign a parameter.
numpy-only: no jax, no fixture, no GPU.
"""
import numpy as np
import pytest

NKX, NKY, NKZ, NC, NV, NMU = 2, 2, 1, 2, 2, 8
GRID = (NKX, NKY, NKZ)
NK = NKX * NKY * NKZ
N = NC * NV * NK


def _q_flat(k, kp):
    ku = np.array(np.unravel_index(k, GRID))
    kpu = np.array(np.unravel_index(kp, GRID))
    return int(np.ravel_multi_index(tuple((ku - kpu) % np.array(GRID)), GRID))


def _payload():
    rng = np.random.default_rng(11)

    def cplx(*shape):
        return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / 2.0

    psi_c = cplx(NK, NC, 1, NMU)
    psi_v = cplx(NK, NV, 1, NMU)
    eps_c = 1.0 + 0.3 * rng.standard_normal((NK, NC))
    eps_v = -1.0 + 0.3 * rng.standard_normal((NK, NV))
    # PHYSICAL kernels: V PSD real symmetric, W = 0.6 V (a screened
    # interaction is a positive kernel weaker than v — sign AND ordering).
    G = rng.standard_normal((NMU, 2 * NMU))
    V = 0.10 * (G @ G.T) / (2 * NMU)
    W = 0.6 * V
    return psi_c, psi_v, eps_c, eps_v, V, W


def _blocks(sign):
    """(Dm, Kx, Kd, Kd_B, Mmat, V) with the rung entering as ``sign * W``."""
    psi_c, psi_v, eps_c, eps_v, V, W = _payload()
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v)
    D = np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))
    Dm = np.diag(D.reshape(-1).astype(np.complex128))
    lhs = np.einsum("kcvM,MN->kcvN", M, V)
    Kx = np.einsum("kcvN,KCVN->cvkCVK", lhs, np.conj(M)).reshape(N, N) / NK
    Kd = np.zeros((NC, NV, NK, NC, NV, NK), dtype=np.complex128)
    Kd_B = np.zeros_like(Kd)
    Wq = sign * W          # q-independent tile: W(-q)=conj(W(q)) trivially
    for k in range(NK):
        for kp in range(NK):
            Pc = np.einsum("ctm,Ctm->cCm", np.conj(psi_c[k]), psi_c[kp])
            Pv = np.einsum("vsn,Vsn->vVn", psi_v[k], np.conj(psi_v[kp]))
            Kd[:, :, k, :, :, kp] = np.einsum("cCm,mn,vVn->cvCV", Pc, Wq, Pv) / NK
            PcB = np.einsum("ctm,Vtm->cVm", np.conj(psi_c[k]), psi_v[kp])
            PvB = np.einsum("vsn,Csn->vCn", psi_v[k], np.conj(psi_c[kp]))
            Kd_B[:, :, k, :, :, kp] = np.einsum("cVm,mn,vCn->cvCV", PcB, Wq, PvB) / NK
    Mmat = np.transpose(M, (1, 2, 0, 3)).reshape(N, NMU)
    return Dm, Kx, Kd.reshape(N, N), Kd_B.reshape(N, N), Mmat, V, float(np.min(D))


def _wc_tile(Dm, Kx, Kd, Kd_B, Mmat, V):
    """Dense 2N resolvent Wc(0) full tile, production block spec + hybrid row."""
    A = Dm + Kx - Kd
    B = Kx - Kd_B
    row = (-(Kx - np.conj(Kd_B)), -(Dm + Kx - np.conj(Kd)))
    H = np.block([[A, B], [row[0], row[1]]])
    lhs = -H  # z = 0
    out = np.zeros((NMU, NMU), dtype=np.complex128)
    snk = np.sqrt(float(NK))
    for nu0 in range(NMU):
        g = np.zeros(NMU)
        g[nu0] = 1.0
        f = (Mmat @ (V @ g)) / snk
        x = np.linalg.solve(lhs, np.concatenate([f, -f]))
        out[:, nu0] = (V @ (np.conj(Mmat).T @ (x[:N] + x[N:]))) / snk
    return out


def test_attractive_rung_pulls_exchange_free_poles_below_the_continuum():
    Dm, _, Kd, _, _, _, minD = _blocks(+1.0)
    ev = np.linalg.eigvalsh(0.5 * ((Dm - Kd) + np.conj((Dm - Kd).T)))
    assert ev[0] < minD - 1e-9, (
        f"exchange-free ladder pole {ev[0]:+.6f} is not below the free "
        f"continuum {minD:+.6f}: the rung is not attractive")


def test_flipped_rung_fails_the_pole_gate():
    """Red twin: the OLD failure mode (repulsive rung) must be caught."""
    Dm, _, Kd, _, _, _, minD = _blocks(+1.0)
    ev = np.linalg.eigvalsh(0.5 * ((Dm + Kd) + np.conj((Dm + Kd).T)))
    assert not (ev[0] < minD - 1e-9), "a repulsive rung passed the pole gate"


def test_production_sign_rung_enhances_screening():
    Dm, Kx, Kd, Kd_B, Mmat, V, _ = _blocks(+1.0)
    wc_rpa = _wc_tile(Dm, Kx, 0.0 * Kd, 0.0 * Kd_B, Mmat, V)
    wc_lad = _wc_tile(Dm, Kx, Kd, Kd_B, Mmat, V)
    d = 0.5 * ((wc_lad - wc_rpa) + np.conj((wc_lad - wc_rpa).T))
    tr = float(np.real(np.trace(d)))
    ev = np.linalg.eigvalsh(d)
    assert tr < 0.0, f"ladder did not enhance screening: tr(dWc) = {tr:+.3e}"
    # Loewner direction: allow numerical zeros, forbid any genuinely
    # positive direction (fraction of the dominant one).
    assert ev[-1] < 0.05 * abs(ev[0]), (
        f"dWc has a positive screening direction: eigs {ev}")


def test_flipped_rung_fails_the_screening_direction_gate():
    """Red twin: a flipped rung REDUCES screening and must be caught."""
    Dm, Kx, Kd, Kd_B, Mmat, V, _ = _blocks(+1.0)
    wc_rpa = _wc_tile(Dm, Kx, 0.0 * Kd, 0.0 * Kd_B, Mmat, V)
    wc_flip = _wc_tile(Dm, Kx, -Kd, -Kd_B, Mmat, V)
    tr = float(np.real(np.trace(wc_flip - wc_rpa)))
    assert tr > 0.0, "the red twin stopped discriminating (flip looks enhancing)"
