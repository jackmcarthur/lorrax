"""Covariance gate for the BSE exchange conjugation.

The exchange kernel is a quadratic form in the pair vertex
``M[k,c,v,mu] = sum_s conj(psi_c) psi_v`` against the BARE Coulomb tile.  The
transition density ``<0|rho|Psi> = sum A_cvk psi_ck conj(psi_vk)`` fixes its
form to

    K^x = M V M^dagger

i.e. the CONJUGATED vertex sits on the forward (encode) leg and the BARE vertex
on the back-contract.  The reverse assignment builds ``conj(M) V M^T``, which is
``conj(K^x)`` whenever V is real — a Hermitian operator with the right norm and
the right hermiticity, so nothing local catches it.  What catches it is
COVARIANCE: the direct/W term is covariant under the physical symmetry operator
U, the flipped exchange is covariant only under ``conj(U)``, and no single
operator commutes with their sum.  That mismatch is what split the exciton
multiplets.

These gates run on a SYNTHETIC fixture carrying an EXACT symmetry by
construction, so the statement is a 1e-16-vs-O(1) discriminator rather than a
converged-physics measurement, and it needs no GPU, no FFI and no restart file.

Construction (cyclic group of order nk, generator g):
    g : k -> k+1 (mod nk),  mu -> sigma(mu) (cyclic shift),
        conduction band c picks up d_c = exp(2i pi a_c / nk),
        valence    band v picks up e_v = exp(2i pi b_v / nk)
    psi_c[j,c,s,mu] = d_c^j psi_c[0,c,s,sigma^-j(mu)]        (likewise psi_v)
    V_q0 and W(q) circulant in mu  =>  sigma-invariant;  eps k-independent.
Then ``M[j,c,v,mu] = conj(d_c)^j e_v^j M[0,c,v,sigma^-j mu]``, so the pair-space
symmetry operator is ``U = (conj(D^c) (x) D^v) (x) shift_k``.

Evidence chain for the physics: /pscratch/sd/j/jackm/seam_0808_x (x1..x4).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

NK, NC, NV, NS, M_REP = 4, 2, 2, 1, 2
NMU = NK * M_REP
A_C = np.array([0, 1])
B_V = np.array([0, 3])
N = NC * NV * NK

# Covariance thresholds.  The fixture symmetry is EXACT, so the correct operator
# commutes to round-off; the band below is pure float64 headroom, and the
# shipping conjugation misses it by ~15 orders of magnitude.
COV_TOL = 1e-12
COV_RED_MIN = 1e-3


def _circulant(col, n):
    return np.array([[col[(i - j) % n] for j in range(n)] for i in range(n)])


def _circulant_herm(n, rng):
    col = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    C = _circulant(col, n)
    return 0.5 * (C + C.conj().T)


def _circulant_real_sym(n, rng):
    col = rng.standard_normal(n)
    for k in range(n):
        col[k] = col[(-k) % n] = 0.5 * (col[k] + col[(-k) % n])
    return _circulant(col, n)


@pytest.fixture(scope="module")
def sym_fixture():
    """psi/eps/V/W carrying the exact cyclic symmetry described in the docstring."""
    rng = np.random.default_rng(20260808)
    psi_c0 = rng.standard_normal((NC, NS, NMU)) + 1j * rng.standard_normal((NC, NS, NMU))
    psi_v0 = rng.standard_normal((NV, NS, NMU)) + 1j * rng.standard_normal((NV, NS, NMU))
    d_c = np.exp(2j * np.pi * A_C / NK)
    e_v = np.exp(2j * np.pi * B_V / NK)
    psi_c = np.empty((NK, NC, NS, NMU), dtype=complex)
    psi_v = np.empty((NK, NV, NS, NMU), dtype=complex)
    for j in range(NK):
        psi_c[j] = (d_c ** j)[:, None, None] * np.roll(psi_c0, -j * M_REP, axis=-1)
        psi_v[j] = (e_v ** j)[:, None, None] * np.roll(psi_v0, -j * M_REP, axis=-1)
    V_q0 = _circulant_herm(NMU, rng)          # complex Hermitian: the harder case
    W_q = np.zeros((NMU, NMU, NK, 1, 1), dtype=complex)
    for q in range(NK):
        W_q[:, :, q, 0, 0] = 0.3 * _circulant_real_sym(NMU, rng)
    for q in range(NK):
        mq = (-q) % NK
        A = 0.5 * (W_q[:, :, q, 0, 0] + W_q[:, :, mq, 0, 0].conj().T)
        W_q[:, :, q, 0, 0], W_q[:, :, mq, 0, 0] = A, A.conj().T
    return dict(psi_c=psi_c, psi_v=psi_v,
                eps_c=np.tile(np.array([0.9, 1.3]), (NK, 1)),
                eps_v=np.tile(np.array([-0.2, -0.5]), (NK, 1)),
                V_q0=V_q0, W_q=W_q, nkx=NK, nky=1, nkz=1)


def _dense(data, exchange="fixed"):
    """(H, Kx, Kd).  exchange='fixed' -> M V M^dag; 'shipping' -> conj(M) V M^T."""
    psi_c, psi_v, V_q0, W_q = (data["psi_c"], data["psi_v"],
                               data["V_q0"], data["W_q"])
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v)
    D = np.transpose(data["eps_c"][:, :, None] - data["eps_v"][:, None, :], (1, 2, 0))
    if exchange == "fixed":
        lhs = np.einsum("kcvM,MN->kcvN", M, V_q0)
        Kx = np.einsum("kcvN,KCVN->cvkCVK", lhs, np.conj(M)) / NK
    else:
        lhs = np.einsum("kcvM,MN->kcvN", np.conj(M), V_q0)
        Kx = np.einsum("kcvN,KCVN->cvkCVK", lhs, M) / NK
    Wflat = W_q.reshape(NMU, NMU, NK)
    Kd = np.zeros((NC, NV, NK, NC, NV, NK), dtype=complex)
    for k in range(NK):
        for kp in range(NK):
            Wq = Wflat[:, :, (k - kp) % NK]
            Pc = np.einsum("ctm,Ctm->cCm", np.conj(psi_c[k]), psi_c[kp])
            Pv = np.einsum("vsn,Vsn->vVn", psi_v[k], np.conj(psi_v[kp]))
            Kd[:, :, k, :, :, kp] = np.einsum("cCm,mn,vVn->cvCV", Pc, Wq, Pv) / NK
    Kx2, Kd2 = Kx.reshape(N, N), Kd.reshape(N, N)
    return np.diag(D.reshape(-1).astype(complex)) + Kx2 - Kd2, Kx2, Kd2


def _sym_U(conjugate=False):
    d_c, e_v = np.exp(2j * np.pi * A_C / NK), np.exp(2j * np.pi * B_V / NK)
    U = np.zeros((N, N), dtype=complex)
    for c in range(NC):
        for v in range(NV):
            fac = np.conj(d_c[c]) * e_v[v]
            if conjugate:
                fac = np.conj(fac)
            for k in range(NK):
                U[(c * NV + v) * NK + (k + 1) % NK, (c * NV + v) * NK + k] = fac
    return U


def _rel_comm(H, U):
    return float(np.linalg.norm(H @ U - U @ H) / np.linalg.norm(H))


def _relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def _matvec(kind, data, X, include_W=True):
    from jax.sharding import Mesh
    from bse.bse_serial import (apply_bse_hamiltonian_single_device,
                                compute_pair_amplitude)
    if kind == "serial":
        return apply_bse_hamiltonian_single_device(
            jnp.asarray(X), jnp.asarray(data["psi_c"]), jnp.asarray(data["psi_v"]),
            jnp.asarray(data["eps_c"]), jnp.asarray(data["eps_v"]),
            jnp.asarray(data["W_q"]), jnp.asarray(data["V_q0"]), NK, 1, 1,
            include_W=include_W)
    from bse.bse_ring_comm import build_bse_ring_matvec, make_bse_shardings
    from bse.bse_simple import build_bse_simple_matvec
    from bse.bse_stack_matvec import build_bse_stack_matvec
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    sh = make_bse_shardings(mesh)
    with mesh:
        pcx = jax.lax.with_sharding_constraint(jnp.asarray(data["psi_c"]), sh.psi_x)
        pcy = jax.lax.with_sharding_constraint(jnp.asarray(data["psi_c"]), sh.psi_y)
        pvx = jax.lax.with_sharding_constraint(jnp.asarray(data["psi_v"]), sh.psi_x)
        pvy = jax.lax.with_sharding_constraint(jnp.asarray(data["psi_v"]), sh.psi_y)
        Wq = jax.lax.with_sharding_constraint(jnp.asarray(data["W_q"]), sh.W)
        Vq = jax.lax.with_sharding_constraint(jnp.asarray(data["V_q0"]), sh.V)
        Xs = jax.lax.with_sharding_constraint(jnp.asarray(X), sh.X)
        W_R = jnp.fft.ifftn(Wq, axes=(2, 3, 4), norm="ortho")
        M_X = jax.lax.with_sharding_constraint(compute_pair_amplitude(pcx, pvx), sh.psi_x)
        M_Y = jax.lax.with_sharding_constraint(compute_pair_amplitude(pcy, pvy), sh.psi_y)
        if kind == "simple":
            mv = build_bse_simple_matvec(mesh, NK, 1, 1, include_W=include_W)
        elif kind == "stack":
            mv = build_bse_stack_matvec(mesh, NK, 1, 1)
        else:
            mv = build_bse_ring_matvec(mesh, NK, 1, 1, include_W=include_W,
                                       low_mem=(kind == "ring"))
        out = mv(Xs, pcx, pcy, pvx, pvy, jnp.asarray(data["eps_c"]),
                 jnp.asarray(data["eps_v"]), W_R, Vq, M_X, M_Y)
        out.block_until_ready()
    return out


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------
def test_fixture_symmetry_is_exact(sym_fixture):
    """The W/direct term — which this fix does NOT touch — is covariant under U.

    This is the anchor: it proves the fixture's symmetry is real and that U is
    the PHYSICAL symmetry operator, so 'the exchange must commute with this same
    U' is a statement about the exchange, not about the fixture.
    """
    _, _, Kd = _dense(sym_fixture)
    U = _sym_U()
    assert _rel_comm(-Kd, U) < COV_TOL, (
        f"W term not covariant under U: {_rel_comm(-Kd, U):.3e} — fixture is broken")
    assert _rel_comm(-Kd, _sym_U(conjugate=True)) > COV_RED_MIN, (
        "RED TWIN FAILED: W must NOT be covariant under conj(U)")


def test_exchange_covariant_under_same_U_as_W(sym_fixture):
    """K^x = M V M† commutes with the same U as W; conj(M) V M^T does not.

    Red twin (the shipping conjugation restored locally) must trip this, and
    must instead be covariant under conj(U) — the precise signature of the bug.
    """
    _, Kx_fix, _ = _dense(sym_fixture, "fixed")
    _, Kx_shp, _ = _dense(sym_fixture, "shipping")
    U, Uc = _sym_U(), _sym_U(conjugate=True)

    assert _rel_comm(Kx_fix, U) < COV_TOL, (
        f"fixed exchange not covariant: {_rel_comm(Kx_fix, U):.3e}")
    # RED TWIN: the old conjugation.
    assert _rel_comm(Kx_shp, U) > COV_RED_MIN, (
        "RED TWIN FAILED: shipping exchange should NOT commute with U")
    # ...and it is covariant under the WRONG (conjugate) operator instead.
    assert _rel_comm(Kx_shp, Uc) < COV_TOL, (
        "shipping exchange should be covariant under conj(U) — that mismatch "
        "against W is the bug")


def test_full_H_covariance_and_red_twins(sym_fixture):
    """D + K^x − K^d commutes with U only with the corrected exchange."""
    H_fix, _, _ = _dense(sym_fixture, "fixed")
    H_shp, _, _ = _dense(sym_fixture, "shipping")
    U, Uc = _sym_U(), _sym_U(conjugate=True)
    assert _relerr(H_fix, H_fix.conj().T) < 1e-12, "fixed H must be Hermitian"
    assert _rel_comm(H_fix, U) < COV_TOL, (
        f"fixed H not covariant: {_rel_comm(H_fix, U):.3e}")
    # RED TWIN 1 — the old conjugation restored.
    assert _rel_comm(H_shp, U) > COV_RED_MIN, (
        "RED TWIN FAILED: shipping H should not commute with U")
    # RED TWIN 2 — the wrong symmetry operator.
    assert _rel_comm(H_fix, Uc) > COV_RED_MIN, (
        "RED TWIN FAILED: fixed H should not commute with conj(U)")


@pytest.mark.parametrize("kind", ["serial", "simple", "stack"])
def test_cross_solver_agreement(sym_fixture, kind):
    """Every live matvec path builds the SAME corrected operator.

    They agreed before this fix only because they shared the bug; this pins that
    they still agree with each other AND now agree with the corrected dense
    build.  ``ring``/``gather`` need the MKL batched-GEMM FFI for their W leg and
    are covered on the GPU/Perlmutter leg instead.
    """
    H_fix, _, _ = _dense(sym_fixture, "fixed")
    rng = np.random.default_rng(11)
    X = rng.standard_normal((1, NC, NV, NK)) + 1j * rng.standard_normal((1, NC, NV, NK))
    hx = np.asarray(_matvec(kind, sym_fixture, X))[0].reshape(-1)
    ref = H_fix @ X[0].reshape(-1)
    assert _relerr(hx, ref) < 1e-10, f"{kind} vs dense: {_relerr(hx, ref):.3e}"
    # RED TWIN: the same matvec must NOT reproduce the shipping operator.
    H_shp, _, _ = _dense(sym_fixture, "shipping")
    assert _relerr(hx, H_shp @ X[0].reshape(-1)) > COV_RED_MIN, (
        "RED TWIN FAILED: matvec still matches the shipping dense H")
