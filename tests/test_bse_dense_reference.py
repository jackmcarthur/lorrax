"""Dense-reference gate for the Q=0 BSE kernel (Phase-2, step 1).

Builds the explicit ⟨cvk|H|c'v'k'⟩ matrix from the SAME padded, head-injected
arrays the production matvecs consume (``bse_io._load_ring_subset``), then
checks each live matvec against it.

The settled physics (reports/bse_refactor_map_2026-07-15/archive/adjudication/
VERDICT.md): the Q=0 exchange is DENSE in (k,k′) —

    ⟨cvk|K^x|c'v'k'⟩ = (1/Nk) Σ_{μν} conj(M_cvk(μ)) V_q0(μν) M_c'v'k'(ν)

with no δ_kk′.  The B1 fix k-SUMS the exchange encode (``S[b,ν]``, k-free) and
broadcasts the decode back at every k, so the full-H / D+V / spectrum checks
match the dense reference exactly.  The W-only positive control is independent
of B1 (W is untouched) and pins the convolution sign q = k − k′.

Piggybacks the session-scoped ``gnppm_session`` GW run (MoS2 3×3×1, nk=9); a
2v2c window ⇒ N = nc·nv·nk = 36, so the dense build + eigh are trivial.
"""
from __future__ import annotations

import numpy as np
import pytest

import harness

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


# The session-scoped ``bse_dense_state`` fixture lives in tests/conftest.py so
# the trial-stack matvec gate (test_bse_stack_matvec.py) can reuse the same
# loaded BSE subset without a second restart load.


# ---------------------------------------------------------------------------
# Dense reference builder — explicit (k,k′) quadrature in numpy.
# ---------------------------------------------------------------------------
def _q_flat(k, kp, grid):
    """Flat index of q = (k − k′) mod grid, C-order over (kx,ky,kz)."""
    ck = np.array(np.unravel_index(k, grid))
    ckp = np.array(np.unravel_index(kp, grid))
    q = (ck - ckp) % np.array(grid)
    return int(np.ravel_multi_index(tuple(q), grid))


def _build_dense_H(data):
    """Return (H, D_flat, Kx, Kd) as (N,N) numpy for flat index I=(c,v,k)."""
    psi_c = np.asarray(data["psi_c"])   # (k,c,s,μ)
    psi_v = np.asarray(data["psi_v"])   # (k,v,s,ν)
    eps_c = np.asarray(data["eps_c"])   # (k,c)
    eps_v = np.asarray(data["eps_v"])   # (k,v)
    V_q0 = np.asarray(data["V_q0"])     # (μ,μ)
    W_q = np.asarray(data["W_q"])       # (μ,μ,nkx,nky,nkz)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    grid = (nkx, nky, nkz)
    nk = nkx * nky * nkz
    nc = psi_c.shape[1]
    nv = psi_v.shape[1]
    nmu = psi_c.shape[3]
    N = nc * nv * nk

    # Pair amplitude M[k,c,v,μ] = Σ_s conj(ψ_c) ψ_v.
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v)

    # D — diagonal (correct as coded): ε_c(k) − ε_v(k), layout (c,v,k).
    D = np.transpose(eps_c[:, :, None] - eps_v[:, None, :], (1, 2, 0))

    # Exchange — DENSE in (k,k′).  Kx[c,v,k, c',v',k'].
    lhs = np.einsum("kcvM,MN->kcvN", np.conj(M), V_q0)   # conj(M)·V
    Kx = np.einsum("kcvN,KCVN->cvkCVK", lhs, M) / nk

    # Direct — screened W_{μν}(k−k′), 1/Nk, q = k − k′ (ortho fft convolution).
    Wflat = W_q.reshape(nmu, nmu, nk)   # (μ,ν,qflat) C-order
    Kd = np.zeros((nc, nv, nk, nc, nv, nk), dtype=np.complex128)
    for k in range(nk):
        for kp in range(nk):
            q = _q_flat(k, kp, grid)
            Wq = Wflat[:, :, q]                                 # (μ,ν)
            Pc = np.einsum("ctm,Ctm->cCm",
                           np.conj(psi_c[k]), psi_c[kp])        # (c,c',μ)
            Pv = np.einsum("vsn,Vsn->vVn",
                           psi_v[k], np.conj(psi_v[kp]))        # (v,v',ν)
            Kd[:, :, k, :, :, kp] = np.einsum(
                "cCm,mn,vVn->cvCV", Pc, Wq, Pv) / nk

    Kx2 = Kx.reshape(N, N)
    Kd2 = Kd.reshape(N, N)
    H = np.diag(D.reshape(-1).astype(np.complex128)) + Kx2 - Kd2
    return H, D.reshape(-1), Kx2, Kd2


# ---------------------------------------------------------------------------
# Matvec drivers (single device; 1×1 mesh for the sharded kinds).
# ---------------------------------------------------------------------------
def _random_X(nb, nc, nv, nk):
    rng = np.random.default_rng(1234)
    x = (rng.standard_normal((nb, nc, nv, nk))
         + 1j * rng.standard_normal((nb, nc, nv, nk)))
    return jnp.asarray(x)


def _serial_matvec(data, X, include_W):
    from bse.bse_serial import apply_bse_hamiltonian_single_device
    return apply_bse_hamiltonian_single_device(
        X, data["psi_c"], data["psi_v"], data["eps_c"], data["eps_v"],
        data["W_q"], data["V_q0"],
        int(data["nkx"]), int(data["nky"]), int(data["nkz"]),
        include_W=include_W)


def _sharded_matvec(kind, data, X, include_W):
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from bse.bse_ring_comm import build_bse_ring_matvec, make_bse_shardings
    from bse.bse_simple import build_bse_simple_matvec
    from bse.bse_serial import compute_pair_amplitude

    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    sh = make_bse_shardings(mesh)
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    with mesh:
        psi_c_X = jax.lax.with_sharding_constraint(data["psi_c"], sh.psi_x)
        psi_c_Y = jax.lax.with_sharding_constraint(data["psi_c"], sh.psi_y)
        psi_v_X = jax.lax.with_sharding_constraint(data["psi_v"], sh.psi_x)
        psi_v_Y = jax.lax.with_sharding_constraint(data["psi_v"], sh.psi_y)
        W_q = jax.lax.with_sharding_constraint(data["W_q"], sh.W)
        V_q0 = jax.lax.with_sharding_constraint(data["V_q0"], sh.V)
        Xs = jax.lax.with_sharding_constraint(X, sh.X)
        W_R = jnp.fft.ifftn(W_q, axes=(2, 3, 4), norm="ortho")
        # Hoisted V-term pair amplitudes (audit P3) — matvec args, not recomputed.
        M_X = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(psi_c_X, psi_v_X), sh.psi_x)
        M_Y = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(psi_c_Y, psi_v_Y), sh.psi_y)
        if kind == "simple":
            mv = build_bse_simple_matvec(mesh, nkx, nky, nkz, include_W=include_W)
        else:
            mv = build_bse_ring_matvec(
                mesh, nkx, nky, nkz, include_W=include_W,
                low_mem=(kind == "ring"))
        HX = mv(Xs, psi_c_X, psi_c_Y, psi_v_X, psi_v_Y,
                data["eps_c"], data["eps_v"], W_R, V_q0, M_X, M_Y)
        HX.block_until_ready()
    return HX


def _run_matvec(kind, data, X, include_W):
    if kind == "serial":
        return _serial_matvec(data, X, include_W)
    return _sharded_matvec(kind, data, X, include_W)


def _relerr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


MATVEC_KINDS = ["serial", "simple", "ring"]


# ---------------------------------------------------------------------------
# Gates.
# ---------------------------------------------------------------------------
@pytest.mark.gpu
@pytest.mark.parametrize("kind", MATVEC_KINDS)
def test_w_positive_control(bse_dense_state, kind):
    """W-only control (passes pre- AND post-fix; pins q = k−k′ sign).

    (matvec_W − matvec_noW)(X) == −(Kd @ X), exactly — W is untouched by B1.
    """
    harness.skip_unless_gpu(pytest)
    data = bse_dense_state
    _, _, _, Kd = _build_dense_H(data)
    nc = int(data["psi_c"].shape[1]); nv = int(data["psi_v"].shape[1])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    X = _random_X(1, nc, nv, nk)
    hxw = np.asarray(_run_matvec(kind, data, X, True))[0].reshape(-1)
    hx0 = np.asarray(_run_matvec(kind, data, X, False))[0].reshape(-1)
    lhs = hxw - hx0
    rhs = -(Kd @ np.asarray(X)[0].reshape(-1))
    assert _relerr(lhs, rhs) < 1e-9, f"{kind}: W control rel-err {_relerr(lhs, rhs):.2e}"


@pytest.mark.gpu
@pytest.mark.parametrize("kind", MATVEC_KINDS)
def test_full_H_matches_dense(bse_dense_state, kind):
    """matvec(X) == H @ X — dense (k,k') exchange (B1 fixed)."""
    harness.skip_unless_gpu(pytest)
    data = bse_dense_state
    H, _, _, _ = _build_dense_H(data)
    nc = int(data["psi_c"].shape[1]); nv = int(data["psi_v"].shape[1])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    X = _random_X(1, nc, nv, nk)
    hx = np.asarray(_run_matvec(kind, data, X, True))[0].reshape(-1)
    ref = H @ np.asarray(X)[0].reshape(-1)
    assert _relerr(hx, ref) < 1e-9, f"{kind}: full-H rel-err {_relerr(hx, ref):.2e}"


@pytest.mark.gpu
@pytest.mark.parametrize("kind", MATVEC_KINDS)
def test_DV_matches_dense(bse_dense_state, kind):
    """matvec_{include_W=False}(X) == (diag(D)+Kx) @ X — dense exchange locus (B1)."""
    harness.skip_unless_gpu(pytest)
    data = bse_dense_state
    _, D, Kx, _ = _build_dense_H(data)
    nc = int(data["psi_c"].shape[1]); nv = int(data["psi_v"].shape[1])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    X = _random_X(1, nc, nv, nk)
    hx = np.asarray(_run_matvec(kind, data, X, False))[0].reshape(-1)
    xf = np.asarray(X)[0].reshape(-1)
    ref = D * xf + Kx @ xf
    assert _relerr(hx, ref) < 1e-9, f"{kind}: D+V rel-err {_relerr(hx, ref):.2e}"


@pytest.mark.gpu
def test_spectrum_matches_dense(bse_dense_state):
    """Materialised serial matvec has the dense-H spectrum (B1).

    Design (d) asked for an *iterative* lowest-4 check, but both single-vector
    and block Lanczos are numerically fragile on this fixture: the q=0 head
    injection makes the V/W ISDF tiles O(1e5) (V_q0[0,0]≈2.3e5) and they
    near-cancel against D, so the Krylov solvers return ghost / below-λ_min
    Ritz values (a solver-conditioning issue orthogonal to B1 — see
    PHASE2_LOG.md).  We instead MATERIALISE the corrected serial matvec into an
    N×N matrix with ONE batched application to the identity basis and compare
    its full spectrum to the dense reference — a robust, solver-independent
    proof that the B1-corrected operator IS the dense Hamiltonian.
    """
    harness.skip_unless_gpu(pytest)
    data = bse_dense_state
    H, _, _, _ = _build_dense_H(data)
    nc = int(data["psi_c"].shape[1]); nv = int(data["psi_v"].shape[1])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    N = nc * nv * nk
    basis = jnp.asarray(np.eye(N, dtype=np.complex128).reshape(N, nc, nv, nk))
    # row i of the batched output is H·e_i = column i of H, so cols == Hᵀ.
    cols = np.asarray(_run_matvec("serial", data, basis, True)).reshape(N, N)
    Hmat = cols.T
    assert _relerr(Hmat, H) < 1e-9, f"materialised matvec ≠ H: {_relerr(Hmat, H):.2e}"
    ev_mat = np.sort(np.linalg.eigvalsh(0.5 * (Hmat + Hmat.conj().T)))
    ev_ref = np.sort(np.linalg.eigvalsh(0.5 * (H + H.conj().T)))
    assert np.allclose(ev_mat, ev_ref, atol=1e-8), \
        f"spectrum mismatch: max|Δ|={np.max(np.abs(ev_mat - ev_ref)):.2e}"


@pytest.mark.gpu
@pytest.mark.extra
def test_report_before_after_eigenvalues(bse_dense_state):
    """Diagnostic (``extra``): print the lowest-20 exciton eigenvalues of the
    k-diagonal-exchange Hamiltonian (pre-B1) vs the dense-exchange one (post-B1).

    Both are eigvalsh of the SAME reference builder — only the exchange kernel's
    off-diagonal k-blocks differ — so this isolates the physical eigenvalue
    shift the B1 fix produces on the gate fixture.  Deselected from the plain
    suite; run with ``-o addopts='' -s`` (or ``-m extra``).
    """
    harness.skip_unless_gpu(pytest)
    data = bse_dense_state
    H, D, Kx, Kd = _build_dense_H(data)
    nc = int(data["psi_c"].shape[1]); nv = int(data["psi_v"].shape[1])
    nk = int(data["nkx"] * data["nky"] * data["nkz"])
    # Pre-B1: zero the off-diagonal k-blocks of the exchange (k-diagonal only).
    K6 = Kx.reshape(nc, nv, nk, nc, nv, nk).copy()
    mask = np.eye(nk)
    K6 *= mask[None, None, :, None, None, :]
    Kx_kdiag = K6.reshape(Kx.shape)
    H_before = np.diag(D.astype(np.complex128)) + Kx_kdiag - Kd
    ev_before = np.sort(np.linalg.eigvalsh(0.5 * (H_before + H_before.conj().T)))[:20]
    ev_after = np.sort(np.linalg.eigvalsh(0.5 * (H + H.conj().T)))[:20]
    RY = 13.6056980659
    print("\n#  eig_before(Ry)  eig_after(Ry)   Δ(eV)   |  before(eV)  after(eV)")
    for i, (b, a) in enumerate(zip(ev_before, ev_after)):
        print(f"{i:2d}  {b:13.6f}  {a:13.6f}  {(a-b)*RY:+7.3f}  | "
              f"{b*RY:9.4f}  {a*RY:9.4f}")
