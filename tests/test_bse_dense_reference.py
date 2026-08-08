"""Dense-reference gate for the Q=0 BSE kernel (Phase-2, step 1).

Builds the explicit ⟨cvk|H|c'v'k'⟩ matrix from the SAME padded, head-injected
arrays the production matvecs consume (``bse_io._load_ring_subset``), then
checks each live matvec against it.

The settled physics (reports/bse_refactor_map_2026-07-15/archive/adjudication/
VERDICT.md): the Q=0 exchange is DENSE in (k,k′) —

    ⟨cvk|K^x|c'v'k'⟩ = (1/Nk) Σ_{μν} M_cvk(μ) V_q0(μν) conj(M_c'v'k'(ν))

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

    # Exchange — DENSE in (k,k′).  Kx[c,v,k, c',v',k'] = M V M†: the bra carries
    # the bare vertex, the ket the conjugate (transition density
    # <0|ρ̂|Ψ> = Σ A_cvk ψ_ck ψ*_vk).
    lhs = np.einsum("kcvM,MN->kcvN", M, V_q0)            # M·V
    Kx = np.einsum("kcvN,KCVN->cvkCVK", lhs, np.conj(M)) / nk

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


def _build_dense_nontda(data):
    """Return (A, B, H_shao) for the full (non-TDA) optical BSE.

    Extends ``_build_dense_H``: A = diag(D)+Kx−Kd (resonant, Hermitian); the
    coupling block B = Kx_B − Kd_B is the FIRST value-validation of the B-blocks
    WITH screened W.  Kx_B un-conjugates the ket pair density relative to A, so
    A = M V M† while B = M V M^T (Henneke 2-20); Kd_B
    swaps c'↔v' in the k' pair densities (the anti-resonant coupling).  The
    physical operator is the para-Hermitian SHAO ``H = [[A, B], [-B*, -A*]]``
    (A Hermitian, B complex-symmetric) — real spectrum, ±ω pairs."""
    psi_c = np.asarray(data["psi_c"]); psi_v = np.asarray(data["psi_v"])
    W_q = np.asarray(data["W_q"]); V_q0 = np.asarray(data["V_q0"])
    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    grid = (nkx, nky, nkz); nk = nkx * nky * nkz
    nc = psi_c.shape[1]; nv = psi_v.shape[1]; nmu = psi_c.shape[3]; N = nc * nv * nk
    H_A, _, _, _ = _build_dense_H(data)                     # A block = resonant H
    A = H_A
    M = np.einsum("kcsm,kvsm->kcvm", np.conj(psi_c), psi_v)
    lhs = np.einsum("kcvM,MN->kcvN", M, V_q0)
    Kx_B = np.einsum("kcvN,KCVN->cvkCVK", lhs, M) / nk             # bare ket
    Wflat = W_q.reshape(nmu, nmu, nk)
    Kd_B = np.zeros((nc, nv, nk, nc, nv, nk), dtype=np.complex128)
    for k in range(nk):
        for kp in range(nk):
            Wq = Wflat[:, :, _q_flat(k, kp, grid)]
            Pc_B = np.einsum("ctm,Vtm->cVm", np.conj(psi_c[k]), psi_v[kp])
            Pv_B = np.einsum("vsn,Csn->vCn", psi_v[k], np.conj(psi_c[kp]))
            Kd_B[:, :, k, :, :, kp] = np.einsum("cVm,mn,vCn->cvCV", Pc_B, Wq, Pv_B) / nk
    B = Kx_B.reshape(N, N) - Kd_B.reshape(N, N)
    H_shao = np.block([[A, B], [-B.conj(), -A.conj()]])
    return A, B, H_shao


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


# ---------------------------------------------------------------------------
# Non-TDA (full BSE) gates — first value validation of the B-blocks WITH W.
# ---------------------------------------------------------------------------
def _materialize_nontda_operator(data):
    """Materialise the full non-TDA operator O (2N x 2N) from the fixed matvec.

    O[:, (part, i)] = matvec([e_i in block ``part``]); columns 0..N-1 = resonant,
    N..2N-1 = anti-resonant.  Chunked so it fits any 1 GPU."""
    from jax.sharding import Mesh
    from bse.bse_ring_comm import build_bse_ring_matvec_full, make_bse_shardings
    from bse.bse_serial import compute_pair_amplitude

    nkx, nky, nkz = int(data["nkx"]), int(data["nky"]), int(data["nkz"])
    nk = nkx * nky * nkz
    nc = int(data["psi_c"].shape[1]); nv = int(data["psi_v"].shape[1]); N = nc * nv * nk
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    sh = make_bse_shardings(mesh)
    eye = np.eye(N, dtype=np.complex128).reshape(N, nc, nv, nk)
    O = np.empty((2 * N, 2 * N), dtype=np.complex128)          # rows=(part,flat), cols
    with mesh:
        pcx = jax.lax.with_sharding_constraint(data["psi_c"], sh.psi_x)
        pcy = jax.lax.with_sharding_constraint(data["psi_c"], sh.psi_y)
        pvx = jax.lax.with_sharding_constraint(data["psi_v"], sh.psi_x)
        pvy = jax.lax.with_sharding_constraint(data["psi_v"], sh.psi_y)
        Wqs = jax.lax.with_sharding_constraint(data["W_q"], sh.W)
        Vqs = jax.lax.with_sharding_constraint(data["V_q0"], sh.V)
        W_R = jnp.fft.ifftn(Wqs, axes=(2, 3, 4), norm="ortho")
        M_X = jax.lax.with_sharding_constraint(compute_pair_amplitude(pcx, pvx), sh.psi_x)
        M_Y = jax.lax.with_sharding_constraint(compute_pair_amplitude(pcy, pvy), sh.psi_y)
        args = (pcx, pcy, pvx, pvy, data["eps_c"], data["eps_v"], W_R, Vqs, M_X, M_Y)
        mv = build_bse_ring_matvec_full(mesh, nkx, nky, nkz, include_W=True, screening=False)
        chunk = 8
        for part in (0, 1):
            for c0 in range(0, N, chunk):
                blk = eye[c0:c0 + chunk]; z = np.zeros_like(blk)
                top, bot = (blk, z) if part == 0 else (z, blk)
                Xf = jax.lax.with_sharding_constraint(
                    jnp.asarray(np.stack([top, bot], axis=0)), sh.X_full)
                out = np.asarray(mv(Xf, *args))               # (2, b, nc, nv, nk)
                b = blk.shape[0]
                col = part * N + c0
                O[:N, col:col + b] = out[0].reshape(b, N).T   # top (X-block)
                O[N:, col:col + b] = out[1].reshape(b, N).T   # bottom (Y-block)
    return O, N


@pytest.mark.gpu
def test_nontda_matvec_matches_dense_shao(bse_dense_state):
    """The FIXED full matvec is the physical SHAO operator [[A,B],[-B*,-A*]].

    First value-validation of the coupling B-block WITH screened W: the
    materialised operator equals the analytic dense build, its (2,1)/(2,2) rows
    are -B*/-A* (not the historical -B/-A), and its spectrum is REAL.  The naive
    [[A,B],[-B,-A]] operator is shown COMPLEX — the bug this fix removes.
    """
    harness.skip_unless_gpu(pytest)
    data = bse_dense_state
    A, B, H_shao = _build_dense_nontda(data)
    O, N = _materialize_nontda_operator(data)
    A_mat, B_mat = O[:N, :N], O[:N, N:]
    # Block identities (B is the first-validated coupling-with-W block).
    assert _relerr(A_mat, A) < 1e-9, f"A block: {_relerr(A_mat, A):.2e}"
    assert _relerr(B_mat, B) < 1e-9, f"B block: {_relerr(B_mat, B):.2e}"
    assert _relerr(A, A.conj().T) < 1e-6, "A must be Hermitian"
    assert _relerr(B, B.T) < 1e-6, "B must be complex-symmetric (B = Bᵀ)"
    # Anti-resonant row is the SHAO fix (-B*, -A*), NOT the old (-B, -A).
    assert _relerr(O[N:, :N], -B.conj()) < 1e-9, "row (2,1) must be -B*"
    assert _relerr(O[N:, N:], -A.conj()) < 1e-9, "row (2,2) must be -A*"
    assert _relerr(O, H_shao) < 1e-9, f"full operator vs dense SHAO: {_relerr(O, H_shao):.2e}"
    # Physical spectrum is REAL; the naive [[A,B],[-B,-A]] is complex.
    ev = np.linalg.eigvals(O)
    assert np.max(np.abs(ev.imag)) < 1e-8, "SHAO spectrum must be real"
    ev_naive = np.linalg.eigvals(np.block([[A, B], [-B, -A]]))
    assert np.max(np.abs(ev_naive.imag)) > 1e-6, "naive [[A,B],[-B,-A]] should be complex (the bug)"


def _nontda_data_from_subset(data):
    """Build the ``solve_bse_nontda_sharded`` data contract (sharded ψ/ε/W/V +
    hoisted M_X/M_Y + pad counts) from the ``bse_dense_state`` subset — no second
    restart load (1×1 mesh, so no band padding)."""
    from jax.sharding import Mesh
    from bse.bse_ring_comm import make_bse_shardings
    from bse.bse_serial import compute_pair_amplitude
    mesh = Mesh(np.array(jax.devices()[:1]).reshape(1, 1), axis_names=("x", "y"))
    sh = make_bse_shardings(mesh)
    nc = int(data["psi_c"].shape[1]); nv = int(data["psi_v"].shape[1])
    with mesh:
        d = {
            "psi_c_X": jax.lax.with_sharding_constraint(data["psi_c"], sh.psi_x),
            "psi_c_Y": jax.lax.with_sharding_constraint(data["psi_c"], sh.psi_y),
            "psi_v_X": jax.lax.with_sharding_constraint(data["psi_v"], sh.psi_x),
            "psi_v_Y": jax.lax.with_sharding_constraint(data["psi_v"], sh.psi_y),
            "eps_c": jnp.asarray(data["eps_c"]), "eps_v": jnp.asarray(data["eps_v"]),
            # COPY, and it must be a real one.  ``solve_bse_nontda_sharded``
            # DONATES W_q (``bse_nontda.py``, ``donate_argnums=(0,)``, landed
            # 2026-08-08), and on this 1x1 mesh ``with_sharding_constraint`` is
            # a no-op that hands straight back the SESSION-scoped
            # ``bse_dense_state`` buffer -- so the donation deleted W_q for
            # every later consumer of that fixture, and the suite went red in
            # whatever order xdist happened to pick.  The fixture's own
            # docstring promises the session state is never mutated; this keeps
            # the promise here, where the borrow happens, instead of giving up
            # a measured production win to defend against a test.
            "W_q": jax.lax.with_sharding_constraint(
                jnp.array(data["W_q"]), sh.W),
            "V_q0": jax.lax.with_sharding_constraint(data["V_q0"], sh.V),
            "nkx": int(data["nkx"]), "nky": int(data["nky"]), "nkz": int(data["nkz"]),
            "n_cond_pad": nc, "n_val_pad": nv,
        }
        d["M_X"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(d["psi_c_X"], d["psi_v_X"]), sh.psi_x)
        d["M_Y"] = jax.lax.with_sharding_constraint(
            compute_pair_amplitude(d["psi_c_Y"], d["psi_v_Y"]), sh.psi_y)
    return d, mesh


@pytest.mark.gpu
def test_nontda_solver_reproduces_dense(bse_dense_state):
    """The structure-preserving non-TDA solver reproduces the dense SHAO positive
    eigenvalues and returns paired (X, Y) with X^H X − Y^H Y = +1."""
    harness.skip_unless_gpu(pytest)
    from bse.bse_nontda import solve_bse_nontda_sharded
    data = bse_dense_state
    _, _, H_shao = _build_dense_nontda(data)
    ev = np.linalg.eigvals(H_shao).real
    ref = np.sort(ev[ev > 1e-9])[:4]
    sdata, mesh = _nontda_data_from_subset(data)
    omega, evecs, _ = solve_bse_nontda_sharded(sdata, mesh, n_eig=4, include_W=True)
    omega = np.sort(np.asarray(jax.device_get(omega)))
    assert np.allclose(omega, ref, atol=1e-6), f"solver ω {omega} vs dense {ref}"
    evecs = np.asarray(jax.device_get(evecs))                 # (n_eig, 2, nc, nv, nk)
    for i in range(4):
        X = evecs[i, 0].reshape(-1); Y = evecs[i, 1].reshape(-1)
        snorm = float(np.real(np.conj(X) @ X - np.conj(Y) @ Y))
        assert abs(snorm - 1.0) < 1e-6, f"state {i}: X^HX−Y^HY = {snorm:.6f} ≠ 1"
