"""SR/LR W densification (``w_head_densify = srlr``) on a synthetic slab, CPU.

A toy ISDF slab: Gaussian centroid functions ζ_μ(q+G), the Ismail-Beigi kernel
on a small G set, a smooth χ(q) = −B(q)B(q)ᴴ.  Checks the three properties the
densifier rests on (``gw.lr_channels``, ``bse.bse_densify._srlr_terms``):

1. the Woodbury dressing equals the direct Schur split, W = J + F̃ᴴ D F̃;
2. the SR remainder J carries no |q| cusp where W does (the slab head);
3. subtracting the fitted MODEL, densifying and adding it back reproduces
   every coarse tile exactly.
"""
import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from ffi import _services  # noqa: E402

_services.ensure_on_path()
from vcoul import inscribed_radius, lr_channels, lr_kernel  # noqa: E402

from bse.bse_densify import build_pair_ws_embedding, pad_W_R_to_grid  # noqa: E402
from gw import lr_channels as LC  # noqa: E402

A_LAT, L_Z, SIGMA = 6.0, 20.0, 0.8
BVEC = 2 * np.pi * np.array([[1 / A_LAT, 1 / (np.sqrt(3) * A_LAT), 0.0],
                             [0.0, 2 / (np.sqrt(3) * A_LAT), 0.0],
                             [0.0, 0.0, 1 / L_Z]])
OMEGA = A_LAT ** 2 * np.sqrt(3) / 2 * L_Z
N_Z = 3
RNG = np.random.default_rng(7)
N_MU = 10
R_MU = (np.c_[RNG.random((N_MU, 2)), 0.08 * RNG.standard_normal(N_MU)]
        @ (2 * np.pi * np.linalg.inv(BVEC).T))
G_SET = np.array([[a, b, n] for a in range(-2, 3) for b in range(-2, 3)
                  for n in range(-N_Z, N_Z + 1)])
B_RAND = RNG.standard_normal((N_MU, 4)) + 1j * RNG.standard_normal((N_MU, 4))


def toy(q_frac):
    """V, χ, W and the channel rows F at one q (prototype normalisation)."""
    q_frac = np.asarray(q_frac, dtype=np.float64)
    K = (q_frac + G_SET) @ BVEC
    v = lr_kernel(q_frac[None], (G_SET)[None], bvec=BVEC, sys_dim=2)[0]
    Z = np.exp(-0.5 * SIGMA ** 2 * np.sum(K * K, 1))[:, None] * np.exp(-1j * K @ R_MU.T)
    Z /= np.sqrt(OMEGA)
    V = (Z.conj().T * v) @ Z
    qc = q_frac @ BVEC
    B = B_RAND * (1.0 + 0.3 * np.cos(R_MU @ qc))[:, None] * np.sqrt(1e-3)
    chi = -(B @ B.conj().T) * (np.sum(qc * qc) + 0.02)
    W = np.linalg.solve(np.eye(N_MU) - V @ chi, V)
    ch = lr_channels(q_frac[None], bvec=BVEC, sys_dim=2, n_z=N_Z)
    rows = [int(np.flatnonzero((G_SET == g).all(1))[0]) for g in ch.gvec[0]]
    return V, chi, W, Z[rows], ch.v[0]


def split(q_frac):
    V, chi, W, F, v = toy(q_frac)
    Ft, P = LC.dress(F[None], (F @ chi)[None], (chi @ F.conj().T)[None],
                     (F @ chi @ W)[None], v[None])
    return V, chi, W, F, v, Ft[0], P[0]


def test_woodbury_is_the_schur_split():
    for q in ([1 / 6, 0, 0], [1 / 3, 1 / 6, 0], [0.04, 0.02, 0]):
        V, chi, W, F, v, Ft, P = split(q)
        Vb = V - (F.conj().T * v) @ F
        J = np.linalg.solve(np.eye(N_MU) - Vb @ chi, Vb)
        Ft_d = np.linalg.solve((np.eye(N_MU) - chi @ Vb).T, F.T).T
        assert np.linalg.norm(Ft - Ft_d) <= 1e-10 * np.linalg.norm(Ft_d)
        assert np.linalg.norm(P - Ft_d @ chi @ F.conj().T) <= 1e-10 * np.linalg.norm(P)
        lr = LC.lr_term(Ft[None], P[None], v[None])[0]
        assert np.linalg.norm(J + lr - W) <= 1e-10 * np.linalg.norm(W)


def test_sr_remainder_has_no_slab_cusp():
    s = np.linspace(0.004, 0.04, 10)
    u = RNG.standard_normal(N_MU)
    coef = {}
    for name in ("W", "J"):
        y = []
        for si in s:
            q = si * np.array([1.0, 0.0, 0.0]) @ np.linalg.inv(BVEC)
            V, chi, W, F, v, Ft, P = split(q)
            J = W - LC.lr_term(Ft[None], P[None], v[None])[0]
            y.append(float(np.real(u @ (W if name == "W" else J) @ u)))
        c = np.polyfit(s, y, 2)[::-1]
        coef[name] = abs(c[1] / c[0])          # linear |q| coefficient / value
    assert coef["W"] > 1.0                      # the head cusp is there in W
    assert coef["J"] < 1e-3 * coef["W"]


def test_model_subtraction_keeps_coarse_tiles():
    cg, fg = (6, 6, 1), (12, 12, 1)
    qf = np.array([[i / cg[0], j / cg[1], 0] for i in range(cg[0]) for j in range(cg[1])])
    ws = lr_channels(qf, bvec=BVEC, sys_dim=2, n_z=N_Z).gvec[:, N_Z]
    qc = (qf + ws) @ BVEC
    data = [split(q + g) for q, g in zip(qf, ws)]
    W = np.stack([d[2] for d in data])
    Ft = np.stack([d[5] for d in data])
    P = np.stack([d[6] for d in data])
    v = np.stack([d[4] for d in data])
    q_ins = inscribed_radius(BVEC, 2)
    r_a, r_b = LC.WINDOW_INNER * q_ins, LC.WINDOW_OUTER * q_ins
    model, info = LC.fit_model(qc, Ft, P, r_mu=R_MU, S_cart=np.zeros((3, 3)),
                               c0=N_Z, sys_dim=2, r_fit=r_b)
    w = LC.window(np.linalg.norm(qc, axis=1), r_a, r_b)
    LR = LC.lr_term(LC.model_rows(model, qc), LC.model_P(model, qc), v) * w[:, None, None]
    LR[0] = LC.lr_term(Ft[:1], P[:1], v[:1])[0]           # exact at Γ
    X = (W - LR).reshape(cg + (N_MU, N_MU)).transpose(3, 4, 0, 1, 2)
    XR = np.fft.ifftn(X, axes=(2, 3, 4), norm="ortho")
    A = 2 * np.pi * np.linalg.inv(BVEC).T                 # rows a_i (bohr)
    emb = build_pair_ws_embedding(cg, fg, A, R_MU @ np.linalg.inv(A))
    Xf = np.fft.fftn(np.asarray(pad_W_R_to_grid(jnp.asarray(XR), fg,
                                                embedding=emb)),
                     axes=(2, 3, 4), norm="ortho")
    for i in range(cg[0]):
        for j in range(cg[1]):
            k = i * cg[1] + j
            Wf = Xf[:, :, 2 * i, 2 * j, 0] + LR[k]
            assert np.linalg.norm(Wf - W[k]) <= 1e-10 * np.linalg.norm(W[k])
    assert info["n_fit"] > 0 and np.count_nonzero(w) > 1


def test_wedge_unfold_of_the_channel_rows():
    """``_unfold_lr_channels`` on bcc Na 3³ (time-reversal rows included).

    Gaussian centroid functions give analytic ζ rows, ζ_μ(K) = e^{−σ²K²/2 − iK·r_μ};
    the rows stored on the q wedge, unfolded, must equal the rows built
    directly at each full-zone channel momentum, and P (a function of |K|)
    must come back unchanged.
    """
    import os
    import jax
    from jax.sharding import Mesh
    from symmetry_maps import QirrTables
    from bandstructure.fh_interp import setup_wfn_and_sym
    from bse.bse_densify import _unfold_lr_channels
    from file_io.centroids import load_centroids
    from gw.qgrid_symmetry import qgrid_trs_policy_for
    from gw.v_q_g_flat import _resolve_ibz_q_list
    from vcoul import CoulombGeometry

    fx = os.path.join(os.path.dirname(__file__), "hsuite", "fixture_na")
    wfn, sym = setup_wfn_and_sym(os.path.join(fx, "WFN.h5"), None)
    fft = tuple(int(x) for x in wfn.fft_grid)
    _, cidx, n_mu = load_centroids(os.path.join(fx, "centroids_frac_56.txt"), fft)
    grid = (3, 3, 3)
    _, q_irr, irr, rows, perm, wraps, reduced = _resolve_ibz_q_list(
        sym=sym, centroid_indices=cidx, kgrid=grid, fft_grid=fft, context="test")
    assert reduced
    n_sp = int(np.asarray(perm).shape[0]) // 2
    policy = qgrid_trs_policy_for(sym=sym, irr_idx_q=irr, sym_idx_q=rows,
                                  kgrid=grid, n_sym_spatial=n_sp, context="test")
    tables = QirrTables(irr_idx_q=np.asarray(irr), sym_idx_q=np.asarray(policy.unfold_sym_idx),
                        q_irr_frac=np.asarray(q_irr), sym_perm=np.asarray(perm),
                        L_table=np.asarray(wraps), n_sym_spatial=n_sp)
    bvec = np.asarray(CoulombGeometry.from_wfn(wfn).bvec)
    r_mu = (np.asarray(cidx, dtype=np.float64) / np.asarray(fft)) @ (2 * np.pi * np.linalg.inv(bvec).T)

    def rows_at(K_frac):                                   # (n, n_L, 3) → (n, n_L, N)
        K = np.asarray(K_frac) @ bvec
        return (np.exp(-0.5 * 0.3 ** 2 * np.sum(K * K, -1))[..., None]
                * np.exp(-1j * K @ r_mu.T))

    ch = lr_channels(np.asarray(q_irr), bvec=bvec, sys_dim=3)
    K_irr = np.asarray(q_irr)[:, None, :] + ch.gvec
    P_irr = -np.sum((K_irr @ bvec) ** 2, -1)[..., None]      # (n_qi, 1, 1), a function of |K|
    lr = dict(Ft=rows_at(K_irr), P=P_irr.astype(np.complex128), gvec=ch.gvec,
              v=ch.v, sys_dim=3, n_z=0, tables=tables)
    mesh = Mesh(np.array(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))
    Ft, Pf, vf, K_full, q_full = _unfold_lr_channels(
        lr, sym, coarse_grid=grid, n_rmu=n_mu, n_rmu_pad=n_mu, mesh_xy=mesh)
    G = K_full - q_full[:, None, :]
    assert np.allclose(G, np.rint(G))                       # each row lands on its own q
    want = rows_at(K_full)
    assert np.linalg.norm(Ft - want) <= 1e-10 * np.linalg.norm(want)
    naive = lr["Ft"][np.asarray(irr)]                      # parent rows, no symmetry action
    assert np.linalg.norm(naive - want) > 1e-2 * np.linalg.norm(want)
    assert np.allclose(Pf[:, 0, 0], -np.sum((K_full[:, 0] @ bvec) ** 2, -1), atol=1e-12)
    K2 = np.sum((K_full[:, 0] @ bvec) ** 2, -1)
    assert np.allclose(vf[:, 0], np.where(K2 > 1e-12, 8 * np.pi / np.maximum(K2, 1e-12), 0.0))
