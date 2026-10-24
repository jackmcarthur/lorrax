"""The srlr long-range P fit keeps the q-direction dependence (``gw.lr_channels.fit_model``).

A synthetic P(q) = −Σ_k u_k u_kᴴ built from a known anisotropic head tensor S
(P_00 → q·S·q) on a bulk orthorhombic grid and a rectangular slab with five
G_z channels.  The fit must return S as a tensor (not its mean diagonal), and
reproduce P at off-grid q; on a radial P (cubic bulk, hexagonal slab) it must
stay the radial answer.  Pure NumPy.
"""
import numpy as np
import pytest

from gw import lr_channels as LC


def _ws_q(bvec, ngrid, dim):
    f = np.stack(np.meshgrid(*[np.arange(n) / n for n in ngrid], indexing="ij"),
                 -1).reshape(-1, 3)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
                       for k in ((-1, 0, 1) if dim == 3 else (0,))])
    cand = (f[:, None, :] + shifts[None]) @ bvec
    q = cand[np.arange(len(f)), np.argmin(np.linalg.norm(cand, axis=2), axis=1)]
    if dim == 2:
        q[:, 2] = 0.0
    return q


def _synth_P(q, S, n_L, c0, *, radial):
    rng = np.random.default_rng(7)
    dim = S.shape[0]
    L = np.linalg.cholesky(-S)
    f = 0.4 * rng.standard_normal((dim, dim))
    g = rng.standard_normal((dim, n_L)) + 1j * rng.standard_normal((dim, n_L))
    h = rng.standard_normal((dim, n_L, dim)) + 1j * rng.standard_normal((dim, n_L, dim))
    qk = q[:, :dim]
    q2 = np.sum(qk * qk, axis=1)
    lead = np.arange(n_L) == c0
    P = np.zeros((len(q), n_L, n_L), complex)
    for k in range(dim):
        if radial:
            u = (qk @ L[:, k] * (1 + 0.3 * q2 - 0.2 * q2 ** 2))[:, None] \
                * np.where(lead, 1.0, 0.4 * g[0])[None]
            V = np.where(lead, 0.0, 0.3 * g[k])[None] * (1 + 0.5 * q2)[:, None]
            P -= V[:, :, None] * np.conj(V[:, None, :])
        else:
            u = 0.3 * g[k][None] * (1 + 0.5 * q2)[:, None] + 0.3j * np.einsum("ad,nd->na", h[k], qk)
            u[:, c0] = (qk @ L[:, k]) * (1 + 0.3 * q2 + (qk @ f[k]) ** 2)
        P -= u[:, :, None] * np.conj(u[:, None, :])
    return P


def _fit(bvec, ngrid, fine, S, n_L, c0, *, radial=False, S_anchor=None):
    dim = S.shape[0]
    qc = _ws_q(bvec, ngrid, dim)
    r_fit = 0.9 * 0.5 * min(np.linalg.norm(bvec[i]) for i in range(dim))
    S3 = np.zeros((3, 3))
    S3[:dim, :dim] = S if S_anchor is None else S_anchor
    Ft = np.ones((len(qc), n_L, 2), complex)
    model, info = LC.fit_model(qc, Ft, _synth_P(qc, S, n_L, c0, radial=radial),
                               r_mu=np.zeros((2, 3)), S_cart=S3, c0=c0,
                               sys_dim=dim, r_fit=r_fit)
    qf = _ws_q(bvec, fine, dim)
    qn = np.linalg.norm(qf, axis=1)
    qf = qf[(qn > 1e-9) & (qn < r_fit / 3)]
    Pt = _synth_P(qf, S, n_L, c0, radial=radial)
    Pm = LC.model_P(model, qf)
    head = lambda P: P[:, c0, c0].real / np.sum(qf[:, :dim] ** 2, 1)   # noqa: E731
    return info, qf, np.linalg.norm(Pm - Pt) / np.linalg.norm(Pt), \
        np.max(np.abs(head(Pm) - head(Pt))) / np.max(np.abs(head(Pt)))


S_BULK = np.array([[-0.30, 0.03, 0.0], [0.03, -0.18, 0.01], [0.0, 0.01, -0.06]])
B_ORTHO = np.diag(2 * np.pi / np.array([5.0, 6.5, 9.0]))
S_SLAB = np.array([[-0.25, 0.02], [0.02, -0.12]])
B_RECT = np.diag([2 * np.pi / 6.2, 2 * np.pi / 8.4, 2 * np.pi / 30.0])


@pytest.mark.parametrize("anchor", ["S", "mean"])
def test_bulk_head_is_q_S_q(anchor):
    # The radial fit this replaced gave S̄ = −0.18 in every direction (head
    # error 34 % near Γ); the direction-resolved fit returns the tensor even
    # when anchored to S̄·1, because the anchor is soft.
    S_anchor = None if anchor == "S" else np.eye(3) * np.trace(S_BULK) / 3
    info, _, err_P, err_head = _fit(B_ORTHO, (6, 6, 6), (11, 11, 11), S_BULK, 1, 0,
                                    S_anchor=S_anchor)
    assert np.linalg.norm(info["S_fit"] - S_BULK) / np.linalg.norm(S_BULK) < 0.03
    assert err_head < 0.03 and err_P < 0.03


def test_slab_head_wings_and_gz_block():
    info, _, err_P, err_head = _fit(B_RECT, (9, 9, 1), (15, 15, 1), S_SLAB, 5, 2)
    assert np.allclose(info["S_fit"], S_SLAB, atol=1e-4)
    assert err_P < 1e-4 and err_head < 1e-4


@pytest.mark.parametrize("case", ["bulk_cubic", "slab_hex"])
def test_radial_P_stays_radial(case):
    if case == "bulk_cubic":
        args = (np.eye(3) * 2 * np.pi / 6.0, (6, 6, 6), (11, 11, 11), -0.2 * np.eye(3), 1, 0)
    else:
        a = 6.0
        b = 2 * np.pi / a * np.array([[1, -1 / np.sqrt(3), 0], [0, 2 / np.sqrt(3), 0],
                                      [0, 0, a / 30.0]])
        args = (b, (9, 9, 1), (12, 12, 1), -0.24 * np.eye(2), 7, 3)
    info, _, err_P, err_head = _fit(*args, radial=True)
    S = args[3]
    assert np.allclose(info["S_fit"], S, atol=1e-5)
    assert err_P < 1e-5 and err_head < 1e-5
