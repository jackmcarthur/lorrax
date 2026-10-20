"""Point-group covariance of the W densifier on an fcc toy (CPU).

48 sites, the orbit of one generic point under O_h folded into the home cell,
so every operation maps the set onto itself with NON-UNIFORM lattice shifts
(r_i M = r_pi(i) + L_i, L_i varying with i) — the folding a lattice-
coordinate box cannot follow.  The tile
W_ij(q) = sum_R e^{-2 pi i q.R} F(r_i + R, r_j) of a distance-only kernel F
obeys, exactly,

    W(q M^-T)_{pi(i) pi(j)} = e^{-2 pi i q'.L_i} W(q)_ij e^{2 pi i q'.L_j}.

Sampled on 4^3 and densified to 8^3, the per-pair embedding
(``bse_densify.pad_W_R_to_grid``) keeps that identity to rounding at every
off-grid q; the previous fftfreq box (Nyquist split) and one Wigner–Seitz set
shared by every pair break it.
"""
import itertools

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from bse.bse_densify import pad_W_R_to_grid  # noqa: E402
from common.ws_images import pair_ws_images  # noqa: E402

A = 10.26 * np.array([[0, .5, .5], [.5, 0, .5], [.5, .5, 0]])   # rows a_i, bohr
CG, FG, SIGMA = (4, 4, 4), (8, 8, 8), 6.0


def _ops():
    """O_h as integer matrices M on fractional rows (f -> f M)."""
    ops, Ainv = [], np.linalg.inv(A)
    for perm in itertools.permutations(range(3)):
        for sg in itertools.product((1, -1), repeat=3):
            Rc = np.zeros((3, 3))
            Rc[range(3), perm] = sg
            M = A @ Rc.T @ Ainv
            assert np.allclose(M, np.round(M))
            ops.append(np.round(M))
    return ops


OPS = _ops()
R_SITE = np.unique(np.round((np.array([0.11, 0.23, 0.37]) @ np.stack(OPS)) % 1.0, 12),
                   axis=0)


def _action(M):
    """Site permutation pi and lattice shifts L (r_i M = r_pi(i) + L_i)."""
    img = R_SITE @ M
    L = np.floor(img + 1e-9)
    key = {tuple(np.round(v, 9)): j for j, v in enumerate(R_SITE)}
    return np.array([key[tuple(np.round(v, 9) % 1.0)] for v in img - L]), L


def _tiles(q):
    """W_ij(q) by direct lattice sum, |T_a| <= 8 (F < 1e-13 beyond)."""
    T = np.stack(np.meshgrid(*[np.arange(-8, 9)] * 3, indexing='ij'), -1).reshape(-1, 3)
    out = np.zeros((len(q), 48, 48), complex)
    for j in range(48):
        d = (R_SITE[:, None, :] + T[None] - R_SITE[j]) @ A           # (48, nT, 3)
        F = np.exp(-np.einsum('itx,itx->it', d, d) / (2 * SIGMA ** 2))
        out[:, :, j] = np.exp(-2j * np.pi * q @ T.T) @ F.T
    return out


def _cov_err(tile, q):
    """max over O_h of |W(q')[pi, pi] - gauge W(q)| / max|W|, q' = q M^-T."""
    Wq, err = tile(q), 0.0
    for M in OPS:
        pi, L = _action(M)
        qp = q @ np.linalg.inv(M).T
        g = np.exp(-2j * np.pi * qp @ L.T)
        want = g[:, :, None] * Wq * np.conj(g)[:, None, :]
        err = max(err, float(np.abs(tile(qp)[:, pi][:, :, pi] - want).max()))
    return err / float(np.abs(Wq).max())


def _box_pad(XR):
    """The previous embedding: fftfreq box, Nyquist split +-n/2."""
    for ax, nf in zip((2, 3, 4), FG):
        nc, s = XR.shape[ax], (XR.shape[ax] + 1) // 2
        lo, hi = XR.take(range(s), ax), XR.take(range(s, nc), ax)
        half = 0.5 * XR.take([s], ax)
        lo = np.concatenate([lo, half], ax)
        hi = np.concatenate([half, XR.take(range(s + 1, nc), ax)], ax)
        z = list(XR.shape)
        z[ax] = nf - lo.shape[ax] - hi.shape[ax]
        XR = np.concatenate([lo, np.zeros(z, XR.dtype), hi], ax)
    return XR * np.sqrt(np.prod(FG) / np.prod(CG))


def test_toy_folds_non_uniformly_and_is_covariant():
    assert R_SITE.shape == (48, 3)
    assert sum(len({tuple(v) for v in _action(M)[1]}) > 1 for M in OPS) >= 24
    assert _cov_err(_tiles, np.random.default_rng(1).random((2, 3))) < 1e-11


def test_pair_embedding_is_covariant_and_box_and_shared_are_not():
    qc = np.stack(np.meshgrid(*[np.arange(n) / n for n in CG], indexing='ij'),
                  -1).reshape(-1, 3)
    XR = np.fft.ifftn(np.moveaxis(_tiles(qc), 0, -1).reshape((48, 48) + CG),
                      axes=(2, 3, 4), norm='ortho')
    fine_R = {
        'pair': pad_W_R_to_grid(jnp.asarray(XR), FG,
                                embedding=pair_ws_images(CG, A, R_SITE, 3)),
        'shared': pad_W_R_to_grid(jnp.asarray(XR), FG,
                                  embedding=pair_ws_images(CG, A, np.full((48, 3), 0.5), 3)),
        'box': _box_pad(XR)}
    q = np.random.default_rng(3).integers(0, 8, size=(4, 3)) / 8.0
    q[np.all(np.abs(q * 4 - np.round(q * 4)) < 1e-9, axis=1)] += 1.0 / 8.0    # off-grid
    err = {}
    for name, WR in fine_R.items():
        Wf = np.fft.fftn(np.asarray(WR), axes=(2, 3, 4), norm='ortho').reshape(48, 48, -1)

        def tile(qq, Wf=Wf):
            n = np.rint(qq * FG).astype(int) % np.asarray(FG)
            return np.moveaxis(Wf[:, :, (n[:, 0] * FG[1] + n[:, 1]) * FG[2] + n[:, 2]], -1, 0)
        err[name] = _cov_err(tile, q)
    assert err['pair'] < 1e-10, err
    assert err['box'] > 0.1 and err['shared'] > 0.1, err
