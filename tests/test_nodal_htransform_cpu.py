"""Nodal htransform: covariance, on-grid exactness, the WS code form, the metric (CPU).

A face-centred-cubic toy with one general orbit of 48 sites (an orbit-closed
node set that the home cell folds NON-uniformly) and a distance-only hopping
model.  The node-pair kernel ``fH`` of the lowest bands, continued off the
4^3 grid with per-pair Wigner-Seitz images and diagonalized directly, gives
star-invariant energies to rounding; a lattice-box R set (the former
htransform R grid) does not.  On the coarse grid it reproduces f(eps)
exactly.  The orbit-tied quadrature metric and the Loewdin step make
sampled nodal values exactly orthonormal.

Covariance is exact only when the window top lies in a gap at every coarse k
(the precondition ``common.band_degeneracy.check_band_window`` reports): a
window that cuts a degenerate multiplet makes the coarse kernels
gauge-dependent.  Here 15 bands end in a 0.039 Ry gap; 14 would cut one.
"""
import itertools

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from bandstructure.fh_interp import f_transform_eigs, newton_inv  # noqa: E402
from common.ws_images import (pair_ws_code_fourier, pair_ws_code_table,  # noqa: E402
                              pair_ws_fourier, pair_ws_images)
from isdf.galerkin import _loewdin, _quadrature_metric  # noqa: E402

A = 0.5 * np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]) * 10.0
KGRID = (4, 4, 4)
N_WINDOW, N_GUARD = 10, 5


def _oh():
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            S = np.zeros((3, 3))
            S[np.arange(3), perm] = signs
            ops.append(S)
    return ops


def _orbit(x0_cart):
    """Distinct images of one Cartesian point, folded into [0,1) crystal."""
    inv = np.linalg.inv(A)
    pts = []
    for S in _oh():
        f = np.mod((S @ x0_cart) @ inv, 1.0)
        f[np.isclose(f, 1.0)] = 0.0
        if not any(np.allclose(f, p) for p in pts):
            pts.append(f)
    return np.array(pts)


SITES = _orbit(np.array([0.61, 0.23, 1.37]))


def _hamiltonian(k_frac):
    """``H(k)_ij = sum_R e^{-2 pi i k.R} t(|r_i + R - r_j|)`` (Ry)."""
    R = np.array(list(itertools.product(range(-3, 4), repeat=3)), float)
    d = SITES[:, None, None, :] + R[None, :, None, :] - SITES[None, None, :, :]
    dist = np.linalg.norm(d @ A, axis=-1)
    t = -0.4 * np.exp(-(dist / 2.2) ** 2) * (dist < 9.0)
    ph = np.exp(-2j * np.pi * (np.atleast_2d(k_frac) @ R.T))
    H = np.einsum('qr,irj->qij', ph, t)
    return 0.5 * (H + np.conj(np.swapaxes(H, -1, -2)))


def _coarse_k():
    return np.stack(np.meshgrid(*[np.arange(n) / n for n in KGRID],
                                indexing='ij'), -1).reshape(-1, 3)


def _kernel():
    """Coarse fH kernel of orthonormal nodal coefficients, class-major R."""
    E, C = np.linalg.eigh(_hamiltonian(_coarse_k()))
    nb = N_WINDOW + N_GUARD
    E, C = E[:, :nb], C[:, :, :nb]                  # C_k[nu, n]
    f_eps, a, n, shift = f_transform_eigs(E.T)
    f = np.asarray(f_eps).T
    X_k = np.einsum('kin,kn,kjn->kij', C, f, np.conj(C))
    X = np.fft.ifftn(X_k.reshape(KGRID + X_k.shape[1:]), axes=(0, 1, 2))
    return X.reshape((-1,) + X_k.shape[1:]), (a, n, shift), E, nb


def _ws_tables():
    images = pair_ws_images(KGRID, A, SITES, 3)
    return images, pair_ws_code_table(images)


def _box(X_R, k):
    axes = [np.where(np.arange(m) >= (m + 1) // 2, np.arange(m) - m,
                     np.arange(m)) for m in KGRID]
    R = np.stack(np.meshgrid(*axes, indexing='ij'), -1).reshape(-1, 3)
    return np.einsum('qc,cij->qij', np.exp(-2j * np.pi * k @ R.T), X_R)


def _energies(fH_R, k, params, nb, fourier):
    X = np.asarray(fourier(fH_R, k))
    X = 0.5 * (X + np.conj(np.swapaxes(X, -1, -2)))
    values = np.linalg.eigvalsh(X)[:, :nb]          # f <= 0 ascending
    E, _res = newton_inv(*params, values)
    return np.asarray(E)[:, :N_WINDOW]


def _star_spread(fourier):
    fH_R, params, _, nb = _kernel()
    rng = np.random.default_rng(3)
    k0 = rng.random((6, 3))
    inv = np.linalg.inv(A)
    E0 = _energies(fH_R, k0, params, nb, fourier)
    worst = 0.0
    for S in _oh():
        M = A @ S.T @ inv                            # R_frac -> R_frac @ M
        kS = k0 @ np.linalg.inv(M).T
        ES = _energies(fH_R, kS, params, nb, fourier)
        worst = max(worst, float(np.abs(ES - E0).max()))
    return worst


def test_node_set_folds_non_uniformly():
    inv = np.linalg.inv(A)
    wraps = {tuple(np.floor((S @ (SITES[0] @ A)) @ inv + 1e-9).astype(int))
             for S in _oh()}
    assert SITES.shape == (48, 3) and len(wraps) > 1


def test_code_fourier_equals_pair_ws_fourier():
    images, table = _ws_tables()
    rng = np.random.default_rng(0)
    n = SITES.shape[0]
    X = rng.standard_normal((64, n, n)) + 1j * rng.standard_normal((64, n, n))
    k = rng.random((5, 3))
    ref = pair_ws_fourier(np.moveaxis(X, 0, -1).reshape((n, n) + KGRID),
                          images, k)
    got = pair_ws_code_fourier(X, table, k)
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref), atol=1e-12)


def test_direct_eigh_is_exact_on_the_coarse_grid():
    fH_R, params, E, nb = _kernel()
    _, table = _ws_tables()
    got = _energies(fH_R, _coarse_k(), params, nb,
                    lambda X, k: pair_ws_code_fourier(X, table, k))
    np.testing.assert_allclose(got, E[:, :N_WINDOW], atol=1e-10)


def test_ws_kernel_is_star_covariant_and_the_box_is_not():
    _, table = _ws_tables()
    ws = _star_spread(lambda X, k: pair_ws_code_fourier(X, table, k))
    box = _star_spread(_box)
    assert ws < 1e-10, ws                            # Ry: rounding
    assert box > 1e-4, box                           # the control fails


def test_quadrature_metric_and_loewdin_orthonormalize_sampled_values():
    """Phi = U / sqrt(v) with v constant on each orbit: NNLS returns w = v,
    the Gram is the identity, and C = W^{1/2} Phi G^{-1/2} is orthonormal."""
    rng = np.random.default_rng(11)
    nk, nb, n_mu = 9, 5, 12
    labels = np.repeat(np.arange(3), 4)
    v = np.array([0.5, 2.0, 1.3])[labels]
    U = np.linalg.qr(rng.standard_normal((nk, 2 * n_mu, nb))
                     + 1j * rng.standard_normal((nk, 2 * n_mu, nb)))[0]
    phi = np.moveaxis(U.reshape(nk, 2, n_mu, nb), -1, 1) / np.sqrt(v)
    w, gram = _quadrature_metric(phi, labels)        # phi (nk, nb, ns, n_mu)
    np.testing.assert_allclose(w, v, rtol=1e-10)
    inv_half, eps = _loewdin(gram)
    assert eps < 1e-10
    C = np.einsum('kmn,kmsv,v->knsv', inv_half, phi, np.sqrt(w)).reshape(
        nk, nb, -1)
    np.testing.assert_allclose(np.einsum('kna,kma->knm', np.conj(C), C),
                               np.broadcast_to(np.eye(nb), (nk, nb, nb)),
                               atol=1e-12)
