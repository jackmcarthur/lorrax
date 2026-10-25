#!/usr/bin/env python3
"""Linear response of a TD-aGW window: BSE modes, TD spectra, magneto-optics.

    python3 tools/td_response.py kerr --td-x X/tdgw.h5 --td-y Y/tdgw.h5 \\
        --eta-ev 0.02 --lz-bohr 30.0 --d-ang 6.6 [--eigenvectors eigenvectors.h5] --out kerr.csv

The one owner of the response formulas that the validation ladder
(``tests/tdagw/test_validation_*.py``) and the CrI3 analysis
(``tools/td_validation_cri3.py``) share.  Hartree atomic units, Gaussian
electromagnetic units (``c = 1/alpha``).

Linear response of the window (``docs/theory/td_agw_propagation.md#validation``).
With ``X_t = drho_k,cv`` and ``Y_t = drho_k,vc`` for every pair ``t = (k, c, v)``,
the TD-aGW equation of motion linearized about ``rho_eq = diag(f)`` is

    i dz/dt = L z + E_b(t) u_b,       z = (X, Y),
    L = [[D + K_XX, K_XY], [-K_YX, -D - K_YY]],   u_b = (r^b_cv, -r^b_vc),

with ``K_XY[t, t'] = K_{k c v, k' v' c'}`` etc. the blocks of the kernel
``dSigma_{k,nm} = sum K_{knm,k'n'm'} drho_{k',n'm'}`` (Ha): the full
(non-TDA) BSE.  The observable is ``d<r_a> = o_a . z``, ``o_a = w (r^a_vc, r^a_cv)``.
With ``L = Z diag(lam) Z^-1`` every mode S contributes the residue

    R^S_ab = (o_a . Z[:, S]) (Z^-1[S, :] . u_b),
    d<r_a>(t) = -i sum_S R^S_ab int_{-inf}^t exp(-i lam_S (t - t')) E_b(t') dt',

independent of how the eigenvectors are normalized.  For a positive mode
``R^S_ab = w D^S_a conj(D^S_b)`` with ``D^S = <0|r|S>`` (per unit cell), so the
campaign's helicity strengths are ``I+- = (R_xx + R_yy -+ 2 Im R_xy)/2``
(sigma+ = E0 (cos wt, sin wt, 0)).  ``P = -g_s d<r>/Omega`` and
``chi_ab(w) = -(g_s/Omega) sum_S R^S_ab / (w - lam_S)``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from typing import NamedTuple

import numpy as np
from scipy.special import wofz

#: Speed of light in atomic units (CODATA 2018 1/alpha, as scipy.constants).
C_AU = 137.035999084
BOHR_IN_ANG = 0.529177210903


# ---------------------------------------------------------------------------
# The window BSE of a TD-aGW kernel
# ---------------------------------------------------------------------------

class WindowModes(NamedTuple):
    """Modes of the linearized window.

    lam : (2N,) complex — eigenvalues of L (Ha), +-Omega pairs, ascending.
    R : (2N, 3, 3) complex — residues R^S_ab (bohr^2, per unit cell).
    Z, Zi : (2N, 2N) complex — right eigenvectors (columns) and Z^-1 (rows);
        a positive mode is scaled to z^H eta z = 1, eta = diag(1, -1) (the
        exciton number of the full BSE), so |Zi[S] . z|^2 counts excitons S.
    L : (2N, 2N) complex — the linearized operator.
    o, u : (3, 2N) complex — observable and drive vectors.
    n_pairs : N = nk * nc * nv.
    """
    lam: np.ndarray
    R: np.ndarray
    Z: np.ndarray
    Zi: np.ndarray
    L: np.ndarray
    o: np.ndarray
    u: np.ndarray
    n_pairs: int


def pair_index(nk: int, nb: int, n_val: int):
    """(k, c, v) of every pair t, c in [n_val, nb), v in [0, n_val), row-major."""
    k, c, v = np.meshgrid(np.arange(nk), np.arange(n_val, nb), np.arange(n_val),
                          indexing="ij")
    return k.ravel(), c.ravel(), v.ravel()


def window_operator(e_ha, n_val: int, K_ha, r, w_k):
    """L, o, u of the linearized TD-aGW window (module docstring).

    Parameters
    ----------
    e_ha : (nk, nb) float — band energies (Ha), valence rows first.
    n_val : int — occupied rows per k.
    K_ha : (nk, nb, nb, nk, nb, nb) complex or None — dSigma = K drho (Ha);
        None is the independent-particle window.
    r : (3, nk, nb, nb) complex — interband position (bohr).
    w_k : (nk,) float — k weights.

    Returns (L (2N, 2N), o (3, 2N), u (3, 2N)).
    """
    e = np.asarray(e_ha, float)
    nk, nb = e.shape
    k, c, v = pair_index(nk, nb, n_val)
    n = k.size
    d = e[k, c] - e[k, v]
    L = np.zeros((2 * n, 2 * n), complex)
    L[:n, :n] = np.diag(d)
    L[n:, n:] = -np.diag(d)
    if K_ha is not None:
        K = np.asarray(K_ha)
        L[:n, :n] += K[k[:, None], c[:, None], v[:, None], k[None], c[None], v[None]]
        L[:n, n:] += K[k[:, None], c[:, None], v[:, None], k[None], v[None], c[None]]
        L[n:, :n] -= K[k[:, None], v[:, None], c[:, None], k[None], c[None], v[None]]
        L[n:, n:] -= K[k[:, None], v[:, None], c[:, None], k[None], v[None], c[None]]
    r = np.asarray(r)
    w = np.asarray(w_k, float)[k]
    o = np.concatenate([w * r[:, k, v, c], w * r[:, k, c, v]], axis=1)
    u = np.concatenate([r[:, k, c, v], -r[:, k, v, c]], axis=1)
    return L, o, u


def window_modes(e_ha, n_val: int, K_ha, r, w_k) -> WindowModes:
    """Diagonalize L and return every mode's residues (module docstring)."""
    L, o, u = window_operator(e_ha, n_val, K_ha, r, w_k)
    lam, Z = np.linalg.eig(L)
    n = L.shape[0] // 2
    norm = np.einsum("is,is->s", np.conj(Z[:n]), Z[:n]) - np.einsum("is,is->s", np.conj(Z[n:]), Z[n:])
    scale = np.where(norm.real > 0, 1.0 / np.sqrt(np.abs(norm)), 1.0)
    Z = Z * scale[None, :]
    Zi = np.linalg.solve(Z, np.eye(Z.shape[0]))
    left = o @ Z                                 # (3, 2N): o_a . Z[:, S]
    right = Zi @ u.T                             # (2N, 3): Z^-1[S, :] . u_b
    R = left.T[:, :, None] * right[:, None, :]   # (2N, a, b)
    order = np.argsort(lam.real)
    return WindowModes(lam=lam[order], R=R[order], Z=Z[:, order], Zi=Zi[order], L=L,
                       o=o, u=u, n_pairs=n)


def sum_rules(modes: WindowModes):
    """Moments of the residues and their closed forms, (3, 3) each.

    m0 = sum_S R^S = o.u (zero on the diagonal: d<r>(0+) = 0),
    m1 = sum_S lam_S R^S = o.L.u (the window's f-sum / TRK value; for the
    independent-particle window 2 sum_t w dE_t |r_t|^2 on the diagonal),
    m_1 = sum_S R^S / lam_S = o.L^-1.u (the static polarizability, up to -g_s/Omega).
    """
    o, u, L = modes.o, modes.u, modes.L
    return dict(m0=o @ u.T, m1=o @ L @ u.T, m_inv=o @ np.linalg.solve(L, u.T))


def helicity(R):
    """I+, I-, eta of residues R (..., 3, 3): I+- = (R_xx + R_yy -+ 2 Im R_xy)/2."""
    R = np.asarray(R)
    base = (R[..., 0, 0] + R[..., 1, 1]).real
    im = R[..., 0, 1].imag
    ip, im_ = 0.5 * (base - 2 * im), 0.5 * (base + 2 * im)
    with np.errstate(invalid="ignore", divide="ignore"):
        eta = (ip - im_) / (ip + im_)
    return ip, im_, eta


def group_degenerate(lam, tol):
    """Labels grouping eigenvalues closer than ``tol`` (sorted input not required)."""
    lam = np.asarray(lam)
    order = np.argsort(lam.real)
    labels = np.empty(lam.size, int)
    g = 0
    for i, j in enumerate(order):
        if i and abs(lam[j] - lam[order[i - 1]]) > tol:
            g += 1
        labels[j] = g
    return labels


# ---------------------------------------------------------------------------
# Gaussian pulses: E_b(t) = exp(-((t - tc)/tau)^2) Re[a_b exp(-i w t)]
# ---------------------------------------------------------------------------

class Pulse(NamedTuple):
    """One Gaussian pulse in a.u.: E(t) = exp(-((t - tc)/tau)^2) Re[a exp(-i w t)].

    a : (3,) complex amplitude (a.u. of field); a = E0 (1, i, 0) is sigma+,
    E0 (1, -i, 0) sigma-, E0 (1, 0, 0) linear x.  w = 0 with real a is a kick.
    """
    a: np.ndarray
    w: float
    tc: float
    tau: float

    def field(self, t):
        t = np.asarray(t, float)
        g = np.exp(-((t - self.tc) / self.tau) ** 2)
        return (g[:, None] * (np.asarray(self.a)[None, :] * np.exp(-1j * self.w * t)[:, None])).real


def _gauss_primitive(om, x, tau):
    """int_{-inf}^{x} exp(i om s - s^2/tau^2) ds, stable for real om, any x (Faddeeva w)."""
    om = np.asarray(om, complex)
    x = np.asarray(x, float)
    pref = 0.5 * tau * np.sqrt(np.pi)
    with np.errstate(over="ignore", invalid="ignore"):
        lo = pref * np.exp(1j * om * x - (x / tau) ** 2) * wofz(-om * tau / 2 - 1j * x / tau)
        hi = (2 * pref * np.exp(-(om * tau) ** 2 / 4)
              - pref * np.exp(1j * om * x - (x / tau) ** 2) * wofz(om * tau / 2 + 1j * x / tau))
    return np.where(x <= 0, lo, hi)


def pulse_integral(lam, pulse: Pulse, t):
    """F_Sb(t) = int_{-inf}^t exp(i lam_S t') E_b(t') dt', (nt, 2N, 3)."""
    lam = np.asarray(lam, complex)[None, :]
    t = np.asarray(t, float)[:, None]
    x = t - pulse.tc
    a = np.asarray(pulse.a, complex)
    gm = np.exp(1j * (lam - pulse.w) * pulse.tc) * _gauss_primitive(lam - pulse.w, x, pulse.tau)
    gp = np.exp(1j * (lam + pulse.w) * pulse.tc) * _gauss_primitive(lam + pulse.w, x, pulse.tau)
    return 0.5 * (gm[..., None] * a[None, None, :] + gp[..., None] * np.conj(a)[None, None, :])


def pulse_spectrum(lam, pulse: Pulse):
    """F_Sb(+inf) = int exp(i lam_S t) E_b(t) dt, (2N, 3)."""
    lam = np.asarray(lam, complex)
    a = np.asarray(pulse.a, complex)
    s = pulse.tau * np.sqrt(np.pi)
    gm = np.exp(1j * (lam - pulse.w) * pulse.tc - ((lam - pulse.w) * pulse.tau) ** 2 / 4) * s
    gp = np.exp(1j * (lam + pulse.w) * pulse.tc - ((lam + pulse.w) * pulse.tau) ** 2 / 4) * s
    return 0.5 * (gm[:, None] * a[None, :] + gp[:, None] * np.conj(a)[None, :])


def modal_response(modes: WindowModes, pulse: Pulse, t):
    """Closed-form d<r_a>(t), (nt, 3) real, of the window driven by ``pulse``."""
    F = pulse_integral(modes.lam, pulse, t)                      # (nt, S, b)
    ph = np.exp(-1j * np.outer(np.asarray(t, float), modes.lam))  # (nt, S)
    y = -1j * np.einsum("sab,tsb,ts->ta", modes.R, F, ph)
    return y


def mode_amplitudes(modes: WindowModes, pulse: Pulse):
    """c_S = -i sum_b (Z^-1 u_b)_S F_Sb(inf): first-order amplitude of mode S after the pulse.

    z(t) = sum_S Z[:, S] c_S e^{-i lam_S t} once the pulse is over, so with the
    symplectic scaling of :func:`window_modes` |c_S|^2 is the number of
    excitons S (summed over the k grid; divide by nk per cell).
    """
    F = pulse_spectrum(modes.lam, pulse)
    return -1j * np.einsum("sb,sb->s", modes.Zi @ modes.u.T, F)


def project_state(modes: WindowModes, drho, n_val: int):
    """c_S(t) = Zi[S] . z(t) of TD density matrices drho (nt, nk, nb, nb), (nt, 2N)."""
    d = np.asarray(drho)
    nt, nk, nb = d.shape[0], d.shape[1], d.shape[2]
    k, c, v = pair_index(nk, nb, n_val)
    z = np.concatenate([d[:, k, c, v], d[:, k, v, c]], axis=1)
    return z @ modes.Zi.T


# ---------------------------------------------------------------------------
# Fitting TD signals
# ---------------------------------------------------------------------------

def fit_fixed_frequencies(t, y, lam, *, group_tol=0.0):
    """Least-squares amplitudes A_S of y(t) = sum_S A_S exp(-i lam_S t).

    t : (nt,) a.u. (after the drive); y : (nt,) or (nt, m) samples.
    Modes closer than ``group_tol`` (Ha) share one column (their sum is the
    only determined amplitude); the group amplitude is returned on every
    member divided equally, plus the labels.  Returns (A (S[, m]), labels, rel_residual).
    """
    t = np.asarray(t, float)
    lam = np.asarray(lam, complex)
    labels = group_degenerate(lam, group_tol) if group_tol > 0 else np.arange(lam.size)
    ng = labels.max() + 1
    rep = np.array([lam[labels == g].mean() for g in range(ng)])
    B = np.exp(-1j * np.outer(t, rep))
    yy = np.asarray(y, complex)
    sol, *_ = np.linalg.lstsq(B, yy, rcond=None)
    resid = np.linalg.norm(B @ sol - yy) / max(np.linalg.norm(yy), 1e-300)
    counts = np.bincount(labels, minlength=ng)
    A = sol[labels] / (counts[labels] if sol.ndim == 1 else counts[labels][:, None])
    return A, labels, float(resid)


def matrix_pencil(t, y, n_modes: int, *, pencil: float = 1.0 / 3.0):
    """Free-frequency exponentials of uniformly sampled y (Hua-Sarkar matrix pencil).

    y(t_j) = sum_m A_m exp(-i lam_m t_j), j = 0..n-1, dt uniform.  Returns
    (lam (n_modes,), A (n_modes,)) with A referred to t = 0, sorted by Re lam.
    """
    t = np.asarray(t, float)
    y = np.asarray(y, complex)
    dt = t[1] - t[0]
    if np.max(np.abs(np.diff(t) - dt)) > 1e-9 * abs(dt):
        raise ValueError("matrix_pencil: samples must be uniform")
    n = y.size
    Lp = int(n * pencil)
    H = np.lib.stride_tricks.sliding_window_view(y, Lp + 1)      # (n - Lp, Lp + 1)
    _, _, vh = np.linalg.svd(H, full_matrices=False)
    V = vh[:n_modes].conj().T                                    # (Lp + 1, M)
    V1, V2 = V[:-1], V[1:]
    z = np.linalg.eigvals(np.linalg.pinv(V1) @ V2)
    lam = 1j * np.log(z) / dt
    Bz = np.exp(-1j * np.outer(t - t[0], lam))
    A, *_ = np.linalg.lstsq(Bz, y, rcond=None)
    A = A * np.exp(1j * lam * t[0])
    order = np.argsort(lam.real)
    return lam[order], A[order]


# ---------------------------------------------------------------------------
# Susceptibility, dielectric tensor, conductivity
# ---------------------------------------------------------------------------

def chi_from_modes(modes: WindowModes, omega, eta, *, g_s: float, cell_volume: float):
    """chi_ab(w + i eta) = -(g_s/Omega) sum_S R^S_ab / (w + i eta - lam_S), (nw, 3, 3)."""
    z = np.asarray(omega, float)[:, None] + 1j * eta - modes.lam[None, :]
    return -(g_s / cell_volume) * np.einsum("sab,ws->wab", modes.R, 1.0 / z)


def chi_from_td(t, P, E, omega, eta):
    """chi_ab(w + i eta) from TD samples, exact in linear response.

    P(t) e^{-eta t} = int chi(t - t') e^{-eta (t - t')} E(t') e^{-eta t'} dt', so
    FT[P e^{-eta t}](w) = chi(w + i eta) FT[E e^{-eta t}](w).  ``P`` (nt, 3)
    for ONE drive direction b given by ``E`` (nt, 3) (only its b component
    nonzero, or any fixed polarization: the returned column is chi . e_hat).
    Trapezoid FT; t must start before the pulse and run until P e^{-eta t}
    is negligible.  Returns (nw, 3): chi_ab e_b along the drive.
    """
    t = np.asarray(t, float)
    damp = np.exp(-eta * (t - t[0]))
    ph = np.exp(1j * np.outer(np.asarray(omega, float), t))       # (nw, nt)
    wts = np.full(t.size, t[1] - t[0])
    wts[[0, -1]] *= 0.5
    Pw = ph @ (np.asarray(P) * (damp * wts)[:, None])
    Ew = ph @ (np.asarray(E) * (damp * wts)[:, None])
    e_amp = np.linalg.norm(np.asarray(E), axis=0)
    b = int(np.argmax(e_amp))
    return Pw / Ew[:, b][:, None]


def circular(chi):
    """chi_+- = chi_xx +- i chi_xy for the (x +- i y) field (sigma+ = (cos, sin))."""
    chi = np.asarray(chi)
    return chi[..., 0, 0] + 1j * chi[..., 0, 1], chi[..., 0, 0] - 1j * chi[..., 0, 1]


def circular_from_strengths(omega, eta, energies, i_plus, i_minus, *, g_s, cell_volume):
    """chi_+- (w + i eta) of an in-plane isotropic window from its exciton table.

    chi_+ = -(g_s/Omega) sum_S [I+(S)/(z - Omega_S) - I-(S)/(z + Omega_S)], chi_- with
    I+ <-> I-, z = w + i eta: a sigma+ field is absorbed through I+ and its
    counter-rotating part through I- (residues of the positive mode and of its
    -Omega partner).  Exact when chi_yy = chi_xx and chi_yx = -chi_xy (C3, C4, C6).
    ``energies`` (Ha), ``i_plus``, ``i_minus`` (bohr^2 per cell, campaign
    convention I+- = |D_x -+ i D_y|^2/2).  Returns (chi_+, chi_-), each (nw,).
    Also chi_xy = (chi_+ - chi_-)/(2i): the BSE eps_xy of an exciton table.
    """
    z = np.asarray(omega, float)[:, None] + 1j * eta
    om = np.asarray(energies, float)[None, :]
    ip, im = (np.asarray(x, float)[None, :] for x in (i_plus, i_minus))
    pref = -g_s / cell_volume
    chi_p = pref * np.sum(ip / (z - om) - im / (z + om), axis=1)
    chi_m = pref * np.sum(im / (z - om) - ip / (z + om), axis=1)
    return chi_p, chi_m


def sheet_conductivity(chi, omega, lz_bohr):
    """2D conductivity sigma^2D_ab(w) = -i w chi_ab(w + i eta) L_z (a.u., Gaussian).

    The broadened chi is formed at w + i eta and sigma at the real w, the
    Lorentzian-broadened BSE convention (eps(w) = 1 + 4 pi chi(w + i eta),
    sigma = -i w (eps - 1)/(4 pi)), so a sheet and a film of the same
    sigma^2D describe one monolayer and coincide as d -> 0.  (Continuing to
    -i (w + i eta) chi instead changes sigma by O(eta/w).)
    """
    w = np.asarray(omega, float)[:, None, None]
    return -1j * w * np.asarray(chi) * lz_bohr


def kerr_faraday_sheet(sigma2d):
    """Lab-frame complex Kerr/Faraday ratios of a free-standing sheet, normal incidence.

    Fields continuous across the sheet, the jump of H_t carries J = sigma E:
    T = (1 + (2 pi/c) sigma)^-1, R = -(2 pi/c) sigma T on the in-plane
    Jones vector.  For x-polarized incidence theta + i*epsilon ~= E_y/E_x of
    the outgoing beam; angles are lab-frame (from x toward y, counter-clockwise
    viewed from +z).  Returns (phi_K (nw,), phi_F (nw,)) complex, radians, exact
    ellipse parameters (not the small-angle ratio).
    """
    s = (2 * np.pi / C_AU) * np.asarray(sigma2d)[..., :2, :2]
    eye = np.eye(2)
    T = np.linalg.inv(eye + s)
    R = -s @ T
    return _ellipse(R[..., :, 0]), _ellipse(T[..., :, 0])


def kerr_faraday_film(eps, omega, d_bohr):
    """The same for a film of thickness d with in-plane dielectric tensor eps, Airy formulas.

    At normal incidence the in-plane field obeys E'' + (w/c)^2 eps E = 0, so the
    eigenpolarizations are the eigenvectors V of the 2 x 2 in-plane eps
    (circular, x +- i y with eps_+- = eps_xx +- i eps_xy, when eps_yy = eps_xx
    and eps_yx = -eps_xy).  Each propagates with n_j = sqrt(eps_j) (Im n >= 0):
    r_j = r01 (1 - e^{2i delta}) / (1 - r01^2 e^{2i delta}),
    t_j = t01 t12 e^{i delta} / (1 - r01^2 e^{2i delta}), r01 = (1 - n)/(1 + n),
    delta = n w d / c, and the lab Jones matrices are M = V diag(m_j) V^-1.
    """
    eps = np.asarray(eps)[..., :2, :2]
    w = np.asarray(omega, float)
    lam, V = np.linalg.eig(eps)                           # (nw, 2), (nw, 2, 2)
    n = np.sqrt(lam)
    n = np.where(n.imag < 0, -n, n)
    r01 = (1 - n) / (1 + n)
    e1 = np.exp(1j * n * (w[:, None] * d_bohr / C_AU))   # e^{i delta}
    den = 1 - r01 ** 2 * e1 ** 2
    r = r01 * (1 - e1 ** 2) / den
    vac = np.exp(1j * w * d_bohr / C_AU)[:, None]        # the vacuum phase of the thickness
    t = (2 / (1 + n)) * (2 * n / (n + 1)) * e1 / den / vac
    Vi = np.linalg.inv(V)
    R = V @ (r[..., :, None] * Vi)
    T = V @ (t[..., :, None] * Vi)
    return _ellipse(R[..., :, 0]), _ellipse(T[..., :, 0])


def _ellipse(J):
    """theta + i*epsilon of Jones vectors J (..., 2): orientation and ellipticity angles.

    E+- = (E_x -+ i E_y)/sqrt2 are the (x +- i y)/sqrt2 components;
    theta = (arg E- - arg E+)/2, tan(epsilon) = (|E+| - |E-|)/(|E+| + |E-|).
    """
    J = np.asarray(J)
    ep = (J[..., 0] - 1j * J[..., 1]) / np.sqrt(2)
    em = (J[..., 0] + 1j * J[..., 1]) / np.sqrt(2)
    theta = 0.5 * np.angle(em / ep)
    ell = np.arctan((np.abs(ep) - np.abs(em)) / (np.abs(ep) + np.abs(em)))
    return theta + 1j * ell


def film_eps(chi, lz_bohr, d_bohr):
    """eps of a film of thickness d from the supercell chi: eps - 1 = 4 pi chi L_z / d."""
    eye = np.eye(3)
    return eye + 4 * np.pi * np.asarray(chi) * (lz_bohr / d_bohr)


# ---------------------------------------------------------------------------
# CLI: TD x/y drives (and optionally the BSE states) -> chi, sigma_xy, Kerr, Faraday
# ---------------------------------------------------------------------------

def _read_td(path):
    import h5py
    with h5py.File(path, "r") as f:
        td = f["tdgw_data/time_domain"]
        t = td["polarization/times"][()]
        P = td["polarization/data"][()]
        E = td["Efield/data"][()]
        omega_cell = abs(np.linalg.det(f["tdgw_header/structure/rprim"][()]))
        lz = float(np.linalg.norm(f["tdgw_header/structure/rprim"][()][2]))
    return t, P.real, E, omega_cell, lz


def _parse(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("kerr", help="chi, sigma_xy, Kerr and Faraday from TD x and y drives")
    k.add_argument("--td-x", required=True)
    k.add_argument("--td-y", required=True)
    k.add_argument("--eta-ev", type=float, default=0.02)
    k.add_argument("--w-min-ev", type=float, default=0.5)
    k.add_argument("--w-max-ev", type=float, default=4.0)
    k.add_argument("--nw", type=int, default=1401)
    k.add_argument("--d-ang", type=float, default=6.6)
    k.add_argument("--out", required=True)
    return p.parse_args(argv)


def main(argv=None) -> int:
    from scipy.constants import physical_constants
    ha = physical_constants["Hartree energy in eV"][0]
    a = _parse(argv)
    tx, Px, Ex, vol, lz = _read_td(a.td_x)
    ty, Py, Ey, _, _ = _read_td(a.td_y)
    w = np.linspace(a.w_min_ev, a.w_max_ev, a.nw) / ha
    eta = a.eta_ev / ha
    chi = np.zeros((w.size, 3, 3), complex)
    chi[:, :, 0] = chi_from_td(tx, Px, Ex, w, eta)
    chi[:, :, 1] = chi_from_td(ty, Py, Ey, w, eta)
    sig = sheet_conductivity(chi, w, lz)
    kerr_s, far_s = kerr_faraday_sheet(sig)
    d = a.d_ang / BOHR_IN_ANG
    kerr_f, far_f = kerr_faraday_film(film_eps(chi, lz, d), w, d)
    with open(a.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["omega_ev", "eps_xx_re", "eps_xx_im", "eps_xy_re", "eps_xy_im",
                     "sigma2d_xy_re_e2_over_hbar", "sigma2d_xy_im_e2_over_hbar",
                     "kerr_rot_sheet_mrad", "kerr_ell_sheet_mrad", "kerr_rot_film_mrad",
                     "kerr_ell_film_mrad", "faraday_rot_sheet_mrad", "faraday_rot_film_mrad"])
        for i in range(w.size):
            e = np.eye(3) + 4 * np.pi * chi[i]
            wr.writerow([f"{w[i] * ha:.6f}", e[0, 0].real, e[0, 0].imag, e[0, 1].real,
                         e[0, 1].imag, sig[i, 0, 1].real, sig[i, 0, 1].imag,
                         1e3 * kerr_s[i].real, 1e3 * kerr_s[i].imag, 1e3 * kerr_f[i].real,
                         1e3 * kerr_f[i].imag, 1e3 * far_s[i].real, 1e3 * far_f[i].real])
    print(f"wrote {a.out}: {w.size} frequencies, supercell L_z {lz:.3f} bohr, film d {a.d_ang} A")
    return 0


if __name__ == "__main__":
    sys.exit(main())
