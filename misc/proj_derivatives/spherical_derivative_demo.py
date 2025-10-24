#!/usr/bin/env python3
"""
Validate analytic gradient of a spherical function f(q) = A_l(q) Y_{lm}(q̂)
against numerical finite differences on the unit sphere at fixed radius q.

We implement the stable split:
  ∇_q f = A'(q) Y_{lm} e_r + (A(q)/q) ∇_Ω Y_{lm}
and compare Cartesian components to central differences of f on small angular
displacements at fixed q.

This is a self-contained demo with no redundancy, structured to be adapted into
isdf.psp.get_dipole_mtxels for computing g_α(q) = ∇β_α(q).
"""

import numpy as np
from dataclasses import dataclass

try:
    import matplotlib.pyplot as plt
    _HAS_PLOT = True
except Exception:
    _HAS_PLOT = False


# --------------------------
# Spherical frames and harmonics (minimal, consistent with QE conventions)
# --------------------------

def cart_unit_from_angles(theta: float, phi: float) -> np.ndarray:
    st = np.sin(theta)
    ct = np.cos(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    # e_r, e_theta, e_phi in Cartesian
    e_r = np.array([st * cp, st * sp, ct], dtype=float)
    e_t = np.array([ct * cp, ct * sp, -st], dtype=float)
    e_p = np.array([-sp, cp, 0.0], dtype=float)
    return e_r, e_t, e_p


def angles_from_cart(v: np.ndarray) -> tuple[float, float, float]:
    q = float(np.linalg.norm(v))
    if q == 0.0:
        return 0.0, 0.0, 0.0
    x, y, z = (v / q)
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return q, theta, phi


def sph_harm_complex(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    # Use scipy if available, else fallback to numpy for a few small l via recursion
    try:
        from scipy.special import sph_harm as _Y
        return _Y(m, l, phi, theta)
    except Exception:
        # Very small fallback: only l=0,1 exact
        if l == 0 and m == 0:
            return np.full_like(theta, 0.5 / np.sqrt(np.pi), dtype=complex)
        if l == 1:
            # Condon-Shortley phase; normalized to scipy
            c = np.sqrt(3/(8*np.pi))
            if m == -1:
                return c * np.exp(-1j*phi) * np.sin(theta)
            if m == 0:
                return np.sqrt(3/(4*np.pi)) * np.cos(theta)
            if m == 1:
                return -c * np.exp(1j*phi) * np.sin(theta)
        raise RuntimeError("Fallback sph_harm only supports l<=1")


def grad_angular_Y_complex(l: int, m: int, theta: float, phi: float) -> np.ndarray:
    # Returns Cartesian 3-vector for ∇_Ω Y_{lm} = e_theta ∂_θ Y + e_phi (1/ sinθ) ∂_φ Y
    # Use numerical d/dθ on P_l^m via scipy derivative rules if available
    th = np.asarray([theta], dtype=float)
    ph = np.asarray([phi], dtype=float)
    Y = sph_harm_complex(l, m, th, ph)[0]
    # Angular partials (complex); use small-step central differences for robustness
    h = 1e-6
    Y_t1 = sph_harm_complex(l, m, th + h, ph)[0]
    Y_t0 = sph_harm_complex(l, m, th - h, ph)[0]
    dth = (Y_t1 - Y_t0) / (2*h)
    Y_p1 = sph_harm_complex(l, m, th, ph + h)[0]
    Y_p0 = sph_harm_complex(l, m, th, ph - h)[0]
    dph = (Y_p1 - Y_p0) / (2*h)
    e_r, e_t, e_p = cart_unit_from_angles(theta, phi)
    st = max(np.sin(theta), 1e-12)
    return (dth * e_t) + (dph / st) * e_p


# --------------------------
# Radial model: hydrogenic 1s or 2p (analytic A, A')
# --------------------------

@dataclass
class RadialModel:
    l: int
    a0: float = 1.0  # Bohr radius scale in reciprocal units for demo

    def A(self, q: float) -> float:
        # Simple smooth radial envelope: A(q) = (q a0)^l exp(-(q a0)^2)
        x = q * self.a0
        return (x ** self.l) * np.exp(-(x ** 2))

    def Aprime(self, q: float) -> float:
        x = q * self.a0
        if q == 0.0:
            # l>0: derivative tends to 0; l=0: derivative 0 at q=0 for this model
            return 0.0
        # d/dq [x^l e^{-x^2}] = a0 [ l x^{l-1} e^{-x^2} - 2 x^{l+1} e^{-x^2} ]
        return self.a0 * (self.l * (x ** (self.l - 1)) - 2 * (x ** (self.l + 1))) * np.exp(-(x ** 2))


# --------------------------
# Gradient assembly and test harness
# --------------------------

def grad_spherical_function(qvec: np.ndarray, l: int, m: int, radial: RadialModel) -> np.ndarray:
    q, theta, phi = angles_from_cart(qvec)
    if q == 0.0:
        return np.zeros(3, dtype=complex)
    e_r, e_t, e_p = cart_unit_from_angles(theta, phi)
    Y = sph_harm_complex(l, m, np.array([theta]), np.array([phi]))[0]
    A = radial.A(q)
    Ap = radial.Aprime(q)
    g_ang = grad_angular_Y_complex(l, m, theta, phi)
    return (Ap * Y) * e_r + (A / q) * g_ang


def finite_diff_grad(qvec: np.ndarray, l: int, m: int, radial: RadialModel) -> np.ndarray:
    # Central difference with small displacements in Cartesian
    h = 1e-5
    base = spherical_function(qvec, l, m, radial)
    g = np.zeros(3, dtype=complex)
    for i in range(3):
        dp = np.zeros(3); dp[i] = h
        dm = -dp
        fp = spherical_function(qvec + dp, l, m, radial)
        fm = spherical_function(qvec + dm, l, m, radial)
        g[i] = (fp - fm) / (2*h)
    return g


def spherical_function(qvec: np.ndarray, l: int, m: int, radial: RadialModel) -> complex:
    q, theta, phi = angles_from_cart(qvec)
    if q == 0.0:
        return 0.0
    Y = sph_harm_complex(l, m, np.array([theta]), np.array([phi]))[0]
    return radial.A(q) * Y


def _supports_lm(l: int, m: int) -> bool:
    try:
        _ = sph_harm_complex(l, m, np.array([1.0]), np.array([0.5]))
        return True
    except Exception:
        return False


def main():
    # Fixed radius and meridian for overlays
    q = 1.0
    phi = 0.3
    n = 361
    thetas = np.linspace(1e-3, np.pi - 1e-3, n)
    qvecs = np.stack([
        q * np.sin(thetas) * np.cos(phi),
        q * np.sin(thetas) * np.sin(phi),
        q * np.cos(thetas),
    ], axis=1)

    # Overlay a few (l,m). If SciPy is missing, l<=1 are supported by fallback
    lm_list = [(0, 0), (1, 0), (2, 1)]

    overlays = []
    for l, m in lm_list:
        if not _supports_lm(l, m):
            print(f"Skipping (l,m)=({l},{m}) due to missing sph_harm support")
            continue
        radial = RadialModel(l=l, a0=0.8)
        g_analytic = np.array([grad_spherical_function(v, l, m, radial) for v in qvecs])
        g_numeric = np.array([finite_diff_grad(v, l, m, radial) for v in qvecs])
        err = np.linalg.norm(g_analytic - g_numeric, axis=1)
        overlays.append(((l, m), g_analytic, g_numeric, err))

    if overlays:
        # Plot gx overlays and errors
        if _HAS_PLOT:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
            for (l, m), ga, gn, err in overlays:
                ax[0].plot(thetas, np.real(ga[:, 0]), label=f'(l,m)=({l},{m}) analytic')
                ax[0].plot(thetas, np.real(gn[:, 0]), '--', label=f'(l,m)=({l},{m}) numeric')
            ax[0].set_ylabel('g_x'); ax[0].legend(ncol=2, fontsize=8)
            for (l, m), ga, gn, err in overlays:
                ax[1].plot(thetas, err, label=f'(l,m)=({l},{m})')
            ax[1].set_xlabel('theta'); ax[1].set_ylabel('||Δg||')
            ax[1].legend(ncol=3, fontsize=8)
            fig.tight_layout()
            fig.savefig('spherical_gradient_demo_overlay.png', dpi=150)
            print('Saved plot to spherical_gradient_demo_overlay.png')

    # Plot g_x, g_y, g_z along a circular path for one representative (l,m)
    lm_pick = None
    for cand in [(2, 1), (1, 0), (0, 0)]:
        if _supports_lm(*cand):
            lm_pick = cand
            break
    if lm_pick is not None and _HAS_PLOT:
        l, m = lm_pick
        radial = RadialModel(l=l, a0=0.8)
        # Less-symmetric inclined circular path at fixed radius q (plane normal not aligned to axes)
        t = np.linspace(0.0, 2 * np.pi, 361)
        n_vec = np.array([0.7, 0.2, 0.67], dtype=float)
        n_vec /= np.linalg.norm(n_vec)
        a = np.array([0.3, -0.9, 0.2], dtype=float)
        u = np.cross(n_vec, a)
        if np.linalg.norm(u) < 1e-8:
            a = np.array([1.0, 0.0, 0.0], dtype=float)
            u = np.cross(n_vec, a)
        u /= np.linalg.norm(u)
        v = np.cross(n_vec, u)
        qvecs_path = q * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
        gA = np.array([grad_spherical_function(v, l, m, radial) for v in qvecs_path])
        gN = np.array([finite_diff_grad(v, l, m, radial) for v in qvecs_path])
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
        labels = ['x', 'y', 'z']
        for i in range(3):
            ax[i].plot(t, np.real(gA[:, i]), label=f'analytic g_{labels[i]}')
            ax[i].plot(t, np.real(gN[:, i]), '--', label=f'numeric g_{labels[i]}')
            ax[i].set_ylabel(f'g_{labels[i]}')
            ax[i].legend(fontsize=8)
        ax[-1].set_xlabel('t (radians) along inclined circular path')
        fig.tight_layout()
        fig.savefig('spherical_gradient_path_components.png', dpi=150)
        print('Saved plot to spherical_gradient_path_components.png')


if __name__ == '__main__':
    main()


