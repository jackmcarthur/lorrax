"""The metal head's Fermi-surface measure on free electrons: Thomas-Fermi and plasma frequency.

The head's intraband term reads one measure, the star-covariant tetrahedron
table of ``delta(E - mu)`` (``fermi_surface.metal_head_surface_weights``):
``kappa_TF^2 = 8 pi N(E_F) / Omega`` and ``D = (C / (Omega N_k)) sum w v v^T``,
``omega_p^2 = 8 pi qhat.D.qhat`` (docs/theory/metal-q0-head.md section 1).  On a
parabolic band ``E = |k|^2`` (Ry, ``m = 1/2``) with ``C = 2`` the exact values
are ``kappa_TF^2 = 4 k_F / pi`` and ``D = 2 n``, ``n = k_F^3 / (3 pi^2)``, i.e.
``omega_p^2 = 16 pi n`` (Na at r_s = 3.93: 6.05 eV).  Checked on simple-cubic
grids (one band, sphere inside the zone): the linear-tetrahedron error is
second order in the k spacing, so both quantities must converge to the exact
values from 12^3 to 24^3 and be within 1.5 % at 24^3; D stays isotropic.
"""
from types import SimpleNamespace

import numpy as np
import pytest

A = 1.0                      # bohr; Omega = A^3
KF = 1.5                     # bohr^-1, inside the zone (half-width pi / A)


def _measure(kgrid):
    """``(kappa_TF^2, D)`` of the parabolic band from the head's table on ``kgrid``."""
    from gw.fermi_surface import metal_head_surface_weights
    n = np.array(kgrid)
    grid = np.stack(np.meshgrid(*[np.arange(m) / m for m in n], indexing="ij"), -1).reshape(-1, 3)
    crys = (grid + 0.5) % 1.0 - 0.5                        # crystal k in [-1/2, 1/2)
    bvec = 2.0 * np.pi / A * np.eye(3)
    k = crys @ bvec
    energies = np.sum(k * k, axis=1)[:, None]              # E = |k|^2 (Ry), one band
    nk, omega, capacity = crys.shape[0], A ** 3, 2.0
    sym = SimpleNamespace(unfolded_kpts=crys, irr_idx_k=np.arange(nk))
    w = metal_head_surface_weights(energies, KF ** 2, sym=sym, kgrid=kgrid, bvec_cart=bvec)
    assert w.shape == (nk, 1) and np.all(w >= 0.0)
    kappa2 = 8.0 * np.pi * capacity * float(w.sum()) / nk / omega
    v = 2.0 * k                                            # dE/dk, Ry bohr
    D = capacity / (omega * nk) * np.einsum("k,ka,kb->ab", w[:, 0], v, v)
    return kappa2, D


def test_thomas_fermi_and_plasma_frequency_free_electrons():
    from common import RYD_TO_EV
    kappa2_exact = 4.0 * KF / np.pi
    n = KF ** 3 / (3.0 * np.pi ** 2)
    errors = []
    for g in (12, 24):
        kappa2, D = _measure((g, g, g))
        assert np.max(np.abs(D - np.diag(np.diag(D)))) < 1e-10 * np.max(np.diag(D))
        errors.append((abs(kappa2 / kappa2_exact - 1.0), float(np.max(np.abs(np.diag(D) / (2 * n) - 1.0)))))
    (k12, d12), (k24, d24) = errors
    assert k24 < k12 and d24 < d12                          # converges with the grid
    assert k24 < 0.015 and d24 < 0.015
    _, D = _measure((24, 24, 24))
    omega_p = np.sqrt(8.0 * np.pi * np.diag(D)) * RYD_TO_EV
    assert omega_p == pytest.approx(np.sqrt(16.0 * np.pi * n) * RYD_TO_EV, rel=0.01)


def test_na_density_plasma_frequency():
    """The free-electron plasma frequency at sodium's density, the comparator the logs quote."""
    from common import RYD_TO_EV
    rs = 3.93
    n = 3.0 / (4.0 * np.pi * rs ** 3)                      # bohr^-3
    assert np.sqrt(16.0 * np.pi * n) * RYD_TO_EV == pytest.approx(6.05, abs=0.01)
