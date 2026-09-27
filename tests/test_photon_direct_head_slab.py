"""The slab (``sys_dim = 2``) direct four-current Γ head.

Limits a slab-truncated head must reproduce, on the production primitives
(``vcoul`` slab ``D``, ``project_first_order_photon_response``,
``metal_intraband_photon_response``, ``_solve_photon_head``):

* insulator: ``eps_CC(q) = 1 + 8 pi (1 - e^{-zc q}) |q.S.q|/q^2 -> 1`` linearly
  in ``q`` (the 2D polarizability form), not the bulk constant ``eps_inf``;
* metal: the CC plasmon of the 4x4 Dyson with Fermi-surface atoms follows
  ``omega^2 = 8 pi (1 - e^{-zc q}) qhat.D.qhat + (3/4) (q u)^2``, so
  ``omega ~ sqrt(q)``;
* the cell average: the exact polygon rule with zero response returns the
  bare ``<D>`` whose TT block is the bare TT Γ tile of V to rounding, and
  the slab Coulomb moment has ``Q_zz = 0``.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.sharding import Mesh  # noqa: E402

_A, _C = 12.0, 40.0
_BVEC = np.diag([2 * np.pi / _A, 2 * np.pi / _A, 2 * np.pi / _C])
_VOLUME = _A * _A * _C
_KGRID = (4, 4, 1)
_ZC = np.pi / _BVEC[2, 2]


def _geometry():
    from ffi import _services
    _services.ensure_on_path()
    from vcoul import CoulombGeometry
    return CoulombGeometry(bvec=_BVEC, cell_volume=_VOLUME)


def _slab_D(q):
    from vcoul.minibz import _photon_D_raw
    return _photon_D_raw(np.asarray(q, np.float64), kind="slab", zc=_ZC)[0]


def _head(D, Pi):
    from gw.head_correction import _solve_photon_head
    return np.asarray(_solve_photon_head(jnp.asarray(D, jnp.complex128),
                                         jnp.asarray(Pi, jnp.complex128))[0])


def test_insulator_slab_screening_tends_to_one_linearly():
    from gw.photon_direct_head import project_first_order_photon_response
    s = 0.02                      # -S along x, per volume (bohr^-3 Ry^-1 scale)
    M = np.zeros((6, 6), np.complex128)
    M[:3, :3] = -s * np.diag([1.0, 1.3, 0.4])     # anisotropic jets
    M[0, 3] = M[3, 0] = 1e-3                        # a CT coupling, O(q)
    M[3:, 3:] = -1e-4 * np.eye(3)
    q = np.array([[x, 0.0, 0.0] for x in (1e-4, 3e-4, 1e-3, 3e-3)])
    Pi = np.asarray(project_first_order_photon_response(jnp.asarray(M), q))
    W = _head(_slab_D(q), Pi)
    eps = _slab_D(q)[:, 0, 0] / W[:, 0, 0].real
    expected = 1.0 + 8 * np.pi * (1 - np.exp(-_ZC * q[:, 0])) * s
    np.testing.assert_allclose(eps, expected, rtol=1e-4)
    slope = (eps - 1.0) / q[:, 0]
    np.testing.assert_allclose(slope[0], 8 * np.pi * _ZC * s, rtol=2e-3)
    assert eps[0] - 1.0 < 1e-2


def test_metal_slab_plasmon_scales_as_sqrt_q():
    from gw.fermi_surface import _intraband_blocks
    from gw.photon_direct_head import metal_intraband_photon_response
    from gw.head_correction import _solve_photon_head
    n, u0, W0 = 24, 0.3, 2e-3
    phi = 2 * np.pi * np.arange(n) / n
    u = u0 * np.stack((np.cos(phi), np.sin(phi), 0 * phi), 1)
    w = np.full(n, W0 / n)
    drude = np.einsum("s,sa,sb->ab", w, u, u)
    blocks = [jnp.asarray(x) for x in _intraband_blocks(w, u)]

    def det(q, omega):
        Pi, _ = metal_intraband_photon_response(
            jnp.asarray(q[None]), jnp.asarray(omega + 0j), jnp.asarray(drude),
            float(np.sum(w)), *blocks)
        lhs = _solve_photon_head(jnp.asarray(_slab_D(q[None]), jnp.complex128), Pi)[1]
        return float(np.linalg.det(np.asarray(lhs[0])).real)

    omegas = []
    qs = np.array([1e-4, 4e-4, 1.6e-3])
    for x in qs:
        q = np.array([x, 0.0, 0.0])
        lo, hi = 2.0 * x * u0, 1.0
        assert det(q, lo) * det(q, hi) < 0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if det(q, lo) * det(q, mid) > 0 else (lo, mid)
        omegas.append(0.5 * (lo + hi))
    omegas = np.asarray(omegas)
    # 2D Lindhard at |z| >> q u: omega^2 = omega_p(q)^2 + (3/4) (q u)^2.
    analytic = np.sqrt(8 * np.pi * (1 - np.exp(-_ZC * qs)) * drude[0, 0]
                       + 0.75 * (qs * u0) ** 2)
    np.testing.assert_allclose(omegas, analytic, rtol=1e-3)
    exponent = np.polyfit(np.log(qs), np.log(omegas), 1)[0]
    assert abs(exponent - 0.5) < 0.02


def test_slab_moment_is_in_plane():
    from vcoul import minibz_coulomb_moment
    Q = minibz_coulomb_moment(_BVEC, _KGRID, is_2d=True)
    assert Q[2, 2] == 0.0 and np.allclose(Q[:2, 2], 0.0)
    np.testing.assert_allclose(Q[0, 0], Q[1, 1], rtol=1e-12)
    # Independent polar rule over the square mini-BZ cell: R(theta) is the
    # distance to the square's edge; Gauss-Legendre in r, midpoint in theta.
    half = np.pi / _A / _KGRID[0]
    theta = (np.arange(20000) + 0.5) / 20000 * 2 * np.pi
    radius = half / np.maximum(np.abs(np.cos(theta)), np.abs(np.sin(theta)))
    t, wt = np.polynomial.legendre.leggauss(64)
    r = 0.5 * (t + 1)[None, :] * radius[:, None]
    wr = 0.5 * wt[None, :] * radius[:, None]
    vq = 8 * np.pi * (1 - np.exp(-_ZC * r)) / r        # v(q) q, the area element
    num = np.sum(wr * vq * r**2 * np.cos(theta)[:, None] ** 2)
    np.testing.assert_allclose(Q[0, 0], num / np.sum(wr * vq), rtol=1e-7)


def test_zero_response_cell_average_is_the_bare_tt_tile():
    from gw.photon_direct_head import _slab_gamma_cell_average
    from gw.v_q_bispinor import _tt_head_tensor
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))
    nz = 2
    zeros = jnp.zeros((nz, 6, 6), jnp.complex128)
    blocks = [jnp.zeros((1, 64)), jnp.zeros((1, 64, 3))]
    operands = (zeros, zeros, jnp.zeros((4, 6, 6), jnp.complex128),
                jnp.asarray([0.0, 0.5j]), jnp.zeros((3, 3), jnp.complex128),
                jnp.asarray(0.0), jnp.zeros((4, 4), jnp.complex128), *blocks)
    fields, spread, residual, rule = _slab_gamma_cell_average(
        _geometry(), _KGRID, operands, mesh, _VOLUME)
    Wc, _, _, _, constant, moments, bare = fields
    assert np.max(np.abs(Wc)) == 0.0 and np.max(np.abs(constant)) == 0.0
    tt = _tt_head_tensor(bvec=_BVEC, cell_volume=_VOLUME, sys_dim=2, kgrid=_KGRID)
    np.testing.assert_allclose(-bare[1:, 1:].real * _VOLUME, tt, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.diag(tt) / tt[2, 2], [0.5, 0.5, 1.0], atol=1e-12)
    assert "Wigner-Seitz" in rule and residual < 1e-12


def test_slab_fermi_surface_table_is_the_2d_dos():
    """The tetrahedron table on an (n, n, 1) grid integrates a z-independent
    band as the 2D free-electron Fermi circle: sum w = pi / A_BZ for E = k^2."""
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import lattice_grid_point_group
    from gw.fermi_surface import tetrahedron_delta_weights
    n = 24
    frac = np.stack(np.meshgrid(np.arange(n) / n, np.arange(n) / n, [0.0],
                                indexing="ij"), -1).reshape(-1, 3)
    wrapped = frac - np.rint(frac)
    k = wrapped @ _BVEC
    energies = np.sum(k[:, :2] ** 2, axis=1)[:, None]
    w = tetrahedron_delta_weights(
        energies, frac, (n, n, 1), 0.05,
        symmetry_matrices=lattice_grid_point_group(_BVEC, (n, n, 1)))
    area = abs(np.linalg.det(_BVEC[:2, :2]))
    np.testing.assert_allclose(np.sum(w), np.pi / area, rtol=2e-2)
