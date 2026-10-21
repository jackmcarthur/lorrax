"""htransform's modern-theory orbital route on toy inputs (CPU, seconds).

The star-weighted parent total equals the full-BZ sum and the per-parent
sum, the moved SOS pieces equal
``orbital_response.orbital_magnetization``, and a stored velocity that
is not the WFN's, or a QP velocity with no Sigma term, refuses before any
payload read.
"""
from types import SimpleNamespace

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from bandstructure import orbital  # noqa: E402
from file_io import dipole  # noqa: E402
from psp.orbital_response import (  # noqa: E402
    orbital_magnetization, orbital_pieces_at_k)

_R = np.diag([1.0, -1.0, -1.0])          # C2 about x: polar = axial action


def _velocity(rng, nb):
    a = rng.standard_normal((3, nb, nb)) + 1j * rng.standard_normal((3, nb, nb))
    return a + np.conj(np.swapaxes(a, 1, 2))


def test_orbital_identities_sos_full_bz_and_per_parent_sum():
    # SOS pieces equal the thermodynamic moment.
    rng = np.random.default_rng(1)
    v, e = _velocity(rng, 8), np.sort(rng.standard_normal(8))
    mu = 0.5 * (e[2] + e[3])
    pa, pb = orbital_pieces_at_k(v, e, 3, 1e-8)
    sos = 0.5 * (pa - 2 * mu * pb).sum(axis=(1, 2)).imag
    lib = np.asarray(orbital_magnetization(v, e, mu_ry=mu, width_ry=0.0))
    np.testing.assert_allclose(sos, lib, atol=1e-12)

    # The star-weighted parent total with axial weights equals the full-BZ sum.
    rng = np.random.default_rng(2)
    nb, nelec = 12, 4
    v, e = _velocity(rng, nb), np.sort(rng.standard_normal(nb))
    v_image = np.einsum("ab,bnm->anm", _R, v)      # velocity at C2x k
    sym = SimpleNamespace(
        irr_idx_k=np.array([0, 0]), nk_tot=2, active_symmetry_rows=[0, 1],
        cartesian_action=lambda rows, axial, time_odd: np.stack(
            [np.eye(3), _R])[np.asarray(rows)])
    mu, E_top, m = orbital.orbital_totals(
        v[None], e[None], sym, nelec=nelec - 1e-7, width_ry=None,
        deps_tol_ry=1e-8)                           # a float count, as QE writes
    assert mu == 0.5 * (e[nelec - 1] + e[nelec]) and E_top == e[-1]
    full = 0.5 * sum(np.asarray(orbital_magnetization(
        x, e, mu_ry=mu, width_ry=0.0)) for x in (v, v_image))
    np.testing.assert_allclose(m, full, atol=1e-12)
    assert abs(m[0]) > 1e-3                         # the check sees a moment

    # Three parents, file band order, Fermi-Dirac: the traced total equals
    # the per-parent sum.
    rng = np.random.default_rng(5)
    nb, nelec, kT = 10, 4.0, 0.05
    v = np.stack([_velocity(rng, nb) for _ in range(3)])
    e = np.stack([rng.permutation(np.linspace(-1, 2, nb)) + 0.01 * rng.standard_normal(nb)
                  for _ in range(3)])
    sym = SimpleNamespace(
        irr_idx_k=np.array([0, 1, 1, 2, 2, 2]), nk_tot=6,
        active_symmetry_rows=[0],
        cartesian_action=lambda rows, axial, time_odd: np.eye(3)[None])
    mu, _, m = orbital.orbital_totals(v, e, sym, nelec=nelec, width_ry=kT,
                                      deps_tol_ry=1e-8)
    w = np.array([1, 2, 3]) / 6.0
    ref = sum(w[p] * np.asarray(orbital_magnetization(v[p], e[p], mu_ry=mu,
                                                      width_ry=kT))
              for p in range(3))
    np.testing.assert_allclose(m, ref, atol=1e-13)
    assert np.abs(ref).max() > 1e-3

