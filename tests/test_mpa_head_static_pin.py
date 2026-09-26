"""The scalar head fit reproduces its static sample exactly.

On shell the band-diagonal head is ``(1/2 - f) W^c(0) / (Omega N_k)`` for
any pole model, so the static sample carries the head's on-shell
self-energy.  The guarded fit's unweighted residue refit missed it by
5.8 Ry bohr^3 on Fe 4^3 (a +/-7.7 meV on-shell error);
``gw.mpa.model._pin_static_head_sample`` re-solves the residues with that
sample as an equality constraint.
"""

from __future__ import annotations

import numpy as np
import pytest

def _pole_model(z, omega, residues):
    return np.sum(2.0 * omega * residues / (z[:, None] ** 2 - omega ** 2),
                  axis=1)


def test_static_pin_makes_the_static_sample_exact_and_is_constrained_lsq():
    from gw.mpa.model import _pin_static_head_sample

    rng = np.random.default_rng(3)
    z = np.concatenate(([2.0e-5j], np.linspace(0.1, 3.0, 7) + 0.2j,
                        [2.0j], np.linspace(0.1, 3.0, 7) + 2.0j))
    omega = np.asarray([0.3 - 0.1j, 0.9 - 0.2j, 1.7 - 0.3j, 2.6 - 0.1j])
    truth = rng.normal(size=4) + 1j * rng.normal(size=4)
    wc = _pole_model(z, omega, truth)
    wc[0] += 5.0 + 0.0j          # a static sample the pole set cannot reach
    fitted = {"Omega_p": omega, "B_p": truth, "max_abs_residual": 5.0}
    got = _pin_static_head_sample(wc, z, fitted)
    model = _pole_model(z, omega, got["B_p"])
    assert abs(model[0] - wc[0]) < 1e-10 * abs(wc[0])
    assert got["static_pin_residual_before"] == pytest.approx(5.0, rel=1e-12)
    assert got["static_pin_z"] == z[0]
    # The equality-constrained optimum is the limit of a heavily weighted
    # least squares on the same poles.
    matrix = 2.0 * omega[None, :] / (z[:, None] ** 2 - omega[None, :] ** 2)
    weight = np.ones(z.size)
    weight[0] = 1.0e7
    heavy, *_ = np.linalg.lstsq(matrix * weight[:, None], wc * weight,
                                rcond=None)
    np.testing.assert_allclose(got["B_p"], heavy, rtol=1e-5, atol=1e-8)
    # A fit that already interpolates every sample is left unchanged.
    exact = _pole_model(z, omega, truth)
    same = _pin_static_head_sample(
        exact, z, {"Omega_p": omega, "B_p": truth, "max_abs_residual": 0.0})
    np.testing.assert_allclose(same["B_p"], truth, rtol=1e-9, atol=1e-12)
