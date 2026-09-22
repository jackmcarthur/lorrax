"""One complete signed response with independent ordered amplitudes."""
import numpy as np
import pytest
import minimax
from gw.response_bank import response_sample_weights, response_occupation_envelope
from common.units import RYD_TO_EV as EV


@pytest.mark.parametrize("metal", [False, True])
def test_signed_response_matches_direct_sum(metal):
    e = np.array([-88., -51., -20., -8., -.1, .1, 8., 20., 70., 300.])/EV
    f = 1/(1+np.exp(np.clip(e/(.02 if metal else .0001), -700, 700)))
    f, u, _ = response_sample_weights(f, 1-f)
    beta, amplitude = response_occupation_envelope(e, f, u, 0.)
    d = e[None, :]-e[:, None]
    weight = f[:, None]*u[None, :]
    live = weight != 0
    d, weight = d[live], weight[live]
    rng = np.random.default_rng(22)
    residues = rng.normal(size=(len(e), len(e)))+1j*rng.normal(size=(len(e), len(e)))
    a, b = residues[live], residues.T[live]
    for z in np.array([1j, 9.91+2.6j])/EV:
        rule = minimax.response_frequency_rule(d.min(), d.max(), z,
            decay_rate=beta, rel_tol=1e-8/amplitude)
        values, slopes = [], []
        for side in (0, 1):
            basis = np.exp(-(d[:, None]-rule['reference_ry'])*rule['t'][side])
            values.append(basis@rule['value'][side])
            slopes.append(basis@rule['derivative'][side])
        got = -np.sum(weight*(a*values[0]+b*values[1]))
        exact = np.sum(weight*(a/(z-d)-b/(z+d)))
        ds = -np.sum(weight*(a*slopes[0]+b*slopes[1]))
        exact_ds = np.sum(weight*(-a/(z-d)**2+b/(z+d)**2))/(2*z)
        assert abs(got-exact)*z.imag < 1e-7
        assert abs(ds-exact_ds)*z.imag**3 < 1e-7
