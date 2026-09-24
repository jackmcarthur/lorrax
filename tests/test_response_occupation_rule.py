"""One complete signed response with independent ordered amplitudes.

One shared node set serves a group of samples: forward rows use exp(-d t),
reverse rows the same Green pair at conj(t).
"""
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
    z = np.array([1j, 9.91+2.6j])/EV
    rules = minimax.response_group_rules(d.min(), d.max(), z,
        decay_rate=beta, rel_tol=1e-8/amplitude)
    assert sorted(m for rule in rules for m in rule['members']) == [0, 1]
    for rule in rules:
        n = rule['count']
        times = (rule['t'][:n], np.conj(rule['t'][:n]))
        for row, sample in enumerate(rule['members']):
            point = z[sample]
            values, slopes = [], []
            for side in (0, 1):
                basis = np.exp(-(d[:, None]-rule['reference_ry'])*times[side])
                values.append(basis@rule['value'][row, side, :n])
                slopes.append(basis@rule['derivative'][row, side, :n])
            got = -np.sum(weight*(a*values[0]+b*values[1]))
            exact = np.sum(weight*(a/(point-d)-b/(point+d)))
            ds = -np.sum(weight*(a*slopes[0]+b*slopes[1]))
            exact_ds = np.sum(weight*(-a/(point-d)**2+b/(point+d)**2))/(2*point)
            assert abs(got-exact)*point.imag < 1e-7
            assert abs(ds-exact_ds)*point.imag**3 < 1e-7
