"""Every transition has one owner, including fractional TR-broken pairs."""
import numpy as np
import pytest
import minimax
from gw.response_bank import response_windows
from common.units import RYD_TO_EV as EV


@pytest.mark.parametrize("metal", [False, True])
def test_crossing_and_remote_cover_ordered_response(metal):
    e = np.array([-88., -51., -20., -8., -.1, .1, 8., 20., 70., 300.])/EV
    f = 1/(1+np.exp(np.clip(e/(.02 if metal else .0001), -700, 700)))
    z = np.array([1j, 3+1j, 9.91+1j, 34j])/EV
    masks, ft, ut, cells, receipt = response_windows(
        e, f, 1-f, chemical_potential_ry=0., z_ry=z)
    np.testing.assert_array_equal(sum(m.astype(int) for m in masks), 1)
    # Moving the imaginary ladder cannot enlarge the crossing state window.
    other = response_windows(e, f, 1-f, chemical_potential_ry=0.,
                             z_ry=np.r_[z, 100j/EV])[0]
    for got, want in zip(masks, other):
        np.testing.assert_array_equal(got, want)
    assert receipt['window_ev_relative_mu'][1] < 25
    assert np.all(masks[0][e*EV < -50])
    # Independent complex amplitudes represent unequal +q/-q residues.
    rng = np.random.default_rng(22)
    amplitude = rng.normal(size=(len(e), len(e)))+1j*rng.normal(size=(len(e), len(e)))
    d = e[None, :]-e[:, None]
    weight = ft[:, None]*ut[None, :]
    exact = np.sum(weight[..., None]*(amplitude[..., None]/(z-d[..., None])
                   - amplitude.T[..., None]/(z+d[..., None])), axis=(0, 1))
    centre = masks[1][:, None] & masks[1][None, :]
    got = np.sum((centre*weight)[..., None]*(amplitude[..., None]/(z-d[..., None])
                 - amplitude.T[..., None]/(z+d[..., None])), axis=(0, 1))
    owners = centre.astype(int)
    for cell in cells:
        lower, upper = masks[cell['lower']], masks[cell['upper']]
        pair = lower[:, None] & upper[None, :]
        owners += pair + pair.T
        assert cell['delta_min_ry'] > max(abs(z.real))
        ref = cell['references_ry'][1]-cell['references_ry'][0]
        rule = minimax.response_laplace_rule(cell['delta_min_ry'], cell['delta_max_ry'],
                                             z, ordered=True, reference_ry=ref)
        dd = d[pair]
        basis = np.exp(-(dd[:, None]-ref)*rule['t'])
        even = basis@rule['projection_value'].T
        odd = basis@rule['odd_projection_value'].T
        # 1/(z-d)=-(even+odd), -1/(z+d)=-(even-odd).
        a, b = amplitude[pair], amplitude.T[pair]
        w, wr = weight[pair], weight.T[pair]
        got += np.sum(-w[:, None]*((a+b)[:, None]*even+(a-b)[:, None]*odd)
                      +wr[:, None]*((a+b)[:, None]*even+(a-b)[:, None]*odd), axis=0)
    np.testing.assert_array_equal(owners[weight != 0], 1)
    np.testing.assert_allclose(got, exact, rtol=1e-8, atol=1e-9)
