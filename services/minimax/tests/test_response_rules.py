"""Independent analytic kernel checks for additive response rule doors."""
import numpy as np
import pytest
import minimax

RY_EV = 13.605693122994


@pytest.mark.parametrize('height,top,imag,delta_max', [
    (1., 5.5, [1., 4., 16.], 45.),
    (1., 15., [1.5, 3., 7., 16.], 60.),
    (.4, 6., [.4, 2., 16., 64.], 45.),
])
def test_hermite_current_points(height, top, imag, delta_max):
    z = np.r_[np.arange(0., top+.01, height/2)+1j*height,
              1j*np.asarray(imag)]/RY_EV
    rule = minimax.response_bank_rule(z, delta_max/RY_EV)
    assert np.all(rule['t'] > 0) and np.all(rule['h'] > 0)
    assert rule['certificate']['status'] == 'PASS'
    eta = z.imag.min()
    d = np.unique(np.r_[np.linspace(-delta_max, delta_max, 1003)/RY_EV,
                        z.real, -z.real, 0.])
    for zi, p, dp in zip(z, rule['projection_value'], rule['projection_derivative']):
        phase = np.exp(1j*d[:, None]*rule['t'])
        actual = phase@p
        derivative = phase@dp
        exact = 1j/(zi+d)
        exact_ds = -1j/(2*zi*(zi+d)**2)
        assert np.max(np.abs(actual-exact))*eta < 1e-8
        assert np.max(np.abs(derivative-exact_ds))*eta**3 < 1e-8
    # The omitted chain factor is a physically different derivative.
    wrong = rule['projection_value'][0]*(1j*rule['t'])
    got = np.exp(1j*d[:, None]*rule['t'])@wrong
    exact = -1j/(2*z[0]*(z[0]+d)**2)
    assert np.max(np.abs(got-exact))*eta**3 > 1e-4


@pytest.mark.parametrize('lo,hi,z_ev', [
    (35., 90., [1j, 2+1j, 5.5+1j, 4j, 16j]),
    (45., 120., [1j, 7+1j, 15+1j, 1.5j, 16j]),
])
def test_remote_current_points(lo, hi, z_ev):
    z = np.asarray(z_ev)/RY_EV
    rule = minimax.response_laplace_rule(lo/RY_EV, hi/RY_EV, z)
    assert np.all(rule['t'] > 0) and np.all(rule['coefficient_rows'] >= 0)
    cert = rule['certificate']
    assert cert['status'] == 'PASS'
    delta = np.geomspace(lo/RY_EV, hi/RY_EV, 3007)
    basis = np.exp(-delta[:, None]*rule['t'])
    value = basis@rule['projection_value'].T
    derivative = basis@rule['projection_derivative'].T
    truth = delta[:, None]/(delta[:, None]**2-z[None, :]**2)
    truth_ds = delta[:, None]/(delta[:, None]**2-z[None, :]**2)**2
    assert np.max(np.abs(value/truth-1)) < 1e-8
    assert np.max(np.abs(derivative/truth_ds-1)) < 1e-8
    assert max(cert['value_bound']) < 1e-8
    assert max(cert['derivative_bound']) < 1e-8


def test_response_refusals():
    with pytest.raises(ValueError, match='upper-half-plane'):
        minimax.response_bank_rule([1.], 2.)
    with pytest.raises(ValueError, match='nonnegative'):
        minimax.response_bank_rule([1j], -1.)
    with pytest.raises(ValueError, match='positive'):
        minimax.response_laplace_rule(0., 1., [1j])
    with pytest.raises(ValueError, match='does not converge'):
        minimax.response_laplace_rule(1., 2., [4+1j])
