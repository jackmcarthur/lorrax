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


def test_fixed_stream_nodes_update_projections_and_check_current_points():
    z = np.array([1j, 2+1j, 8j])/RY_EV
    rule = minimax.response_bank_rule(z, 30/RY_EV, domain_pad_ry=4/RY_EV)
    current = z + np.array([.05j, .03, -.2j])/RY_EV
    reused = minimax.response_bank_rule(current, 32/RY_EV, previous=rule)
    assert reused['reuse_status'] == 'hit'
    assert reused['node_digest'] == rule['node_digest']
    np.testing.assert_array_equal(reused['t'], rule['t'])
    np.testing.assert_array_equal(reused['h'], rule['h'])
    delta = np.linspace(-32, 32, 707)/RY_EV
    phase = np.exp(1j*delta[:, None]*reused['t'])
    truth = 1j/(current[None, :]+delta[:, None])
    truth_ds = -1j/(2*current[None, :]*(current[None, :]+delta[:, None])**2)
    eta = current.imag.min()
    assert np.max(abs(phase@reused['projection_value'].T-truth))*eta < 1e-8
    assert np.max(abs(phase@reused['projection_derivative'].T-truth_ds))*eta**3 < 1e-8
    assert np.max(abs(phase@rule['projection_value'].T-truth))*eta > 1e-4
    escaped = minimax.response_bank_rule(current, 36/RY_EV, previous=reused)
    assert escaped['reuse_status'] == 'rebuild'
    assert escaped['node_digest'] != reused['node_digest']
    corrupted = dict(reused, h=reused['h']*1.01)
    with pytest.raises(ValueError, match='digest mismatch'):
        minimax.response_bank_rule(current, 32/RY_EV, previous=corrupted)
    # A lower imaginary frequency lengthens the infinite tail even though
    # transition energies are still contained.
    lower = minimax.response_bank_rule(current/2, 32/RY_EV, previous=reused)
    assert lower['reuse_status'] == 'rebuild'


def test_fixed_remote_rows_update_current_taylor_projection():
    z = np.array([1j, 2+1j, 5+1j])/RY_EV
    rule = minimax.response_laplace_rule(45/RY_EV, 90/RY_EV, z,
                                        domain_pad_ry=4/RY_EV)
    current = z + np.array([.04j, .03, -.05])/RY_EV
    reused = minimax.response_laplace_rule(44/RY_EV, 92/RY_EV, current, previous=rule)
    assert reused['reuse_status'] == 'hit'
    np.testing.assert_array_equal(rule['coefficient_rows'], reused['coefficient_rows'])
    delta = np.geomspace(44/RY_EV, 92/RY_EV, 503)
    basis = np.exp(-delta[:, None]*reused['t'])
    truth = delta[:, None]/(delta[:, None]**2-current[None, :]**2)
    ds = delta[:, None]/(delta[:, None]**2-current[None, :]**2)**2
    assert np.max(abs(basis@reused['projection_value'].T/truth-1)) < 1e-8
    assert np.max(abs(basis@reused['projection_derivative'].T/ds-1)) < 1e-8
    assert np.max(abs(basis@rule['projection_value'].T/truth-1)) > 1e-6


def test_remote_padding_does_not_cross_taylor_boundary():
    z = np.array([1.4+.1j])
    rule = minimax.response_laplace_rule(2., 3., z, rel_tol=1e-6, domain_pad_ry=1.)
    assert rule['certificate']['delta_ry'] == [2., 4.]
    delta = np.geomspace(2., 3., 101)
    truth = delta/(delta**2-z[0]**2)
    got = np.exp(-delta[:, None]*rule['t'])@rule['projection_value'][0]
    assert np.max(abs(got/truth-1)) < 1e-6
