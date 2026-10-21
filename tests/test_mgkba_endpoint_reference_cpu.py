"""Independent three-level endpoint and correlated-reference diagnostics."""
import numpy as np

from tests.tdagw.cri3_qgrid_750_campaign.diagnostics import audit_mgkba_endpoint_reference as toy


def coherent_density():
    rng = np.random.default_rng(91)
    u, _ = np.linalg.qr(rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)))
    return u@np.diag([.95, .7, .35])@u.conj().T


def test_external_free_propagator_fixes_column_sampling_and_density_order():
    result = toy.endpoint_test(coherent_density())
    assert result['column_integral_error_max'] < 1e-11
    assert abs(result['raw_trace']) < 1e-14
    assert result['wrong_negative_lag_order_error_max'] > .05
    assert abs(result['wrong_negative_lag_order_trace']) > .02


def test_physical_number_constrained_markov_stationary_state_is_not_ground_state():
    warm, narrow = (toy.stationary(eta) for eta in (.1, .001))
    for result in (warm, narrow):
        assert result['root_success'] and result['full_rhs_residual_fro'] < 1e-13
        assert abs(result['trace']-2) < 1e-13
        assert 0 < min(result['eigenvalues']) < max(result['eigenvalues']) < 1
        assert result['tangent_jacobian_real_range'][1] < 0
        assert result['diagonal'][2] > .3
    # Relaxation vanishes with eta; the infinite-time occupation does not.
    assert abs(narrow['diagonal'][2]-warm['diagonal'][2]) < .02
    assert abs(narrow['tangent_jacobian_real_range'][1]) < .02*abs(warm['tangent_jacobian_real_range'][1])
    weak = toy.stationary(.01, .1)
    ground = toy.exact_ground_state(.1, 12)
    assert weak['diagonal'][2] > .3
    assert ground['density_diagonal'][2] < .002
    assert ground['raw_rhs_at_eta_005_fro'] > 1e-5
    # The comparison uses an exact fixed-N fermion Fock sector, not a
    # distinguishable-particle or unconstrained oscillator density surrogate.
    assert abs(ground['trace']-2) < 1e-13


def test_PV_dressing_identity_has_coherent_remainder_and_number_bookkeeping():
    projector = toy.dressing_test(toy.PROJECTOR)
    coherent = toy.dressing_test(coherent_density())
    assert projector['identity_error_max'] < 1e-14
    assert projector['commutator_only_error_fro'] < 1e-14
    assert coherent['identity_error_max'] < 1e-14
    assert coherent['coherent_remainder_fro'] > .1
    traces = toy.spectral_only_test(coherent_density())
    assert abs(traces['raw_fresh_trace']) < 1e-14
    assert abs(traces['spectral_only_fresh_trace']) > .01
