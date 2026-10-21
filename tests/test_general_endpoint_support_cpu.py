"""Exact current full-matrix endpoint supports, never population pruning."""
import numpy as np
import pytest

from tests._full_density_blocks_oracle import four_block_panels, pair_spectrum, pair_frequency
from tests.test_full_density_blocks_oracle_cpu import case


def _blocks(energy, rho):
    from gw.shared_pole_screening import _density_endpoint_column_support
    from gw.response_bank import response_blocks
    f, u = _density_endpoint_column_support(rho, energy.shape[1])
    return response_blocks(energy, f, u, (1, 3), allow_crossing=True, exact_endpoint_support=True)


@pytest.mark.parametrize('coherence', [0., 1e-30, .07j])
def test_exact_support_preserves_literal_all_q_full_complex_four_blocks(coherence):
    from gw.shared_pole_screening import _density_endpoint_column_support
    psi, energy, rho, partners = case(810)
    rho[:, 1:3, 1:3] = np.array([[1., coherence], [np.conj(coherence), 0.]])
    f, u = _density_endpoint_column_support(rho, 4)
    assert bool(f[:, 2].all()) == bool(coherence != 0)
    assert bool(u[:, 1].all()) == bool(coherence != 0)
    if coherence != 0:
        assert f[:, 1:3].all() and u[:, 1:3].all()
    identity = np.broadcast_to(np.eye(4), rho.shape)
    expected = four_block_panels(rho, slice(1, 3))
    sites = np.array([-.37+.23j, .19+.31j])
    for block in _blocks(energy, rho):
        name = block['name']
        left = (identity-rho if name[1] == 'A' else identity)*block['u'][:, None]
        right = (rho if name[0] == 'A' else identity)*block['f'][:, None]
        # The new masks remove only literally ZERO columns of each complete
        # matrix operand, including for infinitesimal off-diagonal coherence.
        np.testing.assert_array_equal(left, expected[name][0])
        np.testing.assert_array_equal(right, expected[name][1])
        for partner in partners:
            actual = pair_frequency(pair_spectrum(psi, energy, left, right, partner), sites, .043)
            literal = pair_frequency(pair_spectrum(psi, energy, *expected[name], partner), sites, .043)
            np.testing.assert_array_equal(actual, literal)


def test_new_live_coherent_column_expands_interval_and_rebuilds_held_rule():
    from gw.response_bank import full_density_sample_rules
    _, energy, rho, _ = case(811)
    rho[:, 1:3, 1:3] = np.diag([1., 0.])
    aa = next(b for b in _blocks(energy, rho) if b['name'] == 'AA')
    assert aa['lo'] > 0
    held = {}
    old = full_density_sample_rules(energy, np.array([.8j]), gamma_ry=.03, paired=True,
        interval_ry=(aa['lo'], aa['hi']), held=held)
    rho[:, 1, 2] = 1e-30
    rho[:, 2, 1] = 1e-30
    aa = next(b for b in _blocks(energy, rho) if b['name'] == 'AA')
    assert aa['lo'] < 0
    new = full_density_sample_rules(energy, np.array([.8j]), gamma_ry=.03, paired=True,
        interval_ry=(aa['lo'], aa['hi']), held=held)
    assert new is not old and held['interval_ry'][0] <= aa['lo']


def test_declared_chi_stop_only_masks_excluded_columns():
    from gw.shared_pole_screening import _density_endpoint_column_support
    rho = np.broadcast_to(np.eye(4), (2, 4, 4))
    f, u = _density_endpoint_column_support(rho, 3)
    np.testing.assert_array_equal(f, [[1., 1., 1., 0.]]*2)
    assert not u.any()


def test_default_response_block_geometry_remains_population_independent():
    from gw.response_bank import response_blocks
    from gw.shared_pole_screening import _density_endpoint_column_support
    _, energy, rho, _ = case(812)
    rho[:, 1:3, 1:3] = np.diag([1., 0.])
    f, u = _density_endpoint_column_support(rho, 4)
    default = response_blocks(energy, f, u, (1, 3), allow_crossing=True)
    explicit = response_blocks(energy, f, u, (1, 3), allow_crossing=True, exact_endpoint_support=False)
    for a, b in zip(default, explicit):
        assert a['lo'] == b['lo'] and a['hi'] == b['hi']
        np.testing.assert_array_equal(a['f'], b['f'])
        np.testing.assert_array_equal(a['u'], b['u'])
    assert next(b for b in default if b['name'] == 'AA')['lo'] < 0


def test_fractional_frozen_weights_are_preserved_and_existing_gate_still_refuses():
    from gw.response_bank import response_blocks
    from gw.shared_pole_screening import _density_endpoint_column_support
    _, energy, rho, _ = case(813)
    rho[:, 0, 0] = .37
    occupations = np.real(rho.diagonal(axis1=-2, axis2=-1))
    f, u = _density_endpoint_column_support(rho, 4, occupations=occupations, active=slice(1, 3))
    np.testing.assert_array_equal(f[:, 0], np.full(3, .37))
    np.testing.assert_array_equal(u[:, 0], np.full(3, 1.-.37))
    with pytest.raises(ValueError, match='response_blocks_frozen_fractional'):
        response_blocks(energy, f, u, (1, 3), allow_crossing=True, exact_endpoint_support=True)
