"""Opt-in exact full endpoint merge; no response, band or gate shortcut."""
import numpy as np
import pytest

from tests._full_density_blocks_oracle import pair_spectrum, pair_frequency


def case(coherence=0.):
    rng = np.random.default_rng(717)
    nk, nb, ns, mu = 3, 8, 2, 4
    psi = .2*(rng.normal(size=(nk, nb, ns, mu))+1j*rng.normal(size=(nk, nb, ns, mu)))
    energy = np.array([-8., -2., -.4, -.1, .2, .7, 1.6, 6.])[None]
    energy = energy+rng.uniform(-.03, .03, (nk, nb))
    rho = np.broadcast_to(np.diag([1., 1., 1., 0., 0., 0., 0., 0.]), (nk, nb, nb)).astype(complex).copy()
    if coherence:
        rho[:, 2:4, 2:4] = np.array([[.7, coherence], [np.conj(coherence), .3]])
    return psi, energy, rho


def partitions(energy, rho, partition):
    from gw.shared_pole_screening import _density_endpoint_column_support, _general_response_blocks
    occ = np.real(rho.diagonal(axis1=-2, axis2=-1))
    f, u = _density_endpoint_column_support(rho, len(energy[0]), occupations=occ, active=slice(2, 4))
    return _general_response_blocks(energy, f, u, (2, 4), partition), f, u


def endpoints(block, rho):
    identity = np.broadcast_to(np.eye(rho.shape[-1]), rho.shape)
    left_density, right_density = block['density_sides']
    left = (identity-rho if left_density else identity)*block['u'][:, None]
    right = (rho if right_density else identity)*block['f'][:, None]
    return left, right


@pytest.mark.parametrize('coherence', [0., 1e-30, .07j])
def test_merged_full_retains_four_components_and_far_virtual_frequency_sum(coherence):
    psi, energy, rho = case(coherence)
    blocks, f, u = partitions(energy, rho, 'blocks')
    merged, _, _ = partitions(energy, rho, 'full')
    assert len(blocks) == 4 and len(merged) == 1
    full = merged[0]
    assert full['name'] == 'full' and full['density_sides'] == (True, True)
    assert full['components'] == ['FF', 'FA', 'AF', 'AA']
    assert full['lo'] == energy[u != 0].min()-energy[f != 0].max()
    assert full['hi'] == energy[u != 0].max()-energy[f != 0].min()
    assert full['hi'] > 10*np.ptp(energy[:, 2:4])
    left, right = endpoints(full, rho)
    sites = np.array([-.37+.23j, .19+.31j])
    for iq in range(len(psi)):
        partner = (np.arange(len(psi))-iq) % len(psi)
        def value(l, r):
            return (pair_frequency(pair_spectrum(psi, energy, l, r, partner), sites, .031)
                    -pair_frequency(pair_spectrum(psi, energy, r, l, partner), sites, .031))
        direct = value(left, right)
        separate = sum(value(*endpoints(b, rho)) for b in blocks)
        np.testing.assert_allclose(direct, separate, rtol=3e-13, atol=3e-14)
        # Far virtual columns remain in the actual operand and bound.
        assert np.any(left[:, -1] != 0)


def test_default_block_arrays_and_component_gates_are_unchanged():
    from gw.response_bank import response_blocks
    from gw.shared_pole_screening import _general_response_blocks
    _, energy, rho = case(.07j)
    blocks, f, u = partitions(energy, rho, 'blocks')
    old = response_blocks(energy, f, u, (2, 4), allow_crossing=True, exact_endpoint_support=True)
    default = _general_response_blocks(energy, f, u, (2, 4))
    for a, b, c in zip(old, blocks, default):
        for key in ('f', 'u', 'refs'):
            np.testing.assert_array_equal(a[key], b[key])
            np.testing.assert_array_equal(a[key], c[key])
        assert a['lo'] == b['lo'] == c['lo'] and a['hi'] == b['hi'] == c['hi']
    f[:, 0], u[:, 0] = .37, .63
    for partition in ('blocks', 'full'):
        with pytest.raises(ValueError, match='response_blocks_frozen_fractional'):
            _general_response_blocks(energy, f, u, (2, 4), partition)
    with pytest.raises(ValueError, match='response_partition'):
        _general_response_blocks(energy, f, u, (2, 4), 'auto')


def test_new_nonzero_coherent_column_expands_full_rule_interval():
    from gw.response_bank import full_density_sample_rules
    _, energy, rho = case()
    old = partitions(energy, rho, 'full')[0][0]
    held = {}
    first = full_density_sample_rules(energy, np.array([.4j]), gamma_ry=.03,
        interval_ry=(old['lo'], old['hi']), paired=True, values_only=True, held=held)
    rho[:, 2, 3] = 1e-30; rho[:, 3, 2] = 1e-30
    current = partitions(energy, rho, 'full')[0][0]
    assert current['lo'] < old['lo']
    second = full_density_sample_rules(energy, np.array([.4j]), gamma_ry=.03,
        interval_ry=(current['lo'], current['hi']), paired=True, values_only=True, held=held)
    assert second is not first and held['interval_ry'][0] <= current['lo']


def test_logical750_full_bounds_keep_far_virtuals_but_not_inert_padding():
    from gw.shared_pole_screening import _general_response_blocks
    energy = np.broadcast_to(np.linspace(-8., 6., 752), (2, 752)).copy()
    energy[:, 750:] = 1e8
    f = np.zeros_like(energy); f[:, :130] = 1.
    u = np.zeros_like(energy); u[:, 130:750] = 1.
    full = _general_response_blocks(energy, f, u, (98, 144), 'full')[0]
    assert full['hi'] == energy[:, 749].max()-energy[:, 0].min()
    assert full['lo'] == energy[:, 130].min()-energy[:, 129].max()
    assert full['u'][:, 749].all() and not full['u'][:, 750:].any()
    assert full['components'] == ['FF', 'FA', 'AF', 'AA']
