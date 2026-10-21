"""Public complex-box value rules and exact common-chi operands, tiny CPU."""
import dataclasses
import numpy as np
import pytest


@pytest.mark.parametrize('interval,sites', [
    ((.0008043023522215642, 5.883438242507291), np.array([.1615+.0183746544j, .646+.0183746544j])),
    ((-.7, .7), np.array([-.3+.08j, .42+.13j])),
    ((2., 5.), np.array([.2+.3j, .4+.7j]))])
def test_both_value_primitives_and_zero_growth_reference_all_complex_nodes(interval, sites):
    from gw.response_bank import full_density_sample_rules
    gamma = .0036749308824745378
    rules = full_density_sample_rules(np.array([[interval[0], interval[1]]]), sites,
        gamma_ry=gamma, interval_ry=interval, paired=True, values_only=True)
    d = np.linspace(*interval, 1001)
    actual = np.zeros((2, len(sites), len(d)), complex)
    for rule in rules:
        assert rule['construction'] == 'analytic_box_values_only'
        assert rule['analytic_box_certificate']['requested_targets'] == ['value']
        for side in (0, 1):
            t = rule['times'].conj() if side else rule['times']
            # Existing native endpoint references use the actual minimum or
            # maximum gap at each node. The origin correction NEVER grows.
            reference = np.where(t.real >= 0, interval[0], interval[1])
            correction = np.exp((rule['reference_ry']-reference)*t)
            assert np.isfinite(correction).all() and np.max(abs(correction)) <= 1+2e-13
            phase = np.exp(-(d[:, None]-reference[None])*t[None])
            assert np.isfinite(phase).all() and np.max(abs(phase)) <= 1+2e-13
            actual[side] += (rule['weights'][side]*correction[None])@phase.T
    z = sites+1j*gamma
    exact = np.stack([1/(z[:, None]-d), -1/(z[:, None]+d)])
    assert np.max(abs(actual-exact)*z.imag[None, :, None]) < 1e-8
    assert max(r['coefficient_mass'] for r in rules) < 5000


def test_default_derivative_rule_never_requests_value_only_box(monkeypatch):
    import minimax
    from gw.response_bank import full_density_sample_rules
    monkeypatch.setattr(minimax, 'analytic_box_rule', lambda *a, **k: pytest.fail('changed default'))
    common = dict(gamma_ry=.03, interval_ry=(2., 5.), paired=True)
    a = full_density_sample_rules(np.array([[0., 5.]]), np.array([.2+.4j]), **common)
    b = full_density_sample_rules(np.array([[0., 5.]]), np.array([.2+.4j]), values_only=False, **common)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x['times'], y['times'])
        np.testing.assert_array_equal(x['weights'], y['weights'])


@pytest.mark.parametrize('failure', ['certificate', 'mass'])
def test_unaccepted_public_box_is_not_admitted(monkeypatch, failure):
    import minimax
    from gw.response_bank import full_density_sample_rules
    real = minimax.analytic_box_rule
    if failure == 'certificate':
        monkeypatch.setattr(minimax, 'analytic_box_rule', lambda *a, **k:
            dataclasses.replace(real(*a, **k), sup_error=1.))
    else:
        monkeypatch.setattr(minimax, 'rule_roundoff_amplification', lambda *a, **k: 84.)
    with pytest.raises(ValueError, match='analytic value box certificate/mass'):
        full_density_sample_rules(np.array([[0., 1.]]), np.array([.2+.1j]),
            gamma_ry=.03, interval_ry=(0., 1.), paired=True, values_only=True)


@pytest.mark.parametrize('qgrid', [(4, 2, 1), (2, 2, 1)])
def test_actual_typed_common_stream_four_blocks_matches_literal_frequency_sum(monkeypatch, qgrid):
    import jax.numpy as jnp
    from ffi.fft import xla_reference
    from gw.greens_function_kernel import density_left_endpoint
    from gw.response_bank import full_density_sample_rules
    from gw.w_isdf import full_density_response_stream
    from tests.test_common_coherent_response_cpu import _fixture
    from tests._full_density_blocks_oracle import four_block_panels, pair_spectrum, pair_frequency
    from tests.test_chi_fejer_window_cpu import _literal
    mesh, grid, psi, energy, rho, parent = _fixture(monkeypatch)
    nk, nb, ns, mu = psi.shape
    coords = np.indices(grid).reshape(3, -1).T
    partners = [np.ravel_multi_index(((coords-q)%grid).T, grid) for q in coords]
    sites = np.array([-.17+.12j, .42+.18j]); gamma = .03
    endpoint = density_left_endpoint(mesh, parent.psi_mun.shape)
    for name, (left, right) in four_block_panels(rho, slice(1, 3)).items():
        support = [np.any(a != 0, axis=1) for a in (left, right)]
        bounds = tuple(v for mask in support for v in (energy[mask].min(), energy[mask].max()))
        lo, hi = bounds[0]-bounds[3], bounds[1]-bounds[2]
        bands = np.arange(nb)[None]
        ranges = np.array([[np.min(np.where(mask, bands, nb), axis=1),
                            np.max(np.where(mask, bands+1, 0), axis=1)] for mask in support])
        rules = full_density_sample_rules(energy, sites, gamma_ry=gamma,
            interval_ry=(lo, hi), paired=True, values_only=True)
        got = None
        l, r = (endpoint(parent.psi_mun, jnp.asarray(a)) for a in (left, right))
        with xla_reference():
            for rule in rules:
                stream = full_density_response_stream(mesh, grid, (nk, nb, mu, ns),
                    q_ids=range(np.prod(qgrid)), n_outputs=len(sites), k_unfold_plan=parent.plan,
                    qgrid=qgrid, response_window='fejer', rule_reference_ry=rule['reference_ry'],
                    pair_bounds_ry=bounds, pair_band_ranges=ranges)
                value = np.asarray(stream(jnp.asarray(rule['times']), jnp.asarray(rule['weights']),
                    l, parent.psi_nmu, r, parent.enk))
                got = value if got is None else got+value
        want = np.array([pair_frequency(pair_spectrum(psi, energy, left, right, p), sites, gamma)
                         -pair_frequency(pair_spectrum(psi, energy, right, left, p), sites, gamma)
                         for p in partners])*np.sqrt(nk)
        if qgrid != grid:
            lattice = np.fft.ifftn(want.reshape(*grid, len(sites), mu, mu), axes=(0, 1, 2), norm='ortho')
            want = _literal(lattice.reshape(nk, len(sites), mu, mu), grid, qgrid)
        np.testing.assert_allclose(got, want, rtol=4e-7, atol=3e-10, err_msg=name)


def test_native_proof_frequency_oracle_and_typed_config_fixture():
    import importlib.util
    import sys
    from pathlib import Path
    from tests.test_full_density_blocks_oracle_cpu import case
    from tests._full_density_blocks_oracle import pair_spectrum, pair_frequency
    directory = Path(__file__).parent/'tdagw/cri3_qgrid_750_campaign/diagnostics'
    spec = importlib.util.spec_from_file_location('complex_box_native_proof',
        directory/'verify_complex_box_chi_p4.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    psi, energy, rho, partners = case(810)
    grid = (3, 1, 1); coords = np.indices(grid).reshape(3, -1).T
    left, right = np.eye(4)[None]-rho, rho
    sites = np.array([-.17+.12j, .42+.18j]); gamma = .03
    want = np.array([pair_frequency(pair_spectrum(psi, energy, left, right, p), sites, gamma)
                    -pair_frequency(pair_spectrum(psi, energy, right, left, p), sites, gamma)
                    for p in partners])*np.sqrt(len(psi))
    np.testing.assert_allclose(module.frequency_literal(psi, energy, left, right,
        coords, grid, sites, gamma), want, rtol=2e-13, atol=2e-14)
    sys.path.insert(0, str(directory))
    try:
        from verify_common_coherent_chi_p4 import bank_config
        config = bank_config()
        assert config.backend.linalg is not None
    finally:
        sys.path.remove(str(directory))
