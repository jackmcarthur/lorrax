"""Adversarial contracts for AFIXES; tiny arrays, plus a separate real-P4 gate."""
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest


def _rule():
    from minimax import UniformRule
    return UniformRule(times=np.array([.1+.02j]), weights=np.array([.3-.01j]),
        box=(-2., -.3, .05, .4), eps=1e-4, relative=True,
        theta_deg=5., rank=1, sup_error=5e-5, kappa_max=1., seconds=0.)


def test_cache_one_ulp_request_reuses_authenticated_stored_eps(tmp_path):
    from gw.sigma_box_plan import _rule_cache_store, _rule_cache_lookup
    rule = _rule()
    _rule_cache_store(str(tmp_path), rule, 1.)
    for eps in (np.nextafter(rule.eps, 0), np.nextafter(rule.eps, np.inf)):
        best, warnings = _rule_cache_lookup(str(tmp_path), rule.box, eps, True,
                                           noise_amplification_cap=1e9)
        assert not warnings and best is not None
        assert best[0].eps == rule.eps


@pytest.mark.parametrize('field', ['eps', 'relative'])
def test_cache_authenticates_stored_fields_before_filtering(tmp_path, field):
    from gw.sigma_box_plan import _rule_cache_store, _rule_cache_lookup
    rule = _rule()
    _rule_cache_store(str(tmp_path), rule, 1.)
    path, = tmp_path.glob('rule_*.npz')
    with np.load(path) as data:
        values = dict(data)
    values[field] = np.nextafter(rule.eps, np.inf) if field == 'eps' else False
    np.savez(path, **values)
    best, warnings = _rule_cache_lookup(str(tmp_path), rule.box, rule.eps, True,
                                       noise_amplification_cap=1e9)
    assert best is None and len(warnings) == 1
    assert 'sigma_rule_integrity' in warnings[0]


def test_restart_local_open_error_agrees_before_payload(monkeypatch, tmp_path):
    from file_io import tagged_arrays as tagged, shared_pole_store as store
    events = []
    def broken(*args, **kwargs):
        raise OSError('injected rank-local open')
    def agree(error, **kwargs):
        events.append(('agree', error))
        assert isinstance(error, OSError)
        raise RuntimeError('agreed injected rank-local open')
    def payload(*args, **kwargs):
        pytest.fail('collective payload entered after local read failure')
    monkeypatch.setattr(tagged.h5py, 'File', broken)
    monkeypatch.setattr(tagged, 'agree_io_error', agree)
    monkeypatch.setattr(store, 'validate_shared_pole_model', payload)
    with pytest.raises(tagged.SharedPoleMemberRefused, match='injected rank-local open'):
        tagged.read_shared_pole_restart_member(tmp_path/'restart.h5',
            expected_identity={}, mesh_xy=object())
    assert len(events) == 1


def test_digest_empty_addressable_shards(monkeypatch):
    from file_io import shared_pole_store as store
    class IO:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read_slab(self, name, **kwargs):
            return np.ones((1, 1)) if name == 'poles2_ry2' else NS(addressable_shards=[])
    monkeypatch.setattr(store, 'SlabIO', IO)
    monkeypatch.setattr(store, '_check_io_capacity', lambda *args: None)
    monkeypatch.setattr(store, '_admit', lambda *args, **kwargs: None)
    monkeypatch.setattr(store, '_check_factor', lambda *args: None)
    monkeypatch.setattr(store, 'psum_replicate', lambda value, mesh: value)
    result = store._model_digest('unused', dict(n_mu_logical=1, Kmax=1,
        n_q_irr=1, K=[1]), NS(size=1, shape={'x':1,'y':1}), capacity=object())
    assert len(result) == 64


def test_pencil_alias_and_distinct_panels():
    import jax.numpy as jnp
    from gw.shared_pole_constructor import finite_pencil_column
    rng = np.random.default_rng(714)
    arrays = [jnp.asarray(rng.normal(size=(1, 4, 3)) +
                          1j*rng.normal(size=(1, 4, 3))) for _ in range(5)]
    q, o, d, qr, ore = arrays
    s = jnp.array([.2+.1j, .4+.1j, .8+.1j])
    calls = []
    def mm(a, b, transa='N'):
        calls.append(1)
        return (jnp.swapaxes(a.conj(), -1, -2) if transa == 'C' else a) @ b
    for right_q, right_o, expected_calls in ((q,o,2),(qr,ore,3)):
        calls.clear()
        g, h = finite_pencil_column((s,q,o), (s,right_q,right_o,d), matmul=mm)
        assert len(calls) == expected_calls
        a = np.swapaxes(np.asarray(o).conj(), -1, -2) @ np.asarray(right_q)
        b = np.swapaxes(np.asarray(q).conj(), -1, -2) @ np.asarray(right_o)
        expected = (a-b)/(np.asarray(s)[None,None,:]-np.asarray(s).conj()[None,:,None])
        np.testing.assert_allclose(g, expected, atol=1e-12)
        np.testing.assert_allclose(h, np.asarray(s)[None,None,:]*expected-a, atol=1e-12)


@pytest.mark.parametrize('key', ['zeta_nband','mpa_sampling_alpha',
    'sigma_quadrature_reduction_steps','number_bands_chi','number_bands_sigma','nband'])
def test_nullable_integer_none_is_uniform(tmp_path, key):
    from gw.gw_config import read_lorrax_input
    path=tmp_path/'input.in'
    path.write_text('[cohsex]\n')
    baseline = read_lorrax_input(str(path))[key]
    path.write_text('[cohsex]\n'+key+' = none\n')
    # Band-count fields are subsequently resolved by their existing owner;
    # explicit None must behave like the absent nullable input there too.
    assert read_lorrax_input(str(path))[key] == baseline


def test_output_source_fails_before_any_screening_work():
    from gw.shared_pole_screening import screen_shared_poles
    with pytest.raises(ValueError, match='before screening'):
        screen_shared_poles(None, None, None, NS(write_w=True, write_poles=False),
            mesh_xy=None, sym=None, centroid_indices=None, run_dir=None,
            label=None, wfn=NS(_filename='private-is-not-public'),
            wfn_fingerprint_binding=None, tensors_filename=None,
            occupation_state=None, print_fn=print)


@pytest.mark.parametrize('family,description', [('fd','kBT'),('mp1','half-width')])
def test_smearing_width_diagnostic_names_family(family, description):
    from gw.efermi import solve_smearing_occupations
    with pytest.raises(ValueError) as caught:
        solve_smearing_occupations(np.zeros((1,2)), [1.], 1., 0.,
            state_capacity=2., family=family)
    assert 'solve_smearing_occupations:' in str(caught.value)
    assert description in str(caught.value)
    assert 'solve_mp1_occupations:' not in str(caught.value)


def test_export_empty_deck_source_refuses_during_parsing(tmp_path):
    from gw.gw_config import read_lorrax_input
    path = tmp_path/'empty_source.in'
    path.write_text('[cohsex]\ncompute_mode = mpa\nsigma_w_model = shared_pole\nwrite_w = true\nwfn_file =\n')
    with pytest.raises(ValueError, match='nonempty wfn_file'):
        read_lorrax_input(str(path))
