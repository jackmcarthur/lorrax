"""Host lifecycle/ordering oracle; NOT a chi-kernel or GPU qualification."""
from types import SimpleNamespace
import time

import numpy as np
import pytest


@pytest.fixture
def stage(monkeypatch):
    import jax
    import jax.numpy as jnp
    import gw.response_bank as rules_owner
    import gw.w_isdf as stream_owner
    from gw.shared_pole_screening import _general_current_site_samples
    monkeypatch.setattr(jax, 'process_index', lambda: 0)
    monkeypatch.setattr(jax, 'block_until_ready', lambda values: values)
    monkeypatch.setattr(jnp, 'asarray', np.asarray)
    planned, planned_energies, dispatched, dyson_calls, events = [], [], [], [], []
    def rules(energy, sites, **kwargs):
        planned.append(np.asarray(sites).copy())
        planned_energies.append(energy.copy())
        assert kwargs['paired'] and kwargs['values_only']
        kwargs['held']['seen_sites'] = np.asarray(sites).copy()
        out = []
        for fraction in (.25, .75):
            weights = np.zeros((2, len(sites), 384), complex)
            weights[0, :, 0] = fraction*(sites+1)
            out.append(dict(members=list(range(len(sites))), times=np.zeros(384, complex),
                weights=weights, reference_ry=.1, active_count=1, carrier_capacity=384,
                sampled_error=1e-10, coefficient_mass=1.))
        return out
    def stream(mesh, grid, shape, **kwargs):
        def run(times, weights, left, right_face, right, energy):
            dispatched.append(dict(kwargs, right_face=right_face, energy=energy,
                                   left=left.copy(), right=right.copy()))
            panel = np.einsum('ksmb,ksnb->mn', left, right.conj())
            scale = weights.sum(axis=(0, 2))
            return np.array([np.array([panel*w*(q+1) for w in scale]) for q in kwargs['q_ids']])
        return run
    monkeypatch.setattr(rules_owner, 'full_density_sample_rules', rules)
    monkeypatch.setattr(stream_owner, 'full_density_response_stream', stream)
    rng = np.random.default_rng(932)
    energy = np.array([[-2., -.4, .2, 1.5], [-2.1, -.35, .25, 1.6]])
    rho = np.broadcast_to(np.diag([1., .7, .3, 0.]), (2, 4, 4)).astype(complex).copy()
    rho[:, 1, 2] = .07j; rho[:, 2, 1] = -.07j
    pm = rng.normal(size=(2, 1, 2, 4))+1j*rng.normal(size=(2, 1, 2, 4))
    ph = np.einsum('ksmb,kbc->ksmc', pm, rho)
    def dyson(root, value):
        dyson_calls.append(value.copy())
        return root*value+3
    current = dict(energy=energy, rho=rho, occupations=rho.diagonal(axis1=-2, axis2=-1).real,
        active=slice(1, 3), response_active=slice(1, 3), stop=4, nb=4, gamma=.03,
        recipe=dict(bank_rule_tolerance=1e-8), label='host-spy', pm=pm, ph=ph,
        pn=pm.transpose(0, 3, 1, 2), energy_d=energy, mesh_xy=object(), grid=(2, 1, 1),
        qgrid=(2, 1, 1), nk=2, nmu=2, ns=1, layout='face', response_window='sharp',
        query_q_ids=np.array([0, 1]), plan=object(), execute_response=object(),
        response_partition='blocks', dyson=SimpleNamespace(value=dyson), H=np.full((2, 2, 2), 2.))
    phase_start = {}
    def begin(name):
        events.append(('begin', name)); phase_start[name] = time.monotonic()
    def end(name, values=None):
        events.append(('end', name)); phase_start.pop(name)
    def evaluate(sites, *, evaluate_dyson=True, cache=None):
        return _general_current_site_samples(current, np.asarray(sites, complex),
            scalar_cache={} if cache is None else cache, begin=begin, end=end,
            phase_start=phase_start, print_fn=lambda *a: None, evaluate_dyson=evaluate_dyson)
    return locals()


def expected_raw(current, sites):
    from gw.shared_pole_screening import _density_endpoint_column_support, _general_response_blocks
    f, u = _density_endpoint_column_support(current['rho'], current['stop'],
        occupations=current['occupations'], active=current['response_active'])
    blocks = _general_response_blocks(current['energy'], f, u, (1, 3), 'blocks')
    output = [None]*len(sites)
    for block in blocks:
        ld, rd = block['density_sides']
        left = (current['pm']-current['ph'] if ld else current['pm'])*block['u'][:, None, None]
        right = (current['ph'] if rd else current['pm'])*block['f'][:, None, None]
        panel = np.einsum('ksmb,ksnb->mn', left, right.conj())
        for fraction in (.25, .75):
            for j, z in enumerate(sites):
                value = np.array([panel*(fraction*(z+1))*(q+1) for q in current['query_q_ids']])
                output[j] = value if output[j] is None else output[j]+value
    return output


def test_initial_stage_preserves_endpoint_order_sums_and_single_dyson_per_site(stage):
    sites = np.array([.1+.1j, 1.2+.5j, .4+.12j])
    raw = expected_raw(stage['current'], sites)
    result = stage['evaluate'](sites)
    for value, expected in zip(result['values'], raw):
        np.testing.assert_array_equal(value, stage['current']['H']*expected+3)
    assert len(stage['dyson_calls']) == len(sites)
    assert stage['events'] == [(a, p) for p in ('planning', 'stream', 'dyson') for a in ('begin', 'end')]
    for call in stage['dispatched']:
        assert call['k_unfold_plan'] is stage['current']['plan']
        assert call['execute'] is stage['current']['execute_response']
        assert call['right_face'] is stage['current']['pn']
        assert call['energy'] is stage['current']['energy_d']
        np.testing.assert_array_equal(call['q_ids'], [0, 1])


def test_raw_epsilon_query_never_executes_dyson(stage):
    stage['current']['query_q_ids'] = np.array([0])
    sites = np.array([-.16+1.006j])
    result = stage['evaluate'](sites, evaluate_dyson=False)
    np.testing.assert_array_equal(result['values'], expected_raw(stage['current'], sites))
    assert not stage['dyson_calls']
    assert stage['events'] == [(a, p) for p in ('planning', 'stream') for a in ('begin', 'end')]
    assert all(np.array_equal(call['q_ids'], [0]) for call in stage['dispatched'])


def test_incremental_query_evaluates_only_new_sites_and_keeps_old_values(stage):
    cache = {}; first_sites = np.array([.2+.1j, .4+.2j])
    first = stage['evaluate'](first_sites, cache=cache)
    old_values = list(first['values']); copies = [v.copy() for v in old_values]
    stage['planned'].clear(); stage['dispatched'].clear(); stage['dyson_calls'].clear()
    new_sites = np.array([.8+.3j])
    second = stage['evaluate'](new_sites, cache={})
    assert len(second['values']) == 1 and len(stage['dyson_calls']) == 1
    assert all(np.array_equal(z, new_sites) for z in stage['planned'])
    for value, previous, copy in zip(first['values'], old_values, copies):
        assert value is previous
        np.testing.assert_array_equal(value, copy)


def test_new_current_energy_and_full_density_are_runtime_operands(stage):
    sites = np.array([.2+.1j])
    first = stage['evaluate'](sites)
    current = stage['current']
    current['energy'] = current['energy']+np.array([.03, -.01, .02, -.02])[None]
    current['energy_d'] = current['energy']
    current['rho'] = current['rho'].copy()
    current['rho'][:, 1, 2] *= 2; current['rho'][:, 2, 1] *= 2
    current['ph'] = np.einsum('ksmb,kbc->ksmc', current['pm'], current['rho'])
    stage['dispatched'].clear(); stage['planned_energies'].clear()
    second = stage['evaluate'](sites)
    expected = [current['H']*v+3 for v in expected_raw(current, sites)]
    np.testing.assert_array_equal(second['values'], expected)
    assert not np.array_equal(first['values'][0], second['values'][0])
    assert all(np.array_equal(e, current['energy']) for e in stage['planned_energies'])
    assert all(call['energy'] is current['energy_d'] for call in stage['dispatched'])
