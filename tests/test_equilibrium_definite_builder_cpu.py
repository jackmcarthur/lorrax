"""Same-state finite/infinity builder algebra; no material admission claim."""
import numpy as np
import pytest

from tests.test_general_frequency_program_cache_cpu import programs


@pytest.fixture
def equilibrium_programs(programs):
    from gw.shared_pole_screening import _equilibrium_frequency_programs
    _equilibrium_frequency_programs.cache_clear()
    yield (*programs, _equilibrium_frequency_programs)
    _equilibrium_frequency_programs.cache_clear()


def data():
    rng = np.random.default_rng(720)
    plus = rng.normal(size=(2, 4, 2))+1j*rng.normal(size=(2, 4, 2))
    omega = np.array([[.6, 1.4], [.8, 1.9]])
    j = np.diag([1., 1., -1., -1.])
    c = np.array([np.concatenate((plus[q], plus[1-q].conj()), axis=1) for q in range(2)])
    m = np.array([np.diag(np.r_[omega[q], omega[1-q]]) for q in range(2)])
    gamma = .037
    sites = np.repeat([.21+.18j, -.21+.18j, .47j, .72+.22j, -.72+.22j], 2)
    directions = rng.normal(size=(2, 4, len(sites)))+1j*rng.normal(size=(2, 4, len(sites)))
    qi = rng.normal(size=(2, 4, 1))+1j*rng.normal(size=(2, 4, 1))
    def exact(q, z):
        return c[q]@np.linalg.solve((z+1j*gamma)*j-m[q], c[q].conj().T)
    output = np.array([np.stack([exact(q, z)@directions[q, :, i]
                                for i, z in enumerate(sites)], axis=-1) for q in range(2)])
    states = np.array([np.stack([np.linalg.solve((z+1j*gamma)*j-m[q],
        c[q].conj().T@directions[q, :, i]) for i, z in enumerate(sites)], axis=-1) for q in range(2)])
    c1 = c@j@c.conj().swapaxes(-1, -2)
    c2 = c@j@m@j@c.conj().swapaxes(-1, -2)
    return locals()


def test_actual_finite_and_infinity_owners_form_same_state_latent_grams(equilibrium_programs):
    from gw.shared_pole_constructor import reduce_general_pencil
    factory, compiled, equilibrium = equilibrium_programs
    existing = len(compiled)
    pencil, infinity, adjoint, _ = factory('mesh', (None, 'x', 'y'), ('native',))
    common, output_gram = equilibrium('mesh', (None, 'x', 'y'), ('native',))
    a = data()
    q, o, zeta = a['directions'], a['output'], a['sites']+1j*a['gamma']
    g, h, b, c = common(zeta, q, o)
    for actual, old in zip((g, h, b, c), pencil(zeta.conj(), zeta, q, adjoint(o), q, o)):
        np.testing.assert_array_equal(actual, old)
    states, signature, metric = a['states'], a['j'], a['m']
    np.testing.assert_allclose(g, states.conj().swapaxes(-1, -2)@signature@states, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(h, states.conj().swapaxes(-1, -2)@metric@states, rtol=2e-13, atol=2e-13)
    # Current gamma moments must be UNDAMPED before the infinity state joins
    # this same space. C1 need not vanish and W(q,z) need not be even in z.
    c2_gamma = a['c2']-1j*a['gamma']*a['c1']
    assert min(np.linalg.norm(x) for x in a['c1']) > .1
    joined = infinity(zeta.conj(), zeta, g, h, b, c, q, adjoint(o), q, o,
        a['qi'], a['c1'], c2_gamma+1j*a['gamma']*a['c1'])
    states = np.concatenate((states, signature@a['c'].conj().swapaxes(-1, -2)@a['qi']), axis=-1)
    gg, hh, bb, cc = joined
    np.testing.assert_allclose(gg, states.conj().swapaxes(-1, -2)@signature@states, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(hh, states.conj().swapaxes(-1, -2)@metric@states, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(cc, a['c']@states, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(bb, cc.conj().swapaxes(-1, -2), rtol=3e-13, atol=3e-13)
    held = [-.43+.2j, .31+.04j, 1.5+.08j]
    grams = output_gram(cc)
    np.testing.assert_array_equal(grams, cc.conj().swapaxes(-1, -2)@cc)
    models = []
    for iq in range(2):
        model = reduce_general_pencil(gg[iq], hh[iq], max_order=4, rank_tolerance=1e-10,
            structure=dict(kind='equilibrium_definite', gamma_ry=a['gamma']),
            output_gram=grams[iq])
        models.append(model)
        l, r = cc[iq]@model['output_map'], cc[iq]@model['input_map']
        for z in held:
            value = (l/(z-model['poles_ry']))@r.conj().T
            np.testing.assert_allclose(value, a['exact'](iq, z), rtol=2e-10, atol=2e-10)
        np.testing.assert_array_equal(model['poles_ry'].imag, np.full(4, -a['gamma']))
    for z in held:
        np.testing.assert_allclose(a['exact'](0, z), a['exact'](1, -z.conjugate()).conj(),
                                   rtol=3e-13, atol=3e-13)
    assert np.linalg.norm(a['exact'](0, held[0])-a['exact'](0, -held[0])) > .1
    assert len(compiled)-existing == 6  # Four old programs, two opt-in programs.


@pytest.mark.parametrize('mode,expected', [('same', True), ('coarse_fejer', True),
    ('coarse_sharp', False), ('coherence', False), ('fractional', False), ('inversion', False)])
def test_exact_member_eligibility_ignores_padding_not_current_physics(mode, expected):
    from gw.shared_pole_screening import _equilibrium_definite_eligibility
    energy = np.array([[-1., -.4, .4, 1., 1e8], [-1.1, -.3, .5, 1.1, 1e8]]*2)
    rho = np.broadcast_to(np.diag([1., 1., 0., 0., 0.]), (4, 5, 5)).astype(complex).copy()
    grid, qgrid, window = (4, 1, 1), (4, 1, 1), 'sharp'
    if mode.startswith('coarse'):
        qgrid, window = (2, 1, 1), mode.split('_')[1]
    elif mode == 'coherence':
        rho[:, 1, 2] = 1e-30; rho[:, 2, 1] = 1e-30
    elif mode == 'fractional':
        rho[:, 1, 1] = .5
    elif mode == 'inversion':
        energy[:, 2] = -2.
    receipt = _equilibrium_definite_eligibility(energy, rho, 4, grid, qgrid, window)
    assert receipt['eligible'] is expected


@pytest.mark.parametrize('sample_error', [1e-10, 1e-9, 1e-8])
def test_redundant_passive_sample_noise_is_not_a_psd_repair_license(equilibrium_programs, sample_error):
    from gw.shared_pole_constructor import reduce_general_pencil
    factory, _, equilibrium = equilibrium_programs
    pencil, infinity, adjoint, _ = factory('mesh', (None, 'x', 'y'), ('native',))
    common, output_gram = equilibrium('mesh', (None, 'x', 'y'), ('native',))
    a = data(); rng = np.random.default_rng(9931)
    q, sites, gamma = a['directions'], a['sites'], a['gamma']
    errors = np.zeros((2, len(sites), 4, 4), complex)
    # Perturb the sampled W MATRICES, not G/H, and preserve the same actual
    # q/-q + negative-frequency covariance used by the builder's mirrors.
    for z in np.unique(sites):
        if z.real < 0:
            continue
        dw = rng.normal(size=(2, 4, 4))+1j*rng.normal(size=(2, 4, 4))
        dw /= np.linalg.norm(dw, axis=(-2, -1))[:, None, None]
        dw *= sample_error*max(np.linalg.norm(a['exact'](i, z)) for i in range(2))
        if z.real == 0:
            dw[1] = dw[0].conj()
        for i in range(2):
            errors[i, sites == z] = dw[i]
            errors[i, sites == -z.conjugate()] = dw[1-i].conj()
    output = a['output']+np.einsum('qimn,qni->qmi', errors, q)
    zeta = sites+1j*gamma
    g, h, b, c = common(zeta, q, output)
    g, h, b, c = infinity(zeta.conj(), zeta, g, h, b, c, q, adjoint(output), q, output,
        a['qi'], a['c1'], a['c2'])
    record = []
    grams = output_gram(c)
    for iq in range(2):
        minimum = float(np.linalg.eigvalsh((h[iq]+h[iq].conj().T)*.5)[0])
        with pytest.raises(ValueError, match='resolved negative H') as refused:
            reduce_general_pencil(g[iq], h[iq], max_order=4, rank_tolerance=1e-10,
                structure=dict(kind='equilibrium_definite', gamma_ry=gamma),
                output_gram=grams[iq])
        record.append(dict(q=iq, H_min=minimum, refusal=str(refused.value)))
    print(dict(relative_sample_perturbation=sample_error, finite_columns=len(sites),
               infinity_columns=a['qi'].shape[-1], results=record))


def test_exact_eligibility_transitions_keep_distinct_held_space_namespaces():
    import ast
    from pathlib import Path
    import gw.shared_pole_screening as owner
    tree = ast.parse(Path(owner.__file__).read_text())
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == 'screen_general_shared_poles')
    expressions = {}
    for node in ast.walk(builder):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('directions', 'spaces')
                and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
                and node.value.func.attr == 'setdefault'):
            expressions[node.targets[0].id] = compile(ast.Expression(node.value), owner.__file__, 'eval')
    assert set(expressions) == {'directions', 'spaces'}
    cache = {}; previous = {}
    for definite in (True, False, True):
        current = {key: eval(code, dict(fit_basis=cache, definite=definite))
                   for key, code in expressions.items()}
        if definite in previous:
            assert all(current[key] is previous[definite][key] for key in current)
            assert current['spaces'][0].shape == (7, 3)
        else:
            current['directions'][0] = np.ones((4, 3 if definite else 2))
            current['spaces'][0] = np.ones((7, 3) if definite else (5, 2))
            previous[definite] = current
    assert previous[True]['directions'] is not previous[False]['directions']
    assert previous[True]['spaces'] is not previous[False]['spaces']
