"""Native Si scalar-rule regression; independent final value/slope replay."""
import numpy as np


def test_native_interval_mass_refinement_on_independent_spectrum():
    import minimax

    lo, hi = .04499315372020445, 30.208688943498455
    z = 4.607368602009248 + .19109640588867596j
    rules = minimax.response_group_rules(lo, hi, np.array([z]), rel_tol=1e-8)
    assert len(rules) == 1 and rules[0]['members'] == [0]
    rule = rules[0]
    count = rule['count']
    assert 0 < count <= len(rule['t']) == 384
    assert rule['reference_ry'] == lo
    assert np.all(np.isfinite(rule['t'])) and np.all(rule['t'].real >= 0)
    assert np.all(rule['t'][count:] == 0)
    assert np.all(rule['value'][...,count:] == 0)
    assert np.all(rule['derivative'][...,count:] == 0)
    assert np.all(np.isfinite(rule['value'])) and np.all(np.isfinite(rule['derivative']))
    d = np.unique(np.r_[lo, hi, np.linspace(lo, hi, 10003),
                       (lo+hi)/2+(hi-lo)/2*np.cos(np.pi*(np.arange(10007)+.371)/10007),
                       z.real+z.imag*np.linspace(-8.001, 8.003, 8009)])
    d = d[(d>=lo)&(d<=hi)]
    # Signed spectral sums test physical cancellation without normalizing by
    # an arbitrarily small exact sum. Their total variation is exactly one.
    weights = np.sin(np.arange(len(d))*.731)
    weights /= np.sum(abs(weights))
    for side in (0, 1):
        times = rule['t'][:count]
        if side: times = times.conj()
        basis = np.exp(-(d[:,None]-lo)*times)
        values = basis@rule['value'][0,side,:count]
        slopes = basis@rule['derivative'][0,side,:count]
        exact = 1/(d+(-z if side==0 else z))
        exact_ds = (1 if side==0 else -1)*exact**2/(2*z)
        assert z.imag*np.max(abs(values-exact)) <= 5e-9
        assert z.imag**3*np.max(abs(slopes-exact_ds)) <= 5e-9
        assert z.imag*abs(weights@(values-exact)) <= 5e-9
        assert z.imag**3*abs(weights@(slopes-exact_ds)) <= 5e-9
        assert z.imag*np.sum(abs(rule['value'][0,side,:count])) <= 5000
        assert np.isfinite(z.imag**3*np.sum(abs(rule['derivative'][0,side,:count])))
