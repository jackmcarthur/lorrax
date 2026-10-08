"""Bounded held-sample evaluation preserves the causal derivative and refusal."""
import numpy as np
import pytest
import jax
import jax.numpy as jnp

from gw.shared_pole_local import evaluate_round_held_samples
from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b


def mm(a, b, *, transa='N', transb='N'):
    if transa == 'C':
        a = a.swapaxes(-1, -2).conj()
    if transb == 'C':
        b = b.swapaxes(-1, -2).conj()
    return a @ b


@pytest.mark.parametrize('ordered', [False, True])
def test_held_derivative_matches_finite_difference(ordered):
    jax.config.update('jax_enable_x64', True)
    factor = np.array([[1., .2], [.3, .8]], complex)
    nodes = np.array([1.1+.3j, 2.1+.4j])
    poles = np.array([.7, 2.9])
    mu = np.array([-1.2, .9])

    def value(z):
        weights = 1/(z*mu-1) if ordered else 1/(z*z-poles)
        return sum(w*np.outer(factor[:, i], factor[:, i].conj())
                   for i, w in enumerate(weights))

    step = 1e-5
    wc = np.array([value(z) for z in nodes])[None]
    dw = np.array([(value(z+step)-value(z-step))/(2*step)/(2*z)
                   for z in nodes])[None]
    model = tuple(map(jnp.asarray, (factor[None], poles[None], np.ones((1, 2), bool))))
    signed = tuple(map(jnp.asarray, (factor[None], mu[None], np.ones((1, 2), bool)))) if ordered else ()
    errors, reciprocity, evaluated = evaluate_round_held_samples(
        model, signed, (jnp.asarray(wc), jnp.asarray(dw)), nodes=nodes,
        matmul=mm, gates=shared_real_pole_gates_v1_r3b, ordered=ordered,
        return_values=True)
    assert np.max(np.asarray(errors)) < 2e-9
    assert evaluated.shape == (1, 2, 2, 2, 2)
    np.testing.assert_allclose(evaluated[:, 0], wc, rtol=3e-15, atol=3e-15)
    if ordered:
        assert reciprocity == {}
    else:
        assert np.asarray(reciprocity['passed']).all()


def test_asymmetric_residue_refuses_symmetric_held_reference():
    jax.config.update('jax_enable_x64', True)
    factor = jnp.asarray([[[1.+0j], [.2+.7j]]])
    model = (factor, jnp.asarray([[2.]]), jnp.asarray([[True]]))
    nodes = jnp.asarray([1.+.3j])
    symmetric = jnp.eye(2, dtype=jnp.complex128)[None, None]
    errors, reciprocity = evaluate_round_held_samples(
        model, (), (symmetric, symmetric), nodes=nodes,
        matmul=mm, gates=shared_real_pole_gates_v1_r3b, ordered=False)
    assert np.asarray(reciprocity['applicable']).all()
    assert not np.asarray(reciprocity['passed']).any()
    assert np.asarray(errors).min() > .1
