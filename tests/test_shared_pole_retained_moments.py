"""Independent projected moment checks for the exact-size pencil owner."""

from functools import partial

import numpy as np


def test_retained_moments_preserve_both_operators_and_current_values():
    """Check noncommuting pencils, inactive columns and a changed H."""
    import jax
    import jax.numpy as jnp
    from gw.shared_pole_constructor import retained_moment_identity

    rng = np.random.default_rng(901)

    def complex_array(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    x = complex_array((2, 8, 8))
    g = x @ x.conj().swapaxes(-1, -2) + np.eye(8)
    x = complex_array((2, 8, 8))
    h = (x + x.conj().swapaxes(-1, -2)) / 2
    coefficients = complex_array((2, 8, 5))
    active = np.array([[True, False, True, True, False],
                       [False, True, True, False, True]])
    poles = np.array([[1., 2., 3., 4., 5.], [6., 5., 4., 3., 2.]])
    selector = np.broadcast_to(np.eye(8)[:, -3:], (2, 8, 3)).copy()
    output = complex_array((2, 3, 8))
    model = (complex_array((2, 3, 5)), poles, active)

    def mm(left, right, *, transa='N'):
        if transa == 'C':
            left = left.conj().swapaxes(-1, -2)
        return jnp.matmul(left, right)

    kernel = jax.jit(partial(retained_moment_identity, matmul=mm))
    y = coefficients * active[:, None, :]
    a = y.conj().swapaxes(-1, -2) @ g @ selector
    projected = y @ a
    for current_h in (h, h + .3 * g):
        got = kernel((g, current_h, output), coefficients, model, selector)
        expected = {}
        for name, operator, weight in (('M1', g, np.ones_like(poles)),
                                       ('M3', current_h, poles)):
            target = projected.conj().swapaxes(-1, -2) @ operator @ projected / 2
            value = a.conj().swapaxes(-1, -2) @ (a * weight[:, :, None]) / 2
            expected[name] = np.linalg.norm(value-target, axis=(-2, -1)) / np.linalg.norm(target, axis=(-2, -1))
            np.testing.assert_allclose(got[name], expected[name], rtol=1e-12, atol=1e-12)
        assert not np.allclose(expected['M1'], expected['M3'])
