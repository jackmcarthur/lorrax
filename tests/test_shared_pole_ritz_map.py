"""Ritz reconstruction checked against a planted latent pole model."""

from functools import partial

import numpy as np


def test_ritz_map_reconstructs_physical_factors_with_rank_and_padding():
    """Recover five latent poles from seven active, dependent port columns."""
    import jax
    import jax.numpy as jnp
    from gw.shared_pole_constructor import reduce_shared_pole_pencil
    from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b as gates

    rng = np.random.default_rng(207)
    x = rng.normal(size=(5, 8)) + 1j * rng.normal(size=(5, 8))
    x[:, -1] = 0
    physical = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    poles = np.arange(1., 6.)
    g = x.conj().T @ x
    h = x.conj().T @ (poles[:, None] * x)
    output = physical @ x

    def mm(a, b, *, transa='N'):
        if transa == 'C':
            a = a.conj().swapaxes(-1, -2)
        return jnp.matmul(a, b)

    kernel = jax.jit(partial(reduce_shared_pole_pencil, eigh=jnp.linalg.eigh,
                             matmul=mm, gates=gates))
    model, diagnostics, y = kernel((g[None], h[None], output[None]),
                                   np.array([[True] * 7 + [False]]))
    b, recovered, active = map(np.asarray, model)
    assert int(active.sum()) == 5
    assert bool(diagnostics['gram_valid'][0])
    assert bool(diagnostics['retained_metric_positive'][0])
    np.testing.assert_allclose(recovered[active], poles, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(b, output[None] @ np.asarray(y), atol=1e-12, rtol=1e-12)
    for weight in (lambda t: np.ones_like(t), lambda t: t,
                   lambda t: 1 / (-.25**2-t)):
        got = (b[0] * np.where(active[0], weight(recovered[0]), 0)) @ b[0].conj().T
        expected = (physical * weight(poles)) @ physical.conj().T
        np.testing.assert_allclose(got, expected, atol=1e-11, rtol=1e-11)
