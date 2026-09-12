"""Moment-loop algebra; production mesh and peak gates remain separate."""

from functools import partial

import numpy as np


def test_moment_diagnostics_tracks_current_poles_and_projected_defects():
    """Distinct M1/M3 defects and a second state expose a swapped/frozen loop."""
    import jax
    import jax.numpy as jnp

    from gw.shared_pole_constructor import _model_diagnostics

    def matmul(a, b, *, transa="N", transb="N"):
        if transa == "C":
            a = a.conj().swapaxes(-1, -2)
        if transb == "C":
            b = b.conj().swapaxes(-1, -2)
        return a @ b

    evaluate = jax.jit(partial(_model_diagnostics, matmul=matmul))
    rng = np.random.default_rng(917)
    factors = rng.normal(size=(2, 5, 3)) + 1j * rng.normal(size=(2, 5, 3))
    directions = rng.normal(size=(2, 5, 2)) + 1j * rng.normal(size=(2, 5, 2))
    poles = np.array([[0.3, 1.4, 2.7], [0.8, 1.1, 3.6]])
    # The two moments have different noncommuting Hermitian perturbations.
    perturbations = {}
    for name in ("M1", "M3"):
        a = rng.normal(size=(2, 5, 5)) + 1j * rng.normal(size=(2, 5, 5))
        perturbations[name] = (a + a.conj().swapaxes(-1, -2)) * 0.02

    for energy_scale in (1.0, 1.17):
        current_poles = poles * energy_scale
        moments, expected = {}, {}
        for name, power in (("M1", 0), ("M3", 1)):
            # Independent rank-one spectral sum, in physical Ry moments.
            exact = np.stack([
                sum(current_poles[q, k] ** power
                    * np.outer(factors[q, :, k], factors[q, :, k].conj()) / 2
                    for k in range(3))
                for q in range(2)
            ])
            moments[name] = exact + perturbations[name]
            adjoint = directions.conj().swapaxes(-1, -2)
            expected[name] = {
                "full_relative": np.linalg.norm(perturbations[name], axis=(-2, -1))
                / np.linalg.norm(moments[name], axis=(-2, -1)),
                "original_infinity_relative": np.linalg.norm(
                    adjoint @ perturbations[name] @ directions, axis=(-2, -1))
                / np.linalg.norm(adjoint @ moments[name] @ directions, axis=(-2, -1)),
            }
        actual = evaluate(
            (jnp.asarray(factors), jnp.asarray(current_poles), jnp.ones((2, 3), bool)),
            {name: jnp.asarray(value) for name, value in moments.items()},
            jnp.asarray(directions))
        for name in expected:
            for field in expected[name]:
                np.testing.assert_allclose(actual[name][field], expected[name][field],
                                           rtol=1e-11, atol=1e-13)
