"""The Gram validity floor is the propagated float64 rounding of the resolvent-identity entries.

Exact Stieltjes samples give a PSD Gram whose computed negative excursion stays inside
gram_rounding_floor; a non-Stieltjes perturbation far above rounding is refused.
"""
import numpy as np
import jax
import jax.numpy as jnp

from gw.shared_pole_local import _mm, solve_parent_pencil, zero_row_safe_eigh
from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b

jax.config.update('jax_enable_x64', True)


def _pencil_inputs(defect=0.0, seed=7):
    rng = np.random.default_rng(seed)
    n, poles = 24, 40
    t = np.sort(rng.uniform(0.1, 50.0, poles))            # Omega^2, Ry^2
    b = rng.normal(size=(n, poles)) + 1j * rng.normal(size=(n, poles))
    # ``defect`` adds s*C with C PSD: dW/ds gains +C, so the resolvent-identity Gram
    # gains -Q^H C Q, which no Stieltjes response (dW/ds <= 0) can produce.
    c = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    c = defect * (c @ c.conj().T) / n
    W = lambda s: (b / (s - t)) @ b.conj().T + s * c
    dW = lambda s: -(b / (s - t) ** 2) @ b.conj().T + c
    cols = [[], [], [], []]
    for u in (0.5, 2.0, 8.0):
        s = -u * u
        w, v = np.linalg.eigh(-(W(s) + W(s).conj().T) / 2)
        q = v[:, -6:]
        o = W(s) @ q
        for store, value in zip(cols, (np.full(6, s + 0j), q, o, dW(s) @ q)):
            store.append(value)
    points, q, o, d = (np.concatenate(c, axis=-1) for c in cols)
    m1, m3 = 0.5 * b @ b.conj().T, 0.5 * (b * t) @ b.conj().T
    qi = np.linalg.eigh(m1)[1][:, -4:]
    infinity = tuple(jnp.asarray(a)[None] for a in (qi, m1 @ qi, m3 @ qi))
    active = jnp.ones((1, points.size + 4), bool)
    return (jnp.asarray(points)[None], jnp.asarray(q)[None], jnp.asarray(o)[None],
            jnp.asarray(d)[None], infinity, active)


def _reduce(args):
    _, _, (reduction, _, _) = solve_parent_pencil(
        *args, eigh=zero_row_safe_eigh(jnp.linalg.eigh), matmul=_mm,
        gates=shared_real_pole_gates_v1_r3b, ordered=False, odd_moments=False, keep_budget=None)
    return {k: np.asarray(v)[0] for k, v in reduction.items()}


def test_exact_samples_sit_inside_the_rounding_floor():
    r = _reduce(_pencil_inputs())
    assert r['gram_valid']
    assert 0 < r['gram_floor_relative'] < 1e-8
    assert r['gram_min_relative'] >= -r['gram_floor_relative']


def test_a_non_stieltjes_defect_above_rounding_is_refused():
    r = _reduce(_pencil_inputs(defect=1e-3))
    assert r['gram_min_relative'] < -r['gram_floor_relative']
    assert not r['gram_valid']
