"""Uniform storage carriers do not add physical Green-pair executions."""
import ast
from pathlib import Path

import numpy as np


def test_complex_rules_use_one_carrier_and_keep_exact_zero_suffix():
    import minimax
    from gw.response_bank import full_density_sample_rules
    rules = full_density_sample_rules(np.array([[0., 2.]]), np.array([.17+.12j, .4+.2j]),
        gamma_ry=.03, interval_ry=(0., 2.), paired=True, values_only=True)
    physical = 0
    for rule in rules:
        n = rule['active_count']; physical += n
        assert 0 < n <= rule['carrier_capacity'] == minimax.RESPONSE_NODE_CAPACITY == 384
        assert rule['times'].shape == (384,) and rule['weights'].shape == (2, 2, 384)
        assert not rule['times'][n:].any() and not rule['weights'][..., n:].any()
        assert np.any(rule['weights'][..., n-1] != 0)
    assert physical < 384*len(rules)
    old = full_density_sample_rules(np.array([[0., 2.]]), np.array([.4j]),
        gamma_ry=.03, interval_ry=(1., 2.), paired=True)
    assert all('active_count' not in r and 'carrier_capacity' not in r for r in old)


def test_actual_live_count_and_rowpass_skip_zero_suffix(monkeypatch):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh
    import gw.w_isdf as owner
    import gw.contour_accumulator as accumulator
    from gw.subtile_stream import PassPlan, stream_passes
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))
    weights = jnp.zeros((2, 1, 384), dtype=jnp.complex128).at[0, 0, :3].set(1.)
    # Execute the current owner's actual live-prefix formula, not a second
    # implementation that could diverge from the native stream.
    tree = ast.parse(Path(owner.__file__).read_text())
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'live_count')
    namespace = dict(jnp=jnp, projection_rows=weights)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 owner.__file__, 'exec'), namespace)
    count = namespace['live_count']()
    assert int(count) == 3
    seen = []
    def node_rows(ops, index):
        jax.debug.callback(lambda n: seen.append(int(n)), index, ordered=True)
        return jnp.ones((1, 2, 1, 1, 1), dtype=jnp.complex128)*(index+1)
    def add(a, r, w, valid, *, m0, n0, sets, terms, outputs):
        mask = jnp.arange(len(r)) < terms
        return a+jnp.einsum('to,tqmn,t->oqmn', w, r, mask)
    monkeypatch.setattr(accumulator, 'contour_block_accumulate_local', add)
    plan = PassPlan(passes=((0, 1),), chunk=8, rows=1, windows=((0, 0, 1),))
    @jax.jit
    def execute(w, n):
        return stream_passes(jnp.zeros((1, 1, 1, 1), dtype=jnp.complex128), mesh=mesh,
            plan=plan, weights=w, count=n, node_rows=node_rows, prepare=lambda window: None)
    result = execute(weights, count)
    jax.block_until_ready(result); jax.effects_barrier()
    assert seen == [0, 1, 2]  # NOT 384 and not the remaining five chunk slots.
    np.testing.assert_array_equal(result, np.full((1, 1, 1, 1), 6.))
