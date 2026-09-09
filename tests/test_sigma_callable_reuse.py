"""Rebuilt Sigma programs must consume current factors, poles and intervals."""
from pathlib import Path
import json
import os
import numpy as np


def check_reuse(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from ffi import ffi_dial_key
    from gw.mpa.sigma import (_shared_pole_panel_unfold,
        _shared_pole_local_kernel, synthesize_shared_pole_parents)
    from common.jax_compile_cache import compile_cache_stats

    def put(a, spec):
        sharding = NamedSharding(mesh, spec)
        return jax.make_array_from_callback(a.shape, sharding, lambda i: a[i])

    header = dict(grid=(2, 2, 2), representation='scalar-trs-even-s',
        n_q_full=8, q_irr_full_idx=np.array([0, 1], np.int32),
        qirr=dict(n_sym_spatial=1, sym_idx_q=np.zeros(8, np.int32)))
    tables = dict(rows=np.array([0, 1], np.int32),
        parent_rows=np.array([0, 1], np.int32), sym_rows=np.zeros(2, np.int32),
        q_frac=np.array([[0., 0., 0.], [0., 0., 0.5]]),
        wraps=np.zeros((1, 8, 3), np.int32), n_sym_spatial=1,
        certificates={a: dict(is_local=True, local_perm=(
            np.arange(8, dtype=np.int32) % (8//mesh.shape[a]))[None, :])
            for a in ('x', 'y')})
    def build():
        _, unfold = _shared_pole_panel_unfold(None, header, (0, 2),
                                             mesh_xy=mesh, tables=tables)
        return unfold, _shared_pole_local_kernel(mesh, unfold, ffi_dial_key())

    rng = np.random.default_rng(331)
    C = (rng.normal(size=(2, 8, 1, 8)) + 1j*rng.normal(size=(2, 8, 1, 8)))*0.1
    def operands(scale, pole_shift):
        return (put(C*scale, P(None, 'x', None, None)),
            put(C*scale, P(None, 'y', None, None)),
            put(np.full((2, 8), 0.7+pole_shift), P()),
            put(np.array([[0, 8], [0, 6]] if pole_shift == 0.
                         else [[1, 7], [2, 5]], np.int32), P()),
            jnp.array(0.1), jnp.array(0.4+0.2j))
    args0, args1 = operands(1., 0.), operands(1.1, 0.2)
    unfold0, first = build()
    start = compile_cache_stats()
    a = first(*args0)
    jax.block_until_ready(a)
    middle = compile_cache_stats()
    unfold1, second = build()
    assert unfold0 is unfold1 and first is second
    b = second(*args1)
    jax.block_until_ready(b)
    end = compile_cache_stats()
    assert end['compiles'] == middle['compiles']
    plus, transposed = synthesize_shared_pole_parents(*args1, mesh_xy=mesh)
    reference = unfold1(plus, transposed)
    error = float(jnp.max(jnp.abs(b-reference)))
    changed = float(jnp.max(jnp.abs(b-a)))
    assert error < 1e-12 and changed > 1e-10
    return dict(status='PASS', maximum_absolute=error,
        changed_input_difference=changed,
        first_compiles=middle['compiles']-start['compiles'],
        rebuilt_compiles=end['compiles']-middle['compiles'],
        scope='P4 rebuilt local Sigma factory, current factors, poles and active intervals; fresh direct service comparison')


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    initialize_communicator_stack(platform='gpu')
    def main():
        import sys, jax
        from common.collectives import resolve_mesh, barrier
        assert jax.process_count() == 4
        result = check_reuse(resolve_mesh())
        result['job_step'] = os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
        if jax.process_index() == 0:
            Path(sys.argv[1]).write_text(json.dumps(result, indent=2)+'\n')
        barrier('sigma-callable-reuse')
    run_main_and_finalize(main)
