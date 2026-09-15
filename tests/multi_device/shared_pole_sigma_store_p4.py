"""P4 canonical-store → full-q W parity, including forced q/K panels.

Schedules here force execution branches; they are not capacity certificates.
The aggregate production admission is verified separately by its owner.
"""
from pathlib import Path
import argparse
import json
import os
import runpy


def main(runtime):
    import jax
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from file_io import shared_pole_store as store
    from file_io.slab_io import SlabIO
    from gw.mpa.sigma import _shared_pole_w_synthesis, _shared_pole_fixed_q_policy
    from gw.mpa.sigma_windows import shared_pole_frequencies

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mesh = runtime.mesh
    assert jax.process_count() == jax.device_count() == 4
    assert mesh.shape['x'] > 1 and mesh.shape['y'] > 1
    helpers = runpy.run_path('tests/test_shared_pole_store.py')
    meta, tables, recipe, identity = helpers['_sigma_fixture'](mesh)
    C, packed, poles, counts = helpers['_model'](meta)
    C=C[...,:4];poles=poles[:,:4];counts=np.minimum(counts,4)
    C[0, :, 0, 1] *= np.exp(1j*np.arange(C.shape[1]))
    packed=meta.mu_basis.pack_host(C,axis=1)
    put = lambda a, spec: helpers['_device'](np.asarray(a), mesh, spec)
    path = args.output / 'model.h5'
    header = store.write_shared_pole_model(
        path, put(packed, P(None, 'x', None, 'y')),
        put(poles, P(None, 'y')), counts, q_span=(0, 3),
        meta=meta, tables=tables, recipe=recipe,
        receipts={'identity': identity, 'scope': 'planted W execution parity'})
    header = store.validate_shared_pole_model(
        path, expected_identity=identity, mesh_xy=mesh, capacity=meta.shared_pole_capacity)
    frequencies = shared_pole_frequencies(poles, counts)
    indices = put(np.arange(3, dtype=np.int32), P())
    bounds_host = np.tile([1., 4., -np.inf, -np.inf, np.inf, np.inf], (3, 1))
    bounds = put(bounds_host, P())
    phase = put(np.ones(3), P())
    e, t = put(np.asarray(.6), P()), put(np.asarray(.7+.2j), P())
    omega = np.sqrt(poles)
    active = (omega > 1) & (omega <= 4) & (np.arange(4)[None, :] < counts[:, None])
    weights = np.where(active, np.exp(-1j*(omega-.6)*(.7+.2j))/(2*omega), 0)
    # Independent endpoint oracle, with the known paired operation rows.
    paired=np.asarray([0,0,6,2,4,0,8,6,10])
    policy=_shared_pole_fixed_q_policy(header)
    assert np.array_equal(policy.unfold_sym_idx,paired) and policy.n_pair_rewired==4, policy.unfold_sym_idx
    oracle_helpers = runpy.run_path('tests/multi_device/shared_pole_dense_oracle.py')
    expected = oracle_helpers['full_q_operator'](C, weights, tables, paired)
    expected=meta.mu_basis.pack_host(meta.mu_basis.pack_host(expected,axis=1),axis=2)
    from symmetry_maps import q_negation_index
    neg=q_negation_index((3,3,1))
    assert np.max(np.abs(expected-expected[neg].transpose(0,2,1)))<1e-10
    oracle = put(expected, P(None, 'x', 'y'))
    results = []
    with SlabIO(path, mode='r', mesh=mesh) as io:
        for b, c in ((3, 5), (1, 2), (2, 3), (3, 1)):
            schedule = dict(status="PASS", parent_capacity=b, column_capacity=c)
            build = _shared_pole_w_synthesis(
                io, meta, header, frequencies,
                schedule, mesh_xy=mesh)
            got = build(None, None, indices, bounds, phase, e, t)
            error = float(jax.numpy.max(jax.numpy.abs(got-oracle)))
            assert error < 1e-10, (b, c, error)
            # A changed window must invalidate cached selectors, including
            # an empty interval that returns the complete zero full-q W.
            empty = put(np.tile([20., 30., -np.inf, -np.inf, np.inf, np.inf], (3, 1)), P())
            zero = build(None, None, indices, empty, phase, e, t)
            assert float(jax.numpy.max(jax.numpy.abs(zero))) == 0
            again = build(None, None, indices, bounds, phase, e, t)
            repeat = float(jax.numpy.max(jax.numpy.abs(again-oracle)))
            assert repeat < 1e-10
            results.append(dict(parent_capacity=b, column_capacity=c,
                                dense_error=error, restored_window_error=repeat,
                                q_pair_rewired=policy.n_pair_rewired))
            del build, got, zero, again
    report = dict(status='PASS', job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        scope='P4 canonical store + local full-q synthesis, ragged K, forced q/K panels and window refresh',
        results=results, aggregate_3U='NOT_MEASURED', nonlocal_fallback='NOT_MEASURED',
        full_sigma='NOT_MEASURED')
    (args.output/f'receipt_rank{jax.process_index()}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime = initialize_communicator_stack()
    run_main_and_finalize(lambda: main(runtime))
