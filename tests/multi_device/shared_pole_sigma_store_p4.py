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
    from gw.mpa.sigma import _shared_pole_w_synthesis
    from gw.mpa.sigma_windows import shared_pole_frequencies

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mesh = runtime.mesh
    assert jax.process_count() == jax.device_count() == 4
    assert mesh.shape['x'] > 1 and mesh.shape['y'] > 1
    helpers = runpy.run_path('tests/test_shared_pole_store.py')
    meta, tables, recipe, identity = helpers['_fixture'](mesh)
    _, packed, poles, counts = helpers['_model'](meta)
    put = lambda a, spec: helpers['_device'](np.asarray(a), mesh, spec)
    path = args.output / 'model.h5'
    header = store.write_shared_pole_model(
        path, put(packed, P(None, 'x', None, 'y')),
        put(poles, P(None, 'y')), counts, q_span=(0, 3),
        meta=meta, tables=tables, recipe=recipe,
        receipts={'identity': identity, 'scope': 'planted W execution parity'})
    store.validate_shared_pole_model(path, expected_identity=identity, mesh_xy=mesh)
    frequencies = shared_pole_frequencies(poles, counts)
    indices = put(np.arange(3, dtype=np.int32), P())
    bounds_host = np.tile([1., 4., -np.inf, -np.inf, np.inf, np.inf], (3, 1))
    bounds = put(bounds_host, P())
    phase = put(np.ones(3), P())
    e, t = put(np.asarray(.6), P()), put(np.asarray(.7+.2j), P())
    omega = np.sqrt(poles)
    active = (omega > 1) & (omega <= 4) & (np.arange(6)[None, :] < counts[:, None])
    weights = np.where(active, np.exp(-1j*(omega-.6)*(.7+.2j))/(2*omega), 0)
    C = packed[:, :, 0, :]
    expected = np.einsum('qik,qk,qjk->qij', C, weights, C.conj())
    oracle = put(expected, P(None, 'x', 'y'))
    results = []
    with SlabIO(path, mode='r', mesh=mesh) as io:
        for b, c in ((3, 5), (1, 2), (2, 3), (3, 1)):
            build = _shared_pole_w_synthesis(
                io, meta, header, frequencies,
                dict(status='PASS', parent_capacity=b, column_capacity=c), mesh_xy=mesh)
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
                                dense_error=error, restored_window_error=repeat))
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
