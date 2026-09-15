"""P4 stored ragged factors to current Gamma W; scalar-head MPA fit owner."""
import argparse
import json
import os
from pathlib import Path
import runpy


def main(runtime):
    import jax
    import numpy as np
    from jax.sharding import PartitionSpec as P
    from file_io import shared_pole_store as store
    from file_io.slab_io import SlabIO
    from gw.shared_pole_head import _gamma_body
    from gw.mpa.model import fit_head_samples
    from types import SimpleNamespace

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    mesh = runtime.mesh
    assert jax.process_count() == jax.device_count() == 4
    helpers = runpy.run_path('tests/test_shared_pole_store.py')
    meta, tables, recipe, identity = helpers['_fixture'](mesh)
    _, packed, poles, counts = helpers['_model'](meta)
    put = lambda a, spec: helpers['_device'](np.asarray(a), mesh, spec)
    path = args.output/'model.h5'
    store.write_shared_pole_model(path, put(packed, P(None, 'x', None, 'y')),
        put(poles, P(None, 'y')), counts, q_span=(0, 3), meta=meta, tables=tables,
        recipe=recipe, receipts={'identity': identity, 'scope': 'planted Gamma head'})
    header = store.validate_shared_pole_model(path, expected_identity=identity,
        mesh_xy=mesh, capacity=meta.shared_pole_capacity)
    with SlabIO(path, mode='r', mesh=mesh) as io:
        b, lam, k = store.read_shared_pole_matrix(io, (0, 1), meta=meta, header=header)
    assert b.sharding.spec == P(None, 'x', 'y')
    v = np.eye(packed.shape[1], dtype=complex)[None]
    vdev = put(v, P(None, 'x', 'y'))
    build = _gamma_body(mesh)
    errors = []
    for factor, z in ((1., 1+.5j), (1.2, .7j), (.8, 0j)):
        c = factor*packed[:1, :, 0, :]
        weights = np.where(np.arange(6)[None, :]<counts[:1, None], 1/(z*z-poles[:1]), 0)
        expected = v+np.einsum('qik,qk,qjk->qij', c, weights, c.conj())
        got = build(put(np.asarray(z*z), P()), factor*b, lam, k, vdev)
        error = float(jax.numpy.max(abs(got-put(expected, P(None, 'x', 'y')))))
        assert error < 1e-9, error
        errors.append(error)
    # The head uses the existing scalar fitter on a planted two-pole model.
    z = np.array([0+.15j, 1+.15j, 0+.6j, 1+.6j])
    omega, residue = np.array([.4-.02j, .9-.03j]), np.array([.2, .1])
    wc = np.sum(2*omega*residue/(z[:, None]**2-omega**2), axis=1)
    samples = [SimpleNamespace(wcoul0=x+2, vc0=2) for x in wc]
    head = fit_head_samples(samples, z, 2, model='qsgw_schur_loewner', solve='loewner')
    fitted = np.sum(2*head['Omega_p']*head['B_p']/(z[:, None]**2-head['Omega_p']**2), axis=1)
    assert np.max(abs(fitted-wc)) < 1e-9
    report = dict(status='PASS', job_step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
        scope='P4 ragged canonical factor read, all-P Gamma W with changed factors/frequencies, planted MPA scalar fit; no full SC verdict',
        gamma_errors=errors, head_sample_error=float(np.max(abs(fitted-wc))))
    (args.output/f'receipt_rank{jax.process_index()}.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime = initialize_communicator_stack()
    run_main_and_finalize(lambda: main(runtime))
