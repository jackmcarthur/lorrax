"""P4/P16 axis photon response against the pre-fix source, including Fe geometry."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    from runtime import initialize_communicator_stack, finalize_process
    rt = initialize_communicator_stack()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host
    from common.wfn_layout import psi_specs
    from gw.w_isdf import _get_chi_fractional_contour_kernel_face as candidate
    mesh = rt.mesh
    assert mesh.size in (4, 16)
    path = args.run / 'reference_w_isdf.py'
    spec = importlib.util.spec_from_file_location('gw._axis_reference', path)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    reference = reference_module._get_chi_fractional_contour_kernel_face

    def put(value, sharding=P()):
        value = np.asarray(value)
        return jax.make_array_from_callback(value.shape, NamedSharding(mesh, sharding),
                                            lambda ix: value[ix])

    def carrier(shape, sharding, seed):
        # Only the local tile exists on host; both executions consume it unchanged.
        def local(ix):
            slices = [part.indices(size) for part, size in zip(ix, shape)]
            dims = tuple(len(range(*part)) for part in slices)
            rng = np.random.default_rng(seed + sum((i+1)*p[0] for i, p in enumerate(slices)))
            return (rng.normal(size=dims) + 1j*rng.normal(size=dims)) / np.sqrt(nb)
        return jax.make_array_from_callback(shape, NamedSharding(mesh, sharding), local)

    rows = []
    for label, grid, nb, n in [('tiny', (2, 2, 2), 8, 16),
                              ('fe304', (4, 4, 4), 304, 3168)]:
        nk, ns = int(np.prod(grid)), 4
        nmu, mun = psi_specs('axis')
        left = tuple(carrier((nk, ns, n, nb), mun, seed) for seed in (477, 478))
        right = tuple(carrier((nk, nb, ns, n), nmu, seed) for seed in (479, 480))
        energy = np.broadcast_to(np.linspace(-1, 2, nb), (nk, nb))
        f = 1 / (1 + np.exp(energy / .3))
        cases = ('moment', 'retarded', 'laplace_ordered', 'kms_static') if label == 'tiny' else ('retarded',)
        for case in cases:
            mode = 'retarded' if case == 'moment' else case
            times = np.array([0.]) if case == 'moment' else np.array([.1, .3, .5])
            if label == 'fe304':
                times = times[:1]
            nout = 2 if case == 'laplace_ordered' else 1
            nrows = 2*nout if case == 'laplace_ordered' else nout
            projection = np.arange(1, nrows*len(times)+1).reshape(nrows, -1) * (.1+.2j)
            lower, upper, ref = f, -1j*(1-f) if case == 'moment' else 1-f, np.array(0.)
            if case == 'laplace_ordered':
                lower, upper, ref = np.stack((f, .4*f)), np.stack((1-f, .7*(1-f))), np.array([-1., 2.])
            elif case == 'kms_static':
                ref = np.array([2., 0.])
            inputs = (put(times), put(projection), left, right, put(energy),
                      put(lower.astype(complex)), put(upper.astype(complex)), put(ref))
            selected = (0, 3, 7) if label == 'tiny' else tuple(range(13))
            outputs, measurements = [], {}
            for name, builder in [('reference', reference), ('candidate', candidate)]:
                kernel = builder(mesh, grid, nout, (nk, nb, n, ns), selected_q=selected,
                                 ordered=True, vertex=True, pair_mode=mode, layout='axis')
                executable = kernel.lower(*inputs).compile()
                stats = executable.memory_analysis()
                output = executable(*inputs)
                jax.block_until_ready(output)
                timings = []
                for _ in range(3):
                    started = time.monotonic()
                    measured = executable(*inputs)
                    jax.block_until_ready(measured)
                    timings.append(time.monotonic()-started)
                    del measured
                assert output.sharding.spec == P(None, None, 'x', 'y'), output.sharding
                outputs.append(output)
                measurements[name] = dict(seconds=timings, arguments=stats.argument_size_in_bytes,
                                          outputs=stats.output_size_in_bytes, temporaries=stats.temp_size_in_bytes)
                if jax.process_index() == 0:
                    (args.run / f'{label}_{case}_{name}.hlo').write_text(executable.as_text())
                del executable, kernel
            relative = float(gather_to_host(jnp.linalg.norm(outputs[1]-outputs[0]) /
                                            jnp.maximum(jnp.linalg.norm(outputs[0]), 1e-300)))
            assert np.isfinite(relative) and relative < 2e-12, (label, case, relative)
            row = dict(geometry=label, case=case, relative=relative, measurements=measurements)
            rows.append(row)
            if jax.process_index() == 0:
                print(json.dumps(row), flush=True)
            del outputs, inputs
        del left, right
    if jax.process_index() == 0:
        result = dict(schema='lorrax.photon-axis-layout.v1', status='PASS', rows=rows,
                      reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                      job=os.environ.get('SLURM_JOB_ID'), step=os.environ.get('SLURM_STEP_ID'),
                      scope='Axis photon contraction parity and synthetic Fe304 correlation timing/memory; not full SC performance')
        (args.run / 'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    finalize_process()


if __name__ == '__main__':
    main()
