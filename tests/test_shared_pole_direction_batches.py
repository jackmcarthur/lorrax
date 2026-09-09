"""Independent spectral cuts on q batches through the existing service doors."""
from pathlib import Path
import json
import os
import numpy as np


def check_direction_batches(mesh):
    """Compare batched singular/eigen projectors to known unequal-rank measures."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la as D
    from runtime.padding import padded_axis

    rng = np.random.default_rng(32324)
    n = 8
    def unitary():
        return np.linalg.qr(rng.normal(size=(n, n))
                            + 1j*rng.normal(size=(n, n)))[0]
    left = np.stack([unitary() for _ in range(3)])
    right = np.stack([unitary() for _ in range(3)])
    sigma = np.array([[8, 4, 4, 1, .5, .1, .02, .005],
                      [8, 5, 3, 2, 1, .4, .2, .1],
                      [8, 7, 6, 5, 4, 1, .5, .1]])
    adjoint = lambda a: a.conj().swapaxes(-1, -2)
    w = (left*sigma[:, None, :]) @ adjoint(right)
    h = (right*sigma[:, None, :]) @ adjoint(right)
    face = NamedSharding(mesh, P(None, 'x', 'y'))
    def put(a):
        layout = face if a.ndim == 3 else NamedSharding(mesh, P('x', 'y'))
        return jax.make_array_from_callback(a.shape, layout, lambda i: a[i])
    def extent(width):
        return padded_axis(width, mesh, name='direction_test_port',
                           specs=((P('x', 'y'), 0), (P('x', 'y'), 1))).carrier
    rows = []
    for label, backend, route in [('local', 'off', 'batch_reshard'),
                                   ('distributed', 'distributed', 'auto')]:
        ep = D.plan('eigh', mesh, n=n, backend=backend, batched_route=route)
        sp = D.plan('eigh', mesh, n=2*n, backend=backend, batched_route=route)
        def project(q):
            return D.matmul(q, q, transb='C', mesh=mesh,
                            backend='auto', batched_route=route)
        for name, matrix, expected_counts in [('svd', w, [3, 2, 5]),
                                               ('eigh', h, [3, 2, 2])]:
            def select(matrix):
                if name == 'svd':
                    return D.right_singular_vectors(matrix, .49, eigh_plan=sp,
                                                     column_extent=extent)
                return D.leading_eigenvectors(matrix, 2, eigh_plan=ep,
                                               column_extent=extent)
            q, values = select(put(matrix))
            counts = [v.shape[-1] for v in values]
            assert counts == expected_counts, (name, counts)
            assert q.sharding.is_equivalent_to(face, 3)
            got = project(q)
            errors = []
            for i, count in enumerate(counts):
                expected = right[i, :, :count] @ adjoint(right[i, :, :count])
                errors.append(float(jnp.max(jnp.abs(got[i] - put(expected)))))
                single, single_values = select(put(matrix[i]))
                single_projector = project(single[None])[0]
                assert float(jnp.max(jnp.abs(got[i] - single_projector))) < 2e-11
                assert np.max(abs(np.asarray(values[i]) - np.asarray(single_values))) < 2e-11
                if count < q.shape[-1]:
                    assert float(jnp.max(jnp.abs(q[i, :, count:]))) == 0
            assert max(errors) < 2e-11, errors
            rows.append(dict(backend=label, kind=name, counts=counts,
                             maximum_projector_error=max(errors)))
        # Same dimensions and cached executable, different current matrix.
        rotate = unitary()
        changed = w @ rotate
        q, values = D.right_singular_vectors(put(changed), .49,
                    eigh_plan=sp, column_extent=extent)
        changed_right = adjoint(rotate) @ right
        got = project(q)
        changed_errors, stale_errors = [], []
        for i, value in enumerate(values):
            count = value.shape[-1]
            expected = changed_right[i, :, :count] @ adjoint(changed_right[i, :, :count])
            stale = right[i, :, :count] @ adjoint(right[i, :, :count])
            changed_errors.append(float(jnp.max(jnp.abs(got[i] - put(expected)))))
            stale_errors.append(float(np.max(abs(expected-stale))))
        assert max(changed_errors) < 2e-11 and min(stale_errors) > .05
        rows.append(dict(backend=label, kind='changed_input',
                         maximum_projector_error=max(changed_errors),
                         stale_projector_red_minimum=min(stale_errors)))
    return dict(status='PASS', rows=rows,
                scope='P4 local/distributed; three unequal spectral cuts, multiplet closure, scalar parity, exact zero tails, current-input red twin')


def test_direction_batches():
    import jax
    from jax.sharding import Mesh
    from lxkit.testing import require_devices
    require_devices(4, 'cpu')
    mesh = Mesh(np.asarray(jax.devices('cpu')[:4]).reshape(2, 2), ('x', 'y'))
    check_direction_batches(mesh)


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    initialize_communicator_stack(platform='gpu')
    def main():
        import sys
        import jax
        from common.collectives import resolve_mesh, barrier
        assert jax.process_count() == 4
        result = check_direction_batches(resolve_mesh())
        result['job_step'] = os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID']
        if jax.process_index() == 0:
            Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
        barrier('direction-batch-gate')
        return 0
    run_main_and_finalize(main)
