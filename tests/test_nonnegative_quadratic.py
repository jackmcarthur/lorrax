"""Distributed non-negative quadratics against independent NNLS and KKT data.

Run as a compute-node P4 script, after the CPU/GPU launch prelude.
"""
import json
import math
import sys
import time


def main():
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from scipy.optimize import nnls
    from distrib_la import solve_nonnegative_quadratic
    from common.gpu_utils import resolve_device_budget_gb
    from common.collectives import all_gather_processes
    resolve_device_budget_gb(linalg='distributed')
    mesh = stack.mesh
    p = math.lcm(int(mesh.shape['x']), int(mesh.shape['y']))
    face, rows = NamedSharding(mesh, P('x', 'y')), NamedSharding(mesh, P('x'))
    started, errors, cases = time.perf_counter(), [], 0

    def check(a, b, expected, name):
        nonlocal cases
        n = b.size
        padded = math.ceil(n / p) * p
        ap = np.pad(a, ((0, padded-n), (0, padded-n)))
        bp = np.pad(b, (0, padded-n))
        aa = jax.make_array_from_callback((padded, padded), face, lambda ix: ap[ix])
        bb = jax.make_array_from_callback((padded,), rows, lambda ix: bp[ix])
        x = solve_nonnegative_quadratic(aa, bb, mesh=mesh, n_logical=n)
        # Oracle tensors are intentionally tiny. Production never gathers A.
        expected = np.pad(expected, (0, padded-n))
        local = np.asarray(x.addressable_data(0))
        ix = next(iter(rows.addressable_devices_indices_map((padded,)).values()))
        error = np.linalg.norm(local-expected[ix]) / max(np.linalg.norm(expected[ix]), 1.)
        errors.append(float(error))
        assert error < 2e-8, (name, error, local, expected[ix])
        assert np.all(local >= 0), name
        # Exact zeros are important for the nodal rank's zero-weight pruning.
        assert np.all(local[expected[ix] == 0] == 0), name
        cases += 1
        if jax.process_index() == 0:
            print('NNLS_CASE ' + json.dumps(dict(name=name, n=n, error=error)), flush=True)

    for n in (7, 17, 33):
        rng = np.random.default_rng(100+n)
        for scale in (1e-8, 1., 1e8):
            q = rng.normal(size=(2*n, n))
            a = q.T@q + np.eye(n)*.1
            truth = rng.random(n)
            truth[rng.random(n) < .6] = 0.
            dual = np.where(truth == 0, rng.random(n)+.2, 0.)
            b = a@truth-dual
            check(a*scale, b*scale, truth, f'KKT-{n}-{scale}')
        check(np.eye(n), -np.ones(n), np.zeros(n), f'zero-{n}')
        # Independent oracle for a Gram matrix with htransform's positive
        # nodal overlaps and an orbit-folded metric, including spin2.
        phi = (rng.normal(size=(4, 5, 2, n*2))
               + 1j*rng.normal(size=(4, 5, 2, n*2))) / np.sqrt(n*2)
        contribution = np.einsum('kmsv,knsv->vkmn', phi.conj(), phi)
        contribution = contribution.reshape(n, 2, -1).sum(axis=1).T
        design = np.concatenate([contribution.real, contribution.imag], axis=0)
        target = np.tile(np.eye(5).reshape(-1), 4)
        target = np.concatenate([target, np.zeros_like(target)])
        a, b = design.T@design, design.T@target
        a += np.eye(n) * (1e-14*np.trace(a)/n)
        chol = np.linalg.cholesky(a)
        oracle, _ = nnls(chol.T, np.linalg.solve(chol, b), maxiter=20*n)
        check(a, b, oracle, f'orbit-gram-{n}')
    # Both gradients initially request entry; the unconstrained answer has
    # a negative component, so one solve cannot satisfy non-negativity.
    n = math.ceil(2/p)*p
    ap = np.pad(np.array([[1., 2.], [2., 5.]]), ((0, n-2), (0, n-2)))
    bp = np.pad(np.ones(2), (0, n-2))
    aa = jax.make_array_from_callback((n, n), face, lambda ix: ap[ix])
    bb = jax.make_array_from_callback((n,), rows, lambda ix: bp[ix])
    try:
        solve_nonnegative_quadratic(aa, bb, mesh=mesh, n_logical=2, maxiter=1)
    except RuntimeError as exc:
        assert 'did not converge' in str(exc)
    else:
        raise AssertionError('iteration exhaustion returned unconverged weights')
    maximum = float(np.max(all_gather_processes(np.array(max(errors)))))
    if jax.process_index() == 0:
        print('NONNEGATIVE_QUADRATIC PASS ' + json.dumps(dict(
            cases=cases, iteration_limit_refuses=True, max_relative_error=maximum,
            seconds=time.perf_counter()-started, ranks=jax.process_count())), flush=True)
    runtime.finalize_process(0)
    return 0


def test_nonnegative_quadratic_requires_compute_node():
    import pytest
    pytest.skip('run tests/test_nonnegative_quadratic.py as a P4 compute-node script')


if __name__ == '__main__':
    sys.exit(main())
