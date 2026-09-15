"""Shared-pole service gates; same checks under pytest and the P4 CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def check_faces(mesh, output=None, profile=False):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la as D

    assert mesh.shape['x'] > 1 and mesh.shape['y'] > 1
    rng = np.random.default_rng(1909)
    c = rng.normal(size=(3, 12, 7)) + 1j * rng.normal(size=(3, 12, 7))
    c[0, :, 5:] = 0
    c[1, :, 3:] = 0
    omega = rng.uniform(.2, 2, (3, 7))
    d = np.exp(-1j * omega * .7) / (2 * omega)
    lo, hi = np.array([1, 0, 2]), np.array([5, 3, 7])
    mask = (np.arange(7) >= lo[:, None]) & (np.arange(7) < hi[:, None])
    dm = d * mask
    want = (c * dm[:, None, :]) @ c.conj().transpose(0, 2, 1)
    wt = (c.conj() * dm[:, None, :]) @ c.transpose(0, 2, 1)

    def put(a, spec):
        sh = NamedSharding(mesh, spec)
        return jax.make_array_from_callback(a.shape, sh, lambda i: a[i])

    rows = []
    for spin in (False, True):
        factors = c.reshape((3, 6, 2, 7)) if spin else c
        xs = P(None, 'x', None, None) if spin else P(None, 'x', None)
        ys = P(None, 'y', None, None) if spin else P(None, 'y', None)
        args = (put(factors, xs), put(factors, ys), put(d, P()),
                put(lo, P()), put(hi, P()))
        fn = jax.jit(lambda *a: D.contract_faces(
            *a, mesh=mesh, return_transpose=True))
        executable = fn.lower(*args).compile()
        if profile and not spin:
            import ctypes
            cudart = ctypes.CDLL('libcudart.so.13')
            jax.block_until_ready(args)
            assert cudart.cudaProfilerStart() == 0
            for _ in range(25):
                profiled = executable(*args)
                jax.block_until_ready(profiled)
            assert cudart.cudaProfilerStop() == 0
        got, trans = executable(*args)
        err = float(jnp.max(jnp.abs(got - put(want, P(None, 'x', 'y')))))
        terr = float(jnp.max(jnp.abs(trans - put(wt, P(None, 'x', 'y')))))
        assert err < 2e-12 and terr < 2e-12, (err, terr)
        hlo = executable.as_text().lower()
        banned = [s for s in ('all-reduce', 'all-gather', 'all-to-all',
                              'collective-permute', 'lorrax_') if s in hlo]
        assert not banned, banned
        if output and jax.process_index() == 0:
            (output / f'faces_spin{spin}.hlo').write_text(hlo)
        rows.append(dict(spin_axis=spin, error=err, transpose_error=terr,
                         optimized_hlo_collectives=banned))
    red = {
        'missing_2omega': (c * (dm * 2 * omega)[:, None, :]) @ c.conj().transpose(0, 2, 1),
        'conjugated_weight': (c * dm.conj()[:, None, :]) @ c.conj().transpose(0, 2, 1),
        'wrong_orientation': wt,
    }
    red_errors = {name: float(np.max(np.abs(value - want)))
                  for name, value in red.items()}
    assert min(red_errors.values()) > .1, red_errors
    return dict(status='PASS', cases=rows, red_errors=red_errors)


def test_face_contraction():
    import jax
    from jax.sharding import Mesh
    from lxkit.testing import require_devices
    require_devices(4, 'cpu')
    mesh = Mesh(np.asarray(jax.devices('cpu')[:4]).reshape(2, 2), ('x', 'y'))
    check_faces(mesh)


def check_directions_and_gemm(mesh):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    import distrib_la as D
    rng = np.random.default_rng(1910)
    def unitary():
        return np.linalg.qr(rng.normal(size=(12, 12))
                            + 1j * rng.normal(size=(12, 12)))[0]
    u, v = unitary(), unitary()
    spectrum = np.array([6, 3, 3*(1-4e-7), .1, .08, .07, .06, .05, .04, .03, .02, .01])
    w = (u * spectrum) @ v.conj().T
    herm = (v * spectrum) @ v.conj().T
    tile = NamedSharding(mesh, P('x', 'y'))
    def put(a):
        return jax.make_array_from_callback(a.shape, tile, lambda i: a[i])
    def error(got, want):
        return float(jnp.max(jnp.abs(got - put(want))))
    rows = []
    for label, backend, route in [('local', 'off', 'batch_reshard'),
                                   ('distributed', 'distributed', 'auto')]:
        eig = D.plan('eigh', mesh, backend=backend, n=24, batched_route=route)
        q, s = D.right_singular_vectors(put(w), .4999999, eigh_plan=eig,
                                        column_extent=lambda r: 6)
        assert s.size == 3, (label, s)
        np.testing.assert_allclose(np.asarray(s), spectrum[:3], rtol=2e-12)
        projector_error = error(q @ q.conj().T, v[:, :3] @ v[:, :3].conj().T)
        assert projector_error < 2e-11, projector_error
        assert float(jnp.max(jnp.abs(q[:, 3:]))) == 0
        # Repeated calls at one shape must use the current matrix, including
        # changed right directions and singular values across SC updates.
        repeated_errors = []
        for matrix, vectors, scale in ((w.conj().T, u, 1.), (2*w, v, 2.)):
            qr, sr = D.right_singular_vectors(put(matrix), .4999999,
                eigh_plan=eig, column_extent=lambda r: 6)
            assert sr.size == 3
            np.testing.assert_allclose(np.asarray(sr), scale*spectrum[:3],
                                       rtol=2e-12)
            repeated_error = error(qr @ qr.conj().T,
                                    vectors[:, :3] @ vectors[:, :3].conj().T)
            assert repeated_error < 2e-11
            assert float(jnp.max(jnp.abs(qr[:, 3:]))) == 0
            repeated_errors.append(repeated_error)
        ep = D.plan('eigh', mesh, backend=backend, n=12, batched_route=route)
        eigen_input = put(herm)
        full_values, full_vectors = ep(eigen_input)
        jax.block_until_ready((full_values, full_vectors))
        input_error = error(eigen_input, herm)
        assert input_error == 0, (label, 'non-donating eigh mutated its input', input_error)
        qe, ev = D.leading_eigenvectors(put(herm), 2, eigh_plan=ep,
                                        column_extent=lambda r: 6)
        assert ev.size == 3
        eigen_error = error(qe @ qe.conj().T, v[:, :3] @ v[:, :3].conj().T)
        assert eigen_error < 2e-11
        # Exercise rectangular adjoints and both operand orientations.
        products = []
        for ta, tb in [('N', 'N'), ('C', 'N'), ('T', 'N'), ('N', 'C'), ('N', 'T'), ('C', 'C')]:
            a = w[:8, :6] if ta == 'N' else w[:6, :8]
            b = herm[:6, :10] if tb == 'N' else herm[:10, :6]
            aa = a if ta == 'N' else (a.conj().T if ta == 'C' else a.T)
            bb = b if tb == 'N' else (b.conj().T if tb == 'C' else b.T)
            got = D.matmul(put(a), put(b), mesh=mesh, backend=backend,
                           batched_route=route, transa=ta, transb=tb)
            err = error(got, aa @ bb)
            assert err < 2e-11, (label, ta, tb, err)
            products.append(dict(transa=ta, transb=tb, error=err))
        rows.append(dict(plan=label, svd_projector_error=projector_error,
                         changing_input_projector_errors=repeated_errors,
                         eigen_projector_error=eigen_error,
                         eigen_input_preservation_error=input_error, retained_rank=3,
                         gemm=products))
    return dict(status='PASS', cases=rows)


def main():
    import argparse
    import subprocess
    import jax
    from jax.sharding import Mesh
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--constructor', action='store_true')
    ap.add_argument('--symmetry', action='store_true')
    ap.add_argument('--profile', action='store_true')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert jax.process_count() == 4 and jax.device_count() == 4
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ('x', 'y'))
    receipt = dict(job=os.environ.get('SLURM_JOB_ID'),
                   step=os.environ.get('SLURM_STEP_ID'),
                   commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   source=str(Path(__file__).resolve()), faces=check_faces(mesh, args.output, profile=args.profile))
    if args.constructor:
        receipt['constructor'] = check_directions_and_gemm(mesh)
    if args.symmetry:
        import importlib.util
        path = Path(__file__).resolve().parents[2] / 'symmetry_maps/tests/test_shared_pole_unfold.py'
        spec = importlib.util.spec_from_file_location('shared_pole_unfold_gate', path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        receipt['symmetry'] = mod.check_shared_pole_unfold(mesh, profile=args.profile)
    if jax.process_index() == 0:
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)
    return 0


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    initialize_communicator_stack(platform='gpu')
    run_main_and_finalize(main)
