"""Shared-pole service gates; same checks under pytest and the P4 CLI."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def check_faces(mesh, output=None):
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


def main():
    import argparse
    import subprocess
    import jax
    from jax.sharding import Mesh
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    assert jax.process_count() == 4 and jax.device_count() == 4
    mesh = Mesh(np.asarray(jax.devices()).reshape(2, 2), ('x', 'y'))
    receipt = dict(job=os.environ.get('SLURM_JOB_ID'),
                   step=os.environ.get('SLURM_STEP_ID'),
                   commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   source=str(Path(__file__).resolve()), faces=check_faces(mesh, args.output))
    if jax.process_index() == 0:
        (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps(receipt), flush=True)
    return 0


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    initialize_communicator_stack(platform='gpu')
    run_main_and_finalize(main)
