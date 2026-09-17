"""Round model checks (``gw.shared_pole_local.round_checks``) against independent dense identities.

A round of four slots in batch layout, three real (slot 3 synthetic and skipped), ragged active columns:
* even route: held W(s) = b (s - Lambda)^-1 b^H and dW/ds at s = z^2 exact to 1e-12, V-whitened passivity
  maximum equal to numpy's, reciprocity not applicable on complex residues, moments M1/M3 exact;
* ordered route: the signed model's held W(z), dW/ds, the Hermitian part of -W(i eta) and the
  moments of (c/|mu|, mu^-2) against numpy;
* RED TWIN: a held sample scaled by 1.01 reads 0.01/1.01 in W only.
CPU (pytest, 2x2 host mesh) and P4 (``python test_shared_pole_local_checks.py OUT.json``).
"""
from pathlib import Path
import json
import os
import numpy as np


def check(mesh):
    import distrib_la
    from gw.shared_pole_local import _batch_put, check_round
    ranks, n, k = 4, 8, 8
    rng = np.random.default_rng(81)
    c = (rng.normal(size=(ranks, n, k)) + 1j * rng.normal(size=(ranks, n, k))) * .01
    mask = np.ones((ranks, k), bool)
    mask[1, -2:] = False
    c[1, :, -2:] = 0
    nodes = np.array([1j, 1.5 + .5j])
    eta = .25
    qi = _batch_put(mesh, np.linalg.qr(rng.normal(size=(ranks, n, 3)))[0].astype(complex))
    inverse = _batch_put(mesh, np.broadcast_to(np.eye(n, dtype=complex), (ranks, n, n)).copy())
    native = distrib_la.plan('eigh', mesh, n=n, backend='off', batched_route='batch_reshard').native_fn
    adj = lambda a: np.conj(np.swapaxes(a, -1, -2))
    rows = {}
    for ordered in (False, True):
        if ordered:
            mu = np.where(np.arange(k) % 2, 1., -1.)[None] * np.linspace(.4, 2., k)[None] * np.ones((ranks, 1))
            weight = np.where(mask[:, None], 1 / (nodes[None, :, None] * mu[:, None] - 1), 0)
            slope = np.where(mask[:, None], -mu[:, None] / (nodes[None, :, None] * mu[:, None] - 1) ** 2
                             / (2 * nodes[None, :, None]), 0)
            passive = np.where(mask, 1 / (1 - 1j * eta * mu), 0)
            scale = np.where(mask, 1 / np.abs(mu), 0)
            m1, m3 = ((c * scale[:, None] ** p) @ adj(c) / 2 for p in (2, 4))
            model = (_batch_put(mesh, np.zeros_like(c)), _batch_put(mesh, np.ones((ranks, k))), _batch_put(mesh, mask))
            signed = (_batch_put(mesh, c), _batch_put(mesh, mu), _batch_put(mesh, mask))
        else:
            poles = np.where(mask, np.arange(1, k + 1, dtype=float)[None], 1.)
            s = nodes ** 2
            weight = np.where(mask[:, None], 1 / (s[None, :, None] - poles[:, None]), 0)
            slope = -weight ** 2
            passive = np.where(mask, 1 / (poles + eta ** 2), 0)
            m1, m3 = c @ adj(c) / 2, (c * poles[:, None]) @ adj(c) / 2
            model = (_batch_put(mesh, c), _batch_put(mesh, poles), _batch_put(mesh, mask))
            signed = ()
        wc = (c[:, None] * weight[:, :, None]) @ adj(c)[:, None]
        dw = (c[:, None] * slope[:, :, None]) @ adj(c)[:, None]
        moments = tuple(_batch_put(mesh, m) for m in (m1, m3))
        response = (c * passive[:, None]) @ adj(c)
        expected = np.linalg.eigvalsh((response + adj(response)) / 2)[:, -1]
        run = lambda w: check_round(model, signed, inverse, tuple(_batch_put(mesh, a) for a in (w, dw)), moments, qi,
                                    real=3, nodes=nodes, eta_ry=eta, mesh_xy=mesh, native_eigh=native,
                                    ordered=ordered)
        passed, held, reciprocity, defects = run(wc)
        assert held.shape == (ranks, 2, 2) and float(np.max(held[:3])) < 1e-12, held
        assert np.all(held[3] == 0) and not passed["passivity"][3]
        np.testing.assert_allclose(passed["passivity_max"][:3], expected[:3], rtol=1e-12, atol=1e-15)
        assert passed["passivity"][:3].all()
        assert max(float(np.max(v["full_relative"][:3])) for v in defects.values()) < 1e-12, defects
        if ordered:
            assert reciprocity == {}
        else:
            assert reciprocity["passed"].shape == (ranks, 2, 2) and not reciprocity["applicable"][:3].any()
        _, red, _, _ = run(wc * 1.01)
        np.testing.assert_allclose(red[:3, 0], .01 / 1.01, rtol=1e-12)
        assert float(np.max(red[:3, 1])) < 1e-12
        rows["ordered" if ordered else "even"] = float(np.max(held[:3]))
    return dict(status='PASS', max_held_relative=rows,
                scope='round checks: held W/dW, passivity and moment identities on both routes, synthetic skip, red twin')


def test_round_checks():
    import jax
    from jax.sharding import Mesh
    from lxkit.testing import require_devices
    require_devices(4, 'cpu')
    check(Mesh(np.asarray(jax.devices('cpu')[:4]).reshape(2, 2), ('x', 'y')))


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    initialize_communicator_stack(platform='gpu')

    def main():
        import sys
        import jax
        from common.collectives import resolve_mesh, barrier
        result = check(resolve_mesh())
        result['job_step'] = os.environ['SLURM_JOB_ID'] + '.' + os.environ['SLURM_STEP_ID']
        if jax.process_index() == 0:
            Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
        barrier('local-checks-test')
    run_main_and_finalize(main)
