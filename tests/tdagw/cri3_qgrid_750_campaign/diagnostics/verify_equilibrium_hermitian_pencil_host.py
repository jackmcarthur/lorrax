"""Tiny HOST-ONLY proof of a proposed equilibrium value-state realization.

This is not a production reducer or a material correctness/accuracy result.
The actual finite_pencil_column owner assembles the small G/H; NumPy/SciPy
then demonstrate the positive-H congruence. No existing fitted pole is edited.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import scipy.linalg as la


def reduce_host_proof(g, h, output, gamma):
    """Proof-only dense congruence; refuse non-dark unresolved metric modes."""
    eps = np.finfo(float).eps
    n = len(h)
    if max(la.norm(g-g.conj().T), la.norm(h-h.conj().T)) > 512*n*eps*max(la.norm(g), la.norm(h), 1.):
        raise ValueError('GATE host_eq_pencil: non-Hermitian input')
    eigen, vectors = la.eigh(h)
    scale = max(float(np.max(abs(eigen))), 1.)
    rounding = 512*n*eps*scale
    if eigen[0] < -rounding:
        raise ValueError('GATE host_eq_pencil: resolved negative H; no PSD repair')
    keep = eigen > rounding
    null = vectors[:, ~keep]
    if null.size and (la.norm(g@null) > 2048*n*eps*max(la.norm(g), 1.)
                     or la.norm(output@null) > 2048*n*eps*max(la.norm(output), 1.)):
        raise ValueError('GATE host_eq_pencil: unresolved live H direction')
    if not keep.any():
        raise ValueError('GATE host_eq_pencil: empty state space')
    z = vectors[:, keep]/np.sqrt(eigen[keep])[None]
    metric = z.conj().T@g@z
    mu, rotation = la.eigh(metric)
    c = output@z@rotation
    mu_rounding = 512*len(mu)*eps*max(float(np.max(abs(mu))), 1.)
    finite = abs(mu) > mu_rounding
    # A zero inverse-frequency with nonzero output is a CONSTANT term, not
    # automatically a removable state. The toy admits only exact dark zeros.
    if np.any(c[:, ~finite] != 0):
        raise ValueError('GATE host_eq_pencil: non-dark zero/unresolved mu gives feedthrough')
    mu, c = mu[finite], c[:, finite]
    return dict(poles=1/mu-1j*gamma, factors=c/np.sqrt(abs(mu))[None],
        signs=np.sign(mu), state_rank=int(keep.sum()), exact_dark_mu=int((~finite).sum()),
        h_min=float(eigen[0]), h_roundoff_bound=rounding)


def model_value(model, z):
    b = model['factors']
    return (b*(model['signs']/(z-model['poles']))[None])@b.conj().T


def run_host_proof():
    """Actual owner versus independent four-state gyrotropic q/−q models."""
    import jax
    from gw.shared_pole_pencil import finite_pencil_column
    jax.config.update('jax_enable_x64', True)
    rng = np.random.default_rng(20261021)
    plus = rng.normal(size=(2, 4, 2))+1j*rng.normal(size=(2, 4, 2))
    omega = np.array([[.6, 1.4], [.8, 1.9]])
    signature = np.diag([1., 1., -1., -1.])
    gamma = .037
    nodes = np.repeat(np.array([.21+.18j, .47j, .72+.22j, 1.22+.31j]), 2)
    directions = rng.normal(size=(4, 8))+1j*rng.normal(size=(4, 8))
    def data(q):
        return np.concatenate((plus[q], plus[1-q].conj()), axis=1), np.r_[omega[q], omega[1-q]]
    def exact(q, z):
        c, energy = data(q)
        return (c/(np.diag(signature)*(z+1j*gamma)-energy)[None])@c.conj().T
    def mm(a, b, transa=None):
        return (a.conj().swapaxes(-1, -2) if transa == 'C' else a)@b
    models, errors, gram_errors, c1_norms = [], [], [], []
    held = np.array([.31+.04j, -.43+.2j, .13+.9j, 1.5+.08j])
    for q in range(2):
        c, energy = data(q)
        c1_norms.append(float(la.norm(c@signature@c.conj().T)))
        zeta = nodes+1j*gamma
        output = np.stack([exact(q, node)@directions[:, i] for i, node in enumerate(nodes)], axis=1)
        states = np.stack([c.conj().T@directions[:, i]/(np.diag(signature)*z-energy)
                           for i, z in enumerate(zeta)], axis=1)
        args = (zeta[None], directions[None], output[None])
        g, h = finite_pencil_column(args, (*args, np.zeros_like(output)[None]), matmul=mm)
        g, h = np.asarray(g)[0], np.asarray(h)[0]
        gram_errors.extend([float(la.norm(g-states.conj().T@signature@states)/la.norm(g)),
            float(la.norm(h-states.conj().T@(energy[:, None]*states))/la.norm(h))])
        model = reduce_host_proof(g, h, output, gamma)
        models.append(model)
        for node in np.r_[nodes, held]:
            errors.append(float(la.norm(model_value(model, node)-exact(q, node))/la.norm(exact(q, node))))
        expected = np.sort(np.r_[omega[q], -omega[1-q]])
        np.testing.assert_allclose(np.sort(model['poles'].real), expected, rtol=3e-11, atol=3e-11)
        np.testing.assert_array_equal(model['poles'].imag, np.full(4, -gamma))
    crossing = max(float(la.norm(model_value(models[0], z)-model_value(models[1], -z.conjugate()).conj())
                          /la.norm(exact(0, z))) for z in held)
    not_even = float(la.norm(exact(0, held[0])-exact(0, -held[0]))/la.norm(exact(0, held[0])))
    zeta_probe = .31+.4j
    undamped_not_even = float(la.norm(exact(0, zeta_probe-1j*gamma)
        -exact(0, -zeta_probe-1j*gamma))/la.norm(exact(0, zeta_probe-1j*gamma)))
    assert (max(errors) < 3e-11 and max(gram_errors) < 3e-12 and crossing < 3e-11
            and not_even > .1 and undamped_not_even > .1 and min(c1_norms) > .1)
    # The two null-mu cases expose why infinite-mode dropping is not implicit.
    dark = reduce_host_proof(np.diag([1., 0.]), np.eye(2), np.array([[1., 0.]]), gamma)
    assert dark['exact_dark_mu'] == 1
    refusals = []
    for g, h, o in ((np.diag([1., 0.]), np.eye(2), np.eye(2)),
                    (np.eye(2), np.diag([1., -.1]), np.eye(2))):
        try:
            reduce_host_proof(g, h, o, gamma)
        except ValueError as error:
            refusals.append(str(error))
        else:
            raise AssertionError('required no-repair refusal missing')
    return dict(status='HOST_ONLY_PROPOSED_EQ_PENCIL_PASS', production_route=False,
        material_qualified=False, native_GPU_test=False, full_complex_held_relative_error=max(errors),
        latent_gram_relative_error=max(gram_errors), q_minus_q_relative_error=crossing,
        not_even_frequency_relative_difference=not_even, gamma_ry=gamma,
        undamped_not_even_frequency_relative_difference=undamped_not_even,
        state_ranks=[m['state_rank'] for m in models], node_count=len(nodes),
        nonzero_C1_norms=c1_norms, zero_C1_assumed=False,
        h_min=[m['h_min'] for m in models], h_roundoff_bound=[m['h_roundoff_bound'] for m in models],
        real_zeta_poles=True, exact_dark_mu_removed=1, required_refusals=refusals,
        finite_owner='gw.shared_pole_pencil.finite_pencil_column',
        scope='Tiny complex gyrotropic signed-channel proof only; no material reduction, speed or nonequilibrium theorem')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    sys.path.insert(0, str(args.source.resolve()/'src'))
    from runtime.source_closure import ensure_source_closure
    ensure_source_closure()
    started = time.perf_counter()
    result = run_host_proof()
    owner = args.source/'src/gw/shared_pole_pencil.py'
    result.update(elapsed_seconds=time.perf_counter()-started, source=str(args.source.resolve()),
        finite_owner_sha256=hashlib.sha256(owner.read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
