"""Receiving Hartree adjoints against independently projected point densities.

The standalone compute proof requires four MPI ranks and four GPUs. It tests
complex four-spinor charge, exact monopoles and both periodic means, not the
accuracy of any reconstructed atomic field.
"""
if __name__ == '__main__':
    from runtime import initialize_communicator_stack
    RUNTIME = initialize_communicator_stack(platform='gpu')

import importlib.util
from pathlib import Path
import numpy as np


def check_point_contraction(mesh):
    from scipy.integrate import lebedev_rule
    from scipy.special import sph_harm_y
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from gw.augmentation_hartree_receiving import _make_point_contraction
    from isdf.atomic_hartree import charge_hartree_functional, make_charge_hartree_tile
    from isdf.atomic_moments import exact_pair_moments
    # Share the existing source fixture, rather than a second source builder.
    path = Path(__file__).with_name('test_atomic_hartree.py')
    spec = importlib.util.spec_from_file_location('hartree_source_fixture', path)
    fixtures = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixtures)
    operand, _, _, _, _, _ = fixtures.local_functional_fixture()
    radius = (np.arange(1, 9)/8)**3
    lm = np.asarray([(l, m) for l in range(2) for m in range(-l, l+1)])
    directions, aw = lebedev_rule(5); directions = directions.T
    theta = np.arccos(directions[:, 2]); phi = np.arctan2(directions[:, 1], directions[:, 0])
    Y = np.asarray([sph_harm_y(l, m, theta, phi) for l, m in lm])
    nb, logical, na = 16, 13, 1
    npoint = len(radius)*len(directions); pad = int(mesh.shape['y'])
    active = np.arange(npoint+pad) < npoint
    rng = np.random.default_rng(218013)
    ps = (rng.normal(size=(nb, 4, npoint))+1j*rng.normal(size=(nb, 4, npoint)))*.03
    delta = (rng.normal(size=ps.shape)+1j*rng.normal(size=ps.shape))*.01
    ps[logical:] = 0.; delta[logical:] = 0.
    C = (rng.normal(size=(nb, 3))+1j*rng.normal(size=(nb, 3)))*.03
    D = (rng.normal(size=C.shape)+1j*rng.normal(size=C.shape))*.005
    C[logical:] = 0.; D[logical:] = 0.
    B = rng.normal(size=(3, 3))+1j*rng.normal(size=(3, 3)); B = B+B.conj().T
    exact = exact_pair_moments(C, D, C, D, B)[None, :, :, None]/np.sqrt(4*np.pi)
    # Literal receiving charge products and spherical projection are the
    # independent reference; no production angular buckets/point adjoints.
    ps_density = np.einsum('ism,jsm->ijm', ps.conj(), ps)
    delta_density = (np.einsum('ism,jsm->ijm', ps.conj(), delta)
        + np.einsum('ism,jsm->ijm', delta.conj(), ps)
        + np.einsum('ism,jsm->ijm', delta.conj(), delta))
    grid_scale = np.prod(operand['fft_grid'])/operand['volume']
    def project(density):
        return grid_scale*np.einsum('ijard,hd,d->ijahr',
            density.reshape(nb, nb, na, len(radius), len(directions)), Y.conj(), aw)[None]
    tp, td = project(ps_density), project(delta_density)
    boxes = np.zeros((1, nb, 4, *operand['fft_grid']), complex)
    _, body, local, mean, _ = map(np.asarray,
        make_charge_hartree_tile(operand)(boxes, boxes, tp, td, exact))
    expected = np.concatenate((body[:, 0][None], local[:, :, 0], mean[:, 0][None]))
    functional = charge_hartree_functional(operand)
    kwargs = dict(radius=radius, directions=directions, angular_weights=aw,
                  lm=lm, Y=Y, band_tile=8, point_active=active)
    contract, _ = _make_point_contraction(mesh, functional=functional, **kwargs)
    put = lambda a, spec: device_put_process_local(np.asarray(a), NamedSharding(mesh, spec))
    # Poison only unused points, before the receiving kernel masks them.
    ps = np.pad(ps[None], ((0, 0), (0, 0), (0, 0), (0, pad)), constant_values=np.inf)
    delta = np.pad(delta[None], ((0, 0), (0, 0), (0, 0), (0, pad)), constant_values=np.nan)
    pf, df = (put(a, P(None, 'x', None, 'y')) for a in (ps, delta))
    ef = put(exact, P(None, 'x', 'y', None))
    actual = np.asarray(gather_to_host(contract(pf, df, ef)))
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-13)
    assert np.isfinite(actual).all()
    assert np.max(abs(actual[:, :, logical:])) == 0
    assert np.max(abs(actual[:, :, :, logical:])) == 0
    assert np.min(np.max(abs(expected), axis=(1, 2, 3))) > 1e-7
    wrong = dict(functional, receiving_components={k: v.conj()
        for k, v in functional['receiving_components'].items()})
    wrong_contract, _ = _make_point_contraction(mesh, functional=wrong, **kwargs)
    wrong_signal = float(np.max(abs(np.asarray(gather_to_host(wrong_contract(pf, df, ef)))-actual)))
    assert wrong_signal > 1e-4
    return dict(max_component_error_Ry=float(np.max(abs(actual-expected))),
        wrong_receiving_adjoint_signal_Ry=wrong_signal,
        poisoned_point_suffix_inert=True, ghost_bands_exact_zero=True,
        four_spinors=True, exact_M0_and_both_means=True)


def test_receiving_point_adjoint_matches_independent_pair_projection():
    from test_hartree_point_trace import cpu_mesh
    check_point_contraction(cpu_mesh())


if __name__ == '__main__':
    import argparse
    import json
    import jax
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if RUNTIME.process_count != 4 or len(jax.local_devices()) != 1:
        raise ValueError('Standalone receiving regression requires actual four-rank/four-GPU P4')
    result = dict(status='PASS', rank=RUNTIME.process_index, ranks=4,
                  **check_point_contraction(RUNTIME.mesh))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output/f'rank{RUNTIME.process_index}.json'
    if path.exists(): raise FileExistsError('Preserve completed receiving regression')
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
