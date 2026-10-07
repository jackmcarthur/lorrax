"""Distributed diagonal sample action against literal complex band pairs."""
from pathlib import Path
import json
import os

import numpy as np
import pytest


def check_diagonal_sample_projection(mesh):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.contract_bands import make_diagonal_sample_projector

    rng = np.random.default_rng(360507)
    # Multiple k rows, four spins, non-prefix sample ghosts, and unequal
    # physical/carrier bands expose slab ordering and padding mistakes.
    nk, nb, ns, nmu = 3, 8, 4, 12
    band_mask = np.arange(nb) < 6
    sample_mask = np.ones(nmu, bool)
    sample_mask[[2, 9]] = False
    clean = (rng.normal(size=(nk, nb, ns, nmu))
             + 1j*rng.normal(size=(nk, nb, ns, nmu))) / 30.
    clean[:, ~band_mask] = 0.
    clean[..., ~sample_mask] = 0.
    real_values = rng.normal(size=(2, nmu))
    complex_values = real_values + 1j*rng.normal(size=(2, nmu))
    real_values[:, ~sample_mask] = 0.
    complex_values[:, ~sample_mask] = 0.

    def put(value, spec):
        return device_put_process_local(value, NamedSharding(mesh, spec))

    face_spec, value_spec = P(None, 'x', None, 'y'), P(None, 'y')
    project = make_diagonal_sample_projector(
        mesh, clean.shape, band_mask=band_mask, sample_mask=sample_mask)

    def literal(face, values):
        # Explicit endpoint pair cloud is permitted only in this tiny oracle.
        pair = face.conj()[:, :, None] * face[:, None, :]
        return np.stack([np.sum(pair * v[None, None, None, None, :],
                                axis=(-1, -2)) for v in values])

    def evaluate(face, values):
        result = project(put(face, face_spec), put(values, value_spec))
        assert result.sharding.is_equivalent_to(
            NamedSharding(mesh, P(None, None, 'x', 'y')), ndim=4)
        return np.asarray(gather_to_host(result))

    expected = literal(clean, complex_values)
    actual = evaluate(clean, complex_values)
    error = float(np.max(abs(actual-expected)))
    assert error < 4e-15
    hermitian = evaluate(clean, real_values)
    assert np.max(abs(hermitian-hermitian.swapaxes(-1, -2).conj())) < 4e-15
    # Complex functionals must retain their literal non-Hermitian action;
    # the projection is not allowed to clip or symmetrize an operand.
    assert np.max(abs(actual-actual.swapaxes(-1, -2).conj())) > 1e-3

    poisoned = clean.copy()
    poisoned[:, ~band_mask] = np.nan + 1j*np.nan
    poisoned[..., ~sample_mask] = np.nan + 1j*np.nan
    bad_values = complex_values.copy()
    bad_values[:, ~sample_mask] = np.nan + 1j*np.nan
    padded = evaluate(poisoned, bad_values)
    np.testing.assert_array_equal(padded, actual)
    assert np.max(abs(padded[:, :, ~band_mask])) == 0.
    assert np.max(abs(padded[:, :, :, ~band_mask])) == 0.

    # Independent band gauges require a bra phase and a ket phase.
    phases = np.exp(1j*rng.normal(size=(nk, nb)))
    gauged = evaluate(clean*phases[:, :, None, None], complex_values)
    gauge_expected = expected*phases.conj()[None, :, :, None]*phases[None, :, None, :]
    gauge_error = float(np.max(abs(gauged-gauge_expected)))
    assert gauge_error < 4e-15

    # A changed centroid order is equivalent only when BOTH operands and
    # the active mask follow that order; no hidden canonical/packed map.
    order = rng.permutation(nmu)
    reordered = make_diagonal_sample_projector(
        mesh, clean.shape, band_mask=band_mask, sample_mask=sample_mask[order])
    moved = np.asarray(gather_to_host(reordered(
        put(clean[..., order], face_spec), put(complex_values[:, order], value_spec))))
    order_error = float(np.max(abs(moved-expected)))
    assert order_error < 4e-15
    wrong_order = evaluate(clean, complex_values[:, order])
    assert np.max(abs(wrong_order-expected)) > 1e-3

    # Physical source occupations and k weights belong to the source, not
    # this receiving contraction. A positive discrete potential formed from
    # fractional occupations is compared to its explicit pair action.
    occupations = np.asarray([[1., .7, .3, 0., 0., 0., 0., 0.],
                              [.9, .6, .2, 0., 0., 0., 0., 0.],
                              [.8, .5, .1, 0., 0., 0., 0., 0.]])
    kweights = np.asarray([.2, .3, .5])
    density = np.sum(abs(clean)**2 * occupations[:, :, None, None]
                     * kweights[:, None, None, None], axis=(0, 1, 2))
    kernel = rng.normal(size=(nmu, nmu))
    potential = (kernel.T @ kernel) @ density
    potential[~sample_mask] = 0.
    occupied = evaluate(clean, potential[None])
    occ_expected = literal(clean, potential[None])
    occupation_error = float(np.max(abs(occupied-occ_expected)))
    assert occupation_error < 4e-15
    assert np.min(np.linalg.eigvalsh(occupied[0, :, :6, :6])) > -4e-15
    zero = evaluate(clean, np.zeros_like(real_values))
    np.testing.assert_array_equal(zero, np.zeros_like(zero))

    # An extra grid normalization or a second receiver conjugation is a
    # material error even though dimensions and diagonal norms still pass.
    wrong_conjugation = literal(clean.conj(), complex_values)
    conjugation_red = float(np.max(abs(wrong_conjugation-expected)))
    normalization_red = float(np.max(abs(actual/nmu-expected)))
    assert min(conjugation_red, normalization_red) > 1e-3

    for keyword, changed in (("band_mask", np.arange(nb)),
                             ("sample_mask", sample_mask[:-1])):
        with pytest.raises(ValueError, match="mask"):
            make_diagonal_sample_projector(mesh, clean.shape, **{keyword: changed})
    with pytest.raises(ValueError, match="shape or dtype"):
        project(put(clean.astype(np.complex64), face_spec), put(real_values, value_spec))
    with pytest.raises(ValueError, match="shape or dtype"):
        project(put(clean, face_spec), put(real_values[:, :-2], value_spec))
    if mesh.shape['x'] > 1:
        with pytest.raises(ValueError, match="layout"):
            project(put(clean, P()), put(real_values, value_spec))
    return dict(maximum_complex_pair_error=error, gauge_error=gauge_error,
        consistent_sample_order_error=order_error,
        nonuniform_occupation_action_error=occupation_error,
        wrong_receiver_conjugation_error=conjugation_red,
        wrong_extra_normalization_error=normalization_red,
        poisoned_band_and_sample_ghosts_bitwise_inert=True,
        zero_functional_exact=True, physical_bands=6, carrier_bands=nb,
        nk=nk, spinor_components=ns, sample_extent=nmu,
        output_sharding="P(None,None,x,y)", mesh=dict(mesh.shape))


def test_diagonal_sample_projection_single_device():
    import jax
    from jax.sharding import Mesh
    jax.config.update('jax_enable_x64', True)
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ('x', 'y'))
    check_diagonal_sample_projection(mesh)


if __name__ == '__main__':
    from runtime import initialize_communicator_stack, run_main_and_finalize
    runtime = initialize_communicator_stack()
    def main():
        import jax
        result = check_diagonal_sample_projection(runtime.mesh)
        if jax.process_index() == 0:
            print(json.dumps(result), flush=True)
            output = os.environ.get('DIAGONAL_SAMPLE_REPORT')
            if output:
                Path(output).write_text(json.dumps(result, indent=2)+'\n')
        return 0
    run_main_and_finalize(main)
