"""Independent finite-q, covariance, and artifact guards for served moments."""
from __future__ import annotations


def _synthetic_cache():
    import numpy as np
    from psp.augmentation_spinors import spinor_function_labels
    from isdf.atomic_moments import _served_operator_identity

    rng = np.random.default_rng(610111)
    kappa = np.repeat((-1, 1, -2, 2, -3), 5)
    ell = np.where(kappa < 0, -kappa-1, kappa)
    labels = spinor_function_labels(ell, kappa)
    opf, mj = labels.T
    radial_B = np.zeros((len(kappa), len(kappa)), complex)
    for k in np.unique(kappa):
        ids = np.flatnonzero(kappa == k)
        x = rng.normal(size=(len(ids), len(ids)))/7
        radial_B[np.ix_(ids, ids)] = x.T@x
    B = radial_B[opf[:, None], opf[None]]*(mj[:, None] == mj[None])
    momentum = np.linspace(0., 16., 8)
    leg = lambda l, k: dict(momentum=momentum.copy(), radial=np.zeros((8, len(kappa)), complex),
        ell=l.copy(), kappa=k.copy(), maximum_absolute_error=0., maximum_scaled_error=0., validation_points=7)
    return dict(B=B, labels=labels, upper=leg(ell, kappa), lower=leg(2*abs(kappa)-1-ell, -kappa),
        support_radius=2.6, lower_radius=0., source_quad_order=5, source_identity=_served_operator_identity(),
        field_model='unwindowed_independent_hermite')


def test_auxiliary_signed_integral_and_finite_q():
    import numpy as np
    from isdf.atomic_moments import auxiliary_charge_geometry, evaluate_auxiliary_charge, exact_pair_moments

    cache = _synthetic_cache()
    geometry = auxiliary_charge_geometry(cache)
    rng = np.random.default_rng(610112)
    random = lambda shape: (rng.normal(size=shape)+1j*rng.normal(size=shape))/11
    n = len(cache['labels'])
    C, D, CC, DD = (random((nb, n)) for nb in (6, 6, 9, 9))
    scale = .73
    plus, minus = evaluate_auxiliary_charge(C, D, geometry, grid_sample_scale=scale)
    phase = np.exp(2j*np.pi*.319)
    right_plus, right_minus = evaluate_auxiliary_charge(CC*phase, DD*phase, geometry, grid_sample_scale=scale)
    values = (np.einsum('nsp,msp,p->nm', plus.conj(), right_plus, geometry['integration_weights'])
              - np.einsum('nsp,msp,p->nm', minus.conj(), right_minus, geometry['integration_weights']))
    exact = scale**2*exact_pair_moments(C, D, CC*phase, DD*phase, cache['B'])
    np.testing.assert_allclose(values, exact, atol=2e-12, rtol=2e-12)
    assert np.count_nonzero(plus[:, 2:]) == np.count_nonzero(minus[:, 2:]) == 0
    assert geometry['carrier'] == 'auxiliary_charge_functional'
    assert len(geometry['relative_points']) == 208
    wrong = scale**2*exact_pair_moments(C, D, CC.conj()*phase, DD.conj()*phase, cache['B'])
    assert np.max(abs(wrong-exact)) > .01


def test_auxiliary_origin_and_quadrature_guards():
    import numpy as np
    import pytest
    from isdf.atomic_moments import auxiliary_charge_geometry, evaluate_auxiliary_charge

    cache = _synthetic_cache()
    with pytest.raises(ValueError, match='radial quadrature'):
        auxiliary_charge_geometry(cache, radial_points=7)
    with pytest.raises(ValueError, match='angular quadrature'):
        auxiliary_charge_geometry(cache, angular_order=3)
    geometry = auxiliary_charge_geometry(cache)
    with pytest.raises(ValueError, match='carrier'):
        evaluate_auxiliary_charge(np.zeros((2, len(cache['labels']))), np.zeros((2, len(cache['labels']))),
                                  dict(geometry, carrier='normalized_rkb'), grid_sample_scale=1.)
    cache['B'][0, 0] += 1e-3j
    with pytest.raises(ValueError, match='real atomic radial metric'):
        auxiliary_charge_geometry(cache)


def test_served_cache_authentication(tmp_path):
    import json
    import numpy as np
    import pytest
    from isdf.atomic_moments import write_served_moment_cache, load_served_moment_cache

    cache = _synthetic_cache()
    path = tmp_path/'served.npz'
    write_served_moment_cache(path, cache, normalized_cache_sha256='a'*64)
    loaded = load_served_moment_cache(path, normalized_cache_sha256='a'*64, support_radius=2.6)
    np.testing.assert_array_equal(loaded['B'], cache['B'])
    with pytest.raises(FileExistsError):
        write_served_moment_cache(path, cache, normalized_cache_sha256='a'*64)
    with pytest.raises(ValueError, match='identity'):
        load_served_moment_cache(path, normalized_cache_sha256='b'*64, support_radius=2.6)
    with pytest.raises(ValueError, match='identity'):
        load_served_moment_cache(path, normalized_cache_sha256='a'*64, support_radius=2.5)
    with np.load(path, allow_pickle=False) as data:
        payload = {key: data[key].copy() for key in data.files}
    metadata = json.loads(str(payload['metadata_json']))
    metadata['source_identity']['operator_source_sha256'] = '0'*64
    payload['metadata_json'] = np.asarray(json.dumps(metadata))
    bad = tmp_path/'bad_source.npz'
    np.savez(bad, **payload)
    with pytest.raises(ValueError, match='identity'):
        load_served_moment_cache(bad, normalized_cache_sha256='a'*64, support_radius=2.6)
    payload['metadata_json'] = np.asarray(json.dumps({**metadata, 'source_identity': cache['source_identity']}))
    payload['upper_radial'][0, 0] = 1.
    bad_payload = tmp_path/'bad_payload.npz'
    np.savez(bad_payload, **payload)
    with pytest.raises(ValueError, match='payload'):
        load_served_moment_cache(bad_payload, normalized_cache_sha256='a'*64, support_radius=2.6)


def test_monopole_units_and_padding():
    import numpy as np
    from isdf.atomic_moments import integrated_auxiliary_monopole

    rng = np.random.default_rng(610113)
    values = rng.normal(size=(4, 6, 11))+1j*rng.normal(size=(4, 6, 11))
    weights = rng.random(11)
    values[3] = 0.
    # Packed auxiliary points with zero weights must not contribute.
    values[..., -2:] = 1e30*(1+1j)
    weights[-2:] = 0.
    got = integrated_auxiliary_monopole(values, weights)
    reference = sum(values[..., p]*weights[p] for p in range(9))/np.sqrt(4*np.pi)
    np.testing.assert_allclose(got, reference, atol=2e-15)
    assert np.count_nonzero(got[3]) == 0


def test_raw_parent_cache_window_and_source_guards(tmp_path):
    from types import SimpleNamespace
    import hashlib
    import numpy as np
    import pytest
    from isdf.atomic_moments import raw_parent_moment_binding, write_raw_parent_moments, load_raw_parent_moments

    wfn = SimpleNamespace(nbands=6, nspinor=2, nelec=2, energies=np.zeros((3, 6)),
                         kpoints=np.asarray([[0., 0., 0.], [1/3, 0., 0.], [2/3, 0., 0.]]), path=None)
    g = np.zeros((3, 9, 3), int)
    g[:, :7, 0] = np.arange(7)
    options = dict(k_parent_frac=wfn.kpoints, gvecs=g, ngk_valid=np.full(3, 7),
        centers_cart=np.asarray([[0., 0., 0.], [.5, 0., 0.]]), atom_types=np.asarray([47, 53]),
        cell_volume=95., physical_bands=6, served_cache_sha256_by_species={47: 'a'*64, 53: 'b'*64})
    binding = raw_parent_moment_binding(wfn, **options)
    g[:, 7:] = 123456
    assert raw_parent_moment_binding(wfn, **options) == binding
    D = (np.ones((3, 6, 4), complex), np.ones((3, 6, 7), complex)*(.3+.2j))
    source = np.arange(96, dtype=np.uint8).reshape(3, 32)
    target = tmp_path/'raw.npz'
    write_raw_parent_moments(target, D, source, binding=binding)
    sha = hashlib.sha256(target.read_bytes()).hexdigest()
    got = load_raw_parent_moments(target, expected_binding=binding, expected_file_sha256=sha)
    np.testing.assert_array_equal(got['atom_D'][1], D[1])
    with pytest.raises(ValueError, match='full physical WFN'):
        raw_parent_moment_binding(wfn, **dict(options, physical_bands=5))
    with pytest.raises(ValueError, match='file identity'):
        load_raw_parent_moments(target, expected_binding=binding, expected_file_sha256='0'*64)
    with pytest.raises(ValueError, match='full-window identity'):
        load_raw_parent_moments(target, expected_binding=dict(binding, band_range=[0, 5]), expected_file_sha256=sha)
    with pytest.raises(ValueError, match='unpadded full physical bands'):
        write_raw_parent_moments(tmp_path/'padded.npz', (np.pad(D[0], ((0,0),(0,2),(0,0))), D[1]), source, binding=binding)
