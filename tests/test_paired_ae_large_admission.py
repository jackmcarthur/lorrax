"""Closed target admission checks; synthetic fields are not atomic evidence."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from psp import augmentation_cache as cache_owner
from psp import reconstruction_overlap as overlap_owner
from psp.augmentation_spinors import free_graph_small_from_large, spinor_function_labels


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, arrays, metadata):
    np.savez(path, **arrays, metadata_json=np.asarray(json.dumps(dict(metadata,
        payload_sha256=cache_owner._payload_hash(arrays)), sort_keys=True)))
    return _sha(path)


def test_closed_policy_contracts_are_distinct():
    native = cache_owner.paired_field_policy_contract(cache_owner.PAIRED_COMPACT_PAULI_FIELD_POLICY)
    ae = cache_owner.paired_field_policy_contract(cache_owner.PAIRED_AE_LARGE_FIELD_POLICY)
    assert native['frame_model'] == overlap_owner.COMPACT_PAULI_FRAME_MODEL
    for key in native:
        assert native[key] != ae[key]


@pytest.mark.parametrize('policy', [None, '', 'ae_large', True, {}])
def test_unknown_paired_policy_refuses(policy):
    with pytest.raises(ValueError, match='Unknown paired'):
        cache_owner.paired_field_policy_contract(policy)


@pytest.mark.parametrize('carrier', ['pauli2embed4', 'normalized_rkb'])
def test_represented_metric_diagnostic_keeps_spectral_and_sphere_domains_distinct(tmp_path, carrier):
    from gw.isdf_augmentation import _represented_target_metric_diagnostic

    target = np.eye(2, dtype=np.complex128)
    native, ae = tmp_path/'native.npz', tmp_path/'ae.npz'
    np.savez(native, **{carrier+'_sphere_B':target+np.diag([.2, -.1])})
    np.savez(ae, represented_spectral_B=target+np.diag([.003, -.002]))
    species = dict(metric_witness=str(native), target_witness_file=str(ae), target_witness_sha256=_sha(ae))
    for policy, expected in ((cache_owner.PAIRED_COMPACT_PAULI_FIELD_POLICY,
                              ('represented_species_B_error', .2)),
                             (cache_owner.PAIRED_AE_LARGE_FIELD_POLICY,
                              ('represented_species_spectral_B_error', .003))):
        name, error = _represented_target_metric_diagnostic(species, target,
            field_policy=policy, carrier=carrier)
        assert name == expected[0]
        assert error == pytest.approx(expected[1])
    assert np.array_equal(target, np.eye(2))


@pytest.mark.parametrize('change', ['shape', 'dtype', 'nonfinite', 'wrong_key', 'wrong_source'])
def test_ae_spectral_diagnostic_refuses_incompatible_witness(tmp_path, change):
    from gw.isdf_augmentation import _represented_target_metric_diagnostic

    value = np.eye(2, dtype=np.complex128)
    if change == 'shape': value = value[:1]
    if change == 'dtype': value = value.real
    if change == 'nonfinite': value[0, 0] = np.nan
    key = 'normalized_rkb_sphere_B' if change == 'wrong_key' else 'represented_spectral_B'
    path = tmp_path/'witness.npz'; np.savez(path, **{key:value})
    species = dict(target_witness_file=str(path), target_witness_sha256=_sha(path))
    if change == 'wrong_source':
        other = tmp_path/'other.npz'
        np.savez(other, represented_spectral_B=2*value)
        species['target_witness_file'] = str(other)
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        _represented_target_metric_diagnostic(species, np.eye(2, dtype=np.complex128),
            field_policy=cache_owner.PAIRED_AE_LARGE_FIELD_POLICY, carrier='normalized_rkb')


def _field_fixture(tmp_path):
    from common.bispinor_init import lift_to_4spinor
    from common import gamma_matrices
    from psp.augmentation_spinors import evaluate_normalized_radials

    r = np.linspace(0., 3., 25)
    fields = dict(radius=r, ell=np.array([0], np.int32), kappa=np.array([-1], np.int32))
    for key in cache_owner.RADIAL_KEYS:
        fields[key] = np.zeros((len(r), 1), np.complex128)
    fields['large_R'][:, 0] = np.exp(-r)
    fields['dlarge_R_dr'][:, 0] = -np.exp(-r)
    data = dict(l=fields['ell'], kappa=fields['kappa'],
        metadata=dict(source_sha256='a'*64, payload_sha256='b'*64))
    witness = tmp_path/'witness'; witness.write_text('constructor witness locator')
    target = dict(kind=cache_owner.AE_LARGE_TARGET, descriptor=dict(nuclear_charge=24))
    metadata = dict(schema='lorrax.dev.paired_ae_large_species_fields.v1',
        source_model='ae_large_preserved_free_graph_with_finiteK_paired_approximation',
        target_kind=cache_owner.AE_LARGE_TARGET, target_binding=target,
        construction_controls=dict(marker='test_only'), controls=dict(support_radius=2.),
        no_post_U_taper=True, carrier='pauli2embed4', source_upf_sha256='a'*64,
        native_payload_sha256='b'*64, native_reconstruction_sha256='c'*64,
        common_spectrum_sha256='d'*64, target_witness_file=str(witness),
        target_witness_sha256=_sha(witness),
        source_owners_sha256={inspect.getsourcefile(owner):_sha(inspect.getsourcefile(owner))
            for owner in (evaluate_normalized_radials, lift_to_4spinor, gamma_matrices)})
    return fields, data, metadata, target


def _load_field(path, data, policy):
    return cache_owner.load_paired_field_cache(path, data, expected_file_sha256=_sha(path),
        common_spectrum_sha256='d'*64, carrier='pauli2embed4', support_radius=2., field_policy=policy)


def test_ae_field_target_bound_and_native_schema_not_relabelled(tmp_path, monkeypatch):
    fields, data, metadata, target = _field_fixture(tmp_path)
    monkeypatch.setattr(cache_owner, '_target_inputs', lambda *_: (data, {}, target['descriptor'], target))
    path = tmp_path/'fields.npz'; _write(path, fields, metadata)
    loaded, binding = _load_field(path, data, cache_owner.PAIRED_AE_LARGE_FIELD_POLICY)
    assert np.array_equal(loaded['large_R'], fields['large_R'])
    assert binding['target_binding'] == target
    assert binding['target_witness_sha256'] == metadata['target_witness_sha256']
    with pytest.raises(ValueError, match='policy/source/spectrum/payload'):
        _load_field(path, data, cache_owner.PAIRED_COMPACT_PAULI_FIELD_POLICY)


@pytest.mark.parametrize('change', ['target', 'witness', 'lower', 'nuclear'])
def test_ae_field_independent_target_and_zero_small_guards(tmp_path, monkeypatch, change):
    fields, data, metadata, target = _field_fixture(tmp_path)
    actual = target
    if change == 'target':
        actual = dict(target, descriptor=dict(nuclear_charge=25))
    if change == 'witness':
        Path(metadata['target_witness_file']).write_text('changed constructor witness')
    if change == 'lower':
        fields['small_R'][1, 0] = 1e-9
    if change == 'nuclear':
        target['descriptor']['nuclear_charge'] = 130
    monkeypatch.setattr(cache_owner, '_target_inputs', lambda *_: (data, {}, actual['descriptor'], actual))
    path = tmp_path/'fields.npz'; _write(path, fields, metadata)
    with pytest.raises(ValueError):
        _load_field(path, data, cache_owner.PAIRED_AE_LARGE_FIELD_POLICY)


def _gram_fixture(tmp_path):
    nodes, weights = np.polynomial.legendre.leggauss(32)
    r, w = (nodes+1)/2, weights/2
    ell = np.array([0, 0, 1], np.int32); kappa = np.array([-1, -1, 1], np.int32)
    f = (1-r*r)**3
    df = -6*r*(1-r*r)**2
    large = np.column_stack((f, (1+.3*r*r)*f, r*f)).astype(np.complex128)
    derivative = np.column_stack((df, .6*r*f+(1+.3*r*r)*df, f+r*df)).astype(np.complex128)
    lower = free_graph_small_from_large(r, large, derivative, kappa)
    upper_B = overlap_owner.atomic_delta_gram(dict(r=r, weights_dr=w,
        delta_u=r[:,None]*large, l=ell, kappa=kappa))
    lower_B = overlap_owner.atomic_delta_gram(dict(r=r, weights_dr=w,
        delta_u=r[:,None]*lower, l=2*np.abs(kappa)-1-ell, kappa=-kappa))
    arrays = dict(source_radius=r, source_weights_dr=w, source_large_R=large,
        source_dlarge_R_dr=derivative, source_lower_free_graph_R=lower,
        ell=ell, kappa=kappa, labels=spinor_function_labels(ell, kappa), target_B=upper_B+lower_B)
    # The combined bank's extra arrays are binding evidence; this test tests
    # the independently remeasured construction B, not spectral accuracy.
    for key in ('momentum', 'weights_dK', 'pauli_radial_spectrum', 'normalized_large_spectrum',
                'normalized_small_spectrum', 'target_large_spectrum', 'native_small_diagnostic_spectrum',
                'source_native_small_R', 'represented_spectral_B'):
        arrays[key] = np.zeros(1, np.complex128)
    target = dict(kind=cache_owner.AE_LARGE_TARGET,
        descriptor=dict(dirac_window_start=.7, dirac_window_stop=1.))
    controls = dict(marker='synthetic_test_only')
    owners = {importlib.util.find_spec(name).origin:_sha(importlib.util.find_spec(name).origin)
        for name in ('psp.augmentation_cache', 'psp.augmentation_spinors', 'common.bispinor_init')}
    metadata = dict(overlap_owner.AE_LARGE_TARGET_WITNESS, construction_controls=controls,
        target_binding=target, construction_window=[.7, 1.], target_radial_domain=[0., 1.],
        source_owners_sha256=owners)
    entry = dict(target_binding=target, construction_controls=controls,
        target_witness_file=str(tmp_path/'target.npz'))
    data = dict(l=ell, kappa=kappa)
    return arrays, metadata, entry, data, target


@pytest.mark.parametrize('change', [None, 'derivative', 'lower', 'B', 'support', 'owner'])
def test_actual_construction_fields_remeasure_sobolev_B(tmp_path, monkeypatch, change):
    arrays, metadata, entry, data, target = _gram_fixture(tmp_path)
    expected = arrays['target_B'].copy()
    if change == 'derivative':
        arrays['source_dlarge_R_dr'][3, 0] += .01
    elif change == 'lower':
        arrays['source_lower_free_graph_R'][3, 0] += .01
    elif change == 'B':
        arrays['target_B'][0, 0] += .01
    elif change == 'support':
        metadata['target_radial_domain'] = [0., 2.]
    elif change == 'owner':
        metadata['source_owners_sha256'] = {name:'0'*64 for name in metadata['source_owners_sha256']}
    path = Path(entry['target_witness_file']); pin = _write(path, arrays, metadata)
    entry['target_witness_sha256'] = pin
    monkeypatch.setattr(cache_owner, '_target_inputs', lambda *_: (data, {}, target['descriptor'], target))
    run = lambda: overlap_owner._ae_large_target_gram_witness(entry, data, {str(path):pin}, support_radius=2.)
    if change is None:
        assert np.array_equal(run(), expected)
    else:
        with pytest.raises(ValueError):
            run()


@pytest.mark.parametrize('change', [None, 'D', 'source_chart', 'gradient_control', 'carrier_spectrum', 'domain'])
def test_independent_full_source_witness_compares_actual_payload(tmp_path, change):
    target = dict(kind=cache_owner.AE_LARGE_TARGET, descriptor=dict(nuclear_charge=24))
    target_hash = hashlib.sha256(json.dumps(target, sort_keys=True, separators=(',',':')).encode()).hexdigest()
    B = np.eye(2, dtype=np.complex128)
    # The frame loader separately requires its complete physical key set.
    # This bounded test isolates the independent-witness actual-array join.
    frame = dict(source_gram=B[None], target_gram=B[None],
        target_gram_quadrature_control=B[None], inverse_sqrt=B[None],
        atom_D_000=np.zeros((1,2,2), np.complex128),
        species_B_24=B, parent_k_frac=np.zeros((1,3), np.float64),
        gvec_parent_000=np.array([[0,0,0],[-1,0,1]], np.int32))
    witness = {key:value.copy() for key,value in frame.items()}
    witness['species_B_gradient_control_24'] = B.copy()
    if change == 'D':
        witness['atom_D_000'][0,0,0] = .01
    elif change == 'source_chart':
        witness['gvec_parent_000'][1,0] = 1
    elif change == 'gradient_control':
        witness['species_B_gradient_control_24'][0,0] += .01
    archive = tmp_path/'independent.npz'; np.savez(archive, **witness)
    program = tmp_path/'producer.py'; program.write_text('independent witness producer\n')
    math = tmp_path/'math.py'; math.write_text('independent literal contractor\n')
    primitive = tmp_path/'primitive.json'; primitive.write_text('{}\n')
    spec = tmp_path/'spec.json'; spec.write_text(json.dumps(dict(math_owner_file=str(math),math_owner_sha256=_sha(math))))
    receipt = dict(schema='lorrax.dev.ae_large_target_witness.v1',
        status='PASS_INDEPENDENT_CONTINUUM_AE_LARGE_TARGET_FULL_SOURCE',
        program_sha256=_sha(program), spec_sha256=_sha(spec), math_owner_sha256=_sha(math),
        arrays_sha256=_sha(archive), wfn_sha256='w'*64, physical_bands=2,
        complete_all_FILE_parents=True, target_binding_by_species={'24':target},
        common_spectrum_sha256_by_species={'24':'s'*64},
        source_files_sha256={str(program):_sha(program),str(math):_sha(math)},
        source_input_files_sha256={str(primitive):_sha(primitive)})
    if change == 'carrier_spectrum':
        receipt['common_spectrum_sha256_by_species']['24'] = 't'*64
    elif change == 'domain':
        receipt['physical_bands'] = 1
    path = tmp_path/'receipt.json'; path.write_text(json.dumps(receipt))
    bound = dict(schema=receipt['schema'], program_file=str(program),program_sha256=_sha(program),
        spec_file=str(spec),spec_sha256=_sha(spec),receipt_file=str(path),receipt_sha256=_sha(path),
        arrays_file=str(archive),arrays_sha256=_sha(archive),target_binding_sha256_by_species={'24':target_hash})
    metadata = dict(physical_bands=2, independent_target_witness=bound,
        atomic_species_inputs={'24':dict(target_binding=target,spectral_witness_sha256='s'*64)})
    pins = {str(p):_sha(p) for p in (program,math,primitive,spec,path,archive)}
    run = lambda: overlap_owner._authenticate_ae_large_frame_witness(metadata,frame,pins,wfn_sha256='w'*64)
    if change is None:
        run()
    else:
        with pytest.raises(ValueError):
            run()
