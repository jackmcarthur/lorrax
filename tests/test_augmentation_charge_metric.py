"""An onsite smooth-neutral metric requires an explicit field representation."""
import json

import pytest


@pytest.mark.parametrize("metric,radial,diagnostic", [
    ({"smooth_neutral_cross": "automatic"}, {"interpolation_degree": 3}, "exactly"),
    ({"smooth_neutral_cross": "onsite", "extra": True}, {"interpolation_degree": 3}, "exactly"),
    ({"smooth_neutral_cross": "onsite"}, {}, "interpolation_degree"),
])
def test_metric_admission_precedes_atomic_data_reads(tmp_path, metric, radial, diagnostic):
    from gw.isdf_augmentation import read_augmentation_manifest
    # If the option were ignored, the deliberately absent sidecar would be
    # read instead of producing the actionable metric-policy refusal.
    control = dict(schema="lorrax.isdf_augmentation.v1", carrier="normalized_rkb",
        frozen_core_policy="reconstruct_valence_only", species={"47": {}},
        radial=radial, angular={}, cache={}, runtime={}, charge_metric=metric)
    (tmp_path / "manifest.json").write_text(json.dumps(control))
    with pytest.raises(ValueError, match=diagnostic):
        read_augmentation_manifest(tmp_path)


def test_retained_smooth_workspace_prices_completion_liveness():
    from gw.isdf_augmentation import _local_rhs_workspace_bytes
    delta = _local_rhs_workspace_bytes(16, 216, 2304, 3320, 16)
    paired = _local_rhs_workspace_bytes(16, 216, 2304, 3320, 16, retain_smooth=True)
    assert delta["full_q_scalar_panels"] == 4
    assert paired["full_q_scalar_panels"] == 6
    assert paired["parent_projectors_and_layout"] == delta["parent_projectors_and_layout"]
    assert paired["total"] - delta["total"] == 2 * delta["full_q_scalar"]


@pytest.mark.parametrize('served,diagnostic', [
    (None,'artifact table'),
    ({'species_files':{},'species_sha256':{},'raw_parent_file':'absent.npz','raw_parent_sha256':'0'*64},'exact species'),
    ({'species_files':{'47':'absent.npz'},'species_sha256':{'47':'0'*64},
      'raw_parent_file':'absent.npz','raw_parent_sha256':'0'*64,'fallback':True},'exact species'),
])
def test_served_moment_admission_has_no_implicit_data_fallback(tmp_path,served,diagnostic):
    from gw.isdf_augmentation import read_augmentation_manifest
    control=dict(schema='lorrax.isdf_augmentation.v1',carrier='normalized_rkb',
        frozen_core_policy='reconstruct_valence_only',species={'47':{}},
        radial={'interpolation_degree':3},angular={},cache={'species_files':{'47':'absent.npz'}},runtime={},
        overlap={'mode':'full_wfn_lowdin'},
        charge_metric={'smooth_neutral_cross':'onsite','moment_enrichment':'served_monopole'})
    if served is not None:control['served_moments']=served
    (tmp_path/'manifest.json').write_text(json.dumps(control))
    with pytest.raises(ValueError,match=diagnostic):read_augmentation_manifest(tmp_path)


def test_prepared_local_cache_requires_pair_before_atomic_reads(tmp_path):
    from gw.isdf_augmentation import read_augmentation_manifest
    control=dict(schema='lorrax.isdf_augmentation.v1',carrier='normalized_rkb',
        frozen_core_policy='reconstruct_valence_only',species={'47':{}},radial={},angular={},
        cache={'local_coulomb_fourier_file':'absent.npz'},runtime={})
    (tmp_path/'manifest.json').write_text(json.dumps(control))
    with pytest.raises(ValueError,match='file/SHA pair'):read_augmentation_manifest(tmp_path)


def test_local_fourier_metadata_does_not_change_normalized_field_controls():
    from gw.isdf_augmentation import _normalized_cache_control
    original={'momentum_max':400.,'radius_points':8800,'species_files':{'47':'normalized.npz'}}
    augmented=dict(original,local_coulomb_fourier_file='local.npz',local_coulomb_fourier_sha256='0'*64)
    assert _normalized_cache_control(augmented)==original
    assert augmented['local_coulomb_fourier_file']=='local.npz'


def test_full_wfn_overlap_requires_served_data_without_monopole(tmp_path):
    from gw.isdf_augmentation import read_augmentation_manifest
    control=dict(schema='lorrax.isdf_augmentation.v1',carrier='normalized_rkb',
        frozen_core_policy='reconstruct_valence_only',species={'47':{}},
        radial={},angular={},cache={},runtime={},overlap={'mode':'full_wfn_lowdin','bands':1})
    (tmp_path/'manifest.json').write_text(json.dumps(control))
    with pytest.raises(ValueError,match='served_moments artifact table'):
        read_augmentation_manifest(tmp_path)


def test_served_overlap_table_admitted_independently_of_monopole(tmp_path,monkeypatch):
    from gw.isdf_augmentation import read_augmentation_manifest
    from psp import atomic_reconstruction
    class AtomicReadReached(Exception):
        pass
    def reached(*args,**kwargs):
        raise AtomicReadReached
    monkeypatch.setattr(atomic_reconstruction,'load_atomic_reconstruction',reached)
    control=dict(schema='lorrax.isdf_augmentation.v1',carrier='normalized_rkb',
        frozen_core_policy='reconstruct_valence_only',
        species={'47':{'source_upf':'absent.upf','reconstruction':'absent.npz'}},
        radial={},angular={},cache={'species_files':{'47':'absent.npz'}},runtime={},
        overlap={'mode':'full_wfn_lowdin','bands':1},served_moments={
            'species_files':{'47':'absent.npz'},'species_sha256':{'47':'0'*64},
            'raw_parent_file':'absent.npz','raw_parent_sha256':'0'*64})
    (tmp_path/'manifest.json').write_text(json.dumps(control))
    with pytest.raises(AtomicReadReached):
        read_augmentation_manifest(tmp_path)


def test_raw_parent_preparation_authenticates_species_with_strict_fitting_default(tmp_path,monkeypatch):
    from gw.isdf_augmentation import read_augmentation_manifest
    from psp import atomic_reconstruction
    class AtomicReadReached(Exception):
        pass
    def reached(*args,**kwargs):
        raise AtomicReadReached
    monkeypatch.setattr(atomic_reconstruction,'load_atomic_reconstruction',reached)
    control=dict(schema='lorrax.isdf_augmentation.v1',carrier='normalized_rkb',
        frozen_core_policy='reconstruct_valence_only',
        species={'47':{'source_upf':'absent.upf','reconstruction':'absent.npz'}},
        radial={},angular={},cache={'species_files':{'47':'absent.npz'}},runtime={},
        overlap={'mode':'full_wfn_lowdin','bands':1},served_moments={
            'species_files':{'47':'absent.npz'},'species_sha256':{'47':'0'*64}})
    (tmp_path/'manifest.json').write_text(json.dumps(control))
    with pytest.raises(ValueError,match='exact species'):
        read_augmentation_manifest(tmp_path)
    with pytest.raises(AtomicReadReached):
        read_augmentation_manifest(tmp_path,load_raw_parent=False)
    control['served_moments']['raw_parent_file']='absent.npz'
    (tmp_path/'manifest.json').write_text(json.dumps(control))
    with pytest.raises(ValueError,match='exact species'):
        read_augmentation_manifest(tmp_path,load_raw_parent=False)


def test_indexed_workspace_keeps_full_native_outputs_in_its_bound():
    from gw.isdf_augmentation import _local_rhs_workspace_bytes
    full = _local_rhs_workspace_bytes(16,216,2304,17280,16,retain_smooth=True)
    indexed = _local_rhs_workspace_bytes(
        16,216,2304,17280,16,retain_smooth=True,nq_accumulator=28)
    scalar = full['full_q_scalar']
    assert indexed['full_q_scalar_panels'] == 2
    assert indexed['accumulator_scalar_panels'] == 6
    assert indexed['accumulator_q_rows'] == 28
    assert indexed['endpoint_and_quarter_outputs'] == 2*scalar+6*scalar*28/216
    assert indexed['parent_projectors_and_layout'] == full['parent_projectors_and_layout']
    assert indexed['total'] < 34e9 < full['total']
    for rows in (0,217):
        with pytest.raises(ValueError,match='full-q subset'):
            _local_rhs_workspace_bytes(16,216,2304,17280,16,nq_accumulator=rows)
