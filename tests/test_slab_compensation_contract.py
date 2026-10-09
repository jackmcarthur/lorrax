"""Kernel and support refusals for the actual periodic-cache metadata owner.

These tests certify metadata acceptance, not Fourier accuracy or slab GW.
The independent compact-polynomial spectral oracle supplies the latter
completion comparison on a compute node.
"""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from isdf.coulomb_fourier_cache import (
    PERIODIC_SCHEMA, PERIODIC_SLAB_SCHEMA, _periodic_geometry_binding,
    _require_periodic_compensation_metadata, periodic_compensation_contract,
    periodic_compensation_geometry,
)


def geometry():
    return dict(reciprocal_rows_bohr_inverse=np.diag(2*np.pi/np.array([8., 8., 20.])),
        cell_volume_bohr3=1280., atom_centres_bohr=[[0., 0., 0.]],
        operator_q_fractional=[[0., 0., 0.], [1/3, 0., 0.]],
        support_radius_bohr=1.5)


def metadata(geo):
    bound, schema, model = periodic_compensation_contract(geo)
    preparation = dict(receipt_sha256='1'*64, payload_sha256='2'*64,
        producer_sources_sha256={'independent_preparation':'3'*64},
        cutoffs=[12., 16.], refinement_max=[[1e-8], [1e-8]])
    return dict(schema=schema, model=copy.deepcopy(model), geometry=bound,
        lm=[[0, 0]], logical_shape=[2, 1, 1],
        row_order='atom_major_canonical_complex_lm', preparation=preparation,
        source_binding=dict(physical_model=copy.deepcopy(model),
            producer_sources_sha256=preparation['producer_sources_sha256']))


def require(record, geo):
    return _require_periodic_compensation_metadata(record, geometry=geo,
        lm=[[0, 0]], stored_shape=(2, 1, 1), stored_dtype=np.complex128)


def test_bulk_v1_metadata_is_unchanged():
    geo = geometry()
    bound = periodic_compensation_geometry(geo)
    assert set(bound) == set(geo)
    record = metadata(bound)
    assert record['schema'] == PERIODIC_SCHEMA
    assert record['model'] == dict(compensation_power=6,
        coulomb='bare_periodic_8pi_over_Omega_K2_Ry', gamma_zero='excluded',
        phase='exp_minus_i_K_dot_center',
        units='physical_Ry_unit_harmonic_multipoles')
    require(record, geo)


def test_slab_metadata_binds_public_orientation_height_and_zero_mode():
    geo = periodic_compensation_geometry(geometry(), sys_dim=2)
    assert geo['kernel'] == dict(sys_dim=2, owner='vcoul.Slab2D',
        normal_cartesian=[0., 0., 1.], truncation_half_height_bohr=10.,
        gamma_zero='excluded')
    record = metadata(geo)
    assert record['schema'] == PERIODIC_SLAB_SCHEMA
    assert record['model']['neutral_mean'] == 'none_compact_support_layer'
    require(record, geo)


def test_legacy_bulk_cache_cannot_answer_slab_request():
    bulk = metadata(geometry())
    slab = periodic_compensation_geometry(geometry(), sys_dim=2)
    with pytest.raises(ValueError, match='model, geometry or axis'):
        require(bulk, slab)
    with pytest.raises(ValueError, match='model, geometry or axis'):
        require(metadata(slab), geometry())


@pytest.mark.parametrize('field,value', [
    ('owner', 'vcoul.Bulk3D'), ('gamma_zero', 'retained'),
    ('normal_cartesian', [0., 0., -1.]), ('truncation_half_height_bohr', 9.),
])
def test_changed_slab_kernel_binding_refuses(field, value):
    geo = periodic_compensation_geometry(geometry(), sys_dim=2)
    geo['kernel'][field] = value
    with pytest.raises(ValueError, match='kernel identity'):
        _periodic_geometry_binding(geo)


def test_bulk_mean_model_cannot_answer_slab_request():
    geo = periodic_compensation_geometry(geometry(), sys_dim=2)
    record = metadata(geo)
    record['model']['neutral_mean'] = 'bulk_both_adjoints'
    with pytest.raises(ValueError, match='model, geometry or axis'):
        require(record, geo)


def test_cutoff_cannot_intersect_compact_layer_pairs():
    geo = geometry()
    geo['atom_centres_bohr'] = [[0., 0., -4.], [2., 0., 4.]]
    with pytest.raises(ValueError, match='GATE slab_compact_support'):
        periodic_compensation_geometry(geo, sys_dim=2)


def test_boundary_wrapped_layer_has_one_unwrapped_support_proof():
    geo = geometry()
    geo['atom_centres_bohr'] = [[0., 0., -9.5], [3., 0., 9.5]]
    bound = periodic_compensation_geometry(geo, sys_dim=2)
    assert bound['atom_centres_bohr'] == geo['atom_centres_bohr']
    assert _periodic_geometry_binding(bound) == bound


@pytest.mark.parametrize('reverse,tilt', [(True, False), (False, True)])
def test_public_slab_orientation_refusal_is_retained(reverse, tilt):
    geo = geometry()
    if reverse:
        geo['reciprocal_rows_bohr_inverse'][2, 2] *= -1
    if tilt:
        geo['reciprocal_rows_bohr_inverse'][0, 2] = .01
    geo['cell_volume_bohr3'] = (2*np.pi)**3/abs(np.linalg.det(
        geo['reciprocal_rows_bohr_inverse']))
    with pytest.raises(ValueError, match='Slab2D requires'):
        periodic_compensation_geometry(geo, sys_dim=2)


def test_out_of_plane_q_refuses():
    geo = geometry()
    geo['operator_q_fractional'][1][2] = 1/3
    with pytest.raises(ValueError, match='in-plane operator q'):
        periodic_compensation_geometry(geo, sys_dim=2)


def test_legacy_local_metric_refuses_before_any_rhs_or_v_access():
    from gw.isdf_augmentation import attach_local_augmentation

    with pytest.raises(ValueError, match='GATE slab_atomic_metric'):
        attach_local_augmentation(object(), {'body_metric':'legacy'},
                                  body_contract={'sys_dim':2})


def test_slab_hartree_source_refuses_before_any_field_or_fit_access():
    from gw.isdf_augmentation import prepare_augmentation

    with pytest.raises(ValueError, match='GATE slab_augmented_hartree_source_unavailable'):
        prepare_augmentation(wfn=None, sym=None, cfg=None, mesh_xy=None,
            meta=SimpleNamespace(nspinor=4, sys_dim=2),
            plan=SimpleNamespace(nspinor=4), centroid_indices=None,
            parent_psi=SimpleNamespace(psi_G=object()), parent_faces=None,
            band_range_left=(0, 1), band_range_right=(0, 1), artifact={},
            hartree_source_request={})
