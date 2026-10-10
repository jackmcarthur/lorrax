"""Source/receiving policy joins for the new direct-kernel slab Hartree.

These metadata tests do not replace the independent numerical Hartree
oracle or admit a public slab GW run.
"""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from isdf.atomic_hartree import charge_hartree_operator_contract
from gw.isdf_augmentation import charge_hartree_response_metadata
from gw.augmentation_hartree import _operator_binding
from gw.augmentation_hartree_receiving import _make_point_contraction


def wfn():
    return SimpleNamespace(bvec=np.diag(2*np.pi/np.array([8., 8., 20.])),
        blat=1., cell_volume=1280., fft_grid=(4, 4, 4), alat=1.,
        avec=np.diag([8., 8., 20.]), atom_crys=np.zeros((1, 3)),
        atom_types=np.array([1]), kvecs=lambda **_: np.zeros((1, 3)))


def bound_source(sys_dim):
    contract = charge_hartree_operator_contract(wfn(), sys_dim=sys_dim)
    binding = dict(fft_grid=[4, 4, 4])
    state = dict(radius=np.array([.1, .2]), weights_dr=np.array([.05, .1]),
        lm=np.array([[0, 0]]), centers_cart=np.zeros((1, 3)),
        fft_points=64, cell_volume=1280., support_radius=1.5)
    if sys_dim == 2:
        binding.update(sys_dim=2, hartree_kernel=contract['kernel'])
        state.update(sys_dim=2, hartree_operator_contract=contract)
    source_identity = hashlib.sha256(json.dumps(binding, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    state['hartree_source'] = dict(exact_monopole=np.zeros((1, 1)),
        source_binding=binding, source_identity=source_identity)
    functional = dict(contract, sys_dim=sys_dim,
        local_feature_order=('delta', 'PS', 'exact_Y00'),
        smooth_potential=np.zeros((1, 4, 4, 4)), local_response=np.zeros((1, 5)))
    return state, functional


def test_bulk_operator_contract_and_identity_fields_remain_unchanged():
    expected = dict(operator='ordinary_3D_periodic_full_FFT_G0_zero',
                    neutral_mean_policy='subtract_free_space_neutral_cell_mean')
    assert charge_hartree_operator_contract(None) == expected
    state, functional = bound_source(3)
    record = charge_hartree_response_metadata(state, functional)
    assert 'kernel' not in record['potential_binding']
    assert all(record['potential_binding'][key] == value for key, value in expected.items())


def test_slab_response_identity_binds_actual_kernel_once():
    state, functional = bound_source(2)
    record = charge_hartree_response_metadata(state, functional)
    assert record['potential_binding']['operator'] == 'ordinary_2D_truncated_full_FFT_G0_zero'
    assert record['potential_binding']['neutral_mean_policy'] == 'none_direct_smooth_neutral_fft'
    assert record['potential_binding']['kernel'] == functional['kernel']
    assert record['source_binding']['hartree_kernel'] == functional['kernel']


@pytest.mark.parametrize('changed', ['functional_kernel', 'source_kernel', 'dimension'])
def test_slab_source_response_kernel_mismatch_refuses(changed):
    state, functional = bound_source(2)
    if changed == 'functional_kernel':
        functional = copy.deepcopy(functional)
        functional['kernel']['truncation_half_height_bohr'] = 9.
    elif changed == 'source_kernel':
        state['hartree_source']['source_binding']['hartree_kernel'] = {'owner': 'wrong'}
        binding = state['hartree_source']['source_binding']
        state['hartree_source']['source_identity'] = hashlib.sha256(json.dumps(binding,
            sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    else:
        functional['sys_dim'] = 3
    with pytest.raises(ValueError, match='Hartree response'):
        charge_hartree_response_metadata(state, functional)


def test_resident_operator_binding_distinguishes_bulk_and_slab():
    artifact = dict(identity='test physical atom',
        radial=dict(radius=[.1, .2, .3, .4], weights_dr=[.05, .1, .1, .05], support_radius=1.5),
        angular=dict(lebedev_order=3, lmax=0, orthogonality_tolerance=1e-10))
    sym = SimpleNamespace(R_cart=np.eye(3)[None], kirr_fullids=np.array([0]),
                          parent_k_domain='file')
    kwargs = dict(wfn=wfn(), sym=sym, artifact=artifact,
                  source_identity='1'*64, band_range=(0, 2))
    bulk = _operator_binding(**kwargs)
    slab = _operator_binding(**kwargs, sys_dim=2)
    assert 'kernel' not in bulk
    assert slab['kernel'] == charge_hartree_operator_contract(wfn(), sys_dim=2)['kernel']
    assert bulk['band_range'] == slab['band_range'] == [0, 2]
    assert bulk['parent_full_rows'] == slab['parent_full_rows'] == [0]


@pytest.mark.parametrize('contract_mode', ['omitted', 'empty', 'wrong_kernel'])
def test_receiving_slab_policy_cannot_be_unbound_or_bypassed(contract_mode):
    _, functional = bound_source(2)
    functional.update(local_geometry=(1, 1, 2), local_to_grid=.05,
        receiving_component_order=('compensation_body', 'difference', 'PS_delta',
                                   'delta_PS', 'enriched', 'periodic_mean'))
    kwargs = {}
    if contract_mode == 'empty':
        kwargs['operator_contract'] = {}
    elif contract_mode == 'wrong_kernel':
        contract = copy.deepcopy(charge_hartree_operator_contract(wfn(), sys_dim=2))
        contract['kernel']['truncation_half_height_bohr'] = 9.
        kwargs['operator_contract'] = contract
    with pytest.raises(ValueError, match='Receiving adjoints'):
        _make_point_contraction(SimpleNamespace(shape={'x': 1, 'y': 1}),
            functional=functional, radius=np.array([.1, .2]),
            directions=np.array([[0., 0., 1.]]), angular_weights=np.array([4*np.pi]),
            lm=np.array([[0, 0]]), Y=np.array([[1/np.sqrt(4*np.pi)]]),
            band_tile=1, **kwargs)


@pytest.mark.parametrize('stale', ['delta_PS', 'bulk_mean'])
def test_receiving_slab_refuses_stale_bulk_adjoints_before_device_placement(stale):
    _, functional = bound_source(2)
    delta = np.zeros((6, 1, 1, 1, 2), complex)
    ps = np.zeros_like(delta)
    m0 = np.zeros((6, 1, 1), complex)
    if stale == 'delta_PS':
        ps[3, 0, 0, 0, 0] = .01
    else:
        m0[5, 0, 0] = .01
    functional.update(local_geometry=(1, 1, 2), local_to_grid=.05,
        receiving_component_order=('compensation_body', 'difference', 'PS_delta',
                                   'delta_PS', 'enriched', 'periodic_mean'),
        receiving_components=dict(delta=delta, PS=ps, exact_Y00=m0))
    with pytest.raises(ValueError, match='Receiving slab adjoints'):
        _make_point_contraction(SimpleNamespace(shape={'x': 1, 'y': 1}),
            functional=functional, radius=np.array([.1, .2]),
            directions=np.array([[0., 0., 1.]]), angular_weights=np.array([4*np.pi]),
            lm=np.array([[0, 0]]), Y=np.array([[1/np.sqrt(4*np.pi)]]),
            band_tile=1, operator_contract=charge_hartree_operator_contract(wfn(), sys_dim=2))
