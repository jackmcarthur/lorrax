"""Paired current identity must join the independently bound charge source."""
import pytest

from gw.augmentation_hartree import _current_operator_binding, _identity


def _records():
    source = dict(augmentation_identity='a'*64, physical_bands=200,
        source_frame_policy='paired fixture', physical_frame_shape=[3,200,200],
        physical_frame_band_domain=[0,200], physical_frame_convention='same factor',
        full150_frame_sha256='b'*64, fft_grid=[80,80,250], cell_volume=6331.,
        wfn_fingerprint_scheme='fixture', wfn_fingerprint='c'*64,
        wfn_sha256='d'*64, sys_dim=2)
    facts = dict(source_binding=source, source_identity=_identity(source),
        band_range=[0,192], parent_full_rows=[0,1,5],
        parent_k_frac=[[0.,0.,0.],[0.,1/3,0.],[1/3,0.,0.]],
        units='Ry', k_domain='file_wedge', trs_rule='conj')
    endpoint = {key:value for key,value in source.items() if key != 'sys_dim'}
    endpoint.update(FILE_parent_rows=facts['parent_full_rows'],
        FILE_parent_k_fractional=facts['parent_k_frac'])
    contract = dict(schema='lorrax.current.physical_Slab_mixed_operator.v1',
        operator='mixed_reciprocal_point_Slab_plus_bulk_local_infinite_remainder',
        endpoint_source_binding=endpoint, current_geometry_binding=dict(
            logical_mu=4262, centroid_hash='e'*64, q_full_indices=[0,1,5],
            q_fractional=facts['parent_k_frac'], kgrid=[3,3,1]))
    current = dict(operator_contract=contract, operator_identity=_identity(contract),
        generation_identity='f'*64)
    return facts, current


def test_public_receiving_192_retains_physical_source_frame_200():
    charge, current = _records()
    result = _current_operator_binding(charge, current)
    assert result['band_range'] == [0,192]
    assert result['physical_current_operator_contract']['endpoint_source_binding']['physical_bands'] == 200
    assert result['variational_convention'] == 'hermitian_density_current_energy_derivative_v1'


@pytest.mark.parametrize('key', [
    'augmentation_identity','physical_bands','source_frame_policy',
    'physical_frame_shape','physical_frame_band_domain','physical_frame_convention',
    'full150_frame_sha256','fft_grid','cell_volume','wfn_fingerprint_scheme',
    'wfn_fingerprint','wfn_sha256','FILE_parent_rows','FILE_parent_k_fractional'])
def test_changed_current_endpoint_refuses_even_after_rehash(key):
    charge, current = _records()
    endpoint = current['operator_contract']['endpoint_source_binding']
    endpoint[key] = 'changed'
    current['operator_identity'] = _identity(current['operator_contract'])
    with pytest.raises(ValueError, match='resident_current_source'):
        _current_operator_binding(charge, current)


def test_opaque_current_descriptor_refuses_even_after_rehash():
    charge, current = _records()
    current['operator_contract']['schema'] = 'opaque_fixture'
    current['operator_identity'] = _identity(current['operator_contract'])
    with pytest.raises(ValueError, match='physical mixed Slab current owner'):
        _current_operator_binding(charge, current)


def test_missing_current_chart_refuses_even_after_rehash():
    charge, current = _records()
    current['operator_contract'].pop('current_geometry_binding')
    current['operator_identity'] = _identity(current['operator_contract'])
    with pytest.raises(ValueError, match='centroid/q geometry'):
        _current_operator_binding(charge, current)
