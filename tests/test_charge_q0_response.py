"""Physical potential/basis provenance guards for the optional charge response."""
from types import SimpleNamespace

import numpy as np
import pytest


def response_inputs():
    import jax
    from jax.sharding import Mesh
    mesh = Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1, 1), ('x', 'y'))
    axis = SimpleNamespace(packed_to_canonical=np.asarray([2, 0, 1, -1]),
                           active_mask=np.asarray([True, True, True, False]))
    return dict(mesh=mesh, meta=SimpleNamespace(fft_grid=(4, 3, 4)),
                q_full_indices=np.asarray([0, 1]),
                q_frac=np.asarray([[0., 0., 0.], [.5, 0., 0.]]),
                typed_plan=SimpleNamespace(layout=SimpleNamespace(axis=axis),coordinate_kind='fractional'),
                centroid_indices=np.asarray([[0.,0.,0.],[.1,.2,.3],[.4,.5,.6]]))


def test_physical_response_records_canonical_placement_without_changing_the_potential():
    import jax
    from gw.isdf_fitting import _prepare_charge_q0_response
    value = np.linspace(-2., 3., 48).reshape(4, 3, 4)
    placed, metadata = _prepare_charge_q0_response(
        value, dict(source_identity='physical_occupation_trace',
                    potential_identity='fullFFT_G0_zero'), **response_inputs())
    np.testing.assert_array_equal(np.asarray(jax.device_get(placed))[0], value)
    assert metadata['q0_full_index'] == 0 and metadata['q0_slot'] == 0
    assert metadata['source_count'] == 1 and metadata['fft_points'] == 48
    assert metadata['raw_rhs_order'] == 'packed'
    assert metadata['complete'] is False
    assert metadata['units'] == 'physical_Ry_potential_times_grid_density_sum'
    assert len(metadata['packed_to_canonical_sha256']) == 64


@pytest.mark.parametrize('change', ('complex', 'nonfinite', 'shape', 'no_source',
                                   'metadata_nonfinite', 'override', 'no_gamma', 'two_gamma'))
def test_q0_response_refuses_unbound_or_nonphysical_potential(change):
    from gw.isdf_fitting import _prepare_charge_q0_response
    value = np.ones((4, 3, 4), np.float64)
    metadata = dict(source_identity='physical_occupation_trace', potential_identity='fullFFT_G0_zero')
    inputs = response_inputs()
    if change == 'complex': value = value.astype(complex)
    elif change == 'nonfinite': value[0, 0, 0] = np.nan
    elif change == 'shape': value = value[:, :2]
    elif change == 'no_source': metadata.pop('source_identity')
    elif change == 'metadata_nonfinite': metadata['source_charge'] = np.inf
    elif change == 'override': metadata['q0_slot'] = 1
    elif change == 'no_gamma': inputs['q_full_indices'] = np.asarray([1, 2])
    else:
        inputs['q_full_indices'] = np.asarray([0, 0])
        inputs['q_frac'] = np.zeros((2, 3))
    with pytest.raises(ValueError, match='q0'):
        _prepare_charge_q0_response(value, metadata, **inputs)


def test_response_finalize_refuses_partial_or_current_fit():
    from isdf.zeta_mubatch import finalize_charge_q0_response
    for vertex, complete in ((0, False), (1, True)):
        zeta = SimpleNamespace(q0_response_rhs=np.ones((1, 2, 1), complex),
            q0_response_metadata=dict(schema='lorrax.charge_q0_response.v1', complete=complete),
            fit_vertex_mu_L=vertex)
        with pytest.raises(ValueError, match='complete charge fit'):
            finalize_charge_q0_response(zeta)
