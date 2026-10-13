"""Paired direct-field transport shares the charge writer and band owner."""
import copy
import hashlib
import json

import numpy as np
import pytest


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _record():
    source = dict(schema='physical_source_fixture', common_frame='a'*64,
                  occupations=[1., 1., 0.])
    facts = dict(source_identity=_identity(source), band_range=[2, 5],
        parent_full_rows=[0, 3], parent_k_frac=[[0., 0., 0.], [0., 0., .5]],
        units='Ry', k_domain='file_wedge', trs_rule='conj')
    charge = dict(facts, schema='lorrax.resident_charge_hartree_operator.v1',
                  kernel='charge_fixture')
    transverse = dict(facts, schema='lorrax.resident_current_hartree_operator.v1',
        current_operator_generation='b'*64, current_basis='Cartesian',
        variational_convention='hermitian_density_current_energy_derivative_v1')
    binding = dict(schema='lorrax.resident_paired_direct_operator.v1',
        source_identity=facts['source_identity'], components={name:dict(
            operator_identity=_identity(value), operator_binding=value)
            for name, value in (('charge', charge), ('transverse', transverse))})
    return dict(facts, schema='lorrax.resident_direct_hartree.v2',
        source_binding=source, operator_binding=binding,
        operator_identity=_identity(binding))


def _literal():
    raw = np.arange(18).reshape(2, 3, 3).astype(complex)
    charge = raw + raw.conj().swapaxes(1, 2)
    transverse = (2j*raw - 2j*raw.swapaxes(1, 2)) * .013
    return np.stack((charge, transverse), axis=-1)


def exercise_paired_hartree_io(mesh, path):
    """Actual all-rank writer/read, with distinct fields and padded band ghosts."""
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import barrier, device_put_process_local, gather_to_host
    from file_io.tagged_arrays import write_restart_state_to_h5, _hartree_band_axis
    from file_io.restart_bundle import read_resident_hartree_from_h5

    axis = _hartree_band_axis(3, mesh)
    carrier = np.full((2, axis.carrier, axis.carrier, 2), np.nan+1j*np.nan)
    carrier[:, :3, :3] = _literal()
    value = device_put_process_local(carrier,
        NamedSharding(mesh, P(None, 'x', 'y', None)))
    write_restart_state_to_h5(str(path), n_rmu_logical=2, mesh=mesh,
        hartree_parent_kij_ry=value, hartree_provenance=_record())
    actual = read_resident_hartree_from_h5(path, mesh, required=True)
    assert set(actual) == {'charge_parent_kij_ry', 'transverse_parent_kij_ry', 'provenance'}
    expected = np.zeros_like(carrier)
    expected[:, :3, :3] = _literal()
    for i, name in enumerate(('charge', 'transverse')):
        field = actual[name+'_parent_kij_ry']
        assert field.sharding.is_equivalent_to(NamedSharding(mesh, P(None, 'x', 'y')), 3)
        np.testing.assert_array_equal(np.asarray(gather_to_host(field)), expected[..., i])
    # This record only transports fixture facts, not a production source.
    from gw.augmentation_hartree import bind_resident_hartree
    try:
        bind_resident_hartree(actual, wfn=None, sym=None, plan=None,
            artifact=None, wfn_fingerprint_binding=None, band_range=(2, 5))
    except ValueError:
        pass
    else:
        raise AssertionError('Legacy source gate admitted unpaired fixture fields')
    barrier('paired_hartree_io_complete')
    return dict(status='PASS_PAIRED_DIRECT_FIELD_IO_ONLY',
        fields=['charge', 'transverse'], physical_shape=[2, 3, 3, 2],
        packed_bands=axis.carrier, poison_padding_inert=True,
        legacy_physical_source_gate_preserved=True,
        scope='IO, distinct fields and band padding only; no material source/operator/Hartree/GW admission')


def test_paired_direct_field_roundtrip(tmp_path):
    import jax
    from jax.sharding import Mesh
    mesh = Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1, 1), ('x', 'y'))
    exercise_paired_hartree_io(mesh, tmp_path/'paired.h5')


@pytest.mark.parametrize('defect', ['source', 'domain', 'operator', 'derivative', 'names'])
def test_paired_component_metadata_refuses_even_with_rehashed_outer_record(defect):
    from file_io.tagged_arrays import normalize_resident_hartree_provenance
    record = copy.deepcopy(_record())
    binding = record['operator_binding']
    member = binding['components']['transverse']
    if defect == 'source':
        member['operator_binding']['source_identity'] = 'c'*64
    elif defect == 'domain':
        member['operator_binding']['band_range'] = [0, 3]
    elif defect == 'operator':
        member['operator_identity'] = 'c'*64
    elif defect == 'derivative':
        member['operator_binding']['variational_convention'] = 'unprojected_complex_field'
    else:
        binding['components']['current'] = binding['components'].pop('transverse')
    if defect != 'operator':
        member['operator_identity'] = _identity(member['operator_binding'])
    record['operator_identity'] = _identity(binding)
    with pytest.raises(ValueError, match='paired'):
        normalize_resident_hartree_provenance(record)


def test_changed_transverse_payload_refuses_shared_checksum(tmp_path):
    import h5py
    import jax
    from jax.sharding import Mesh
    from file_io.tagged_arrays import HARTREE_PARENT_DATASET
    from file_io.restart_bundle import read_resident_hartree_metadata
    mesh = Mesh(np.asarray(jax.devices('cpu')[:1]).reshape(1, 1), ('x', 'y'))
    path = tmp_path/'paired.h5'
    exercise_paired_hartree_io(mesh, path)
    with h5py.File(path, 'a') as f:
        f[HARTREE_PARENT_DATASET][0, 0, 0, 1] += 1
    with pytest.raises(ValueError, match='checksum'):
        read_resident_hartree_metadata(path)
