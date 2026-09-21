"""Collective retained-star direct-field write, plus receipt refusal twin."""
import sys
from pathlib import Path

from runtime import initialize_communicator_stack, finalize_process

runtime = initialize_communicator_stack(platform="gpu")
import h5py
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import device_put_process_local, rank0_transaction
from file_io.sigma_output import (
    DIRECT_FIELD_SUM_FULL_BZ_ATTR, DIRECT_FIELD_SUM_STAR_WEDGE_ATTR,
    SIGMA_DIRECT_COMPONENT_DATASETS, _validate_raw_direct_component_contract,
    write_sigma_omega_h5,
)


def main():
    path = Path(sys.argv[1])
    face = NamedSharding(runtime.mesh, P(None, "x", "y"))
    cube = NamedSharding(runtime.mesh, P(None, None, "x", "y"))
    scalar = device_put_process_local(np.ones((2, 4, 4), complex), face)
    transverse = 0.125j * scalar
    correlation = device_put_process_local(np.zeros((3, 2, 4, 4), complex), cube)
    write_sigma_omega_h5(str(path), np.array([-1., 0., 1.]), None,
        sigma_c_kij_ev=correlation, sigma_sx_kij_ev=scalar,
        hartree_kij_ev=scalar + transverse, hartree_scalar_kij_ev=scalar,
        hartree_transverse_kij_ev=transverse, mesh=runtime.mesh,
        star=(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), 2),
        star_already_selected=True)

    def check():
        with h5py.File(path, "r+") as h5:
            assert _validate_raw_direct_component_contract(h5, required=True)
            for name in ("hartree_kij_ev",) + SIGMA_DIRECT_COMPONENT_DATASETS:
                assert not h5[name].attrs[DIRECT_FIELD_SUM_FULL_BZ_ATTR]
                assert h5[name].attrs[DIRECT_FIELD_SUM_STAR_WEDGE_ATTR]
            np.testing.assert_array_equal(h5['hartree_kij_ev'][:],
                h5['hartree_scalar_kij_ev'][:] + h5['hartree_transverse_kij_ev'][:])
            attrs = h5['hartree_kij_ev'].attrs
            del attrs[DIRECT_FIELD_SUM_STAR_WEDGE_ATTR]
            try:
                _validate_raw_direct_component_contract(h5, required=True)
            except ValueError:
                pass
            else:
                raise AssertionError('Missing input-scope authentication accepted')
            attrs[DIRECT_FIELD_SUM_STAR_WEDGE_ATTR] = True
        print('PASS retained-star direct-field writer and missing-authentication refusal', flush=True)
    rank0_transaction(str(path), stage='retained_direct_receipt', write=check)


status = 1
try:
    main()
    status = 0
except BaseException:
    import traceback
    traceback.print_exc()
finalize_process(status)
