"""P4 CT writer gate: both ordered endpoints persist one exact pole census."""

import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from runtime import initialize_communicator_stack,finalize_process
    initialize_communicator_stack()
    import jax
    import numpy as np
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import resolve_mesh,device_put_process_local,gather_to_host
    from gw.shared_pole_local import batch_to_face,face_rows,canonical_factors
    from gw.shared_pole_sectors import _host_sector_census
    from file_io import shared_pole_store as store
    from file_io.slab_io import SlabIO
    from gw.shared_pole_recipe import CapacityLedger
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from test_shared_pole_store import _fixture

    mesh=resolve_mesh()
    assert mesh.size==4 and int(mesh.shape['y'])==2
    meta,tables,recipe,identity=_fixture(mesh)
    meta.nspinor=4
    meta.shared_pole_capacity=CapacityLedger(meta,mesh_xy=mesh)
    meta.shared_pole_capacity.live_stages=()
    source=np.ones((4,8,7),np.complex128)
    batched=jax.make_array_from_callback(source.shape,NamedSharding(mesh,P(('x','y'))),
                                          lambda index:source[index])
    face=face_rows(mesh,(0,1,2),width=8)(batch_to_face(mesh)(batched))
    charge=canonical_factors(mesh,(0,1,2))(face)
    current=jax.jit(lambda a:jax.numpy.repeat(2*a,3,axis=2),
                    out_shardings=NamedSharding(mesh,P(None,'x',None,'y')))(charge)
    poles=np.pad(np.tile(np.arange(1,8,dtype=np.float64),(3,1)),
                 ((0,0),(0,1)),constant_values=1.0)
    drift=poles.copy()
    drift[1,4]=np.nextafter(drift[1,4],np.inf)
    assert poles[1,4]!=drift[1,4]
    mask=np.tile(np.arange(8)<7,(3,1))
    repl=NamedSharding(mesh,P())
    charge_census=_host_sector_census(
        device_put_process_local(poles,repl),device_put_process_local(mask,repl),mesh,3)
    current_census=_host_sector_census(
        device_put_process_local(drift,repl),device_put_process_local(mask,repl),mesh,3,
        common=charge_census)
    assert current_census is charge_census
    assert charge_census[3]==8
    np.testing.assert_array_equal(charge_census[0],poles)
    args.output.mkdir(parents=True,exist_ok=True)
    for name,factor,census in (('CT_C',charge,charge_census),
                               ('CT_T',current,current_census)):
        host_poles,_,counts,width=census
        header=store.write_shared_pole_model(
            args.output/(name+'.h5'),factor,
            device_put_process_local(host_poles[:,:width],repl),counts,
            q_span=(0,3),meta=meta,tables=tables,recipe=recipe,
            receipts={'identity':identity,'scope':'CT-exact-host-census'},
            ordered=True,basis=meta.mu_basis,sector=name)
        assert header['finalized'] and header['Kmax']==7
    stored=[]
    for name in ('CT_C','CT_T'):
        path=args.output/(name+'.h5')
        header=store.validate_shared_pole_model(
            path,expected_identity=identity,mesh_xy=mesh,
            capacity=meta.shared_pole_capacity)
        with SlabIO(path,mode='r',mesh=mesh) as io:
            value,counts=store.read_shared_pole_census(
                io,header=header,capacity=meta.shared_pole_capacity)
        np.testing.assert_array_equal(np.asarray(gather_to_host(value)),poles[:,:7])
        np.testing.assert_array_equal(np.asarray(gather_to_host(counts)),np.full(3,7))
        stored.append(np.asarray(gather_to_host(value)))
    assert stored[0].tobytes()==stored[1].tobytes()
    if jax.process_index()==0:
        import h5py
        with h5py.File(args.output/'CT_C.h5','r') as c, h5py.File(args.output/'CT_T.h5','r') as t:
            assert c['poles2_ry2'].shape==t['poles2_ry2'].shape==(3,7)
            assert c['poles2_ry2'][...].tobytes()==t['poles2_ry2'][...].tobytes()
            assert c['factor'].shape==(3,7,1,7)
            assert t['factor'].shape==(3,7,3,7)
        print(json.dumps(dict(status='PASS',gate='CT host census exact finalized bytes',
                              Kmax=7,carrier=8,job=os.environ.get('SLURM_JOB_ID'),
                              step=os.environ.get('SLURM_STEP_ID'))),flush=True)
    finalize_process()


if __name__=='__main__':
    main()
