"""Stage placement and selected-q integration versus the proved full-q owner."""
from pathlib import Path
import json
import os


def check_served_monopole_stage(runtime):
    from types import SimpleNamespace
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.staged_reshard import face_to_batch_reshard
    from gw.centroid_k_unfold import build_centroid_k_unfold_plan
    from gw.isdf_augmentation import _served_monopole_rhs
    from isdf.atomic_moments import auxiliary_charge_geometry, evaluate_auxiliary_charge, make_auxiliary_monopole_compressor
    from isdf.local_rhs import local_density_rhs
    from symmetry_maps import q_negation_index
    from tests.test_atomic_moments import _synthetic_cache

    mesh = runtime.mesh
    assert mesh.size == 4
    cache = _synthetic_cache()
    aux = auxiliary_charge_geometry(cache)
    rng = np.random.default_rng(610221)
    nb,kgrid,grid = 8,(3,1,1),(8,8,8)
    centers,lattice = np.array([[0.,0.,0.],[.5,0.,0.]]),8*np.eye(3)
    points = (centers[:,None]+aux['relative_points'][None] @ np.linalg.inv(lattice)).reshape(-1,3)
    mu = np.asarray([[0,0,0],[1,0,0],[7,0,0],[2,0,0],[0,1,0],[0,7,0],[0,0,1],[0,0,7]])
    wl,wr = np.array([1,1,1,0,0,0,0,0.]),np.array([1,1,1,1,1,0,0,0.])
    qneg,qids = q_negation_index(kgrid),np.array([1,0],np.int32)
    put = lambda a,s: device_put_process_local(np.asarray(a),NamedSharding(mesh,s))
    face = lambda a: (put(a,P(None,'x',None,'y')),put(a.transpose(0,2,3,1),P(None,None,'x','y')))
    random = lambda shape: (rng.normal(size=shape)+1j*rng.normal(size=shape))/11
    receipts=[]
    for kind in ('identity','time_reversal'):
        npar=3 if kind=='identity' else 2
        irr=np.arange(3) if npar==3 else np.array([0,1,1])
        actions=np.array([np.eye(4),np.kron(np.eye(2),np.array([[0,1],[-1,0]]))],complex)
        sym=SimpleNamespace(sym_matrices=np.eye(3,dtype=int)[None],translations=np.zeros((1,3)),
            irr_idx_k=irr,sym_idx_k=np.zeros(3,int) if npar==3 else np.array([0,0,1]),
            kirr_fullids=np.arange(npar),spinor_action=lambda rows,nspinor: actions[rows])
        kp=np.zeros((npar,3));kp[:,0]=np.arange(npar)/3
        left=build_centroid_k_unfold_plan(sym,mu,grid,mesh,nspinor=4,parent_k_frac=kp)
        right=build_centroid_k_unfold_plan(sym,points,grid,mesh,nspinor=4,parent_k_frac=kp,coordinate_kind='fractional')
        ni=len(cache['labels'])
        C,D=[random((2,npar,nb,ni)) for _ in range(2)]
        C[:,:,-1]=1e5*(1+1j);D[:,:,-1]=1e5*(1-1j)
        L=random((npar,nb,4,len(mu)))
        L[:,-1]=1e5
        L=left.layout.axis.pack_host(L,axis=3,fill_value=0.)
        scale=.73
        plus,minus=zip(*(evaluate_auxiliary_charge(c,d,aux,grid_sample_scale=scale) for c,d in zip(C,D)))
        plus=right.layout.axis.pack_host(np.concatenate(plus,axis=-1),axis=3,fill_value=0.)
        minus=right.layout.axis.pack_host(np.concatenate(minus,axis=-1),axis=3,fill_value=0.)
        weights=np.zeros((2,len(points)));npnt=len(aux['relative_points'])
        for a in range(2):weights[a,a*npnt:(a+1)*npnt]=aux['integration_weights']
        full=local_density_rhs(centroid_faces=face(L),atom_ae_faces=face(plus),atom_ps_faces=face(minus),
            left_plan=left,right_plan=right,weight_l=wl,weight_r=wr,kgrid=kgrid,mesh_xy=mesh,q_neg_idx=qneg)
        compressed=make_auxiliary_monopole_compressor(mesh,right,weights)['compress'](full)
        compressed=jax.jit(lambda a:jnp.pad(a,((0,1),(0,0),(0,0))),
                           out_shardings=NamedSharding(mesh,P(None,'x','y')))(compressed)
        ref=np.asarray(gather_to_host(face_to_batch_reshard(mesh)(compressed)))
        expected=np.zeros((4,left.n_centroid_packed,2),complex);expected[:2]=ref[qids,...,:2]
        result,receipt=_served_monopole_rhs(left,face(L),[put(c,P(None,('x','y'),None)) for c in C],
            [put(d,P(None,('x','y'),None)) for d in D],
            dict(caches={47:cache},atom_types=np.array([47,47]),centers=centers,lattice=lattice,scale=scale,
                 weight_l=wl,weight_r=wr,kgrid=kgrid,qneg=qneg,indexed_q=qids,q_indices=qids,qpad=4),mesh)
        actual=np.asarray(gather_to_host(result))
        error=float(np.max(abs(actual-expected)))
        assert error<3e-12,(kind,error)
        assert np.count_nonzero(actual[2:])==0
        assert receipt['points_per_atom']==[208,208]
        receipts.append(dict(kind=kind,maximum_absolute_error=error,selected_q=qids.tolist(),q_padding_zero=True))
    if jax.process_index()==0:
        Path(os.environ['SERVED_MONOPOLE_STAGE_REPORT']).write_text(json.dumps(dict(cases=receipts),indent=2)+'\n')
    return receipts


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(Path.cwd()))
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    run_main_and_finalize(lambda: (check_served_monopole_stage(runtime),0)[1])
