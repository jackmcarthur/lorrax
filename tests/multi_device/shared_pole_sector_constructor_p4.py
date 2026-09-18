"""Physical signed photon bank through the public sector constructor/store seam."""


def check_sector_constructor(mesh, root, *, linalg="local", parents=16, return_observables=False):
    from types import SimpleNamespace
    import os
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.centroid_basis import PackedCentroidBasis
    from common.collectives import rank0_transaction, gather_to_host
    from symmetry_maps import QirrTables, centroid_source_map_and_wrap
    from gw.photon_layout import PhotonBasisLayout
    from gw.shared_pole_recipe import CapacityLedger, ROLE_CODES
    from gw.shared_pole_sectors import construct_sector_poles, positive_cross_models
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1 as gates
    from file_io import shared_pole_store as store
    from file_io.slab_io import SlabIO

    run=root/f"constructor_{os.environ['SLURM_STEP_ID']}_{linalg}"
    rank0_transaction(run,stage='plant.directory',write=lambda:run.mkdir())
    rotation=np.eye(3,dtype=np.int32)[None]
    sym=SimpleNamespace(sym_matrices=rotation,translations=np.zeros((1,3)),
        trs_allowed=False,active_symmetry_rows=np.array([0],np.int32),
        operation_typing_source='planted authenticated identity magnetic group')
    sym.operation_rows=lambda rows:(np.repeat(rotation,len(rows),axis=0),
                                    np.zeros((len(rows),3)),np.zeros(len(rows),bool))
    sym.spinor_action=lambda rows,nspinor:np.ones((len(rows),1,1),complex)
    sym.cartesian_action=lambda rows,axial,time_odd:np.repeat(rotation,len(rows),axis=0)
    nc,nt,nq=32,4,parents
    fft=(32,1,1)
    coordinates=np.column_stack((np.arange(nc),np.zeros((nc,2),int))).astype(np.int32)
    bases=tuple(PackedCentroidBasis.build(coordinates[:n],sym,fft,mesh) for n in (nc,nt))
    tables=[]
    for basis in bases:
        perm,wrap=centroid_source_map_and_wrap(basis.canonical_indices,rotation,
            sym.translations,np.asarray(fft),extend_trs=False)
        qt=QirrTables(irr_idx_q=np.arange(nq,dtype=np.int32),sym_idx_q=np.zeros(nq,np.int32),
            q_irr_frac=np.column_stack((np.arange(nq)/nq,np.zeros((nq,2)))),
            sym_perm=perm,L_table=wrap,n_sym_spatial=1)
        tables.append(dict(qirr=qt,q_irr_full_idx=np.arange(nq,dtype=np.int64),sym=sym))
    meta=SimpleNamespace(mu_basis=bases[0],nspinor=4,nkx=nq,nky=1,nkz=1,
        fft_grid=fft,nk_tot=nq,n_rmu=nc,n_rmu_padded=bases[0].n_packed)
    recipe=dict(version='shared_real_pole_v1_r3b',gate_version='shared_real_pole_gates_ordered_v1',
        accuracy='production',direction_cutoff=1e-3,multiplet_relative_tolerance=1e-6,
        operator_realization='little-group-reynolds-v1',role_codes=ROLE_CODES,
        z_ry=np.array([.7+.4j,.8j,1.1+.3j,1.3j],complex),
        role=np.array([0,1,3,4],np.int8),distinct_id=np.arange(4,dtype=np.int64),
        held=np.array([False,False,True,True]),
        support_pair=np.array([[-1,-1],[-1,-1],[0,1],[0,1]],np.int64),
        fit_ids=np.array([0,1],np.int64),held_ids=np.array([2,3],np.int64))
    meta.shared_pole_recipe=recipe
    # The production ledger receives a resolved per-device budget. For this
    # tiny synthetic bank, use the live GPU limit instead of its 3U fallback;
    # the unchanged 3U scaling predicate is still reported independently.
    device_limit=int(jax.local_devices()[0].memory_stats()['bytes_limit'])
    meta.shared_pole_capacity=CapacityLedger(meta,mesh_xy=mesh,
        device_budget_bytes=device_limit)
    meta.shared_pole_capacity.live_stages=()
    layout=PhotonBasisLayout.from_centroid_extents(nc,nt,mesh)
    identity={key:'signed-constructor-'+key for key in store._IDENTITY_KEYS}
    path=run/'bank.h5'
    bank=dict(path=str(path),identity=identity,tables=tables[0],sector_tables=tables,
              mu_bases=bases,photon_layout=layout)
    store.initialize_shared_pole_bank(path,meta=meta,tables=tables[0],recipe=recipe,
        identity=identity,mesh_xy=mesh,photon_layout=layout,mu_bases=bases)
    # Charge/current endpoint amplitudes are independent, complex and small
    # enough for a strictly stable signed interaction. Cminus=conj(Cplus)
    # supplies the exact q/-q ordered identity, with broken time reversal.
    rng=np.random.default_rng(479)
    # The production policy retains four infinity and several independent
    # finite directions even in CC. Supply a full-rank physical plant so an
    # active direction does not lie in an exact null space of the toy model.
    npositive=10
    positive=.025*(rng.normal(size=(nc+3*nt,npositive))+1j*rng.normal(size=(nc+3*nt,npositive)))
    positive*=np.sqrt(np.linspace(.4,.9,npositive))[None]
    c=np.concatenate((positive,positive.conj()),axis=1)
    j=np.diag([1.]*npositive+[-1.]*npositive)
    v=np.diag([1.2]*nc+[-.3]*(3*nt))
    d=np.diag([0.]*nc+[.05]*(3*nt))
    u=np.linalg.solve(np.eye(len(v))+v@d,v)
    energies=np.linspace(.5,1.8,npositive)
    h=np.diag(np.tile(energies,2))+c.conj().T@u@c
    out=u@c
    assert np.linalg.eigvalsh(h).min()>.45
    def value(z):
        resolvent=np.linalg.inv(z*j-h)
        return out@resolvent@out.conj().T,-out@resolvent@j@resolvent@out.conj().T/(2*z)
    # Native photon layout is mesh-major/channel-major; the exact model above
    # uses charge rows followed by mu-major Cartesian current rows.
    index=[]
    for owner in range(layout.mesh_side):
        index.extend(range(owner*(nc//layout.mesh_side),(owner+1)*(nc//layout.mesh_side)))
        for component in range(3):
            index.extend(nc+3*mu+component for mu in range(owner*(nt//layout.mesh_side),(owner+1)*(nt//layout.mesh_side)))
    def packed(a,samples=False):
        a=np.asarray(a,dtype=np.complex128)[...,index,:][...,index]
        a=np.broadcast_to(a,(nq,)+a.shape).copy()
        spec=P(None,None,'x','y') if samples else P(None,'x','y')
        return jax.make_array_from_callback(a.shape,NamedSharding(mesh,spec),lambda ix:a[ix])
    values=[value(z) for z in recipe['z_ry']]
    mirrors=[value(-z.conjugate()) for z in recipe['z_ry']]
    moments=[out@np.linalg.matrix_power(j@h,k)@j@out.conj().T/2 for k in range(4)]
    store.write_shared_pole_bank(path,q_span=(0,nq),sample_span=(0,4),
        Wc=packed([a[0] for a in values],True),dWc_ds=packed([a[1] for a in values],True),
        Wc_mirror=packed([a[0] for a in mirrors],True),
        dWc_mirror_ds=packed([a[1] for a in mirrors],True),
        constant=packed(u-v),**{f'M{k}':packed(m) for k,m in enumerate(moments)},
        meta=meta,expected_identity=identity,mesh_xy=mesh)
    result=construct_sector_poles(bank,meta,SimpleNamespace(backend=SimpleNamespace(linalg=linalg)),
                                  mesh_xy=mesh,output=str(run/'model.h5'))
    rounds=[row for row in result['q_receipts'] if 'held' in row]
    assert all(row['execution']==('face' if linalg=='distributed' else 'local') for row in rounds)
    if linalg=='distributed':
        assert len(rounds)==nq and all(len(row['parents'])==1 for row in rounds)
    handle=result['handle']
    manifest=store.validate_shared_pole_sector_manifest(handle['path'],expected_identity=identity,
        mesh_xy=mesh,capacity=meta.shared_pole_capacity)
    assert manifest['digest']==handle['digest']
    headers=manifest['model_headers']
    factors={}
    for sector,family in (('CC',0),('TT',1),('CT_C',0),('CT_T',1)):
        with SlabIO(handle['sectors'][sector]['path'],mode='r',mesh=mesh) as io:
            b,_,poles,k=store.read_shared_pole_faces(io,(0,1),meta=meta,header=headers[sector],basis=bases[family])
        # Tiny oracle only: production never gathers a factor panel.
        factors[sector]=(gather_to_host(b)[0].reshape((-1,b.shape[-1])),np.asarray(poles)[0],int(np.asarray(k)[0]))
    errors={};observables={}
    for name,left,right,sl,sr in (('CC','CC','CC',slice(0,nc),slice(0,nc)),
            ('TT','TT','TT',slice(nc,None),slice(nc,None)),
            ('CT','CT_C','CT_T',slice(0,nc),slice(nc,None))):
        bl,poles,k=factors[left];br,other,kr=factors[right]
        np.testing.assert_array_equal(poles,other);assert k==kr
        z=.9+.31j;omega=np.sqrt(poles[:k])
        got=(bl[:,:k]/(2*omega*(z-omega)))@br[:,:k].conj().T
        got-=(bl[:,:k].conj()/(2*omega*(z+omega)))@br[:,:k].T
        exact=value(z)[0][sl,sr]
        observables[name]=got
        errors[name]=float(np.linalg.norm(got-exact)/np.linalg.norm(exact))
        assert errors[name]<1e-7,(name,errors[name])
    # Endpoint-weight asymmetry: one low pole violates only the current
    # dropped-weight budget. The combined model must refuse admission.
    def batch(a):
        a=np.broadcast_to(np.asarray(a),(mesh.size,)+np.shape(a)).copy()
        return jax.make_array_from_callback(a.shape,NamedSharding(mesh,P(('x','y'))),lambda ix:a[ix])
    signed=(batch(np.array([[1.,1e-12]],complex)),batch(np.array([[1.,10.]],complex)),
            batch(np.array([1.,2000.])),batch(np.ones(2,bool)))
    pair,zero=positive_cross_models(signed,mesh_xy=mesh,gates=gates)
    assert bool(jnp.all(zero['charge']['zero_policy']))
    assert not bool(jnp.any(zero['current']['zero_policy']))
    assert not bool(jnp.any(zero['zero_policy']))
    assert bool(jnp.all(pair[0][1]==pair[1][1])) and bool(jnp.all(pair[0][2]==pair[1][2]))
    receipt=dict(name='production_sector_constructor_manifest',held_W_relative=errors,
                 parents=nq,manifest=handle['path'],linalg=linalg,asymmetric_endpoint_loss_refused=True)
    return (receipt,observables) if return_observables else receipt


def main():
    import argparse
    import json
    import os
    from pathlib import Path
    from runtime import initialize_communicator_stack, finalize_process
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    initialize_communicator_stack()
    import jax
    from common.collectives import resolve_mesh
    from shared_pole_sectors_p4 import run_extent_span_checks
    mesh=resolve_mesh()
    rows=run_extent_span_checks(mesh)
    if jax.process_index()==0:
        print(json.dumps(dict(status='RETAINED_SPAN_PASS',checks=rows)),flush=True)
    rows.append(check_sector_constructor(mesh,args.output.parent))
    result=dict(status='PASS',checks=rows,job=os.environ.get('SLURM_JOB_ID'),
        step=os.environ.get('SLURM_STEP_ID'),
        scope='P4 signed sector public constructor and authenticated stores; unequal parent coefficient spans; no production deck or integrated Sigma')
    if jax.process_index()==0:
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finalize_process()


if __name__=='__main__':
    main()
