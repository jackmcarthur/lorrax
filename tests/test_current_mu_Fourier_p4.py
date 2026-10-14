"""Complete current Fourier layout parity with physical Slab and OWN3 fixtures."""
from pathlib import Path
from contextlib import contextmanager
import argparse
import json


def check_current_fourier(runtime):
    import numpy as np
    import jax
    from types import SimpleNamespace
    from common.collectives import gather_to_host
    from common.centroid_basis import PackedCentroidBasis
    from common import gpu_utils
    import isdf.zeta_mubatch as zm
    from tests.test_atomic_breit_p4 import check_breit_group
    host=lambda value:np.asarray(gather_to_host(value))
    first=np.asarray([[.11,.17,.23],[.19,.31,.43],[.27,.41,.59]])
    sym=SimpleNamespace(sym_matrices=np.asarray([np.eye(3),-np.eye(3)],np.int32),translations=np.zeros((2,3)))
    basis=PackedCentroidBasis.build(np.concatenate((first,1.-first)),sym,(4,4,6),runtime.mesh,
                                   coordinate_kind='fractional')
    literal=np.argsort(np.asarray([0,3,2,5,1,4]))
    assert basis.n_logical==6 and basis.n_packed==8 and not basis.is_identity
    receipts=[];comparisons=[];written=[]
    original=zm.contract_v_group
    def paired(zetas,pairs,v_tables,**kwargs):
        provider=getattr(zetas[0],'local_augmentation',None)
        if provider is None:return original(zetas,pairs,v_tables,**kwargs)
        group=provider['group_provider']
        prepare=group['prepare_fourier_coefficients']
        group['prepare_fourier_coefficients']=lambda state,**options:state
        try:
            reference=tuple(host(v) for v in original(zetas,pairs,v_tables,**kwargs))
            shell=tuple(host(z.shell) for z in zetas)
            writers=kwargs.get('zeta_ios')
            payload=None if writers is None else tuple(v.payload.copy() for v in writers)
        finally:group['prepare_fourier_coefficients']=prepare
        actual=original(zetas,pairs,v_tables,**kwargs)
        receipt=group['fourier_distribution_receipt']
        assert receipt['mode']=='mu',receipt
        pair_receipt=group.get('pair_distribution_receipt')
        if pair_receipt is not None:
            expected='face' if group['body_metric']=='mixed_reciprocal' else 'q'
            assert pair_receipt['mode']==expected,pair_receipt
        for quote in receipt['actual_AOT']:
            assert quote['collective_counts']['all-gather']==0,quote
            if quote['owner']=='mu_Fourier':assert not any(quote['collective_counts'].values()),quote
        errors=[float(np.max(abs(host(v)-r))) for v,r in zip(actual,reference)]
        errors += [float(np.max(abs(host(z.shell)-r))) for z,r in zip(zetas,shell)]
        if payload is not None:errors += [float(np.max(abs(v.payload-r))) for v,r in zip(writers,payload)]
        assert max(errors)<2e-11,errors
        comparisons.append(max(errors));receipts.append(dict(receipt,pair_distribution=pair_receipt))
        # All-rank admission must decline safely, leaving onsite banks usable.
        saved=gpu_utils.device_budget_bytes()/1e9
        def declined(state,**options):
            try:
                gpu_utils.set_device_budget_gb(1e-7 if jax.process_index()==0 else saved)
                result=prepare(state,**options)
                assert result is state and group['fourier_distribution_receipt']['mode']=='q'
                return result
            finally:gpu_utils.set_device_budget_gb(saved)
        group['prepare_fourier_coefficients']=declined
        try:
            fallback=original(zetas,pairs,v_tables,**kwargs)
            assert group['pair_distribution_receipt']['mode']=='q'
            assert max(float(np.max(abs(host(v)-r))) for v,r in zip(fallback,reference))<2e-11
        finally:group['prepare_fourier_coefficients']=prepare
        group['pair_distribution_receipt']=pair_receipt
        return actual

    class Writer:
        def __init__(self,zeta):
            self.mesh=zeta.mesh;self.path=zeta.path
            self.payload=np.zeros((3,8,7),complex)
        def write_slab(self,name,value,offset):
            assert name=='zeta_q_G'
            first=offset[2];n=min(value.shape[-1],7-first)
            self.payload[...,first:first+n]=host(value)[...,:n]
        def sync_writes(self):pass
    @contextmanager
    def opened(zetas,facts):
        writers=tuple(Writer(z) for z in zetas)
        written.append(writers)
        yield writers
    def verify(zetas,result,facts):
        for writer,physical,compensation in zip(written[-1],facts['physical'],facts['compensation']):
            assert np.max(abs(writer.payload-physical))<2e-11
            assert np.max(abs(writer.payload-compensation))>1e-7
            assert not np.any(writer.payload[:,6:])
    publication=dict(open=opened,verify=verify)
    zm.contract_v_group=paired
    try:
        cases=[]
        for options in (dict(circular=False),dict(circular=True,compensated=True,
                averaged_heads=True,embedded_gamma=True,sys_dim=2,streamed_coefficients=True)):
            cases.append(check_breit_group(runtime,mu_basis=basis,canonical_to_packed_reference=literal,
                publication_fixture=publication,**options))
    finally:zm.contract_v_group=original
    # At Q=P the ordinary owner already occupies the complete mesh.
    from jax.sharding import NamedSharding,PartitionSpec as P
    from runtime.padding import padded_axis
    from isdf.atomic_breit import radial_breit_providers
    qa=padded_axis(4,runtime.mesh,name='Fourier fallback q')
    store=SimpleNamespace(Q=4,Q_pad=4,mu_pad=8,g_tile=4,q_axis=qa,
        g_axis=padded_axis(4,4,name='Fourier fallback G'))
    zetas=tuple(SimpleNamespace(store=store,mesh=runtime.mesh,mu_basis=None,
        ngk_per_q=np.full(4,4,np.int32),gvec_components=np.zeros((4,3,4),np.int32)) for _ in range(3))
    sh=NamedSharding(runtime.mesh,P(('x','y'),None,None))
    cloud=tuple(jax.ShapeDtypeStruct((4,8,8),np.complex128,sharding=sh) for _ in range(3))
    radius=np.linspace(.01,1.,8)
    providers=radial_breit_providers(zetas,cloud,radius=radius,lm=np.asarray([[0,0]],np.int32),
        centers_cart=np.zeros((1,3)),q_plus_G_cart=np.zeros((4,4,3)),cell_volume=1000.,
        fft_points=96,support_radius=1.,minimum_atom_image_distance=3.,reciprocal_prefactor=-8*np.pi/1000.)
    group=providers[0]['group_provider'];state=dict(coefficients=cloud,moments=())
    assert group['prepare_fourier_coefficients'](state) is state
    assert group['fourier_distribution_receipt']['mode']=='q'
    return dict(cases=cases,q_covers_mesh_fallback=True,layout_max_errors=comparisons,admission_receipts=receipts,
        scope='P4 Q3 complex OWN3, six V blocks, physical delta writer, packed centroid ghosts, Slab nn/tn/nt and Gamma; no material accuracy or speed claim')


def quote_actual(runtime,manifest,g_tile):
    """Compile actual shapes only; the deliberately tiny budget forbids dispatch."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from types import SimpleNamespace
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common import gpu_utils
    from runtime.padding import padded_axis
    from gw.isdf_augmentation import _radial_grid
    from isdf.atomic_breit import radial_breit_providers
    mesh=runtime.mesh;Q=3;Qp=padded_axis(Q,mesh,name='actual current q').carrier
    mu,na,nh=4272,8,64
    radial=json.loads(Path(manifest).read_text())['radial']
    radius,_,R=_radial_grid(radial);nr=len(radius)
    assert nr==256
    qs=NamedSharding(mesh,P(('x','y'),None,None))
    cloud=tuple(jax.ShapeDtypeStruct((Qp,mu,na*nh*nr),jnp.complex128,sharding=qs) for _ in range(3))
    moments=tuple(jax.ShapeDtypeStruct((Qp,mu,na*nh*2),jnp.complex128,sharding=qs) for _ in range(3))
    qa=padded_axis(Q,mesh,name='actual current q')
    store=SimpleNamespace(Q=Q,Q_pad=Qp,mu_pad=mu,g_tile=g_tile,q_axis=qa,
        g_axis=padded_axis(g_tile,g_tile,name='actual current G'))
    counts=np.full(Q,g_tile,np.int32);G=np.zeros((Q,3,g_tile),np.int32)
    zetas=tuple(SimpleNamespace(store=store,mesh=mesh,mu_basis=None,ngk_per_q=counts,gvec_components=G) for _ in range(3))
    K=np.zeros((Q,g_tile,3));K[:,:,0]=np.linspace(0.,.3,g_tile)
    lm=np.asarray([(l,m) for l in range(8) for m in range(-l,l+1)],np.int32)
    providers=radial_breit_providers(zetas,cloud,radius=radius,lm=lm,
        centers_cart=np.column_stack((np.arange(na)*(2*R+1),np.zeros((na,2)))),q_plus_G_cart=K,
        cell_volume=1000.,fft_points=96,support_radius=R,minimum_atom_image_distance=2*R+1,
        reciprocal_prefactor=-8*np.pi/1000.,fourier_points=4097,
        interpolation_degree=radial['interpolation_degree'],quadrature_order=radial['quadrature_order'])
    group=providers[0]['group_provider'];state=dict(coefficients=cloud,moments=moments)
    saved=gpu_utils.device_budget_bytes()/1e9
    try:
        gpu_utils.set_device_budget_gb(1e-7)
        assert group['prepare_fourier_coefficients'](state) is state
    finally:gpu_utils.set_device_budget_gb(saved)
    receipt=group['fourier_distribution_receipt']
    return dict(receipt,shape=dict(Q=Q,Qp=Qp,mu=mu,atoms=na,Lmax=7,radial_points=nr,g_tile=g_tile,P=mesh.size),
        scope='Shape-only AOT; add production live baseline and caller workspace to phase increments; no numerical bank dispatch')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--manifest');parser.add_argument('--g-tile',type=int)
    args=parser.parse_args()
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    def run():
        import jax
        result=quote_actual(runtime,args.manifest,args.g_tile) if args.manifest else check_current_fourier(runtime)
        if jax.process_index()==0:
            Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result),flush=True)
        return 0
    run_main_and_finalize(run)
