"""All-six mixed-current face Grams, q/μ transport, shell and writer parity."""
from pathlib import Path
import argparse
import json


def check_pair_face(runtime):
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from runtime.aot_memory import aot_kernel_peak_bytes
    from isdf.zeta_mubatch import _pair_tile_kernel, _pair_face_tile_kernel, _finish_v
    mesh=runtime.mesh
    assert mesh.size==4
    Q,Qp,mu,gt,ng=3,4,8,8,13
    pairs=tuple((a,b) for a in range(3) for b in range(a,3))
    qspec=P(('x','y'),None,None)
    put=lambda a,spec=qspec:device_put_process_local(a,NamedSharding(mesh,spec))
    host=lambda a:np.asarray(gather_to_host(a))
    rng=np.random.default_rng(941)
    complex_values=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
    tables=complex_values((6,Qp,ng));tables[:,Q:]=0
    counts=np.asarray([13,10,7,0],np.int32)
    slots=np.asarray([[0,7,12],[1,8,9],[0,5,6],[0,0,0]],np.int32)
    v=put(tables,P(None,('x','y'),None));ngk=put(counts,P(('x','y')))
    sl=put(slots,P(('x','y'),None))
    initial=complex_values((6,Qp,mu,mu))*.03;initial[:,Q:]=0
    initial[:,:,6:]=0;initial[:,:,:,6:]=0
    packets=[]
    for _ in range(2):
        packet=complex_values((9,Qp,mu,gt))*.1
        packet[:,:,6:]=0;packet[:,Q:]=0
        packets.append(packet)
    cases=[]
    for physical in (False,True):
        qkernel=_pair_tile_kernel(mesh,gt,pairs,use_local_augmentation=True,return_physical=physical)
        face,native=_pair_face_tile_kernel(mesh,Q,mu,gt,pairs,return_physical=physical)
        Vq=tuple(put(a) for a in initial)
        Vf=tuple(_finish_v(mesh,Q)(put(a)) for a in initial)
        Sq=tuple(put(np.zeros((Qp,mu,3),complex)) for _ in range(3))
        Sf=tuple(put(np.zeros((Qp,mu,3),complex)) for _ in range(3))
        expected=initial.copy();expected_shell=np.zeros((3,Qp,mu,3),complex)
        negative_gap=0.;errors=[];quote=None
        for t,packet in enumerate(packets):
            Z=tuple(put(a) for a in packet[:3])
            local=(tuple(put(a) for a in packet[3:6]),tuple(put(a) for a in packet[6:]))
            args=(Z,v,ngk,sl,jnp.int32(t),Vf,Sf,local)
            if quote is None:
                compiled=face.lower(*args).compile()
                price=aot_kernel_peak_bytes(compiled)
                hlo=compiled.as_text().lower()
                quote=dict(total=price.total,increment=price.resident_increment,native=native,
                    collectives={kind:hlo.count(kind+'(') for kind in
                        ('all-gather','all-to-all','all-reduce','collective-permute')})
            qresult=qkernel(Z,v,ngk,sl,jnp.int32(t),Vq,Sq,local)
            result=face(*args);jax.block_until_ready((qresult,result))
            Vq,Sq=qresult[:2];Vf,Sf=result[:2]
            idx=t*gt+np.arange(gt);mask=idx[None,:]<counts[:,None]
            for i,(a,b) in enumerate(pairs):
                weight=np.where(mask,tables[i,:,np.minimum(idx,ng-1)].T,0)
                product=lambda left,right:np.einsum('qmg,qg,qng->qmn',left.conj(),weight,right)
                cross=product(packet[a],packet[3+b])+product(packet[3+a],packet[b])
                expected[i]+=product(packet[a],packet[b])+(cross+product(packet[6+a],packet[6+b]))
                negative_gap=max(negative_gap,float(np.max(abs(cross))))
            for a in range(3):
                full=packet[a]+packet[3+a]
                expected_shell[a]+=np.einsum('qmg,qgs->qms',full,(slots[:,None,:]==idx[None,:,None]))
                errors.extend((float(np.max(abs(host(Sf[a])-expected_shell[a]))),
                               float(np.max(abs(host(Sf[a])-host(Sq[a]))))))
                if physical:
                    want=np.where(mask[:,None,:],full,0)
                    errors.extend((float(np.max(abs(host(result[2][a])-want))),
                                   float(np.max(abs(host(result[2][a])-host(qresult[2][a]))))))
            for i in range(6):
                errors.extend((float(np.max(abs(host(Vf[i])-expected[i,:Q]))),
                               float(np.max(abs(host(Vf[i])-host(Vq[i])[:Q])))))
        assert max(errors)<2e-12,errors
        assert negative_gap>1e-3,negative_gap
        cases.append(dict(return_physical=physical,max_error=max(errors),
            omitted_cross_term_negative_control=negative_gap,AOT=quote))
    return cases


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    args=parser.parse_args()
    from runtime import initialize_communicator_stack,run_main_and_finalize
    runtime=initialize_communicator_stack()
    def run():
        import jax
        from tests.test_current_mu_Fourier_p4 import check_current_fourier
        result=dict(pair=check_pair_face(runtime),complete=check_current_fourier(runtime))
        if jax.process_index()==0:
            Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        return 0
    run_main_and_finalize(run)
