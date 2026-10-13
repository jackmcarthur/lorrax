"""Complete complex onsite blocks: whole-q reference versus bounded face GEMM.

Run this small diagnostic under the canonical four-rank runtime. It is an
execution/layout test of identical finite radial factors, not a CrI3 fit,
timing or physical-field convergence certificate.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import hashlib
import os
import time


def check_face_onsite(runtime, *, circular, compensated=False, maximum=7):
    import numpy as np
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from runtime.padding import padded_axis
    from isdf.atomic_breit import radial_breit_providers
    from isdf.zeta_mubatch import _to_mu_owner

    mesh=runtime.mesh
    if dict(mesh.shape)!={'x':2,'y':2} or runtime.process_count!=4:
        raise ValueError('Actual P4 square mesh required')
    Q=3;qa=padded_axis(Q,mesh,name='onsite test q')
    logical_mu=10;ma=padded_axis(logical_mu,mesh,name='onsite test centroid')
    mu=ma.carrier;na=2;nr=8;R=1.1
    lm=np.asarray([(l,m) for l in range(maximum+1) for m in range(-l,l+1)],np.int32)
    nh=len(lm);nf=na*nh*nr
    r=R*(np.arange(1,nr+1)/nr)**3
    B=(np.asarray([[1,1j,0],[1,-1j,0],[0,0,np.sqrt(2)]])/np.sqrt(2)
       if circular else np.eye(3,dtype=complex))
    sh=NamedSharding(mesh,P(('x','y'),None,None))
    put=lambda value:device_put_process_local(np.asarray(value),sh)
    rng=np.random.default_rng(160391+int(circular)+3*int(compensated))
    random=lambda shape:rng.normal(size=shape)+1j*rng.normal(size=shape)
    rhs=[];ps=[]
    for _ in range(3):
        value=random((qa.carrier,mu,na,nh,nr))*.02
        value*=r[None,None,None,None]**lm[None,None,None,:,0,None]
        value*=1-(r/R)**2
        value[:Q,logical_mu:]=0.;value[Q:]=np.nan+1j*np.nan
        smooth=random(value.shape)*.03
        smooth[:Q,logical_mu:]=0.;smooth[Q:]=np.nan+1j*np.nan
        rhs.append(put(value.reshape(qa.carrier,mu,nf)))
        ps.append(put(smooth.reshape(qa.carrier,mu,nf)))
    store=SimpleNamespace(Q=Q,Q_pad=qa.carrier,mu_pad=mu,g_tile=4,
        g_axis=padded_axis(4,4,name='onsite test G'))
    counts=np.full(Q,4,np.int32);G=np.zeros((Q,3,4),np.int32)
    zetas=tuple(SimpleNamespace(store=store,mesh=mesh,mu_basis=None,
        ngk_per_q=counts,gvec_components=G) for _ in range(3))
    K=rng.normal(size=(Q,4,3))*.3;K[0,0]=0.
    providers=radial_breit_providers(zetas,tuple(rhs),
        smooth_rhs=tuple(ps) if compensated else None,
        body_metric='compensated' if compensated else 'mixed_reciprocal',
        radius=r,lm=lm,centers_cart=np.asarray([[0,0,0],[4.8,.2,-.1]]),
        q_plus_G_cart=K,cell_volume=1000.,fft_points=96,support_radius=R,
        minimum_atom_image_distance=4.5,reciprocal_prefactor=-8*np.pi/1000.,
        current_basis_rows=B,field_tile=7,centroid_tile=5,fourier_points=4097)
    group=providers[0]['group_provider']
    state=group['prepare'](tuple(p['rhs'] for p in providers))
    pairs=tuple((a,b) for a in range(3) for b in range(a,3))
    old=group['onsite'](state,pairs)
    face=group['onsite_face'](state,pairs)
    jax.block_until_ready(face)
    expected=tuple(_to_mu_owner(mesh,Q,'xy')(value) for value in old)
    target=NamedSharding(mesh,P(None,'x','y'))
    if any(value.sharding!=target or value.shape!=(Q,mu,mu) for value in face):
        raise ValueError('Onsite output did not remain on the centroid face')
    # This bounded 3q×6×12² fixture alone is gathered for comparison.
    actual=np.stack([gather_to_host(value) for value in face])
    reference=np.stack([gather_to_host(value) for value in expected])
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        raise ValueError('Poisoned q-padding reached physical output')
    maximum_error=float(np.max(abs(actual-reference)))
    scale=max(1.,float(np.max(abs(reference))))
    envelope=32768*np.finfo(float).eps*scale
    if maximum_error>envelope:
        raise ValueError('Complete complex face/whole-q onsite blocks differ')
    ghost=max(float(np.max(abs(actual[:,:,logical_mu:]))),
              float(np.max(abs(actual[:,:,:,logical_mu:]))))
    if ghost!=0.:
        raise ValueError('Inactive centroid rows contributed to onsite energy')
    # Complex cross blocks make an extra right conjugation detectable.
    if float(np.max(abs(reference.imag)))<1e-8:
        raise ValueError('The equality witness has no resolved complex cross block')
    return dict(Lmax=maximum,circular=circular,compensated=compensated,
        logical_mu=logical_mu,carrier_mu=mu,q_logical=Q,q_carrier=qa.carrier,
        max_complex_error=maximum_error,arithmetic_envelope=envelope,
        inactive_centroid_max_abs=ghost,output_sharding='P(None,x,y)',
        packet_diagnostics=group['onsite_face_packet_diagnostics'])


def quote_actual_onsite_shape(runtime, manifest):
    """Actual 4272/L7/r256 shape, without allocating coefficient banks."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from runtime.padding import padded_axis
    from common.gpu_utils import device_budget_bytes
    from gw.isdf_augmentation import _radial_grid
    from isdf.atomic_breit import radial_breit_providers
    mesh=runtime.mesh;Q=3;Qp=4;mu=4272;na=8;nh=64
    radial=json.loads(Path(manifest).read_text())['radial']
    radius,_,R=_radial_grid(radial);nr=len(radius)
    if nr!=256:
        raise ValueError('Actual radial256 control changed')
    lm=np.asarray([(l,m) for l in range(8) for m in range(-l,l+1)],np.int32)
    sh=NamedSharding(mesh,P(('x','y'),None,None))
    cloud=tuple(jax.ShapeDtypeStruct((Qp,mu,na*nh*nr),jnp.complex128,sharding=sh) for _ in range(3))
    moments=tuple(jax.ShapeDtypeStruct((Qp,mu,na*nh*2),jnp.complex128,sharding=sh) for _ in range(3))
    store=SimpleNamespace(Q=Q,Q_pad=Qp,mu_pad=mu,g_tile=4,
        g_axis=padded_axis(4,4,name='onsite AOT G'))
    counts=np.full(Q,4,np.int32);G=np.zeros((Q,3,4),np.int32)
    zetas=tuple(SimpleNamespace(store=store,mesh=mesh,mu_basis=None,
        ngk_per_q=counts,gvec_components=G) for _ in range(3))
    # K and separated centres do not enter the onsite executable. This is
    # a shape/memory quote, not a second physical geometry certificate.
    K=np.zeros((Q,4,3));K[:,:,0]=np.linspace(0.,.3,4)
    B=np.asarray([[1,1j,0],[1,-1j,0],[0,0,np.sqrt(2)]])/np.sqrt(2)
    t=time.perf_counter()
    providers=radial_breit_providers(zetas,cloud,body_metric='mixed_reciprocal',
        radius=radius,lm=lm,centers_cart=np.column_stack((np.arange(na)*(2*R+1),np.zeros((na,2)))),
        q_plus_G_cart=K,cell_volume=1000.,fft_points=96,support_radius=R,
        minimum_atom_image_distance=2*R+1,reciprocal_prefactor=-8*np.pi/1000.,
        current_basis_rows=B,fourier_points=4097,
        interpolation_degree=radial['interpolation_degree'],quadrature_order=radial['quadrature_order'])
    group=providers[0]['group_provider']
    pairs=tuple((a,b) for a in range(3) for b in range(a,3))
    executable=group['onsite_face_kernel'](pairs).lower(cloud,moments,
        *group['onsite_kernel_arguments']).compile()
    m=executable.memory_analysis()
    quote=dict(arguments=int(m.argument_size_in_bytes),outputs=int(m.output_size_in_bytes),
        temporary=int(m.temp_size_in_bytes),aliases=int(m.alias_size_in_bytes))
    quote['total_addressed_bytes']=quote['arguments']+quote['outputs']+quote['temporary']-quote['aliases']
    # Same conservative seven whole-q matrix residence as the consuming
    # public AOT owner. Actual source/factor leaves still require its normal
    # full-phase preflight; they are not silently assumed absent here.
    matrix=16*Qp*mu*mu/mesh.size
    quote['with_seven_q_matrices_bytes']=quote['total_addressed_bytes']+7*matrix
    quote['budget_bytes']=float(device_budget_bytes())
    if quote['with_seven_q_matrices_bytes']>quote['budget_bytes']:
        raise MemoryError('Actual-shape onsite plus matrix residence exceeds live device budget')
    return dict(Q=Q,Qpad=Qp,mu=mu,atoms=na,Lmax=7,radial_points=nr,
        quote=quote,packet_diagnostics=group['onsite_face_packet_diagnostics'],
        elapsed_seconds=time.perf_counter()-t,
        qualification='Shape-only onsite AOT; actual source/factor leaves and complete Fourier/writer phase retain mandatory public preflight')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--spec',required=True)
    args=p.parse_args()
    spec=json.loads(Path(args.spec).read_text())
    if spec['status']!='READY_P4_SOURCE_REVIEWED' or not os.environ.get('SLURM_STEP_ID'):
        raise ValueError('Exact reviewed P4 compute request required')
    for filename,expected in spec['source_sha256'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
            raise ValueError('Pinned source changed: '+filename)
    from runtime import initialize_communicator_stack,run_main_and_finalize
    rt=initialize_communicator_stack()
    def run():
        import jax
        results=[]
        for case in (dict(circular=True),dict(circular=False),dict(circular=True,compensated=True,maximum=3)):
            results.append(check_face_onsite(rt,**case))
            if jax.process_index()==0:print(json.dumps(results[-1]),flush=True)
        actual_quote=quote_actual_onsite_shape(rt,spec['manifest'])
        if jax.process_index()==0:
            out=Path(args.output);out.mkdir(exist_ok=False)
            (out/'receipt.json').write_text(json.dumps(dict(
                status='PASS_SAME_FACTOR_COMPLEX_ONSITE_FACE_AND_PADDING',cases=results,
                actual_shape_AOT=actual_quote,
                scope='Static execution/layout equality only; no actual fit, speedup or physical current accuracy admission.'),
                sort_keys=True,indent=2)+'\n')
            print(json.dumps(results),flush=True)
        return 0
    return run_main_and_finalize(run)


if __name__=='__main__':
    main()
