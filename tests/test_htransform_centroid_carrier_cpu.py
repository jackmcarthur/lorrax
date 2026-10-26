"""Fine-wavefunction projection must never drop mu sharding for odd targets."""
import re
import runtime
from runtime.source_closure import ensure_source_closure
ensure_source_closure()
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh,NamedSharding,PartitionSpec as P
from bandstructure.bse_setup import _project_fine_wfns
jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('side,ns,output',[(2,1,14),(2,2,16),(3,1,15),(3,2,18)])
def test_physical_targets_project_directly_to_distributed_carrier(side,ns,output):
    if jax.device_count()<side*side:
        pytest.skip('requires nine virtual CPU devices')
    mesh=Mesh(np.array(jax.devices()[:side*side]).reshape(side,side),('x','y'))
    logical,nq,rank,nb=13,1,side*side*2,side
    target=-(-(ns*logical)//(side*side))*(side*side)
    rng=np.random.default_rng(537+side+ns)
    L=rng.normal(size=(nq,target,rank))+1j*rng.normal(size=(nq,target,rank))
    L[:,ns*logical:]=np.nan
    U=np.eye(rank,dtype=complex)[None]
    lam=-np.arange(1,rank+1,dtype=float)[None]
    face=NamedSharding(mesh,P(None,'x','y'))
    rep=NamedSharding(mesh,P())
    y=NamedSharding(mesh,P(None,None,None,'y'))
    x=NamedSharding(mesh,P(None,'x',None,None))
    @jax.jit(in_shardings=(rep,face,face),out_shardings=(y,x))
    def project(lam,U,L):
        _,psi,_=_project_fine_wfns(lam,U,L,0,nb,ns,logical,output)
        return psi,jnp.transpose(psi,(0,3,1,2))
    args=(jax.device_put(lam,rep),jax.device_put(U,face),jax.device_put(L,face))
    exe=project.lower(*args).compile()
    py,px=exe(*args)
    exact=(L[:,:ns*logical,:nb]/(-lam[:,None,:nb])).reshape(nq,ns,logical,nb).transpose(0,3,1,2)
    wanted=np.pad(exact,((0,0),(0,0),(0,0),(0,output-logical)))
    np.testing.assert_allclose(np.asarray(py),wanted,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(np.asarray(px),wanted.transpose(0,3,1,2),rtol=1e-14,atol=1e-14)
    assert py.sharding.is_equivalent_to(y,4)
    assert px.sharding.is_equivalent_to(x,4)
    # Reject both full-face materialization and an odd physical-mu complex
    # intermediate in the actual optimized projector (not a mock kernel).
    hlo=exe.as_text()
    assert not re.search(r'c128\[[^\]]*\b13\b[^\]]*\]',hlo)
    full_psi = nq*ns*output*nb
    for line in hlo.splitlines():
        if ' all-gather(' not in line or not any(owner in line for owner in ('jit(project)/div','jit(_take)/gather')):
            continue
        dims=re.search(r'= c128\[([^\]]+)\]',line)
        assert dims is not None
        assert np.prod([int(n) for n in dims[1].split(',')]) < full_psi,line



@pytest.mark.parametrize('side,ns', [(2,1),(2,2),(3,1),(3,2)])
def test_actual_bundle_epilogue_crops_q_on_legal_output_shardings(side,ns):
    import ast
    from functools import partial
    from pathlib import Path
    if jax.device_count()<side*side:
        pytest.skip('requires nine virtual CPU devices')
    mesh=Mesh(np.array(jax.devices()[:side*side]).reshape(side,side),('x','y'))
    nqpad,nb,rank=3*side*side,2*side+1,3*side*side
    mu=-(-13//(side*side))*(side*side)
    nq=2*side*side+1
    rep=NamedSharding(mesh,P())
    q4=NamedSharding(mesh,P(('x','y'),None,None,None))
    q3=NamedSharding(mesh,P(('x','y'),None,None))
    q2=NamedSharding(mesh,P(('x','y'),None))
    outY=NamedSharding(mesh,P(None,None,None,'y'))
    outX=NamedSharding(mesh,P(None,'x',None,None))
    cs=NamedSharding(mesh,P(None,('y','x'),None))
    source=Path(__file__).parents[1]/'src/bandstructure/bse_setup.py'
    node=next(n for n in ast.walk(ast.parse(source.read_text()))
              if isinstance(n,ast.FunctionDef) and n.name=='_build_bundle')
    from common.staged_reshard import batch_to_face_reshard
    namespace=dict(jax=jax,jnp=jnp,partial=partial,out_Y=outY,out_X=outX,
                   rep=rep,coeff_sh=cs,return_coeffs=True,nq=nq,nb_fi=nb,
                   nspinor=ns,mu_out=mu,mu_divisor=side,mesh_xy=mesh,
                   NamedSharding=NamedSharding,P=P,batch_to_face_reshard=batch_to_face_reshard)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(source),'exec'),namespace)
    program=namespace['_build_bundle']()
    psi=np.arange(nqpad*nb*ns*mu).reshape(nqpad,nb,ns,mu).astype(complex)
    c=np.arange(nqpad*rank*nb).reshape(nqpad,rank,nb).astype(complex)
    e=np.arange(nqpad*nb).reshape(nqpad,nb).astype(float)
    args=(jax.device_put(psi,q4),jax.device_put(e,q2),jax.device_put(e,q2),jax.device_put(c,q3))
    exe=program.lower(*args).compile()
    actual=exe(*args)
    for observed,wanted in zip(actual,(psi[:nq],psi[:nq].transpose(0,3,1,2),e[:nq],e[:nq],c[:nq])):
        np.testing.assert_array_equal(np.asarray(observed),wanted)
    assert actual[0].sharding.is_equivalent_to(outY,4)
    assert actual[1].sharding.is_equivalent_to(outX,4)
    assert actual[4].sharding.is_equivalent_to(cs,3)
    hlo=exe.as_text()
    # These exact full-payload local shapes are the former eager-crop defect.
    for shape in ((nq,nb,ns,mu),(nqpad,nb,ns,mu),(nq,rank,nb),(nqpad,rank,nb)):
        assert 'c128['+','.join(map(str,shape))+']' not in hlo,hlo


@pytest.mark.parametrize('side,ns', [(2,1),(2,2),(2,4),(3,1),(3,2),(3,4)])
def test_fine_wavefunction_chunks_export_only_requested_coefficients(monkeypatch,side,ns):
    """Actual public solve/projection, odd bands/targets, with a fixed-gauge oracle.

    The optional bank must disappear at the compiled chunk boundary, before
    Python drops it. Internal eigenvectors are still needed to form psi.
    """
    from bandstructure import bse_setup as B
    from ffi.fft import xla_reference
    from isdf.galerkin import NodalInterp
    if jax.device_count()<side*side:
        pytest.skip('requires nine virtual CPU devices')
    mesh=Mesh(np.array(jax.devices()[:side*side]).reshape(side,side),('x','y'))
    processes,nb,logical=side*side,5,13
    divisor=int(np.lcm(processes,ns))
    rank=-(-nb//divisor)*divisor
    target_carrier=-(-(ns*logical)//processes)*processes
    rng=np.random.default_rng(847+side+ns)
    target=rng.normal(size=(1,nb,target_carrier))+1j*rng.normal(size=(1,nb,target_carrier))
    target[:,:,ns*logical:]=0
    coeff_sh=NamedSharding(mesh,P(None,None,('x','y')))
    coeff=jax.device_put(np.eye(rank,dtype=complex)[None,:nb],coeff_sh)
    basis=NodalInterp(coeff,np.zeros((rank//ns,3)),np.eye(3),3,ns,
        target=jax.device_put(target,coeff_sh),target_frac=np.zeros((logical,3)))
    chunks=[]
    original=B.admit_node_route

    def observe(stage,program,operands,**kw):
        route,executable=original(stage,program,operands,**kw)
        chunks.append(executable)
        return route,executable

    monkeypatch.setattr(B,'admit_node_route',observe)
    outputs=[]
    with xla_reference():
        for keep in (True,False):
            outputs.append(B.compute_wfns_fi(ctilde=coeff,B_at_mu=basis,
                enk_sigma=jnp.arange(-3.,2.)[:,None],kgrid_co=(1,1,1),
                q_list=np.zeros((1,3)),band_window_fi=(1,4),mesh_xy=mesh,
                return_coeffs=keep,use_low_mem_eigh=(side==3 and ns==4)))
    kept,omitted=outputs
    assert len(chunks)==2
    assert chunks[0].out_info[2].shape==(processes,rank,3)
    assert chunks[1].out_info[2] is None,'discarded coefficients still leave the priced chunk'
    assert not hasattr(omitted,'coeffs_fi')
    for name in ('enk_full','lam_fi','psi_rmu_Y','psi_rmuT_X'):
        left,right=np.asarray(getattr(kept,name)),np.asarray(getattr(omitted,name))
        np.testing.assert_array_equal(left,right)
    np.testing.assert_allclose(np.asarray(kept.enk_full),[[-2.,-1.,0.]],atol=5e-13)
    c=np.asarray(kept.coeffs_fi)[0]
    phase=c[np.arange(1,4),np.arange(3)]
    np.testing.assert_allclose(np.abs(phase),1.,atol=3e-13)
    np.testing.assert_allclose(c,np.eye(rank)[:,1:4]*phase,atol=3e-13)
    exact=target[0,1:4,:ns*logical].reshape(3,ns,logical)*phase[:,None,None]
    padded=np.pad(exact,((0,0),(0,0),(0,kept.n_mu_carrier-logical)))[None]
    np.testing.assert_allclose(np.asarray(kept.psi_rmu_Y),padded,rtol=3e-13,atol=3e-13)
    np.testing.assert_allclose(np.asarray(kept.psi_rmuT_X),padded.transpose(0,3,1,2),rtol=3e-13,atol=3e-13)
