"""P4 actual-batch/live-input Dyson admission against independent storage/math."""
def main():
    import argparse,json,gc,weakref
    from pathlib import Path
    from types import SimpleNamespace
    import runtime
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    mesh=runtime.initialize_communicator_stack().mesh
    import jax
    import numpy as np
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local,gather_to_host
    from common.gpu_utils import device_budget_bytes
    from lxkit import memory
    from gw.tangential_recipe import CapacityLedger
    import gw.response_bank as owner
    assert mesh.size==4
    budget=int(device_budget_bytes());face=NamedSharding(mesh,P(None,'x','y'))
    n,q=32,3
    meta=SimpleNamespace(nk_tot=64,nspinor=2,nspinor_wfnfile=2,nspin=1,n_rmu=n,cell_volume=2.5)
    ledger=CapacityLedger(meta,mesh_xy=mesh,device_budget_bytes=budget)
    ledger.live_stages=();meta.tangential_capacity=ledger
    h=np.broadcast_to(np.diag(1+.01*np.arange(n)),(q,n,n)).astype(complex).copy()
    chi=np.stack([np.diag(-.1-.001*np.arange(n)-.01*k) for k in range(q)]).astype(complex)
    dc=chi*.2
    put=lambda a:device_put_process_local(a,face)
    H,C,D=map(put,(h,chi,dc));jax.block_until_ready((H,C,D))
    ledger.at_phase();base=memory.room(bucketed=False)
    expected=q*n*n*16//mesh.size
    abstract=jax.ShapeDtypeStruct((q,n,n),np.complex128,sharding=face)
    assert owner._dyson_room(ledger,mesh,(H,C,D))==base
    assert owner._dyson_room(ledger,mesh,(H,abstract,abstract))==base-2*expected
    assert owner._dyson_room(ledger,mesh,(abstract,abstract,abstract))==base-3*expected
    contact_abstract=jax.ShapeDtypeStruct((1,n,n),np.complex128,sharding=face)
    assert owner._dyson_room(ledger,mesh,(H,abstract,contact_abstract))==base-expected-n*n*16//mesh.size
    full=jax.ShapeDtypeStruct((meta.nk_tot,n,n),np.complex128,sharding=face)
    assert owner._dyson_room(ledger,mesh,(full,full,full))==base-3*meta.nk_tot*n*n*16//mesh.size
    unknown=CapacityLedger(meta,mesh_xy=mesh,device_budget_bytes=budget)
    assert owner._dyson_room(unknown,mesh,(H,C)) is None
    assert owner._dyson_room(None,mesh,(H,C)) is None

    receipt=lambda:dict(seconds={},memory=[],compiled=[])
    config=SimpleNamespace(backend=SimpleNamespace(linalg='distributed'))
    records=[]
    # Physical normalization depends on the FULL grid, while storage uses q.
    for fullq in (64,512):
        meta.nk_tot=fullq
        dyson,slope,moments,info=owner.response_algebra(meta,config,mesh_xy=mesh,n=n)
        execute=owner._bank_execution(meta,mesh,receipt0:=receipt(),config)
        result=execute(dyson.value,(H,put(chi)),"sample_dyson")
        got=np.asarray(gather_to_host(result))
        pref=2/(np.sqrt(fullq)*meta.nspin*meta.nspinor_wfnfile)
        x=pref*(h@chi@h)
        wanted=h@np.linalg.solve(np.eye(n)[None]-x,x)@h
        np.testing.assert_allclose(got,wanted,rtol=3e-11,atol=3e-12)
        records.append(dict(full_q=fullq,batch=q,prefactor=info['prefactor'],
            max_abs=float(np.max(np.abs(got-wanted))),compiled=receipt0['compiled'],
            native=receipt0.get('native_queries',[])))
        assert info['prefactor']==pref
        # Actual planning roots are already live; only future chi/dchi
        # operands spend room. The real composed program is priced here.
        phase=owner._dyson_phase(dyson,slope,H,H,mesh,'distributed',nq=q,n=n)
        assert phase[0]>=expected and phase[1]>=0
        try:dyson.value.resolve(H,jax.ShapeDtypeStruct((q+1,n,n),np.complex128,sharding=face))
        except ValueError:pass
        else:raise AssertionError('mismatched actual q batch was accepted')

    # A bounded injected-room policy fixture; it is NOT physical capacity.
    class CappedLedger(CapacityLedger):
        def at_phase(self):self.room_bytes=self.cap
    capped=CappedLedger(meta,mesh_xy=mesh,device_budget_bytes=budget)
    capped.live_stages=();capped.cap=128<<20;capped.at_phase()
    meta.tangential_capacity=capped
    dyson,_,moments,_=owner.response_algebra(meta,config,mesh_xy=mesh,n=n)
    first=dyson.value.resolve(H,C)
    n_variant=len(owner._VARIANTS);n_compiled=len(owner._COMPILED)
    capped.cap-=1<<20
    second=dyson.value.resolve(H,C)
    assert first is second and len(owner._VARIANTS)==n_variant
    try:jax.make_jaxpr(lambda h,c:dyson.value(h,c))(H,C)
    except TypeError as error:assert 'admission is eager' in str(error)
    else:raise AssertionError('nested admission traced an allocator collective')
    # The same exact resolved program owns compilation and finish.
    executable=owner._compiled(first,(H,put(chi)))
    inputs=(H,put(chi));raw=executable(*inputs)
    capped.cap=0
    value=dyson.value.finish(inputs,raw)
    np.testing.assert_allclose(np.asarray(gather_to_host(value)),wanted,rtol=3e-11,atol=3e-12)
    # Charge moments contain no solve: their construction must not trip a
    # zero-room LU guard before the streamed q width is even selected.
    capped.cap=0
    _,_,moments,_=owner.response_algebra(meta,config,mesh_xy=mesh,n=n)
    assert not isinstance(moments,owner._Decided)
    marg=(H,put(chi),put(dc));mex=owner._compiled(moments,marg)
    mgot=moments.finish(marg,mex(*marg))
    b0=h@chi@h;b1=h@dc@h
    for a,w in zip(mgot,(.5*h@b0@h,.5*h@(b1+b0@b0)@h)):
        np.testing.assert_allclose(np.asarray(gather_to_host(a)),w,rtol=3e-11,atol=3e-12)
    # Photon moments DO solve, and must resolve against their actual batch.
    capped.cap=128<<20
    photon,_,pmoments,_=owner.response_algebra(meta,config,mesh_xy=mesh,n=n,photon=True)
    zeros=put(np.zeros_like(h));contact=put(.01*np.eye(n,dtype=complex)[None])
    pargs=(H,zeros,put(chi),zeros,put(dc),contact)
    winf=np.linalg.solve(np.eye(n)[None]+h*(.01*meta.cell_volume),h)
    coefficients=(np.zeros_like(chi),np.zeros_like(chi),dc,chi)
    series=[]
    for k,c in enumerate(coefficients):
        rhs=c@winf
        for i in range(k):rhs+=coefficients[i]@series[k-i-1]
        series.append(winf@rhs)
    for batch in (q,1):
        operands=tuple(a[:batch] if a.shape[0]==q else a for a in pargs)
        pprogram=pmoments.resolve(*operands)
        pexe=owner._compiled(pprogram,operands);pout=pprogram.finish(operands,pexe(*operands))
        for a,w in zip(pout,(winf-h,*[.5*c for c in series])):
            np.testing.assert_allclose(np.asarray(gather_to_host(a)),w[:batch],rtol=3e-11,atol=3e-12)
    # A global variant/program may keep plans and tables, never this map's ledger.
    def ephemeral():
        l=CappedLedger(meta,mesh_xy=mesh,device_budget_bytes=budget)
        l.live_stages=();l.cap=128<<20
        m0=SimpleNamespace(**vars(meta));m0.tangential_capacity=l
        d,_,_,_=owner.response_algebra(m0,config,mesh_xy=mesh,n=n)
        return d.value.resolve(H,C),weakref.ref(l)
    cached,wledger=ephemeral();gc.collect();assert wledger() is None
    # Actual provider workspace and layout fallback, on one bounded matrix.
    large=512
    from distrib_la import plan,lu_workspace_bytes
    proposed=plan('solve_lu',mesh,backend='distributed',n=large,budget_bytes=0)
    native=(lu_workspace_bytes(mesh,large,np.complex128) if proposed.backend=='cusolvermp'
            else min(2_000_000_000,16*large*large//mesh.size))
    minimum=16*large*large//mesh.size+native
    whole=13*16*large*large
    context=int(memory.contexts().at(mesh.size))
    face_route=-(-whole//mesh.size)+context
    meaningful=max(minimum,face_route)<whole
    fallback={'native_minimum':int(minimum),'whole_parent':whole,'face_with_context':face_route,
              'contexts_not_withheld_by_budget':context,'meaningful':meaningful}
    if meaningful:
        meta.n_rmu=large
        capped=CappedLedger(meta,mesh_xy=mesh,device_budget_bytes=budget)
        capped.live_stages=();meta.tangential_capacity=capped
        capped.cap=max(int(minimum)+1,int(face_route),min(whole-1,2*int(minimum)))
        capped.at_phase();local=SimpleNamespace(backend=SimpleNamespace(linalg='local'))
        assert owner.dense_layout(meta,local,mesh,large)=='distributed'
        d,_,_,_=owner.response_algebra(meta,local,mesh_xy=mesh,n=large)
        mat=np.broadcast_to(np.eye(large),(1,large,large)).astype(complex).copy()
        lh,lc=put(mat),put(-.001*mat)
        resolved=d.value.resolve(lh,lc)
        execute=owner._bank_execution(meta,mesh,fallback_receipt:=receipt(),local)
        vv=execute(d.value,(lh,lc),'sample_dyson')
        np.testing.assert_allclose(np.asarray(gather_to_host(vv)),
            (-.001*pref/(1+.001*pref))*mat,rtol=3e-11,atol=3e-12)
        fallback.update(plan=resolved.lu.describe(),compiled=fallback_receipt['compiled'],
                        native=fallback_receipt.get('native_queries',[]))
    # A small explicit-provider control uses the SAME production Dyson body.
    # P4's size router may prefer one_device even after the bank's larger
    # whole-parent layout falls back; no oversized matrix is needed to test
    # the native query and its runtime cap under the actual allocator budget.
    from distrib_la import plan
    meta.n_rmu=large
    real=CapacityLedger(meta,mesh_xy=mesh,device_budget_bytes=budget)
    real.live_stages=();meta.tangential_capacity=real
    dyson,_,_,_=owner.response_algebra(meta,config,mesh_xy=mesh,n=large)
    native=plan('solve_lu',mesh,backend='cusolvermp',n=large,batched_route='auto')
    forced=owner._response_programs(*dyson.value.builder).program_for('value',native.batched)
    forced.lu=native
    mat=np.broadcast_to(np.eye(large),(1,large,large)).astype(complex).copy()
    lh=put(mat)
    execute=owner._bank_execution(meta,mesh,native_receipt:=receipt(),config)
    import distrib_la
    queried=[];original_query=distrib_la.lu_workspace_query
    def record_query(mesh,n,nrhs,dtype,cap):
        value=original_query(mesh,n,nrhs,dtype,cap)
        queried.append(dict(n=n,nrhs=nrhs,cap=int(cap),bytes=int(value)))
        return value
    distrib_la.lu_workspace_query=record_query
    try:vv=execute(forced,(lh,put(-.001*mat)),'sample_dyson')
    finally:distrib_la.lu_workspace_query=original_query
    assert real.room_bytes_per_rank(())>context+queried[-1]['bytes']
    np.testing.assert_allclose(np.asarray(gather_to_host(vv)),
        (-.001*pref/(1+.001*pref))*mat,rtol=3e-11,atol=3e-12)
    # Admission may refuse after the native query has posted a cap.
    # It must withdraw that cap before the next map uses this shape.
    events=[];cls=type(native)
    original_chunk,original_reserve=cls.chunk_room,owner._reserve
    def record_chunk(self,n,nrhs,nbytes):
        events.append(nbytes)
        return original_chunk(self,n,nrhs,nbytes)
    def refuse(*a,**k):raise RuntimeError('injected admission refusal')
    cls.chunk_room=record_chunk;owner._reserve=refuse
    try:
        try:execute(forced,(lh,put(-.001*mat)),'sample_dyson')
        except RuntimeError as error:assert 'injected admission refusal' in str(error)
        else:raise AssertionError('injected admission refusal was lost')
        assert events and events[-1] is None and any(v is not None for v in events)
        owner._reserve=original_reserve;events.clear()
        execute(forced,(lh,put(-.001*mat)),'sample_dyson')
        assert events and events[-1] is None
    finally:
        cls.chunk_room=original_chunk;owner._reserve=original_reserve
    result=dict(verdict='PASS',rank=jax.process_index(),expected_face_bytes=expected,
        actual_materialized_room=base,future_parent_charge=3*expected,
        future_full_q_charge=3*64*n*n*16//mesh.size,programs=records,
        cache_reused_across_room_change=True,cached_ledger_weak_dead=True,charge_moments_no_lu=True,
        photon_moments_actual_batches=[q,1],nested_admission_refused_before_collective=True,fallback=fallback,
        explicit_native=dict(compiled=native_receipt['compiled'],native=native_receipt.get('native_queries',[]),
                             lu_workspace_queries=queried,cap_cleared_on_refusal_and_success=True,
                             budget=memory.run(),contexts_not_withheld_by_budget=context))
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/f'rank{jax.process_index()}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('DYSON_ACTUAL_ROOM PASS',jax.process_index(),flush=True)

if __name__=='__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
