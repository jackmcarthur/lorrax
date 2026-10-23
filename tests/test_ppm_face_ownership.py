"""P4 producer/placement and metadata-only partner-cache oracles."""
def run_checks(mesh):
    import gc
    import weakref
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.gpu_utils import device_budget_bytes
    from symmetry_maps import QirrOperator, q_negation_index
    import gw.ppm_sigma as ppm
    device_budget_bytes()
    assert mesh.size == 4
    ptr=lambda a:int(a.addressable_data(0).unsafe_buffer_pointer())
    face=NamedSharding(mesh,P(None,'x','y'))
    q,m=4,32
    r=np.arange(m);active=r<m-2;block=active[:,None]&active[None,:]
    wc=-.02/(1+(r[:,None]-r[None,:])**2)
    omega=.6+2.8*(r[:,None]+r[None,:])/(2*(m-3))
    wi=wc*omega**2/(omega**2+4)
    odd=1j*.003/(1+(r[:,None]-r[None,:])**2)
    v=.2*np.eye(m)
    original=ppm.fit_gn_ppm_from_wc_pair
    captured={}
    def producer(*args,**kwargs):
        result=original(*args,**kwargs)
        captured.update(fit=result,wc0=args[0])
        return result
    ppm.fit_gn_ppm_from_wc_pair=producer
    fits=[]
    try:
        for ordered,coarsen,sh in [(False,True,face),(True,True,face),
                                  (False,False,NamedSharding(mesh,P()))]:
            make=lambda a:device_put_process_local(np.broadcast_to(np.where(block,a,0),(q,m,m)).astype(complex).copy(),sh)
            V,W0,Wp=make(v),make(v+wc),make(v+wi+(odd if ordered else 0))
            out=ppm.fit_ppm(W0,Wp,V,2j,mesh,n_mu_logical=m-2,mu_active_mask=active,
                ordered_orientations=ordered,coarsen_extreme_tails=coarsen,
                q_neg_index=np.asarray(q_negation_index((2,2,1))),print_fn=lambda *a,**k:None)
            pairs=[('Omega_q','omega_qmunu'),('B_q','B_qmunu'),('valid_mask_q','valid_qmunu')]
            if ordered:pairs.append(('B_odd_q','B_odd_qmunu'))
            for dst,src in pairs:
                a,b=getattr(out,dst),getattr(captured['fit'],src)
                assert a.sharding.spec==face.spec
                np.testing.assert_array_equal(np.asarray(gather_to_host(a)),np.asarray(gather_to_host(b)))
                if sh==face:assert ptr(a)==ptr(b)
            np.testing.assert_array_equal(np.asarray(gather_to_host(out.Wc0_q)),np.asarray(gather_to_host(captured['wc0'])))
            if sh==face:assert ptr(out.Wc0_q)==ptr(captured['wc0'])
            for a in [out.Omega_q,out.B_q,out.valid_mask_q]+([out.B_odd_q] if ordered else []):
                host=np.asarray(gather_to_host(a))
                assert np.all(host[:,~block]==0) and np.all(np.isfinite(host))
            fits.append(dict(ordered=ordered,tail=coarsen,input_spec=str(sh.spec),
                             changed_tails=int(captured['fit'].n_tail_low+captured['fit'].n_tail_high)))
            captured.clear()
    finally:ppm.fit_gn_ppm_from_wc_pair=original

    # Independently enumerate the real -q lane map, including an antiunitary
    # centroid permutation and a remapping of stored q-row labels.
    m=12;grid=(3,1,1)
    perm=np.stack([np.arange(m),np.arange(m).reshape(-1,2)[:,::-1].ravel()]).astype(np.int32)
    tables=dict(irr_idx=np.array([0,1,1],np.int32),sym_idx=np.array([0,0,1],np.int32),
        sym_perm=perm,L_table=np.zeros((2,m,3)),q_irr_frac=np.array([[0,0,0],[1/3,0,0]]),
        n_sym_spatial=1,full_rows=np.array([0,1],np.int32))
    def bind(t,number):
        a=device_put_process_local(np.full((2,m,m),number,complex),face)
        op=QirrOperator(values=a,**t)
        return op.minus_q_partner(grid,mesh),weakref.ref(a),weakref.ref(op)
    fn,wval,wop=bind(tables,1)
    fn2,wval2,wop2=bind(tables,7)
    assert fn2 is fn
    remap=dict(tables,irr_idx=np.array([1,0,0],np.int32),
               q_irr_frac=tables['q_irr_frac'][::-1].copy(),full_rows=np.array([1,0],np.int32))
    fn3,wval3,wop3=bind(remap,9)
    assert fn3 is not fn
    gc.collect()
    assert all(w() is None for w in [wval,wop,wval2,wop2,wval3,wop3])
    for cached in [fn,fn3]:
        ops=[c.cell_contents for c in cached.__closure__ or () if isinstance(c.cell_contents,QirrOperator)]
        assert len(ops)==1 and ops[0].values is None and ops[0].load is None
    lane=np.arange(2*m*m).reshape(2,m,m)
    masks=[lane%3==0,lane%5==1]
    neg=np.asarray(q_negation_index(grid))
    for call,t in [(fn,tables),(fn3,remap)]:
        compiled=jax.jit(call)
        for mask in masks:
            wanted=[]
            for full in t['full_rows']:
                row=t['irr_idx'][neg[full]];action=t['sym_idx'][neg[full]]
                p=t['sym_perm'][action]
                wanted.append(mask[row][np.ix_(p,p)])
            actual=compiled(device_put_process_local(mask,face))
            np.testing.assert_array_equal(np.asarray(gather_to_host(actual)),np.asarray(wanted))
    return dict(verdict='PASS',fits=fits,partner_changed_values=True,partner_remapped_rows=True,
                retired_input_weakrefs=6,partner_templates_metadata_only=True)

def main():
    import argparse,json
    from pathlib import Path
    import runtime
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    mesh=runtime.initialize_communicator_stack().mesh
    result=run_checks(mesh)
    import jax
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/f'rank{jax.process_index()}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PPM_OWNERSHIP PASS',jax.process_index(),flush=True)

if __name__=='__main__':
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
