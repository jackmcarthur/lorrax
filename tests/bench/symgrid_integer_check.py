"""CPU native geometry parity, refusal controls and actual WFN startup timing."""
from runtime import initialize_communicator_stack, run_main_and_finalize, rank0_print
R = initialize_communicator_stack()
import argparse, hashlib, json, time, subprocess, warnings
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack
import numpy as np
from file_io import WFNReader
from common.parallel_transport import wfn_fingerprint
import symmetry_maps.maps as maps

p=argparse.ArgumentParser();p.add_argument('--wfn',required=True);p.add_argument('--output',required=True)
a=p.parse_args()

def outcome(full,q,old=False):
    try:
        with patch.object(maps,'_uniform_kminusq_index_map',return_value=None) if old else warnings.catch_warnings():
            return maps.SymMaps._get_kminusq_index_map(full,q,name='geometry fixture'),None
    except ValueError as exc:
        return None,str(exc)

def main():
    rng=np.random.default_rng(493);checks=[]
    for grid in ((3,4,2),(1,3,1),(4,4,3)):
        ints=np.indices(grid).reshape(3,-1).T
        for shift in ((0.,0.,0.),(.5,.5,.5),(.3,0.,.5)):
            for permuted in (False,True):
                order=rng.permutation(len(ints)) if permuted else np.arange(len(ints))
                full=(ints[order]+shift)/np.asarray(grid)
                full+=rng.integers(-2,3,full.shape)
                q=ints[rng.permutation(len(ints))]/np.asarray(grid)
                q+=rng.integers(-2,3,q.shape)
                fast=maps._uniform_kminusq_index_map(full,q)
                assert fast is not None,(grid,shift,permuted)
                old,error=outcome(full,q,True);new,new_error=outcome(full,q)
                assert error is None and new_error is None and np.array_equal(old,new)
                vectors,labels=maps._unwrapped_grid_q_tables(ints[order],grid)
                direct=ints[order,None,:]-ints[None,order,:]
                v,inv=np.unique(direct.reshape(-1,3),axis=0,return_inverse=True)
                assert np.array_equal(vectors,v) and np.array_equal(labels,inv.reshape(len(ints),len(ints)))
                checks.append(dict(grid=grid,shift=shift,permuted=permuted,parity=True))
    # Tiny noisy coordinates keep the native nearest-match acceptance;
    # larger displacements/duplicates retain the exact native refusal.
    grid=np.array((3,4,2));ints=np.indices(grid).reshape(3,-1).T
    base=ints/grid; q=base[::3]
    cases=[('noisy',base.copy(),q),('invalid',base.copy(),q),
           ('duplicate',base.copy(),q),('shifted_q',base+.5/grid,q+.5/grid)]
    cases[0][1][1,0]+=1e-5;cases[1][1][1,0]+=.002;cases[2][1][1]=cases[2][1][0]
    for name,full,q in cases:
        assert maps._uniform_kminusq_index_map(full,q) is None,name
        old,error=outcome(full,q,True);new,new_error=outcome(full,q)
        assert (error is None)==(new_error is None),(name,error,new_error)
        if error is None:assert np.array_equal(old,new),name
        else:assert error==new_error,(name,error,new_error)
        checks.append(dict(name=name,fallback=True,refused=error is not None))
    bad=ints.copy();bad[-1]=bad[0]
    vectors,labels=maps._unwrapped_grid_q_tables(bad,grid)
    pair=bad[:,None]-bad[None,:];v,inv=np.unique(pair.reshape(-1,3),axis=0,return_inverse=True)
    assert np.array_equal(vectors,v) and np.array_equal(labels,inv.reshape(len(bad),len(bad)))
    negative=maps._uniform_kminusq_index_map(base,-base)
    positive=maps._uniform_kminusq_index_map(base,base)
    assert not np.array_equal(negative,positive)
    # Actual Fe header and native SymMaps constructor; verify selected columns
    # against the unchanged old door without a second full quadratic scan.
    w=WFNReader(a.wfn,mesh=R.mesh)
    full=maps.SymMaps._generate_uniform_full_kpoints(maps.SymMaps.__new__(maps.SymMaps),w)
    assert maps._uniform_kminusq_index_map(full,np.asarray(w.kpoints)[:19]) is not None
    stage_times={};uniform_calls=[]
    original_uniform=maps._uniform_kminusq_index_map
    def uniform_timed(full,q):
        start=time.monotonic();value=original_uniform(full,q)
        uniform_calls.append(dict(nk=len(full),nq=len(q),fast=value is not None,wall_s=time.monotonic()-start))
        return value
    def timed(name,original):
        def wrapper(self,*args,**kwargs):
            start=time.monotonic();value=original(self,*args,**kwargs)
            stage_times[name]=time.monotonic()-start
            return value
        return wrapper
    with ExitStack() as stack:
        stack.enter_context(patch.object(maps,'_uniform_kminusq_index_map',uniform_timed))
        for name in ('_initialize_symmetry_provenance','_initialize_active_operations',
                     '_validate_identity_grid','_initialize_spatial_operators',
                     '_initialize_k_maps','_initialize_q_maps'):
            stack.enter_context(patch.object(maps.SymMaps,name,timed(name,getattr(maps.SymMaps,name))))
        start=time.monotonic();sym=w.symmetry();elapsed=time.monotonic()-start
    sample=rng.choice(len(full),19,replace=False)
    old,error=outcome(sym.unfolded_kpts,sym.unfolded_kpts[sample],True)
    assert error is None and np.array_equal(sym.kqfull_map[:,sample],old)
    rows=rng.choice(len(full),19,replace=False)
    actual_delta=sym.kvecs_asints[rows,None]-sym.kvecs_asints[None,sample]
    encoded=sym.all_unfolded_qpts[sym.all_unfolded_qpt_ids[rows[:,None],sample[None,:]]]
    assert np.array_equal(actual_delta,encoded)
    receipt=dict(schema='symgrid.integer_geometry_check.v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_wfn_fingerprint=wfn_fingerprint(w),wfn=a.wfn,checks=checks,negative_wrong_q_sign_detected=True,
        actual=dict(kgrid=list(map(int,w.kgrid)),nk=len(full),nparent=len(w.kpoints),nq=len(sym.q_irr_full_idx),
            trs_allowed=bool(sym.trs_allowed),symmetry_initialization_s=elapsed,unwrapped_q_count=len(sym.all_unfolded_qpts),
            stages_seconds=stage_times,uniform_calls=uniform_calls,
            fullmap_sha256=hashlib.sha256(sym.kqfull_map.tobytes()).hexdigest(),q_ids_sha256=hashlib.sha256(sym.all_unfolded_qpt_ids.tobytes()).hexdigest()),
        scope='exact integer geometry and existing noisy/irregular fallback; actual Fe20cubed native metadata constructor; no GW/DFT numerical calculation or cache')
    out=Path(a.output)
    if out.exists():raise RuntimeError('immutable receipt exists')
    out.write_text(json.dumps(receipt,indent=2)+'\n');rank0_print(json.dumps(receipt),flush=True)
    w.close();return 0
run_main_and_finalize(main)
