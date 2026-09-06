"""Receipt first and warm chi classes and capture one TT repeat on rank zero."""
import ctypes,hashlib,json,time
from pathlib import Path
from gw import gw_jax,w_isdf
import jax
import numpy as np
from common import jax_compile_cache
from runtime import run_main_and_finalize

name=('compute_no_pair_dirac_current_blocks' if hasattr(w_isdf,'compute_no_pair_dirac_current_blocks') else 'compute_no_pair_dirac_current_block')
original=getattr(w_isdf,name);records=[];captured=False

def timed(label,call):
    state=jax_compile_cache._STATE;before=state.compiles,state.compile_secs
    start=time.perf_counter()
    with jax.profiler.TraceAnnotation(label):
        value=call();jax.block_until_ready(value)
    row={'label':label,'seconds':time.perf_counter()-start,'compiles':state.compiles-before[0],'compile_seconds':state.compile_secs-before[1]}
    records.append(row);print('[prof-zw] '+json.dumps(row),flush=True);return value

def measured(*args,**kwargs):
    global captured
    pairs=kwargs.get('vertex_pairs',((kwargs.get('vertex_left'),kwargs.get('vertex_right')),))
    quad=args[2];meta={'pairs':pairs,'nodes':len(quad.tau),'node_digest':hashlib.sha256(np.asarray(quad.tau).tobytes()+np.asarray(quad.alpha).tobytes()).hexdigest()}
    for key,wf in zip(('left','right'),args[:2]):
        parent=getattr(wf,'green_parent',None);face=parent.psi_mun if parent is not None else wf.psi_mun
        meta[key+'_shape']=list(face.shape);meta[key+'_plan_id']=id(parent.plan) if parent is not None else None
    print('[prof-zw-metadata] '+json.dumps(meta),flush=True)
    label='chi_'+str(tuple(tuple(p) for p in pairs));value=timed(label+'_first',lambda:original(*args,**kwargs))
    for index in range(3):
        capture=not captured and tuple(pairs[0])==(1,1) and index==0
        cudart=ctypes.CDLL('libcudart.so') if capture and jax.process_index()==0 else None
        if cudart and cudart.cudaProfilerStart()!=0:raise RuntimeError('cudaProfilerStart failed')
        try:repeat=timed(label+f'_warm{index}',lambda:original(*args,**kwargs))
        finally:
            jax.effects_barrier()
            if cudart and cudart.cudaProfilerStop()!=0:raise RuntimeError('cudaProfilerStop failed')
        if capture:captured=True
        for a,b in zip(jax.tree.leaves(value),jax.tree.leaves(repeat)):
            if float(jax.numpy.max(jax.numpy.abs(a-b)))!=0:raise AssertionError('Repeated class differs')
    return value
setattr(w_isdf,name,measured)
def main():
    try:return gw_jax.main()
    finally:
        Path(f'unit_timings.rank{jax.process_index()}.json').write_text(json.dumps(records,indent=2)+'\n');jax_compile_cache._report()
run_main_and_finalize(main)
