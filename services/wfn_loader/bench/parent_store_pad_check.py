"""Check parent-store padding against build_axis_stress_fixture.py values."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import runtime
stack = runtime.initialize_communicator_stack()
import jax
import numpy as np
from common.collectives import barrier
from common.jax_compile_cache import compile_cache_stats
from common.meta import Meta
from common.psi_G_store import load_parent_psi_G
from jax.sharding import PartitionSpec as P
from wfn_loader import WfnLoader

p = argparse.ArgumentParser()
p.add_argument('--wfn', required=True)
p.add_argument('--out', required=True)
p.add_argument('--loops', type=int, default=2)
p.add_argument('--band-chunk', type=int, default=64)
p.add_argument('--stop', type=int, default=20002)
a = p.parse_args()
out = Path(a.out); out.mkdir(parents=True,exist_ok=True)
rank = jax.process_index()
stream = (out/f'rank{rank}.jsonl').open('w',buffering=1)
with WfnLoader(a.wfn,mesh=stack.mesh) as wfn:
    sym = wfn.symmetry()
    lo,hi = 3,min(a.stop,int(wfn.nbands)-1)
    assert wfn.ngkmax==7 and wfn.nspinor==1
    meta = Meta.from_system(wfn,sym,nval=1,ncond=int(wfn.nbands)-1,
                           nband=int(wfn.nbands),n_rmu=19,bispinor=False)
    for repeat in range(a.loops):
        barrier(f'parent-pad-{repeat}')
        before=compile_cache_stats(); started=time.perf_counter()
        result=load_parent_psi_G(wfn=wfn,mesh_xy=stack.mesh,meta=meta,
            band_range=(lo,hi),band_chunk=a.band_chunk,centroid_indices=None,
            placement='device',k_domain='ibz',print_fn=lambda x:None)
        jax.block_until_ready(result.psi_G)
        seconds=time.perf_counter()-started; after=compile_cache_stats()
        assert result.psi_G.sharding.spec==P(None,None,None,('x','y'))
        assert tuple(result.psi_G.shape)==(int(wfn.nkpts),-(-(hi-lo)//int(stack.mesh.size))*int(stack.mesh.size),1,
                                         -(-7//int(stack.mesh.size))*int(stack.mesh.size))
        digest=hashlib.sha256()
        for shard in result.psi_G.addressable_shards:
            actual=np.asarray(shard.data)
            ks,bs,ss,gs=[np.arange(n)[idx] for n,idx in zip(result.psi_G.shape,shard.index)]
            assert len(ss)==1
            for k0 in range(0,len(ks),16):
                for b0 in range(0,len(bs),512):
                    k=ks[k0:k0+16,None,None]
                    b=bs[None,b0:b0+512,None]+lo
                    g=gs[None,None,:]
                    re=np.ldexp(((b+1)*8192+(k+1)*8+g+1).astype(float),-29)
                    im=np.ldexp(((b*17+k*131+g*7)&((1<<20)-1))-(1<<19),-21)
                    expected=re+1j*im
                    expected=np.where((b<hi)&(g<3+2*(k%3)),expected,0)
                    block=actual[k0:k0+16,b0:b0+512,0,:]
                    assert np.array_equal(block,expected),('coefficient/pad mismatch',rank,k0,b0)
                    digest.update(np.ascontiguousarray(block))
            del actual,block,expected
        del shard
        record=dict(status='PASS',rank=rank,repeat=repeat,seconds=seconds,
            shape=list(result.psi_G.shape),band_chunk=a.band_chunk,logical_bands=hi-lo,
            rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            sha256=digest.hexdigest(),compile_delta={k:after[k]-before[k] for k in
                ('compiles','compile_secs','probes','hits','read_secs','compile_fingerprint_secs','compile_agreement_secs')})
        stream.write(json.dumps(record)+'\n')
        if rank==0: print(json.dumps(record),flush=True)
        result.psi_G.delete(); result.sphere_index.delete()
        del result
stream.close()
runtime.finalize_process(0)
