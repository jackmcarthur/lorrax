"""Time synchronized tau units and capture exactly the second native unit."""
from pathlib import Path
import ctypes,functools,json,os,time
if Path('rule_replay.py').exists():
 import rule_replay
import jax
from gw import gw_jax,ppm_tau_kernel
from gw.mpa import sigma as mpa_sigma
from common import jax_compile_cache
from runtime import run_main_and_finalize
original=ppm_tau_kernel.get_shared_sigma_tau_kernel
calls=0

def factory(*args,**kwargs):
 kernel=original(*args,**kwargs)
 @functools.wraps(kernel)
 def measured(*a,**kw):
  global calls
  i=calls;calls+=1
  before=jax_compile_cache.compile_cache_stats();start=time.perf_counter()
  cudart=None
  if cudart and cudart.cudaProfilerStart()!=0:raise RuntimeError('capture start')
  with jax.profiler.TraceAnnotation('sigma_tau'):
   result=kernel(*a,**kw);jax.block_until_ready(result)
  if cudart and cudart.cudaProfilerStop()!=0:raise RuntimeError('capture stop')
  after=jax_compile_cache.compile_cache_stats()
  if int(os.environ['SLURM_PROCID'])==0:
   with Path('boundary.jsonl').open('a') as f:f.write(json.dumps(dict(label='sigma_tau',unit=i,host_ms=1000*(time.perf_counter()-start),compiles=after['compiles']-before['compiles'],compile_s=after['compile_secs']-before['compile_secs']))+'\n')
  return result
 if hasattr(kernel,'lower'):measured.lower=kernel.lower
 return measured
ppm_tau_kernel.get_shared_sigma_tau_kernel=factory
mpa_sigma.get_shared_sigma_tau_kernel=factory
run_main_and_finalize(gw_jax.main)
