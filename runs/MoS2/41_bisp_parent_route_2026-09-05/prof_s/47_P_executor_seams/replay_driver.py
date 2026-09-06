from gw import gw_jax
import rule_replay
import functools,json,os,time
from pathlib import Path
import jax
from common import jax_compile_cache
from gw.mpa import sigma
from gw.ppm_accumulators import DeviceOmegaAccumulator
from common.centroid_basis import PackedCentroidBasis
from runtime import run_main_and_finalize

def measured(fn,label,sync=True):
 @functools.wraps(fn)
 def call(*args,**kwargs):
  before=jax_compile_cache.compile_cache_stats();start=time.perf_counter()
  result=fn(*args,**kwargs)
  if sync:
   value=result.sigma_c_kij if hasattr(result,'sigma_c_kij') else result
   jax.block_until_ready(value)
  after=jax_compile_cache.compile_cache_stats()
  row=dict(label=label,host_ms=1000*(time.perf_counter()-start),compiles=after['compiles']-before['compiles'],compile_s=after['compile_secs']-before['compile_secs'])
  if int(os.environ['SLURM_PROCID'])==0:
   with Path('boundary.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
  return result
 return call
factory=sigma.get_shared_sigma_tau_kernel
def factory_call(*args,**kwargs):
 kernel=measured(factory,'sigma_tau_factory',False)(*args,**kwargs)
 wrapped=measured(kernel,'sigma_tau')
 if hasattr(kernel,'lower'):wrapped.lower=kernel.lower
 return wrapped
sigma.get_shared_sigma_tau_kernel=factory_call
for name in ['__init__','begin_window','precompile_tau_add','add_tau','end_window','finalize']:
 setattr(DeviceOmegaAccumulator,name,measured(getattr(DeviceOmegaAccumulator,name),'accumulator_'+name))
original=sigma._integrate_sigma_batches
def executor(wfns,batches,*args,**kwargs):
 iterator=iter(batches)
 def values():
  while True:
   try:value=measured(lambda:next(iterator),'pole_batch_read')()
   except StopIteration:return
   yield value
 pack=PackedCentroidBasis.pack_operator
 PackedCentroidBasis.pack_operator=measured(pack,'executor_pack_operator')
 try:return original(wfns,values(),*args,**kwargs)
 finally:PackedCentroidBasis.pack_operator=pack
sigma._integrate_sigma_batches=measured(executor,'sigma_executor')
run_main_and_finalize(gw_jax.main)
