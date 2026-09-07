import compile_receipts
import functools,json,time
from pathlib import Path
import jax
from gw import gw_jax,isdf_fitting
from isdf import core
from common import jax_compile_cache
from runtime import run_main_and_finalize

def measured(original,label):
 @functools.wraps(original)
 def call(*args,**kwargs):
  before=jax_compile_cache.compile_cache_stats();start=time.perf_counter()
  value=original(*args,**kwargs)
  jax.block_until_ready(value)
  after=jax_compile_cache.compile_cache_stats()
  if jax.process_index()==0:
   with Path('coupled_units.jsonl').open('a') as f:
    f.write(json.dumps(dict(label=label,vertex=kwargs.get('vertex_mu_L',0),batch=getattr(args[0],'nbatch',getattr(args[0],'shape',[None])[0]),ms=1000*(time.perf_counter()-start),compiles=after['compiles']-before['compiles'],compile_s=after['compile_secs']-before['compile_secs']))+'\n')
  return value
 return call
for module in (core,isdf_fitting):
 module.solve_zeta=measured(module.solve_zeta,'solve_zeta')
 module.factor_c_q=measured(module.factor_c_q,'factor_c_q')
run_main_and_finalize(gw_jax.main)
