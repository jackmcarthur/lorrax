"""Time the common three-channel transaction boundary without overlapping thread sums."""
from pathlib import Path
import json,time
from gw import gw_jax,gw_init,isdf_fitting
import jax
from common import jax_compile_cache
from runtime import run_main_and_finalize
rows=[]
coordinator=gw_init._CoupledMu123ZqCoordinator
original_init=coordinator.__init__;original_finish=coordinator.finish_channel
original_fit=isdf_fitting.fit_zeta_to_h5

def start(self):
    state=jax_compile_cache._STATE
    self._prof_start=(time.perf_counter(),state.compiles,state.compile_secs)
    original_init(self)

def receipt(label,before):
    state=jax_compile_cache._STATE
    row={'label':label,'seconds':time.perf_counter()-before[0],'compiles':state.compiles-before[1],'compile_seconds':state.compile_secs-before[2]}
    rows.append(row);print('[prof-zw-transaction] '+json.dumps(row),flush=True)

def finish(self,mu):
    original_finish(self,mu)
    if int(mu)==3:receipt('transverse_transaction',self._prof_start)

def fit(*args,**kwargs):
    if kwargs.get('vertex_mu_L',0):return original_fit(*args,**kwargs)
    state=jax_compile_cache._STATE;before=time.perf_counter(),state.compiles,state.compile_secs
    value=original_fit(*args,**kwargs);receipt('charge_fit',before);return value
coordinator.__init__=start;coordinator.finish_channel=finish
isdf_fitting.fit_zeta_to_h5=fit

def main():
    try:return gw_jax.main()
    finally:
        Path(f'unit_timings.rank{jax.process_index()}.json').write_text(json.dumps(rows,indent=2)+'\n');jax_compile_cache._report()
run_main_and_finalize(main)
