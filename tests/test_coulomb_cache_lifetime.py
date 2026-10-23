"""Observe Coulomb-cache ownership across the normal GW driver; retain weakrefs only."""
import gc,json,os,time,weakref
from pathlib import Path

def main():
 import jax
 import gw.gw_jax as gw
 import gw.response_bank as rb
 import gw.tangential_screening as screening
 out=Path(os.environ['CACHE_PROBE_OUT']);out.mkdir(parents=True,exist_ok=True)
 result={'rank':jax.process_index(),'cache_calls':[],'checkpoints':[]}
 refs=[]
 def save():
  p=out/f'rank{jax.process_index()}.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(p)
 def checkpoint(label):
  jax.effects_barrier();gc.collect()
  result['checkpoints'].append({'label':label,'wall_time_ns':time.time_ns(),
   'root_entries':len(rb._COULOMB_ROOTS),'operator_entries':len(rb._COULOMB_OPERATORS),
   'parent_entries':len(screening._PARENT_ROWS),
   'digest_entries':len(rb._OPERATOR_DIGESTS),
   'watched':[dict(row,alive=ref() is not None) for row,ref in refs],
   'allocator':{k:int(v) for k,v in (jax.local_devices()[0].memory_stats() or {}).items() if isinstance(v,int)}});save()
 original_batch=rb._coulomb_batch
 def batch(*args,**kwargs):
  before={id(a) for v in rb._COULOMB_ROOTS.values() for a in v[:2]}
  value=original_batch(*args,**kwargs)
  result['cache_calls'].append({'span':list(args[4]),'reused_both':all(id(a) in before for a in value[:2]),'root_entries':len(rb._COULOMB_ROOTS)})
  return value
 rb._coulomb_batch=batch
 original_persist=gw._persist_screening
 def persist(*args,**kwargs):
  value=original_persist(*args,**kwargs)
  seen=set()
  for kind,arrays in [('root',[a for v in rb._COULOMB_ROOTS.values() for a in v[:2]]),('operator',list(rb._COULOMB_OPERATORS.values()))]:
   for a in arrays:
    if id(a) in seen:continue
    seen.add(id(a));refs.append(({'kind':kind,'shape':list(a.shape),'local_bytes':sum(a.addressable_data(i).nbytes for i in range(len(a.sharding.addressable_devices)))},weakref.ref(a)))
  checkpoint('after_persist_before_driver_cleanup')
  return value
 gw._persist_screening=persist
 original_sigma=gw._run_oneshot_sigma
 def sigma(*args,**kwargs):
  checkpoint('sigma_entry')
  expected=bool(int(os.environ['CACHE_EXPECT_RELEASE']))
  row=result['checkpoints'][-1]
  roots=[x for x in row['watched'] if x['kind']=='root']
  assert roots, 'fresh shared-pole fixture did not produce cached Coulomb roots'
  assert all(x['alive'] != expected for x in roots), row
  if expected:assert row['root_entries']==row['operator_entries']==row['digest_entries']==row['parent_entries']==0,row
  return original_sigma(*args,**kwargs)
 gw._run_oneshot_sigma=sigma
 try:gw.main()
 finally:
  checkpoint('after_main');rb._coulomb_batch=original_batch;gw._persist_screening=original_persist;gw._run_oneshot_sigma=original_sigma

if __name__=='__main__':
 from runtime import run_main_and_finalize
 run_main_and_finalize(main)
