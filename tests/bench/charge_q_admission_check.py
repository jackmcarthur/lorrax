"""Native measured response-floor correction and one-parent refusal, no fields."""
from runtime import initialize_communicator_stack,run_main_and_finalize,rank0_print
R=initialize_communicator_stack()
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import jax
from jax.sharding import NamedSharding,PartitionSpec as P
from common.collectives import rank0_transaction
from common.grouped_layout import identity_square_grouped_shard_layout
from file_io import WFNReader
from gw.centroid_k_unfold import CentroidKUnfoldPlan
from gw.w_isdf import _get_chi_fractional_contour_kernel_face as factory
from gw.response_bank import _check_response_q_panel
from symmetry_maps import q_negation_index
ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);a=ap.parse_args()
def main():
 base=Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14/runs/Fe/73_soc80_20nscf_closed_buffer_20260930/qe')
 w=WFNReader(str(base/'WFN.h5'),mesh=R.mesh,qe_schema=str(base/'data-file-schema.xml'));s=w.symmetry();kg=tuple(map(int,w.kgrid));np_,m,nb=1062,904,90
 pl=CentroidKUnfoldPlan(R.mesh,identity_square_grouped_shard_layout(m,m,(2,2)),np.asarray(s.irr_idx_k),np.asarray(s.sym_idx_k),np.tile(np.arange(m),(2*int(w.ntran),1)),np.zeros((2*int(w.ntran),m,3)),np.asarray(s.unfolded_kpts)[np.asarray(s.kirr_fullids)],np.asarray(s.spinor_action(s.sym_idx_k,nspinor=2)),int(w.ntran),2,parent_full_rows=np.asarray(s.kirr_fullids,dtype=np.int32))
 parent=np.asarray(s.kirr_fullids);neg=np.asarray(q_negation_index(kg))
 def rows(width):
  def union(p):return tuple(dict.fromkeys(p.tolist()+neg[p].tolist()))
  return max((union(parent[i:i+width]) for i in range(0,len(parent),width)),key=len)
 # This native local-volume proxy keeps every Fe k/spin action. Only the
 # caller's room is controlled to exercise correction without a GPU draw.
 cache={};builds=[]
 def build(wfns,meta,mesh,support,*,q_ids,n_outputs,ordered,vertex):
  q=tuple(q_ids);builds.append(len(q))
  if q not in cache:
   import minimax
   cap=minimax.RESPONSE_NODE_CAPACITY
   fn=factory(mesh,kg,2,(np_,nb,m,2),k_unfold_plan=pl,layout='face',selected_q=q,pair_mode='direct',ordered=True,bank_carry=True)
   sd=lambda sh,dt,sp:jax.ShapeDtypeStruct(sh,dt,sharding=NamedSharding(mesh,sp))
   args=(sd((cap,),np.complex128,P()),sd((2,2,cap),np.complex128,P()),sd((np_,2,m,nb),np.complex128,P(None,None,'x','y')),sd((np_,nb,2,m),np.complex128,P(None,'x',None,'y')),sd((np_,nb),np.float64,P()),sd((np_,nb),np.complex128,P()),sd((np_,nb),np.complex128,P()),sd((2,),np.float64,P()),sd((2,len(q),m,m),np.complex128,P(None,None,'x','y')))
   cache[q]=fn.lower(*args).compile()
  return cache[q]
 results=[]
 for room in (60_000_000_000,56_280_000_000):
  builds.clear();ledger=SimpleNamespace(live_stages=(),reserve_bytes_per_rank=0,room_bytes_per_rank=lambda stages:room);meta=SimpleNamespace(shared_pole_capacity=ledger)
  started=time.monotonic()
  with patch('gw.response_bank._stream_executable',side_effect=build),patch('common.gpu_utils.device_room_bytes',return_value=room):
   try:
    check=_check_response_q_panel(None,meta,R.mesh,{},width=325,rows_for_width=rows,ordered=True,vertex=None,face_bytes=16*452**2)
   except MemoryError as e:
    assert room==56_280_000_000 and 'GATE compiled_chunk_capacity' in str(e),(room,str(e))
    assert builds[-1]<=2,builds
    row=dict(room=room,minimum_refused=True,compiled_q_rows=builds.copy(),refusal=str(e),wall_s=time.monotonic()-started)
   else:
    assert room==60_000_000_000 and check.recompiled and 1<check.chunk<325 and check.compiled_bytes<=room
    ma=check.compiled.memory_analysis()
    row=dict(room=room,parent_width=check.chunk,recompiled=check.recompiled,compiled_q_rows=builds.copy(),new_bytes=check.compiled_bytes,temp_bytes=ma.temp_size_in_bytes,output_bytes=ma.output_size_in_bytes,alias_bytes=ma.alias_size_in_bytes,wall_s=time.monotonic()-started)
  results.append(row);rank0_print(json.dumps(row),flush=True)
 rank0_transaction(a.output,stage='native response q admission',write=lambda:Path(a.output).write_text(json.dumps(results,indent=2)+'\n'));w.close();return 0
run_main_and_finalize(main)
