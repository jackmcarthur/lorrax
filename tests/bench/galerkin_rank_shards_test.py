"""P4 parity and allocation-free production-shape Galerkin capacity proof."""
from pathlib import Path
import json
import numpy as np
from runtime import initialize_communicator_stack
rt = initialize_communicator_stack(platform='gpu')
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import device_put_process_local, gather_to_host
from isdf.galerkin import (_reduce_projection_partials, _assemble_coefficient_chunks,
    _solve_coefficient_projection, _selected_gram_from_projection,
    _galerkin_rank_metrics, _coefficients_from_projection)
from bandstructure.fh_interp import _apply_qp_block_to_compact_state
from file_io.slab_io import SlabIO
mesh=rt.mesh
assert mesh.size == 4
sh = NamedSharding(mesh,P(None,None,('x','y')))
rep=NamedSharding(mesh,P())
rng=np.random.default_rng(192)
nk,nb,rank=7,5,12
proj=rng.normal(size=(nk,nb,rank))+1j*rng.normal(size=(nk,nb,rank))
L=np.tril(rng.normal(size=(rank,rank))+1j*rng.normal(size=(rank,rank)))
L[np.diag_indices(rank)] = np.arange(rank)+20
pd=device_put_process_local(proj,sh);ld=device_put_process_local(L,rep)
solved=_solve_coefficient_projection(pd,ld,mesh)
reference=np.linalg.solve(L,proj.reshape(-1,rank).conj().T).conj().T.reshape(proj.shape)
err=float(np.max(np.abs(gather_to_host(solved)-reference)))
assert err<2e-15,err
partials=np.tile(proj/4,(4,1,1))
acc=device_put_process_local(partials,NamedSharding(mesh,P(('x','y'),None,None)))
reduced=_reduce_projection_partials(acc,mesh)
assert np.max(np.abs(gather_to_host(reduced)-proj))<1e-15
chunks=[device_put_process_local(proj[:,:3],sh),device_put_process_local(proj[:,3:],sh)]
assembled=_assemble_coefficient_chunks(chunks,logical_widths=(3,2),nk=nk,rank=rank,mesh_xy=mesh)
assert np.array_equal(gather_to_host(assembled),proj)
selected=np.array([1,7,12,30])
picked=_selected_gram_from_projection(assembled,selected_states=selected,rank_carrier=rank,mesh_xy=mesh)
expected=np.pad(proj.reshape(-1,rank)[selected],((0,rank-len(selected)),(0,0)))
assert np.array_equal(gather_to_host(picked),expected)
U=np.broadcast_to(np.eye(3),(nk,3,3)).copy();U[:,0,0]=0;U[:,1,1]=0;U[:,0,1]=1;U[:,1,0]=-1
energy=rng.normal(size=(nb,nk));qp=rng.normal(size=(nk,3))
cq,eq=_apply_qp_block_to_compact_state(solved,device_put_process_local(energy,rep),device_put_process_local(U,rep),device_put_process_local(qp,rep),band_offset=1)
expected=reference.copy();expected[:,1:4]=np.einsum('kmn,kma->kna',U,reference[:,1:4]);energy[1:4]=qp.T
assert np.max(np.abs(gather_to_host(cq)-expected))<2e-15
assert np.array_equal(gather_to_host(eq),energy)
physical=9
path=Path(__file__).resolve().parents[2]/'runs'/'galerkin_rank_shards_roundtrip.h5'
# Use the shared carrier>dataset contract: physical rank is deliberately not P-divisible.
with SlabIO(path,mode='w',mesh=mesh) as io:
    io.create_dataset('C',shape=(nk,nb,physical),dtype=solved.dtype)
    io.write_slab('C',solved)
with SlabIO(path,mode='r',mesh=mesh) as io:
    reread=io.read_slab('C',shape=(nk,nb,rank),partition_spec=sh.spec)
expected=np.pad(reference[:,:,:physical],((0,0),(0,0),(0,rank-physical)))
assert np.max(np.abs(gather_to_host(reread)-expected))<2e-15
rows=[]
for bands in (184,198):
    shape=(8000,bands,2520)
    abstract=jax.ShapeDtypeStruct(shape,jnp.complex128,sharding=sh)
    factor=jax.ShapeDtypeStruct((2520,2520),jnp.complex128,sharding=rep)
    fn=jax.jit(lambda C,L:_solve_coefficient_projection(C,L,mesh),donate_argnums=(0,))
    exe=fn.lower(abstract,factor).compile();mem=exe.memory_analysis();hlo=exe.as_text()
    local=8000*bands*2520*16//4
    assert 'all-gather' not in hlo
    assert mem.temp_size_in_bytes<4*local+2520*2520*16,(bands,mem)
    metrics=_galerkin_rank_metrics.lower(abstract,factor,selected_rows=tuple(range(2500)),physical=2500).compile()
    m=metrics.memory_analysis()
    assert m.temp_size_in_bytes<local+2*2520*2520*16,(bands,m)
    rows.append(dict(bands=bands,shape=shape,abstract_only=True,argument_bytes=mem.argument_size_in_bytes,output_bytes=mem.output_size_in_bytes,alias_bytes=mem.alias_size_in_bytes,temp_bytes=mem.temp_size_in_bytes,metrics_temp_bytes=m.temp_size_in_bytes,full_C_bytes=local*4,P36_C_bytes=local*4//36,no_solve_all_gather=True))
if jax.process_index()==0:
    target=Path(__file__).resolve().parent/'galerkin_rank_shards_result.json'
    target.write_text(json.dumps(dict(parity_max_abs=err,rows=rows),indent=2)+'\n')
    print('GALERKIN RANK SHARDS PASS',json.dumps(rows),flush=True)
