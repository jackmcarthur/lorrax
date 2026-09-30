"""P4 parity and allocation-free production-shape Galerkin capacity proof."""
from pathlib import Path
import json
import argparse
import hashlib
from dataclasses import replace
from runtime import initialize_communicator_stack
rt = initialize_communicator_stack(platform='gpu')
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import device_put_process_local, gather_to_host
from isdf.galerkin import (_reduce_projection_partials, _assemble_coefficient_chunks,
    _solve_coefficient_projection, _selected_gram_from_projection,
    _galerkin_rank_metrics_kernel, _coefficients_from_projection)
from isdf.galerkin import (GalerkinBasis, _basis_check,
    _coefficient_null_tail_max, QRCP_RNG_VERSION)
from bandstructure.fh_interp import _apply_qp_block_to_compact_state
from file_io.slab_io import SlabIO
parser=argparse.ArgumentParser();parser.add_argument("--output-dir",type=Path,required=True)
out=parser.parse_args().output_dir
out.mkdir(parents=True,exist_ok=True)
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
path=out/'roundtrip.h5'
# Use the shared carrier>dataset contract: physical rank is deliberately not P-divisible.
with SlabIO(path,mode='w',mesh=mesh) as io:
    io.create_dataset('C',shape=(nk,nb,physical),dtype=solved.dtype)
    io.write_slab('C',solved)
with SlabIO(path,mode='r',mesh=mesh) as io:
    reread=io.read_slab('C',shape=(nk,nb,rank),partition_spec=sh.spec)
expected=np.pad(reference[:,:,:physical],((0,0),(0,0),(0,rank-physical)))
assert np.max(np.abs(gather_to_host(reread)-expected))<2e-15
# Publication checks an unaligned all-P coefficient tail before SlabIO.
nodes=np.zeros((rank,2,3),dtype=np.complex128)
nodes[:physical]=rng.normal(size=(physical,2,3))
factor=np.eye(rank,dtype=np.complex128)
pivots=np.arange(physical,dtype='<i8')
basis=GalerkinBasis(ctilde=reread,
    basis_at_nodes=device_put_process_local(nodes,NamedSharding(mesh,P(None,None,'y'))),
    rank_physical=physical,band_range=(0,nb),selected_state_indices=tuple(pivots),
    selection_factor=device_put_process_local(factor,rep),qrcp_seed=0,
    qrcp_rng_version=QRCP_RNG_VERSION,qrcp_eps=1e-3,qrcp_raw_rank=physical,
    qrcp_search_rank=rank,candidate_hash='a'*64,
    pivot_hash=hashlib.sha256(pivots.tobytes()).hexdigest())
provenance=dict(band_range=(0,nb),nk=nk,nb=nb,nspinor=2,
    centroid_shape=(3,),qrcp_seed=0,qrcp_eps=1e-3,qrcp_rng=QRCP_RNG_VERSION)
_basis_check(basis,provenance)
corrupt=expected.copy();corrupt[-1,-1,-1]=1e-14
corrupt=device_put_process_local(corrupt,sh)
assert float(_coefficient_null_tail_max(corrupt,physical=physical))==1e-14
try:
    _basis_check(replace(basis,ctilde=corrupt),provenance)
    raise AssertionError('nonzero synthetic coefficient was accepted')
except ValueError as exc:
    assert 'exact-null/identity' in str(exc)
tail_shape=(8000,184,2108)
tail_abstract=jax.ShapeDtypeStruct(tail_shape,jnp.complex128,sharding=sh)
tail_exe=_coefficient_null_tail_max.lower(tail_abstract,physical=2105).compile()
tail_mem=tail_exe.memory_analysis()
assert tail_mem.temp_size_in_bytes<64*1024**2,tail_mem
assert 'all-gather' not in tail_exe.as_text()
tail_receipt=dict(shape=tail_shape,physical_rank=2105,carried_rank=2108,
    temp_bytes=tail_mem.temp_size_in_bytes,argument_bytes=tail_mem.argument_size_in_bytes,
    output_bytes=tail_mem.output_size_in_bytes,no_all_gather=True,
    exact_zero_passed=True,nonzero_1e_minus14_refused=True)
metrics_small=_galerkin_rank_metrics_kernel(mesh,solved.shape,selected_rows=tuple(selected),physical=len(selected))(solved,ld)
gram=np.einsum('kna,kma->knm',reference,reference.conj())
norm=np.diagonal(gram,axis1=1,axis2=2).real
cpu=(np.diag(L).real[:len(selected)].min(),np.abs(gram-np.eye(nb)).max(),np.abs(norm-1).max(),np.sqrt(max(0,1-norm.mean())),np.abs(reference.reshape(-1,rank)[selected]-L[:len(selected)]).max(),max(1,np.abs(L[:len(selected)]).max()))
assert np.max(np.abs(np.asarray(tuple(float(x) for x in metrics_small))-np.asarray(cpu)))<5e-14
try:
    _galerkin_rank_metrics_kernel(mesh,(nk,nb,11),selected_rows=tuple(selected),physical=len(selected))
    raise AssertionError("nondivisible rank must refuse")
except ValueError:
    pass
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
    metrics=_galerkin_rank_metrics_kernel(mesh,shape,selected_rows=tuple(range(2500)),physical=2500).lower(abstract,factor).compile()
    m=metrics.memory_analysis()
    assert m.temp_size_in_bytes<3.1*local+2*2520*2520*16,(bands,m.temp_size_in_bytes)
    assert "all-gather" not in metrics.as_text()
    rows.append(dict(bands=bands,shape=shape,abstract_only=True,argument_bytes=mem.argument_size_in_bytes,output_bytes=mem.output_size_in_bytes,alias_bytes=mem.alias_size_in_bytes,temp_bytes=mem.temp_size_in_bytes,metrics_temp_bytes=m.temp_size_in_bytes,full_C_bytes=local*4,P36_C_bytes=local*4//36,no_solve_all_gather=True))
legacy=jax.jit(lambda C:C,out_shardings=rep).lower(abstract).compile()
legacy_mem=legacy.memory_analysis()
assert legacy_mem.output_size_in_bytes==local*4
assert "all-gather" in legacy.as_text()
if jax.process_index()==0:
    target=out/'result.json'
    target.write_text(json.dumps(dict(parity_max_abs=err,rows=rows,
        publication_tail=tail_receipt,
        replicated_negative_output_bytes=legacy_mem.output_size_in_bytes),indent=2)+'\n')
    print('GALERKIN RANK SHARDS PASS',json.dumps(rows),flush=True)
