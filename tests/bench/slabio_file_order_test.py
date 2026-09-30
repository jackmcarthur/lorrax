"""P4 SlabIO exact file-order staging and bounded scratch verification.

Production-sized inputs are abstract compiler probes, not runtime allocations.
"""
from pathlib import Path
import argparse
import json
from runtime import initialize_communicator_stack
rt = initialize_communicator_stack(platform='gpu')
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import gather_to_host, device_put_process_local
from file_io.slab_io import SlabIO
from file_io._slab_io_ffi import (_file_order_take, _file_order_move_supported,
                                _file_order_plan, _spec_axes)
mesh = rt.mesh
ap=argparse.ArgumentParser()
ap.add_argument('--out',type=Path,required=True)
root=ap.parse_args().out
root.mkdir(parents=True,exist_ok=True)
assert mesh.size == 4
rows = []
for index, (shape, spec, k, sizes, height, starts) in enumerate([
    ((13,16,18), P(None,'x','y'), 0, (5,16,18), 8, (7,0,0)),
    ((3,13,16,18), P(None,None,'x','y'), 1, (2,5,16,18), 8, (1,7,0,0)),
    ((12,16,18), P('x',None,'y'), 0, (12,16,18), 12, (0,0,0)),
    ((13,16,18), P(None,'y','x'), 0, (5,16,18), 8, (7,0,0)),
    ((13,16,18), P(None,'x',None), 0, (5,16,18), 8, (7,0,0)),
    ((13,16,20), P(None,None,('x','y')), 0, (5,16,20), 8, (7,0,0)),
    ((3,13,16,20), P(None,None,None,('x','y')), 1,
     (2,5,16,20), 8, (1,7,0,0)),
    ((13,20,18), P(None,('x','y'),None), 0, (5,20,18), 8, (7,0,0)),
]):
    sh = NamedSharding(mesh,spec)
    @jax.jit(out_shardings=sh)
    def make():
        v = jnp.arange(np.prod(shape),dtype=jnp.float64).reshape(shape)
        return v + 1j*(v*.5+3)
    a = make()
    assert _file_order_move_supported(spec,shape,k,mesh)
    take = _file_order_take(mesh,shape,a.dtype,k,sizes,height,spec)
    control = device_put_process_local(np.asarray(starts,np.int32),NamedSharding(mesh,P()))
    actual = gather_to_host(take(a,control))
    ref = np.arange(np.prod(shape),dtype=np.float64).reshape(shape)
    ref = ref+1j*(ref*.5+3)
    ref = ref[tuple(slice(s,s+n) for s,n in zip(starts,sizes))]
    pad = [(0,0)]*len(shape);pad[k]=(0,height-sizes[k]);ref=np.pad(ref,pad)
    assert np.array_equal(actual,ref), (index,np.max(np.abs(actual-ref)))
    rows.append({'case':index,'shape':shape,'spec':str(spec),'max_error':0})
assert not _file_order_move_supported(P(None,('y','x'),None),(13,16,18),0,mesh)
assert not _file_order_move_supported(P('y',None,'x'),(12,16,18),0,mesh)
# Real transport uses >1MiB runs so this exercises the production optimization.
shape=(13,512,512);spec=P(None,'x','y');sh=NamedSharding(mesh,spec)
@jax.jit(out_shardings=sh)
def make_real():
    v=jnp.arange(np.prod(shape),dtype=jnp.float64).reshape(shape)
    return v+1j*(v*.25+2)
a=make_real()
path=root/'roundtrip.h5'
with SlabIO(path,mode='w',mesh=mesh) as io:
    io.create_dataset('V',shape=(15,511,510),dtype=np.complex128)
    io.write_slab('V',a,offset=(1,1,2),valid_shape=(11,507,505))
with SlabIO(path,mode='r',mesh=mesh) as io:
    b=io.read_slab('V',shape=shape,offset=(1,1,2),valid_shape=(11,507,505),partition_spec=spec)
actual=gather_to_host(b)
ref=np.arange(np.prod(shape),dtype=np.float64).reshape(shape)
ref=ref+1j*(ref*.25+2)
expect=np.zeros(shape,np.complex128);expect[:11,:507,:505]=ref[:11,:507,:505]
assert np.array_equal(actual,expect),np.max(np.abs(actual-expect))
rows.append({'roundtrip_max_error':0,'logical_write':(11,507,505)})
# A differently ordered compound source retains the collective route.
spec_f=P(None,('y','x'),None);sh_f=NamedSharding(mesh,spec_f)
@jax.jit(out_shardings=sh_f)
def make_fallback():
    v=jnp.arange(np.prod(shape),dtype=jnp.float64).reshape(shape)
    return v+1j*(v*.25+2)
af=make_fallback()
with SlabIO(root/'fallback.h5',mode='w',mesh=mesh) as io:
    io.create_dataset('V',shape=(15,511,510),dtype=np.complex128)
    io.write_slab('V',af,offset=(1,1,2),valid_shape=(11,507,505))
with SlabIO(root/'fallback.h5',mode='r',mesh=mesh) as io:
    bf=io.read_slab('V',shape=shape,offset=(1,1,2),valid_shape=(11,507,505),partition_spec=spec_f)
assert np.array_equal(gather_to_host(bf),expect)
rows.append({'native_compound_fallback_max_error':0})
# The Galerkin layout uses one combined rank axis and a ragged physical tail.
spec_c=P(None,None,('x','y'));sh_c=NamedSharding(mesh,spec_c)
@jax.jit(out_shardings=sh_c)
def make_combined():
    v=jnp.arange(np.prod(shape),dtype=jnp.float64).reshape(shape)
    return v+1j*(v*.25+2)
ac=make_combined()
with SlabIO(root/'combined.h5',mode='w',mesh=mesh) as io:
    io.create_dataset('C',shape=(11,507,505),dtype=np.complex128)
    io.write_slab('C',ac,valid_shape=(11,507,505))
with SlabIO(root/'combined.h5',mode='r',mesh=mesh) as io:
    bc=io.read_slab('C',shape=shape,valid_shape=(11,507,505),partition_spec=spec_c)
assert np.array_equal(gather_to_host(bc),expect)
combined_plan=_file_order_plan(shape,(11,507,505),_spec_axes(spec_c,3),
                               16,4,tuple(mesh.axis_names),(11,507,505))
assert combined_plan not in (None,'as-is'),combined_plan
rows.append({'native_combined_rank_roundtrip_max_error':0})
# Exact Fe184 physical-basis carrier: output is a bounded 40-k-row piece.
cshape=(8000,184,2108);csizes=(40,184,2108)
cfn=_file_order_take(mesh,cshape,np.dtype('complex128'),0,csizes,40,spec_c)
cabs=jax.ShapeDtypeStruct(cshape,jnp.complex128,sharding=sh_c)
cstart=jax.ShapeDtypeStruct((3,),jnp.int32,sharding=NamedSharding(mesh,P()))
cexe=cfn.lower(cabs,cstart).compile();cma=cexe.memory_analysis();chlo=cexe.as_text()
assert 'all-gather' not in chlo
assert cma.temp_size_in_bytes < 256*2**20,cma
rows.append({'actual_C_shape':cshape,'argument_bytes':cma.argument_size_in_bytes,
             'output_bytes':cma.output_size_in_bytes,'temp_bytes':cma.temp_size_in_bytes,
             'all_gather':False})
# Abstract compilation allocates no production-sized tensor.
large=(8001,1800,1800);sizes=(4,1800,1800)
fn=_file_order_take(mesh,large,np.dtype('complex128'),0,sizes,4,spec)
abstract=jax.ShapeDtypeStruct(large,jnp.complex128,sharding=sh)
start_abs=jax.ShapeDtypeStruct((3,),jnp.int32,sharding=NamedSharding(mesh,P()))
exe=fn.lower(abstract,start_abs).compile();m=exe.memory_analysis()
hlo=exe.as_text()
assert 'all-gather' not in hlo
assert m.temp_size_in_bytes < 256*2**20,m
rows.append({'large_shape':large,'argument_bytes':m.argument_size_in_bytes,'output_bytes':m.output_size_in_bytes,'temp_bytes':m.temp_size_in_bytes,'all_gather':False})
# The same scratch instrument must reject the old global-slice route.
control_shape=(513,512,512);control_sizes=(4,512,512)
def old(a,s):return jax.lax.dynamic_slice(a,[s[i] for i in range(3)],control_sizes)
old_fn=jax.jit(old,out_shardings=NamedSharding(mesh,P(('x','y'),None,None)))
c=old_fn.lower(jax.ShapeDtypeStruct(control_shape,jnp.complex128,sharding=sh),start_abs).compile()
cm=c.memory_analysis();ch=c.as_text()
assert 'all-gather' in ch and cm.temp_size_in_bytes>256*2**20,cm
rows.append({'negative_control_temp_bytes':cm.temp_size_in_bytes,'negative_control_all_gather':True})
if jax.process_index()==0:
    (root/'result.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('SLABIO BOUNDED STAGING PASS',json.dumps(rows),flush=True)
