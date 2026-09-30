"""Native compact-row read parity, donation and actual Ni230 dimensions."""
from pathlib import Path
import argparse
import json
import resource
import time
from runtime import initialize_communicator_stack
rt = initialize_communicator_stack(platform='gpu')
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import gather_to_host, device_put_process_local
from file_io.slab_io import SlabIO
from file_io._slab_io_ffi import _file_order_read_insert, _file_order_read_plan
mesh = rt.mesh
ap = argparse.ArgumentParser()
ap.add_argument('--out', type=Path, required=True)
ap.add_argument('--native-ni', action='store_true')
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
rows=[]
for shape, valid, spec, k, compact, carrier, real, start in [
    ((17,18,20),(17,17,19),P(None,'x','y'),0,(8,17,19),(8,18,20),7,(9,0,0)),
    ((3,17,18,20),(3,17,17,19),P(None,None,'y','x'),1,(2,8,17,19),(2,8,18,20),7,(1,9,0,0)),
    ((17,18,20),(17,17,19),P(None,None,('x','y')),0,(8,17,19),(8,18,20),7,(9,0,0)),
    ((17,20,18),(17,19,17),P(None,('x','y'),None),0,(8,19,17),(8,20,18),7,(9,0,0)),
]:
    sh=NamedSharding(mesh,spec)
    fs=NamedSharding(mesh,P(*([None]*k),tuple(mesh.axis_names),*([None]*(len(shape)-k-1))))
    @jax.jit(out_shardings=sh)
    def zeros(): return jnp.zeros(shape,jnp.complex128)
    @jax.jit(out_shardings=fs)
    def make():
        v=jnp.arange(np.prod(compact),dtype=jnp.float64).reshape(compact)
        return v+1j*(v*.125+1)
    dest=zeros(); part=make()
    ctl=device_put_process_local(np.asarray(start,np.int32),NamedSharding(mesh,P()))
    fn=_file_order_read_insert(mesh,shape,jnp.dtype('complex128'),spec,k,compact,carrier,real)
    actual=gather_to_host(fn(dest,part,ctl))
    ref=np.arange(np.prod(compact),dtype=np.float64).reshape(compact)
    ref=ref+1j*(ref*.125+1)
    ref=np.pad(ref,[(0,y-x) for x,y in zip(compact,carrier)])
    ref=np.take(ref,range(real),axis=k)
    expect=np.zeros(shape,np.complex128)
    expect[tuple(slice(s,s+n) for s,n in zip(start,ref.shape))]=ref
    assert np.array_equal(actual,expect)
    exe=fn.lower(jax.ShapeDtypeStruct(shape,jnp.complex128,sharding=sh),
        jax.ShapeDtypeStruct(compact,jnp.complex128,sharding=fs),
        jax.ShapeDtypeStruct((len(shape),),jnp.int32,sharding=NamedSharding(mesh,P()))).compile()
    ma=exe.memory_analysis()
    assert ma.alias_size_in_bytes==np.prod(shape)*16//4,ma
    assert 'all-gather' not in exe.as_text()
    rows.append({'shape':shape,'spec':str(spec),'bitwise':True,'alias_bytes':ma.alias_size_in_bytes})
# Replicated and noncanonical compound layouts must retain the original reader.
assert _file_order_read_plan((17,512,512),(17,511,511),P(None,'x',None),16,mesh,(17,511,511)) is None
assert _file_order_read_plan((17,512,512),(17,511,511),P(None,('y','x'),None),16,mesh,(17,511,511)) is None
# Exact production volume, no allocation at AOT: output buffer must alias.
shape=(8000,3,232,232); valid=(8000,3,230,230); spec=P(None,None,'x','y')
plan=_file_order_read_plan(shape,valid,spec,16,mesh,valid)
assert plan is not None
k,height,lead=plan
compact=(height,3,230,230);carrier=(height,3,232,232)
fn=_file_order_read_insert(mesh,shape,jnp.dtype('complex128'),spec,k,compact,carrier,height)
sh=NamedSharding(mesh,spec);fs=NamedSharding(mesh,P(tuple(mesh.axis_names),None,None,None))
exe=fn.lower(jax.ShapeDtypeStruct(shape,jnp.complex128,sharding=sh),
    jax.ShapeDtypeStruct(compact,jnp.complex128,sharding=fs),
    jax.ShapeDtypeStruct((4,),jnp.int32,sharding=NamedSharding(mesh,P()))).compile()
ma=exe.memory_analysis(); assert ma.alias_size_in_bytes==np.prod(shape)*16//4,ma
assert 'all-gather' not in exe.as_text()
assert ma.temp_size_in_bytes < 4*64*2**20,ma
rows.append({'native_ni_shape':shape,'logical':valid,'plan':plan,'alias_bytes':ma.alias_size_in_bytes,'temp_bytes':ma.temp_size_in_bytes,'output_bytes':ma.output_size_in_bytes,'argument_bytes':ma.argument_size_in_bytes,'all_gather':False})
if a.native_ni:
    @jax.jit(out_shardings=fs)
    def make_file():
        i=jax.lax.broadcasted_iota(jnp.float64,valid,0)
        c=jax.lax.broadcasted_iota(jnp.float64,valid,1)
        b=jax.lax.broadcasted_iota(jnp.float64,valid,2)
        d=jax.lax.broadcasted_iota(jnp.float64,valid,3)
        v=i*1e6+c*1e5+b*500+d
        return v+1j*(v*.125+1)
    path=a.out/'ni_dimension_fixture.h5'
    value=make_file();value.block_until_ready();t=time.monotonic()
    with SlabIO(path,mode='w',mesh=mesh) as io:
        io.create_dataset('links',shape=valid,dtype=np.complex128)
        io.write_slab('links',value)
    write_s=time.monotonic()-t;del value
    t=time.monotonic()
    with SlabIO(path,mode='r',mesh=mesh) as io:
        read=io.read_slab('links',shape=shape,partition_spec=spec)
        read.block_until_ready()
    read_s=time.monotonic()-t
    @jax.jit
    def check(v):
        i=jax.lax.broadcasted_iota(jnp.float64,shape,0)
        c=jax.lax.broadcasted_iota(jnp.float64,shape,1)
        b=jax.lax.broadcasted_iota(jnp.float64,shape,2)
        d=jax.lax.broadcasted_iota(jnp.float64,shape,3)
        ref=i*1e6+c*1e5+b*500+d
        ref=jnp.where((b<230)&(d<230),ref+1j*(ref*.125+1),0)
        return jnp.max(jnp.abs(v-ref))
    err=float(gather_to_host(check(read)));assert err==0,err
    rows.append({'native_ni_logical_bytes':int(np.prod(valid)*16),'write_seconds':write_s,'read_seconds':read_s,'bitwise':True,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
if jax.process_index()==0:
    (a.out/'result.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('FILE ROW READ PASS',json.dumps(rows),flush=True)
