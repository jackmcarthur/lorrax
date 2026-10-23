"""Nonphysical encoded coefficients: exact sweep/padding/memory/I/O probe."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import runtime
stack = runtime.initialize_communicator_stack()
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P
from common.collectives import barrier
from common.jax_compile_cache import compile_cache_stats
from common import mtxel_sweep as ms
from common.wfn_layout import band_sphere_spec

p=argparse.ArgumentParser()
p.add_argument("--bands",default="1027")
p.add_argument("--nk",type=int,default=1)
p.add_argument("--ns",type=int,default=1)
p.add_argument("--operator",choices=("kinetic","dipole","local"),default="dipole")
p.add_argument("--compile-only",action="store_true")
p.add_argument("--write",action="store_true")
p.add_argument("--loops",type=int,default=3)
p.add_argument("--tile-mib",type=float,default=0)
p.add_argument("--ng",type=int,default=67)
p.add_argument("--grid-edge",type=int,default=8)
p.add_argument("--out",required=True)
a=p.parse_args()
if a.tile_mib:
    import runtime.tiles
    runtime.tiles.TILE_BYTES=int(a.tile_mib*2**20)
rank=jax.process_index(); world=jax.process_count()
out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
stream=(out/f"rank{rank:04d}.jsonl").open("w",buffering=1)
grid=(a.grid_edge,)*3; ng=a.ng; valid=ng-6; n_grid=int(np.prod(grid))
gvec=np.column_stack(np.unravel_index(np.arange(ng),grid)).astype(np.int32)
gvec[valid:]=a.grid_edge-1
gv=np.broadcast_to(gvec,(a.nk,ng,3)).copy()
mask=np.broadcast_to((np.arange(ng)<valid).astype(float),(a.nk,ng)).copy()
box=np.broadcast_to(np.arange(ng,dtype=np.int32),(a.nk,ng)).copy()
box[:,valid:]=n_grid
kv=np.broadcast_to(np.array([.125,-.0625,.09375]),(a.nk,3)).copy()
bdot=np.diag([1.,1.25,.75])
gv_weight=(1+np.arange(ng)%7)/8
gv_weight[valid:]=0
spin=2.**(-np.arange(a.ns))
factor_g=(gv_weight**2).sum()*np.square(spin).sum()
if a.operator=="kinetic":
    kg=gv[0]+kv[0]
    scalar=np.sum(gv_weight**2*np.einsum("gi,ij,gj->g",kg,bdot,kg))*np.square(spin).sum()
elif a.operator=="dipole":
    scalar=2*np.einsum("g,ga->a",gv_weight**2,gv[0]+kv[0])*np.square(spin).sum()
else:
    x,y,z=np.indices(grid)
    potential=1.25+.125*np.sin(2*np.pi*x/8)+.25*np.cos(2*np.pi*y/8)
    basis=np.zeros(n_grid,complex); basis[:valid]=gv_weight[:valid]
    transformed=np.fft.fftn(potential*np.fft.ifftn(basis.reshape(grid),norm="ortho"),norm="ortho")
    scalar=np.vdot(basis,transformed.reshape(-1))*np.square(spin).sum()

original=ms._cached_jit
executables={}; memories={}
def capture(name,key,build):
    fn=original(name,key,build)
    if name!="sweep_matrix_elements": return fn
    def call(*args):
        if fn not in executables:
            executable=fn.lower(*args).compile()
            executables[fn]=executable
            mem=executable.memory_analysis()
            memories[fn]={key:int(getattr(mem,key)) for key in
                ("argument_size_in_bytes","output_size_in_bytes","temp_size_in_bytes","alias_size_in_bytes")}
        if a.compile_only: return (jnp.zeros((),jnp.complex128),)
        return executables[fn](*args)
    return call
ms._cached_jit=capture
for nb in map(int,a.bands.split(",")):
    geom=ms.SweepGeometry(mesh=stack.mesh,fft_grid=grid,ngkmax=ng,nb=nb,
                         ns=a.ns,nk=a.nk,cell_volume=512.)
    b=np.arange(geom.nb)
    amp=(1+b%13)/16+1j*(b%7)/32
    amp[b>=nb]=0
    shape=(a.nk,geom.nb,a.ns,ng)
    sharding=NamedSharding(stack.mesh,band_sphere_spec())
    def coefficients(index):
        return np.broadcast_to(amp[index[1]][None,:,None,None]*spin[index[2]][None,None,:,None]
            *gv_weight[index[3]][None,None,None,:],
            tuple(len(range(n)[i]) for n,i in zip(shape,index))).copy()
    psi=jax.make_array_from_callback(shape,sharding,coefficients)
    if a.operator=="kinetic": op=ms.kinetic_operator(geom,bdot)
    elif a.operator=="dipole": op=ms.dipole_operator(geom,bvec=np.eye(3),blat=1.)
    else: op=ms.local_potential_operator(geom,potential)
    plan=ms.plan_sweep(geom,op)
    old_sha=None
    for repeat in range(1 if a.compile_only else a.loops):
        barrier(f"mtxel-{nb}-{repeat}")
        before=compile_cache_stats(); start=time.perf_counter()
        result=ms.sweep_matrix_elements(psi,geom=geom,operator=op,gvecs=gv,
            gmask=mask,box_index=box,kvecs=kv)
        result.block_until_ready()
        elapsed=time.perf_counter()-start; after=compile_cache_stats()
        digest=hashlib.sha256(); error=0.
        if not a.compile_only:
            for shard in result.addressable_shards:
                actual=np.asarray(shard.data)
                r=np.arange(geom.nb)[shard.index[-2]]
                c=np.arange(geom.nb)[shard.index[-1]]
                expected=np.conj(amp[r,None])*amp[None,c]
                expected=(expected[None,None]*scalar[None,:,None,None]
                          if a.operator=="dipole" else expected[None]*scalar)
                expected=np.broadcast_to(expected,actual.shape)
                np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)
                error=max(error,float(np.max(np.abs(actual-expected))))
                digest.update(np.ascontiguousarray(actual))
            sha=digest.hexdigest()
            if old_sha is not None: assert sha==old_sha
            old_sha=sha
        row=dict(status="PASS",rank=rank,world=world,nb=nb,nb_pad=geom.nb,
            nk=a.nk,ns=a.ns,operator=a.operator,repeat=repeat,compile_only=a.compile_only,
            seconds=elapsed,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            memory=list(memories.values())[-1],plan=list(plan),max_abs_error=error,
            sha256=None if a.compile_only else old_sha,
            compile_delta={k:after[k]-before[k] for k in ("compiles","probes","hits")})
        stream.write(json.dumps(row)+"\n")
        if rank==0: print(json.dumps(row),flush=True)
        if a.write and repeat==a.loops-1:
            from file_io.dipole import write_dipole
            from file_io.kin_ion import write_kin_ion
            started=time.perf_counter()
            path=out/("dipole.h5" if a.operator=="dipole" else "kin_ion.h5")
            if a.operator=="dipole":
                write_dipole(path,result,np.zeros((a.nk,nb)),mesh=stack.mesh,
                    attrs={"fixture_kind":"nonphysical_encoded_scaling_probe"})
            else:
                write_kin_ion(path,result,mesh=stack.mesh,nb=nb,
                    star=(np.arange(a.nk),np.zeros(a.nk,dtype=int),1),
                    attrs={"fixture_kind":"nonphysical_encoded_scaling_probe"})
            stream.write(json.dumps(dict(status="PASS",rank=rank,kind="write",
                seconds=time.perf_counter()-started,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                path=str(path)))+"\n")
            # Read only small independent rows after collective close. Never
            # gather the full large matrix just to validate its file format.
            if rank==0:
                import h5py
                with h5py.File(path,"r") as h5:
                    ds=h5["dipole_cart" if a.operator=="dipole" else "kin_ion"]
                    ds_shape=ds.shape
                    assert ds_shape==((3,a.nk,nb,nb) if a.operator=="dipole" else (a.nk,nb,nb))
                    for r0 in (0,nb//2,nb-1):
                        ref=np.conj(amp[r0])*amp[:nb]
                        expected=(scalar[:,None]*ref[None] if a.operator=="dipole" else scalar*ref)
                        actual=ds[:,0,r0,:] if a.operator=="dipole" else ds[0,r0,:]
                        np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)
                stream.write(json.dumps(dict(status="PASS",rank=0,kind="file_rows",
                    shape=list(ds_shape),logical_bytes=3*a.nk*nb*nb*16 if a.operator=="dipole" else a.nk*nb*nb*16))+"\n")
        result.delete()
    psi.delete()
stream.close()
barrier("mtxel-probe-done")
runtime.finalize_process(0)
