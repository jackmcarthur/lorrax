"""Native P4 full coupled Dirac parity and bounded component/pole/q seams."""
from pathlib import Path
from types import SimpleNamespace as NS
import argparse,json,time
from runtime import initialize_communicator_stack
rt=initialize_communicator_stack(platform='gpu')
import jax,jax.numpy as jnp,numpy as np
from jax.sharding import NamedSharding,PartitionSpec as P
from common.collectives import device_put_process_local,gather_to_host
from common.contract_bands import contract_bands_block_reshard
from common.gamma_matrices import gamma_projector_half,gamma_perm_phase_host
from gw.centroid_k_unfold import CentroidKUnfoldPlan
from gw.greens_function_kernel import build_G_parents,_weighted_tau_phases
from gw.cohsex_sigma import make_lorentz_convolution
from gw.mpa.sector_sigma import sector_tau_factory,_sector_component_kernel
from gw.mpa.sigma import _shared_pole_weights,_chunk_major
from gw.shared_pole_recipe import CapacityLedger
from distrib_la import panel_matmul
from functools import partial
from runtime.padding import padded_axis
ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out
out.mkdir(parents=True,exist_ok=True)
mesh=rt.mesh;assert mesh.size==4
rng=np.random.default_rng(9281)
kg=(20,20,20);nk=int(np.prod(kg));q=4;m=6;n=10;nb=6
row=(np.arange(nk)%q).astype(np.int32);sym=((np.arange(nk)//q)%2).astype(np.int32)
# Stored parents are their own identity rows; every other star row exercises TR.
sym[:q]=0
kparent=rng.uniform(-.4,.4,size=(q,3));u=np.zeros((nk,2,2),complex)
theta=.31;u[:,0,0]=u[:,1,1]=np.cos(theta);u[:,0,1]=np.sin(theta);u[:,1,0]=-np.sin(theta)
p=np.where(np.arange(nk)%3==0,-1.,1.);spin=np.zeros((nk,4,4),complex)
spin[:,:2,:2]=u;spin[:,2:,2:]=p[:,None,None]*u
plans=[]
for extent in (m,n):
    perm=np.stack([np.arange(extent),np.concatenate([np.arange(extent//2)[::-1],np.arange(extent//2,extent)[::-1]])]).astype(np.int32)
    wraps=rng.integers(-1,2,size=(2,extent,3))
    plans.append(CentroidKUnfoldPlan(mesh,NS(axis=NS(n_logical=extent,n_padded=extent)),
        row,sym,perm,wraps,kparent,spin,1,4,parent_full_rows=np.arange(q,dtype=np.int32)))

def random(shape):return (rng.normal(size=shape)+1j*rng.normal(size=shape))/np.sqrt(np.prod(shape[-2:]))
def put(x,spec):return device_put_process_local(np.ascontiguousarray(x),NamedSharding(mesh,spec))
# Distinct endpoint families, complex phases, nontrivial alpha_y.
xh=random((q,4,m,nb));yh=random((q,nb,4,n));rh=random((q,nb,4,m));zh=random((q,4,n,nb))
x,y,r,z=put(xh,P(None,None,'x','y')),put(yh,P(None,'x',None,'y')),put(rh,P(None,'x',None,'y')),put(zh,P(None,None,'x','y'))
energy=put(rng.uniform(-1,1,(q,nb)),P());weight=put(rng.uniform(0,1,(q,nb)),P())
wh=random((nk,m,4,n,4));w=put(wh,P(None,'x',None,'y',None))
class Carrier:
    def __init__(self,plan,mun,nmu,proj_mun,proj_nmu):
        self.plan,self.psi_mun,self.psi_nmu=plan,mun,nmu
        self._projection=(proj_nmu,proj_mun);self.enk=energy;self.occ=weight;self.layout='face'
    def projection_faces(self):return self._projection
left=NS(green_parent=Carrier(plans[0],x,r,x,r))
right=NS(green_parent=Carrier(plans[1],z,y,z,y))
meta=NS(kgrid=kg,nk_tot=nk,nspinor=4,n_rmu=m)
meta.shared_pole_capacity=CapacityLedger(meta,mesh_xy=mesh,device_budget_bytes=75_000_000_000)
meta.shared_pole_capacity.live_stages=()
axis=padded_axis(nb,mesh,name='testbands',specs=((P(None,'x','y'),1),(P(None,'x','y'),2)))
class Interactions:
    def __init__(self,w,lefts,rights):self.w,self.lefts,self.rights=w,lefts,rights
    def component(self,a,b):
        A=jnp.asarray(self.lefts)[a];B=jnp.asarray(self.rights)[b]
        return jnp.take(jnp.take(self.w,A,axis=2),B,axis=3)
class DummySynthesis:
    native=0
results=[]
for ls,rs in [((0,),(0,)),((1,2,3),(1,2,3)),((0,),(1,2,3)),((1,2,3),(0,))]:
    keys=tuple((A,B) for A in ls for B in rs)
    tau=sector_tau_factory(left,right,keys,meta,mesh)(DummySynthesis(),axis)
    @jax.jit
    def bounded(x,y,r,z,energy,weight,w):
        return tau._spatial(x,y,r,z,energy,weight,.17,.23-.11j,Interactions(w,ls,rs))
    conv=make_lorentz_convolution(mesh,kg,nk,keys,*plans)
    project=contract_bands_block_reshard(mesh,layout='face',face_shape=(q,nb,m,4),
        right_face_shape=(q,nb,n,4),face_band_extent=nb)
    @jax.jit
    def full(x,y,r,z,energy,weight,w):
        phases=_weighted_tau_phases(energy,1j*(.23-.11j),e_ref=.17,band_weight=weight)
        green=build_G_parents(x,y,phases=phases,layout='face',
            gemm=partial(panel_matmul,mesh=mesh,panel_bytes=1<<20),k_unfold_plan=plans[0])
        selected=jnp.take(jnp.take(w,jnp.asarray(ls),axis=2),jnp.asarray(rs),axis=4)
        return project(r,conv(green,selected),z)
    t=time.perf_counter();got=bounded(x,y,r,z,energy,weight,w);got.block_until_ready();elapsed=time.perf_counter()-t
    expected=full(x,y,r,z,energy,weight,w);expected.block_until_ready()
    gh,eh=map(gather_to_host,(got,expected));norm=max(np.max(np.abs(eh)),1e-30)
    error=float(np.max(np.abs(gh-eh))/norm);assert error<2e-11,(keys,error)
    # Independent gamma contraction identity on arbitrary full spin matrices.
    small=random((4,4));lp=random((3,4));rp=random((4,3));direct=np.zeros((3,3),complex);split=direct.copy()
    for A,B in keys:
        pa,pha=gamma_perm_phase_host(A);pb,phb=gamma_perm_phase_host(B)
        ga=pha[:,None]*np.eye(4)[pa];gb=phb[:,None]*np.eye(4)[pb]
        direct+=np.conj(lp)@ga@small@gb.conj().T@rp
        for h in range(2):
            for g in range(2):
                la=np.argsort(pa)[2*h:2*h+2];rb=np.argsort(pb)[2*g:2*g+2]
                l=lp[:,la]*np.conj(pha[la])[None];r_=rp[rb]*np.conj(phb[rb])[:,None]
                split+=np.conj(l)@small[2*h:2*h+2,2*g:2*g+2]@r_
    independent=float(np.max(np.abs(direct-split)));assert independent<1e-12
    mem=bounded.lower(x,y,r,z,energy,weight,w).compile().memory_analysis()
    results.append(dict(keys=keys,relative_error=error,independent_gamma_max_abs=independent,
        cold_seconds=elapsed,temp_bytes=mem.temp_size_in_bytes))
    if jax.process_index()==0:print('SECTOR_QUARTER_PASS',results[-1],flush=True)
# q/pole panel seams, mixed widths and ordered hole phase.
qg=(2,2,2);Q=8;qp=3;M=6;N=10;nc=3;nt=1;K=6;width=2
parent=np.arange(Q,dtype=np.int32)%qp
bxh=random((qp,M,nc,K));byh=random((qp,N,nt,K));ph=np.linspace(.3,3,qp*K).reshape(qp,K)
x0,y0,po=put(bxh,P(None,'x',None,'y')),put(byh,P(None,'y',None,'x')),put(ph,P())
xc=_chunk_major(mesh,P(None,'x',None,'y'),3,width)(x0);yc=_chunk_major(mesh,P(None,'y',None,'x'),3,width)(y0);pc=_chunk_major(mesh,P(),3,width)(po)
# Native shardwise row gathers, preserving the original face specs.
panels=[]
for lo,hi in [(0,2),(2,3)]:
    rows_=np.flatnonzero((parent>=lo)&(parent<hi));ids=parent[rows_]-lo
    routes=tuple(jax.jit(lambda f,ids=ids:jnp.take(f,jnp.asarray(ids),axis=0),
        out_shardings=NamedSharding(mesh,spec)) for spec in (P(None,'x',None,'y'),P(None,'y',None,'x')))
    panels.append(dict(span=(lo,hi),rows=rows_,parent=ids,routes=routes))
ker,native=_sector_component_kernel(panels,mesh,qg,Q,M,N,width,_shared_pole_weights)
interval=np.array([[1,5],[0,3],[3,6]],np.int32);intervald=put(interval,P())
from symmetry_maps import q_negation_index
minus=np.asarray(q_negation_index(qg))
for hole in (False,True):
    for A in range(nc):
        value=ker(xc,yc,pc,intervald,.19,.22-.13j,hole,A,0);value.block_until_ready();vh=gather_to_host(value)
        weights=gather_to_host(jax.jit(_shared_pole_weights)(po,intervald,.19,.22-.13j))
        xx=bxh[parent,:,A,:];yy=byh[parent,:,0,:];ww=weights[parent]
        if hole:xx,yy,ww=np.conj(xx[minus]),np.conj(yy[minus]),ww[minus]
        reference=np.einsum('qmk,qk,qnk->qmn',xx,ww,np.conj(yy))
        error=float(np.max(np.abs(vh-reference))/max(np.max(np.abs(reference)),1e-30));assert error<2e-12,error
        results.append(dict(panel_hole=hole,component=A,relative_error=error))
# Named quarter guards, including mismatched endpoint actions.
for h,g,right in [(2,0,None),(0,-1,None)]:
    try:plans[0].dirac_quarter_load_tables(h,g,right);raise AssertionError('quarter accepted')
    except ValueError as e:assert 'GATE dirac_quarter' in str(e)
if jax.process_index()==0:(out/'result.json').write_text(json.dumps(results,indent=2)+'\n');print('SECTOR_STREAM_PASS',flush=True)
