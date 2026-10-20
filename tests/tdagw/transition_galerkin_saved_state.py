"""Verification-only P4 q0 transition-Galerkin saved-state diagnostic.

Never computes Sigma, writes a production bank, propagates time, or admits an
all-q model. Run only after review/push/explicit launch approval. The launcher
must use kill-on-bad-exit and an external wall cap <=600 seconds as well.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import traceback

import runtime


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source','snapshot','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--trial',type=int,default=0)
    p.add_argument('--b-lo',type=int,default=128)
    p.add_argument('--basis-capacity',type=int,default=128)
    p.add_argument('--seed-width',type=int,default=8)
    p.add_argument('--panel-width',type=int,default=32)
    p.add_argument('--gamma-ev',type=float,default=.05)
    p.add_argument('--eta-ev',type=float,default=.1)
    p.add_argument('--rank-tolerance',type=float,default=1e-8)
    p.add_argument('--fit-tolerance',type=float,default=.02)
    p.add_argument('--wall-seconds',type=float,default=600.)
    p.add_argument('--source-bound-bytes-per-rank',type=int,required=True)
    p.add_argument('--lu-bound-bytes-per-rank',type=int,required=True)
    p.add_argument('--device-bound-bytes-per-rank',type=int,required=True)
    p.add_argument('--host-bound-bytes-per-rank',type=int,default=4*1024**3)
    return p


def validate_args(a):
    import math
    if (not 1<=a.basis_capacity<=512 or not 1<=a.seed_width<=8
            or not 1<=a.panel_width<=32 or not 0<a.wall_seconds<=600
            or not 0<a.rank_tolerance<1 or not 0<a.fit_tolerance<=.02
            or not math.isfinite(a.gamma_ev+a.eta_ev) or a.gamma_ev<=0 or a.eta_ev<=0
            or abs(a.gamma_ev-.05)>1e-12 or abs(a.eta_ev-.1)>1e-12
            or a.trial<0 or a.b_lo!=128
            or min(a.source_bound_bytes_per_rank,a.lu_bound_bytes_per_rank,
                   a.device_bound_bytes_per_rank,a.host_bound_bytes_per_rank)<=0):
        raise ValueError('GATE transition_q0_cli: bounded P4 active128:132 policy required')
    if a.source_bound_bytes_per_rank+a.lu_bound_bytes_per_rank>=a.device_bound_bytes_per_rank:
        raise ValueError('GATE transition_capacity: source/LU declarations exhaust device bound')


def known_device_array_bytes(objects):
    """Public reachable-array accounting only; NOT a complete allocation census.

    Private provider/runtime allocations are outside this lower-bound scope
    and remain the explicit source-ceiling caller's responsibility.
    """
    import jax
    seen=set()
    def visit(value):
        if id(value) in seen:return 0
        seen.add(id(value))
        if isinstance(value,jax.Array):
            return sum(s.data.size*s.data.dtype.itemsize for s in value.addressable_shards)
        if isinstance(value,dict):return sum(visit(v) for v in value.values())
        if isinstance(value,(tuple,list)):return sum(visit(v) for v in value)
        if hasattr(value,'__dict__') and not callable(value):
            return sum(visit(v) for k,v in vars(value).items() if not k.startswith('_'))
        return 0
    return int(visit(objects))


def bind_device_budget(declared,actual):
    if min(int(declared),int(actual))<=0 or int(declared)>int(actual):
        raise ValueError('GATE transition_capacity: declared device bound exceeds runtime budget')
    return int(actual)


def tiny_collective_preflight(mesh,*,linalg='distributed'):
    """Actual-mesh protocol exercise before reading the costly material source.

    Only four-port toy arrays may reach the host here. No CPU fixture mesh,
    material port gather, fitted response, or production admission is involved.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from types import SimpleNamespace
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import gather_to_host,device_put_process_local
    from common.wfn_layout import psi_specs
    from gw import transition_galerkin as tg
    from gw.response_bank import response_algebra
    from gw.shared_pole_recipe import CapacityLedger
    from gw.shared_pole_execution import face_matmul
    jax.config.update('jax_enable_x64',True)
    rng=np.random.default_rng(20261020)
    nk,nb,nmu=4,4,4
    psi=.2*(rng.normal(size=(nk,nb,1,nmu))+1j*rng.normal(size=(nk,nb,1,nmu)))
    energy=np.broadcast_to(np.array([-.8,-.2,.3,.9]),(nk,nb)).copy()
    rho=np.empty((nk,nb,nb),complex)
    for k in range(nk):
        u=np.linalg.qr(rng.normal(size=(nb,nb))+1j*rng.normal(size=(nb,nb)))[0]
        rho[k]=u@np.diag([.9,.6,.3,.1])@u.conj().T
    partner=np.arange(nk);gamma=.04;z=.23+.37j
    cat=tg.transition_catalogue(energy,rho,partner,pair_tile=4)
    bridge=tg.make_transition_bridge(mesh,psi.shape,cat['b'].shape[1],
        normalization=tg.physical_transition_normalization(nk,1,1),pair_tile=4)
    psi_device=device_put_process_local(psi,NamedSharding(mesh,psi_specs('face')[0]))
    t=bridge['vertex'](psi_device,partner,cat['b'],cat['d'],cat['valid'])
    pi,_=tg.transition_polarization(bridge,t,cat,energy,rho,z,gamma=gamma,
        max_block_size=3,source_resident_bytes_per_rank=2**20)
    # Independent literal ordered transition residue: rho is LEFT of phase.
    literal=np.zeros((nmu,nmu),complex);eye=np.eye(nb)
    for k in range(nk):
        for b in range(nb):
            for d in range(nb):
                right=psi[k,b,0]*psi[k,d,0].conjugate()
                left=np.zeros(nmu,complex)
                for a in range(nb):
                    for c in range(nb):
                        left+=(eye[a,b]*rho[k,c,d].conjugate()-rho[k,a,b]*eye[c,d])*psi[k,a,0]*psi[k,c,0].conjugate()
                literal+=2/nk*np.outer(left,right.conjugate())/(z-energy[k,b]+energy[k,d]+1j*gamma)
    meta=SimpleNamespace(nk_tot=nk,nspin=1,nspinor_wfnfile=1,nspinor=1,n_rmu=nmu)
    meta.shared_pole_capacity=CapacityLedger(meta,mesh_xy=mesh,device_budget_bytes=2**30)
    meta.shared_pole_capacity.live_stages=()
    config=SimpleNamespace(backend=SimpleNamespace(linalg=linalg))
    dyson,_,_,algebra=response_algebra(meta,config,mesh_xy=mesh,n=nmu,ordered=True)
    rep=bridge['port_sharding'];face=NamedSharding(mesh,P(None,'x','y'))
    h=device_put_process_local(.2*np.eye(nmu,dtype=complex)[None],face)
    v=.04*np.eye(nmu,dtype=complex)
    vf=device_put_process_local(v,NamedSharding(mesh,P('x','y')))
    wc=tg.transition_dyson_value(pi,h,dyson,raw_prefactor=algebra['prefactor'])
    w=np.linalg.solve(np.eye(nmu)-v@literal,v)
    ur=device_put_process_local(np.eye(nmu,dtype=complex)[:,:2],rep)
    ul=device_put_process_local(np.eye(nmu,dtype=complex)[:,2:],rep)
    mm=face_matmul(mesh)
    narrow=jax.jit(lambda a:a[0],out_shardings=rep)
    full=vf[None]+wc
    x,y=tg.dyson_transition_seeds(bridge,t,cat,energy,rho,z,
        narrow(mm(full,jax.device_put(ur[None],face))),
        narrow(mm(full,jax.device_put(ul[None],face),transa='C')),gamma)
    q,basis=tg.transition_basis(bridge,[x,y],cat,rho,capacity=16,
        rank_tolerance=1e-10,max_block_size=2,self_crossing=tg.transition_crossing_map(cat,cat))
    model=tg.transition_parent_model(bridge,t,cat,energy,rho,vf,q,gamma,max_block_size=2)
    denominator=device_put_process_local(np.asarray(z-model['poles_ry'])[None,None],rep)
    reduced=mm(model['left']/denominator,model['right'],transb='C')
    pi_host=np.asarray(gather_to_host(pi))[0]
    wc_host=np.asarray(gather_to_host(wc))[0]
    reduced_host=np.asarray(gather_to_host(reduced))[0]
    error=dict(Pi=float(np.linalg.norm(pi_host-literal)),
        checked_Dyson=float(np.linalg.norm(wc_host-(w-v))),
        right_interpolation=float(np.linalg.norm((reduced_host+v-w)[:,:2])),
        dual_interpolation=float(np.linalg.norm((reduced_host+v-w)[2:,:])))
    if not all(np.isfinite(x) and x<1e-8 for x in error.values()):
        raise ValueError('GATE transition_real_P4_preflight: '+str(error))
    return dict(status='PASS_TINY_ONLY',errors_absolute=error,shape=[nk,nb,nmu],
        numerical_order=basis['order'],dense_backend=linalg,material_admission=False)


def main():
    args=parser().parse_args();validate_args(args)
    if not all(os.environ.get(k) for k in ('LEG_HEAD','LEG_SOURCE_TREE','LEG_SOURCE_ARCHIVE_SHA256')):
        raise ValueError('GATE transition_q0_provenance: landing must pin commit/tree/archive before launch')
    root=Path(args.out).resolve();started=time.monotonic()
    # Fatal timeout does no MPI work: a killed rank must terminate its step.
    # Each phase is committed before entering native work, so incomplete work
    # never becomes a completion/admission receipt even on timeout.
    def timeout():
        try:
            if root.is_dir():
                (root/f'timeout_pid{os.getpid()}.json').write_text(json.dumps(dict(
                    status='NOT_ADMITTED_TIMEOUT',sigma_computed=False,physical_time_fs=0,
                    wall_cap_seconds=args.wall_seconds)))
        finally:
            os._exit(124)
    watchdog=threading.Timer(args.wall_seconds,timeout);watchdog.daemon=True;watchdog.start()
    stack=runtime.initialize_communicator_stack()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import gather_to_host,all_gather_processes,device_put_process_local
    from common.units import RYD_TO_EV
    from file_io.commit_state import agree_io_refusal
    from gw import transition_galerkin as tg
    from gw.response_bank import (response_algebra,_coulomb_algebra,dense_layout,
        dielectric_physical_metrics,dielectric_inverse_bounds)
    from gw.shared_pole_execution import face_matmul
    from gw.wavefunction_bundle import rotate_wavefunctions
    from tdagw import dynsig,snapshot_runtime as sr
    from distrib_la import plan_subspace,plan_orthogonalization,plan
    from distrib_la.workspace import lu_workspace_bytes,workspace_bytes_per_rank
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_constructor import general_pencil_workspace_bytes
    rank0=jax.process_index()==0;say=print if rank0 else lambda *a,**k:None
    if jax.process_count()!=4 or jax.device_count()!=4 or jax.local_device_count()!=1:
        raise ValueError('GATE transition_q0_cli: exactly four ranks/one GPU each required')
    error=None
    if rank0:
        try:
            if root.exists():raise FileExistsError(str(root))
            root.mkdir(parents=True)
        except OSError as exc:error=exc
    agree_io_refusal(error,path=root,stage='transition_q0/output')
    report=dict(status='NOT_ADMITTED_RUNNING',q=0,admitted=False,sigma_computed=False,
        physical_time_propagated_fs=0,production_bank_written=False,adaptive_tuning=False,
        source=str(Path(args.source).resolve()),snapshot=str(Path(args.snapshot).resolve()),
        source_input_sha256=hashlib.sha256(Path(args.source).read_bytes()).hexdigest(),
        snapshot_sha256=hashlib.sha256(Path(args.snapshot).read_bytes()).hexdigest(),
        commit=os.environ.get('LEG_HEAD'),source_tree=os.environ.get('LEG_SOURCE_TREE'),
        archive_sha256=os.environ.get('LEG_SOURCE_ARCHIVE_SHA256'),walls_s={},args=vars(args))
    def save():
        error=None
        if rank0:
            try:(root/'receipt.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
            except OSError as exc:error=exc
        agree_io_refusal(error,path=root/'receipt.json',stage='transition_q0/receipt')
    def phase(name,fn):
        elapsed=np.asarray(all_gather_processes(np.asarray(time.monotonic()-started))).reshape(-1)
        if elapsed.max()>=args.wall_seconds:raise TimeoutError('transition proof total wall cap')
        report['phase']=name;save();before=time.monotonic()
        try:
            result=fn();jax.block_until_ready(result);return result
        finally:
            report['walls_s'][name]=time.monotonic()-before;save()
            say(f'[transition q0 NOT ADMITTED] {name}: {report["walls_s"][name]:.3f}s',flush=True)
    rc=0
    try:
        report['real_collective_preflight']=phase('tiny_real_P4_preflight',lambda:tiny_collective_preflight(stack.mesh))
        report['numerical_proof_scope']=dict(basis_capacity=args.basis_capacity,
            seed_width=args.seed_width,underconverged=True,full_spatial_ports=True,
            production_shared_pole_recipe=False,all_q_admission=False)
        with np.load(args.snapshot) as f:
            U,e,rhoa=(np.asarray(f[k][args.trial]) for k in ('U','E_ry','rho_in_basis'))
            report['saved_time_fs']=float(f['time_fs'])
        if U.shape!=(36,4,4) or e.shape!=(36,4) or rhoa.shape!=U.shape:
            raise ValueError('GATE transition_q0_snapshot: expected36k active2v2c tables')
        if (not all(np.isfinite(a).all() for a in (U,e,rhoa))
                or np.max(abs(U.conj().swapaxes(-1,-2)@U-np.eye(4)))>1e-7
                or np.max(abs(rhoa-rhoa.conj().swapaxes(-1,-2)))>1e-8):
            raise ValueError('GATE transition_q0_snapshot: invalid current basis/density')
        source=phase('source',lambda:dynsig.gw_window_source(args.source,stack.mesh,
            workdir=str(root/'driver'),screening=True,say=say))
        prepared=phase('prepare',lambda:sr.prepare_native_source(source,stack.mesh))
        meta=source['meta'];mask=np.asarray(meta.mu_basis.active_mask,bool)
        ledger=getattr(meta,'shared_pole_capacity',None)
        runtime_budget=getattr(ledger,'device_budget_bytes_per_rank',None)
        budget_origin='source.shared_pole_capacity'
        if runtime_budget is None:
            from common.gpu_utils import device_budget_bytes
            runtime_budget=device_budget_bytes();budget_origin='common.gpu_utils.device_budget_bytes'
        runtime_budget=bind_device_budget(args.device_bound_bytes_per_rank,runtime_budget)
        report['runtime_device_budget']=dict(bytes_per_rank=runtime_budget,origin=budget_origin,
            declared_bytes_per_rank=args.device_bound_bytes_per_rank)
        if (prepared['nk']!=36 or prepared['nb']!=160 or prepared['nb_chi']!=160
                or prepared['nb_sigma']!=160 or tuple(prepared['kgrid'])!=(6,6,1)
                or mask.sum()!=978 or len(mask)!=980):
            raise ValueError('GATE transition_q0_shape: only authenticated36/160/978logical/980packed source supported')
        if abs(float(source['config'].sigma.regularization_ev)-args.eta_ev)>1e-12:
            raise ValueError('GATE transition_q0_eta: source and declared collision eta differ')
        energy=prepared['energy'].copy();active=slice(128,132)
        energy[:,:128]+=(e[:,0]-energy[:,128])[:,None]
        energy[:,132:]+=(e[:,-1]-energy[:,131])[:,None];energy[:,active]=e
        rho=np.eye(160,dtype=complex)[None]*prepared['occupations'][:,:,None]
        rho[:,active,active]=rhoa
        current=phase('rotate_current',lambda:rotate_wavefunctions(source['wfns'],jnp.asarray(U),
            enk_active_new=jnp.asarray(e),enk_base=jnp.asarray(energy),efermi=None,
            mesh_xy=stack.mesh,active_slice=active,occupations=prepared['occupations']))
        psi=current.green_parent.projection_faces()[0]
        if psi.shape!=(36,160,int(meta.nspinor_wfnfile),980):
            raise ValueError('GATE transition_q0_shape: unexpected current native orbital carrier')
        known_source=known_device_array_bytes((source,prepared,current))
        # The face-to-whole-k conversion may transiently replicate source psi
        # and its k-q partner. These are orbital panels, never mu-squared data.
        known_source+=int(2*psi.size*psi.dtype.itemsize)
        if known_source>args.source_bound_bytes_per_rank:
            raise ValueError('GATE transition_capacity: explicit source bound below known arrays/redistribution')
        cat=tg.transition_catalogue(energy,rho,np.arange(36),coherent_slice=active,pair_tile=32)
        if cat['valid'].size>304128 or cat['receipt']['live_transitions']>303552:
            raise ValueError('GATE transition_q0_catalogue: unpriced transition support')
        bridge=tg.make_transition_bridge(stack.mesh,psi.shape,cat['b'].shape[1],
            normalization=tg.physical_transition_normalization(36,int(meta.nspin),int(meta.nspinor_wfnfile)),
            layout=current.green_parent.layout,pair_tile=32)
        qr=plan_subspace(capacity=args.basis_capacity,n_eig=1,max_block_size=args.seed_width,
                        vector_sharding=bridge['state_sharding'])
        cgs=plan_orthogonalization(capacity=args.basis_capacity,max_block_size=args.seed_width,
                                 vector_sharding=bridge['state_sharding'])
        provider=sum(np.prod(s.shape)*np.dtype(s.dtype).itemsize for p in (qr,cgs) for s in p.workspace_specs.values())
        nmu=len(mask);p=4;nt=cat['valid'].size;cap=args.basis_capacity;width=args.seed_width
        panel=tg.transition_panel_capacity(cat,nmu,retained_basis_rows=cap,processes=p,
            max_block_size=args.panel_width,source_resident_bytes_per_rank=args.source_bound_bytes_per_rank)
        # Conservative sum rather than optimistic lifetime overlap. Input
        # seeds (two sites*right+dual), both basis buffers, small modal arrays,
        # full exact sample faces and redistribution/provider scratch are bound.
        extras=int(2*16*nt*cap/p+16*nt*(4*width+8*width)/p+provider
            +32*16*nmu*nmu/p+16*12*cap*cap+32*nmu*cap)
        vendor_lu=int(lu_workspace_bytes(stack.mesh,nmu,np.complex128))
        layout=dense_layout(meta,source['config'],stack.mesh,nmu)
        resolution=linalg_resolution({'linalg':layout})
        if resolution.layout!='distributed':
            raise ValueError('GATE transition_q0_layout: only explicitly priced distributed root/LU route supported')
        root_plan=plan('eigh',stack.mesh,backend='distributed',n=nmu,batched_route=resolution.batched_route)
        root_workspace=int(workspace_bytes_per_rank(root_plan,'eigh',[(1,nmu,nmu)],np.complex128))
        extras+=root_workspace
        host_modal=int(general_pencil_workspace_bytes((cap,cap)))
        if vendor_lu>args.lu_bound_bytes_per_rank:
            raise ValueError('GATE transition_capacity: declared LU bound below public owner shape price')
        if host_modal>args.host_bound_bytes_per_rank:
            raise ValueError('GATE transition_capacity: reduced modal price exceeds explicit host bound')
        price=int(panel['working_bytes_per_rank']+extras+args.lu_bound_bytes_per_rank)
        report['price']=dict(panel=panel,provider_bytes_per_rank=int(provider),
            extra_conservative_bytes_per_rank=extras,vendor_LU_shape_bytes_per_rank=vendor_lu,
            known_source_arrays_and_redistribution_bytes_per_rank=known_source,
            source_array_census_scope='incomplete public-array lower bound; private runtime/provider allocations excluded',
            coulomb_root_workspace_bytes_per_rank=root_workspace,
            reduced_modal_host_bytes_per_rank=host_modal,
            total_declared_bytes_per_rank=price,source_bound_is_explicit=True,
            all_parent_retention_admitted=False,source_redistribution_included_in_source_bound=True)
        report['catalogue']=cat['receipt'];report['microscopic_space']=dict(
            n_k=36,n_q_requested=1,q=0,full_q_count=36,nmu_logical=978,nmu_packed=980,
            bands=160,active=[128,132],spatial_projection=False)
        save()
        if price>args.device_bound_bytes_per_rank:
            raise ValueError('GATE transition_capacity: complete conservative proof price exceeds declared device bound')
        T=phase('vertex',lambda:bridge['vertex'](psi,cat['kmq'],cat['b'],cat['d'],cat['valid']))
        V=prepared['V_q'][0];mm=face_matmul(stack.mesh)
        roots=phase('coulomb_root',lambda:_coulomb_algebra(stack.mesh,nmu,978,
            dense_layout(meta,source['config'],stack.mesh,nmu))(V[None]))
        H,_,negative,_=roots
        if bool(jnp.any(negative)):raise ValueError('GATE transition_q0_coulomb: negative bare spectrum')
        dyson,_,_,algebra=response_algebra(meta,source['config'],mesh_xy=stack.mesh,n=nmu,ordered=True)
        pref=float(algebra['prefactor']);face=NamedSharding(stack.mesh,P(None,'x','y'))
        rep=bridge['port_sharding'];gamma=args.gamma_ev/RYD_TO_EV;eta=args.eta_ev/RYD_TO_EV
        # Fixed, declared sites; no adaptive support/rank/pole search.
        sites=np.array([.35+.15j,1.10+.20j,.65+1j*args.eta_ev,1.40+.25j])/RYD_TO_EV
        report['sites']=[dict(role='train' if i<2 else 'held',z_ry=[z.real,z.imag]) for i,z in enumerate(sites)]
        @jax.jit
        def direction_panel():
            rows=jnp.cumsum(jnp.asarray(mask))-1
            column_carrier=((width+int(stack.mesh.shape['y'])-1)//int(stack.mesh.shape['y']))*int(stack.mesh.shape['y'])
            columns=jnp.arange(column_carrier)
            right=jnp.exp(2j*jnp.pi*rows[:,None]*columns[None]/978)/jnp.sqrt(978.)
            left=jnp.exp(2j*jnp.pi*rows[:,None]*(columns[None]+width)/978)/jnp.sqrt(978.)
            return right*jnp.asarray(mask)[:,None],left*jnp.asarray(mask)[:,None]
        ur,ul=jax.jit(direction_panel,out_shardings=(rep,rep))()
        narrow=jax.jit(lambda a:a[0,:,:width],out_shardings=rep)
        metric=jax.jit(lambda a,b:dielectric_physical_metrics(a,jnp.asarray(mask),matmul=mm,inverse=b),
                       in_shardings=(face,face),out_shardings=rep)
        exact=[];seeds=[];records=[]
        for i,z in enumerate(sites):
            Pi,note=phase(f'Pi_{i}',lambda z=z:tg.transition_polarization(bridge,T,cat,energy,rho,z,
                gamma=gamma,max_block_size=args.panel_width,retained_basis_rows=cap,
                source_resident_bytes_per_rank=args.source_bound_bytes_per_rank,
                max_device_bytes=args.device_bound_bytes_per_rank))
            wc=phase(f'checked_Dyson_{i}',lambda:tg.transition_dyson_value(Pi,H,dyson,raw_prefactor=pref))
            epsilon,inv=phase(f'dielectric_{i}',lambda:dyson.dielectric(H,Pi/pref))
            vals={k:float(v) for k,v in metric(epsilon,inv).items()}
            bounds=dielectric_inverse_bounds(vals['epsilon_fro'],vals['inverse_fro'],
                vals['inverse_residual_fro'],978,padding_defect=max(vals['padding_defect'],vals['inverse_padding_defect']))
            records.append(dict(site=i,Pi=note,dielectric=vals,inverse_bounds=bounds));exact.append(wc)
            if i<2:
                full=V[None]+wc
                r=narrow(mm(full,jax.device_put(ur[None],face)))
                l=narrow(mm(full,jax.device_put(ul[None],face),transa='C'))
                x,y=phase(f'seeds_{i}',lambda:tg.dyson_transition_seeds(bridge,T,cat,energy,rho,z,r,l,gamma))
                seeds.extend((x,y))
            report['direct_samples']=records;save()
        crossing=tg.transition_crossing_map(cat,cat)
        Q,br=phase('TSQR_CGS2',lambda:tg.transition_basis(bridge,seeds,cat,rho,capacity=cap,
            rank_tolerance=args.rank_tolerance,max_block_size=width,
            max_device_bytes=args.device_bound_bytes_per_rank,self_crossing=crossing))
        model=phase('structured_parent',lambda:tg.transition_parent_model(bridge,T,cat,energy,rho,V,Q,gamma,
            max_block_size=width,host_workspace_bytes=args.host_bound_bytes_per_rank))
        count=model['count'];poles=np.asarray(model['poles_ry'])[:count]
        report['basis']=br;report['model']=model['receipt']
        growth=float(poles.imag.max(initial=-np.inf));errors=[]
        for i,z in enumerate(sites):
            denominator=device_put_process_local(np.asarray(z-model['poles_ry'])[None,None],rep)
            factors=mm(model['left']/denominator,model['right'],transb='C')
            errors.append(dict(site=i,role='train' if i<2 else 'held',relative_frobenius=float(
                jnp.linalg.norm(factors-exact[i])/jnp.maximum(jnp.linalg.norm(exact[i]),1e-300))))
        report.update(status='NOT_ADMITTED_COMPLETE_ONE_PARENT_NO_SIGMA_NO_DYNAMICS',
            checks=errors,max_growth_ry=None if not count else growth,
            collision_laplace_margin_ry=None if not count else eta-growth,
            support_margin_ry=None if not count else float(sites.imag.min())-growth,
            response_checks_pass=all(x['relative_frobenius']<=args.fit_tolerance for x in errors),
            laplace_height_checks_pass=bool(not count or growth+1e-12<min(eta,sites.imag.min())),
            no_native_pole_certification_from_sample_norms=True)
        error=None
        if rank0:
            try:np.savez(root/'frequency_only.npz',poles_ry=poles,count=count,
                projected_generator=model['projected_generator'])
            except OSError as exc:error=exc
        agree_io_refusal(error,path=root/'frequency_only.npz',stage='transition_q0/frequency_only')
    except Exception:
        report.update(status='NOT_ADMITTED_REFUSED',error=traceback.format_exc());rc=1;say(report['error'],flush=True)
    report['total_wall_s']=time.monotonic()-started
    report['active_gpu_hours_upper']=4*report['total_wall_s']/3600
    save();watchdog.cancel();runtime.finalize_process(rc)


if __name__=='__main__':main()
