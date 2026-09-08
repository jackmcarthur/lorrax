"""Full balanced round trip and modal exports; no production Sigma implementation."""
from pathlib import Path
import importlib.util
import json
import time
import numpy as np

EV = 13.605693122994
ETA = .25/EV
S = Path('/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14')

def reduce_models(a, B, C, balancing, context):
    """Project H(s)=C(sI-A)^-1 B and remove evaluation eta from modal poles.

    Parameters
    ----------
    a : jax.Array, (d,)
        Diagonal stable generator, Ry, on the assigned single GPU.
    B, C : jax.Array
        Input (d,n) and output (n,d) arrays in physical coordinates.
    balancing : tuple
        Cholesky roots Lp,Lq and U,h,Vh from SVD(Lq†Lp).
    context : dict
        Out directory, receipt, 16 z_Ry samples, original physical factor
        and positive Omega_Ry arrays, and the existing mesh.
    """
    import jax
    import jax.numpy as jnp
    import scipy.linalg
    from jax.sharding import PartitionSpec as P
    from file_io.slab_io import SlabIO
    lp,lq,u,h,vh = balancing
    out,receipt,z = context['out'],context['receipt'],context['z']
    factor,om = context['factor'],context['omega']
    shift=context.get('shift_ry',ETA)
    floor_damping=context.get('floor_damping',False)
    adj=lambda x:x.conj().T
    relative=lambda x,y:float(jnp.linalg.norm(x-y)/jnp.linalg.norm(y))
    def exact(zi):
        return (factor/(zi**2-om**2)[None,:])@adj(factor)
    # Exact transformed resolvent is Ti diag(1/(s-a)) T. Contract C T Ti
    # once, then stream the 16 diagonal resolvents. This measures the full
    # balancing transform without 16 redundant dense d-side factorizations.
    clp=C@lp
    lqb=adj(lq)@B
    ctv=clp@adj(vh)
    returned=((ctv/h[None,:])@adj(u))@adj(lq)
    full_errors=[relative((returned/(-1j*zi-shift-a)[None,:])@B,exact(zi)) for zi in z]
    receipt['full_balancing_W16_roundtrip_max']=max(full_errors)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if max(full_errors)>1e-10:
        raise RuntimeError('Full balanced transform failed W16 gate; no truncation trusted')
    ownerpath=S/'runs/frequency_integration_sandbox/299_shared_residue_ls_20260907/anchor_input.py'
    spec=importlib.util.spec_from_file_location('order_coulomb_owner',ownerpath)
    owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
    assert owner.sha(owner.COULOMB)==owner.COULOMB_SHA
    with SlabIO(owner.COULOMB,mode='r',mesh=context['mesh']) as io:
        v=io.read_slab('V_canonical_qwedge',shape=(1,896,896),offset=(receipt['q'],0,0),partition_spec=P(None,'x','y'))[0]
    ve,vu=jnp.linalg.eigh((v+adj(v))/2)
    cutoff=896*np.finfo(float).eps*max(float(ve[-1]),1.)
    active=np.flatnonzero(np.asarray(ve)>cutoff)
    whiten=adj(vu[:,active])/jnp.sqrt(ve[active])[:,None]
    targets=[(f'eps{eps:.0e}',int(np.count_nonzero(np.asarray(h)>eps*float(h[0]))),None,eps)
             for eps in (1e-2,1e-3,1e-4)]
    targets += [(f'K{k}',2*k,k,None) for k in (224,448,896) if 2*k<=len(a)]
    targets=context.get('targets',targets)
    for label,d,budget,epsilon in targets:
        started=time.monotonic()
        invsqrt=1/jnp.sqrt(h[:d])
        T=(lp@adj(vh[:d]))*invsqrt[None,:]
        Ti=invsqrt[:,None]*(adj(u[:,:d])@adj(lq))
        Ar=Ti@(a[:,None]*T)
        Br=invsqrt[:,None]*(adj(u[:,:d])@lqb)
        Cr=ctv[:,:d]*invsqrt[None,:]
        identity=float(jnp.linalg.norm(Ti@T-jnp.eye(d))/np.sqrt(d))
        errors=[relative(Cr@jnp.linalg.solve((-1j*zi-shift)*jnp.eye(d)-Ar,Br),exact(zi)) for zi in z]
        # At nu=eta, s=nu-shift. Check the assembled general-residue matrix.
        response=-Cr@jnp.linalg.solve((.25/EV-shift)*jnp.eye(d)-Ar,Br)
        x=whiten@response@adj(whiten)
        anti=float(jnp.linalg.norm(x-adj(x))/max(float(jnp.linalg.norm(x)),1e-300))
        vals=jnp.linalg.eigvalsh((x+adj(x))/2)
        passivity=dict(min_eigenvalue=float(vals[0]),max_eigenvalue=float(vals[-1]),
                       relative_antihermitian=anti,resolved_rank=len(active),
                       status='PASS' if float(vals[0])>=-1e-6 and float(vals[-1])<=1+1e-6 and anti<=1e-8 else 'FAIL',
                       coulomb_sha256=owner.COULOMB_SHA,nu_ev=.25)
        # The general nonnormal eigensolve uses the compute host only.
        ah,bh,ch=np.asarray(Ar),np.asarray(Br),np.asarray(Cr)
        ar,vr=scipy.linalg.eig(ah)
        left=ch@vr
        right=scipy.linalg.solve(vr,bh)
        physical=1j*(ar+shift)
        positive=physical.real>1e-10
        modal_errors=[]
        for zi in z:
            wm=(left/(zi-physical)[None,:])@(1j*right)
            direct=ch@scipy.linalg.solve((-1j*zi-shift)*np.eye(d)-ah,bh)
            modal_errors.append(np.linalg.norm(wm-direct)/np.linalg.norm(direct))
        if max(modal_errors)>1e-8:
            raise RuntimeError('Modal conversion lost reduced W')
        floor_receipt=None
        if floor_damping:
            original=physical.copy()
            # Explicit coordinator perturbation, never hidden as roundoff.
            eligible=(physical.imag>0)&(physical.imag<=shift+1e-4/EV)
            physical[eligible]=physical[eligible].real+0j
            floor_change=[];floor_parent=[]
            for zi in z:
                before=(left/(zi-original)[None,:])@(1j*right)
                after=(left/(zi-physical)[None,:])@(1j*right)
                floor_change.append(float(np.linalg.norm(after-before)/np.linalg.norm(before)))
                floor_parent.append(relative(jnp.asarray(after),exact(zi)))
            response=-jnp.asarray((left/(.25j/EV-physical)[None,:])@(1j*right))
            x=whiten@response@adj(whiten)
            vals=jnp.linalg.eigvalsh((x+adj(x))/2)
            anti=float(jnp.linalg.norm(x-adj(x))/max(float(jnp.linalg.norm(x)),1e-300))
            passivity=dict(min_eigenvalue=float(vals[0]),max_eigenvalue=float(vals[-1]),relative_antihermitian=anti,
                           resolved_rank=len(active),status='PASS' if float(vals[0])>=-1e-3 and float(vals[-1])<=1+1e-3 and anti<=1e-8 else 'FAIL',
                           coulomb_sha256=owner.COULOMB_SHA,nu_ev=.25,tolerance=1e-3)
            floor_receipt=dict(count=int(np.sum(eligible)),minimum_gamma_before_ev=float(-original.imag.max()*EV),
                               count_beyond_shift_margin=int(np.sum(original.imag>shift+1e-4/EV)),
                               W16_change_relative_to_unfloored=floor_change,W16_parent_relative_after_floor=floor_parent,
                               policy='Coordinator explicitly requested zero physical damping floor; unchanged eta=.25eV')
            np.savez(out/f'{label}_unfloored_model.npz',poles_ry=original,residue_left=left,residue_right=1j*right,q_parent=np.int32(receipt['q_full_row']))
        path=out/f'{label}_model.npz'
        # W(z)=sum L[:,p] R[p,:]/(z-pole[p]); no implicit conjugation.
        np.savez(path,poles_ry=physical,residue_left=left,residue_right=1j*right,
                 q_parent=np.int32(receipt['q_full_row']))
        info=dict(q=receipt['q'],q_parent=receipt['q_full_row'],jobid=receipt['jobid'],stepid=receipt['stepid'],
                  requested_positive_K=budget,relative_hsv_threshold=epsilon,model_label=label,state_order=d,positive_pole_K=int(positive.sum()),
                  J_positive=len(np.unique(physical[positive])),port_width=896,
                  modal_eigenvector_condition=float(np.linalg.cond(vr)),balanced_identity_error=identity,
                  W16_parent_relative_errors=errors,modal_roundtrip_max=float(max(modal_errors)),
                  full_balancing_W16_roundtrip_max=max(full_errors),
                  damping_fraction_gt_0p1ev=float(np.mean(-physical[positive].imag*EV>.1)) if np.any(positive) else None,
                  negative_physical_damping_count=int(np.sum(physical.imag>1e-10)),
                  minimum_physical_gamma_ev=float(np.min(-physical.imag*EV)),
                  maximum_abs_real_pole_ev=float(np.max(np.abs(physical.real))*EV),
                  stable_broadened=bool(np.all(ar.real<0)),
                  storage_bytes=int(physical.nbytes+left.nbytes+right.nbytes),
                  passivity=passivity,elapsed_seconds=time.monotonic()-started,
                  export_convention='W(z)=L diag(1/(z-poles_ry)) R; all signed poles, no conjugation of R',
                  weight=receipt['weight'],realization_shift_ev=shift*EV,floor=floor_receipt,
                  star_tail_target=context.get('target_metadata',{}).get(label),
                  admissibility='REFUSED_PHYSICAL_UPPER_HALF_PLANE' if np.any(physical.imag>1e-10) else ('PASS_POINT_AND_POLES' if passivity['status']=='PASS' else 'REFUSED_POINT_PASSIVITY'))
        (out/f'{label}_receipt.json').write_text(json.dumps(info,indent=2)+'\n')
        print(json.dumps(info),flush=True)
