"""Passive shared-real-pole and general ordered-pole constructors.

The physical convention is W(s) = b (s - Lambda)^-1 b.H, s = z_Ry**2;
S_m = 2 M_(2m+1).  Thus b has units Ry**(3/2), Lambda Ry**2, and
M1/M3 are physical moments in Ry**3/Ry**5, never bare chi coefficients.

Sample matrices are consumed in bounded batches.  Only narrow direction,
output and derivative-action panels survive to the pencil assembly.  Dense
products and eigensolves of the passive constructor use resolved distrib_la
operations. The explicit general, non-Hermitian route instead gathers only
its capacity-bounded reduced frequency pencil for canonical host SciPy/LAPACK
reduction; microscopic port factors remain distributed.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
import time

import jax
from common import timing
from gw.shared_pole_capacity import ConstructorCapacity
from gw.shared_pole_directions import (_round_kernels, _sample_point, line_panel_states, port_extent,
                                       select_round_states, infinity_directions)
from gw.shared_pole_reduction import ORIENTATION_PAIR_REFUSAL

EQUILIBRIUM_METRIC_SOLVER='original-columns-extended-newton-v1'
EQUILIBRIUM_DEVICE_SOLVER='same-state-device-metric-quotient-v1'


def _spectral_cut_receipt(closure):
    """An end cut has no neighbouring gap; publish null rather than infinity."""
    import math
    if closure is None:return None
    return {key:(None if key in ('gap_rel','gap_rel_closed') and
                 not math.isfinite(value) else value) for key,value in closure.items()}


@lru_cache(maxsize=32)
def _equilibrium_device_programs(mesh_xy, width, dial_key, room=None, eigh_backend='auto'):
    """Cache shape-specialized native metric/Ritz programs, never current data."""
    import jax.numpy as jnp
    import distrib_la
    from jax.sharding import NamedSharding, PartitionSpec as P
    from .shared_pole_execution import face_matmul
    from .shared_pole_reduction import equilibrium_metric_stage, equilibrium_ritz_stage
    face = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    rep = NamedSharding(mesh_xy, P())
    mm = face_matmul(mesh_xy)
    metric = jax.jit(lambda g,h,n,b: equilibrium_metric_stage(g,h,n,matmul=mm,space=b),
        out_shardings=(face,rep,rep))
    eig = distrib_la.plan('eigh',mesh_xy,n=width,backend=eigh_backend,budget_bytes=room)
    ritz = distrib_la.checked_program(lambda g,h,n,z,a: equilibrium_ritz_stage(g,h,n,z,a,
        matmul=mm,eigh=eig.batched),mesh_xy,out_shardings=(rep,face,rep,rep,rep))
    scale_columns = jax.jit(lambda a,w:a*w, out_shardings=face)
    return metric,ritz,scale_columns


def reduce_equilibrium_pencil_device(G, H, output_gram, *, mesh_xy, max_order,
                                    rank_tolerance, gamma_ry, fixed_spaces=None,
                                    multiplet_tolerance=1e-6, capacity=None,eigh_backend='auto'):
    """Native face reduction of the equilibrium same-state value pencil.

    G/H/output_gram are complex128 [1,R,R] faces, in original coordinates.
    A positive diagonal congruence and Hermitian metric cut cap amplification
    using the source rank tolerance. Removing weak positive metric modes is
    an explicit approximation, admitted only by the caller's original full-W
    train/held checks; full complex Sigma refinement is separately required.
    Only spectra/scalar diagnostics reach host. Maps and held spaces remain
    native [1,R,Kcarrier] faces, with exact-zero carrier columns.
    """
    import math
    import numpy as np
    import jax.numpy as jnp
    import distrib_la
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host, device_put_process_local
    from common.rank_criterion import select_rank
    from common.spectral_closure import resolve_spectral_cut
    from common.staged_reshard import permute_sharded_axis
    from .shared_pole_execution import face_matmul
    started = time.monotonic()
    if (G.ndim != 3 or G.shape[0] != 1 or G.shape[-2] != G.shape[-1]
            or H.shape != G.shape or output_gram.shape != G.shape
            or not 0 < float(rank_tolerance) < 1 or not np.isfinite(gamma_ry)
            or gamma_ry < 0 or int(max_order) < 1):
        raise ValueError('GATE shared_equilibrium_pencil: one square native G/H/output Gram required')
    face = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    rep = NamedSharding(mesh_xy, P())
    mm = face_matmul(mesh_xy)
    side = G.shape[-1]
    rounding = 128*side*np.finfo(float).eps
    room=None
    if capacity is not None:
        from lxkit import memory
        ambient=capacity.preview(resident_bytes_per_rank=0,workspace_bytes_per_rank=0,
            concurrent_with=tuple(capacity.live_stages))
        room=memory.bucket(max(0,ambient['available_device_bytes_per_rank']-
                              ambient['aggregate_bytes_per_rank']))
    base = None if fixed_spaces is None else fixed_spaces['right']
    if base is not None and (base.ndim != 3 or base.shape[:2] != (1, side)
                             or fixed_spaces['left'] is not base):
        raise ValueError('GATE shared_equilibrium_basis: one current native common space required')
    def admit(program, operands, phase, vendor=0):
        if capacity is None:
            return
        executable=program.lower(*operands).compile()
        analysis=executable.memory_analysis()
        workspace=max(0,int(analysis.temp_size_in_bytes)+int(analysis.output_size_in_bytes)
                      -int(analysis.alias_size_in_bytes))
        capacity.reserve(f'equilibrium_device_{phase}.{len(capacity.entries)}',
            resident_bytes_per_rank=0,workspace_bytes_per_rank=workspace+int(vendor),
            concurrent_with=tuple(capacity.live_stages))
    metric_stage,_,scale_columns = _equilibrium_device_programs(
        mesh_xy,side,distrib_la.dial_key(),room,eigh_backend)
    admit(metric_stage,(G,H,output_gram,base),'metric')
    metric, scale, initial = metric_stage(G,H,output_gram,base)
    initial = {k:np.asarray(gather_to_host(v)).item() for k,v in initial.items()}
    if not initial['input_finite'] or max(initial['G_hermiticity'],initial['H_hermiticity']) > rounding:
        raise ValueError('GATE shared_equilibrium_hermiticity: finite current Hermitian G/H required')
    if not np.isfinite(initial['output_norm']) or initial['output_hermiticity']>rounding:
        raise ValueError('GATE shared_equilibrium_output: finite Hermitian output Gram required')
    eig = distrib_la.plan('eigh', mesh_xy, n=metric.shape[-1],backend=eigh_backend,budget_bytes=room)
    spectrum, vectors = eig.batched(metric, cut=float(rank_tolerance) if base is None else rounding)
    spectrum = np.asarray(gather_to_host(spectrum))[0]
    largest = float(np.max(abs(spectrum), initial=0.))
    if not np.isfinite(spectrum).all() or largest == 0 or spectrum[0] < -rounding*largest:
        raise ValueError('GATE shared_equilibrium_metric: unresolved or negative H; no PSD repair')
    closure = None
    if base is None:
        cut = max(float(rank_tolerance), rounding)
        rank = select_rank(spectrum, cut, ceiling=side)
        rank, closure = resolve_spectral_cut(np.maximum(spectrum,0), rank,
            rtol=float(multiplet_tolerance), where='shared equilibrium device metric', rcond=cut)
    else:
        # A held numerical span is fixed within the inner solve. Refactor
        # its CURRENT H; never reuse old whitening or silently cut it again.
        norms = np.asarray(gather_to_host(jnp.sum(abs(base)**2, axis=-2)))[0]
        rank = int(np.count_nonzero(norms))
        if np.count_nonzero(spectrum > rounding*largest) != rank:
            raise ValueError('GATE shared_equilibrium_basis: current held H unresolved; no held rank drop')
        cut = rounding
    if not 0 < rank <= int(max_order):
        raise ValueError('GATE shared_equilibrium_order: resolved rank empty or exceeds physical pole budget')
    width = math.ceil(rank/int(mesh_xy.shape['y']))*int(mesh_xy.shape['y'])
    indices = np.r_[np.arange(len(spectrum)-rank,len(spectrum)),
                    np.repeat(len(spectrum)-1,width-rank)].astype(np.int32)
    selected = permute_sharded_axis(vectors, 2, indices, mesh_xy, face.spec)
    weights = np.r_[1/np.sqrt(spectrum[-rank:]), np.zeros(width-rank)]
    weights = device_put_process_local(weights[None,None], rep)
    selected = scale_columns(selected,weights)
    space = scale_columns(selected,scale[:,:,None]) if base is None else mm(base,selected)
    active = device_put_process_local((np.arange(width)<rank)[None],rep)
    _,ritz,_ = _equilibrium_device_programs(mesh_xy,width,distrib_la.dial_key(),room,eigh_backend)
    vendor=0
    if capacity is not None:
        ritz_plan=distrib_la.plan('eigh',mesh_xy,n=width,backend=eigh_backend,budget_bytes=room)
        vendor=distrib_la.workspace_bytes_per_rank(ritz_plan,'eigh',((1,width,width),),np.complex128)
    admit(ritz,(G,H,output_gram,space,active),'ritz',vendor)
    mu, maps, physical, output_weight, diagnostics = ritz(G,H,output_gram,space,active)
    diagnostics = {k:np.asarray(gather_to_host(v)).reshape(-1)[0].item()
                   for k,v in diagnostics.items()}
    if (not diagnostics['eigen_finite'] or not diagnostics['metric_ok']
            or not np.isfinite(diagnostics['eigen_residual'])
            or diagnostics['eigen_residual'] > 64*width*np.finfo(float).eps):
        raise ValueError('GATE shared_equilibrium_eigenvectors: retained-space metric/backward check failed')
    mu = np.asarray(gather_to_host(mu))[0]
    physical = np.asarray(gather_to_host(physical))[0]
    bound = rounding*float(np.max(abs(mu[physical]), initial=0.))
    finite = physical & (abs(mu)>bound)
    output_weight = np.asarray(gather_to_host(output_weight))[0]
    output_scale=float(np.sum(abs(output_weight[physical])))
    if not np.isfinite(output_weight).all() or np.any(output_weight < -rounding*output_scale):
        raise ValueError('GATE shared_equilibrium_output: nonfinite or negative output weight')
    constant_weight=float(np.sum(abs(output_weight[physical & ~finite])))
    if constant_weight > (rank_tolerance**2+rounding)*output_scale:
        raise ValueError('GATE shared_equilibrium_infinite: live unresolved mu gives constant feedthrough')
    positions = np.flatnonzero(finite)
    positions = positions[np.argsort(1/mu[positions], kind='stable')]
    poles = 1/mu[positions]-1j*float(gamma_ry)
    count = len(poles)
    if count == 0:
        raise ValueError('GATE shared_equilibrium_infinite: no finite pole remains')
    out_width = math.ceil(count/int(mesh_xy.shape['y']))*int(mesh_xy.shape['y'])
    selection = np.r_[positions, np.repeat(positions[0],out_width-count)].astype(np.int32)
    modal = permute_sharded_axis(maps, 2, selection, mesh_xy, face.spec)
    normalization = np.r_[1/np.sqrt(abs(mu[positions])),np.zeros(out_width-count)]
    output_map = scale_columns(modal,device_put_process_local(normalization[None,None],rep))
    signs = np.r_[np.sign(mu[positions]),np.zeros(out_width-count)]
    input_map = scale_columns(output_map,device_put_process_local(signs[None,None],rep))
    receipt = dict(order=count,held_state_rank=rank,construction='equilibrium-definite-value',
        metric_solver=EQUILIBRIUM_DEVICE_SOLVER,numerical_spaces='original-coordinate-native-common-span',
        rank_tolerance=float(rank_tolerance),relative_metric_cut=cut,
        H_min_eigenvalue=float(spectrum[0]),H_max_eigenvalue=largest,
        H_condition=largest/float(spectrum[-rank]),original_order=side,
        zero_mu_output_weight=constant_weight,projected_output_weight=output_scale,
        discarded_metric_directions=side-rank if base is None else 0,
        approximation='positive metric quotient; all original full-W targets remain mandatory',
        physical_accuracy_certified=False,sigma_accuracy_certified=False,
        approximation_requires_full_W_validation=True,
        required_validation='original full-W train/held and full complex column-Sigma/collision refinement',
        spectral_closure=_spectral_cut_receipt(closure),spectral_variable='z+i_gamma',gamma_ry=float(gamma_ry),
        host_frequency_matrix_bytes=0,host_reduce_seconds=0.,
        device_reduce_seconds=time.monotonic()-started,**initial,**diagnostics)
    held_space=space if base is None else base
    return dict(poles_ry=poles,output_map=output_map,input_map=input_map,
        spaces=dict(right=held_space,left=held_space),receipt=receipt)


@lru_cache(maxsize=32)
def _general_device_programs(mesh_xy, left, right, dial_key, room, eigh_backend, eig_backend):
    """Native general descriptor programs; all current pencils are operands."""
    import jax.numpy as jnp
    import distrib_la
    from jax.sharding import NamedSharding,PartitionSpec as P
    from .shared_pole_execution import face_matmul
    from .shared_pole_reduction import (general_equilibration_stage,general_descriptor_dilation,
        general_projected_stage,general_modal_stage)
    from .shared_pole_pencil import tile_split
    face=NamedSharding(mesh_xy,P(None,'x','y'));rep=NamedSharding(mesh_xy,P())
    mm=face_matmul(mesh_xy)
    eig=distrib_la.plan('eigh',mesh_xy,n=left+right,backend=eigh_backend,budget_bytes=room)
    equilibrate=jax.jit(general_equilibration_stage,out_shardings=(face,face,rep,rep))
    def svd(e):
        values,vectors=eig.batched(general_descriptor_dilation(e,matrix_sharding=face))
        u,v=tile_split(vectors,(left,right),-2,face)
        return values,u*2**.5,v*2**.5
    svd=distrib_la.checked_program(svd,mesh_xy,out_shardings=(rep,face,face))
    project=jax.jit(lambda e,a,x,y:general_projected_stage(e,a,x,y,matmul=mm),
        out_shardings=(face,face))
    def local_modal(a):
        w,vr=distrib_la.eig_local(a,backend=eig_backend)
        dual=distrib_la.solve_local(vr,jnp.broadcast_to(jnp.eye(a.shape[-1],dtype=a.dtype),a.shape))
        info=distrib_la.eig_local_diagnostics(a,w,vr,dual)
        p=int(mesh_xy.shape['y'])
        small=tuple(jnp.broadcast_to(i[:,None,None],(a.shape[0],p,p)) for i in info)
        return jnp.eye(a.shape[-1],dtype=a.dtype)*w[:,None,:],vr,dual,small
    modal=distrib_la.local_batch(local_modal,mesh_xy)
    run_modal=jax.jit(lambda a,x,y,active:general_modal_stage(a,x,y,active,matmul=mm,modal=modal),
        out_shardings=(face,face,face,(face,)*4,rep))
    scale=jax.jit(lambda a,w:a*w,out_shardings=face)
    lift=jax.jit(lambda x,y,o,i:(mm(x,o),mm(y,i)),out_shardings=(face,face))
    def checks(e,a,rb,lb,o,values,active):
        metric=mm(lb,mm(e,rb),transa='C')
        want=jnp.eye(e.shape[-1],dtype=e.dtype)*active[:,None,:]
        lhs=mm(a,o);rhs=mm(e,o)*values[:,None,:]
        error=jnp.linalg.norm(lhs-rhs)/jnp.maximum(
            jnp.linalg.norm(a)*jnp.linalg.norm(o)+jnp.linalg.norm(e)*jnp.linalg.norm(o)*jnp.max(abs(values)),
            jnp.finfo(e.real.dtype).tiny)
        return jnp.linalg.norm(metric-want),error
    checks=jax.jit(checks,out_shardings=(rep,rep))
    return equilibrate,svd,project,run_modal,scale,lift,checks


def reduce_general_pencil_device(E,A,*,mesh_xy,max_order,rank_tolerance=1e-8,
                                fixed_spaces=None,frequency_scale_ry=1.,coordinate_units=None,
                                capacity=None,eigh_backend='auto'):
    """Two-sided native reduction of one general complex descriptor pencil.

    E/A [1,L,R] and returned original-coordinate spaces/maps [1,L/R,K]
    remain face tiled. Hermitian dilations supply direct singular vectors;
    the current restricted descriptor is balanced on every solve. The
    ordinary non-Hermitian eigensystem uses an explicit device provider and
    its inverse right eigenvectors, preserving signed/complex/growing poles.
    Only O(L+R+K) spectra and scalar diagnostics reach the host.
    """
    import math
    import numpy as np
    import jax.numpy as jnp
    import distrib_la
    from jax.sharding import NamedSharding,PartitionSpec as P
    from common.collectives import device_put_process_local,gather_to_host
    from common.rank_criterion import select_rank
    from common.spectral_closure import resolve_spectral_cut
    from common.staged_reshard import permute_sharded_axis
    from .shared_pole_execution import face_matmul
    started=time.monotonic()
    if (E.ndim!=3 or E.shape[0]!=1 or A.shape!=E.shape or min(E.shape[-2:])<1
            or not 0<float(rank_tolerance)<1 or not np.isfinite(frequency_scale_ry)
            or frequency_scale_ry<=0 or int(max_order)<1):
        raise ValueError('GATE shared_general_pencil: one finite native descriptor pair required')
    backend=distrib_la.resolve_eig_local_backend()
    nl,nr=E.shape[-2:];p=int(mesh_xy.shape['y'])
    face=NamedSharding(mesh_xy,P(None,'x','y'));rep=NamedSharding(mesh_xy,P())
    put=lambda a:device_put_process_local(np.asarray(a),rep)
    mm=face_matmul(mesh_xy)
    room=None
    if capacity is not None:
        from lxkit import memory
        ambient=capacity.preview(resident_bytes_per_rank=0,workspace_bytes_per_rank=0,
            concurrent_with=tuple(capacity.live_stages))
        room=memory.bucket(max(0,ambient['available_device_bytes_per_rank']-ambient['aggregate_bytes_per_rank']))
    admitted=[];modal_workspace=None
    def admit(program,operands,phase,local_order=None,eigh_order=None):
        nonlocal modal_workspace
        if capacity is None:return
        executable=program.lower(*operands).compile()
        analysis=executable.memory_analysis()
        price=max(0,int(analysis.temp_size_in_bytes)+int(analysis.output_size_in_bytes)-int(analysis.alias_size_in_bytes))
        if local_order is not None:
            workspace=distrib_la.eig_local_workspace(local_order,backend=backend)
            price+=workspace['device_bytes']
            modal_workspace=workspace
        if eigh_order is not None:
            plan=distrib_la.plan('eigh',mesh_xy,n=eigh_order,backend=eigh_backend,budget_bytes=room)
            price+=distrib_la.workspace_bytes_per_rank(plan,'eigh',((1,eigh_order,eigh_order),),np.complex128)
        row=capacity.reserve(f'general_device_{phase}.{len(capacity.entries)}',
            resident_bytes_per_rank=0,workspace_bytes_per_rank=price,
            concurrent_with=tuple(capacity.live_stages),whole=local_order is not None)
        admitted.append(row['stage'])
    programs=_general_device_programs(mesh_xy,nl,nr,distrib_la.dial_key(),room,eigh_backend,backend)
    if fixed_spaces is None:
        if coordinate_units is None:coordinate_units=(np.ones(nl),np.ones(nr))
        ul,ur=(np.asarray(v,float) for v in coordinate_units)
        if (ul.shape!=(nl,) or ur.shape!=(nr,) or not np.isfinite(ul).all()
                or not np.isfinite(ur).all() or np.any(ul<=0) or np.any(ur<=0)):
            raise ValueError('GATE shared_general_units: positive original equation-unit vectors required')
        operands=(E,A,put(ul),put(ur),put(frequency_scale_ry))
        admit(programs[0],operands,'equilibration')
        eq,_,dl,dr=programs[0](*operands)
        admit(programs[1],(eq,),'descriptor_svd',eigh_order=nl+nr)
        spectrum,u,v=programs[1](eq)
        spectrum=np.asarray(gather_to_host(spectrum))[0]
        singular=spectrum[-min(nl,nr):]
        if not np.isfinite(singular).all() or singular[-1]<=0:
            raise ValueError('GATE shared_general_rank: nonfinite or null descriptor')
        natural_rank=select_rank(singular,float(rank_tolerance),ceiling=min(nl,nr))
        rank=min(natural_rank,int(max_order))
        rank,closure=resolve_spectral_cut(np.maximum(singular,0),rank,
            where='shared general device descriptor',rcond=float(rank_tolerance))
        if not 0<rank<=int(max_order):
            raise ValueError('GATE shared_general_order: resolved descriptor rank empty or exceeds pole budget')
        width=math.ceil(rank/p)*p
        index=np.r_[np.arange(len(spectrum)-rank,len(spectrum)),np.repeat(len(spectrum)-1,width-rank)].astype(np.int32)
        active=np.arange(width)<rank
        x=permute_sharded_axis(v,2,index,mesh_xy,face.spec)
        y=permute_sharded_axis(u,2,index,mesh_xy,face.spec)
        x=programs[4](programs[4](x,dr[:,:,None]),put(active[None,None]))
        y=programs[4](programs[4](y,dl[:,:,None]),put(active[None,None]))
        spaces=dict(right=x,left=y)
    else:
        x,y=fixed_spaces['right'],fixed_spaces['left']
        if (x.ndim!=3 or y.ndim!=3 or x.shape[:2]!=(1,nr) or y.shape[:2]!=(1,nl)
                or x.shape[-1]!=y.shape[-1]):
            raise ValueError('GATE shared_general_basis: native original-coordinate held spaces required')
        width=x.shape[-1]
        norms=jax.jit(lambda a:jnp.sum(abs(a)**2,axis=-2),out_shardings=rep)(x)
        active=np.asarray(gather_to_host(norms))[0]>0
        rank=int(np.count_nonzero(active));spaces=fixed_spaces;closure=None;natural_rank=rank
        if not 0<rank<=int(max_order):raise ValueError('GATE shared_general_basis: invalid held rank')
    # Original pencils and original-coordinate spaces define CURRENT physics.
    admit(programs[2],(E,A,x,y),'current_projection')
    ep,ap=programs[2](E,A,x,y)
    small=_general_device_programs(mesh_xy,width,width,distrib_la.dial_key(),room,eigh_backend,backend)
    admit(small[1],(ep,),'current_descriptor_svd',eigh_order=2*width)
    spectrum,u,v=small[1](ep)
    spectrum=np.asarray(gather_to_host(spectrum))[0]
    singular=spectrum[-width:]
    rounding=128*max(nl,nr,width)*np.finfo(float).eps
    if (not np.isfinite(singular).all() or singular[-rank]<=max(float(rank_tolerance),rounding)*singular[-1]):
        raise ValueError('GATE shared_general_basis: current projected descriptor unresolved; no held rank drop')
    index=np.r_[np.arange(len(spectrum)-rank,len(spectrum)),np.repeat(len(spectrum)-1,width-rank)].astype(np.int32)
    weights=np.r_[1/np.sqrt(singular[-rank:]),np.zeros(width-rank)]
    rb=small[4](permute_sharded_axis(v,2,index,mesh_xy,face.spec),put(weights[None,None]))
    lb=small[4](permute_sharded_axis(u,2,index,mesh_xy,face.spec),put(weights[None,None]))
    active=put((np.arange(width)<rank)[None])
    admit(small[3],(ap,rb,lb,active),'modal',width)
    poles,omap,imap,info,sentinels=small[3](ap,rb,lb,active)
    info=[np.asarray(gather_to_host(v)).reshape(-1)[0].item() for v in info]
    if int(info[0]):
        raise ValueError(f'GATE shared_general_eigenvectors: native modal backward/inverse status={int(info[0])}')
    poles=np.asarray(gather_to_host(jax.jit(lambda a:jnp.diagonal(a,axis1=-2,axis2=-1),out_shardings=rep)(poles)))[0]
    sentinels=np.asarray(gather_to_host(sentinels))[0]
    padding=width-rank
    inert=np.zeros(width,bool)
    for sentinel in sentinels[rank:]:
        candidates=np.flatnonzero(abs(poles-sentinel)<=rounding*max(1,abs(sentinel)))
        if len(candidates)!=1 or inert[candidates[0]]:
            raise ValueError('GATE shared_general_padding: modal inert sentinel unresolved')
        inert[candidates[0]]=True
    positions=np.flatnonzero(~inert)
    positions=positions[np.lexsort((poles[positions].imag,poles[positions].real))]
    logical=poles[positions]
    index=np.r_[positions,np.repeat(positions[0],padding)].astype(np.int32)
    mask=put((np.arange(width)<rank)[None,None])
    output_coefficients=small[4](permute_sharded_axis(omap,2,index,mesh_xy,face.spec),mask)
    input_coefficients=small[4](permute_sharded_axis(imap,2,index,mesh_xy,face.spec),mask)
    check_operands=(ep,ap,rb,lb,output_coefficients,put(np.pad(logical,(0,padding))[None]),active)
    admit(small[6],check_operands,'descriptor_checks')
    metric_error,generalized_error=small[6](*check_operands)
    metric_error=float(gather_to_host(metric_error));generalized_error=float(gather_to_host(generalized_error))
    if (not np.isfinite([metric_error,generalized_error]).all()
            or generalized_error>64*width*np.finfo(float).eps):
        raise ValueError('GATE shared_general_eigenvectors: nonfinite current descriptor reconstruction')
    admit(programs[5],(x,y,output_coefficients,input_coefficients),'original_lift')
    output,input_=programs[5](x,y,output_coefficients,input_coefficients)
    return dict(poles_ry=logical,output_map=output,input_map=input_,spaces=spaces,
        receipt=dict(order=rank,held_state_rank=rank,construction='general-two-sided-tangential-loewner',
            eigensolver=f'explicit-device-{backend}-right-eigenvectors-inverse',
            numerical_spaces='original-coordinate-native-two-sided-v1',
            rank_tolerance=float(rank_tolerance),spectral_closure=_spectral_cut_receipt(closure),
            natural_descriptor_rank=natural_rank,pole_budget_binds=natural_rank>int(max_order),
            eigen_residual=float(info[1]),inverse_residual=float(info[2]),
            eigenvector_condition_estimate=float(info[3]),host_frequency_matrix_bytes=0,
            current_descriptor_metric_error=metric_error,
            projected_generalized_backward_error=generalized_error,
            host_reduce_seconds=0.,device_reduce_seconds=time.monotonic()-started,
            physical_accuracy_certified=False,sigma_accuracy_certified=False,
            required_validation='all original full-W fit/held, growth and full complex column-Sigma',
            modal_provider_workspace=modal_workspace,
            provider_host_workspace_in_device_ledger=False,
            capacity_stages=admitted))


def collective_general_pencil(E, A, *, max_order, rank_tolerance=1e-10,
                             max_workspace_bytes=4*1024**3, fixed_spaces=None,
                             frequency_scale_ry=1., coordinate_units=None,
                             structure=None, output_gram=None):
    """One canonical owner of a reduced host pencil's modal gauges.

    Every rank supplies the same gathered ``E,A[l,r]``; ONLY rank zero runs
    LAPACK. A bounded success/error record crosses first, then the exact
    uint64 bits of the five reduced complex arrays (poles, two coefficient
    maps and two held frequency spaces). No microscopic port or q stack
    crosses this handoff. Distributed faces must not mix independent host
    SVD/eigenvector gauges, even on homogeneous hardware.
    """
    from common.collectives import process_count
    if int(process_count()) == 1:
        return reduce_general_pencil(E, A, max_order=max_order, rank_tolerance=rank_tolerance,
            max_workspace_bytes=max_workspace_bytes, fixed_spaces=fixed_spaces,
            frequency_scale_ry=frequency_scale_ry,coordinate_units=coordinate_units,
            structure=structure,output_gram=output_gram)
    return collective_general_pencil_batch([(E,A)], max_order=max_order,
        rank_tolerance=rank_tolerance, max_workspace_bytes=max_workspace_bytes,
        fixed_spaces=[fixed_spaces],frequency_scale_ry=frequency_scale_ry,coordinate_units=coordinate_units,
        structure=structure,output_grams=None if output_gram is None else [output_gram])[0]


def general_pencil_workspace_bytes(shape):
    """Conservative host bytes for current descriptor SVD/balance/modal maps.

    Covers reduced E/A, unit/equilibration copies and frequency spaces, both SVDs and their work,
    balanced ordinary-eigenvector/metric buffers and reconstruction products.
    It never prices or gathers microscopic port matrices.
    """
    l,r=map(int,shape)
    return 16*(20*l*r+24*min(l,r)**2)


def general_pencil_batch_plan(shapes, *, max_order, max_workspace_bytes=4*1024**3,
                             processes=None, retained_host_bytes=0, input_matrix_count=2):
    """Price a prefix of at most P SMALL reduced pencils before host gather.

    Two disjoint phases are priced, not summed: one assigned LAPACK solve
    per process beside the other input pencils; then all returned models,
    one owner's original result and one bit-transfer buffer. The same
    numerical order bound applies at every width; capacity never cuts poles.
    ``retained_host_bytes`` accounts for the caller's existing held spaces.
    """
    from common.collectives import process_count
    world=int(process_count()) if processes is None else int(processes)
    if world<1 or int(max_order)<1 or int(retained_host_bytes)<0 or input_matrix_count not in (2,3):
        raise ValueError('GATE shared_general_capacity: invalid batch capacity metadata')
    shapes=[tuple(map(int,s)) for s in shapes]
    if not shapes or any(len(s)!=2 or min(s)<1 for s in shapes):
        raise ValueError('GATE shared_general_capacity: nonempty reduced pencil shapes required')
    accepted=None
    for width in range(1,min(world,len(shapes))+1):
        batch=shapes[:width]
        inputs=[16*input_matrix_count*l*r for l,r in batch]
        reduce=[general_pencil_workspace_bytes((l,r)) for l,r in batch]
        orders=[min(int(max_order),l,r) for l,r in batch]
        models=[16*k*(1+2*l+2*r) for (l,r),k in zip(batch,orders)]
        words=[16*max(l,r,1)*k for (l,r),k in zip(batch,orders)]
        control=8192*(world+1+width)
        reduction=int(retained_host_bytes)+sum(inputs)+max(x-y for x,y in zip(reduce,inputs))+control
        handoff=int(retained_host_bytes)+sum(inputs)+sum(models)+max(models)+max(words)+control
        price=max(reduction,handoff)
        if price<=int(max_workspace_bytes):
            accepted=dict(batch_size=width,processes=world,reduced_shapes=[list(s) for s in batch],
                reduction_peak_bytes=reduction,handoff_peak_bytes=handoff,
                host_workspace_bytes=price,host_workspace_bound_bytes=int(max_workspace_bytes),
                retained_host_bytes=int(retained_host_bytes),
                input_bytes=sum(inputs),model_bound_bytes=sum(models),
                full_microscopic_host_payload=False)
    if accepted is None:
        raise ValueError('GATE shared_general_capacity: even one canonical reduced pencil exceeds host bound')
    return accepted


def _share_general_pencil(model, error, *, owner_rank, price, shape):
    """Ordered canonical metadata/raw-bit handoff; no factorization or port gather."""
    import json
    import numpy as np
    from common.collectives import all_gather_processes, process_rank
    from jax.experimental.multihost_utils import broadcast_one_to_all
    rank=int(process_rank());started=time.perf_counter()
    status=None;arrays=None
    if rank==owner_rank:
        if error is not None:
            status=dict(error=f'{type(error).__name__}: {str(error)[:1000]}')
        else:
            arrays=[model[k] for k in ('poles_ry','output_map','input_map')]
            arrays += [model['spaces'][k] for k in ('right','left')]
            status=dict(error=None,shapes=[list(a.shape) for a in arrays],
                receipt=dict(model['receipt'],canonical_owner_rank=owner_rank,
                    canonical_host_workspace_bytes=price['host_workspace_bytes'],
                    canonical_batch_size=price['batch_size'],reduced_pencil_shape=list(shape)))
    payload=b'' if rank!=owner_rank else json.dumps(status,separators=(',',':')).encode('ascii')
    lengths = np.asarray(all_gather_processes(np.asarray(len(payload), np.int64))).reshape(-1)
    count = int(lengths[owner_rank])
    # Status is bounded metadata, never eigenvector payload in a control RPC.
    if not 0 < count <= 8192:
        raise ValueError("GATE shared_general_owner: invalid canonical status extent")
    record = np.zeros(count, np.uint8)
    if rank == owner_rank:
        record[:] = np.frombuffer(payload, np.uint8)
    status = json.loads(np.asarray(all_gather_processes(record))[owner_rank].tobytes())
    if status["error"] is not None:
        failure=ValueError(f"GATE shared_general_owner: owner_rank={owner_rank}, "
                           f"reduced_shape={shape}: "+status["error"])
        # Every rank decoded the SAME bounded canonical status. Expose its
        # owner without parsing prose, so the driver can preserve ONLY that
        # already-gathered frequency pencil before rethrowing this gate.
        failure.owner_rank=int(owner_rank)
        failure.reduced_shape=tuple(shape)
        failure.canonical_host_failure=True
        raise failure
    result = []
    for i, shape in enumerate(status["shapes"]):
        array = (np.ascontiguousarray(arrays[i], dtype=np.complex128) if rank == owner_rank
                 else np.zeros(shape, np.complex128))
        # uint8 broadcast promotes under x64; uint64 preserves IEEE bits,
        # including tiny signs/complex phases and the zero-length model.
        bits = array.view(np.uint64)
        if bits.size:
            bits = np.asarray(broadcast_one_to_all(bits, is_source=rank == owner_rank))
        result.append(np.ascontiguousarray(bits).view(np.complex128).reshape(shape))
    wall=np.asarray(all_gather_processes(np.asarray(time.perf_counter()-started,np.float64))).reshape(-1)
    status['receipt']['canonical_handoff_seconds']=float(wall[owner_rank])
    return dict(zip(("poles_ry", "output_map", "input_map"), result[:3]),
                spaces=dict(right=result[3], left=result[4]), receipt=status["receipt"])


def collective_general_pencil_batch(pencils, *, max_order, rank_tolerance=1e-10,
                                   max_workspace_bytes=4*1024**3, fixed_spaces=None,
                                   retained_host_bytes=0,frequency_scale_ry=1.,coordinate_units=None,
                                   structure=None,output_grams=None):
    """Reduce at most P gathered frequency pencils concurrently, one owner each.

    The DRIVER gathers this admitted SMALL prefix on ALL ranks before entry.
    Each rank runs only its assigned pure-host LAPACK solve, BEFORE any
    ordered per-parent handoff. Every model/space/pole is then canonicalized
    from that parent's unique owner, with consistent all-rank failures.
    No JAX gather or full-microscopic matrix enters a rank-only branch.
    """
    from common.collectives import process_count, process_rank
    world,rank=int(process_count()),int(process_rank())
    pencils=list(pencils)
    if structure is not None and (not isinstance(structure,dict)
                                 or structure.get('kind')!='equilibrium_definite'):
        raise ValueError('GATE shared_equilibrium_structure: explicit equilibrium_definite kind required')
    grams=[None]*len(pencils) if output_grams is None else list(output_grams)
    if len(grams)!=len(pencils) or (structure is not None and any(g is None for g in grams)):
        raise ValueError('GATE shared_equilibrium_output: one output Gram per pencil required')
    plan=general_pencil_batch_plan([e.shape for e,a in pencils],max_order=max_order,
        max_workspace_bytes=max_workspace_bytes,processes=world,retained_host_bytes=retained_host_bytes,
        input_matrix_count=3 if structure is not None else 2)
    if len(pencils)>world or plan['batch_size']!=len(pencils):
        raise ValueError('GATE shared_general_capacity: caller must gather only the metadata-admitted batch prefix')
    fixed=[None]*len(pencils) if fixed_spaces is None else list(fixed_spaces)
    if len(fixed)!=len(pencils):
        raise ValueError('GATE shared_general_basis: one held space per batch parent required')
    model=None;error=None
    if rank<len(pencils):
        try:
            if world>1 and not jax.config.x64_enabled:
                raise ValueError('GATE shared_general_precision: canonical complex128 handoff requires JAX x64')
            e,a=pencils[rank]
            model=reduce_general_pencil(e,a,max_order=max_order,rank_tolerance=rank_tolerance,
                max_workspace_bytes=max_workspace_bytes,fixed_spaces=fixed[rank],
                frequency_scale_ry=frequency_scale_ry,coordinate_units=coordinate_units,
                structure=structure,output_gram=grams[rank])
        except Exception as exc:
            error=exc
    if world==1:
        if error is not None:
            error.owner_rank=0
            error.reduced_shape=tuple(pencils[0][0].shape)
            error.canonical_host_failure=True
            raise error
        return [model]
    results=[]
    for owner in range(len(pencils)):
        results.append(_share_general_pencil(model if rank==owner else None,
            error if rank==owner else None,owner_rank=owner,price=plan,shape=pencils[owner][0].shape))
    return results


def reduce_general_pencil(E, A, *, max_order, rank_tolerance=1e-10,
                          max_workspace_bytes=4 * 1024 ** 3, fixed_spaces=None,
                          frequency_scale_ry=1.,coordinate_units=None,
                          structure=None,output_gram=None):
    """Bounded host reduction of ONE general tangential frequency pencil.

    Only ``E,A[l,r]`` (complex128) cross this seam, never full microscopic
    response samples, q stacks, or port factors. A two-sided SVD of the
    descriptor E chooses numerical frequency spaces for this strictly proper
    finite-pole Loewner bank. A-visible directions unresolved in E are not
    promoted into nearly infinite modal states. This is numerical rank
    resolution, not a stable-pole projection: resolved growing poles remain.
    Before rank selection, finite/infinity row units and eight deterministic
    Ruiz sweeps equilibrate the combined E,A/frequency_scale_ry magnitudes.
    ``coordinate_units=(left[l],right[r])`` contains positive equation-unit
    factors, never microscopic port weights. Returned/held spaces are ALWAYS
    recovered to original E/A coordinates. Current original pencils supply
    the eigensolve, so physical poles never need frequency rescaling.
    Every CURRENT projected descriptor, including held spaces, is SVD-balanced
    to an ordinary eigenproblem. The recovered maps retain unchanged complex
    poles and independent input/output coefficients. No pole or residue is clipped.
    ``max_order`` bounds realization states, not the full spatial port count.

    Returns a plain mapping with ``poles_ry[K]``, ``output_map[r,K]`` and
    ``input_map[l,K]``. Native GEMMs form full-port factors as
    ``left=C@output_map`` and ``right=B.H@input_map``, so
    ``Wc(z)=left@diag(1/(z-poles_ry))@right.H``. Workspace is conservatively
    priced before any factorization. Infinite/defective poles refuse.
    """
    if structure is not None:
        if coordinate_units is not None:
            raise ValueError('GATE shared_equilibrium_coordinates: independent row/column units are not a congruence')
        return _reduce_equilibrium_pencil(E,A,output_gram,structure=structure,max_order=max_order,
            rank_tolerance=rank_tolerance,max_workspace_bytes=max_workspace_bytes,fixed_spaces=fixed_spaces)
    import numpy as np
    import scipy.linalg as la
    import hashlib
    started=time.perf_counter();svd_seconds=0.;eigen_started=None
    E, A = np.asarray(E, complex), np.asarray(A, complex)
    if (E.ndim != 2 or A.shape != E.shape or min(E.shape) < 1
            or not np.isfinite(E).all() or not np.isfinite(A).all()
            or int(max_order) < 1 or not 0 < rank_tolerance < 1):
        raise ValueError("GATE shared_general_pencil: finite compatible pencils required")
    nl, nr = E.shape
    # Conservative inherited bound for descriptor factors/workspace, the
    # projected descriptor and both eigenspaces. This is admission, not a
    # claim that LAPACK's hidden workspace is zero.
    price = general_pencil_workspace_bytes(E.shape)
    if price > int(max_workspace_bytes):
        raise ValueError(f"GATE shared_general_capacity: host pencil workspace {price} "
                         f"exceeds {int(max_workspace_bytes)} bytes")
    scale = max(float(np.linalg.norm(E)), float(np.linalg.norm(A)))
    if scale == 0:
        return dict(poles_ry=np.empty(0, complex), output_map=np.zeros((nr, 0), complex),
                    input_map=np.zeros((nl, 0), complex),
                    spaces=dict(right=np.zeros((nr, 0), complex), left=np.zeros((nl, 0), complex)),
                    receipt=dict(order=0, workspace_bytes=price, rank_tolerance=rank_tolerance,
                        svd_seconds=0.,eigen_seconds=0.,host_reduce_seconds=time.perf_counter()-started))
    if fixed_spaces is None:
        svd_started=time.perf_counter()
        omega=float(frequency_scale_ry)
        if not np.isfinite(omega) or omega<=0:
            raise ValueError('GATE shared_general_equilibration: positive physical frequency reference required')
        dl,dr=(np.ones(nl),np.ones(nr)) if coordinate_units is None else (
            np.asarray(coordinate_units[0],float).copy(),np.asarray(coordinate_units[1],float).copy())
        if (dl.shape!=(nl,) or dr.shape!=(nr,) or not np.isfinite(dl).all()
                or not np.isfinite(dr).all() or np.any(dl<=0) or np.any(dr<=0)):
            raise ValueError('GATE shared_general_equilibration: positive finite equation-unit factors required')
        es=dl[:,None]*E*dr[None,:]
        aa=dl[:,None]*(A/omega)*dr[None,:]
        for _ in range(8):
            magnitude=np.hypot(np.abs(es),np.abs(aa))
            rows=magnitude.max(axis=1);cols=magnitude.max(axis=0)
            rl=1/np.sqrt(np.where(rows>0,rows,1.))
            rr=1/np.sqrt(np.where(cols>0,cols,1.))
            dl*=rl;dr*=rr
            es*=rl[:,None];es*=rr[None,:]
            aa*=rl[:,None];aa*=rr[None,:]
        if not all(np.isfinite(a).all() for a in (es,aa,dl,dr)):
            raise ValueError('GATE shared_general_equilibration: unrepresentable frequency coordinates')
        y, singular, xh = la.svd(es, full_matrices=False, check_finite=False)
        n = min(int(max_order), int(np.count_nonzero(singular > rank_tolerance * singular[0])))
        if not n:
            raise ValueError("GATE shared_general_rank: nonzero response has no resolved frequency span")
        y, x = dl[:,None]*y[:, :n], dr[:,None]*xh[:n].conj().T
        equilibration=dict(method='unit-aware-combined-pencil-Ruiz-infinity-8',frequency_scale_ry=omega,
            left_scale_range=[float(dl.min()),float(dl.max())],right_scale_range=[float(dr.min()),float(dr.max())],
            left_scale_sha256=hashlib.sha256(dl.tobytes()).hexdigest(),
            right_scale_sha256=hashlib.sha256(dr.tobytes()).hexdigest(),
            held_space_coordinates='original-pencil')
        del es,aa,magnitude
        svd_seconds=time.perf_counter()-svd_started
    else:
        equilibration=dict(method='held-original-pencil-spaces',held_space_coordinates='original-pencil')
        x, y = (np.asarray(fixed_spaces[key], complex) for key in ("right", "left"))
        n = x.shape[1]
        if (x.shape != (nr, n) or y.shape != (nl, n) or not n
                or n>min(int(max_order),nl,nr)):
            raise ValueError("GATE shared_general_basis: fixed spaces differ from current pencil")
    eigen_started=time.perf_counter()
    er, ar = y.conj().T @ E @ x, y.conj().T @ A @ x
    balance_started=time.perf_counter()
    ub, sb, vbh = la.svd(er,full_matrices=False,check_finite=False)
    if (not np.isfinite(sb).all() or not sb.size or sb[0]<=0
            or sb[-1]<=rank_tolerance*sb[0]):
        # A held frequency space is NEVER silently rank-truncated. Its
        # current descriptor must resolve every previously retained state.
        raise ValueError("GATE shared_general_balance: current descriptor is unresolved or singular")
    d=1/np.sqrt(sb)
    right_balance=vbh.conj().T*d[None,:]
    left_balance=ub*d[None,:]
    balanced=left_balance.conj().T@ar@right_balance
    balance_seconds=time.perf_counter()-balance_started
    geev_started=time.perf_counter()
    poles, wl, vr = la.eig(balanced,left=True,right=True,check_finite=False)
    geev_seconds=time.perf_counter()-geev_started
    if not np.isfinite(poles).all():
        raise ValueError("GATE shared_general_poles: descriptor has an unresolved infinite pole")
    metric = wl.conj().T @ vr
    diagonal = np.diag(metric)
    norm = np.linalg.norm(wl, axis=0) * np.linalg.norm(vr, axis=0)
    if np.any(np.abs(diagonal) <= 128 * np.finfo(float).eps * norm):
        raise ValueError("GATE shared_general_poles: defective or unresolved eigenvector normalization")
    # Left/right ordinary eigenvectors are biorthogonal for distinct poles.
    # In a degenerate eigenspace use its full metric, never per-column division.
    inverse = la.solve(metric, np.eye(n, dtype=complex), check_finite=False)
    original_right=right_balance@vr
    output = x @ original_right
    inputs = y @ left_balance @ wl @ inverse.conj().T
    # Check the ORIGINAL descriptor, not just the better-scaled ordinary
    # equation. Balancing is an invertible coordinate change, not pole repair.
    residual = np.linalg.norm(ar @ original_right - (er @ original_right) * poles[None]) / max(
        np.linalg.norm(ar @ original_right), np.finfo(float).tiny)
    offdiag = metric - np.diag(diagonal)
    # Mixing distinct eigenvalues would change the transfer; only roundoff
    # off-diagonal metric is acceptable outside exactly degenerate blocks.
    commutator = inverse @ np.diag(poles) - np.diag(poles) @ inverse
    commutator_error = np.linalg.norm(commutator) / max(
        np.linalg.norm(inverse) * np.max(np.abs(poles)), np.finfo(float).tiny)
    if residual > 1e-9 or commutator_error > 1e-9:
        raise ValueError("GATE shared_general_eigenvectors: unresolved general modal reconstruction")
    order = np.lexsort((poles.imag, poles.real))
    return dict(poles_ry=poles[order], output_map=output[:, order], input_map=inputs[:, order],
                spaces=dict(right=x, left=y),
                receipt=dict(order=n, workspace_bytes=price, rank_tolerance=rank_tolerance,
                             eigen_residual=float(residual), metric_offdiagonal=float(np.linalg.norm(offdiag)),
                             modal_commutator_error=float(commutator_error),
                             max_growth_ry=float(np.max(poles.imag)),
                             balance_seconds=balance_seconds,ordinary_eigen_seconds=geev_seconds,
                             current_descriptor_condition=float(sb[0]/sb[-1]),
                             svd_seconds=svd_seconds,eigen_seconds=time.perf_counter()-eigen_started,
                             host_reduce_seconds=time.perf_counter()-started,
                             numerical_spaces="two-sided-descriptor-svd",
                             frequency_equilibration=equilibration,
                             eigensolver="current-descriptor-balanced-ordinary-geev",
                             construction="general-two-sided-tangential-loewner"))


def _equilibrium_state_subset(g, h, output_gram, *, units, frequency_scale,
                              rank_tolerance, rounding):
    """RRQR independent ORIGINAL states using joint metric/output evidence.

    This chooses a Galerkin basis, not a positive-H spectrum. No input
    matrix or sample is repaired; the selected current H must still pass
    the strict reducer and every original full-port sample remains checked.
    O†O supplies output evidence without a host microscopic-port gather.
    Its roundoff-resolved square root is ONLY a rank witness.
    """
    import hashlib
    import numpy as np
    import scipy.linalg as la
    d=np.asarray(units,float)
    try:
        frequency_scale=float(frequency_scale)
    except (TypeError,ValueError) as error:
        raise ValueError('GATE shared_equilibrium_selection: explicit frequency scale required') from error
    if (d.shape!=(g.shape[0],) or not np.isfinite(d).all() or np.any(d<=0)
            or not np.isfinite(frequency_scale) or frequency_scale<=0):
        raise ValueError('GATE shared_equilibrium_selection: positive common state units/energy scale required')
    gs=d[:,None]*g*d[None];hs=d[:,None]*h*d[None]/frequency_scale
    ns=d[:,None]*output_gram*d[None]
    if any(not np.isfinite(a).all() for a in (gs,hs,ns)):
        raise ValueError('GATE shared_equilibrium_selection: nonfinite scaled rank evidence')
    eigen,u=la.eigh(ns,check_finite=False)
    if not np.isfinite(eigen).all() or not np.isfinite(u).all():
        raise ValueError('GATE shared_equilibrium_output: nonfinite output eigenpairs')
    scale=max(float(np.max(abs(eigen))),np.finfo(float).tiny)
    if eigen[0]<-rounding*scale:
        raise ValueError('GATE shared_equilibrium_output: scaled output Gram has resolved negative eigenvalue')
    resolved=eigen>rounding*scale
    factor=np.sqrt(eigen[resolved])[:,None]*u[:,resolved].conj().T
    blocks=[a/max(la.norm(a),np.finfo(float).tiny) for a in (gs,hs,factor)]
    evidence=np.concatenate(blocks,axis=0)
    _,r,pivot=la.qr(evidence,mode='economic',pivoting=True,check_finite=False)
    diagonal=abs(r.diagonal())
    cutoff=(rank_tolerance+rounding)*float(np.max(diagonal,initial=0.))
    below=np.flatnonzero(diagonal<=cutoff)
    rank=int(below[0]) if len(below) else len(diagonal)
    selected=np.asarray(pivot[:rank],np.int64)
    if not rank:
        raise ValueError('GATE shared_equilibrium_selection: no resolved common state')
    receipt=dict(method='original-state-joint-RRQR-v1',original_order=g.shape[0],selected_order=rank,
        selected_columns_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
        rank_tolerance=float(rank_tolerance),relative_cutoff=float(rank_tolerance+rounding),
        smallest_kept_pivot=float(diagonal[rank-1]),
        largest_discarded_pivot=float(np.max(diagonal[rank:],initial=0.)),
        output_gram_roundoff_rank=int(np.count_nonzero(resolved)),
        discarded_output_gram_norm=float(la.norm(eigen[~resolved])),
        output_gram_roundoff_bound=float(rounding*scale),
        frequency_scale_ry=float(frequency_scale),
        scope='numerical state independence; NOT an operator-W sample-error certificate',
        required_validation='all original full-matrix W fit/held and moment checks')
    return selected,receipt


def equilibrium_metric_workspace_bytes(full_order, selected_order):
    """Host array envelope before extended selected-metric copies.

    Eight full complex128 squares cover original/normalized pencils and
    selection storage; 32 selected squares cover sixteen complex256 cores,
    eigensolver/maps/scratch; four full-by-selected panels cover selectors
    and returned maps. This fits the existing admitted general envelope.
    No microscopic port matrix is priced or gathered here.
    """
    n,m=int(full_order),int(selected_order)
    if not 0<m<=n:
        raise ValueError('GATE shared_equilibrium_capacity: invalid selected metric shape')
    return 16*(8*n*n+32*m*m+4*n*m)


def _equilibrium_extended_product(x,y):
    """Complex extended product, inner32 Kahan sums; small host pencils only."""
    import numpy as np
    out=np.empty((x.shape[0],y.shape[1]),np.clongdouble)
    for first in range(0,len(x),64):
        rows=x[first:first+64]
        total=np.zeros((len(rows),y.shape[1]),np.clongdouble)
        compensation=np.zeros_like(total)
        for k in range(0,x.shape[1],32):
            term=rows[:,k:k+32]@y[k:k+32]
            corrected=term-compensation;updated=total+corrected
            compensation=(updated-total)-corrected;total=updated
        out[first:first+64]=total
    return out


def _equilibrium_original_null_span(v):
    """Thin QR of an ORIGINAL-coordinate null span, independent of units."""
    import numpy as np
    import scipy.linalg as la
    if not np.isfinite(v).all():
        raise ValueError('GATE shared_equilibrium_null: nonfinite original null span')
    q=la.qr(v,mode='economic',check_finite=False)[0] if v.shape[1] else v
    if not np.isfinite(q).all():
        raise ValueError('GATE shared_equilibrium_null: nonfinite original null QR')
    return q


def _equilibrium_accurate_metric(g,h, *,full_order,max_workspace_bytes,rounding):
    """All-selected current-H congruence with unchanged 1e-9 validation.

    Extended initialization, TWO fixed current-H Newton congruences, then
    rounded original maps. Neither H nor a pole spectrum is truncated.
    The SPD witness concerns the computed input pencil, not sample error.
    """
    import numpy as np
    import scipy.linalg as la
    n=len(h);price=equilibrium_metric_workspace_bytes(full_order,n)
    if price>int(max_workspace_bytes):
        raise ValueError('GATE shared_equilibrium_capacity: extended metric exceeds admitted host envelope before copies')
    ext=np.finfo(np.longdouble);fp=np.finfo(float);started=time.perf_counter();phases={}
    if ext.nmant<=fp.nmant:
        raise ValueError('GATE shared_equilibrium_precision: extended host precision required')
    def gamma(k):
        v=np.longdouble(k)*ext.eps
        if v>=1:raise ValueError('GATE shared_equilibrium_precision: invalid roundoff envelope')
        return v/(1-v)
    def norm(a):
        a=np.asarray(a,np.clongdouble)
        return np.sqrt(np.sum(a.real*a.real+a.imag*a.imag,dtype=np.longdouble))/(1-gamma(8*a.size+8))
    mm=_equilibrium_extended_product
    diagonal=h.diagonal().real
    if np.any(diagonal<=0):
        raise ValueError('GATE shared_equilibrium_metric: nonpositive/negative H diagonal; no PSD repair')
    d=(1/np.sqrt(diagonal)).astype(np.longdouble)
    hx=d[:,None]*np.asarray(h,np.clongdouble)*d[None]
    if not np.isfinite(hx).all():
        raise ValueError('GATE shared_equilibrium_metric: nonfinite scaled current H')
    lower=np.zeros_like(hx)
    for j in range(n):
        pivot=hx[j,j].real-np.sum(abs(lower[j,:j])**2,dtype=np.longdouble)
        if not np.isfinite(pivot) or pivot<=0:
            raise ValueError('GATE shared_equilibrium_metric: nonpositive/negative H pivot; no PSD repair')
        lower[j,j]=np.sqrt(pivot)
        if j+1<n:lower[j+1:,j]=(hx[j+1:,j]-lower[j+1:,:j]@lower[j,:j].conj())/lower[j,j]
    inverse=np.zeros_like(lower)
    for j in range(n):
        inverse[j,j]=1/lower[j,j]
        if j:inverse[j,:j]=-(lower[j,:j]@inverse[:j,:j])/lower[j,j]
    phases['factor_inverse']=time.perf_counter()-started;stage=time.perf_counter()
    ll=mm(lower,lower.conj().T);bl=mm(inverse,lower)
    delta=norm(hx-ll)+gamma(16*n+64)*norm(lower)**2
    delta+=gamma(8)*(2*norm(hx)+norm(ll))
    r=norm(bl-np.eye(n))+gamma(16*n+64)*norm(inverse)*norm(lower)
    r+=gamma(8)*(norm(bl)+np.sqrt(np.longdouble(n)))
    singular=max(np.longdouble(0),1-r)/norm(inverse)
    positive=singular**2-delta
    positive-=gamma(16)*(singular**2+abs(delta))
    lower_bound=float(np.nextafter(float(positive),-np.inf))
    if (not all(np.isfinite(a) for a in (delta,r,singular,positive))
            or not r<1 or not lower_bound>0):
        raise ValueError('GATE shared_equilibrium_metric: current computed H lacks positive residual witness')
    del ll,bl
    phases['SPD_witness']=time.perf_counter()-stage;stage=time.perf_counter()
    z=np.asarray(d[:,None]*inverse.conj().T,np.complex128)
    del lower,inverse
    history=[]
    for _ in range(2):
        y=np.asarray(z,np.clongdouble)/d[:,None]
        s=mm(y.conj().T,mm(hx,y))
        error=norm(s-np.eye(n));history.append(float(error))
        if not np.isfinite(error) or error>=np.longdouble(.5):
            raise ValueError('GATE shared_equilibrium_congruence: current H outside fixed Newton near-I domain')
        s=(s+s.conj().T)*np.longdouble(.5)
        correction=np.longdouble(1.5)*np.eye(n,dtype=np.clongdouble)-np.longdouble(.5)*s
        z=np.asarray(d[:,None]*mm(y,correction),np.complex128)
    del y,s,correction
    phases['fixed_two_congruences']=time.perf_counter()-stage;stage=time.perf_counter()
    zx=np.asarray(z,np.clongdouble);he=np.asarray(h,np.clongdouble);ge=np.asarray(g,np.clongdouble)
    white=mm(zx.conj().T,mm(he,zx))
    y=zx/d[:,None];balanced=mm(y.conj().T,mm(hx,y))
    k=mm(zx.conj().T,mm(ge,zx))
    del ge
    kd=np.asarray((k+k.conj().T)*np.longdouble(.5),np.complex128)
    if not np.isfinite(kd).all():
        raise ValueError('GATE shared_equilibrium_eigenvectors: nonfinite current G metric')
    mu,rotation=la.eigh(kd,check_finite=False)
    if not np.isfinite(mu).all() or not np.isfinite(rotation).all():
        raise ValueError('GATE shared_equilibrium_eigenvectors: nonfinite metric eigenpairs')
    del kd
    vectors=np.asarray(mm(zx,np.asarray(rotation,np.clongdouble)),np.complex128)
    if not np.isfinite(vectors).all():
        raise ValueError('GATE shared_equilibrium_eigenvectors: nonfinite original rounded maps')
    del zx
    vx=np.asarray(vectors,np.clongdouble)
    final=mm(vx.conj().T,mm(he,vx))
    y=vx/d[:,None];final_balanced=mm(y.conj().T,mm(hx,y))
    errors=[float(norm(a-np.eye(n))) for a in (white,balanced,final,final_balanced)]
    if not np.isfinite(errors).all() or max(errors)>1e-9:
        raise ValueError('GATE shared_equilibrium_eigenvectors: original rounded-map metric exceeds 1e-9')
    del balanced,final,final_balanced,y,vx,hx,he
    rr=np.asarray(rotation,np.clongdouble)
    kr=mm(k,rr);hr=mm(white,rr)*mu[None]
    residual=float(norm(kr-hr)/max(norm(kr),np.finfo(np.longdouble).tiny))
    if not np.isfinite(residual) or residual>1e-9:
        raise ValueError('GATE shared_equilibrium_eigenvectors: original generalized residual exceeds 1e-9')
    phases['metric_eigen_validation']=time.perf_counter()-stage
    binary64_white=float(la.norm(z.conj().T@h@z-np.eye(n)))
    if not np.isfinite(binary64_white):
        raise ValueError('GATE shared_equilibrium_eigenvectors: nonfinite original metric diagnostic')
    receipt=dict(metric_solver=EQUILIBRIUM_METRIC_SOLVER,
        metric_workspace_bytes=price,extended_mantissa_bits=int(ext.nmant),
        computed_scaled_H_min_lower=lower_bound,computed_H_SPD_residual_upper=float(np.nextafter(float(delta),np.inf)),
        inverse_BL_residual_upper=float(np.nextafter(float(r),np.inf)),
        fixed_congruence_count=2,Newton_metric_errors_before=history,
        original_H_extended_whitening_errors=errors,
        original_H_binary64_whitening_error=binary64_white,
        eigen_residual=residual,whitening_error=max(errors),metric_phase_seconds=phases,
        metric_scope='computed current H, not physical sample-error or full-W admission')
    return z,mu,vectors,receipt


def _reduce_equilibrium_pencil(G, H, output_gram, *, structure, max_order,
                               rank_tolerance, max_workspace_bytes, fixed_spaces):
    """Opt-in same-state ``O[(z+i gamma)G-H]^-1 O.H`` host reduction.

    G/H and O.H O are small square frequency matrices, never microscopic
    ports. The caller must authenticate the equilibrium definite-member
    premise; this function does not classify a physical density. The native
    builder selects it only for its authenticated equilibrium subset. Null directions require current
    metric/output loss witnesses, not a pole or positive-eigenvalue repair.
    """
    import numpy as np
    import scipy.linalg as la
    started=time.perf_counter()
    shape=np.shape(G)
    if (not isinstance(structure,dict) or structure.get('kind')!='equilibrium_definite'
            or len(shape)!=2 or shape[0]!=shape[1] or shape[0]<1 or np.shape(H)!=shape
            or output_gram is None or np.shape(output_gram)!=shape
            or int(max_order)<1 or not 0<float(rank_tolerance)<1):
        raise ValueError('GATE shared_equilibrium_pencil: square G/H and explicit output Gram required')
    price=general_pencil_workspace_bytes(shape)
    if price>int(max_workspace_bytes):
        raise ValueError('GATE shared_equilibrium_capacity: host workspace exceeds bound before copies/eigh')
    gamma=float(structure.get('gamma_ry',np.nan))
    if not np.isfinite(gamma) or gamma<0:
        raise ValueError('GATE shared_equilibrium_gamma: finite nonnegative uniform lag required')
    g,h,n=(np.asarray(a,np.complex128) for a in (G,H,output_gram))
    if any(not np.isfinite(a).all() for a in (g,h,n)):
        raise ValueError('GATE shared_equilibrium_pencil: nonfinite frequency matrices')
    eps=np.finfo(float).eps;size=shape[0];rounding=128*size*eps
    for a in (g,h,n):
        if la.norm(a-a.conj().T)>rounding*max(la.norm(a),np.finfo(float).tiny):
            raise ValueError('GATE shared_equilibrium_hermiticity: non-Hermitian frequency matrix')
    # Only roundoff antisymmetry is normalized; no spectrum is floored.
    g,h,n=((a+a.conj().T)*.5 for a in (g,h,n))
    hn=la.norm(h);gn=la.norm(g);on=la.norm(n)
    if not np.isfinite([hn,gn,on]).all():
        raise ValueError('GATE shared_equilibrium_pencil: nonfinite pencil norms')
    nev=la.eigvalsh(n,check_finite=False)
    if not np.isfinite(nev).all():
        raise ValueError('GATE shared_equilibrium_output: nonfinite output spectrum')
    if nev[0]<-rounding*max(on,np.finfo(float).tiny):
        raise ValueError('GATE shared_equilibrium_output: output Gram has resolved negative eigenvalue')
    held=fixed_spaces is not None
    selection=None
    accurate=structure.get('metric_solver')==EQUILIBRIUM_METRIC_SOLVER
    if structure.get('metric_solver') is not None and (not accurate
            or structure.get('state_selection')!='principal-rrqr-v1'):
        raise ValueError('GATE shared_equilibrium_solver: explicit original-column principal policy required')
    if held:
        base=np.asarray(fixed_spaces['right'],complex)
        if (base.ndim!=2 or base.shape[0]!=size or not 0<base.shape[1]<=int(max_order)
                or not np.array_equal(base,np.asarray(fixed_spaces['left'])) or not np.isfinite(base).all()):
            raise ValueError('GATE shared_equilibrium_basis: held left/right spaces must be the SAME congruence')
        if accurate:
            columns=np.argmax(abs(base),axis=0)
            selector=np.zeros_like(base);selector[columns,np.arange(len(columns))]=1
            if len(np.unique(columns))!=len(columns) or not np.array_equal(base,selector):
                raise ValueError('GATE shared_equilibrium_basis: original-column P_I selector required; legacy dense spaces cannot be reused')
        if not accurate:hp=base.conj().T@h@base
    else:
        base=np.eye(size,dtype=complex)
        if structure.get('state_selection') is not None:
            if structure['state_selection']!='principal-rrqr-v1':
                raise ValueError('GATE shared_equilibrium_selection: unknown original-state selection policy')
            columns,selection=_equilibrium_state_subset(g,h,n,units=structure.get('state_units'),
                frequency_scale=structure.get('frequency_scale_ry'),
                rank_tolerance=float(rank_tolerance),rounding=rounding)
            base=base[:,columns]
        if not accurate:hp=base.conj().T@h@base
    def output_loss(v):
        loss=float(la.norm(v.conj().T@n@v)) if v.shape[1] else 0.
        if not np.isfinite(loss):
            raise ValueError('GATE shared_equilibrium_output: nonfinite original output loss')
        return loss
    accurate_receipt={}
    if accurate:
        rank=base.shape[1]
        if rank>int(max_order):
            raise ValueError('GATE shared_equilibrium_order: selected state space exceeds budget; no truncation to cap')
        # Exact original selectors avoid a poorly conditioned dense unit
        # reinterpretation and allow the core to validate ORIGINAL entries.
        gp=g[np.ix_(columns,columns)];hp=h[np.ix_(columns,columns)]
        metric_price=equilibrium_metric_workspace_bytes(size,rank)
        if metric_price>price:
            raise ValueError('GATE shared_equilibrium_capacity: extended cores exceed metadata-admitted batch envelope')
        small_z,mu,small_vectors,accurate_receipt=_equilibrium_accurate_metric(gp,hp,
            full_order=size,max_workspace_bytes=min(price,int(max_workspace_bytes)),rounding=rounding)
        z=np.zeros((size,rank),complex);z[columns]=small_z
        vectors=np.zeros_like(z);vectors[columns]=small_vectors
        spaces=base;null=np.empty((size,0),complex);original_null=null;eigen=None
    else:
        eigen,u=la.eigh(hp,check_finite=False)
        if not np.isfinite(eigen).all() or not np.isfinite(u).all():
            raise ValueError('GATE shared_equilibrium_metric: nonfinite H eigenpairs')
        hs=max(float(np.max(abs(eigen))),np.finfo(float).tiny)
        if eigen[0]<-rounding*hs:
            raise ValueError('GATE shared_equilibrium_metric: resolved negative H; no PSD repair')
        cut=rounding*hs if held else max(rounding,float(rank_tolerance))*hs
        keep=eigen>cut
        null=base@u[:,~keep]
        if held and not np.all(keep):
            raise ValueError('GATE shared_equilibrium_basis: current held H is unresolved; no held rank drop')
        original_null=_equilibrium_original_null_span(null)
        g_null=float(la.norm(g@original_null))
        if not np.isfinite(g_null):
            raise ValueError('GATE shared_equilibrium_null: nonfinite original G null loss')
        if null.shape[1] and (g_null>(rank_tolerance+rounding)*max(gn,np.finfo(float).tiny)
                or output_loss(original_null)>(rank_tolerance**2+rounding)*max(on,np.finfo(float).tiny)):
            raise ValueError('GATE shared_equilibrium_null: live H-null/unresolved state')
        rank=int(np.count_nonzero(keep))
    if rank>int(max_order):
        raise ValueError('GATE shared_equilibrium_order: resolved state space exceeds budget; no truncation to cap')
    if not accurate:
        z=base@(u[:,keep]/np.sqrt(eigen[keep])[None])
        spaces=base if held else z
    if not rank:
        return dict(poles_ry=np.empty(0,complex),output_map=z,input_map=z,
            spaces=dict(right=spaces,left=spaces),receipt=dict(order=0,construction='equilibrium-definite-value',
            workspace_bytes=price,host_reduce_seconds=time.perf_counter()-started))
    if not accurate:
        metric=z.conj().T@g@z
        mu,rotation=la.eigh((metric+metric.conj().T)*.5,check_finite=False)
        vectors=z@rotation
    if not np.isfinite(mu).all() or not np.isfinite(vectors).all():
        raise ValueError('GATE shared_equilibrium_eigenvectors: nonfinite eigenpairs or maps')
    bound=rounding*max(float(np.max(abs(mu))),np.finfo(float).tiny)
    finite=abs(mu)>bound
    if np.any(~finite):
        v=_equilibrium_original_null_span(vectors[:,~finite])
        if output_loss(v)>(rank_tolerance**2+rounding)*max(on,np.finfo(float).tiny):
            raise ValueError('GATE shared_equilibrium_infinite: live zero/unresolved mu gives constant feedthrough')
    if accurate:
        residual=accurate_receipt['eigen_residual'];whitening=accurate_receipt['whitening_error']
    else:
        gp=z.conj().T@g@z;hp=z.conj().T@h@z
        residual=la.norm(gp@rotation-(hp@rotation)*mu[None])/max(la.norm(gp@rotation),np.finfo(float).tiny)
        whitening=la.norm(hp-np.eye(rank))
    if not np.isfinite([residual,whitening]).all() or residual>1e-9 or whitening>1e-9:
        raise ValueError('GATE shared_equilibrium_eigenvectors: unresolved metric reconstruction')
    mu,vectors=mu[finite],vectors[:,finite]
    poles=1/mu-1j*gamma
    t=vectors/np.sqrt(abs(mu))[None]
    if not np.isfinite(poles).all() or not np.isfinite(t).all():
        raise ValueError('GATE shared_equilibrium_eigenvectors: nonfinite poles or normalized maps')
    order=np.argsort(poles.real,kind='stable')
    receipt=dict(order=len(poles),held_state_rank=rank,
        construction='equilibrium-definite-value',spectral_variable='z+i_gamma',gamma_ry=gamma,
        rank_tolerance=float(rank_tolerance),H_min_eigenvalue=None if accurate else float(eigen[0]),
        H_null_output_squared=output_loss(original_null),null_output_basis='original-coordinate-orthonormal',
        H_null_roundoff_allowance=rounding*on,zero_mu_output_checked=True,
        numerical_spaces='same-state-H-congruence',state_selection=selection,
        eigen_residual=float(residual),whitening_error=float(whitening),
        workspace_bytes=price,host_reduce_seconds=time.perf_counter()-started)
    if accurate:
        import hashlib
        receipt.update(accurate_receipt,space_representation='original-column-selector-v1',
            original_selector_columns_sha256=hashlib.sha256(np.asarray(columns,np.int64).tobytes()).hexdigest())
    return dict(poles_ry=poles[order],output_map=t[:,order],input_map=(t*np.sign(mu)[None])[:,order],
        spaces=dict(right=spaces,left=spaces),receipt=receipt)


def constructor_route(meta, config, recipe, *, mesh_xy, ledger, upstream, ordered,
                      odd_moments, nq):
    """Resolve local or whole-mesh parent execution for one map's bank.

    The single admission the constructor applies before its first bank read,
    also consulted by the map owner before the bank exists (bank residence).
    Returns ``(execution, receipt, column_extent, selection_faces,
    moment_fields)``; ``upstream`` names the accepted live reservations.
    The selection holds the dense fitted samples (W, dW/ds), the moments and
    every line sample's stored panels (``selection_face_count``).
    """
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_execution import constructor_execution, line_panel_count, selection_face_count
    from file_io.shared_pole_store import line_panel_geometry

    column_extent = port_extent(mesh_xy)
    moment_fields = ("M0", "M1", "M2", "M3") if odd_moments else ("M1", "M3")
    families, states = line_panel_geometry(meta, ordered=ordered)
    faces = selection_face_count(recipe, n=int(meta.n_rmu_padded), logical_n=int(meta.n_rmu),
        states=states, rows=int(meta.n_rmu_padded), dense_fields=2, moment_fields=len(moment_fields),
        column_extent=column_extent)
    execution, receipt = constructor_execution(
        meta, linalg_resolution({"linalg": config.backend.linalg}), recipe,
        mesh=mesh_xy, ledger=ledger, upstream=upstream, ordered=ordered,
        odd_moments=odd_moments, selection_faces=faces,
        sample_batch=len(recipe["fit_ids"]) - line_panel_count(recipe), parent_count=int(nq),
        column_extent=column_extent,
        ritz_budget=recipe.get("pole_budget") if ordered else None)
    return execution, receipt, column_extent, faces, moment_fields


def construct_shared_poles(bank, moments, meta, config, *, mesh_xy, output, residence=None):
    """Construct and write a current-state, bounded-batch real-pole model.

    Parameters
    ----------
    bank : mapping
        Authenticated resource descriptor: ``path``, ``identity``, ``tables``,
        ``coulomb`` (response-owner authenticated Coulomb resource). Dense
        workspace is queried from the resolved service plans. A producer
        certificate is carried as ``rule_receipt``. No dense bank is passed
        as a jit argument. The recipe is read only from meta.
    moments : mapping
        ``path`` to committed physical M1/M3 in the same scratch schema.
    meta : Meta
        Current packed centroid basis, scalar representation and the once-
        resolved ``shared_pole_recipe`` with current state identities.
    config : LorraxConfig
        Cached dense policy is read once, before any per-q plans.
    mesh_xy : Mesh
        Supplied named x/y mesh, never reconstructed by this driver.
    output : path-like
        New immutable compact model file, written by the store owner.
    residence : callable, optional
        The map owner's model residence rule, ``residence(nq, width)`` ->
        ``(ResidentSectorModel or None, receipt)``, asked once the model's
        column width is known and before it is written. A resident target
        replaces ``output`` and its stage stays reserved for the caller; the
        receipt is returned as ``model_residence``.

    Returns
    -------
    dict
        Versioned construction receipt with all gate rows, per-q diagnostics,
        capacity prices and the store header/digest. Structural failures raise
        before the affected q is written; a partial file is never finalized.
    """
    walls = {}

    @contextmanager
    def phase(name):
        """One named constructor stage; its wall adds to ``walls[name]`` (the per-map summary)."""
        started = time.monotonic()
        if getattr(meta, "shared_pole_capacity", None) is not None:
            meta.shared_pole_capacity.at_phase()
        with timing.section("spole." + name):
            yield
        walls[name] = walls.get(name, 0.0) + time.monotonic() - started

    with phase("entry"):
        import numpy as np
        from jax.sharding import NamedSharding, PartitionSpec as P
        from file_io.shared_pole_store import (
            charge_representation, validate_shared_pole_bank,
            RoundReader, read_round_samples, write_shared_pole_model,
        )
        from gw.gw_config import linalg_resolution
        from common.staged_reshard import face_to_batch_reshard
        from gw.shared_pole_local import (batch_to_face, canonical_factors, check_round, face_rows,
                                          own_extent_receipts, reduce_round, round_tables,
                                          recipe_panel_widths, recipe_infinity_width, pad_states,
                                          carrier_history)
        from gw.shared_pole_capacity import round_padding_output_bytes
        from gw.shared_pole_recipe import (
            build_construction_row, charge4_gates, construction_receipt,
            shared_real_pole_gates_v1_r3b as gates,
            shared_real_pole_gates_ordered_v1,
        )
        from common.units import RYD_TO_EV
        from gw.w_isdf import response_coulomb_powers

    with phase("setup"):
        recipe = meta.shared_pole_recipe
        # Time-reversal-broken scalar states take the ordered particle-hole route.
        ordered = not bool(bank["tables"]["sym"].trs_allowed)
        # Every charge carrier contracts to one mu x mu CC operator.
        if not charge_representation(meta):
            raise ValueError("GATE shared_pole_representation: expected an authenticated charge carrier")
        charge4 = recipe.get("charge_operator") == "four-component-spin-traced-v1"
        if (int(meta.nspinor) == 4) != charge4:
            raise ValueError("GATE shared_pole_representation: charge4 recipe and carrier disagree")
        if charge4:
            gates = charge4_gates(ordered)
        elif ordered:
            gates = shared_real_pole_gates_ordered_v1
        from gw.shared_pole_recipe import table_hash
        if recipe["gate_hash"] != table_hash(gates):
            raise ValueError("GATE shared_pole_representation: recipe gate table differs from charge carrier")
        resolution = linalg_resolution({"linalg": config.backend.linalg})
        identity = bank["identity"]
        ledger = meta.shared_pole_capacity
        upstream = ledger.live_stages
        n = int(meta.n_rmu_padded)
        # Every device byte this construction admits is priced here; the driver
        # below keeps only `budget.retained_panels` and `budget.batch_width`
        # current as it moves from selection to reduction to the model checks.
        header = validate_shared_pole_bank(bank["path"], expected_identity=identity,
                                           mesh_xy=mesh_xy, require_complete=True)
        moment_header = validate_shared_pole_bank(moments["path"], expected_identity=identity,
                                                  mesh_xy=mesh_xy)
        # The stored plan must bind the current physical points and role census.
        stored_recipe = header["recipe"]
        for name in ("recipe_hash", "gate_hash", "fit_ids", "held_ids", "role", "role_codes",
                     "distinct_id", "held", "support_pair", "census"):
            current = recipe[name]
            previous = stored_recipe[name]
            if isinstance(current, np.ndarray):
                equal = np.array_equal(current, previous)
            else:
                equal = current == previous
            if not equal:
                raise ValueError(f"GATE shared_pole_bank_state: got: stale {name}; want: current resolved recipe; why: no frozen SC inputs")
        for sample_id in (*recipe["fit_ids"], *recipe["held_ids"]):
            if _sample_point(recipe, sample_id) != _sample_point(stored_recipe, sample_id):
                raise ValueError("GATE shared_pole_bank_state: got: changed z; want: current physical sample point; why: recipe hashes alone do not bind resolved points")
        if bool(header.get("ordered", False)) != ordered:
            raise ValueError(f"GATE shared_pole_representation: got: bank ordered={bool(header.get('ordered', False))} with trs_allowed={not ordered}; want: an ordered bank exactly when time reversal is broken (both orientations, -q as its own parent); why: particle-hole pairing across parents")
        # Odd z-moments M0/M2 certify the ordered infinity block; a finite-state
        # ordered bank builds without it and records the moments NOT_MEASURED.
        odd_moments = ordered and bool(header.get("odd_moments", False))
        logical_n = int(meta.n_rmu)
        from gw.shared_pole_execution import constructor_side_upper_bound
        execution, execution_receipt, column_extent, selection_faces, moment_fields = constructor_route(
            meta, config, recipe, mesh_xy=mesh_xy, ledger=ledger, upstream=upstream,
            ordered=ordered, odd_moments=odd_moments,
            nq=int(header['bank_shape']['nq']))
        # Line supports off the imaginary axis arrive as their stored states;
        # the dense fitted samples (imaginary axis) are selected per round.
        line_lo, line_hi = (int(v) for v in header["line_panels"]["sample_span"])
        dense_fit = [int(i) for i in recipe["fit_ids"] if not line_lo <= int(i) < line_hi]
        budget = ConstructorCapacity(meta, resolution, mesh_xy=mesh_xy,
                                     ledger=ledger, upstream=upstream,
                                     execution=execution)
        budget.ritz_budget = recipe.get("pole_budget") if ordered and execution == 'local' else None
        # A face constructor keeps the Coulomb eigensolve on the same complete
        # mesh even when the deck's default dense policy is local.
        coulomb_config = ({'linalg': 'distributed'}
                          if execution == 'face' else config)
        conservative_side = constructor_side_upper_bound(
            recipe, ordered=ordered, odd_moments=odd_moments,
            logical_n=logical_n, column_extent=column_extent)
        # A face plan waits for its room (face_eigh_room, below).
        eig = budget.eigenplan(n) if execution != 'face' else None
        receipts, factors, store_poles, store_counts, placed = {}, [], [], [], []
        receipt_entry_start = 0
        held_ids = [int(i) for i in recipe["held_ids"]]
        if not held_ids:
            raise ValueError("GATE shared_pole_held: got: no held samples; want: at least one held support; why: the model checks compare W and dW/ds there")
        held_lo, held_hi = min(held_ids), max(held_ids) + 1
        nq = int(header["bank_shape"]["nq"])
        # A local round runs one parent per rank from its sample read to its
        # sorted model. A face round runs a budget-sized batch of parents over
        # all ranks. Both take fixed-width rounds from the one schedule
        # (``parent_rounds``); a short last round repeats its last parent.
        face_batch = 1
        if execution == 'face':
            from gw.shared_pole_capacity import check_eigh_room, face_eigh_room
            from gw.shared_pole_execution import face_batch_width, face_ritz_carrier, face_round_compiled_bytes
            keep_budget = recipe.get("pole_budget")
            # The conservative round's infinity carrier and its block in the pencil side
            # (constructor_side_upper_bound), for the width's compiled admission.
            infinity_cap = column_extent(max(1, int(recipe["infinity_width"])))
            infinity_block = (2 * infinity_cap if odd_moments else 0) if ordered else infinity_cap
            sizing = dict(rows=n, side=conservative_side,
                          carrier=face_ritz_carrier(mesh_xy, keep_budget) if ordered else None)
            face_batch, execution_receipt['face_batch'] = face_batch_width(
                meta, resolution, mesh=mesh_xy, ledger=ledger, upstream=upstream,
                side=conservative_side, nq=int(header['n_q_irr']),
                held=execution_receipt['retained_output_upper_bound_bytes_per_rank'],
                selection=dict(sample_batch=len(dense_fit), selection_faces=selection_faces),
                compiled=lambda width, room: face_round_compiled_bytes(
                    mesh_xy, width, rows=n, finite=conservative_side - infinity_block, infinity_width=infinity_cap,
                    infinities=1 + len(moment_fields), ordered=ordered, odd_moments=odd_moments,
                    keep_budget=keep_budget, carrier=sizing['carrier'],
                    eigh_backend=resolution.eigh_backend, room=room))
            # Every face round's reduction is priced at this size.
            budget.program_bytes = execution_receipt['face_batch']['program_bytes_per_rank']
            reduction_room = execution_receipt['face_batch'].pop('eigh_room')
            # distrib_la decides each eigh stack against the room beside the
            # admitted batch and the retained factors: whole matrices per rank
            # where they fit, else the whole mesh.
            retained_bound = execution_receipt['retained_output_upper_bound_bytes_per_rank']
            # The selection row reserves its eigh stacks as native workspace: handed back to them.
            selected = execution_receipt['face_batch']['selection']
            budget.face_room = face_eigh_room(selected, retained_bound - int(selected['workspace_bytes_per_rank']),
                                              stage="shared-pole constructor selection")
            execution_receipt['face_eigh_room_bytes_per_rank'] = dict(selection=int(budget.face_room))
            eig = budget.eigenplan(n)
        from gw.shared_pole_local import parent_rounds
        # One fixed width per route; a short last round repeats its last real parent.
        rounds = parent_rounds(header['n_q_irr'], mesh_xy.size if execution == 'local' else face_batch)
        batch_spec = P(("x", "y"))
        read_spec = batch_spec if execution == 'local' else None
        kernels = _round_kernels(mesh_xy, 'batch' if execution == 'local' else 'face')
        to_face, to_batch = batch_to_face(mesh_xy), face_to_batch_reshard(mesh_xy)
    # One read handle per bank for every round (RoundReader): the bank and
    # moments files are opened, and their datasets opened, once.
    with RoundReader(bank["path"], mesh_xy=mesh_xy) as bank_rounds, \
            RoundReader(moments["path"], mesh_xy=mesh_xy) as moment_rounds:
        for ids, real, slots in rounds:
            with phase("batch_admission"):
                budget.batch_width = len(ids)
                budget.retained_panels = tuple(factors)
                budget.plan(
                    conservative_side, phase="selection",
                    sample_batch=len(dense_fit), selection_faces=selection_faces,
                    runs_in="spole.direction_selection")
                budget.live(())
            with phase("scratch_read"):
                with moment_rounds.phase() as moment_io:
                    exact, _ = read_round_samples(moment_io, meta=meta, header=moment_header,
                                                  q_ids=ids, partition_spec=read_spec,
                                                  fields=moment_fields)
            with phase("infinity_selection"):
                width = min(logical_n, max(1, int(recipe["infinity_width"])))
                qi, round_infinity_values = infinity_directions(
                    kernels, exact["M1"], width, eigh_plan=eig, column_extent=column_extent,
                    multiplet_tol=recipe["multiplet_relative_tolerance"],
                    real_rows=real if execution == 'local' else None)
                infinity = (qi, *(kernels.apply(exact[name], qi)
                                  for name in moment_fields))
                del exact
            with phase("sample_batch_read"):
                budget.live(infinity)
                # The round's dense fitted samples and line panels in one
                # collective read (SlabIO.read_multi) on the rounds' one handle.
                with bank_rounds.phase() as bank_io:
                    samples, line = read_round_samples(
                        bank_io, meta=meta, header=header, q_ids=ids, partition_spec=read_spec,
                        sample_ids=dense_fit, fields=("Wc", "dWc_ds"),
                        line_family="charge", line_samples=range(line_lo, line_hi))
                # The store admits this complete bounded scratch batch before
                # allocation. Charge it while directions/actions are selected;
                # release it before admitting the dense pencil.
            with phase("direction_selection"):
                # Synthetic local slots select nothing, as a dense selection's do.
                line_states = {sid: line_panel_states(panels, np.where(np.arange(len(counts)) < real, counts, 0),
                                                      recipe, sid=sid, ordered=ordered, mesh_xy=mesh_xy)
                               for sid, (panels, counts) in line.items()}
                round_states, round_counts, round_roles = select_round_states(
                    samples, recipe, sample_ids=dense_fit, real=real, mesh_xy=mesh_xy, eigh_plan=eig,
                    column_extent=column_extent, logical_n=logical_n, ordered=ordered,
                    line_states=line_states)
                del samples, line, line_states, qi
            with phase("reduction_admission"):
                infinity_counts = [int(v.shape[-1]) for v in round_infinity_values]
                budget.retained_panels = tuple(factors)
                budget.batch_width = len(ids)
                # Every state panel is padded to its recipe carrier, so the round
                # program's inputs have one shape; the pencil extent is grow-only
                # over this model's rounds and SC maps (round_tables): discovered
                # in map 0, held from map 1. The table is host metadata, known
                # before any panel is allocated.
                widths = recipe_panel_widths(round_roles[0], round_states, recipe,
                                             column_extent=column_extent, logical_n=logical_n)
                infinity_width = recipe_infinity_width(infinity, recipe, column_extent=column_extent,
                                                       logical_n=logical_n)
                tables = round_tables(
                    round_counts, widths, [st[0] for st in round_states], infinity_counts, infinity_width,
                    column_extent=column_extent, ordered=ordered, odd_moments=odd_moments,
                    key=("scalar", logical_n, ordered, odd_moments), history=carrier_history(meta))
                side = int(tables["active"].shape[-1])
                # Resolve before either reduction program is traced; the ledger
                # warns when the route price is over the budget.
                local_eigh = budget.eigenplan(side)
                # The reservation holds the round's operands; a face round's program is priced by
                # its compiled figure where it runs (face_reduce_round). The eigh rooms below still
                # subtract the admitted (conservative) program.
                admitted = budget.program_bytes
                try:
                    budget.program_bytes = 0          # the round program: its compiled figure
                    reduction_row = budget.plan(side, phase="reduction", padding_output_bytes_per_rank=round_padding_output_bytes(
                        round_states, infinity, widths, infinity_width), runs_in="spole.gram_reduction")
                finally:
                    budget.program_bytes = admitted
                round_states, infinity = pad_states(round_states, widths, infinity, infinity_width)
            with phase("gram_reduction"):
                if execution == 'face':
                    from gw.shared_pole_execution import face_reduce_round
                    # The kept span on the budget's carrier, as the local round
                    # solves it on its Ritz carrier: the Schur and final eigh run
                    # at about 2 x budget instead of the pencil side.
                    # The room the batch width compiled this program in: a room beside the
                    # admitted row would hold the program's own stacks twice (room 0).
                    room = reduction_room
                    round_model, round_signed, vectors, round_diagnostics = face_reduce_round(
                        round_states, infinity, tables, mesh=mesh_xy,
                        budget=budget, ordered=ordered, odd_moments=odd_moments,
                        keep_budget=keep_budget, eigh_backend=resolution.eigh_backend,
                        admit=False, room=room, carrier=sizing['carrier'],
                        beside=budget.ledger.price(reduction_row))
                else:
                    round_model, round_signed, vectors, round_diagnostics = reduce_round(
                        round_states, infinity, tables, real=real, mesh_xy=mesh_xy,
                        native_eigh=local_eigh.native_fn, ordered=ordered,
                        odd_moments=odd_moments, keep_budget=recipe.get("pole_budget"),
                        beside=budget.ledger.price(reduction_row))
                qi = infinity[0]
                # One device-to-host read of the round's replicated diagnostics and vectors.
                from gw.shared_pole_local import host_tree
                (round_reduction, round_zero, round_retained, round_permutation), (poles, active) = host_tree(
                    (tuple(round_diagnostics), tuple(vectors)))
                # rocSOLVER zheevd can return finite eigenvalues with NaN eigenvectors and info 0
                # (distrib_la.local_eigh). A local even-route round that reports it is solved once
                # more by Jacobi on the same, still-live inputs (no allocation lies between the solve
                # and this read, so the peak is the round's own); every other round is untouched.
                finite = round_reduction.get("ritz_eigh_finite")
                if execution != 'face' and finite is not None and not all(bool(v) for v in finite[:real]):
                    from distrib_la import native_eigh_jacobi
                    bad = [ids[i] for i in range(real) if not bool(finite[i])]
                    import warnings
                    warnings.warn(f"shared-pole round: local eigh returned non-finite eigenpairs at q={bad}; "
                                  "re-solved the round by Jacobi", RuntimeWarning, stacklevel=1)
                    del round_model, round_signed, vectors, round_diagnostics
                    round_model, round_signed, vectors, round_diagnostics = reduce_round(
                        round_states, infinity, tables, real=real, mesh_xy=mesh_xy,
                        native_eigh=native_eigh_jacobi, ordered=ordered,
                        odd_moments=odd_moments, keep_budget=recipe.get("pole_budget"))
                    (round_reduction, round_zero, round_retained, round_permutation), (poles, active) = host_tree(
                        (tuple(round_diagnostics), tuple(vectors)))
                del round_states, infinity
                budget.retained_panels = tuple(factors)
            with phase("gates"):
                import numpy as np
                # Host-side context for the zero-Ritz refusal: the round's own diagnostics.
                _sp_report = lambda slot: " ".join(
                    f"{k}={np.array2string(np.asarray(d[k][slot]), precision=3, max_line_width=400)}"
                    for d in (round_zero, round_reduction) for k in (
                        "dropped_factor_weight_fraction", "dropped_count", "factor_weight", "retained_rank",
                        "infinite_weight_fraction", "positive_count", "negative_count", "paired_rank",
                        "gram_min_relative", "paired_min_relative", "gram_condition") if k in d)
                for slot, q in enumerate(ids[:real]):
                    if ordered and not round_reduction["orientation_paired"][slot]:
                        raise ValueError(ORIENTATION_PAIR_REFUSAL + f" (q={q})")
                    for name in ("gram_diagonal_positive", "gram_valid", "retained_metric_positive"):
                        if not round_reduction[name][slot]:
                            raise ValueError(
                                f"GATE shared_pole_{name}: got: failed at q={q}, "
                                f"Gram min/max={round_reduction['gram_min_relative'][slot]}, "
                                f"rounding floor/max={round_reduction['gram_floor_relative'][slot]}, "
                                f"paired Schur S min/max={round_reduction['paired_min_relative'][slot] if ordered else 'n/a'}, "
                                f"metric infinity norm={round_reduction['metric_initial_infinity_norm'][slot]}, "
                                f"inverse-root residual={round_reduction['metric_inverse_root_residual_relative'][slot]}; "
                                f"want: Gram min >= -floor (normalized_gram_validity; even route: the larger of it and gram_rounding_validity) "
                                "and valid diagonal/retained metric; why: no PSD repair")
                    if not round_reduction.get("ritz_eigh_finite", [True] * (slot + 1))[slot]:
                        raise ValueError(f"GATE shared_pole_ritz_eigh_finite: got: non-finite Ritz eigenpairs at q={q}; "
                                         "want: finite eigenvalues and eigenvectors of the reduced pencil; "
                                         "why: a silent eigensolver failure, not a physical zero-Ritz weight")
                    if not round_zero["zero_policy"][slot]:
                        raise ValueError(f"GATE shared_pole_zero_ritz: got: failed at q={q} [{_sp_report(slot)}]; want: finite positive response within dropped-weight budget; why: no pole clipping")
                    # The ordered identity is on the ORIGINAL infinity directions: exact only for the
                    # full Galerkin span, projection accuracy after the keep/retention cuts. It is
                    # reported beside the full_m1/full_m3 diagnostic bands, as the TRS route reports
                    # its original-direction defects; the retained Ritz algebra is the metric gate above.
                    if not ordered and not all(value[slot] <= gates["retained_subspace_moments"]["threshold"]
                                               for value in round_retained.values()):
                        raise ValueError(f"GATE shared_pole_retained_moments: got: failed at q={q}; want: projected latent moment identity <=1e-10; why: corrected Ritz algebra")
                reductions = own_extent_receipts(round_reduction, tables["own"][:real])
            with phase("coulomb"):
                budget.batch_width = len(ids)
                model_row = budget.plan(side, phase="model", sample_batch=len(held_ids))
                budget.live((*round_model, *round_signed, qi))
                # V^-1/2 of the round's parents: one owner call per contiguous run of ids, rows in slot order.
                runs = []
                for q in sorted(ids[:real]):
                    if runs and q == runs[-1][1]:
                        runs[-1][1] += 1
                    else:
                        runs.append([q, q + 1])
                parts, coulomb = [], {}
                for lo, hi in runs:
                    coulomb_sqrt, part, receipt = response_coulomb_powers(
                        meta, coulomb_config, mesh_xy=mesh_xy, bank_io=bank,
                        q_span=(lo, hi), cache_config=config)
                    del coulomb_sqrt
                    parts.append(part)
                    # One parent's support rank and the resource hash; the resource
                    # itself (path, q order) is the bank's, stated once.
                    coulomb.update({q: dict(sha256=receipt["coulomb_identity"].get("sha256"),
                                            support_ranks=receipt["support_ranks"][q - lo:q - lo + 1],
                                            root_cache=receipt.get("root_cache"))
                                    for q in range(lo, hi)})
                inverse_sqrt = face_rows(mesh_xy, tuple(sorted(ids[:real]).index(q) for q in ids))(*parts)
                if execution == 'local':
                    inverse_sqrt = to_batch(inverse_sqrt)
                del parts
                # Callee I/O admission needs the actual arrays that survive the
                # Coulomb call, not the earlier pre-call live set.
                budget.live((*round_model, *round_signed, qi, inverse_sqrt))
            with phase("sample_batch_read"):
                # Read here, not with the fitted samples: the held panels would
                # otherwise stay resident through the reduction.
                with bank_rounds.phase() as bank_io:
                    held, _ = read_round_samples(bank_io, meta=meta, header=header, q_ids=ids,
                                                 partition_spec=read_spec,
                                                 sample_span=(held_lo, held_hi), fields=("Wc", "dWc_ds"))
                pick = kernels.take(tuple(i - held_lo for i in held_ids))
                held = tuple(pick(held[name]) for name in ("Wc", "dWc_ds"))
                budget.live((*round_model, *round_signed, qi, inverse_sqrt, *held))
            with phase("moment_read"):
                with moment_rounds.phase() as moment_io:
                    exact, _ = read_round_samples(moment_io, meta=meta, header=moment_header, q_ids=ids,
                                                  partition_spec=read_spec, fields=("M1", "M3"))
                budget.live((*round_model, *round_signed, qi, inverse_sqrt, *held,
                             *exact.values()))
            with phase("passivity_held"):
                passive, held_errors, reciprocity, moment_defects = check_round(
                    round_model, round_signed, inverse_sqrt, held, (exact["M1"], exact["M3"]), qi,
                    real=real, nodes=[_sample_point(recipe, i) for i in held_ids], eta_ry=recipe["eta_ev"] / RYD_TO_EV,
                    mesh_xy=mesh_xy, eigh_plan=eig if execution == 'face' else local_eigh, ordered=ordered,
                    # A face check decides its eighs beside the model phase and its own program.
                    room=lambda price: check_eigh_room(model_row, price))
                del inverse_sqrt, held, exact
            with phase("gates"):
                for slot, q in enumerate(ids[:real]):
                    if not passive["passivity"][slot]:
                        raise ValueError(f"GATE shared_pole_passivity: got: failed at q={q}; want: 0 <= V-whitened -W(i eta) <= I; why: passive screening")
                    if not ordered and not np.all(reciprocity["passed"][slot]):
                        raise ValueError(f"GATE shared_pole_model_reciprocity: got: { {k: v[slot].tolist() for k, v in reciprocity.items()} } at q={q}; want: model preserves transpose symmetry of symmetric held data; why: conjugate-port closure must survive reduction")
            with phase("receipts"):
                price = budget.plan(side)
                counts = active.sum(axis=-1, dtype=np.int64)
                for slot, q in enumerate(ids[:real]):
                    row_of = lambda tree: {key: value[slot:slot + 1] for key, value in tree.items()}
                    row, measurements = build_construction_row(
                        (None, poles[slot:slot + 1], active[slot:slot + 1]), counts[slot:slot + 1],
                        dict(reduction=reductions[slot], zero=row_of(round_zero), passive=row_of(passive),
                             retained=row_of(round_retained),
                             moment_defects={name: row_of(value) for name, value in moment_defects.items()},
                             held=[{"sample_id": sample_id, "Wc": float(held_errors[slot, 0, i]),
                                    "dWc_ds": float(held_errors[slot, 1, i])} for i, sample_id in enumerate(held_ids)],
                             # Preserve the receipt's sample-major W,dW rows, each with one parent.
                             reciprocity={key: value[slot].T.reshape(-1, 1).tolist()
                                          for key, value in reciprocity.items()},
                             permutation=round_permutation[slot:slot + 1]),
                        span=(q, q + 1), roles=round_roles[slot], price=price, coulomb=coulomb[q],
                        native_queries=budget.native_queries, identity=identity,
                        gates=gates, nspinor=int(meta.nspinor), logical_n=logical_n,
                        ordered=ordered, odd_moments=odd_moments,
                        detail=bool(config.debug.sigma_freq_debug_output))
                    receipt = construction_receipt(
                        measurements, capacity=ledger,
                        capacity_entry_start=receipt_entry_start, ordered=ordered,
                        charge4=charge4)
                    receipt.update(identity=identity, constructor=row)
                    receipts[q] = receipt
                receipt_entry_start = len(ledger.entries)
            with phase("export_prepare"):
                # The sorted model is an active prefix: keep the round's widest K, then restore to the face.
                width = column_extent(int(counts[:real].max()))
                factors.append(face_rows(mesh_xy, tuple(range(real)), width)(round_model[0]) if execution == 'face' else
                               face_rows(mesh_xy, tuple(range(real)), width)(to_face(round_model[0])))
                # The round's sorted poles stay on the device; the writer joins them there.
                store_poles.append(((vectors[0], vectors[1]), (int(real), int(width))))
                store_counts.append(counts[:real])
                placed += list(ids[:real])
                del round_model, round_signed, qi, reductions, vectors
                ledger.live_stages = upstream
                budget.retained_panels = tuple(factors)
    with phase("writer_stack"):
        if sorted(placed) != list(range(nq)):
            raise ValueError(f"GATE shared_pole_rounds: got: parents {sorted(placed)}; want: each of {nq} parents once; why: one canonical store")
        order = np.argsort(placed)
        width = max(block.shape[-1] for block in factors)
        public_b = canonical_factors(mesh_xy, tuple(int(i) for i in order))(*factors)
        from gw.shared_pole_local import store_pole_rows
        store_poles, store_device_counts = store_pole_rows(
            mesh_xy, tuple(b for _, b in store_poles), int(width), tuple(int(i) for i in order))(
            tuple(p for (p, _), _ in store_poles), tuple(a for (_, a), _ in store_poles))
        budget.batch_width = nq
        budget.retained_panels = ()
        budget.plan(width, phase="model")
        budget.live((public_b,))
        del factors
    with phase("writer"):
        target, model_residence = output, dict(residence="file")
        if residence is not None:
            resident, model_residence = residence(nq, width)
            if resident is not None:
                target = resident
                ledger.live_stages = (*ledger.live_stages, model_residence["stage"])
        store_header = write_shared_pole_model(
            target, public_b, store_poles,
            np.concatenate(store_counts)[order], q_span=(0, nq), meta=meta, tables=bank["tables"], recipe=recipe,
            receipts={"identity": identity, "q_receipts": [receipts[q] for q in range(nq)]}, ordered=ordered,
            device_counts=store_device_counts)
        ledger.live_stages = upstream
        del public_b
    with phase("return"):
        # Stage walls summed over the rounds (host clock; a stage that only
        # dispatches lends its device time to the next stage that reads back).
        seconds = {name: round(value, 3) for name, value in walls.items()}
        seconds["rounds"] = len(rounds)
        return {"q_receipts": [receipts[q] for q in range(nq)], "model_header": store_header,
                "model": target if model_residence.get("stage") else str(output),
                "model_residence": model_residence,
                "capacity": ledger.receipt(),
                "identity": identity, "status": "CONSTRUCTED", "seconds": seconds,
                "execution": dict(mode=execution, **execution_receipt)}
