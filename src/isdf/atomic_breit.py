"""Static compensated three-current provider on the existing whole-q owners.

No current-wavefunction or GW admission is selected here. Callers supply
three consistently reconstructed current RHS, their separate canonical C
factors, complete geometry and an explicit reciprocal kernel identity.
The static Poisson/biharmonic locality certificate requires matched Q0/Q2,
ordinary three-dimensional geometry and disjoint compact spheres.
The optional Slab body completes the finite reciprocal nn/tn/nt terms;
the analytic local remainder remains the same bulk infinite-space reference.
This hybrid operator is not by itself a transverse-convergence certificate.
"""
from __future__ import annotations

import hashlib
import numpy as np

STATIC_KERNEL = 'ordinary_3d_static_transverse_P_over_K2'
BODY_HEAD_SCHEMA = 'vcoul_static_current_body_heads_v2'
SLAB_STATIC_KERNEL = 'slab_2d_static_transverse_bulk_local_completion'
SLAB_BODY_HEAD_SCHEMA = 'vcoul_static_current_body_heads_slab_v1'
CURRENT_GAMMA_K2_EPS = 1e-30


def _body_head_digest(policy):
    schema=policy.get('schema',BODY_HEAD_SCHEMA)
    digest=hashlib.sha256(schema.encode('ascii'))
    for key in ('q_frac','gvec_components','bvec','q_plus_G_cart','ngk_per_q',
                'sel','mask','mult','len2','v_bare','v_avg',
                'gamma_head_cartesian','gamma_kgrid'):
        value=np.ascontiguousarray(policy[key])
        digest.update(key.encode('ascii'))
        digest.update(str(value.dtype).encode('ascii'))
        digest.update(np.asarray(value.shape,dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    digest.update(np.asarray([policy['cell_volume'],policy['cutoff_ry'],
                              policy['tie_rtol']],dtype=np.float64).tobytes())
    digest.update(policy['gamma_placement'].encode('ascii'))
    digest.update(policy['gamma_source_receipt'].encode('utf8'))
    if schema==SLAB_BODY_HEAD_SCHEMA:
        digest.update(np.asarray([policy['sys_dim'],policy['slab_half_height']],np.float64).tobytes())
    return digest.hexdigest()


def static_breit_body_head_slots(*,q_frac,gvec_components,ngk_per_q,geometry,
                                vcoul_cutoff_ry,v_head_fn=None,head_tie_rtol=1e-9,
                                gamma_head=None,sys_dim=3):
    """Bind the public scalar body-head owner to actual current q/G slots.

    The caller passes the same canonical head callback used to construct V.
    Each public table sees only physical G slots, so padding cannot create
    artificial argmin ties. Cartesian P(K) and the current basis are applied
    by the current provider after the scalar slot values are authenticated.
    ``gamma_head`` optionally carries the SAME public current-head owner's
    signed physical Cartesian tensor in Ry, its integer kgrid and opaque
    source receipt. It is embedded once at the unique physical Gamma slot.
    Otherwise Gamma remains separate; scalar charge heads never supply it.
    ``sys_dim=2`` authenticates the public Slab2D orientation/half-height;
    the default bulk schema and its payload bytes remain unchanged.
    """
    from vcoul import get_kernel,head_slot_table
    q=np.asarray(q_frac,dtype=np.float64)
    G=np.asarray(gvec_components,dtype=np.float64)
    counts=np.asarray(ngk_per_q)
    cutoff,tie=float(vcoul_cutoff_ry),float(head_tie_rtol)
    bvec=np.asarray(geometry.bvec,dtype=np.float64);volume=float(geometry.cell_volume)
    if isinstance(sys_dim,bool) or not isinstance(sys_dim,(int,np.integer)) or sys_dim not in (2,3):
        raise ValueError('static current body requires explicit bulk3D or Slab2D dimensionality')
    kernel=get_kernel(sys_dim)
    if (q.ndim!=2 or q.shape[1]!=3 or not len(q) or not np.isfinite(q).all()
            or G.ndim!=3 or G.shape[:2]!=(len(q),3) or not np.isfinite(G).all()
            or np.any(G!=np.rint(G)) or counts.shape!=(len(q),) or counts.dtype.kind not in 'iu'
            or np.any(counts<1) or np.any(counts>G.shape[2])
            or bvec.shape!=(3,3) or not np.isfinite(bvec).all()
            or not np.isfinite(volume) or volume<=0
            or not np.isfinite(cutoff) or cutoff<=0 or not np.isfinite(tie) or tie<0):
        raise ValueError('invalid canonical static current body-head geometry')
    tables=[head_slot_table(kernel,q[i:i+1],G[i:i+1,:,:int(n)],
                           geometry=geometry,vcoul_cutoff_ry=cutoff,
                           v_head_fn=v_head_fn,head_tie_rtol=tie)
            for i,n in enumerate(counts)]
    width=max(t.sel.shape[1] for t in tables)
    sel=np.zeros((len(q),width),np.int32);mask=np.zeros((len(q),width),np.float64)
    for i,t in enumerate(tables):
        n=t.sel.shape[1];sel[i,:n]=t.sel[0];mask[i,:n]=t.mask[0]
        sel[i,n:]=t.sel[0,0]
    policy=dict(schema=BODY_HEAD_SCHEMA,
        q_frac=q.copy(),gvec_components=G.copy(),bvec=bvec.copy(),
        q_plus_G_cart=(G.transpose(0,2,1)+q[:,None])@bvec,
        ngk_per_q=counts.astype(np.int32,copy=True),sel=sel,mask=mask,
        cell_volume=volume,cutoff_ry=cutoff,tie_rtol=tie,
        gamma_placement='separate_measured_current',
        gamma_head_cartesian=np.zeros((len(q),3,3),np.complex128),
        gamma_kgrid=np.zeros(3,np.int32),gamma_source_receipt='')
    if sys_dim==2:
        policy.update(schema=SLAB_BODY_HEAD_SCHEMA,sys_dim=2,
                      slab_half_height=kernel.truncation_half_height(geometry))
    if gamma_head is not None:
        if not isinstance(gamma_head,dict) or set(gamma_head)!={'cartesian_ry','kgrid','source_receipt'}:
            raise ValueError('canonical Gamma current head requires its tensor, kgrid and source receipt')
        tensor=np.asarray(gamma_head['cartesian_ry'],dtype=np.complex128)
        grid=np.asarray(gamma_head['kgrid'])
        receipt=gamma_head['source_receipt']
        active=np.arange(G.shape[2])[None]<counts[:,None]
        zero=active&(np.einsum('qgi,qgi->qg',policy['q_plus_G_cart'],policy['q_plus_G_cart'])<=CURRENT_GAMMA_K2_EPS)
        if (tensor.shape!=(3,3) or not np.isfinite(tensor).all() or not np.any(tensor)
                or not np.allclose(tensor,tensor.conj().T,rtol=0,atol=1e-12)
                or grid.shape!=(3,) or grid.dtype.kind not in 'iu' or np.any(grid<1)
                or np.any(grid>np.iinfo(np.int32).max)
                or not isinstance(receipt,str) or not receipt.strip() or len(receipt.encode('utf8'))>65536
                or not np.allclose(q*grid,np.rint(q*grid),rtol=0,atol=2e-10)
                or np.count_nonzero(zero)!=1):
            raise ValueError('canonical Gamma current head geometry, tensor or source receipt is invalid')
        policy.update(gamma_placement='canonical_embedded_physical_current',
            gamma_kgrid=grid.astype(np.int32,copy=True),gamma_source_receipt=receipt)
        policy['gamma_head_cartesian'][zero.any(axis=1)]=tensor
    for key in ('mult','len2','v_bare','v_avg'):
        policy[key]=np.concatenate([np.asarray(getattr(t,key)) for t in tables])
    policy['payload_sha256']=_body_head_digest(policy)
    return policy


def _authenticate_body_head_slots(policy,K,counts,volume,body_mask):
    """Refuse changed geometry, missed ties, and double Gamma placement."""
    from vcoul import CoulombGeometry,get_kernel,head_slot_table
    required={'q_frac','gvec_components','bvec','q_plus_G_cart','ngk_per_q',
              'sel','mask','mult','len2','v_bare','v_avg',
              'cell_volume','cutoff_ry','tie_rtol','payload_sha256',
              'gamma_head_cartesian','gamma_kgrid','gamma_source_receipt','gamma_placement'}
    if not isinstance(policy,dict) or not required.issubset(policy):
        raise ValueError('static Breit canonical body-head payload is incomplete')
    if policy.get('schema')==SLAB_BODY_HEAD_SCHEMA and not {'sys_dim','slab_half_height'}.issubset(policy):
        raise ValueError('static current Slab body requires its dimensionality and truncation geometry')
    if (not isinstance(policy,dict) or policy.get('schema') not in (BODY_HEAD_SCHEMA,SLAB_BODY_HEAD_SCHEMA)
            or policy.get('gamma_placement') not in ('separate_measured_current','canonical_embedded_physical_current')
            or not isinstance(policy['gamma_source_receipt'],str)
            or policy.get('payload_sha256')!=_body_head_digest(policy)):
        raise ValueError('static Breit canonical body-head payload is missing or changed')
    active=np.arange(K.shape[1])[None]<counts[:,None]
    q=np.asarray(policy['q_frac']);G=np.asarray(policy['gvec_components']);bvec=np.asarray(policy['bvec'])
    if (q.shape!=(K.shape[0],3) or G.shape!=(K.shape[0],3,K.shape[1])
            or bvec.shape!=(3,3) or not np.isfinite(q).all() or not np.isfinite(G).all()
            or not np.isfinite(bvec).all() or np.any(G!=np.rint(G))
            or not np.allclose(((G.transpose(0,2,1)+q[:,None])@bvec)[active],
                               K[active],rtol=0,atol=2e-12)):
        raise ValueError('static Breit body-head fractional/reciprocal geometry mismatch')
    if (np.asarray(policy['q_plus_G_cart']).shape!=K.shape
            or not np.allclose(np.asarray(policy['q_plus_G_cart'])[active],K[active],rtol=0,atol=2e-12)
            or not np.array_equal(policy['ngk_per_q'],counts)
            or float(policy['cell_volume'])!=volume):
        raise ValueError('static Breit body-head q/G/count/volume binding mismatch')
    Q,ng=K.shape[:2];sel,mask=np.asarray(policy['sel']),np.asarray(policy['mask'])
    if (sel.ndim!=2 or sel.shape[0]!=Q or not sel.shape[1] or sel.dtype.kind not in 'iu'
            or mask.shape!=sel.shape or np.any((mask!=0)&(mask!=1))
            or np.any(sel<0) or np.any(sel>=ng)
            or any(np.asarray(policy[key]).shape!=(Q,) for key in ('mult','len2','v_bare','v_avg'))
            or not np.isfinite(policy['v_avg']).all() or np.any(policy['v_avg']<0)):
        raise ValueError('invalid static Breit canonical body-head table')
    if (np.asarray(policy['mult']).dtype.kind not in 'iu'
            or not np.isfinite(policy['cutoff_ry']) or policy['cutoff_ry']<=0
            or not np.isfinite(policy['tie_rtol']) or policy['tie_rtol']<0):
        raise ValueError('invalid static Breit canonical cutoff/tie/multiplicity policy')
    length2=np.einsum('qgi,qgi->qg',K,K)
    if not np.array_equal(body_mask,active&(length2<=policy['cutoff_ry'])):
        raise ValueError('static Breit body cutoff disagrees with its canonical head owner')
    gamma=np.asarray(policy['gamma_head_cartesian']);grid=np.asarray(policy['gamma_kgrid'])
    if (gamma.shape!=(Q,3,3) or not np.isfinite(gamma).all()
            or not np.allclose(gamma,gamma.conj().transpose(0,2,1),rtol=0,atol=1e-12)
            or grid.shape!=(3,) or grid.dtype.kind not in 'iu'):
        raise ValueError('static Breit canonical Gamma tensor/grid is invalid')
    if policy['gamma_placement']=='separate_measured_current':
        if np.any(gamma) or np.any(grid) or policy['gamma_source_receipt']:
            raise ValueError('separate measured current Gamma head cannot also be embedded')
    else:
        zero=active&(length2<=CURRENT_GAMMA_K2_EPS)
        if (np.count_nonzero(zero)!=1 or np.any(grid<1)
                or not np.allclose(q*grid,np.rint(q*grid),rtol=0,atol=2e-10)
                or not policy['gamma_source_receipt'].strip()
                or len(policy['gamma_source_receipt'].encode('utf8'))>65536
                or np.any(gamma[~zero.any(axis=1)]) or not np.any(gamma[zero.any(axis=1)])):
            raise ValueError('static Breit canonical Gamma placement/source/grid mismatch')
    selected=np.zeros((Q,ng),bool)
    geometry=CoulombGeometry(bvec=bvec,cell_volume=volume)
    dimension=2 if policy['schema']==SLAB_BODY_HEAD_SCHEMA else 3
    kernel=get_kernel(dimension)
    if dimension==2 and (policy.get('sys_dim')!=2
            or policy.get('slab_half_height')!=kernel.truncation_half_height(geometry)):
        raise ValueError('static current Slab dimensionality or truncation geometry changed')
    for row,n in enumerate(counts):
        # Reuse the same owner as the producer. Relative-tolerance ties
        # need not have identical lengths; their bare value is the public
        # tied-set mean, not the value at the shortest individual vector.
        expected=head_slot_table(kernel,q[row:row+1],G[row:row+1,:,:int(n)],
            geometry=geometry,vcoul_cutoff_ry=policy['cutoff_ry'],
            head_tie_rtol=policy['tie_rtol'])
        wanted=expected.sel[0,expected.mask[0]==1]
        got=sel[row,mask[row]==1]
        if (not np.array_equal(got,wanted) or int(policy['mult'][row])!=int(expected.mult[0])
                or np.any(sel[row,mask[row]==0]!=(int(got[0]) if len(got) else 0))):
            raise ValueError('static Breit body heads must contain every canonical tied physical slot')
        if (not np.isclose(policy['len2'][row],expected.len2[0],rtol=2e-12,atol=1e-14)
                or not np.isclose(policy['v_bare'][row],expected.v_bare[0],rtol=2e-12,atol=1e-14)
                or (not len(wanted) and policy['v_avg'][row]!=0)):
            raise ValueError('static Breit body-head bare scale or Gamma skip mismatch')
        selected[row,wanted]=True
    return selected


def _current_point_body_tile(policy, q_indices, first, width, pairs, B, prefactor, *, sys_dim):
    """Public point-body scalar times B P(K) B† on bounded physical q/G rows.

    Returns (local_q, pair, G_tile) complex128 on the host. No averaged
    scalar head or physical Gamma tensor enters the bulk reference. The
    public scalar owner already supplies 8*pi/Omega and the cutoff mask.
    """
    from vcoul import CoulombGeometry,get_kernel,v_qG_table
    rows=np.asarray(q_indices,dtype=int)
    geometry=CoulombGeometry(bvec=policy['bvec'],cell_volume=policy['cell_volume'])
    scale=float(prefactor)/(8*np.pi/float(policy['cell_volume']))
    output=np.zeros((len(rows),len(pairs),int(width)),complex)
    for slot,q in enumerate(rows):
        if q>=len(policy['ngk_per_q']):continue
        stop=min(first+width,int(policy['ngk_per_q'][q]))
        if stop<=first:continue
        vectors=np.asarray(policy['q_plus_G_cart'])[q,first:stop]
        length=np.linalg.norm(vectors,axis=-1)
        unit=np.divide(vectors,length[:,None],out=np.zeros_like(vectors),where=length[:,None]>0)
        scalar=v_qG_table(get_kernel(sys_dim),np.asarray(policy['q_frac'])[q:q+1],
            np.asarray(policy['gvec_components'])[q:q+1,:,first:stop],
            geometry=geometry,vcoul_cutoff_ry=policy['cutoff_ry'])[0]*scale
        for index,(a,b) in enumerate(pairs):
            ua=np.einsum('i,gi->g',B[a],unit);ub=np.einsum('i,gi->g',B[b],unit)
            output[slot,index,:stop-first]=scalar*(float(a==b)-ua*ub.conj())
    return output


def authenticate_group(zetas,pairs,v_tables):
    """Return one complete shared static provider or refuse partial admission."""
    local=tuple(getattr(z,'local_augmentation',None) for z in zetas)
    if not any(p is not None for p in local):
        return None
    if len(zetas)!=3 or any(not isinstance(p,dict) for p in local):
        raise ValueError("GATE augmented_current_coulomb: augmented currents require all three local providers together")
    shared=local[0].get('group_provider')
    if (not isinstance(shared,dict) or shared.get('kernel') not in (STATIC_KERNEL,SLAB_STATIC_KERNEL)
            or any(p.get('group_provider') is not shared for p in local)
            or any(('rhs' not in p) != (shared.get('coefficient_owner') is not None) for p in local)
            or tuple(p.get('channel') for p in local)!=(0,1,2)
            or tuple(id(z) for z in zetas)!=shared.get('zeta_identity')):
        raise ValueError("GATE augmented_current_coulomb: missing or mismatched shared current geometry/basis references")
    owned = shared.get('coefficient_owner')
    if owned is not None:
        from isdf.current_coefficients import CurrentCoefficientBanks
        if (not isinstance(owned, CurrentCoefficientBanks)
                or any(p.get('coefficient_owner') is not owned for p in local)):
            raise ValueError('GATE augmented_current_coulomb: solved current banks require their stream owner')
        owned.authenticate(zetas)
    for key in ('prepare','fourier_tile','onsite','authenticate_kernel'):
        if not callable(shared.get(key)):
            raise ValueError(f"GATE augmented_current_coulomb: missing group provider {key}")
    shared['authenticate_kernel'](pairs,v_tables)
    return shared


def radial_breit_providers(zetas,rhs,*,smooth_rhs=None,body_metric='mixed_reciprocal',
                          smooth_support=None,
                          radius,lm,centers_cart,q_plus_G_cart,
                          cell_volume,fft_points,support_radius,
                          minimum_atom_image_distance,reciprocal_prefactor,
                          current_basis_rows=None,head_cartesian=None,body_mask=None,
                          body_head_slots=None,
                          interpolation_degree=3,quadrature_order=16,
                          fourier_points=4097,field_tile=128,centroid_tile=512):
    r"""Procedural per-current references to a coherent static group provider.

    Every RHS is (Q_pad,mu,atom*LM*radius) complex128, whole-q owned.
    The caller solves each with its OWN current C factor; no charge factor
    or second fit is introduced. B is the canonical row basis J_fit=B J_cart.

    Body kernel: v_ab(K)=reciprocal_prefactor [B P(K) B^dagger]_ab/K².
    Explicit head tensors are physical Cartesian v at K=0; default zero.
    The local geometric scale is derived as reciprocal_prefactor*N_fft²/Omega.
    For the physical TT prefactor -8*pi/Omega this becomes -8*pi*(N/Omega)².
    The factory never infers this scale from an arbitrary passed v table.

    Fourier tiles contain grid-sum delta/g. Source delta uses the existing
    physical cardinal Fourier owner; analytic two-moment g uses exact Sonine
    transforms. Onsite streams positive vector-harmonic radial Gram differences;
    matching Q0 cancels their exterior blocks, while Q2 certifies offsite locality.
    field_tile bounds retained radial factor coordinates, centroid_tile bounds
    samples. The full radial rank is retained; input coefficient clouds are
    gathered directly by sample/feature tile and never padded or replicated.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding,PartitionSpec as P
    from scipy.special import sph_harm_y
    from jax import shard_map
    from common.collectives import device_put_process_local
    from isdf.zeta_mubatch import _require_q_owned
    from isdf.atomic_coulomb import _radial_fourier_cache
    from isdf.augmentation_breit import (static_transverse_geometry,
        build_two_moment_compensation,evaluate_two_moment_compensation,
        two_moment_compensation_radial_fourier,static_transverse_block_factors)

    from isdf.current_coefficients import CurrentCoefficientBanks
    coefficient_owner = rhs if isinstance(rhs, CurrentCoefficientBanks) else None
    owned_banks = coefficient_owner is not None
    separate_halves = owned_banks and coefficient_owner.operand_kind == 'delta_and_ps'
    physical_mixed = False
    if smooth_support is not None:
        from isdf.current_support import SmoothCurrentSupport
        if not isinstance(smooth_support, SmoothCurrentSupport) or not smooth_support.physical_mixed:
            raise ValueError('physical mixed current requires the actual Slab layer/support witness')
        smooth_support.authenticate(zetas, body_head_slots, centers_cart=centers_cart,
                                    support_radius=support_radius)
        if body_metric != 'mixed_reciprocal':
            raise ValueError('physical mixed current support cannot alter another body policy')
        physical_mixed = True
    if owned_banks:
        expected_metric = 'compensated' if separate_halves else 'mixed_reciprocal'
        if (smooth_rhs is not None or body_metric != expected_metric
                or not separate_halves and (not physical_mixed or coefficient_owner.smooth_support is not smooth_support)):
            raise ValueError('streamed current banks require one complete compensated owner')
        owned = coefficient_owner.authenticate(zetas)
        rhs = tuple(v[0] for v in owned)
        smooth_rhs = tuple(v[1] for v in owned) if separate_halves else None
    if len(zetas)!=3 or len(rhs)!=3:
        raise ValueError("static Breit provider requires exactly three current channels and RHS")
    if body_metric not in ('mixed_reciprocal','compensated'):
        raise ValueError("static Breit provider requires a declared supported body metric")
    compensated=body_metric=='compensated'
    if compensated != (smooth_rhs is not None):
        raise ValueError("compensated static Breit requires the smooth local RHS for all three channels")
    if smooth_rhs is not None and (not isinstance(smooth_rhs,(tuple,list)) or len(smooth_rhs)!=3):
        raise ValueError("static Breit smooth local RHS requires all three current channels")
    slab=isinstance(body_head_slots,dict) and body_head_slots.get('schema')==SLAB_BODY_HEAD_SCHEMA
    if slab and not (compensated or physical_mixed):
        raise ValueError('Slab current requires compensated operands or authenticated physical mixed support')
    z0=zetas[0];st,mesh=z0.store,z0.mesh
    Q,Qp,mu,gt=int(st.Q),int(st.Q_pad),int(st.mu_pad),int(st.g_tile)
    ng=int(st.g_axis.logical);XY=('x','y');qspec=P(XY,None,None)
    qsh=NamedSharding(mesh,qspec)
    for z in zetas:
        other=z.store
        if ((other.Q,other.Q_pad,other.mu_pad,other.g_tile,other.g_axis.logical)
                !=(Q,Qp,mu,gt,ng) or z.mesh is not mesh or z.mu_basis is not z0.mu_basis
                or not np.array_equal(z.ngk_per_q,z0.ngk_per_q)
                or not np.array_equal(z.gvec_components,z0.gvec_components)):
            raise ValueError("all current channels must share exact store, centroid and q/G geometry")
    r=np.asarray(radius,dtype=float);labels=np.asarray(lm)
    centers=np.asarray(centers_cart,dtype=float);K=np.asarray(q_plus_G_cart,dtype=float)
    volume=float(cell_volume);N=int(fft_points);R=float(support_radius);k0=float(reciprocal_prefactor)
    ft,mt=int(field_tile),int(centroid_tile)
    B=np.eye(3,dtype=complex) if current_basis_rows is None else np.asarray(current_basis_rows,dtype=complex)
    if (centers.ndim!=2 or centers.shape[1]!=3 or not len(centers)
            or not np.all(np.isfinite(centers)) or K.shape!=(Q,ng,3) or not np.all(np.isfinite(K))
            or not np.isfinite(volume) or volume<=0 or N!=fft_points or N<1
            or not np.isfinite(k0) or k0==0 or not np.isfinite(minimum_atom_image_distance)
            or minimum_atom_image_distance<=2*R or ft!=field_tile or ft<1
            or mt!=centroid_tile or mt<1 or B.shape!=(3,3) or not np.all(np.isfinite(B))
            or not np.allclose(B@B.conj().T,np.eye(3),rtol=0,atol=1e-12)):
        raise ValueError("invalid authenticated static Breit geometry, scale or row current basis")
    geometry=static_transverse_geometry(r,labels,support_radius=R,
        interpolation_degree=interpolation_degree,quadrature_order=quadrature_order)
    angular,radial=geometry['angular'],geometry['radial'];labels=angular['lm']
    nh,nr,na=len(labels),len(r),len(centers);nf=na*nh*nr
    if owned_banks and coefficient_owner.radial_shape[:3] != (na, nh, nr):
        raise ValueError('streamed current coefficient radial geometry disagrees with provider')
    for value in rhs:
        _require_q_owned(value,mesh,(Qp,mu,nf),name='static current RHS')
    if compensated:
        for value in smooth_rhs:
            _require_q_owned(value,mesh,(Qp,mu,nf),name='static smooth current RHS')
    counts=np.asarray(z0.ngk_per_q,dtype=int)
    if counts.shape!=(Q,) or np.any(counts<1) or np.any(counts>ng):
        raise ValueError("static Breit source G counts are invalid")
    active=np.arange(ng)[None]<counts[:,None]
    mask=active if body_mask is None else np.asarray(body_mask)
    if mask.shape!=active.shape or mask.dtype!=bool or np.any(mask&~active):
        raise ValueError("static Breit body mask must authenticate physical G slots")
    head=np.zeros((Q,3,3),complex) if head_cartesian is None else np.array(head_cartesian,dtype=complex,copy=True)
    if (head.shape!=(Q,3,3) or not np.all(np.isfinite(head))
            or not np.allclose(head,head.conj().transpose(0,2,1),rtol=0,atol=1e-12)):
        raise ValueError("static Breit head must be an explicit Hermitian Cartesian tensor")
    head_selected=None
    body_head_identity=None
    if body_head_slots is not None:
        head_selected=_authenticate_body_head_slots(body_head_slots,K,counts,volume,mask)
        bound_head=np.asarray(body_head_slots['gamma_head_cartesian'])
        if head_cartesian is None:
            head=bound_head.copy()
        elif not np.array_equal(head,bound_head):
            raise ValueError('static Breit Gamma tensor differs from its canonical body contract')
        if (body_head_slots['gamma_placement']=='canonical_embedded_physical_current'
                and not np.isclose(k0,-8*np.pi/volume,rtol=2e-12,atol=0)):
            raise ValueError('canonical embedded Gamma head requires the physical signed static TT prefactor')
        policy_G=np.asarray(body_head_slots['gvec_components'])
        stored_G=np.asarray(z0.gvec_components)
        if (policy_G.shape!=stored_G.shape
                or not np.array_equal(policy_G.transpose(0,2,1)[active],
                                      stored_G.transpose(0,2,1)[active])):
            raise ValueError('static Breit body-head G slot order disagrees with its current zeta store')
        body_head_identity=body_head_slots['payload_sha256']
    if slab and not 2*R<float(body_head_slots['slab_half_height']):
        raise ValueError('Slab current local remainder requires sphere diameter below the truncation half-height')
    norm=np.linalg.norm(K,axis=-1);maximum=float(np.max(np.where(active,norm,0)))
    compensation=build_two_moment_compensation(labels[:,0],support_radius=R)
    # Reuse the existing source-cardinal Fourier algorithm unchanged. Its
    # g0 cache is only a validation companion; actual two-moment g FT is analytic.
    cache_tables=dict(radial,moments=radial['moments0'],
        compensation_quadrature_shapes=evaluate_two_moment_compensation(
            compensation,radial['quadrature_radius'])[:,0])
    cache=_radial_fourier_cache(cache_tables,maximum,fourier_points)
    gamma_origin_error=None
    if slab or (body_head_slots is not None and body_head_slots['gamma_placement']=='canonical_embedded_physical_current'):
        source0=np.asarray(cache['density'](np.zeros(1)))[:,0,:]
        g0=two_moment_compensation_radial_fourier(compensation,np.zeros(1))[...,0]
        maps=np.stack((radial['moments0'],radial['moments2']),axis=1)
        compensated0=np.einsum('ld,ldr->lr',g0,maps)
        gamma_origin_error=float(np.max(abs(source0-compensated0)))
        scale=max(float(np.max(abs(source0))),float(np.max(abs(compensated0))),np.finfo(float).tiny)
        if gamma_origin_error>256*np.finfo(float).eps*scale:
            raise ValueError('canonical embedded current Gamma requires delta(0)=g(0) on the same radial map')
    grow=np.searchsorted(radial['degrees'],labels[:,0])
    moments=np.stack((radial['moments0'][grow],radial['moments2'][grow]),axis=1)
    blocks=static_transverse_block_factors(geometry,compensation)
    mup=((mu+mt-1)//mt)*mt;nmu=mup//mt
    put=lambda value:device_put_process_local(np.asarray(value),NamedSharding(mesh,P()))
    M,rows=put(moments),put(grow)
    # Each VSH column has at most three orbital m=M-s samples. Its spin
    # weights also rotate the canonical fitted row basis into Cartesian space.
    mode=blocks['basis']['modes'];U=blocks['basis']['transform'].reshape(3,nh,-1)
    hrows=np.zeros((len(mode),3),np.int32);spin_weights=np.zeros((3,len(mode),3),complex)
    lm_rows={tuple(v):i for i,v in enumerate(labels)}
    for k,(_,L,m) in enumerate(mode):
        for slot,s in enumerate(range(-1,2)):
            h=lm_rows.get((int(L),int(m-s)))
            if h is not None:
                hrows[k,slot]=h
                spin_weights[:,k,slot]=np.einsum('b,cb->c',U[:,h,k].conj(),B.conj())
    sparse_rows,sparse_weights=put(hrows),put(spin_weights)
    # Magnetic and edge sectors have one L; electric interior sectors have
    # two. Separate buckets avoid padding every magnetic nr block to 2*nr.
    bucket_arrays=[];bucket_meta=[]
    for width in (1,2):
        sectors=[s for s in blocks['sectors'] if len(s['Ls'])==width]
        if not sectors:continue
        dimension=sectors[0]['delta_factor'].shape[0]
        if any(s['delta_factor'].shape[0]!=dimension
               or s['compensation_factor'].shape[0]!=dimension for s in sectors):
            raise ValueError("static Breit vector blocks do not share their common field coordinates")
        tile=min(ft,dimension);padded=((dimension+tile-1)//tile)*tile
        df=np.stack([np.pad(s['delta_factor'],((0,padded-dimension),(0,0))) for s in sectors])
        gf=np.stack([np.pad(s['compensation_factor'],((0,padded-dimension),(0,0))) for s in sectors])
        steps=np.asarray([(i,*cols) for i,s in enumerate(sectors) for cols in s['indices_by_M']],np.int32)
        bucket_arrays.append(tuple(map(put,(df,gf,steps))))
        bucket_meta.append((width,tile,padded//tile,len(steps)))
    bucket_arrays=tuple(bucket_arrays);bucket_meta=tuple(bucket_meta)
    onsite_scale=k0*N*N/volume;fourier_scale=N/volume
    coefficient_specs = ((qspec, qspec),)*3 if separate_halves else (qspec,)*3

    def live(local_q):
        rank=jax.lax.axis_index('x')*int(mesh.shape['y'])+jax.lax.axis_index('y')
        return rank*local_q+jnp.arange(local_q)<Q

    def prepare_local(coefficients,moment_rows):
        out=[]
        for pair in coefficients:
            values = pair[0] if separate_halves else pair
            if compensated:
                # Gather one radial row directly from the paired operand.
                # Even an in-jit full-half slice needs a contiguous cloud
                # temporary because PS alternates with delta per centroid.
                zero=jnp.zeros((values.shape[0],mu,na*nh*2),jnp.complex128)
                def add_moment(acc,step):
                    h=step%nh;feature=step*nr+jnp.arange(nr)
                    source=values[:,:,feature]
                    q=jnp.einsum('dn,bmn->bmd',moment_rows[h],source)
                    return jax.lax.dynamic_update_slice(acc,q,(0,0,2*step)),None
                q=jax.lax.scan(add_moment,zero,jnp.arange(na*nh),unroll=1)[0]
                q=jnp.where(live(values.shape[0])[:,None,None],q,0)
            else:
                shaped=values.reshape(values.shape[0],mu,na,nh,nr)
                q=jnp.einsum('idn,bmain->bmaid',moment_rows,shaped)
                q=jnp.where(live(values.shape[0])[:,None,None,None,None],q,0)
                q=q.reshape(values.shape[0],mu,na*nh*2)
            out.append(q)
        return tuple(out)
    prepare_kernel=jax.jit(shard_map(prepare_local,mesh=mesh,
        in_specs=(coefficient_specs,P()),out_specs=(qspec,)*3,check_vma=False))

    def fourier_local(coefficients,moment_coefficients,bessel,angle,g_radial,degree_rows):
        outputs=[];compensations=[]
        for c in range(3):
            values=coefficients[c][0] if separate_halves else coefficients[c]
            shaped=None if compensated else values.reshape(values.shape[0],mu,na,nh,nr)
            qm=moment_coefficients[c].reshape(values.shape[0],mu,na,nh,2)
            zero=jnp.zeros((values.shape[0],mu,gt),jnp.complex128)
            def add(acc,step):
                atom,h=step//nh,step%nh;lrow=degree_rows[h]
                source=(values[:,:,step*nr+jnp.arange(nr)] if compensated else shaped[:,:,atom,h])
                d=jnp.einsum('bmr,bgr->bmg',source,bessel[:,lrow])
                g=jnp.einsum('bmd,bdg->bmg',qm[:,:,atom,h],g_radial[:,lrow])
                a=angle[:,atom,h]
                return (acc[0]+d*a[:,None],acc[1]+g*a[:,None]),None
            d,g=jax.lax.scan(add,(zero,zero),jnp.arange(na*nh),unroll=1)[0]
            outputs.append(d);compensations.append(g)
        return tuple(outputs),tuple(compensations)
    fourier_kernel=jax.jit(shard_map(fourier_local,mesh=mesh,
        in_specs=(coefficient_specs,(qspec,)*3,P(XY,None,None,None),
                  P(XY,None,None,None),P(XY,None,None,None),P()),
        out_specs=((qspec,)*3,(qspec,)*3),check_vma=False))

    def coupled_field(values,c,atom,m0,modes_,radial_width,hrows_,spin_weights_,feature_offset=0):
        # Same sparse Cartesian/VSH projection for both onsite layouts.
        wanted=m0+jnp.arange(mt);sample=jnp.minimum(wanted,mu-1)
        feature=feature_offset+(atom*nh+hrows_[modes_])[:,:,None]*radial_width+jnp.arange(radial_width)
        panel=values[:,sample[:,None],feature.reshape(-1)[None]]
        panel=panel.reshape(values.shape[0],mt,modes_.shape[0],3,radial_width)
        panel=jnp.sum(panel*spin_weights_[c,modes_][None,None,:,:,None],axis=3)
        qlive=live(values.shape[0])
        panel=jnp.where(qlive[:,None,None,None]&(wanted<mu)[None,:,None,None],panel,0)
        return panel.reshape(values.shape[0],mt,-1)

    def field_panels(coefficients,moment_coefficients,hrows_,spin_weights_,df,gf,atom,modes_,m0):
        d=[];g=[];ps=[]
        for c in range(3):
            delta_values=coefficients[c][0] if separate_halves else coefficients[c]
            source=coupled_field(delta_values,c,atom,m0,modes_,nr,hrows_,spin_weights_)
            moment=coupled_field(moment_coefficients[c],c,atom,m0,modes_,2,hrows_,spin_weights_)
            d.append(jnp.einsum('fr,bmr->bmf',df,source))
            g.append(jnp.einsum('fr,bmr->bmf',gf,moment))
            if compensated:
                smooth_values=coefficients[c][1] if separate_halves else coefficients[c]
                smooth=coupled_field(smooth_values,c,atom,m0,modes_,nr,hrows_,spin_weights_,
                                     0 if separate_halves else nf)
                ps.append(jnp.einsum('fr,bmr->bmf',df,smooth))
        return tuple(d),tuple(g),tuple(ps)

    onsite_kernels={}
    def make_onsite(pairs):
        pairs=tuple((int(a),int(b)) for a,b in pairs)
        if pairs in onsite_kernels:return onsite_kernels[pairs]
        def onsite_local(coefficients,moment_coefficients,hrows_,spin_weights_,buckets_):
            first_values = coefficients[0][0] if separate_halves else coefficients[0]
            zero=jnp.zeros((first_values.shape[0],len(pairs),mup,mup),jnp.complex128)
            result=zero
            for (width,tile,nfields,nsteps),(delta_factor,comp_factor,steps) in zip(bucket_meta,buckets_):
                def sector_field(acc,step):
                    atom=step//(nsteps*nfields);entry=(step//nfields)%nsteps;fblock=step%nfields
                    sector=steps[entry,0];modes_=steps[entry,1:];r0=fblock*tile
                    df=jax.lax.dynamic_slice_in_dim(delta_factor[sector],r0,tile,axis=0)
                    gf=jax.lax.dynamic_slice_in_dim(comp_factor[sector],r0,tile,axis=0)
                    def fields(m0):
                        return field_panels(coefficients,moment_coefficients,hrows_,spin_weights_,
                                            df,gf,atom,modes_,m0)
                    def left_block(acc,lblock):
                        m0=lblock*mt;dleft,gleft,psleft=fields(m0)
                        def right_block(acc,rblock):
                            n0=rblock*mt;dright,gright,psright=fields(n0)
                            values=[]
                            for a,b in pairs:
                                inner=lambda l,r:jnp.einsum('bmf,bnf->bmn',l.conj(),r)
                                signed=inner(dleft[a],dright[b])
                                signed-=inner(gleft[a],gright[b])
                                if compensated:
                                    signed+=(inner(psleft[a],dright[b]-gright[b])
                                             +inner(dleft[a]-gleft[a],psright[b]))
                                values.append(onsite_scale*signed)
                            block=jnp.stack(values,axis=1)
                            old=jax.lax.dynamic_slice(acc,(0,0,m0,n0),(acc.shape[0],len(pairs),mt,mt))
                            return jax.lax.dynamic_update_slice(acc,old+block,(0,0,m0,n0)),None
                        return jax.lax.scan(right_block,acc,jnp.arange(nmu),unroll=1)[0],None
                    return jax.lax.scan(left_block,acc,jnp.arange(nmu),unroll=1)[0],None
                result=jax.lax.scan(sector_field,result,jnp.arange(na*nsteps*nfields),unroll=1)[0]
            return tuple(result[:,i,:mu,:mu] for i in range(len(pairs)))
        kernel=jax.jit(shard_map(onsite_local,mesh=mesh,
            in_specs=(coefficient_specs,(qspec,)*3,P(),P(),tuple((P(),P(),P()) for _ in bucket_arrays)),
            out_specs=(qspec,)*len(pairs),check_vma=False))
        onsite_kernels[pairs]=kernel;return kernel

    # Concatenated independent field coordinates add their bilinears exactly.
    # Keep 32 existing bounded field blocks per packet: no full field bank and
    # no collective for every magnetic/Cartesian/radial micro-iteration.
    from common.collectives import transpose_xy
    from distrib_la import gemm_plan
    from runtime.padding import padded_axis,pad_to_axis
    from isdf.zeta_mubatch import _to_mu_owner
    face_spec=P(None,'x','y');face_sharding=NamedSharding(mesh,face_spec)
    to_face=_to_mu_owner(mesh,Q,'xy')
    field_blocks_per_packet=32
    face_onsite_kernels={}
    face_packet_owners=[]
    for width,tile,nfields,nsteps in bucket_meta:
        count=na*nsteps*nfields
        coordinate_axis=padded_axis(field_blocks_per_packet*tile,mesh,
            name='current onsite packet field coordinates',
            specs=((face_spec,1),(face_spec,2)))
        kfield=coordinate_axis.carrier
        positive=gemm_plan(mesh,m=mu,n=mu,k=kfield,nq=Q,dtype=jnp.complex128,
            layout='face',alpha=onsite_scale,beta=1.,warmup=False)
        negative=gemm_plan(mesh,m=mu,n=mu,k=kfield,nq=Q,dtype=jnp.complex128,
            layout='face',alpha=-onsite_scale,beta=1.,warmup=False)

        def packet_owner(tile,nfields,nsteps,count,coordinate_axis):
            def project(coefficients,moment_coefficients,hrows_,spin_weights_,
                        delta_factor,comp_factor,steps,packet):
                first_values=coefficients[0][0] if separate_halves else coefficients[0]
                shape=(first_values.shape[0],mup,coordinate_axis.carrier)
                fields=tuple(jnp.zeros(shape,jnp.complex128) for _ in range(9 if compensated else 6))
                def field_block(acc,slot):
                    wanted=packet*field_blocks_per_packet+slot
                    step=jnp.minimum(wanted,count-1)
                    atom=step//(nsteps*nfields);entry=(step//nfields)%nsteps;fblock=step%nfields
                    sector=steps[entry,0];modes_=steps[entry,1:];r0=fblock*tile
                    df=jax.lax.dynamic_slice_in_dim(delta_factor[sector],r0,tile,axis=0)
                    gf=jax.lax.dynamic_slice_in_dim(comp_factor[sector],r0,tile,axis=0)
                    def centroid_block(current,block):
                        m0=block*mt
                        d,g,ps=field_panels(coefficients,moment_coefficients,hrows_,spin_weights_,
                                            df,gf,atom,modes_,m0)
                        values=d+g+ps
                        values=tuple(jnp.where(wanted<count,v,0.) for v in values)
                        return tuple(jax.lax.dynamic_update_slice(a,v,(0,m0,slot*tile))
                                     for a,v in zip(current,values)),None
                    return jax.lax.scan(centroid_block,acc,jnp.arange(nmu),unroll=1)[0],None
                fields=jax.lax.scan(field_block,fields,
                    jnp.arange(field_blocks_per_packet),unroll=1)[0]
                return tuple(pad_to_axis(value[:,:mu],coordinate_axis,axis=2) for value in fields)
            return jax.jit(shard_map(project,mesh=mesh,
                in_specs=(coefficient_specs,(qspec,)*3,P(),P(),P(),P(),P(),P()),
                out_specs=(qspec,)*(9 if compensated else 6),check_vma=False))

        face_packet_owners.append((packet_owner(tile,nfields,nsteps,count,coordinate_axis),
            positive,negative,(count+field_blocks_per_packet-1)//field_blocks_per_packet,
            coordinate_axis))

    def make_onsite_face(pairs):
        pairs=tuple((int(a),int(b)) for a,b in pairs)
        if pairs in face_onsite_kernels:return face_onsite_kernels[pairs]
        def contract(coefficients,moment_coefficients,hrows_,spin_weights_,buckets_):
            result=tuple(jax.lax.with_sharding_constraint(
                jnp.zeros((Q,mu,mu),jnp.complex128),face_sharding) for _ in pairs)
            for (project,positive,negative,npackets,_),bucket in zip(face_packet_owners,buckets_):
                def packet(acc,index):
                    panels=project(coefficients,moment_coefficients,hrows_,spin_weights_,*bucket,index)
                    panels=tuple(to_face(value) for value in panels)
                    d,g=panels[:3],panels[3:6]
                    ps=panels[6:]
                    dt=tuple(transpose_xy(value,mesh) for value in d)
                    gt=tuple(transpose_xy(value,mesh) for value in g)
                    pt=tuple(transpose_xy(value,mesh) for value in ps)
                    values=[]
                    for previous,(a,b) in zip(acc,pairs):
                        value=positive(d[a].conj(),dt[b],previous)
                        value=negative(g[a].conj(),gt[b],value)
                        if compensated:
                            value=positive(ps[a].conj(),dt[b]-gt[b],value)
                            value=positive((d[a]-g[a]).conj(),pt[b],value)
                        values.append(value)
                    return tuple(values),None
                result=jax.lax.scan(packet,result,jnp.arange(npackets),unroll=1)[0]
            return result
        kernel=jax.jit(contract,
            in_shardings=(tuple((qsh,qsh) if separate_halves else qsh for _ in range(3)),(qsh,)*3,
                          NamedSharding(mesh,P()),NamedSharding(mesh,P()),
                          tuple((NamedSharding(mesh,P()),)*3 for _ in bucket_arrays)),
            out_shardings=(face_sharding,)*len(pairs))
        face_onsite_kernels[pairs]=kernel
        return kernel

    tail_observer = None
    if physical_mixed:
        from isdf.current_tail import laplacian3_positive_map,make_sobolev_norm_kernel
        tail_map = laplacian3_positive_map(compensation)
        tail_kernel,tail_positive,tail_absolute = make_sobolev_norm_kernel(mesh,mu=mu,atoms=na,
            lm=labels,positive_map=tail_map,fft_points=N,volume=volume)
        tail_observer = (tail_kernel,put(tail_positive),put(tail_absolute))

    def prepare(coefficients):
        if owned_banks:
            if coefficients is not None:
                raise ValueError('streamed current provider consumes only its authenticated half-bank owner')
            owned = coefficient_owner.authenticate(zetas)
            coefficients = owned if separate_halves else tuple(v[0] for v in owned)
        if len(coefficients)!=3:raise ValueError("static current coefficients require all three channels")
        for v in coefficients:
            for half in v if separate_halves else (v,):
                _require_q_owned(half,mesh,(Qp,mu,nf if separate_halves else (2 if compensated else 1)*nf),
                                 name='static current coefficients')
        coefficients=tuple(coefficients)
        moments = prepare_kernel(coefficients,M)
        if tail_observer is not None:
            kernel,L,A = tail_observer
            norms = kernel(moments,L,A)
            sink = bundle.get('compensation_tail_diagnostics')
            if not isinstance(sink,dict):
                raise ValueError('physical current tail observer requires its actual diagnostic sink')
            sink['weighted_fourier_K12_norm_upper'] = norms
            sink['body_cutoff_Ry'] = float(body_head_slots['cutoff_ry'])
            sink['norm_tail_divisor'] = float(body_head_slots['cutoff_ry'])**6
            sink['bare_Sigma_energy_prefactor'] = 2*abs(k0)/float(body_head_slots['cutoff_ry'])**7
            sink['rounding_relative_envelope'] = tail_map['rounding_relative_envelope']
            sink['qualification'] = 'Actual fitted analytic compensation moments; occupied-current Sigma covectors/weights still required; no screened response bound'
        return dict(coefficients=coefficients,moments=moments)

    def put_q_rows(builder,tail):
        # Build only this process's addressed q rows, never a Q*G*radial
        # host table on every rank. The public callback transports no peers.
        shape=(Qp,*tail);sh=NamedSharding(mesh,P(XY,*([None]*len(tail))))
        return jax.make_array_from_callback(shape,sh,lambda index:builder(index[0]))

    def fourier_tile(t,state):
        first=int(t)*gt
        def vectors(qslice):
            qidx=np.arange(Qp)[qslice];vec=np.zeros((len(qidx),gt,3))
            valid=np.zeros((len(qidx),gt),bool);n=max(0,min(gt,ng-first))
            real=qidx<Q
            if n and np.any(real):
                vec[real,:n]=K[qidx[real],first:first+n]
                valid[real]=first+np.arange(gt)[None]<counts[qidx[real],None]
            return vec,valid
        def source_rows(qslice):
            vec,_=vectors(qslice)
            return np.moveaxis(cache['density'](np.linalg.norm(vec,axis=-1)),0,1)
        def comp_rows(qslice):
            vec,_=vectors(qslice);length=np.linalg.norm(vec,axis=-1)
            transformed=two_moment_compensation_radial_fourier(compensation,length.ravel())
            return transformed.reshape(len(radial['degrees']),2,*length.shape).transpose(2,0,1,3)
        def angle_rows(qslice):
            vec,valid=vectors(qslice);length=np.linalg.norm(vec,axis=-1)
            theta=np.arccos(np.clip(np.divide(vec[...,2],length,out=np.ones_like(length),where=length>0),-1,1))
            phi=np.arctan2(vec[...,1],vec[...,0])
            Y=np.stack([4*np.pi*(-1j)**int(l)*sph_harm_y(l,m,theta,phi) for l,m in labels],axis=1)
            phase=np.exp(-1j*np.einsum('bgi,ai->bag',vec,centers))
            return phase[:,:,None]*Y[:,None]*valid[:,None,None]*fourier_scale
        result=fourier_kernel(state['coefficients'],state['moments'],
            put_q_rows(source_rows,(len(radial['degrees']),gt,nr)),
            put_q_rows(angle_rows,(na,nh,gt)),
            put_q_rows(comp_rows,(len(radial['degrees']),2,gt)),rows)
        if slab and np.any(active[:,first:first+gt]&(norm[:,first:first+gt]**2<=CURRENT_GAMMA_K2_EPS)):
            from common.collectives import gather_to_host
            def head_rows(qslice):
                vec,valid=vectors(qslice)
                return valid&(np.einsum('qgi,qgi->qg',vec,vec)<=CURRENT_GAMMA_K2_EPS)
            marker=put_q_rows(head_rows,(gt,))
            error,scale=neutral_head_kernel(*result,marker)
            error,scale=float(gather_to_host(error)),float(gather_to_host(scale))
            if error>2e-12+256*np.finfo(float).eps*scale:
                raise ValueError('Slab current completion requires actual delta(0)=g(0) in its represented Fourier tiles')
            bundle['gamma_neutral_representation_error']=max(
                bundle.get('gamma_neutral_representation_error',0.),error)
        return result

    if slab:
        def neutral_head_local(delta,g,marker):
            error=jnp.max(jnp.stack([jnp.max(jnp.where(marker[:,None],abs(a-b),0))
                                   for a,b in zip(delta,g)]))
            scale=jnp.max(jnp.stack([jnp.max(jnp.where(marker[:,None],jnp.maximum(abs(a),abs(b)),0))
                                   for a,b in zip(delta,g)]))
            return jax.lax.pmax(error,XY),jax.lax.pmax(scale,XY)
        neutral_head_kernel=jax.jit(shard_map(neutral_head_local,mesh=mesh,
            in_specs=((qspec,)*3,(qspec,)*3,P(XY,None)),out_specs=(P(),P()),check_vma=False))

    def kernel_difference_tile(t,pairs,v_tables):
        """D_target - D_bulk-point, keeping only the addressed q/G packet."""
        first=int(t)*gt
        def build(qslice):
            qids=np.arange(Qp)[qslice]
            difference=-_current_point_body_tile(body_head_slots,qids,first,gt,pairs,B,k0,sys_dim=3)
            n=max(0,min(gt,ng-first))
            for slot,q in enumerate(qids):
                if q>=Q:continue
                for index,table in enumerate(v_tables):
                    difference[slot,index,:n]+=np.asarray(table)[q,first:first+n]
            return difference
        return put_q_rows(build,(len(pairs),gt))

    def onsite(state,pairs):
        return make_onsite(pairs)(state['coefficients'],state['moments'],
                                  sparse_rows,sparse_weights,bucket_arrays)

    def onsite_face(state,pairs):
        return make_onsite_face(pairs)(state['coefficients'],state['moments'],
                                     sparse_rows,sparse_weights,bucket_arrays)

    def authenticate_kernel(pairs,v_tables):
        if (bundle.get('kernel')!=(SLAB_STATIC_KERNEL if slab else STATIC_KERNEL)
                or bundle.get('body_metric')!=body_metric
                or bundle.get('local_cross_policy')!=('onsite_smooth_neutral' if compensated
                                                     else 'mixed_reciprocal')):
            raise ValueError("static Breit provider body/cross policy was changed after construction")
        if physical_mixed:
            if bundle.get('smooth_support') is not smooth_support:
                raise ValueError('physical mixed support owner changed')
            smooth_support.authenticate(zetas, body_head_slots, centers_cart=centers, support_radius=R)
        completion_kind = 'nn_tn_nt_slab_minus_bulk_point_body'
        if slab and not physical_mixed and (bundle.get('completion')!=completion_kind
                or bundle.get('bulk_reference')!='public_Bulk3D_point_body_no_head'
                or bundle.get('kernel_difference_tile') is not kernel_difference_tile):
            raise ValueError('Slab current completion or point-body reference was changed after construction')
        if body_head_slots is not None:
            if (body_head_slots.get('payload_sha256')!=body_head_identity
                    or bundle.get('body_head_slots_sha256')!=body_head_identity):
                raise ValueError('static Breit canonical body-head identity was changed after construction')
            _authenticate_body_head_slots(body_head_slots,K,counts,volume,mask)
        pairs=tuple((int(a),int(b)) for a,b in pairs)
        if (not pairs or len(pairs)!=len(v_tables) or any(a<0 or b<0 or a>=3 or b>=3 for a,b in pairs)
                or len(set(pairs))!=len(pairs) or {i for p in pairs for i in p}!={0,1,2}):
            raise ValueError("static Breit requires valid distinct current-kernel pairs")
        if physical_mixed:
            smooth_support.authenticate_point_kernel(pairs, v_tables, B, k0)
            return
        hfit=np.einsum('ai,qij,bj->qab',B,head,B.conj())
        for (a,b),passed in zip(pairs,v_tables):
            value=np.asarray(passed,dtype=complex)
            if value.shape!=(Q,ng) or not np.all(np.isfinite(value)):
                raise ValueError("static Breit kernel tables disagree with physical q/G geometry")
            for first in range(0,ng,gt):
                last=min(first+gt,ng);v=K[:,first:last];length=norm[:,first:last]
                unit=np.divide(v,length[:,:,None],out=np.zeros_like(v),where=length[:,:,None]>0)
                ua=np.einsum('i,qgi->qg',B[a],unit);ub=np.einsum('i,qgi->qg',B[b],unit)
                projection=float(a==b)-ua*ub.conj()
                expected=np.divide(k0*projection,length*length,out=np.zeros_like(projection),where=length>0)
                if slab:
                    expected=_current_point_body_tile(body_head_slots,np.arange(Q),first,last-first,
                        ((a,b),),B,k0,sys_dim=2)[:,0]
                if head_selected is not None:
                    scalar=np.asarray(body_head_slots['v_avg'])*k0/(8*np.pi/volume)
                    expected=np.where(head_selected[:,first:last],scalar[:,None]*projection,expected)
                    expected=np.where(length*length<1e-12,0,expected)
                expected*=mask[:,first:last]
                expected=np.where(length*length<=CURRENT_GAMMA_K2_EPS,hfit[:,a,b,None],expected)
                expected*=active[:,first:last]
                error=np.abs(value[:,first:last]-expected)
                if np.any(error>2e-12+2e-11*np.abs(expected)):
                    raise ValueError("static Breit provider kernel identity/scale/basis/head mismatch")

    digest=hashlib.sha256()
    digest.update(body_metric.encode('ascii'))
    if physical_mixed:
        import json
        digest.update(json.dumps(smooth_support.descriptor,sort_keys=True).encode('utf8'))
    if body_head_slots is not None:
        digest.update(body_head_slots['payload_sha256'].encode('ascii'))
    for v in (r,labels,centers,K,B,head,mask,
              np.asarray([volume,N,R,k0,minimum_atom_image_distance,
                          interpolation_degree,quadrature_order,ft,mt])):
        digest.update(np.ascontiguousarray(v).tobytes())
    bundle=dict(kernel=SLAB_STATIC_KERNEL if slab else STATIC_KERNEL,geometry_sha256=digest.hexdigest(),
        zeta_identity=tuple(id(z) for z in zetas),prepare=prepare,
        fourier_tile=fourier_tile,onsite=onsite,authenticate_kernel=authenticate_kernel,
        reciprocal_prefactor=k0,local_geometry_scale=onsite_scale,current_basis_rows=B,
        body_metric=body_metric,local_cross_policy=('onsite_smooth_neutral' if compensated
                                                   else 'mixed_reciprocal'),
        body_head_slots_sha256=body_head_identity,
        gamma_placement=(None if body_head_slots is None else body_head_slots['gamma_placement']),
        gamma_zero_moment_operator_error=gamma_origin_error,
        radial_fourier_diagnostics={key:cache[key] for key in ('points','maximum_wavevector',
            'retained_spline_bytes','max_density_validation_error','validation_points')},
        field_tile=ft,centroid_tile=mt,geometry=geometry,compensation=compensation,
        prepare_kernel=prepare_kernel,fourier_kernel=fourier_kernel,
        onsite_kernel=make_onsite,
        onsite_face=onsite_face,onsite_face_kernel=make_onsite_face,
        onsite_face_packet_diagnostics=dict(
            field_blocks_per_packet=field_blocks_per_packet,
            packet_field_carriers=[axis.carrier for *_,axis in face_packet_owners],
            packets=sum(owner[3] for owner in face_packet_owners),
            q_count=Q,output_layout='P(None,x,y)',
            field_projection='same exact sparse Cartesian/VSH and radial delta/compensation factors',
            contract='D_a.conj() @ transpose_xy(D_b) - G_a.conj() @ transpose_xy(G_b)',
            coordinate_padding='runtime PaddedAxis; only zero added field coordinates'),
        onsite_kernel_arguments=(sparse_rows,sparse_weights,bucket_arrays),
        onsite_block_diagnostics={key:blocks[key] for key in ('angular_forbidden_error',
            'angular_M_covariance_error','retained_columns','retained_compensation_columns',
            'factor_bytes','no_cutoff')},onsite_sparse_max_orbitals=3,coefficient_cloud_padding=False)
    if slab and not physical_mixed:
        bundle.update(kernel_difference_tile=kernel_difference_tile,
            completion='nn_tn_nt_slab_minus_bulk_point_body',
            bulk_reference='public_Bulk3D_point_body_no_head',
            gamma_neutral_representation_error=0.,
            completion_workspace_bytes_per_rank=16.*Qp/int(mesh.size)*(9*gt+2*mu*gt+2*mu*mu))
    # All geometry/head/source contracts have refused invalid inputs before
    # allocating the paired clouds. The consuming stage blocks these arrays
    # and releases its raw six-cloud operands before grouped solves begin.
    if physical_mixed:
        bundle['smooth_support'] = smooth_support
        bundle['physical_mixed_support'] = smooth_support.descriptor
        bundle['physical_tail_qualification'] = 'Slab high-G local remainder proved by layer geometry; global compensation tail needs full fitted/occupied stopping bound'
    if owned_banks:
        bundle['coefficient_owner'] = coefficient_owner
        return tuple(dict(coefficient_owner=coefficient_owner, group_provider=bundle,
                          channel=c) for c in range(3))
    provider_rhs=(tuple(jnp.concatenate((delta,smooth),axis=-1)
                        for delta,smooth in zip(rhs,smooth_rhs)) if compensated else tuple(rhs))
    return tuple(dict(rhs=provider_rhs[c],group_provider=bundle,channel=c) for c in range(3))
