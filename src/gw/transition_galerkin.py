"""Internal full-port transition actions; no production route or pole fitting.

The ordered response is P(z)=T K_rho (z-D)^-1 T.H, not T(z-D)^-1K T.H.
Native wavefunction faces remain distributed. Transition rows are k-major
and sharded over the entire mesh; every physical ISDF port remains present.
"""
from functools import partial
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P


def physical_transition_normalization(nk, nspin=1, nspinor=2):
    """Square-root physical spin/k weight, INCLUDING raw FFT normalization."""
    if min(nk,nspin,nspinor)<1:raise ValueError('GATE transition_normalization: positive census required')
    return float(np.sqrt(2/(int(nk)*int(nspin)*int(nspinor))))


def transition_catalogue(energy, rho, kmq, *, coherent_slice=None, pair_tile=32):
    """Exact nonzero K-column catalogue, padded independently inside each k.

    Optional coherent_slice reserves ALL its columns for a held numerical
    catalogue, regardless of current diagonal occupation. With no reservation
    an ensuing state change must rebuild if it activates an omitted column.
    No numerical occupation threshold is used. Return a plain array mapping.
    """
    energy=np.asarray(energy,float);rho=np.asarray(rho,complex);kmq=np.asarray(kmq,np.int32)
    nk,nb=energy.shape
    if (rho.shape!=(nk,nb,nb) or kmq.shape!=(nk,) or pair_tile<1
            or not np.array_equal(np.sort(kmq),np.arange(nk))
            or not np.isfinite(energy).all() or not np.isfinite(rho).all()
            or np.max(abs(rho-rho.conj().swapaxes(-1,-2)))>1e-10):
        raise ValueError('GATE transition_catalogue: finite Hermitian density and exact k-q permutation required')
    diag=rho.diagonal(axis1=-2,axis2=-1)
    off=rho.copy();ids=np.arange(nb);off[:,ids,ids]=0
    mixed=np.any(off!=0,axis=1)
    live=(mixed[:,:,None]|mixed[kmq,None,:]|(diag[:,:,None]!=diag[kmq,None,:]))
    if coherent_slice is not None:
        lo,hi=coherent_slice.start,coherent_slice.stop
        if not 0<=lo<hi<=nb:raise ValueError('GATE transition_catalogue: coherent slice outside band carrier')
        active=(ids>=lo)&(ids<hi)
        live|=active[None,:,None]|active[None,None,:]
    counts=live.sum(axis=(1,2));maximum=int(counts.max(initial=0))
    width=max(pair_tile,((maximum+pair_tile-1)//pair_tile)*pair_tile)
    b=np.zeros((nk,width),np.int32);d=b.copy();valid=np.zeros((nk,width),bool)
    for k in range(nk):
        bk,dk=np.nonzero(live[k]);n=len(bk)
        b[k,:n]=bk;d[k,:n]=dk;valid[k,:n]=True
    return dict(b=b,d=d,valid=valid,kmq=kmq,
        receipt=dict(n_k=nk,n_bands=nb,live_transitions=int(counts.sum()),
            transition_carrier=nk*width,pairs_per_k=width,per_k_live=counts.tolist(),
            pair_tile=int(pair_tile),population_threshold=None,
            coherent_reservation=None if coherent_slice is None else [lo,hi]))


def transition_capacity(catalogue, nmu, capacity, *, processes=1, max_block_size=32):
    """Literal live-array prices, not a measured performance/admission claim."""
    nk,width=catalogue['valid'].shape;nt=nk*width;p=int(processes)
    if min(nmu,capacity,p,max_block_size)<1:raise ValueError('GATE transition_capacity: positive sizes required')
    return dict(transition_rows=nt,live_transitions=int(catalogue['valid'].sum()),full_ports=int(nmu),
        basis_bytes_global=16*nt*capacity,vertex_bytes_global=16*nt*nmu,
        basis_bytes_per_rank=(16*nt*capacity+p-1)//p,
        vertex_bytes_per_rank=(16*nt*nmu+p-1)//p,
        K_scratch_bytes_per_rank=(16*nk*(3*int(catalogue['receipt']['n_bands'])**2+2*width)*max_block_size+p-1)//p,
        retained_parent_count=1,all_parent_retention_admitted=False,
        note='Add provider QR/CGS workspace and source resident arrays before material admission')


def validate_transition_catalogue(catalogue, rho):
    """Refuse a held catalogue if current rho activates any omitted K column."""
    nk,nb,_=np.asarray(rho).shape
    needed=transition_catalogue(np.zeros((nk,nb)),rho,catalogue['kmq'],pair_tile=1)
    present=np.zeros((nk,nb,nb),bool)
    for k in range(nk):
        mask=catalogue['valid'][k]
        present[k,catalogue['b'][k,mask],catalogue['d'][k,mask]]=True
        live=needed['valid'][k]
        if not present[k,needed['b'][k,live],needed['d'][k,live]].all():
            raise ValueError('GATE transition_catalogue_changed: current density activates omitted transition rows')
    return True


def make_transition_bridge(mesh, psi_shape, n_pairs, *, normalization,
                           layout='face', pair_tile=32):
    """Return distributed full-port T/T.H, K_rho and D actions on plain arrays.

    psi_shape=(nk,nb,spin,mu), psi_nmu uses the native layout. T[k,p,mu]
    and states[r,k,p] shard WHOLE k over ('x','y'); port panels[mu,r] are
    bounded replicated blocks, not a truncated port basis. All current arrays
    (including catalogue/permutation) are function operands, never closures.
    """
    from common.wfn_layout import psi_specs
    nk,nb,ns,nmu=map(int,psi_shape);width=int(n_pairs)
    if (min(nk,nb,ns,nmu,width,pair_tile)<1 or nk%int(mesh.size) or width%pair_tile
            or not np.isfinite(normalization) or normalization<=0):
        raise ValueError('GATE transition_layout: whole-k sharding and tiled pairs required')
    rep=NamedSharding(mesh,P());kpsi=NamedSharding(mesh,P(('x','y'),None,None,None))
    tsh=NamedSharding(mesh,P(('x','y'),None,None));state=NamedSharding(mesh,P(None,('x','y'),None))
    nmu_spec,_=psi_specs(layout)
    psi_input=NamedSharding(mesh,nmu_spec)
    @partial(jax.shard_map,mesh=mesh,
             in_specs=(P(('x','y'),None,None,None),)*2+(P(('x','y'),None),)*3,
             out_specs=P(('x','y'),None,None),check_vma=False)
    def vertex_local(psi,partner,b,d,valid):
        local_nk=psi.shape[0]
        zero=jnp.zeros((local_nk,width,nmu),jnp.complex128)
        def step(out,index):
            begin=index*pair_tile
            bb=jax.lax.dynamic_slice(b,(0,begin),(local_nk,pair_tile))
            dd=jax.lax.dynamic_slice(d,(0,begin),(local_nk,pair_tile))
            mask=jax.lax.dynamic_slice(valid,(0,begin),(local_nk,pair_tile))
            left=jax.vmap(lambda x,i:x[i])(psi,bb)
            right=jax.vmap(lambda x,i:x[i])(partner,dd)
            value=normalization*jnp.einsum('kpsm,kpsm->kpm',left,right.conj())*mask[:,:,None]
            return jax.lax.dynamic_update_slice(out,value,(0,begin,0)),None
        return jax.lax.scan(step,zero,jnp.arange(width//pair_tile))[0]
    @partial(jax.jit,in_shardings=(psi_input,rep,rep,rep,rep),out_shardings=tsh)
    def vertex(psi,kmq,b,d,valid):
        psi=jax.lax.with_sharding_constraint(psi,kpsi)
        partner=jax.lax.with_sharding_constraint(psi[kmq],kpsi)
        return vertex_local(psi,partner,b,d,valid)
    @partial(jax.jit,in_shardings=(tsh,rep),out_shardings=state)
    def to_state(t,u):
        return jnp.einsum('kpm,mr->rkp',t.conj(),u)
    @partial(jax.shard_map,mesh=mesh,in_specs=(P(('x','y'),None,None),P(None,('x','y'),None)),
             out_specs=P(),check_vma=False)
    def to_port_local(t,x):
        from lxkit import ordered_psum
        return ordered_psum(jnp.einsum('kpm,rkp->mr',t,x),('x','y'))
    to_port=jax.jit(to_port_local,in_shardings=(tsh,state),out_shardings=rep)
    @partial(jax.shard_map,mesh=mesh,in_specs=(P(None,('x','y'),None),)*2,
             out_specs=P(),check_vma=False)
    def gram_local(x,y):
        from lxkit import ordered_psum
        return ordered_psum(jnp.einsum('ikp,jkp->ij',x.conj(),y),('x','y'))
    gram=jax.jit(gram_local,in_shardings=(state,state),out_shardings=rep)
    @partial(jax.jit,in_shardings=(state,rep,rep),out_shardings=state)
    def cross(x,indices,valid):
        mirrored=jnp.take(x.conj().reshape(x.shape[0],-1),indices.reshape(-1),axis=1)
        return mirrored.reshape(x.shape)*valid[None]
    @partial(jax.jit,in_shardings=(state,rep,rep,rep,rep,rep),out_shardings=state)
    def K(x,rho,kmq,b,d,valid):
        def unpack(row,bi,di,mask):
            return jnp.zeros((nb,nb),x.dtype).at[bi,di].add(row*mask)
        full=jax.vmap(jax.vmap(unpack,in_axes=(0,0,0,0)),in_axes=(0,None,None,None))(x,b,d,valid)
        value=jnp.einsum('rkbd,kde->rkbe',full,rho[kmq])-jnp.einsum('kab,rkbd->rkad',rho,full)
        indices=(b*nb+d)[None]
        return jnp.take_along_axis(value.reshape((x.shape[0],nk,nb*nb)),indices,axis=-1)*valid[None]
    @partial(jax.jit,in_shardings=(state,rep,rep,rep,rep,rep,rep),out_shardings=state)
    def D(x,energy,kmq,b,d,valid,gamma):
        eb=jnp.take_along_axis(energy,b,axis=1)
        ed=jnp.take_along_axis(energy[kmq],d,axis=1)
        return x*(eb-ed-1j*gamma)[None]*valid[None]
    return dict(vertex=vertex,to_state=to_state,to_port=to_port,K=K,D=D,
        gram=gram,
        cross=cross,
        state_sharding=state,vertex_sharding=tsh,port_sharding=rep,
        mesh=mesh,
        receipt=dict(full_ports=nmu,normalization=float(normalization),physical_transition_projection=False,
                     k_sharding='whole-k over x,y',pair_tile=pair_tile))


def transition_crossing_map(source, target):
    """Exact q -> -q state-row permutation (conjugation supplied by caller)."""
    b,d,mask=source['b'],source['d'],source['valid'];nk,width=b.shape
    inverse=np.argsort(source['kmq']);out=np.zeros(target['b'].shape,np.int32)
    for k in range(nk):
        original=int(inverse[k]);lookup={(int(bb),int(dd)):i for i,(bb,dd,v) in
            enumerate(zip(b[original],d[original],mask[original])) if v}
        for j in np.flatnonzero(target['valid'][k]):
            pair=(int(target['d'][k,j]),int(target['b'][k,j]))
            if pair not in lookup:raise ValueError('GATE transition_crossing: unmatched exact live row')
            out[k,j]=original*width+lookup[pair]
    return out


def dyson_transition_seeds(bridge, t, catalogue, energy, rho, z,
                           right_potential, left_potential, gamma):
    """Exact right/dual states from CURRENT full-W port actions.

    right_potential=W(z)r and left_potential=W(z).H l retain all ports.
    The dual order is R(z).H K T.H W(z).H l; K never commutes through R.
    """
    validate_transition_catalogue(catalogue,np.asarray(rho))
    b,d,valid,kmq=(jnp.asarray(catalogue[k]) for k in ('b','d','valid','kmq'))
    right=bridge['to_state'](t,right_potential)
    left=bridge['K'](bridge['to_state'](t,left_potential),rho,kmq,b,d,valid)
    diagonal=bridge['D'](jnp.ones_like(right[:1]),energy,kmq,b,d,valid,jnp.asarray(gamma))
    return right/(z-diagonal)*valid[None],left/(complex(z).conjugate()-diagonal.conj())*valid[None]


def _canonical_qr_rotation(r, rank_tolerance, reference_norm):
    """Canonical small-R SVD with scale-relative probe independence.

    All ranks gather before pure host root work. This is a residual-probe
    admission, not an overlap inverse: ``reference_norm`` is the original
    seed block norm before CGS2. Refuse a cut through a singular multiplet.
    """
    import scipy.linalg as la
    from common.collectives import gather_to_host,all_gather_processes
    from common.rank_criterion import probe_is_independent
    from common.spectral_closure import resolve_spectral_cut
    from jax.experimental.multihost_utils import broadcast_one_to_all
    small=np.asarray(gather_to_host(r));width=small.shape[0]
    transform=None;message='';count=0
    if jax.process_index()==0:
        try:
            u,s,_=la.svd(small,full_matrices=False,check_finite=False)
            family_scale=float(s[0]) if len(s) else 0.
            count=sum(probe_is_independent(value,reference_norm,family_scale,
                rtol=rank_tolerance) for value in s)
            count,_=resolve_spectral_cut(s,count,mode='strict',
                where='transition residual seed QR',rcond=rank_tolerance)
            transform=np.ascontiguousarray(u[:,:count],dtype=complex)
        except Exception as error:message=str(error).encode()[:1000].decode(errors='replace')
    error_bytes=message.encode()
    control=np.asarray(all_gather_processes(np.array([count,len(error_bytes)],np.int64)))[0]
    count,length=map(int,control)
    if length:
        error=np.zeros(length,np.uint8)
        if jax.process_index()==0:error[:]=np.frombuffer(error_bytes,np.uint8)
        message=np.asarray(all_gather_processes(error))[0].tobytes().decode()
        raise ValueError('GATE transition_qr: '+message)
    if not jax.config.x64_enabled:raise ValueError('GATE transition_precision: x64 required')
    if transform is None:transform=np.zeros((width,count),complex)
    if count:
        bits=np.asarray(broadcast_one_to_all(transform.view(np.uint64),is_source=jax.process_index()==0))
        transform=bits.view(complex).reshape(width,count)
    return transform


def transition_basis(bridge, seed_blocks, catalogue, rho, *, capacity,
                     rank_tolerance, max_block_size=32, max_device_bytes=None,
                     self_crossing=None):
    """One-parent CGS2 + TSQR basis with canonical small-R rank admission.

    Binary stationary populations use separate exact K=-1,0,+1 sectors.
    The zero sector remains in the held basis, even when its current transfer
    is dark. Each normalized block is restored to its exact structural support.
    For a self-inverse q, self_crossing is the authenticated row permutation and
    seeds are completed by their antiunitary partners BEFORE QR. No state is
    truncated to meet capacity: a retained block that does not fit refuses.
    """
    from distrib_la import plan_subspace,plan_orthogonalization
    from common.collectives import device_put_process_local
    if not np.isfinite(rank_tolerance) or not 0 < rank_tolerance < 1:
        raise ValueError('GATE transition_rank: relative probe tolerance must lie in (0,1)')
    validate_transition_catalogue(catalogue,np.asarray(rho))
    mesh=bridge['mesh'];state=bridge['state_sharding'];rep=bridge['port_sharding']
    qr=plan_subspace(capacity=int(capacity),n_eig=1,max_block_size=max_block_size,vector_sharding=state)
    cgs=plan_orthogonalization(capacity=int(capacity),max_block_size=max_block_size,vector_sharding=state)
    workspace=sum(int(np.prod(v.shape))*np.dtype(v.dtype).itemsize for v in qr.workspace_specs.values())
    workspace+=sum(int(np.prod(v.shape))*np.dtype(v.dtype).itemsize for v in cgs.workspace_specs.values())
    price=transition_capacity(catalogue,bridge['receipt']['full_ports'],capacity,
        processes=mesh.size,max_block_size=max_block_size)
    price['provider_workspace_bytes_per_rank']=workspace
    nk,width=catalogue['valid'].shape
    price['streaming_state_bytes_per_rank']=(6*16*nk*width*max_block_size+mesh.size-1)//mesh.size
    price['working_bytes_per_rank']=2*price['basis_bytes_per_rank']+price['vertex_bytes_per_rank']+price['K_scratch_bytes_per_rank']+workspace+price['streaming_state_bytes_per_rank']
    price['external_residency_included']=False
    price['external_residency_note']='Caller must additionally price source psi/W/V, source-psi reshard transients, retained seed arrays and all other parent bases; NOT a material admission'
    if max_device_bytes is not None and price['working_bytes_per_rank']>int(max_device_bytes):
        raise ValueError('GATE transition_capacity: explicit one-parent working bound exceeded')
    basis=jax.jit(lambda:jnp.zeros((capacity,nk,width),complex),out_shardings=state)()
    active=0;records=[]
    diag=np.asarray(rho).diagonal(axis1=-2,axis2=-1)
    stationary=np.array_equal(rho,np.eye(rho.shape[1])[None]*diag[:,:,None]) and np.isin(diag,[0,1]).all()
    b,d,valid,kmq=(catalogue[k] for k in ('b','d','valid','kmq'))
    sectors=np.ones((nk,width),int)
    if stationary:
        sectors=(np.take_along_axis(diag[kmq],d,1)-np.take_along_axis(diag,b,1)).real.astype(int)
    sector_values=(-1,0,1) if stationary else (None,)
    sector_masks=[valid&(sectors==s) for s in sector_values] if stationary else [valid]
    for block in seed_blocks:
        partners=[block]
        if self_crossing is not None:
            partners.append(bridge['cross'](block,jnp.asarray(self_crossing),jnp.asarray(valid)))
        for sector_value,sector in zip(sector_values,sector_masks):
            for candidate in partners:
                for lo in range(0,candidate.shape[0],max_block_size):
                    work=candidate[lo:lo+max_block_size]*jnp.asarray(sector)[None]
                    reference_norm=float(jnp.linalg.norm(work))
                    work=cgs(basis,work,jnp.asarray(active,np.int32))
                    q,r=qr.qr(work)
                    rotate=_canonical_qr_rotation(r,rank_tolerance,reference_norm)
                    keep=rotate.shape[1]
                    if active+keep>capacity:raise ValueError('GATE transition_capacity: retained QR block exceeds capacity')
                    if keep:
                        coeff=device_put_process_local(rotate,rep)
                        # QR may place roundoff outside an exactly known
                        # coordinate sector; restoring these structural zeros
                        # is not a population or small-amplitude cutoff.
                        normalized=jnp.einsum('ar,akp->rkp',coeff,q)*jnp.asarray(sector)[None]
                        basis=basis.at[active:active+keep].set(normalized);active+=keep
                        jax.block_until_ready(basis)
                    records.append(dict(input_rows=int(work.shape[0]),retained=keep,active=active,
                        original_seed_norm=reference_norm,
                        exact_K_sector=sector_value,
                        independence_policy='common.rank_criterion.probe_is_independent',
                        independence_rtol=float(rank_tolerance),
                        spectral_closure='strict whole residual singular blocks; not an overlap inverse'))
    if not active:raise ValueError('GATE transition_rank: no resolved seed state')
    basis=basis[:active]
    gram=bridge['gram'](basis,basis)
    defect=float(jnp.linalg.norm(gram-jnp.eye(active)))
    if defect>1e-9:raise ValueError('GATE transition_orthogonality: unresolved global basis')
    crossing_defect=0.
    if self_crossing is not None:
        squared=0.
        for lo in range(0,active,max_block_size):
            block=basis[lo:lo+max_block_size]
            mirrored=bridge['cross'](block,jnp.asarray(self_crossing),jnp.asarray(valid))
            overlap=bridge['gram'](basis,mirrored)
            projected=jnp.einsum('ij,ikp->jkp',overlap,basis)
            squared+=float(jnp.sum(abs(mirrored-projected)**2))
        crossing_defect=float(np.sqrt(squared/active))
        if crossing_defect>max(1e-9,32*rank_tolerance):
            raise ValueError('GATE transition_crossing: retained basis is not antiunitary-closed')
    return basis,dict(price=price,order=active,orthogonality_defect=defect,
        self_crossing_defect=crossing_defect,
        stationary_binary_sectors=bool(stationary),qr_blocks=records,
        held_scope='one parent; rebuild if catalogue changes; basis carries original transition labels',
        production_default=False)


def transition_parent_model(bridge, t, catalogue, energy, rho, V, basis, gamma, *,
                            max_block_size=32, host_workspace_bytes=4*1024**3):
    """ONE full-port Galerkin parent in the existing complex-pole convention.

    Current D, K and T are projected on the supplied original transition
    basis; NO W interpolation pencil is assembled. Only the small projected
    frequency matrices leave the device. V[mu,nu] stays on native XY faces.
    Returned left/right[1,mu,Kcarrier] remain distributed; exact zero padding
    is distinguished by count. This internal proof neither stores nor admits
    a production bank and does not switch any TD route. At binary stationary
    rho only, exactly coordinate-separated K=0 basis rows may be omitted from
    modal materialization after exact D-crossblock/A0 checks; the full held
    basis and projected generator remain available for the next current rho.
    """
    from common.collectives import gather_to_host,device_put_process_local
    from .shared_pole_execution import face_matmul
    from .shared_pole_constructor import collective_general_pencil,general_pencil_workspace_bytes
    validate_transition_catalogue(catalogue,np.asarray(rho))
    mesh=bridge['mesh'];mm=face_matmul(mesh);rep=bridge['port_sharding']
    face=NamedSharding(mesh,P(None,'x','y'));nmu=t.shape[-1];n=basis.shape[0]
    orthogonality=float(jnp.linalg.norm(bridge['gram'](basis,basis)-jnp.eye(n)))
    if orthogonality>1e-9:raise ValueError('GATE transition_orthogonality: supplied Galerkin basis is not orthonormal')
    if general_pencil_workspace_bytes((n,n))>host_workspace_bytes:
        raise ValueError('GATE transition_capacity: reduced modal workspace exceeds explicit bound')
    carrier=((n+int(mesh.shape['y'])-1)//int(mesh.shape['y']))*int(mesh.shape['y'])
    allocate=jax.jit(lambda:jnp.zeros((1,nmu,carrier),complex),out_shardings=face)
    update=jax.jit(lambda target,block,start:jax.lax.dynamic_update_slice(target,block[None],(0,0,start)),
        in_shardings=(face,rep,rep),out_shardings=face)
    aq,bq=allocate(),allocate()
    b,d,valid,kmq=(jnp.asarray(catalogue[k]) for k in ('b','d','valid','kmq'))
    diagonal=np.asarray(rho).diagonal(axis1=-2,axis2=-1)
    stationary=np.array_equal(rho,np.eye(rho.shape[1])[None]*diagonal[:,:,None]) and np.isin(diagonal,[0,1]).all()
    sector_defect=0.
    for lo in range(0,n,max_block_size):
        block=basis[lo:lo+max_block_size]
        bare=bridge['to_port'](t,block)
        kblock=bridge['K'](block,rho,kmq,b,d,valid)
        weighted=bridge['to_port'](t,kblock)
        if stationary:
            coefficients=bridge['gram'](basis,kblock)
            difference=kblock-jnp.einsum('ij,ikp->jkp',coefficients,basis)
            sector_defect+=float(jnp.sum(abs(difference)**2))
        bq=update(bq,bare,jnp.asarray(lo));aq=update(aq,weighted,jnp.asarray(lo))
        jax.block_until_ready((aq,bq))
    sector_defect=float(np.sqrt(sector_defect/n))
    if stationary and sector_defect>1e-9:
        raise ValueError('GATE transition_sectors: stationary basis does not preserve K-sign sectors')
    dq=bridge['gram'](basis,bridge['D'](basis,energy,kmq,b,d,valid,jnp.asarray(gamma)))
    va=mm(V[None],aq);vb=mm(V[None],bq)
    feedback=mm(bq,va,transa='C')
    # EVERY rank gathers before the canonical pure-host modal owner.
    d_host=np.asarray(gather_to_host(dq))
    m=d_host+np.asarray(gather_to_host(feedback[0,:n,:n]))
    indices=np.arange(n);dark_indices=np.array([],int)
    dark_certificate=False
    if stationary:
        sector=(np.take_along_axis(diagonal[np.asarray(catalogue['kmq'])],catalogue['d'],1)
                -np.take_along_axis(diagonal,catalogue['b'],1)).real
        dark_mask=np.asarray(catalogue['valid'])&(sector==0)
        # Only n booleans leave the state/port devices, never Q or full ports.
        inside=np.asarray(gather_to_host(jnp.any((basis!=0)&jnp.asarray(dark_mask)[None],axis=(1,2))))
        outside=np.asarray(gather_to_host(jnp.any((basis!=0)&jnp.asarray(~dark_mask)[None],axis=(1,2))))
        weighted_live=np.asarray(gather_to_host(jnp.any(aq[0,:,:n]!=0,axis=0)))
        dark_indices=np.flatnonzero(inside&~outside)
        indices=np.flatnonzero(~inside&outside)
        separated=len(dark_indices)+len(indices)==n
        dark_certificate=(separated and not np.any(weighted_live[dark_indices])
            and not np.any(d_host[np.ix_(indices,dark_indices)]!=0)
            and not np.any(d_host[np.ix_(dark_indices,indices)]!=0))
        if not dark_certificate:
            indices=np.arange(n);dark_indices=np.array([],int)
    order=len(indices)
    model_carrier=max(int(mesh.shape['y']),((order+int(mesh.shape['y'])-1)//int(mesh.shape['y']))*int(mesh.shape['y']))
    if order:
        eye=np.eye(order,dtype=complex)
        modal=collective_general_pencil(eye,m[np.ix_(indices,indices)],max_order=order,
            max_workspace_bytes=host_workspace_bytes,fixed_spaces=dict(right=eye,left=eye))
        om=np.zeros((carrier,model_carrier),complex);im=om.copy()
        om[indices,:order]=modal['output_map'];im[indices,:order]=modal['input_map']
        left=mm(va,device_put_process_local(om[None],face))
        right=mm(vb,device_put_process_local(im[None],face))
        poles=np.pad(modal['poles_ry'],(0,model_carrier-order))
        modal_receipt=modal['receipt']
    else:
        left=jax.jit(lambda:jnp.zeros((1,nmu,model_carrier),complex),out_shardings=face)()
        right=left;poles=np.zeros(model_carrier,complex)
        modal_receipt=dict(construction='exact structurally dark zero transfer; no eigensolve')
    return dict(left=left,right=right,poles_ry=poles,count=order,
        projected_generator=m,receipt=dict(construction='full-port-transition-Galerkin-one-parent-proof',
            full_ports=nmu,transition_rows=int(catalogue['valid'].size),live_transitions=int(catalogue['valid'].sum()),
            free_lag_decay_ry=float(gamma),modal=modal_receipt,production_default=False,
            orthogonality_defect=orthogonality,stationary_binary=bool(stationary),K_sector_defect=sector_defect,
            held_order=n,materialized_order=order,materialized_indices=indices.tolist(),
            exact_dark_Kzero_indices=dark_indices.tolist(),exact_dark_certificate=bool(dark_certificate),
            retained_parent_count=1,all_parent_retention_admitted=False))
