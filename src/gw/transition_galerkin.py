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


def transition_panel_capacity(catalogue, nmu, *, retained_basis_rows=0,
                              processes=1, max_block_size=32,
                              source_resident_bytes_per_rank=None):
    """Price direct physical-Pi panels BEFORE T or Pi allocation.

    The source price must include source psi/redistribution, V/H, existing
    W samples and all other retained parents. The checked Dyson owner's LU
    workspace remains an additional ledger obligation, not a zero here.
    """
    if int(retained_basis_rows)!=retained_basis_rows or retained_basis_rows < 0:
        raise ValueError('GATE transition_capacity: nonnegative retained basis rows required')
    price=transition_capacity(catalogue,nmu,max(1,retained_basis_rows),
        processes=processes,max_block_size=max_block_size)
    nt=int(catalogue['valid'].size);p=int(processes)
    width=min(int(max_block_size),int(nmu))
    q=(16*nt*int(retained_basis_rows)+p-1)//p
    packed=(16*nt*(4*width+2)+p-1)//p
    # Old Pi, updated Pi and the local gathered panel tile can overlap.
    faces=(48*int(nmu)**2+p-1)//p
    ports=32*int(nmu)*width
    source=source_resident_bytes_per_rank
    if source is not None and (not np.isfinite(source) or source<0):
        raise ValueError('GATE transition_capacity: finite nonnegative source residency required')
    known=price['vertex_bytes_per_rank']+q+price['K_scratch_bytes_per_rank']+packed+faces+ports
    return dict(price,retained_basis_rows=int(retained_basis_rows),
        retained_basis_bytes_per_rank=q,streaming_state_bytes_per_rank=packed,
        overlapping_Pi_faces_bytes_per_rank=faces,narrow_port_bytes_per_rank=ports,
        source_resident_bytes_per_rank=None if source is None else int(source),
        known_working_bytes_per_rank=known,
        working_bytes_per_rank=None if source is None else known+int(source),
        checked_Dyson_workspace_included=False,material_admission=False,
        note='One-parent shape price only; source and checked-LU ledger obligations must both be bound')


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
    @partial(jax.jit,in_shardings=(tsh,rep),out_shardings=state,
             static_argnums=(2,))
    def columns(t,start,count):
        panel=jax.lax.dynamic_slice(t,(0,0,start),(nk,width,count))
        return jnp.moveaxis(panel.conj(),2,0)
    port_face=NamedSharding(mesh,P(None,'x','y'))
    allocate_ports=jax.jit(lambda:jnp.zeros((1,nmu,nmu),complex),out_shardings=port_face)
    @partial(jax.shard_map,mesh=mesh,in_specs=(P(None,'x','y'),P(),P()),
             out_specs=P(None,'x','y'),check_vma=False)
    def update_ports_local(target,panel,start):
        nr,nc=target.shape[-2:]
        row_start=jax.lax.axis_index('x')*nr
        rows=jax.lax.dynamic_slice(panel,(row_start,jnp.asarray(0,row_start.dtype)),(nr,panel.shape[1]))
        column=jax.lax.axis_index('y')*nc+jnp.arange(nc)
        offset=column-start
        # The clipped index only makes inactive local loads legal. The exact
        # global interval mask decides every written column, including seams.
        selected=rows[:,jnp.clip(offset,0,panel.shape[1]-1)]
        mask=(offset>=0)&(offset<panel.shape[1])
        return jnp.where(mask[None,None,:],selected[None],target)
    update_ports=jax.jit(update_ports_local,in_shardings=(port_face,rep,rep),out_shardings=port_face)
    allocate_diagonal=jax.jit(lambda:jnp.ones((1,nk,width),complex),out_shardings=state)
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
        columns=columns,allocate_ports=allocate_ports,update_ports=update_ports,
        allocate_diagonal=allocate_diagonal,
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
    denominator=jnp.where(valid[None],z-diagonal,1.)
    dual_denominator=jnp.where(valid[None],complex(z).conjugate()-diagonal.conj(),1.)
    return right/denominator*valid[None],left/dual_denominator*valid[None]


def transition_polarization(bridge, t, catalogue, energy, rho, z, *, gamma,
                            max_block_size=32, retained_basis_rows=0,
                            source_resident_bytes_per_rank=None,
                            max_device_bytes=None):
    """Direct physical Pi=T K_rho (z-D)^-1 T.H on native full-port faces.

    T is the physical-normalized ``[k,p,mu]`` bridge vertex. All current
    arrays are operands. Exact T.H column slices replace T.H@I; 32-wide
    panels resolve the RIGHT-index phases before K. No independent noise,
    extra eta/gamma, spatial projection or pole model enters. Return Pi
    ``[1,mu,mu]`` at P(None,'x','y') and a timing/capacity receipt.
    """
    import time
    validate_transition_catalogue(catalogue,np.asarray(rho))
    if (not np.isfinite(complex(z)) or not np.isfinite(gamma) or gamma<0
            or int(max_block_size)!=max_block_size or max_block_size<1):
        raise ValueError('GATE transition_sample: finite site/width and nonnegative lag gamma required')
    mesh=bridge['mesh'];nmu=t.shape[-1]
    if nmu%int(mesh.shape['x']) or nmu%int(mesh.shape['y']):
        raise ValueError('GATE transition_layout: use the centroid owner packed XY port extent')
    price=transition_panel_capacity(catalogue,nmu,retained_basis_rows=retained_basis_rows,
        processes=mesh.size,max_block_size=max_block_size,
        source_resident_bytes_per_rank=source_resident_bytes_per_rank)
    if max_device_bytes is not None:
        if price['working_bytes_per_rank'] is None:
            raise ValueError('GATE transition_capacity: bind source residency before explicit panel admission')
        if price['working_bytes_per_rank']>int(max_device_bytes):
            raise ValueError('GATE transition_capacity: direct panel working bound exceeded')
    pi=bridge['allocate_ports']()
    ones=bridge['allocate_diagonal']()
    b,d,valid,kmq=(jnp.asarray(catalogue[k]) for k in ('b','d','valid','kmq'))
    diagonal=bridge['D'](ones,energy,kmq,b,d,valid,jnp.asarray(gamma))
    started=time.perf_counter();panels=0
    for start in range(0,nmu,max_block_size):
        count=min(max_block_size,nmu-start)
        source=bridge['columns'](t,jnp.asarray(start),count)
        denominator=jnp.where(valid[None],complex(z)-diagonal,1.)
        resolved=source/denominator*valid[None]
        weighted=bridge['K'](resolved,rho,kmq,b,d,valid)
        panel=bridge['to_port'](t,weighted)
        pi=bridge['update_ports'](pi,panel,jnp.asarray(start))
        jax.block_until_ready(pi)  # bound raw/resolved/K panels across steps
        panels+=1
    return pi,dict(construction='direct-full-port-physical-transition-Pi',
        full_ports=nmu,panels=panels,max_block_size=int(max_block_size),
        site_ry=[float(complex(z).real),float(complex(z).imag)],
        free_lag_decay_ry=float(gamma),extra_electronic_eta_ry=0.,
        physical_prefactor_in_T=True,seconds=time.perf_counter()-started,
        price=price,production_default=False,material_admission=False)


def transition_dyson_value(polarization, root, dyson, *, raw_prefactor):
    """Use the existing checked-LU owner with physical Pi/pref, exactly once.

    Pi[1,mu,mu] is already physically normalized. H=root[mu,mu] is the
    authenticated native Coulomb root used by response_algebra. No fitted
    pole or additional electronic eta enters this direct Dyson sample.
    """
    if not np.isfinite(raw_prefactor) or raw_prefactor<=0:
        raise ValueError('GATE transition_normalization: finite positive raw response prefactor required')
    h=root[None] if root.ndim==2 else root
    return dyson.value(h,polarization/float(raw_prefactor))


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
    update=bridge['update_ports']
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


def transport_transition_basis(bridge, catalogue, basis, U_reference, U_current, *,
                               active_slice, source_resident_bytes_per_rank,
                               max_device_bytes, max_block_size=32):
    """Transport held numerical states: X_new = S_k.H X_ref S_(k-q).

    ``basis[r,k,p]`` is complex128 on ``P(None,('x','y'),None)``; r is
    processed in blocks <=32, never gathered. ``U_reference,U_current`` are
    small unitary ``[k,active,active]`` tables, S=Uref.H Ucurrent, embedded
    as identity on frozen bands. Catalogue b,d label exact physical rows.
    Complete reserved active rows/columns are mandatory: no omitted entry
    is silently discarded after rotation. Only the numerical space moves;
    CURRENT energies, density, T, D and K must still be supplied by callers.

    Source residency (bytes/rank) must include every other live source/T/W
    array. Explicit device budget is checked before any new device array.
    Returns transported states and a literal one-parent workspace receipt;
    this internal helper does not wire a production route or admit all q.
    """
    from common.collectives import device_put_process_local
    mesh=bridge['mesh'];state=bridge['state_sharding'];nk,width=catalogue['valid'].shape
    nb=int(catalogue['receipt']['n_bands'])
    if not isinstance(active_slice,slice) or active_slice.step not in (None,1):
        raise ValueError('GATE transition_transport_indices: contiguous exact active slice required')
    lo,hi=active_slice.start,active_slice.stop
    block=int(max_block_size)
    if (not 0<=lo<hi<=nb or not 1<=block<=32 or block!=max_block_size
            or nk%int(mesh.size) or not isinstance(basis,jax.Array) or basis.ndim!=3
            or basis.shape[1:]!=(nk,width) or basis.shape[0]<1
            or basis.dtype!=jnp.complex128 or not basis.sharding.is_equivalent_to(state,3)):
        raise ValueError('GATE transition_transport: exact full-mesh complex state layout required')
    reserved=catalogue['receipt'].get('coherent_reservation')
    if reserved is None or not reserved[0]<=lo<hi<=reserved[1]:
        raise ValueError('GATE transition_transport_reservation: held catalogue must reserve the active block')
    b,d,valid=(np.asarray(catalogue[key]) for key in ('b','d','valid'))
    kmq=np.asarray(catalogue['kmq'])
    if (b.shape!=(nk,width) or d.shape!=b.shape or valid.dtype!=np.bool_
            or b.dtype.kind not in 'iu' or d.dtype.kind not in 'iu'
            or kmq.dtype.kind not in 'iu' or kmq.shape!=(nk,)
            or np.any(b<0) or np.any(b>=nb) or np.any(d<0) or np.any(d>=nb)
            or not np.array_equal(np.sort(kmq),np.arange(nk))):
        raise ValueError('GATE transition_transport_indices: exact integer catalogue required')
    required=np.zeros((nb,nb),bool);required[lo:hi,:]=True;required[:,lo:hi]=True
    for k in range(nk):
        bb,dd=b[k,valid[k]],d[k,valid[k]]
        if (np.any(bb>=nb) or np.any(dd>=nb) or np.any(bb<0) or np.any(dd<0)
                or len(np.unique(bb*nb+dd))!=len(bb)):
            raise ValueError('GATE transition_transport_indices: invalid or duplicate physical rows')
        present=np.zeros((nb,nb),bool);present[bb,dd]=True
        if not present[required].all():
            raise ValueError('GATE transition_transport_reservation: active coordinate omitted')
    ur,uc=np.asarray(U_reference,complex),np.asarray(U_current,complex)
    if (ur.shape!=(nk,hi-lo,hi-lo) or uc.shape!=ur.shape
            or not np.isfinite(ur).all() or not np.isfinite(uc).all()):
        raise ValueError('GATE transition_transport_unitary: finite active rotations required')
    defect=max(float(np.max(abs(u.conj().swapaxes(-1,-2)@u-np.eye(hi-lo)))) for u in (ur,uc))
    if defect>1e-10:
        raise ValueError('GATE transition_transport_unitary: active rotations are not unitary')
    if (not np.isfinite(source_resident_bytes_per_rank) or source_resident_bytes_per_rank<0
            or not np.isfinite(max_device_bytes) or max_device_bytes<=0):
        raise ValueError('GATE transition_transport_capacity: explicit source residency and device budget required')
    nr=int(basis.shape[0]);block=min(block,nr);p=int(mesh.size)
    qbytes=16*nr*nk*width//p
    # Two held Q arrays plus unpack, left/right intermediates, packed block,
    # small active rotations and integer catalogue (all local whole-k slabs).
    table_bytes=b.dtype.itemsize+d.dtype.itemsize+valid.dtype.itemsize
    workspace=(16*block*nk*(5*nb*nb+2*width)+32*nk*(hi-lo)**2+table_bytes*nk*width+p-1)//p
    working=2*qbytes+workspace+int(source_resident_bytes_per_rank)
    if working>int(max_device_bytes):
        raise ValueError('GATE transition_transport_capacity: explicit full working budget exceeded')
    price=dict(input_and_output_basis_bytes_per_rank=2*qbytes,
        unpack_thin_workspace_bytes_per_rank=workspace,
        source_resident_bytes_per_rank=int(source_resident_bytes_per_rank),
        working_bytes_per_rank=working,device_budget_bytes_per_rank=int(max_device_bytes),
        max_block_size=block,whole_k_sharding=True,material_all_q_admission=False)
    pair_spec=P(('x','y'),None);rotation_spec=P(('x','y'),None,None)
    ss=ur.conj().swapaxes(-1,-2)@uc
    rotations=device_put_process_local(ss,NamedSharding(mesh,rotation_spec))
    partners=device_put_process_local(ss[kmq],NamedSharding(mesh,rotation_spec))
    args=[device_put_process_local(a,NamedSharding(mesh,pair_spec)) for a in (b,d,valid)]

    @partial(jax.shard_map,mesh=mesh,
        in_specs=(P(None,('x','y'),None),rotation_spec,rotation_spec,pair_spec,pair_spec,pair_spec),
        out_specs=P(None,('x','y'),None),check_vma=False)
    def local(q,s,sp,bi,di,mask):
        local_nk=q.shape[1]
        def rotate(x):
            def unpack(row,bb,dd,keep):
                return jnp.zeros((nb,nb),q.dtype).at[bb,dd].add(row*keep)
            full=jax.vmap(jax.vmap(unpack,in_axes=(0,0,0,0)),in_axes=(0,None,None,None))(x,bi,di,mask)
            left=jnp.einsum('kji,rkjb->rkib',s.conj(),full[:,:,lo:hi,:])
            full=full.at[:,:,lo:hi,:].set(left)
            right=jnp.einsum('rkbj,kji->rkbi',full[:,:,:,lo:hi],sp)
            full=full.at[:,:,:,lo:hi].set(right)
            rows=(bi*nb+di)[None]
            return jnp.take_along_axis(full.reshape((x.shape[0],local_nk,nb*nb)),rows,axis=-1)*mask[None]
        def step(out,index):
            begin=index*block
            chunk=jax.lax.dynamic_slice(q,(begin,0,0),(block,local_nk,width))
            return jax.lax.dynamic_update_slice(out,rotate(chunk),(begin,0,0)),None
        result=jax.lax.scan(step,jnp.zeros_like(q),jnp.arange(nr//block))[0]
        tail=nr%block
        if tail:
            result=result.at[-tail:].set(rotate(q[-tail:]))
        return result
    # Scalar reductions only; no Q/transition/mu face is staged on a host.
    check=jax.jit(lambda q,m:jnp.all(jnp.isfinite(q))&jnp.all(jnp.where(m[None],True,q==0)))
    if not bool(check(basis,args[-1])):
        raise ValueError('GATE transition_transport: nonfinite states or live padded coordinates')
    moved=jax.jit(local,out_shardings=state)(basis,rotations,partners,*args)
    jax.block_until_ready(moved)
    return moved,dict(price=price,unitarity_defect=defect,
        transport='Xnew=S_k^dagger Xref S_kminusq',active_bands=[lo,hi],
        numerical_space_only=True,current_energies_held=False,production_wired=False)
