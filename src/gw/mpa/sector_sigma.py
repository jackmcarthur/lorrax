"""Ordered photon sectors in the common frequency-quadrature Sigma executor.

A sector has two centroid families and Lorentz components, not a new
frequency-integration algorithm. Factors and rectangular operators retain
both processor axes; one endpoint class is consumed at a time.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.gamma_matrices import gamma_perm_phase
from runtime.padding import pad_to_axis, combined_divisor, round_up
from gw.ppm_tau_kernel import get_shared_sigma_tau_kernel
from gw.wavefunction_bundle import parent_sigma_operands, sigma_face_kernel_kwargs


def _native_workspace(mesh_xy, shapes):
    """Query distributed GEMM scratch for the actual three contraction shapes."""
    from distrib_la import plan, workspace_bytes_per_rank
    context=plan('eigh',mesh_xy,n=max(max(a[-2:]+b[-2:]) for a,b in shapes),
                 backend='distributed',batched_route='auto')
    return max(workspace_bytes_per_rank(context,'gemm',(a,b),np.complex128)
               for a,b in shapes)


def _admit_compiled(kernel,args,meta,stage,*,native=0,resident=0):
    from runtime.aot_memory import aot_kernel_peak_bytes
    compiled=kernel.lower(*args).compile()
    peak=aot_kernel_peak_bytes(compiled)
    if not peak.cufft_measured:
        raise ValueError('GATE shared_pole_capacity: sector FFT workspace unavailable')
    meta.shared_pole_capacity.reserve(stage,resident_bytes_per_rank=resident,
        workspace_bytes_per_rank=peak.total+native,
        concurrent_with=meta.shared_pole_capacity.live_stages)
    return compiled


def sector_tau_factory(left, right, keys, meta, mesh_xy):
    """Bind Gamma_A G_AB(t) Gamma_B to the established tau executor.

    G[k,s,mu_X,s',nu_Y] has rectangular centroid endpoints. Each of the
    at-most-nine Lorentz blocks uses the existing FFT convolution owner.
    Only the small projected band operator survives the call.
    """
    from distrib_la import gemm_plan
    from common.contract_bands import contract_bands_block_reshard
    from gw.greens_function_kernel import build_G, _weighted_tau_phases
    from gw.cohsex_sigma import _make_static_convolution

    a, b = left.green_parent, right.green_parent
    plans = a.plan, b.plan
    shapes = tuple((p.n_parent, c.psi_nmu.shape[1], p.n_centroid_packed, p.nspinor)
                   for c, p in zip((a, b), plans))
    q=shapes[0][0];m=shapes[0][2]*shapes[0][3];n=shapes[1][2]*shapes[1][3];k=shapes[0][1]
    native=_native_workspace(mesh_xy,(((q,m,k),(q,k,n)),))
    meta.shared_pole_capacity.reserve(f'sigma.sector.tau.warm.{keys[0]}',
        resident_bytes_per_rank=0,
        workspace_bytes_per_rank=2*16*q*(m*k+k*n+m*n)//mesh_xy.size+native,
        concurrent_with=meta.shared_pole_capacity.live_stages)
    gemm = gemm_plan(mesh_xy, m=shapes[0][2]*shapes[0][3], k=shapes[0][1],
                    n=shapes[1][2]*shapes[1][3], nq=shapes[0][0],
                    dtype=jnp.complex128, layout=a.layout)
    convolve = _make_static_convolution(mesh_xy, meta.kgrid, meta.nk_tot, lorentz=True)
    vertices = jax.tree.map(lambda *xs: jnp.stack(xs),
        *((gamma_perm_phase(A), gamma_perm_phase(B)) for A, B in keys))
    rows = jnp.asarray(plans[0].parent_full_rows)

    def factory(synthesis, band_axis):
        project = contract_bands_block_reshard(mesh_xy, layout=a.layout,
            face_shape=shapes[0], right_face_shape=shapes[1],
            face_band_extent=band_axis.padded)
        _, right_yr, _, right_proj, _, _ = parent_sigma_operands(right)
        right_proj = pad_to_axis(right_proj, band_axis, axis=3)

        @jax.jit
        def kernel(xn, yr, xr, yn, energies, weight, reference, time, interactions):
            phases = _weighted_tau_phases(energies, 1j*time, e_ref=reference,
                                         band_weight=weight)
            green = build_G(xn, yr, phases=phases, layout=a.layout,
                gemm=gemm, k_unfold_plan=plans[0], right_k_unfold_plan=plans[1])
            sigma = convolve(green, interactions, 1.0, vertices)
            return project(xr, jnp.take(sigma, rows, axis=0), yn)

        admitted=None
        def spatial(*args):
            nonlocal admitted
            args=(args[0],right_yr,args[2],right_proj,*args[4:])
            if admitted is None:
                q=shapes[0][0]; m=shapes[0][2]*shapes[0][3]
                n=shapes[1][2]*shapes[1][3]; k=shapes[0][1]; b=band_axis.padded
                native=_native_workspace(mesh_xy,(((q,m,k),(q,k,n)),
                    ((q,b,m),(q,m,n)),((q,b,n),(q,n,b))))
                admitted=_admit_compiled(kernel,args,meta,
                    f'sigma.sector.tau.{keys[0]}',native=native)
            return admitted(*args)
        return get_shared_sigma_tau_kernel(mesh_xy=mesh_xy, kgrid=meta.kgrid,
            brackets=None, w_synthesis=synthesis, _sigma_kij=spatial,
            cache=False, **sigma_face_kernel_kwargs(left))
    return factory


def _endpoint_route(header, basis, sym, span, rows, mesh_xy, axis, width):
    """Bind the symmetry service's current/charge endpoint action, once per panel."""
    from symmetry_maps import unfold_endpoint_panel, endpoint_panel_cost
    from gw.qgrid_symmetry import shared_pole_packed_action
    proxy = SimpleNamespace(mu_basis=basis)
    perm, wraps, _ = shared_pole_packed_action(proxy, header, mesh_xy=mesh_xy)
    qt = header['qirr']
    lo, hi = span
    parent = np.asarray(qt['irr_idx_q'])[rows]-lo
    operations = np.asarray(qt['sym_idx_q'])[rows]
    nc = int(header.get('factor_components', 1))
    action = (sym.cartesian_action(operations, axial=False, time_odd=True)
              if nc == 3 else np.ones((len(rows),1,1)))
    cost = endpoint_panel_cost((hi-lo,basis.n_packed,nc,width),len(rows),
                              mesh=mesh_xy,mesh_axis=axis,dtype=np.complex128)
    kwargs = dict(irr_idx=parent,sym_idx=operations,
        q_irr_frac=np.asarray(qt['q_irr_frac'])[lo:hi],source_perm=perm,L_table=wraps,
        spin_action_full=action,n_sym_spatial=int(qt['n_sym_spatial']),
        active_mask=basis.active_mask,mesh=mesh_xy,mesh_axis=axis,
        max_live_bytes=cost['estimated_live_bytes_per_rank'])
    return jax.jit(lambda face: unfold_endpoint_panel(face,**kwargs)[0]), cost


def sector_synthesis(readers, headers, bases, families, frequencies, meta, mesh_xy):
    """Read bounded q/K panels and synthesize one full-q Lorentz class.

    Hole windows use conj(b_A(-q)) d(t) b_B(-q)^T, equivalently
    W_BA,+(-q)^T. The scalar phase is never conjugated.
    """
    from distrib_la import gemm_plan
    from file_io.shared_pole_store import read_shared_pole_faces
    from symmetry_maps import q_negation_index
    from .sigma import _shared_pole_weights, _shared_pole_contract
    from .sigma_windows import shared_pole_intervals

    left, right = headers
    nc, nt = (int(h.get('factor_components',1)) for h in headers)
    m, n = (b.n_packed for b in bases)
    nq, nk = int(left['n_q_irr']),int(left['n_q_full'])
    kmax = int(left['Kmax'])
    if any(left[k] != right[k] for k in ('K','Kmax','q_irr_full_idx','identity')):
        raise ValueError('GATE shared_pole_sector_census: endpoint identities differ')
    multiple=combined_divisor(mesh_xy.shape['x'],mesh_xy.shape['y'])
    # Pole panels are bounded by one endpoint dimension, retaining cubic
    # contraction and O(mu^2/P) transient storage even if K grows further.
    width=max(multiple,round_up(min(max(1,kmax),max(m,n)),multiple))
    shape=(nk,m*nc,n*nt)
    sharding=NamedSharding(mesh_xy,P(None,'x','y'))
    zero=jax.jit(lambda:jnp.zeros(shape,jnp.complex128),out_shardings=sharding)
    minus=jnp.asarray(q_negation_index(tuple(left['grid'])))
    kernels=[]
    for parent in range(nq):
        rows=np.flatnonzero(np.asarray(left['qirr']['irr_idx_q'])==parent).astype(np.int32)
        routes=[]; costs=[]
        for h,b,f,axis in zip(headers,bases,families,('x','y')):
            route,cost=_endpoint_route(h,b,f.green_parent.plan.sym,(parent,parent+1),rows,
                                      mesh_xy,axis,width)
            routes.append(route);costs.append(cost)
        amount=16*nk*m*nc*n*nt//mesh_xy.size
        native=_native_workspace(mesh_xy,(((len(rows),m*nc,width),(len(rows),width,n*nt)),))
        meta.shared_pole_capacity.reserve(
            f'sigma.sector.warm.{left.get("sector")}.{right.get("sector")}.{parent}',
            resident_bytes_per_rank=amount,workspace_bytes_per_rank=native+
                2*16*len(rows)*(m*nc*width+width*n*nt+m*nc*n*nt)//mesh_xy.size,
            concurrent_with=meta.shared_pole_capacity.live_stages)
        gemm=gemm_plan(mesh_xy,m=m*nc,n=n*nt,k=width,nq=len(rows),dtype=np.complex128)
        meta.shared_pole_capacity.reserve(f'sigma.sector.panel.{left.get("sector")}.{right.get("sector")}.{parent}',
            resident_bytes_per_rank=amount,
            workspace_bytes_per_rank=(3*16*len(rows)*m*nc*n*nt//mesh_xy.size
                +sum(c['estimated_live_bytes_per_rank'] for c in costs)),
            concurrent_with=meta.shared_pole_capacity.live_stages)
        @partial(jax.jit,static_argnums=(6,))
        def kernel(x,y,p,interval,ref,time,hole,routes=routes,gemm=gemm):
            x,y=routes[0](x),routes[1](y)
            if hole:x,y=jnp.conj(x),jnp.conj(y)
            weights=_shared_pole_weights(p,interval,ref,time)
            return _shared_pole_contract(x,y,jnp.broadcast_to(weights,(x.shape[0],weights.shape[1])),gemm=gemm)
        def abstract(shape,dtype,spec=P()):
            return jax.ShapeDtypeStruct(shape,dtype,sharding=NamedSharding(mesh_xy,spec))
        args=(abstract((1,m,nc,width),np.complex128,P(None,'x',None,'y')),
              abstract((1,n,nt,width),np.complex128,P(None,'y',None,'x')),
              abstract((1,width),np.float64),abstract((1,2),np.int32),
              abstract((),np.float64),abstract((),np.complex128))
        for hole in (False,True):
            _admit_compiled(kernel,(*args,hole),meta,
                f'sigma.sector.compiled.{left.get("sector")}.{right.get("sector")}.{parent}.{hole}',
                native=native,resident=amount)
        kernels.append((rows,kernel))

    @partial(jax.jit,donate_argnums=(0,))
    def add(total,rows,value):return total.at[rows].add(value,indices_are_sorted=True,unique_indices=True)

    def evaluate(space,_omega,indices,bounds,_real,ref,time,_count=None):
        intervals=shared_pole_intervals(frequencies,np.asarray(indices),np.asarray(bounds))
        total=zero()
        for parent,(rows,kernel) in enumerate(kernels):
            for start in range(0,kmax,width):
                stop=min(start+width,kmax)
                selected=np.clip(intervals[parent:parent+1]-start,0,stop-start)
                if not np.any(selected[:,1]>selected[:,0]):continue
                if readers[0] is readers[1] and headers[0] is headers[1]:
                    # A diagonal sector owns one store and needs both
                    # orientations of the same factor. Read it once.
                    left_faces=read_shared_pole_faces(
                        readers[0],(parent,parent+1),meta=meta,header=headers[0],
                        basis=bases[0],column_span=(start,stop))
                    right_faces=left_faces
                else:
                    # An ordered mixed sector consumes only left-X/right-Y.
                    # The store owns face placement and admission; requesting
                    # one orientation avoids an unused collective slab read.
                    left_faces=read_shared_pole_faces(
                        readers[0],(parent,parent+1),meta=meta,header=headers[0],
                        basis=bases[0],column_span=(start,stop),orientations=('x',))
                    right_faces=read_shared_pole_faces(
                        readers[1],(parent,parent+1),meta=meta,header=headers[1],
                        basis=bases[1],column_span=(start,stop),orientations=('y',))
                if left_faces is not right_faces and not bool(
                        jnp.all(left_faces[2]==right_faces[2])):
                    raise ValueError('GATE shared_pole_sector_census: unequal pole values')
                # Reader may pad the final chunk more narrowly than the full
                # panel. Pad through the common padding owner to fixed K.
                x,y=left_faces[0],right_faces[1]
                # Every stored Kmax is mesh-padded; final width gets its own
                # GEMM shape by selecting a fixed full-size padded face below.
                x=jnp.pad(x,((0,0),(0,0),(0,0),(0,width-x.shape[-1])))
                y=jnp.pad(y,((0,0),(0,0),(0,0),(0,width-y.shape[-1])))
                poles=jnp.pad(left_faces[2],((0,0),(0,width-left_faces[2].shape[-1])),constant_values=1)
                value=kernel(x,y,poles,jnp.asarray(selected),ref,time,space=='val')
                total=add(total,jnp.asarray(rows),value)
                total.block_until_ready()
                del left_faces,right_faces,x,y,poles,value
        if space=='val':total=total[minus]
        # mu-major/component-minor -> one bounded stack of Lorentz tiles.
        return jnp.transpose(total.reshape(nk,m,nc,n,nt),(2,4,0,1,3)).reshape(nc*nt,nk,m,n)
    capacity=meta.shared_pole_capacity
    live=f'sigma.sector.live.{left.get("sector")}.{right.get("sector")}'
    capacity.reserve(live,resident_bytes_per_rank=
        16*nk*m*nc*n*nt//mesh_xy.size+32*width*(m*nc+n*nt)//mesh_xy.size,
        workspace_bytes_per_rank=0,concurrent_with=capacity.live_stages)
    def build(*args):
        ambient=capacity.live_stages
        capacity.live_stages=(*ambient,live)
        try:
            return evaluate(*args)
        finally:
            capacity.live_stages=ambient
    build.ordered=True
    return build


def instantaneous_sector_sigma(handle, families, bases, meta, mesh_xy, *, occupation_state):
    """Exchange-like equal-time contraction of W_infinity-V, exactly once."""
    from file_io.slab_io import SlabIO
    from file_io.shared_pole_store import validate_shared_pole_bank, read_shared_pole_bank
    from gw.photon_layout import PhotonBasisLayout, photon_block_view, pack_photon_operator
    from gw.photon_sigma import contract_lorentz_blocks, _TERM_X
    from gw.cohsex_sigma import _resolve_Gij
    from gw.qgrid_symmetry import qgrid_trs_policy_from_shared_pole_store
    from symmetry_maps import unfold_file_wedge_band_operator
    header=validate_shared_pole_bank(handle['path'],expected_identity=handle['identity'],
                                    mesh_xy=mesh_xy,require_complete=True)
    raw_layout=PhotonBasisLayout.from_centroid_extents(bases[0].n_logical,bases[1].n_logical,mesh_xy)
    layout=PhotonBasisLayout.from_centroid_extents(bases[0].n_packed,bases[1].n_packed,mesh_xy)
    nq=int(header['bank_shape']['nq'])
    amount=16*nq*max(raw_layout.packed_extent,layout.packed_extent)**2//mesh_xy.size
    meta.shared_pole_capacity.reserve('sigma.sector.constant.pack',
        resident_bytes_per_rank=2*amount,workspace_bytes_per_rank=2*amount,
        concurrent_with=meta.shared_pole_capacity.live_stages)
    with SlabIO(handle['path'],mode='r',mesh=mesh_xy) as io:
        raw=read_shared_pole_bank(io,(0,nq),meta=meta,header=header,fields=('constant',))['constant']
    def block(A,B):
        value=photon_block_view(raw,raw_layout,A,B,mesh_xy)
        value=bases[bool(A)].pack_axis(value,1,spec=P(None,'x','y'))
        return bases[bool(B)].pack_axis(value,2,spec=P(None,'x','y'))
    packed=pack_photon_operator(block,nq,layout,mesh_xy)
    packed.block_until_ready()
    del raw
    response=SimpleNamespace(V_packed=packed,W_packed=packed,layout=layout,
        family_plans=tuple(f.green_parent.plan for f in families),head_completion=None,
        qgrid_policy=qgrid_trs_policy_from_shared_pole_store(header,announce=False))
    gij=_resolve_Gij(None,meta,mesh_xy,occupation_state)
    keys=tuple((a,b) for a in range(4) for b in range(4))
    def admit(kernel,args,key):
        a,b=map(bool,key)
        left,right=(families[i].green_parent for i in (a,b))
        q=left.plan.n_parent;m=left.plan.n_centroid_packed*left.plan.nspinor
        n=right.plan.n_centroid_packed*right.plan.nspinor;k=left.psi_nmu.shape[1]
        bs=families[a].slices.nb_sigma
        native=_native_workspace(mesh_xy,(((q,m,k),(q,k,n)),
            ((q,bs,m),(q,m,n)),((q,bs,n),(q,n,bs))))
        _admit_compiled(kernel,args,meta,f'sigma.sector.constant.{key}',
                        native=native,resident=amount)
    total=None
    for _,value,_ in contract_lorentz_blocks(keys,families=families,term=_TERM_X,
            response=response,Gij=gij,meta=meta,mesh_xy=mesh_xy,admit_kernel=admit):
        total=value if total is None else total+value
    from gw.cohsex_sigma import _replicate_band_sigma
    nb=families[0].slices.nb_sigma
    @jax.jit
    def finish(value):
        parent=_replicate_band_sigma(value,mesh_xy)[:,:nb,:nb]
        return unfold_file_wedge_band_operator(families[0].green_parent.plan.sym,
                                               parent,trs_rule='transpose')
    return finish(total)


def compute_sector_sigma(handle, families, bases, meta, mesh_xy, **options):
    """Integrate CC, TT and both ordered mixed endpoints on their own pole sets.

    ``options`` is the common MPA/shared-pole quadrature contract; its live
    occupation state and fixed-rule sessions remain owned by the caller.
    The scalar charge entry is unchanged. No model is kept across SC maps.
    """
    from file_io.slab_io import SlabIO
    from file_io.shared_pole_store import validate_shared_pole_sector_manifest
    from .sigma import compute_sigma_c_mpa_omega_grid
    if handle.get('representation')!='sector-ordered-ph':
        raise ValueError('GATE shared_pole_sectors: missing ordered sector handle')
    if families[1] is None or len(bases)!=2:
        raise ValueError('GATE shared_pole_sectors: both endpoint families are required')
    manifest=validate_shared_pole_sector_manifest(handle['path'],
        expected_identity=handle['identity'],mesh_xy=mesh_xy,capacity=meta.shared_pole_capacity)
    for key in ('digest','sectors','constant'):
        if manifest[key]!=handle[key]:
            raise ValueError(f'GATE shared_pole_sector_identity: stale handle {key}')
    sectors=manifest['sectors']
    headers=manifest['model_headers']
    total=None
    for names,endpoints in ((('CC','CC'),(0,0)),(('TT','TT'),(1,1)),
                            (('CT_C','CT_T'),(0,1)),(('CT_T','CT_C'),(1,0))):
        a,b=endpoints
        pair=tuple(headers[n] for n in names)
        keys=tuple((A,B) for A in (range(1,4) if a else (0,))
                   for B in (range(1,4) if b else (0,)))
        with ExitStack() as stack:
            def synthesis(reader, _header, freq, _schedule):
                # All serial metadata authentication precedes collective file
                # opens. Diagonal sectors share the already-open first reader.
                other=(reader if names[0]==names[1] else stack.enter_context(
                    SlabIO(sectors[names[1]]['path'],mode='r',mesh=mesh_xy)))
                return sector_synthesis((reader,other),pair,(bases[a],bases[b]),
                    (families[a],families[b]),freq,meta,mesh_xy)
            context=dict(schedule=lambda _header:dict(route='sector-panels'),
                synthesis=synthesis,
                tau_kernel=sector_tau_factory(families[a],families[b],keys,meta,mesh_xy))
            opts=dict(options)
            sessions=opts.pop('fixed_quadrature_session',None)
            if sessions is not None:opts['fixed_quadrature_session']=sessions.setdefault('_'.join(names),{})
            value=compute_sigma_c_mpa_omega_grid(families[a],sectors[names[0]]['path'],meta,mesh_xy,
                sigma_w_model='shared_pole',fit_identity=sectors[names[0]]['identity'],
                fit_digest=sectors[names[0]]['digest'],sector_context=context,**opts)
            total=value if total is None else replace(total,sigma_c_kij=total.sigma_c_kij+value.sigma_c_kij)
    constant=instantaneous_sector_sigma(handle['constant'],families,bases,meta,mesh_xy,
                                      occupation_state=options.get('occupation_state'))
    # Static band axes use the same carrier as dynamic Sigma; pad only through
    # the existing semantic band-axis owner before broadcasting in omega.
    constant=pad_to_axis(pad_to_axis(constant,total.band_axis,axis=1),total.band_axis,axis=2)
    return replace(total,sigma_c_kij=total.sigma_c_kij+constant[None])
