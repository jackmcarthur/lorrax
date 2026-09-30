"""Ordered photon sectors in the common frequency-quadrature Sigma executor.

A sector has two centroid families and Lorentz components, not a new
frequency-integration algorithm. Its configured factor layout follows the
Green carrier; rectangular interaction operators retain both processor axes.
One endpoint class is consumed at a time.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import replace
from functools import lru_cache, partial
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

from common.collectives import device_put_process_local
from runtime.padding import pad_to_axis, padded_axis
from gw.wavefunction_bundle import parent_sigma_operands
from .sigma import SynthesisTau, WSynthesis, _admit, _static_key


def _native_workspace(mesh_xy, shapes):
    """Query distributed GEMM scratch for the supplied contraction shapes."""
    from distrib_la import plan, workspace_bytes_per_rank
    context=plan('eigh',mesh_xy,n=max(max(a[-2:]+b[-2:]) for a,b in shapes),
                 backend='distributed',batched_route='auto')
    return max(workspace_bytes_per_rank(context,'gemm',(a,b),np.complex128)
               for a,b in shapes)


_ADMIT_COMPILED = {}


def _admit_compiled(kernel,args,meta,stage,*,native=0,resident=0):
    """AOT-compile ``kernel`` at ``args`` (once per signature) and reserve its peak.

    ``lower().compile()`` bypasses jit's executable cache, so an admission
    repeated every SC map would recompile an unchanged program; the
    reservation itself is still taken on every call.
    """
    leaves=jax.tree.leaves(args)
    key=(kernel,jax.tree.structure(args),tuple(
        (tuple(x.shape),str(x.dtype),getattr(x,'sharding',None))
        if hasattr(x,'shape') else x for x in leaves))
    if key not in _ADMIT_COMPILED:
        _ADMIT_COMPILED[key]=kernel.lower(*args).compile()
    return _admit(_ADMIT_COMPILED[key],meta,stage,native=native,resident=resident)


# ---- executables built once per process ------------------------------------
# Every SC map rebuilds the sector models, but not the programs that consume
# them: the builders below are keyed on static configuration (shapes, mesh,
# layout, symmetry tables by content), so a later map dispatches the same jit
# objects and XLA compiles each program once per run.  Data (factors, poles,
# intervals) always enters as an argument, never as a closure constant.

_ENDPOINT_UNFOLD = {}


def _endpoint_unfold(kwargs):
    """``jit(face -> unfold_endpoint_panel(face, **kwargs)[0])``, one per table set."""
    from symmetry_maps import unfold_endpoint_panel
    key = _static_key(kwargs)
    if key not in _ENDPOINT_UNFOLD:
        _ENDPOINT_UNFOLD[key] = jax.jit(
            lambda face: unfold_endpoint_panel(face, **kwargs)[0])
    return _ENDPOINT_UNFOLD[key]


@lru_cache(maxsize=None)
def _placer(mesh_xy, spec):
    """Reshard to ``spec`` (identity values)."""
    return jax.jit(lambda x: x, out_shardings=NamedSharding(mesh_xy, spec))


@lru_cache(maxsize=None)
def _zeros(mesh_xy, shape):
    return jax.jit(lambda: jnp.zeros(shape, jnp.complex128),
                   out_shardings=NamedSharding(mesh_xy, P(None, 'x', 'y')))


@lru_cache(maxsize=None)
def _w_contraction(mesh_xy, grid, nk, mc, nt_n, kcarrier, layout, weights_fn):
    """W(t) = B_A d(t) B_B^T on the full-q grid; the valence branch reads -q.

    ``(nk, m*nc, n*nt)`` from ``(x, y, omega, interval, ref, time, hole)``
    with ``hole`` static.  One GEMM plan per configuration.  ``weights_fn``
    is d: the causal d(t) for Sigma, the omega = 0 coefficient for W0.
    """
    from distrib_la import gemm_plan
    from symmetry_maps import q_negation_index
    from .sigma import _shared_pole_contract
    gemm = gemm_plan(mesh_xy, m=mc, n=nt_n, k=kcarrier, nq=nk,
                     dtype=np.complex128, layout=layout)
    minus = jnp.asarray(q_negation_index(grid))

    @partial(jax.jit, static_argnums=(6,))
    def kernel(x, y, omega, interval, ref, time, hole):
        if hole:
            x = jnp.conj(jnp.take(x, minus, axis=0))
            y = jnp.conj(jnp.take(y, minus, axis=0))
            omega = jnp.take(omega, minus, axis=0)
            interval = jnp.take(interval, minus, axis=0)
        weights = weights_fn(omega, interval, ref, time)
        return _shared_pole_contract(x, y, weights, gemm=gemm, layout=layout)
    return kernel


def sector_tau_factory(left, right, keys, meta, mesh_xy, *, real_weights=False, stage='sigma.sector.tau'):
    """Stream exact Dirac quarters and Lorentz components through mode7.

    Fixed monomial vertices act on projector faces, so each convolution
    carries two spinors and one scalar W_AB, including the mixed sectors.
    The native typed load owns quarter parity and antiunitary partners.
    No full four-spinor Sigma or 3x3 Lorentz interaction is constructed.
    """
    from distrib_la import panel_matmul
    from common.contract_bands import contract_bands_block_reshard
    from common.gamma_matrices import gamma_projector_half
    from common.fft_helpers import make_kconv_klead_unfold, make_kfft_klead
    from gw.greens_function_kernel import (build_G_parents, _weighted_tau_phases,
                                           green_panel_bytes, sigma_row_blocks)
    from gw.cohsex_sigma import lorentz_class_vertices

    a, b = left.green_parent, right.green_parent
    plans = a.plan, b.plan
    if a.layout != 'face' or b.layout != 'face' or any(p.nspinor != 4 for p in plans):
        raise ValueError('GATE sector_sigma_stream: four-spinor all-P face operands required')
    halves = tuple(p.dirac_halves()[0] for p in plans)
    shapes = tuple((p.n_parent, c.psi_nmu.shape[1], p.n_centroid_packed, 2)
                   for c, p in zip((a, b), plans))
    q, k, m, ns = shapes[0]
    n = shapes[1][2]
    if shapes[1][:2] != shapes[0][:2]:
        raise ValueError('GATE sector_sigma_stream: parent/band extents differ')
    lefts, rights = lorentz_class_vertices(keys)
    grid = tuple(int(v) for v in meta.kgrid)
    prep = make_kfft_klead(mesh_xy, grid, P(None, None, None, 'x', 'y'), kind='ifftn', norm='ortho')
    doors = tuple(make_kconv_klead_unfold(mesh_xy, grid,
        plans[0].dirac_quarter_load_tables(0, g, None if plans[1] is plans[0] else plans[1]),
        store_rows=plans[0].parent_full_rows, norm='ortho',
        mult=-1.0/np.sqrt(float(meta.nk_tot))) for g in (0, 1))
    ledger = meta.shared_pole_capacity
    # Both raw-parent quarters and their SUMMA faces; the W stage reserves
    # its one scalar tile independently. The final window AOT owns admission.
    warm = 2*16*q*(2*m*k+k*2*n+4*m*n)//mesh_xy.size
    panel = green_panel_bytes(n_rows=q,m=2*m,n=2*n,mesh=mesh_xy,
        room=ledger.room_bytes_per_rank(ledger.live_stages)-warm)
    ledger.reserve(f'{stage}.warm.{keys[0]}',resident_bytes_per_rank=0,
        workspace_bytes_per_rank=warm+panel,concurrent_with=ledger.live_stages)
    gemm = partial(panel_matmul, mesh=mesh_xy, panel_bytes=panel)
    blocks = sigma_row_blocks(n_rmu=m, ns=2, d=1, mesh=mesh_xy)

    def factory(synthesis, band_axis):
        # The output x blocks keep even the quarter-spin Sigma bounded.
        project = contract_bands_block_reshard(mesh_xy,layout='face',
            face_shape=shapes[0],right_face_shape=shapes[1],
            face_band_extent=band_axis.padded,row_block=blocks[0][1]*blocks[0][3])
        _, right_yr, _, right_proj, _, _ = parent_sigma_operands(right)
        right_proj = pad_to_axis(right_proj, band_axis, axis=3)

        def spatial(xn, yr, xr, yn, energies, weight, reference, time, interactions):
            phases = _weighted_tau_phases(energies,1j*time,e_ref=reference,band_weight=weight)
            if real_weights:phases=jnp.real(phases)
            band_shape=(q,band_axis.padded,band_axis.padded)
            result=jax.lax.with_sharding_constraint(jnp.zeros(band_shape,jnp.complex128),
                NamedSharding(mesh_xy,P(None,'x','y')))

            def quarter(index,total):
                h,g=index//2,index%2
                xhalf=jax.lax.dynamic_slice_in_dim(xn,2*h,2,axis=1)
                yhalf=jax.lax.dynamic_slice_in_dim(yr,2*g,2,axis=2)
                green=build_G_parents(xhalf,yhalf,phases=phases,layout='face',
                                      gemm=gemm,k_unfold_plan=halves[0],real_weights=real_weights)
                # Both doors have the same typed two-spinor action; only the
                # authenticated p^(h+g) endpoint sign differs.
                def component(index,total):
                    ia,ib=index//len(rights),index%len(rights)
                    A=jnp.asarray(lefts)[ia];B=jnp.asarray(rights)[ib]
                    faces=project.prepare(gamma_projector_half(xr,A,h,axis=2),
                        gamma_projector_half(yn,B,g,axis=1))
                    interaction=prep(interactions.component(ia,ib))
                    partial_band=None
                    for rows in blocks:
                        calls=tuple(partial(door,green.G,green.transpose,interaction,
                            conj_partner=green.conj_partner,rows=rows) for door in doors)
                        operator=jax.lax.cond(h==g,calls[0],calls[1])
                        partial_band=project.accumulate(faces,operator,rows=rows,acc=partial_band)
                    return total+project.finish(partial_band)
                return jax.lax.fori_loop(0,len(keys),component,total)
            return jax.lax.fori_loop(0,4,quarter,result)

        native=_native_workspace(mesh_xy,(((q,band_axis.padded,2*m),(q,2*m,2*n)),
            ((q,band_axis.padded,2*n),(q,2*n,band_axis.padded))))
        key=('quarter-stream',mesh_xy,shapes,int(band_axis.padded),tuple(keys),grid,real_weights,
             int(meta.nk_tot),id(plans[0]),id(plans[1]))
        return SynthesisTau(spatial,synthesis,right_yr,right_proj,
            native+synthesis.native,f'{stage}.{keys[0]}',meta,key,plans)
    factory.workspace_bytes=warm+panel
    return factory


def _endpoint_route(header, basis, sym, span, rows, mesh_xy, axis, width):
    """Bind the symmetry service's current/charge endpoint action, once per panel."""
    from symmetry_maps import endpoint_panel_cost
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
    return _endpoint_unfold(kwargs), cost


class _SectorComponents:
    """Trace-time recipe for one W_AB; no Lorentz-sized device buffer."""
    def __init__(self, kernel, operands, ref, time, hole):
        self.kernel,self.operands,self.ref,self.time,self.hole=kernel,operands,ref,time,hole

    def component(self, left, right):
        return self.kernel(*self.operands,self.ref,self.time,self.hole,left,right)


def _sector_component_kernel(panels, mesh_xy, grid, nk, m, n, width, weights_fn, *, column_starts=None):
    """One scalar W_AB, from bounded parent-q and pole-column panels.

    Endpoint routing retains every physical current component: the spatial
    action may mix them. Selection of A/B follows that action, then GEMM.
    Hole rows land at -q, conjugating faces but retaining causal weights.
    """
    from distrib_la import gemm_plan
    from symmetry_maps import q_negation_index
    from .sigma import _shared_pole_contract
    programs=[]
    minus=np.asarray(q_negation_index(grid),np.int32)
    native=0
    for item in panels:
        span,rows,routes=item['span'],item['rows'],item['routes']
        count=len(rows)
        parent=np.asarray(item['parent'],np.int32)
        gemm=gemm_plan(mesh_xy,m=m,n=n,k=width,nq=count,dtype=np.complex128,
                       layout='face',enable_active_range=True,warmup=False)
        native=max(native,_native_workspace(mesh_xy,(((count,m,width),(count,width,n)),)))
        programs.append((span,rows,minus[rows],routes,parent,gemm))
    face=NamedSharding(mesh_xy,P(None,'x','y'))

    @partial(jax.jit,static_argnums=(6,))
    def kernel(x,y,poles,intervals,ref,time,hole,A,B):
        out=jnp.zeros((nk,m,n),jnp.complex128)
        for span,rows,hole_rows,routes,parent,gemm in programs:
            lo,hi=span
            # One panel has all pole chunks; the loop slices one and routes
            # it on all P, never replicating its contraction-column axis.
            def column(chunk,out):
                sx=jax.lax.dynamic_index_in_dim(x,chunk,axis=0,keepdims=False)[lo:hi]
                sy=jax.lax.dynamic_index_in_dim(y,chunk,axis=0,keepdims=False)[lo:hi]
                omega=jax.lax.dynamic_index_in_dim(poles,chunk,axis=0,keepdims=False)[lo:hi]
                offset=chunk*width if column_starts is None else jnp.asarray(column_starts)[chunk]
                low=jnp.maximum(intervals[lo:hi,0],chunk*width)
                bounds=jnp.clip(jnp.stack((low,intervals[lo:hi,1]),axis=1)-offset,0,width)[parent]
                omega=omega[parent]
                bx=routes[0](sx);by=routes[1](sy)
                # Dynamic selection only touches the unsharded component
                # axis; its original face placement remains all-P.
                bx=jax.lax.dynamic_slice_in_dim(bx,A,1,axis=2)
                by=jax.lax.dynamic_slice_in_dim(by,B,1,axis=2)
                if hole:bx=jnp.conj(bx);by=jnp.conj(by)
                weights=weights_fn(omega,bounds,ref,time)
                value=_shared_pole_contract(bx,by,weights,gemm=gemm,layout='face',intervals=bounds)
                return out.at[jnp.asarray(hole_rows if hole else rows)].add(value)
            out=jax.lax.fori_loop(0,x.shape[0],column,out)
            out=jax.lax.optimization_barrier(out)
        return jax.lax.with_sharding_constraint(out,face)
    return kernel,native


def _sector_stream_synthesis(readers,headers,bases,syms,layout,frequencies,meta,mesh_xy,*,spatial_workspace=0):
    """All-P factor residency plus a bounded component synthesis schedule."""
    from file_io.shared_pole_store import read_shared_pole_faces,face_width
    from .sigma import _chunk_major,_shared_pole_weights
    from .sigma_windows import shared_pole_intervals
    if layout!='face':
        raise ValueError('GATE sector_sigma_stream: endpoint factors must use all-P faces')
    left,right=headers
    for key in ('K','Kmax','q_irr_full_idx','identity'):
        if left[key]!=right[key]:
            raise ValueError('GATE shared_pole_sector_census: endpoint identities differ')
    nc,nt=(int(h.get('factor_components',1)) for h in headers)
    m,n=(int(b.n_packed) for b in bases)
    nq,nk,kmax=int(left['n_q_irr']),int(left['n_q_full']),int(left['Kmax'])
    grid=tuple(int(v) for v in left['grid'])
    ledger=meta.shared_pole_capacity;ambient=ledger.live_stages
    tag=f'{left.get("sector")}.{right.get("sector")}'
    if not kmax:
        zero=_zeros(mesh_xy,(nk,m,n))
        return WSynthesis(lambda ref,time,hole:_SectorComponents(
            lambda *_args:zero(),(),ref,time,hole),lambda *_args:(),lambda:(),
            lambda _result=None:None,0,('sector-zero',mesh_xy,nk,m,n,nc,nt),ordered=True)
    carrier=face_width(mesh_xy,kmax)
    resident=16*nq*carrier*(m*nc+n*nt)//mesh_xy.size+8*nq*carrier
    # The physical faces are read at their native two-axis placement and
    # then split into chunk-major carriers. Count both generations at setup.
    setup=f'sigma.sector.setup.{tag}';held=f'sigma.sector.resident.{tag}'
    room=ledger.room_bytes_per_rank(ambient)-resident-int(spatial_workspace)
    tile=16*nk*m*n//mesh_xy.size
    parent_map=np.asarray(left['qirr']['irr_idx_q'],np.int32)
    divisor=max(int(mesh_xy.shape['x']),int(mesh_xy.shape['y']))
    bcap=nq;width=carrier
    while True:
        panels=[];peak=0
        for lo in range(0,nq,bcap):
            hi=min(lo+bcap,nq)
            rows=np.flatnonzero((parent_map>=lo)&(parent_map<hi)).astype(np.int32)
            routes=[];costs=[]
            for header,basis,sym,axis in zip(headers,bases,syms,('x','y')):
                route,cost=_endpoint_route(header,basis,sym,(lo,hi),rows,mesh_xy,axis,width)
                routes.append(route);costs.append(cost)
            peak=max(peak,sum(c['estimated_live_bytes_per_rank'] for c in costs)
                +16*len(rows)*m*n//mesh_xy.size)
            panels.append(dict(span=(lo,hi),rows=rows,routes=tuple(routes),parent=parent_map[rows]-lo))
        # The window owns compiled admission of W, raw-quarter Greens and
        # projection. This search only bounds endpoint/GEMM staging beside
        # its one scalar interaction and the raw-factor residents.
        if 2*tile+peak<=room:break
        if bcap>1:bcap=max(1,bcap//2)
        elif width>divisor:width=max(divisor,((width//2+divisor-1)//divisor)*divisor)
        else:
            raise MemoryError(f'GATE shared_pole_capacity: scalar sector component minimum '
                              f'{2*tile+peak} bytes exceeds room {room}')
    n_chunks=-(-carrier//width)
    # A final rounded chunk may have extra zero columns; read and split
    # capacities include them in the retained residency.
    resident=16*nq*n_chunks*width*(m*nc+n*nt)//mesh_xy.size+8*nq*n_chunks*width
    ledger.reserve(setup,resident_bytes_per_rank=2*resident,
        workspace_bytes_per_rank=0,concurrent_with=ambient)
    ledger.live_stages=(*ambient,setup)
    try:
        # Each pole chunk is read directly at its all-P native face
        # placement. A global K reshape can all-gather its old K shard;
        # no such whole-face staging is allowed here.
        xs=[];ys=[];ps=[];column_starts=[]
        for chunk in range(n_chunks):
            offset=chunk*width
            span=None if n_chunks==1 else (min(offset,kmax-width),min(offset,kmax-width)+width)
            lhs=read_shared_pole_faces(readers[0],(0,nq),meta=meta,header=left,basis=bases[0],
                                      column_span=span,orientations=('x',))
            rhs=read_shared_pole_faces(readers[1],(0,nq),meta=meta,header=right,basis=bases[1],
                                      column_span=span,orientations=('y',))
            if not bool(jnp.all(lhs[2]==rhs[2])):
                raise ValueError('GATE shared_pole_sector_census: unequal pole values')
            xs.append(lhs[0]);ys.append(rhs[1]);ps.append(lhs[2])
            column_starts.append(0 if span is None else span[0])
        x=jnp.stack(xs);y=jnp.stack(ys);poles=jnp.stack(ps)
        jax.block_until_ready((x,y,poles));del lhs,rhs,xs,ys,ps
        ledger.reserve(held,resident_bytes_per_rank=resident,workspace_bytes_per_rank=0,concurrent_with=ambient)
        ledger.live_stages=(*ambient,held)
        kernel,native=_sector_component_kernel(panels,mesh_xy,grid,nk,m,n,width,_shared_pole_weights,
                                                column_starts=tuple(column_starts))
    except BaseException:
        ledger.live_stages=ambient
        raise
    replicated=NamedSharding(mesh_xy,P())
    def window_operands(space,indices,bounds):
        intervals=shared_pole_intervals(frequencies,np.asarray(indices),np.asarray(bounds))
        return x,y,poles,device_put_process_local(np.ascontiguousarray(intervals),replicated)
    closed=False
    def close(result=None):
        nonlocal x,y,poles,closed
        if closed:return
        try:jax.block_until_ready(result if result is not None else (x,y,poles))
        finally:
            x=y=poles=None;ledger.live_stages=ambient;closed=True
    def w_kernel(x,y,poles,intervals,ref,time,hole):
        return _SectorComponents(kernel,(x,y,poles,intervals),ref,time,hole)
    return WSynthesis(w_kernel,window_operands,lambda:(x,y,poles),close,native,
        ('sector-components',mesh_xy,grid,nk,m,n,nc,nt,carrier,width,bcap,
         tuple(column_starts),tuple((item['span'],tuple(map(int,item['rows'])),item['routes']) for item in panels)),ordered=True)


def sector_synthesis(readers, headers, bases, syms, layout, frequencies, meta, mesh_xy,
                     *, weights_fn=None, stage='sigma', spatial_workspace=0):
    """Retain full-q endpoint factors and form one W(t) tile per tau.

    The store and symmetry services are called once at setup.  The factors
    are placed once, with pole columns replicated (axis orientation) whenever
    the capacity ledger admits it, so each tau is a local GEMM; otherwise the
    configured ``layout`` (the endpoint families' Green layout) is kept.
    ``syms`` are the endpoint families' symmetry maps (a current endpoint's
    Cartesian action; a charge endpoint reads none).  ``weights_fn`` is d:
    the causal d(t) by default, the omega = 0 coefficient for
    :func:`sector_static_wc`, whose ledger stages ``stage`` prefixes.
    Occupied windows use conj(B_A(-q)) d(t) B_B(-q)^T; d is never conjugated.
    """
    if stage=='sigma' and weights_fn is None:
        return _sector_stream_synthesis(readers,headers,bases,syms,layout,frequencies,meta,mesh_xy,
                                        spatial_workspace=spatial_workspace)
    from file_io.shared_pole_store import read_shared_pole_faces
    from .sigma import _shared_pole_factor_specs, _shared_pole_weights
    from .sigma_windows import shared_pole_intervals

    left,right=headers
    nc,nt=(int(h.get('factor_components',1)) for h in headers)
    m,n=(b.n_packed for b in bases)
    nq,nk=int(left['n_q_irr']),int(left['n_q_full'])
    kmax=int(left['Kmax'])
    if any(left[k]!=right[k] for k in ('K','Kmax','q_irr_full_idx','identity')):
        raise ValueError('GATE shared_pole_sector_census: endpoint identities differ')
    capacity=meta.shared_pole_capacity
    ambient=capacity.live_stages
    tag=f'{left.get("sector")}.{right.get("sector")}'
    shape=(nk,m*nc,n*nt)
    if not kmax:
        zero=_zeros(mesh_xy,shape)
        return WSynthesis(lambda _ref,_time,_hole:zero().reshape(nk,m,nc,n,nt),
                          lambda _space,_indices,_bounds:(),lambda:(),lambda _result=None:None,0,
                          ('zero',mesh_xy,shape,nc,nt),ordered=True)
    # The store reader pads physical Kmax for both endpoint face shardings.
    # Keep that carrier through unfolding and GEMM; K and the interval bounds
    # remain physical, so the padded pole columns have identically zero weight.
    kcarrier=padded_axis(kmax,mesh_xy,name='sector_sigma_K',specs=(
        (P(None,'x',None,'y'),3),(P(None,'y',None,'x'),3))).carrier
    rows=np.arange(nk,dtype=np.int32)
    routes=[];costs=[]
    for h,b,sym,axis in zip(headers,bases,syms,('x','y')):
        route,cost=_endpoint_route(h,b,sym,(0,nq),rows,mesh_xy,axis,kcarrier)
        routes.append(route);costs.append(cost)
    def place(value,spec):
        return _placer(mesh_xy,spec)(value)
    px,py=int(mesh_xy.shape['x']),int(mesh_xy.shape['y'])
    face_bytes=16*nq*kcarrier*(m*nc+n*nt)//mesh_xy.size

    def resident_for(factor_layout):
        # Each factor has one centroid axis. Pole columns divide over the
        # other mesh axis only in the face orientation.
        split=factor_layout=='face'
        return (16*nk*((m//px)*nc*(kcarrier//py if split else kcarrier)
                       +(n//py)*nt*(kcarrier//px if split else kcarrier))
                +8*nk*kcarrier+16*nk*m*nc*n*nt//mesh_xy.size)
    native=_native_workspace(mesh_xy,(((nk,m*nc,kcarrier),(nk,kcarrier,n*nt)),))
    workspace=sum(c['estimated_live_bytes_per_rank'] for c in costs)+native
    # Sector factors retain both processor axes, including charge W0.
    factor_layout=layout
    factor_spec=_shared_pole_factor_specs(factor_layout)
    resident_bytes=resident_for(factor_layout)
    setup=f'{stage}.sector.setup.{tag}'
    resident=f'{stage}.sector.resident.{tag}'
    capacity.reserve(setup,resident_bytes_per_rank=resident_bytes+2*face_bytes,
        workspace_bytes_per_rank=workspace,concurrent_with=ambient)
    capacity.live_stages=(*ambient,setup)
    try:
        same=readers[0] is readers[1] and headers[0] is headers[1]
        if same:
            lhs=read_shared_pole_faces(readers[0],(0,nq),meta=meta,header=left,basis=bases[0])
            rhs=lhs
        else:
            lhs=read_shared_pole_faces(readers[0],(0,nq),meta=meta,header=left,
                                       basis=bases[0],orientations=('x',))
            rhs=read_shared_pole_faces(readers[1],(0,nq),meta=meta,header=right,
                                       basis=bases[1],orientations=('y',))
            if not bool(jnp.all(lhs[2]==rhs[2])):
                raise ValueError('GATE shared_pole_sector_census: unequal pole values')
        b_x=place(routes[0](lhs[0]),factor_spec[0])
        b_y=place(routes[1](rhs[1]),factor_spec[1])
        parent=np.asarray(left['qirr']['irr_idx_q'],dtype=np.int32)
        poles=jnp.take(lhs[2],parent,axis=0)
        jax.block_until_ready((b_x,b_y,poles))
        del lhs,rhs
    except BaseException:
        capacity.live_stages=ambient
        raise
    try:
        capacity.reserve(resident,resident_bytes_per_rank=resident_bytes,
                         workspace_bytes_per_rank=0,concurrent_with=ambient)
        capacity.live_stages=(*ambient,resident)
    except BaseException:
        b_x=b_y=poles=None
        capacity.live_stages=ambient
        raise
    try:
        # The window runner inlines this contraction and SynthesisTau.admit
        # reserves the runner's peak plus this GEMM's native workspace; a
        # standalone AOT compile per hole would only repeat that work.
        kernel=_w_contraction(mesh_xy,tuple(left['grid']),nk,m*nc,n*nt,kcarrier,factor_layout,
                              weights_fn or _shared_pole_weights)
    except BaseException:
        capacity.live_stages=ambient
        b_x=b_y=poles=None
        raise
    replicated=NamedSharding(mesh_xy,P())
    def w_kernel(x,y,omega,interval,ref,time,hole):
        # (nk, m*nc, n*nt) is centroid-major per endpoint: the four-current
        # door reads it as (nk, m, nc, n, nt) without a transpose.
        return kernel(x,y,omega,interval,ref,time,hole).reshape(nk,m,nc,n,nt)
    def window_operands(space,indices,bounds):
        # Host intervals once per window; every tau node of the window reuses them.
        intervals=shared_pole_intervals(frequencies,np.asarray(indices),np.asarray(bounds))
        return (b_x,b_y,poles,device_put_process_local(
            np.ascontiguousarray(intervals[parent]),replicated))
    closed=False
    def close(result=None):
        nonlocal b_x,b_y,poles,closed
        if closed:return
        try:
            if result is not None:result.block_until_ready()
            else:jax.block_until_ready((b_x,b_y,poles))
        finally:
            b_x=b_y=poles=None
            capacity.live_stages=ambient
            closed=True
    return WSynthesis(w_kernel,window_operands,lambda:(b_x,b_y,poles),close,native,
                      ('w',mesh_xy,tuple(left['grid']),nk,m,nc,n,nt,kcarrier,layout),ordered=True)


_CONSTANT_COMPONENT_CONTRACT = {}


def _constant_component_contract(tau, rows, nk, m, n):
    """Stable immutable program; retains geometry/physics, never wavefunction buffers."""
    key=(tau._key,tuple(map(int,rows)),int(nk),int(m),int(n))
    if key not in _CONSTANT_COMPONENT_CONTRACT:
        spatial=tau._spatial
        mesh=tau._plans[0].mesh_xy
        class ConstantComponents:
            def __init__(self,data):self.data=data
            def component(self,A,B):
                return jax.lax.with_sharding_constraint(
                    jnp.zeros((nk,m,n),jnp.complex128).at[jnp.asarray(rows)].set(self.data[A,B]),
                    NamedSharding(mesh,P(None,'x','y')))
        @jax.jit
        def contract(xn,yr,xr,yn,energy,weight,data):
            return spatial(xn,yr,xr,yn,energy,weight,0.,0.,ConstantComponents(data))
        _CONSTANT_COMPONENT_CONTRACT[key]=(tau._plans,contract)
    return _CONSTANT_COMPONENT_CONTRACT[key][1]


def instantaneous_sector_sigma(handle, families, bases, meta, mesh_xy, *,
                               occupation_state, return_components=False):
    """Equal-time W_infinity-V, from bounded native q panels and Dirac quarters.

    Read each packed parent panel once. Its Lorentz mixing remains coupled;
    restore one endpoint class on its child panel, then stream each scalar
    component through the same exact quarter-spin Sigma owner as dynamic W.
    No whole-bank photon matrix or full-q TT operator is materialized.
    """
    from gw.photon_layout import PhotonBasisLayout,photon_block_view,pack_photon_operator
    from gw.cohsex_sigma import _resolve_Gij,_occ_diag_full
    from gw.qgrid_symmetry import qgrid_trs_policy_from_shared_pole_store
    from gw.w_isdf import photon_blocks_full_q
    from gw.photon_sigma import band_sigma_finish
    from file_io.shared_pole_store import read_bank_constant_header,read_bank_constant
    from gw.ppm_sigma import sigma_band_axis
    header=read_bank_constant_header(handle,mesh_xy=mesh_xy)
    raw_layout=PhotonBasisLayout.from_centroid_extents(bases[0].n_logical,bases[1].n_logical,mesh_xy)
    layout=PhotonBasisLayout.from_centroid_extents(bases[0].n_packed,bases[1].n_packed,mesh_xy,packed=True)
    nq=int(header['bank_shape']['nq']);d=max(raw_layout.packed_extent,layout.packed_extent)
    plans=tuple(f.green_parent.plan for f in families)
    parents=np.asarray(plans[0].sym.irr_idx_q,np.int32)
    policy=qgrid_trs_policy_from_shared_pole_store(header,announce=False)
    ledger=meta.shared_pole_capacity;ambient=ledger.live_stages
    factories={}
    for a,b in ((0,0),(0,1),(1,0),(1,1)):
        keys=tuple((A,B) for A in ((1,2,3) if a else (0,))
                   for B in ((1,2,3) if b else (0,)))
        factories[a,b]=sector_tau_factory(families[a],families[b],keys,meta,mesh_xy,
            real_weights=True,stage='sigma.sector.constant.plan')
    room=ledger.room_bytes_per_rank(ambient)-max(f.workspace_bytes for f in factories.values())
    tile=16*meta.nk_tot*max(b.n_packed for b in bases)**2//mesh_xy.size
    bcap=nq
    while True:
        child=max(np.count_nonzero((parents>=lo)&(parents<min(lo+bcap,nq))) for lo in range(0,nq,bcap))
        # Raw + packed conversions, all coupled current-child components,
        # one full-q scalar W and its FFT form. The compiled contraction
        # admits the actual Green/projector workspace before dispatch.
        amount=16*bcap*d*d//mesh_xy.size
        stage_bytes=4*amount+16*child*9*max(b.n_packed for b in bases)**2//mesh_xy.size
        if stage_bytes+2*tile<=room:break
        if bcap==1:
            raise MemoryError('GATE shared_pole_capacity: constant q-panel minimum exceeds room')
        bcap=max(1,bcap//2)
    gij=_resolve_Gij(None,meta,mesh_xy,occupation_state)
    total=None;currents=[None,None]
    try:
        for lo in range(0,nq,bcap):
            hi=min(lo+bcap,nq);q_span=(lo,hi)
            rows=np.flatnonzero((parents>=lo)&(parents<hi)).astype(np.int32)
            panel_stage=f'sigma.sector.constant.q{lo}'
            ledger.reserve(panel_stage,resident_bytes_per_rank=stage_bytes,
                workspace_bytes_per_rank=0,concurrent_with=ambient)
            ledger.live_stages=(*ambient,panel_stage)
            raw=read_bank_constant(handle,header,meta=meta,mesh_xy=mesh_xy,q_span=q_span)
            def block(A,B):
                value=photon_block_view(raw,raw_layout,A,B,mesh_xy)
                value=bases[bool(A)].pack_axis(value,1,spec=P(None,'x','y'))
                return bases[bool(B)].pack_axis(value,2,spec=P(None,'x','y'))
            packed=pack_photon_operator(block,hi-lo,layout,mesh_xy)
            packed.block_until_ready();del raw
            for a,b in ((0,0),(0,1),(1,0),(1,1)):
                lefts=(1,2,3) if a else (0,);rights=(1,2,3) if b else (0,)
                keys=tuple((A,B) for A in lefts for B in rights)
                values=tuple(value for _,value in photon_blocks_full_q(packed,keys,
                    layout=layout,family_plans=plans,qgrid_policy=policy,q_span=q_span))
                children=jnp.stack(values).reshape(len(lefts),len(rights),len(rows),
                    bases[a].n_packed,bases[b].n_packed)
                children.block_until_ready();del values
                family=families[a]
                weight=plans[a].parent_rows(_occ_diag_full(gij,family.slices.nb_sigma,family.slices.nb_full))
                energy=jnp.zeros_like(weight)
                xn,_,xr,_,_,_=parent_sigma_operands(family)
                axis=sigma_band_axis(int(family.slices.nb_sigma),mesh_xy,ansatz='dynamic')
                synthesis=SimpleNamespace(native=0)
                tau=factories[a,b](synthesis,axis)
                m,n=bases[a].n_packed,bases[b].n_packed
                contract=_constant_component_contract(tau,rows,meta.nk_tot,m,n)
                args=(xn,tau._right[0],pad_to_axis(xr,axis,axis=1),tau._right[1],energy,weight,children)
                _admit_compiled(contract,args,meta,f'sigma.sector.constant.{lo}.{keys[0]}',native=tau._native)
                value=contract(*args)
                total=value if total is None else total+value
                if return_components and (a or b):
                    channel=int(bool(a and b))
                    currents[channel]=value if currents[channel] is None else currents[channel]+value
                total.block_until_ready();del children
            del packed
    finally:ledger.live_stages=ambient
    finish=band_sigma_finish(mesh_xy,int(families[0].slices.nb_sigma),plans[0].sym)
    return (finish(total),finish(currents[0]),finish(currents[1])) if return_components else finish(total)


def compute_sector_sigma(handle, families, bases, meta, mesh_xy, *,
                         on_shell=None, **options):
    """Integrate CC, TT and both ordered mixed endpoints on their own pole sets.

    ``options`` is the common MPA/shared-pole quadrature contract; its live
    occupation state and fixed-rule sessions remain owned by the caller.
    The scalar charge entry is unchanged. No model is kept across SC maps.
    """
    from file_io.shared_pole_store import (ResidentSectorModel, open_shared_pole_model,
                                           validate_shared_pole_sector_manifest)
    from .sigma import compute_sigma_c_mpa_omega_grid
    if handle.get('representation')!='sector-ordered-ph':
        raise ValueError('GATE shared_pole_sectors: missing ordered sector handle')
    if families[1] is None or len(bases)!=2:
        raise ValueError('GATE shared_pole_sectors: both endpoint families are required')
    resident={name:sector['path'] for name,sector in handle['sectors'].items()
              if isinstance(sector['path'],ResidentSectorModel)}
    manifest=validate_shared_pole_sector_manifest(handle['path'],
        expected_identity=handle['identity'],mesh_xy=mesh_xy,capacity=meta.shared_pole_capacity,
        resident=resident)
    for key in ('digest','sectors','constant'):
        if manifest[key]!=handle[key]:
            raise ValueError(f'GATE shared_pole_sector_identity: stale handle {key}')
    sectors=manifest['sectors']
    headers=manifest['model_headers']
    # One Sigma-rule scope for the map's sector calls. Their windows differ
    # only at each sector's pole extremes, and the in-process scope serves any
    # rule whose certified box contains the request, so TT and CT reuse CC's fits (Fe 4^3
    # bispinor: 27 cold fits -> 9 per map). Deterministic: fixed sector order.
    from file_io.shared_pole_store import read_shared_pole_census
    census=[]
    for name in ('CC','TT','CT_C'):
        with open_shared_pole_model(sectors[name]['path'],mesh_xy=mesh_xy) as io:
            poles,counts=read_shared_pole_census(io,header=headers[name],
                                                 capacity=meta.shared_pole_capacity)
        census.append(tuple(np.asarray(a) for a in jax.device_get((poles,counts))))
    rule_census=([np.concatenate([p[q,:int(c[q])] for p,c in census]) for q in range(len(census[0][1]))],
                 [sum(int(c[q]) for _,c in census) for q in range(len(census[0][1]))])
    total=None
    currents=[None,None]
    for names,endpoints in ((('CC','CC'),(0,0)),(('TT','TT'),(1,1)),
                            (('CT_C','CT_T'),(0,1)),(('CT_T','CT_C'),(1,0))):
        a,b=endpoints
        pair=tuple(headers[n] for n in names)
        keys=tuple((A,B) for A in (range(1,4) if a else (0,))
                   for B in (range(1,4) if b else (0,)))
        with ExitStack() as stack:
            bound=[]
            def synthesis(reader, _header, freq, _schedule):
                # All serial metadata authentication precedes collective file
                # opens. Diagonal sectors share the already-open first reader.
                other=(reader if names[0]==names[1] else stack.enter_context(
                    open_shared_pole_model(sectors[names[1]]['path'],mesh_xy=mesh_xy)))
                if families[a].layout!=families[b].layout:
                    raise ValueError('GATE shared_pole_sectors: endpoint wavefunction layouts differ')
                builder=sector_synthesis((reader,other),pair,(bases[a],bases[b]),
                    tuple(f.green_parent.plan.sym for f in (families[a],families[b])),
                    families[a].layout,freq,meta,mesh_xy,
                    spatial_workspace=tau_factory.workspace_bytes)
                bound.append(builder)
                stack.callback(builder.close)
                return builder
            tau_factory=sector_tau_factory(families[a],families[b],keys,meta,mesh_xy)
            context=dict(schedule=lambda _header:dict(route='sector-panels'),
                synthesis=synthesis,rule_census=rule_census,
                tau_kernel=tau_factory)
            opts=dict(options)
            sessions=opts.pop('fixed_quadrature_session',None)
            if sessions is not None:opts['fixed_quadrature_session']=sessions.setdefault('_'.join(names),{})
            value=compute_sigma_c_mpa_omega_grid(families[a],sectors[names[0]]['path'],meta,mesh_xy,
                sigma_w_model='shared_pole',fit_identity=sectors[names[0]]['identity'],
                fit_digest=sectors[names[0]]['digest'],sector_context=context,**opts)
            for builder in bound:builder.close(value.sigma_c_kij)
            if on_shell is not None and (a or b):
                channel=int(bool(a and b))  # 0: CT+TC, 1: TT
                shell=on_shell(value)
                currents[channel]=(shell if currents[channel] is None
                                   else currents[channel]+shell)
            total=value if total is None else replace(total,sigma_c_kij=total.sigma_c_kij+value.sigma_c_kij)
    # Resident models are read once per map: release them and their stage.
    # An SC map holds CC for the accepted final map's W0 persist
    # (sector_static_wc); the next map's entry or the SC end releases it.
    held=resident.pop('CC',None) if handle.get('hold_charge_model') else None
    for model in resident.values():
        model.release()
    if handle.get('model_stage'):
        ledger=meta.shared_pole_capacity
        ledger.live_stages=tuple(s for s in ledger.live_stages if s!=handle['model_stage'])
    if held is not None:
        from gw.shared_pole_screening import hold_resident_model
        cc=headers['CC']
        hold_resident_model(held,meta,ResidentSectorModel.payload_bytes(
            mesh_xy,cc['n_q_irr'],bases[0].n_canonical,cc['Kmax']),
            stage=f"{handle['model_stage']}.CC")
    constant=instantaneous_sector_sigma(handle['constant'],families,bases,meta,mesh_xy,
        occupation_state=options.get('occupation_state'),
        return_components=on_shell is not None)
    if on_shell is not None:
        constant, ct_constant, tt_constant=constant
        for channel, part in enumerate((ct_constant, tt_constant)):
            # QSGW Hermitises the total constant after interpolation.
            hermitian=0.5*(part+jnp.swapaxes(part.conj(),-1,-2))
            currents[channel]=currents[channel]+hermitian
    # Static band axes use the same carrier as dynamic Sigma; pad only through
    # the existing semantic band-axis owner before broadcasting in omega.
    constant=pad_to_axis(pad_to_axis(constant,total.band_axis,axis=1),total.band_axis,axis=2)
    result=replace(total,sigma_c_kij=total.sigma_c_kij+constant[None])
    return (result, tuple(currents)) if on_shell is not None else result


def sector_static_wc(handle, meta, *, mesh_xy):
    """Wc_CC(q, omega = 0) of a four-current sector model, full q grid.

    The charge-sector body of the restart's ``W0_qmunu = V + Wc_CC(0)`` for
    ``bispinor_gw = full_shared_pole``, as :func:`gw.mpa.sigma.shared_pole_static_wc`
    is for a scalar store.  It is the Sigma sector synthesis
    (:func:`sector_synthesis`: one factor read, endpoint unfold, contraction)
    with ``_shared_pole_omega0_weights`` in place of d(t), summing W_+(q)
    and the valence branch from the -q factors as the Sigma consumer reads
    them.  No ``W_inf - V`` term enters: the Ward contact is TT-only and V is
    block diagonal, so the CC block of ``(I + V c)^-1 V - V`` is zero
    (``gw.response_bank``).  CT/TC/TT are not in it: BSE screens with the
    charge sector only, as after the charge route.

    The q = 0 body already carries the direct Gamma head: the bank adds the
    head field W_h - V_h to every sample through the packed zeta(G = 0)
    vectors (``photon_direct_head.add_direct_gamma_field``), so the CC poles
    fit it.  The BSE loader adds the stored whead once as the same rank-one
    field, so ``gw_output.persist_w0_and_head`` stores whead = v_h (the bare
    head) beside this W0; storing the screened head would count W_h - V_h
    twice.

    Returns ``(Q, m, m)`` complex128 at ``P(None,'x','y')`` in the run's
    packed charge-centroid order (``meta.mu_basis``, the CC endpoint basis).
    """
    from file_io.shared_pole_store import open_shared_pole_model, validate_shared_pole_model
    from .sigma import _shared_pole_omega0_weights
    cc=handle['sectors']['CC']
    capacity=meta.shared_pole_capacity
    header=validate_shared_pole_model(cc['path'],expected_identity=cc['identity'],
                                      mesh_xy=mesh_xy,capacity=capacity)
    if header.get('sector')!='CC' or header['digest']!=cc['digest']:
        raise ValueError('GATE shared_pole_identity: W0 handle is not the published CC model')
    Q,m=int(header['n_q_full']),int(meta.mu_basis.n_packed)
    if not int(header['Kmax']):
        return _zeros(mesh_xy,(Q,m,m))()
    counts=np.asarray(header['K'],np.int64)
    parent=np.asarray(header['qirr']['irr_idx_q'],dtype=np.int32)
    intervals=device_put_process_local(np.ascontiguousarray(
        np.stack([np.zeros_like(counts),counts],axis=1)[parent]),NamedSharding(mesh_xy,P()))
    ambient=capacity.live_stages
    tile=-(-16*Q*m*m//int(mesh_xy.size))
    capacity.reserve('w0.static_output',resident_bytes_per_rank=2*tile,
                     workspace_bytes_per_rank=0,concurrent_with=ambient)
    capacity.live_stages=(*ambient,'w0.static_output')
    try:
        with open_shared_pole_model(cc['path'],mesh_xy=mesh_xy) as reader:
            synthesis=sector_synthesis((reader,reader),(header,header),(meta.mu_basis,)*2,
                (None,None),'face',None,meta,mesh_xy,weights_fn=_shared_pole_omega0_weights,
                stage='w0')
            wc=None
            try:
                x,y,poles=synthesis.resident_operands()
                wc=sum(synthesis.w_kernel(x,y,poles,intervals,0.0,0.0,hole).reshape(Q,m,m)
                       for hole in (False,True))
            finally:
                synthesis.close(wc)
    finally:
        capacity.live_stages=ambient
    return wc
