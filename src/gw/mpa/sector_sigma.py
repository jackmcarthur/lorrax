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
from runtime.padding import ladder_extent, pad_to_axis, padded_axis
from gw.wavefunction_bundle import parent_sigma_operands


def _native_workspace(mesh_xy, shapes):
    """Query distributed GEMM scratch for the supplied contraction shapes."""
    from distrib_la import plan, workspace_bytes_per_rank
    context=plan('eigh',mesh_xy,n=max(max(a[-2:]+b[-2:]) for a,b in shapes),
                 backend='distributed',batched_route='auto')
    return max(workspace_bytes_per_rank(context,'gemm',(a,b),np.complex128)
               for a,b in shapes)


def _admit(compiled,meta,stage,*,native=0,resident=0,counted=0):
    """Reserve a compiled executable's peak; ``counted`` argument bytes are charged elsewhere."""
    from runtime.aot_memory import aot_kernel_peak_bytes
    peak=aot_kernel_peak_bytes(compiled)
    if not peak.cufft_measured:
        raise ValueError('GATE shared_pole_capacity: sector FFT workspace unavailable')
    meta.shared_pole_capacity.reserve(stage,resident_bytes_per_rank=resident,
        workspace_bytes_per_rank=max(0,peak.total-counted)+native,
        concurrent_with=meta.shared_pole_capacity.live_stages)
    return compiled


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

def _static_key(value):
    """Hashable content key of a small table tree (arrays by bytes digest)."""
    import hashlib
    if isinstance(value, dict):
        return tuple(sorted((k, _static_key(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_static_key(v) for v in value)
    if isinstance(value, (np.ndarray, jax.Array)):
        a = np.asarray(value)
        return ('array', a.shape, a.dtype.str,
                hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
    return value


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
def _w_contraction(mesh_xy, grid, nk, mc, nt_n, kcarrier, layout):
    """W(t) = B_A d(t) B_B^T on the full-q grid; the valence branch reads -q.

    ``(nk, m*nc, n*nt)`` from ``(x, y, omega, interval, ref, time, hole)``
    with ``hole`` static.  One GEMM plan per configuration.
    """
    from distrib_la import gemm_plan
    from symmetry_maps import q_negation_index
    from .sigma import _shared_pole_weights, _shared_pole_contract
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
        weights = _shared_pole_weights(omega, interval, ref, time)
        return _shared_pole_contract(x, y, weights, gemm=gemm, layout=layout)
    return kernel


_SECTOR_TAU = {}


class SectorTau:
    """One sector's τ kernel for the window executable: W(τ) synthesis and Σ(τ) in one body.

    ``window_kernel(space)`` is the traceable ``fn(*arguments, t, active_count)``
    of :meth:`DeviceOmegaAccumulator.integrate_window` (one per branch space,
    since the valence branch reads the q-negated factors); ``window_arguments``
    swaps in the right endpoint's operands and the synthesis's per-window
    operands.  Neither closes over a device buffer, so the accumulator's
    runner cache retains no resident factors.
    """

    def __init__(self, spatial, synthesis, right_yr, right_proj, native, stage, meta, key, plans):
        self._spatial, self._synthesis = spatial, synthesis
        self._right = (right_yr, right_proj)
        self._native, self._stage, self._meta = native, stage, meta
        self._key, self._plans = key, plans
        self._admitted = False

    def window_kernel(self, space):
        """The τ body for ``space``, one function object per static configuration.

        The window runner is cached on this object, so returning the first
        map's body for an equal configuration (same shapes, mesh, layout,
        parent plans and W contraction) lets every later SC map dispatch the
        compiled window executable instead of recompiling it.  The body
        closes over no device buffer.
        """
        hole = space == 'val'
        key = (self._key, self._synthesis.key, hole)
        if key not in _SECTOR_TAU:
            spatial, w_kernel = self._spatial, self._synthesis.w_kernel

            def tau(xn, yr, xr, yn, energies, weight, w_operands, e_ref_a, e_ref_b, t, _active):
                interactions = w_kernel(*w_operands, e_ref_b, t, hole)
                return spatial(xn, yr, xr, yn, energies, weight, e_ref_a, t, interactions)
            # The plans ride along so the ids in the key cannot be reused.
            _SECTOR_TAU[key] = (self._plans, tau)
        return _SECTOR_TAU[key][1]

    def window_arguments(self, xn, xr, energies, weight, e_ref_a, e_ref_b, space, indices, bounds):
        w_operands = self._synthesis.window_operands(space, indices, bounds)
        return (xn, self._right[0], xr, self._right[1], energies, weight, w_operands,
                e_ref_a, e_ref_b)

    def admit(self, compiled, arguments):
        """Reserve the first window executable; the resident factors are the synthesis's stage."""
        if self._admitted:
            return
        counted = sum(int(x.addressable_shards[0].data.nbytes)
                      for x in jax.tree.leaves(self._synthesis.resident_operands()))
        _admit(compiled, self._meta, self._stage, native=self._native, counted=counted)
        self._admitted = True


def sector_tau_factory(left, right, keys, meta, mesh_xy):
    """Bind Gamma_A G_AB(t) Gamma_B to the window executor.

    G[k,mu_X,s,nu_Y,s'] has rectangular centroid endpoints. The at-most-nine
    Lorentz blocks share one transform of the raw-parent Green in the
    four-current door (``gw.cohsex_sigma.make_lorentz_convolution``). Only
    the small projected band operator survives the call.
    """
    from distrib_la import gemm_plan, panel_matmul
    from common.contract_bands import contract_bands_block_reshard
    from gw.greens_function_kernel import build_G_parents, _weighted_tau_phases
    from gw.cohsex_sigma import make_lorentz_convolution

    a, b = left.green_parent, right.green_parent
    plans = a.plan, b.plan
    shapes = tuple((p.n_parent, c.psi_nmu.shape[1], p.n_centroid_packed, p.nspinor)
                   for c, p in zip((a, b), plans))
    q=shapes[0][0];m=shapes[0][2]*shapes[0][3];n=shapes[1][2]*shapes[1][3];k=shapes[0][1]
    # The face Green has a narrow band contraction. Its persistent ψ and G
    # remain two-axis tiled; only a bounded contraction panel is gathered.
    face_green = a.layout == 'face'
    green_panel_bytes = 32 << 20
    native=(0 if face_green else
            _native_workspace(mesh_xy,(((q,m,k),(q,k,n)),)))
    meta.shared_pole_capacity.reserve(f'sigma.sector.tau.warm.{keys[0]}',
        resident_bytes_per_rank=0,
        workspace_bytes_per_rank=(2*16*q*(m*k+k*n+m*n)//mesh_xy.size
                                  +native+(green_panel_bytes if face_green else 0)),
        concurrent_with=meta.shared_pole_capacity.live_stages)
    if face_green:
        gemm = partial(panel_matmul, mesh=mesh_xy,
                       panel_bytes=green_panel_bytes)
    else:
        gemm = gemm_plan(mesh_xy, m=m, k=k, n=n, nq=q,
                         dtype=jnp.complex128, layout=a.layout)
    convolve = make_lorentz_convolution(mesh_xy, meta.kgrid, meta.nk_tot, keys,
                                        plans[0], plans[1])

    def factory(synthesis, band_axis):
        project = contract_bands_block_reshard(mesh_xy, layout=a.layout,
            face_shape=shapes[0], right_face_shape=shapes[1],
            face_band_extent=band_axis.padded)
        _, right_yr, _, right_proj, _, _ = parent_sigma_operands(right)
        right_proj = pad_to_axis(right_proj, band_axis, axis=3)

        def spatial(xn, yr, xr, yn, energies, weight, reference, time, interactions):
            phases = _weighted_tau_phases(energies, 1j*time, e_ref=reference,
                                         band_weight=weight)
            green = build_G_parents(xn, yr, phases=phases, layout=a.layout,
                                    gemm=gemm, k_unfold_plan=plans[0])
            return project(xr, convolve(green, interactions), yn)

        b=band_axis.padded
        projector_shapes=(((q,b,m),(q,m,n)),((q,b,n),(q,n,b)))
        native=_native_workspace(mesh_xy,projector_shapes if face_green
            else (((q,m,k),(q,k,n)),*projector_shapes))
        # Everything spatial() closes over is a function of this key; SC maps
        # keep the parent plans, so their identities are stable.
        key=(mesh_xy,a.layout,shapes,int(b),tuple(keys),tuple(int(v) for v in meta.kgrid),
             int(meta.nk_tot),id(plans[0]),id(plans[1]))
        return SectorTau(spatial, synthesis, right_yr, right_proj,
                         native+synthesis.native, f'sigma.sector.tau.{keys[0]}', meta, key, plans)
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


def sector_synthesis(readers, headers, bases, families, frequencies, meta, mesh_xy):
    """Retain full-q endpoint factors and form one W(t) tile per tau.

    The store and symmetry services are called once at setup.  The factors
    are placed once, with pole columns replicated (axis orientation) whenever
    the capacity ledger admits it, so each tau is a local GEMM; otherwise the
    configured face placement is kept.
    Occupied windows use conj(B_A(-q)) d(t) B_B(-q)^T; d is never conjugated.
    """
    from file_io.shared_pole_store import read_shared_pole_faces
    from .sigma import _shared_pole_factor_specs
    from .sigma_windows import shared_pole_intervals

    left,right=headers
    nc,nt=(int(h.get('factor_components',1)) for h in headers)
    m,n=(b.n_packed for b in bases)
    nq,nk=int(left['n_q_irr']),int(left['n_q_full'])
    kmax=int(left['Kmax'])
    if any(left[k]!=right[k] for k in ('K','Kmax','q_irr_full_idx','identity')):
        raise ValueError('GATE shared_pole_sector_census: endpoint identities differ')
    layout=families[0].layout
    if layout!=families[1].layout:
        raise ValueError('GATE shared_pole_sectors: endpoint wavefunction layouts differ')
    capacity=meta.shared_pole_capacity
    ambient=capacity.live_stages
    tag=f'{left.get("sector")}.{right.get("sector")}'
    shape=(nk,m*nc,n*nt)
    if not kmax:
        zero=_zeros(mesh_xy,shape)
        return _SectorW(lambda _ref,_time,_hole:zero().reshape(nk,m,nc,n,nt),
                        lambda _space,_indices,_bounds:(),lambda:(),lambda _result=None:None,0,
                        ('zero',mesh_xy,shape,nc,nt))
    # The store reader pads physical Kmax for both endpoint face shardings.
    # Keep that carrier through unfolding and GEMM; K and the interval bounds
    # remain physical, so the padded pole columns have identically zero weight.
    kcarrier=padded_axis(ladder_extent(kmax),mesh_xy,name='sector_sigma_K',specs=(
        (P(None,'x',None,'y'),3),(P(None,'y',None,'x'),3))).carrier
    rows=np.arange(nk,dtype=np.int32)
    routes=[];costs=[]
    for h,b,f,axis in zip(headers,bases,families,('x','y')):
        route,cost=_endpoint_route(h,b,f.green_parent.plan.sym,(0,nq),rows,
                                   mesh_xy,axis,kcarrier)
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
    # A face input is required by the established symmetry route. After it
    # completes the factors are placed once for every tau: they do not depend
    # on tau, only d(tau) does. With K replicated (axis orientation) each tau
    # is a local batched GEMM with no collective; the face GEMM's per-q SUMMA
    # re-broadcast the same panels every call (Fe 4^3 bispinor: 158k NCCL
    # broadcasts, 9.9 s of the first sector sweep). Face stays the fallback
    # when the ledger cannot admit the replicated pole columns.
    factor_layout=layout
    if layout=='face' and capacity.preview(
            resident_bytes_per_rank=resident_for('axis')+2*face_bytes,
            workspace_bytes_per_rank=workspace,
            concurrent_with=ambient)['device_budget_status']=='PASS':
        factor_layout='axis'
    factor_spec=_shared_pole_factor_specs(factor_layout)
    resident_bytes=resident_for(factor_layout)
    setup=f'sigma.sector.setup.{tag}'
    resident=f'sigma.sector.resident.{tag}'
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
        # The window runner inlines this contraction and SectorTau.admit
        # reserves the runner's peak plus this GEMM's native workspace; a
        # standalone AOT compile per hole would only repeat that work.
        kernel=_w_contraction(mesh_xy,tuple(left['grid']),nk,m*nc,n*nt,kcarrier,factor_layout)
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
    return _SectorW(w_kernel,window_operands,lambda:(b_x,b_y,poles),close,native,
                    ('w',mesh_xy,tuple(left['grid']),nk,m,nc,n,nt,kcarrier,layout))


class _SectorW:
    """A sector's bound W(tau) synthesis: its kernel, per-window operands and lifetime.

    ``w_kernel(*window_operands(space, indices, bounds), ref, time, hole)`` is
    W(tau) as the four-current door's ``(nk, m, nc, n, nt)`` operand;
    ``resident_operands()`` are the factors the synthesis stage already
    charged; ``native`` is its GEMM's native workspace; ``close`` releases them.
    """
    ordered=True

    def __init__(self,w_kernel,window_operands,resident_operands,close,native,key):
        self.key=key
        self.w_kernel=w_kernel
        self.window_operands=window_operands
        self.resident_operands=resident_operands
        self.close=close
        self.native=int(native)


def instantaneous_sector_sigma(handle, families, bases, meta, mesh_xy, *,
                               occupation_state, return_components=False):
    """Exchange-like equal-time contraction of W_infinity-V, exactly once."""
    from gw.photon_layout import PhotonBasisLayout, photon_block_view, pack_photon_operator
    from gw.photon_sigma import contract_lorentz_blocks, _TERM_X
    from gw.cohsex_sigma import _resolve_Gij
    from gw.qgrid_symmetry import qgrid_trs_policy_from_shared_pole_store
    from file_io.shared_pole_store import read_bank_constant_header, read_bank_constant
    header=read_bank_constant_header(handle,mesh_xy=mesh_xy)
    raw_layout=PhotonBasisLayout.from_centroid_extents(bases[0].n_logical,bases[1].n_logical,mesh_xy)
    layout=PhotonBasisLayout.from_centroid_extents(bases[0].n_packed,bases[1].n_packed,mesh_xy)
    nq=int(header['bank_shape']['nq'])
    amount=16*nq*max(raw_layout.packed_extent,layout.packed_extent)**2//mesh_xy.size
    meta.shared_pole_capacity.reserve('sigma.sector.constant.pack',
        resident_bytes_per_rank=2*amount,workspace_bytes_per_rank=2*amount,
        concurrent_with=meta.shared_pole_capacity.live_stages)
    raw=read_bank_constant(handle,header,meta=meta,mesh_xy=mesh_xy)
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
        # The static face projector contracts O @ psi_right, then
        # psi_left† @ T, both over the padded carrier (k), before the
        # final logical-band slice. Query those actual GEMM shapes.
        native=_native_workspace(mesh_xy,(((q,m,k),(q,k,n)),
            ((q,m,n),(q,n,k)),((q,k,m),(q,m,k))))
        _admit_compiled(kernel,args,meta,f'sigma.sector.constant.{key}',
                        native=native,resident=amount)
    total=None
    currents=[None,None]
    for key,value,_ in contract_lorentz_blocks(keys,families=families,term=_TERM_X,
            response=response,Gij=gij,meta=meta,mesh_xy=mesh_xy,admit_kernel=admit):
        total=value if total is None else total+value
        if return_components and key != (0,0):
            channel=int(key[0] != 0 and key[1] != 0)  # 0: CT+TC, 1: TT
            currents[channel]=(value if currents[channel] is None
                               else currents[channel]+value)
    from gw.photon_sigma import band_sigma_finish
    finish=band_sigma_finish(mesh_xy,int(families[0].slices.nb_sigma),
                             families[0].green_parent.plan.sym)
    if return_components:
        return finish(total), finish(currents[0]), finish(currents[1])
    return finish(total)


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
    # only at each sector's pole extremes, and the cache serves any rule whose
    # certified box contains the request, so TT and CT reuse CC's fits (Fe 4^3
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
                builder=sector_synthesis((reader,other),pair,(bases[a],bases[b]),
                    (families[a],families[b]),freq,meta,mesh_xy)
                bound.append(builder)
                stack.callback(builder.close)
                return builder
            context=dict(schedule=lambda _header:dict(route='sector-panels'),
                synthesis=synthesis,rule_census=rule_census,
                tau_kernel=sector_tau_factory(families[a],families[b],keys,meta,mesh_xy))
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
    for model in resident.values():
        model.release()
    if handle.get('model_stage'):
        ledger=meta.shared_pole_capacity
        ledger.live_stages=tuple(s for s in ledger.live_stages if s!=handle['model_stage'])
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
