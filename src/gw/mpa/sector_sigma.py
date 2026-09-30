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
def _zeros(mesh_xy, shape, spec=P(None, 'x', 'y')):
    return jax.jit(lambda: jnp.zeros(shape, jnp.complex128),
                   out_shardings=NamedSharding(mesh_xy, spec))


@lru_cache(maxsize=None)
def _other_face(mesh_xy, spec, other):
    """A factor's face in the other orientation: rank (i, j)'s tile to rank (j, i).

    One collective permute of one tile (``common.collectives.to_transpose_partner``);
    a diagonal sector's right factor is its left factor, so only one
    orientation is kept resident and this forms the other per Lorentz panel.
    """
    from common.collectives import to_transpose_partner
    from common.shard_map import shard_map
    p = int(mesh_xy.shape['x'])
    return shard_map(lambda t: to_transpose_partner(t, p), mesh=mesh_xy,
                     in_specs=spec, out_specs=other, check_vma=False)


@lru_cache(maxsize=None)
def _w_contraction(mesh_xy, grid, nk, mc, nt_n, kcarrier, layout, weights_fn, rows=None):
    """W(t) = B_A d(t) B_B^T on the full-q grid; the valence branch reads -q.

    ``(nk, m*nc, n*nt)`` from ``(x, y, omega, interval, ref, time, hole)``
    with ``hole`` static.  One GEMM plan per configuration.  ``weights_fn``
    is d: the causal d(t) for Sigma, the omega = 0 coefficient for W0.
    ``rows`` plans the GEMM for a q panel of that many rows (``hole`` False:
    the caller has read the panel's rows).
    """
    from distrib_la import gemm_plan
    from symmetry_maps import q_negation_index
    from .sigma import _shared_pole_contract
    gemm = gemm_plan(mesh_xy, m=mc, n=nt_n, k=kcarrier, nq=rows or nk,
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


#: Test hook: the largest Lorentz-block panel the synthesis may choose
#: (``None``: the ledger decides).  A forced small panel checks the streamed
#: sum against the one-panel sum on a deck where one panel fits.
_TEST_MAX_PANEL_BLOCKS = None
#: Test hook: the largest parent-q panel the instantaneous constant is read
#: and packed in (``None``: the ledger decides).
_TEST_CONSTANT_Q_SPAN = None
#: Test hook: build the antiunitary partner by its GEMM even when the ledger
#: holds lax.cond's copy of it.
_TEST_DIRECT_PARTNER = False
#: Test hook: form each W(t) panel in q panels of at most this many rows.
_TEST_W_Q_ROWS = None


class LorentzPanels:
    """W(t) of one endpoint class, formed one Lorentz-block panel at a time.

    ``panels`` are ``((a0, a1), (b0, b1))`` component ranges of the class's
    ``A x B`` blocks; ``block(i)`` is panel ``i`` as the four-current door's
    ``(nk, m, |A|, n, |B|)`` operand.  Nothing is formed until a consumer asks,
    so one panel's tile is live at a time.  ``operands`` may be passed back
    through an optimization barrier to order the panels.
    """

    def __init__(self, panels, form, operands):
        self.panels, self._form, self.operands = panels, form, operands

    def block(self, index, operands=None):
        return self._form(self.panels[index], *(self.operands if operands is None else operands))


def sector_tau_factory(left, right, keys, meta, mesh_xy):
    """Bind Gamma_A G_AB(t) Gamma_B to the window executor.

    G[k,mu_X,s,nu_Y,s'] has rectangular centroid endpoints. The at-most-nine
    Lorentz blocks share one transform of the raw-parent Green in the
    four-current door (``gw.cohsex_sigma.make_lorentz_convolution``). Only
    the small projected band operator survives the call. When the synthesis
    forms W(t) in Lorentz-block panels (its class tile does not fit), each
    panel is one call of its sub-product's door and the projected Sigma
    sums over panels; a class that fits is one call, as before.
    """
    from distrib_la import gemm_plan, panel_matmul
    from common.contract_bands import contract_bands_block_reshard
    from gw.greens_function_kernel import (build_G_parents, _weighted_tau_phases,
                                           green_panel_bytes)
    from gw.cohsex_sigma import make_lorentz_convolution, lorentz_class_vertices

    a, b = left.green_parent, right.green_parent
    plans = a.plan, b.plan
    shapes = tuple((p.n_parent, c.psi_nmu.shape[1], p.n_centroid_packed, p.nspinor)
                   for c, p in zip((a, b), plans))
    q=shapes[0][0];m=shapes[0][2]*shapes[0][3];n=shapes[1][2]*shapes[1][3];k=shapes[0][1]
    # The face Green has a narrow band contraction. Its persistent ψ and G
    # remain two-axis tiled; only a bounded contraction panel is gathered.
    face_green = a.layout == 'face'
    native=(0 if face_green else
            _native_workspace(mesh_xy,(((q,m,k),(q,k,n)),)))
    ledger = meta.shared_pole_capacity
    warm = 2*16*q*(m*k+k*n+m*n)//mesh_xy.size + native
    # The face Green's SUMMA band panels (distrib_la.panel_matmul, two live, at
    # most N_b/p_x bands each) are bounded by one parent Green tile when the
    # ledger's room beside the warm workspace holds it, else by what the room
    # holds (green_panel_bytes).
    room = ledger.room_bytes_per_rank(ledger.live_stages) - warm
    panel = (green_panel_bytes(n_rows=q, m=m, n=n, mesh=mesh_xy, room=room)
             if face_green else 0)
    # The antiunitary partner behind build_G_parents' device predicate
    # (lax.cond: conj(G) at real weights, else its own GEMM) is held twice at
    # the cond's exit, branch value and output.  When the room beside the warm
    # workspace cannot hold that copy and the door's Sigma_k (two parent tiles,
    # Fe 20^3/P36 24.5 GB/rank each), the partner is built by its GEMM with no
    # predicate: the same values (conj of a product is exact), one tile.
    tile = 16*q*m*n//mesh_xy.size
    direct_partner = bool(_TEST_DIRECT_PARTNER) or room - panel < 2*tile
    ledger.reserve(f'sigma.sector.tau.warm.{keys[0]}',
        resident_bytes_per_rank=0, workspace_bytes_per_rank=warm + panel,
        concurrent_with=ledger.live_stages)
    if face_green:
        gemm = partial(panel_matmul, mesh=mesh_xy, panel_bytes=panel)
    else:
        gemm = gemm_plan(mesh_xy, m=m, k=k, n=n, nq=q,
                         dtype=jnp.complex128, layout=a.layout)
    convolve = make_lorentz_convolution(mesh_xy, meta.kgrid, meta.nk_tot, keys,
                                        plans[0], plans[1])
    lefts, rights = lorentz_class_vertices(keys)

    def factory(synthesis, band_axis):
        project = contract_bands_block_reshard(mesh_xy, layout=a.layout,
            face_shape=shapes[0], right_face_shape=shapes[1],
            face_band_extent=band_axis.padded)
        _, right_yr, _, right_proj, _, _ = parent_sigma_operands(right)
        right_proj = pad_to_axis(right_proj, band_axis, axis=3)
        # One mode-8 door per Lorentz-block panel of the synthesis; a class
        # that fits is one panel, the class's own door, as before.
        panels = synthesis.panels
        doors = ((convolve,) if len(panels) == 1 else tuple(
            make_lorentz_convolution(mesh_xy, meta.kgrid, meta.nk_tot,
                tuple((lefts[i], rights[j]) for i in range(*ia) for j in range(*ib)),
                plans[0], plans[1]) for ia, ib in panels))

        def spatial(xn, yr, xr, yn, energies, weight, reference, time, interactions):
            whole = interactions.block(0) if len(doors) == 1 else None
            phases = _weighted_tau_phases(energies, 1j*time, e_ref=reference,
                                         band_weight=weight)
            green = build_G_parents(xn, yr, phases=phases, layout=a.layout,
                                    gemm=gemm, k_unfold_plan=plans[0],
                                    real_weights=False if direct_partner else None)
            if whole is not None:
                return project(xr, doors[0](green, whole), yn)
            # Panels in sequence: the barrier makes panel i+1's W(t) wait for
            # panel i's projected Sigma, so one panel tile is live at a time.
            total, operands = None, interactions.operands
            for index, door in enumerate(doors):
                value = project(xr, door(green, interactions.block(index, operands)), yn)
                total = value if total is None else total + value
                if index + 1 < len(doors):
                    total, operands = jax.lax.optimization_barrier((total, operands))
            return total

        b=band_axis.padded
        projector_shapes=(((q,b,m),(q,m,n)),((q,b,n),(q,n,b)))
        native=_native_workspace(mesh_xy,projector_shapes if face_green
            else (((q,m,k),(q,k,n)),*projector_shapes))
        # Everything spatial() closes over is a function of this key; SC maps
        # keep the parent plans, so their identities are stable.
        key=(mesh_xy,a.layout,shapes,int(b),tuple(keys),tuple(int(v) for v in meta.kgrid),
             int(meta.nk_tot),id(plans[0]),id(plans[1]),panels,direct_partner)
        return SynthesisTau(spatial, synthesis, right_yr, right_proj,
                         native+synthesis.native, f'sigma.sector.tau.{keys[0]}', meta, key, plans)
    return factory


def _endpoint_route(header, basis, sym, span, rows, mesh_xy, axis, width):
    """Bind the symmetry service's current/charge endpoint action, once per panel.

    Returns ``(route, cost, kwargs)``: the bound unfold of every child row in
    ``rows``, its analytical cost, and the tables it binds (a child-q panel
    of the full grid reads their rows through the service's eager kernel).
    """
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
    return _endpoint_unfold(kwargs), cost, kwargs


#: Test hook: the largest child-q panel the full-grid factor unfold may take
#: (``None``: the ledger decides; every child in one call when it fits).
_TEST_UNFOLD_SPAN = None
#: Test hook: a diagonal sector keeps one factor face even when both fit.
_TEST_MIRROR = False


def _unfold_full_grid(face, route, kwargs, span, mesh_xy, axis):
    """The full-q factor from its parent face, ``span`` child rows per call.

    ``span`` covering the grid is the bound route, one call.  Otherwise each
    child-q panel is unfolded by the symmetry service's eager kernel (its
    tables as operands, so equal panels share one executable) and written in
    place into the full-q face, so the transients are one panel's.
    """
    from symmetry_maps import unfold_endpoint_panel, endpoint_panel_cost
    nk=len(kwargs['irr_idx'])
    if span>=nk:
        return route(face)
    spec=P(None,axis,None,'y' if axis=='x' else 'x')
    if not face.sharding.is_equivalent_to(NamedSharding(mesh_xy,spec),face.ndim):
        face=_placer(mesh_xy,spec)(face)
    out=_zeros(mesh_xy,(nk,*face.shape[1:]),spec)()
    write=_set_q_rows(mesh_xy,spec)
    for lo in range(0,nk,span):
        hi=min(lo+span,nk)
        part,_=unfold_endpoint_panel(face,**{**kwargs,
            'irr_idx':kwargs['irr_idx'][lo:hi],'sym_idx':kwargs['sym_idx'][lo:hi],
            'spin_action_full':kwargs['spin_action_full'][lo:hi],
            'max_live_bytes':endpoint_panel_cost(face.shape,hi-lo,mesh=mesh_xy,mesh_axis=axis,
                dtype=face.dtype)['estimated_live_bytes_per_rank']})
        out=write(out,part,lo)
        del part
    return out


_PANELS_HELD = {}


def _blocks(panel):
    (a0,a1),(b0,b1)=panel
    return (a1-a0)*(b1-b0)


def sector_synthesis(readers, headers, bases, syms, layout, frequencies, meta, mesh_xy,
                     *, weights_fn=None, stage='sigma'):
    """Retain full-q endpoint factors and form W(t) per tau, one Lorentz panel at a time.

    The store and symmetry services are called once at setup.  The factors
    are placed once, with pole columns replicated (axis orientation) whenever
    the capacity ledger admits it, so each tau is a local GEMM; otherwise the
    configured ``layout`` (the endpoint families' Green layout) is kept.
    ``syms`` are the endpoint families' symmetry maps (a current endpoint's
    Cartesian action; a charge endpoint reads none).  ``weights_fn`` is d:
    the causal d(t) by default, the omega = 0 coefficient for
    :func:`sector_static_wc`, whose ledger stages ``stage`` prefixes.
    Occupied windows use conj(B_A(-q)) d(t) B_B(-q)^T; d is never conjugated.
    ``w_kernel`` returns :class:`LorentzPanels`; ``panels`` is the schedule
    the ledger admitted (one panel when the class tile fits).
    """
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
    whole=(((0,nc),(0,nt)),)
    if not kmax:
        zero=_zeros(mesh_xy,shape)
        synthesis=WSynthesis(lambda _ref,_time,_hole:LorentzPanels(
                              whole,lambda _panel:zero().reshape(nk,m,nc,n,nt),()),
                          lambda _space,_indices,_bounds:(),lambda:(),lambda _result=None:None,0,
                          ('zero',mesh_xy,shape,nc,nt),ordered=True)
        synthesis.panels=whole
        return synthesis
    # The store reader pads physical Kmax for both endpoint face shardings.
    # Keep that carrier through unfolding and GEMM; K and the interval bounds
    # remain physical, so the padded pole columns have identically zero weight.
    kcarrier=padded_axis(kmax,mesh_xy,name='sector_sigma_K',specs=(
        (P(None,'x',None,'y'),3),(P(None,'y',None,'x'),3))).carrier
    # A diagonal sector (CC, TT, the W0 charge body) contracts one factor
    # with itself: B_B is B_A on the other face.  When both faces do not fit,
    # only B_A is read, unfolded and kept, and each Lorentz panel forms its
    # B_B columns by one tile exchange per tau (_other_face).
    same=readers[0] is readers[1] and headers[0] is headers[1]
    diagonal=same and bases[0] is bases[1] and syms[0] is syms[1]
    rows=np.arange(nk,dtype=np.int32)
    routes=[];tables=[]
    for h,b,sym,axis in zip(headers,bases,syms,('x','y')):
        route,_,kwargs=_endpoint_route(h,b,sym,(0,nq),rows,mesh_xy,axis,kcarrier)
        routes.append(route);tables.append(kwargs)
    def place(value,spec):
        return _placer(mesh_xy,spec)(value)
    px,py=int(mesh_xy.shape['x']),int(mesh_xy.shape['y'])
    def face_bytes(mirror):
        return 16*nq*kcarrier*(m*nc+(0 if mirror else n*nt))//mesh_xy.size

    def resident_for(factor_layout,panel,mirror):
        # Each factor has one centroid axis. Pole columns divide over the
        # other mesh axis only in the face orientation. One W(t) panel of
        # |A| x |B| Lorentz blocks is live at a time; a mirrored right
        # factor is the panel's |B| columns of the left one, moved.
        split=factor_layout=='face'
        right_components=panel[1][1]-panel[1][0] if mirror else nt
        return (16*nk*((m//px)*nc*(kcarrier//py if split else kcarrier)
                       +(n//py)*right_components*(kcarrier//px if split else kcarrier))
                +8*nk*kcarrier+16*nk*m*n*_blocks(panel)//mesh_xy.size)
    native=_native_workspace(mesh_xy,(((nk,m*nc,kcarrier),(nk,kcarrier,n*nt)),))

    def unfold_workspace(span,mirror):
        # The full-grid unfold's transients for ``span`` child rows per call.
        from symmetry_maps import endpoint_panel_cost
        return sum(endpoint_panel_cost((nq,extent,comps,kcarrier),span,mesh=mesh_xy,
                   mesh_axis=axis,dtype=np.complex128)['estimated_live_bytes_per_rank']
                   for extent,comps,axis in ((m,nc,'x'),(n,nt,'y'))[:1 if mirror else 2])+native
    spans=[nk]
    while spans[-1]>1:spans.append(-(-spans[-1]//2))
    if _TEST_UNFOLD_SPAN is not None:
        spans=[s for s in spans if s<=_TEST_UNFOLD_SPAN] or spans[-1:]
    mirrors=(False,True) if diagonal else (False,)
    if diagonal and _TEST_MIRROR:
        mirrors=(True,)
    # A face input is required by the established symmetry route. After it
    # completes the factors are placed once for every tau: they do not depend
    # on tau, only d(tau) does. With K replicated (axis orientation) each tau
    # is a local batched GEMM with no collective; the face GEMM's per-q SUMMA
    # re-broadcast the same panels every call (Fe 4^3 bispinor: 158k NCCL
    # broadcasts, 9.9 s of the first sector sweep). Face stays the fallback
    # when the ledger cannot admit the replicated pole columns.
    # The full-q W(t) of a current class is nc*nt Lorentz blocks (TT: 9 at
    # 16 nk m n bytes each; Fe 20^3 at P36 is 103.7 GB/rank). When the class
    # tile does not fit, W(t) is formed in A x B panels (one component row,
    # then one block), each consumed by its own mode-8 door call. The first
    # schedule that fits wins, so a deck that fits keeps one panel.  Within a
    # schedule, a diagonal sector keeps both faces when they fit, else one
    # (Fe 20^3/P36 TT: 92.2 -> 46.1 GB/rank of factors).  The full-q factor
    # is unfolded from its parent face at setup: every child in one call
    # when its transients fit, else in child-q panels written in place
    # (Fe 20^3/P36 TT: 276 GB/rank of transients in one call).
    ladder=[whole]
    if nc>1:ladder.append(tuple(((i,i+1),(0,nt)) for i in range(nc)))
    if nt>1:ladder.append(tuple(((i,i+1),(j,j+1)) for i in range(nc) for j in range(nt)))
    if _TEST_MAX_PANEL_BLOCKS is not None:
        ladder=[p for p in ladder if _blocks(p[0])<=_TEST_MAX_PANEL_BLOCKS] or ladder[-1:]
    # An SC map first tries the previous map's streamed choice, so the
    # window executables keep their shapes while it still fits.
    held=_PANELS_HELD.get((tag,nk,m,n,nc,nt,layout))
    choice=None
    for schedule in ([held[1]] if held else [])+ladder:
        for candidate in ((held[0],) if held and schedule is held[1] else
                          ('axis','face') if layout=='face' else (layout,)):
            for mirror in ((held[2],) if held and schedule is held[1] else mirrors):
                for span in spans:
                    if capacity.preview(
                            resident_bytes_per_rank=resident_for(candidate,schedule[0],mirror)
                            +2*face_bytes(mirror),
                            workspace_bytes_per_rank=unfold_workspace(span,mirror),
                            concurrent_with=ambient)['device_budget_status']=='PASS':
                        choice=candidate,schedule,mirror,span
                        break
                if choice is not None:break
            if choice is not None:break
        if choice is not None:break
    factor_layout,panels,mirror,span=choice or (layout,ladder[-1],mirrors[-1],spans[-1])
    if weights_fn is None and (len(panels)>1 or mirror):
        _PANELS_HELD[(tag,nk,m,n,nc,nt,layout)]=factor_layout,panels,mirror
    factor_spec=_shared_pole_factor_specs(factor_layout)
    resident_bytes=resident_for(factor_layout,panels[0],mirror)
    setup=f'{stage}.sector.setup.{tag}'
    resident=f'{stage}.sector.resident.{tag}'
    capacity.reserve(setup,resident_bytes_per_rank=resident_bytes+2*face_bytes(mirror),
        workspace_bytes_per_rank=unfold_workspace(span,mirror),concurrent_with=ambient)
    capacity.live_stages=(*ambient,setup)
    try:
        if same:
            lhs=read_shared_pole_faces(readers[0],(0,nq),meta=meta,header=left,basis=bases[0],
                                       orientations=('x',) if mirror else ('x','y'))
            rhs=lhs
        else:
            lhs=read_shared_pole_faces(readers[0],(0,nq),meta=meta,header=left,
                                       basis=bases[0],orientations=('x',))
            rhs=read_shared_pole_faces(readers[1],(0,nq),meta=meta,header=right,
                                       basis=bases[1],orientations=('y',))
            if not bool(jnp.all(lhs[2]==rhs[2])):
                raise ValueError('GATE shared_pole_sector_census: unequal pole values')
        b_x=place(_unfold_full_grid(lhs[0],routes[0],tables[0],span,mesh_xy,'x'),factor_spec[0])
        b_y=None if mirror else place(_unfold_full_grid(
            rhs[1],routes[1],tables[1],span,mesh_xy,'y'),factor_spec[1])
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
        (a0,a1),(b0,b1)=panels[0]
        kernel=_w_contraction(mesh_xy,tuple(left['grid']),nk,m*(a1-a0),n*(b1-b0),kcarrier,
                              factor_layout,weights_fn or _shared_pole_weights)
        # A streamed class (Lorentz panels or one factor face) also forms each
        # panel's W(t) in q panels whose GEMM operand copies (the panel's
        # factor columns, both faces) are at most one W(t) panel tile; a class
        # that fits forms it in one call, as before.
        q_rows=nk
        if len(panels)>1 or mirror or _TEST_W_Q_ROWS is not None:
            split=factor_layout=='face'
            copies=2*16*nk*((m//px)*(a1-a0)+(n//py)*(b1-b0))*(kcarrier//px if split else kcarrier)
            tile=16*nk*m*n*_blocks(panels[0])//mesh_xy.size
            want=max(1,-(-copies//max(tile,1))) if _TEST_W_Q_ROWS is None else -(-nk//_TEST_W_Q_ROWS)
            q_rows=max(d for d in range(1,nk+1) if nk%d==0 and d<=nk//want)
        row_kernel=(kernel if q_rows>=nk else _w_contraction(mesh_xy,tuple(left['grid']),nk,
            m*(a1-a0),n*(b1-b0),kcarrier,factor_layout,weights_fn or _shared_pole_weights,rows=q_rows))
        from symmetry_maps import q_negation_index
        minus=np.asarray(q_negation_index(tuple(left['grid'])),dtype=np.int32)
    except BaseException:
        capacity.live_stages=ambient
        b_x=b_y=poles=None
        raise
    replicated=NamedSharding(mesh_xy,P())
    def w_kernel(x,y,omega,interval,ref,time,hole):
        # (nk, m*|A|, n*|B|) is centroid-major per endpoint: the four-current
        # door reads it as (nk, m, |A|, n, |B|) without a transpose. The
        # component axis of the factors is unsharded, so a panel is a local slice.
        def form(panel,x,y,omega,interval):
            (a0,a1),(b0,b1)=panel
            if q_rows<nk:
                return q_panels(panel,x,y,omega,interval)
            if y is None:
                y=_other_face(mesh_xy,factor_spec[0],factor_spec[1])(
                    x if panels==whole else x[:,:,b0:b1])
            elif panels!=whole:
                y=y[:,:,b0:b1]
            if panels!=whole:
                x=x[:,:,a0:a1]
            return kernel(x,y,omega,interval,ref,time,hole).reshape(nk,m,a1-a0,n,b1-b0)
        def q_panels(panel,x,y,omega,interval):
            # W(t) of one Lorentz panel in q panels of q_rows rows: each reads
            # its rows (at -q on the valence branch), slices the panel's
            # components, contracts and is written in place, so the GEMM's
            # operand copies are one q panel's.
            (a0,a1),(b0,b1)=panel
            order=jnp.asarray((minus if hole else np.arange(nk,dtype=np.int32)).reshape(-1,q_rows))
            def body(carry,rows):
                xc=jnp.take(x,rows,axis=0)
                yc=(_other_face(mesh_xy,factor_spec[0],factor_spec[1])(xc[:,:,b0:b1])
                    if y is None else jnp.take(y,rows,axis=0)[:,:,b0:b1])
                xc=xc[:,:,a0:a1]
                if hole:
                    xc,yc=jnp.conj(xc),jnp.conj(yc)
                return carry,row_kernel(xc,yc,jnp.take(omega,rows,axis=0),
                                        jnp.take(interval,rows,axis=0),ref,time,False)
            # Stacked scan outputs: the panel's W(t) exists only from its own
            # loop on (no zero-filled buffer the scheduler could hoist).
            _,w=jax.lax.scan(body,None,order,unroll=1)
            return w.reshape(nk,m,a1-a0,n,b1-b0)
        return LorentzPanels(panels,form,(x,y,omega,interval))
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
    synthesis=WSynthesis(w_kernel,window_operands,lambda:(b_x,b_y,poles),close,native,
                      ('w',mesh_xy,tuple(left['grid']),nk,m,nc,n,nt,kcarrier,layout,
                       factor_layout,panels),ordered=True)
    synthesis.panels=panels
    return synthesis


@lru_cache(maxsize=None)
def _set_q_rows(mesh_xy, spec=P(None, 'x', 'y')):
    """Write a q panel into a full-q operand in place (q is unsharded)."""
    return jax.jit(lambda full, part, lo: jax.lax.dynamic_update_slice_in_dim(full, part, lo, axis=0),
                   donate_argnums=0, out_shardings=NamedSharding(mesh_xy, spec))


def instantaneous_sector_sigma(handle, families, bases, meta, mesh_xy, *,
                               occupation_state, return_components=False):
    """Exchange-like equal-time contraction of W_infinity-V, exactly once.

    The constant is read and packed in parent-q panels, and each class is
    restored and convolved in Lorentz-block panels, only when the whole does
    not fit the ledger; a deck that fits takes one read and one call per class.
    """
    from gw.photon_layout import PhotonBasisLayout, photon_block_view, pack_photon_operator
    from gw.photon_sigma import contract_lorentz_blocks, _TERM_X
    from gw.cohsex_sigma import _resolve_Gij
    from gw.qgrid_symmetry import qgrid_trs_policy_from_shared_pole_store
    from file_io.shared_pole_store import read_bank_constant_header, read_bank_constant
    header=read_bank_constant_header(handle,mesh_xy=mesh_xy)
    raw_layout=PhotonBasisLayout.from_centroid_extents(bases[0].n_logical,bases[1].n_logical,mesh_xy)
    layout=PhotonBasisLayout.from_centroid_extents(bases[0].n_packed,bases[1].n_packed,mesh_xy,packed=True)
    nq=int(header['bank_shape']['nq'])
    amount=16*nq*max(raw_layout.packed_extent,layout.packed_extent)**2//mesh_xy.size
    ledger=meta.shared_pole_capacity
    # The raw constant is read and packed in parent-q panels beside the
    # packed operator when raw + packed (with their workspace, 4x) do not
    # fit (Fe 20^3/P36: 97.87 GB/rank); a deck that fits reads it once.
    room=ledger.room_bytes_per_rank(ledger.live_stages)
    span=nq if 4*amount<=room else max(1,min(nq,(room-amount)*nq//(4*amount)))
    if _TEST_CONSTANT_Q_SPAN is not None:
        span=min(span,_TEST_CONSTANT_Q_SPAN)
    part=-(-amount*span//nq)
    ledger.reserve('sigma.sector.constant.pack',
        resident_bytes_per_rank=2*amount if span==nq else amount+2*part,
        workspace_bytes_per_rank=2*amount if span==nq else 2*part,
        concurrent_with=ledger.live_stages)
    def packer(raw):
        def block(A,B):
            value=photon_block_view(raw,raw_layout,A,B,mesh_xy)
            value=bases[bool(A)].pack_axis(value,1,spec=P(None,'x','y'))
            return bases[bool(B)].pack_axis(value,2,spec=P(None,'x','y'))
        return block
    if span==nq:
        raw=read_bank_constant(handle,header,meta=meta,mesh_xy=mesh_xy)
        packed=pack_photon_operator(packer(raw),nq,layout,mesh_xy)
        packed.block_until_ready()
        del raw
    else:
        packed=_zeros(mesh_xy,(nq,layout.packed_extent,layout.packed_extent))()
        for lo in range(0,nq,span):
            hi=min(lo+span,nq)
            raw=read_bank_constant(handle,header,meta=meta,mesh_xy=mesh_xy,q_span=(lo,hi))
            packed=_set_q_rows(mesh_xy)(packed,pack_photon_operator(packer(raw),hi-lo,layout,mesh_xy),lo)
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
    # The class restore is the full-q (nk, m, |A|, n, |B|) interaction (TT
    # at Fe 20^3/P36: 103.7 GB/rank). Split a class into A x B panels (one
    # component row, then one block) when its restore, the door's transform
    # of it and one temporary do not fit beside the packed constant.
    room=meta.shared_pole_capacity.room_bytes_per_rank(meta.shared_pole_capacity.live_stages)-amount
    nk=int(meta.nk_tot)
    def panels(class_keys):
        lefts=tuple(dict.fromkeys(A for A,_ in class_keys))
        rights=tuple(dict.fromkeys(B for _,B in class_keys))
        m,n=(int(bases[bool(v[0])].n_packed) for v in (lefts,rights))
        ladder=[(class_keys,)]
        if len(lefts)>1:ladder.append(tuple(tuple((A,B) for B in rights) for A in lefts))
        if len(rights)>1:ladder.append(tuple(((A,B),) for A in lefts for B in rights))
        if _TEST_MAX_PANEL_BLOCKS is not None:
            ladder=[p for p in ladder if len(p[0])<=_TEST_MAX_PANEL_BLOCKS] or ladder[-1:]
        for schedule in ladder:
            if 3*16*nk*m*n*len(schedule[0])//mesh_xy.size<=room:
                return schedule
        return ladder[-1]
    total=None
    currents=[None,None]
    for key,value,_ in contract_lorentz_blocks(keys,families=families,term=_TERM_X,
            response=response,Gij=gij,meta=meta,mesh_xy=mesh_xy,admit_kernel=admit,
            panels=panels):
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
                    families[a].layout,freq,meta,mesh_xy)
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
                wc=sum(synthesis.w_kernel(x,y,poles,intervals,0.0,0.0,hole).block(0).reshape(Q,m,m)
                       for hole in (False,True))
            finally:
                synthesis.close(wc)
    finally:
        capacity.live_stages=ambient
    return wc
