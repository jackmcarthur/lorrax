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
# symmetry tables by content), so a later map dispatches the same jit objects
# and XLA compiles each program once per run.  Data (factors, poles,
# intervals) always enters as an argument, never as a closure constant.


@lru_cache(maxsize=None)
def _placer(mesh_xy, spec):
    """Reshard to ``spec`` (identity values); the scalar route's synthesis places with it."""
    return jax.jit(lambda x: x, out_shardings=NamedSharding(mesh_xy, spec))


@lru_cache(maxsize=None)
def _zeros(mesh_xy, shape, spec=P(None, 'x', 'y')):
    return jax.jit(lambda: jnp.zeros(shape, jnp.complex128),
                   out_shardings=NamedSharding(mesh_xy, spec))


#: Test hook: the largest parent-q panel the instantaneous constant is read
#: and packed in (``None``: the ledger decides).
_TEST_CONSTANT_Q_SPAN = None


class ParentW:
    """W(t) of one endpoint class on the irreducible q, as the Green is on the parent k.

    ``W`` ``(nq_irr, m, nA, n, nB)`` centroid-major at ``P(None,'x',None,'y',None)``
    is ``B_A d(t) B_B^dagger`` and ``partner`` ``conj(B_A) d(t) B_B^T`` (the
    antiunitary rows' tile), both from ``gw.greens_function_kernel.build_G_parents``.
    ``hole`` (static) selects the valence branch, W_-(q) = partner(-q): the
    same pair read through the q-negated tables.  The Sigma door unfolds W on
    its load; no full-q W or full-grid W_R exists.
    """

    def __init__(self, W, partner, hole):
        self.W, self.partner, self.hole = W, partner, bool(hole)


_W_TABLES = {}


def _w_tables(headers, bases, syms, mesh_xy):
    """The W q-plan of an endpoint class: ``(plan, particle tables, hole tables)``, cached by content.

    The store's ``qirr`` rows (irr_idx_q, sym_idx_q, q_irr_frac) and each
    endpoint's packed centroid action (``shared_pole_packed_action``); the
    endpoint's Lorentz action is Cartesian (time-odd, polar) for a current
    endpoint and 1 for the charge endpoint.  The tables are
    ``symmetry_maps.unfold_load_tables`` with the pair-transpose rule, so an
    antiunitary q row reads the partner tile.  The hole tables are the same
    tables read at -q with the particle/partner roles swapped (the flag
    flipped and both phases conjugated): W_-(q) = partner(-q), no -q gather.
    """
    import hashlib
    from symmetry_maps import unfold_load_tables
    from gw.qgrid_symmetry import shared_pole_packed_action
    from gw.centroid_k_unfold import CentroidKUnfoldPlan
    qt = headers[0]['qirr']
    irr = np.asarray(qt['irr_idx_q'], np.int32)
    ops = np.asarray(qt['sym_idx_q'], np.int32)
    q_frac = np.asarray(qt['q_irr_frac'], np.float64)
    n_spatial = int(qt['n_sym_spatial'])
    ends = []
    for header, basis, sym in zip(headers, bases, syms):
        perm, wraps, _ = shared_pole_packed_action(SimpleNamespace(mu_basis=basis), header, mesh_xy=mesh_xy)
        comps = int(header.get('factor_components', 1))
        action = (np.asarray(sym.cartesian_action(ops, axial=False, time_odd=True), np.complex128)
                  if comps == 3 else np.ones((ops.size, 1, 1), np.complex128))
        ends.append((np.asarray(perm, np.int32), np.asarray(wraps), action))
    digest = hashlib.sha256()
    for a in (irr, ops, q_frac, *(x for e in ends for x in e), np.asarray(headers[0]['grid'])):
        digest.update(np.ascontiguousarray(a).tobytes())
        digest.update(str(a.shape).encode())
    key = (digest.hexdigest(), n_spatial, tuple(d.id for d in np.asarray(mesh_xy.devices).flat))
    hit = _W_TABLES.get(key)
    if hit is not None:
        return hit
    (perm_l, wraps_l, act_l), (perm_r, wraps_r, act_r) = ends
    for axis, perm, action in (('x', perm_l, act_l), ('y', perm_r, act_r)):
        # Refuse at plan time a spin action that is not block-diagonal on the
        # 2+2 Dirac split's Lorentz partner: every Cartesian or scalar action is.
        if action.shape[-1] not in (1, 3):
            raise ValueError(f"GATE shared_pole_w_tables: the {axis} endpoint action has "
                             f"{action.shape[-1]} components; want 1 (charge) or 3 (current)")
    particle = unfold_load_tables(
        irr_idx=irr, sym_idx=ops, sym_perm=perm_l, L_table=wraps_l, k_irr_frac=q_frac,
        spin_action_full=act_l, n_sym_spatial=n_spatial, mesh_xy=mesh_xy,
        right_sym_perm=perm_r, right_L_table=wraps_r, trs_rule='pair_transpose',
        right_spin_action_full=act_r)
    if any(np.any(np.imag(a) != 0) for a in (act_l, act_r)):
        # The hole tables keep U unconjugated: W_-(k) = R partner R^T needs a real action.
        raise ValueError("GATE shared_pole_w_tables: an endpoint action is not real; the hole "
                         "branch's q-negated tables need real (Cartesian or scalar) actions")
    hole = hole_tables(particle, headers[0]['grid'])
    plan = CentroidKUnfoldPlan(
        mesh_xy=mesh_xy, layout=bases[0].layout, irr_idx=irr, sym_idx=ops, sym_perm=perm_l,
        L_table=wraps_l, k_parent_frac=q_frac, spin_action_full=act_l,
        n_sym_spatial=n_spatial, nspinor=int(act_l.shape[-1]))
    _W_TABLES[key] = (plan, particle, hole)
    return _W_TABLES[key]


def hole_tables(particle, grid):
    """The valence branch's load tables, W_-(q) = partner(-q): ``particle`` read at -q.

    The flag is flipped (a row that read W reads its partner) and both phases
    conjugated, so no -q gather of W is formed.  Needs a real endpoint action
    (the caller's check); the scalar route (``gw.mpa.sigma``) reads its
    ordered store's W_+(-q)^T through the same tables.
    """
    from symmetry_maps import q_negation_index
    qn = np.asarray(q_negation_index(tuple(int(v) for v in grid)), np.int64)
    return particle._replace(
        row=particle.row[qn], trs=(1 - particle.trs[qn]).astype(np.int32),
        lsrc=particle.lsrc[qn], rsrc=particle.rsrc[qn],
        mph=np.conj(particle.mph[qn]), nph=np.conj(particle.nph[qn]),
        spin=particle.spin[qn], spin_r=None if particle.spin_r is None else particle.spin_r[qn])


_W_PARENTS = {}


def _w_parents(mesh_xy, plan, m, nc, n, nt, kcarrier, weights_fn, panel_bytes):
    """``jit((b_x, b_y, poles, intervals, ref, time) -> (W, partner))`` on the irreducible q.

    The factors enter ``build_G_parents`` as the Green's faces do
    (``_shared_pole_contract``'s transpose-and-build, on the parent rows):
    components merged with their own centroid axis, the pole axis as the band
    axis, ``d(t)`` as the phase row.  The contraction is the face Green's
    batched SUMMA (``distrib_la.panel_matmul``) with its two live pole panels
    bounded by ``panel_bytes`` (the ledger's room; one W tile at most).  The
    partner is ``conj(B_A) d B_B^T`` at the same d: on the face route
    (``face_green_product(partner=True)``) d scales each gathered panel slice
    and the partner comes from the same exchange, so no scaled or conjugated
    copy of a face is formed.  The route bounds its two live panels by one
    Green tile of ``n_full`` rows; the builder's plan here says ``n_full`` =
    the rows whose tile is ``panel_bytes`` (both whole faces when they fit:
    one panel step).
    """
    key = (mesh_xy, id(plan), m, nc, n, nt, kcarrier, weights_fn, int(panel_bytes))
    if key in _W_PARENTS:
        return _W_PARENTS[key][1]
    from gw.greens_function_kernel import build_G_parents
    nq = int(plan.n_parent)
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])
    rows = int(max(1, int(panel_bytes) // (16 * (m * nc // px) * (n * nt // py))))
    gemm = SimpleNamespace(backend='face', mesh=mesh_xy)
    face_plan = SimpleNamespace(sym_idx=plan.sym_idx, n_sym_spatial=plan.n_sym_spatial,
                                mesh_xy=mesh_xy, n_full=rows)
    antiunitary = bool(np.any(np.asarray(plan.sym_idx) >= int(plan.n_sym_spatial)))

    @jax.jit
    def kernel(b_x, b_y, poles, intervals, ref, time):
        d = weights_fn(poles, intervals, ref, time)
        x = b_x.reshape(nq, m * nc, 1, kcarrier).transpose(0, 2, 1, 3)
        y = b_y.reshape(nq, n * nt, 1, kcarrier).transpose(0, 3, 2, 1)
        pg = build_G_parents(x, y, phases=d, layout='face', gemm=gemm, k_unfold_plan=face_plan,
                             real_weights=False)
        partner = pg.partner() if antiunitary else build_G_parents(
            jnp.conj(x), jnp.conj(y), phases=d, layout='face', gemm=gemm,
            k_unfold_plan=face_plan, real_weights=False).G
        shape = (nq, m, nc, n, nt)
        return pg.G.reshape(shape), partner.reshape(shape)
    _W_PARENTS[key] = (plan, kernel)
    return kernel


#: The sector Σ doors' placed load tables (Green, W particle/hole) per mesh, plans and
#: W tables: every SC map's τ programs read the same device tables.  Bounded.
_SECTOR_DOOR_LOADS = {}


def sector_tau_factory(left, right, keys, meta, mesh_xy):
    """Bind Gamma_A G_AB(t) Gamma_B to the window executor, row pass by row pass (``gw.subtile_stream``).

        Σ_mn(k) = Σ_{μ ∈ passes} Σ_ν ψ*_m(μ) [Σ_AB γ̃_A G γ̃_B† ⋆ W_AB](k)_{μν} ψ_n(ν)

    is linear in the μ rows, so each rank's ``(μ_X, ν_Y)`` tile of the class
    runs in row passes of whole centroid orbits (cuts admissible for the
    Green's tables and for both branches' W tables).  Per pass:

    - the four-spinor parent Green on the pass's band-complete ψ rows: one
      local GEMM against the right operand formed once per Σ call
      (``greens_function_kernel.green_right_operand``), its antiunitary
      partner from the same operands at conjugate weights;
    - one ns = 4 mode-8 Lorentz door (``common.fft_helpers.make_kconv_lorentz_unfold``,
      every vertex of the class in its Mid) reading the pass's rows of W(t)
      on the irreducible q, with the Green's and W's tables placed once on the
      devices and cut to the pass there (``subtile_stream.pass_load``), so no
      program holds table constants;
    - the axis band projection of the pass's rows into a rank-local partial;

    and one band-block reduce-scatter ends the node.  No whole-tile
    four-spinor Green exists.  Pass sizes come from
    :data:`runtime.tiles.TILE_BYTES` and the shapes
    (``subtile_stream.plan_rows``), the rule the scalar Σ τ engine uses.
    The caller places the left operands band-complete once per Σ call
    (``ppm_tau_kernel.sigma_subtile_operands``); the right ones are placed here.
    """
    from distrib_la import gemm_plan
    from common.contract_bands import contract_bands_block_reshard
    from common.fft_helpers import make_kconv_lorentz_unfold
    from common.gamma_matrices import gamma_perm_phase_host
    from gw.cohsex_sigma import lorentz_class_vertices
    from gw.greens_function_kernel import (build_G_parents, _weighted_tau_phases,
                                           green_right_operand, has_antiunitary_rows)
    from symmetry_maps import device_load_tables
    from gw.subtile_stream import (fold_passes, orbit_cuts, pass_load, pass_rows, pass_tables,
                                   plan_rows)

    a, b = left.green_parent, right.green_parent
    plans = a.plan, b.plan
    n_parent, nb = int(plans[0].n_parent), int(a.psi_nmu.shape[1])
    m, n = int(plans[0].n_centroid_packed), int(plans[1].n_centroid_packed)
    ns = int(plans[0].nspinor)
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])
    local_rows, nu = m // px, n // py
    kgrid = tuple(int(v) for v in meta.kgrid)
    lefts, rights = lorentz_class_vertices(keys)
    vertices = ([gamma_perm_phase_host(A) for A in lefts], [gamma_perm_phase_host(B) for B in rights])
    n_a, n_b = (1 if lefts == (0,) else 3), (1 if rights == (0,) else 3)
    g_tables = plans[0].unfold_load_tables(right_plan=None if plans[1] is plans[0] else plans[1])
    partner = int(has_antiunitary_rows(plans[0]))
    mult = -1.0 / np.sqrt(float(meta.nk_tot))

    def factory(synthesis, band_axis):
        _, right_yr, _, right_proj, _, _ = parent_sigma_operands(right)
        right_proj = pad_to_axis(right_proj, band_axis, axis=3)
        nb_sig = int(band_axis.padded)
        # The right operands, placed once per Σ call: the Green's right
        # operand (conj ψ_nmu with every band, merged) and the projector's.
        @jax.jit
        def place(yr, yn):
            # subtile_stream.band_complete's columns and projection_complete's right operand.
            cols = jax.lax.with_sharding_constraint(yr, NamedSharding(mesh_xy, P(None, None, None, 'y')))
            right_p = jax.lax.with_sharding_constraint(yn, NamedSharding(mesh_xy, P(None, None, 'y', None)))
            return green_right_operand(cols), right_p
        right_g, right_p = place(right_yr, right_proj)
        w_tables = tuple(synthesis.w_tables)
        n_w = int(np.max(np.asarray(w_tables[0].row))) + 1     # the W parents (irreducible q)
        # One local row's live set: the parent Green and its partner, the
        # door's Σ rows, the pass's rows of W(t) and its partner, the pass's
        # ψ rows (Green and projection).
        row_bytes = 16 * ((1 + partner) * n_parent * ns * ns * nu + n_parent * ns * ns * nu
                          + 2 * n_w * n_a * n_b * nu + n_parent * ns * (nb + nb_sig))
        passes = plan_rows(local_rows, row_bytes, lambda: sorted(
            set(orbit_cuts(g_tables.lsrc, px, ns))
            & set.intersection(*(set(orbit_cuts(t.lsrc, px, n_a)) for t in w_tables))))
        stages = []
        for x0, xr in passes:
            whole = (x0, xr) == (0, local_rows)
            cut = (lambda t, k: t) if whole else (lambda t, k: pass_tables(t, x0, xr, px, k))
            gemm = gemm_plan(mesh_xy, m=px * xr * ns, k=nb, n=n * ns, nq=n_parent,
                             dtype=jnp.complex128, layout='axis', warmup=False)
            doors = {hole: make_kconv_lorentz_unfold(
                mesh_xy, kgrid, cut(g_tables, ns), left_vertices=vertices[0],
                right_vertices=vertices[1], store_rows=plans[0].parent_full_rows,
                norm='ortho', mult=mult, w_tables=cut(w, n_a))
                for hole, w in ((False, w_tables[0]), (True, w_tables[1]))}
            project = contract_bands_block_reshard(
                mesh_xy, channels="none", layout="axis", face_shape=(n_parent, nb, px * xr, ns),
                right_face_shape=(n_parent, nb, n, ns), face_band_extent=nb_sig)
            stages.append((whole, gemm, doors, project))
        finish = stages[0][3].finish
        key_t = (mesh_xy, id(plans[0]), id(plans[1]), id(w_tables[0]), id(w_tables[1]))
        if key_t not in _SECTOR_DOOR_LOADS:
            while len(_SECTOR_DOOR_LOADS) >= 8:
                _SECTOR_DOOR_LOADS.pop(next(iter(_SECTOR_DOOR_LOADS)))
            # The plans and tables ride along so their ids in the key cannot be reused.
            _SECTOR_DOOR_LOADS[key_t] = ((plans, w_tables), (
                device_load_tables(g_tables, mesh_xy),
                tuple(device_load_tables(t, mesh_xy) for t in w_tables)))
        door_loads = _SECTOR_DOOR_LOADS[key_t][1]
        psi_bytes = 16 * n_parent * ns * (nb + nb_sig) * (local_rows + nu)
        price = dict(d=ns, ns=ns, passes=len(passes),
                     new=float(max(xr for _, xr in passes) * row_bytes + psi_bytes))
        from common.gpu_utils import record_stage_price
        record_stage_price(f"Sigma tau {keys[0]}, {len(passes)} row pass(es)", price["new"],
                           section="sigma.tau_sweep")
        if jax.process_index() == 0:
            print(f"Sigma tau stream {''.join('CT'[f] for f in (int(lefts != (0,)), int(rights != (0,))))}: "
                  f"{len(passes)} row pass(es) of {max(xr for _, xr in passes)} local rows "
                  f"({local_rows} per rank)", flush=True)
        partial_spec = NamedSharding(mesh_xy, P(None, ('x', 'y')))
        w_spec = P(None, 'x', None, 'y', None)

        def spatial(xn, yr, xr, yn, energies, weight, reference, time, interactions, loads):
            # xn, xr: the left band-complete ψ rows and projection rows (placed
            # by the caller); yr, yn: the right Green operand and projection
            # operand, and ``loads`` the doors' placed tables (window arguments).
            phases = _weighted_tau_phases(energies, 1j*time, e_ref=reference, band_weight=weight)
            hole = interactions.hole
            g_load, w_load = loads[0], loads[1][int(hole)]
            zero = jax.lax.with_sharding_constraint(
                jnp.zeros((1, px * py * n_parent, nb_sig, nb_sig), jnp.complex128), partial_spec)

            def step(p, x0, xr_, acc, operands):
                whole, gemm, doors, project = stages[p]
                W, Wt, rows, left_p = operands
                if not whole:
                    W, Wt = (pass_rows(w, mesh_xy, x0, xr_, axis=1, spec=w_spec) for w in (W, Wt))
                    rows = pass_rows(rows, mesh_xy, x0, xr_, axis=2)
                    left_p = pass_rows(left_p, mesh_xy, x0, xr_, axis=3)
                green = build_G_parents(rows, None, phases=phases, layout='axis', gemm=gemm,
                                        k_unfold_plan=plans[0], real_weights=False, right=yr)
                sigma = doors[hole](green.G, green.transpose, W, Wt,
                                    conj_partner=green.conj_partner,
                                    load=g_load if whole else pass_load(g_load, mesh_xy, x0, xr_, ns),
                                    w_load=(w_load if whole else
                                            pass_load(w_load, mesh_xy, x0, xr_, n_a)))
                return project.accumulate((jnp.conj(left_p), yn), sigma, acc=acc)
            return finish(fold_passes(passes, step, zero,
                                      (interactions.W, interactions.partner, xn, xr)))
        spatial.price = price
        key = (mesh_xy, shapes_key, int(nb_sig), tuple(keys), kgrid, int(meta.nk_tot),
               id(plans[0]), id(plans[1]), id(w_tables[0]), id(w_tables[1]), tuple(passes))
        return SynthesisTau(spatial, synthesis, right_g, right_p, synthesis.native,
                            f'sigma.sector.tau.{keys[0]}', meta, key, (*plans, *w_tables),
                            door=door_loads)
    shapes_key = (n_parent, nb, m, n, ns)
    return factory


def sector_synthesis(readers, headers, bases, syms, layout, frequencies, meta, mesh_xy,
                     *, weights_fn=None, stage='sigma'):
    """Keep the endpoint factors on the irreducible q and form W(t) there, per tau.

    The store is read once at setup: both factor faces on the store's own
    parent rows (``nq_irr``), never unfolded.  Each tau builds
    ``W = B_A d(t) B_B^dagger`` and its partner on those rows with the Green's
    builder (``gw.greens_function_kernel.build_G_parents``), and the Sigma
    door unfolds W on its load through the class's W tables
    (:func:`_w_tables`).  No full-q W, full-q W factor, unfolded pole table or
    full-grid W_R is held.  ``syms`` are the endpoint families' symmetry maps
    (a current endpoint's Cartesian action; a charge endpoint reads none).
    ``weights_fn`` is d: the causal d(t) by default, the omega = 0
    coefficient for :func:`sector_static_wc`, whose ledger stages ``stage``
    prefixes.  The occupied windows read W_-(q) = partner(-q) through the
    hole tables; d is never conjugated.  ``layout`` is kept for the caller's
    census (the factors stay on the store's face layout).
    """
    from file_io.shared_pole_store import read_shared_pole_faces
    from .sigma import _shared_pole_weights
    from .sigma_windows import shared_pole_intervals

    del layout
    left,right=headers
    nc,nt=(int(h.get('factor_components',1)) for h in headers)
    m,n=(b.n_packed for b in bases)
    nq=int(left['n_q_irr'])
    kmax=int(left['Kmax'])
    if any(left[k]!=right[k] for k in ('K','Kmax','q_irr_full_idx','identity')):
        raise ValueError('GATE shared_pole_sector_census: endpoint identities differ')
    capacity=meta.shared_pole_capacity
    ambient=capacity.live_stages
    tag=f'{left.get("sector")}.{right.get("sector")}'
    w_plan,*w_tables=_w_tables(headers,bases,syms,mesh_xy)
    shape=(nq,m,nc,n,nt)
    if not kmax:
        zero=_zeros(mesh_xy,shape,P(None,'x',None,'y',None))
        synthesis=WSynthesis(lambda _ref,_time,hole:ParentW(zero(),zero(),hole),
                          lambda _space,_indices,_bounds:(),lambda:(),lambda _result=None:None,0,
                          ('zero',mesh_xy,shape),ordered=True)
        synthesis.w_tables=tuple(w_tables)
        return synthesis
    # The store reader pads physical Kmax for both endpoint face shardings.
    # Keep that carrier through the GEMM; K and the interval bounds remain
    # physical, so the padded pole columns have identically zero weight.
    kcarrier=padded_axis(kmax,mesh_xy,name='sector_sigma_K',specs=(
        (P(None,'x',None,'y'),3),(P(None,'y',None,'x'),3))).carrier
    same=readers[0] is readers[1] and headers[0] is headers[1]
    faces=16*nq*kcarrier*(m*nc+n*nt)//mesh_xy.size
    tile=16*nq*m*nc*n*nt//mesh_xy.size
    native=0
    # The W pair's two live pole panels hold at most both whole faces (one
    # step, the fastest), bounded by what the room leaves beside the faces
    # and the pair; never below one pole column.
    column=16*nq*(m*nc//int(mesh_xy.shape['x'])+n*nt//int(mesh_xy.shape['y']))
    panel=max(column,min(faces,capacity.room_bytes_per_rank(ambient)-faces-8*nq*kcarrier-2*tile))
    setup=f'{stage}.sector.resident.{tag}'
    # Resident: both parent faces and the parent poles.  The W pair (2 tiles)
    # is the window executable's, priced there with the Green.
    capacity.reserve(setup,resident_bytes_per_rank=faces+8*nq*kcarrier,
        workspace_bytes_per_rank=2*tile+panel,concurrent_with=ambient)
    capacity.live_stages=(*ambient,setup)
    try:
        if same:
            lhs=read_shared_pole_faces(readers[0],(0,nq),meta=meta,header=left,basis=bases[0])
            b_x,b_y,poles=lhs[0],lhs[1],lhs[2]
        else:
            lhs=read_shared_pole_faces(readers[0],(0,nq),meta=meta,header=left,
                                       basis=bases[0],orientations=('x',))
            rhs=read_shared_pole_faces(readers[1],(0,nq),meta=meta,header=right,
                                       basis=bases[1],orientations=('y',))
            if not bool(jnp.all(lhs[2]==rhs[2])):
                raise ValueError('GATE shared_pole_sector_census: unequal pole values')
            b_x,b_y,poles=lhs[0],rhs[1],lhs[2]
        jax.block_until_ready((b_x,b_y,poles))
        del lhs
        kernel=_w_parents(mesh_xy,w_plan,m,nc,n,nt,kcarrier,weights_fn or _shared_pole_weights,panel)
    except BaseException:
        capacity.live_stages=ambient
        raise
    replicated=NamedSharding(mesh_xy,P())

    def w_kernel(x,y,omega,interval,ref,time,hole):
        W,partner=kernel(x,y,omega,interval,ref,time)
        return ParentW(W,partner,hole)

    def window_operands(space,indices,bounds):
        # Host intervals once per window, on the parent q rows; every tau node reuses them.
        intervals=shared_pole_intervals(frequencies,np.asarray(indices),np.asarray(bounds))
        return (b_x,b_y,poles,device_put_process_local(np.ascontiguousarray(intervals),replicated))
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
                      ('w-parent',mesh_xy,tuple(left['grid']),nq,m,nc,n,nt,kcarrier,id(w_plan)),
                      ordered=True)
    synthesis.w_tables=tuple(w_tables)
    return synthesis


@lru_cache(maxsize=None)
def _set_q_rows(mesh_xy, spec=P(None, 'x', 'y')):
    """Write a q panel into a full-q operand in place (q is unsharded)."""
    return jax.jit(lambda full, part, lo: jax.lax.dynamic_update_slice_in_dim(full, part, lo, axis=0),
                   donate_argnums=0, out_shardings=NamedSharding(mesh_xy, spec))


def instantaneous_sector_sigma(handle, families, bases, meta, mesh_xy, *,
                               occupation_state, return_components=False):
    """Exchange-like equal-time contraction of W_infinity-V, exactly once.

    The constant is read and packed on its irreducible q (in parent-q panels
    only when raw + packed do not fit the ledger); each endpoint class is its
    parent pair ``(W, conj W)``, which the four-current door unfolds on its
    load with the occupied Green (``gw.photon_sigma.contract_lorentz_blocks``,
    d = 1, one branch).  No full-q class operand is formed.
    """
    from gw.photon_layout import PhotonBasisLayout, pack_photon_operator
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
            from gw.photon_layout import photon_block_view
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


def _unfold_w_rows(W, Wt, tables, rows, mesh_xy):
    """``(len(rows), m*nA, n*nB)`` at ``P(None,'x','y')``: the parent pair through ``tables``
    at the full-q rows ``rows`` only (the service's reference unfold, on each rank's
    tiles).  Every per-q table is cut to those rows, so no full-q tile is formed."""
    from symmetry_maps import apply_unfold_load_tables_local, local_unfold_load_tables
    from common.shard_map import shard_map
    rows = np.asarray(rows, np.int64)
    tables = tables._replace(
        row=tables.row[rows], trs=tables.trs[rows], lsrc=tables.lsrc[rows],
        rsrc=tables.rsrc[rows], mph=tables.mph[rows], nph=tables.nph[rows],
        spin=tables.spin[rows], spin_r=None if tables.spin_r is None else tables.spin_r[rows])
    na, nb = int(W.shape[2]), int(W.shape[4])
    spin_l = np.asarray(tables.spin)
    spin_r = None if tables.spin_r is None else np.asarray(tables.spin_r)

    def local(w, wt):
        t = local_unfold_load_tables(tables)
        flat = lambda a: a.reshape(a.shape[0], a.shape[1] * na, a.shape[3] * nb)
        O = apply_unfold_load_tables_local(flat(w), flat(wt), t, spin_l, spin_r)
        return O.reshape(O.shape[0], O.shape[1] * na, O.shape[3] * nb)
    spec = P(None, 'x', None, 'y', None)
    return shard_map(local, mesh=mesh_xy, in_specs=(spec, spec), out_specs=P(None, 'x', 'y'),
                     check_vma=False)(W, Wt)


def sector_static_wc(handle, meta, *, mesh_xy, rows):
    """Wc_CC(q, omega = 0) of a four-current sector model at the full-q rows ``rows``.

    The charge-sector body of the restart's ``W0_qmunu = V + Wc_CC(0)`` for
    ``bispinor_gw = full_shared_pole``, as :func:`gw.mpa.sigma.shared_pole_static_wc`
    is for a scalar store.  It is the Sigma sector synthesis
    (:func:`sector_synthesis`: one factor read, the parent contraction) with
    ``_shared_pole_omega0_weights`` in place of d(t), summing W_+(q) and the
    valence branch W_-(q) = partner(-q), unfolded through the same tables the
    Sigma door reads.  No ``W_inf - V`` term enters: the Ward contact is TT-only and V is
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

    ``rows`` are the full-q rows the restart stores: the q parents of the
    run's V wedge (``QirrOperator.full_rows``), or every q on a deck whose q
    axis does not reduce.  Both branches are unfolded at those rows only, so
    no full-q W is formed (TASTE 97); BSE unfolds the stored parents on load
    (``file_io.restart_bundle.read_interaction``).

    Returns ``(len(rows), m, m)`` complex128 at ``P(None,'x','y')`` in the
    run's packed charge-centroid order (``meta.mu_basis``, the CC endpoint
    basis).
    """
    from file_io.shared_pole_store import open_shared_pole_model, validate_shared_pole_model
    from .sigma import _shared_pole_omega0_weights
    cc=handle['sectors']['CC']
    capacity=meta.shared_pole_capacity
    header=validate_shared_pole_model(cc['path'],expected_identity=cc['identity'],
                                      mesh_xy=mesh_xy,capacity=capacity)
    if header.get('sector')!='CC' or header['digest']!=cc['digest']:
        raise ValueError('GATE shared_pole_identity: W0 handle is not the published CC model')
    rows=np.asarray(rows,np.int64).reshape(-1)
    Q,m=int(header['n_q_full']),int(meta.mu_basis.n_packed)
    if rows.size==0 or rows.min()<0 or rows.max()>=Q:
        raise ValueError(f'GATE shared_pole_static_w: rows must be full-q rows in [0, {Q})')
    if not int(header['Kmax']):
        return _zeros(mesh_xy,(rows.size,m,m))()
    counts=np.asarray(header['K'],np.int64)
    intervals=device_put_process_local(np.ascontiguousarray(
        np.stack([np.zeros_like(counts),counts],axis=1)),NamedSharding(mesh_xy,P()))
    ambient=capacity.live_stages
    tile=-(-16*rows.size*m*m//int(mesh_xy.size))
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
                # Both branches at the stored rows only: W_+ through the particle
                # tables and W_-(q) = partner(-q) through the hole tables.
                x,y,poles=synthesis.resident_operands()
                pair=synthesis.w_kernel(x,y,poles,intervals,0.0,0.0,False)
                wc=sum(_unfold_w_rows(pair.W,pair.partner,tables,rows,mesh_xy)
                       for tables in synthesis.w_tables)
            finally:
                synthesis.close(wc)
    finally:
        capacity.live_stages=ambient
    return wc
