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
from runtime.padding import pad_to_axis
from gw.wavefunction_bundle import parent_sigma_operands
from .sigma import SynthesisTau, WSynthesis, _admit, _static_key


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
#: and packed in (``None``: one tile decides).
_TEST_CONSTANT_Q_SPAN = None


class ParentW:
    """W(t) of one endpoint class on the irreducible q, as the Green is on the parent k.

    ``W`` ``(nq_irr, m, nA, n, nB)`` centroid-major at ``P(None,'x',None,'y',None)``
    is ``B_A d(t) B_B^dagger`` and ``partner`` ``conj(B_A) d(t) B_B^T`` (the
    antiunitary rows' tile), both from ``gw.greens_function_kernel.build_G_parents``.
    ``hole`` (static) selects the valence branch, W_-(q) = partner(-q): the
    same pair read through the q-negated tables.  The Sigma kconv call unfolds W on
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


_W_PROGRAMS = {}


def _w_program(mesh_xy, route, same, m, nc, n, nt, nq, kcarrier, weights_fn):
    """``jit((b_x, b_y, poles, intervals, ref, time) -> (W, partner))`` on the irreducible q.

    The scalar W(τ) owner, :func:`gw.mpa.sigma.synthesize_shared_pole_parents`,
    with the placement ``route`` its schedule chose (``axis``: replicated pole
    columns, one local GEMM per rank and no exchange; ``local``: whole parents
    per rank, ``distrib_la.batch_gram``, only W moves; ``face``: the Green's
    batched SUMMA, panels bounded by one W tile).  Only each parent's live
    pole columns are contracted.  The partner ``conj(B_A) d B_B^T`` is the
    transpose of W on a diagonal sector (``same``) and, on a mixed one, the
    same contraction on the conjugate factors (the face route's same panel
    exchange).  Components are merged with their own centroid axis, so W is
    ``(nq, m, nc, n, nt)`` at ``P(None,'x',None,'y',None)``.  One program per
    configuration and process.
    """
    key = (mesh_xy, route, bool(same), m, nc, n, nt, nq, kcarrier, weights_fn)
    hit = _W_PROGRAMS.get(key)
    if hit is not None:
        return hit
    from distrib_la import gemm_plan
    from .sigma import synthesize_shared_pole_parents
    gemm = None if route == 'local' else gemm_plan(
        mesh_xy, m=m * nc, k=kcarrier, n=n * nt, nq=nq, dtype=np.complex128, layout=route,
        enable_active_range=True, warmup=False)
    shape = (nq, m, nc, n, nt)
    spec = NamedSharding(mesh_xy, P(None, 'x', None, 'y', None))

    @jax.jit
    def kernel(b_x, b_y, poles, intervals, ref, time):
        pair = synthesize_shared_pole_parents(
            b_x, b_y, poles, jnp.clip(intervals, 0, kcarrier), ref, time, mesh_xy=mesh_xy,
            gemm=gemm, layout=route, weights_fn=weights_fn, active_range=True, same_factor=same,
            right_formed=route == 'axis')
        return tuple(jax.lax.with_sharding_constraint(w.reshape(shape), spec) for w in pair)
    _W_PROGRAMS[key] = kernel
    return kernel


#: The sector Σ kconv calls' placed load tables (Green, W particle/hole) per mesh, plans and
#: W tables: every SC map's τ programs read the same device tables.  Bounded.
_SECTOR_NODES = {}


def sector_node(left, right, keys, meta, mesh_xy, w_tables, band_axis, *, static=False,
                brackets=None):
    """One endpoint class's Σ node on the sub-tile engine (``gw.subtile_stream``), row pass by row pass.

        Σ_mn(k) = Σ_{μ ∈ passes} Σ_ν ψ*_m(μ) [Σ_AB γ̃_A G γ̃_B† ⋆ W_AB](k)_{μν} ψ_n(ν)

    is linear in the μ rows, so each rank's ``(μ_X, ν_Y)`` tile of the class
    runs in row passes of whole centroid orbits (cuts admissible for the
    Green's tables and for every branch's W tables).  Per pass:

    - the four-spinor parent Green on the pass's band-complete ψ rows: one
      local GEMM against the right operand formed once per Σ call
      (``greens_function_kernel.green_right_operand``), its antiunitary
      partner from the same operands at conjugate weights;
    - one ns = 4 mode-8 Lorentz k-convolution (``common.fft_helpers.make_kconv_lorentz_unfold``,
      every vertex of the class in its Mid) reading the pass's rows of W on
      the irreducible q, with the Green's and W's tables placed once on the
      devices and cut to the pass's window there (``subtile_stream.window_load``), so no
      program holds table constants;
    - the axis band projection of the pass's rows into a rank-local partial;

    and one band-block reduce-scatter ends the node.  No whole-tile
    four-spinor Green exists.  Pass sizes come from
    :data:`runtime.tiles.TILE_BYTES` and the shapes
    (``subtile_stream.plan_windows``), the rule the scalar Σ τ engine uses.

    ``w_tables`` are W's q tables, one per branch ``ParentW.hole`` selects: the
    particle and hole tables of an ordered W(t) (:func:`sector_tau_factory`),
    or the one table of a static class (``static``: the photon static classes,
    ``gw.photon_sigma.contract_lorentz_blocks``, the τ = 0 node).  Built once
    per configuration; returns ``SimpleNamespace(spatial, loads, key, plans)``
    with ``spatial(xn, yr, xr, yn, energies, weight, reference, time,
    interactions, loads)``: the left operands placed by
    :func:`sector_left_operands`, the right by :func:`sector_right_operands`,
    ``interactions`` a :class:`ParentW` and ``loads`` the node's placed tables.

    ``brackets`` (the band-extrapolation plan's disjoint band brackets,
    ``gw.ppm_pipeline.plan_sigma_band_brackets``): the Green band sum is split
    as the scalar Σ τ kernel splits it (``ppm_tau_kernel.bracket_selectors``).
    Inside each row pass every live bracket builds its own Green over its own
    bands (the active-range GEMM) and runs its own mode-8 convolution against
    the same W(t) rows, one bracket's Green at a time; the node returns the
    brackets on a leading axis.  W(t) is formed once per node.
    """
    from distrib_la import gemm_plan
    from common.contract_bands import contract_bands_block_reshard
    from common.fft_helpers import make_kconv_lorentz_unfold
    from common.gamma_matrices import gamma_perm_phase_host
    from gw.cohsex_sigma import lorentz_class_vertices
    from gw.greens_function_kernel import build_G_tau, has_antiunitary_rows
    from gw.ppm_tau_kernel import bracket_selectors
    from symmetry_maps import device_load_tables
    from gw.subtile_stream import (orbit_cuts, plan_windows, scan_passes, window_green_rows,
                                   window_load, window_rows, window_tables)

    a, b = left.green_parent, right.green_parent
    plans = a.plan, b.plan
    n_parent, nb = int(plans[0].n_parent), int(a.psi_nmu.shape[1])
    m, n = int(plans[0].n_centroid_packed), int(plans[1].n_centroid_packed)
    ns = int(plans[0].nspinor)
    nb_sig = int(band_axis.padded)
    kgrid = tuple(int(v) for v in meta.kgrid)
    w_tables = tuple(w_tables)
    selectors = None if brackets is None else tuple(
        (int(lo), None if hi is None else int(hi)) for lo, hi in brackets)
    key = (mesh_xy, id(plans[0]), id(plans[1]), (n_parent, nb, m, n, ns), nb_sig, tuple(keys),
           kgrid, int(meta.nk_tot), tuple(map(id, w_tables)), bool(static), selectors)
    hit = _SECTOR_NODES.get(key)
    if hit is not None:
        return hit[1]
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])
    local_rows, nu = m // px, n // py
    lefts, rights = lorentz_class_vertices(keys)
    vertices = ([gamma_perm_phase_host(A) for A in lefts], [gamma_perm_phase_host(B) for B in rights])
    n_a, n_b = (1 if lefts == (0,) else 3), (1 if rights == (0,) else 3)
    g_tables = plans[0].unfold_load_tables(right_plan=None if plans[1] is plans[0] else plans[1])
    partner = int(has_antiunitary_rows(plans[0]))
    mult = -1.0 / np.sqrt(float(meta.nk_tot))
    n_w = int(np.max(np.asarray(w_tables[0].row))) + 1     # the W parents (irreducible q)
    # One local row's live set: the parent Green and its partner, the
    # k-convolution's Σ rows, the pass's rows of W and its partner, the pass's
    # ψ rows (Green and projection).
    row_bytes = 16 * ((1 + partner) * n_parent * ns * ns * nu + n_parent * ns * ns * nu
                      + 2 * n_w * n_a * n_b * nu + n_parent * ns * (nb + nb_sig))
    R, windows = plan_windows(local_rows, row_bytes, lambda: sorted(
        set(orbit_cuts(g_tables.lsrc, px, ns))
        & set.intersection(*(set(orbit_cuts(t.lsrc, px, n_a)) for t in w_tables))))
    whole = len(windows) == 1
    # One window shape for every pass: host tables give the shapes, each pass
    # reads the placed tables' cut (subtile_stream.window_load).
    cut = (lambda t, k: t) if whole else (lambda t, k: window_tables(t, R, px, k))
    # Every Green contracts only its window's live bands (the active-range GEMM): a
    # bracket's own, else the window's (TT, CT, TC: 130-622 of 752 bands at CrI3 24x24).
    gemm = gemm_plan(mesh_xy, m=px * R * ns, k=nb, n=n * ns, nq=n_parent,
                     dtype=jnp.complex128, layout='axis', warmup=False, enable_active_range=True)
    kconv = tuple(make_kconv_lorentz_unfold(
        mesh_xy, kgrid, cut(g_tables, ns), left_vertices=vertices[0],
        right_vertices=vertices[1], store_rows=plans[0].parent_full_rows,
        norm='ortho', mult=mult, w_tables=cut(w, n_a)) for w in w_tables)
    project = contract_bands_block_reshard(
        mesh_xy, channels="none", layout="axis", face_shape=(n_parent, nb, px * R, ns),
        right_face_shape=(n_parent, nb, n, ns), face_band_extent=nb_sig)
    finish = project.finish
    loads = (device_load_tables(g_tables, mesh_xy),
             tuple(device_load_tables(t, mesh_xy) for t in w_tables))
    psi_bytes = 16 * n_parent * ns * (nb + nb_sig) * (local_rows + nu)
    price = dict(d=ns, ns=ns, passes=len(windows), new=float(R * row_bytes + psi_bytes))
    from common.gpu_utils import record_stage_price
    # A static node is priced in its caller's section (exchange, static COHSEX, W∞ − V).
    label = "Sigma static" if static else "Sigma tau"
    record_stage_price(f"{label} {keys[0]}, {len(windows)} row pass(es)", price["new"],
                       section=None if static else "sigma.tau_sweep")
    if jax.process_index() == 0:
        print(f"{label} stream {''.join('CT'[f] for f in (int(lefts != (0,)), int(rights != (0,))))}: "
              f"{len(windows)} row pass(es) of {R} local rows ({local_rows} per rank)", flush=True)
    partial_spec = NamedSharding(mesh_xy, P(None, ('x', 'y')))
    w_spec = P(None, 'x', None, 'y', None)

    def spatial(xn, yr, xr, yn, energies, weight, reference, time, interactions, loads):
        # xn, xr: the left band-complete ψ rows and projection rows; yr, yn: the
        # right Green operand and projection operand; ``loads`` the placed tables.
        if selectors is not None:
            bracketed = bracket_selectors(weight, selectors)
        hole = int(interactions.hole)
        g_load, w_load = loads[0], loads[1][hole]
        zero = jax.lax.with_sharding_constraint(
            jnp.zeros((1, px * py * n_parent, nb_sig, nb_sig), jnp.complex128), partial_spec)

        def contract(acc, green, W, Wt, left_p, g_pass, w_pass, rows_live):
            # ``rows_live``: a padded window's live rows [lo, hi), skipped outside.
            sigma = kconv[hole](green.G, green.transpose, W, Wt,
                                conj_partner=green.conj_partner, load=g_pass, w_load=w_pass,
                                live=rows_live)
            return project.accumulate((jnp.conj(left_p), yn), sigma, acc=acc)

        def one_pass(acc, W, Wt, rows, left_p, g_pass, w_pass, rows_live=None):
            if selectors is None:
                green = build_G_tau(rows, None, energies, 1j*time, e_ref=reference,
                                    band_weight=weight, layout='axis', gemm=gemm,
                                    k_unfold_plan=plans[0], trim_zero_bands=True, unfold=False,
                                    real_weights=False, right=yr)
                return contract(acc, green, W, Wt, left_p, g_pass, w_pass, rows_live)
            out = []
            for b, (sel, band_range, live) in enumerate(zip(*bracketed)):
                if b:
                    # One bracket's Green at a time.
                    prev, rows = jax.lax.optimization_barrier((tuple(out), rows))
                    out = list(prev)

                def add(acc, sel=sel, band_range=band_range, rows=rows):
                    green = build_G_tau(rows, None, energies, 1j*time, e_ref=reference,
                                        band_weight=sel, layout='axis', gemm=gemm,
                                        k_unfold_plan=plans[0], band_range=band_range,
                                        trim_zero_bands=True, unfold=False, real_weights=False,
                                        right=yr)
                    return contract(acc, green, W, Wt, left_p, g_pass, w_pass, rows_live)
                out.append(jax.lax.cond(live, add, lambda a: a, acc[b]))
            return tuple(out)
        accs = zero if selectors is None else (zero,) * len(selectors)
        W, Wt = interactions.W, interactions.partner
        if whole:
            accs = one_pass(accs, W, Wt, window_green_rows(xn, mesh_xy), xr, g_load, w_load)
        else:
            def step(s, lo, hi, acc):
                # The pass's window: W's and the ψ rows, the projection rows zeroed
                # outside the live rows, and the tables cut there.
                return one_pass(acc, *(window_rows(w, mesh_xy, s, R, axis=1, spec=w_spec)
                                       for w in (W, Wt)),
                                window_green_rows(xn, mesh_xy, s, R),
                                window_rows(xr, mesh_xy, s, R, axis=3, live=(lo, hi)),
                                window_load(g_load, mesh_xy, s, lo, hi, R, ns),
                                window_load(w_load, mesh_xy, s, lo, hi, R, n_a),
                                jnp.stack([lo, hi]).astype(jnp.int32))
            accs = scan_passes(windows, step, accs)
        if selectors is None:
            return finish(accs)
        # Every bracket's partial in one reduce-scatter (stacked on the channel axis).
        return jax.lax.with_sharding_constraint(
            project.finish(jnp.concatenate(accs, axis=0), stacked=True),
            NamedSharding(mesh_xy, P(None, None, 'x', 'y')))
    spatial.price = price
    node = SimpleNamespace(spatial=spatial, loads=loads, key=key + (tuple(windows),), plans=plans)
    while len(_SECTOR_NODES) >= 16:
        _SECTOR_NODES.pop(next(iter(_SECTOR_NODES)))
    # The plans and tables ride along so their ids in the key cannot be reused.
    _SECTOR_NODES[key] = ((plans, w_tables), node)
    return node


@lru_cache(maxsize=None)
def _place_left(mesh_xy):
    """``subtile_stream.green_rows`` (μ-major) and ``projection_complete``'s left operand."""
    from gw.subtile_stream import green_rows

    @jax.jit
    def place(xn, xr):
        rows = jax.lax.with_sharding_constraint(xn, NamedSharding(mesh_xy, P(None, None, 'x', None)))
        return (green_rows(rows, mesh_xy),
                jax.lax.with_sharding_constraint(xr, NamedSharding(mesh_xy, P(None, None, None, 'x'))))
    return place


@lru_cache(maxsize=None)
def _place_right(mesh_xy):
    """``band_complete``'s columns as the Green's right operand, and ``projection_complete``'s right."""
    from gw.greens_function_kernel import green_right_operand

    @jax.jit
    def place(yr, yn):
        cols = jax.lax.with_sharding_constraint(yr, NamedSharding(mesh_xy, P(None, None, None, 'y')))
        right_p = jax.lax.with_sharding_constraint(yn, NamedSharding(mesh_xy, P(None, None, 'y', None)))
        return green_right_operand(cols), right_p
    return place


def sector_left_operands(family, band_axis, mesh_xy):
    """A node's left operands ``(xn, xr)``, placed once per Σ call: band-complete ψ rows,
    μ-major (``subtile_stream.green_rows``), and the projector's left face, padded to ``band_axis``."""
    xn, _, xr, _, _, _ = parent_sigma_operands(family)
    return _place_left(mesh_xy)(xn, pad_to_axis(xr, band_axis, axis=1))


def sector_right_operands(family, band_axis, mesh_xy):
    """A node's right operands ``(yr, yn)``, placed once per Σ call: the Green's right
    operand (conj ψ_nmu with every band, merged) and the projector's right face."""
    _, yr, _, yn, _, _ = parent_sigma_operands(family)
    return _place_right(mesh_xy)(yr, pad_to_axis(yn, band_axis, axis=3))


def sector_tau_factory(left, right, keys, meta, mesh_xy, brackets=None):
    """Bind Gamma_A G_AB(t) Gamma_B to the window executor: the class's :func:`sector_node`
    reading W(t)'s particle and hole branches.  The caller places the left operands
    band-complete once per Σ call (``ppm_tau_kernel.sigma_subtile_operands``); the right
    ones are placed here.  ``brackets``: the Σ call's band brackets (the executor's own)."""
    def factory(synthesis, band_axis):
        w_tables = tuple(synthesis.w_tables)
        node = sector_node(left, right, keys, meta, mesh_xy, w_tables, band_axis,
                           brackets=brackets)
        right_g, right_p = sector_right_operands(right, band_axis, mesh_xy)
        return SynthesisTau(node.spatial, synthesis, right_g, right_p, synthesis.native,
                            f'sigma.sector.tau.{keys[0]}', meta, node.key, (*node.plans, *w_tables),
                            kconv_tables=node.loads)
    return factory


def fused_mixed_tau_factory(families, headers, bases, meta, mesh_xy):
    """Bind the mixed pair CT + TC to one window executor.

    CT's W pair is synthesized once per τ node; TC's pair is its transposed pair,
    ``W_TC = partner_CTᵀ`` and ``partner_TC = W_CTᵀ`` (one X↔Y exchange per node in
    place of a second synthesis); both classes' nodes run in the same loop trip on
    the same rule and their Σ(τ) are summed, so the pair costs one window's fixed
    costs and one synthesis.  Round-off against two sweeps: the transposed GEMM
    sums in another order (claim 3286: the identity holds to 1.9e-15).  The caller
    places the charge family's left operands (``_integrate_sigma_batches``); the
    current family's left operands and both right operands ride the window's
    resident arguments, packed as pytrees.
    """
    c, t = families
    syms = tuple(f.green_parent.plan.sym for f in families)
    tables_tc = _w_tables((headers['CT_T'], headers['CT_C']), (bases[1], bases[0]),
                          (syms[1], syms[0]), mesh_xy)[1:]
    keys_ct = tuple((0, B) for B in range(1, 4))
    keys_tc = tuple((A, 0) for A in range(1, 4))
    perm = (0, 3, 4, 1, 2)
    spec = NamedSharding(mesh_xy, P(None, 'x', None, 'y', None))

    def factory(synthesis, band_axis):
        tables_ct = tuple(synthesis.w_tables)
        node_ct = sector_node(c, t, keys_ct, meta, mesh_xy, tables_ct, band_axis)
        node_tc = sector_node(t, c, keys_tc, meta, mesh_xy, tuple(tables_tc), band_axis)
        yr_t, yn_t = sector_right_operands(t, band_axis, mesh_xy)
        xn_t, xr_t = sector_left_operands(t, band_axis, mesh_xy)
        yr_c, yn_c = sector_right_operands(c, band_axis, mesh_xy)

        def spatial(xn, yr, xr, yn, energies, weight, reference, time, interactions, loads):
            yr_t, xn_t, yr_c = yr
            yn_t, xr_t, yn_c = yn
            sigma = node_ct.spatial(xn, yr_t, xr, yn_t, energies, weight, reference, time,
                                    interactions, loads[0])
            flip = lambda w: jax.lax.with_sharding_constraint(jnp.transpose(w, perm), spec)
            mirrored = ParentW(flip(interactions.partner), flip(interactions.W), interactions.hole)
            return sigma + node_tc.spatial(xn_t, yr_c, xr_t, yn_c, energies, weight, reference, time,
                                           mirrored, loads[1])
        p1, p2 = node_ct.spatial.price, node_tc.spatial.price
        spatial.price = dict(d=p1['d'], ns=p1['ns'], passes=p1['passes'] + p2['passes'],
                             new=p1['new'] + p2['new'] + 2 * synthesis.tile_bytes)
        return SynthesisTau(spatial, synthesis, (yr_t, xn_t, yr_c), (yn_t, xr_t, yn_c), synthesis.native,
                            'sigma.sector.tau.(0, 1)+(1, 0)', meta, (node_ct.key, node_tc.key, 'fused'),
                            (*node_ct.plans, *node_tc.plans, *tables_ct, *tables_tc),
                            kconv_tables=(node_ct.loads, node_tc.loads))
    return factory


def sector_synthesis(readers, headers, bases, syms, layout, frequencies, meta, mesh_xy,
                     *, weights_fn=None, stage='sigma', linalg=None):
    """Keep the endpoint factors on the irreducible q and form W(t) there, per tau.

    The store is read once at setup: the factors on the store's own parent
    rows (``nq_irr``), never unfolded, placed as the scalar model's schedule
    (:func:`gw.mpa.sigma._shared_pole_memory_schedule`) places its own, with
    this sector's tile extents: replicated pole columns when they fit, else
    whole parents per rank on a ``linalg = local`` deck (``linalg``, resolved)
    when those fit, else both faces.  Each tau builds ``W = B_A d(t) B_B^dagger``
    and its partner on those rows through the one W(τ) owner
    (:func:`_w_program`), and the Σ kconv unfolds W on its load through the
    class's W tables (:func:`_w_tables`).  No full-q W, full-q W factor,
    unfolded pole table or full-grid W_R is held.  ``syms`` are the endpoint
    families' symmetry maps (a current endpoint's Cartesian action; a charge
    endpoint reads none).  ``weights_fn`` is d: the causal d(t) by default,
    the omega = 0 coefficient for :func:`sector_static_wc`, whose ledger stages
    ``stage`` prefixes.  The occupied windows read W_-(q) = partner(-q)
    through the hole tables; d is never conjugated.  ``layout`` is kept for
    the caller's census.
    """
    from file_io.shared_pole_store import face_width, read_shared_pole_faces
    from .sigma import _shared_pole_memory_schedule, _shared_pole_weights
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
    tile=16*nq*m*nc*n*nt//mesh_xy.size
    if not kmax:
        zero=_zeros(mesh_xy,shape,P(None,'x',None,'y',None))
        synthesis=WSynthesis(lambda _ref,_time,hole:ParentW(zero(),zero(),hole),
                          lambda _space,_indices,_bounds:(),lambda:(),lambda _result=None:None,0,
                          ('zero',mesh_xy,shape),ordered=True,tile_bytes=tile)
        synthesis.w_tables=tuple(w_tables)
        return synthesis
    # The store reader pads physical Kmax for both endpoint face shardings.
    # Keep that carrier through the GEMM; K and the interval bounds remain
    # physical, so the padded pole columns have identically zero weight.
    kcarrier=face_width(mesh_xy,kmax)
    same=readers[0] is readers[1] and headers[0] is headers[1]
    schedule=_shared_pole_memory_schedule(meta,left,mesh_xy=mesh_xy,stage=f'{stage}.sector.{tag}',
        linalg=linalg,extents=(m*nc,n*nt),factors=1 if same else 2)
    route=schedule['factor_layout']
    setup=f'{stage}.sector.resident.{tag}'
    # Resident: the placed factors and the parent poles; workspace: the
    # route's synthesis.  The W pair (2 tiles) is the window executable's,
    # priced there with the Green.
    row=schedule['capacity_receipt']
    capacity.reserve(setup,resident_bytes_per_rank=row['resident_bytes_per_rank'],
        workspace_bytes_per_rank=row['workspace_bytes_per_rank'],concurrent_with=ambient)
    capacity.live_stages=(*ambient,setup)
    local=route=='local'
    try:
        # Whole parents per rank read every factor in the x orientation;
        # the face and axis routes read the right factor in the y orientation.
        rhs_axis='x' if local else 'y'
        if same and not local:
            lhs=read_shared_pole_faces(readers[0],(0,nq),meta=meta,header=left,basis=bases[0])
            b_x,b_y,poles=lhs[0],lhs[1],lhs[2]
        else:
            lhs=read_shared_pole_faces(readers[0],(0,nq),meta=meta,header=left,
                                       basis=bases[0],orientations=('x',))
            b_x,b_y,poles=lhs[0],None,lhs[2]
            if not same:
                rhs=read_shared_pole_faces(readers[1],(0,nq),meta=meta,header=right,
                                           basis=bases[1],orientations=(rhs_axis,))
                if not bool(jnp.all(lhs[2]==rhs[2])):
                    raise ValueError('GATE shared_pole_sector_census: unequal pole values')
                b_y=rhs[0] if local else rhs[1]
                del rhs
        del lhs
        if local:
            # Components merged with their centroids, one copy per factor in
            # distrib_la's batch layout, placed once per Σ call.
            from distrib_la import batch_layout
            merge=lambda a:_merge_components(mesh_xy,tuple(a.shape))(a)
            b_x=batch_layout(merge(b_x),mesh_xy)
            b_y=None if b_y is None else batch_layout(merge(b_y),mesh_xy)
            poles=batch_layout(poles,mesh_xy)
        elif route=='axis':
            from .sigma import shared_pole_right_operand
            b_x=_placer(mesh_xy,P(None,'x',None,None))(b_x)
            # The right GEMM operand, formed once per Σ call: no τ node copies it.
            b_y=jax.jit(shared_pole_right_operand,out_shardings=NamedSharding(mesh_xy,P(None,None,'y')))(
                _placer(mesh_xy,P(None,'y',None,None))(b_y))
        jax.block_until_ready((b_x,b_y,poles))
        kernel=_w_program(mesh_xy,route,same,m,nc,n,nt,nq,kcarrier,weights_fn or _shared_pole_weights)
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
    synthesis=WSynthesis(w_kernel,window_operands,lambda:(b_x,b_y,poles),
                      close,0,('w-parent',mesh_xy,tuple(left['grid']),nq,m,nc,n,nt,kcarrier,id(w_plan),
                               route,same),ordered=True,tile_bytes=tile)
    synthesis.w_tables=tuple(w_tables)
    synthesis.route=route
    return synthesis


@lru_cache(maxsize=None)
def _merge_components(mesh_xy, shape):
    """``(nq, m, nc, K)`` at ``P(None,'x',None,'y')`` -> ``(nq, m*nc, K)`` at ``P(None,'x','y')``."""
    nq, m, nc, k = shape
    return jax.jit(lambda a: a.reshape(nq, m * nc, k),
                   out_shardings=NamedSharding(mesh_xy, P(None, 'x', 'y')))


@lru_cache(maxsize=None)
def _set_q_rows(mesh_xy, spec=P(None, 'x', 'y')):
    """Write a q panel into a full-q operand in place (q is unsharded)."""
    return jax.jit(lambda full, part, lo: jax.lax.dynamic_update_slice_in_dim(full, part, lo, axis=0),
                   donate_argnums=0, out_shardings=NamedSharding(mesh_xy, spec))


def instantaneous_sector_sigma(handle, families, bases, meta, mesh_xy, *,
                               occupation_state, return_components=False):
    """Exchange-like equal-time contraction of W_infinity-V, exactly once.

    The constant is read and packed on its irreducible q (in parent-q panels
    when raw + packed exceed one tile, ``runtime.tiles``); each endpoint class is its
    parent pair ``(W, conj W)``, read with the occupied Green by the class's
    :func:`sector_node` at τ = 0 on one branch (``gw.photon_sigma.contract_lorentz_blocks``).
    No full-q class operand is formed.
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
    # packed operator: the most q whose raw + packed (with their workspace,
    # 4x) fit one tile (runtime.tiles, from the shapes, never the budget;
    # Fe 20^3/P36: 97.87 GB/rank in all); a deck under one tile reads it once.
    from runtime.tiles import tile_units
    span=tile_units(4*amount/nq,nq)
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
        # The sector node's GEMMs are local XLA programs: the compiled figure is the peak.
        _admit_compiled(kernel,args,meta,f'sigma.sector.constant.{key}',resident=amount)
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
                         on_shell=None, linalg=None, **options):
    """Integrate CC, TT and the mixed pair CT + TC (one fused window) on their own pole sets.

    ``options`` is the common MPA/shared-pole quadrature contract; its live
    occupation state and fixed-rule sessions remain owned by the caller.
    ``linalg`` (the deck's resolved dense layout) places each sector's W(τ)
    synthesis as it places the scalar model's (:func:`sector_synthesis`).
    The scalar charge entry is unchanged. No model is kept across SC maps.

    Band brackets (``options['band_brackets']``, the band-extrapolation plan)
    split the CC class's Green band sum only (:func:`sector_node`): the result
    is CC's band-count cube (``gw.ppm_sigma.BandCountCube``), and TT, CT, TC
    and the W∞ − V constant are its ``term``, the same at every count.  The
    fit's differences then see CC alone and the extrapolated Σ carries the
    other classes once, at their sum to N.  The β = 3 tail law is the charge
    vertex's; the current classes are c⁻² of CC and their tails smaller still
    (docs/theory/band-extrapolation.md#four-current).
    """
    from file_io.shared_pole_store import (ResidentSectorModel, open_shared_pole_model,
                                           validate_shared_pole_sector_manifest)
    from gw.ppm_sigma import BandCountCube
    from .sigma import compute_sigma_c_mpa_omega_grid
    if handle.get('representation')!='sector-ordered-ph':
        raise ValueError('GATE shared_pole_sectors: missing ordered sector handle')
    if families[1] is None or len(bases)!=2:
        raise ValueError('GATE shared_pole_sectors: both endpoint families are required')
    brackets=options.get('band_brackets')
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
    total=counts=None
    currents=[None,None]
    for names,endpoints in ((('CC','CC'),(0,0)),(('TT','TT'),(1,1)),(('CT_C','CT_T'),(0,1))):
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
                    families[a].layout,freq,meta,mesh_xy,linalg=linalg)
                bound.append(builder)
                stack.callback(builder.close)
                return builder
            # Only CC's band sum is bracketed (the extrapolated class).
            charge=names==('CC','CC')
            # The mixed pair CT + TC runs as one fused window (fused_mixed_tau_factory).
            context=dict(schedule=lambda _header:dict(route='sector-panels'),
                synthesis=synthesis,rule_census=rule_census,
                tau_kernel=(fused_mixed_tau_factory(families,headers,bases,meta,mesh_xy) if a!=b else
                            sector_tau_factory(families[a],families[b],keys,meta,mesh_xy,
                                               brackets=brackets if charge else None)))
            opts=dict(options)
            if not charge:
                opts.pop('band_brackets',None)
                opts.pop('band_counts',None)
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
            if isinstance(value.sigma_c_kij,BandCountCube):
                counts=value
            else:
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
    if counts is not None:
        # TT, CT, TC and W∞ − V enter every CC count alike.
        result=replace(counts,sigma_c_kij=replace(counts.sigma_c_kij,term=result.sigma_c_kij))
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
    Sigma kconv call reads.  No ``W_inf - V`` term enters: the Ward contact is TT-only and V is
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
