"""LORRAX static W -> BerkeleyGW 4.0 ``bsemat.h5`` (TDA or extended); see docs/theory/bgw-bsemat-export.md.

The outline, in the order it runs. Once per restart (``_export_context``):

1. the restart bundle at full q (``bse_loading``): ψ at the centroids, V(q),
   W(q) and the head-less Γ tiles, all canonical centroid order;
2. BerkeleyGW's head momentum of every (k, k′) pair (``fullbz`` + ``sortbyq``,
   ties included) and the ζ column at it, unfolded from the q wedge;
3. the file-only split E3–E5 (``bse.bsemat_algebra``): one batched eigh of V per
   wedge q, the solve carried to every head key as ζ is, E3–E5 on each key's
   own V/W faces with its gates (``head_split_keys``);
4. the gauge map of ψ into the coarse WFN's coefficient gauge.

Then once per file written (``_write_output``; ``run_bsemat_exports`` writes
several from one load): the q → 0 body, ψ in the file's gauge and band order,
the optional downfold, and one k′ row of the file at a time: E7 against
W(k − k′) for all k, then the exchange row, each written as one hyperslab per
matrix through SlabIO. Every stage is a ``common.timing`` row
(``SHARED_STAGES``, ``OUTPUT_STAGES``).

Staging: W and V stay resident at ``P('x','y',None)``. A row materialises
C (n_k, n²·μ/√P), W permuted to the row's q order, and the (n_k, n², n²) outputs,
so device memory is two W copies plus O(n_k n² μ/√P).
"""
from __future__ import annotations

import os
import time
from functools import partial

import h5py
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

import common.timing as timing
from . import bsemat_algebra as A
from .exchange_path import exchange_tile_momentum
from common.log import get_logger

log = get_logger(__name__)

EIGHT_PI = 8.0 * np.pi
#: BerkeleyGW ``nrtype.f90``: ``ncell`` (fullbz's Wigner–Seitz scan) and TOL_Small.
BGW_NCELL = 3
BGW_TOL_SMALL = 1.0e-6
#: Relative band-energy tolerance between the restart and the coarse WFN (Ry).
ENERGY_MATCH_TOL_RY = 2.0e-3


def _log(msg):
    log.info(msg)


# ---------------------------------------------------------------------------
# 2. BerkeleyGW's head momentum per pair
# ---------------------------------------------------------------------------

def _bgw_length(f, bdot):
    """``DOT_PRODUCT(f, MATMUL(bdot, f))`` in BerkeleyGW's sequential order (no FMA)."""
    m = [bdot[i][0] * f[0] + bdot[i][1] * f[1] + bdot[i][2] * f[2] for i in range(3)]
    return f[0] * m[0] + f[1] * m[1] + f[2] * m[2]


def _k_range(x):
    """``Common/misc.f90`` ``k_range``: components into [−TOL_Small, 1 − TOL_Small)."""
    t = [float(v) for v in x]
    for i in range(3):
        while t[i] < -BGW_TOL_SMALL:
            t[i] += 1.0
        while t[i] >= 1.0 - BGW_TOL_SMALL:
            t[i] -= 1.0
    return t


def ws_bgw(t, bdot):
    """``fullbz(wigner_seitz=.true.)`` on one [0,1) point: the first strict minimum of
    ``|t − n|²`` over n ∈ [−ncell+1, ncell]³ in (i1, i2, i3) order, with BerkeleyGW's
    own arithmetic (it decides the zone-boundary ties)."""
    bd = [[float(bdot[i][j]) for j in range(3)] for i in range(3)]
    best, lmin = None, 1.0e10
    rng = range(-BGW_NCELL + 1, BGW_NCELL + 1)
    for i1 in rng:
        for i2 in rng:
            for i3 in rng:
                f = [t[0] - i1, t[1] - i2, t[2] - i3]
                L = _bgw_length(f, bd)
                if L < lmin:
                    lmin, best = L, f
    return np.asarray(best)


def bgw_kernel_q_list(q_wedge, mtrx, bdot):
    """BerkeleyGW's kernel q list ``qg%f``: ``fullbz`` of ``[0] + q_wedge`` over the WFN ops.

    ``q_wedge`` (nq, 3) the epsmat q points (crystal) or a stand-in wedge;
    ``mtrx`` (ntran, 3, 3) as h5py reads ``mf_header/symmetry/mtrx`` (so the
    Fortran ``mtrx(:,:,it)`` is ``mtrx[it].T``). Rotation, ``k_range``,
    first-occurrence dedupe (``TOL_Small``) and the Wigner–Seitz fold follow
    ``Common/fullbz.f90`` line by line.
    """
    r = [np.zeros(3)] + [np.asarray(q, dtype=np.float64) for q in q_wedge
                         if np.linalg.norm(q) > BGW_TOL_SMALL]
    fk = []
    for q in r:
        for it in range(len(mtrx)):
            M = np.asarray(mtrx[it], dtype=np.float64).T
            t = _k_range([M[i, 0] * q[0] + M[i, 1] * q[1] + M[i, 2] * q[2] for i in range(3)])
            if any(all(abs(t[i] - g[i]) < BGW_TOL_SMALL for i in range(3)) for g in fk):
                continue
            fk.append(t)
    return np.array([ws_bgw(t, bdot) for t in fk])


def pair_head_momenta(kpts, bdot, q_list):
    """Per BerkeleyGW pair (ik, ikp): head momentum K (crystal) and |q| (bohr⁻¹).

    ``sortbyq.f90``: fq = σ(k_ik − k_ikp), σ = −1 when ik > ikp, matched to the
    first ``q_list`` entry equal mod G; the pair's G₁ plane wave sits at σ·q.
    Returns ``K (nk, nk, 3)`` indexed [ik, ikp] and ``qlen (nk, nk)``.
    """
    nk = len(kpts)
    K = np.empty((nk, nk, 3))
    qlen = np.empty((nk, nk))
    bd = np.asarray(bdot, dtype=np.float64)
    for i in range(nk):
        for j in range(nk):
            s = 1.0 if i <= j else -1.0
            d = s * (kpts[i] - kpts[j])[None, :] - q_list
            d -= np.rint(d)
            hit = np.flatnonzero(np.einsum("qi,ij,qj->q", d, bd, d) <= 1.0e-12)
            if hit.size == 0:
                raise ValueError(f"GATE bsemat_q_list: pair ({i},{j}) has no q in the "
                                 "kernel q list; the wedge does not cover the grid.")
            f = q_list[hit[0]]
            K[i, j] = s * f
            qlen[i, j] = np.sqrt(max(float(f @ bd @ f), 0.0))
    return K, qlen


def grid_index(frac, grid):
    """Flat fft-order index of crystal points on the uniform grid (refuses off-grid)."""
    n = np.asarray(frac, dtype=np.float64) * np.asarray(grid)[None, :]
    ni = np.rint(n).astype(np.int64)
    if not np.allclose(n, ni, atol=1e-6):
        raise ValueError("GATE bsemat_k_on_grid: a WFN_co k point is off the restart grid")
    ni = np.mod(ni, grid)
    return (ni[:, 0] * grid[1] + ni[:, 1]) * grid[2] + ni[:, 2]


def finite_q_legs(kpts, grid, exciton_Q):
    """``(qflag, c_rows, v_rows)``: where the two legs of each file row live at exciton momentum Q.

    File row j is BerkeleyGW's ik, the WFN_co point k_j. Its conduction leg is
    at k_j and its valence leg at k_j − Q: BerkeleyGW's ``exciton_Q_shift`` is
    −Q (``BSE/input_kernel.f90``: valence at ``kg%f + finiteq``). The pair set
    is LORRAX's |v k, c k+Q⟩ (``bse.exciton_bands``), labelled by the electron.
    The hole is always on the restart grid: ``v_rows`` is its flat index.
    Q = 0 gives ``qflag`` 1, and Q on the grid gives 2 (``c_rows`` the flat
    index of k_j). Q off the grid gives 0: WFN_co is the restart grid shifted
    by Q, and ``c_rows`` is None (the conduction leg comes from the htransform at k_j).
    docs/theory/bgw-bsemat-export.md#finite-q.
    """
    Q = np.asarray(exciton_Q, dtype=np.float64).reshape(3)
    g = np.asarray(grid, dtype=np.float64)
    try:
        v_rows = grid_index(np.asarray(kpts, dtype=np.float64) - Q[None, :], grid)
    except ValueError:
        raise ValueError(f"GATE bsemat_finite_q_grid: a WFN_co point minus exciton_Q {Q.tolist()} is "
                         "off the restart grid; the hole leg must be on it. fix: for Q off the grid, "
                         "WFN_co is the restart grid shifted by +Q (the electron leg) and the "
                         "restart grid's own WFN is --wfnq-co.") from None
    if len(set(v_rows.tolist())) != len(v_rows):
        raise ValueError("GATE bsemat_finite_q_grid: WFN_co minus Q does not cover the restart grid once")
    if np.allclose(Q - np.rint(Q), 0.0, atol=1e-9):
        qflag = 1
    elif np.allclose(Q * g, np.rint(Q * g), atol=1e-6):
        qflag = 2
    else:
        qflag = 0
    return qflag, (None if qflag == 0 else grid_index(kpts, grid)), v_rows


def require_offgrid_zeta(restart, nk, exciton_Q):
    """``GATE bsemat_finite_q_offgrid``: the off-grid route (qflag 0) runs only on a full-zone ζ.

    That route is implemented but not validated against ``kernel.x``
    (docs/theory/bgw-bsemat-export.md#finite-q). Its exchange is the per-Q ζ refit,
    whose reader labels an IBZ-stored ζ by the first WFN k-points, and gw_init stores
    the IBZ by default (``write_ibz_only_charge``). Until the route is certified it
    refuses unless the restart's ``zeta_q.h5`` holds every q of the grid
    (``zeta_loader.ZetaLoader.q_layout == 'full_bz'``). Returns the stored q count.
    """
    from zeta_loader import ZetaLoader
    path = os.path.join(os.path.dirname(os.path.abspath(restart)), "zeta_q.h5")
    Q = np.asarray(exciton_Q, dtype=np.float64).reshape(3).tolist()
    why = (f"exciton_Q {Q} is off the restart grid (qflag 0), a route not yet validated against "
           "kernel.x; it runs only on a full-zone zeta")
    fix = ("fix: put Q on the restart grid (every N_i Q_i an integer: qflag 2), or rerun GW with "
           "full-zone zeta storage")
    if not os.path.exists(path):
        raise ValueError(f"GATE bsemat_finite_q_offgrid: {why}, and {path} is absent. {fix}")
    with ZetaLoader(path) as zl:
        layout, nq = str(zl.q_layout), int(zl.n_q_on_disk)
    if layout != "full_bz" or nq != int(nk):
        raise ValueError(f"GATE bsemat_finite_q_offgrid: {why}, and {path} stores {nq} of the grid's "
                         f"{int(nk)} q (q_layout {layout!r}; gw_init writes the IBZ by default). {fix}")
    _log(f"[bsemat] qflag 0 route on a full-zone zeta ({nq} q): implemented, NOT validated against "
         "kernel.x (docs/theory/bgw-bsemat-export.md#finite-q)")
    return nq


def htransform_window(input_file, kpts, *, n_val, n_cond, n_rmu, mesh_xy):
    """ψ and ε of the BSE window at explicit points, from the deck's htransform.

    Built exactly as ``bse.exciton_bands`` builds its conduction caches. The
    fit is ``fh_interp.initialize_wfns`` on the deck's band window, the BSE
    bands are returned from inside it (``bse_setup.resolve_conduction_window``),
    and ``bse_setup.compute_wfns_fi`` runs with ``q_list = kpts`` (crystal). One
    f(H) covers every band the deck loads. Returns host ``psi`` with shape
    (nk, n_val + n_cond, ns, n_rmu), bands ascending, ``enk`` with shape
    (nk, n_val + n_cond) in Ry, and the receipts. Only the conduction block is
    consumed; the valence block enters the gauge fit
    (docs/how-to/htransform-and-exciton-bands.md §7).
    """
    from gw.gw_config import (linalg_resolution, read_lorrax_input, resolve_distrib_la_batched_route,
                              resolve_eigh_backend)
    from bandstructure import fh_interp as ht
    from bandstructure.bse_setup import compute_wfns_fi, resolve_conduction_window
    from common.band_degeneracy import DEFAULT_MODE, DEGENERACY_TOL_RY
    params = read_lorrax_input(input_file)
    base = os.path.dirname(os.path.abspath(input_file))
    absp = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
    params["centroids_file"] = absp(str(params["centroids_file"]))
    route = resolve_distrib_la_batched_route(params)
    wfn, sym = ht.setup_wfn_and_sym(absp(str(params["wfn_file"])), mesh_xy=mesh_xy)
    receipts = []
    wfn, sym, meta, _, basis, enk_sigma = ht.initialize_wfns(
        input_file, params, _log, mesh_xy=mesh_xy, centroid_subset_idx=None, wfn_sym=(wfn, sym),
        rank_record_fn=receipts.append, distrib_la_batched_route=route)
    nval_in = int(params["nval"])
    if nval_in < int(n_val):
        raise ValueError(f"GATE bsemat_finite_q_window: the deck's nval {nval_in} < n_val {n_val}; the "
                         "htransform window must hold the BSE valence bands")
    _, b_max, _ = resolve_conduction_window(
        enk_sigma, nb_window=int(basis.ctilde.shape[1]), nval_in=nval_in, n_cond=int(n_cond),
        tol_ry=DEGENERACY_TOL_RY, mode=DEFAULT_MODE, input_file=input_file, log_fn=_log)
    bundle = compute_wfns_fi(
        ctilde=basis.ctilde, B_at_mu=basis.basis_at_nodes, enk_sigma=enk_sigma,
        kgrid_co=(int(meta.nkx), int(meta.nky), int(meta.nkz)),
        band_window_fi=(nval_in - int(n_val), b_max), mesh_xy=mesh_xy,
        q_list=np.asarray(kpts, dtype=np.float64),
        batch_size=int(params.get("wfn_fi_q_chunk", 0)), eigh_backend=resolve_eigh_backend(params),
        use_low_mem_eigh=linalg_resolution(params).layout == "distributed", log_fn=_log,
        distrib_la_batched_route=route, htransform_quality_record_fn=receipts.append)
    rep = jax.jit(lambda a: a, out_shardings=NamedSharding(mesh_xy, P()))
    psi, enk = np.asarray(rep(bundle.psi_rmu_Y)), np.asarray(rep(bundle.enk_full))
    if psi.shape[-1] != int(n_rmu):
        raise ValueError(f"GATE bsemat_finite_q_basis: the htransform fitted ψ at {psi.shape[-1]} "
                         f"centroids, the restart stores μ = {n_rmu} (a downfolded bundle needs its "
                         "parent's table)")
    wfn.close()
    return psi, enk, receipts


def exchange_tile(Q, qflag, *, data, input_file, restart, grid, n_rmu, n_rmu_pad, mesh_xy):
    """The exchange tile the file contracts (μ, ν) at ``P('x','y')``, and its receipt.

    At Q = 0 this is the head-less Γ tile: kernel.x drops G = 0 (E1d). At
    finite Q it is V at tile momentum :func:`exchange_tile_momentum` (Q) =
    wrap(+Q) with every G kept: BerkeleyGW's ``energy_loss`` exchange, whose
    plane wave e^{i(Q_s+G)·r} (``kernel_main.f90``: ``vcoularray(:, nf+1)`` at
    ``finiteq`` = Q_s = −Q) picks the density component of momentum Q − G
    (E12). The tile comes from ``bse.exchange_path.exchange_tiles``, the
    evaluator ``bse.exciton_bands`` solves with, using the same +Q. On the
    grid it is the stored tile (``ongrid``, exact); off the grid it is the
    per-Q ζ refit on the BSE window (``refit``, the arbitrary-Q route).
    """
    from .exchange_path import exchange_tiles
    if qflag == 1:
        return data["V_q0"], "head-less Gamma tile (G = 0 dropped, E1d)"
    rst = None
    if qflag == 2:
        route, refit = "ongrid", None
    else:
        from . import vq_interp
        zx = vq_interp.read_vq_payload(restart, os.path.join(os.path.dirname(restart), "zeta_q.h5"),
                                       mesh=mesh_xy, log_fn=_log, input_file=input_file,
                                       require_slab=False, require_full_bz_zeta=False,
                                       include_interactions=False)
        rst = vq_interp.refit_prepare(input_file, mesh_xy, zx, log_fn=_log, policy=zx.get("policy"),
                                      keep_idx=None, window_mode="bse")
        route, refit = "refit", (rst["zx_fit"], rst)
    q_tile = exchange_tile_momentum(Q)
    V_stack, _, _, _ = exchange_tiles(
        q_tile.reshape(1, 3), route=route, mesh_xy=mesh_xy,
        V_q0=data["V_q0"], V_ongrid=data.get("V_q_full"), kgrid_vq=np.asarray(grid),
        n_rmu=n_rmu, n_rmu_pad=n_rmu_pad, refit=refit, log_fn=_log)
    V = jax.jit(lambda s: s[0], out_shardings=NamedSharding(mesh_xy, P("x", "y")))(V_stack)
    if rst is not None:
        from . import vq_interp
        vq_interp.close_refit_state(rst)
    return V, (f"stored tile V[wrap(+Q) = {np.round(q_tile, 9).tolist()}], every G (energy_loss)"
               if qflag == 2 else
               f"per-Q zeta refit on the BSE window at wrap(+Q) = {np.round(q_tile, 9).tolist()}, "
               "every G (energy_loss)")


# ---------------------------------------------------------------------------
# 2b. ζ columns at the head momenta, unfolded from the wedge
# ---------------------------------------------------------------------------

def _head_parents(keys, *, tables, sym, grid, n_rmu_pad):
    """Parent ``(wedge q, ζ slot)`` of every head key, from SymMaps' star rows (host).

    A key ``(iq, G)`` sits at ``q + G = S(q_p + G_p)`` with ``S`` its star row,
    so ``G_p = S⁻¹(q + G) − q_p``. Returns the padded tables, the full and
    wedge q (BerkeleyGW-wrapped crystal) and ``need[kk] = (p, G_p)``.
    """
    from symmetry_maps import bgw_integer_q_to_fractional
    tp = tables.padded(int(n_rmu_pad))
    g3 = np.asarray(grid, dtype=np.int64).reshape(3)
    q_full = bgw_integer_q_to_fractional(np.asarray(sym.kvecs_asints, np.int64), g3)
    q_parent = np.asarray(tp.q_irr_frac, dtype=np.float64)
    if not np.allclose(q_parent, bgw_integer_q_to_fractional(
            np.asarray(sym.q_irr_kgrid_int, np.int64), g3), rtol=0.0, atol=1e-12):
        raise ValueError("GATE bsemat_head_parent: the restart's q wedge is not SymMaps' q-IBZ")
    idx = np.asarray(sym.irr_idx_q, dtype=np.int64)
    rows = np.asarray(tp.sym_idx_q, dtype=np.int64)
    S_all = np.asarray(sym.sym_mats_k, dtype=np.int64)
    if idx.shape != (q_full.shape[0],) or rows.shape != idx.shape or np.any(rows >= S_all.shape[0]):
        raise ValueError("GATE bsemat_head_parent: SymMaps star rows do not cover the full q grid")
    need = []
    for iq, G in keys:
        S = np.asarray(S_all[int(rows[iq])], dtype=np.float64)
        Gp_f = np.linalg.solve(S, q_full[iq] + np.asarray(G, float)) - q_parent[int(idx[iq])]
        Gp = np.rint(Gp_f).astype(np.int64)
        if not np.allclose(Gp_f, Gp, atol=1e-8):
            raise ValueError(f"GATE bsemat_head_parent: key {(iq, tuple(G))} has parent G {Gp_f}")
        need.append((int(idx[iq]), tuple(int(v) for v in Gp)))
    return tp, q_full, q_parent, idx, need


def _parent_columns(need, *, zeta_path, n_qi, n_rmu, n_rmu_pad):
    """``ζ_{q_p}(μ, G_p)/√Ω`` at every distinct parent slot of ``need`` (host, μ_pad each), and Ω."""
    from zeta_loader import ZetaLoader
    cols = {}
    with ZetaLoader(zeta_path) as zl:
        omega = float(zl.cell_volume)
        gcomp = np.asarray(zl.gvec_components, dtype=np.int64)
        ngk = np.asarray(zl.ngk_per_q, dtype=np.int64)
        if gcomp.shape[0] != n_qi:
            raise ValueError(f"GATE bsemat_zeta_wedge: {zeta_path} holds {gcomp.shape[0]} q, "
                             f"the W0 wedge {n_qi}")
        for p, g in sorted(set(need)):
            lut = {tuple(v): i for i, v in enumerate(gcomp[p, :, :ngk[p]].T)}
            if g not in lut:
                raise ValueError(f"GATE bsemat_head_in_sphere: wedge q {p} slot {g} "
                                 "is outside the stored ζ sphere; fix: raise the ζ cutoff.")
            c = np.zeros(int(n_rmu_pad), np.complex128)
            c[:n_rmu] = np.asarray(zl.read_zeta_G_local((p, slice(None), lut[g])))
            cols[(p, g)] = c / np.sqrt(omega)
    return cols, omega


def _unfold_to_keys(cols, need, keys, *, tp, sym, grid, n_qi, n_rmu_pad, mesh_xy):
    """Carry parent vectors to their keys by ``symmetry_maps.unfold_isdf_one_leg`` (host, (n_keys, μ_pad)).

    ``cols[(p, G_p)]`` is a ζ-like parent column: it moves as ``ζ_q(·, G_p)``
    does (centroid permutation, L and τ phases, conjugation on antiunitary
    rows). Keys of one wedge q that need different parent slots (zone-boundary
    ties) go through separate unfold layers; a parent with no key carries zeros.
    """
    from symmetry_maps import unfold_isdf_one_leg
    from common.collectives import device_put_process_local
    per_p = {p: sorted({g for (pp, g) in need if pp == p}) for p in range(n_qi)}
    n_layer = max(len(v) for v in per_p.values() if v) if need else 0
    out = np.zeros((len(keys), int(n_rmu_pad)), np.complex128)
    rep = jax.jit(lambda a: a, out_shardings=NamedSharding(mesh_xy, P()))
    for j in range(n_layer):
        src = np.zeros((n_qi, 3), np.int64)
        col = np.zeros((n_qi, int(n_rmu_pad)), np.complex128)
        for p in range(n_qi):
            gl = per_p[p]
            if gl:
                src[p] = gl[min(j, len(gl) - 1)]
                col[p] = cols[(p, tuple(int(v) for v in src[p]))]
        u = unfold_isdf_one_leg(
            device_put_process_local(col, NamedSharding(mesh_xy, P(None, "x"))),
            source_gvec_components=src.astype(np.int32), sym=sym, sym_idx=tp.sym_idx_q,
            sym_perm=tp.sym_perm, L_table=tp.L_table, q_irr_frac=tp.q_irr_frac,
            kgrid=tuple(int(g) for g in grid), mesh_xy=mesh_xy, component_action="scalar")
        u = np.asarray(rep(u))
        for kk, (p, g) in enumerate(need):
            if src[p].tolist() == list(g):
                out[kk] = u[keys[kk][0]]
    return out


def head_columns(keys, *, zeta_path, tables, sym, grid, n_rmu, n_rmu_pad, mesh_xy):
    """ζ_q(μ, G)/√Ω at each requested ``(iq, G)`` of the full grid (host, (n_keys, μ_pad)).

    Each key's parent column is ``G_p = S⁻¹(q + G) − q_p`` at its wedge q_p, read
    from ``zeta_q.h5`` and carried to q by ``symmetry_maps.unfold_isdf_one_leg``
    (preselected carrier). Keys of one wedge q that need different parent
    slots (zone-boundary ties) go through separate unfold layers.
    """
    from ffi import _services
    _services.ensure_on_path()
    tp, q_full, q_parent, _, need = _head_parents(keys, tables=tables, sym=sym, grid=grid,
                                                  n_rmu_pad=n_rmu_pad)
    cols, omega = _parent_columns(need, zeta_path=zeta_path, n_qi=q_parent.shape[0],
                                  n_rmu=n_rmu, n_rmu_pad=n_rmu_pad)
    out = _unfold_to_keys(cols, need, keys, tp=tp, sym=sym, grid=grid, n_qi=q_parent.shape[0],
                          n_rmu_pad=n_rmu_pad, mesh_xy=mesh_xy)
    return out, q_full, omega


# ---------------------------------------------------------------------------
# 3. the split per key from one eigendecomposition per wedge q
# ---------------------------------------------------------------------------

def _self_images(parents, *, q_parent, idx, tp, sym, grid):
    """Full-grid index of each wedge q in ``parents`` where its star row is the identity (host).

    There the centroid action is trivial (S = 1, τ = 0, unitary, identity
    permutation, L = 0), so the bundle's full-q V and W tiles ARE the parent
    tiles that every star member's tiles, and the one-leg ζ unfold, are built from.
    """
    S_all = np.asarray(sym.sym_mats_k, dtype=np.int64)
    n_sp = int(np.asarray(sym.sym_matrices).shape[0])
    tau = np.asarray(sym.translations, dtype=np.float64)
    rows = np.asarray(tp.sym_idx_q, dtype=np.int64)
    perm, L = np.asarray(tp.sym_perm), np.asarray(tp.L_table)
    out = {}
    for p in parents:
        iq = int(grid_index(q_parent[p][None], grid)[0])
        r = int(rows[iq])
        if not (int(idx[iq]) == p and r < n_sp and np.array_equal(S_all[r], np.eye(3, dtype=np.int64))
                and not np.any(np.abs(tau[r]) > 1e-12)
                and np.array_equal(perm[r], np.arange(perm.shape[1])) and not np.any(L[r])):
            raise ValueError(f"GATE bsemat_head_parent_self: wedge q {p} (full q {iq}) is reached by "
                             f"star row {r}, not the identity; the split's one solve per wedge q needs "
                             "the parent's own tile. fix: regenerate the restart's q-star tables "
                             "(symmetry_maps.write_tables) so each IBZ point is its own identity image.")
        out[p] = iq
    return out


def head_split_keys(keys, z_keys, V_flat, W_flat, *, zeta_path, tables, sym, grid, n_rmu,
                    n_rmu_pad, v_head_q, q_frac, head_K, bvec, rcond, mesh_xy):
    """E3–E5 at every head key from ONE eigendecomposition per wedge q (docs/theory/bgw-bsemat-export.md#split-ibz).

    A star member S q of wedge q_p carries the parent tiles by the centroid action
    (``unfold_isdf_operator``: V_Sq = M V_q M†, conjugated on antiunitary rows) and
    its head column by the one-leg action (``unfold_isdf_one_leg``, the same
    permutation and phases). So V_Sq⁺ = M V_q⁺ M† and conj(a) = conj(V⁺ z*)
    moves exactly as z does: conj(a_Sq) = unfold(conj(a_q)) at the key's own
    parent slot G_p, time reversal and zone-boundary ties included. Here:

    1. one batched ``distrib_la`` eigh over the wedge tiles that some key needs
       (the bundle's tiles at the wedge points' identity images);
    2. one E3 solve per parent slot (p, G_p), with the parent's gates;
    3. conj(a) carried to every key through head_columns' unfold layers;
    4. per key, E3's residual, E4, E5 on the key's OWN V and W tiles (O(μ²)):
       ``check_split`` as before, and s against the parent's
       (``check_rotation``) certifies the rotation where it is consumed.

    ``V_flat``/``W_flat`` (μ, ν, n_q) at ``P('x','y',None)``; ``z_keys`` (n_keys, μ_pad)
    host from :func:`head_columns`; ``v_head_q`` (n_q,) v(q + G₁); ``q_frac`` (n_q, 3) the
    bundle's full q (crystal), ``head_K`` (n_q, 3) V's head momentum per q and
    ``bvec`` (3, 3) for the shortest-q+G gate. Returns ``W11 (n_keys,)``,
    ``y_keys (n_keys, μ_pad)`` host, the per-key figures and a stage summary
    (eigh faces, parent slots, keys, wall).
    """
    import distrib_la as dla
    from ffi import _services
    _services.ensure_on_path()
    from common.collectives import device_put_process_local

    t0 = time.time()
    tp, _, q_parent, idx, need = _head_parents(keys, tables=tables, sym=sym, grid=grid,
                                               n_rmu_pad=n_rmu_pad)
    n_qi = q_parent.shape[0]
    slots = sorted({need[kk] for kk, (iq, _) in enumerate(keys) if iq != 0})
    parents = sorted({p for p, _ in slots})
    self_iq = _self_images(parents, q_parent=q_parent, idx=idx, tp=tp, sym=sym, grid=grid)
    cols, _ = _parent_columns(slots, zeta_path=zeta_path, n_qi=n_qi, n_rmu=n_rmu,
                              n_rmu_pad=n_rmu_pad)

    face = NamedSharding(mesh_xy, P("x", "y"))
    rs = NamedSharding(mesh_xy, P())
    rep = jax.jit(lambda a: a, out_shardings=rs)
    tile = jax.jit(lambda a, i: jnp.take(a, i, axis=2), out_shardings=face)
    herm = jax.jit(lambda a: 0.5 * (a + jnp.conj(a).T), out_shardings=face)
    put = lambda a: device_put_process_local(np.ascontiguousarray(a), rs)

    # 1. one eigendecomposition per wedge q, as one batched call
    plan = dla.plan("eigh", mesh_xy, backend="distributed" if jax.process_count() > 1 else "native",
                    n=n_rmu_pad)
    iqs = np.asarray([self_iq[p] for p in parents], dtype=np.int32)
    t_e, lam_all, U_all = time.time(), None, None
    if parents:                                       # a Γ-only grid has no split to make
        stack = jax.jit(lambda v, i: (lambda t: 0.5 * (t + jnp.conj(jnp.swapaxes(t, 1, 2))))(
            jnp.moveaxis(jnp.take(v, i, axis=2), 2, 0)),
            out_shardings=NamedSharding(mesh_xy, P(None, "x", "y")))(V_flat, jnp.asarray(iqs))
        lam_all, U_all = plan.batched(stack)
        lam_all = np.asarray(rep(lam_all))
        del stack
    t_eigh = time.time() - t_e
    pick = jax.jit(lambda u, i: u[i], out_shardings=face)

    # 2. one solve per parent slot, with the parent's own gates
    abar, s_par, rank_par, par_fig = {}, {}, {}, {}
    for p in parents:
        i = parents.index(p)
        U_p = pick(U_all, i)
        V_p = herm(tile(V_flat, self_iq[p]))
        W_p = tile(W_flat, self_iq[p])
        for (pp, g) in slots:
            if pp != p:
                continue
            zr = put(cols[(p, g)])
            a, rank = A.head_solve(put(lam_all[i]), U_p, zr, xp=jnp, rcond=rcond)
            sp = A.head_rows(V_p, W_p, zr, a, float(v_head_q[self_iq[p]]), rank=rank, xp=jnp)
            A.check_split(sp, where=f"wedge q {p} slot {g}")
            abar[(p, g)] = np.asarray(rep(jnp.conj(a)))
            s_par[(p, g)], rank_par[(p, g)] = sp.figures["s"], sp.figures["rank"]
            par_fig[(p, g)] = sp.figures
    del U_all
    for (p, g) in set(need) - set(slots):
        abar[(p, g)] = np.zeros(int(n_rmu_pad), np.complex128)     # q = 0 keys: no split

    # 3. a at every key, carried exactly as z
    a_keys = np.conj(_unfold_to_keys(abar, need, keys, tp=tp, sym=sym, grid=grid, n_qi=n_qi,
                                     n_rmu_pad=n_rmu_pad, mesh_xy=mesh_xy))

    # 4. E3 residual, E4, E5 on each key's own tiles
    W11 = np.zeros(len(keys))
    y_keys = np.zeros((len(keys), int(n_rmu_pad)), np.complex128)
    figs = []
    for kk, (iq, G) in enumerate(keys):
        if iq == 0:
            figs.append({"iq": 0, "G": G, "q0": True})
            continue
        Kc = (q_frac[iq] + np.asarray(G)) @ bvec
        if abs(np.linalg.norm(Kc) - np.linalg.norm(head_K[iq] @ bvec)) > 1e-8 * np.linalg.norm(Kc):
            raise ValueError(f"GATE bsemat_head_length: key {(iq, G)} is not a shortest q+G")
        par = need[kk]
        sp = A.head_rows(herm(tile(V_flat, iq)), tile(W_flat, iq), put(z_keys[kk]), put(a_keys[kk]),
                         float(v_head_q[iq]), rank=rank_par[par], xp=jnp)
        A.check_split(sp, where=f"q {iq} G {G}")
        rel = A.check_rotation(sp, s_par[par], where=f"q {iq} G {G} (wedge q {par[0]} slot {par[1]})")
        W11[kk] = sp.W11
        y_keys[kk] = np.asarray(rep(sp.y))
        figs.append({"iq": int(iq), "G": G, "qlen": float(np.linalg.norm(Kc)),
                     "v1": float(v_head_q[iq]), **sp.figures,
                     "parent": [int(par[0]), list(par[1])], "s_parent_rel": rel})
    stats = {"eigh_faces": len(parents), "eigh_extent": int(n_rmu_pad), "parent_slots": len(slots),
             "keys_split": sum(1 for iq, _ in keys if iq != 0), "wall_eigh_s": t_eigh,
             "wall_s": time.time() - t0,
             "parent_solve_residual_max": max((f["solve_residual"] for f in par_fig.values()), default=0.0)}
    _log(f"[bsemat] split: {stats['eigh_faces']} wedge-q eigh (one batched call, {t_eigh:.2f} s) "
         f"for {stats['parent_slots']} parent slots -> {stats['keys_split']} keys; "
         f"max |s/s_parent - 1| {max((f.get('s_parent_rel', 0.0) for f in figs), default=0.0):.1e}; "
         f"{stats['wall_s']:.2f} s")
    return W11, y_keys, figs, stats


# ---------------------------------------------------------------------------
# 4. gauge map into the coarse WFN's coefficient gauge
# ---------------------------------------------------------------------------

#: Bytes of one FFT-box chunk in the gauge map's ψ_file(r_μ) evaluation (``gflat_to_rmu`` scan).
GAUGE_BOX_CHUNK_BYTES = 1 << 28
#: Relative singular-value cut of the block polar factor (the parallel-transport links' value).
GAUGE_POLAR_RCOND = 1.0e-10


def gauge_map(psi, *, wfn_co, k_rows, bands_abs, centroid_frac, fft_grid, nv, enk_win, mesh_xy,
              guard=8):
    """Window unitaries ``U[k] (nb, nb)`` with ψ_LX ≈ ψ_file U, valence and conduction blocks.

    ``psi`` (nk_L, nb, ns, μ) host, LORRAX ψ at the centroids (canonical order,
    bands ascending); ``k_rows[j]`` the LORRAX k of WFN row j; ``bands_abs``
    the absolute 0-based file bands of ψ's band axis; ``enk_win`` (nk_L, nb) Ry
    the restart's energies of those bands (gated against the file's). The file states over the
    window plus ``guard`` bands each side are evaluated at the (grid-rounded)
    centroids, ψ_LX is fitted by least squares (the centroids are not a
    quadrature, dipfix), and each block's polar factor is kept. Returns
    ``(U (nk_wfn, nb, nb) host, figures)``.

    Device route, all existing LORRAX paths: ``WfnLoader.load(k='ibz')`` reads the
    file's own rows (no unfold; the collective read at P > 1) and the GW face
    transform ``gflat_to_rmu`` evaluates them at the centroid grid points,
    ``ψ_file,b(r_μ) = Σ_G c_b(G) e^{2πi(G+k)·r_μ}`` (``norm='forward'``, the
    loader's own k table). The normal equations ``(A*Aᵀ) X = A*Bᵀ`` are formed
    with μ over the whole mesh and solved batched over k; the block polar factors
    go through ``distrib_la.plan_polar_factor(...).batched`` on the batch-reshard
    route, as the parallel-transport links do. ``mesh_xy`` is the run's mesh.
    """
    import distrib_la as dla
    from wfn_loader import WfnLoader
    from common.collectives import device_put_process_local
    from common.gvec_fft_box import build_sphere_box_index
    from common.wfn_layout import band_sphere_spec
    from common.wfn_transforms import gflat_to_rmu, load_psi_gflat_padded
    from runtime.padding import padded_axis
    t0 = time.time()
    grid = tuple(int(n) for n in fft_grid)
    r_mu = np.mod(np.rint(np.asarray(centroid_frac) * np.asarray(grid)[None, :]).astype(np.int64),
                  np.asarray(grid)[None, :]).astype(np.int32)
    nb, ns, n_mu = int(psi.shape[1]), int(psi.shape[2]), len(r_mu)
    nc = nb - nv
    k_rows = np.asarray(k_rows, dtype=np.int64)
    with h5py.File(wfn_co, "r") as w:
        el = np.asarray(w["mf_header/kpoints/el"][()], dtype=np.float64)
    el = el.reshape(-1, el.shape[-2], el.shape[-1])[0]                 # (nk_wfn, mnband) Ry, spin 1
    nk_w, nb_file = el.shape
    b_lo, b_hi = max(0, int(bands_abs[0]) - guard), min(nb_file, int(bands_abs[-1]) + 1 + guard)
    nbg, w0 = b_hi - b_lo, int(bands_abs[0]) - b_lo
    de = np.max(np.abs(el[:, int(bands_abs[0]):int(bands_abs[-1]) + 1] - enk_win[k_rows]), axis=1)
    if np.any(de > 5.0e-2):
        j = int(np.flatnonzero(de > 5.0e-2)[0])
        raise ValueError(f"GATE bsemat_band_offset: WFN_co row {j} window energies differ "
                         f"from the restart's by {de[j]:.3e} Ry; want the same bands.")

    # ψ_file(r_μ): (nk_wfn, nbg_pad, ns, μ), bands over XY (zero pad bands).
    with WfnLoader(wfn_co, mesh=mesh_xy) as loader:
        if int(loader.nspinor) != ns or int(loader.nkpts) != nk_w:
            raise ValueError(f"GATE bsemat_gauge_layout: WFN_co has nspinor {int(loader.nspinor)}, "
                             f"{int(loader.nkpts)} k; ψ_LX has {ns}, the file's energies {nk_w}.")
        ngk = loader.ngk_valid(k="ibz")
        g_index = build_sphere_box_index(loader.gvecs(k="ibz"), grid, int(loader.ngkmax),
                                         ngk_valid=ngk)
        for j in range(nk_w):
            if np.unique(g_index[j, :ngk[j]]).size != int(ngk[j]):
                raise ValueError(f"GATE bsemat_gauge_box_alias: WFN_co row {j}'s G sphere aliases "
                                 f"on the FFT grid {grid}; the centroid grid must hold the sphere.")
        psi_G = load_psi_gflat_padded(loader, (b_lo, b_hi), mesh_xy=mesh_xy, bispinor=False, k="ibz")
        A_r = gflat_to_rmu(
            psi_G, device_put_process_local(g_index, NamedSharding(mesh_xy, P(None, None))), r_mu,
            mesh=mesh_xy, fft_grid=grid, kvecs_frac=loader.kvecs(k="ibz"), norm="forward",
            chunk_size=max(1, GAUGE_BOX_CHUNK_BYTES // (ns * int(np.prod(grid)) * 16)))
        del psi_G

    # Least squares ψ_LX ≈ ψ_file X per k, μ over the whole mesh (zero μ pad is inert).
    mu_w = padded_axis(n_mu, mesh_xy, name="bsemat gauge centroid axis").carrier
    s_mu = NamedSharding(mesh_xy, P(None, None, None, ("x", "y")))
    rep = NamedSharding(mesh_xy, P())
    B = np.zeros((nk_w, nb, ns, mu_w), np.complex128)
    B[..., :n_mu] = np.asarray(psi)[k_rows][..., :n_mu]
    B = device_put_process_local(B, s_mu)

    @partial(jax.jit, out_shardings=(rep, rep, rep))
    def fit(a, b):
        a = jnp.pad(a, ((0, 0), (0, 0), (0, 0), (0, mu_w - n_mu)))
        a = jax.lax.with_sharding_constraint(a, s_mu)[:, :nbg]
        gram = jax.lax.with_sharding_constraint(jnp.einsum("kbsm,kcsm->kbc", jnp.conj(a), a), rep)
        rhs = jax.lax.with_sharding_constraint(jnp.einsum("kbsm,knsm->kbn", jnp.conj(a), b), rep)
        x = jnp.linalg.solve(gram, rhs)                                   # (nk, nbg, nb)
        ssq = lambda m: jnp.sum(jnp.abs(m) ** 2, axis=tuple(range(1, m.ndim)))   # per k
        res = jnp.sqrt(ssq(b - jnp.einsum("kbn,kbsm->knsm", x, a)) / ssq(b))
        xw = x[:, w0:w0 + nb]
        cross = (jnp.sqrt(ssq(xw[:, :nv, nv:])) + jnp.sqrt(ssq(xw[:, nv:, :nv]))) / jnp.sqrt(ssq(xw))
        return xw, res, cross

    Xw, res, cross = fit(A_r, B)
    del A_r, B

    # Block polar factors, one (2·nk, n, n) stack at P(None, x, y); zero pad directions drop.
    n_blk = padded_axis(max(nv, nc), mesh_xy, name="bsemat gauge polar block",
                        spec=band_sphere_spec(), axis=1).carrier
    pol = dla.plan_polar_factor(mesh_xy, n=n_blk, backend="native", rcond=GAUGE_POLAR_RCOND,
                                batched_route=dla.ROUTE_BATCH_RESHARD)
    pad = lambda m: jnp.pad(m, ((0, 0), (0, n_blk - m.shape[1]), (0, n_blk - m.shape[2])))
    stack = jax.jit(lambda x: jnp.concatenate([pad(x[:, :nv, :nv]), pad(x[:, nv:, nv:])]),
                    out_shardings=NamedSharding(mesh_xy, P(None, "x", "y")))(Xw)
    link, sv = pol.batched(stack)
    link = np.asarray(jax.jit(lambda a: a, out_shardings=rep)(link))
    sv = np.asarray(sv)
    U = np.zeros((nk_w, nb, nb), np.complex128)
    U[:, :nv, :nv] = link[:nk_w, :nv, :nv]
    U[:, nv:, nv:] = link[nk_w:, :nc, :nc]
    s_v, s_c = sv[:nk_w, :nv], sv[nk_w:, :nc]
    spread = max(float(np.max((s.max(1) - s.min(1)) / s.max(1))) for s in (s_v, s_c))
    fig = {"lsq_residual_max": float(np.max(np.asarray(res))), "sv_spread_max": spread,
           "cross_block_max": float(np.max(np.asarray(cross))),
           "energy_mismatch_max_ry": float(np.max(de)), "wall_s": round(time.time() - t0, 3)}
    return U, fig


# ---------------------------------------------------------------------------
# 5. one k′ row: E7 and the exchange row
# ---------------------------------------------------------------------------

@partial(jax.jit, static_argnames=("omega",))
def _direct_row(psiL_X, psiL_Y, psiR_X, psiR_Y, W_flat, iq_pair, key_pair, z_keys, y_keys,
                W11_keys, lam_pair, kp, *, omega):
    """BerkeleyGW (head, wing, body) of row k′ = kp for all k, (nk, n2′, n2, n1′, n1) each.

    ``psiL`` the n1 (valence-role) leg, ``psiR`` the n2 leg, each (nk, n, ns, μ)
    at μ on 'x' (``_X``) and on 'y' (``_Y``); ``W_flat`` (μ, ν, nq) at
    ``P('x','y',None)``; ``iq_pair``/``key_pair``/``lam_pair`` (nk, nk) the q
    index, head-column key and |q| of each (k, k′); ``z_keys``/``y_keys``
    (n_keys, μ) and ``W11_keys`` (n_keys,) the split per key (replicated).
    """
    iq_row, kidx, lam_row = iq_pair[:, kp], key_pair[:, kp], lam_pair[:, kp]
    z_row, y_row, W11_row = z_keys[kidx], y_keys[kidx], W11_keys[kidx]
    Ccc = A.pair_amplitudes(psiR_X, jnp.broadcast_to(psiR_X[kp], psiR_X.shape), xp=jnp)
    Cvv = A.pair_amplitudes(psiL_Y, jnp.broadcast_to(psiL_Y[kp], psiL_Y.shape), xp=jnp)
    Wq = jnp.moveaxis(jnp.take(W_flat, iq_row, axis=2), 2, 0)
    head, wing, body = A.direct_parts(Ccc, Cvv, Wq, z_row, y_row, W11_row,
                                      omega=omega, lam_q=lam_row, xp=jnp)
    order = lambda a: jnp.transpose(a, (0, 2, 1, 4, 3))
    return order(head), order(wing), order(body)


def _downfold_stage(*, mu_small, rcond, restart, psi_win, psi_1, psi_2, n_rmu, W_flat, V_flat,
                    z_keys, y_keys, W11, keys, v_head_q, Vbar, grid, mesh_xy):
    """The downfold (``gw.downfold``; docs/downfold.md) as the writer's first kernel stage.

    The window is the file's own band set (``psi_win``: the n_v + n_c bands, bundle k
    order); the pair-density Gram, the orbit-closed CUR selection and the transfer
    T_q = S_SS⁺ S_SL are the driver's functions, unmodified. Every linear object of the
    kernel then rides T: W_S = T W T†, V̄_Γ,S = T₀ V̄_Γ T₀†, and the split's vectors
    z_S = T̄ z, y_S = T̄ y (Z_S = Z T†, so the plane-wave head W₁₁ is unchanged and the
    child's head, wing and body are the parent's carried by T; docs/theory/bgw-bsemat-export.md
    #downfold). ψ at the kept centroids is the column slice. The child split
    (a = V_S⁺ z_S*) runs as a gated diagnostic: its residual is exact by construction,
    its W₁₁ equals the parent's only where head_row_S = 1.
    """
    from gw.downfold import (BandWindow, pair_density_gram, select_cur_centroids,
                             slice_psi_to_centroids, build_transfer, congruence, epsilon_w,
                             record_kept_rank)
    from file_io.restart_bundle import read_interaction_orbits
    from runtime.padding import padded_mu_extent
    from common.collectives import device_put_process_local

    t0 = time.time()
    sh = lambda *spec: NamedSharding(mesh_xy, P(*spec))
    put = lambda a, *spec: device_put_process_local(np.ascontiguousarray(a), sh(*spec))
    rep = jax.jit(lambda a: a, out_shardings=sh())
    nw = int(psi_win.shape[1])
    window = BandWindow(left=(0, nw), right=(0, nw))
    kg = tuple(int(g) for g in grid)
    psiY = put(psi_win, None, None, None, "y")
    psiX = put(np.conj(psi_win).transpose(0, 3, 1, 2), None, "x", None, None)
    S_LL = pair_density_gram(psiX, psiY, window, kgrid=kg, mesh_xy=mesh_xy)
    S_q0 = jax.jit(lambda a: a[0], out_shardings=sh("x", "y"))(S_LL)
    tables = read_interaction_orbits(restart, n_rmu, _log)
    keep_idx, sel = select_cur_centroids(
        S_q0, mu_small, rcond=rcond, select_tol=None, mesh_xy=mesh_xy, mu_large_logical=n_rmu,
        print_fn=_log, sym_perm=None if tables is None else np.asarray(tables.sym_perm)[:, :n_rmu])
    _log(sel.describe())
    mu_S = int(sel.mu_small)
    mu_S_pad = int(padded_mu_extent(mu_S, mesh_xy))
    keep = np.asarray(keep_idx, dtype=np.int64)
    psiSX, psiSY = slice_psi_to_centroids(psiX, psiY, keep, mu_S_pad, mesh_xy)
    S_cross = pair_density_gram(psiSX, psiY, window, kgrid=kg, mesh_xy=mesh_xy)
    S_SS = pair_density_gram(psiSX, psiSY, window, kgrid=kg, mesh_xy=mesh_xy)
    T_x, T_y, reports, _ = build_transfer(S_SS, S_cross, mesh_xy, rcond=rcond, print_fn=_log)
    record_kept_rank(sel, reports[0])                          # the kept-block rank, from the q = 0 solve
    operands, project = congruence(mesh_xy)                    # A_S = project(A_L, *operands(T_x, T_y))
    ops = operands(T_x, T_y)
    qmaj = jax.jit(lambda w: jnp.moveaxis(w, 2, 0), out_shardings=sh(None, "x", "y"))
    flat = jax.jit(lambda w: jnp.moveaxis(w, 0, 2), out_shardings=sh("x", "y", None))
    Wq = qmaj(W_flat)
    W_Sq = project(Wq, *ops)
    eps_W, _, _ = epsilon_w(Wq, S_LL, W_Sq, S_SS, mesh_xy)
    del Wq, S_LL
    V_Sq = project(qmaj(V_flat), *ops)
    V0 = jax.jit(lambda v: v[None], out_shardings=sh(None, "x", "y"))(Vbar)
    Vbar_S = jax.jit(lambda v: v[0], out_shardings=sh("x", "y"))(
        project(V0, *operands(T_x[:1], T_y[:1])))
    T_h = np.asarray(rep(T_x))                                  # (nq, μ_S_pad, μ_L_pad)
    zS = np.stack([np.conj(T_h[iq]) @ z_keys[k] for k, (iq, _) in enumerate(keys)])
    yS = np.stack([np.conj(T_h[iq]) @ y_keys[k] for k, (iq, _) in enumerate(keys)])
    del T_h
    t_df = time.time() - t0

    # The child split, as a diagnostic of the identity the downfold must keep.
    V_h, W_h = np.asarray(rep(V_Sq)), np.asarray(rep(W_Sq))
    res, hrow, w11 = [], [], []
    for k, (iq, _) in enumerate(keys):
        if iq == 0:
            continue
        sp = A.head_split(V_h[iq], W_h[iq], zS[k], float(v_head_q[iq]),
                          eigh=np.linalg.eigh, xp=np, rcond=A.RCOND)
        res.append(float(sp.figures["solve_residual"]))
        hrow.append(float(sp.figures["head_row"]))
        w11.append(abs(float(sp.W11) / float(W11[k]) - 1.0))
    del V_h, W_h, V_Sq
    if max(res) > 1e-6:
        raise ValueError(f"GATE bsemat_downfold_head_solve: |V_S a - z_S*|/|z_S*| = {max(res):.2e} "
                         "> 1e-6; z_S* must lie in range(V_S) = range(Z_S†) by construction.")
    cut = lambda a: np.ascontiguousarray(np.pad(np.take(a, keep, axis=-1),
                                                [(0, 0)] * (a.ndim - 1) + [(0, mu_S_pad - mu_S)]))
    eps = np.asarray(eps_W, dtype=np.float64).ravel()
    fig = {"mu_L": int(n_rmu), "mu_S_requested": int(sel.mu_small_requested or mu_S),
           "mu_S": mu_S, "mu_S_pad": mu_S_pad, "N_b": nw, "rcond": float(rcond),
           "ceiling_eigen_rank_pool": int(sel.eigen_rank_pool), "eigen_rank_kept": int(sel.eigen_rank_kept),
           "orbits_kept": sel.n_orbits_kept, "orbits_pool": sel.n_orbits_pool,
           "rank_per_q_min": int(min(r.rank_criterion for r in reports)),
           "rank_per_q_max": int(max(r.rank_criterion for r in reports)),
           "eps_W_min": float(np.nanmin(eps)), "eps_W_median": float(np.nanmedian(eps)),
           "eps_W_max": float(np.nanmax(eps)),
           "child_solve_residual_max": max(res), "child_head_row_min": min(hrow),
           "child_head_row_max": max(hrow), "child_W11_rel_max": max(w11),
           "wall_downfold_s": t_df}
    _log(f"[bsemat] downfold: mu_L {n_rmu} -> mu_S {mu_S} (requested {fig['mu_S_requested']}, "
         f"ceiling {fig['ceiling_eigen_rank_pool']} at rcond {rcond:g}); eps_W "
         f"{fig['eps_W_min']:.2e}/{fig['eps_W_median']:.2e}/{fig['eps_W_max']:.2e}; child split: residual "
         f"{max(res):.1e}, head_row {min(hrow):.6f}-{max(hrow):.6f}, |W11_S/W11 - 1| {max(w11):.1e}; "
         f"{t_df:.1f} s")
    return dict(W_flat=flat(W_Sq), z_keys=zS, y_keys=yS, Vbar=Vbar_S,
                psi_1=cut(psi_1), psi_2=cut(psi_2), fig=fig)


@partial(jax.jit, static_argnames=("omega",))
def _exchange_row(T_cv, Ccv, kp, *, omega):
    """BerkeleyGW exchange of row k′ for all k, (nk, n2′, n2, n1′, n1) (E7).

    ``T_cv = C_cv(k) V̄_Γ`` (nk, n2, n1, ν) precomputed; ``Ccv`` (nk, n2, n1, ν).
    """
    return jnp.einsum("kcvn,dun->kdcuv", T_cv, jnp.conj(Ccv[kp])) * (omega / EIGHT_PI)


def _ri(a):
    return jnp.stack([a.real, a.imag], axis=-1)[None]


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

#: The writer's ``common.timing`` rows, in run order. The shared stages run once
#: per process (top-level ``bsemat.<stage>``); the output stages run once per file
#: written (children of ``bsemat.output``). The CLI adds the pre-``main`` rows
#: ``bsemat.runtime_stack.<phase>`` and ``bsemat.imports`` (``runtime.run_session``).
SHARED_STAGES = ("deferred_imports", "restart_load", "wfn_co_read", "head_columns", "head_split",
                 "psi_gauge")
OUTPUT_STAGES = ("q0_body", "psi_map", "downfold", "rows", "exchange", "write")
_OPEN_STAGE = []


def _stage(name, *, error=None):
    """Close the running shared-stage section, then open ``bsemat.<name>`` (``None``: close only).

    The shared stages are consecutive spans of one body (``_export_context``),
    marked in sequence rather than indented into ``with`` blocks so each span's
    code reads as it does in a one-stage edit. ``error`` closes a span an
    exception left open (``run_bsemat_exports``). Rank 0 announces each span's
    entry and exit, with a 60 s heartbeat while it runs (``common.timing``).
    """
    if _OPEN_STAGE:
        sec = _OPEN_STAGE.pop()
        sec.__exit__(None if error is None else type(error), error, None)
    if name is not None:
        sec = timing.section(f"bsemat.{name}", announce=True)
        sec.__enter__()
        _OPEN_STAGE.append(sec)


def _stage_walls(parent=()):
    """``{name: inclusive s}`` of the ``common.timing`` rows directly under the path ``parent``."""
    d = len(parent)
    return {str(r["name"]): float(r["inclusive"]) for r in timing.records()
            if r["depth"] == d and tuple(r["path"][:d]) == tuple(parent)}


def output_spec(out_path, *, extended=False, use_gauge_map=True, q0_body="head_less",
                q0_vector=None, q0_body_update=None, mu_small=None, downfold_rcond=None):
    """One file of an export: ``run_bsemat_export``'s per-file options as a plain dict."""
    if q0_body not in ("head_less", "dipole"):
        raise ValueError(f"q0_body must be head_less or dipole, got {q0_body!r}")
    return dict(out_path=out_path, extended=bool(extended), use_gauge_map=bool(use_gauge_map),
                q0_body=q0_body, q0_vector=q0_vector, q0_body_update=q0_body_update,
                mu_small=mu_small, downfold_rcond=downfold_rcond)


def run_bsemat_export(*, input_file, wfn_co, out_path, n_val, n_cond, extended=False,
                      use_gauge_map=True, rcond=A.RCOND, bgw_eps=None, q0_body="head_less",
                      q0_vector=None, q0_body_update=None, mu_small=None,
                      downfold_rcond=None, exciton_Q=(0.0, 0.0, 0.0), wfnq_co=None, mesh_xy):
    """Write ``out_path`` (BerkeleyGW bsemat.h5) from the deck's restart. Returns a summary dict.

    ``bgw_eps`` (an ``epsmat.h5``) supplies BerkeleyGW's own q wedge, so the
    head slot of every zone-boundary tie is BerkeleyGW's; without it the
    restart's full q grid stands in (same physics, possibly another tied slot).
    ``q0_body_update`` ``(A (n, μ), D (n, n))`` is the q → 0 hook
    (``bsemat_algebra.add_q0_body_update``): the wing-mediated body term of
    BerkeleyGW's eps0mat, from the eps export's dipole wing vector. ``None``
    keeps the head-less Γ body. ``q0_body = "dipole"`` builds it from the
    restart's stored wings (``head_wing_L``/``head_wing_R``, ``S_cart_head``,
    ``S_cart_head_direct``; ``bsemat_algebra.q0_dipole_body_rows``) along
    BerkeleyGW's q0 (``q0_vector`` crystal, else the ``eps0mat.h5`` beside
    ``bgw_eps``); 3-D only, the term is O(q0) in a slab.
    ``mu_small`` (int or ``"auto"``) runs the downfold on the file's band window
    before the rows (``_downfold_stage``), at ``downfold_rcond`` (default the
    downfold driver's); ``None`` builds the rows on the parent's μ.
    ``exciton_Q`` (crystal) writes the finite-momentum kernel of the pairs
    |v k−Q, c k⟩ on WFN_co's k list (``finite_q_legs``); off the restart grid,
    WFN_co is the grid shifted by Q and ``wfnq_co`` the restart grid's own WFN
    (BerkeleyGW's WFNq_co).
    One file of ``run_bsemat_exports``, which shares the load across files.
    """
    return run_bsemat_exports(
        input_file=input_file, wfn_co=wfn_co, n_val=n_val, n_cond=n_cond, rcond=rcond,
        bgw_eps=bgw_eps, exciton_Q=exciton_Q, wfnq_co=wfnq_co, mesh_xy=mesh_xy,
        outputs=[output_spec(out_path, extended=extended, use_gauge_map=use_gauge_map,
                             q0_body=q0_body, q0_vector=q0_vector,
                             q0_body_update=q0_body_update, mu_small=mu_small,
                             downfold_rcond=downfold_rcond)])[0]


def run_bsemat_exports(*, input_file, wfn_co, n_val, n_cond, outputs, rcond=A.RCOND,
                       bgw_eps=None, exciton_Q=(0.0, 0.0, 0.0), wfnq_co=None, mesh_xy):
    """Write several bsemat files from one load of the deck's restart; one summary per file.

    ``outputs`` are ``output_spec`` dicts with distinct ``out_path``; they may differ
    in kernel form, gauge, q → 0 body and downfold. The bundle, BerkeleyGW's head
    momenta and ζ columns, the split (E3–E5) and the gauge map are built once
    (``_export_context``); each file then applies its own q → 0 body, ψ map,
    downfold and rows (``_write_output``), and files of one shape reuse the
    compiled row kernels. The split reads W only at q ≠ 0, so a per-file q → 0
    body leaves it exact, and every file equals its one-file export bit for bit
    (tests/test_bsemat_multi_cpu.py). Each summary carries ``stages`` (the shared
    stages' and its own walls, s), ``wall_shared_s``, and ``wall_total_s`` from
    the start of the export to the end of that file.
    """
    outputs = list(outputs)
    paths = [os.path.abspath(o["out_path"]) for o in outputs]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError(f"run_bsemat_exports: want one or more distinct output paths, got {paths}")
    Q = np.asarray(exciton_Q, dtype=np.float64).reshape(3)
    if np.any(np.abs(Q - np.rint(Q)) > 1e-9):
        bad = [o["out_path"] for o in outputs if o["extended"] or o["mu_small"] is not None]
        if bad:
            raise ValueError(f"GATE bsemat_finite_q_form: {bad} ask for extended_kernel or the downfold "
                             "at exciton_Q != 0; the finite-Q file is the TDA kernel on the parent's "
                             "centroids (kernel.x's finite-Q route). fix: extended_kernel = false and --mu-small off")
    t0, w0 = time.time(), _stage_walls()
    try:
        ctx = _export_context(input_file=input_file, wfn_co=wfn_co, n_val=n_val, n_cond=n_cond,
                              rcond=rcond, bgw_eps=bgw_eps, exciton_Q=Q, wfnq_co=wfnq_co,
                              use_gauge_map=any(o["use_gauge_map"] for o in outputs),
                              mesh_xy=mesh_xy)
    except BaseException as exc:
        _stage(None, error=exc)
        raise
    if all(o["mu_small"] is None for o in outputs):
        ctx["V_flat"] = None                         # past the split only the downfold reads V
    w1, t_shared = _stage_walls(), time.time() - t0
    shared = {n: w1[f"bsemat.{n}"] - w0.get(f"bsemat.{n}", 0.0)
              for n in SHARED_STAGES if f"bsemat.{n}" in w1}
    summaries = []
    for i, out in enumerate(outputs):
        before = _stage_walls(("bsemat.output",))
        s = _write_output(ctx, out, release=i == len(outputs) - 1)
        after = _stage_walls(("bsemat.output",))
        s["stages"] = {"shared": shared, "output": {n: after[n] - before.get(n, 0.0)
                                                    for n in OUTPUT_STAGES if n in after}}
        s["wall_shared_s"] = t_shared
        s["wall_total_s"] = time.time() - t0
        summaries.append(s)
    ctx["wfn"].close()
    return summaries


def _export_context(*, input_file, wfn_co, n_val, n_cond, rcond, bgw_eps, use_gauge_map,
                    mesh_xy, exciton_Q=(0.0, 0.0, 0.0), wfnq_co=None):
    """The shared stages of an export, once per restart; returns the plain dict ``_write_output`` reads.

    The bundle at full q (1), BerkeleyGW's head momenta and ζ columns (2), the
    split per key on V's face (3) and the gauge map (4), fitted when
    ``use_gauge_map`` (some file wants the coarse WFN's gauge). W and V stay
    resident flattened at ``P('x','y',None)``; the bundle's own copies are dropped.
    """
    _stage("deferred_imports")
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader
    from vcoul import lr_channels as vc_lr_channels
    from symmetry_maps import read_tables
    from file_io import bgw_bsemat as F
    from file_io.restart_bundle import (_find_restart_file, read_metadata,
                                        refuse_mpa_static_w)
    from file_io.centroids import load_centroids
    from gw.lr_channels import slab_shells
    from .bse_loading import load_bse_data_from_restart_sharded
    from .bse_window import resolve_n_occ

    _stage("restart_load")
    restart = _find_restart_file(input_file)
    refuse_mpa_static_w(restart)                         # GATE bse_mpa_w0
    params = _deck_params(input_file)
    sys_dim = int(params.get("sys_dim", 3))
    if sys_dim == 3 and str(params.get("mc_average_vcoul_body", "false")).lower() in ("true", "1", "yes"):
        raise ValueError("GATE bsemat_point_head: mc_average_vcoul_body = true puts a cell "
                         "average in V's head slot at every q; the split needs the point "
                         "v(q+G1) that BerkeleyGW's kernel uses. fix: a deck with it false.")
    wfn_path = params["wfn_file"] if os.path.isabs(params["wfn_file"]) else os.path.join(
        os.path.dirname(os.path.abspath(input_file)), params["wfn_file"])
    wfn = WfnLoader(wfn_path)
    sym = wfn.symmetry()
    header = read_metadata(restart)

    # ── 1. the bundle at full q ───────────────────────────────────────────
    data = load_bse_data_from_restart_sharded(
        restart, n_val, n_cond, mesh_xy=mesh_xy, pad_bands=False, inject_head=False,
        load_v_full=True, input_file=input_file)
    n_val, n_cond = int(data["n_val"]), int(data["n_cond"])
    grid = np.array([data["nkx"], data["nky"], data["nkz"]], dtype=np.int64)
    nk = int(np.prod(grid))
    n_rmu, n_rmu_pad = int(data["n_rmu"]), int(data["n_rmu_pad"])
    rep = jax.jit(lambda a: a, out_shardings=NamedSharding(mesh_xy, P()))
    psi_v = np.asarray(rep(data["psi_v_X"]))          # (nk, nv, ns, μ_pad), bands ascending
    psi_c = np.asarray(rep(data["psi_c_X"]))
    k_int = np.asarray(sym.kvecs_asints, dtype=np.int64)
    fft_int = np.stack(np.meshgrid(*[np.arange(n) for n in grid], indexing="ij"), -1).reshape(-1, 3)
    if k_int.shape != fft_int.shape or not np.array_equal(np.mod(k_int, grid), fft_int):
        raise ValueError("GATE bsemat_k_order: SymMaps' k rows are not the bundle's flat fft order")

    _stage("wfn_co_read")
    # BerkeleyGW's coarse k list and its map onto the bundle's k rows.
    kpts, kgrid_w, bdot, celvol = F.read_wfn_k_list(wfn_co)
    if len(kpts) != nk:
        raise ValueError(f"GATE bsemat_full_grid: {wfn_co} has {len(kpts)} k, the bundle {nk}; "
                         "want an explicit full-grid WFN_co (no_symmetries_coarse_grid).")
    # Finite Q: row j holds the electron at k_j (WFN_co) and the hole at k_j − Q (the grid).
    exciton_Q = np.asarray(exciton_Q, dtype=np.float64).reshape(3)
    qflag, c_rows, v_rows = finite_q_legs(kpts, grid, exciton_Q)
    if (qflag == 0) != bool(wfnq_co):
        raise ValueError(f"GATE bsemat_finite_q_wfnq: exciton_Q {exciton_Q.tolist()} is "
                         + ("off the restart grid and no WFNq_co was given; fix: --wfnq-co, the "
                            "restart grid's explicit WFN (BerkeleyGW reads the hole leg from it)"
                            if qflag == 0 else "on the restart grid, where WFNq_co is not read; "
                            "fix: drop --wfnq-co"))
    if qflag == 0:
        require_offgrid_zeta(restart, nk, exciton_Q)     # GATE bsemat_finite_q_offgrid
    if c_rows is not None and len(set(c_rows.tolist())) != nk:
        raise ValueError("GATE bsemat_k_bijection: WFN_co k points do not cover the grid once")
    wfn_v = wfnq_co if qflag == 0 else wfn_co            # the file whose gauge the hole leg takes
    kv = F.read_wfn_k_list(wfn_v)[0] if qflag == 0 else kpts
    kv_rows = grid_index(kv, grid)
    if len(kv) != nk or len(set(kv_rows.tolist())) != nk:
        raise ValueError(f"GATE bsemat_k_bijection: {wfn_v} k points do not cover the grid once")
    row_of = np.empty(nk, np.int64)
    row_of[kv_rows] = np.arange(nk)                      # flat grid index -> row of wfn_v
    with h5py.File(wfn_co, "r") as w:
        el_all = np.asarray(w["mf_header/kpoints/el"][()]).reshape(-1, nk, int(w["mf_header/kpoints/mnband"][()]))[0]
        ecutwfc = float(w["mf_header/kpoints/ecutwfc"][()])

    _stage("head_columns")
    # ── 2. head momenta and ζ columns ─────────────────────────────────────
    with h5py.File(restart, "r") as f:
        tables = read_tables(f, "W0_qmunu")
    with h5py.File(wfn_co, "r") as w:
        mtrx = np.asarray(w["mf_header/symmetry/mtrx"][()])[:int(w["mf_header/symmetry/ntran"][()])]
    from symmetry_maps import bgw_integer_q_to_fractional
    q_full_L = bgw_integer_q_to_fractional(fft_int, grid)
    if bgw_eps:
        with h5py.File(bgw_eps, "r") as e:
            q_wedge = np.asarray(e["eps_header/qpoints/qpts"][()], dtype=np.float64)
        q_src = f"BerkeleyGW eps q list ({bgw_eps})"
    else:
        # The restart wedge may use authenticated antiunitary operations that
        # the coarse WFN's spatial-only header cannot unfold. All q of the
        # validated restart grid provide the standalone head-order carrier.
        q_wedge = q_full_L
        q_src = "restart full q grid"
    q_list = bgw_kernel_q_list(q_wedge, mtrx, bdot)
    K, qlen = pair_head_momenta(kpts, bdot, q_list)
    iq_pair = np.empty((nk, nk), np.int64)
    keys, key_of = [], {}
    for i in range(nk):
        for j in range(nk):
            dq = kpts[i] - kpts[j]
            iq = int(grid_index(dq[None], grid)[0])
            G = K[i, j] - q_full_L[iq]
            Gi = np.rint(G).astype(np.int64)
            if not np.allclose(G, Gi, atol=1e-6):
                raise ValueError("GATE bsemat_head_label: head momentum off the q's lattice")
            kk = (iq, tuple(int(v) for v in Gi))
            if kk not in key_of:
                key_of[kk] = len(keys)
                keys.append(kk)
            iq_pair[i, j] = iq
    key_pair = np.array([[key_of[(int(iq_pair[i, j]), tuple(int(v) for v in np.rint(
        K[i, j] - q_full_L[iq_pair[i, j]]).astype(np.int64)))] for j in range(nk)] for i in range(nk)])
    zeta_path = os.path.join(os.path.dirname(restart), "zeta_q.h5")
    z_keys, _, omega = head_columns(keys, zeta_path=zeta_path, tables=tables, sym=sym, grid=grid,
                                    n_rmu=n_rmu, n_rmu_pad=n_rmu_pad, mesh_xy=mesh_xy)
    n_ties = len(keys) - len({k[0] for k in keys})
    _log(f"[bsemat] {nk} k, {len(keys)} head columns ({n_ties} extra from zone-boundary ties, "
         f"kernel q list of {len(q_list)} from the {q_src}); Ω = {omega:.4f} bohr³")

    # v(q + G₁) as V carries it: the point kernel of the head channel.
    from vcoul import CoulombGeometry
    bvec = np.asarray(CoulombGeometry.from_wfn(wfn).bvec, dtype=np.float64)
    n_z = slab_shells(bvec) if sys_dim == 2 else 0
    ch = vc_lr_channels(q_full_L, bvec=bvec, sys_dim=sys_dim, n_z=n_z, head_fn=None)
    c0 = n_z if sys_dim == 2 else 0
    v_head_q = np.asarray(ch.v)[:, c0]
    K_ch = q_full_L + np.asarray(ch.gvec)[:, c0]

    _stage("head_split")
    # ── 3. the split per key, on V's face ─────────────────────────────────
    W_flat = jax.jit(lambda w: w.reshape(w.shape[0], w.shape[1], -1),
                     out_shardings=NamedSharding(mesh_xy, P("x", "y", None)))(data["W_q"])
    V_flat = jax.jit(lambda w: w.reshape(w.shape[0], w.shape[1], -1),
                     out_shardings=NamedSharding(mesh_xy, P("x", "y", None)))(data["V_q_full"])
    W11, y_keys, figs, split_stats = head_split_keys(
        keys, z_keys, V_flat, W_flat, zeta_path=zeta_path, tables=tables, sym=sym, grid=grid,
        n_rmu=n_rmu, n_rmu_pad=n_rmu_pad, v_head_q=v_head_q, q_frac=q_full_L, head_K=K_ch,
        bvec=bvec, rcond=rcond, mesh_xy=mesh_xy)
    _log("[bsemat] split: max solve residual %.2e, head_row min/max %.6f/%.6f, "
         "W11/v1 min/max %.4f/%.4f" % (
             max(f.get("solve_residual", 0) for f in figs),
             min(f.get("head_row", 1) for f in figs), max(f.get("head_row", 0) for f in figs),
             min(f.get("eps_inv_head", 1) for f in figs), max(f.get("eps_inv_head", 0) for f in figs)))

    _stage("psi_gauge")
    # ── 4. gauge map ──────────────────────────────────────────────────────
    bw = header.get("band_window")
    b0 = int(np.asarray(bw).reshape(-1)[0]) if bw is not None else 0
    psi_win = np.concatenate([psi_v, psi_c], axis=1)        # ascending energy
    enk = np.asarray(header["energies"])
    n_occ_w = int(resolve_n_occ(enk, input_file=input_file))
    bands_abs = b0 + np.arange(n_occ_w - n_val, n_occ_w + n_cond)
    enk_win = enk[:, n_occ_w - n_val:n_occ_w + n_cond]
    cent_frac = load_centroids(params["centroids_file"] if os.path.isabs(params["centroids_file"])
                               else os.path.join(os.path.dirname(os.path.abspath(input_file)),
                                                 params["centroids_file"]),
                               tuple(int(x) for x in wfn.fft_grid))[0]
    nb = n_val + n_cond
    eye = lambda: np.broadcast_to(np.eye(nb), (nk, nb, nb)).copy()
    gmap = lambda psi, wfile, rows, e: gauge_map(
        psi, wfn_co=wfile, k_rows=rows, bands_abs=bands_abs, centroid_frac=cent_frac,
        fft_grid=wfn.fft_grid, nv=n_val, enk_win=e, mesh_xy=mesh_xy)
    if use_gauge_map:
        U_file, gfig = gmap(psi_win, wfn_v, kv_rows, enk_win)
    else:
        U_file, gfig = eye(), {"gauge_map": "identity (control variant)"}
    _log(f"[bsemat] gauge map ({os.path.basename(wfn_v)}): {gfig}")
    # The two legs of every file row, in its own k order: source ψ and window unitary.
    src_v, Uv = psi_win[v_rows], U_file[row_of[v_rows]]
    if qflag == 0:
        psi_h, enk_h, ht_rec = htransform_window(input_file, kpts, n_val=n_val, n_cond=n_cond,
                                                 n_rmu=n_rmu, mesh_xy=mesh_xy)
        src_c = np.zeros_like(psi_win)
        src_c[..., :n_rmu] = psi_h
        eps_c = enk_h[:, n_val:]
        if use_gauge_map:
            Uc, gfig_c = gmap(src_c, wfn_co, np.arange(nk), enk_h)
        else:
            Uc, gfig_c = eye(), {"gauge_map": "identity (control variant)"}
        _log(f"[bsemat] gauge map ({os.path.basename(wfn_co)}, htransform electron leg): {gfig_c}")
        gfig = {"hole_leg": gfig, "electron_leg": gfig_c,
                "htransform": [str(r)[:400] for r in ht_rec]}
    else:
        src_c, Uc, eps_c = psi_win[c_rows], U_file[row_of[c_rows]], enk_win[c_rows, n_val:]

    # q = 0 head column check against the bundle's G = 0 vector.
    q0 = [kk for kk, (iq, G) in enumerate(keys) if iq == 0 and G == (0, 0, 0)][0]
    if data.get("g0_X") is not None:
        g0 = np.asarray(rep(data["g0_X"]))[:n_rmu]
        dz = np.linalg.norm(z_keys[q0][:n_rmu] * np.sqrt(omega) - g0) / np.linalg.norm(g0)
        _log(f"[bsemat] q=0 column vs bundle G0 vector: rel. diff {dz:.2e}")

    nocc_abs = int(bands_abs[n_val])
    efermi = 0.5 * (el_all[:, :nocc_abs].max() + el_all[:, nocc_abs:].min()) * 13.605693122994
    # The exchange tile: head-less Γ at Q = 0, V at wrap(+Q) with every G otherwise (E12).
    V_ex, ex_rec = exchange_tile(exciton_Q, qflag, data=data, input_file=input_file, restart=restart,
                                 grid=grid, n_rmu=n_rmu, n_rmu_pad=n_rmu_pad, mesh_xy=mesh_xy)
    _log(f"[bsemat] exchange: {ex_rec}; exciton_Q {exciton_Q.tolist()} (qflag {qflag}, "
         f"exciton_Q_shift {(-exciton_Q).tolist()})")
    _stage(None)
    data["W_q"] = data["V_q_full"] = None            # the flattened copies above stay resident
    return dict(restart=restart, sys_dim=sys_dim, bgw_eps=bgw_eps, wfn=wfn, wfn_co=wfn_co,
                mesh_xy=mesh_xy, rep=rep, n_val=n_val, n_cond=n_cond, nk=nk, grid=grid,
                n_rmu=n_rmu, n_rmu_pad=n_rmu_pad, nspinor=int(psi_v.shape[2]), omega=omega,
                bvec=bvec, kpts=kpts, kgrid_w=kgrid_w, ecutwfc=ecutwfc, efermi=efermi,
                exciton_Q=exciton_Q, qflag=qflag, exchange=ex_rec, eps_c=eps_c,
                src_c=src_c, src_v=src_v, Uc=Uc, Uv=Uv,
                keys=keys, n_ties=n_ties, iq_pair=iq_pair, key_pair=key_pair,
                lam_pair=qlen if sys_dim == 3 else np.ones_like(qlen),
                z_keys=z_keys, y_keys=y_keys, W11=W11, figs=figs, split_stats=split_stats,
                v_head_q=v_head_q,
                W_flat=W_flat, V_flat=V_flat, Vbar=V_ex, psi_win=psi_win,
                use_gauge_map=use_gauge_map, gfig=gfig, q0_dipole={},
                q0_update=jax.jit(lambda w, a, d: w.at[:, :, 0].set(
                    A.add_q0_body_update(w[:, :, 0], a, d, xp=jnp)), out_shardings=W_flat.sharding))


def _q0_dipole_rows(ctx, q0_vector, W_flat):
    """``(A, D, figures)`` of BerkeleyGW's 3-D q0 body term along q0 (``bsemat_algebra.q0_dipole_body_rows``).

    From the restart's stored q → 0 wings and the head-less Γ tile of ``W_flat``
    (before any q0 update), gated; computed once per q0 direction per export
    (``ctx['q0_dipole']``). ``q0_vector`` (crystal) or ``None`` for the
    ``eps0mat.h5`` beside ``bgw_eps``.
    """
    from common.collectives import device_put_process_local
    restart, sys_dim, bgw_eps, bvec = ctx["restart"], ctx["sys_dim"], ctx["bgw_eps"], ctx["bvec"]
    n_rmu, n_rmu_pad, omega, mesh_xy = ctx["n_rmu"], ctx["n_rmu_pad"], ctx["omega"], ctx["mesh_xy"]
    memo = None if q0_vector is None else tuple(float(x) for x in q0_vector)
    if memo in ctx["q0_dipole"]:
        return ctx["q0_dipole"][memo]
    if sys_dim != 3:
        raise ValueError("GATE bsemat_q0_body_dim: the dipole q0 body term is O(q0) in a slab "
                         "(BerkeleyGW's own body moves 8e-8); fix: --q0-body head-less.")
    names = ("head_wing_L", "head_wing_R", "S_cart_head", "S_cart_head_direct")
    with h5py.File(restart, "r") as f:
        miss = [n for n in names if n not in f]
        if miss:
            raise ValueError(f"GATE bsemat_q0_wings_source: {restart} lacks {miss}; the dipole "
                             "q0 body needs the GW bundle's stored q -> 0 wings. fix: rerun GW "
                             "with wing storage, or --q0-body head-less.")
        Lw, Rw, S_eff, S_d = (np.asarray(f[n][()]) for n in names)
    if q0_vector is None:
        eps0 = os.path.join(os.path.dirname(os.path.abspath(bgw_eps)), "eps0mat.h5") if bgw_eps else None
        if not eps0 or not os.path.exists(eps0):
            raise ValueError("GATE bsemat_q0_vector: the dipole q0 body is direction dependent; "
                             "give --q0-vector (crystal) or an eps0mat.h5 beside --bgw-eps.")
        with h5py.File(eps0, "r") as e:
            q0_vector = np.asarray(e["eps_header/qpoints/qpts"][()], np.float64)[0]
    q0_cart = np.asarray(q0_vector, np.float64) @ bvec
    if Lw.shape != (3, n_rmu) or Rw.shape != (n_rmu, 3):
        raise ValueError(f"GATE bsemat_q0_wing_fold: wings {Lw.shape}/{Rw.shape}, the bundle has {n_rmu} μ")
    Lp = np.zeros((3, n_rmu_pad), np.complex128); Lp[:, :n_rmu] = Lw
    Rp = np.zeros((n_rmu_pad, 3), np.complex128); Rp[:n_rmu] = Rw
    rs = NamedSharding(mesh_xy, P())
    dip = jax.jit(lambda w, l, r, se, sd: A.q0_dipole_body_rows(
        w[:, :, 0], l, r, se, sd, q0_cart, omega, xp=jnp), out_shardings=rs)
    A_d, D_d, fig = dip(W_flat, *(device_put_process_local(np.asarray(x, np.complex128), rs)
                                  for x in (Lp, Rp, S_eff, S_d)))
    q0_fig = {k: (complex(np.asarray(v)) if np.iscomplexobj(np.asarray(v)) else float(np.asarray(v)))
              for k, v in fig.items()}
    q0_fig["q0_vector"] = [float(x) for x in q0_vector]
    if not q0_fig["fold_gate"] <= 1e-3 or not q0_fig["R_vs_LH"] <= 1e-10:
        raise ValueError(f"GATE bsemat_q0_wing_fold: Ω q̂(S_eff − S_d)q̂ vs q̂ L W R q̂ on this W "
                         f"differs by {q0_fig['fold_gate']:.2e} (R vs Lᴴ {q0_fig['R_vs_LH']:.2e}); "
                         "L's μ order or normalisation is not this bundle's.")
    out = (np.asarray(A_d)[:, :n_rmu], np.asarray(D_d), q0_fig)
    _log(f"[bsemat] q0 body: dipole term along q0 = {q0_fig['q0_vector']} (crystal), "
         f"ε_M = {q0_fig['eps_M']:.4f}, fold gate {q0_fig['fold_gate']:.1e}")
    ctx["q0_dipole"][memo] = out
    return out


def _write_output(ctx, out, *, release):
    """One bsemat file from the shared context: q → 0 body, ψ map, downfold, rows (E7), write.

    ``ctx`` (``_export_context``) is only read, except that ``release`` (the last
    file) hands its W and V to this file, so the rows run with the resident
    copies a one-file export holds, and the dipole term is memoised per q0.
    ``out`` an ``output_spec``. Returns the file's summary dict.
    """
    from common.collectives import device_put_process_local
    from file_io import bgw_bsemat as F
    from file_io.slab_io import SlabIO

    mesh_xy, rep, omega, nk = ctx["mesh_xy"], ctx["rep"], ctx["omega"], ctx["nk"]
    n_val, n_cond, n_rmu, n_rmu_pad = ctx["n_val"], ctx["n_cond"], ctx["n_rmu"], ctx["n_rmu_pad"]
    out_path, extended, mu_small = out["out_path"], out["extended"], out["mu_small"]
    q0_body, q0_body_update = out["q0_body"], out["q0_body_update"]
    W_flat, V_flat = ctx["W_flat"], ctx["V_flat"]
    if release:
        ctx["W_flat"] = ctx["V_flat"] = None
    with timing.section("bsemat.output", announce=True,
                        label=f"bsemat.output {os.path.basename(out_path)}"):
        # ── the q → 0 body of this file (the split never reads q = 0) ──────
        with timing.section("q0_body"):
            q0_fig = None
            if q0_body == "dipole":
                A_d, D_d, q0_fig = _q0_dipole_rows(ctx, out["q0_vector"], W_flat)
                q0_body_update = (A_d, D_d)
            if q0_body_update is not None:
                A0 = np.zeros((np.asarray(q0_body_update[0]).shape[0], n_rmu_pad), np.complex128)
                A0[:, :n_rmu] = np.asarray(q0_body_update[0])[:, :n_rmu]
                rs = NamedSharding(mesh_xy, P())
                W_flat = ctx["q0_update"](
                    W_flat, device_put_process_local(A0, rs),
                    device_put_process_local(np.asarray(q0_body_update[1], np.complex128), rs))
                _log(f"[bsemat] q -> 0 body update of rank {A0.shape[0]} applied")

        # ── ψ in the file's gauge, k order and band order ─────────────────
        with timing.section("psi_map"):
            Uc, Uv, gfig = ctx["Uc"], ctx["Uv"], ctx["gfig"]
            if ctx["use_gauge_map"] and not out["use_gauge_map"]:     # control variant beside mapped files
                nb = n_val + n_cond
                Uc = Uv = np.broadcast_to(np.eye(nb), (nk, nb, nb)).copy()
                gfig = {"gauge_map": "identity (control variant)"}
                _log(f"[bsemat] gauge map: {gfig}")
            # ψ_file,n = Σ_m ψ_LX,m conj(U[n, m]); rows of the file in WFN_co order, the hole leg
            # at k − Q and the electron leg at k (the same rows at Q = 0).
            to_file = lambda src, U: np.einsum("knsr,kmn->kmsr", src, np.conj(U))
            psi_v_b = to_file(ctx["src_v"], Uv)[:, :n_val][:, ::-1]   # BerkeleyGW: iv = 1 highest
            psi_c_b = to_file(ctx["src_c"], Uc)[:, n_val:]
            psi_1 = np.concatenate([psi_v_b, psi_c_b], axis=1) if extended else psi_v_b
            psi_2 = psi_1 if extended else psi_c_b
            sx = NamedSharding(mesh_xy, P(None, None, None, "x"))
            sy = NamedSharding(mesh_xy, P(None, None, None, "y"))
            put = lambda a, s: device_put_process_local(np.ascontiguousarray(a), s)
            p1x, p1y, p2x, p2y = put(psi_1, sx), put(psi_1, sy), put(psi_2, sx), put(psi_2, sy)

        z_keys, y_keys, Vbar = ctx["z_keys"], ctx["y_keys"], ctx["Vbar"]
        df_fig = None
        if mu_small is not None:
            with timing.section("downfold"):
                from gw.downfold import DEFAULT_RCOND
                dfo = _downfold_stage(
                    mu_small=mu_small, rcond=float(out["downfold_rcond"] or DEFAULT_RCOND),
                    restart=ctx["restart"], psi_win=ctx["psi_win"], psi_1=psi_1, psi_2=psi_2,
                    n_rmu=n_rmu, W_flat=W_flat, V_flat=V_flat, z_keys=z_keys, y_keys=y_keys,
                    W11=ctx["W11"], keys=ctx["keys"], v_head_q=ctx["v_head_q"], Vbar=Vbar,
                    grid=ctx["grid"], mesh_xy=mesh_xy)
                W_flat = dfo["W_flat"]
                z_keys, y_keys, Vbar, psi_1, psi_2 = (dfo[k] for k in ("z_keys", "y_keys", "Vbar", "psi_1", "psi_2"))
                df_fig = dfo["fig"]
                p1x, p1y, p2x, p2y = put(psi_1, sx), put(psi_1, sy), put(psi_2, sx), put(psi_2, sy)
        V_flat = None

        # Exchange operands: C_{n2 n1}(k) = Σ conj(ψ_n2) ψ_n1, T = C V̄_Γ.
        with timing.section("exchange"):
            vim = float(jnp.linalg.norm(Vbar.imag) / jnp.linalg.norm(Vbar))
            Ccv = A.pair_amplitudes(p2x, p1x, xp=jnp)                  # (nk, n2, n1, μ)
            T_cv = jnp.einsum("kcvm,mn->kcvn", Ccv, Vbar)

        # ── 5. rows ───────────────────────────────────────────────────────
        n1b, n2b = psi_1.shape[1], psi_2.shape[1]
        with timing.section("write"):
            if jax.process_index() == 0:
                F.create_bsemat_file(
                    out_path, wfn_file=ctx["wfn_co"], kpts=ctx["kpts"], kgrid=ctx["kgrid_w"],
                    nvb=n_val, ncb=n_cond, extended=extended,
                    icutv=F.ICUTV_OF_SYS_DIM[ctx["sys_dim"]], ecuts=ctx["ecutwfc"],
                    ecutg=ctx["ecutwfc"], efermi=ctx["efermi"], nspinor=ctx["nspinor"],
                    qflag=ctx["qflag"], exciton_Q_shift=-ctx["exciton_Q"])
            shape = F.matrix_shape(nk, n1b, n2b)
            _barrier()
        fro = np.zeros(4)
        with timing.section("rows"):
            r_sh = NamedSharding(mesh_xy, P())
            zk, yk, w11k = (device_put_process_local(np.ascontiguousarray(a), r_sh)
                            for a in (z_keys, y_keys, ctx["W11"]))
            iqp, kyp, lmp = (device_put_process_local(np.ascontiguousarray(a), r_sh)
                             for a in (ctx["iq_pair"].astype(np.int32), ctx["key_pair"].astype(np.int32),
                                       ctx["lam_pair"]))
        t_rows = time.time()
        with SlabIO(out_path, mode="a", mesh=mesh_xy) as io:
            F.register_matrices(io, shape)
            for jp in range(nk):
                # Each matrix of row k′ = jp: the replicated (Re, Im) block and its Frobenius
                # sum (the host read that ends the stage's device work), then one hyperslab.
                with timing.section("rows"):
                    h, w_, b = _direct_row(p1x, p1y, p2x, p2y, W_flat, iqp, kyp, zk, yk, w11k, lmp,
                                           jp, omega=omega)
                for m, arr in enumerate((h, w_, b)):
                    with timing.section("rows"):
                        arr = rep(_ri(arr))
                        fro[m] += float(jnp.sum(arr ** 2))
                    with timing.section("write"):
                        io.write_slab(F.MATRICES[m], arr, offset=(jp, 0, 0, 0, 0, 0, 0))
                with timing.section("exchange"):
                    arr = rep(_ri(_exchange_row(T_cv, Ccv, jp, omega=omega)))
                    fro[3] += float(jnp.sum(arr ** 2))
                with timing.section("write"):
                    io.write_slab(F.MATRICES[3], arr, offset=(jp, 0, 0, 0, 0, 0, 0))
            with timing.section("write"):
                io.sync_writes()                      # the queued hyperslabs, before the close
        t_end = time.time()
        names = ("Head", "Wing", "Body", "X   ")
        for nm, v in zip(names, np.sqrt(fro)):
            _log(f" - {nm} : {v:21.12E}")
    return {"out_path": out_path, "extended": extended, "use_gauge_map": out["use_gauge_map"],
            "mu_small": mu_small, "nk": nk, "n1b": n1b, "n2b": n2b, "keys": len(ctx["keys"]),
            "ties": ctx["n_ties"], "q0_body": q0_body, "q0_dipole": q0_fig, "downfold": df_fig,
            "frobenius": dict(zip(("head", "wing", "body", "exchange"), np.sqrt(fro).tolist())),
            "split": ctx["figs"], "head_split": ctx["split_stats"], "gauge": gfig, "vbar_imag_rel": vim,
            "exciton_Q": ctx["exciton_Q"].tolist(), "qflag": int(ctx["qflag"]),
            "exciton_Q_shift": (-ctx["exciton_Q"]).tolist(), "exchange": ctx["exchange"],
            "kpts": np.asarray(ctx["kpts"]).tolist(),
            "eps_c_ry": np.asarray(ctx["eps_c"]).tolist(),
            "wall_rows_s": t_end - t_rows, "peak_device_bytes": _peak_device_bytes()}


def _deck_params(input_file):
    from .bse_densify import _read_lorrax_input_quietly
    return _read_lorrax_input_quietly(input_file)


def _barrier():
    from jax.experimental import multihost_utils
    if jax.process_count() > 1:
        multihost_utils.sync_global_devices("bsemat_header")


def _peak_device_bytes():
    try:
        st = jax.local_devices()[0].memory_stats() or {}
        return int(st.get("peak_bytes_in_use", 0))
    except Exception:
        return 0
