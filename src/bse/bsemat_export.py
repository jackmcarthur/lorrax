"""LORRAX static W -> BerkeleyGW 4.0 ``bsemat.h5`` (TDA or extended); see docs/theory/bgw-bsemat-export.md.

The outline, in the order it runs. Once per restart (``_export_context``):

1. the restart bundle at full q (``bse_loading``): ψ at the centroids, V(q),
   W(q) and the head-less Γ tiles, all canonical centroid order;
2. BerkeleyGW's head momentum of every (k, k′) pair (``fullbz`` + ``sortbyq``,
   ties included) and the ζ column at it, unfolded from the q wedge;
3. per q the file-only split E3–E5 (``bse.bsemat_algebra.head_split``) on
   V's face, with its gates;
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

EIGHT_PI = 8.0 * np.pi
#: BerkeleyGW ``nrtype.f90``: ``ncell`` (fullbz's Wigner–Seitz scan) and TOL_Small.
BGW_NCELL = 3
BGW_TOL_SMALL = 1.0e-6
#: Relative band-energy tolerance between the restart and the coarse WFN (Ry).
ENERGY_MATCH_TOL_RY = 2.0e-3


def _log(msg):
    if jax.process_index() == 0:
        print(msg, flush=True)


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


# ---------------------------------------------------------------------------
# 2b. ζ columns at the head momenta, unfolded from the wedge
# ---------------------------------------------------------------------------

def head_columns(keys, *, zeta_path, tables, sym, grid, n_rmu, n_rmu_pad, mesh_xy):
    """ζ_q(μ, G)/√Ω at each requested ``(iq, G)`` of the full grid (host, (n_keys, μ_pad)).

    Each key's parent column is ``G_p = S⁻¹(q + G) − q_p`` at its wedge q_p, read
    from ``zeta_q.h5`` and carried to q by ``symmetry_maps.unfold_isdf_one_leg``
    (preselected carrier). Keys of one wedge q that need different parent
    slots (zone-boundary ties) go through separate unfold layers.
    """
    from ffi import _services
    _services.ensure_on_path()
    from symmetry_maps import bgw_integer_q_to_fractional, unfold_isdf_one_leg
    from zeta_loader import ZetaLoader
    from common.collectives import device_put_process_local

    tp = tables.padded(int(n_rmu_pad))
    # the one-leg unfold's star rows, from SymMaps' public tables
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
    n_qi = q_parent.shape[0]
    need = []
    for iq, G in keys:
        S = np.asarray(S_all[int(rows[iq])], dtype=np.float64)
        Gp_f = np.linalg.solve(S, q_full[iq] + np.asarray(G, float)) - q_parent[int(idx[iq])]
        Gp = np.rint(Gp_f).astype(np.int64)
        if not np.allclose(Gp_f, Gp, atol=1e-8):
            raise ValueError(f"GATE bsemat_head_parent: key {(iq, tuple(G))} has parent G {Gp_f}")
        need.append((int(idx[iq]), tuple(int(v) for v in Gp)))
    per_p = {p: sorted({g for (pp, g) in need if pp == p}) for p in range(n_qi)}
    n_layer = max(len(v) for v in per_p.values() if v) if need else 0
    out = np.zeros((len(keys), int(n_rmu_pad)), np.complex128)
    rep = jax.jit(lambda a: a, out_shardings=NamedSharding(mesh_xy, P()))
    with ZetaLoader(zeta_path) as zl:
        omega = float(zl.cell_volume)
        gcomp = np.asarray(zl.gvec_components, dtype=np.int64)
        ngk = np.asarray(zl.ngk_per_q, dtype=np.int64)
        if gcomp.shape[0] != n_qi:
            raise ValueError(f"GATE bsemat_zeta_wedge: {zeta_path} holds {gcomp.shape[0]} q, "
                             f"the W0 wedge {n_qi}")
        for j in range(n_layer):
            src = np.zeros((n_qi, 3), np.int64)
            col = np.zeros((n_qi, int(n_rmu_pad)), np.complex128)
            for p in range(n_qi):
                gl = per_p[p]
                g = gl[min(j, len(gl) - 1)] if gl else (0, 0, 0)
                src[p] = g
                lut = {tuple(v): i for i, v in enumerate(gcomp[p, :, :ngk[p]].T)}
                if g not in lut:
                    raise ValueError(f"GATE bsemat_head_in_sphere: wedge q {p} slot {g} "
                                     "is outside the stored ζ sphere; fix: raise the ζ cutoff.")
                col[p, :n_rmu] = np.asarray(zl.read_zeta_G_local((p, slice(None), lut[g])))
            u = unfold_isdf_one_leg(
                device_put_process_local(col / np.sqrt(omega), NamedSharding(mesh_xy, P(None, "x"))),
                source_gvec_components=src.astype(np.int32), sym=sym, sym_idx=tp.sym_idx_q,
                sym_perm=tp.sym_perm, L_table=tp.L_table, q_irr_frac=tp.q_irr_frac,
                kgrid=tuple(int(g) for g in grid), mesh_xy=mesh_xy, component_action="scalar")
            u = np.asarray(rep(u))
            for kk, (p, g) in enumerate(need):
                if src[p].tolist() == list(g):
                    out[kk] = u[keys[kk][0]]
    return out, q_full, omega


# ---------------------------------------------------------------------------
# 4. gauge map into the coarse WFN's coefficient gauge
# ---------------------------------------------------------------------------

def gauge_map(psi, *, wfn_co, k_rows, bands_abs, centroid_frac, fft_grid, nv, enk_win,
              guard=8):
    """Window unitaries ``U[k] (nb, nb)`` with ψ_LX ≈ ψ_file U, valence and conduction blocks.

    ``psi`` (nk_L, nb, ns, μ) host, LORRAX ψ at the centroids (canonical order,
    bands ascending); ``k_rows[j]`` the LORRAX k of WFN row j; ``bands_abs``
    the absolute 0-based file bands of ψ's band axis; ``enk_win`` (nk_L, nb) Ry
    the restart's energies of those bands (gated against the file's). The file states over the
    window plus ``guard`` bands each side are evaluated at the (grid-rounded)
    centroids, ψ_LX is fitted by least squares (the centroids are not a
    quadrature, dipfix), and each block's polar factor is kept. Returns
    ``(U (nk_wfn, nb, nb), figures)``.
    """
    from file_io.bgw_bsemat import read_wfn_coefficients
    grid = np.asarray(fft_grid, dtype=np.int64)
    rmu = np.mod(np.rint(np.asarray(centroid_frac) * grid[None, :]).astype(np.int64),
                 grid[None, :]) / grid[None, :]
    with h5py.File(wfn_co, "r") as w:
        nb_file = int(w["mf_header/kpoints/mnband"][()])
        rk = np.asarray(w["mf_header/kpoints/rk"][()], dtype=np.float64)
    if rk.shape[0] == 3 and rk.shape[1] != 3:
        rk = rk.T
    b_lo, b_hi = max(0, int(bands_abs[0]) - guard), min(nb_file, int(bands_abs[-1]) + 1 + guard)
    nb, n_mu = psi.shape[1], len(rmu)
    sl_v, sl_c = slice(0, nv), slice(nv, nb)
    w0 = int(bands_abs[0]) - b_lo
    U = np.zeros((len(rk), nb, nb), np.complex128)
    fig = {"lsq_residual_max": 0.0, "sv_spread_max": 0.0, "cross_block_max": 0.0,
           "energy_mismatch_max_ry": 0.0}
    for j in range(len(rk)):
        g, c, el = read_wfn_coefficients(wfn_co, j, b_lo, b_hi)
        de = float(np.max(np.abs(el[int(bands_abs[0]) - b_lo:int(bands_abs[-1]) + 1 - b_lo]
                                  - enk_win[k_rows[j]])))
        fig["energy_mismatch_max_ry"] = max(fig["energy_mismatch_max_ry"], de)
        if de > 5.0e-2:
            raise ValueError(f"GATE bsemat_band_offset: WFN_co row {j} window energies differ "
                             f"from the restart's by {de:.3e} Ry; want the same bands.")
        ph = np.exp(2j * np.pi * ((g + rk[j][None, :]) @ rmu.T))            # full-Bloch convention
        Af = np.einsum("bsg,gm->bsm", c, ph).reshape(b_hi - b_lo, -1)
        Bm = psi[k_rows[j], :, :, :n_mu].reshape(nb, -1)
        X = np.linalg.solve(Af.conj() @ Af.T, Af.conj() @ Bm.T)             # ψ_LX ≈ ψ_file X
        res = np.linalg.norm(Bm - X.T @ Af) / np.linalg.norm(Bm)
        Xw = X[w0:w0 + nb]
        for sl in (sl_v, sl_c):
            Wl, s, Vh = np.linalg.svd(Xw[sl, sl])
            U[j][sl, sl] = Wl @ Vh
            fig["sv_spread_max"] = max(fig["sv_spread_max"], float((s.max() - s.min()) / s.max()))
        cross = (np.linalg.norm(Xw[sl_v, sl_c]) + np.linalg.norm(Xw[sl_c, sl_v])) / np.linalg.norm(Xw)
        fig["lsq_residual_max"] = max(fig["lsq_residual_max"], float(res))
        fig["cross_block_max"] = max(fig["cross_block_max"], float(cross))
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
                             slice_psi_to_centroids, build_transfer, congruence, epsilon_w)
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
    T_x, T_y, reports = build_transfer(S_SS, S_cross, mesh_xy, rcond=rcond, print_fn=_log)
    project = congruence(mesh_xy, T_x, T_y)
    qmaj = jax.jit(lambda w: jnp.moveaxis(w, 2, 0), out_shardings=sh(None, "x", "y"))
    flat = jax.jit(lambda w: jnp.moveaxis(w, 0, 2), out_shardings=sh("x", "y", None))
    Wq = qmaj(W_flat)
    W_Sq = project(Wq)
    eps_W, _, _ = epsilon_w(Wq, S_LL, W_Sq, S_SS, mesh_xy)
    del Wq, S_LL
    V_Sq = project(qmaj(V_flat))
    V0 = jax.jit(lambda v: v[None], out_shardings=sh(None, "x", "y"))(Vbar)
    Vbar_S = jax.jit(lambda v: v[0], out_shardings=sh("x", "y"))(
        congruence(mesh_xy, T_x[:1], T_y[:1])(V0))
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
                      downfold_rcond=None, mesh_xy):
    """Write ``out_path`` (BerkeleyGW bsemat.h5) from the deck's restart. Returns a summary dict.

    ``bgw_eps`` (an ``epsmat.h5``) supplies BerkeleyGW's own q wedge, so the
    head slot of every zone-boundary tie is BerkeleyGW's; without it the
    restart's q wedge stands in (same physics, possibly another tied slot).
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
    One file of ``run_bsemat_exports``, which shares the load across files.
    """
    return run_bsemat_exports(
        input_file=input_file, wfn_co=wfn_co, n_val=n_val, n_cond=n_cond, rcond=rcond,
        bgw_eps=bgw_eps, mesh_xy=mesh_xy,
        outputs=[output_spec(out_path, extended=extended, use_gauge_map=use_gauge_map,
                             q0_body=q0_body, q0_vector=q0_vector,
                             q0_body_update=q0_body_update, mu_small=mu_small,
                             downfold_rcond=downfold_rcond)])[0]


def run_bsemat_exports(*, input_file, wfn_co, n_val, n_cond, outputs, rcond=A.RCOND,
                       bgw_eps=None, mesh_xy):
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
    t0, w0 = time.time(), _stage_walls()
    try:
        ctx = _export_context(input_file=input_file, wfn_co=wfn_co, n_val=n_val, n_cond=n_cond,
                              rcond=rcond, bgw_eps=bgw_eps,
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
                    mesh_xy):
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
    from file_io.restart_bundle import _find_restart_file, read_metadata
    from file_io.centroids import load_centroids
    from gw.lr_channels import slab_shells
    from common.collectives import device_put_process_local
    from .bse_loading import load_bse_data_from_restart_sharded
    from .bse_window import resolve_n_occ
    import distrib_la as dla

    _stage("restart_load")
    restart = _find_restart_file(input_file)
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
    b2l = grid_index(kpts, grid)
    if len(set(b2l.tolist())) != nk:
        raise ValueError("GATE bsemat_k_bijection: WFN_co k points do not cover the grid once")
    with h5py.File(wfn_co, "r") as w:
        el_all = np.asarray(w["mf_header/kpoints/el"][()]).reshape(-1, nk, int(w["mf_header/kpoints/mnband"][()]))[0]
        ecutwfc = float(w["mf_header/kpoints/ecutwfc"][()])

    _stage("head_columns")
    # ── 2. head momenta and ζ columns ─────────────────────────────────────
    with h5py.File(restart, "r") as f:
        tables = read_tables(f, "W0_qmunu")
    with h5py.File(wfn_co, "r") as w:
        mtrx = np.asarray(w["mf_header/symmetry/mtrx"][()])[:int(w["mf_header/symmetry/ntran"][()])]
    if bgw_eps:
        with h5py.File(bgw_eps, "r") as e:
            q_wedge = np.asarray(e["eps_header/qpoints/qpts"][()], dtype=np.float64)
        q_src = f"BerkeleyGW eps q list ({bgw_eps})"
    else:
        q_wedge = np.asarray(tables.q_irr_frac, dtype=np.float64)
        q_src = "restart q wedge"
    q_list = bgw_kernel_q_list(q_wedge, mtrx, bdot)
    K, qlen = pair_head_momenta(kpts, bdot, q_list)
    iq_pair = np.empty((nk, nk), np.int64)
    keys, key_of = [], {}
    from symmetry_maps import bgw_integer_q_to_fractional
    q_full_L = bgw_integer_q_to_fractional(fft_int, grid)
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
    face = NamedSharding(mesh_xy, P("x", "y"))
    tile = jax.jit(lambda a, i: jnp.take(a, i, axis=2), out_shardings=face)
    plan = dla.plan("eigh", mesh_xy, backend="distributed" if jax.process_count() > 1 else "native",
                    n=n_rmu_pad)
    herm = jax.jit(lambda a: 0.5 * (a + jnp.conj(a).T), out_shardings=face)
    W11 = np.zeros(len(keys))
    y_keys = np.zeros_like(z_keys)
    figs = []
    eig_cache = {}
    for kk, (iq, G) in enumerate(keys):
        if iq == 0:
            figs.append({"iq": 0, "G": G, "q0": True})
            continue
        Kc = (q_full_L[iq] + np.asarray(G)) @ bvec
        if abs(np.linalg.norm(Kc) - np.linalg.norm(K_ch[iq] @ bvec)) > 1e-8 * np.linalg.norm(Kc):
            raise ValueError(f"GATE bsemat_head_length: key {(iq, G)} is not a shortest q+G")
        Vq = herm(tile(V_flat, iq))
        if iq not in eig_cache:
            eig_cache = {iq: plan(Vq)}
        lam, U = eig_cache[iq]
        zr = device_put_process_local(z_keys[kk], NamedSharding(mesh_xy, P()))
        sp = A.head_split(Vq, tile(W_flat, iq), zr, float(v_head_q[iq]),
                          eigh=lambda _m, _l=lam, _u=U: (_l, _u), xp=jnp, rcond=rcond)
        A.check_split(sp, where=f"q {iq} G {G}")
        W11[kk] = sp.W11
        y_keys[kk] = np.asarray(rep(sp.y))
        figs.append({"iq": int(iq), "G": G, "qlen": float(np.linalg.norm(Kc)),
                     "v1": float(v_head_q[iq]), **sp.figures})
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
    if use_gauge_map:
        Ug, gfig = gauge_map(psi_win, wfn_co=wfn_co, k_rows=b2l, bands_abs=bands_abs,
                             centroid_frac=cent_frac, fft_grid=wfn.fft_grid, nv=n_val,
                             enk_win=enk_win)
    else:
        Ug = np.broadcast_to(np.eye(n_val + n_cond), (nk, n_val + n_cond, n_val + n_cond)).copy()
        gfig = {"gauge_map": "identity (red twin)"}
    _log(f"[bsemat] gauge map: {gfig}")

    # q = 0 head column check against the bundle's G = 0 vector.
    q0 = [kk for kk, (iq, G) in enumerate(keys) if iq == 0 and G == (0, 0, 0)][0]
    if data.get("g0_X") is not None:
        g0 = np.asarray(rep(data["g0_X"]))[:n_rmu]
        dz = np.linalg.norm(z_keys[q0][:n_rmu] * np.sqrt(omega) - g0) / np.linalg.norm(g0)
        _log(f"[bsemat] q=0 column vs bundle G0 vector: rel. diff {dz:.2e}")

    nocc_abs = int(bands_abs[n_val])
    efermi = 0.5 * (el_all[:, :nocc_abs].max() + el_all[:, nocc_abs:].min()) * 13.605693122994
    _stage(None)
    data["W_q"] = data["V_q_full"] = None            # the flattened copies above stay resident
    return dict(restart=restart, sys_dim=sys_dim, bgw_eps=bgw_eps, wfn=wfn, wfn_co=wfn_co,
                mesh_xy=mesh_xy, rep=rep, n_val=n_val, n_cond=n_cond, nk=nk, grid=grid,
                n_rmu=n_rmu, n_rmu_pad=n_rmu_pad, nspinor=int(psi_v.shape[2]), omega=omega,
                bvec=bvec, kpts=kpts, kgrid_w=kgrid_w, b2l=b2l, ecutwfc=ecutwfc, efermi=efermi,
                keys=keys, n_ties=n_ties, iq_pair=iq_pair, key_pair=key_pair,
                lam_pair=qlen if sys_dim == 3 else np.ones_like(qlen),
                z_keys=z_keys, y_keys=y_keys, W11=W11, figs=figs, v_head_q=v_head_q,
                W_flat=W_flat, V_flat=V_flat, Vbar=data["V_q0"], psi_win=psi_win,
                use_gauge_map=use_gauge_map, Ug=Ug, gfig=gfig, q0_dipole={},
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
            Ug, gfig = ctx["Ug"], ctx["gfig"]
            if ctx["use_gauge_map"] and not out["use_gauge_map"]:     # red twin beside mapped files
                nb = n_val + n_cond
                Ug = np.broadcast_to(np.eye(nb), (nk, nb, nb)).copy()
                gfig = {"gauge_map": "identity (red twin)"}
                _log(f"[bsemat] gauge map: {gfig}")
            # ψ_file,n = Σ_m ψ_LX,m conj(U[n, m]); rows of the file in WFN_co order.
            psi_map = np.einsum("knsr,kmn->kmsr", ctx["psi_win"][ctx["b2l"]], np.conj(Ug))
            psi_v_b = psi_map[:, :n_val][:, ::-1]                     # BerkeleyGW: iv = 1 highest
            psi_c_b = psi_map[:, n_val:]
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
                    ecutg=ctx["ecutwfc"], efermi=ctx["efermi"], nspinor=ctx["nspinor"])
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
            "split": ctx["figs"], "gauge": gfig, "vbar_imag_rel": vim,
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
