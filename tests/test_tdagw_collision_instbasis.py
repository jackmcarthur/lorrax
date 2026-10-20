"""CPU gates of the collision term's non-diagonal G builder and instantaneous basis (TD-aGW §1.5).

Runs as a script on a 2x2 CPU mesh, on the GN-PPM synthetic magnet of
``tests/test_tdagw_dynsig_sigma.py`` (3 x 2 x 1 k, 9 bands, the 2v + 2c window [3, 7))
with the GW driver's own W0, W(i w_p) and pole fit, as
``tests/test_tdagw_collision_fullrho.py`` (whose rows check the symmetric pairing):

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/test_tdagw_collision_instbasis.py

IB  U = 1, diagonal rho' (an excited population state): both pairings == the
    populations' sweep (pair, both branches, static term, static limit)  rel <= 1e-12
ID  vertex pairing at a random Hermitian rho' (spectrum in [0, 1]): both branches
    and the static term == the dense reference, Sigma_ij(e_i) with the internal energy
    of the index on i's vertex and Sigma_ij(e_j) with that on j's       rel <= 1e-5
IE  covariance at a unitary mixing degenerate window states (e_v1 = e_v2, e_c1 = e_c2
    per k), both pairings: U^H X[rho] U == X'[U^H rho U], X = gamma, injection
                                                                         rel <= 1e-10
IF  sum_k Tr C[rho] / flux on reciprocal poles, active intermediates:
    vertex code <= 1e-6; dense vertex reference <= 1e-12; the symmetric
    comparison is a red twin and exceeds the vertex tolerance by > 100x
IG  gamma and injection Hermitian, both pairings                        abs == 0
IS  the full-rho snapshot basis on a synthetic linear dSigma: at rho_eq U = 1,
    E = e_QP, f = f_eq (one evaluation); at an excited rho,
    [diag(e_QP)+K(rho-rho_eq)] U = U diag(E), f = diag(U^H rho U)       <= 1e-9
IC  ``tdagw.dynsig.window_collision`` end to end on hand-built propagator operators (2
    trials, TD k order permuted against the bundle's, a synthetic linear dSigma), vertex
    pairing: with dSigma = 0 and e_QP = eps^G the equilibrium instantaneous basis is
    the band basis (U = 1; t0 held operators and dissipator)           rel <= 1e-12;
    with dSigma != 0 and e_QP != eps^G a refresh at rho_eq reproduces t0 (dissipator)
    rel <= 1e-12 and holds Delta Sigma^h = 0 (<= 1e-9 meV), and a refresh at an excited
    rho converges its basis (residual <= DRESS_TOL)
IH  the q -> 0 head of a non-diagonal line (``gw.ppm_sigma._line_head``, a synthetic
    ``HeadSigmaDiag``): at a diagonal C it is the populations' head rows (hole, particle,
    static) and zero off the diagonal; covariant at a unitary mixing degenerate states
                                                                         rel <= 1e-14
IW  the walls of the vertex and symmetric sweeps over the populations' (reported)

The equilibrium path's bitwise identity with the previous tree and the symmetric
pairing's agreement with the pseudo-band carrier it replaces are measured by dumping
the same calls from both trees (the track's ``ib_gate_dump.py``).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runtime  # noqa: F401  (before numpy)
import numpy as np

import test_tdagw_collision_fullrho as FR
import test_tdagw_dynsig_sigma as F

GATES = dict(IB=1e-12, ID=1e-5, IE=1e-10, IF_code_vertex=1e-6, IF_dense=1e-12, IG=0.0, IS=1e-9,
             IC_band=1e-12, IC_eq=1e-12, IC_eq_held_mev=1e-9, IC_basis=0.0, IH=1e-14)
_rel = FR._rel


def _dense_vertex(psi_g, e_g, M_int, proj, e_proj, poles, eta, branch):
    """(2, nk, nP, nP) pair table of one branch with the vertex pairing (Ry).

    ``FR._dense_full``'s sums with the internal energy of m (the index on i's vertex) in
    Sigma_ij(e_i) and that of n (on j's) in Sigma_ij(e_j).
    """
    Om, B, D, good, Wc0 = poles
    nk = psi_g.shape[0]
    kmq = F._kmq()
    Rp, Rm = np.where(good, B + D, 0.0), np.where(good, B - D, 0.0)
    nP = proj.shape[1]
    S = np.zeros((2, nk, nP, nP), complex)
    for k in range(nk):
        for q in range(nk):
            kb = kmq[q, k]
            a = np.einsum("isu,asu->iau", np.conj(proj[k]), psi_g[kb])
            M = M_int[kb]
            for slot in (0, 1):
                for i in range(nP):
                    for j in range(nP):
                        E = e_proj[k, i] if slot == 0 else e_proj[k, j]
                        if branch == "val":
                            K = Rm[q][None] / (E - e_g[kb][:, None, None] + Om[q][None] - 1j * eta)
                        else:
                            K = Rp[q][None] / (E - e_g[kb][:, None, None] - Om[q][None] + 1j * eta)
                        if slot == 0:
                            b = np.einsum("mn,nv->mv", M, np.conj(a[j]))
                            val = np.einsum("mu,muv,mv->", a[i], K, b)
                        else:
                            c = np.einsum("mu,mn->nu", a[i], M)
                            val = np.einsum("nu,nuv,nv->", c, K, np.conj(a[j]))
                        S[slot, k, i, j] += val / nk
    return S


def _slow_basis_rows():
    """IS: the collision basis map on a synthetic linear dSigma (no mesh)."""
    import jax.numpy as jnp
    from tdagw import dynsig as D
    rng = np.random.default_rng(20261019)
    nk, nb, ntr = 3, 4, 2
    e_qp = np.sort(rng.uniform(-0.3, 0.3, (nk, nb)), axis=1)
    e_qp[1] = e_qp[1][[1, 0, 2, 3]]                    # an unsorted k: the order is tracked
    f_eq = np.tile(np.array([1.0, 1.0, 0.0, 0.0]), (nk, 1))
    rho_eq = np.einsum("ki,ij->kij", f_eq, np.eye(nb)).astype(complex)
    A = rng.standard_normal((nk,) + (nb,) * 4) + 1j * rng.standard_normal((nk,) + (nb,) * 4)
    K = 0.02 * (A + np.conj(np.transpose(A, (0, 2, 1, 4, 3))))   # Hermitian in, Hermitian out
    apply = lambda params, rho, t: jnp.einsum("kabcd,tkcd->tkab", params,
                                              rho - jnp.asarray(rho_eq)[None])
    step = D.slow_basis_step(apply, jnp.asarray(rho_eq), jnp.asarray(e_qp), jnp.asarray(f_eq))
    rows = {}
    r0 = np.broadcast_to(rho_eq, (ntr, nk, nb, nb)).copy()
    U, E, f, _, rec = D.collision_basis_state(step, jnp.asarray(K), jnp.asarray(r0), 0.0)
    U, E, f = D._track_bands(U, E, f)
    rows["IS_eq"] = max(float(np.max(np.abs(U - np.eye(nb)))), float(np.max(np.abs(E - e_qp))),
                        float(np.max(np.abs(f - f_eq))), float(rec["residual"]))
    rows["IS_eq_iterations"] = rec["iterations"]
    rho = np.stack([0.85 * rho_eq + 0.15 * FR._random_rho(rng, nk, nb, coh=0.3)
                    for _ in range(ntr)])
    rho = FR._herm(rho)
    rec_prod = D.collision_basis_state(step, jnp.asarray(K), jnp.asarray(rho), 0.0)[4]
    rows["IS_production_tol"] = dict(tol=D.DRESS_TOL, **rec_prod)
    U, E, f, _, rec = D.collision_basis_state(step, jnp.asarray(K), jnp.asarray(rho), 0.0,
                                              steps=64, tol=1e-12)
    U, E, f = D._track_bands(U, E, f)
    Uh = np.conj(np.swapaxes(U, -1, -2))
    H = np.einsum("ki,ij->kij", e_qp, np.eye(nb))[None] + np.asarray(apply(K, rho, 0.0))
    rows["IS_fixed_point"] = max(float(np.max(np.abs(H @ U - U * E[..., None, :]))),
                                 float(np.max(np.abs(f - np.real(np.einsum("tkii->tki", Uh @ rho @ U))))))
    rows["IS_residual"] = float(rec["residual"])
    rows["IS_iterations"] = rec["iterations"]
    rows["IS_max_u_minus_1"] = float(np.max(np.abs(U - np.eye(nb))))
    rows["IS"] = max(rows["IS_eq"], rows["IS_fixed_point"], 0.0 if rec["residual"] <= 1e-10 else 1.0)
    return rows


def _head_rows():
    """IH: ``_line_head`` against ``_head_branch_rows`` (no mesh)."""
    from gw import ppm_sigma as PS
    from gw.head_correction import HeadSigmaDiag
    rng = np.random.default_rng(20261023)
    nk, nb, b_lo, n = 3, 7, 2, 4
    win = slice(b_lo, b_lo + n)
    e = np.sort(rng.uniform(-0.5, 0.5, (nk, nb)), axis=1)
    e[:, b_lo:b_lo + 2] = e[:, b_lo:b_lo + 2].mean(axis=1, keepdims=True)    # degenerate pairs
    e[:, b_lo + 2:b_lo + 4] = e[:, b_lo + 2:b_lo + 4].mean(axis=1, keepdims=True)
    f = rng.uniform(0.0, 1.0, (nk, nb))
    head = HeadSigmaDiag.build(omega_grid_ry=np.linspace(-1, 1, 5), enk_ry=e, efermi_ry=0.0,
                               occupations=f, poles_ry=np.array([0.7 - 0.01j, 1.3 - 0.02j]),
                               residues_ry=np.array([0.2, 0.05]), cell_volume=300.0, nk_tot=nk)
    table = e + 0.003 * rng.standard_normal((nk, nb))          # per-row evaluation energies
    table[:, b_lo:b_lo + 2] = table[:, b_lo:b_lo + 2].mean(axis=1, keepdims=True)
    table[:, b_lo + 2:b_lo + 4] = table[:, b_lo + 2:b_lo + 4].mean(axis=1, keepdims=True)
    fw = f[:, win]
    D_h = np.einsum("ki,ij->kij", fw, np.eye(n)).astype(complex)
    hv, hc, hs = PS._line_head(head, table, win, D_h, np.eye(n)[None] - D_h)
    hole, particle, static = PS._head_branch_rows(head, table, win)
    dg = lambda X: np.einsum("...ii->...i", X)
    off = lambda X: float(np.max(np.abs(X - dg(X)[..., None] * np.eye(n))))
    rows = dict(IH_diag=max(_rel(dg(hv[0]), hole), _rel(dg(hv[1]), hole),
                            _rel(dg(hc[0]), particle), _rel(dg(hc[1]), particle),
                            _rel(dg(hs), static)),
                IH_offdiag=max(off(hv[0]), off(hc[1]), off(hs)))
    rho = FR._random_rho(rng, nk, n, coh=1.0)
    V = np.zeros((nk, n, n), complex)
    for k in range(nk):
        for blk in (slice(0, 2), slice(2, 4)):
            a = rng.standard_normal((2, 2)) + 1j * rng.standard_normal((2, 2))
            V[k, blk, blk] = np.linalg.qr(a)[0]
    Vh = np.conj(np.swapaxes(V, -1, -2))
    A = PS._line_head(head, table, win, rho, np.eye(n)[None] - rho)
    B = PS._line_head(head, table, win, Vh @ rho @ V, np.eye(n)[None] - Vh @ rho @ V)
    rows["IH_cov"] = max(max(_rel(Vh @ A[i][s_] @ V, B[i][s_]) for s_ in (0, 1)) for i in (0, 1))
    rows["IH_cov"] = max(rows["IH_cov"], _rel(Vh @ A[2] @ V, B[2]))
    rows["IH"] = max(rows["IH_diag"], rows["IH_offdiag"], rows["IH_cov"])
    return rows


def _collision_rows(mesh, wfns, meta, poles, cfg, splan, kplan, e, f, rng):
    """IC: ``window_collision`` end to end, the propagator's operators built by hand."""
    from typing import NamedTuple
    import jax
    import jax.numpy as jnp
    from common.collectives import gather_to_host
    from common.units import RYD_TO_EV
    from tdagw import collision as C
    from tdagw import dynsig as D
    from tdagw import propagate as tp
    from tdagw.self_energy import SelfEnergyModel
    from tdagw.tdgw_io import HA_IN_EV

    class Ops(NamedTuple):
        e_qp: object
        rho_eq: object
        w_k: object
        collision: object = None

    B_LO, NA = F.B_LO, F.NA
    nk = int(np.prod(F.KGRID))
    nk_pad, ntr = 8, 2
    order = rng.permutation(nk)                       # TD k i is bundle row order[i]
    k_td = np.asarray(kplan.k_parent_frac, np.float64)[order]
    e_w, f_w = e[:, B_LO:B_LO + NA][order], f[:, B_LO:B_LO + NA][order]
    rho_eq = np.einsum("ki,ij->kij", f_w, np.eye(NA)).astype(complex)
    place = lambda a, ax=0: tp._place(tp._pad_k(a, ax, nk_pad), mesh, ax)
    blocks = lambda a: tp.place_window_blocks(a, mesh, nk_pad)
    A = rng.standard_normal((nk,) + (NA,) * 4) + 1j * rng.standard_normal((nk,) + (NA,) * 4)
    K_h = 0.004 * (A + np.conj(np.transpose(A, (0, 2, 1, 4, 3))))

    def setup(dsigma, shift):
        e_qp = (e_w + shift) * RYD_TO_EV / HA_IN_EV                    # Ha, the header's
        ops = Ops(e_qp=place(e_qp), rho_eq=place(rho_eq), w_k=place(np.full(nk, 1.0 / nk)))
        K = place(K_h * dsigma)
        req = ops.rho_eq

        def apply(params, rho, t):
            kernel, reference = params
            return jnp.einsum("kabcd,tkcd->tkab", kernel, rho - reference[None])
        return ops, SelfEnergyModel(apply=apply, params=(K, req), linear=True, name="synthetic K")

    src = dict(wfns=wfns, meta=meta, poles=poles, plan=splan, config=cfg, head=None,
               request=None, efermi_ry=None, member="gn_ppm")

    def build(basis, pairing, ops, model):
        return D.window_collision(
            src, ops, mesh=mesh, k_frac=k_td, b_lo=B_LO, ntr=ntr, form="subtracted",
            hermitian="dynamic", policy=C.RefreshPolicy(every=1.0), sweep="active",
            in_scattering="full", rate_intermediates="active", basis=basis, pairing=pairing,
            model=model, say=lambda line: None)

    def refresh(built, model, rho, t):
        recs = []
        built["state"]["t_last"] = 0.0
        out = built["refresh"](t, blocks(rho), built_ops[0], (model.params, built["held"]))
        recs.append(built["state"]["history"][-1])
        leaves = [np.asarray(gather_to_host(x)) for x in
                  jax.tree_util.tree_leaves(out["ops"].collision)]
        return dict(held=np.asarray(gather_to_host(out["params"][1])), diss=leaves, rec=recs[-1])

    rows = {}
    rho_x = np.stack([0.9 * rho_eq + 0.1 * FR._herm(FR._random_rho(rng, nk, NA, coh=1.0))
                      for _ in range(ntr)])
    rel_list = lambda a, b: max(_rel(x, y) for x, y in zip(a, b))
    built_ops = [None]
    ops, model = setup(0.0, 0.0)
    built_ops[0] = ops
    ba = build("band", "vertex", ops, model)
    bi = build("instantaneous", "vertex", ops, model)
    diss_host = lambda b: [np.asarray(gather_to_host(x))
                          for x in jax.tree_util.tree_leaves(b["diss"])]
    rows["IC_band_vertex"] = max(
        rel_list(diss_host(bi), diss_host(ba)),
        *[_rel(bi["eq"][key], ba["eq"][key]) for key in ("gamma", "injection", "m_h", "static")])
    rows["IC_t0_basis_vertex"] = bi["basis_t0"]
    rows["IC_band"] = rows["IC_band_vertex"]
    shift = np.where(f_w > 0.5, -0.02, 0.05)                      # Ry, a QP-like scissor
    ops, model = setup(1.0, shift)
    built_ops[0] = ops
    bv = build("instantaneous", "vertex", ops, model)
    t0_diss = [np.asarray(gather_to_host(x)) for x in jax.tree_util.tree_leaves(bv["diss"])]
    req = refresh(bv, model, np.broadcast_to(rho_eq, (ntr,) + rho_eq.shape).copy(), 2.0)
    rows["IC_eq"] = rel_list(req["diss"], t0_diss)
    rows["IC_eq_held_mev"] = float(req["rec"]["held_max_mev"])
    rx = refresh(bv, model, rho_x, 4.0)
    b = rx["rec"]["basis"]
    rows["IC_excited"] = dict(basis=b, edges=rx["rec"]["edges"], frozen_s=rx["rec"]["frozen_s"],
                              active_s=rx["rec"]["active_s"], wall_s=rx["rec"]["wall_s"],
                              held_max_mev=rx["rec"]["held_max_mev"],
                              frozen_energies=bv["frozen_energies"])
    rows["IC_basis"] = 0.0 if (b["residual"] <= D.DRESS_TOL and b["max_u_minus_1"] > 0.0) else 1.0
    return rows


def run_checks(mesh):
    import jax
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding
    from common.collectives import device_put_process_local
    from common.units import RYD_TO_EV
    from ffi import fft as FF
    from symmetry_maps import q_negation_index
    from gw import ppm_sigma as PS
    from gw.minimax_screening import build_imag_probe_response_rule, build_static_quadrature
    from gw.w_isdf import compute_chi0, compute_chi0_imag_ordered, solve_w
    from tdagw import collision as C
    from tdagw import dynsig as D
    F.RYD = RYD_TO_EV
    KG, B_LO, NA, NMU = F.KGRID, F.B_LO, F.NA, F.NMU
    rng = np.random.default_rng(20261017)
    psi, e, f, V, zeta, v = F._synthetic(rng)
    nk = int(np.prod(KG))
    kf = np.stack(np.unravel_index(np.arange(nk), KG), axis=1) / np.asarray(KG, float)
    kplan, perm = F._plan(mesh, kf, rng)
    psi, V = psi[..., perm], V[:, perm][:, :, perm]
    gather = lambda x: np.asarray(multihost_utils.process_allgather(x, tiled=True))
    putc = lambda a: device_put_process_local(np.ascontiguousarray(a), NamedSharding(mesh, D.CHI_SPEC))
    q_neg = np.asarray(q_negation_index(KG))
    z = 1j * F.OMEGA_P
    rows = _slow_basis_rows()
    rows.update(_head_rows())
    rng2 = np.random.default_rng(20261020)
    with FF.xla_reference():
        wfns, meta = F._gw_bundle(mesh, psi, e, f, kplan)
        mm = __import__("types").SimpleNamespace(target_error=F.TARGET, max_nodes=64,
                                                 energy_reference="midgap")
        quad, e_ref = build_static_quadrature(wfns, mm)
        quad_p = build_imag_probe_response_rule(quad, F.OMEGA_P, mm, with_odd_kernel=True)
        V_dev = putc(V)
        W0_g = solve_w(V_dev, compute_chi0(wfns, quad, meta, mesh, energy_reference=e_ref), meta, mesh)
        Wp_g = solve_w(V_dev, compute_chi0_imag_ordered(wfns, quad_p, meta, mesh, q_neg_index=q_neg,
                                                        energy_reference=e_ref), meta, mesh)
        rule = D.static_rule(e, f, target_error=F.TARGET, max_nodes=64)
        plan = PS.pole_fit_plan(V_dev, z, mesh, n_mu_logical=NMU, fallback_omega=2.0, ordered=True,
                                coarsen_extreme_tails=True, q_neg_index=q_neg,
                                n_nodes_static=len(rule["tau"]))
        poles = PS.fit_pole_model(W0_g, Wp_g, plan)
        po = {k_: gather(getattr(poles, k_)) for k_ in ("Omega_q", "B_q", "B_odd_q", "valid_mask_q")}
        dpo = (po["Omega_q"], po["B_q"], po["B_odd_q"], po["valid_mask_q"], gather(poles.Wc0_q))
        eta = F.ETA_EV / RYD_TO_EV
        omega_ry = np.arange(-120, 121) * 0.25 / RYD_TO_EV
        splan = PS.window_sigma_plan(F._config(omega_ry))
        eye = np.broadcast_to(np.eye(NA, dtype=complex), (nk, NA, NA)).copy()
        e_w = e[:, B_LO:B_LO + NA]
        kw = dict(meta=meta, mesh_xy=mesh, branches=True, static_limit=True)

        def sweep(win, internal="active", pairing="symmetric"):
            t = time.perf_counter()
            out = PS.window_sigma_c(wfns, poles, win, splan, internal=internal, pairing=pairing, **kw)
            out["wall_s"] = time.perf_counter() - t
            return out

        def operators(out):
            return tuple(np.asarray(g) for g in C.branch_operators(
                out["branch_pair"]["cond"], out["branch_pair"]["val"])[:2])

        def tables_rel(a, b):
            parts = [_rel(a["pair"], b["pair"])]
            parts += [_rel(a["branch_pair"][n], b["branch_pair"][n]) for n in ("cond", "val")]
            st_b = b["branch_pair"]["static"]
            parts += [float(np.max(np.abs(a["branch_pair"]["static"] - st_b))
                            / max(float(np.max(np.abs(st_b))), 1e-300))]
            sl = lambda o: o["static_limit"]["sx"] + np.einsum(
                "ki,ij->kij", o["static_limit"]["head"], np.eye(NA))
            parts += [_rel(sl(a), sl(b))]
            return parts

        # ---- IC: window_collision end to end (fitted poles)
        rows.update(_collision_rows(mesh, wfns, meta, poles, F._config(omega_ry), splan, kplan,
                                    e, f, np.random.default_rng(20261022)))

        # ---- IB: U = 1, diagonal rho' == populations, both pairings
        f_x = np.clip(f[:, B_LO:B_LO + NA] + np.array([-0.06, -0.035, 0.05, 0.045])
                      * (1.0 + 0.3 * rng2.standard_normal((nk, NA))), 0.0, 1.0)
        win_p = dict(b_lo=B_LO, U=eye, e=e_w, f=f_x)
        pop = sweep(win_p)
        dia = dict(win_p, rho=np.einsum("ki,ij->kij", f_x, np.eye(NA)).astype(complex))
        rows["IB_parts"] = {pr: tables_rel(sweep(dia, pairing=pr), pop)
                            for pr in ("vertex", "symmetric")}
        rows["IB"] = max(max(v_) for v_ in rows["IB_parts"].values())
        rows["IW_populations_s"] = min(sweep(win_p)["wall_s"] for _ in range(2))
        for pr in ("vertex", "symmetric"):
            rows[f"IW_{pr}_s"] = min(sweep(dia, pairing=pr)["wall_s"] for _ in range(2))
        rows["IW_vertex_sweeps"] = len(sweep(dia, pairing="vertex")["sweep"]["lines"])

        # ---- ID / IG: vertex vs the dense reference (fitted poles)
        rho = FR._random_rho(rng2, nk, NA, coh=1.0)
        f_r = np.real(np.einsum("kii->ki", rho))
        win_r = dict(b_lo=B_LO, U=eye, e=e_w, f=f_r, rho=rho)
        e_g = e.copy()
        e_g[:, B_LO:B_LO + NA] = e_w
        M_h = np.zeros((nk, F.NBF, F.NBF), complex)
        M_h[:, B_LO:B_LO + NA, B_LO:B_LO + NA] = rho
        M_p = np.zeros_like(M_h)
        M_p[:, B_LO:B_LO + NA, B_LO:B_LO + NA] = np.eye(NA)[None] - rho
        proj = psi[:, B_LO:B_LO + NA]
        vx = sweep(win_r, pairing="vertex")
        ref = dict(val=_dense_vertex(psi, e_g, M_h, proj, e_w, dpo, eta, "val"),
                   cond=_dense_vertex(psi, e_g, M_p, proj, e_w, dpo, eta, "cond"),
                   static=FR._dense_full(psi, e_g, M_h, proj, e_w, dpo, eta, "static")[0])
        rows["ID_parts"] = [_rel(vx["branch_pair"][n], ref[n]) for n in ("cond", "val", "static")]
        rows["ID"] = max(rows["ID_parts"])
        sym = sweep(win_r, pairing="symmetric")
        rows["IG"] = max(float(np.max(np.abs(g - np.conj(np.swapaxes(g, -1, -2)))))
                         for o in (vx, sym) for g in operators(o))
        rows["ID_vertex_vs_symmetric"] = _rel(vx["pair"], sym["pair"])

        # ---- IE: covariance at a unitary mixing degenerate window states
        e_d = e_w.copy()
        e_d[:, 0:2] = e_w[:, 0:2].mean(axis=1, keepdims=True)
        e_d[:, 2:4] = e_w[:, 2:4].mean(axis=1, keepdims=True)
        Ud = np.zeros((nk, NA, NA), complex)
        for k in range(nk):
            for blk in (slice(0, 2), slice(2, 4)):
                a = rng2.standard_normal((2, 2)) + 1j * rng2.standard_normal((2, 2))
                Ud[k, blk, blk] = np.linalg.qr(a)[0]
        Uh = np.conj(np.swapaxes(Ud, -1, -2))
        rho_u = Uh @ rho @ Ud
        ie = {}
        for pr in ("vertex", "symmetric"):
            ga = operators(sweep(dict(b_lo=B_LO, U=eye, e=e_d, f=f_r, rho=rho), pairing=pr))
            gb = operators(sweep(dict(b_lo=B_LO, U=Ud, e=e_d, f=np.real(np.einsum("kii->ki", rho_u)),
                                  rho=rho_u), pairing=pr))
            ie[pr] = max(_rel(Uh @ ga[i] @ Ud, gb[i]) for i in range(2))
        rows["IE_parts"] = ie
        rows["IE"] = max(ie.values())

        # ---- IF: the closed-window trace identity on reciprocal poles
        neg = lambda a: np.swapaxes(a[q_neg], -1, -2)
        good = po["valid_mask_q"]
        g_s = good & neg(good)
        Om_s = 0.5 * (po["Omega_q"] + neg(po["Omega_q"]))
        B_s = 0.5 * (po["B_q"] + neg(po["B_q"]))
        D_s = 0.5 * (po["B_odd_q"] - neg(po["B_odd_q"]))
        dev = lambda name, a: jax.device_put(np.asarray(a, dtype=np.asarray(po[name]).dtype),
                                             getattr(poles, name).sharding)
        fields = dict(Omega_q=dev("Omega_q", np.where(g_s, Om_s, po["Omega_q"])),
                      B_q=dev("B_q", np.where(g_s, B_s, 0.0)),
                      B_odd_q=dev("B_odd_q", np.where(g_s, D_s, 0.0)),
                      valid_mask_q=dev("valid_mask_q", g_s))
        poles = (poles._replace(**fields) if hasattr(poles, "_replace")
                 else __import__("dataclasses").replace(poles, **fields))
        dpo = (np.asarray(np.where(g_s, Om_s, po["Omega_q"])), np.where(g_s, B_s, 0.0),
               np.where(g_s, D_s, 0.0), g_s, dpo[4])
        for pr in ("vertex", "symmetric"):
            out = sweep(win_r, pairing=pr)
            rc, flux = FR._trace_ratio(*operators(out), rho)
            rows[f"IF_code_{pr}"] = rc
        rows["IF_flux_mev"] = flux * RYD_TO_EV * 1e3
        refv = dict(val=_dense_vertex(psi, e_g, M_h, proj, e_w, dpo, eta, "val"),
                    cond=_dense_vertex(psi, e_g, M_p, proj, e_w, dpo, eta, "cond"))
        rows["IF_dense"] = FR._trace_ratio(
            *FR._dense_operators(refv["cond"], refv["val"])[:2], rho)[0]
        rows["RI_symmetric_nonconserving"] = rows["IF_code_symmetric"]
    return rows


def verdict(rows):
    out = {g: rows[g] <= tol for g, tol in GATES.items()}
    out["RI"] = rows["RI_symmetric_nonconserving"] > 100.0 * GATES["IF_code_vertex"]
    return out


def main() -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    from common.collectives import resolve_mesh
    mesh = resolve_mesh()
    if tuple(mesh.devices.shape) != (2, 2):
        print(f"SKIP: needs a 2x2 mesh, got {tuple(mesh.devices.shape)}", flush=True)
        return 0
    t0 = time.perf_counter()
    rows = run_checks(mesh)
    ok = verdict(rows)
    print("RESULT " + json.dumps(dict(rows=rows, ok=ok, all=all(ok.values()),
                                      wall_s=time.perf_counter() - t0,
                                      platform=jax.devices()[0].platform), default=float),
          flush=True)
    return 0 if all(ok.values()) else 1


def test_tdagw_collision_instbasis_needs_2x2_mesh():
    import pytest
    pytest.skip("script on a 2x2 mesh: JAX_PLATFORMS=cpu XLA_FLAGS="
                "--xla_force_host_platform_device_count=4 python3 tests/test_tdagw_collision_instbasis.py")


if __name__ == "__main__":
    sys.exit(main())
