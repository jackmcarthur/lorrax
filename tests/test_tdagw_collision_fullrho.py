"""CPU gates of the full-ρ in-scattering and the active-only rate intermediates (TD-aGW §1.5).

Runs as a script on a 2x2 CPU mesh, on the GN-PPM synthetic magnet of
``tests/test_tdagw_dynsig_sigma.py`` (3 x 2 x 1 k, 9 bands: 3 frozen occupied,
the 2v + 2c window [3, 7), 2 frozen empty; 2 spinors, 8 centroids), the GW
driver's own W0, W(i w_p) and pole fit:

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/test_tdagw_collision_fullrho.py

FA  diagonal rho' (an excited population state): window_sigma_c(rho = diag f) ==
    the populations' sweep for the pair, both branches, the static term and the
    static limit                                                        rel <= 1e-12
FB  gamma and injection of the full-ρ sweep are Hermitian               abs == 0
FD  random Hermitian rho' (spectrum in [0, 1], coherences O(0.3)): both branches
    and the static term == the dense full-ρ reference (hole rho_mn, particle
    (1 - rho)_mn, 1/2 g(e_m) + 1/2 g(e_n); the static term -sum rho_mn <i m| W_s
    |n j>)                                                              rel <= 1e-5
RC  the earlier symmetric internal ordering is retained as a red twin.  With
    the corrected column-sampled external operator it is not conserving even
    on reciprocal poles (code and dense residual/flux both > 1e-2).  The
    conserving full-rho vertex gate is IF in test_tdagw_collision_instbasis.py.
    The fitted-pole and comparison rows remain diagnostic.
FU  covariance at a unitary U mixing degenerate window states (e_v1 = e_v2,
    e_c1 = e_c2 per k): U^H X[rho] U == X'[U^H rho U], X = gamma, injection
                                                                         rel <= 1e-10;
    the populations' internal line fails it (control variant, > 100 x)
FW  the wall of an active full-ρ sweep over the populations' one (reported)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runtime  # noqa: F401  (before numpy)
import numpy as np

import test_tdagw_dynsig_sigma as F

GATES = dict(FA=1e-12, FB=0.0, FD=1e-5, FU=1e-10)


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(float(np.max(np.abs(b))), 1e-300))


def _herm(a):
    return 0.5 * (a + np.conj(np.swapaxes(a, -1, -2)))


def _random_rho(rng, nk, n, coh=1.0):
    """Hermitian rho per k with spectrum in [0, 1]; ``coh`` scales the off-diagonal part."""
    out = np.zeros((nk, n, n), complex)
    for k in range(nk):
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        v, _ = np.linalg.qr(a)
        lam = rng.uniform(0.0, 1.0, n)
        r = (v * lam) @ np.conj(v.T)
        d = np.diag(np.diag(r))
        out[k] = d + coh * (r - d)
    return out


def _dense_full(psi_g, e_g, M_int, proj, e_proj, poles, eta, branch, rule="sym"):
    """(2, nk, nP, nP) pair table of one branch with a matrix internal weight (Ry).

    ``M_int`` (nk, nb, nb): the internal line's weight on every band (hole rho, particle
    1 - rho on the window; zero elsewhere); ``rule``: ``sym`` 1/2 (g(e_m) + g(e_n)),
    ``left`` g(e_m).  ``branch``: ``val`` (R_-, time ordered) | ``cond`` (R_+,
    retarded) | ``static`` (-W_s on the invalid modes, frequency independent).
    """
    Om, B, D, good, Wc0 = poles
    nk = psi_g.shape[0]
    kmq = F._kmq()
    Rp, Rm, Ws = np.where(good, B + D, 0.0), np.where(good, B - D, 0.0), np.where(good, 0.0, Wc0)
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
                        elif branch == "cond":
                            K = Rp[q][None] / (E - e_g[kb][:, None, None] - Om[q][None] + 1j * eta)
                        else:
                            K = np.broadcast_to(-Ws[q][None], (e_g.shape[1],) + Ws[q].shape)
                        b = np.einsum("mn,nv->mv", M, np.conj(a[j]))
                        t1 = np.einsum("mu,muv,mv->", a[i], K, b)
                        if rule == "left":
                            val = t1
                        else:
                            c = np.einsum("mu,mn->nu", a[i], M)
                            t2 = np.einsum("nu,nuv,nv->", c, K, np.conj(a[j]))
                            val = 0.5 * (t1 + t2)
                        S[slot, k, i, j] += val / nk
    return S


def _dense_operators(pair_cond, pair_val, static=0.0):
    """Independent NumPy one-sided operators from dense branch tables."""
    dagger = lambda a: np.conj(np.swapaxes(a, -1, -2))
    rp, rh = pair_cond[1], dagger(pair_val[0])
    total = rp + rh
    injection = 1j * rh + dagger(1j * rh)
    return 1j * (total - dagger(total)), injection, _herm(total + static)


def _trace_ratio(gamma, injection, rho):
    """Trace residual relative to the particle flux, C = -1/2 {gamma, rho} + injection."""
    tr = sum(np.trace(injection[k]) - np.trace(gamma[k] @ rho[k]) for k in range(len(rho)))
    flux = sum(abs(np.trace((gamma[k] - injection[k]) @ rho[k])) for k in range(len(rho)))
    return float(abs(tr) / flux), float(flux)


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
    rows = {}
    rng2 = np.random.default_rng(20261019)
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
        good = po["valid_mask_q"]
        Rp = np.where(good, po["B_q"] + po["B_odd_q"], 0.0)
        Rm = np.where(good, po["B_q"] - po["B_odd_q"], 0.0)
        rows["reciprocity_Rm_vs_RpT"] = _rel(Rm, np.swapaxes(Rp[q_neg], -1, -2))
        rows["reciprocity_Om"] = _rel(np.where(good, po["Omega_q"], 0),
                                      np.where(good[q_neg], po["Omega_q"][q_neg], 0)
                                      .swapaxes(-1, -2))
        rows["n_invalid_modes"] = int(np.sum(~good))
        eta = F.ETA_EV / RYD_TO_EV
        omega_ry = np.arange(-120, 121) * 0.25 / RYD_TO_EV
        splan = PS.window_sigma_plan(F._config(omega_ry))
        eye = np.broadcast_to(np.eye(NA, dtype=complex), (nk, NA, NA)).copy()
        e_w = e[:, B_LO:B_LO + NA]
        kw = dict(meta=meta, mesh_xy=mesh, branches=True, static_limit=True)

        def sweep(win, internal="active"):
            t = time.perf_counter()
            out = PS.window_sigma_c(wfns, poles, win, splan, internal=internal, **kw)
            out["wall_s"] = time.perf_counter() - t
            return out

        def operators(out):
            gamma, injection, _ = C.branch_operators(out["branch_pair"]["cond"],
                                                     out["branch_pair"]["val"])
            return np.asarray(gamma), np.asarray(injection)

        # ---- FA: diagonal rho' == populations (an excited population state)
        f_x = np.clip(f[:, B_LO:B_LO + NA] + np.array([-0.06, -0.035, 0.05, 0.045])
                      * (1.0 + 0.3 * rng2.standard_normal((nk, NA))), 0.0, 1.0)
        win_p = dict(b_lo=B_LO, U=eye, e=e_w, f=f_x)
        pop = sweep(win_p)
        dia = sweep(dict(win_p, rho=np.einsum("ki,ij->kij", f_x, np.eye(NA)).astype(complex)))
        fa = [_rel(dia["pair"], pop["pair"])]
        fa += [_rel(dia["branch_pair"][n], pop["branch_pair"][n]) for n in ("cond", "val")]
        fa += [float(np.max(np.abs(dia["branch_pair"]["static"] - pop["branch_pair"]["static"]))
                     / max(float(np.max(np.abs(pop["branch_pair"]["static"]))), 1e-300))]
        fa += [_rel(dia["static_limit"]["sx"], pop["static_limit"]["sx"])]
        rows["FA"] = max(fa)
        rows["FA_parts"] = fa
        rows["FW_cold_s"] = [pop["wall_s"], dia["wall_s"]]
        rows["FW_populations_s"] = min(sweep(win_p)["wall_s"] for _ in range(2))
        rows["FW_full_s"] = min(sweep(dict(win_p, rho=np.einsum("ki,ij->kij", f_x, np.eye(NA))
                                            .astype(complex)))["wall_s"] for _ in range(2))
        rows["n_insert"] = int(dia["full_rho"]["n_insert"])

        # ---- FD / FB: random Hermitian rho' vs the dense full-rho reference
        rho = _random_rho(rng2, nk, NA, coh=1.0)
        f_r = np.real(np.einsum("kii->ki", rho))
        win_r = dict(b_lo=B_LO, U=eye, e=e_w, f=f_r, rho=rho)
        full = sweep(win_r)
        rows["FW_full_offdiag_s"] = full["wall_s"]
        e_g = e.copy()
        e_g[:, B_LO:B_LO + NA] = e_w
        M_h = np.zeros((nk, F.NBF, F.NBF), complex)
        M_h[:, B_LO:B_LO + NA, B_LO:B_LO + NA] = rho
        M_p = np.zeros_like(M_h)
        M_p[:, B_LO:B_LO + NA, B_LO:B_LO + NA] = np.eye(NA)[None] - rho
        proj = psi[:, B_LO:B_LO + NA]
        ref = dict(val=_dense_full(psi, e_g, M_h, proj, e_w, dpo, eta, "val"),
                   cond=_dense_full(psi, e_g, M_p, proj, e_w, dpo, eta, "cond"),
                   static=_dense_full(psi, e_g, M_h, proj, e_w, dpo, eta, "static")[0])
        rows["FD_parts"] = [_rel(full["branch_pair"][n], ref[n]) for n in ("cond", "val", "static")]
        rows["FD"] = max(rows["FD_parts"])
        gamma, injection = operators(full)
        rows["FB"] = max(float(np.max(np.abs(g - np.conj(np.swapaxes(g, -1, -2)))))
                         for g in (gamma, injection))

        # ---- FC on the fitted (non-reciprocal) poles: code == dense residual
        rows["FC_fit_code"] = _trace_ratio(gamma, injection, rho)[0]
        rows["FC_fit_dense"] = _trace_ratio(*_dense_operators(ref["cond"], ref["val"])[:2], rho)[0]
        gamma_p, injection_p = operators(sweep(dict(b_lo=B_LO, U=eye, e=e_w, f=f_r)))
        rows["FC_fit_populations_active"] = _trace_ratio(gamma_p, injection_p, rho)[0]

        # ---- the fixture's poles made reciprocal
        neg = lambda a: np.swapaxes(a[q_neg], -1, -2)
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
        Rp_s = np.where(g_s, dpo[1] + dpo[2], 0.0)
        Rm_s = np.where(g_s, dpo[1] - dpo[2], 0.0)
        rows["reciprocity_sym_Rm_vs_RpT"] = _rel(Rm_s, neg(Rp_s))
        rows["n_invalid_modes_sym"] = int(np.sum(~g_s))
        full = sweep(win_r)
        gamma, injection = operators(full)
        ref = dict(val=_dense_full(psi, e_g, M_h, proj, e_w, dpo, eta, "val"),
                   cond=_dense_full(psi, e_g, M_p, proj, e_w, dpo, eta, "cond"))
        rows["FD_sym_parts"] = [_rel(full["branch_pair"][n], ref[n]) for n in ("cond", "val")]

        # ---- FC: the closed-window trace identity (reciprocal poles)
        rc, flux = _trace_ratio(gamma, injection, rho)
        rows["FC_code"] = rc
        rows["FC_flux_mev"] = flux * RYD_TO_EV * 1e3
        gamma_d, injection_d, _ = _dense_operators(ref["cond"], ref["val"])
        rows["FC_dense"] = _trace_ratio(gamma_d, injection_d, rho)[0]
        left = dict(val=_dense_full(psi, e_g, M_h, proj, e_w, dpo, eta, "val", rule="left"),
                    cond=_dense_full(psi, e_g, M_p, proj, e_w, dpo, eta, "cond", rule="left"))
        gamma_l, injection_l, _ = _dense_operators(left["cond"], left["val"])
        rows["FC_dense_left_index"] = _trace_ratio(gamma_l, injection_l, rho)[0]
        popr = sweep(dict(b_lo=B_LO, U=eye, e=e_w, f=f_r))
        gamma_p, injection_p = operators(popr)
        fro = sweep(dict(b_lo=B_LO, U=eye, e=e_w, f=f_r), internal="frozen")
        gamma_f, injection_f = operators(fro)
        rows["FC_populations_active"] = _trace_ratio(gamma_p, injection_p, rho)[0]
        rows["FC_full_all"] = _trace_ratio(gamma + gamma_f, injection + injection_f, rho)[0]
        rows["FC_populations_all"] = _trace_ratio(gamma_p + gamma_f, injection_p + injection_f, rho)[0]
        rows["FC_frozen_flux_share"] = float(
            sum(abs(np.trace((gamma_f[k] - injection_f[k]) @ rho[k])) for k in range(nk)) / flux)
        # the same identity at a near-equilibrium state (coherences 0.05)
        rho_s = np.einsum("ki,ij->kij", f[:, B_LO:B_LO + NA], np.eye(NA)) * 0.9 + 0.05 * _random_rho(
            rng2, nk, NA, coh=1.0)
        rho_s = _herm(rho_s)
        lam = np.linalg.eigvalsh(rho_s)
        rows["FC_small_spectrum"] = [float(lam.min()), float(lam.max())]
        out_s = sweep(dict(b_lo=B_LO, U=eye, e=e_w, f=np.real(np.einsum("kii->ki", rho_s)),
                           rho=rho_s))
        rows["FC_code_small"] = _trace_ratio(*operators(out_s), rho_s)[0]
        pop_s = sweep(dict(b_lo=B_LO, U=eye, e=e_w, f=np.real(np.einsum("kii->ki", rho_s))))
        rows["FC_populations_active_small"] = _trace_ratio(*operators(pop_s), rho_s)[0]

        # ---- FU: covariance at a unitary mixing degenerate window states
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
        A = sweep(dict(b_lo=B_LO, U=eye, e=e_d, f=f_r, rho=rho))
        Bq = sweep(dict(b_lo=B_LO, U=Ud, e=e_d, f=np.real(np.einsum("kii->ki", rho_u)), rho=rho_u))
        ga, gb = operators(A), operators(Bq)
        rows["FU"] = max(_rel(Uh @ ga[i] @ Ud, gb[i]) for i in range(2))
        Ap = sweep(dict(b_lo=B_LO, U=eye, e=e_d, f=f_r))
        Bp = sweep(dict(b_lo=B_LO, U=Ud, e=e_d, f=np.real(np.einsum("kii->ki", rho_u))))
        gap, gbp = operators(Ap), operators(Bp)
        rows["RU_populations"] = max(_rel(Uh @ gap[i] @ Ud, gbp[i]) for i in range(2))
    return rows


def verdict(rows):
    ok = {g: rows[g] <= tol for g, tol in GATES.items()}
    ok["RU"] = rows["RU_populations"] > 100.0 * GATES["FU"]
    # Column sampling requires the vertex internal ordering.  This file keeps
    # the earlier symmetric builder as a lower-level red twin; the vertex
    # conservation gate is IF in test_tdagw_collision_instbasis.py.
    ok["RC"] = rows["FC_code"] > 1e-2 and rows["FC_dense"] > 1e-2
    return ok


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


if __name__ == "__main__":
    sys.exit(main())
