"""Sigma^c of the CrI3 516 frontier bands on a real-frequency grid, for the QP broadening (measured).

    lx run -N 1 -G 4 -n 4 -- python3 -u tests/tdagw/_qp_broadening_516_case.py \\
        --spm-ref <516>/lorrax/04_gw_oneshot_mpa_eqp --gn-ref <516>/lorrax/07_gw_gnppm_bundle \\
        --out DIR

The window Sigma (``gw.ppm_sigma.window_sigma_c``) on the band block [--win), U = 1,
e = eps^DFT, f = the populations of the state, on the on-shell list (Sigma_ij(e_i),
Sigma_ij(e_j), exact) and on a real grid (``omega_ry``: --step-ev from the DFT VBM -
--below-ev to the DFT CBM + --above-ev; Sigma_ij(omega + i eta), the Lorentzian of the
sweep).  Arms: the 04 shared-pole run at each --etas, the source rerun at that eta
(``sigma_regularization_ev`` override: the recipe couples eta to W, so W is refitted at
every eta), at rho_eq and at the hot state (--hot-amount of band 129 moved to 130 at
every k) with W refitted there (``tdagw.dynsig.spm_refit``; V1 the production ladder,
V2 --v2-* supports); the 07 GN-PPM run (W independent of eta) at each eta, rho_eq.
--qp-g: also the 04 arm at those etas with the window's energies in G set to the run's
own eqp1 (an eigenvalue-updated G on the window bands only).  Rank 0 writes one npz per
sweep (DIR/sig_<arm>_<state>_eta<eta>.npz: pair, omega cube of the block, grid, the
block's eps, f) and DIR/qp_516.json; the eigenvalue analysis is host-side
(``qp_broadening_analysis.py``).
"""
import argparse
import json
import os
import sys
import time

import runtime  # noqa: F401
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _eqp(path, ivbm=130):
    """(nk, nb) eqp1 QP energies (eV) and DFT energies, BGW format."""
    rows = []
    with open(path) as f:
        lines = [l.split() for l in f if l.strip() and not l.lstrip().startswith("#")]
    i = 0
    while i < len(lines):
        nb = int(lines[i][3])
        rows.append(np.array([[float(x) for x in l] for l in lines[i + 1:i + 1 + nb]]))
        i += 1 + nb
    return np.array([r[:, 3] for r in rows]), np.array([r[:, 2] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spm-ref", required=True)
    ap.add_argument("--gn-ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--win", type=int, nargs=2, default=(119, 144))
    ap.add_argument("--etas", default="0.25,0.1,0.05,0.025")
    ap.add_argument("--step-ev", type=float, default=0.01)
    ap.add_argument("--below-ev", type=float, default=3.5)
    ap.add_argument("--above-ev", type=float, default=4.5)
    ap.add_argument("--hot-amount", type=float, default=0.05)
    ap.add_argument("--v2-line", default="0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.3")
    ap.add_argument("--v2-imag", default="0.25,0.5")
    ap.add_argument("--v2-height", type=float, default=1.0)
    ap.add_argument("--arms", default="spm,gn")
    ap.add_argument("--states", default="eq,hot_V1,hot_V2")
    ap.add_argument("--qp-g", default="0.05,0.025", help="etas of the 04 eqp1-in-G sweep ('' none)")
    a = ap.parse_args()
    stack = runtime.initialize_communicator_stack()
    import jax
    from common.units import RYD_TO_EV
    from gw import ppm_sigma as PS
    from tdagw import dynsig as D
    from _spm_refresh_516_case import _v2_sites
    mesh = stack.mesh
    rank0 = jax.process_index() == 0
    say = print if rank0 else (lambda *x, **k: None)
    os.makedirs(a.out, exist_ok=True)
    out = dict(win=list(a.win), etas=a.etas, head=os.environ.get("LEG_HEAD", ""), sweeps=[])
    jname = os.path.join(a.out, "qp_516.json")

    def save():
        if rank0:
            with open(jname + ".tmp", "w") as f:
                json.dump(out, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
            os.replace(jname + ".tmp", jname)

    lo, hi = (int(b) for b in a.win)
    nA = hi - lo
    etas = [float(x) for x in a.etas.split(",") if x]
    qpg = [float(x) for x in a.qp_g.split(",") if x]

    def eta_plan(plan, eta_ev):
        import dataclasses
        return dict(plan, sigma_cfg=dataclasses.replace(plan["sigma_cfg"], regularization_ev=float(eta_ev)))

    def sweep(tag, src, poles, f_blk, eta, *, e_blk=None):
        wfns, meta = src["wfns"], src["meta"]
        nk = int(meta.nk_tot)
        enk = np.asarray(jax.device_get(wfns.enk), np.float64)
        e = enk[:, lo:hi] if e_blk is None else e_blk
        efermi = float(src["efermi_ry"])
        vbm, cbm = enk[:, 129].max(), enk[:, 130].min()
        grid_abs = np.arange(vbm - a.below_ev / RYD_TO_EV, cbm + a.above_ev / RYD_TO_EV + 1e-12,
                             a.step_ev / RYD_TO_EV)
        eye = np.broadcast_to(np.eye(nA, dtype=complex), (nk, nA, nA)).copy()
        t1 = time.perf_counter()
        o = PS.window_sigma_c(wfns, poles, dict(b_lo=lo, U=eye, e=e, f=f_blk), eta_plan(src["plan"], eta),
                              meta=meta, mesh_xy=mesh, omega_ry=grid_abs - efermi)
        wall = time.perf_counter() - t1
        rec = dict(tag=tag, eta_ev=eta, wall_s=wall, n_omega=int(grid_abs.size), sweep=o["sweep"],
                   efermi_ry=efermi, max_abs_onshell_ev=float(np.max(np.abs(o["diag"]))) * RYD_TO_EV)
        if rank0:
            np.savez(os.path.join(a.out, f"sig_{tag}.npz"), pair=o["pair"], omega=o["omega"].astype(np.complex64),
                     omega_diag=np.einsum("wkii->wki", o["omega"]), grid_abs_ry=grid_abs, efermi_ry=efermi,
                     e_blk_ry=e, f_blk=f_blk, enk_ry=enk, eta_ev=eta, band_lo=lo)
        out["sweeps"].append(rec)
        say(f"[qp_516] sweep {tag}: eta {eta:g} eV, {grid_abs.size} omega, {wall:.1f} s, "
            f"{o['sweep']['distinct_tau']} tau (on shell)")
        save()
        return o

    t_all = time.perf_counter()
    arms = [x for x in a.arms.split(",") if x]
    states = [x for x in a.states.split(",") if x]
    if "spm" in arms:
        deck = os.path.join(os.path.abspath(a.spm_ref), "oneshot.in")
        e_qp_ev, _ = _eqp(os.path.join(os.path.abspath(a.spm_ref), "eqp1.dat"))
        for eta in etas:
            src = D.gw_window_source(deck, mesh, workdir=os.path.join(os.path.abspath(a.out), f"driver_spm_{eta:g}"),
                                     screening=any(s.startswith("hot") for s in states),
                                     overrides={"sigma_regularization_ev": eta}, say=say)
            occ = np.asarray(jax.device_get(src["wfns"].occ), np.float64)
            if "eqp1_vs_enk_ev" not in out:      # eqp1's k order and DFT column against the carrier's
                _, e_dft_ev = _eqp(os.path.join(os.path.abspath(a.spm_ref), "eqp1.dat"))
                enk_ev = np.asarray(jax.device_get(src["wfns"].enk), np.float64) * RYD_TO_EV
                nb_ = min(e_dft_ev.shape[1], enk_ev.shape[1])
                out["eqp1_vs_enk_ev"] = float(np.max(np.abs(e_dft_ev[:, :nb_] - enk_ev[:, :nb_])))
            f_eq = occ[:, lo:hi].copy()
            f_hot = f_eq.copy()
            f_hot[:, 129 - lo] -= a.hot_amount
            f_hot[:, 130 - lo] += a.hot_amount
            out.setdefault("sources", []).append(dict(arm="spm", eta_ev=eta, wall_s=src["wall_s"],
                                                      K=[min(src["model"]["K"]), max(src["model"]["K"])]))
            if "eq" in states:
                sweep(f"spm_eq_eta{eta:g}", src, src["poles"], f_eq, eta)
            if eta in qpg:
                efermi_ev = float(src["efermi_ry"]) * RYD_TO_EV
                e_qp = (e_qp_ev[:, lo:hi]) / RYD_TO_EV                 # absolute Ry (eqp1 is absolute eV)
                sweep(f"spm_eqQPG_eta{eta:g}", src, src["poles"], f_eq, eta, e_blk=e_qp)
                out.setdefault("qp_g_note", f"window energies in G = eqp1 (abs eV); E_F {efermi_ev:.4f} eV")
            for st in states:
                if not st.startswith("hot"):
                    continue
                V = st.split("_")[1]
                sites = "" if V == "V1" else _v2_sites(
                    src["meta"].shared_pole_recipe, [float(x) for x in a.v2_line.split(",") if x],
                    [float(x) for x in a.v2_imag.split(",") if x], a.v2_height if a.v2_height > 0 else None)
                scr = D.spm_screening(src, mesh=mesh, b_lo=128, n_active=4, support_sites=sites,
                                      scratch=os.path.join(os.path.abspath(a.out), f"refits_{V}_{eta:g}"))
                t1 = time.perf_counter()
                try:
                    at, rrec = D.spm_refit(scr, f_hot[:, 128 - lo:132 - lo], label="hot")
                except Exception as exc:
                    out.setdefault("refusals", []).append(dict(state=st, eta_ev=eta, error=repr(exc)[:600]))
                    say(f"[qp_516] refit {st} eta {eta:g} REFUSED: {exc!r}"[:400])
                    save()
                    continue
                out.setdefault("refits", []).append(dict(state=st, eta_ev=eta, wall_s=time.perf_counter() - t1,
                                                         K=[min(rrec["K"]), max(rrec["K"])],
                                                         census=(rrec.get("census") or {}).get("frac_residue_below"),
                                                         omega_min_ev=(rrec.get("census") or {}).get("omega_min_ev")))
                sweep(f"spm_{st}_eta{eta:g}", src, at, f_hot, eta)
                del at, scr
            del src
    if "gn" in arms:
        deck = os.path.join(os.path.abspath(a.gn_ref), "oneshot.in")
        src = D.gw_window_source(deck, mesh, workdir=os.path.join(os.path.abspath(a.out), "driver_gn"),
                                 screening=False, say=say)
        occ = np.asarray(jax.device_get(src["wfns"].occ), np.float64)
        out.setdefault("sources", []).append(dict(arm="gn", eta_ev=float(src["config"].sigma.regularization_ev),
                                                  wall_s=src["wall_s"]))
        for eta in etas:
            sweep(f"gn_eq_eta{eta:g}", src, src["poles"], occ[:, lo:hi].copy(), eta)
    out["wall_s"] = time.perf_counter() - t_all
    save()
    if rank0:
        print("RESULT " + json.dumps(dict(n=len(out["sweeps"]), refusals=len(out.get("refusals", [])),
                                          wall_s=out["wall_s"])), flush=True)
    runtime.finalize_process(0)


if __name__ == "__main__":
    main()
