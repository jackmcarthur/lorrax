#!/usr/bin/env python3
"""The TD-aGW validation ladder on CrI3 (516: 6x6 k, PBE+SOC spinors, static-COHSEX ISDF bundle).

    python3 tools/td_validation_cri3.py dry     --out DIR   # login node, JAX_PLATFORMS=cpu, 4 host devices
    python3 -u tools/td_validation_cri3.py leg  --out DIR   # P = 4 under lx: kernel samples + drives
    python3 tools/td_validation_cri3.py analyze --out DIR   # host: window BSE + every comparison

Rungs 2, 4, 5 and 6 of ``docs/theory/td_agw_propagation.md#validation`` on the
production operator: the TD system and the ISDF Hartree + SEX model of the
bundle route (``[lorrax] system = 'bundle'``, ``model = 'isdf'``, one
``load_td_model`` for header and kernel), propagated by the driver's stages
on the run's mesh.  Two windows:

* ``w8`` — bands 123..138 (8v8c, DFT edges 68 and 89 meV clear at every k):
  x and y kicks; the linearized response against the full BSE of the same
  kernel (sampled column by column, ``tdagw.model.dense_td_kernel``), and the
  window's f-sum against the full-band dipole set.
* ``w2`` — bands 129..132 (2v2c): sigma+ and sigma- pulses centred on the
  lowest bright sigma+ exciton, x and y kicks; per-state populations, eps_xy,
  sigma_xy, Kerr and Faraday of a suspended monolayer (sheet and d = 6.6 A film).

The leg writes per window ``kernel_bse.npy`` (the kernel on the c-v / v-c
pairs, Ha), ``system.npz`` (header energies, dipoles, weights, cell) and the
propagated ``tdgw.h5`` files; ``analyze`` needs no device.  Every drive is run
at +E0 and -E0 and analysed through the odd part.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

S2 = "/pscratch/sd/j/jackm/sandbox_v2_docs_consolidation_2026-08-14"
R516 = f"{S2}/runs/CrI3/516_tdgw_ref_6x6_nosym_20261016"
B05 = f"{R516}/lorrax/05_gw_cohsex_bundle"
BUNDLE = dict(bundle_input=f"{B05}/oneshot.in", dipole_file=f"{B05}/dipole.h5",
              eqp_file=f"{R516}/tdgw/indata/eqp.dat", isdf_zeta=f"{B05}/tmp/zeta_q.h5")
WINDOWS = {"w8": (123, 138), "w2": (129, 132)}        # 1-based ibndmin, ibndmax
IVBM = 130
KICK = dict(e0=1e6, tc=1200.0, tau=200.0)            # V/m, as, as
PULSE = dict(e0=1e6, tc=12000.0, tau=2000.0)
TF_AS, MDT_AS, DT_AS = 60000.0, 20.0, 10.0           # etdrk4 at 10 as, a sample every 2 steps


def _parse(argv):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("cmd", choices=("dry", "leg", "analyze"))
    p.add_argument("--out", required=True)
    p.add_argument("--degeneracy-mode", default="w8=strict,w2=off",
                   help="[lorrax] degeneracy_mode per window, 'w8=strict,w2=off' or one mode for "
                        "both; strict refuses w2: the QP spectrum has bands 128|129 (1-based) "
                        "0.548 meV apart at k = 24 (TDPROP's 516 runs: off)")
    p.add_argument("--windows", default="w8,w2")
    return p.parse_args(argv)


def _degeneracy_mode(spec, window):
    if "=" not in spec:
        return spec
    return dict(kv.split("=", 1) for kv in spec.split(","))[window]


def _tdr():
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("lorrax_tools_td_response",
                                                  os.path.join(here, "td_response.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _trial(e0, w_ev, tc, tau, pol):
    from tdagw.tdgw_io import HBAR_IN_EV_AS
    env = f"{float(e0)!r} * np.exp(-((t - {float(tc)!r}) / {float(tau)!r})**2)"
    w = float(w_ev) / HBAR_IN_EV_AS
    cos, sin = f"np.cos({w!r} * t)", f"np.sin({w!r} * t)"
    if pol in ("x", "y"):
        row = {"E_x": 0, "E_y": 0, "E_z": 0}
        row["E_" + pol] = f"{env} * {cos}"
        return row
    s = 1 if pol == "+" else -1
    return {"E_x": f"{env} * {cos}", "E_y": f"{s} * {env} * {sin}", "E_z": 0}


def _drives(window, sigma_plus_ev):
    """[(label, pol, e0, w_ev, tc, tau)] of a window, each to be run at +-E0."""
    k = [("kick_x", "x", KICK["e0"], 0.0, KICK["tc"], KICK["tau"]),
         ("kick_y", "y", KICK["e0"], 0.0, KICK["tc"], KICK["tau"])]
    if window == "w8":
        return k
    return k + [("sp_at_Splus", "+", PULSE["e0"], sigma_plus_ev, PULSE["tc"], PULSE["tau"]),
                ("sm_at_Splus", "-", PULSE["e0"], sigma_plus_ev, PULSE["tc"], PULSE["tau"])]


def _deck(window, outdir, drives, degeneracy_mode):
    from tdagw.td_config import parse_td_deck
    lo, hi = WINDOWS[window]
    trials = []
    for _, pol, e0, w, tc, tau in drives:
        trials += [_trial(e0, w, tc, tau, pol), _trial(-e0, w, tc, tau, pol)]
    nt = int(round(TF_AS / MDT_AS))
    meas = ["Efield", "polarization"] + (["density_matrix"] if window == "w2" else [])
    raw = {"control": {"calc_type": "COHSEX",
                       "timesteps": {"t0": 0, "tf": TF_AS, "dt_measure": MDT_AS, "ntcheck": nt},
                       "ivp": {"rtol": 1e-8, "atol": 1e-15},
                       "files": {"outdir": outdir}},
           "system": {"ibndmin": lo, "ibndmax": hi, "ivbm": IVBM, "geometry": "slab",
                      "esmear": 0.05},
           "measurements": {"types": meas},
           "lorrax": dict(BUNDLE, system="bundle", pair_rule="cv", model="isdf",
                          degeneracy_mode=degeneracy_mode, method="etdrk4", dt=DT_AS,
                          intraband=False, trials=trials)}
    if window == "w2":
        raw["measurements"]["2"] = {"itrange": [0, nt, 50]}
    return parse_td_deck(raw, source=f"td_validation_cri3:{window}")


def _host_operators(deck, sysd):
    """The propagator's host operators (``stages.load_system``'s own call)."""
    from tdagw import tdgw_io
    v = deck.values
    return tdgw_io.header_operators(sysd.hdr, geometry=v["system.geometry"],
                                    esmear_ha=v["system.esmear"], intraband=False)


def _system_npz(path, sysd, host, n_val):
    import numpy as np
    h = sysd.hdr
    np.savez(path, e_ha=np.asarray(host["e_qp"]), f_kn=np.asarray(host["f_kn"]),
             r=np.asarray(host["r_inter"]), w_k=np.asarray(host["w_k"]),
             cell_volume=float(host["cell_volume"]), nspinor=int(host["nspinor"]),
             n_val=int(n_val), rprim=np.asarray(h["rprim"]), kpts=np.asarray(h["kpts"]),
             bands=np.asarray(h["bands"]), itype=int(h["itype"]))


def _bse_index(nk, nb, n_val):
    import numpy as np
    k, c, v = np.meshgrid(np.arange(nk), np.arange(n_val, nb), np.arange(n_val), indexing="ij")
    k, c, v = k.ravel(), c.ravel(), v.ravel()
    # flat (k, n, m) index of every X = (k, c, v) and Y = (k, v, c) entry
    return np.concatenate([(k * nb + c) * nb + v, (k * nb + v) * nb + c])


# ---------------------------------------------------------------------------
# leg / dry
# ---------------------------------------------------------------------------

def run_leg(args, *, propagate: bool):
    from runtime import debug_print, initialize_communicator_stack
    runtime = initialize_communicator_stack(print_fn=debug_print)
    import numpy as np
    from jax.experimental import multihost_utils
    from common.collectives import process_rank
    from tdagw import stages
    from tdagw.model import build_td_agw_apply, dense_td_kernel
    from tdagw.tdgw_io import HA_IN_EV, efield_function

    tdr = _tdr()
    mesh = runtime.mesh
    rank0 = process_rank() == 0
    say = (lambda s: print(s, flush=True)) if rank0 else (lambda s: None)
    receipts = dict(mesh=list(mesh.devices.shape), platform=str(mesh.devices.flat[0].platform),
                    degeneracy_mode=args.degeneracy_mode, windows={})
    for window in args.windows.split(","):
        t0 = time.perf_counter()
        wdir = os.path.join(args.out, window)
        os.makedirs(wdir, exist_ok=True)
        rec = {"degeneracy_mode": _degeneracy_mode(args.degeneracy_mode, window)}
        mode = _degeneracy_mode(args.degeneracy_mode, window)
        deck0 = _deck(window, os.path.join(wdir, "td"), _drives(window, 1.0), mode)
        sysd = stages.load_system(deck0, mesh)
        rec["load_s"] = time.perf_counter() - t0
        rec["receipts"] = {k: v for k, v in (sysd.receipts or {}).items()
                           if k in ("bands", "n_val", "n_cond", "energy_source", "pairs",
                                    "qp_minus_dft_gap_ev")}
        say(f"[tdval] {window}: system nk {sysd.nk} nb {sysd.nb} nspinor {sysd.nspinor} "
            f"load {rec['load_s']:.1f} s; {rec['receipts']}")
        if not propagate:
            receipts["windows"][window] = rec
            continue
        t1 = time.perf_counter()
        apply = build_td_agw_apply(mesh, sysd.model)
        K = dense_td_kernel(apply, sysd.model, mesh_xy=mesh, batch=16)
        rec["kernel_s"] = time.perf_counter() - t1
        nk, nb = sysd.nk, sysd.nb
        n_val = stages._n_val(sysd.hdr)
        host = _host_operators(deck0, sysd)
        idx = _bse_index(nk, nb, n_val)
        Kf = K.reshape(nk * nb * nb, nk * nb * nb)
        K_bse = 0.5 * Kf[np.ix_(idx, idx)]                   # Ry -> Ha, the isdf_model's dSigma
        herm = float(np.abs(Kf - Kf.conj().T).max() / np.abs(Kf).max())
        rec["kernel_hermitian_dev"] = herm
        del K, Kf
        sigma_plus_ev = 0.0
        if window == "w2":                                   # the drive sits on the lowest bright sigma+ state
            n = idx.size // 2
            L, o, u = tdr.window_operator(host["e_qp"], n_val, None, host["r_inter"],
                                          host["w_k"])
            L = L + np.block([[K_bse[:n, :n], K_bse[:n, n:]], [-K_bse[n:, :n], -K_bse[n:, n:]]])
            m = _modes_from_L(tdr, L, o, u)
            pos = m.lam.real > 0
            ip, im, _ = tdr.helicity(m.R[pos])
            bright = (ip + im) > 1e-2 * (ip + im).max()
            cand = np.where(bright & (ip > im))[0]
            sigma_plus_ev = float(m.lam.real[pos][cand[0]] * HA_IN_EV) if cand.size else 0.0
            sigma_plus_ev = float(multihost_utils.broadcast_one_to_all(np.asarray(sigma_plus_ev)))
            rec["sigma_plus_carrier_ev"] = sigma_plus_ev
            rec["lowest_bse_ev"] = [float(x) for x in m.lam.real[pos][:6] * HA_IN_EV]
        if rank0:
            np.save(os.path.join(wdir, "kernel_bse.npy"), K_bse)
            _system_npz(os.path.join(wdir, "system.npz"), sysd, host, n_val)
        del K_bse
        deck = _deck(window, os.path.join(wdir, "td"), _drives(window, sigma_plus_ev), mode)
        model = stages.build_self_energy(deck, sysd, mesh)
        outs = stages.open_outputs(deck, sysd)
        t2 = time.perf_counter()
        stats = stages.propagate(deck, sysd, model, efield_function(deck.trials), outs, mesh,
                                 say=say)
        rec["propagate_s"] = time.perf_counter() - t2
        rec["stats"] = {k: (float(v) if isinstance(v, (int, float)) else v)
                        for k, v in stats.items() if k in ("accepted", "dsigma", "wall",
                                                            "wall_first_chunk", "simulated_fs")}
        rec["drives"] = [list(d) for d in _drives(window, sigma_plus_ev)]
        rec["paths"] = list(outs.paths)
        rec["model"] = model.name
        receipts["windows"][window] = rec
        say(f"[tdval] {window}: kernel {rec['kernel_s']:.1f} s (herm dev {herm:.1e}), "
            f"propagate {rec['propagate_s']:.1f} s, {rec['stats']}")
    if rank0:
        with open(os.path.join(args.out, f"receipts_{args.cmd}.json"), "w") as fh:
            json.dump(receipts, fh, indent=1, default=str)
    return 0


def _modes_from_L(tdr, L, o, u):
    """WindowModes of an assembled L (the analysis's own diagonalization)."""
    import numpy as np
    lam, Z = np.linalg.eig(L)
    n = L.shape[0] // 2
    norm = (np.einsum("is,is->s", np.conj(Z[:n]), Z[:n])
            - np.einsum("is,is->s", np.conj(Z[n:]), Z[n:]))
    Z = Z * np.where(norm.real > 0, 1.0 / np.sqrt(np.abs(norm)), 1.0)[None, :]
    Zi = np.linalg.solve(Z, np.eye(Z.shape[0]))
    R = (o @ Z).T[:, :, None] * (Zi @ u.T)[:, None, :]
    order = np.argsort(lam.real)
    return tdr.WindowModes(lam=lam[order], R=R[order], Z=Z[:, order], Zi=Zi[order], L=L,
                           o=o, u=u, n_pairs=n)


# ---------------------------------------------------------------------------
# analyze (host)
# ---------------------------------------------------------------------------

def _load_window(out, window):
    import numpy as np
    sysz = dict(np.load(os.path.join(out, window, "system.npz")))
    K = np.load(os.path.join(out, window, "kernel_bse.npy"))
    return sysz, K


def _td_odd(paths, n_drives):
    import h5py
    import numpy as np
    data = []
    for p in paths:
        with h5py.File(p, "r") as f:
            td = f["tdgw_data/time_domain"]
            data.append({k: (td[k]["times"][()], td[k]["data"][()]) for k in td})
    out = []
    for i in range(n_drives):
        a, b = data[2 * i], data[2 * i + 1]
        out.append({k: (a[k][0], 0.5 * (a[k][1] - b[k][1])) for k in a})
    return out


def analyze(args):
    import numpy as np
    tdr = _tdr()
    from tdagw.tdgw_io import EFIELD_IN_VPERM as e_vm, HA_IN_EV as ha, \
        HBAR_IN_EV_AS as hbar_ev_as, T_IN_ATTOSEC as t_as
    rec = json.load(open(os.path.join(args.out, "receipts_leg.json")))
    res = {}
    for window, wrec in rec["windows"].items():
        sysz, K = _load_window(args.out, window)
        e, w_k, r = sysz["e_ha"], sysz["w_k"], sysz["r"]
        nk, nb = e.shape
        n_val = int(sysz["n_val"])
        vol = float(sysz["cell_volume"])
        g_s = 2.0 if int(sysz["nspinor"]) == 1 else 1.0
        L0, o, u = tdr.window_operator(e, n_val, None, r, w_k)
        n = L0.shape[0] // 2
        L = L0 + np.block([[K[:n, :n], K[:n, n:]], [-K[n:, :n], -K[n:, n:]]])
        t0 = time.perf_counter()
        m = _modes_from_L(tdr, L, o, u)
        out = dict(n_pairs=int(n), eig_s=time.perf_counter() - t0,
                   max_abs_im_lambda_ev=float(np.abs(m.lam.imag).max() * ha))
        pos = np.where(m.lam.real > 0)[0]
        ip, im, eta = tdr.helicity(m.R[pos])
        drives = wrec["drives"]
        td = _td_odd(wrec["paths"], len(drives))
        pulses = []
        for lab, pol, e0, w_ev, tc, tau in drives:
            a = {"x": (1, 0, 0), "y": (0, 1, 0), "+": (1, 1j, 0), "-": (1, -1j, 0)}[pol]
            pulses.append(tdr.Pulse(a=np.asarray(a, complex) * (e0 / e_vm),
                                    w=(w_ev / hbar_ev_as) * t_as, tc=tc / t_as, tau=tau / t_as))
        # rung 2: P(t) == the full-BSE closed form, per drive
        out["P_rel_dev"] = {}
        for (lab, *_), d, pl in zip(drives, td, pulses):
            t, P = d["polarization"][0], d["polarization"][1].real
            ref = -(g_s / vol) * tdr.modal_response(m, pl, t).real
            out["P_rel_dev"][lab] = float(np.abs(P - ref).max() / np.abs(ref).max())
        # spectra from the kicks, eta 0.2 eV
        eta_s = 0.2 / ha
        w = np.linspace(0.8, 4.5, 741) / ha
        chi = np.zeros((w.size, 3, 3), complex)
        for b, lab in enumerate(("kick_x", "kick_y")):
            i = [d[0] for d in drives].index(lab)
            chi[:, :, b] = tdr.chi_from_td(td[i]["polarization"][0], td[i]["polarization"][1].real,
                                           td[i]["Efield"][1], w, eta_s)
        ref = tdr.chi_from_modes(m, w, eta_s, g_s=g_s, cell_volume=vol)
        out["chi_rel_dev"] = float(np.abs(chi[:, :2, :2] - ref[:, :2, :2]).max()
                                   / np.abs(ref[:, :2, :2]).max())
        out["truncation_e_minus_eta_T"] = float(np.exp(-eta_s * TF_AS / t_as))
        out["isotropy"] = dict(
            xx_minus_yy=float(np.abs(ref[:, 0, 0] - ref[:, 1, 1]).max() / np.abs(ref[:, 0, 0]).max()),
            xy_plus_yx=float(np.abs(ref[:, 0, 1] + ref[:, 1, 0]).max()
                             / max(np.abs(ref[:, 0, 1]).max(), 1e-300)))
        # rung 4: the window's f-sum, interacting and IP
        sr = tdr.sum_rules(m)
        sr0 = tdr.sum_rules(tdr.WindowModes(lam=None, R=None, Z=None, Zi=None, L=L0, o=o, u=u,
                                            n_pairs=n))
        out["f_sum"] = dict(m1_xx=float(sr["m1"][0, 0].real), m1_yy=float(sr["m1"][1, 1].real),
                            s_ip_xx=float(sr0["m1"][0, 0].real), s_ip_yy=float(sr0["m1"][1, 1].real),
                            interacting_over_ip_minus_1=float(sr["m1"][0, 0].real
                                                              / sr0["m1"][0, 0].real - 1))
        # exciton table
        order = np.argsort(m.lam.real[pos])
        tot = ip + im
        out["states"] = [dict(e_ev=float(m.lam.real[pos][s] * ha), i_plus=float(ip[s]),
                              i_minus=float(im[s]), eta=float(eta[s]))
                         for s in order[:24]]
        out["bright_max_minor_over_major"] = float(
            (np.minimum(ip, im) / np.maximum(ip, im))[tot > 1e-2 * tot.max()].max())
        if window == "w2":
            out.update(_helicity_and_mo(tdr, m, pos, ip, im, eta, td, drives, pulses, w, chi,
                                        ref, eta_s, vol, g_s, sysz, n_val, ha, args.out))
        res[window] = out
        np.savez(os.path.join(args.out, window, "spectra.npz"), w_ev=w * ha, chi_td=chi,
                 chi_bse=ref)
    res["f_sum_truncation"] = (_f_sum_truncation(tdr, ha) if os.path.isfile(BUNDLE["dipole_file"])
                               else f"not run: {BUNDLE['dipole_file']} not on this host")
    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:6000])
    return 0


def _rot_dev(a, b):
    """max |(theta_a - theta_b) mod pi + i (eps_a - eps_b)| / max |theta_b + i eps_b|.

    Rotation angles are defined modulo pi (``td_response._ellipse``), so a
    rotation near +-pi/2 is compared on the circle.
    """
    import numpy as np
    d_th = (a.real - b.real + np.pi / 2) % np.pi - np.pi / 2
    return float(np.abs(d_th + 1j * (a.imag - b.imag)).max() / np.abs(b).max())


def _helicity_and_mo(tdr, m, pos, ip, im, eta, td, drives, pulses, w, chi, ref, eta_s, vol,
                     g_s, sysz, n_val, ha, out):
    import numpy as np
    labs = [d[0] for d in drives]
    tot = ip + im
    bright = tot > 1e-2 * tot.max()
    plus, minus = bright & (eta > 0), bright & (eta < 0)
    s_plus = int(np.where(plus)[0][0])
    s_minus = int(np.where(minus)[0][0])
    res = dict(S_plus=dict(index=s_plus, e_ev=float(m.lam.real[pos][s_plus] * ha),
                           i_plus=float(ip[s_plus]), i_minus=float(im[s_plus])),
               S_minus=dict(index=s_minus, e_ev=float(m.lam.real[pos][s_minus] * ha),
                            i_plus=float(ip[s_minus]), i_minus=float(im[s_minus])))
    for lab in ("sp_at_Splus", "sm_at_Splus"):
        i = labs.index(lab)
        drho = td[i]["density_matrix"][1]
        n = np.abs(tdr.project_state(m, drho, n_val)[-1, pos]) ** 2
        pt = np.abs(tdr.mode_amplitudes(m, pulses[i])[pos]) ** 2
        res[lab] = dict(N_Splus=float(n[s_plus]), N_Sminus=float(n[s_minus]),
                        N_Splus_vs_first_order=float(abs(n[s_plus] / pt[s_plus] - 1)),
                        N_Sminus_vs_first_order=float(abs(n[s_minus] / pt[s_minus] - 1)
                                                      if pt[s_minus] > 0 else np.nan),
                        sum_plus=float(n[plus].sum()), sum_minus=float(n[minus].sum()),
                        sum_plus_first_order=float(pt[plus].sum()),
                        sum_minus_first_order=float(pt[minus].sum()))
    # magneto-optics: eps_xy, sigma_xy, Kerr, Faraday (sheet and 6.6 A film)
    lz = float(np.linalg.norm(sysz["rprim"][2]))
    eps_t, eps_b = np.eye(3) + 4 * np.pi * chi, np.eye(3) + 4 * np.pi * ref
    sig_t, sig_b = (tdr.sheet_conductivity(x, w, lz) for x in (chi, ref))
    (kt, ft), (kb, fb) = tdr.kerr_faraday_sheet(sig_t), tdr.kerr_faraday_sheet(sig_b)
    d = 6.6 / tdr.BOHR_IN_ANG
    kf, ff = tdr.kerr_faraday_film(tdr.film_eps(ref, lz, d), w, d)
    cp, cm = tdr.circular_from_strengths(w, eta_s, m.lam.real[pos], ip, im, g_s=g_s,
                                         cell_volume=vol)
    pt_, mt_ = tdr.circular(chi)
    res["mo"] = dict(
        eps_xy_rel_dev=float(np.abs(eps_t[:, 0, 1] - eps_b[:, 0, 1]).max() / np.abs(eps_b[:, 0, 1]).max()),
        sigma_xy_rel_dev=float(np.abs(sig_t[:, 0, 1] - sig_b[:, 0, 1]).max() / np.abs(sig_b[:, 0, 1]).max()),
        kerr_rel_dev=_rot_dev(kt, kb), faraday_rel_dev=_rot_dev(ft, fb),
        film_vs_sheet_kerr=_rot_dev(kf, kb), film_vs_sheet_faraday=_rot_dev(ff, fb),
        chi_plus_vs_table=float(np.abs(pt_ - cp).max() / np.abs(cp).max()),
        chi_minus_vs_table=float(np.abs(mt_ - cm).max() / np.abs(cm).max()),
        max_kerr_rot_mrad=float(1e3 * np.abs(kb.real).max()),
        max_faraday_rot_mrad=float(1e3 * np.abs(fb.real).max()),
        lz_bohr=lz, d_bohr=d)
    import csv
    with open(os.path.join(out, "w2", "magnetooptics_w2.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["omega_ev", "eps_xx_im_td", "eps_xx_im_bse", "eps_xy_re_td", "eps_xy_re_bse",
                     "eps_xy_im_td", "eps_xy_im_bse", "kerr_rot_mrad_td", "kerr_rot_mrad_bse",
                     "kerr_ell_mrad_bse", "faraday_rot_mrad_td", "faraday_rot_mrad_bse",
                     "kerr_rot_mrad_film", "faraday_rot_mrad_film"])
        for i in range(w.size):
            wr.writerow([w[i] * ha, eps_t[i, 0, 0].imag, eps_b[i, 0, 0].imag, eps_t[i, 0, 1].real,
                         eps_b[i, 0, 1].real, eps_t[i, 0, 1].imag, eps_b[i, 0, 1].imag,
                         1e3 * kt[i].real, 1e3 * kb[i].real, 1e3 * kb[i].imag, 1e3 * ft[i].real,
                         1e3 * fb[i].real, 1e3 * kf[i].real, 1e3 * ff[i].real])
    return res


def _f_sum_truncation(tdr, ha):
    """The IP oscillator sum of the 2v2c and 8v8c windows against the whole dipole.h5 band set.

    S_a(window) = 2 sum_k w_k sum_{c, v in window} (E_c - E_v) |<c|r_a|v>|^2 (Ha bohr^2 per
    cell), r = v/(i dE) from the bundle's own velocity and DFT energies (the bundle
    route's r).  The TRK sum of a complete basis with a local potential is N_e = 130
    electrons per cell; the nonlocal pseudopotential velocity and the 160-band cut
    both move the full-set value off N_e.
    """
    import numpy as np
    from file_io.restart_bundle import load_dipole_h5
    from bse.bse_window import _parse_wfn_path
    v, de, _ = load_dipole_h5(BUNDLE["dipole_file"],
                              wfn_path=_parse_wfn_path(BUNDLE["bundle_input"]))
    v = np.asarray(v)                                    # (3, nk, nb, nb) Ry bohr
    de = np.asarray(de)                                  # (nk, nb, nb) Ry, E_n - E_m
    nk, nb = de.shape[0], de.shape[1]
    out = {}
    for name, (c0, c1, v0, v1) in {"2v2c": (130, 132, 128, 130), "8v8c": (130, 138, 122, 130),
                                   "v1-130_c131-%d" % nb: (130, nb, 0, 130)}.items():
        d_ry = de[:, c0:c1, v0:v1]
        r = v[:2, :, c0:c1, v0:v1] / (1j * d_ry[None])   # bohr
        s = 2.0 * np.sum(0.5 * d_ry[None] * np.abs(r) ** 2, axis=(1, 2, 3)) / nk
        out[name] = [float(s[0]), float(s[1])]
    out["bands_in_dipole_file"] = int(nb)
    out["N_e"] = 130
    return out


def main(argv=None) -> int:
    args = _parse(argv)
    os.makedirs(args.out, exist_ok=True)
    if args.cmd == "analyze":
        return analyze(args)
    return run_leg(args, propagate=args.cmd == "leg")


if __name__ == "__main__":
    sys.exit(main())
