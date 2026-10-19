"""The shared-pole collision refresh on CrI3 516 (6x6 k, nosym), measured.

    lx run -N 1 -G 4 -n 4 -- python3 -u tests/tdagw/_spm_refresh_516_case.py \\
        --spm-ref <516>/lorrax/04_gw_oneshot_mpa_eqp --ref <516>/lorrax/07_gw_gnppm_bundle \\
        --out DIR --sections SP0,SPC,SPF

docs/theory/td_agw_dynamical.md §1.4, §7.1 (shared-pole member).  One JSON file
DIR/spm_516.json (rank 0; rewritten after every section) and ``RESULT {...}``.  The
04 and 07 arms are read-only: the source is the production one
(``tdagw.dynsig.gw_window_source`` on a copy of the 04 deck in DIR/driver), the TD
system is the 07 bundle (same WFN.h5, centroids and bands), the TD runs go through
``tdagw.stages.propagate(collision=...)``.  Variants: V1 the production ladder
re-resolved at every state; V2 the low-energy support set (--v2-*: line sites added
below the production ladder, the whole line at --v2-height, small imaginary sites).

SP0  the t0 collision term per variant: the refit at rho_eq against the run's model
     (bitwise), its census, the t0 rates; the V2 equilibrium model's W^c(0) against V1's
SPC  states (rho_eq and the hot state, --hot-amount of band b_lo+1 moved to b_lo+2 at
     every k): per variant the refit (walls, K, gates, supports moved, the census per q
     and q-summed), the band rates on W(state) and on the equilibrium W at the same
     populations, and the band rates at --eta-list (the window sweep's Lorentzian only,
     the model fixed)
SPF  rho_eq propagated --eq-fs with the collision on (refresh every --refresh-fs):
     max|rho - rho_eq| and the held Delta Sigma^h per variant (both zero to round-off)
SPD  driven runs: --cd-pulses x --cd-cadences x --cd-variants (V1, V2, frozen = the
     equilibrium shared-pole W throughout): P(t), n_c, Tr rho, every refresh's record
     (walls, census, K, gates), compiles; the populations at the pulse peak and the end
     of each run are saved (DIR/states_<tag>.npz) and refitted (SPC's measurements)
SPE  eta: the source rerun at sigma_regularization_ev = --eta-low (deck override; the
     recipe couples eta to W) and the refits at rho_eq, the hot state and the --states
     files (the peak / end populations of SPD runs): band rates at --eta-low vs 0.25 eV
Variants with the suffix B (V1B, V2B, V2PB) take the refits' chi0 in four blocks on the
     active bands --chi0-active (``collision_spm_chi0_active``); V2P is V2 with the imaginary
     sites --v2p-imag.  SP0 and SPC then also record the B model's W at the recipe sites
     against the run's model (rho_eq) and against the one-rule refit (the same state).
Variants with the suffix H (V1H, V2PH, ...) refit the q -> 0 head at every state as well
     (``collision_spm_head = refit``, td_agw_dynamical.md §1.4 "Head"): SP0 checks the head refit
     at rho_eq against the run's head (bitwise), SPC records the head of each state (its samples
     W^c_h(z) and eps^-1_h(z) = <W>/<v> with and without the Drude term, its poles, D and the
     diagnostics of ``tdagw.dynsig.spm_head_diagnostics``) and the rates on W(state) with that head.
SPH  per variant, the refit at rho_eq, the hot state and the --states files' peak / end
     populations: the refit's gates (a refused refit is recorded by name, the section goes on),
     the q = 0 census, and for H variants the head of the state (as in SPC)
SV3  the V3 comparator at the hot state (rank 0, host): the equilibrium poles held,
     one residue scale per pole fitted to the V1 refit's W^c at its fit sites; the
     misfit at the fit and held sites against V1's and the weight it cannot place
"""
import argparse
import glob
import json
import os
import sys
import time

import runtime  # noqa: F401  (before numpy: the runtime tunes BLAS spin at import)
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _collision_516_case as base  # noqa: E402  (the 516 deck helpers)

AS_AU, FS_AU = base.AS_AU, base.FS_AU


def _v2_sites(recipe, low_line, low_imag, height_ev):
    """V2's support text: low line sites below the run's line ladder, low imaginary sites below its imaginary ladder."""
    line = sorted(set(float(x) for x in low_line) | set(float(x) for x in recipe["line_ev"]))
    imag = sorted(set(float(x) for x in low_imag) | set(float(x) for x in recipe["imaginary_ev"]))
    text = ",".join(f"{x:.6g}" for x in line) + " | " + ",".join(f"{x:.6g}" for x in imag)
    return text + ("" if height_ev is None else f" | {float(height_ev):.6g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spm-ref", required=True, help="the shared-pole GW arm (04)")
    ap.add_argument("--ref", required=True, help="the TD bundle arm (07): bundle_input, dipoles")
    ap.add_argument("--bands", type=int, nargs=2, default=(128, 132))
    ap.add_argument("--out", required=True)
    ap.add_argument("--sections", default="SP0,SPC,SPF")
    ap.add_argument("--variants", default="V1,V2", help="SP0/SPC/SPF variants")
    ap.add_argument("--v2-line", default="0.25,0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.3",
                    help="V2: line sites (eV) added below the production ladder")
    ap.add_argument("--v2-imag", default="0.1,0.25,0.5", help="V2: imaginary sites (eV) added")
    ap.add_argument("--v2-height", type=float, default=1.0, help="V2: line height (eV); 0 = the run's")
    ap.add_argument("--v2p-imag", default="0.25,0.5", help="V2P: imaginary sites (eV) added")
    ap.add_argument("--chi0-active", default="88,144",
                    help="B variants: the active bands [a0, a1) of the four chi0 blocks")
    ap.add_argument("--hot-amount", type=float, default=0.05)
    ap.add_argument("--eta-list", default="", help="SPC: window-sweep Lorentzians, eV (default: the "
                    "source's; the shared-pole Sigma stage refuses any other: the recipe stamps eta)")
    ap.add_argument("--eq-fs", type=float, default=2.5)
    ap.add_argument("--refresh-fs", type=float, default=0.5)
    ap.add_argument("--refresh-df", type=float, default=1e-3)
    ap.add_argument("--drive-fs", type=float, default=8.0)
    ap.add_argument("--peak-fs", type=float, default=3.0, help="the pulse envelope's centre, fs")
    ap.add_argument("--cd-pulses", default="5e8,3e9")
    ap.add_argument("--cd-cadences", default="0.5,0.25,0.125")
    ap.add_argument("--cd-variants", default="V1")
    ap.add_argument("--cd-tag", default="")
    ap.add_argument("--eta-low", default="0.1,0.05", help="SPE: the lower etas (eV), each a rerun source")
    ap.add_argument("--states", default="", help="SPE, SPH: glob of SPD states_*.npz files")
    ap.add_argument("--spf-variants", default="", help="SPF variants (default: --variants)")
    a = ap.parse_args()
    sections = set(a.sections.split(","))

    stack = runtime.initialize_communicator_stack()
    import h5py
    import jax
    import logging
    from common.collectives import gather_to_host
    from common.units import RYD_TO_EV
    from gw import ppm_sigma as PS
    from tdagw import dynsig as D
    from tdagw import propagate as tp
    from tdagw import stages
    from tdagw.td_config import parse_td_deck
    from tdagw.tdgw_io import efield_function
    mesh = stack.mesh
    rank0 = jax.process_index() == 0
    say = print if rank0 else (lambda *x, **k: None)
    out = dict(mesh=list(mesh.devices.shape), spm_ref=os.path.abspath(a.spm_ref),
               ref=os.path.abspath(a.ref), bands=list(a.bands), head=os.environ.get("LEG_HEAD", ""),
               sections=sorted(sections))
    ref, sref = os.path.abspath(a.ref), os.path.abspath(a.spm_ref)
    spm_deck = os.path.join(sref, "oneshot.in")
    os.makedirs(a.out, exist_ok=True)
    b_lo, b_hi = (int(b) for b in a.bands)
    nA = b_hi - b_lo
    ev, mev = RYD_TO_EV, RYD_TO_EV * 1e3
    jname = os.path.join(a.out, "spm_516.json")

    def _save():
        if rank0:
            with open(jname + ".tmp", "w") as f:
                json.dump(out, f, indent=1, default=base._jsonable)
            os.replace(jname + ".tmp", jname)

    def section(name):
        class _S:
            def __enter__(self):
                self.t = time.perf_counter()
                say(f"[spm_516] section {name} ...", flush=True)
                return self

            def __exit__(self, et, e, tb):
                out.setdefault("section_wall_s", {})[name] = time.perf_counter() - self.t
                if et is not None:
                    import traceback
                    out.setdefault("errors", {})[name] = "".join(
                        traceback.format_exception(et, e, tb))[-4000:]
                    say(f"[spm_516] section {name} FAILED:\n{out['errors'][name]}", flush=True)
                try:
                    _save()
                except Exception as exc:
                    say(f"[spm_516] save after {name} failed: {exc}", flush=True)
                return True
        return _S()

    def mem():
        try:
            st = jax.local_devices()[0].memory_stats() or {}
            return dict(peak_gb=st.get("peak_bytes_in_use", 0) / 1e9,
                        in_use_gb=st.get("bytes_in_use", 0) / 1e9)
        except Exception:
            return {}

    # ---- the source: the 04 deck through the GW driver in this process ----
    src = None
    with section("source"):
        src = D.gw_window_source(spm_deck, mesh, workdir=os.path.join(os.path.abspath(a.out), "driver"),
                                 screening=True, say=say)
        meta = src["meta"]
        rec0 = meta.shared_pole_recipe
        out["source"] = dict(wall_s=src["wall_s"], rc=src["rc"], member=src["member"],
                             nk=int(meta.nk_tot), n_mu=int(meta.n_rmu), memory_after=mem(),
                             eta_ev=float(src["config"].sigma.regularization_ev),
                             K=[min(src["model"]["K"]), max(src["model"]["K"])],
                             recipe=dict(line_ev=list(map(float, rec0["line_ev"])),
                                         imaginary_ev=list(map(float, rec0["imaginary_ev"])),
                                         height_ev=float(rec0["height_ev"]),
                                         held_line_ev=list(map(float, rec0["held_line_ev"])),
                                         held_imaginary_ev=list(map(float, rec0["held_imaginary_ev"])),
                                         recipe_hash=rec0["recipe_hash"]))
    if src is None:
        out["verdict"] = "no source"
        _save()
        runtime.finalize_process(1)
        return
    wfns, meta, splan = src["wfns"], src["meta"], src["plan"]
    nk = int(meta.nk_tot)
    e_full = np.asarray(jax.device_get(wfns.enk), np.float64)
    f_full = np.asarray(jax.device_get(wfns.occ), np.float64)
    v2_text = _v2_sites(rec0, [float(x) for x in a.v2_line.split(",") if x],
                        [float(x) for x in a.v2_imag.split(",") if x],
                        a.v2_height if a.v2_height > 0 else None)
    v2p_text = _v2_sites(rec0, [float(x) for x in a.v2_line.split(",") if x],
                         [float(x) for x in a.v2p_imag.split(",") if x],
                         a.v2_height if a.v2_height > 0 else None)
    sites = dict(V1="", V2=v2_text, V2P=v2p_text)
    sites.update({k + "B": v for k, v in list(sites.items())})
    sites.update({k + "H": v for k, v in list(sites.items())})
    active = [int(x) for x in a.chi0_active.split(",")]
    head_of = lambda V: V.endswith("H")
    core = lambda V: V[:-1] if head_of(V) else V
    blocks_of = lambda V: active if core(V).endswith("B") else None
    out["variants"] = sites

    # ---- the TD system of the window (the 07 bundle, deck route) ----
    sysw = model = None
    with section("system"):
        raw0 = base._td_deck(ref, (b_lo, b_hi), os.path.join(a.out, "td", "probe"), tf_fs=0.1)
        deck0 = parse_td_deck(raw0, source="spm_516")
        sysw = stages.load_system(deck0, mesh)
        model = stages.build_self_energy(deck0, sysw, mesh)
        out["system"] = dict(nk=sysw.nk, nk_pad=sysw.nk_pad, nb=sysw.nb, bands=list(sysw.bands),
                             model=model.name)
    rho_eq_w = np.asarray(gather_to_host(sysw.ops.rho_eq))[:sysw.nk]

    def model_path(V, label):
        """The model file of variant V's refit ``label`` (its scratch slot)."""
        hits = sorted(glob.glob(os.path.join(os.path.abspath(a.out), f"refits_{V}",
                                             label + "_shared_pole", "**", "*.h5"), recursive=True))
        hits = [h for h in hits if "model" in os.path.basename(h)] or hits
        return hits[0] if hits else None

    def w_sites_cmp(path_a, path_b):
        """W^c of model b against model a at a's recipe sites: max over q of ||W_b - W_a||_F / ||W_a||_F.

        Ordered store (header normalization, scripts/sv3_host.py): W^c(q, z) = sum_j C_j C_j^H /
        (2 W_j (z - W_j)) - sum_j conj(C_j(-q)) C_j(-q)^T / (2 W_j(-q) (z + W_j(-q))).  Rank 0, on
        its own device; per site role (fit / held) the max over q and the site of the max.
        """
        if not rank0 or path_a is None or path_b is None:
            return None
        import jax.numpy as jnp
        from symmetry_maps import q_negation_index
        t1 = time.perf_counter()
        dev = jax.local_devices()[0]

        def load(path):
            with h5py.File(path, "r") as f:
                raw = f["header_json"][()]
                hdr = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))
                K = np.asarray(hdr["K"], np.int64)
                return hdr, K, f["poles2_ry2"][()], f["factor"][()]
        ha, Ka, la, Ca = load(path_a)
        hb, Kb, lb, Cb = load(path_b)
        rec = ha["recipe"]
        codes = rec["role_codes"]
        z = np.array([complex(d["real"], d["imag"]) for d in rec["z_ry"]])
        role = np.asarray(rec["role"])
        fit = np.isin(role, [codes["line"], codes["imaginary"]])
        held = np.isin(role, [codes["held_line"], codes["held_imaginary"]])
        keep = fit | held
        neg = np.asarray(q_negation_index((6, 6, 1)))

        def w_all(K, lam, C, q):
            om = jnp.sqrt(jax.device_put(np.asarray(lam[q, :K[q]], np.float64), dev))
            Cq = jax.device_put(np.asarray(C[q, :, :, :K[q]]).reshape(-1, int(K[q])), dev)
            p = int(neg[q])
            omm = jnp.sqrt(jax.device_put(np.asarray(lam[p, :K[p]], np.float64), dev))
            Cm = jax.device_put(np.asarray(C[p, :, :, :K[p]]).reshape(-1, int(K[p])), dev)
            out = []
            for zz in z[keep]:
                a_ = 1.0 / (2.0 * om * (zz - om))
                b_ = 1.0 / (2.0 * omm * (zz + omm))
                out.append((Cq * a_) @ Cq.conj().T - (Cm.conj() * b_) @ Cm.T)
            return out
        worst = dict(fit=0.0, held=0.0)
        per_q = []
        for q in range(int(la.shape[0])):
            wa, wb = w_all(Ka, la, Ca, q), w_all(Kb, lb, Cb, q)
            rel = [float(jnp.linalg.norm(y - x) / jnp.linalg.norm(x)) for x, y in zip(wa, wb)]
            per_q.append(max(rel))
            for r_, f_ in zip(rel, fit[keep]):
                k_ = "fit" if f_ else "held"
                worst[k_] = max(worst[k_], r_)
        return dict(max_rel_fit=worst["fit"], max_rel_held=worst["held"], per_q_max=per_q,
                    n_sites=int(keep.sum()), K=[[int(Ka.min()), int(Ka.max())], [int(Kb.min()), int(Kb.max())]],
                    paths=[path_a, path_b], seconds=time.perf_counter() - t1)

    def band_rates(g_out, g_in):
        """Per band: k-mean, k-min, k-max of the diagonal rates (meV), TD rows."""
        d_o, d_i = (np.real(np.einsum("kii->ki", g)) * mev for g in (g_out, g_in))
        herm = lambda g: 0.5 * (g + np.conj(np.swapaxes(g, -1, -2)))
        return dict(out_mean=d_o.mean(0).tolist(), out_min=d_o.min(0).tolist(),
                    out_max=d_o.max(0).tolist(), in_mean=d_i.mean(0).tolist(),
                    in_min=d_i.min(0).tolist(), in_max=d_i.max(0).tolist(),
                    min_eig_total_mev=float(np.linalg.eigvalsh(herm(g_out + g_in)).min() * mev))

    def eta_plan(plan, eta_ev):
        import dataclasses
        return dict(plan, sigma_cfg=dataclasses.replace(plan["sigma_cfg"],
                                                        regularization_ev=float(eta_ev)))

    def eta_rates(source, at, f_win, eta_ev, bundle_of):
        """Band rates of the window at populations f_win (bundle rows) on ``at``, Lorentzian eta_ev."""
        eye = np.broadcast_to(np.eye(nA, dtype=complex), (nk, nA, nA)).copy()
        w = dict(b_lo=b_lo, U=eye, e=e_full[:, b_lo:b_hi], f=f_win)
        t1 = time.perf_counter()
        o = PS.window_sigma_c(source["wfns"], at, w, eta_plan(source["plan"], eta_ev),
                              meta=source["meta"], mesh_xy=mesh, branches=True)
        go, gi, _ = D.window_branch_rates(o["branch_pair"])
        return dict(band_rates(go[bundle_of], gi[bundle_of]), eta_ev=float(eta_ev),
                    wall_s=time.perf_counter() - t1, tau=int(o["sweep"]["distinct_tau"]))

    def census_out(c, per_q=False):
        keep = ("K", "omega_min_ev", "frac_residue_below", "frac_static_below", "n_below",
                "hist_residue", "hist_static", "residue_total", "static_total")
        d = {k: c[k] for k in keep if k in c}
        d["K"] = [min(c["K"]), max(c["K"])]
        if per_q:
            d["hist_residue_q"] = c["hist_residue_q"]
        return d

    def refit_out(rec, per_q=False):
        d = {k: v for k, v in rec.items() if k not in ("census", "K")}
        d["K_range"] = [min(rec["K"]), max(rec["K"])]
        d["K_per_q"] = rec["K"]
        d["walls"] = D._refit_walls(rec)
        if rec.get("census") is not None:
            d["census"] = census_out(rec["census"], per_q=per_q)
        return d

    def eta_split(rs):
        """Gamma(eta) = a + b eta per band (least squares over the eta list): a = electronic."""
        etas = np.asarray([r["eta_ev"] for r in rs])
        res = {}
        for key in ("out_mean", "in_mean"):
            G = np.asarray([r[key] for r in rs])            # (n_eta, nA)
            A = np.stack([np.ones_like(etas), etas], 1)
            coef, *_ = np.linalg.lstsq(A, G, rcond=None)
            res[key] = dict(electronic_mev=coef[0].tolist(), slope_mev_per_ev=coef[1].tolist(),
                            electronic_fraction_at_max_eta=(coef[0] / np.maximum(
                                np.abs(G[np.argmax(etas)]), 1e-300)).tolist(),
                            log_slope=(np.log(np.maximum(np.abs(G[np.argmax(etas)]), 1e-300)
                                              / np.maximum(np.abs(G[np.argmin(etas)]), 1e-300))
                                       / np.log(etas.max() / etas.min())).tolist())
        return res

    # ---- the t0 collision term per variant (production member, window_collision) ----
    colls = {}
    variants = [v for v in a.variants.split(",") if v]
    if "SPD" in sections:          # the driven runs' variants need their t0 objects (states)
        variants = sorted(set(variants) | {v for v in a.cd_variants.split(",") if v and v != "frozen"})
    if sections & {"SP0", "SPC", "SV3", "SPF", "SPD", "SPH"}:
        with section("SP0"):
            out["SP0"] = {}
            for V in variants:
                t1 = time.perf_counter()
                colls[V] = D.window_collision(
                    src, sysw.ops, mesh=mesh, k_frac=sysw.hdr["kpts"], b_lo=b_lo, ntr=1,
                    form="subtracted", hermitian="dynamic", sweep="active",
                    screening="shared_pole", spm_support_sites=sites[V],
                    spm_scratch=os.path.join(os.path.abspath(a.out), f"refits_{V}"),
                    spm_chi0_active=blocks_of(V), spm_head="refit" if head_of(V) else "frozen",
                    say=say)
                t0r = colls[V]["t0"]
                sc = t0r.get("screen_eq") or {}
                bo = colls[V]["bundle_of"]
                out["SP0"][V] = dict(
                    wall_s=time.perf_counter() - t1, line=t0r.get("line", ""),
                    receipt={k: v for k, v in t0r.items() if k not in ("line", "screen_eq")},
                    refit_eq=refit_out(sc, per_q=True) if sc else None,
                    rates_eq=band_rates(colls[V]["eq"]["g_out"], colls[V]["eq"]["g_in"]),
                    memory=mem())
                if core(V).endswith("B"):
                    out["SP0"][V]["w_sites_vs_run"] = w_sites_cmp(
                        src["model"]["path"], model_path(V, "eq"))
                    if V[:-1] in colls:
                        out["SP0"][V]["w_sites_vs_one_rule"] = w_sites_cmp(
                            model_path(V[:-1], "eq"), model_path(V, "eq"))
                _save()
            if "V1" in colls and "V2" in colls:
                import jax.numpy as jnp
                w1, w2 = colls["V1"]["W0c"], colls["V2"]["W0c"]
                rep_ = lambda x: float(np.asarray(x.addressable_shards[0].data))   # replicated scalar
                out["SP0"]["Wc0_V2_vs_V1_rel"] = (rep_(jnp.max(jnp.abs(w1 - w2)))
                                                  / rep_(jnp.max(jnp.abs(w1))))

    f_eq_b = f_full[:, b_lo:b_hi]
    f_hot_b = f_eq_b.copy()
    f_hot_b[:, 1] -= a.hot_amount
    f_hot_b[:, 2] += a.hot_amount
    etas = ([float(x) for x in a.eta_list.split(",") if x]
            or [float(src["config"].sigma.regularization_ev)])

    def head_measure(hctx, scr, f_b):
        """The head of window populations f_b (bundle rows): samples with and without D, diagnostics."""
        f_full = scr["f_eq"].copy()
        f_full[scr["rows"], scr["win"]] = np.asarray(f_b, np.float64)
        t1 = time.perf_counter()
        head_t, hr = D.spm_head_refit(hctx, f_full)
        t2 = time.perf_counter()
        _, hb = D.spm_head_refit(hctx, f_full, drude=False)
        diag = D.spm_head_diagnostics(hctx, f_full, hr, hb)
        eps_inv = lambda x: ((x["wc"] + x["vc0"]) / x["vc0"])
        return head_t, dict(
            refit_s=t2 - t1, z_ry=hr["z"], wc=hr["wc"], wc_no_drude=hb["wc"],
            eps_inv=eps_inv(hr), eps_inv_no_drude=eps_inv(hb), vc0=hr["vc0"],
            poles_ry=hr["poles"], residues=hr["residues"], poles_no_drude_ry=hb["poles"],
            residues_no_drude=hb["residues"], drude=hr["drude"], diagnostics=diag)

    def state_measure(V, f_b, label, *, per_q=True, at_eq_too=True):
        """The refit at window populations f_b (bundle rows): record, rates on W(state), on W_eq, eta list."""
        cv = colls[V]
        scr, bo = cv["screening"], cv["bundle_of"]
        at, rec = D.spm_refit(scr, f_b, label=label)
        r = dict(refit=refit_out(rec, per_q=per_q))
        head_t = None
        if head_of(V) and cv.get("head_ctx") is not None:
            head_t, r["head"] = head_measure(cv["head_ctx"], scr, f_b)
        t1 = time.perf_counter()
        tabs = cv["sweep"](f_b, at=at, head=head_t)
        r["rates_W_state"] = band_rates(tabs["g_out"], tabs["g_in"])
        r["sweep_W_state_s"] = time.perf_counter() - t1
        if at_eq_too:
            tabs0 = cv["sweep"](f_b, at=None)
            r["rates_W_eq"] = band_rates(tabs0["g_out"], tabs0["g_in"])
        r["eta"] = [eta_rates(src, at, f_b, e_, bo) for e_ in etas]
        if len(etas) > 1:
            r["eta_split"] = eta_split(r["eta"])
        if at_eq_too:
            r["eta_W_eq"] = [eta_rates(src, cv["poles"], f_b, e_, bo) for e_ in etas]
        del at
        return r

    if "SPC" in sections:
        with section("SPC"):
            spc = out.setdefault("SPC", {})
            for V in variants:
                if V not in colls:
                    continue
                spc[V] = dict(eq=state_measure(V, f_eq_b, "c_eq", at_eq_too=False),
                              hot=state_measure(V, f_hot_b, "c_hot"))
                spc[V]["hot"]["state"] = dict(from_band=b_lo + 1, to_band=b_lo + 2,
                                              amount=a.hot_amount)
                if V.endswith("B") and V[:-1] in colls:
                    for st in ("eq", "hot"):
                        spc[V][st]["w_sites_vs_one_rule"] = w_sites_cmp(
                            model_path(V[:-1], "c_" + st), model_path(V, "c_" + st))
                _save()

    log_compiles = base.CompileLog()
    logging.getLogger("jax").addHandler(log_compiles)

    def td_deck(tag, *, tf_fs, dtm_as=50.0, efield=None, refresh_fs=0.0, refresh_df=0.0,
                variant="V1"):
        raw = base._td_deck(ref, (b_lo, b_hi), os.path.join(a.out, "td", tag), tf_fs=tf_fs,
                            dtm_as=dtm_as, efield=efield, collision="subtracted",
                            refresh_fs=refresh_fs, refresh_df=refresh_df, sweep="active")
        if variant != "frozen":
            raw["lorrax"]["collision_screening"] = "shared_pole"
            raw["lorrax"]["collision_source"] = spm_deck
            if sites[variant]:
                raw["lorrax"]["collision_spm_support_sites_ev"] = sites[variant]
            if blocks_of(variant) is not None:
                raw["lorrax"]["collision_spm_chi0_active"] = blocks_of(variant)
            if head_of(variant):
                raw["lorrax"]["collision_spm_head"] = "refit"
        return raw

    def td_run(tag, raw, source):
        deck = parse_td_deck(raw, source=tag)
        outs = stages.open_outputs(deck, sysw)
        lines = []
        jax.config.update("jax_log_compiles", True)
        n0 = len(log_compiles.names)
        t1 = time.perf_counter()
        err = None
        stats = {}
        try:
            stats = stages.propagate(deck, sysw, model, efield_function(deck.trials), outs, mesh,
                                     say=lines.append, collision=source)
        except Exception as exc:              # a refit refused by a gate: recorded, the matrix goes on
            import traceback
            err = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-3000:]
            say(f"[spm_516] run {tag} REFUSED/FAILED:\n{err}", flush=True)
        finally:
            jax.config.update("jax_log_compiles", False)
        res = dict(tag=tag, wall_s=time.perf_counter() - t1, compiles=log_compiles.names[n0:],
                   lines=lines[-20:], error=err,
                   refresh_lines=[l for l in lines if "collision" in l])
        if stats:
            res.update(steps=int(stats["accepted"]), programs=int(stats["programs"]),
                       jit_cache=int(stats["jit_cache"]), refreshes=int(stats["refreshes"]),
                       dt_au=float(stats["dt"]), collision=stats.get("collision"),
                       rows=[{k: r[k] for k in ("t_fs", "wall_s", "s_per_fs", "new_program", "dsigma")}
                             for r in stats["rows"]])
        if rank0 and os.path.exists(outs.paths[0]):
            with h5py.File(outs.paths[0], "r") as f:
                td_ = f["tdgw_data/time_domain"]
                res["rho_t"] = np.asarray(td_["rho_full/data"][()])
                res["pol"] = np.asarray(td_["polarization/data"][()])
                res["nc"] = np.asarray(td_["carrier_density/data"][()])
        res["times_fs"] = np.asarray(outs.mtsteps) / FS_AU
        return res

    def refresh_rows(r):
        """The refresh records of a run (its t0 receipt excluded), on every rank."""
        hist = (r.get("collision") or {}).get("history") or []
        keys = ("n", "t_fs", "why", "df", "wall_s", "sweep_s", "rates_s", "build_s", "hermitian_s",
                "rebase_s", "held_max_mev", "min_rate_mev", "frozen_s", "active_s", "out_per_fs",
                "in_per_fs", "d_trace", "c_eq_now", "screen")
        return [{k: h.get(k) for k in keys} for h in hist if "n" in h]

    def summarize(r):
        """Host summary of a run (rank 0 has the arrays); never raises (the ranks stay in step)."""
        keep = {k: v for k, v in r.items() if k not in ("rho_t", "pol", "nc", "times_fs", "collision")}
        try:
            keep["refresh_summary"] = refresh_rows(r)
            col = r.get("collision") or {}
            if col.get("history"):
                keep["t0"] = {k: v for k, v in col["history"][0].items() if k != "line"}
            if "rho_t" not in r:
                return keep
            rho_t = r["rho_t"]
            n = rho_t.shape[0]
            tr = np.real(np.einsum("tkii->t", rho_t)) / rho_t.shape[1]
            pops = np.real(np.einsum("tkii->ti", rho_t)) / rho_t.shape[1]
            dev = np.max(np.abs(rho_t - rho_eq_w[None]), axis=(1, 2, 3))
            pol, nc = (np.asarray(r[k]).reshape(n, -1) for k in ("pol", "nc"))
            keep.update(max_dev=float(dev.max()), dev_t=dev.tolist(), trace_t=(tr - tr[0]).tolist(),
                        pops_t=pops.tolist(), pol_t=np.real(pol).tolist(), nc_t=np.real(nc).tolist(),
                        times_fs=np.asarray(r["times_fs"])[:n].tolist())
        except Exception as exc:                   # a summary bug must not desynchronize the ranks
            keep["summary_error"] = repr(exc)
        return keep

    if "SPF" in sections:
        with section("SPF"):
            spf = out.setdefault("SPF", {})
            for V in ([v for v in a.spf_variants.split(",") if v] or variants):
                raw = td_deck(f"eq_{V}", tf_fs=a.eq_fs, refresh_fs=a.refresh_fs, variant=V)
                r = summarize(td_run(f"eq_{V}", raw, src))
                spf[V] = {k: r.get(k) for k in ("max_dev", "dev_t", "trace_t", "refreshes", "steps",
                                                "wall_s", "programs", "jit_cache", "compiles",
                                                "refresh_summary", "error", "lines", "refresh_lines",
                                                "t0", "summary_error")}
                _save()

    def save_states(tag, r):
        """The window populations (bundle rows) at the pulse peak and at the end of a run.

        Rank 0 reads the run's output file; the states are broadcast so every rank
        takes the same branch (the refits that follow are collective).
        """
        from jax.experimental import multihost_utils
        bo = next(iter(colls.values()))["bundle_of"]
        have, buf, tt = np.zeros(1), np.zeros((2, nk, nA)), np.zeros(2)
        if "rho_t" in r:
            n = r["rho_t"].shape[0]
            t = np.asarray(r["times_fs"])[:n]
            for j, i in enumerate((int(np.argmin(np.abs(t - a.peak_fs))), n - 1)):
                buf[j][bo] = np.real(np.einsum("kii->ki", r["rho_t"][i]))
                tt[j] = t[i]
            have[0] = 1.0
        have, buf, tt = (np.asarray(multihost_utils.broadcast_one_to_all(x)) for x in (have, buf, tt))
        if have[0] == 0.0:
            return None
        sts = dict(peak=dict(t_fs=float(tt[0]), f_b=buf[0]), end=dict(t_fs=float(tt[1]), f_b=buf[1]))
        if rank0:
            np.savez(os.path.join(a.out, f"states_{tag}.npz"),
                     **{f"{k}_f_b": v["f_b"] for k, v in sts.items()},
                     **{f"{k}_t_fs": v["t_fs"] for k, v in sts.items()})
        return sts

    if "SPD" in sections:
        with section("SPD"):
            spd = out.setdefault("SPD", {})
            cdv = [v for v in a.cd_variants.split(",") if v]
            for amp in (float(v) for v in a.cd_pulses.split(",")):
                for V in cdv:
                    cads = [float(v) for v in a.cd_cadences.split(",")] if V != "frozen" else [0.25]
                    for cad in cads:
                        tag = f"E{amp:g}_r{cad:g}_{V}{a.cd_tag}"
                        dtm = 25.0 if cad < 0.25 else 50.0
                        raw = td_deck(tag, tf_fs=a.drive_fs, dtm_as=dtm, efield=base_pulse(amp),
                                      refresh_fs=cad, refresh_df=a.refresh_df, variant=V)
                        r = td_run(tag, raw, src)
                        sts = save_states(tag, r)
                        rs = summarize(r)
                        rs.update(amp=amp, cadence_fs=cad, variant=V, dtm_as=dtm,
                                  states_t_fs=None if sts is None else
                                  {k: v["t_fs"] for k, v in sts.items()})
                        if sts is not None and V != "frozen" and cad == 0.25 and V in colls:
                            rs["states"] = {k: state_measure(V, v["f_b"], f"d_{k}")
                                            for k, v in sts.items()}
                        spd[tag] = rs
                        _save()

    if "SPH" in sections:
        with section("SPH"):
            sph = out.setdefault("SPH", {})
            states = [("eq", f_eq_b), ("hot", f_hot_b)]
            for p_ in sorted(glob.glob(a.states)) if a.states else []:
                z_ = np.load(p_)
                tg = os.path.basename(p_)[len("states_"):-len(".npz")]
                for nm in ("peak", "end"):
                    if f"{nm}_f_b" in z_:
                        states.append((f"{tg}:{nm}", np.asarray(z_[f"{nm}_f_b"])))
            for V in variants:
                if V not in colls:
                    continue
                cv = colls[V]
                sv = sph.setdefault(V, {})
                for nm, f_b in states:
                    row = dict(n_c=float(np.sum(np.clip(np.asarray(f_b) - f_eq_b, 0.0, None)) / nk))
                    try:
                        at, rec = D.spm_refit(cv["screening"], f_b, label="h_state")
                        row["refit"] = refit_out(rec, per_q=True)
                        del at
                    except Exception as exc:          # a refused refit: recorded, the section goes on
                        row["refit_error"] = f"{type(exc).__name__}: {exc}"[:1500]
                    if head_of(V) and cv.get("head_ctx") is not None:
                        _, row["head"] = head_measure(cv["head_ctx"], cv["screening"], f_b)
                    sv[nm] = row
                    _save()

    if "SPE" in sections:
        with section("SPE"):
            import gc
            spe = out.setdefault("SPE", {})
            states = [("eq", f_eq_b), ("hot", f_hot_b)]
            for p_ in sorted(glob.glob(a.states)) if a.states else []:
                z = np.load(p_)
                tg = os.path.basename(p_)[len("states_"):-len(".npz")]
                for nm in ("peak", "end"):
                    if f"{nm}_f_b" in z:
                        states.append((f"{tg}:{nm}", np.asarray(z[f"{nm}_f_b"])))
            for eta_e in (float(x) for x in a.eta_low.split(",") if x):
                key = f"eta{eta_e:g}"
                src_e = D.gw_window_source(
                    spm_deck, mesh, workdir=os.path.join(os.path.abspath(a.out), f"driver_{key}"),
                    screening=True, overrides={"sigma_regularization_ev": eta_e}, say=say)
                re_ = src_e["meta"].shared_pole_recipe
                spe[key] = dict(eta_ev=eta_e, source=dict(
                    wall_s=src_e["wall_s"], eta_ev=float(src_e["config"].sigma.regularization_ev),
                    K=[min(src_e["model"]["K"]), max(src_e["model"]["K"])],
                    line_ev=list(map(float, re_["line_ev"])),
                    imaginary_ev=list(map(float, re_["imaginary_ev"])),
                    height_ev=float(re_["height_ev"])))
                for V in variants:
                    v2e = "" if V == "V1" else _v2_sites(
                        re_, [float(x) for x in a.v2_line.split(",") if x],
                        [float(x) for x in a.v2_imag.split(",") if x],
                        a.v2_height if a.v2_height > 0 else None)
                    ce = D.window_collision(
                        src_e, sysw.ops, mesh=mesh, k_frac=sysw.hdr["kpts"], b_lo=b_lo, ntr=1,
                        form="subtracted", hermitian="dynamic", sweep="active",
                        screening="shared_pole", spm_support_sites=v2e,
                        spm_scratch=os.path.join(os.path.abspath(a.out), f"refits_{key}_{V}"),
                        spm_chi0_active=blocks_of(V), say=say)
                    se = dict(t0_line=ce["t0"].get("line", ""), sites=v2e, states={})
                    for nm, f_b in states:
                        at, rec = D.spm_refit(ce["screening"], f_b, label="e_state")
                        se["states"][nm] = dict(
                            refit=refit_out(rec, per_q=False),
                            rates=eta_rates(src_e, at, f_b, eta_e, ce["bundle_of"]))
                        del at
                        _save()
                    spe[key][V] = se
                    del ce
                    gc.collect()
                    _save()
                del src_e
                gc.collect()

    if "SV3" in sections and "V1" in colls:
        with section("SV3"):
            out["SV3"] = sv3(colls["V1"], f_hot_b, src, say)

    out["compiles_total"] = len(log_compiles.names)
    out["memory_end"] = mem()
    out["wall_s"] = sum(out.get("section_wall_s", {}).values())
    _save()
    if rank0:
        print("RESULT " + json.dumps(dict(errors=sorted(out.get("errors", {})),
                                          sections=out.get("section_wall_s")), default=float),
              flush=True)
    runtime.finalize_process(0 if not out.get("errors") else 1)


def base_pulse(amp):
    """The 516 case's pulse with peak field ``amp`` V/m."""
    return f"{float(amp)!r} * " + base.PULSE.split(" * ", 1)[1]


def sv3(coll, f_hot_b, src, say):
    """V3: the equilibrium poles held, one residue scale per pole fitted to V1's W^c at its fit sites.

    Rank 0, on its first local device (no collective).  W^c(z) = sum_j s_j b_j b_j^H /
    (z^2 - Omega_j^2) in the store's factor basis (the census convention, the even
    store); the scales s_j solve the least-squares problem over the recipe's fit sites
    against the V1 refit of the hot state.  Returns per q: the misfit of V3 relative to
    the change W_V1 - W_eq at the fit and the held sites, and the V1 weight below the
    equilibrium Omega_min (which no V3 can place).
    """
    import jax
    import jax.numpy as jnp
    import h5py
    from common.units import RYD_TO_EV
    from tdagw import dynsig as D
    scr = coll["screening"]
    at, rec = D.spm_refit(scr, f_hot_b, label="v3_hot")
    if jax.process_index() != 0:
        return None
    dev = jax.local_devices()[0]
    recipe = scr["meta"].shared_pole_recipe
    codes = recipe["role_codes"]
    z, role = np.asarray(recipe["z_ry"]), np.asarray(recipe["role"])
    fit_m = np.isin(role, [codes["line"], codes["imaginary"]])
    held_m = np.isin(role, [codes["held_line"], codes["held_imaginary"]])

    def load(path, q):
        with h5py.File(path, "r") as f:
            raw = f["header_json"][()]
            K = json.loads(raw.decode() if isinstance(raw, bytes) else str(raw))["K"]
            lam = np.asarray(f["poles2_ry2"][q, :K[q]], np.float64)
            b = np.asarray(f["factor"][q, :, :, :K[q]])
        return lam, b.reshape(-1, b.shape[-1])

    put = lambda x: jax.device_put(jnp.asarray(x), dev)

    def wc(lam, b, zz, scale=None):
        g = 1.0 / (zz * zz - lam)
        if scale is not None:
            g = g * scale
        return (b * g[None, :]) @ jnp.conj(b).T

    res = {}
    t0 = time.perf_counter()
    for q in (0, int(len(at["model"]["K"]) // 2)):
        lam0, b0 = (put(x) for x in load(coll["poles"]["model"]["path"], q))
        lamh, bh = (put(x) for x in load(at["model"]["path"], q))
        fit_z, held_z = z[fit_m], z[held_m]
        gram = jnp.abs(jnp.conj(b0).T @ b0) ** 2
        A = jnp.zeros(gram.shape, jnp.complex128)
        y = jnp.zeros(gram.shape[0], jnp.complex128)
        for zz in fit_z:
            g = 1.0 / (zz * zz - lam0)
            A = A + jnp.conj(g)[:, None] * gram * g[None, :]
            y = y + jnp.conj(g) * jnp.einsum("ij,ij->j", jnp.conj(b0), wc(lamh, bh, zz) @ b0)
        s = np.linalg.lstsq(np.asarray(A), np.asarray(y), rcond=1e-12)[0]
        sd = put(s)
        mis = {}
        for nm, zs in (("fit", fit_z), ("held", held_z)):
            rows = []
            for zz in zs:
                Wt, We, W3 = wc(lamh, bh, zz), wc(lam0, b0, zz), wc(lam0, b0, zz, sd)
                d = float(jnp.linalg.norm(Wt - We))
                rows.append(dict(z_ev=[float(np.real(zz) * RYD_TO_EV), float(np.imag(zz) * RYD_TO_EV)],
                                 change_V1_rel=d / float(jnp.linalg.norm(We)),
                                 V3_misfit_of_change=float(jnp.linalg.norm(W3 - Wt)) / max(d, 1e-300),
                                 eq_misfit_of_change=1.0))
            mis[nm] = rows
        om0 = np.sqrt(np.asarray(lam0)) * RYD_TO_EV
        omh = np.sqrt(np.asarray(lamh)) * RYD_TO_EV
        wr = np.asarray(jnp.sum(jnp.abs(bh) ** 2, 0)) / (2.0 * np.sqrt(np.asarray(lamh)))
        res[f"q{q}"] = dict(omega_min_eq_ev=float(om0.min()), omega_min_V1_ev=float(omh.min()),
                            V1_residue_below_eq_min=float(wr[omh < om0.min()].sum() / wr.sum()),
                            scales=dict(min=float(np.min(np.real(s))), max=float(np.max(np.real(s))),
                                        n_negative=int(np.sum(np.real(s) < 0))),
                            misfit=mis)
        del lam0, b0, lamh, bh, gram, A, y, sd
    res["wall_s"] = time.perf_counter() - t0
    say(f"[spm_516] SV3 done in {res['wall_s']:.1f} s")
    return res


if __name__ == "__main__":
    main()
