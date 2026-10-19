"""The GN-PPM collision term on a real GN-PPM GW run (CrI3 516, 6x6 k, nosym), measured.

    lx run -N 1 -G 4 -n 4 -- python3 -u tests/tdagw/_collision_516_case.py \\
        --ref <516>/lorrax/07_gw_gnppm_bundle --bands 128 132 --wide 120 144 --out DIR

docs/theory/td_agw_dynamical.md §1.5.  One JSON file DIR/collision_516.json (rank
0; rewritten after every section) and ``RESULT {...}``.  The 07 arm is read-only:
the collision source is the production one (``tdagw.dynsig.gw_window_source``: the
GW driver in this process on a copy of the deck in DIR/driver), the TD runs go
through ``tdagw.stages.propagate(collision=...)``.

D0  the driver run vs the 07 arm: every sigma_mnk.h5 dataset bitwise,
    sigma_diag.dat identical (generator line excluded)
A   (a) at rho_eq, one sweep with both branches: cond + val + static == pair;
    pair (branches kept) == pair (incumbent call) and == the file's sigma_c_kij_ev
B   (b) an independent rate check: the body-only branch tables (head None) vs an
    explicit NumPy (q, s, mu, nu) sum with the run's GN-PPM residues R+- = B +- D,
    elementwise Omega and the sweep's Lorentzian (eta = sigma_regularization_ev),
    hole branch time ordered, for the window at a few k: G~>, G~<, the branch pairs
C   (c) t0 rates: min eigenvalues and diagonal ranges of G~>, G~< (meV), per band,
    at the run's eta and at --eta2 (the regularization tails); ||C[rho_eq]||
    diagonal vs off-diagonal; rho_eq propagated --eq-fs with the collision on:
    subtracted (deviation) and raw (drift)
DR  (d) a driven run (the 5e8 V/m pulse of the 516 dynsig case) with the collision
    off / subtracted (refreshed) / raw: populations, Tr rho, P(t), refresh walls
    (sweep, rates, dissipator, hermitian) vs the incumbent window_sigma_c call,
    chunk programs and compiles, the Magnus history rebase (the hook's delta
    dropped), hermitian = full vs dynamic, the per-step cost of the flow
H   the chunk program's collectives, collision on vs off (k-locality)
W   the wide window (--wide): rates at rho_eq per band (both eta), the hot-carrier
    state (--hot-from band, --hot-to band, --hot-amount) relaxed with the collision
    on (refreshed), populations and Tr rho
S   the refresh sweeps (td_agw_dynamical.md §1.6) on --bands and --wide: tau nodes
    per window and warm walls of the incumbent sweep (GW run's on-shell list), of
    the whole sum planned on the window's own energies, of the frozen part (once)
    and of the active part (per refresh, held session); the split Sigma^F +
    Sigma^A against the incumbent at rho_eq and at a hot-carrier state (rates,
    S~, static limit)
SD  the driven run of DR with the split refresh (collision_refresh_sweep =
    active), every refresh checked against a full sweep at the same rho
CS  the refreshed screening (collision_screening = refreshed) per window: its rules,
    the rho_eq fixed point (dchi0 = 0, W = W0, the run's poles), a hot state's pole
    census and walls, the window Sigma on W(t) (split vs full, the frozen part
    planned on the window vs on the GW list, the frozen-band truncation), and the
    size of the dW effect on the band-edge rates against Re Sigma
C0  the run's chi0 rebuilt from the split at both supports (chi0^ff + fa + aa):
    W of the Dyson solve with V against the run's W samples
CD  driven runs: pulses x refresh cadences x screening (frozen / refreshed), the
    split refresh, a max|df| trigger; per-refresh census, P(t), populations
"""
import argparse
import copy
import json
import logging
import os
import re
import time

import runtime  # noqa: F401  (before numpy: the runtime tunes BLAS spin at import)
import numpy as np

AS_AU = 1.0 / 24.188843265857           # a.u. of time per as
FS_AU = 1000.0 * AS_AU
PULSE = "5e8 * np.exp(-((t - 3000.0) / 1000.0)**2) * np.cos(0.07 * t / 24.188843265857)"


def _jsonable(o):
    """json.dump's default: complex -> [re, im], numpy scalars and arrays -> Python."""
    if isinstance(o, complex) or np.iscomplexobj(o):
        a = np.asarray(o)
        return [np.real(a).tolist(), np.imag(a).tolist()]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return str(o)


class CompileLog(logging.Handler):
    PAT = re.compile(r"Compiling (\S+)")

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.names = []

    def emit(self, record):
        m = self.PAT.search(record.getMessage())
        if m:
            self.names.append(m.group(1))


def _td_deck(ref, bands, outdir, *, tf_fs, dtm_as=50.0, efield=None, collision="off",
             refresh_fs=0.0, refresh_df=0.0, hermitian="dynamic", method="cfm4", dt_as=25.0,
             chunk=5, degeneracy="", ivbm=130, sweep="full", screening="frozen",
             frozen_bands=None, gamma_ev=None):
    b_lo, b_hi = bands
    ex = efield or 0.0
    raw = {"control": {"calc_type": "COHSEX",
                       "timesteps": {"t0": 0.0, "tf": float(tf_fs) * 1000.0, "dt_measure": dtm_as,
                                     "ntcheck": chunk},
                       "files": {"outdir": outdir}},
           "system": {"ibndmin": b_lo + 1, "ibndmax": b_hi, "ivbm": ivbm, "geometry": "slab",
                      "esmear": 0.05, "Efunc": {"E_x": ex, "E_y": 0.0, "E_z": 0.0}},
           "measurements": {"types": ["Efield", "polarization", "carrier_density",
                                      "density_matrix"],
                            "3": {"dataset_name": "rho_full", "brange1": [b_lo + 1, b_hi + 1],
                                  "brange2": [b_lo + 1, b_hi + 1]}},
           "lorrax": {"method": method, "dt": dt_as, "n_corr": 2, "chunk": chunk,
                      "system": "bundle", "bundle_input": os.path.join(ref, "oneshot.in"),
                      "dipole_file": os.path.join(ref, "dipole.h5"), "model": "isdf",
                      "collision": collision, "collision_refresh_fs": refresh_fs,
                      "collision_refresh_df": refresh_df, "collision_hermitian": hermitian}}
    if degeneracy:
        raw["lorrax"]["degeneracy_mode"] = degeneracy
    if sweep != "full":
        raw["lorrax"]["collision_refresh_sweep"] = sweep
    if screening != "frozen":
        raw["lorrax"]["collision_screening"] = screening
    if frozen_bands is not None:
        raw["lorrax"]["collision_frozen_bands"] = [int(v) for v in frozen_bands]
    if gamma_ev is not None and screening == "refreshed":
        raw["lorrax"]["collision_screening_gamma_ev"] = float(gamma_ev)
    return raw


def _toml(raw):
    """A TOML deck of ``raw`` (strings, numbers, booleans, nested tables, lists)."""
    lines = []

    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, (list, tuple)):
            return "[" + ", ".join(val(x) for x in v) + "]"
        return repr(v)

    def table(prefix, d):
        scal = {k: v for k, v in d.items() if not isinstance(v, dict)}
        if prefix:
            lines.append(f"[{prefix}]")
        for k, v in scal.items():
            lines.append(f"{k} = {val(v)}")
        lines.append("")
        for k, v in d.items():
            if isinstance(v, dict):
                table(f"{prefix}.{k}" if prefix else k, v)
    table("", raw)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--bands", type=int, nargs=2, default=(128, 132))
    ap.add_argument("--wide", type=int, nargs=2, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eta2", type=float, default=0.05, help="second regularization width, eV")
    ap.add_argument("--dense-k", default="0,7", help="k rows of the NumPy reference")
    ap.add_argument("--eq-fs", type=float, default=2.5)
    ap.add_argument("--drive-fs", type=float, default=8.0)
    ap.add_argument("--refresh-fs", type=float, default=0.5)
    ap.add_argument("--refresh-df", type=float, default=1e-3)
    ap.add_argument("--hot-from", type=int, default=129)
    ap.add_argument("--hot-to", type=int, default=139)
    ap.add_argument("--hot-amount", type=float, default=0.1)
    ap.add_argument("--hot-fs", type=float, default=4.0)
    ap.add_argument("--hot2v-amount", type=float, default=0.05,
                    help="S: the 2v2c hot state moves this from band 129 to 130 (0-based)")
    ap.add_argument("--sections", default="D0,A,B,C,DR,H,W")
    ap.add_argument("--cs-trunc", default="4,16", help="CS: frozen-band truncation widths")
    ap.add_argument("--cd-window", default="2v2c", choices=("2v2c", "wide"))
    ap.add_argument("--cd-pulses", default="5e8,3e9", help="CD: peak fields, V/m")
    ap.add_argument("--cd-cadences", default="0.5,0.25,0.125", help="CD: refresh cadences, fs")
    ap.add_argument("--cd-screenings", default="frozen,refreshed")
    ap.add_argument("--cd-frozen-bands", default="", help="CD refreshed: 'n_below,n_above'")
    ap.add_argument("--cd-check-trunc", action="store_true",
                    help="CD refreshed with frozen bands: also sweep the whole frozen part")
    ap.add_argument("--cd-tag", default="", help="CD: suffix of the run tags")
    ap.add_argument("--cd-dtm-as", type=float, default=50.0,
                    help="CD: the sample interval (as); a refresh happens at chunk ends, chunk = 5 samples")
    ap.add_argument("--cd-rebase", action="store_true",
                    help="CD refreshed: rebase the subtracted form's C[rho_eq] on W(t)")
    ap.add_argument("--cd-gamma-ev", type=float, default=None,
                    help="CD refreshed: collision_screening_gamma_ev (default: the deck's)")
    ap.add_argument("--drive-cases", default="off,sub,raw,sub_full,sub_norebase")
    a = ap.parse_args()
    sections = set(a.sections.split(","))

    stack = runtime.initialize_communicator_stack()
    import h5py
    import jax
    import jax.numpy as jnp
    from jax.experimental import multihost_utils
    from common.collectives import gather_to_host
    from common.units import RYD_TO_EV
    from gw import ppm_sigma as PS
    from tdagw import collision as C
    from tdagw import dynsig as D
    from tdagw import propagate as tp
    from tdagw import stages
    from tdagw.td_config import parse_td_deck
    from tdagw.tdgw_io import efield_function
    mesh = stack.mesh
    rank0 = jax.process_index() == 0
    say = print if rank0 else (lambda *x, **k: None)
    gather = lambda x: np.asarray(multihost_utils.process_allgather(x, tiled=True))
    out = dict(mesh=list(mesh.devices.shape), ref=os.path.abspath(a.ref), bands=list(a.bands),
               wide=a.wide, head=os.environ.get("LEG_HEAD", ""))
    ref = os.path.abspath(a.ref)
    os.makedirs(a.out, exist_ok=True)
    b_lo, b_hi = (int(b) for b in a.bands)
    nA = b_hi - b_lo
    ev, mev = RYD_TO_EV, RYD_TO_EV * 1e3

    def _save():
        if rank0:
            with open(os.path.join(a.out, "collision_516.json"), "w") as f:
                json.dump(out, f, indent=1, default=_jsonable)

    def section(name):
        class _S:
            def __enter__(self):
                self.t = time.perf_counter()
                say(f"[collision_516] section {name} ...", flush=True)
                return self

            def __exit__(self, et, e, tb):
                out.setdefault("section_wall_s", {})[name] = time.perf_counter() - self.t
                if et is not None:
                    import traceback
                    out.setdefault("errors", {})[name] = "".join(
                        traceback.format_exception(et, e, tb))[-3000:]
                    say(f"[collision_516] section {name} FAILED:\n{out['errors'][name]}", flush=True)
                try:
                    _save()
                except Exception as exc:        # a record that does not serialize stays unsaved
                    say(f"[collision_516] save after {name} failed: {exc}", flush=True)
                return True                     # the next section still runs
        return _S()

    def mem():
        try:
            st = jax.local_devices()[0].memory_stats() or {}
            return dict(peak_gb=st.get("peak_bytes_in_use", 0) / 1e9,
                        in_use_gb=st.get("bytes_in_use", 0) / 1e9,
                        limit_gb=st.get("bytes_limit", 0) / 1e9)
        except Exception:
            return {}

    # ---- the collision source: the GW driver in this process (production path) ----
    src = None
    with section("source"):
        src = D.gw_window_source(os.path.join(ref, "oneshot.in"), mesh,
                             workdir=os.path.join(os.path.abspath(a.out), "driver"),
                             screening=bool({"CS", "C0", "CD"} & sections), say=say)
        meta = src["meta"]
        out["source"] = dict(wall_s=src["wall_s"], rc=src["rc"], nk=int(meta.nk_tot),
                             n_mu=int(meta.n_rmu), memory_after=mem(),
                             eta_ev=float(src["config"].sigma.regularization_ev))
    if src is None:
        out["verdict"] = "no source"
        _save()
        runtime.finalize_process(1)
        return
    wfns, poles, splan = src["wfns"], src["poles"], src["plan"]
    nk = int(meta.nk_tot)
    e_full = np.asarray(jax.device_get(wfns.enk), np.float64)
    f_full = np.asarray(jax.device_get(wfns.occ), np.float64)

    if "D0" in sections:
        with section("D0"):
            d0 = {}
            drv = os.path.join(os.path.abspath(a.out), "driver")
            with h5py.File(os.path.join(ref, "sigma_mnk.h5"), "r") as fa, \
                    h5py.File(os.path.join(drv, "sigma_mnk.h5"), "r") as fb:
                for k in sorted(fa.keys()):
                    if not isinstance(fa[k], h5py.Dataset):
                        continue
                    x = np.asarray(fa[k][()])
                    y = np.asarray(fb[k][()]) if k in fb else None
                    d0[k] = bool(y is not None and x.shape == y.shape and np.array_equal(x, y))
            strip = lambda p: [l for l in open(p) if not l.startswith("# Generated")]
            same = strip(os.path.join(ref, "sigma_diag.dat")) == strip(os.path.join(drv,
                                                                                  "sigma_diag.dat"))
            out["D0"] = dict(datasets=d0, all_bitwise=all(d0.values()), n=len(d0),
                             sigma_diag_identical=bool(same))

    # ---- the TD system of the window (the deck route) ----
    sysw = model = None
    with section("system"):
        raw0 = _td_deck(ref, (b_lo, b_hi), os.path.join(a.out, "td", "probe"), tf_fs=0.1)
        deck0 = parse_td_deck(raw0, source="collision_516")
        sysw = stages.load_system(deck0, mesh)
        model = stages.build_self_energy(deck0, sysw, mesh)
        out["system"] = dict(nk=sysw.nk, nk_pad=sysw.nk_pad, nb=sysw.nb, bands=list(sysw.bands),
                             model=model.name)

    # ---- the t0 collision term on the window (production member) ----
    coll = None
    with section("t0"):
        coll = D.window_collision(src, sysw.ops, mesh=mesh, k_frac=sysw.hdr["kpts"], b_lo=b_lo,
                                 ntr=1, form="subtracted", hermitian="dynamic", say=say)
        out["t0"] = {k: v for k, v in coll["t0"].items() if k != "line"}
    bundle_of = None if coll is None else coll["bundle_of"]
    eq = None if coll is None else coll["eq"]

    if "A" in sections and eq is not None:
        with section("A"):
            r = eq["raw"]
            tabs, pair = r["branch_pair"], r["pair"]
            sumb = tabs["cond"] + tabs["val"] + tabs["static"][None]
            eye = np.broadcast_to(np.eye(nA, dtype=complex), (nk, nA, nA)).copy()
            win = dict(b_lo=b_lo, U=eye, e=e_full[:, b_lo:b_hi], f=f_full[:, b_lo:b_hi])
            t1 = time.perf_counter()
            inc = PS.window_sigma_c(wfns, poles, win, splan, meta=meta, mesh_xy=mesh)
            t_inc = time.perf_counter() - t1
            t1 = time.perf_counter()
            br = PS.window_sigma_c(wfns, poles, win, splan, meta=meta, mesh_xy=mesh, branches=True)
            t_br = time.perf_counter() - t1
            with h5py.File(os.path.join(ref, "sigma_mnk.h5"), "r") as f:
                fc = np.asarray(f["sigma_c_kij_ev"][()])[:, :, b_lo:b_hi, b_lo:b_hi]
            mx = lambda x: float(np.max(np.abs(x)))
            out["A"] = dict(
                split_rel=mx(sumb - pair) / mx(pair), split_max_abs_ev=mx(sumb - pair) * ev,
                kept_vs_incumbent_max_abs_ev=mx(br["pair"] - inc["pair"]) * ev,
                kept_vs_incumbent_bitwise=bool(np.array_equal(br["pair"], inc["pair"])),
                incumbent_vs_file_max_abs_ev=mx(inc["pair"] * ev - fc),
                branches_vs_file_max_abs_ev=mx(sumb * ev - fc), max_abs_sigma_c_ev=mx(fc),
                max_abs_branch_ev={k: mx(v) * ev for k, v in tabs.items()},
                wall_incumbent_s=t_inc, wall_branches_s=t_br)

    if "B" in sections and eq is not None:
        with section("B"):
            ks = [int(x) for x in a.dense_k.split(",")]
            plan_nohead = dict(splan, head=None)
            eye = np.broadcast_to(np.eye(nA, dtype=complex), (nk, nA, nA)).copy()
            win = dict(b_lo=b_lo, U=eye, e=e_full[:, b_lo:b_hi], f=f_full[:, b_lo:b_hi])
            body = PS.window_sigma_c(wfns, poles, win, plan_nohead, meta=meta, mesh_xy=mesh,
                                     branches=True)["branch_pair"]
            psi = gather(wfns.green_parent.psi_nmu)                     # (nk, nb_pad, ns, mu)
            nbf = int(wfns.slices.nb_full_logical)
            nbs = int(src["request"].tables_ry.shape[-1]) if src["request"] is not None else nbf
            Om, Bq, Dq, good = (gather(x) for x in (poles.Omega_q, poles.B_q, poles.B_odd_q,
                                                    poles.valid_mask_q))
            if Om.shape[0] != nk or Om.shape[-1] != psi.shape[-1]:
                raise ValueError(f"B: poles {Om.shape} vs carrier {psi.shape} (want nq = nk, "
                                 "one mu axis)")
            eta = float(src["config"].sigma.regularization_ev) / ev
            from tdagw.model import flat_k_rows
            kf = np.asarray(wfns.green_parent.plan.k_parent_frac, np.float64)
            kg = tuple(int(n) for n in meta.kgrid)
            rows = np.asarray(flat_k_rows(kf, kg))
            if not np.array_equal(rows, np.arange(nk)):
                raise ValueError("B needs the parent rows in flat C order (as 516's)")
            ijk = np.stack(np.unravel_index(np.arange(nk), kg), axis=1)
            kmq = np.stack([np.ravel_multi_index(tuple(((ijk - ijk[q]) % np.asarray(kg)).T), kg)
                            for q in range(nk)])                      # [q, k] -> k - q
            Rp = np.where(good, Bq + Dq, 0.0)
            Rm = np.where(good, Bq - Dq, 0.0)
            Om = np.where(good, Om, 1.0)                              # invalid: no pole (R = 0)
            nsum = int(wfns.slices.nb_sigma_sum) if hasattr(wfns.slices, "nb_sigma_sum") else nbf
            res = {}
            mine = [k for i, k in enumerate(ks) if i % jax.process_count() == jax.process_index()]
            for k in mine:
                proj = psi[k, b_lo:b_hi]                              # (nA, ns, mu)
                Ek = e_full[k, b_lo:b_hi]
                S = {p: np.zeros((nA, nA, nA), complex) for p in ("val", "cond")}   # [E, i, j]
                for q in range(nk):
                    kb = kmq[q, k]
                    amat = np.einsum("isu,asu->aiu", np.conj(proj), psi[kb, :nsum])  # (s, i, mu)
                    for s_ in range(nsum):
                        es, fs = e_full[kb, s_], f_full[kb, s_]
                        ai = amat[s_]                                  # (i, mu)
                        for iE, E in enumerate(Ek):
                            if fs > 0.0:
                                X = fs * Rm[q] / (E - es + Om[q] - 1j * eta)
                                S["val"][iE] += ai @ X @ np.conj(ai).T / nk
                            if fs < 1.0:
                                X = (1.0 - fs) * Rp[q] / (E - es - Om[q] + 1j * eta)
                                S["cond"][iE] += ai @ X @ np.conj(ai).T / nk
                res[k] = {p: np.stack([np.stack([S[p][i, i, :] for i in range(nA)]),   # T1_ij = S(e_i)
                                       np.stack([S[p][j, :, j] for j in range(nA)]).T])  # T2_ij = S(e_j)
                          for p in S}
            allres = multihost_utils.process_allgather(
                np.stack([np.stack([res.get(k, {"val": np.zeros((2, nA, nA), complex)})["val"],
                                    res.get(k, {"cond": np.zeros((2, nA, nA), complex)})["cond"]])
                          for k in ks]))
            dense = np.sum(np.asarray(allres), axis=0)                 # (len(ks), 2 parts, 2, nA, nA)
            rows_b = []
            for ik, k in enumerate(ks):
                dv, dc = dense[ik, 0], dense[ik, 1]
                sv, sc = body["val"][:, k], body["cond"][:, k]
                g_out_d, g_in_d = (np.asarray(g) for g in C.rates(dc, dv))
                g_out_s, g_in_s = (np.asarray(g) for g in C.rates(sc, sv))
                mxa = lambda x: float(np.max(np.abs(x)))
                rows_b.append(dict(
                    k=k, val_rel=mxa(sv - dv) / mxa(dv), cond_rel=mxa(sc - dc) / mxa(dc),
                    g_out_rel=mxa(g_out_s - g_out_d) / max(mxa(g_out_d), 1e-300),
                    g_in_rel=mxa(g_in_s - g_in_d) / max(mxa(g_in_d), 1e-300),
                    g_out_diag_mev=list(np.real(np.diag(g_out_s)) * mev),
                    g_in_diag_mev=list(np.real(np.diag(g_in_s)) * mev),
                    g_out_diag_dense_mev=list(np.real(np.diag(g_out_d)) * mev),
                    g_in_diag_dense_mev=list(np.real(np.diag(g_in_d)) * mev)))
            out["B"] = dict(eta_ev=eta * ev, sum_bands=nsum, rows=rows_b,
                            n_invalid=int(np.sum(~good)), note="body only (head None), U = 1")
            del psi, Om, Bq, Dq, good, Rp, Rm

    def rates_at(plan, f_win, band_lo=b_lo, n=nA):
        eye = np.broadcast_to(np.eye(n, dtype=complex), (nk, n, n)).copy()
        w = dict(b_lo=band_lo, U=eye, e=e_full[:, band_lo:band_lo + n], f=f_win)
        o = PS.window_sigma_c(wfns, poles, w, plan, meta=meta, mesh_xy=mesh, branches=True)
        g_out, g_in, _ = D.window_branch_rates(o["branch_pair"])
        return g_out, g_in, o

    def rate_summary(g_out, g_in):
        herm = lambda g: 0.5 * (g + np.conj(np.swapaxes(g, -1, -2)))
        d_o, d_i = (np.real(np.einsum("kii->ki", g)) for g in (g_out, g_in))
        return dict(min_eig_out_mev=float(np.linalg.eigvalsh(herm(g_out)).min() * mev),
                    min_eig_in_mev=float(np.linalg.eigvalsh(herm(g_in)).min() * mev),
                    min_eig_total_mev=float(np.linalg.eigvalsh(herm(g_out + g_in)).min() * mev),
                    out_diag_per_band_mev=[[float(d_o[:, b].min() * mev), float(d_o[:, b].max() * mev)]
                                           for b in range(d_o.shape[1])],
                    in_diag_per_band_mev=[[float(d_i[:, b].min() * mev), float(d_i[:, b].max() * mev)]
                                          for b in range(d_i.shape[1])],
                    offdiag_max_mev=float(max(np.max(np.abs(g - np.einsum("kii->ki", g)[..., None]
                                                            * np.eye(g.shape[-1]))) for g in
                                              (g_out, g_in)) * mev))

    def eta_plan(eta_ev):
        import dataclasses
        return dict(splan, sigma_cfg=dataclasses.replace(splan["sigma_cfg"],
                                                         regularization_ev=float(eta_ev)))

    log_compiles = CompileLog()
    logging.getLogger("jax").addHandler(log_compiles)

    def td_run(tag, raw, system, mdl, *, wrap=None, spy=None, rho0=None):
        deck = parse_td_deck(raw, source=tag)
        outs = stages.open_outputs(deck, system)
        lines = []
        orig_attach, orig_place, orig_chunk = (stages._attach_collision, tp.place_state,
                                               tp._chunk_program)
        if wrap is not None:
            def attach(*x, **kw):
                built, s2, m2, c2 = orig_attach(*x, **kw)
                return dict(built, refresh=wrap(built["refresh"])), s2, m2, c2
            stages._attach_collision = attach
        if rho0 is not None:                  # the initial state only (the gate places its own)
            eq_pad = np.asarray(gather_to_host(system.ops.rho_eq))[None]

            def place(r, m):
                rr = np.asarray(r)
                if rr.shape == rho0.shape and np.array_equal(rr, eq_pad):
                    return orig_place(rho0, m)
                return orig_place(r, m)
            tp.place_state = place
        if spy is not None:
            def chunk_spy(*x, **kw):
                fn = orig_chunk(*x, **kw)

                def wrapped(*args):
                    if "text" not in spy:
                        spy["text"] = fn.lower(*args).compile().as_text()
                    return fn(*args)
                wrapped._cache_size = fn._cache_size
                return wrapped
            tp._chunk_program = chunk_spy
        jax.config.update("jax_log_compiles", True)
        n0 = len(log_compiles.names)
        t1 = time.perf_counter()
        try:
            stats = stages.propagate(deck, system, mdl, efield_function(deck.trials), outs, mesh,
                                     say=lines.append,
                                     collision=None if deck.values["lorrax.collision"] == "off"
                                     else src)
        finally:
            stages._attach_collision, tp.place_state, tp._chunk_program = (orig_attach, orig_place,
                                                                           orig_chunk)
            jax.config.update("jax_log_compiles", False)
        wall = time.perf_counter() - t1
        res = dict(tag=tag, wall_s=wall, compiles=log_compiles.names[n0:], lines=lines,
                   steps=int(stats["accepted"]), programs=int(stats["programs"]),
                   jit_cache=int(stats["jit_cache"]), refreshes=int(stats["refreshes"]),
                   rows=[{k: r[k] for k in ("t_fs", "wall_s", "s_per_fs", "new_program", "dsigma")}
                         for r in stats["rows"]], dt_au=float(stats["dt"]),
                   collision=stats.get("collision"))
        if rank0:
            with h5py.File(outs.paths[0], "r") as f:
                td_ = f["tdgw_data/time_domain"]
                res["rho_t"] = np.asarray(td_["rho_full/data"][()])
                res["pol"] = np.asarray(td_["polarization/data"][()])
                res["nc"] = np.asarray(td_["carrier_density/data"][()])
        res["times_fs"] = np.asarray(outs.mtsteps) / FS_AU
        return res

    def summarize_run(r, rho_eq):
        if "rho_t" not in r:
            return {k: v for k, v in r.items() if k not in ("lines",)}
        rho_t = r["rho_t"]
        tr = np.real(np.einsum("tkii->t", rho_t)) / rho_t.shape[1]
        pops = np.real(np.einsum("tkii->ti", rho_t)) / rho_t.shape[1]
        dev = np.max(np.abs(rho_t - rho_eq[None]), axis=(1, 2, 3))
        keep = {k: v for k, v in r.items() if k not in ("rho_t", "pol", "nc", "lines", "times_fs")}
        pol, nc = (np.asarray(r[k]).reshape(len(tr), -1) for k in ("pol", "nc"))
        keep.update(max_dev=float(dev.max()), dev_t=dev.tolist(), trace_t=(tr - tr[0]).tolist(),
                    pops_t=pops.tolist(), pol_t=np.real(pol).tolist(),
                    pol_imag_max=float(np.max(np.abs(np.imag(pol)))),
                    nc_t=np.real(nc).tolist(), nc_imag_max=float(np.max(np.abs(np.imag(nc)))),
                    times_fs=np.asarray(r["times_fs"]).tolist(), lines=r["lines"][-40:])
        return keep

    rho_eq_w = None
    if sysw is not None:
        rho_eq_w = np.asarray(gather_to_host(sysw.ops.rho_eq))[:sysw.nk]

    if "C" in sections and eq is not None:
        with section("C"):
            c = dict(eta_ev=out["source"]["eta_ev"], at_eta=rate_summary(eq["g_out"], eq["g_in"]),
                     c_eq_diag=coll["t0"]["c_eq_diag"], c_eq_offdiag=coll["t0"]["c_eq_offdiag"],
                     c_eq_diag_mev=coll["t0"]["c_eq_diag"] * 2 * mev,
                     c_eq_offdiag_mev=coll["t0"]["c_eq_offdiag"] * 2 * mev)
            fw = f_full[:, b_lo:b_hi]
            g2o, g2i, _ = rates_at(eta_plan(a.eta2), fw)
            c["eta2_ev"] = a.eta2
            c["at_eta2"] = rate_summary(g2o[bundle_of], g2i[bundle_of])
            out["C"] = c
            _save()
            runs = {}
            for form in ("subtracted", "raw"):
                raw = _td_deck(ref, (b_lo, b_hi), os.path.join(a.out, "td", f"eq_{form}"),
                               tf_fs=a.eq_fs, collision=form, refresh_fs=a.refresh_fs)
                runs[form] = summarize_run(td_run(f"eq_{form}", raw, sysw, model), rho_eq_w)
            out["C"]["eq_runs"] = {k: {kk: vv for kk, vv in v.items()
                                       if kk in ("max_dev", "dev_t", "trace_t", "refreshes",
                                                 "steps", "wall_s", "programs", "jit_cache",
                                                 "times_fs", "collision")}
                                   for k, v in runs.items()}

    if "DR" in sections and coll is not None:
        with section("DR"):
            dr = {}
            # the incumbent window call at rho_eq (the refresh's sweep without branches), warm
            eye = np.broadcast_to(np.eye(nA, dtype=complex), (nk, nA, nA)).copy()
            win = dict(b_lo=b_lo, U=eye, e=e_full[:, b_lo:b_hi], f=f_full[:, b_lo:b_hi])
            walls = []
            for rep in range(3):
                t1 = time.perf_counter()
                PS.window_sigma_c(wfns, poles, win, splan, meta=meta, mesh_xy=mesh)
                walls.append(time.perf_counter() - t1)
            walls_b = []
            for rep in range(3):
                t1 = time.perf_counter()
                PS.window_sigma_c(wfns, poles, win, splan, meta=meta, mesh_xy=mesh, branches=True,
                                  static_limit=True)
                walls_b.append(time.perf_counter() - t1)
            dr["incumbent_window_sigma_s"] = walls
            dr["branches_static_limit_window_sigma_s"] = walls_b
            cases = [("off", "off", {}), ("sub", "subtracted", {}), ("raw", "raw", {}),
                     ("sub_full", "subtracted", dict(hermitian="full")),
                     ("sub_norebase", "subtracted", {})]
            rr = {}
            keep = set(a.drive_cases.split(","))
            cases = [c_ for c_ in cases if c_[0] in keep]
            for tag, form, kw in cases:
                raw = _td_deck(ref, (b_lo, b_hi), os.path.join(a.out, "td", f"drive_{tag}"),
                               tf_fs=a.drive_fs, efield=PULSE, collision=form,
                               refresh_fs=a.refresh_fs, refresh_df=a.refresh_df, **kw)
                wrap = None
                if tag == "sub_norebase":
                    def wrap(hook):
                        def h(*x):
                            u = hook(*x)
                            return None if u is None else dict(u, delta=None)
                        return h
                rr[tag] = summarize_run(td_run(f"drive_{tag}", raw, sysw, model, wrap=wrap),
                                        rho_eq_w)
                _save()
            dr["runs"] = rr
            if rank0 and {"off", "sub", "sub_full", "sub_norebase"} <= set(rr):
                P = {k: np.asarray(v["pol_t"]) for k, v in rr.items() if "pol_t" in v}
                pm = max(float(np.max(np.abs(P["off"]))), 1e-300)
                dr["pol_rel_vs_off"] = {k: float(np.max(np.abs(P[k] - P["off"]))) / pm for k in P}
                dr["norebase_vs_rebase_pol_rel"] = float(np.max(np.abs(P["sub_norebase"]
                                                                        - P["sub"]))) / pm
                dr["full_vs_dynamic_pol_rel"] = float(np.max(np.abs(P["sub_full"] - P["sub"]))) / pm
                pops = {k: np.asarray(v["pops_t"]) for k, v in rr.items() if "pops_t" in v}
                dr["final_pops"] = {k: v[-1].tolist() for k, v in pops.items()}
            # the per-step cost of the held flow vs one dSigma apply (warm, k-sharded)
            flow, _ = tp.make_collision_maps(mesh)
            x0 = tp.place_state(np.zeros((1, sysw.nk_pad, nA, nA), complex), mesh)
            fj = jax.jit(lambda d, x: flow(d, x, 0.5))
            jax.block_until_ready(fj(coll["diss"], x0))
            t1 = time.perf_counter()
            for _ in range(200):
                y = fj(coll["diss"], x0)
            jax.block_until_ready(y)
            dr["flow_ms"] = (time.perf_counter() - t1) / 200 * 1e3
            r0 = tp.place_state(np.broadcast_to(np.asarray(gather_to_host(sysw.ops.rho_eq)),
                                                (1, sysw.nk_pad, nA, nA)).copy(), mesh)
            ap_ = jax.jit(lambda p, r: model.apply(p, r, 0.0))
            jax.block_until_ready(ap_(model.params, r0))
            t1 = time.perf_counter()
            for _ in range(50):
                y = ap_(model.params, r0)
            jax.block_until_ready(y)
            dr["dsigma_apply_ms"] = (time.perf_counter() - t1) / 50 * 1e3
            dr["memory"] = mem()
            out["DR"] = dr

    if "S" in sections:
        with section("S"):
            from gw import quadrature_log
            cfg = src["config"]
            mxa = lambda x: float(np.max(np.abs(x)))
            eye_n = lambda n: np.broadcast_to(np.eye(n, dtype=complex), (nk, n, n)).copy()

            def call(lo, n, f_win, plan, internal=None):
                w = dict(b_lo=lo, U=eye_n(n), e=e_full[:, lo:lo + n], f=f_win)
                t1 = time.perf_counter()
                o = PS.window_sigma_c(wfns, poles, w, plan, meta=meta, mesh_xy=mesh,
                                      branches=True, static_limit=True, internal=internal)
                o["wall_s"] = time.perf_counter() - t1
                return o

            def census(o):
                s = o["sweep"]
                return dict(distinct_tau=s["distinct_tau"], window_tau_pairs=s["window_tau_pairs"],
                            n_windows=s["n_windows"], windows=s["windows"],
                            rebuilds=s.get("rebuilds", 0), wall_s=o["wall_s"])

            def diffs(a, b):
                ga, gb = D.window_branch_rates(a["branch_pair"]), D.window_branch_rates(b["branch_pair"])
                sl = lambda o: o["static_limit"]["sx"] + np.einsum(
                    "ki,ij->kij", o["static_limit"]["head"], np.eye(o["static_limit"]["head"].shape[1]))
                scale = max(mxa(gb[0]), mxa(gb[1]))
                return dict(rate_out_mev=mxa(ga[0] - gb[0]) * mev, rate_in_mev=mxa(ga[1] - gb[1]) * mev,
                            rate_rel=max(mxa(ga[0] - gb[0]), mxa(ga[1] - gb[1])) / scale,
                            s_mev=mxa(ga[2] - gb[2]) * mev, pair_rel=mxa(a["pair"] - b["pair"]) / mxa(b["pair"]),
                            static_limit_mev=mxa(sl(a) - sl(b)) * mev, max_rate_mev=scale * mev)

            ss = {}
            wins = [("2v2c", (b_lo, b_hi))] + ([("wide", tuple(a.wide))] if a.wide else [])
            for tag, (lo, hi) in wins:
                n = hi - lo
                f_eq = f_full[:, lo:hi]
                f_hot = f_eq.copy()
                if tag == "wide":
                    f_hot[:, a.hot_from - lo] -= a.hot_amount
                    f_hot[:, a.hot_to - lo] += a.hot_amount
                else:
                    f_hot[:, 1] -= a.hot2v_amount
                    f_hot[:, 2] += a.hot2v_amount
                rec = {}
                inc = [call(lo, n, f_eq, splan) for _ in range(3)]          # the incumbent refresh
                rec["incumbent"] = dict(census(inc[-1]), walls_s=[o["wall_s"] for o in inc])
                own = [call(lo, n, f_eq, PS.window_sigma_plan(cfg, head=src["head"],
                                                              window_only=True)) for _ in range(2)]
                rec["window_only_full"] = dict(census(own[-1]), walls_s=[o["wall_s"] for o in own])
                pf = PS.window_sigma_plan(cfg, head=None, session={}, window_only=True)
                pa = PS.window_sigma_plan(cfg, head=src["head"], session={}, window_only=True)
                fr0 = call(lo, n, f_eq, pf, "frozen")
                fro = call(lo, n, f_eq, pf, "frozen")
                rec["frozen"] = dict(census(fro), plan_call=census(fr0))
                ac0 = call(lo, n, np.full_like(f_eq, 0.5), pa, "active")
                act = [call(lo, n, f_eq, pa, "active") for _ in range(3)]
                rec["active"] = dict(census(act[-1]), walls_s=[o["wall_s"] for o in act],
                                     plan_call=census(ac0))
                rec["split_vs_incumbent_eq"] = diffs(PS.add_window_parts(fro, act[-1]), inc[-1])
                rec["split_vs_window_only_eq"] = diffs(PS.add_window_parts(fro, act[-1]), own[-1])
                rec["window_only_vs_incumbent_eq"] = diffs(own[-1], inc[-1])
                hot_inc = call(lo, n, f_hot, splan)
                hot_act = call(lo, n, f_hot, pa, "active")
                rec["split_vs_incumbent_hot"] = diffs(PS.add_window_parts(fro, hot_act), hot_inc)
                rec["hot_active"] = census(hot_act)
                rec["hot"] = (dict(from_band=a.hot_from, to_band=a.hot_to, amount=a.hot_amount)
                              if tag == "wide" else dict(from_band=lo + 1, to_band=lo + 2,
                                                         amount=a.hot2v_amount))
                ss[tag] = rec
                out["S"] = ss
                _save()

    if "SD" in sections and coll is not None:
        with section("SD"):
            raw = _td_deck(ref, (b_lo, b_hi), os.path.join(a.out, "td", "drive_sub_active"),
                           tf_fs=a.drive_fs, efield=PULSE, collision="subtracted",
                           refresh_fs=a.refresh_fs, refresh_df=a.refresh_df, sweep="active")
            orig = D.window_collision
            D.window_collision = lambda *x, **kw: orig(*x, **dict(kw, check_full=True))
            try:
                rr = summarize_run(td_run("drive_sub_active", raw, sysw, model), rho_eq_w)
            finally:
                D.window_collision = orig
            sd = dict(run=rr)
            hist = (rr.get("collision") or {}).get("history", [])
            fv = [h["full_vs_split"] for h in hist if "full_vs_split" in h]
            if fv:
                sd["full_vs_split_max"] = {k: max(x[k] for x in fv) for k in fv[0]
                                           if isinstance(fv[0][k], (int, float))}
            prev = out.get("DR", {}).get("runs", {}).get("sub")
            if rank0 and prev is not None and "pol_t" in prev and "pol_t" in rr:
                P0, P1 = np.asarray(prev["pol_t"]), np.asarray(rr["pol_t"])
                m = min(len(P0), len(P1))
                sd["pol_rel_vs_full_refresh"] = float(np.max(np.abs(P1[:m] - P0[:m]))) / max(
                    float(np.max(np.abs(P0[:m]))), 1e-300)
                sd["pops_max_diff_vs_full_refresh"] = float(np.max(np.abs(
                    np.asarray(rr["pops_t"])[:m] - np.asarray(prev["pops_t"])[:m])))
            out["SD"] = sd

    def _pulse(amp):
        """The deck's pulse with peak field ``amp`` V/m (PULSE's envelope and carrier)."""
        return f"{float(amp)!r} * " + PULSE.split(" * ", 1)[1]

    if "CS" in sections or "C0" in sections:
        with section("CS"):
            cfg = src["config"]
            req_ = src["request"]
            freqs = None if req_ is None else (np.asarray(req_.frequencies_ry)
                                               + float(src["efermi_ry"]))
            W0c = poles.Wc0_q
            mxa = lambda x: float(np.max(np.abs(x)))
            eye_n = lambda n: np.broadcast_to(np.eye(n, dtype=complex), (nk, n, n)).copy()
            herm = lambda m: 0.5 * (m + np.conj(np.swapaxes(m, -1, -2)))
            dg = lambda m: np.real(np.einsum("...ii->...i", m))

            def callp(lo, n, f_win, plan, at, internal=None, keep=None, static=True):
                w = dict(b_lo=lo, U=eye_n(n), e=e_full[:, lo:lo + n], f=f_win)
                t1 = time.perf_counter()
                o = PS.window_sigma_c(wfns, at, w, plan, meta=meta, mesh_xy=mesh, branches=True,
                                      static_limit=static, internal=internal, keep_frozen=keep,
                                      static_limit_wc0=W0c if static else None)
                o["wall_s"] = time.perf_counter() - t1
                return o

            def census(o):
                s_ = o["sweep"]
                return dict(distinct_tau=s_["distinct_tau"], window_tau_pairs=s_["window_tau_pairs"],
                            n_windows=s_["n_windows"], windows=s_["windows"],
                            rebuilds=s_.get("rebuilds", 0), wall_s=o["wall_s"])

            def diffs(x, y):
                gx, gy = D.window_branch_rates(x["branch_pair"]), D.window_branch_rates(y["branch_pair"])
                scale = max(mxa(gy[0]), mxa(gy[1]))
                return dict(rate_out_mev=mxa(gx[0] - gy[0]) * mev, rate_in_mev=mxa(gx[1] - gy[1]) * mev,
                            rate_rel=max(mxa(gx[0] - gy[0]), mxa(gx[1] - gy[1])) / scale,
                            s_mev=mxa(gx[2] - gy[2]) * mev,
                            herm_mev=mxa(herm(gx[2]) - herm(gy[2])) * mev,
                            pair_rel=mxa(x["pair"] - y["pair"]) / mxa(y["pair"]), max_rate_mev=scale * mev)

            def dw_effect(x, y):
                """W(t) (x) vs W0 (y) at the same populations: band-edge rates vs Re Sigma, meV."""
                gx, gy = D.window_branch_rates(x["branch_pair"]), D.window_branch_rates(y["branch_pair"])
                return dict(
                    d_rate_out_diag_mev=mxa(dg(gx[0]) - dg(gy[0])) * mev,
                    d_rate_in_diag_mev=mxa(dg(gx[1]) - dg(gy[1])) * mev,
                    rate_out_diag_max_mev=mxa(dg(gy[0])) * mev, rate_in_diag_max_mev=mxa(dg(gy[1])) * mev,
                    d_rate_out_mev=mxa(gx[0] - gy[0]) * mev, d_rate_in_mev=mxa(gx[1] - gy[1]) * mev,
                    d_re_sigma_mev=mxa(herm(gx[2]) - herm(gy[2])) * mev,
                    d_re_sigma_diag_mev=mxa(dg(herm(gx[2])) - dg(herm(gy[2]))) * mev,
                    re_sigma_diag_max_mev=mxa(dg(herm(gy[2]))) * mev)

            cs, scrs = {}, {}
            wins = [("2v2c", (b_lo, b_hi))] + ([("wide", tuple(a.wide))] if a.wide else [])
            widths = [int(v) for v in a.cs_trunc.split(",") if v]
            for tag, (lo, hi) in wins:
                n = hi - lo
                f_eq = f_full[:, lo:hi]
                f_hot = f_eq.copy()
                if tag == "wide":
                    f_hot[:, a.hot_from - lo] -= a.hot_amount
                    f_hot[:, a.hot_to - lo] += a.hot_amount
                else:
                    f_hot[:, 1] -= a.hot2v_amount
                    f_hot[:, 2] += a.hot2v_amount
                rec = {}
                t1 = time.perf_counter()
                scr = D.collision_screening(src, mesh=mesh, b_lo=lo, n_active=n,
                                            gamma_ry=0.001 / ev)
                rec["setup_s"] = time.perf_counter() - t1
                ru = scr["rules"]
                rec["rules"] = dict(static_nodes=len(ru["static"]["tau"]),
                                    probe_nodes=len(ru["probe"]["tau"]),
                                    probe_odd=bool(np.any(ru["probe"]["beta"] != 0.0)),
                                    x_min_ry=ru["static"]["x_min"], x_max_ry=ru["static"]["x_max"],
                                    omega_p_ry=ru["probe"]["omega_p"], nbf=scr["nbf"])
                _, _, r1 = D.screen_populations(scr, f_eq)
                pe, _, re_ = D.screen_populations(scr, f_eq)
                same = lambda u, v: (u is None and v is None) or (
                    u is not None and v is not None and bool(jax.device_get(jnp.array_equal(u, v))))
                rec["eq"] = dict(re_, first_call=dict(dchi_s=r1["dchi_s"], fit_s=r1["fit_s"]),
                                 poles_bitwise=all(same(getattr(pe, k_), getattr(poles, k_))
                                                   for k_ in ("Omega_q", "B_q", "B_odd_q",
                                                              "valid_mask_q", "Wc0_q")))
                del pe
                ph, _, rh = D.screen_populations(scr, f_hot)
                rec["hot"] = dict(rh, state=(dict(from_band=a.hot_from, to_band=a.hot_to,
                                                  amount=a.hot_amount) if tag == "wide" else
                                             dict(from_band=lo + 1, to_band=lo + 2,
                                                  amount=a.hot2v_amount)))
                full_t = callp(lo, n, f_hot, splan, ph)
                full_0 = callp(lo, n, f_hot, splan, poles)
                rec["full_W_t"], rec["full_W0"] = census(full_t), census(full_0)
                rec["dW_effect_hot"] = dw_effect(full_t, full_0)
                pf = PS.window_sigma_plan(cfg, head=None, session={}, window_only=True)
                pa = PS.window_sigma_plan(cfg, head=src["head"], session={}, window_only=True)
                callp(lo, n, f_eq, pf, poles, "frozen")
                fro0 = callp(lo, n, f_eq, pf, poles, "frozen")
                callp(lo, n, np.full_like(f_eq, 0.5), pa, poles, "active")
                fro_t = callp(lo, n, f_hot, pf, ph, "frozen", static=False)
                fro_t["static_limit"] = fro0["static_limit"]
                act_t = callp(lo, n, f_hot, pa, ph, "active")
                split_t = PS.add_window_parts(fro_t, act_t)
                rec["frozen_W0_held"], rec["frozen_W_t"] = census(fro0), census(fro_t)
                rec["active_W_t"] = census(act_t)
                rec["split_vs_full_W_t"] = diffs(split_t, full_t)
                pg = PS.window_sigma_plan(cfg, head=None, frequencies_abs_ry=freqs)
                fro_g = callp(lo, n, f_hot, pg, ph, "frozen", static=False)
                fro_g["static_limit"] = fro0["static_limit"]
                rec["frozen_gw_list_W_t"] = census(fro_g)
                rec["frozen_gw_list_vs_window"] = diffs(PS.add_window_parts(fro_g, act_t), split_t)
                nfull = f_full[:, wfns.slices.full].shape[1] if hasattr(wfns.slices, "full") else f_full.shape[1]
                for w_ in widths:
                    keep = np.zeros(nfull, bool)
                    keep[max(0, lo - w_):lo] = True
                    keep[hi:min(nfull, hi + w_)] = True
                    pk = PS.window_sigma_plan(cfg, head=None, session={}, window_only=True)
                    callp(lo, n, f_eq, pk, poles, "frozen", keep=keep, static=False)
                    fk0 = callp(lo, n, f_eq, pk, poles, "frozen", keep=keep, static=False)
                    fkt = callp(lo, n, f_hot, pk, ph, "frozen", keep=keep, static=False)
                    tr_ = D._shift_window_part(fro0, fkt, fk0)
                    tr_["static_limit"] = fro0["static_limit"]
                    rec[f"trunc_{w_}"] = dict(diffs(PS.add_window_parts(tr_, act_t), split_t),
                                              kept_bands=int(keep.sum()), census=census(fkt))
                cs[tag] = rec
                scrs[tag] = scr
                del ph, full_t, full_0, fro_t, act_t, split_t, fro_g
                out["CS"] = cs
                _save()
            if "C0" in sections:
                scr = scrs["2v2c"]
                c0 = {}
                t1 = time.perf_counter()
                chis = []

                def _split_fn(r_):
                    return jax.jit(lambda A_: D.chi0_split({**scr["dyn"], **A_}, scr["kern"],
                                                           rule=r_))

                for r_ in (None, scr["rules"]["probe"]):
                    sp = _split_fn(r_)(scr["arrays"])
                    chis.append(sp["ff"] + sp["fa"] + sp["aa"])
                    del sp
                jax.block_until_ready(chis)
                c0["chi0_split_s"] = time.perf_counter() - t1
                Vf = scr["plan"]["V_q"]
                _, Wr = PS.screening_update((Vf, Vf), chis, scr["plan"])
                c0["W_rel_vs_run"] = [float(jax.device_get(jnp.max(jnp.abs(Wr[i] - scr["W0"][i]))
                                                           / jnp.max(jnp.abs(scr["W0"][i]))))
                                      for i in range(2)]
                c0["Wc_rel_vs_run"] = [float(jax.device_get(
                    jnp.max(jnp.abs(Wr[i] - scr["W0"][i])) / jnp.max(jnp.abs(scr["W0"][i] - Vf))))
                    for i in range(2)]
                del Wr, chis
                out["C0"] = c0
            del scrs

    if "CD" in sections and coll is not None:
        with section("CD"):
            if a.cd_window == "2v2c":
                lo, hi, sys_cd, model_cd, rho_eq_cd, deg = b_lo, b_hi, sysw, model, rho_eq_w, ""
            else:
                lo, hi = (int(b) for b in a.wide)
                deg = "off"
                rawp = _td_deck(ref, (lo, hi), os.path.join(a.out, "td", "cd_wide_probe"), tf_fs=0.1,
                                degeneracy=deg)
                deckp = parse_td_deck(rawp, source="collision_516_cd_wide")
                sys_cd = stages.load_system(deckp, mesh)
                model_cd = stages.build_self_energy(deckp, sys_cd, mesh)
                rho_eq_cd = np.asarray(gather_to_host(sys_cd.ops.rho_eq))[:sys_cd.nk]
            fb = tuple(int(v) for v in a.cd_frozen_bands.split(",")) if a.cd_frozen_bands else None
            cd = out.setdefault("CD", {})
            for amp in (float(v) for v in a.cd_pulses.split(",")):
                for cad in (float(v) for v in a.cd_cadences.split(",")):
                    for scr_ in a.cd_screenings.split(","):
                        tag = f"{a.cd_window}_E{amp:g}_r{cad:g}_{scr_}{a.cd_tag}"
                        raw = _td_deck(ref, (lo, hi), os.path.join(a.out, "td", tag),
                                       tf_fs=a.drive_fs, dtm_as=a.cd_dtm_as,
                                       efield=_pulse(amp), collision="subtracted",
                                       refresh_fs=cad, refresh_df=a.refresh_df, sweep="active",
                                       screening=scr_, degeneracy=deg,
                                       frozen_bands=fb if scr_ == "refreshed" else None,
                                       gamma_ev=a.cd_gamma_ev)
                        orig = D.window_collision
                        extra = {}
                        if a.cd_check_trunc and scr_ == "refreshed" and fb is not None:
                            extra["check_truncation"] = True
                        if a.cd_rebase and scr_ == "refreshed":
                            extra["rebase"] = True
                        if extra:
                            D.window_collision = lambda *x, **kw: orig(*x, **dict(kw, **extra))
                        try:
                            r = summarize_run(td_run(tag, raw, sys_cd, model_cd), rho_eq_cd)
                        finally:
                            D.window_collision = orig
                        cd[tag] = dict(r, window=[lo, hi], amp=amp, cadence_fs=cad, screening=scr_,
                                       frozen_bands=fb)
                        _save()

    if "H" in sections and sysw is not None:
        with section("H"):
            hh = {}
            for form in ("off", "subtracted"):
                spy = {}
                raw = _td_deck(ref, (b_lo, b_hi), os.path.join(a.out, "td", f"hlo_{form}"),
                               tf_fs=0.25, efield=PULSE, collision=form)
                td_run(f"hlo_{form}", raw, sysw, model, spy=spy)
                text = spy.get("text", "")
                hh[form] = {op: len(re.findall(rf"\b{op}(?:-start)?\(", text)) for op in
                            ("all-reduce", "all-gather", "collective-permute", "all-to-all",
                             "reduce-scatter")}
            hh["same"] = hh["off"] == hh["subtracted"]
            out["H"] = hh

    if "W" in sections and a.wide:
        with section("W"):
            w_lo, w_hi = (int(b) for b in a.wide)
            nW = w_hi - w_lo
            fw = f_full[:, w_lo:w_hi]
            ww = dict(bands=[w_lo, w_hi])
            for tag, plan in (("eta", splan), ("eta2", eta_plan(a.eta2))):
                go, gi, _ = rates_at(plan, fw, band_lo=w_lo, n=nW)
                ww[tag] = rate_summary(go, gi)
            # the hot-carrier state: hot_amount moved from hot_from to hot_to at every k
            fh = fw.copy()
            i_from, i_to = a.hot_from - w_lo, a.hot_to - w_lo
            fh[:, i_from] -= a.hot_amount
            fh[:, i_to] += a.hot_amount
            go, gi, _ = rates_at(splan, fh, band_lo=w_lo, n=nW)
            ww["hot_rates"] = rate_summary(go, gi)
            ww["hot"] = dict(from_band=a.hot_from, to_band=a.hot_to, amount=a.hot_amount)
            _save()
            raw = _td_deck(ref, (w_lo, w_hi), os.path.join(a.out, "td", "wide_probe"), tf_fs=0.1,
                           degeneracy="off")
            deckw = parse_td_deck(raw, source="collision_516_wide")
            sys_wide = stages.load_system(deckw, mesh)
            model_w = stages.build_self_energy(deckw, sys_wide, mesh)
            rho_eq_wide = np.asarray(gather_to_host(sys_wide.ops.rho_eq))[:sys_wide.nk]
            rho_hot = np.zeros((1, sys_wide.nk_pad, nW, nW), complex)
            rho_hot[0, :sys_wide.nk] = rho_eq_wide
            for kk in range(sys_wide.nk):
                rho_hot[0, kk, i_from, i_from] -= a.hot_amount
                rho_hot[0, kk, i_to, i_to] += a.hot_amount
            runs = {}
            for form in ("off", "subtracted"):
                raw = _td_deck(ref, (w_lo, w_hi), os.path.join(a.out, "td", f"hot_{form}"),
                               tf_fs=a.hot_fs, collision=form, refresh_fs=a.refresh_fs,
                               refresh_df=a.refresh_df, degeneracy="off")
                runs[form] = summarize_run(td_run(f"hot_{form}", raw, sys_wide, model_w,
                                                  rho0=rho_hot), rho_eq_wide)
                _save()
            ww["runs"] = runs
            out["W"] = ww
    out["compiles_total"] = len(log_compiles.names)
    out["memory_end"] = mem()
    out["wall_s"] = sum(out.get("section_wall_s", {}).values())
    _save()
    if rank0:
        print("RESULT " + json.dumps(dict(errors=sorted(out.get("errors", {})),
                                          sections=out.get("section_wall_s")), default=float),
              flush=True)
    runtime.finalize_process(0 if not out.get("errors") else 1)


if __name__ == "__main__":
    main()
