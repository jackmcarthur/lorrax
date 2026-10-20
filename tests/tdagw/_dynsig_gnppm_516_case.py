"""The GN-PPM groundwork at t = 0 on a real GN-PPM GW run (CrI3 516, 6x6 k, nosym, 2v2c window).

    lx run -N 1 -G 4 -n 4 -- python3 -u tests/tdagw/_dynsig_gnppm_516_case.py \\
        --ref <516>/lorrax/07_gw_gnppm_bundle --bands 128 132 --out DIR

One JSON line (rank 0), ``RESULT {...}``, and DIR/dynsig_gnppm_516.json.  The 07
arm is read-only.  A GN-PPM run persists W(0) (``W0_qmunu``) but neither
W(i w_p) nor its pole model, so this process first runs the GW driver itself
(``gw.gw_jax.main``, every stage) on a copy of the 07 deck in DIR with
``write_restart_tensors = false`` (I/O only), observing the GN-PPM seam:
``gw.ppm_pipeline.fit_pole_model`` (the run's W(0), W(i w_p), its fit plan and
poles) and ``compute_ppm_sigma_pipeline`` (its Sigma_c body and head).  The
observers call through and snapshot what they read.

D0  this process's driver run (the branch) vs the 07 arm (run on the BEFORE
    tree 3c58c3fdb): every sigma_mnk.h5 dataset bitwise, sigma_diag.dat lines
    (generator stamp excluded), and W(0) vs the arm's W0_qmunu (canonical)
W1  tdagw.dynsig.chi0_split at the run's static and probe rules (re-solved from
    the run's interval, w_p and target) -> gw.ppm_sigma.screening_update with V
    in W0's place: W(0), W(i w_p) vs the run's, relative per q (q = 0 included);
    W(0) also vs the 07 file's W0_qmunu
W2  the poles of W1 vs the run's fit (Omega, B, D; valid masks)
S1  items 4 + 5 chained at rho_eq: gw.ppm_sigma.window_sigma_c on W1's poles (the
    run's head, its on-shell frequency list) + tdagw.dynsig.exchange_sigma
    (heads whead/vhead) vs the run: the window block of sigma_c_kij_ev (pair,
    off-diagonal included), Sigma_x and their sum (sigma_total_kij_ev minus
    hartree_kij_ev: the writer's total carries V_H), from sigma_mnk.h5 (float64)
    and from this process's in-memory Sigma_c; eV with common.units.RYD_TO_EV
    (the writer's constant)
S1d the diagonal vs sigma_diag.dat (sigX, Re sigC; eV printed to 1e-6)
S1w window_sigma_c on the run's own poles (separates the sweep from chi0)
RW  chi0^fa dropped (W, and Sigma through its poles); RP pole order swapped
    (D -> -D); RF frozen bands omitted from G: each must fail by orders
T   warm walls of screening_update and window_sigma_c
"""
import argparse
import dataclasses
import json
import os
import re
import shutil
import time

import runtime  # noqa: F401  (before numpy: the runtime tunes BLAS spin at import)
import numpy as np


def _read_sigma_diag(path, b_lo, b_hi):
    """sigma_diag.dat -> kcrys (nk, 3), sigX, Re sigC, Im sigC, Eo (nk, nb) in eV, file k order."""
    ks, rows, cur = [], [], None
    pat = re.compile(r"^n=(\d+)\s+sigX=\s*(\S+)\s+sigC=\s*(\S+)\+\s*(\S+)i\s+sigXC=.*Eo=\s*(\S+)")
    with open(path) as f:
        for line in f:
            if line.startswith("# kcrys"):
                ks.append([float(v) for v in line.split()[2:5]])
                cur = np.full((b_hi - b_lo, 4), np.nan)
                rows.append(cur)
                continue
            m = pat.match(line.strip())
            if m and cur is not None:
                n = int(m.group(1))
                if b_lo <= n < b_hi:
                    cur[n - b_lo] = [float(m.group(2)), float(m.group(3)), float(m.group(4)),
                                     float(m.group(5))]
    a = np.asarray(rows)
    return np.asarray(ks), a[..., 0], a[..., 1], a[..., 2], a[..., 3]


def _copy_deck(ref, out):
    """The 07 deck in ``out`` with write_restart_tensors = false (the 07 arm holds the restart)."""
    os.makedirs(out, exist_ok=True)
    lines = open(os.path.join(ref, "oneshot.in")).read().splitlines()
    lines = [("write_restart_tensors = false" if l.strip().startswith("write_restart_tensors")
              else l) for l in lines]
    open(os.path.join(out, "oneshot.in"), "w").write("\n".join(lines) + "\n")
    dip = os.path.join(out, "dipole.h5")
    if not os.path.lexists(dip):
        os.symlink(os.path.realpath(os.path.join(ref, "dipole.h5")), dip)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="the 07 arm (read-only)")
    ap.add_argument("--bands", type=int, nargs=2, default=(128, 132))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    stack = runtime.initialize_communicator_stack()      # before any JAX call (multi-process)
    import jax
    import jax.numpy as jnp
    import h5py
    from jax.experimental import multihost_utils
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.units import RYD_TO_EV
    mesh = stack.mesh
    rank0 = jax.process_index() == 0
    gather = lambda x: np.asarray(multihost_utils.process_allgather(x, tiled=True))
    out = dict(mesh=list(mesh.devices.shape), ref=os.path.abspath(a.ref), bands=list(a.bands))
    b_lo, b_hi = (int(b) for b in a.bands)
    nA = b_hi - b_lo
    t_start = time.perf_counter()

    def _save():
        if rank0:
            os.makedirs(a.out, exist_ok=True)
            with open(os.path.join(a.out, "dynsig_gnppm_516.json"), "w") as f:
                json.dump(out, f, indent=1, default=float)

    def _failed(name, t0):
        import traceback
        out.setdefault("errors", {})[name] = traceback.format_exc()[-2500:]
        out.setdefault("section_wall_s", {})[name] = time.perf_counter() - t0
        if rank0:
            print(f"[gnppm_516] section {name} FAILED:\n{out['errors'][name]}", flush=True)
        _save()

    def _done(name, t0):
        out.setdefault("section_wall_s", {})[name] = time.perf_counter() - t0
        _save()

    # ---- the GW driver in this process, observed at the GN-PPM seam ----
    import gw.gw_jax as GJ
    import gw.ppm_pipeline as PP
    cap = {}
    _fit, _pipe = PP.fit_pole_model, PP.compute_ppm_sigma_pipeline

    def fit_spy(W0, Wp, plan, *, print_fn=None):
        cap.update(W0=jnp.copy(W0), Wp=jnp.copy(Wp), plan=dict(plan, V_q=jnp.copy(plan["V_q"])))
        res = _fit(W0, Wp, plan, print_fn=print_fn)
        cap.update(poles=dataclasses.replace(
            res, B_q=jnp.copy(res.B_q), Omega_q=jnp.copy(res.Omega_q),
            valid_mask_q=jnp.copy(res.valid_mask_q), Wc0_q=jnp.copy(res.Wc0_q),
            B_odd_q=None if res.B_odd_q is None else jnp.copy(res.B_odd_q)))
        return res

    def pipe_spy(**kw):
        res = _pipe(**kw)
        body = res.sigma_c_body_omega
        cap.update(kw=kw, efermi_ry=res.efermi_ry, head=res.head_sigma_diag_w_kn_ry,
                   request=body.request,
                   body_pair=np.asarray(gather(body.pair))[:, :, b_lo:b_hi, b_lo:b_hi],
                   body_head_pair=np.asarray(gather(body.with_head(
                       res.head_sigma_diag_w_kn_ry).pair))[:, :, b_lo:b_hi, b_lo:b_hi],
                   wfns=jax.tree.map(lambda y: jnp.copy(y) if isinstance(y, jax.Array) else y,
                                     kw["wfns"]))
        return res

    PP.fit_pole_model, PP.compute_ppm_sigma_pipeline = fit_spy, pipe_spy
    _t = time.perf_counter()
    run_dir = os.path.join(os.path.abspath(a.out), "driver")
    try:
        if rank0:
            _copy_deck(a.ref, run_dir)
        multihost_utils.sync_global_devices("deck copied")
        cwd = os.getcwd()
        os.chdir(run_dir)
        try:
            rc = GJ.main(["-i", "oneshot.in"])
        finally:
            os.chdir(cwd)
            PP.fit_pole_model, PP.compute_ppm_sigma_pipeline = _fit, _pipe
        out["driver"] = dict(rc=rc, dir=run_dir, captured=sorted(cap))
        _done("driver", _t)
    except Exception:
        _failed("driver", _t)
        _stop(out, _save, 1)
        return

    kw = cap["kw"]
    wfns, meta, config, quad = cap["wfns"], kw["meta"], kw["config"], kw["quad"]
    plan = cap["plan"]
    nk = int(meta.nk_tot)
    restart = os.path.join(a.ref, "tmp", f"isdf_tensors_{int(meta.n_rmu)}.h5")
    basis = getattr(meta, "mu_basis", None)
    W0_file = None
    n_log = int(meta.n_rmu)

    def rel_logical(X, R):
        """max |X - R| / max |R| on the logical [0, n_rmu) block (pads may differ)."""
        X, R = X[..., :n_log, :n_log], R[..., :n_log, :n_log]
        return float(np.max(np.abs(X - R)) / np.max(np.abs(R)))

    # ---- D0: the driver path before/after ----
    _t = time.perf_counter()
    try:
        d0 = {}
        with h5py.File(os.path.join(a.ref, "sigma_mnk.h5"), "r") as fa, \
                h5py.File(os.path.join(run_dir, "sigma_mnk.h5"), "r") as fb:
            names = sorted(k for k in fa.keys() if isinstance(fa[k], h5py.Dataset))
            for k in names:
                if k not in fb:
                    d0[k] = "missing"
                    continue
                x, y = np.asarray(fa[k][()]), np.asarray(fb[k][()])
                if x.dtype.kind in "fc" and x.shape == y.shape:
                    d0[k] = dict(bitwise=bool(np.array_equal(x, y)),
                                 max_abs=float(np.max(np.abs(x - y))) if x.size else 0.0)
                else:
                    d0[k] = dict(bitwise=bool(np.array_equal(x, y)))
        strip = lambda p: [l for l in open(p) if not l.startswith("# Generated")]
        la, lb = (strip(os.path.join(d, "sigma_diag.dat")) for d in (a.ref, run_dir))
        out["D0"] = dict(sigma_mnk=d0, all_bitwise=all(isinstance(v, dict) and v["bitwise"]
                                                       for v in d0.values()),
                         sigma_diag_identical=bool(la == lb), sigma_diag_lines=len(la))
        from file_io.restart_bundle import read_interaction
        W0_file = gather(read_interaction(restart, "screened", mesh))
        W0_run = cap["W0"]
        W0_run_c = gather(W0_run if basis is None else basis.unpack_operator(W0_run))
        out["D0"]["W0_run_vs_file"] = dict(
            shapes=[list(W0_run_c.shape), list(W0_file.shape)],
            bitwise=bool(np.array_equal(W0_run_c[..., :n_log, :n_log], W0_file[..., :n_log, :n_log])),
            rel_max=rel_logical(W0_run_c, W0_file))
        _done("D0", _t)
    except Exception:
        _failed("D0", _t)

    # ---- the TD bundle on the run's own carrier (packed centroid order) ----
    from tdagw import dynsig as D
    from gw import ppm_sigma as PS
    kplan = wfns.green_parent.plan
    kf = np.asarray(kplan.k_parent_frac, np.float64)
    from tdagw.model import flat_k_rows
    rows = np.asarray(flat_k_rows(kf, tuple(int(n) for n in meta.kgrid)))
    out["orders"] = dict(parent_rows_c_order=bool(np.array_equal(rows, np.arange(nk))),
                         irr_identity=bool(np.array_equal(np.asarray(kplan.irr_idx), np.arange(nk))),
                         mu_packed=basis is not None, n_rmu=int(meta.n_rmu),
                         mu_carrier=int(cap["W0"].shape[-1]),
                         mu_solve_extent=int(getattr(meta, "mu_solve_extent", meta.n_rmu)))
    if not out["orders"]["parent_rows_c_order"] or not out["orders"]["irr_identity"]:
        out.setdefault("errors", {})["orders"] = "the parent k rows are not the flat C order"
        _stop(out, _save, 1)
        return
    e_full = np.asarray(jax.device_get(wfns.enk), np.float64)
    f_full = np.asarray(jax.device_get(wfns.occ), np.float64)
    nbf = int(wfns.slices.nb_full_logical)
    e_full, f_full = e_full[:, :nbf], f_full[:, :nbf]
    psi_full = jax.jit(lambda p: p, out_shardings=NamedSharding(mesh, P(None, None, None, "x")))(
        wfns.green_parent.psi_nmu)
    V_q = plan["V_q"]
    td = dict(kgrid=tuple(int(n) for n in meta.kgrid), band_lo=b_lo, n_bands=nA,
              n_rmu=int(meta.n_rmu))
    _t = time.perf_counter()
    try:
        mm = config.minimax_config
        rule = D.static_rule(e_full, f_full, target_error=float(mm.target_error),
                             max_nodes=int(mm.max_nodes))
        prule = D.probe_rule(rule, float(config.ppm.omega_p), target_error=float(mm.target_error))
        qtau = getattr(quad, "tau", None)
        out["rules"] = dict(
            static_nodes=int(len(rule["tau"])), probe_nodes=int(len(prule["tau"])),
            static_vs_run=(float(np.max(np.abs(np.asarray(qtau) - rule["tau"])))
                           if qtau is not None and len(qtau) == len(rule["tau"])
                           else f"run nodes {None if qtau is None else len(qtau)}"),
            x_min=[getattr(quad, "x_min", None), rule["x_min"]],
            x_max=[getattr(quad, "x_max", None), rule["x_max"]],
            probe_max_error=prule["max_error"], omega_p=float(config.ppm.omega_p))
        dyn = D.assemble_dynsig_bundle(td, psi_full, e_full, f_full, cap["W0"], mesh_xy=mesh,
                                       rule=rule, V_q=V_q)
        kern = D.make_dynsig_kernels(mesh, td["kgrid"])
        A_, S_ = D.split_bundle(dyn)
        _done("bundle", _t)
    except Exception:
        _failed("bundle", _t)
        _stop(out, _save, 1)
        return

    # ---- W1 / W2 / RW: the split at both supports -> screening_update ----
    _t = time.perf_counter()
    split = {}
    try:
        for tag, r in (("0", None), ("p", prule)):
            sp = jax.jit(lambda A__: D.chi0_split({**S_, **A__}, kern, rule=r))(A_)
            split[tag] = sp
        cs = jax.jit(lambda s: s["ff"] + s["fa"] + s["aa"])
        cb = jax.jit(lambda s: s["ff"] + s["aa"])
        nrm = lambda x: float(jnp.max(jnp.abs(x)))
        shares = {t: dict(fa=nrm(split[t]["fa"]) / nrm(cs(split[t])),
                          aa=nrm(split[t]["aa"]) / nrm(cs(split[t]))) for t in split}
        W0h, Wph = gather(cap["W0"]), gather(cap["Wp"])
        relq = lambda W, R: [float(np.max(np.abs(W[q] - R[q])) / np.max(np.abs(R[q])))
                             for q in range(nk)]
        t0 = time.perf_counter()
        poles_td, (W0_td, Wp_td) = PS.screening_update(
            (V_q, V_q), (cs(split["0"]), cs(split["p"])), plan)
        jax.block_until_ready(poles_td.B_q)
        t_first = time.perf_counter() - t0
        W0t, Wpt = gather(W0_td), gather(Wp_td)
        r0, rp = relq(W0t, W0h), relq(Wpt, Wph)
        out["W1"] = dict(rel_W0_max=max(r0), rel_W0_q0=r0[0], rel_Wp_max=max(rp), rel_Wp_q0=rp[0],
                         rel_W0_per_q=r0, rel_Wp_per_q=rp, chi0_shares=shares,
                         Wp_antihermitian=float(np.max(np.abs(Wph - np.conj(np.swapaxes(Wph, -1, -2))))
                                                / np.max(np.abs(Wph))))
        if W0_file is not None:
            try:
                W0t_c = gather(W0_td if basis is None else basis.unpack_operator(W0_td))
                out["W1"]["rel_W0_vs_file"] = rel_logical(W0t_c, W0_file)
            except Exception as exc:                       # reported, not fatal to W1
                out["W1"]["rel_W0_vs_file"] = f"error: {exc!r}"[:400]
        pr = cap["poles"]
        g = lambda x: gather(x)
        live = g(pr.valid_mask_q)
        rel_live = lambda x, y: float(np.max(np.abs(np.where(live, x - y, 0))) / np.max(np.abs(np.where(live, y, 0))))
        out["W2"] = dict(valid_mismatch=int(np.sum(g(poles_td.valid_mask_q) != live)),
                         n_valid=int(np.sum(live)), n_modes=int(live.size),
                         rel_Omega=rel_live(g(poles_td.Omega_q), g(pr.Omega_q)),
                         rel_B=rel_live(g(poles_td.B_q), g(pr.B_q)),
                         rel_D=rel_live(g(poles_td.B_odd_q), g(pr.B_odd_q)),
                         odd_even=float(np.max(np.abs(np.where(live, g(pr.B_odd_q), 0)))
                                        / np.max(np.abs(np.where(live, g(pr.B_q), 0)))))
        # RW: chi0^fa dropped at both supports
        poles_bad, (W0_b, Wp_b) = PS.screening_update(
            (V_q, V_q), (cb(split["0"]), cb(split["p"])), plan)
        out["RW"] = dict(rel_W0_max=max(relq(gather(W0_b), W0h)), rel_Wp_max=max(relq(gather(Wp_b), Wph)))
        t0 = time.perf_counter()
        warm = PS.screening_update((V_q, V_q), (cs(split["0"]), cs(split["p"])), plan)
        jax.block_until_ready(warm[0].B_q)
        out["T"] = dict(screening_update_first_s=t_first,
                        screening_update_warm_s=time.perf_counter() - t0)
        del warm
        _done("W1", _t)
    except Exception:
        _failed("W1", _t)
        _stop(out, _save, 1)
        return

    # ---- S1 / S1d / S1w / RP / RF: the window Sigma at rho_eq ----
    _t = time.perf_counter()
    try:
        eye = np.broadcast_to(np.eye(nA, dtype=complex), (nk, nA, nA)).copy()
        window = dict(b_lo=b_lo, U=eye, e=e_full[:, b_lo:b_hi], f=f_full[:, b_lo:b_hi])
        freqs_abs = np.asarray(cap["request"].frequencies_ry) + float(cap["efermi_ry"])
        splan = PS.window_sigma_plan(config, head=cap["head"], frequencies_abs_ry=freqs_abs)
        say = print if rank0 else None
        t0 = time.perf_counter()
        s_td = PS.window_sigma_c(wfns, poles_td, window, splan, meta=meta, mesh_xy=mesh,
                                 print_fn=say)
        t_first = time.perf_counter() - t0
        t0 = time.perf_counter()
        s_run = PS.window_sigma_c(wfns, cap["poles"], window, splan, meta=meta, mesh_xy=mesh,
                                  print_fn=say)
        out["T"].update(window_sigma_first_s=t_first, window_sigma_warm_s=time.perf_counter() - t0)
        with h5py.File(restart, "r") as f:
            whead = float(np.asarray(f["whead"][()]).reshape(-1)[0])
            vhead = float(np.asarray(f["vhead"][()]).reshape(-1)[0])
        omega_cell = float(meta.cell_volume)
        zero = jnp.zeros((nk, nA, nA), jnp.complex128)
        sx = gather(jax.jit(lambda A__: D.exchange_sigma({**S_, **A__}, kern, zero,
                                                         heads=(whead, vhead, omega_cell)))(A_))
        ev = RYD_TO_EV
        pair_td, pair_w = s_td["pair"] * ev, s_run["pair"] * ev
        with h5py.File(os.path.join(a.ref, "sigma_mnk.h5"), "r") as f:
            dsets = sorted(k for k in f.keys())
            fc = np.asarray(f["sigma_c_kij_ev"][()])[:, :, b_lo:b_hi, b_lo:b_hi]
            fx_name = next((k for k in ("sigma_x_kij_ev", "sigma_sx_kij_ev") if k in f), None)
            fx = None if fx_name is None else np.asarray(f[fx_name][()])[:, b_lo:b_hi, b_lo:b_hi]
            ft = None                     # the writer's total is sigma_x + sigma_c + V_H: drop V_H
            if "sigma_total_kij_ev" in f and "hartree_kij_ev" in f:
                ft = (np.asarray(f["sigma_total_kij_ev"][()])[:, :, b_lo:b_hi, b_lo:b_hi]
                      - np.asarray(f["hartree_kij_ev"][()])[None, :, b_lo:b_hi, b_lo:b_hi])
        mx = lambda x: float(np.max(np.abs(x)))
        out["S1"] = dict(
            datasets=dsets, exchange_dataset=fx_name,
            max_abs_c_vs_file_ev=mx(pair_td - fc),
            max_abs_c_diag_vs_file_ev=mx(np.einsum("skii->ski", pair_td - fc)),
            max_abs_c_offdiag_vs_file_ev=mx((pair_td - fc) * (1 - np.eye(nA))[None, None]),
            max_abs_c_vs_inmemory_ev=mx(pair_td - cap["body_head_pair"] * ev),
            max_abs_c_file_ev=mx(fc),
            max_abs_x_vs_file_ev=None if fx is None else mx(sx * ev - fx),
            max_abs_x_file_ev=None if fx is None else mx(fx),
            max_abs_xc_vs_file_total_minus_hartree_ev=(
                None if ft is None else mx(pair_td + (sx * ev)[None] - ft)),
            efermi_ry=[float(cap["efermi_ry"]), s_td["efermi_ry"]])
        out["S1w"] = dict(max_abs_c_vs_file_ev=mx(pair_w - fc),
                          max_abs_c_vs_inmemory_ev=mx(pair_w - cap["body_head_pair"] * ev),
                          max_abs_td_vs_runpoles_ev=mx(pair_td - pair_w))
        ks, sxf, scr, sci, eo = _read_sigma_diag(os.path.join(a.ref, "sigma_diag.dat"), b_lo, b_hi)
        srow = np.asarray(flat_k_rows(ks, td["kgrid"]))
        ref_x = np.empty((nk, nA)); ref_x[srow] = sxf
        ref_c = np.empty((nk, nA)); ref_c[srow] = scr
        ref_eo = np.empty((nk, nA)); ref_eo[srow] = eo
        dg = np.real(np.einsum("kii->ki", pair_td[0]))
        dx = np.real(np.einsum("kii->ki", sx)) * ev
        out["S1d"] = dict(max_abs_sigC_ev=mx(dg - ref_c), max_abs_sigX_ev=mx(dx - ref_x),
                          max_abs_sigXC_ev=mx(dg + dx - ref_c - ref_x),
                          max_abs_Eo_vs_window_ev=mx(e_full[:, b_lo:b_hi] * ev - ref_eo),
                          max_abs_sigX_file_ev=mx(ref_x), max_abs_sigC_file_ev=mx(ref_c))
        _done("S1", _t)
        # red twins: the pole order swapped; the frozen bands omitted from G; chi0^fa dropped
        sw = dataclasses.replace(poles_td, B_odd_q=-poles_td.B_odd_q)
        rp = PS.window_sigma_c(wfns, sw, window, splan, meta=meta, mesh_xy=mesh)["pair"] * ev
        rf = PS.window_sigma_c(wfns, poles_td, window, splan, meta=meta, mesh_xy=mesh,
                               omit_frozen=True)["pair"] * ev
        rw = PS.window_sigma_c(wfns, poles_bad, window, splan, meta=meta, mesh_xy=mesh)["pair"] * ev
        out["RP"] = dict(max_abs_c_vs_file_ev=mx(rp - fc))
        out["RF"] = dict(max_abs_c_vs_file_ev=mx(rf - fc))
        out["RW"].update(max_abs_c_vs_file_ev=mx(rw - fc))
        _done("twins", _t)
    except Exception:
        _failed("S1", _t)

    out["t_total_s"] = time.perf_counter() - t_start
    _stop(out, _save, 1 if out.get("errors") else 0)


def _stop(out, save, rc):
    import jax
    save()
    if jax.process_index() == 0:
        print("RESULT " + json.dumps(out, default=float), flush=True)
    runtime.finalize_process(rc)


if __name__ == "__main__":
    main()
