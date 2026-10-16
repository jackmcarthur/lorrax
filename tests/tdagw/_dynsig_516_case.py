"""tdagw.dynsig on a real COHSEX restart (CrI3 516, 6x6 k, 2v2c): t = 0 consistency + frozen-W pulse.

    lx run -N 1 -G 4 -n 4 -- python3 -u tests/tdagw/_dynsig_516_case.py \\
        --deck <05>/oneshot.in --sigma-diag <05>/sigma_diag.dat --dipole <05>/dipole.h5 \\
        --bands 128 132 --out DIR [--eqp eqp1.dat] [--dry]

One JSON line (rank 0), ``RESULT {...}``.  The restart is read-only.
H0 is the DFT spectrum unless ``--eqp`` is given: every check below compares
self-energies (or two models over the same H0), so none depends on it, and the
04 run's diagonal QP spectrum puts a 0.548 meV near-degeneracy at the 128 edge
(k = 24), which the band-window gate refuses at its 1 meV tolerance.

L0  orders: read_wavefunctions rows, the metadata energies and the stored
    V_qmunu/W0_qmunu rows mapped onto the TD model's flat C-order k/q rows;
    the window psi equal the TD model's (exact), W0 at q != 0 equal the TD
    model's W (exact).  --dry stops here.
B1  chi0^ff + chi0^fa(0) + chi0^aa(0) on the run's own Laplace rule (re-solved
    from the run's interval and target), Dyson with the run's V: W(0) vs the
    run's W0_qmunu, per q (q = 0 reported apart: its head is a separate channel)
B2  Sigma^SX and Sigma^COH of the window at rho_eq from W(0), plus the run's
    exact band-diagonal heads, vs sigma_diag.dat (sigSX, sigCOH; eV, printed
    to 1e-6): max |d| over the window bands and the 36 k
B2w the same from the run's W0 (separates the chi0 reproduction from the
    Sigma assembly)
RB1 chi0^fa dropped;  RB2 the active valence counted empty in G (a wrong
    occupation update): both must fail B1/B2 by orders of magnitude
A1  frozen-W pulse through the propagator: dynsig(screening=frozen) vs
    isdf_model, P(t) and rho(t_f)
A2  the literal Eq. (1) in a rotated instantaneous basis at rho(t_f), frozen W:
    Sigma^SX[rho] - Sigma^SX[rho_eq] vs -K^W drho of the TD model;  RA2 the
    coherences dropped
T1  one static_rpa apply at rho(t_f): flags, |dSigma^dyn| / |dSigma^static|,
    warm wall time (s), the same with the exact same-side pairs off
"""
import argparse
import functools
import json
import os
import time

import runtime  # noqa: F401  (before numpy: the runtime tunes BLAS spin at import)
import numpy as np

RYD_EV = 13.605693122994


def _read_sigma_diag(path, n_bands):
    """sigma_diag.dat -> (kcrys (nk, 3), sx (nk, nb), coh (nk, nb)) in eV, file k order."""
    ks, sx, coh, cur = [], [], [], None
    with open(path) as f:
        for line in f:
            if line.startswith("# kcrys"):
                ks.append([float(v) for v in line.split()[2:5]])
                cur = (np.full(n_bands, np.nan), np.full(n_bands, np.nan))
                sx.append(cur[0]); coh.append(cur[1])
            elif line.startswith("n=") and cur is not None:
                tok = line.replace("=", " = ").split()
                n = int(tok[2])
                vals = dict(zip(tok[3::3], tok[5::3]))
                if n < n_bands:
                    cur[0][n] = float(vals["sigSX"]); cur[1][n] = float(vals["sigCOH"])
    return np.asarray(ks), np.asarray(sx), np.asarray(coh)


def _rows_of(k_frac, kgrid):
    from tdagw.model import flat_k_rows
    return np.asarray(flat_k_rows(np.asarray(k_frac), kgrid), np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", required=True)
    ap.add_argument("--sigma-diag", required=True)
    ap.add_argument("--dipole", required=True)
    ap.add_argument("--eqp", default=None, help="eqp1.dat for a QP H0 (default: DFT)")
    ap.add_argument("--bands", type=int, nargs=2, default=(128, 132))
    ap.add_argument("--nbf", type=int, default=160)
    ap.add_argument("--target-error", type=float, default=1e-7)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    stack = runtime.initialize_communicator_stack()      # before any JAX call (multi-process)
    import jax
    import h5py
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jax.experimental import multihost_utils
    from file_io.restart_bundle import (_find_restart_file, read_interaction, read_metadata,
                                        read_wavefunctions)
    from tdagw import dynsig as D, propagate as tp
    from tdagw.self_energy import isdf_model
    from tdagw.td_system import build_td_system
    from tdagw.tdgw_io import header_operators
    from tdagw.model import build_td_agw_apply, pad_td_state, strip_td_state, td_state_sharding
    from gw.gw_config import LorraxConfig
    from vcoul import CoulombGeometry
    from wfn_loader import WfnLoader

    mesh = stack.mesh
    rank0 = jax.process_index() == 0
    out = dict(mesh=list(mesh.devices.shape), deck=os.path.abspath(a.deck), bands=list(a.bands))
    gather = lambda x: np.asarray(multihost_utils.process_allgather(x, tiled=True))
    t_start = time.perf_counter()

    # ---- the TD system and model of the window (TDPROP / TDMODEL loaders) ----
    sysm = build_td_system(a.deck, mesh, bands=tuple(a.bands), dipole_file=a.dipole,
                           eqp_file=a.eqp, log=(print if rank0 else (lambda s: None)))
    td, hdr = sysm.model, sysm.hdr
    kgrid = tuple(int(n) for n in td["kgrid"])
    nk = int(np.prod(kgrid))
    restart = _find_restart_file(a.deck)
    cfg = LorraxConfig.from_input_file(a.deck)
    wfn = WfnLoader(cfg.paths.wfn_file)
    omega = float(CoulombGeometry.from_wfn(wfn).cell_volume)
    k_file = np.asarray(wfn.kvecs(k="full_bz"))
    wfn.close()
    rows_file = _rows_of(k_file, kgrid)               # WFN full-BZ row i -> C-order row
    order = np.argsort(rows_file)                     # C-order row j <- WFN row order[j]
    b_lo, b_hi = (int(b) for b in a.bands)
    nA = b_hi - b_lo

    # ---- every band at the centroids, energies, V, W0 (read-only) ----
    # psi rows are matched against the TD model's own window psi (the BSE
    # loader's C-order rows; the same stored numbers, so the match is exact),
    # not assumed; the energies follow the WFN k coordinates and are checked
    # against the TD header below.
    psi_file = read_wavefunctions(restart, a.deck, mesh, bands=np.arange(a.nbf))
    pw = gather(psi_file[:, b_lo:b_hi])
    px = gather(td["psi_X"])[:, :nA]
    mu_c = min(pw.shape[-1], px.shape[-1])
    dmat = np.array([[np.max(np.abs(pw[i, ..., :mu_c] - px[j, ..., :mu_c])) for i in range(nk)]
                     for j in range(nk)])
    psi_rows = np.argmin(dmat, axis=1)                # C-order row j <- loader row psi_rows[j]
    psi_full = jax.jit(lambda p: jnp.take(p, jnp.asarray(psi_rows), axis=0),
                       out_shardings=NamedSharding(mesh, P(None, None, None, "x")))(psi_file)
    with h5py.File(restart, "r") as f:
        enk = np.asarray(f["enk_full"][()], np.float64)[:, :a.nbf]      # parent rows
        prow = (np.asarray(f["psi_parent_k_rows"][()], np.int64)
                if "psi_parent_k_rows" in f else np.arange(enk.shape[0]))
        whead = float(np.asarray(f["whead"][()]).reshape(-1)[0])
        vhead = float(np.asarray(f["vhead"][()]).reshape(-1)[0])
    if enk.shape[0] != nk:
        raise ValueError(f"GATE dynsig_516_parent_rows: got: {enk.shape[0]} parent k rows; "
                         f"want: the full {nk} (identity-only grid); fix: this case reads a "
                         "nosym bundle only")
    enk_file = np.empty_like(enk); enk_file[prow] = enk                  # file full-k rows
    e_full = enk_file[order]
    n_occ = int(td["receipts"]["n_occ"])
    f_full = np.zeros_like(e_full); f_full[:, :n_occ] = 1.0
    V_q = read_interaction(restart, "bare", mesh)
    W0_q = read_interaction(restart, "screened", mesh)

    # L0: orders against the TD model's own arrays
    dpsi = float(np.max(dmat[np.arange(nk), psi_rows]))
    e_hdr = np.empty((nk, nA)); e_hdr[_rows_of(np.asarray(hdr["kpts"]), kgrid)] = \
        2.0 * np.asarray(hdr["e_kn"], np.float64)[:nk, :nA]            # Ha -> Ry, C rows
    kern = D.make_dynsig_kernels(mesh, kgrid)
    W_td_q = gather(kern.c_fft(jnp.moveaxis(td["W_R"].reshape(td["W_R"].shape[0],
                                                              td["W_R"].shape[1], nk), -1, 0)))
    W0_h = gather(W0_q)
    q_perm = []
    for j in range(nk):
        d = [np.max(np.abs(W0_h[i] - W_td_q[j])) for i in range(nk)]
        q_perm.append(int(np.argmin(d)))
    q_perm = np.asarray(q_perm)
    dW_rows = [float(np.max(np.abs(W0_h[q_perm[j]] - W_td_q[j])) / np.max(np.abs(W_td_q[j])))
               for j in range(nk)]
    out["L0"] = dict(dpsi_window=dpsi, psi_shapes=[list(pw.shape), list(px.shape)],
                     psi_rows_unique=bool(len(set(psi_rows.tolist())) == nk),
                     psi_rows_is_wfn_order=bool(np.all(psi_rows == order)),
                     psi_rows_identity=bool(np.all(psi_rows == np.arange(nk))),
                     e_window_vs_header_ry=float(np.max(np.abs(e_full[:, b_lo:b_hi] - e_hdr))),
                     q_perm_identity=bool(np.all(q_perm == np.arange(nk))),
                     q_perm_unique=bool(len(set(q_perm.tolist())) == nk),
                     w0_vs_td_W_rel_q0=dW_rows[0], w0_vs_td_W_rel_qne0=max(dW_rows[1:]),
                     omega_bohr3=omega, whead=whead, vhead=vhead, n_occ=n_occ,
                     parent_rows_identity=bool(np.all(prow == np.arange(nk))),
                     file_rows_identity=bool(np.all(order == np.arange(nk))),
                     e_vs_metadata_ry=float(np.max(np.abs(e_full - np.asarray(
                         read_metadata(restart)["energies"], np.float64)[:, :a.nbf]))))
    if not out["L0"]["q_perm_unique"]:
        raise ValueError("GATE dynsig_516_q_order: got: W0_qmunu rows that do not map one-to-one "
                         "onto the TD model's q rows; want: a permutation; fix: inspect the store")
    if not np.all(q_perm == np.arange(nk)):
        V_q = jax.jit(lambda v: jnp.take(v, jnp.asarray(q_perm), axis=0),
                      out_shardings=NamedSharding(mesh, D.CHI_SPEC))(V_q)
        W0_q = jax.jit(lambda v: jnp.take(v, jnp.asarray(q_perm), axis=0),
                       out_shardings=NamedSharding(mesh, D.CHI_SPEC))(W0_q)
    out["L0"]["t_load_s"] = time.perf_counter() - t_start

    def _save():
        if rank0:
            os.makedirs(a.out, exist_ok=True)
            with open(os.path.join(a.out, "dynsig_516.json"), "w") as f:
                json.dump(out, f, indent=1, default=float)

    def _failed(name, t0):
        """A failed check is recorded in the JSON (never silent); the next one still runs."""
        import traceback
        out.setdefault("errors", {})[name] = traceback.format_exc()[-2000:]
        out.setdefault("section_wall_s", {})[name] = time.perf_counter() - t0
        if rank0:
            print(f"[dynsig_516] section {name} FAILED:\n{out['errors'][name]}", flush=True)
        _save()

    def _done(name, t0):
        out.setdefault("section_wall_s", {})[name] = time.perf_counter() - t0
        _save()

    def _stop(rc):
        _save()
        if rank0:
            print("RESULT " + json.dumps(out, default=float), flush=True)
        runtime.finalize_process(rc)

    _save()
    if a.dry:
        return _stop(0)

    _t = time.perf_counter()
    try:
        rule = D.static_rule(e_full, f_full, target_error=a.target_error, max_nodes=64)
        out["rule"] = dict(nodes=int(len(rule["tau"])), x_min=rule["x_min"], x_max=rule["x_max"],
                           max_error=rule["max_error"], e_sep=rule["e_sep"])
        dyn = D.assemble_dynsig_bundle(td, psi_full, e_full, f_full, W0_q, mesh_xy=mesh,
                                       rule=rule, V_q=V_q)
        A, S = D.split_bundle(dyn)                   # arrays are jit operands (multi-process)
        _done("bundle", _t)
    except Exception:
        _failed("bundle", _t)
        return _stop(1)
    from gw.w_isdf import solve_w
    from runtime.padding import padded_mu_axis
    axis = padded_mu_axis(int(td["n_rmu"]), mesh)
    dyson = lambda chi: solve_w(V_q, chi, None, mesh, pref=1.0, axis=axis)
    W0h = gather(W0_q)
    rel_q = lambda W: [float(np.max(np.abs(W[q] - W0h[q])) / np.max(np.abs(W0h[q])))
                       for q in range(nk)]

    _t = time.perf_counter()
    try:
        # ---- B1 / RB1 / RB2: chi0 split -> W(0) ----
        t0 = time.perf_counter()
        sp = jax.jit(lambda A_: D.chi0_split({**S, **A_}, kern))(A)
        jax.block_until_ready(sp["ff"])
        t_chi = time.perf_counter() - t0
        chi = sp["ff"] + sp["fa"] + sp["aa"]
        W = gather(dyson(chi))
        nrm = lambda x: float(np.max(np.abs(gather(x))))
        out["B1"] = dict(rel_q0=rel_q(W)[0], rel_qne0_max=max(rel_q(W)[1:]),
                         rel_all_max=max(rel_q(W)), rel_per_q=rel_q(W),
                         fa_share=nrm(sp["fa"]) / nrm(chi), aa_share=nrm(sp["aa"]) / nrm(chi),
                         chi_build_s=t_chi)
        _done("B1", _t)
    except Exception:
        _failed("B1", _t)

    _t = time.perf_counter()
    W_nofa = W_bad = None
    try:
        W_nofa = gather(dyson(sp["ff"] + sp["aa"]))                   # RB1: chi0^fa dropped
        out["RB1"] = dict(rel_all_max=max(rel_q(W_nofa)))
        # RB2: active valence counted empty -> G_eq loses those holes (and the aa/fa terms)
        f_bad = f_full.copy(); f_bad[:, b_lo:n_occ] = 0.0
        dyn_bad = D.assemble_dynsig_bundle(td, psi_full, e_full, f_bad, W0_q, mesh_xy=mesh,
                                           rule=rule, V_q=V_q)
        Ab, Sb = D.split_bundle(dyn_bad)
        spb = jax.jit(lambda A_: D.chi0_split({**Sb, **A_}, kern))(Ab)
        W_bad = gather(dyson(spb["ff"] + spb["fa"] + spb["aa"]))
        out["RB2"] = dict(rel_all_max=max(rel_q(W_bad)))
        _done("RB1_RB2", _t)
    except Exception:
        _failed("RB1_RB2", _t)

    _t = time.perf_counter()
    try:
        # ---- B2 / B2w: Sigma at rho_eq vs sigma_diag.dat ----
        ks, sxf, cohf = _read_sigma_diag(a.sigma_diag, b_hi)
        srow = _rows_of(ks, kgrid)
        ref_sx, ref_coh = np.empty((nk, nA)), np.empty((nk, nA))
        ref_sx[srow], ref_coh[srow] = sxf[:, b_lo:b_hi], cohf[:, b_lo:b_hi]
        from common.collectives import device_put_process_local
        put = lambda x: device_put_process_local(np.asarray(x), NamedSharding(mesh, D.CHI_SPEC))
        heads = (whead, vhead, omega)

        def sigma_err(Wq):
            sx, coh = jax.jit(lambda A_, w: D.equilibrium_sigma({**S, **A_}, kern, w,
                                                                heads=heads))(A, Wq)
            sx = np.real(np.einsum("kii->ki", gather(sx))) * RYD_EV
            coh = np.real(np.einsum("kii->ki", gather(coh))) * RYD_EV
            return dict(max_abs_sx_ev=float(np.max(np.abs(sx - ref_sx))),
                        max_abs_coh_ev=float(np.max(np.abs(coh - ref_coh))),
                        max_abs_tot_ev=float(np.max(np.abs(sx + coh - ref_sx - ref_coh))))
        out["B2w"] = sigma_err(W0_q)
        out["B2"] = sigma_err(put(W))
        sx0, _ = jax.jit(lambda A_, w: D.equilibrium_sigma({**S, **A_}, kern, w,
                                                            heads=None))(A, W0_q)
        out["B2w_nohead"] = dict(max_abs_sx_ev=float(np.max(np.abs(
            np.real(np.einsum("kii->ki", gather(sx0))) * RYD_EV - ref_sx))))
        out["B2_ref"] = dict(file=os.path.abspath(a.sigma_diag),
                             max_abs_sx_ev=float(np.max(np.abs(ref_sx))),
                             max_abs_coh_ev=float(np.max(np.abs(ref_coh))))
        _done("B2", _t)
        for tag, Wt in (("RB1", W_nofa), ("RB2", W_bad)):
            if Wt is not None:
                out[tag].update(sigma_err(put(Wt)))
        _done("B2", _t)
    except Exception:
        _failed("B2", _t)

    _t = time.perf_counter()
    try:
        # ---- A1: frozen-W pulse (isdf vs dynsig frozen) ----
        host = header_operators(hdr, geometry="slab", esmear_ha=0.05 / 27.211386, intraband=False)
        ops, nk_pad = tp.place_operators(host, mesh)
        k_hdr = np.asarray(hdr["kpts"])
        rho_eq_h = np.zeros((nk_pad, nA, nA), complex)
        rho_eq_h[:nk] = gather(ops.rho_eq)[:nk]
        rho_eq = tp._place(rho_eq_h, mesh, 0)
        as_ = 41.341374575751 / 1000.0                    # a.u. per as
        dt, n_sub, n_meas = 20 * as_, 5, 60               # 20 as steps, 6 fs
        cfg_p = tp.PropagatorConfig(method="rk4", t0=0.0, tf=dt * n_sub * (n_meas - 1),
                                    dt_measure=dt * n_sub, dt=dt, n_sub=n_sub, n_corr=1, atol=1e-10,
                                    rtol=1e-8, max_step=np.inf, chunk=n_meas, remainder=False)
        spec = tp.SampleSpec(kinds=(("polarization", "polarization", ()),))
        e0 = 5e8 / 5.14220674763e11                       # 5e8 V/m in a.u.

        def drive(t):
            g = e0 * jnp.exp(-((t - 3.0 * 41.34) / (1.0 * 41.34)) ** 2) * jnp.cos(0.07 * t)
            return jnp.stack([g, 0.0 * g, 0.0 * g])[None]

        def run(model):
            res = []
            stats, carry = tp.run_propagation(
                cfg_p, spec, tp.place_state(rho_eq_h[None], mesh), ops, model, drive, mesh,
                cell_volume=float(hdr["cell_volume"]) if "cell_volume" in hdr else omega,
                nspinor=int(td["nspinor"]), count=1, mtsteps=np.arange(n_meas) * dt * n_sub,
                on_chunk=lambda i0, i1, s, rf, t: res.append((np.asarray(s["polarization"]), rf())))
            return np.concatenate([r[0] for r in res]), res[-1][1], stats

        iso = isdf_model(td, mesh, k_frac=k_hdr, rho_eq=rho_eq)
        frz = D.dynsig_model(dyn, mesh, k_frac=k_hdr, rho_eq=rho_eq, screening="frozen")
        p_iso, r_iso, _ = run(iso)
        p_frz, r_frz, st = run(frz)
        relp = float(np.max(np.abs(p_frz - p_iso)) / np.max(np.abs(p_iso)))
        out["A1"] = dict(rel_P=relp, rel_rho=float(np.max(np.abs(r_frz - r_iso))
                                                   / np.max(np.abs(r_iso - rho_eq_h[None]))),
                         max_abs_P=float(np.max(np.abs(p_iso))), dsigma_calls=int(st["dsigma"]))

        _done("A1", _t)
    except Exception:
        _failed("A1", _t)

    _t = time.perf_counter()
    try:
        # ---- A2 / RA2: literal static limit at rho(t_f) in a rotated basis ----
        rows_m = _rows_of(k_hdr, kgrid)                   # header row -> model row (identity here)
        x = np.zeros((nk, nA, nA), complex)
        x[rows_m] = np.asarray(r_iso)[0, :nk] - rho_eq_h[:nk]
        from common.collectives import device_put_process_local as _dpl
        xd = _dpl(x, NamedSharding(mesh, P()))
        rng = np.random.default_rng(516)
        hrot = np.stack([np.diag(e_full[k, b_lo:b_hi]) for k in range(nk)]) + 0.01 * rng.standard_normal(
            (nk, nA, nA))
        hrot = _dpl(0.5 * (hrot + np.swapaxes(hrot, -1, -2)), NamedSharding(mesh, P()))
        sex = build_td_agw_apply(mesh, td, sex=True, hartree=False)
        xs = _dpl(pad_td_state(x[None], td), td_state_sharding(mesh))
        kw_ref = strip_td_state(gather(sex(xs)), td)[0]
        WR = td["W_R"]
        for tag, cohs in (("A2", True), ("RA2", False)):
            lit = jax.jit(lambda A_, x_, w_, h_: D.literal_sx({**S, **A_}, kern, x_, w_, h=h_,
                                                             coherences=cohs)
                          - D.literal_sx({**S, **A_}, kern, jnp.zeros_like(x_), w_, h=h_))(
                A, xd, WR, hrot)
            out[tag] = dict(rel=float(np.max(np.abs(gather(lit) - kw_ref)) / np.max(np.abs(kw_ref))),
                            max_abs_drho=float(np.max(np.abs(x))))

        _done("A2", _t)
    except Exception:
        _failed("A2", _t)

    _t = time.perf_counter()
    try:
        # ---- T1: one static_rpa apply at rho(t_f) ----
        from tdagw.self_energy import _ISDF_OPERANDS
        td_ops = tuple(td[k] for k in _ISDF_OPERANDS)
        td_static = dict(kgrid=td["kgrid"], nspinor=td["nspinor"])
        td_apply = build_td_agw_apply(mesh, td)

        def correction(A_, ops_, xx, ss):
            model = dict(zip(_ISDF_OPERANDS, ops_), **td_static)
            app = build_td_agw_apply(mesh, model)

            def static_ds(y):
                yp = jnp.pad(y, ((0, 0), (0, td["nb_pad"] - nA), (0, td["nb_pad"] - nA)))[None]
                return app(yp)[0, :, :nA, :nA]
            d, fl = D.dynamic_correction({**S, **A_}, kern, xx, static_dsigma=static_ds,
                                         same_side=ss)
            return d, {k: fl[k] for k in ("crossed", "out_of_rule", "degenerate")}

        for tag, ss in (("T1", True), ("T1_noss", False)):
            fn = jax.jit(functools.partial(correction, ss=ss))
            r_ = fn(A, td_ops, xd); jax.block_until_ready(r_[0])
            t0 = time.perf_counter(); r_ = fn(A, td_ops, xd); jax.block_until_ready(r_[0])
            wall = time.perf_counter() - t0
            dyn_c = gather(r_[0])
            stat = strip_td_state(gather(td_apply(xs)), td)[0]                     # Ry
            out[tag] = dict(wall_s=wall, flags={k: int(gather(r_[1][k]))
                                                for k in ("crossed", "out_of_rule", "degenerate")},
                            max_abs_dyn_ry=float(np.nanmax(np.abs(dyn_c))),
                            ratio_dyn_static=float(np.nanmax(np.abs(dyn_c)) / np.max(np.abs(stat))),
                            finite=bool(np.all(np.isfinite(dyn_c))))
        _done("T1", _t)
    except Exception:
        _failed("T1", _t)

    out["t_total_s"] = time.perf_counter() - t_start
    _stop(1 if out.get("errors") else 0)


if __name__ == "__main__":
    main()
