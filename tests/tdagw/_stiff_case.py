"""The stiff split of cfm4 through the driver's stages (child of test_prop_stiff_split_cpu.py).

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/tdagw/_stiff_case.py OUTDIR

System: ``_validation_systems.synthetic_window`` (3v3c spinors, 3x3 k padded
to 12, ISDF-form Hartree + SEX full kernel) with its kernel scaled x6 plus a
rank-2 local-field Hartree term of 250 eV: the coupling map L has one outlier
quartet near +-5.8 +- 1.0i eV over a bulk below 2.8 eV (the shape of the
30x30 CrI3 model's: few large outliers, then a dense rest).  Prints one JSON
line (RESULT ...): the gate receipts and probes of the plain and split
schemes, runs at the same steps (P(t) against a 10 as plain cfm4
reference), 'auto' steps, the zero-basis run (Pi = 0: the split path with u
= b = 0) against the plain run, the closing-solve red twin (tolerance
unreachable, probe bypassed) and the parser's refusals.
"""
import json
import os
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from runtime.source_closure import ensure_source_closure   # noqa: E402

ensure_source_closure()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _validation_systems as vs   # noqa: E402

T_AS = vs.T_AS


def stiff_window():
    s = vs.synthetic_window()
    rng = np.random.default_rng(20261017)
    nk, nb = s["e_ev"].shape
    c = rng.standard_normal((nk, nb, 2, 8)) + 1j * rng.standard_normal((nk, nb, 2, 8))
    c /= np.sqrt(np.sum(np.abs(c) ** 2, axis=(2, 3), keepdims=True))
    m = np.einsum("knsu,kmsu->knmu", np.conj(c), c)
    q = rng.standard_normal((8, 2))
    q -= q.mean(0, keepdims=True)
    q, _ = np.linalg.qr(q)
    vb = (q * (250.0 / vs.HA_IN_EV) * np.array([1.0, 0.85])) @ q.T
    s["K_ha"] = 6.0 * s["K_ha"] + np.einsum("knmu,uv,KNMv->knmKNM", m, vb, np.conj(m)) / nk
    return s


def deck(outdir, header, kernel, nb, n_val, *, dt, n_corr=2, stiff=0, tol=1e-9, it=6,
         tf=6000.0, method="cfm4", e0=2e8):
    trial = vs.trial_for(e0, 2.2, 2500.0, 700.0, "x")
    lorrax = {"method": method, "n_corr": int(n_corr), "intraband": False, "trials": [trial],
              "kernel_file": str(kernel), "model": "tdgw_kernel", "dt": dt}
    if stiff:
        lorrax.update(stiff_modes=int(stiff), stiff_tol=float(tol), stiff_iter=int(it))
    return {"control": {"calc_type": "COHSEX",
                        "timesteps": {"t0": 0, "tf": float(tf), "dt_measure": 100.0,
                                      "ntcheck": 20},
                        "ivp": {"rtol": 1e-10, "atol": 1e-14},
                        "files": {"outdir": str(outdir), "load_header_from": str(header)}},
            "system": {"ibndmin": 1, "ibndmax": nb, "ivbm": int(n_val), "geometry": "slab"},
            "measurements": {"types": ["polarization", "carrier_density"]}, "lorrax": lorrax}


def main():
    import h5py
    from common.collectives import resolve_mesh
    from tdagw import propagate as tp, stages, tdgw_io
    from tdagw.td_config import parse_td_deck
    root = sys.argv[1]
    mesh = resolve_mesh()
    s = stiff_window()
    nb, n_val = int(s["e_ev"].shape[1]), int(s["n_val"])
    header = vs.write_header(os.path.join(root, "hdr"), s)
    kernel = vs.write_kernel(os.path.join(root, "kernel.h5"), s, tda=False)

    def run(tag, hook=None, **kw):
        raw = deck(os.path.join(root, tag), header, kernel, nb, n_val, **kw)
        try:
            d = parse_td_deck(raw, source=tag)
        except ValueError as exc:
            return dict(refused=str(exc).split(":")[0], message=str(exc))
        system = stages.load_system(d, mesh)
        model = stages.build_self_energy(d, system, mesh)
        outs = stages.open_outputs(d, system)
        lines = []
        try:
            if hook is not None:
                hook()
            st = stages.propagate(d, system, model, tdgw_io.efield_function(d.trials), outs, mesh,
                                  say=lines.append)
        except ValueError as exc:
            return dict(refused=str(exc).split(":")[0], message=str(exc), log=lines)
        with h5py.File(outs.paths[0], "r") as f:
            td = f["tdgw_data/time_domain"]
            p = td["polarization/data"][()][:, 0]
            n = td["carrier_density/data"][()][:, 0]
        return dict(dt_as=st["dt"] * T_AS, p=p, n_c=n, probes=st["check_probes"], log=lines,
                    fast=st.get("fast"), dsigma=st["dsigma"], stiff_dsigma=st.get("stiff_dsigma", 0))

    out = {}
    # the full and the slow radius on the operators the gate sees
    d0 = parse_td_deck(deck(os.path.join(root, "sig"), header, kernel, nb, n_val, dt=50.0), source="sig")
    sys0 = stages.load_system(d0, mesh)
    mod0 = stages.build_self_energy(d0, sys0, mesh)
    bas = tp.stiff_basis(sys0.ops, mod0, mesh, t0=0.0, n_s=4)
    off = lambda t: jax.numpy.zeros((1, 3))
    out["sigma_full_ev"] = tp.coupling_radius(None, sys0.ops, mod0, off, mesh, t0=0.0)[0] * vs.HA_IN_EV
    out["sigma_slow_ev"] = tp.coupling_radius(None, sys0.ops._replace(stiff=(bas["u"], bas["b"])), mod0,
                                              off, mesh, t0=0.0)[0] * vs.HA_IN_EV
    out["basis"] = dict(n_s=bas["n_s"], cut_ev=bas["cut_ev"], residual_ev=bas["residual_ev"],
                        top_ev=[float(abs(z)) for z in bas["ritz_ev"][:8]])
    ref = run("ref", dt=10.0)
    out["ref_max_p"] = float(np.max(np.abs(ref["p"])))
    out["ref_max_nc"] = float(np.max(np.abs(ref["n_c"])))
    err = lambda r: float(np.linalg.norm(r["p"] - ref["p"]) / np.linalg.norm(ref["p"]))
    for tag, kw in (("plain_50", dict(dt=50.0)), ("plain_100", dict(dt=100.0)),
                    ("split_100", dict(dt=100.0, stiff=4)), ("split_50", dict(dt=50.0, stiff=4)),
                    ("plain_auto", dict(dt="auto")), ("split_auto", dict(dt="auto", stiff=4))):
        r = run(tag, **kw)
        if "p" in r:
            r["rel_err"] = err(r)
            r.pop("p"); r.pop("n_c")
        out[tag] = r
        print("RUN", tag, json.dumps({k: v for k, v in r.items() if k in ("dt_as", "rel_err", "refused", "message", "fast", "probes", "stiff_dsigma")}, default=str)[:900], flush=True)
    # Pi = 0: the split path with a zero basis against the plain step (same dt, gate bypassed alike)
    real_basis = tp.stiff_basis

    def zero_basis(ops, model, mesh, **kw):
        b = real_basis(ops, model, mesh, **kw)
        return dict(b, u=0.0 * b["u"], b=0.0 * b["b"])
    real_growth = tp.step_growth
    tp.stiff_basis = zero_basis
    z = run("zero_30", dt=30.0, stiff=4)
    tp.stiff_basis = real_basis
    pl = run("plain_30", dt=30.0)
    out["zero_vs_plain"] = dict(
        bitwise=bool(np.array_equal(z["p"], pl["p"]) and np.array_equal(z["n_c"], pl["n_c"])),
        max_abs=float(np.max(np.abs(z["p"] - pl["p"]))), max_p=float(np.max(np.abs(pl["p"]))),
        fast=z["fast"])
    # red twin: an unreachable closing tolerance with the probe bypassed -> td_stiff_solve at a chunk end
    tp.step_growth = lambda *a, **k: (1.0, np.ones(1), None) if k.get("counters") else (1.0, np.ones(1))
    out["unreachable_tol"] = run("twin_tol", dt=50.0, stiff=4, tol=1e-300, it=2)
    tp.step_growth = real_growth
    for key in ("unreachable_tol",):
        out[key].pop("log", None)
    out["bad_method"] = run("bad_method", dt=50.0, stiff=4, method="rk4")
    print("RESULT " + json.dumps(out, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
          flush=True)


if __name__ == "__main__":
    main()
