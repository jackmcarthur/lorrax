"""The corrected-Magnus step limit on the TDGW fixture (child of test_prop_stability_cpu.py).

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/tdagw/_stability_case.py OUTDIR SCALE

The fixture's TDA kernel is scaled by SCALE (a strong coupling, the shape of
the 30x30 42v x 14c model's).  Prints one JSON line: sigma from the power
iteration and from the dense spectrum of L, the gate's dt_max, the outcome
of cfm4 runs at 2 dt_max (expect the refusal) and dt_max / 2 (expect a
run), dt = 'auto', and the same 2 dt_max run with the gate bypassed (the
red twin: max |P| against the stable run's).
"""
import copy
import json
import os
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from runtime.source_closure import ensure_source_closure   # noqa: E402

ensure_source_closure()

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "syn2v2c")


def _deck(outdir, dt, tf=5000.0):
    meta = json.load(open(os.path.join(FIXTURE, "deck_tdgw.json")))
    raw = copy.deepcopy(meta["deck"])
    raw["control"]["files"]["indir"] = os.path.join(FIXTURE, "indata")
    raw["control"]["files"]["load_header_from"] = os.path.join(FIXTURE, "tdgw_ref")
    raw["control"]["files"]["outdir"] = outdir
    raw["control"]["timesteps"]["tf"] = tf
    raw["lorrax"] = {"method": "cfm4", "n_corr": 2, "dt": dt}
    return raw


def main():
    import h5py
    from common.collectives import resolve_mesh
    from tdagw import propagate as tp, stages, tdgw_io
    from tdagw.td_config import parse_td_deck
    from tdagw.tdgw_io import efield_function, T_IN_ATTOSEC, HA_IN_EV
    outdir, scale = sys.argv[1], float(sys.argv[2])
    mesh = resolve_mesh()

    def build(dt, tag):
        deck = parse_td_deck(_deck(os.path.join(outdir, tag), dt), source=tag)
        system = stages.load_system(deck, mesh)
        model = stages.build_self_energy(deck, system, mesh)
        model = model._replace(params=(scale * model.params[0],) + tuple(model.params[1:]))
        return deck, system, model

    def run(dt, tag):
        deck, system, model = build(dt, tag)
        outs = stages.open_outputs(deck, system)
        try:
            st = stages.propagate(deck, system, model, efield_function(deck.trials), outs,
                                  mesh, say=lambda s: None)
        except ValueError as exc:
            return dict(refused=str(exc).split(":")[0])
        with h5py.File(outs.paths[0], "r") as f:
            p = f["tdgw_data/time_domain/polarization/data"][()]
        return dict(dt_as=st["dt"] * T_IN_ATTOSEC, max_p=float(np.max(np.abs(p))),
                    finite=bool(np.all(np.isfinite(p))))

    deck, system, model = build(1.0, "probe")
    sigma, est = tp.coupling_radius(None, system.ops, model, efield_function(deck.trials),
                                    mesh, t0=0.0)
    # the dense spectrum of L(x) = -i (K x) o (f_m - f_n) on the cv / vc blocks
    from tdagw.tdgw_io import read_bgw_kernel
    kern, tda, cs, vs, _ = read_bgw_kernel(os.path.join(FIXTURE, "indata", "kernel.h5"),
                                           ivbm=int(deck.values["system.ivbm"]), bands=system.bands)
    nk, nc, nv = kern.shape[0], cs[1] - cs[0], vs[1] - vs[0]
    km = scale * kern.reshape(nk * nc * nv, nk * nc * nv)
    exact = float(np.max(np.abs(np.linalg.eigvals(km))))    # TDA: L = -i K on the cv block
    cfg = tp.PropagatorConfig("cfm4", 0.0, 1.0, 1.0, 1.0, 2, 1e-6, 1e-6, np.inf, 10, True)
    dt_max = tp.magnus_dt_limit(cfg, sigma)[0] * T_IN_ATTOSEC
    out = dict(sigma_ev=sigma * HA_IN_EV, exact_ev=exact * HA_IN_EV, dt_max_as=dt_max,
               over=run(2.0 * dt_max, "over"), under=run(0.5 * dt_max, "under"),
               auto=run("auto", "auto"))
    tp.magnus_dt_limit = lambda cfg, sigma: (np.inf, 0.0)    # the twin: no gate
    out["over_ungated"] = run(2.0 * dt_max, "over_ungated")
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
