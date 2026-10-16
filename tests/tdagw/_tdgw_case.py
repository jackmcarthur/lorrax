"""Run the TD-aGW propagator on the frozen TDGW fixture; one JSON line per case.

A child process of ``test_prop_tdgw_parity_cpu.py`` (the host device count
must be fixed before jax starts)::

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/tdagw/_tdgw_case.py OUTDIR '[{"tag": "base"}]'

Each case dict may set method, dt (as), n_corr, intraband (bool) and
kernel_sign (multiplies the dense kernel K only, not rho_eq).
"""
import json
import os
import sys

import h5py
import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from runtime.source_closure import ensure_source_closure   # noqa: E402

ensure_source_closure()

FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "syn2v2c")


def run_case(case, outdir, mesh):
    from tdagw import stages
    from tdagw.td_config import parse_td_deck
    from tdagw.tdgw_io import efield_function

    meta = json.load(open(os.path.join(FIXTURE, "deck_tdgw.json")))
    raw = meta["deck"]
    files = raw["control"]["files"]
    files["indir"] = os.path.join(FIXTURE, "indata")
    files["load_header_from"] = os.path.join(FIXTURE, "tdgw_ref")
    files["outdir"] = os.path.join(outdir, case["tag"])
    raw["lorrax"] = {"method": case.get("method", "dp54"), "n_corr": case.get("n_corr", 1),
                     "intraband": case.get("intraband", True)}
    if case.get("dt") is not None:
        raw["lorrax"]["dt"] = case["dt"]
    deck = parse_td_deck(raw, source=case["tag"])
    system = stages.load_system(deck, mesh)
    model = stages.build_self_energy(deck, system, mesh)
    sign = case.get("kernel_sign", 1)
    if sign != 1:
        model = model._replace(params=(sign * model.params[0],) + tuple(model.params[1:]))
    outs = stages.open_outputs(deck, system)
    stats = stages.propagate(deck, system, model, efield_function(deck.trials), outs, mesh,
                             say=lambda s: None)
    res = dict(tag=case["tag"], dsigma=int(stats["dsigma"]), tdgw_nfev=int(meta["tdgw_nfev"]),
               accepted=int(stats["accepted"]), rel={}, series={})
    with h5py.File(os.path.join(FIXTURE, "tdgw_ref", "tdgw.h5"), "r") as ref, \
            h5py.File(outs.paths[0], "r") as new:
        for name in ref["tdgw_data/time_domain"]:
            a = ref["tdgw_data/time_domain"][name]["data"][()]
            b = new["tdgw_data/time_domain"][name]["data"][()]
            res["rel"][name] = float(np.max(np.abs(a - b)) / (np.max(np.abs(a)) or 1.0))
        p = new["tdgw_data/time_domain/polarization/data"][()]
        res["series"]["polarization"] = [[float(v.real), float(v.imag)] for v in p.ravel()]
        ca, cb = ref["tdgw_data/checkpoint"], new["tdgw_data/checkpoint"]
        res["it_next"] = [int(ca["it_next"][()]), int(cb["it_next"][()])]
        res["tchkp"] = [float(ca["tchkp"][()]), float(cb["tchkp"][()])]
        res["lessG"] = float(np.max(np.abs(ca["lessG_knn"][()] - cb["lessG_knn"][()]))
                             / np.max(np.abs(ca["lessG_knn"][()])))
    return res


def main():
    from common.collectives import resolve_mesh
    outdir, cases = sys.argv[1], json.loads(sys.argv[2])
    mesh = resolve_mesh()
    for case in cases:
        res = run_case(case, outdir, mesh)
        res["mesh"] = list(mesh.devices.shape)
        print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
