"""The Magnus stability gate without coupling (child of test_prop_stability_cpu.py).

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/tdagw/_zero_coupling_case.py OUTDIR

The TDGW fixture's kernel scaled by 0 (dSigma = 0: an IP run, TDVAL's Rabi decks) and by
NaN (a broken model), cfm4 in the interaction frame at an explicit 20 as.  Prints one JSON
line: the power iteration's radius and estimates on the zero map; the zero-map run (the
gate accepts it without a probe and says so); the NaN-map run (refuses by name); and the
no-probe path (a finite radius with a NaN sigma limit: refuses by name, not IndexError).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _stability_case import _deck  # noqa: E402  (also enables x64 and the source closure)


def main():
    import h5py
    from common.collectives import resolve_mesh
    from tdagw import propagate as tp, stages
    from tdagw.td_config import parse_td_deck
    from tdagw.tdgw_io import efield_function, T_IN_ATTOSEC
    outdir = sys.argv[1]
    mesh = resolve_mesh()

    def build(tag, scale):
        deck = parse_td_deck(_deck(os.path.join(outdir, tag), 20.0, tf=2000.0), source=tag)
        system = stages.load_system(deck, mesh)
        model = stages.build_self_energy(deck, system, mesh)
        model = model._replace(params=(scale * model.params[0],) + tuple(model.params[1:]))
        return deck, system, model

    def run(tag, scale):
        deck, system, model = build(tag, scale)
        outs = stages.open_outputs(deck, system)
        log = []
        try:
            st = stages.propagate(deck, system, model, efield_function(deck.trials), outs,
                                  mesh, say=log.append)
        except ValueError as exc:
            return dict(refused=str(exc).split(":")[0], message=str(exc))
        with h5py.File(outs.paths[0], "r") as f:
            p = f["tdgw_data/time_domain/polarization/data"][()]
        return dict(dt_as=st["dt"] * T_IN_ATTOSEC, max_p=float(np.max(np.abs(p))),
                    finite=bool(np.all(np.isfinite(p))), probes=st["check_probes"],
                    log=[l for l in log if "stability" in l])

    deck, system, model = build("radius", 0.0)
    sigma, est = tp.coupling_radius(None, system.ops, model, efield_function(deck.trials),
                                    mesh, t0=0.0)
    out = dict(sigma=sigma, est_finite=bool(np.all(np.isfinite(est))), est_max=float(np.max(est)),
               ip=run("ip", 0.0), nan=run("nan", np.nan))
    tp.magnus_dt_limit = lambda cfg, sigma: (np.nan, np.nan)   # no usable sigma limit
    out["no_probe"] = run("no_probe", 1.0)
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
