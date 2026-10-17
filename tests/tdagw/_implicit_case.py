"""Converged Gauss nodes on the TDGW fixture (child of test_prop_implicit_cpu.py).

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/tdagw/_implicit_case.py OUTDIR SCALE

The fixture's TDA kernel scaled by SCALE (the strong coupling of
_stability_case.py).  Prints one JSON line: the truncated step's sigma limit
dt_max; cfm4 with node_solver = anderson at 2 dt_max (where the truncated
step refuses) and at dt_max (refused by the truncated step's probe) through
the gate, receipts and counters; the truncated run at dt_max / 4 as the
reference replaced by the converged run at dt_max / 16; both schemes at
dt_max / 4; a converged
run with node_max_iter = 2 (the gate's probe refuses: its solves do not
converge); the same with the probe bypassed (the run itself refuses,
GATE td_node_solve, before the chunk is written: rows left in the file).
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


def _deck(outdir, dt, lorrax, tf=5000.0):
    meta = json.load(open(os.path.join(FIXTURE, "deck_tdgw.json")))
    raw = copy.deepcopy(meta["deck"])
    raw["control"]["files"]["indir"] = os.path.join(FIXTURE, "indata")
    raw["control"]["files"]["load_header_from"] = os.path.join(FIXTURE, "tdgw_ref")
    raw["control"]["files"]["outdir"] = outdir
    raw["control"]["timesteps"]["tf"] = tf
    raw["lorrax"] = dict({"method": "cfm4", "n_corr": 2, "dt": dt, "chunk": 10}, **lorrax)
    return raw


def main():
    import h5py
    from tdagw import propagate as tp, stages
    from tdagw.td_config import parse_td_deck
    from tdagw.tdgw_io import efield_function, T_IN_ATTOSEC
    from common.collectives import resolve_mesh
    outdir, scale = sys.argv[1], float(sys.argv[2])
    mesh = resolve_mesh()

    def build(dt, tag, lorrax):
        deck = parse_td_deck(_deck(os.path.join(outdir, tag), dt, lorrax), source=tag)
        system = stages.load_system(deck, mesh)
        model = stages.build_self_energy(deck, system, mesh)
        model = model._replace(params=(scale * model.params[0],) + tuple(model.params[1:]))
        return deck, system, model

    def run(dt, tag, lorrax=None):
        deck, system, model = build(dt, tag, lorrax or {})
        outs = stages.open_outputs(deck, system)
        lines = []
        try:
            st = stages.propagate(deck, system, model, efield_function(deck.trials), outs,
                                  mesh, say=lines.append)
        except ValueError as exc:
            with h5py.File(outs.paths[0], "r") as f:
                it = int(f["tdgw_data/checkpoint/it_next"][()]) if "tdgw_data/checkpoint/it_next" in f else -1
            return dict(refused=str(exc).split(":")[0], message=str(exc), it_next=it)
        with h5py.File(outs.paths[0], "r") as f:
            p = f["tdgw_data/time_domain/polarization/data"][()]
        return dict(dt_as=st["dt"] * T_IN_ATTOSEC, p=np.asarray(p[:, 0]).tolist() if False else None,
                    p_re=np.real(p[:, 0]).tolist(), p_im=np.imag(p[:, 0]).tolist(),
                    max_p=float(np.max(np.abs(p))), finite=bool(np.all(np.isfinite(p))),
                    probes=st["check_probes"], nodes=st.get("nodes"), dsigma=st["dsigma"],
                    steps=st["accepted"], receipt=[l for l in lines if "stability" in l])

    deck, system, model = build(1.0, "probe", {})
    sigma, _ = tp.coupling_radius(None, system.ops, model, efield_function(deck.trials), mesh, t0=0.0)
    cfg = tp.PropagatorConfig("cfm4", 0.0, 1.0, 1.0, 1.0, 2, 1e-6, 1e-6, np.inf, 10, True)
    dt_max = tp.magnus_dt_limit(cfg, sigma)[0] * T_IN_ATTOSEC
    conv = {"node_solver": "anderson", "node_tol": 1e-11}
    out = dict(dt_max_as=dt_max,
               trunc_over=run(2.0 * dt_max, "trunc_over"),
               conv_over=run(2.0 * dt_max, "conv_over", conv),
               conv_at=run(1.0 * dt_max, "conv_at", conv),
               ref=run(dt_max / 16.0, "ref", dict(conv, node_tol=1e-12)),
               trunc_q=run(0.25 * dt_max, "trunc_q"),
               conv_q=run(0.25 * dt_max, "conv_q", conv),
               conv_q_lab=run(0.25 * dt_max, "conv_q_lab", dict(conv, node_frame="lab")),
               conv_q_slow=run(0.25 * dt_max, "conv_q_slow", dict(conv, node_frame="slow",
                                                                    node_shift_ev=1.0)),
               starved=run(2.0 * dt_max, "starved", dict(conv, node_max_iter=2)))
    tp.step_growth = lambda *a, **k: ((1.0, np.ones(1), dict(n_eval=0, n_eval_max=0, n_fail=0,
                                                             dsigma=0))
                                      if k.get("with_counts") else (1.0, np.ones(1)))
    out["starved_unprobed"] = run(2.0 * dt_max, "starved_unprobed", dict(conv, node_max_iter=2))
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
