"""Exact restart of the TD-aGW propagator on the TDGW fixture (child of test_prop_restart_cpu.py).

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/tdagw/_restart_case.py OUTDIR '[{"tag": "cfm4", "method": "cfm4", "dt": 2.5}]'

Per case: run A uninterrupted; run B interrupted after its first chunk (the
writer has written that chunk's checkpoint, then the run is stopped); run C
restarted from B into a new outdir.  Prints one JSON line per case with the
largest |A - C| over every time-domain dataset and the final checkpoint.
A case may set ``twin`` = "zero_history" (cfm4/magnus2: the dSigma history
of the carry zeroed) or "fresh_controller" (dp54: the controller restarted
from the state alone, as TDGW restarts), and ``mutate`` = a deck key path
and value applied to C only (a settings mismatch); ``from_tdgw`` restarts
from the fixture's TDGW output (TDGW's checkpoint, no LORRAX carry).
"""
import copy
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


class _Stop(Exception):
    pass


def _deck(case, outdir):
    meta = json.load(open(os.path.join(FIXTURE, "deck_tdgw.json")))
    raw = copy.deepcopy(meta["deck"])
    raw["control"]["files"]["indir"] = os.path.join(FIXTURE, "indata")
    raw["control"]["files"]["load_header_from"] = os.path.join(FIXTURE, "tdgw_ref")
    raw["control"]["files"]["outdir"] = outdir
    raw["lorrax"] = {"method": case["method"], "n_corr": 2, "chunk": 20}
    if case.get("dt") is not None:
        raw["lorrax"]["dt"] = case["dt"]
    raw["lorrax"].update(case.get("lorrax", {}))       # e.g. node_solver = anderson
    return raw


def _run(raw, mesh, *, stop_after=None):
    from tdagw import stages
    from tdagw.td_config import parse_td_deck
    from tdagw.tdgw_io import efield_function
    deck = parse_td_deck(raw, source="restart_case")
    system = stages.load_system(deck, mesh)
    model = stages.build_self_energy(deck, system, mesh)
    outs = stages.open_outputs(deck, system)
    seen = []

    def say(s):                                     # stop after the stop_after-th chunk
        if not s.lstrip().startswith("samples"):    # (the stability receipt is not one)
            return
        seen.append(s)
        if stop_after is not None and len(seen) >= stop_after:
            raise _Stop()
    try:
        stats = stages.propagate(deck, system, model, efield_function(deck.trials), outs,
                                 mesh, say=say)
    except _Stop:
        stats = None
    src = raw["control"]["files"].get("load_checkpoint_from", "")
    resume = dict(i_next=-1) if not src else stages.tdgw_io.read_checkpoint(
        os.path.join(src, "tdgw.h5"))
    return outs.paths[0], stats, dict(i_next=resume.get("i_next", resume.get("it_next")))


def _twin(path, kind):
    """Corrupt the carry of a checkpoint the way a non-exact restart would."""
    with h5py.File(path, "a") as f:
        g = f["tdgw_data/checkpoint/lorrax"]
        if kind == "zero_history":
            g["leaf_001"][...] = 0.0                   # (x, history, k): the history
        elif kind == "fresh_controller":
            # Dp54State leaves: t, y, f, h_abs, t_old, h_last, y_old, ks x7, counters, failed
            g["leaf_003"][...] = 50.0 / 24.18884326585747   # h_abs: dt_measure, a fresh start
            g["leaf_005"][...] = 0.0                    # h_last 0: no interpolant
            for i in range(7, 14):
                g[f"leaf_{i:03d}"][...] = 0.0           # ks


def _compare(pa, pc):
    out = {}
    with h5py.File(pa, "r") as a, h5py.File(pc, "r") as c:
        for name in a["tdgw_data/time_domain"]:
            x = a["tdgw_data/time_domain"][name]["data"][()]
            y = c["tdgw_data/time_domain"][name]["data"][()]
            out[name] = float(np.max(np.abs(x - y)))
        ca, cc = a["tdgw_data/checkpoint"], c["tdgw_data/checkpoint"]
        out["lessG"] = float(np.max(np.abs(ca["lessG_knn"][()] - cc["lessG_knn"][()])))
        out["it_next"] = [int(ca["it_next"][()]), int(cc["it_next"][()])]
        out["tchkp"] = [float(ca["tchkp"][()]), float(cc["tchkp"][()])]
        # TDGW's reader (BaseData._load_data) reads exactly these: present and well formed
        hdr = c["tdgw_header"]
        out["tdgw_layout"] = bool(all(k in hdr for k in ("version", "tgrid", "tsteps", "structure",
                                                         "wfn_header", "dipole"))
                                  and all(k in cc for k in ("it_next", "tchkp", "lessG_knn"))
                                  and cc["lessG_knn"].shape == ca["lessG_knn"].shape)
    return out


def main():
    from common.collectives import resolve_mesh
    outdir, cases = sys.argv[1], json.loads(sys.argv[2])
    mesh = resolve_mesh()
    for case in cases:
        tag = case["tag"]
        d = {k: os.path.join(outdir, f"{tag}_{k}") for k in "ABC"}
        res = dict(tag=tag, mesh=list(mesh.devices.shape))
        try:
            if case.get("from_tdgw"):            # TDGW's own file: a state, no carry
                rc = _deck(case, d["C"])
                rc["control"]["files"]["load_checkpoint_from"] = os.path.join(FIXTURE, "tdgw_ref")
                _run(rc, mesh)
            pa, sa, _ = _run(_deck(case, d["A"]), mesh)
            pb, _, _ = _run(_deck(case, d["B"]), mesh, stop_after=1)
            if case.get("twin"):
                _twin(pb, case["twin"])
            rc = _deck(case, d["C"])
            rc["control"]["files"]["load_checkpoint_from"] = d["B"]
            if case.get("mutate"):
                key, val = case["mutate"]
                node = rc
                *path, last = key.split(".")
                for p_ in path:
                    node = node[p_]
                node[last] = val
            pc, sc, resume = _run(rc, mesh)
            res.update(_compare(pa, pc), resumed_at=int(resume["i_next"]),
                       steps_a=int(sa["accepted"]), steps_c=int(sc["accepted"]))
        except ValueError as exc:
            res["refused"] = str(exc).split(":")[0]
        print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
