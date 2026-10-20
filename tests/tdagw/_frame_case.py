"""magnus2's two frames on the TDGW fixture: the post-pulse energy (child of test_prop_magnus_frame_cpu.py).

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/tdagw/_frame_case.py OUTDIR

The fixture with Gamma = 0 and its 5e8 V/m pulse (over by 400 as), magnus2 (2 corrections) at
5 and 2.5 as in the whole_rho and the interaction frames, the whole density matrix sampled.
After the pulse the dynamics conserves E = sum_k w_k [Tr(E^QP_k rho_k) + (1/2) Tr(dSigma_k
Delta rho_k)] (dSigma linear in Delta rho through a Hermitian kernel).  Prints one JSON line:
per frame and step, the largest |E(t) - E(400 as)| over 400-4000 as relative to the excitation
energy E(400 as) - E(rho_eq), and the default frame's run.
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


def _deck(outdir, dt, frame, scale):
    meta = json.load(open(os.path.join(FIXTURE, "deck_tdgw.json")))
    raw = copy.deepcopy(meta["deck"])
    raw["control"]["files"]["indir"] = os.path.join(FIXTURE, "indata")
    raw["control"]["files"]["load_header_from"] = os.path.join(FIXTURE, "tdgw_ref")
    raw["control"]["files"]["outdir"] = outdir
    raw["control"]["timesteps"]["tf"] = 4000.0
    raw["system"]["depopulation"] = 0.0
    raw["system"]["dephasing"] = 0.0
    for key in ("E_x", "E_y"):
        raw["system"]["Efunc"][key] = f"{scale:g} * ({raw['system']['Efunc'][key]})"
    raw["measurements"] = {"types": ["density_matrix"],
                           "0": {"dataset_name": "rho", "brange1": [5, 9], "brange2": [5, 9]}}
    raw["lorrax"] = {"method": "magnus2", "n_corr": 2, "dt": dt}
    if frame:
        raw["lorrax"]["magnus_frame"] = frame
    return raw


def main():
    import h5py
    from common.collectives import resolve_mesh
    from tdagw import propagate as tp, stages
    from tdagw.td_config import parse_td_deck
    from tdagw.tdgw_io import efield_function, T_IN_ATTOSEC
    outdir, scale = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    mesh = resolve_mesh()
    out = {}
    for frame in ("whole_rho", "interaction", ""):
        for dt in ((5.0, 2.5) if frame else (5.0,)):
            tag = f"{frame or 'default'}_{dt:g}"
            deck = parse_td_deck(_deck(os.path.join(outdir, tag), dt, frame, scale), source=tag)
            system = stages.load_system(deck, mesh)
            model = stages.build_self_energy(deck, system, mesh)
            outs = stages.open_outputs(deck, system)
            stages.propagate(deck, system, model, efield_function(deck.trials), outs, mesh,
                             say=lambda s: None)
            with h5py.File(outs.paths[0], "r") as f:
                t = f["tdgw_data/time_domain/rho/times"][()] * T_IN_ATTOSEC
                rho = f["tdgw_data/time_domain/rho/data"][()]          # (nt, nk, nb, nb)
            ops = system.ops
            nk = system.nk
            w = np.asarray(tp.gather_to_host(ops.w_k))[:nk]
            e = np.asarray(tp.gather_to_host(ops.e_qp))[:nk]
            req = np.asarray(tp.gather_to_host(ops.rho_eq))[:nk]
            nk_pad = int(ops.e_qp.shape[0])

            def energy(r):
                pad = np.zeros((1, nk_pad) + r.shape[1:], complex)
                pad[0, :nk] = r
                ds = np.asarray(tp.gather_to_host(model.apply(model.params, tp.place_state(pad, mesh), 0.0)))[0, :nk]
                band = np.einsum("k,kn,knn->", w, e, r).real
                inter = 0.5 * np.einsum("k,knm,kmn->", w, ds, r - req).real
                return band + inter
            sel = np.nonzero(t >= 400.0 - 1e-9)[0]
            es = np.array([energy(rho[i]) for i in sel])
            e_eq = float(np.einsum("k,kn,knn->", w, e, req).real)
            out[tag] = dict(frame=frame or "default", dt_as=dt,
                            drift=float(np.max(np.abs(es - es[0])) / abs(es[0] - e_eq)),
                            excitation=float(es[0] - e_eq))
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
