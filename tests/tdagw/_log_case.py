"""Decks for the run-log test (child of ``test_td_log_cpu.py``).

    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 \\
        python3 tests/tdagw/_log_case.py OUTDIR

Writes, on ``_stiff_case.stiff_window`` (3v3c spinors, 3x3 k padded to 12, a
full kernel with one stiff outlier quartet; no Γ and no intraband term, so
the flow is Hamiltonian), the TDGW header ``OUTDIR/hdr``, ``OUTDIR/kernel.h5``
and one TOML deck per case, and prints ``RESULT {tag: deck path}``.  The
test runs ``python -m tdagw.tdgw_jax -i <deck>`` on each, the driver a user
runs:

split    cfm4, 4 stiff modes, dt = 'auto'       (gate auto + the stiff basis)
plain    cfm4, dt 35.78 as, n_corr unset        (gate at a fixed step: the plain auto step of
                                                 this window; 25 as fails its probe, 1.003)
dp54     dp54, rtol 1e-8                        (no gate; predictor/n_corr set, unused)
refused  cfm4, dt 50 as                         (above the static limit: td_magnus_stability)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _stiff_case as sc            # noqa: E402  (seals the source closure, x64)
import _validation_systems as vs    # noqa: E402
from tdagw.tdgw_io import dump_toml  # noqa: E402

CASES = {
    "split": dict(dt="auto", stiff=4),
    "plain": dict(dt=35.78),
    "dp54": dict(dt=None, method="dp54"),
    "refused": dict(dt=50.0),
}


def main():
    root = sys.argv[1]
    os.makedirs(root, exist_ok=True)
    s = sc.stiff_window()
    nb, n_val = int(s["e_ev"].shape[1]), int(s["n_val"])
    header = vs.write_header(os.path.join(root, "hdr"), s)
    kernel = vs.write_kernel(os.path.join(root, "kernel.h5"), s, tda=False)
    out = {}
    for tag, kw in CASES.items():
        raw = sc.deck(os.path.join(root, f"out_{tag}"), header, kernel, nb, n_val, tf=2400.0, **kw)
        lx = raw["lorrax"]
        lx["chunk"] = 6
        if tag == "plain":
            lx.pop("n_corr")                       # the default, marked so in the log
        if tag == "dp54":
            lx.pop("dt")
            raw["control"]["ivp"] = {"rtol": 1e-8, "atol": 1e-12}
        if not raw["control"]["files"]["load_checkpoint_from"]:
            raw["control"]["files"].pop("load_checkpoint_from")
        path = os.path.join(root, f"{tag}.toml")
        dump_toml(path, raw)
        out[tag] = path
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
