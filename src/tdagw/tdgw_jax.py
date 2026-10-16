"""tdagw/tdgw_jax.py — real-time TD-aGW propagation, TDGW-compatible.

    python -m tdagw.tdgw_jax -i tdgw.toml

Reads a TDGW deck (``tdgw.toml``; the ``[lorrax]`` table picks integrator,
step and self-energy model, ``docs/input_reference.md``), takes the header
from the TDGW run named by ``load_header_from``, propagates

    d rho/dt = -i([H(t), rho] + E·[r^intra, rho]) - Gamma∘(rho - rho^eq),
    H(t) = diag(E^QP) + E(t)·r^inter + dSigma[rho(t)]

and writes ``tdgw.h5`` + ``tdgw.toml`` that ``tdgw-pp`` reads unchanged
(one directory per batched drive); ``load_checkpoint_from`` continues a
run from its last checkpoint, bitwise.  Theory and integrators:
``docs/theory/td_agw_propagation.md``.
"""
from __future__ import annotations

import argparse


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m tdagw.tdgw_jax", allow_abbrev=False,
        description="Real-time TD-aGW propagation in TDGW's conventions")
    p.add_argument("-i", "--input", required=True,
                   help="TDGW deck (TOML) with an optional [lorrax] table")
    return p


if __name__ == "__main__":
    from runtime.cli_seam import refuse_bad_argv
    refuse_bad_argv(build_parser())

from runtime import debug_print, initialize_communicator_stack
RUNTIME = initialize_communicator_stack(print_fn=debug_print)

from runtime.run_session import RunSession                       # noqa: E402
from tdagw import stages                                           # noqa: E402
from tdagw.td_config import read_td_deck                           # noqa: E402
from tdagw.tdgw_io import efield_function                          # noqa: E402

STAGES = (("propagation", "tdagw.propagate"),)


def main(argv=None):
    args = build_parser().parse_args(argv)
    deck = read_td_deck(args.input)
    with RunSession(RUNTIME, "tdagw", None, stages=STAGES) as run:
        system = stages.load_system(deck, RUNTIME.mesh)          # header, Eq. (1)-(2) operators
        model = stages.build_self_energy(deck, system, RUNTIME.mesh)  # dSigma[rho]
        drive = efield_function(deck.trials)                     # E(t), one row per drive
        outs = stages.open_outputs(deck, system)                 # tdgw.h5 + tdgw.toml
        run.emit(stages.describe(deck, system, model, RUNTIME))
        stats = stages.propagate(deck, system, model, drive, outs,
                                 RUNTIME.mesh, say=run.emit)     # from t0 or the checkpoint
        run.emit(stages.summarize(deck, stats))
        run.complete(files=[("tdgw.h5", "written", p) for p in outs.paths])
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
