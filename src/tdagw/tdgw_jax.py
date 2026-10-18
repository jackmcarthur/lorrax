"""tdagw/tdgw_jax.py — real-time TD-aGW propagation, TDGW-compatible.

    python -m tdagw.tdgw_jax -i tdgw.toml

Reads a TDGW deck (``tdgw.toml``; the ``[lorrax]`` table picks integrator,
step and self-energy model, ``docs/input_reference.md``), takes the header
from the TDGW run named by ``load_header_from``, propagates

    d rho/dt = -i([H(t), rho] + E·[r^intra, rho]) - Gamma∘(rho - rho^eq) [+ C[rho]],
    H(t) = diag(E^QP) + E(t)·r^inter + dSigma[rho(t)]

(``[lorrax] collision``: the on-shell GW collision term of the GN-PPM bundle,
whose poles the GW driver recovers in this process before the TD run,
``docs/theory/td_agw_dynamical.md`` §1.5)

and writes ``tdgw.h5`` + ``tdgw.toml`` that ``tdgw-pp`` reads unchanged
(one directory per batched drive); ``load_checkpoint_from`` continues a
run from its last checkpoint, bitwise.  Process 0 writes the run log
``outdir/tdagw.out`` (``tdagw.td_log``): stdout's lines plus provenance,
the resolved deck, the stability gate and a progress table.  Theory and
integrators: ``docs/theory/td_agw_propagation.md``; running it:
``docs/how-to/td-agw-run.md``.
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
from tdagw.td_log import TdReport, log_path                        # noqa: E402
from tdagw.tdgw_io import efield_function                          # noqa: E402

STAGES = (("TD system", "tdagw.system"), ("self-energy model", "tdagw.self_energy"),
          ("outputs", "tdagw.outputs"), ("stiff basis", "tdagw.stiff_basis"),
          ("stability gate", "tdagw.stability_gate"), ("propagation", "tdagw.propagate"))
#: The collision term's t0 build (``[lorrax] collision``); the GW driver run that
#: recovers its poles precedes the session and has its own log.
COLLISION_STAGE = ("collision term", "tdagw.collision")


def _deck_and_source(path):
    """The deck and its ``[lorrax] collision`` source (None when off).

    The source is the GN-PPM GW run's own poles: the GW driver runs here, in
    this process, before the TD run's session (``stages.collision_source``).
    """
    deck = read_td_deck(path)
    return deck, stages.collision_source(deck, RUNTIME.mesh)


def _stage_rows(source):
    """The stage table: STAGES, with the collision term's build when it is on."""
    return STAGES if source is None else STAGES[:-1] + (COLLISION_STAGE, STAGES[-1])


def main(argv=None):
    args = build_parser().parse_args(argv)
    deck, source = _deck_and_source(args.input)                  # [lorrax] collision: GW poles
    with RunSession(RUNTIME, "tdagw", TdReport, log_path(deck), stages=_stage_rows(source),
                    deck=deck) as run:                           # the log: outdir/tdagw.out
        system = stages.load_system(deck, RUNTIME.mesh)          # header, Eq. (1)-(2) operators
        model = stages.build_self_energy(deck, system, RUNTIME.mesh)  # dSigma[rho]
        drive = efield_function(deck.trials)                     # E(t), one row per drive
        outs = stages.open_outputs(deck, system)                 # tdgw.h5 + tdgw.toml
        run.report.header(deck, system, model, RUNTIME)          # stdout: stages.describe
        stats = stages.propagate(deck, system, model, drive, outs, RUNTIME.mesh,
                                 say=run.report.echo, log=run.report,  # from t0 or the checkpoint
                                 collision=source)
        run.report.summary(deck, stats)                          # stdout: stages.summarize
        run.complete(files=stages.output_files(outs))
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
