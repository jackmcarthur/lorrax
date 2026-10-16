"""``epsmat`` — BerkeleyGW ε⁻¹ files from a LORRAX GW bundle, and back.

    python3 -m gw.epsmat_cli -i epsmat.in

``mode = export`` writes ``eps0mat.h5`` / ``epsmat.h5`` (BerkeleyGW 4.0,
static, complex) from the bundle's V, W(0) and ζ, one q at a time; Sigma
and Kernel read them in place of their own epsilon run.  ``mode = project``
reads BerkeleyGW's files on the same WFN and writes a copy of the bundle
whose ``W0_qmunu`` is BerkeleyGW's screening projected on the bundle's ζ.
The equations are ``docs/theory/bgw_epsmat_export.md``; the keys are the
``[epsmat]`` section of ``docs/input_reference.md``.
"""
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """The CLI: the input file, the schema; every physical choice is in the file."""
    p = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Convert between LORRAX W_μν and BerkeleyGW eps^-1(G,G').")
    p.add_argument("-i", "--input", help="the [epsmat] input file")
    p.add_argument("--print-schema", action="store_true",
                   help="list every input-file key with its default and exit")
    return p


if __name__ == "__main__":
    # Argv is answered before any runtime exists — runtime/cli_seam.py.
    from runtime.cli_seam import refuse_bad_argv
    refuse_bad_argv(build_parser())

from runtime import initialize_communicator_stack

RUNTIME = initialize_communicator_stack()

from common import timing                                    # noqa: E402
from common.collectives import barrier, process_rank         # noqa: E402
from gw.epsmat_config import EPSMAT_DEFAULTS, EpsmatConfig   # noqa: E402

__all__ = ["build_parser", "main"]


def _json_safe(x):
    """Plain-JSON copy of a figures tree: complex as ``[re, im]``, numpy as Python."""
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in (x.tolist() if isinstance(x, np.ndarray) else x)]
    if isinstance(x, (complex, np.complexfloating)):
        return [float(x.real), float(x.imag)]
    if isinstance(x, np.generic):
        return x.item()
    return x


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    timing.reset()
    rank0 = process_rank() == 0

    def print0(*a, **kw):
        if rank0:
            print(*a, **kw, flush=True)

    if args.print_schema:
        for key, default in EPSMAT_DEFAULTS.items():
            print0(f"  {key:<18} {'<required>' if default is None else repr(default)}")
        return 0
    if not args.input:
        build_parser().error("-i/--input is required (an [epsmat] file)")
    cfg = EpsmatConfig.from_input_file(args.input)
    mesh = RUNTIME.mesh
    print0("=" * 72)
    print0(f"LORRAX epsmat — {cfg.mode}: "
           + ("W_μν -> BerkeleyGW eps^-1(G,G')" if cfg.mode == "export"
              else "BerkeleyGW eps^-1(G,G') -> W_μν"))
    print0("=" * 72)
    print0(cfg.describe())
    print0(f"  mesh               {dict(mesh.shape)} on {RUNTIME.n_devices} device(s)")
    print0("-" * 72)

    from gw.epsmat_export import run_epsmat_export, run_epsmat_projection

    with timing.section(f"epsmat.{cfg.mode}"):
        if cfg.mode == "export":
            result = run_epsmat_export(cfg, mesh, print_fn=print0)
            for name, r in result.items():
                worst = max(q["projection_err"] for q in r["q"])
                print0(f"  wrote {r['path']}: {len(r['q'])} q, ecuts {r['ecuts']} Ry, "
                       f"worst block identity {worst:.2e}")
        else:
            result = run_epsmat_projection(cfg, mesh, print_fn=print0)
            worst = max(q["rel_Wc_bgw_vs_lorrax"] for q in result["q"])
            print0(f"  wrote {result['path']}: W0_qmunu from BerkeleyGW on "
                   f"{len(result['q'])} stored q; max |Wc_BGW − Wc_LX|/|Wc_LX| = {worst:.4e}")
    if rank0:
        import json
        import os
        receipt = os.path.join(cfg.output_dir, f"epsmat_{cfg.mode}.json")
        with open(receipt, "w") as f:
            json.dump(_json_safe({"mode": cfg.mode, "config": cfg.describe().splitlines(),
                                  "result": result}), f, indent=1)
        print0(f"  receipt {receipt}")
        timing.report(print_fn=print, title="--- epsmat timing (s) ---")
    barrier("epsmat.outputs_written")
    return 0


if __name__ == "__main__":
    from runtime import finalize_process

    finalize_process(main())
