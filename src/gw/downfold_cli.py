"""``downfold`` — compress a finished GW calculation onto a smaller ISDF basis.

Run a GW calculation ONCE with thousands of bands and a large centroid set,
then do exciton bands and BSE work on a few-dozen-band subspace far more
cheaply.  This driver is the step in between: it reads the large run's
restart bundle, chooses a small centroid set against the bands you actually
intend to consume, refits the wavefunction-at-centroid coefficients onto that
smaller set, and writes a restart bundle in the SAME FORMAT at the smaller
size — so every BSE consumer downstream is a drop-in with no code change and
no flag.

    python3 -m gw.downfold_cli -i downfold.in

It takes its OWN input file, not a GW deck.  The six things it needs to be
told are where the parent calculation is, which bands must stay faithful, how
small the answer should be, how much round-off amplification is authorised,
and where to put the result; ``docs/downfold.md`` documents each key and
``gw/downfold_config.py`` is the schema itself.  Pointing it at a
``[cohsex]`` deck is refused by name, because a GW deck's hundred and forty
keys would all look like inputs to a compression none of them bears on.

WHAT IT PRINTS THAT YOU SHOULD READ
-----------------------------------
Three numbers, every run, and they are not interchangeable:

* the EIGENVALUE rank of the retained window's Gram — how many independent
  pair-density directions the window actually holds.  ``mu_small`` is
  validated against this one, and the run REFUSES when you ask for more.
* pivoted Cholesky's SELECTION certificate — a necessary but not sufficient
  condition, which runs about three times the eigenvalue rank at the same
  nominal tolerance.  A basis that passes the selection can still be two
  thirds rank-deficient at the solve.
* ``eps_W(q)``, the Pythagorean error bar — the relative error of the
  downfolded observable on the retained window, computed exactly from
  mu x mu traces, with no reference calculation anywhere.

If you read only one, read the third.
"""
from __future__ import annotations

from runtime import initialize_communicator_stack

RUNTIME = initialize_communicator_stack()

import argparse                                              # noqa: E402
import os                                                    # noqa: E402

from common import timing                                    # noqa: E402
from common.collectives import barrier, process_rank         # noqa: E402
from gw.downfold_config import (                             # noqa: E402
    DOWNFOLD_DEFAULTS, DownfoldConfig,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """The CLI.  Extracted from :func:`main` so a test can read its defaults.

    Deliberately thin.  Everything physical lives in the input file — the
    only flags here are the ones that describe the MACHINE (how many devices
    to lay the mesh over) or ask the driver to describe itself.  A knob that
    changes the answer belongs in the input file, where it is recorded
    beside the run, not in a shell history nobody keeps.
    """
    p = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Downfold a GW restart bundle onto a smaller ISDF basis.")
    p.add_argument("-i", "--input",
                   help="the downfold input file (a [downfold] section)")
    p.add_argument("--px", type=int, default=None,
                   help="mesh rows; default = the run's square startup mesh")
    p.add_argument("--py", type=int, default=None,
                   help="mesh columns; must equal --px (square meshes only)")
    p.add_argument("--print-schema", action="store_true",
                   help="list every input-file key with its default and exit")
    return p


def _print_schema(print_fn=print) -> None:
    print_fn("[downfold] input-file schema — section [downfold]")
    for key, default in DOWNFOLD_DEFAULTS.items():
        shown = ("<required, no default>" if default is None
                 else ("<empty>" if default == "" else repr(default)))
        print_fn(f"  {key:<24} {shown}")
    print_fn("Every key is documented in gw/downfold_config.py and in "
             "docs/downfold.md.  Unknown keys are REFUSED, not ignored.")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _pre_main = timing.process_elapsed_s()
    timing.reset()

    rank0 = process_rank() == 0

    def print0(*a, **kw):
        if rank0:
            print(*a, **kw)

    if args.print_schema:
        if rank0:
            _print_schema()
        return 0
    if not args.input:
        build_parser().error("-i/--input is required (a [downfold] file)")

    if _pre_main is not None:
        timing.record("downfold.imports_and_runtime", _pre_main)

    cfg = DownfoldConfig.from_input_file(args.input, print_fn=print0)

    mesh = RUNTIME.mesh
    if args.px is not None or args.py is not None:
        if args.px is None or args.py is None:
            build_parser().error("--px and --py must be given together")
        mesh = RUNTIME.reshape(int(args.px), int(args.py), print_fn=print0)

    print0("=" * 72)
    print0("LORRAX downfold — compress V and W onto a smaller ISDF basis")
    print0("=" * 72)
    print0(cfg.describe())
    print0(f"  mesh              {dict(mesh.shape)} on "
           f"{RUNTIME.n_devices} device(s), {RUNTIME.process_count} "
           f"process(es), {RUNTIME.device_kind}")
    print0("-" * 72)

    # Imported HERE, after the banner, because importing the pipeline pulls
    # in jax and every kernel behind it: a bad input file should be refused
    # before that cost, not after it.
    from gw.downfold_run import run_downfold

    result = run_downfold(cfg, mesh, print_fn=print0)

    print0("-" * 72)
    print0(f"  downfolded  mu {result.mu_large} -> {result.mu_small}  "
           f"over {result.n_q} q, band window "
           f"[{cfg.band_range_left[0]}, {cfg.band_range_left[1]}) x "
           f"[{cfg.band_range_right[0]}, {cfg.band_range_right[1]})")
    print0(f"  output      {result.output_file}")
    if result.zeta_file:
        print0(f"  zeta        {result.zeta_file}  (transported to the small "
               f"basis; bse.exciton_bands --vq-mode interp reads it)")
    if result.centroid_file:
        print0(f"  centroids   {result.centroid_file}  (kept rows, for a "
               f"fresh GW on the small basis)")
    print0("  point any BSE consumer at "
           f"{os.path.dirname(os.path.dirname(result.output_file))} — the "
           "bundle format is unchanged, so nothing downstream needs a flag.")
    if not result.parent_centroid_file:
        print0("  NOTE: no parent centroid table was resolved, so "
               "bse.exciton_bands cannot open this bundle — its htransform "
               "leg refits psi in the PARENT ISDF basis and needs those "
               "coordinates.  Set parent_centroids_file and re-run.")
    if rank0:
        timing.report(print_fn=print, title="--- downfold timing (s) ---")
    barrier("downfold.outputs_written")
    return 0


if __name__ == "__main__":
    from runtime import finalize_process

    finalize_process(main())
