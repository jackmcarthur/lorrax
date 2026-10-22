"""``python -m bse.bsemat_cli``: LORRAX static W -> BerkeleyGW 4.0 bsemat.h5.

    python3 -u -m bse.bsemat_cli -i cohsex.in --wfn-co WFN_co.h5 \\
        [--bgw-eps epsmat.h5] [--gauge map identity] \\
        [--q0-body head-less dipole] [--mu-small off|N|Nx|auto ...] [--downfold-rcond R] \\
        [--wfnq-co WFNq_co.h5] [-o bsemat.h5]

The kernel's settings are kernel.x's keys in the deck: ``number_val_bands``,
``number_cond_bands`` (required), ``extended_kernel`` (default false: the TDA
blocks) and ``exciton_Q_shift Qflag Qx Qy Qz`` (default ``1 0 0 0``).

``-i`` is the producer GW deck. Its run directory holds the restart
(``tmp/isdf_tensors_*.h5``: a static-Dyson ``W0_qmunu``, ``V_qmunu`` and, beside
it, ``zeta_q.h5``). ``--wfn-co`` is the explicit full-grid coarse WFN that
BerkeleyGW's absorption reads (``no_symmetries_coarse_grid``); the kernel is
written in its coefficient gauge and on its k order. ``--bgw-eps`` is the
``epsmat.h5`` whose q list BerkeleyGW's kernel would fold. It fixes the head
slot at zone-boundary ties exactly as ``kernel.x`` does; without it the
restart's q wedge stands in. ``--gauge identity`` (``--no-gauge-map``) writes
LORRAX's own Bloch gauge, the red twin of the gauge test. ``--q0-body dipole`` adds
BerkeleyGW's 3-D q0 body term from the bundle's stored q → 0 wings
(``head_wing_L``/``R``, ``S_cart_head``, ``S_cart_head_direct``). ``--mu-small``
(default ``off``) first downfolds the restart onto μ_S of its centroids against
the file's own band window (the ``gw.downfold`` driver's selection and transfer,
docs/downfold.md) and builds the rows by GEMMs over the small W.

``exciton_Q_shift Qflag Qx Qy Qz`` (crystal) writes kernel.x's finite-momentum
kernel on the same coarse k count, with kernel.x's sign: Q = -(Qx, Qy, Qz) is the
exciton momentum, and file row k holds the electron at k (WFN_co) and the hole at
k − Q, the pair |v k−Q, c k⟩ of ``bse.exciton_bands``' |v k, c k+Q⟩. Qflag 1 is
Q = 0. Qflag 2: Q on the restart grid, both legs the restart's. Qflag 0: off the
grid, WFN_co is the restart grid shifted by +Q, its states come from the
htransform, the hole gauge from ``--wfnq-co`` and the exchange tile from the per-Q
ζ refit. docs/theory/bgw-bsemat-export.md#finite-q.

Several files from one load: ``--gauge``, ``--q0-body`` and ``--mu-small`` each
take one or more values, and the run writes one file per combination
(``bse.bsemat_export.run_bsemat_exports``):
the restart is read, the head split solved and the gauge fitted once, and the row
kernels compile once per file shape. One combination writes ``-o``; several write
``<stem>_<tag><ext>``, the tag naming each varying value (``map``/``nomap``, ``q0hl``/``q0dip``, ``mu<value>``). Each file gets
``<file>.json`` (gates, Frobenius norms and ``stages``, the per-stage walls in s);
rank 0 prints the stage table (``common.timing``) at the end.
Physics and conventions:
docs/theory/bgw-bsemat-export.md.
"""
import argparse
import itertools
import os

#: File-name tag of each multi-valued flag value (``--mu-small`` values: ``mu<value>``).
TAGS = {"tda": "tda", "extended": "ext", "map": "map", "identity": "nomap",
        "head-less": "q0hl", "dipole": "q0dip"}


def build_parser():
    ap = argparse.ArgumentParser(prog="bse.bsemat_cli", allow_abbrev=False,
                                 description="LORRAX static W -> BerkeleyGW 4.0 bsemat.h5")
    ap.add_argument("-i", "--input", required=True, help="producer GW deck of the restart")
    ap.add_argument("--wfn-co", required=True,
                    help="explicit full-grid coarse WFN (BerkeleyGW WFN_co)")
    ap.add_argument("--bgw-eps", default=None,
                    help="BerkeleyGW epsmat.h5 whose q list fixes the head slot of zone-boundary ties")
    ap.add_argument("--gauge", nargs="+", choices=("map", "identity"), default=None,
                    help="psi gauge(s): map = the coarse WFN's coefficient gauge (default); identity = "
                         "LORRAX's Bloch gauge (red twin of the gauge test)")
    ap.add_argument("--no-gauge-map", action="store_true", help="= --gauge identity")
    ap.add_argument("--q0-body", nargs="+", choices=("head-less", "dipole"), default=["head-less"],
                    help="q = 0 body(ies): head-less Γ W, or + the 3-D dipole term from the bundle's stored wings")
    ap.add_argument("--q0-vector", type=float, nargs=3, default=None, metavar=("Q1", "Q2", "Q3"),
                    help="BerkeleyGW q0 (crystal) for --q0-body dipole; default: eps0mat.h5 beside --bgw-eps")
    ap.add_argument("--mu-small", nargs="+", default=["off"],
                    help="one or more downfold requests, each run on the file's band window before the "
                         "rows: an integer, 'kx' = k x N_b with N_b = n_val + n_cond, 'auto' (the "
                         "eigenvalue-rank ceiling), or 'off' (default: rows on the parent's centroids; "
                         "docs/theory/bgw-bsemat-export.md#downfold has the measured accuracy)")
    ap.add_argument("--downfold-rcond", type=float, default=None,
                    help="downfold transfer cut, relative eigenvalue (default: gw.downfold's 1.1e-6)")
    ap.add_argument("--wfnq-co", default=None,
                    help="exciton_Q_shift Qflag 0 only: the restart grid's explicit WFN, BerkeleyGW's "
                         "WFNq_co (the hole leg's coefficient gauge)")
    ap.add_argument("-o", "--output", default="bsemat.h5",
                    help="output file; with several combinations, the <stem><ext> of <stem>_<tag><ext>")
    ap.add_argument("--summary", default=None,
                    help="JSON summary path of a one-file run (default: <output>.json); with several "
                         "files each gets <file>.json and this path the list of them")
    return ap


def kernel_settings(ap, a):
    """kernel.x's deck keys onto ``a``: ``n_val``, ``n_cond``, ``extended_kernel``,
    ``exciton_Q`` (the exciton momentum, ``-(Qx, Qy, Qz)`` of ``exciton_Q_shift``)."""
    from .bse_deck import DeckError, Key, deck_params, read_settings
    keys = {"number_val_bands": Key("n_val", "int"),
            "number_cond_bands": Key("n_cond", "int"),
            "extended_kernel": Key("extended_kernel", "bool"),
            "exciton_Q_shift": Key("exciton_Q_shift", "floats")}
    defaults = argparse.Namespace(n_val=None, n_cond=None, extended_kernel=False,
                                  exciton_Q_shift=[1.0, 0.0, 0.0, 0.0])
    try:
        params, named = deck_params(a.input)
        settings, _ = read_settings(a.input, defaults, keys,
                                    required=("number_val_bands", "number_cond_bands"),
                                    params=params, named=named)
    except DeckError as exc:
        ap.error(str(exc))
    shift = settings.exciton_Q_shift
    if len(shift) != 4 or shift[0] not in (0.0, 1.0, 2.0):
        ap.error(f"exciton_Q_shift takes Qflag Qx Qy Qz with Qflag 0, 1 or 2; got {shift}")
    qflag, q_shift = int(shift[0]), [float(x) for x in shift[1:]]
    if qflag == 1 and any(q_shift):
        ap.error("exciton_Q_shift Qflag 1 is the Q = 0 kernel; give Qx Qy Qz = 0 0 0")
    if qflag == 0 and a.wfnq_co is None:
        ap.error("exciton_Q_shift Qflag 0 reads the hole leg's gauge from WFNq_co: pass --wfnq-co")
    if qflag != 0 and a.wfnq_co is not None:
        ap.error("--wfnq-co is read by exciton_Q_shift Qflag 0 only")
    a.n_val, a.n_cond = settings.n_val, settings.n_cond
    a.extended_kernel = settings.extended_kernel
    a.exciton_Q = [-x + 0.0 for x in q_shift]
    a.exciton_qflag = qflag


def mu_small_of(spec, n_b):
    """``--mu-small`` -> the downfold request: None (off), "auto", or a point count."""
    t = str(spec).strip().lower()
    if t == "off":
        return None
    if t == "auto":
        return "auto"
    if t.endswith("x"):
        return int(round(float(t[:-1]) * n_b))
    return int(t)


def file_combinations(ap, a):
    """The files of one run, one dict per combination of the multi-valued flags (no runtime).

    Axes: the kernel form (``extended_kernel``, one value), ``--gauge`` (or
    ``--no-gauge-map``), ``--q0-body``, ``--mu-small``. One combination writes
    ``-o``; several write ``<stem>_<tag><ext>``, the tag naming each axis that
    varies (``TAGS``).  Contradictory or repeated values end the process through
    ``ap.error``.
    """
    if a.no_gauge_map and a.gauge:
        ap.error("--no-gauge-map is --gauge identity; give one of them")
    axes = (["extended"] if getattr(a, "extended_kernel", False) else ["tda"],
            a.gauge or (["identity"] if a.no_gauge_map else ["map"]),
            list(a.q0_body), list(a.mu_small))
    for flag, vals in zip(("extended_kernel", "--gauge", "--q0-body", "--mu-small"), axes):
        if len({str(v).strip().lower() for v in vals}) != len(vals):
            ap.error(f"{flag} repeats a value: {vals}")
    for v in a.mu_small:
        try:
            mu_small_of(v, 1)
        except ValueError:
            ap.error(f"--mu-small {v!r}: want off, auto, an integer or <number>x")
    stem, ext = os.path.splitext(a.output)
    combos = list(itertools.product(*axes))
    files = []
    for kernel, gauge, q0_body, mu in combos:
        tag = "_".join(TAGS[v] if i < 3 else f"mu{str(v).strip().lower()}"
                       for i, (v, vals) in enumerate(zip((kernel, gauge, q0_body, mu), axes))
                       if len(vals) > 1)
        files.append({"path": a.output if len(combos) == 1 else f"{stem}_{tag}{ext}",
                      "kernel": kernel, "gauge": gauge, "q0_body": q0_body, "mu_small": mu})
    return files


if __name__ == "__main__":
    # Argv is answered before any runtime exists — runtime/cli_seam.py.
    from runtime.cli_seam import refuse_bad_argv
    refuse_bad_argv(build_parser())
    _ap = build_parser()
    file_combinations(_ap, _ap.parse_args())

from runtime import debug_print, initialize_communicator_stack  # noqa: E402

RUNTIME = initialize_communicator_stack(print_fn=debug_print)

import json  # noqa: E402
import time  # noqa: E402

import jax   # noqa: E402

import common.timing as timing  # noqa: E402
from common.log import get_logger  # noqa: E402
from runtime.run_session import _record_pre_main  # noqa: E402

log = get_logger(__name__)


def main(argv=None) -> int:
    _t_main = time.perf_counter()
    _pre_main = timing.process_elapsed_s()
    timing.reset()
    _record_pre_main(RUNTIME, "bsemat", _pre_main)    # bsemat.runtime_stack.<phase>, bsemat.imports
    ap = build_parser()
    a = ap.parse_args(argv)
    kernel_settings(ap, a)
    files = file_combinations(ap, a)
    from .bse_ring_comm import create_mesh_2d
    from .bsemat_export import output_spec, run_bsemat_exports
    outputs = [output_spec(f["path"], extended=f["kernel"] == "extended",
                           use_gauge_map=f["gauge"] == "map",
                           q0_body=f["q0_body"].replace("-", "_"), q0_vector=a.q0_vector,
                           mu_small=mu_small_of(f["mu_small"], a.n_val + a.n_cond),
                           downfold_rcond=a.downfold_rcond)
               for f in files]
    ss = run_bsemat_exports(input_file=a.input, wfn_co=a.wfn_co, n_val=a.n_val, n_cond=a.n_cond,
                            bgw_eps=a.bgw_eps, outputs=outputs, exciton_Q=a.exciton_Q,
                            wfnq_co=a.wfnq_co, mesh_xy=create_mesh_2d())
    wall = time.perf_counter() - _t_main + (_pre_main or 0.0)
    startup = {r["name"][len("bsemat."):]: float(r["inclusive"]) for r in timing.records()
               if r["depth"] == 0 and (r["name"] == "bsemat.imports"
                                       or r["name"].startswith("bsemat.runtime_stack."))}
    if jax.process_index() == 0:
        timing.report(print_fn=log.info, title="--- bsemat timing (s) ---", max_depth=1, wall=wall)
        for s in ss:
            s["stages"]["startup"] = startup
            s["wall_process_s"] = wall
            path = a.summary if a.summary and len(ss) == 1 else os.path.splitext(s["out_path"])[0] + ".json"
            with open(path, "w") as fh:
                json.dump(s, fh, indent=1, default=str)
            log.info(f"[bsemat] wrote {s['out_path']} in {s['wall_total_s']:.1f} s from the load "
                     f"(own stages {sum(s['stages']['output'].values()):.1f} s, rows "
                     f"{s['wall_rows_s']:.1f} s, peak device {s['peak_device_bytes'] / 2**30:.2f} GiB)")
        if a.summary and len(ss) > 1:
            with open(a.summary, "w") as fh:
                json.dump(ss, fh, indent=1, default=str)
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
