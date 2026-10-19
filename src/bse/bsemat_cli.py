"""``python -m bse.bsemat_cli``: LORRAX static W -> BerkeleyGW 4.0 bsemat.h5.

    python3 -u -m bse.bsemat_cli -i cohsex.in --wfn-co WFN_co.h5 --n-val 8 --n-cond 8 \\
        [--extended] [--bgw-eps epsmat.h5] [--no-gauge-map] [--q0-body dipole] [-o bsemat.h5]

``-i`` is the producer GW deck. Its run directory holds the restart
(``tmp/isdf_tensors_*.h5``: a static-Dyson ``W0_qmunu``, ``V_qmunu`` and, beside
it, ``zeta_q.h5``). ``--wfn-co`` is the explicit full-grid coarse WFN that
BerkeleyGW's absorption reads (``no_symmetries_coarse_grid``); the kernel is
written in its coefficient gauge and on its k order. ``--bgw-eps`` is the
``epsmat.h5`` whose q list BerkeleyGW's kernel would fold. It fixes the head
slot at zone-boundary ties exactly as ``kernel.x`` does; without it the
restart's q wedge stands in. ``--no-gauge-map`` writes LORRAX's own Bloch gauge,
the red twin of the gauge test. ``--q0-body dipole`` adds BerkeleyGW's 3-D
q0 body term from the bundle's stored q → 0 wings (``head_wing_L``/``R``,
``S_cart_head``, ``S_cart_head_direct``). Physics and conventions:
docs/theory/bgw-bsemat-export.md.
"""
import argparse


def build_parser():
    ap = argparse.ArgumentParser(prog="bse.bsemat_cli", allow_abbrev=False,
                                 description="LORRAX static W -> BerkeleyGW 4.0 bsemat.h5")
    ap.add_argument("-i", "--input", required=True, help="producer GW deck of the restart")
    ap.add_argument("--wfn-co", required=True,
                    help="explicit full-grid coarse WFN (BerkeleyGW WFN_co)")
    ap.add_argument("--n-val", type=int, required=True,
                    help="valence (spinor) bands = BerkeleyGW number_val_bands")
    ap.add_argument("--n-cond", type=int, required=True,
                    help="conduction (spinor) bands = BerkeleyGW number_cond_bands")
    ap.add_argument("--extended", action="store_true",
                    help="extended_kernel: every (n1,n2)->(n1',n2') block (non-TDA absorption)")
    ap.add_argument("--bgw-eps", default=None,
                    help="BerkeleyGW epsmat.h5 whose q list fixes the head slot of zone-boundary ties")
    ap.add_argument("--no-gauge-map", action="store_true",
                    help="write LORRAX's Bloch gauge (red twin of the gauge test)")
    ap.add_argument("--q0-body", choices=("head-less", "dipole"), default="head-less",
                    help="q = 0 body: head-less Γ W, or + the 3-D dipole term from the bundle's stored wings")
    ap.add_argument("--q0-vector", type=float, nargs=3, default=None, metavar=("Q1", "Q2", "Q3"),
                    help="BerkeleyGW q0 (crystal) for --q0-body dipole; default: eps0mat.h5 beside --bgw-eps")
    ap.add_argument("-o", "--output", default="bsemat.h5")
    ap.add_argument("--summary", default=None, help="JSON summary path (default: <output>.json)")
    return ap


if __name__ == "__main__":
    # Argv is answered before any runtime exists — runtime/cli_seam.py.
    from runtime.cli_seam import refuse_bad_argv
    refuse_bad_argv(build_parser())

from runtime import debug_print, initialize_communicator_stack  # noqa: E402

RUNTIME = initialize_communicator_stack(print_fn=debug_print)

import json  # noqa: E402
import os    # noqa: E402

import jax   # noqa: E402


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    from .bse_ring_comm import create_mesh_2d
    from .bsemat_export import run_bsemat_export
    s = run_bsemat_export(input_file=a.input, wfn_co=a.wfn_co, out_path=a.output,
                          n_val=a.n_val, n_cond=a.n_cond, extended=a.extended,
                          use_gauge_map=not a.no_gauge_map, bgw_eps=a.bgw_eps,
                          q0_body=a.q0_body.replace("-", "_"), q0_vector=a.q0_vector,
                          mesh_xy=create_mesh_2d())
    if jax.process_index() == 0:
        with open(a.summary or os.path.splitext(a.output)[0] + ".json", "w") as fh:
            json.dump(s, fh, indent=1, default=str)
        print(f"[bsemat] wrote {a.output} in {s['wall_total_s']:.1f} s "
              f"(rows {s['wall_rows_s']:.1f} s, peak device "
              f"{s['peak_device_bytes'] / 2**30:.2f} GiB)")
    return 0


if __name__ == "__main__":
    from runtime import run_main_and_finalize
    run_main_and_finalize(main)
