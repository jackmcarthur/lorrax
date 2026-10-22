#!/usr/bin/env python3
"""``python -m bse.dump_window_dipoles``: a ``kgrid_fine`` bundle's window dipoles, per fine q, to HDF5.

The bundle is loaded exactly as ``bse.bse_jax`` loads it (same deck, restart,
``--eqp``, ``--dipole``, window and degeneracy options, the square startup
mesh), so the written ``d_alpha`` is in the same deterministic htransform
gauge as that run's ``eigenvectors.h5``, and
``absorption_common.exciton_dipole_projections(A, d_alpha)`` re-contracts its
states on CPU without a BSE rerun (CrI3 30x30, 112 states, a run from
f48831528 against a dump from 5c375042c: 1.7e-15 of the largest dipole).

Datasets (rank 0 writes):

* ``d_alpha`` (3, nk, nc, nv) complex128, bohr: the dipoles the BSE uses,
  ``⟨c̃|v̂|ṽ⟩/ΔE`` (Rayleigh-quotient ΔE under ``--eqp``), or the exact
  ``i⟨c̃|r̂|ṽ⟩`` of the fine QP states with ``--eqp --dipole-position-fix``
  (``bse_densify.densified_dipoles``);
* ``de_cv`` (nk, nc, nv) float64, Ry; ``q_frac`` (nk, 3);
* with ``--eqp --dipole-position-fix``: ``d_alpha_rq``, ``v/ΔE_RQ`` (the
  default dipoles; ``--check-key d_alpha_rq`` checks a default run);
* with ``--full-window``: ``v_window``, ``r_window`` (3, nk, nb, nb) and
  ``e_dft_window`` (nk, nb).

Attributes: ``route``, ``source``, ``velocity``, ``on_grid_rel``,
``n_on_grid``, ``rfix`` (JSON), ``deck``, ``eqp``, ``n_val``, ``n_cond``,
``n_occ``, ``lorrax_head``.  ``--check-eigenvectors E.h5`` re-contracts
``--check-key`` with E.h5's ``A`` and reports ``max|P − stored|/max|stored|``
against its ``exciton_data/dipoles`` (the gauge identity; it must be round-off
for the dipole set the run used).
"""
import argparse
import json
import os
import subprocess

# THE startup call (runtime module docstring), before any device is touched.
from runtime import debug_print, initialize_communicator_stack
RUNTIME = initialize_communicator_stack(print_fn=debug_print)

import jax  # noqa: E402
import numpy as np  # noqa: E402

from common.band_degeneracy import DEFAULT_MODE, DEGENERACY_TOL_RY, MODES  # noqa: E402

from .bse_densify import DENSIFIED_POSITION_PAIR_RULES  # noqa: E402
from common.log import get_logger

log = get_logger(__name__)


def _head():
    root = os.environ.get("LORRAX_CHECKOUT") or os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    try:
        return subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=30).stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def check_eigenvectors(eigenvectors_h5, d_alpha):
    """``max|P − stored| / max|stored|`` of ``P = exciton_dipole_projections(A, d_alpha)``."""
    from .absorption_common import (exciton_dipole_projections, load_eigenvectors_h5,
                                    load_exciton_dipoles_h5)
    _, A, _ = load_eigenvectors_h5(eigenvectors_h5)
    _, stored = load_exciton_dipoles_h5(eigenvectors_h5)
    P = np.asarray(exciton_dipole_projections(A, d_alpha))
    stored = stored[:P.shape[0]]
    return float(np.abs(P - stored).max() / np.abs(stored).max()), int(P.shape[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("-i", "--input", required=True, help="the BSE run's deck (with kgrid_fine)")
    ap.add_argument("--n-val", type=int, required=True)
    ap.add_argument("--n-cond", type=int, required=True)
    ap.add_argument("--n-occ", type=int, default=None)
    ap.add_argument("--dipole", required=True, help="the coarse WFN's dipole.h5 (as --dipole)")
    ap.add_argument("--eqp", default=None, help="eqp1.dat (as --eqp)")
    ap.add_argument("--band-degeneracy", choices=MODES, default=DEFAULT_MODE)
    ap.add_argument("--degeneracy-tol-ry", type=float, default=DEGENERACY_TOL_RY)
    ap.add_argument("--full-window", action="store_true",
                    help="also write v_window, r_window, e_dft_window")
    ap.add_argument("--pair-rule", choices=DENSIFIED_POSITION_PAIR_RULES, default="mask")
    ap.add_argument("--pair-gap-ev", type=float, default=0.05)
    ap.add_argument("--dipole-position-fix", action="store_true",
                    help="as bse_jax --dipole-position-fix (exact r of the fine QP states)")
    ap.add_argument("--out", required=True, help="output .h5")
    ap.add_argument("--check-eigenvectors", default=None,
                    help="eigenvectors.h5 of the BSE run to re-contract")
    ap.add_argument("--check-key", default="d_alpha", choices=("d_alpha", "d_alpha_rq"))
    args = ap.parse_args(argv)

    from file_io.restart_bundle import _find_restart_file
    from .bse_io import load_bse_data_from_restart_sharded
    from .bse_ring_comm import create_mesh_2d
    mesh_xy = create_mesh_2d()
    data = load_bse_data_from_restart_sharded(
        _find_restart_file(args.input), n_val=args.n_val, n_cond=args.n_cond, mesh_xy=mesh_xy,
        input_file=args.input, n_occ=args.n_occ, degeneracy_mode=args.band_degeneracy,
        degeneracy_tol_ry=args.degeneracy_tol_ry, eqp_file=args.eqp, velocity_file=args.dipole,
        velocity_full_window=bool(args.full_window), velocity_pair_rule=args.pair_rule,
        velocity_pair_gap_ev=args.pair_gap_ev,
        velocity_position_fix=bool(args.dipole_position_fix))
    dip = data.get("dipoles")
    if not data.get("densified") or dip is None:
        raise SystemExit(
            f"GATE dump_window_dipoles_densified: {args.input} loads an undensified bundle "
            f"(no kgrid_fine); want a kgrid_fine deck; why: an undensified run slices "
            f"{args.dipole} itself (absorption_common.window_dipoles), there is nothing "
            f"to dump; fix: pass the deck of the kgrid_fine run.")
    if jax.process_index() != 0:
        return 0
    import h5py
    with h5py.File(args.out, "w") as f:
        for key in ("d_alpha", "de_cv", "q_frac", "d_alpha_rq", "v_window", "r_window", "e_dft_window"):
            if dip.get(key) is not None:
                f.create_dataset(key, data=np.asarray(dip[key]))
        for key in ("route", "source", "velocity"):
            f.attrs[key] = str(dip[key])
        f.attrs["on_grid_rel"] = float(dip["on_grid_rel"])
        f.attrs["n_on_grid"] = int(dip["n_on_grid"])
        f.attrs["rfix"] = json.dumps(dip.get("rfix"))
        f.attrs["deck"] = os.path.abspath(args.input)
        f.attrs["eqp"] = "" if args.eqp is None else os.path.abspath(args.eqp)
        f.attrs["n_val"], f.attrs["n_cond"] = int(args.n_val), int(args.n_cond)
        f.attrs["n_occ"] = -1 if args.n_occ is None else int(args.n_occ)
        f.attrs["lorrax_head"] = _head()
    d = np.asarray(dip["d_alpha"])
    nk = d.shape[1]
    log.info(f"[dump_window_dipoles] wrote {args.out}: d_alpha {d.shape}; route {dip['route']}")
    log.info(f"[dump_window_dipoles] IP sums per cell (bohr^2) x {np.sum(np.abs(d[0])**2)/nk:.6f} "
             f"y {np.sum(np.abs(d[1])**2)/nk:.6f} z {np.sum(np.abs(d[2])**2)/nk:.6f} "
             f"sigma+ {np.sum(np.abs(d[0] + 1j*d[1])**2)/(2*nk):.6f} "
             f"sigma- {np.sum(np.abs(d[0] - 1j*d[1])**2)/(2*nk):.6f}")
    if dip.get("rfix") is not None:
        log.debug(f"[dump_window_dipoles] rfix {json.dumps(dip['rfix'])}")
    if args.check_eigenvectors:
        key = args.check_key
        if dip.get(key) is None:
            raise SystemExit(f"GATE dump_window_dipoles_check_key: {key} is not in this bundle "
                             f"(d_alpha_rq exists only with --eqp --dipole-position-fix)")
        rel, n = check_eigenvectors(args.check_eigenvectors, np.asarray(dip[key]))
        log.info(f"[dump_window_dipoles] gauge identity {key} vs {args.check_eigenvectors}: "
                 f"max|P - stored|/max|stored| = {rel:.3e} over {n} states")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
