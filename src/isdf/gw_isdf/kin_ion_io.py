#!/usr/bin/env python3
"""
Utility to compute and write kin+ion (T + V_loc + V_NL) matrices to kin_ion.h5.
Optionally adds the valence Hartree contribution when --do-hartree is enabled.

This script loads the wavefunctions, symmetry maps, pseudopotentials, and
constructs the required sharded wavefunctions before calling:
  - get_kin_ion(...) -> (nk, nb, nb) complex array
  - write_kin_ion_h5(...)

Usage:
  python -m isdf.gw_isdf.kin_ion_io -i tests/cohsex_debug/cohsex_test.in [-o kin_ion.h5]
"""

import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from isdf.common.wfnreader import WFNReader
from isdf.common import symmetry_maps
from isdf.common import Meta
from isdf.common.load_wfns import read_Gvecs_to_devices

from isdf.psp.get_DFT_mtxels import (
    read_cohsex_input,
    get_bandranges,
    load_pseudopotentials,
    get_kin_ion,
    write_kin_ion_h5,
)


def _resolve_against(path: str, base_dir: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def main(argv=None):
    argp = argparse.ArgumentParser(description="Compute kin+ion matrices and write kin_ion.h5")
    argp.add_argument("-i", "--input", default="tests/cohsex_debug/cohsex_test.in", help="cohsex input file")
    argp.add_argument("-o", "--output", default=None, help="output HDF5 path (default: kin_ion.h5 next to input)")
    argp.add_argument("-n", "--nb", type=int, default=None, help="number of bands to use starting from band 0")
    argp.add_argument("--do-hartree", action="store_true", help="Include Hartree potential when building H(k)")
    args = argp.parse_args(argv)

    print("== kin_ion_io: prepare inputs ==")
    input_dir = os.path.dirname(os.path.abspath(args.input))
    params = read_cohsex_input(args.input)
    wfn_path = _resolve_against(params.get("wfn_file", "WFN.h5"), input_dir)

    print(f"Loading WFN: {os.path.basename(wfn_path)}")
    wfn = WFNReader(wfn_path)
    sym = symmetry_maps.SymMaps(wfn)

    nval = int(params.get("nval", 5))
    ncond = int(params.get("ncond", 5))
    nband = int(params.get("nband", 100))
    bispinor = bool(params.get("bispinor", False))
    print(f"Params: nval={nval} ncond={ncond} nband={nband} bispinor={bispinor}")

    # Determine effective band count: override input nband with --nb when provided
    nb_req = int(args.nb) if args.nb is not None else int(nband)
    nb_eff = max(1, min(int(wfn.nbands), nb_req))
    if args.do_hartree:
        if args.nb is not None and int(args.nb) < int(wfn.nelec):
            raise ValueError(f"--do-hartree requested with nb={args.nb}, but nelec={wfn.nelec}. Increase --nb to at least {wfn.nelec}.")
        if nb_eff < int(wfn.nelec):
            promoted_nb = min(int(wfn.nbands), int(wfn.nelec))
            print(f"--do-hartree requires at least nelec={wfn.nelec} bands; promoting nb from {nb_eff} to {promoted_nb}")
            nb_eff = promoted_nb
    meta = Meta.from_system(wfn, sym, nval, ncond, nb_eff, 0, bispinor)
    print(f"FFT grid: {meta.fft_grid}; spinor: {meta.nspinor}")

    # Device mesh (same heuristic as elsewhere)
    total_devices = jax.process_count() * jax.local_device_count()
    grid_x = int(np.sqrt(total_devices))
    while grid_x > 1 and total_devices % grid_x != 0:
        grid_x -= 1
    grid_y = max(1, total_devices // grid_x)
    devices_2d = np.array(jax.devices()).reshape(grid_x, grid_y)
    mesh_xy = Mesh(devices_2d, ['x', 'y'])
    print(f"Device mesh: {grid_x}x{grid_y} ({total_devices})")

    # Load wavefunctions to devices
    # Load lowest nb_eff bands directly (independent of sigma ranges)
    brange = (0, int(nb_eff))
    print("Loading G-space coefficients to devices...")
    global_psi_G, nb_actual = read_Gvecs_to_devices(wfn, sym, brange, meta, bispinor, mesh_xy)
    print(f"Loaded bands: {nb_actual}, array shape: {tuple(global_psi_G.shape)}")

    # Load pseudopotentials from input directory
    pseudos = load_pseudopotentials(input_dir)
    if not pseudos:
        print("Warning: no pseudopotentials found; kin+ion will be kinetic-only")

    print("Computing kin+ion matrices...")
    kin_ion = get_kin_ion(
        global_psi_G,
        wfn,
        sym,
        pseudos,
        meta,
        mesh_xy,
        include_hartree=bool(args.do_hartree),
        nb_limit=args.nb,
    )
    out_path = args.output or os.path.join(input_dir, 'kin_ion.h5')
    write_kin_ion_h5(kin_ion, out_path)
    print(f"Wrote kin+ion to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
