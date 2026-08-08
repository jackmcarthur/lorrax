"""COMPAT SHIM — cell box truncation of the Coulomb potential for 0D systems.

The FFT itself moved to :mod:`vcoul.box_fft` (2026-08-07):
:func:`compute_vcoul_box` and :func:`_round_up_fft_size`, plus the three
BGW parameters (``N_IN_BOX``, ``NCELL``, ``TRUNC_SHIFT``, from
``Common/nrtype.f90``).  They are re-exported here because
:mod:`gw.coulomb.box_0d` is not the only importer — this module's own
``main()`` is a CLI that compares against a BGW ``vcoul`` file.

The CLI STAYS.  It reads a WFN.h5 through ``file_io.WfnLoader`` and takes
argv, which is exactly the kind of edge the service must not have; a
service that could parse a deck path would not be standalone.  So the
arithmetic went and the driver stayed, which is the split the whole
extraction is made of.

Reference: BerkeleyGW source files:
  - Common/trunc_cell_box.f90 (serial implementation)
  - Common/vcoul_generator.f90 (driver; confirms no miniBZ avg for box trunc)
  - Common/nrtype.f90 (n_in_box=2, ncell=3, trunc_shift=0.5)

Units: output is in Rydberg, matching BerkeleyGW convention: v(G) = 8π/|G|²
for the untruncated case.
"""

import numpy as np

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                               # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from vcoul import (                                         # noqa: E402,F401
    N_IN_BOX,
    NCELL,
    TRUNC_SHIFT,
    _round_up_fft_size,
    compute_vcoul_box,
)

__all__ = ["compute_vcoul_box", "_round_up_fft_size",
           "N_IN_BOX", "NCELL", "TRUNC_SHIFT", "main"]


def main():
    """Load WFN.h5, compute box-truncated vcoul, compare with BGW vcoul file."""
    import argparse
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader

    parser = argparse.ArgumentParser(allow_abbrev=False,
        description="Compute cell-box-truncated Coulomb potential and compare with BGW vcoul"
    )
    parser.add_argument("--wfn", default="WFN.h5", help="WFN HDF5 file")
    parser.add_argument("--vcoul", default="vcoul", help="BGW vcoul text file for comparison")
    args = parser.parse_args()

    wfn = WfnLoader(args.wfn)

    # bdot: reciprocal metric in Bohr^-2, stored directly in the WFN file.
    # This is bdot[i,j] = b_i · b_j where b_i are in 2π/alat units,
    # NOT the same as bvec @ bvec.T (which would be in normalized units).
    bdot = wfn.bdot

    # G-vectors for q=0: all G-vectors in the WFN file for k-point 0
    gvecs_all = wfn.get_gvec_nk(0)  # (ngk, 3) integer crystal coords

    print(f"FFT grid: {wfn.fft_grid}")
    print(f"Number of G-vectors: {gvecs_all.shape[0]}")
    print(f"Cell volume: {wfn.cell_volume:.4f} bohr³")

    vcoul = compute_vcoul_box(bdot, wfn.fft_grid, gvecs_all)

    # Load BGW reference if available
    try:
        data = np.loadtxt(args.vcoul)
        gvecs_ref = data[:, 3:6].astype(int)
        vcoul_ref = data[:, 6]
        print(f"\nBGW vcoul loaded: {len(vcoul_ref)} G-vectors")

        # Match G-vectors between our computation and the reference
        # Build a dict from G-vector tuple -> index for our results
        our_gvec_map = {}
        for i, g in enumerate(gvecs_all):
            our_gvec_map[tuple(g)] = i

        matched = 0
        max_err = 0.0
        max_rel_err = 0.0
        for i in range(len(vcoul_ref)):
            key = tuple(gvecs_ref[i])
            if key in our_gvec_map:
                j = our_gvec_map[key]
                err = abs(vcoul[j] - vcoul_ref[i])
                ref_val = abs(vcoul_ref[i])
                rel = err / ref_val if ref_val > 1e-10 else 0.0
                max_err = max(max_err, err)
                max_rel_err = max(max_rel_err, rel)
                matched += 1
                if i < 10:
                    print(f"  G={key}  ours={vcoul[j]:.8e}  BGW={vcoul_ref[i]:.8e}  err={err:.2e}  rel={rel:.2e}")

        print(f"\nMatched {matched}/{len(vcoul_ref)} G-vectors")
        print(f"Max absolute error: {max_err:.6e}")
        print(f"Max relative error: {max_rel_err:.6e}")

    except FileNotFoundError:
        print(f"\nNo BGW vcoul file '{args.vcoul}' found; printing first 20 values:")
        for i in range(min(20, len(vcoul))):
            print(f"  G={tuple(gvecs_all[i])}  v={vcoul[i]:.8e}")


if __name__ == "__main__":
    raise SystemExit(main())
