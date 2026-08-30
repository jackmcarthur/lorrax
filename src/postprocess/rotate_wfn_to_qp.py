#!/usr/bin/env python3
"""
Rotate DFT wavefunctions to QP basis using rotation matrices from COHSEX.

This is a thin command-line adapter around
``file_io.qp_wfn.write_qp_wfn_h5``.  The file-format owner performs the
coefficient rotation, k-streamed WFN read, BGW-compatible write, energy
replacement, and positive QP-WFN stamping.  This module only authenticates
the companion artifact and selects its full-BZ rows for the source WFN's
file wedge.

Usage:
    python rotate_wfn_to_qp.py WFN.h5 qp_wfn_rotations.h5 [--output WFN_qp.h5]
"""

import argparse
import os
import numpy as np
import h5py


def read_kirr_to_kfull(rot_file, wfn_kpoints, rot_kpoints):
    """The wedge→full-BZ map, READ from the rotation file, never re-derived.

    ``qp_wfn_rotations.h5`` already carries ``kirr_to_kfull``, written
    straight from ``sym.kirr_fullids`` by both producers
    (``gw.gw_jax``/``gw_output.write_results`` and
    ``gw.sc_iteration``).  The symmetry service builds that table by
    EXACT periodic match and RAISES on a miss (``maps.py:1340-1359``),
    and its contract is ``unfolded_kpts[kirr_fullids] == wfn.kpoints``.

    WHAT THIS REPLACED.  This module used to rebuild the same table with
    a ``np.argmin`` over summed coordinate distances at ``tol=1e-6``
    (``find_kpoint_mapping``), taking the nearest full-BZ k to each
    reduced one with no uniqueness check — so two k within tolerance of
    one another resolved silently to whichever came first, and the
    rebuilt table then selected the QP rotation ``U_mnk[ik_full]`` and
    the QP energy for that reduced k.  Those energies reach
    ``eqp{0,1}.dat`` through ``gw.eqp_bgw``, which reads
    ``kirr_to_kfull`` from this very file — so the two disagreeing about
    a k meant the eqp columns and the rotated WFN disagreed too.  There
    is now one table, produced by the service, and this module reads it.

    The coordinates are CHECKED against it rather than searched: the map
    must reproduce ``wfn.kpoints`` from ``kpoints_crys``, which is the
    service's own contract restated at the point of use.
    """
    with h5py.File(rot_file, 'r') as f_rot:
        if 'kirr_to_kfull' not in f_rot:
            raise ValueError(
                f"{os.path.basename(rot_file)} has no 'kirr_to_kfull' "
                f"dataset.  Every rotation file the current drivers write "
                f"carries it (from sym.kirr_fullids); a file without it "
                f"predates that and cannot be used here — the mapping is "
                f"NOT re-derived, because deriving it by nearest-coordinate "
                f"search is the defect this function exists to remove.  "
                f"Regenerate the rotation file.")
        kirr_to_kfull = np.asarray(f_rot['kirr_to_kfull'][:], dtype=np.int32)

    nk_red = len(wfn_kpoints)
    if kirr_to_kfull.shape != (nk_red,):
        raise ValueError(
            f"kirr_to_kfull has shape {kirr_to_kfull.shape}, expected "
            f"({nk_red},) — the rotation file and {nk_red}-point WFN are "
            f"not the same calculation.")
    if kirr_to_kfull.max(initial=-1) >= len(rot_kpoints):
        raise ValueError(
            f"kirr_to_kfull reaches full-BZ row {int(kirr_to_kfull.max())} "
            f"but kpoints_crys has only {len(rot_kpoints)} rows.")

    # The service's contract, checked here: unfolded_kpts[kirr_fullids]
    # IS wfn.kpoints.  Comparison is modulo a lattice vector because the
    # two files may hold a k in different periodic images.
    d = np.asarray(rot_kpoints)[kirr_to_kfull] - np.asarray(wfn_kpoints)
    d -= np.rint(d)
    worst = float(np.max(np.abs(d))) if d.size else 0.0
    if worst > 1e-6:
        bad = int(np.argmax(np.max(np.abs(d), axis=1)))
        raise ValueError(
            f"kirr_to_kfull[{bad}] = {int(kirr_to_kfull[bad])} points at "
            f"{np.asarray(rot_kpoints)[kirr_to_kfull[bad]].tolist()} but "
            f"WFN reduced k-point {bad} is {np.asarray(wfn_kpoints)[bad].tolist()} "
            f"(worst |Δk| = {worst:.2e}).  The rotation file and the WFN "
            f"disagree about the k-set.")
    return kirr_to_kfull


def rotate_wfn_coefficients(wfn_file, rot_file, output_file, verbose=True):
    """Write one authenticated QP WFN through the canonical format owner."""
    from file_io.qp_wfn import (
        authenticate_qp_rotations_source_wfn,
        read_qp_rotations_artifact,
        write_qp_wfn_h5,
    )
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader

    artifact = read_qp_rotations_artifact(rot_file)
    with WfnLoader(wfn_file) as source_wfn:
        authenticate_qp_rotations_source_wfn(
            artifact, source_wfn, artifact_path=rot_file)
        kirr_to_kfull = read_kirr_to_kfull(
            rot_file, source_wfn.kpoints, artifact["kpoints_crys"])

        source_kgrid = np.asarray(source_wfn.kgrid, dtype=np.int64)
        artifact_kgrid = np.asarray(artifact["kgrid"], dtype=np.int64)
        if not np.array_equal(source_kgrid, artifact_kgrid):
            raise ValueError(
                f"QP rotations kgrid {artifact_kgrid.tolist()} does not "
                f"match source WFN kgrid {source_kgrid.tolist()}.")

        band_start, band_stop = (
            int(x) for x in np.asarray(artifact["band_range"]).tolist())
        if not (0 <= band_start < band_stop <= int(source_wfn.nbands)):
            raise ValueError(
                f"QP rotations band range [{band_start}, {band_stop}) is "
                f"outside source WFN [0, {int(source_wfn.nbands)}).")

        U_wedge = np.asarray(
            artifact["U_mnk"][kirr_to_kfull], dtype=np.complex128)
        E_wedge_ry = np.asarray(
            artifact["E_qp_nk_rydberg"][kirr_to_kfull], dtype=np.float64)
        if verbose:
            print(f"Rotation file: {rot_file}")
            print(f"  Full-BZ U shape: {artifact['U_mnk'].shape}")
            print(f"  WFN wedge rows: {len(kirr_to_kfull)}")
            print(f"  Band range: [{band_start}, {band_stop})")
            print(f"  K-grid: {artifact_kgrid.tolist()}")

        write_qp_wfn_h5(
            output_file,
            wfn=source_wfn,
            U_kmn=U_wedge,
            enk_active_qp_ry=E_wedge_ry,
            band_start=band_start,
            band_stop=band_stop,
        )

    if verbose:
        print(f"Wrote authenticated QP WFN: {output_file}")
    return kirr_to_kfull


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False,
        description='Rotate DFT wavefunctions to QP basis using COHSEX rotation matrices.'
    )
    parser.add_argument('wfn_file', help='Input WFN.h5 file')
    parser.add_argument('rotation_file', help='QP rotation file (qp_wfn_rotations.h5)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file (default: WFN_qp.h5 in same directory as WFN.h5)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')
    
    args = parser.parse_args()
    
    # Resolve paths relative to input file directory (per user preference)
    input_dir = os.path.dirname(os.path.abspath(args.wfn_file))
    
    wfn_file = os.path.abspath(args.wfn_file)
    
    # Handle rotation file path
    if os.path.isabs(args.rotation_file):
        rotation_file = args.rotation_file
    else:
        rotation_file = os.path.join(input_dir, args.rotation_file)
    
    # Handle output file path
    if args.output is None:
        output_file = os.path.join(input_dir, 'WFN_qp.h5')
    elif os.path.isabs(args.output):
        output_file = args.output
    else:
        output_file = os.path.join(input_dir, args.output)
    
    verbose = not args.quiet
    
    if verbose:
        print("=" * 60)
        print("Rotate WFN to QP basis")
        print("=" * 60)
        print(f"Input WFN:       {wfn_file}")
        print(f"Rotation file:   {rotation_file}")
        print(f"Output WFN_qp:   {output_file}")
        print("=" * 60)
    
    # NOTE: there is deliberately no --add-mapping any more.  It recomputed
    # kirr_to_kfull by nearest-coordinate search and OVERWROTE the dataset
    # the symmetry service had already written from sym.kirr_fullids — i.e.
    # it replaced the exact table with an approximation of itself.  Every
    # rotation file the drivers write carries the real one.
    kirr_to_kfull = rotate_wfn_coefficients(
        wfn_file, rotation_file, output_file, verbose=verbose
    )
    
    if verbose:
        print("\nDone!")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
