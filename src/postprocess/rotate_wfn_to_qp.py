#!/usr/bin/env python3
"""
Rotate DFT wavefunctions to QP basis using rotation matrices from COHSEX.

This is a thin command-line adapter around
``file_io.qp_wfn.write_qp_wfn_h5``.  The file-format owner performs the
coefficient rotation, the sharded read and write, energy replacement, and
positive QP-WFN stamping.  This module only authenticates the companion
artifact and selects its full-BZ rows for the source WFN's file wedge.

SPMD like every driver: run it with one process per GPU and every rank
writes its own slab (``lx run -n P -- python -m postprocess.rotate_wfn_to_qp
...``).

Usage:
    python rotate_wfn_to_qp.py WFN.h5 qp_wfn_rotations.h5 [--output WFN_qp.h5]
"""

import argparse
import os
from types import SimpleNamespace
import numpy as np


def _stored_final_state_args(artifact, kirr_to_kfull):
    """Select a companion's complete final E/f state onto the WFN wedge."""
    result = {}
    if "E_full_nk_rydberg" in artifact:
        result["enk_full_base_ry"] = np.asarray(
            artifact["E_full_nk_rydberg"][kirr_to_kfull], dtype=np.float64)
    provenance = artifact["occupation_provenance"]
    if provenance is None:
        return result
    occupations_full = np.asarray(artifact["occupations_kn"], dtype=np.float64)
    result.update({
        "occupations_kn": np.asarray(
            occupations_full[kirr_to_kfull], dtype=np.float64),
        # ``write_qp_wfn_h5`` intentionally accepts this record by protocol:
        # file_io must not import the GW occupation solver merely to transport
        # a completed solve's provenance.
        "occupation_state": SimpleNamespace(
            f_kn=occupations_full, **provenance),
    })
    return result


def rotate_wfn_coefficients(wfn_file, rot_file, output_file, verbose=True,
                            mesh=None):
    """Write one authenticated QP WFN through the canonical format owner.

    COLLECTIVE over ``mesh`` (default: the run's canonical mesh).
    """
    from common.collectives import resolve_mesh
    from file_io.qp_wfn import (authenticate_qp_rotations_source_wfn, write_qp_wfn_h5)
    from file_io.restart_bundle import (read_kirr_to_kfull, read_qp_rotations_artifact)
    from ffi import _services
    _services.ensure_on_path()
    from wfn_loader import WfnLoader
    mesh = resolve_mesh(mesh)
    verbose = verbose and _is_rank0()

    artifact = read_qp_rotations_artifact(rot_file)
    with WfnLoader(wfn_file) as source_wfn:
        authenticate_qp_rotations_source_wfn(
            artifact, source_wfn, artifact_path=rot_file)
        kirr_to_kfull = read_kirr_to_kfull(
            rot_file, source_wfn.kpoints, artifact["kpoints_crys"], artifact=artifact)

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
        final_state_kwargs = _stored_final_state_args(
            artifact, kirr_to_kfull)
        if verbose:
            print(f"Rotation file: {rot_file}")
            print(f"  Full-BZ U shape: {artifact['U_mnk'].shape}")
            print(f"  WFN wedge rows: {len(kirr_to_kfull)}")
            print(f"  Band range: [{band_start}, {band_stop})")
            print(f"  K-grid: {artifact_kgrid.tolist()}")
            if artifact["occupation_provenance"] is not None:
                print("  Occupations: stored final fixed-N table "
                      f"({artifact['occupation_provenance']['occ_hash']})")
            if "E_full_nk_rydberg" in artifact:
                print("  Energies: stored complete final ladder")

        write_qp_wfn_h5(
            output_file,
            wfn=source_wfn,
            U_kmn=U_wedge,
            enk_active_qp_ry=E_wedge_ry,
            band_start=band_start,
            band_stop=band_stop,
            mesh=mesh,
            **final_state_kwargs,
        )

    if verbose:
        print(f"Wrote authenticated QP WFN: {output_file}")
    return kirr_to_kfull


def _is_rank0() -> bool:
    from common.collectives import process_rank
    return process_rank() == 0


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
    # The runtime (jax.distributed, one process per GPU) before any jax
    # import; every import below this line is function-local for that reason.
    from runtime import initialize_communicator_stack
    initialize_communicator_stack()
    
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
    
    verbose = not args.quiet and _is_rank0()
    
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
