#!/usr/bin/env python3
"""
Rotate DFT wavefunctions to QP basis using rotation matrices from COHSEX.

This script:
1. Reads WFN.h5 (DFT wavefunctions in reduced BZ)
2. Reads qp_wfn_rotations.h5 (rotation matrices U and QP energies from COHSEX)
3. Rotates wavefunction coefficients: c_qp[n,G] = Σ_m U[k,m,n] * c_dft[m,G]
4. Writes WFN_qp.h5 with rotated coefficients and updated energies

Usage:
    python rotate_wfn_to_qp.py WFN.h5 qp_wfn_rotations.h5 [--output WFN_qp.h5]
"""

import argparse
import os
import shutil
import numpy as np
import h5py

from file_io.mf_header import kpt_starts


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


def rotate_wfn_coefficients(wfn_file, rot_file, output_file, verbose=True, energy_only=False):
    """
    Rotate WFN coefficients from DFT to QP basis.
    
    Args:
        wfn_file: Path to input WFN.h5
        rot_file: Path to qp_wfn_rotations.h5
        output_file: Path to output WFN_qp.h5
        verbose: Print progress information
        energy_only: If True, only update energies, don't rotate wavefunctions
    """
    # Read and authenticate BEFORE copying or mutating an output.  U_mnk is
    # labelled in a particular DFT-band basis; matching shapes and k-points
    # cannot make a rotation from a different source WFN safe.
    from file_io.qp_wfn import (
        authenticate_qp_rotations_source_wfn,
        read_qp_rotations_artifact,
    )
    from wfn_loader import WfnLoader
    _arr = read_qp_rotations_artifact(rot_file)
    with WfnLoader(wfn_file) as _source_wfn:
        authenticate_qp_rotations_source_wfn(
            _arr, _source_wfn, artifact_path=rot_file)

    # Copy WFN.h5 only after the source-basis contract succeeds.
    if verbose:
        print(f"Copying {wfn_file} -> {output_file}")
    shutil.copy2(wfn_file, output_file)

    # THE ARRAYS COME BACK THROUGH THE ADAPTER, the
    # coordinates straight off the file: ``U_mnk`` and ``E_qp_nk_rydberg``
    # may be stored on the file wedge (``k_storage='ibz'``) and are unfolded
    # here, while ``kpoints_crys`` and ``kirr_to_kfull`` are ALWAYS full-BZ
    # and keep their old meaning, so everything below indexes by full-BZ k
    # exactly as it always did.  A file with no ``k_storage`` attr is read
    # verbatim, so a pre-format file is untouched by this.
    U_mnk = _arr['U_mnk']                      # (nk_full, nb, nb)
    E_qp_ry = _arr['E_qp_nk_rydberg']          # (nk_full, nb) in Rydberg
    band_range = _arr['band_range']  # [band_start, band_stop]
    rot_kpoints = _arr['kpoints_crys']  # (nk_full, 3)
    kgrid = _arr['kgrid']  # [nkx, nky, nkz]

    band_start, band_stop = band_range
    nb_sigma = band_stop - band_start

    if verbose:
        print(f"Rotation file: {rot_file}")
        print(f"  U_mnk shape: {U_mnk.shape}")
        print(f"  Band range: [{band_start}, {band_stop})")
        print(f"  K-grid: {kgrid}")
        print(f"  Number of full-zone k-points: {len(rot_kpoints)}")
    
    # Open WFN file and get k-point info
    with h5py.File(wfn_file, 'r') as f_wfn:
        wfn_kpoints = f_wfn['mf_header/kpoints/rk'][:]  # (nk_red, 3) - transposed from file
        wfn_shift = f_wfn['mf_header/kpoints/shift'][:]
        wfn_kgrid = f_wfn['mf_header/kpoints/kgrid'][:]
        nk_red = f_wfn['mf_header/kpoints/nrk'][()]
        nbands = f_wfn['mf_header/kpoints/mnband'][()]
        ngk = f_wfn['mf_header/kpoints/ngk'][:]
        nspin = f_wfn['mf_header/kpoints/nspin'][()]
        nspinor = f_wfn['mf_header/kpoints/nspinor'][()]
        
        if verbose:
            print(f"\nWFN file: {wfn_file}")
            print(f"  Number of reduced k-points: {nk_red}")
            print(f"  Number of bands: {nbands}")
            print(f"  K-grid: {wfn_kgrid}")
            print(f"  nspin={nspin}, nspinor={nspinor}")
    
    # Find mapping from reduced to full zone k-points
    kirr_to_kfull = read_kirr_to_kfull(rot_file, wfn_kpoints, rot_kpoints)
    
    if verbose:
        print(f"\nK-point mapping (reduced -> full zone):")
        for ik in range(min(5, nk_red)):
            print(f"  k_red[{ik}] = {wfn_kpoints[ik]} -> k_full[{kirr_to_kfull[ik]}]")
        if nk_red > 5:
            print(f"  ...")
    
    # Calculate k-point starts for indexing into coefficients
    k_starts = kpt_starts(ngk)
    
    # Open output file for modification
    with h5py.File(output_file, 'r+') as f_out:
        # Get coefficients dataset - shape is (mnband, nspin*nspinor, ngktot, 2) for complex
        coeffs = f_out['wfns/coeffs']
        coeffs_shape = coeffs.shape
        
        if verbose:
            print(f"\nCoefficients shape: {coeffs_shape}")
            print(f"  (mnband, nspin*nspinor, ngktot, real/imag)")
        
        if energy_only:
            if verbose:
                print("\n[energy-only mode] Skipping wavefunction rotation")
        
        # Process each reduced k-point
        for ik_red in range(nk_red):
            if energy_only:
                continue  # Skip rotation, only update energies below
            ik_full = kirr_to_kfull[ik_red]
            U_k = U_mnk[ik_full]  # (nb_sigma, nb_sigma), U[m,n] = <m_DFT|n_QP>
            
            # Get slice indices for this k-point's G-vectors
            start = k_starts[ik_red]
            end = start + ngk[ik_red]
            ng_k = ngk[ik_red]
            
            # Read DFT coefficients for bands in sigma range
            # coeffs shape: (mnband, nspin*nspinor, ngktot, 2)
            # We need bands [band_start:band_stop]
            c_dft = np.zeros((nb_sigma, nspinor, ng_k), dtype=np.complex128)
            
            for ib_local, ib in enumerate(range(band_start, band_stop)):
                for ispinor in range(nspinor):
                    c_dft[ib_local, ispinor, :] = (
                        coeffs[ib, ispinor, start:end, 0] +
                        1j * coeffs[ib, ispinor, start:end, 1]
                    )
            
            # Rotate: c_qp[n, G] = Σ_m U[m, n] * c_dft[m, G]
            # U[m, n] = <m_DFT|n_QP>, meaning |n_QP> = Σ_m U[m,n] |m_DFT>
            # So: c_qp[n,G] = <G|n_QP> = Σ_m U[m,n] <G|m_DFT> = Σ_m U[m,n] c_dft[m,G]
            # Matrix form: c_qp = U^T @ c_dft  (NOT U^H!)
            c_qp = np.zeros_like(c_dft)
            for ispinor in range(nspinor):
                # c_dft[ib, G] shape (nb_sigma, ng_k)
                # U_k shape (nb_sigma, nb_sigma)
                # c_qp[n, G] = sum_m U[m, n] c_dft[m, G] = (U^T @ c_dft)[n, G]
                c_qp[:, ispinor, :] = U_k.T @ c_dft[:, ispinor, :]
            
            # Diagnostic: check unitarity of U and norm preservation
            if ik_red == 0 and verbose:
                # U should be unitary: U^H @ U = I
                UhU = np.conj(U_k.T) @ U_k
                unitarity_err = np.max(np.abs(UhU - np.eye(nb_sigma)))
                print(f"  [Diagnostic k=0] U unitarity error: {unitarity_err:.2e}")
                
                # Check norm preservation (should be same before/after rotation)
                dft_norms = np.sum(np.abs(c_dft[:, 0, :])**2, axis=1)
                qp_norms = np.sum(np.abs(c_qp[:, 0, :])**2, axis=1)
                norm_diff = np.max(np.abs(dft_norms - qp_norms))
                print(f"  [Diagnostic k=0] Max norm diff (DFT vs QP): {norm_diff:.2e}")
                
                # Check if U is close to identity (small rotation)
                diag_weight = np.sum(np.abs(np.diag(U_k))**2) / nb_sigma
                print(f"  [Diagnostic k=0] Avg |U_nn|^2 (1.0 = identity): {diag_weight:.4f}")
                
                # Show which DFT bands contribute most to each QP band (top 3)
                print(f"  [Diagnostic k=0] Band mixing (QP band <- DFT contributions):")
                for n_qp in range(min(5, nb_sigma)):  # First 5 QP bands
                    weights = np.abs(U_k[:, n_qp])**2
                    top_indices = np.argsort(weights)[::-1][:3]
                    top_str = ", ".join([f"{i}({weights[i]:.2f})" for i in top_indices])
                    print(f"    QP {n_qp} <- DFT bands: {top_str}")
            
            # Write back to file
            for ib_local, ib in enumerate(range(band_start, band_stop)):
                for ispinor in range(nspinor):
                    coeffs[ib, ispinor, start:end, 0] = np.real(c_qp[ib_local, ispinor, :])
                    coeffs[ib, ispinor, start:end, 1] = np.imag(c_qp[ib_local, ispinor, :])
            
            if verbose and (ik_red % 10 == 0 or ik_red == nk_red - 1):
                print(f"  Rotated k-point {ik_red + 1}/{nk_red}")
        
        # Update energies for the sigma bands
        # Energies in WFN.h5: h5py reads as (nspin, nk_red, nbands)
        # WFNReader accesses as energies[ispin, ik, ib]
        el = f_out['mf_header/kpoints/el']
        el_shape = el.shape
        
        if verbose:
            print(f"\nEnergies shape: {el_shape} (nspin, nk_red, nbands)")
        
        # Read current energies
        energies = el[:]
        
        # Update with QP energies for each reduced k-point
        if verbose:
            print(f"\n  Energy comparison at k=0 (Rydberg):")
            print(f"  {'Band':>4s}  {'DFT':>12s}  {'QP':>12s}  {'Diff':>12s}")
        
        for ik_red in range(nk_red):
            ik_full = kirr_to_kfull[ik_red]
            for ib_local, ib in enumerate(range(band_start, band_stop)):
                e_dft = energies[0, ik_red, ib]
                e_qp = E_qp_ry[ik_full, ib_local]
                # energies[ispin, ik, ib] - update all spins with the same QP energy
                energies[:, ik_red, ib] = e_qp
                
                if ik_red == 0 and verbose and ib_local < 10:
                    print(f"  {ib:4d}  {e_dft:12.6f}  {e_qp:12.6f}  {e_qp - e_dft:12.6f}")
        
        # Write updated energies
        el[...] = energies
        
        if verbose:
            print(f"\nUpdated energies for bands [{band_start}, {band_stop})")
    
    if verbose:
        print(f"\nWrote {output_file}")
    
    return kirr_to_kfull


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False,
        description='Rotate DFT wavefunctions to QP basis using COHSEX rotation matrices.'
    )
    parser.add_argument('wfn_file', help='Input WFN.h5 file')
    parser.add_argument('rotation_file', help='QP rotation file (qp_wfn_rotations.h5)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file (default: WFN_qp.h5 in same directory as WFN.h5)')
    parser.add_argument('--energy-only', action='store_true',
                        help='Only update energies, do not rotate wavefunctions (for debugging)')
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
    # Rotate wavefunctions (or just update energies if --energy-only)
    kirr_to_kfull = rotate_wfn_coefficients(
        wfn_file, rotation_file, output_file, verbose=verbose,
        energy_only=args.energy_only
    )
    
    if verbose:
        print("\nDone!")
    
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
