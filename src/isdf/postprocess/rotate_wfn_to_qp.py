#!/usr/bin/env python3
"""
Rotate DFT wavefunctions to QP basis using rotation matrices from COHSEX.

This script:
1. Reads WFN.h5 (DFT wavefunctions in reduced BZ)
2. Reads qp_wfn_rotations.h5 (rotation matrices U and QP energies from COHSEX)
3. Rotates wavefunction coefficients: c_qp[n,G] = Σ_m conj(U[k,m,n]) * c_dft[m,G]
4. Writes WFN_qp.h5 with rotated coefficients and updated energies

Usage:
    python rotate_wfn_to_qp.py WFN.h5 qp_wfn_rotations.h5 [--output WFN_qp.h5]
"""

import argparse
import os
import shutil
import numpy as np
import h5py


def find_kpoint_mapping(wfn_kpoints, rot_kpoints, kgrid, shift, tol=1e-6):
    """
    Map reduced k-points from WFN.h5 to full zone indices in the rotation file.
    
    Args:
        wfn_kpoints: Reduced k-points from WFN.h5 (nk_red, 3)
        rot_kpoints: Full zone k-points from rotation file (nk_full, 3)
        kgrid: k-grid dimensions [nkx, nky, nkz]
        shift: k-grid shift
        tol: Tolerance for k-point matching
        
    Returns:
        kirr_to_kfull: Array mapping reduced k-point index to full zone index
    """
    nk_red = len(wfn_kpoints)
    nk_full = len(rot_kpoints)
    
    # Generate full k-point grid
    kx = np.linspace(0, 1, kgrid[0], endpoint=False) + shift[0] / kgrid[0]
    ky = np.linspace(0, 1, kgrid[1], endpoint=False) + shift[1] / kgrid[1]
    kz = np.linspace(0, 1, kgrid[2], endpoint=False) + shift[2] / kgrid[2]
    
    kpts_mesh = np.meshgrid(kx, ky, kz, indexing='ij')
    full_kpoints_gen = np.stack([k.flatten() for k in kpts_mesh]).T
    
    kirr_to_kfull = np.zeros(nk_red, dtype=np.int32)
    
    for ik_red, k_red in enumerate(wfn_kpoints):
        # Wrap reduced k-point to [0, 1)
        k_wrapped = k_red % 1.0
        k_wrapped[k_wrapped > 1.0 - tol] = 0.0
        
        # Find matching k-point in the full zone
        diffs = np.abs(rot_kpoints - k_wrapped[None, :])
        diffs = np.minimum(diffs, 1.0 - diffs)  # Account for periodicity
        total_diff = np.sum(diffs, axis=1)
        
        min_idx = np.argmin(total_diff)
        min_diff = total_diff[min_idx]
        
        if min_diff > tol:
            raise ValueError(
                f"Reduced k-point {ik_red}: {k_red} not found in rotation file k-points. "
                f"Minimum difference: {min_diff}"
            )
        
        kirr_to_kfull[ik_red] = min_idx
    
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
    # Copy WFN.h5 to output file first
    if verbose:
        print(f"Copying {wfn_file} -> {output_file}")
    shutil.copy2(wfn_file, output_file)
    
    # Open rotation file
    with h5py.File(rot_file, 'r') as f_rot:
        U_mnk = f_rot['U_mnk'][:]  # (nk_full, nb, nb)
        E_qp_ry = f_rot['E_qp_nk_rydberg'][:]  # (nk_full, nb) in Rydberg
        band_range = f_rot['band_range'][:]  # [band_start, band_stop]
        rot_kpoints = f_rot['kpoints_crys'][:]  # (nk_full, 3)
        kgrid = f_rot['kgrid'][:]  # [nkx, nky, nkz]
        
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
    kirr_to_kfull = find_kpoint_mapping(wfn_kpoints, rot_kpoints, wfn_kgrid, wfn_shift)
    
    if verbose:
        print(f"\nK-point mapping (reduced -> full zone):")
        for ik in range(min(5, nk_red)):
            print(f"  k_red[{ik}] = {wfn_kpoints[ik]} -> k_full[{kirr_to_kfull[ik]}]")
        if nk_red > 5:
            print(f"  ...")
    
    # Calculate k-point starts for indexing into coefficients
    kpt_starts = np.zeros(nk_red, dtype=np.int64)
    for ik in range(1, nk_red):
        kpt_starts[ik] = kpt_starts[ik - 1] + ngk[ik - 1]
    
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
            start = kpt_starts[ik_red]
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


def add_kpoint_mapping_to_rotation_file(rot_file, wfn_file, verbose=True):
    """
    Add the k-point mapping from reduced to full zone to the rotation file.
    This allows easy lookup of which rotation matrix to use for each reduced k-point.
    
    Args:
        rot_file: Path to qp_wfn_rotations.h5
        wfn_file: Path to WFN.h5 (to get reduced k-points)
        verbose: Print progress information
    """
    # Read WFN k-points
    with h5py.File(wfn_file, 'r') as f_wfn:
        wfn_kpoints = f_wfn['mf_header/kpoints/rk'][:]
        wfn_shift = f_wfn['mf_header/kpoints/shift'][:]
        wfn_kgrid = f_wfn['mf_header/kpoints/kgrid'][:]
    
    # Read rotation file k-points
    with h5py.File(rot_file, 'r') as f_rot:
        rot_kpoints = f_rot['kpoints_crys'][:]
    
    # Compute mapping
    kirr_to_kfull = find_kpoint_mapping(wfn_kpoints, rot_kpoints, wfn_kgrid, wfn_shift)
    
    # Add to rotation file
    with h5py.File(rot_file, 'r+') as f_rot:
        # Delete if exists
        if 'kirr_to_kfull' in f_rot:
            del f_rot['kirr_to_kfull']
        if 'kpoints_reduced' in f_rot:
            del f_rot['kpoints_reduced']
        
        f_rot.create_dataset('kirr_to_kfull', data=kirr_to_kfull, dtype=np.int32)
        f_rot.create_dataset('kpoints_reduced', data=wfn_kpoints, dtype=np.float64)
        f_rot.attrs['mapping_description'] = (
            'kirr_to_kfull[ik_red] gives the index into kpoints_crys/U_mnk/E_qp_nk '
            'for the reduced k-point ik_red from WFN.h5'
        )
    
    if verbose:
        print(f"Added kirr_to_kfull mapping to {rot_file}")
        print(f"  Shape: {kirr_to_kfull.shape}")


def main():
    parser = argparse.ArgumentParser(
        description='Rotate DFT wavefunctions to QP basis using COHSEX rotation matrices.'
    )
    parser.add_argument('wfn_file', help='Input WFN.h5 file')
    parser.add_argument('rotation_file', help='QP rotation file (qp_wfn_rotations.h5)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output file (default: WFN_qp.h5 in same directory as WFN.h5)')
    parser.add_argument('--add-mapping', action='store_true',
                        help='Also add k-point mapping to the rotation file')
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
    
    # Optionally add mapping to rotation file first
    if args.add_mapping:
        add_kpoint_mapping_to_rotation_file(rotation_file, wfn_file, verbose=verbose)
    
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

