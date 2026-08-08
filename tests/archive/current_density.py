#!/usr/bin/env python
"""
Calculate ground state current density in real space.

Current density formula (atomic units, ħ=m=1):
  j_i(r) = (1/2i) * sum_{n,k} [ Psi*_nk(r) nabla_i Psi_nk(r) - (nabla_i Psi_nk(r))* Psi_nk(r) ]
         = Im[ sum_{n,k} Psi*_nk(r) nabla_i Psi_nk(r) ]

where:
  Psi_nk(r) = exp(ik.r) u_nk(r)  (Bloch wavefunction)
  u_nk(r) = sum_G c_nk(G) exp(iG.r)  (periodic part)
  
  nabla Psi_nk = exp(ik.r) [ (ik + nabla) u_nk ]
               = exp(ik.r) sum_G c_nk(G) i(k+G) exp(iG.r)
               
So in reciprocal space:
  (nabla_i Psi_nk)_G = i(k+G)_{cart,i} * c_nk(G)

This script:
1. Reads WFN.h5 wavefunction file
2. Computes nabla Psi in G-space for occupied states
3. FFTs to real space
4. Computes j = Im[ Psi* nabla Psi ]
5. Saves to HDF5 and plots vector slices
"""

import os
import sys
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Add the src directory to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from wfn_loader import WfnLoader                                    # noqa: E402
from symmetry_maps import SymMaps


def get_psi_and_grad_psi_realspace(wfn, sym, ib, ik_full, fft_grid):
    """
    Get both Psi and nabla Psi in real space for band ib, full-zone k-point ik_full.
    
    Args:
        wfn: WFNReader instance
        sym: SymMaps instance
        ib: band index (0-based)
        ik_full: k-point index in full zone
        fft_grid: FFT grid size (nx, ny, nz)
        
    Returns:
        psi_r: complex array (nspinor, nx, ny, nz) - wavefunction in real space
        grad_psi_r: complex array (3, nspinor, nx, ny, nz) - gradient in real space (Cartesian)
    """
    # Get k-point in crystal coordinates
    kvec = sym.unfolded_kpts[ik_full]  # (3,) crystal coords
    
    # Get the irreducible k-point index and symmetry operation
    kbar_idx = sym.irk_to_k_map[ik_full]
    
    # Get rotated G-vectors and coefficients for this k-point
    gvecs_k = sym.get_gvecs_kfull(wfn, ik_full)  # (ngk, 3) integer G-vectors
    cnk = sym.get_cnk_fullzone(wfn, ib, ik_full)  # (nspinor, ngk) coefficients
    
    # Convert G-vectors to be within FFT grid bounds
    gvecs_mod = gvecs_k.copy()
    for i in range(3):
        gvecs_mod[:, i] = gvecs_k[:, i] % fft_grid[i]
    
    # Reciprocal lattice vectors (in Cartesian, with 2pi/alat factor)
    bvec = wfn.bvec  # Already in units where bvec.T @ avec = 2pi I
    blat = wfn.blat  # 2pi/alat
    bvec_cart = blat * bvec  # (3,3) Cartesian reciprocal vectors
    
    # Compute (k+G) in Cartesian coordinates
    # k+G in crystal coords: kvec + gvecs (broadcasting)
    kpG_crys = kvec[None, :] + gvecs_k.astype(float)  # (ngk, 3)
    kpG_cart = kpG_crys @ bvec_cart  # (ngk, 3) in 1/bohr
    
    # Initialize FFT boxes
    nspinor = cnk.shape[0]
    nx, ny, nz = fft_grid
    
    psi_G = np.zeros((nspinor, nx, ny, nz), dtype=np.complex128)
    grad_psi_G = np.zeros((3, nspinor, nx, ny, nz), dtype=np.complex128)
    
    # Populate FFT boxes
    for ispin in range(nspinor):
        # Wavefunction: c_nk(G)
        psi_G[ispin, gvecs_mod[:, 0], gvecs_mod[:, 1], gvecs_mod[:, 2]] = cnk[ispin]
        
        # Gradient: i(k+G)_i * c_nk(G)  (Cartesian components)
        for idir in range(3):
            grad_psi_G[idir, ispin, gvecs_mod[:, 0], gvecs_mod[:, 1], gvecs_mod[:, 2]] = (
                1j * kpG_cart[:, idir] * cnk[ispin]
            )
    
    # FFT to real space (IFFT of G-space coefficients)
    psi_r = np.fft.ifftn(psi_G, axes=(-3, -2, -1))
    grad_psi_r = np.fft.ifftn(grad_psi_G, axes=(-3, -2, -1))
    
    # Apply Bloch phase exp(ik.r)
    # r in crystal coordinates: (ix/nx, iy/ny, iz/nz)
    fx = np.arange(nx) / nx
    fy = np.arange(ny) / ny
    fz = np.arange(nz) / nz
    rx, ry, rz = np.meshgrid(fx, fy, fz, indexing='ij')
    
    # k.r in crystal coordinates
    kr = kvec[0] * rx + kvec[1] * ry + kvec[2] * rz
    phase = np.exp(2j * np.pi * kr)
    
    # Apply phase
    psi_r = psi_r * phase[None, :, :, :]
    grad_psi_r = grad_psi_r * phase[None, None, :, :, :]
    
    # Normalize by sqrt(N_r) for proper normalization
    # The FFT convention: sum_r |u(r)|^2 = N_r sum_G |c(G)|^2
    # We want: int |Psi|^2 d^3r = 1 (per unit cell)
    # With real-space grid: (1/N_r) sum_r |Psi(r)|^2 * Omega = 1
    # So |Psi(r)|^2 should be normalized as N_r/Omega
    # The ifft gives u(r), and Psi(r) = exp(ikr) u(r)
    # IFFTN divides by N_r, so we need to multiply by sqrt(N_r) for proper norm
    norm = np.sqrt(nx * ny * nz)
    psi_r = psi_r * norm
    grad_psi_r = grad_psi_r * norm
    
    return psi_r, grad_psi_r


def compute_current_density(wfn_file, output_h5=None, plot_slices=True, plot_dir=None, 
                            band_start=None, band_end=None, scale_last_band=None):
    """
    Compute the ground state current density from WFN.h5.
    
    Args:
        wfn_file: Path to WFN.h5 file
        output_h5: Path to output HDF5 file (default: current_density.h5 in same dir)
        plot_slices: Whether to generate vector plots of slices
        plot_dir: Directory for plots (default: same as output_h5)
        band_start: First band to include (1-indexed, default: 1)
        band_end: Last band to include (1-indexed, default: all occupied)
        scale_last_band: Scale factor for the last occupied band (default: 1.0)
        
    Returns:
        j_r: Current density array (3, nx, ny, nz) in atomic units
    """
    print(f"Loading wavefunction file: {wfn_file}")
    wfn = WfnLoader(wfn_file)
    sym = SymMaps(wfn)
    
    # System info
    fft_grid = tuple(wfn.fft_grid)
    nx, ny, nz = fft_grid
    nspinor = wfn.nspinor
    nk_full = sym.nk_tot
    nelec = wfn.nelec
    cell_volume = wfn.cell_volume
    
    print(f"FFT grid: {fft_grid}")
    print(f"Number of spinor components: {nspinor}")
    print(f"Number of k-points (full zone): {nk_full}")
    print(f"Number of electrons: {nelec}")
    print(f"Cell volume (bohr^3): {cell_volume:.6f}")
    
    # Number of occupied bands (assuming spin-unpolarized/spinor case)
    # For spinor: each band has 2 spinor components but is singly occupied
    # For collinear spin: need to handle nspin
    if nspinor == 2:
        # Spinor case: nelec electrons fill nelec bands
        n_occ = int(nelec)
    else:
        # Collinear or spinless: nelec/2 bands are occupied (factor of 2 from spin)
        n_occ = int(nelec // 2)
    
    # Handle band range (convert from 1-indexed to 0-indexed)
    if band_start is None:
        band_start_idx = 0
    else:
        band_start_idx = int(band_start) - 1  # Convert to 0-indexed
    
    if band_end is None:
        band_end_idx = n_occ
    else:
        band_end_idx = int(band_end)  # 1-indexed end becomes 0-indexed exclusive end
    
    # Clamp start to valid range, but allow end to go beyond n_occ for conduction bands
    band_start_idx = max(0, band_start_idx)
    band_end_idx = min(wfn.nbands, band_end_idx)  # Allow up to total bands in file
    
    print(f"Number of occupied bands: {n_occ}")
    print(f"Total bands in file: {wfn.nbands}")
    print(f"Summing over bands {band_start_idx + 1} to {band_end_idx} (1-indexed)")
    
    # Initialize current density accumulator
    j_r = np.zeros((3, nx, ny, nz), dtype=np.complex128)
    
    # Determine scale factors for bands
    # scale_last_band can be a single value or "band1:scale1,band2:scale2,..." format
    band_scales = {}  # 0-indexed band -> scale factor
    if scale_last_band is not None:
        scale_str = str(scale_last_band)
        if ':' in scale_str:
            # Format: "69:0.01,70:0.01"
            for part in scale_str.split(','):
                band_num, scale = part.split(':')
                band_scales[int(band_num) - 1] = float(scale)  # Convert to 0-indexed
        else:
            # Just a single scale for last band
            band_scales[n_occ - 1] = float(scale_last_band)
        
        for band_idx, scale in sorted(band_scales.items()):
            print(f"Scaling band {band_idx + 1} by {scale}")
    
    # Loop over k-points and selected bands
    print("Computing current density...")
    for ik in range(nk_full):
        if ik % max(1, nk_full // 10) == 0:
            print(f"  Processing k-point {ik+1}/{nk_full}")
        
        for ib in range(band_start_idx, band_end_idx):
            # Get Psi and nabla Psi in real space
            psi_r, grad_psi_r = get_psi_and_grad_psi_realspace(wfn, sym, ib, ik, fft_grid)
            
            # Apply scale factor if specified for this band
            band_weight = band_scales.get(ib, 1.0)
            
            # Current density contribution: j_i = Im[ Psi* nabla_i Psi ]
            # For spinor: sum over spinor components
            # psi_r: (nspinor, nx, ny, nz)
            # grad_psi_r: (3, nspinor, nx, ny, nz)
            for idir in range(3):
                # Sum over spinor components
                for ispin in range(nspinor):
                    j_r[idir] += band_weight * np.conj(psi_r[ispin]) * grad_psi_r[idir, ispin]
    
    # Take imaginary part (the current density is real)
    j_r = np.imag(j_r)
    
    # Normalize by number of k-points for BZ average
    j_r = j_r / nk_full
    
    # For spinor case, no extra factor
    # For collinear/spinless, factor of 2 for spin degeneracy is already in n_occ
    
    print(f"Current density computed!")
    print(f"  Max |j|: {np.max(np.sqrt(np.sum(j_r**2, axis=0))):.6e} a.u.")
    print(f"  Total current (should be ~0): {np.sum(j_r, axis=(1,2,3))}")
    
    # Save to HDF5
    if output_h5 is None:
        output_h5 = os.path.join(os.path.dirname(wfn_file), "current_density.h5")
    
    print(f"Saving to {output_h5}")
    with h5py.File(output_h5, 'w') as f:
        # Current density
        f.create_dataset('current_density', data=j_r, dtype=np.float64)
        
        # Metadata
        f.create_dataset('fft_grid', data=np.array(fft_grid, dtype=np.int32))
        f.create_dataset('cell_volume', data=cell_volume)
        f.create_dataset('avec', data=wfn.avec)
        f.create_dataset('bvec', data=wfn.bvec)
        f.create_dataset('alat', data=wfn.alat)
        f.create_dataset('atom_types', data=wfn.atom_types)
        f.create_dataset('atom_positions', data=wfn.atom_positions)
        
        # Attributes
        f.attrs['description'] = (
            'Ground state current density j(r) in atomic units (1/bohr^2/time). '
            'j_i = Im[sum_{nk} Psi*_nk nabla_i Psi_nk] summed over occupied states.'
        )
        f.attrs['units'] = 'atomic units (1/bohr^2 * hartree/hbar)'
        f.attrs['nk_full'] = nk_full
        f.attrs['n_occ'] = n_occ
        f.attrs['nspinor'] = nspinor
    
    # Generate plots
    if plot_slices:
        if plot_dir is None:
            plot_dir = os.path.dirname(output_h5)
        os.makedirs(plot_dir, exist_ok=True)
        plot_current_slices(j_r, wfn, output_dir=plot_dir)
    
    return j_r


def plot_current_slices(j_r, wfn, output_dir='.', n_slices=3):
    """
    Generate vector plots of current density slices.
    
    Args:
        j_r: Current density (3, nx, ny, nz)
        wfn: WFNReader instance for crystal structure info
        output_dir: Directory to save plots
        n_slices: Number of slices per direction
    """
    print(f"Generating vector plots in {output_dir}")
    
    nx, ny, nz = j_r.shape[1:]
    
    # Real-space grid in crystal coordinates
    fx = np.arange(nx) / nx
    fy = np.arange(ny) / ny
    fz = np.arange(nz) / nz
    
    # Convert to Cartesian coordinates (in bohr)
    avec = wfn.avec * wfn.alat  # Cartesian lattice vectors
    
    # Compute current magnitude for colorscale reference
    j_mag = np.sqrt(j_r[0]**2 + j_r[1]**2 + j_r[2]**2)
    j_max = np.max(j_mag)
    if j_max < 1e-12:
        print("  Current density is essentially zero, skipping plots")
        return
    
    # Colormap setup: use middle portion of coolwarm for better contrast
    from matplotlib.colors import LinearSegmentedColormap
    cmap_base = plt.colormaps['RdYlBu_r']
    # Use range 0.15 to 0.85 of the colormap for better visibility
    colors = cmap_base(np.linspace(0.15, 0.85, 256))
    cmap = LinearSegmentedColormap.from_list('current_cmap', colors)
    
    # Plot XY slices at specific z fractions (0 and 0.05 for 2D materials)
    z_fracs = [0.0, 0.05]
    z_indices = [int(round(zf * nz)) % nz for zf in z_fracs]
    for iz in z_indices:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get slice data
        jx_slice = j_r[0, :, :, iz]
        jy_slice = j_r[1, :, :, iz]
        j_mag_slice = np.sqrt(jx_slice**2 + jy_slice**2)
        
        # Create coordinate grids
        XX, YY = np.meshgrid(fx, fy, indexing='ij')
        
        # Convert to Cartesian (approximate for visualization)
        X_cart = XX * avec[0, 0] + YY * avec[1, 0]
        Y_cart = XX * avec[0, 1] + YY * avec[1, 1]
        
        # Subsample for cleaner quiver plot
        skip = max(1, min(nx, ny) // 15)
        
        # Keep actual vector values (arrow length = magnitude)
        jx_sub = jx_slice[::skip, ::skip]
        jy_sub = jy_slice[::skip, ::skip]
        j_mag_sub = j_mag_slice[::skip, ::skip]
        
        # Quiver plot with arrows scaled by magnitude
        q = ax.quiver(
            X_cart[::skip, ::skip], Y_cart[::skip, ::skip],
            jx_sub, jy_sub,
            j_mag_sub,
            cmap=cmap,
            clim=[0, j_max * 0.8],  # Saturate top 20% for visibility
            angles='xy',
            scale_units='xy',
            scale=j_max * 0.08,  # Larger scale = shorter arrows
            width=0.006,
            headwidth=4,
            headlength=5,
            headaxislength=4,
            minlength=0.02,
        )
        
        # Plot atom positions
        for iat in range(wfn.nat):
            apos = wfn.atom_crys[iat]
            ax_pos = apos[0] * avec[0, 0] + apos[1] * avec[1, 0]
            ay_pos = apos[0] * avec[0, 1] + apos[1] * avec[1, 1]
            ax.scatter(ax_pos, ay_pos, c='#2d3436', s=200, zorder=5, edgecolors='white', linewidths=2)
        
        ax.set_xlabel('x (bohr)', fontsize=12)
        ax.set_ylabel('y (bohr)', fontsize=12)
        ax.set_title(f'Current density (XY slice at z = {fz[iz]:.3f} c)', fontsize=14)
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        cbar = plt.colorbar(q, ax=ax, label='|j| (a.u.)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'current_xy_z{iz:03d}.png'), dpi=150)
        plt.close()
    
    # Plot XZ slices at different y values
    y_indices = np.linspace(0, ny-1, n_slices+2, dtype=int)[1:-1]
    for iy in y_indices:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get slice data
        jx_slice = j_r[0, :, iy, :]
        jz_slice = j_r[2, :, iy, :]
        j_mag_slice = np.sqrt(jx_slice**2 + jz_slice**2)
        
        # Create coordinate grids
        XX, ZZ = np.meshgrid(fx, fz, indexing='ij')
        
        # Convert to Cartesian
        X_cart = XX * avec[0, 0] + ZZ * avec[2, 0]
        Z_cart = XX * avec[0, 2] + ZZ * avec[2, 2]
        
        skip = max(1, min(nx, nz) // 15)
        
        jx_sub = jx_slice[::skip, ::skip]
        jz_sub = jz_slice[::skip, ::skip]
        j_mag_sub = j_mag_slice[::skip, ::skip]
        
        q = ax.quiver(
            X_cart[::skip, ::skip], Z_cart[::skip, ::skip],
            jx_sub, jz_sub,
            j_mag_sub,
            cmap=cmap,
            clim=[0, j_max * 0.8],
            angles='xy',
            scale_units='xy',
            scale=j_max * 0.08,  # Larger scale = shorter arrows
            width=0.006,
            headwidth=4,
            headlength=5,
            headaxislength=4,
            minlength=0.02,
        )
        
        # Plot atom positions
        for iat in range(wfn.nat):
            apos = wfn.atom_crys[iat]
            ax_pos = apos[0] * avec[0, 0] + apos[2] * avec[2, 0]
            az_pos = apos[0] * avec[0, 2] + apos[2] * avec[2, 2]
            ax.scatter(ax_pos, az_pos, c='#2d3436', s=200, zorder=5, edgecolors='white', linewidths=2)
        
        ax.set_xlabel('x (bohr)', fontsize=12)
        ax.set_ylabel('z (bohr)', fontsize=12)
        ax.set_title(f'Current density (XZ slice at y = {fy[iy]:.3f} b)', fontsize=14)
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        cbar = plt.colorbar(q, ax=ax, label='|j| (a.u.)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'current_xz_y{iy:03d}.png'), dpi=150)
        plt.close()
    
    # Plot YZ slices at different x values
    x_indices = np.linspace(0, nx-1, n_slices+2, dtype=int)[1:-1]
    for ix in x_indices:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Get slice data
        jy_slice = j_r[1, ix, :, :]
        jz_slice = j_r[2, ix, :, :]
        j_mag_slice = np.sqrt(jy_slice**2 + jz_slice**2)
        
        # Create coordinate grids
        YY, ZZ = np.meshgrid(fy, fz, indexing='ij')
        
        # Convert to Cartesian
        Y_cart = YY * avec[1, 1] + ZZ * avec[2, 1]
        Z_cart = YY * avec[1, 2] + ZZ * avec[2, 2]
        
        skip = max(1, min(ny, nz) // 15)
        
        jy_sub = jy_slice[::skip, ::skip]
        jz_sub = jz_slice[::skip, ::skip]
        j_mag_sub = j_mag_slice[::skip, ::skip]
        
        q = ax.quiver(
            Y_cart[::skip, ::skip], Z_cart[::skip, ::skip],
            jy_sub, jz_sub,
            j_mag_sub,
            cmap=cmap,
            clim=[0, j_max * 0.8],
            angles='xy',
            scale_units='xy',
            scale=j_max * 0.08,  # Larger scale = shorter arrows
            width=0.006,
            headwidth=4,
            headlength=5,
            headaxislength=4,
            minlength=0.02,
        )
        
        # Plot atom positions
        for iat in range(wfn.nat):
            apos = wfn.atom_crys[iat]
            ay_pos = apos[1] * avec[1, 1] + apos[2] * avec[2, 1]
            az_pos = apos[1] * avec[1, 2] + apos[2] * avec[2, 2]
            ax.scatter(ay_pos, az_pos, c='#2d3436', s=200, zorder=5, edgecolors='white', linewidths=2)
        
        ax.set_xlabel('y (bohr)', fontsize=12)
        ax.set_ylabel('z (bohr)', fontsize=12)
        ax.set_title(f'Current density (YZ slice at x = {fx[ix]:.3f} a)', fontsize=14)
        ax.set_aspect('equal')
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        cbar = plt.colorbar(q, ax=ax, label='|j| (a.u.)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'current_yz_x{ix:03d}.png'), dpi=150)
        plt.close()
    
    print(f"  Generated {3 * n_slices} slice plots")


def main():
    parser = argparse.ArgumentParser(
        description='Calculate ground state current density from WFN.h5',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python current_density.py WFN.h5
  python current_density.py WFN.h5 -o my_current.h5 --no-plot
  python current_density.py WFN.h5 --plot-dir ./plots --n-slices 5
        """
    )
    parser.add_argument('wfn_file', help='Path to WFN.h5 wavefunction file')
    parser.add_argument('-o', '--output', help='Output HDF5 file path')
    parser.add_argument('--no-plot', action='store_true', help='Skip generating plots')
    parser.add_argument('--plot-dir', help='Directory for output plots')
    parser.add_argument('--n-slices', type=int, default=3, help='Number of slices per direction')
    parser.add_argument('--band-start', type=int, help='First band to include (1-indexed, default: 1)')
    parser.add_argument('--band-end', type=int, help='Last band to include (1-indexed, default: all occupied)')
    parser.add_argument('--scale-last-band', type=str, help='Scale factor: single value for last band, or "band1:scale1,band2:scale2" format (1-indexed)')
    
    args = parser.parse_args()
    
    # Resolve paths relative to input file directory [[memory:8089783]]
    input_dir = os.path.dirname(os.path.abspath(args.wfn_file))
    
    wfn_file = args.wfn_file
    if not os.path.isabs(wfn_file):
        wfn_file = os.path.abspath(wfn_file)
    
    output_h5 = args.output
    if output_h5 and not os.path.isabs(output_h5):
        output_h5 = os.path.join(input_dir, output_h5)
    
    plot_dir = args.plot_dir
    if plot_dir and not os.path.isabs(plot_dir):
        plot_dir = os.path.join(input_dir, plot_dir)
    
    compute_current_density(
        wfn_file,
        output_h5=output_h5,
        plot_slices=not args.no_plot,
        plot_dir=plot_dir,
        band_start=args.band_start,
        band_end=args.band_end,
        scale_last_band=args.scale_last_band,
    )


if __name__ == '__main__':
    main()

