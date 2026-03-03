"""Write BSE eigenvectors to HDF5 file in BerkeleyGW format.

Creates an eigenvectors.h5 file compatible with BerkeleyGW's format
as specified in eigenvectors.h5.spec.

For now:
- Assumes spinor wavefunctions (nspin=1, spin_kernel=3)
- Assumes TDA is used
- No Q shifts (single Q=0 point)
"""
from __future__ import annotations
import argparse
import os
from typing import Optional

import h5py
import numpy as np
import jax.numpy as jnp


def write_eigenvectors_h5(
    output_file: str,
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    kpts: np.ndarray,
    n_val: int,
    n_cond: int,
    nkx: int = 1,
    nky: int = 1,
    nkz: int = 1,
    version: int = 1,
    exciton_Q_shifts: Optional[np.ndarray] = None,
) -> None:
    """Write BSE eigenvectors to HDF5 file.
    
    Args:
        output_file: Path to output HDF5 file
        eigenvalues: (n_eig,) exciton energies in Ry
        eigenvectors: (n_eig, nc, nv, nk) or (n_eig, ns, nc, nv, nk) exciton coefficients
                      Complex array with A_cvk amplitudes
        kpts: (nk, 3) k-point coordinates in crystal units
        n_val: Number of valence bands
        n_cond: Number of conduction bands
        nkx, nky, nkz: k-grid dimensions
        version: File version number
        exciton_Q_shifts: (nQ, 3) Q-shift vectors. Default: single Q=0
    """
    eigenvalues = np.asarray(eigenvalues)
    eigenvectors = np.asarray(eigenvectors)
    kpts = np.asarray(kpts)
    
    n_eig = eigenvalues.shape[0]
    
    # Handle eigenvector shape - add spin dimension if missing
    # Expected input: (n_eig, nc, nv, nk) or (n_eig, ns, nc, nv, nk)
    if eigenvectors.ndim == 4:
        # (n_eig, nc, nv, nk) -> (n_eig, ns=1, nc, nv, nk)
        eigenvectors = eigenvectors[:, np.newaxis, :, :, :]
    
    n_eig_check, ns, nc, nv, nk = eigenvectors.shape
    assert n_eig_check == n_eig, f"Mismatch: {n_eig_check} vs {n_eig} eigenvalues"
    assert nc == n_cond, f"Mismatch: nc={nc} vs n_cond={n_cond}"
    assert nv == n_val, f"Mismatch: nv={nv} vs n_val={n_val}"
    
    # Default Q-shift: single Q=0
    if exciton_Q_shifts is None:
        exciton_Q_shifts = np.zeros((1, 3), dtype=np.float64)
    nQ = exciton_Q_shifts.shape[0]
    
    # BSE Hamiltonian size (TDA): ns * nk * nv * nc
    bse_hamiltonian_size = ns * nk * nv * nc
    evec_sz = bse_hamiltonian_size  # TDA: same as Hamiltonian size
    
    # Determine flavor (1 = real, 2 = complex)
    is_complex = np.iscomplexobj(eigenvectors)
    flavor = 2 if is_complex else 1
    
    # Spinor calculation: spin_kernel = 3
    spin_kernel = 3
    
    # Create k-points array in Fortran order (3, nk)
    kpts_fortran = kpts.T.copy()  # (3, nk)
    
    # Prepare eigenvectors in Fortran storage order
    # Spec says dims are: [2 if complex], ns, nv, nc, nk, nevecs, nQ
    # But we're in Python so we store in C order and HDF5 handles it
    # Shape for storage: (nQ, nevecs, nk, nc, nv, ns) or (nQ, nevecs, nk, nc, nv, ns, 2)
    
    # Current: (n_eig, ns, nc, nv, nk)
    # Need: (nQ, nevecs, nk, nc, nv, ns) for real
    #    or (nQ, nevecs, nk, nc, nv, ns, 2) for complex
    
    # Add nQ dimension (at the front for now, we'll transpose)
    # eigenvectors shape: (n_eig, ns, nc, nv, nk)
    # Transpose to: (n_eig, nk, nc, nv, ns)
    evecs_reordered = eigenvectors.transpose(0, 4, 2, 3, 1)  # (n_eig, nk, nc, nv, ns)
    
    # Add Q dimension
    evecs_with_Q = evecs_reordered[np.newaxis, ...]  # (1, n_eig, nk, nc, nv, ns)
    # Transpose to Fortran-like order for spec: (nQ, nevecs, nk, nc, nv, ns)
    # Actually, keep as is since HDF5 will handle the ordering
    
    if is_complex:
        # Split into real/imag for storage: (..., 2)
        evecs_storage = np.stack([evecs_with_Q.real, evecs_with_Q.imag], axis=-1)
    else:
        evecs_storage = evecs_with_Q.real
    
    # Write HDF5 file
    with h5py.File(output_file, 'w') as f:
        # Create empty header groups (to be filled by user if needed)
        f.create_group('mf_header')
        f.create_group('eps_header')
        f.create_group('bse_header')
        
        # exciton_header group
        exciton_header = f.create_group('exciton_header')
        exciton_header.create_dataset('version', data=version)
        exciton_header.create_dataset('flavor', data=flavor)
        
        # params subgroup
        params = exciton_header.create_group('params')
        params.create_dataset('bse_hamiltonian_size', data=bse_hamiltonian_size)
        params.create_dataset('evec_sz', data=evec_sz)
        params.create_dataset('spin_kernel', data=spin_kernel)
        params.create_dataset('nevecs', data=n_eig)
        params.create_dataset('ns', data=ns)
        params.create_dataset('nc', data=nc)
        params.create_dataset('nv', data=nv)
        params.create_dataset('use_tda', data=1)  # TDA is used
        
        # kpoints subgroup
        kpoints = exciton_header.create_group('kpoints')
        kpoints.create_dataset('nk', data=nk)
        kpoints.create_dataset('kpts', data=kpts_fortran)  # (3, nk)
        kpoints.create_dataset('nQ', data=nQ)
        kpoints.create_dataset('exciton_Q_shifts', data=exciton_Q_shifts.T)  # (3, nQ)
        
        # exciton_data group
        exciton_data = f.create_group('exciton_data')
        exciton_data.create_dataset('eigenvalues', data=eigenvalues)
        exciton_data.create_dataset('eigenvectors', data=evecs_storage)
        
        # Note: eigenvectors_left, eigenvectors_deexcitation, etc. 
        # are only needed for non-TDA calculations
    
    print(f"Wrote {n_eig} eigenvectors to {output_file}")
    print(f"  Dimensions: ns={ns}, nc={nc}, nv={nv}, nk={nk}, nQ={nQ}")
    print(f"  Hamiltonian size: {bse_hamiltonian_size}")
    print(f"  Flavor: {'complex' if flavor == 2 else 'real'}")
    ryd2ev = 13.6056980659
    print(f"  Eigenvalue range: [{eigenvalues.min():.6f}, {eigenvalues.max():.6f}] Ry")
    print(f"                  = [{eigenvalues.min() * ryd2ev:.4f}, {eigenvalues.max() * ryd2ev:.4f}] eV")


def generate_kpts_grid(nkx: int, nky: int, nkz: int) -> np.ndarray:
    """Generate a Monkhorst-Pack style k-point grid.
    
    Args:
        nkx, nky, nkz: Grid dimensions
        
    Returns:
        kpts: (nk, 3) array of k-points in crystal coordinates [0, 1)
    """
    kpts = []
    for ix in range(nkx):
        for iy in range(nky):
            for iz in range(nkz):
                kx = ix / nkx
                ky = iy / nky
                kz = iz / nkz if nkz > 0 else 0.0
                kpts.append([kx, ky, kz])
    return np.array(kpts, dtype=np.float64)


def main():
    """Command-line interface for writing eigenvectors."""
    parser = argparse.ArgumentParser(
        description="Write BSE eigenvectors to HDF5 (BerkeleyGW format)"
    )
    parser.add_argument(
        'input_npz',
        help="Input .npz file with 'eigenvalues' and 'eigenvectors' arrays"
    )
    parser.add_argument(
        '-o', '--output',
        default='eigenvectors.h5',
        help="Output HDF5 file (default: eigenvectors.h5)"
    )
    parser.add_argument(
        '--n-val', type=int, required=True,
        help="Number of valence bands"
    )
    parser.add_argument(
        '--n-cond', type=int, required=True,
        help="Number of conduction bands"
    )
    parser.add_argument(
        '--nkx', type=int, default=1,
        help="k-grid dimension x"
    )
    parser.add_argument(
        '--nky', type=int, default=1,
        help="k-grid dimension y"
    )
    parser.add_argument(
        '--nkz', type=int, default=1,
        help="k-grid dimension z"
    )
    args = parser.parse_args()
    
    # Load input data
    data = np.load(args.input_npz)
    eigenvalues = data['eigenvalues']
    eigenvectors = data['eigenvectors']
    
    # Generate k-points
    kpts = generate_kpts_grid(args.nkx, args.nky, args.nkz)
    
    # Write output
    write_eigenvectors_h5(
        args.output,
        eigenvalues,
        eigenvectors,
        kpts,
        n_val=args.n_val,
        n_cond=args.n_cond,
        nkx=args.nkx,
        nky=args.nky,
        nkz=args.nkz,
    )


if __name__ == '__main__':
    main()

