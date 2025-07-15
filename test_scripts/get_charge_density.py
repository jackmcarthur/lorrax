import numpy as np
from gpu_utils import cp, xp, GPU_AVAILABLE
import h5py as h5
import datetime

# Generating a reliable charge density is the first step toward self-consistent
# COHSEX iterations.  This routine will later be called repeatedly as the
# quasiparticle wavefunctions are updated.

from wfnreader import WFNReader
import symmetry_maps

wfnpath = 'WFN.h5'
epspath = 'epsmat.h5'
eps0path = 'eps0mat.h5'

def perform_fft_3d(data_1d, gvecs, fft_grid):
    """Transform 1D complex array to real space using an FFT.
    
    Args:
        data_1d: 1D complex array of coefficients (length ngk[ik])
        gvecs: G-vector components for this k-point (ranging from ~ -10 to 10)
        fft_grid: 3D FFT grid dimensions for zero-padding
    """
    # Create 3D shape tuple from FFTgrid
    shape3d = tuple(int(x) for x in fft_grid)
    
    if GPU_AVAILABLE:
        fft_box = cp.zeros(shape3d, dtype=np.complex64)
    else:
        fft_box = np.zeros(shape3d, dtype=np.complex64)

    
    # Convert G-vectors to positive FFT grid indices all at once
    ix = gvecs[:, 0] #% shape3d[0]
    iy = gvecs[:, 1] #% shape3d[1]
    iz = gvecs[:, 2] #% shape3d[2]
    
    # Use advanced indexing to assign all values at once
    fft_box[ix, iy, iz] = data_1d
    
    # Perform inverse FFT using CuPy if available, otherwise NumPy
    # Note: ifftn performs the inverse FFT which is what we want for G-space to real-space
    fft_result = xp.fft.ifftn(fft_box)
    
    return fft_result

def calculate_charge_density(wfn, sym, nval=None, ncond=None):
    """
    Calculate charge density in real space from wavefunctions using WFNReader: goes over all occupied states c_nk(G),
    FFTs them to c_nk(R) (using GPU acceleration when available), squares and sums to get rho(R).
    k-point symmetries are used. The loop order is (nband, nk_irr, n_sym).
    n_sym is done on the GPU since symmetry operations over Gvecs can be parallelized.
    """

    fft_grid = xp.asarray(wfn.fft_grid)
    # Convert fft_grid from numpy array to tuple of integers
    fft_grid_tuple = tuple(int(x) for x in fft_grid) #tuple(2*int(x) for x in fft_grid)
    # NOTE THE TWO! this is zero padding (ecutrho = 4*ecutwfc due to convolution in G-space)
    charge_density = xp.zeros(fft_grid_tuple, dtype=xp.double)
    
    # Loop over bands
    nelec = int(np.sum(wfn.occs[0,0]))
    if ncond is None and nval is None:
        bandrange = range(nelec) # 0 to nelec-1
    elif ncond is not None and nval is None:
        bandrange = range(nelec + ncond) # 0 to nelec+ncond-1
    elif ncond is None and nval is not None:
        bandrange = range(nelec-nval, nelec) # nelec-nval to nelec-1
    elif ncond is not None and nval is not None:
        bandrange = range(nelec-nval, nelec+ncond)

    for ib in bandrange:  # Using first k-point's occupations
        # Loop over k-points
        for ik in range(1): # paper suggests only using k0
            # Get G-vectors for this k-point
            gvecs_k = sym.get_gvecs_kfull(wfn, ik)
            if GPU_AVAILABLE:
                gvecs_k = cp.asarray(gvecs_k)

            # Get wavefunction coefficients for this k-point and band
            coeffs_kb = sym.get_cnk_fullzone(wfn, ib, ik)
            if GPU_AVAILABLE:
                coeffs_kb = cp.asarray(coeffs_kb)

            # Transform each spinor component to real space
            for jspinor in range(2):
                spinor_density = perform_fft_3d(coeffs_kb[jspinor], gvecs_k, fft_grid_tuple)
                charge_density += (spinor_density*xp.conj(spinor_density)).real

    # normalize charge density to n_electrons.
    normrho = np.prod(wfn.fft_grid)#/np.prod(wfn.kgrid)
    if GPU_AVAILABLE:
        charge_density = cp.asarray(normrho) * charge_density
    else:
        charge_density = normrho * charge_density

    return charge_density

def save_charge_density(charge_density):
    """Save the charge density to an HDF5 file."""
    # Convert CuPy array to NumPy array
    charge_density_cpu = charge_density.get()
    
    with h5.File('charge_density.h5', 'w') as f:
        f.create_dataset('charge_density', data=charge_density_cpu)

def analyze_gvectors(gvecs):
    """
    Analyze the range and distribution of G-vectors.
    
    Args:
        gvecs: Array of G-vectors, shape (ngvecs, 3)
    """
    # Transpose to match miller indices format (3, ngvecs)
    gvecs = gvecs.T
    
    # Get min and max for each direction
    min_indices = np.min(gvecs, axis=1)
    max_indices = np.max(gvecs, axis=1)
    

if __name__ == "__main__":
    
    print(f"Beginning charge density calculation. Using {xp.__name__} backend.")
    nval = 5
    ncond = 5
    print(f"Including {ncond if ncond is not None else 'no'} conduction states and {nval if nval is not None else 'all'} valence states.")

    # Initialize WFNReader
    wfn = WFNReader(wfnpath)
    
    # Initialize symmetry maps
    sym = symmetry_maps.SymMaps(wfn)
    
    # Analyze G-vectors before calculation
    print("\nAnalyzing G-vectors from wavefunction file:")
    analyze_gvectors(wfn.gvecs)
    
    # Calculate charge density using the reader
    charge_density = calculate_charge_density(wfn, sym, nval=nval, ncond=ncond)

    print(f"\nTotal electron number: {xp.sum(charge_density)}")

    # Save results
    save_charge_density(charge_density)

    print(f"Charge density saved to charge_density.h5")