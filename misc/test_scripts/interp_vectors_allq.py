import numpy as np
from gpu_utils import cp, xp, fftx, GPU_AVAILABLE
import h5py
from wfnreader import WFNReader
#from get_charge_density import perform_fft_3d
import symmetry_maps
#import matplotlib.pyplot as plt


def fft_bandrange(wfn, sym, bandrange, centroid_indices, is_left, psi_rtot_out, xp=cp):
    """Process wavefunctions for all k-points in the full Brillouin zone.
    
    Args:
        wfn: WFNReader object containing wavefunctions
        sym: SymMaps object for symmetry operations
        bandrange: Tuple (start, end) for band range
        centroid_indices: Array of centroid positions
        is_left: Bool indicating if psi = psi_l (gets)
        xp: numpy or cupy module
    
    Returns:
        psi_rtot_out: Array of real-space wavefunctions for all k-points
        psi_rmu_out: Array of centroid values for all k-points
    """
    # Get dimensions
    nb = bandrange[1] - bandrange[0]
    n_rtot = int(xp.prod(wfn.fft_grid))
    n_rmu = int(centroid_indices.shape[0])
    
    # Initialize temporary arrays
    psi_rtot = xp.zeros((nb, 2, *wfn.fft_grid), dtype=xp.complex128)

    #########################################
        # Initialize exp(iGr) phase factor arrays outside kq loops
    fft_nx, fft_ny, fft_nz = wfn.fft_grid
    fx = xp.arange(fft_nx, dtype=float)[None, :, None, None] / fft_nx  # Shape: (nx,1,1)
    fy = xp.arange(fft_ny, dtype=float)[None, None, :, None] / fft_ny  # Shape: (1,ny,1)
    fz = xp.arange(fft_nz, dtype=float)[None, None, None, :] / fft_nz  # Shape: (1,1,nz)

    # Pre-allocate phase arrays
    px = xp.zeros((1,fft_nx, 1, 1), dtype=xp.complex128)
    py = xp.zeros((1,1, fft_ny, 1), dtype=xp.complex128)
    pz = xp.zeros((1,1, 1, fft_nz), dtype=xp.complex128)
    #########################################
    
    # Loop over all k-points in full BZ
    for k_idx in range(sym.nk_tot):
        # Get reduced k-point index and symmetry operation
        # note these both take the unfolded k-point index
        k_red = sym.irk_to_k_map[k_idx]
        sym_op = sym.irk_sym_map[k_idx]
        
        # Get G-vectors and rotate them
        #gvecs_k = xp.asarray(wfn.get_gvec_nk(k_red))
        gvecs_k_rot = xp.asarray(sym.get_gvecs_kfull(wfn,k_idx))
        #xp.einsum('ij,kj->ki', sym.Rinv_grid[sym_op], gvecs_k)
        
        # Initialize G-space wavefunctions
        psi_Gspace = xp.zeros((nb, 2, wfn.ngk[k_red]), dtype=xp.complex128)
        
        # Get wavefunction coefficients
        for ib, band_idx in enumerate(range(bandrange[0], bandrange[1])):
            psi_Gspace[ib, :, :] = xp.asarray(sym.get_cnk_fullzone(wfn,band_idx,k_idx))
        
        # Rotate spinor components
        #psi_Gspace = xp.einsum('ij,bjk->bik', sym.U_spinor[sym_op], psi_Gspace)
        
        # FFT to real space
        psi_rtot.fill(0)
        for ib in range(nb):
            for ispinor in range(2):
                # Place G-space coefficients
                psi_rtot[ib,ispinor,gvecs_k_rot[:,0],gvecs_k_rot[:,1],gvecs_k_rot[:,2]] = psi_Gspace[ib,ispinor,:]
                # Perform FFT
                psi_rtot[ib,ispinor] = fftx.fft.ifftn(psi_rtot[ib,ispinor])
        
        # Normalize
        psi_rtot *= xp.sqrt(n_rtot)

        #####################################
        xp.exp(-2j * xp.pi * float(sym.unfolded_kpts[k_idx][0]) * fx, out=px)
        xp.exp(-2j * xp.pi * float(sym.unfolded_kpts[k_idx][1]) * fy, out=py)
        xp.exp(-2j * xp.pi * float(sym.unfolded_kpts[k_idx][2]) * fz, out=pz)
        psi_rtot *= px
        psi_rtot *= py
        psi_rtot *= pz
        #####################################
        
        
        # Store results with new ordering
        if is_left:
            psi_rtot_out[k_idx] = xp.conj(psi_rtot)
        else:
            psi_rtot_out[k_idx] = psi_rtot


def fft_bandrange_psimu(wfn, sym, bandrange, centroid_indices, psi_rmu_out, xp=cp):
    """Process wavefunctions for all k-points in the full Brillouin zone, return psi_nk(r_mu)
    for ideal memory layout, shape psi_rmu_out: (nkpts, nrmu, nspinor, nbands)
    psi_rmu_out shape: (nkpts, nbands, nspinor, nrmu)
    """
    print("Doing FFTs for all wfns")
    # Get dimensions
    nb = bandrange[1] - bandrange[0]
    n_rtot = int(xp.prod(wfn.fft_grid))
    n_rmu = int(centroid_indices.shape[0])
    
    # Initialize temporary arrays
    psi_rtot = xp.zeros((nb, 2, *wfn.fft_grid), dtype=xp.complex128)
    
    # Loop over all k-points in full BZ
    for k_idx in range(sym.nk_tot):
        # Get reduced k-point index and symmetry operation
        # note these both take the unfolded k-point index
        k_red = sym.irk_to_k_map[k_idx]
        sym_op = sym.irk_sym_map[k_idx]
        
        # Get G-vectors and rotate them
        #gvecs_k = xp.asarray(wfn.get_gvec_nk(k_red))
        gvecs_k_rot = xp.asarray(sym.get_gvecs_kfull(wfn,k_idx))
        #xp.einsum('ij,kj->ki', sym.Rinv_grid[sym_op], gvecs_k)
        
        # Initialize G-space wavefunctions
        psi_Gspace = xp.zeros((nb, 2, wfn.ngk[k_red]), dtype=xp.complex128)
        
        # Get wavefunction coefficients
        for ib, band_idx in enumerate(range(bandrange[0], bandrange[1])):
            psi_Gspace[ib, :, :] = xp.asarray(sym.get_cnk_fullzone(wfn,band_idx,k_idx))
        
        # FFT to real space
        psi_rtot.fill(0)
        psi_Gspace = psi_Gspace.reshape(nb*2,-1)
        psi_rtot = psi_rtot.reshape(nb*2, *wfn.fft_grid)

        for ibspin in range(nb*2):
            # Place G-space coefficients
            psi_rtot[ibspin,gvecs_k_rot[:,0],gvecs_k_rot[:,1],gvecs_k_rot[:,2]] = psi_Gspace[ibspin]
            # Perform FFT
            psi_rtot[ibspin] = fftx.fft.ifftn(psi_rtot[ibspin])
        
        # Normalize
        psi_rtot *= xp.sqrt(n_rtot)
        
        # Store results
        psi_rmu_out[k_idx,:] = psi_rtot[:,centroid_indices[:,0],centroid_indices[:,1],centroid_indices[:,2]].reshape(nb,2,-1)
    print("FFTs complete")

def get_interp_vectors_allq(wfn, wfnq, sym, centroid_indices, bandrange_l, bandrange_r, xp):
    """Find the interpolative separable density fitting representation."""
    # Get dimensions
    n_rtot = int(np.prod(wfn.fft_grid))
    n_rmu = int(centroid_indices.shape[0])
    nb_l = bandrange_l[1] - bandrange_l[0]
    nb_r = bandrange_r[1] - bandrange_r[0]
    
    # Initialize output arrays with (nk, nb) ordering
    psi_l_rtot_out = xp.zeros((sym.nk_tot, nb_l, 2, *wfn.fft_grid), dtype=xp.complex128)
    psi_r_rtot_out = xp.zeros((sym.nk_tot, nb_r, 2, *wfn.fft_grid), dtype=xp.complex128)
    psi_l_rmu_out = xp.zeros((sym.nk_tot, nb_l, 2*n_rmu), dtype=xp.complex128)
    psi_r_rmu_out = xp.zeros((sym.nk_tot, nb_r, 2*n_rmu), dtype=xp.complex128)
    
    # Initialize temporary arrays for processing
    # notice combined band/spinor index
    psi_l_rtot = xp.zeros((nb_l*2, *wfn.fft_grid), dtype=xp.complex128)
    psi_r_rtot = xp.zeros((nb_r*2, *wfn.fft_grid), dtype=xp.complex128)
    psi_l_rmu = xp.zeros((nb_l*2, n_rmu), dtype=xp.complex128)
    psi_r_rmu = xp.zeros((nb_r*2, n_rmu), dtype=xp.complex128)
    
    # Initialize ZC^T and CC^T
    Z_C_T = xp.zeros((n_rtot, n_rmu), dtype=xp.complex128)
    C_C_T = xp.zeros((n_rmu, n_rmu), dtype=xp.complex128)
    zeta_q = xp.zeros((sym.nk_tot, n_rmu, n_rtot), dtype=xp.complex128)

    # Initialize exp(iGr) phase factor arrays outside kq loops
    fft_nx, fft_ny, fft_nz = wfn.fft_grid
    fx = xp.arange(fft_nx, dtype=float)[None, :, None, None] / fft_nx  # Shape: (nx,1,1)
    fy = xp.arange(fft_ny, dtype=float)[None, None, :, None] / fft_ny  # Shape: (1,ny,1)
    fz = xp.arange(fft_nz, dtype=float)[None, None, None, :] / fft_nz  # Shape: (1,1,nz)

    # Pre-allocate phase arrays
    px = xp.zeros((1,fft_nx, 1, 1), dtype=xp.complex128)
    py = xp.zeros((1,1, fft_ny, 1), dtype=xp.complex128)
    pz = xp.zeros((1,1, 1, fft_nz), dtype=xp.complex128)
    
    # fill psi_l/r_rtot_out with respective psi(*)_l/r(r) for all k
    print(f"Performing FFTs for wavefunction ranges {bandrange_l} and {bandrange_r}")
    fft_bandrange(wfn, sym, bandrange_l, centroid_indices, True, psi_l_rtot_out, xp=cp)
    fft_bandrange(wfn, sym, bandrange_r, centroid_indices, False, psi_r_rtot_out, xp=cp)
    print("FFTs complete")
    #fft_bandrange(wfnq, sym, bandrange_r, centroid_indices, False, psi_r_rtot_out, xp=cp)
    
    for iq in range(sym.nk_tot):
        Z_C_T.fill(0.0+0.0j)
        C_C_T.fill(0.0+0.0j)

        #if iq == 1:
            # wfnq used for q=0, now generate psi_r_rtot_out for non-shifted wfns
            # not needed because no shifted wfnq used in sigma.x?
            #fft_bandrange(wfnq, sym, bandrange_r, centroid_indices, False, psi_r_rtot_out, xp=cp)
        
        for k_r in range(sym.nk_tot):
            k_l_1bz = sym.kqfull_map[k_r, iq] # this is just the kpoint id, but k_l_full is a vector
            k_l_full = xp.asarray(sym.unfolded_kpts[k_r] - sym.unfolded_kpts[iq])
            # Replace values very close to integers with those integers
            rounded = xp.round(k_l_full)
            k_l_full = xp.where(xp.abs(k_l_full - rounded) < 1e-8, rounded, k_l_full)
            #if iq == 3:
            #    print(sym.unfolded_kpts[k_l_1bz])
            #    print(k_l_full)


            # important! the psi's here store u_nk(r) for k in 1BZ but k-q can be outside the 1BZ.
            # if this is the case, we need to get the full u_nk-q(r) from the stored u_nk-q+G(r).
            # since psiprod = psi^*_nk-q(r) psi_nk(r) = e^{iqr} u^*_nk-q(r) u_nk(r) only for the real k-q, not k-q+G.
            # to fix this, (map u_nk-q+G(r) -> u_nk-q(r)), we use the phase factor e^(iG.r).
            Gkk = xp.asarray(k_l_full%1.0 - k_l_full, dtype=float)
            # Calculate phase factors
            xp.exp(-2j * xp.pi * float(Gkk[0]) * fx, out=px)
            xp.exp(-2j * xp.pi * float(Gkk[1]) * fy, out=py)
            xp.exp(-2j * xp.pi * float(Gkk[2]) * fz, out=pz)

            psi_l_rtot = psi_l_rtot.reshape(nb_l*2,*wfn.fft_grid)
            psi_r_rtot = psi_r_rtot.reshape(nb_r*2,*wfn.fft_grid)
            
            # apply phase factors to get the real psi_l
            psi_l_rtot[:] = psi_l_rtot_out[k_l_1bz].reshape(nb_l*2,*wfn.fft_grid)
            psi_l_rtot *= px
            psi_l_rtot *= py
            psi_l_rtot *= pz
            psi_r_rtot[:] = psi_r_rtot_out[k_r].reshape(nb_r*2,*wfn.fft_grid)

            psi_l_rmu = psi_l_rtot[:,centroid_indices[:,0],centroid_indices[:,1],centroid_indices[:,2]]#.reshape(nb_l*2, -1)
            psi_r_rmu = psi_r_rtot[:,centroid_indices[:,0],centroid_indices[:,1],centroid_indices[:,2]]#.reshape(nb_r*2, -1)

            psi_l_rtot = psi_l_rtot.reshape(nb_l*2, -1)
            psi_r_rtot = psi_r_rtot.reshape(nb_r*2, -1)
            
            # Add contribution from this k,q pair to ZC^T and CC^T
            # combine band and spinor indices, so that decomp is of form \sum_\mu \zeta^q_\mu(r) \psi^*_mk-q,a(r_\mu) \psi_nk,b(r_\mu)
            # a.k.a. all four possible M_sigma,sigma' combinations are included
            # in the future it may be good to do this with a cuBLAS call, but that's nontrivial w/cupy
            #P_l = xp.einsum('mi,mj->ij', psi_l_rtot.reshape(nb_l*2, -1), psi_l_rmu.reshape(nb_l*2, -1))
            P_l = xp.matmul(psi_l_rtot.T, psi_l_rmu)
            P_r = xp.matmul(psi_r_rtot.T, psi_r_rmu)
            Z_C_T += P_l * P_r
            
            Pmu_l = xp.matmul(psi_l_rmu.T, psi_l_rmu)
            Pmu_r = xp.matmul(psi_r_rmu.T, psi_r_rmu)
            C_C_T += Pmu_l * Pmu_r
        
        # Solve for zeta_q
        # C_C_T shape (nrmu, nrmu), Z_C_T shape (nrmu, nrtot)
        # zeta_q shape (nq, nrmu, nrtot) 
        # (in this order so that FFT's of each zeta_mu to G space can be done w/contiguous memory)
        print(f"qpoint {iq}")
        zeta_q[iq,:,:] = xp.linalg.lstsq(C_C_T.T, Z_C_T.T, rcond=-1)[0]
        #zeta_q[iq,:,:] = xp.linalg.solve(C_C_T.T, Z_C_T.T)

    zeta_q = zeta_q.reshape(sym.nk_tot, n_rmu,*wfn.fft_grid)

    psi_l_rmu_out = psi_l_rtot_out[:,:,:,centroid_indices[:,0],centroid_indices[:,1],centroid_indices[:,2]]
    psi_r_rmu_out = psi_r_rtot_out[:,:,:,centroid_indices[:,0],centroid_indices[:,1],centroid_indices[:,2]]

    return zeta_q, xp.conj(psi_l_rmu_out), psi_r_rmu_out, xp.conj(psi_l_rtot_out), psi_r_rtot_out

if __name__ == "__main__":
    # Initialize WFNReader and SymMaps
    wfn = WFNReader('WFN.h5')
    wfnq = WFNReader('WFNq.h5')
    sym = symmetry_maps.SymMaps(wfn)
    
    # Load centroids
    centroids_frac = np.loadtxt('centroids_frac.txt')
    n_centroids = len(centroids_frac)
    n_rmu = n_centroids
    
    # Convert to indices with periodic boundary conditions
    raw_indices = np.round(centroids_frac * wfn.fft_grid)
    centroid_indices = raw_indices.astype(int)
    # Replace any index equal to the grid size with 0 (periodic boundary)
    for i in range(3):
        centroid_indices[centroid_indices[:, i] == wfn.fft_grid[i], i] = 0

        
    # Get interpolation vectors (using the already-computed indices)
    bandrange_l = (27, 33)
    bandrange_r = (21, 26)
    k_idx = 1  # index of k-point in fullBZ
    q_idx = 0  # index of q-point in fullBZ
    spinl = 0
    spinr = 0
    kminusq_idx = 1 #sym.kq_map[k_idx, q_idx]

    #zeta_q = get_interp_vectors(wfn, sym, centroid_indices, bandrange_l, bandrange_r, xp)
    zeta_q, psi_l_rmu_out, psi_r_rmu_out, psi_l_rtot_out, psi_r_rtot_out = get_interp_vectors_allq(wfn, wfnq, sym, centroid_indices, bandrange_l, bandrange_r, xp)
    
    # After zeta_q calculation is complete
    # with h5py.File(f'zeta_q_{n_centroids}.h5', 'w') as f:
    #     f.create_dataset('zeta_q', data=zeta_q.get() if xp == cp else zeta_q)
    #     f.attrs['n_centroids'] = n_centroids
    #     f.attrs['fft_grid'] = wfn.fft_grid
    #     f.attrs['bandrange_l'] = bandrange_l
    #     f.attrs['bandrange_r'] = bandrange_r

    #print(psi_l_rmu_out.shape)
    psi_l_rmu_out = psi_l_rmu_out.get()
    psi_r_rmu_out = psi_r_rmu_out.get()
    psi_l_rtot_out = psi_l_rtot_out.get()
    psi_r_rtot_out = psi_r_rtot_out.get()

    # Create a grid of projections for bands 25-30
    print("\nProjection grid for bands in range, <M_ij|M_ij_approx>/<M_ij|M_ij>:")
    print("      ", end="")
    for j in range(bandrange_r[0], bandrange_r[0]+5):
        print(f"j={j:<6}", end="")
    print("\n" + "-" * 50)

    for i in range(5):
        print(f"i={i+bandrange_l[0]:<2} |", end=" ")
        for j in range(5):
            # Compute exact M_ia,jb(r)
            M_exact = np.zeros(wfn.fft_grid, dtype=np.complex128)
            M_exact = np.conj(psi_l_rtot_out[kminusq_idx,i,spinl]) * psi_r_rtot_out[k_idx,j,spinr]
            
            # Compute interpolated M_ia,jb(r)
            M_interp = np.zeros_like(M_exact)
            
            # Step 1: get zeta_q for this q-point
            zeta_qmu = zeta_q[q_idx].get()

            # Compute psi*_mk-q psi_nk(spinor,rmu) for all ispinor
            # Shape: (ispinor=2, mu=300)
            psi_prod_mu = np.conj(psi_l_rmu_out[kminusq_idx,i,spinl]) * psi_r_rmu_out[k_idx,j,spinr]

            # do \sum_\mu zeta^q_\mu psi^*_l(r_\mu) psi_r(r_\mu)
            M_interp = np.einsum('i...,i->...', zeta_qmu, psi_prod_mu)
            #np.einsum('j...im,im->i...', zeta_reshaped, psi_prod_mu)
            # Calculate projection
            overlap = np.sum(np.conj(M_exact) * M_interp)
            norm = np.sum(np.abs(M_exact)**2)
            projection = np.real(overlap / norm)
            print(f"{projection:.4f}", end="  ")
        print()


# to-do tomorrow:
# wfnq used for q=0 zeta_q
# v_mu,nu
# sigma