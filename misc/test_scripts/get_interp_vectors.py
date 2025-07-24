import numpy as np
from gpu_utils import cp, xp, fftx, GPU_AVAILABLE
import h5py
from wfnreader import WFNReader
from get_charge_density import perform_fft_3d
import symmetry_maps
import matplotlib.pyplot as plt

def build_interp_array(wfn, sym, centroid_indices):
    """Build array of wavefunction values at centroids for all bands and k-points."""
    
    # Get dimensions
    nk_red = wfn.nkpts
    nb = 41 #wfn.nbands
    n_centroids = len(centroid_indices)
    nk_tot = np.prod(wfn.kgrid)
    max_ngk = max(wfn.ngk)
    
    # Initialize output arrays
    psi_values = np.zeros((nk_tot, nb, 2, n_centroids), dtype=np.complex128)
    psi_real_space = np.zeros((nk_tot, nb, 2, *wfn.fft_grid), dtype=np.complex128)
    
    # Move to GPU if available
    if GPU_AVAILABLE:
        sym.R_grid = cp.asarray(sym.R_grid)
        sym.Rinv_grid = cp.asarray(sym.Rinv_grid)
        sym.U_spinor = cp.asarray(sym.U_spinor)
    
    # Calculate FFT normalization factor
    N = cp.prod(wfn.fft_grid)  # Total number of grid points
    
    # Loop over reduced k-points
    for ik in range(nk_red):
        # Get G-vectors for this k-point
        gvecs_k = wfn.get_gvec_nk(ik)
        if GPU_AVAILABLE:
            gvecs_k = cp.asarray(gvecs_k)
        
        # Get symmetry operations for this k-point
        sym_ops = sym.irk_sym_map[ik]
        k_indices = sym.irk_to_k_map[ik]
        
        # Loop over symmetry operations
        for isym_idx, (isym, k_idx) in enumerate(zip(sym_ops, k_indices)):
            # Apply symmetry operation to G-vectors
            gvecs_k_rot = cp.einsum('ij,kj->ki', sym.Rinv_grid[isym], gvecs_k)
            
            if ik == 0 and isym_idx == 0:
                print(f"Processing k-points with {len(gvecs_k)} G-vectors")
            
            # Loop over bands
            for ib in range(nb):
                # Get coefficients for this band
                coeffs_kb = wfn.get_cnk(ik, ib)
                if GPU_AVAILABLE:
                    coeffs_kb = cp.asarray(coeffs_kb)
                
                # Apply symmetry operation to coefficients
                coeffs_kb_rot = cp.einsum('ij,jk->ik', sym.U_spinor[isym], coeffs_kb)
                
                # Calculate real-space wavefunction for both spinor components
                for ispinor in range(2):
                    # Perform FFT to get real-space wavefunction
                    psi_r = perform_fft_3d(coeffs_kb_rot[ispinor], gvecs_k_rot, wfn.fft_grid)
                    
                    # Normalize for FFT scaling
                    psi_r *= cp.sqrt(N) 
                    
                    # Store full real-space wavefunction
                    psi_real_space[k_idx, ib, ispinor] = psi_r.get() if GPU_AVAILABLE else psi_r
                    
                    # Extract values at centroid positions
                    psi_values[k_idx, ib, ispinor] = psi_r[
                        centroid_indices[:, 0],
                        centroid_indices[:, 1],
                        centroid_indices[:, 2]
                    ].get() if GPU_AVAILABLE else psi_r[
                        centroid_indices[:, 0],
                        centroid_indices[:, 1],
                        centroid_indices[:, 2]
                    ]
    
    # Save to HDF5 file
    with h5py.File('psi_centroids.h5', 'w') as f:
        f.create_dataset('psi_values', data=psi_values)
        f.create_dataset('psi_real_space', data=psi_real_space)
        f.create_dataset('centroids_frac', data=centroids_frac)
        f.create_dataset('centroid_indices', data=centroid_indices)
        f.attrs['nk_tot'] = nk_tot
        f.attrs['nk_red'] = nk_red
        f.attrs['nb'] = nb
        f.attrs['n_centroids'] = n_centroids
        f.attrs['kgrid'] = wfn.kgrid
        f.attrs['fft_grid'] = wfn.fft_grid

def get_interp_vectors(wfn, sym, centroid_indices, bandrange_l, bandrange_r, xp):
    """Find the interpolative separable density fitting representation.
    
    Computes M_mn(k,q,r) = psi_mk+q^*(r) psi_nk(r) ~= sum_\mu psi_mk+q^*(r_\mu) psi_nk(r_\mu) \zeta_\mu(r)
    where \zeta_\mu(r) are the interpolation vectors.
    
    Args:
        wfn: WFNReader object
        sym: SymMaps object
        centroid_indices: Indices of centroids in FFT grid
        bandrange_l: Tuple (start, end) for left bands (e.g., valence)
        bandrange_r: Tuple (start, end) for right bands (e.g., conduction)
    """
    # Get dimensions
    n_rtot = int(np.prod(wfn.fft_grid))  # Total number of real-space points
    n_rmu = int(centroid_indices.shape[0])    # Number of centroid points
    nb_l = bandrange_l[1] - bandrange_l[0]  # Number of left bands
    nb_r = bandrange_r[1] - bandrange_r[0]  # Number of right bands
    # Load wavefunctions from psi_centroids.h5
    with h5py.File('psi_centroids.h5', 'r') as f:
        psi_real_space = xp.asarray(f['psi_real_space'][:])  # (nk, nb, 2, nx, ny, nz)
        psi_values = xp.asarray(f['psi_values'][:])          # (nk, nb, 2, n_rmu)
    
    # Extract and reshape left wavefunctions (psi_l)
    psi_l_rtot = xp.zeros((nb_l, 2*n_rtot), dtype=np.complex128)
    psi_l_rmu = xp.zeros((nb_l, 2*n_rmu), dtype=np.complex128)
    
    # Extract and reshape right wavefunctions (psi_r)
    psi_r_rtot = xp.zeros((nb_r, 2*n_rtot), dtype=np.complex128)
    psi_r_rmu = xp.zeros((nb_r, 2*n_rmu), dtype=np.complex128)
    
    # Fixed k-point (k=0 as specified in comments)
    k_l = 3
    k_r = 3
    
    # Fill left wavefunctions
    for ib, band_idx in enumerate(range(bandrange_l[0], bandrange_l[1])):
        # Reshape real-space wavefunction to combine spatial dimensions
        psi_l_rtot[ib, :n_rtot] = xp.conj(psi_real_space[k_l, band_idx, 0].reshape(-1))
        psi_l_rtot[ib, n_rtot:] = xp.conj(psi_real_space[k_l, band_idx, 1].reshape(-1))
        
        # Get centroid values
        psi_l_rmu[ib, :n_rmu] = xp.conj(psi_values[k_l, band_idx, 0])
        psi_l_rmu[ib, n_rmu:] = xp.conj(psi_values[k_l, band_idx, 1])
    
    # Fill right wavefunctions
    for ib, band_idx in enumerate(range(bandrange_r[0], bandrange_r[1])):
        psi_r_rtot[ib, :n_rtot] = psi_real_space[k_r, band_idx, 0].reshape(-1)
        psi_r_rtot[ib, n_rtot:] = psi_real_space[k_r, band_idx, 1].reshape(-1)
        
        psi_r_rmu[ib, :n_rmu] = psi_values[k_r, band_idx, 0]
        psi_r_rmu[ib, n_rmu:] = psi_values[k_r, band_idx, 1]

    print(f"els of psi_lmu {psi_l_rtot[0,:4]}")

    
    # Move to GPU if available
    if GPU_AVAILABLE:
        xp = cp
        psi_l_rtot = cp.asarray(psi_l_rtot)
        psi_l_rmu = cp.asarray(psi_l_rmu)
        psi_r_rtot = cp.asarray(psi_r_rtot)
        psi_r_rmu = cp.asarray(psi_r_rmu)
    else:
        xp = np
    
    # Step 1: Compute quasi-density matrices P_phi and P_psi (Equation 15)
    P_phi = xp.einsum('mi,mj->ij', psi_l_rtot, psi_l_rmu)  # (2*n_rtot x n_lmu)
    P_psi = xp.einsum('mi,mj->ij', psi_r_rtot, psi_r_rmu)   # (2*n_rtot x n_rmu)
    
    # Step 2: Compute ZC^T (Equation 14)
    Z_C_T = P_phi * P_psi  # (n_rtot x n_rmu)

    Pmu_phi = xp.einsum('mi,mj->ij', psi_l_rmu, psi_l_rmu)  # (n_rmu x n_rmu)
    Pmu_psi = xp.einsum('mi,mj->ij', psi_r_rmu, psi_r_rmu)  # (n_rmu x n_rmu)
    
    # Step 3: Compute CC^T (Equation 16)
    C_C_T = Pmu_phi * Pmu_psi  # (n_rmu x n_rmu)
    
    # Step 4: Solve (CC^T)^-1
    #C_C_T_inv = xp.linalg.inv(C_C_T)  # (n_rmu x n_rmu)
    
    # Step 5: Compute zeta (Equation 13)
    #zeta = Z_C_T @ C_C_T_inv  # (n_rtot x n_rmu)
    zeta = xp.linalg.solve(C_C_T.T, Z_C_T.T).T
    # Reshape zeta to separate spinor components and spatial dimensions
    # 2*r_mu because there's implicitly r-up and r-down
    zeta = zeta.reshape(2, int(wfn.fft_grid[0]), int(wfn.fft_grid[1]), int(wfn.fft_grid[2]), 2*n_rmu)
    
    if GPU_AVAILABLE:
        zeta = zeta.get()
    
    return zeta

def get_interp_vectors_q(wfn, wfnq, sym, iq, centroid_indices, bandrange_l, bandrange_r, xp):
    """Find the interpolative separable density fitting representation.
    
    Computes M_mn(k,q,r) = psi_mk-q^*(r) psi_nk(r) ~= sum_\mu psi_mk-q^*(r_\mu) psi_nk(r_\mu) \zeta_\mu(r)
    where \zeta_\mu(r) are the interpolation vectors.

    This is done by 
    
    Args:
        wfn: WFNReader object
        sym: SymMaps object
        centroid_indices: Indices of centroids in FFT grid
        bandrange_l: Tuple (start, end) for left bands (e.g., valence)
        bandrange_r: Tuple (start, end) for right bands (e.g., conduction)
    """
    # Get dimensions
    n_rtot = int(np.prod(wfn.fft_grid))  # Total number of real-space points
    n_rmu = int(centroid_indices.shape[0])    # Number of centroid points
    nb_l = bandrange_l[1] - bandrange_l[0]  # Number of left bands
    nb_r = bandrange_r[1] - bandrange_r[0]  # Number of right bands

    # for q=0, psi_r contains wfnq wavefunctions (valence only).
    if iq == 0 and any(wfnq.kpoints[0] != wfn.kpoints[0]):
        is_q0 = True
        wfnr = wfnq
    else:
        is_q0 = False
        wfnr = wfn

    # at the end, use this to return the full set of psi_nk(r_mu) for all k
    psi_l_rmu_out = xp.zeros((nb_l, sym.nk_tot, 2*n_rmu), dtype=np.complex128)
    psi_r_rmu_out = xp.zeros((nb_r, sym.nk_tot, 2*n_rmu), dtype=np.complex128)
    psi_l_rtot_out = xp.zeros((nb_l, sym.nk_tot, 2, *wfn.fft_grid), dtype=np.complex128)
    psi_r_rtot_out = xp.zeros((nb_r, sym.nk_tot, 2, *wfn.fft_grid), dtype=np.complex128)
    
    # Left (conjugated) **real space** wfns (psi_l)
    #psi_l_rtot = xp.zeros((nb_l, 2*n_rtot), dtype=np.complex128)
    psi_l_rtot = xp.zeros((nb_l, 2, *wfn.fft_grid), dtype=np.complex128)
    psi_l_rmu = xp.zeros((nb_l, 2*n_rmu), dtype=np.complex128)
    # Extract and reshape right wavefunctions (psi_r)
    #psi_r_rtot = xp.zeros((nb_r, 2*n_rtot), dtype=np.complex128)
    psi_r_rtot = xp.zeros((nb_r, 2, *wfn.fft_grid), dtype=np.complex128)
    psi_r_rmu = xp.zeros((nb_r, 2*n_rmu), dtype=np.complex128)

    # Initialize ZC^T and CC^T
    Z_C_T = xp.zeros((2*n_rtot, 2*n_rmu), dtype=np.complex128)
    C_C_T = xp.zeros((2*n_rmu, 2*n_rmu), dtype=np.complex128)
    
    # Big loop over fullBZ k-points to construct ZCT/CCT, all bands are done at once in each.
    # (for MPI: give each proc a band range to rotate/FFT/sum over)
    # ----------------------
    # k_l = k-q, k_r = k
    for k_r in range(sym.nk_tot): # k_l = k in fullBZ
        k_l = sym.kq_map[k_r, iq] # k-q in fullBZ
        k_l_red = sym.kpoint_map[k_l] # IBZ
        # Find the index where irk_to_k_map matches our k-point
        k_l_sym_idx = np.where(np.array(sym.irk_to_k_map[k_l_red]) == k_l)[0][0]
        k_l_sym_op = sym.irk_sym_map[k_l_red][k_l_sym_idx]

        k_r_red = sym.kpoint_map[k_r] # IBZ
        k_r_sym_idx = np.where(np.array(sym.irk_to_k_map[k_r_red]) == k_r)[0][0]
        k_r_sym_op = sym.irk_sym_map[k_r_red][k_r_sym_idx]

        print(f"k_l = {k_l}, k_l_red = {k_l_red}, k_l_sym_op = {k_l_sym_op}")
        print(f"k_r = {k_r}, k_r_red = {k_r_red}, k_r_sym_op = {k_r_sym_op}")

        gvecs_kl = xp.asarray(wfn.get_gvec_nk(k_l_red))
        gvecs_kr = xp.asarray(wfnr.get_gvec_nk(k_r_red))

        # Get wavefunction G coefficients for SYM REDUCED k-points
        psi_l_Gspace = xp.zeros((nb_l, 2, wfn.ngk[k_l_red]), dtype=np.complex128)
        psi_r_Gspace = xp.zeros((nb_r, 2, wfnr.ngk[k_r_red]), dtype=np.complex128)

        for ib, band_idx in enumerate(range(bandrange_l[0], bandrange_l[1])):
            psi_l_Gspace[ib, :, :] = xp.conj(xp.asarray(wfn.get_cnk(k_l_red, band_idx)))
        for ib, band_idx in enumerate(range(bandrange_r[0], bandrange_r[1])):
            psi_r_Gspace[ib, :, :] = xp.asarray(wfnr.get_cnk(k_r_red, band_idx))

        # rotate G-vectors according to symmetry operation for current k-point
        # rotating the G-component grid
        gvecs_kl_rot = xp.einsum('ij,kj->ki', sym.Rinv_grid[k_l_sym_op], gvecs_kl)
        gvecs_kr_rot = xp.einsum('ij,kj->ki', sym.Rinv_grid[k_r_sym_op], gvecs_kr)
        # rotating the spinor components
        psi_l_Gspace = xp.einsum('ij,bjk->bik', sym.U_spinor[k_l_sym_op], psi_l_Gspace)
        psi_r_Gspace = xp.einsum('ij,bjk->bik', sym.U_spinor[k_r_sym_op], psi_r_Gspace)

        # fill psi_l_rtot and psi_r_rtot with zeros here
        psi_l_rtot.fill(0)
        psi_r_rtot.fill(0)
        psi_l_rmu.fill(0)
        psi_r_rmu.fill(0)

        # FFT (note second index is spinor index)
        for ib, band_idx in enumerate(range(bandrange_l[0], bandrange_l[1])):
            psi_l_rtot[ib,0,gvecs_kl_rot[:,0],gvecs_kl_rot[:,1],gvecs_kl_rot[:,2]] = psi_l_Gspace[ib,0,:]
            psi_l_rtot[ib,1,gvecs_kl_rot[:,0],gvecs_kl_rot[:,1],gvecs_kl_rot[:,2]] = psi_l_Gspace[ib,1,:]
            psi_l_rtot[ib,0] = fftx.fft.ifftn(psi_l_rtot[ib,0])
            psi_l_rtot[ib,1] = fftx.fft.ifftn(psi_l_rtot[ib,1])
        for ib, band_idx in enumerate(range(bandrange_r[0], bandrange_r[1])):
            psi_r_rtot[ib,0,gvecs_kr_rot[:,0],gvecs_kr_rot[:,1],gvecs_kr_rot[:,2]] = psi_r_Gspace[ib,0,:]
            psi_r_rtot[ib,1,gvecs_kr_rot[:,0],gvecs_kr_rot[:,1],gvecs_kr_rot[:,2]] = psi_r_Gspace[ib,1,:]
            psi_r_rtot[ib,0] = fftx.fft.ifftn(psi_r_rtot[ib,0])
            psi_r_rtot[ib,1] = fftx.fft.ifftn(psi_r_rtot[ib,1])
        
        # Check normalization
        for ib in range(psi_l_Gspace.shape[0]):
            norm = xp.sqrt(xp.sum(xp.abs(psi_l_Gspace[ib])**2))
            if not xp.isclose(norm, 1.0, rtol=1e-4):
                print(f"Warning: Left coefficients not normalized for k={k_l_red}, band={bandrange_l[0]+ib}")
                print(f"Norm = {norm}")

        for ib in range(psi_r_Gspace.shape[0]):
            norm = xp.sqrt(xp.sum(xp.abs(psi_r_Gspace[ib])**2))
            if not xp.isclose(norm, 1.0, rtol=1e-4):
                print(f"Warning: Right coefficients not normalized for k={k_r_red}, band={bandrange_r[0]+ib}")
                print(f"Norm = {norm}")
        
        psi_l_rtot *= xp.sqrt(n_rtot)
        psi_r_rtot *= xp.sqrt(n_rtot)
        print(f"psi_l_rtot norm: {xp.linalg.norm(psi_l_rtot[0])}\n")

        psi_l_rtot_out[:, k_l] = xp.conj(psi_l_rtot)
        psi_r_rtot_out[:, k_r] = psi_r_rtot

        # get psi_l_rmu and reshape psi_l_rtot to flatten last four dimensions
        psi_l_rmu = psi_l_rtot[:,:,centroid_indices[:,0],centroid_indices[:,1],centroid_indices[:,2]]
        psi_l_rmu = psi_l_rmu.reshape(nb_l, -1)
        psi_l_rtot = psi_l_rtot.reshape(nb_l, -1)
        psi_r_rmu = psi_r_rtot[:,:,centroid_indices[:,0],centroid_indices[:,1],centroid_indices[:,2]]
        psi_r_rmu = psi_r_rmu.reshape(nb_r, -1)
        psi_r_rtot = psi_r_rtot.reshape(nb_r, -1)

        psi_l_rmu_out[:,k_l,:] = xp.conj(psi_l_rmu)
        psi_r_rmu_out[:,k_r,:] = psi_r_rmu
        
        # Step 1: Compute quasi-density matrices P_phi,k and P_psi,k by sum over bands (Equation 15)
        P_l = xp.einsum('mi,mj->ij', psi_l_rtot, psi_l_rmu)  # (2*n_rtot x n_lmu)
        P_r = xp.einsum('mi,mj->ij', psi_r_rtot, psi_r_rmu)   # (2*n_rtot x n_rmu)
        
        # Step 2: Compute (ZC^T)_q contribution *from this k* (Equation 14)
        Z_C_T += P_l * P_r  # (n_rtot x n_rmu)

        Pmu_l = xp.einsum('mi,mj->ij', psi_l_rmu, psi_l_rmu)  # (n_rmu x n_rmu)
        Pmu_r = xp.einsum('mi,mj->ij', psi_r_rmu, psi_r_rmu)  # (n_rmu x n_rmu)
        
        # Step 3: Compute CC^T (Equation 16)
        C_C_T += Pmu_l * Pmu_r  # (n_rmu x n_rmu)

        # (for next loop)
        psi_l_rtot = psi_l_rtot.reshape(nb_l, 2, *wfn.fft_grid)
        psi_r_rtot = psi_r_rtot.reshape(nb_r, 2, *wfn.fft_grid)

    # Step 4: Solve (CC^T)zeta = (ZC^T)
    zeta_q = xp.linalg.solve(C_C_T.T, Z_C_T.T).T
    # Reshape zeta to separate spinor components and spatial dimensions
    # 2*r_mu because there's implicitly r-up and r-down
    zeta_q = zeta_q.reshape(2, int(wfn.fft_grid[0]), int(wfn.fft_grid[1]), int(wfn.fft_grid[2]), 2*n_rmu)
    
    #if cp.cuda.is_available():
    #    zeta_q = zeta_q.get()
    # zeta_q: shape (2, *fft_grid, 2*n_rmu)
    # psi_l_rmu_out: shape (nb_l, nk_fullbz, 2*n_rmu)
    # psi_r_rmu_out: shape (nb_r, nk_fullbz, 2*n_rmu)

    return zeta_q, psi_l_rmu_out, psi_r_rmu_out, psi_l_rtot_out, psi_r_rtot_out


def analyze_psi_centroids():
    """Analyze the psi_centroids.h5 file to check for zero/nonzero values."""
    
    with h5py.File('psi_centroids.h5', 'r') as f:
        psi_values = f['psi_values'][:]
        psi_real_space = f['psi_real_space'][:]
        
        # Analyze centroid values
        print("\nCentroid values statistics:")
        print_array_stats(psi_values, "psi_values")
        
        # Analyze full real-space values
        print("\nReal-space wavefunction statistics:")
        print_array_stats(psi_real_space, "psi_real_space")

def print_array_stats(arr, name):
    """Helper function to print array statistics."""
    arr_abs = np.abs(arr)
    print(f"\n{name} statistics:")
    print(f"Shape: {arr.shape}")
    print(f"Min absolute value: {arr_abs.min()}")
    print(f"Max absolute value: {arr_abs.max()}")
    print(f"Number of exact zeros: {np.sum(arr_abs == 0)}")
    
    if np.sum(arr_abs == 0) > 0:
        zero_locs = np.where(arr_abs == 0)
        print("\nFirst few zero locations:")
        for i in range(min(5, len(zero_locs[0]))):
            print(f"Location: {tuple(coord[i] for coord in zero_locs)}")

def plot_interpolating_vectors(zeta, centroids_frac, fft_grid, n_vectors=2):
    """Plot spatial distribution of randomly chosen interpolating vectors.
    
    Args:
        zeta: Interpolating vectors (2, nx, ny, nz, n_mu)
        centroids_frac: Fractional coordinates of centroids (n_mu, 3)
        fft_grid: FFT grid dimensions (3,)
        n_vectors: Number of random vectors to plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get actual number of centroids
    n_mu = len(centroids_frac)  # Changed from zeta.shape[-1]
    
    # Randomly choose centroids to visualize
    n_vectors = min(n_vectors, n_mu)  # Ensure we don't try to plot more than we have
    chosen_indices = np.random.choice(n_mu, size=n_vectors, replace=False)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_vectors, figsize=(6*n_vectors, 5))
    if n_vectors == 1:
        axes = [axes]
    
    for i, mu in enumerate(chosen_indices):
        # Sum over spinor components and z-direction for visualization
        zeta_mu = np.abs(zeta[0, ..., mu])# + zeta[1, ..., mu])
        zeta_mu_xy = np.sum(zeta_mu, axis=2)  # Integrate over z
        
        # Plot interpolating vector
        im = axes[i].imshow(zeta_mu_xy.T, origin='lower', cmap='RdBu_r')
        plt.colorbar(im, ax=axes[i])
        
        # Mark centroid position
        centroid_pos = np.round(centroids_frac[mu] * fft_grid).astype(int)
        axes[i].plot(centroid_pos[0], centroid_pos[1], 'k.', markersize=10, label='Centroid')
        
        axes[i].set_title(f'Interpolating Vector {mu}')
        axes[i].set_xlabel('x')
        axes[i].set_ylabel('y')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('interpolating_vectors.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_centroids_3d(centroid_indices, fft_grid):
    """Create a 2D scatter plot of centroid positions in the xz plane.
    
    Args:
        centroid_indices: Array of centroid indices (N, 3)
        fft_grid: FFT grid dimensions (3,)
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot of centroids in xz plane
    ax.scatter(centroid_indices[:, 0],  # x coordinates
              centroid_indices[:, 2],  # z coordinates
              c='b', marker='o')
    
    # Set axis labels and limits
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_xlim(0, fft_grid[0])
    ax.set_ylim(0, fft_grid[2])
    
    # Add title
    ax.set_title(f'Centroid Positions in XZ Plane (Total: {len(centroid_indices)})')
    
    # Make aspect ratio equal
    ax.set_aspect('equal')
    
    # Save plot
    plt.savefig('centroids_xz.png', dpi=300, bbox_inches='tight')
    plt.close()

def calculate_spin_expectation(wfn, nb=30):
    """Calculate spin expectation values for the first nb bands.
    
    Args:
        wfn: WFNReader object
        nb: Number of bands to analyze
    """
    # Pauli matrices
    sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    
    print("\nSpin expectation values for k=0:")
    print("Band    <σx>      <σy>      <σz>")
    print("-" * 35)
    
    # For k=0
    ik = 3
    
    for ib in range(nb):
        # Get wavefunction coefficients
        coeffs = wfn.get_cnk(ik, ib)  # Shape: (2, ngk)
        
        # Normalize the spinor
        #norm = np.sqrt(np.sum(np.abs(coeffs[0])**2 + np.abs(coeffs[1])**2))
        #spinor = coeffs / norm
        
        # Calculate expectation values using spinor components
        spinor_vec = np.array([np.sum(coeffs[0]), np.sum(coeffs[1])])
        spinor_vec = spinor_vec / np.linalg.norm(spinor_vec)
        
        sx = np.real(spinor_vec.conj() @ sigma_x @ spinor_vec)
        sy = np.real(spinor_vec.conj() @ sigma_y @ spinor_vec)
        sz = np.real(spinor_vec.conj() @ sigma_z @ spinor_vec)
        
        print(f"{ib:3d}    {sx:8.4f}  {sy:8.4f}  {sz:8.4f}")

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
    bandrange_l = (27, 40)
    bandrange_r = (21, 26)
    k_idx = 3  # index of k-point in fullBZ
    q_idx = 0  # index of q-point in fullBZ
    kminusq_idx = sym.kq_map[k_idx, q_idx]

    #zeta_q = get_interp_vectors(wfn, sym, centroid_indices, bandrange_l, bandrange_r, xp)
    zeta_q, psi_l_rmu_out, psi_r_rmu_out, psi_l_rtot_out, psi_r_rtot_out = get_interp_vectors_q(wfn, wfnq, sym, q_idx, centroid_indices, bandrange_l, bandrange_r, xp)
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
        print(f"i={i:<2} |", end=" ")
        for j in range(5):
            # Compute exact M_ij(r)
            M_exact = np.zeros((2, *wfn.fft_grid), dtype=np.complex128)
            M_exact = np.conj(psi_l_rtot_out[i,kminusq_idx]) * psi_r_rtot_out[j,k_idx]
            
            # Compute interpolated M_ij(r)
            M_interp = np.zeros_like(M_exact)
            # Get product at all centroids for this spinor (n_centroids,)
            #psi_prod_mu = np.conj(psi_values[k_idx-q_idx, i]) * psi_values[k_idx, j]
            #psi_prod_mu = psi_prod_mu.ravel() #flat
            
            # Step 1: Reshape zeta to separate jspinor and mu
            zeta_reshaped = zeta_q.reshape(2, *wfn.fft_grid, 2,-1).get()

            # Compute psi*_mk-q psi_nk(spinor,rmu) for all ispinor
            # Shape: (ispinor=2, mu=300)
            psi_prod_mu = (np.conj(psi_l_rmu_out[i,kminusq_idx, :]) * psi_r_rmu_out[j,k_idx, :]).reshape(2, -1)

            # Perform tensor contraction using einsum to get M_interp with shape (2, 25, 25, 100)
            M_interp = np.einsum('j...im,im->i...', zeta_reshaped, psi_prod_mu)
            # Calculate projection
            overlap = np.sum(np.conj(M_exact) * M_interp)
            norm = np.sum(np.abs(M_exact)**2)
            projection = np.real(overlap / norm)
            print(f"{projection:.4f}", end="  ")
        print()
