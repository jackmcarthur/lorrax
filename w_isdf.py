import numpy as np
from gpu_utils import cp, xp
from wfnreader import WFNReader
from tagged_arrays import LabeledArray
from gamma_matrices import gammas_sparse
# The routines here construct chi^0 and the screened interaction W using the
# CTSP approach in the static limit.  Once the frequency grids are restored, the
# same machinery will let us tackle full dynamical GW.

# do chi_lm,0(r,r',Yt) = \sum_ab Gc_lm,R(ra,r'b,Yt)Gv_lm,-R(r'b,ra,-Yt) (a,b=spin indices)
# Now applies quadrature weights and returns integrated result
def get_chi_lm_Yt(psi_v, psi_c, win, wfn, xp):
    ntau = win.ntau
    nspinor = psi_v.psi.shape('nspinor')
    npol = 4 if nspinor == 4 else 1 # JM: check this
    nrmu = psi_v.psi.shape('nrmu')
    psi_v.psi.join('nspinor', 'nrmu')
    psi_c.psi.join('nspinor', 'nrmu')
    
    # Now G arrays only need to hold one tau at a time
    Gv_lm = LabeledArray(shape=(*wfn.kgrid, nspinor, nrmu, nspinor, nrmu), axes=('nkx', 'nky', 'nkz', 'nspinor1', 'nrmu1', 'nspinor2', 'nrmu2'))
    Gc_lm = LabeledArray(shape=(*wfn.kgrid, nspinor, nrmu, nspinor, nrmu), axes=('nkx', 'nky', 'nkz', 'nspinor1', 'nrmu1', 'nspinor2', 'nrmu2'))
    
    # Create integrated chi array (no tau dimension) and compute quadrature weights
    chi_lm_integrated = LabeledArray(shape=(npol, nrmu, npol, nrmu, *wfn.kgrid), axes=('npol1', 'nrmu1', 'npol2', 'nrmu2', 'nkx', 'nky', 'nkz'))
    chi_lm_integrated.data[:] = 0.0  # Initialize to zero
    
    # Precompute quadrature weights: -2 z_lm w_i exp(-(z_lm (E_c - E_v) - 1) tau_i)
    quad_weights = xp.asarray(-2.*win.z_lm*win.w_i*np.exp(-(win.z_lm*(win.cond_window.start_energy - win.val_window.end_energy)-1.)*win.tau_i), dtype=xp.complex128)

    Gv_lm.join('nkx', 'nky', 'nkz')
    Gc_lm.join('nkx', 'nky', 'nkz')
    Gv_lm.join('nspinor1', 'nrmu1')
    Gv_lm.join('nspinor2', 'nrmu2')
    Gc_lm.join('nspinor1', 'nrmu1')
    Gc_lm.join('nspinor2', 'nrmu2')

    # Precompute masks and find maximum number of bands in any window
    val_mask_all = (psi_v.enk.data >= win.val_window.start_energy) & (psi_v.enk.data <= win.val_window.end_energy)
    cond_mask_all = (psi_c.enk.data >= win.cond_window.start_energy) & (psi_c.enk.data <= win.cond_window.end_energy)
    
    max_val_bands = int(xp.max(xp.sum(val_mask_all, axis=1)))
    max_cond_bands = int(xp.max(xp.sum(cond_mask_all, axis=1)))
    
    nk = psi_v.psi.shape('nk')
    norb = psi_v.psi.shape('nspinor*nrmu')
    
    # Allocate compressed arrays - only store bands within energy windows
    psi_v_masked = xp.zeros((nk, max_val_bands, norb), dtype=psi_v.psi.data.dtype)
    psi_v_conj = xp.zeros((nk, norb, max_val_bands), dtype=psi_v.psi.data.dtype)
    psi_c_masked = xp.zeros((nk, norb, max_cond_bands), dtype=psi_c.psi.data.dtype)  
    psi_c_conj = xp.zeros((nk, max_cond_bands, norb), dtype=psi_c.psi.data.dtype)
    
    # Also compress the energy arrays to match
    enk_v_compressed = xp.zeros((nk, max_val_bands), dtype=psi_v.enk.data.dtype)
    enk_c_compressed = xp.zeros((nk, max_cond_bands), dtype=psi_c.enk.data.dtype)
    
    # Fill compressed arrays with only the bands within energy windows
    for ik in range(nk):
        val_indices = xp.where(val_mask_all[ik])[0]
        cond_indices = xp.where(cond_mask_all[ik])[0]
        
        if len(val_indices) > 0:
            psi_v_masked[ik, :len(val_indices)] = psi_v.psi.data[ik, val_indices]
            psi_v_conj[ik, :, :len(val_indices)] = xp.conj(psi_v.psi.data[ik, val_indices]).T
            enk_v_compressed[ik, :len(val_indices)] = psi_v.enk.data[ik, val_indices]
            
        if len(cond_indices) > 0:
            psi_c_conj[ik, :len(cond_indices)] = xp.conj(psi_c.psi.data[ik, cond_indices])
            psi_c_masked[ik, :, :len(cond_indices)] = psi_c.psi.data[ik, cond_indices].T
            enk_c_compressed[ik, :len(cond_indices)] = psi_c.enk.data[ik, cond_indices]
    
    # Make arrays contiguous for optimal performance
    psi_v_masked = xp.ascontiguousarray(psi_v_masked)
    psi_v_conj = xp.ascontiguousarray(psi_v_conj)
    psi_c_masked = xp.ascontiguousarray(psi_c_masked)
    psi_c_conj = xp.ascontiguousarray(psi_c_conj)
    
    # Allocate temporary exponential arrays for each tau iteration
    exp_v_tmp = xp.zeros((nk, max_val_bands), dtype=xp.complex128)
    exp_c_tmp = xp.zeros((nk, max_cond_bands), dtype=xp.complex128)
    
    # Loop over tau values to save memory
    for itau in range(ntau):
        tau_val = win.tau_i[itau]
        
        # Compute exponentials directly from compressed energy arrays into temporary arrays
        exp_v_tmp[:] = xp.exp(-win.z_lm * tau_val * (win.val_window.end_energy - enk_v_compressed))
        exp_c_tmp[:] = xp.exp(-win.z_lm * tau_val * (enk_c_compressed - win.cond_window.start_energy))
        
        # Apply exponential to compressed arrays - note shapes:
        # psi_v_conj(nk,norb,max_val_bands), psi_c_conj(nk,max_cond_bands,norb)  
        xp.multiply(psi_v_conj, exp_v_tmp[:, np.newaxis, :], out=psi_v_conj)  # Apply exponential
        xp.multiply(psi_c_conj, exp_c_tmp[:, :, np.newaxis], out=psi_c_conj)  # Apply exponential
        
        # Efficient batched matmuls without any transposes:
        # psi_v_conj(nk,norb,max_val_bands) @ psi_v_masked(nk,max_val_bands,norb) -> (nk,norb,norb)
        Gv_lm.data = xp.matmul(psi_v_conj, psi_v_masked)
        # psi_c_masked(nk,norb,max_cond_bands) @ psi_c_conj(nk,max_cond_bands,norb) -> (nk,norb,norb)  
        Gc_lm.data = xp.matmul(psi_c_masked, psi_c_conj)
        
        # Remove exponential from arrays for reuse
        xp.divide(psi_v_conj, exp_v_tmp[:, np.newaxis, :], out=psi_v_conj)  # Remove exponential
        xp.divide(psi_c_conj, exp_c_tmp[:, :, np.newaxis], out=psi_c_conj)  # Remove exponential
        
        # No tau-specific arrays to free now
        if hasattr(xp, 'get_default_memory_pool'):
            xp.get_default_memory_pool().free_all_blocks()

        # Transform to real space for this tau
        Gv_lm.unjoin('nkx', 'nky', 'nkz')
        Gv_lm = Gv_lm.kgrid_to_last()
        Gv_lm.ifft_kgrid() # G_k -> G_R

        Gc_lm.unjoin('nkx', 'nky', 'nkz')
        Gc_lm = Gc_lm.kgrid_to_last()
        Gc_lm.ifft_kgrid()

        # flip Gv_R -> Gv_-R, keeping Gv_R=0 in the 0th index
        for ik in range(2,5):
            Gv_lm.data = xp.flip(Gv_lm.data, axis=ik)
            Gv_lm.data = xp.roll(Gv_lm.data, 1, axis=ik)

        Gv_lm.unjoin('nspinor1', 'nrmu1')
        Gv_lm.unjoin('nspinor2', 'nrmu2')
        Gc_lm.unjoin('nspinor1', 'nrmu1')
        Gc_lm.unjoin('nspinor2', 'nrmu2')

        # Compute chi contribution for this tau and accumulate with quadrature weight
        current_weight = quad_weights[itau]
        
        if npol == 4:
            scratch = xp.empty_like(Gc_lm.slice_many({'nspinor1': 0, 'nspinor2': 0}))

            for I, (rI, cI, vI) in enumerate(gammas_sparse):
                for J, (rJ, cJ, vJ) in enumerate(gammas_sparse):
                    target = chi_lm_integrated.data[I,:,J,:,:,:,:] 
                    for p in range(len(vI)):
                        a = int(rI[p])
                        c = int(cI[p])
                        gI = vI[p]
                        for q in range(len(vJ)):
                            b = int(rJ[q])
                            d = int(cJ[q])
                            gJ = vJ[q]
                            xp.multiply(Gc_lm.slice_many({'nspinor1': a, 'nspinor2': b}),
                                        Gv_lm.slice_many({'nspinor1': c, 'nspinor2': d}),
                                        out=scratch)
                            # Apply quadrature weight and accumulate
                            xp.add(target, current_weight * gI * gJ * scratch, out=target)
        else:
            for a in range(nspinor):
                for b in range(nspinor):
                    chi_contribution = xp.multiply(Gc_lm.slice_many({'nspinor1':a,'nspinor2':b}), Gv_lm.slice_many({'nspinor1':b,'nspinor2':a}))
                    # Apply quadrature weight and accumulate
                    chi_lm_integrated.data[0,:,0,:,:,:,:] += current_weight * chi_contribution

        # Prepare for next tau iteration - rejoin for k-space operations
        Gv_lm.join('nspinor1', 'nrmu1')
        Gv_lm.join('nspinor2', 'nrmu2')
        Gc_lm.join('nspinor1', 'nrmu1')
        Gc_lm.join('nspinor2', 'nrmu2')
        Gv_lm.join('nkx', 'nky', 'nkz')
        Gc_lm.join('nkx', 'nky', 'nkz')
        # TODO: this is wasteful data rearrangement because the data doesn't matter, only the axes
        Gc_lm = Gc_lm.transpose('nkx*nky*nkz', 'nspinor1*nrmu1', 'nspinor2*nrmu2')
        Gv_lm = Gv_lm.transpose('nkx*nky*nkz', 'nspinor1*nrmu1', 'nspinor2*nrmu2')

    # Clean up masks and reused arrays
    del val_mask_all, cond_mask_all, psi_v_masked, psi_v_conj, psi_c_masked, psi_c_conj
    del exp_v_tmp, exp_c_tmp, enk_v_compressed, enk_c_compressed
    if hasattr(xp, 'get_default_memory_pool'):
        xp.get_default_memory_pool().free_all_blocks()

    psi_v.psi.unjoin('nspinor', 'nrmu')
    psi_c.psi.unjoin('nspinor', 'nrmu')

    # note it would be more efficient to only fft chi0 in get_chi0
    chi_lm_integrated.fft_kgrid() # chi_R -> chi_q
    chi_out = chi_lm_integrated.transpose('nkx', 'nky', 'nkz', 'npol1', 'nrmu1', 'npol2', 'nrmu2')
    #oneoverkgrid = xp.complex128(np.power(np.complex128(wfn.kgrid[0]*wfn.kgrid[1]*wfn.kgrid[2]),0.5))
    #xp.multiply(chi_out.data, oneoverkgrid, out=chi_out.data)
    #xp.multiply(chi_out.data, 0.45, out=chi_out.data)
    print('one chi_lm element ', chi_out.data[0,0,0,0,0,0,0].item())
    return chi_out.data


# sums contributions from all windows
def get_chi0(psi_v, psi_c, windows, wfn, xp):
    nspinor = psi_v.psi.shape('nspinor')
    npol = 4 if nspinor == 4 else 1
    nrmu = psi_v.psi.shape('nrmu')
    chi0 = LabeledArray(shape=(1, *wfn.kgrid, npol, nrmu, npol, nrmu), axes=('ntau', 'nkx', 'nky', 'nkz', 'npol1', 'nrmu1', 'npol2', 'nrmu2'))
    #chi0.join('nkx', 'nky', 'nkz')
    #chi0.join('nspinor1', 'nrmu1')
    #chi0.join('nspinor2', 'nrmu2')

    for win in windows:
        chi_lm_integrated = get_chi_lm_Yt(psi_v, psi_c, win, wfn, xp)
        # Quadrature weights are now applied inside get_chi_lm_Yt, so just add the result
        xp.add(chi0.data[0,:,:,:,:,:,:,:], chi_lm_integrated, out=chi0.data[0,:,:,:,:,:,:,:])

    chi = chi0.transpose('nkx', 'nky', 'nkz', 'ntau', 'npol1', 'nrmu1', 'npol2', 'nrmu2')
    return chi

def get_static_w_q(chi_q, Vq, wfn, sym, xp, n_mult=10, block_f=1, bispinor=False):
    # w_q(omega) = (1-v_q @ chi_q)^{-1} @ v_q
    # This implementation performs the CTSP matrix inversion in the static limit.
    # Once the frequency mesh is restored this routine will compute W(omega) on
    # the full imaginary-time grid.
    # if A = v_q @ chi_q, then (1-A)^{-1} = 1 + A + A^2 + A^3 + ... (iterative matrix inversion faster + more stable than direct)
    # A^N is done with blocked GEMMs along the frequency axis; since we currently do COHSEX we set block_q=1

    #if bispinor:
        # die because no chi_munu = gamma_mu gamma_nu G G yet
    #    raise ValueError("bispinor not implemented yet")
    npol_w = chi_q.shape('npol1')
    nrmu = chi_q.shape('nrmu1')
    print('one chi element: ', chi_q.data[0,0,0,0,0,0,0,0].item())

    # does not matter if bispinor or not
    V_q = Vq.transpose('nfreq','nkx','nky','nkz','npol1','nrmu1','npol2','nrmu2')
    V_q.join('nkx', 'nky', 'nkz')
    V_q.join('npol1', 'nrmu1')
    V_q.join('npol2', 'nrmu2')

    W_q = LabeledArray(shape=(*wfn.kgrid, 1, npol_w, nrmu, npol_w, nrmu), axes=('nkx', 'nky', 'nkz', 'nfreq','npol1', 'nrmu1', 'npol2', 'nrmu2'))
    W_q.join('nkx', 'nky', 'nkz')
    W_q.join('npol1', 'nrmu1')
    W_q.join('npol2', 'nrmu2')


    chi_q.join('nkx', 'nky', 'nkz')
    chi_q.join('npol1', 'nrmu1')
    chi_q.join('npol2', 'nrmu2')
    
    nk_tot, nfreq, N, _ = chi_q.data.shape

    # pick a block‐size along the frequency axis
    if block_f is None:
        # e.g. cap at 128 MB of scratch:
        max_bytes = 128 * 1024**2
        per_mat   = 16 * N * N       # bytes per (N×N) complex128
        block_f   = max(1, int(max_bytes // per_mat))
    block_f = min(block_f, nfreq)

    # allocate scratch buffers once
    A   = xp.empty((block_f, N, N), dtype=xp.complex128)
    Wb  = xp.empty((block_f, N, N), dtype=xp.complex128)
    P   = xp.empty((block_f, N, N), dtype=xp.complex128)
    I   = xp.eye(N, dtype=xp.complex128)[None, :, :]

    # loop over q‐points
    for iq in range(nk_tot):
        Vf = V_q.data[0,iq]  # shape = (N, N)
        ch = chi_q.data[iq]  # shape = (nfreq, N, N)
        #Wf = W_q.data[iq]    # shape = (nfreq, N, N)

        # chunk over freq‐axis
        # for f0 in range(0, nfreq, block_f):
        #     f1 = min(f0+block_f, nfreq)
        #     B  = f1 - f0

        #     cb = ch[f0:f1]      # (B, N, N)
        #     wb = Wb[:B]         # view into scratch
        #     a  = A[:B]

        #     # 1) A := Vb @ cb
        #     xp.matmul(Vf, cb, out=a)

        #     # 2) Wb := I + A
        #     wb[:] = I           # broadcast eye
        #     wb += a

        #     # 3) Build powers A^2 … A^(n_mult+1)
        #     # P = a.copy()        # P == A^1
        #     # for _ in range(n_mult):
        #     #     xp.matmul(P, cb, out=P)
        #     #     wb += P
        #     #     cb = a.copy() # chi array now contains vchi
        #     #     for _ in range(n_mult-1):
        #     #         xp.matmul(cb, a, out=P)
        #     #         wb += P
        #     #         cb = P.copy()
        #     #         #print('mtx norm P: ', xp.linalg.norm(P))
        #     #     # 4) Multiply by Vb → W = (1 - Vχ)^(-1) V
        #     #     xp.matmul(wb, Vf, out=Wf[f0:f1])

        #     # 5) write‐back
        #     #Wf[f0:f1] = wb

        W_q.data[iq] = xp.matmul(xp.linalg.inv(I - xp.matmul(Vf, ch)), Vf)

    W_q.unjoin('nkx', 'nky', 'nkz')
    W_q.kgrid_to_last()
    #W_q.ifft_kgrid() # W_q -> W_R
    W_q.unjoin('npol1', 'nrmu1')
    W_q.unjoin('npol2', 'nrmu2')
    # could do W_q -> W_R here but it's already done in the get_sigma function
    W = W_q.transpose('nfreq', 'nkx', 'nky', 'nkz', 'npol1', 'nrmu1', 'npol2', 'nrmu2')

    V_q.unjoin('nkx', 'nky', 'nkz')
    
    V_q.unjoin('npol1', 'nrmu1')
    V_q.unjoin('npol2', 'nrmu2')

    return W
