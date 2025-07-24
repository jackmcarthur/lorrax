import numpy as np
from gpu_utils import cp, xp, fftx
from wfnreader import WFNReader
from epsreader import EPSReader
import symmetry_maps
import matplotlib.pyplot as plt


# return ranges of bands necessary for \sigma_{X,SX,COH}
def get_bandranges(nv,nc,nband,nelec):
    """Return ranges of bands necessary for \sigma_{X,SX,COH}"""
    nvrange = [int(nelec-nv), int(nelec)]
    ncrange = [int(nelec), int(nelec+nc)]
    nsigmarange = [int(nelec-nv), int(nelec+nc)]
    n_fullrange = [0, int(nband)]
    n_valrange = [0, int(nelec)]
    return nvrange, ncrange, nsigmarange, n_fullrange, n_valrange

def wrap_points_to_voronoi(randcart, bvec,xp, nmax=1):
    """
    Helper function to get test q-points for mini-BZ average with correct voronoi cell.
    """
    # 1. Generate all candidate integer translations.
    grid = xp.arange(-nmax, nmax+1)
    # meshgrid in 3D; shape will be (3, M) with M = (2*nmax+1)**3 candidates.
    shifts = xp.stack(xp.meshgrid(grid, grid, grid, indexing='ij'), axis=-1).reshape(-1, 3)
    
    # 2. Convert integer translations into Cartesian shift vectors.
    #    Here bvec.T has lattice vectors as rows.
    candidate_shifts = shifts @ bvec  # shape (M, 3)

    # 3. For each point, compute its distance to each candidate image.
    #    randcart[:, None, :] has shape (N,1,3), candidate_shifts[None,:,:] has shape (1,M,3)
    diff = randcart[:, None, :] - candidate_shifts[None, :, :]  # shape (N, M, 3)
    dists = xp.linalg.norm(diff, axis=2)  # shape (N, M)

    # 4. Select, for each point, the candidate that minimizes the distance.
    best_idx = xp.argmin(dists, axis=1)  # shape (N,)
    best_shifts = candidate_shifts[best_idx]  # shape (N, 3)

    # 5. Wrap the points by subtracting the chosen lattice translation.
    wrapped = randcart - best_shifts
    return wrapped


def get_V_qG(wfn, sym, q0, xp, epshead, sys_dim):
    # first: V(q,G,G') = 4\pi/|q+G|^2 \delta_{G,G'} * trunc part in 2D, (1-exp(-zc*kxy)*cos(kz*zc))
    # (times one other factor, 1/(N_ktot * cell_volume))
    #print("vqg start")
    print(q0)
    bvec = xp.asarray(wfn.blat * wfn.bvec, dtype=xp.float64)
    q0xp = xp.asarray(q0, dtype=xp.float64)
    qvec = xp.array([xp.float64(0.),xp.float64(0.),xp.float64(0.)])
    zc = xp.pi/bvec[2,2] # note that the crystal z axis must align with the cartesian z axis

    #print("vqg qvec done")
    G_q_crys = xp.zeros((int(wfn.ngkmax),3), dtype=xp.float64)
    G_cart = xp.zeros((int(wfn.ngkmax),3), dtype=xp.float64)
    #print("vqg G_q_crys done")
    V_qG = xp.zeros((sym.nk_tot, int(wfn.ngkmax)), dtype=xp.float64)
    ngks = xp.asarray(wfn.ngk, dtype=xp.int32)
    #print("vqg all arrays done")

    # if sys dim == 3 return error not implemented
    if sys_dim == 3:
        raise NotImplementedError("3D system calculation not yet implemented")
    #print('trying vqg')
    # get V(q,G) array for all sym-reduced q
    if sys_dim == 2:
        for iq in range(wfn.nkpts):
            qvec = xp.asarray(wfn.kpoints[iq])
            print(qvec.shape)
            if iq == 0:
                qvec = q0xp

            Gmax_q = ngks[iq]

            G_q_crys.fill(0.)
            G_cart.fill(0.)
            # this saves memory in the case of many kpts but requires a lot of HtoD transfers. revisit.
            G_q_crys[:Gmax_q] = xp.asarray(wfn.get_gvec_nk(iq).astype(np.float64),dtype=xp.float64) # stored as int32, trying to convert efficiently
            G_cart[:Gmax_q] = xp.matmul(G_q_crys[:Gmax_q] + qvec, bvec) # @ is super slow, probably using numpy

            #print("done with gcart")
            V_qG[iq,:Gmax_q] = xp.divide(4*xp.pi, xp.sum(G_cart*G_cart, axis=1)[:Gmax_q])
            #print("done with vqg no trunc")
            kxy = xp.linalg.norm(G_cart[:Gmax_q,:2], axis=1)
            kz = G_cart[:Gmax_q,2]
            # NOT SURE WHY THERES AN EXTRA 2. 8PI NOT 4PI? I\neq J probably? but i compared to an epsmat.h5 file
            V_qG[iq,:Gmax_q] *= 2 * (1-xp.exp(-zc*kxy)*xp.cos(kz*zc))

        # mini-BZ voronoi monte carlo integration for V_q=0,G=0
        k0_vol_crys = xp.diag(xp.divide(1.0, xp.asarray(wfn.kgrid)))
        randlims = xp.matmul(bvec.T, xp.matmul(k0_vol_crys, xp.linalg.inv(bvec.T)))

        # BGW VORONOI CELL AVERAGE
        randvals = xp.random.rand(2500000,3)
        randcart = xp.einsum('ik,jk->ji', bvec.T, randvals)
        wrapped_cart = wrap_points_to_voronoi(randcart, bvec, xp, nmax=1)

        randqcart = xp.einsum('ik,jk->ji', randlims, wrapped_cart) # set of non-grid qpts closer to q=0 than any other qpt
        #randqcart = xp.einsum('ik,jk->ji', bvec.T, randqs)
        # DEBUG: possibly necessary in 2d?
        randqcart[:,2] = 0.0
        rand_vq = xp.divide(4*xp.pi, xp.einsum('ij,ij->i',randqcart,randqcart))
        kxy_q0 = xp.linalg.norm(randqcart[:,:2],axis=1)
        rand_vq *= 2 * (1. - xp.exp(-xp.pi/bvec[2,2] * kxy_q0) * xp.cos(randqcart[:,2] * xp.pi/bvec[2,2]))
        #print(f"V_q=0,G=0 from q0: {V_qG[0,0]:.4f}")
        V_qG[0,0] = xp.mean(rand_vq)
        print(f"V_q=0,G=0 from miniBZ monte carlo: {V_qG[0,0]:.4f}")

        ##############################################################
        # this is wcoul0 used in BGW/Common/fixwing.f90 (generated in minibzaverage.f90)
        # equations here are: (Ismail-Beigi PRB 2006)
        # W(q,G=G'=0) = epsinv(q,G=G'=0) * vc(q)
        # 1/epsinv(q,G=G'=0) = 1 + vc(q)*f(q)
        # f(q) = gamma |q|^2 exp(-a|q|) (a=0 in minibzaverage.f90)

        q0len = xp.linalg.norm(xp.matmul(q0xp, bvec))
        vc_qtozero = (1.-xp.exp(-q0len*zc))/q0len**2
        gamma = xp.float64((1./xp.asarray(epshead.real, dtype=xp.float64) - 1.)/(q0len**2 * vc_qtozero))
        alpha = xp.float64(0.)

        rand_wq = (1. - xp.exp(-kxy_q0*zc))/(kxy_q0**2) # actually vc(q)
        rand_wq = xp.divide(rand_wq, (1. + rand_wq * kxy_q0**2 * gamma *xp.exp(-alpha*kxy_q0)))
        wcoul0 = 8*xp.pi*xp.mean(rand_wq)

        print(f"W_q=0(G=G'=0) from miniBZ monte carlo: {wcoul0:.4f}")

        fact = xp.float64(1./(sym.nk_tot*wfn.cell_volume)) # won't work if nonuniform grid
        V_qG *= fact
        wcoul0 *= fact
    return V_qG.astype(xp.complex128), wcoul0.astype(xp.complex128)


def get_V_soc(wfn, sym, xp, V_qG, current_i, spin_j):
    factor = xp.complex128(1j*5.32513544E-5) # fine structure const. ^2
    levi_civita = lambda i,j,k:(i-j)*(j-k)*(k-i)/2 # code golf

    bvec = xp.asarray(wfn.blat * wfn.bvec, dtype=xp.float64)

    #print("vqg qvec done")
    G_q_cart = xp.zeros((int(wfn.ngkmax),3), dtype=xp.float64)
    G_cart = xp.zeros((int(wfn.ngkmax),3), dtype=xp.float64)

    V_soc = xp.zeros((sym.nk_tot, int(wfn.ngkmax)), dtype=xp.float64) # convert to complex128 later

    for q_h in range(3):
        perm = levi_civita(q_h,current_i,spin_j)
        
        if perm != 0:
            for q_idx in range(sym.nk_tot):
                q_red = sym.irk_to_k_map[q_idx]
                vcoul_q = V_qG[q_idx]
                G_q_cart.fill(0.)
                Gmax_k = xp.int32(wfn.ngks[q_red])
                qvec = xp.asarray(sym.unfolded_kpts[q_idx])
                gvecs_q_rot = xp.asarray(sym.get_gvecs_kfull(wfn,q_idx))
                xp.matmul(gvecs_q_rot[:Gmax_k] + qvec[None,:], bvec, out=G_q_cart[:Gmax_k]) # @ is super slow, probably using numpy
                
                # Extract k-component and multiply with wavefunction coefficients
                h_component = xp.ascontiguousarray(G_q_cart[:Gmax_k,q_h])  # shape (Gmax_k,)


# say for vectors in 1BZ kbar, k, we have a function f(kbar,G) and we want to know f(kbar@S+G_S,G).
# the expression for this is: f(kbar@S+G_S,G) = f(kbar,(G-G_S)@Sinv)

# current: sigma_k ~ sum_q M*_q(G)V_q(G)M_q(G)
# do: sum_R G_R(r,r')W_R(r,r')


def fft_bandrange(wfn, sym, bandrange, is_left, psi_rtot_out, xp=cp, current_k=None):
    """Process wavefunctions for all k-points in the full Brillouin zone.
    Args:
        wfn/sym: WFNReader/SymMaps objects
        bandrange: Tuple (start, end) for band range
        is_left: Bool indicating if psi = psi_l (gets conjugated)
    Returns:
        psi_rtot_out: Array of real-space wavefunctions for all k-points
    """
    # Get dimensions
    nb = bandrange[1] - bandrange[0]
    n_rtot = int(xp.prod(wfn.fft_grid))
    nspinor = wfn.nspinor

    # Initialize temporary arrays
    psi_rtot = xp.zeros((nb, nspinor, *wfn.fft_grid), dtype=xp.complex128)

    if current_k is not None:
        # get cartesian gcomps
        bvec = xp.asarray(wfn.blat * wfn.bvec, dtype=xp.float64)
        kvec = xp.array([xp.float64(0.),xp.float64(0.),xp.float64(0.)])
        G_k_cart = xp.zeros((int(wfn.ngkmax),3), dtype=xp.float64)

    
    # Loop over all k-points in full BZ
    for k_idx in range(sym.nk_tot):
        # Get reduced k-point index and symmetry operation
        # note these both take the unfolded k-point index
        k_red = sym.irk_to_k_map[k_idx]
        # Initialize G-space wavefunctions
        psi_Gspace = xp.zeros((nb, nspinor, wfn.ngk[k_red]), dtype=xp.complex128)
        
        # Get G-vectors and rotate them
        gvecs_k_rot = xp.asarray(sym.get_gvecs_kfull(wfn,k_idx))
        # Get wavefunction coefficients (symmetry unfolded)
        for ib, band_idx in enumerate(range(bandrange[0], bandrange[1])):
            psi_Gspace[ib, :, :] = xp.asarray(sym.get_cnk_fullzone(wfn,band_idx,k_idx))

        # if we want Jhat_k|nk> = \sum_G exp(i(k+G)r)[(k+G)u_nk(G)] for k=xyz
        if current_k in (0, 1, 2): # won't accept True/False/None etc
            G_k_cart.fill(0.)
            Gmax_k = xp.int32(wfn.ngks[k_red])
            kvec = xp.asarray(sym.unfolded_kpts[k_idx])
            xp.matmul(gvecs_k_rot[:Gmax_k] + kvec[None,:], bvec, out=G_k_cart[:Gmax_k]) # @ is super slow, probably using numpy
            
            # Extract k-component and multiply with wavefunction coefficients
            k_component = xp.ascontiguousarray(G_k_cart[:Gmax_k,current_k])  # shape (Gmax_k,)
            psi_Gspace *= k_component[None,None,:]  # Broadcasting over (nb,nspinor,Gmax_k)

        # FFT to real space
        psi_rtot.fill(0.+0.j)
        for ib in range(nb):
            for ispinor in range(nspinor):
                # Place G-space coefficients
                psi_rtot[ib,ispinor,gvecs_k_rot[:,0],gvecs_k_rot[:,1],gvecs_k_rot[:,2]] = psi_Gspace[ib,ispinor,:]
                # Perform FFT
                psi_rtot[ib,ispinor] = fftx.fft.ifftn(psi_rtot[ib,ispinor])
        
        # Normalize
        psi_rtot *= n_rtot # was xp.sqrt(1.0/n_rtot) with fftn
        
        # Store results with new ordering
        if is_left:
            psi_rtot_out[k_idx] = xp.conj(psi_rtot)
        else:
            psi_rtot_out[k_idx] = psi_rtot



def get_sigma_x_exact(wfn, sym, k_r, bandrange_l, bandrange_r, V_qG, xp):
    """Get the exchange self-energy, Sigma_X, for the 0th band in bandrange_r, for valbands = bandrange_l"""
    # Get dimensions
    n_rtot = int(np.prod(wfn.fft_grid))

    nb_l = bandrange_l[1] - bandrange_l[0]
    nb_r = bandrange_r[1] - bandrange_r[0]
    nspinor = wfn.nspinor
    
    # Initialize output arrays that hold all relevant u_nk(r)with (nk, nb) ordering
    psi_l_rtot_out = xp.zeros((sym.nk_tot, nb_l, nspinor, *wfn.fft_grid), dtype=xp.complex128)
    psi_r_rtot_out = xp.zeros((sym.nk_tot, nb_r, nspinor, *wfn.fft_grid), dtype=xp.complex128)

    # Initialize temporary arrays for processing
    psi_l_rtot = xp.zeros((nb_l*nspinor, *wfn.fft_grid), dtype=xp.complex128)
    psi_r_rtot = xp.zeros((nb_r*nspinor, *wfn.fft_grid), dtype=xp.complex128)

    # initialize array that holds M_mn(k,-q,-G) in reciprocal space
    psiG_vcoul_tmp = xp.zeros(int(wfn.ngkmax), dtype=xp.complex128)

    # fill psi_l/r_rtot_out with respective psi(*)_l/r(r) for all k
    fft_bandrange(wfn, sym, bandrange_l, True, psi_l_rtot_out, xp=cp)
    fft_bandrange(wfn, sym, bandrange_r, False, psi_r_rtot_out, xp=cp)

    psi_r_rtot[:] = psi_r_rtot_out[k_r].reshape(nb_r*2,*wfn.fft_grid)
    sigma_out = 0.0+0.0j
    psiprod = xp.zeros(wfn.fft_grid, dtype=xp.complex128)
    
    ##########################################
    # Loop over all (unfolded) (k-q)-points (no symmetries used for q here)
    # for each q, get M_(vrange k-q,nk)(G)
    ##########################################
    for k_l in range(int(sym.nk_tot)):
        # qvec in extended zone (= k-(k-q))
        q_ext = xp.asarray(sym.unfolded_kpts[k_r] - sym.unfolded_kpts[k_l])
        q_rounded = xp.round(q_ext)
        q_ext = xp.where(xp.abs(q_ext - q_rounded) < 1e-8, q_rounded, q_ext)
        # iq is the qvec index in sym.unfolded_kpts
        iq = find_qpoint_index(q_ext, sym, tol=1e-6) 
        iq_cpu = iq.get()

        ######################################################
        # IF V_q SYMMETRY REDUCED:
        ######################################################
        # get qbar_idx, Sq and G_Sq such that q_ext = Sq @ q_ext + G_Sq.
        iqbar = sym.irk_to_k_map[iq_cpu]
        Sq = sym.sym_mats_k[sym.irk_sym_map[iq_cpu]] # now, qbar @ Sq = q
        G_Sq = np.round(q_ext.get() - Sq @ wfn.kpoints[iqbar]).astype(np.int32)
        #G_Sq = np.round((Sq @ wfn.kpoints[iqbar])%1.0 - Sq @ wfn.kpoints[iqbar]).astype(np.int32)
        # need M_q(-G) V_q(G) M_q(-G) but we have V_qbar(G) = V_q((G-Gq)Sinv)
        vcoul_psiG_comps = xp.asarray(np.einsum('ij,kj->ki',Sq.astype(np.int32),wfn.get_gvec_nk(iqbar)) - G_Sq[np.newaxis,:],dtype=xp.int32)


        psi_l_rtot = psi_l_rtot.reshape(nb_l,nspinor,*wfn.fft_grid)
        psi_r_rtot = psi_r_rtot.reshape(nb_r,nspinor,*wfn.fft_grid)
            
        psi_l_rtot[:] = psi_l_rtot_out[k_l].reshape(nb_l,2,*wfn.fft_grid)

        ######################################################
        # MAIN SIGMA LOOP
        ######################################################
        # here we get:
        # M_vn(k,-q,G) = \sum_a FFT[u_vk-q,a(r) u_nk,a(r)]
        # <nk|Sigma_X|nk> = \sum_G M_vn(k,-q,-G)^* V_q(G) M_vn(k,-q,G)
        for ib in range(psi_l_rtot.shape[0]):
            for ispinor in range(2):
                psiprod = psi_l_rtot[ib,ispinor] * psi_r_rtot[0,ispinor]
                psiprod = fftx.fft.fftn(psiprod) # not normalized! (would normally do *= 1/sqrt(N_FFT))
                psiprod *= 1./n_rtot
                # get G-space matrix elements
                #G_q_comps = xp.asarray(wfn.get_gvec_nk(iqbar), dtype=xp.int32)
                psiG_vcoul_tmp[:vcoul_psiG_comps.shape[0]] += psiprod[vcoul_psiG_comps[:,0],vcoul_psiG_comps[:,1],vcoul_psiG_comps[:,2]]


            # v contribution, all G vectors
            sigma_out += xp.sum(xp.conj(psiG_vcoul_tmp) * V_qG[iqbar] * psiG_vcoul_tmp)
            psiG_vcoul_tmp[:] = 0.0+0.0j

    return -sigma_out


def get_sigma_sex_exact(wfn, sym, epsmat, eps0mat, k_r, bandrange_l, bandrange_r,V_qG,wcoul0,xp):
    """Get the screened exchange self-energy, Sigma_SEX, for the 0th band in bandrange_r, for valbands = bandrange_l"""
    # Get dimensions
    n_rtot = int(np.prod(wfn.fft_grid))

    nb_l = bandrange_l[1] - bandrange_l[0]
    nb_r = bandrange_r[1] - bandrange_r[0]
    nspinor = wfn.nspinor
    
    # Initialize output arrays that hold all relevant u_nk(r)with (nk, nb) ordering
    psi_l_rtot_out = xp.zeros((sym.nk_tot, nb_l, nspinor, *wfn.fft_grid), dtype=xp.complex128)
    psi_r_rtot_out = xp.zeros((sym.nk_tot, nb_r, nspinor, *wfn.fft_grid), dtype=xp.complex128)

    # Initialize temporary arrays for processing
    # notice combined band/spinor index, so we can use a single cublas matmul call later
    psi_l_rtot = xp.zeros((nb_l*nspinor, *wfn.fft_grid), dtype=xp.complex128)
    psi_r_rtot = xp.zeros((nb_r*nspinor, *wfn.fft_grid), dtype=xp.complex128)

    # initialize array that holds M_mn(k,-q,-G) in reciprocal space
    psiG_vcoul_tmp = xp.zeros(int(wfn.ngkmax), dtype=xp.complex128)

    # fill psi_l/r_rtot_out with respective psi(*)_l/r(r) for all k
    fft_bandrange(wfn, sym, bandrange_l, True, psi_l_rtot_out, xp=cp)
    fft_bandrange(wfn, sym, bandrange_r, False, psi_r_rtot_out, xp=cp)

    psi_r_rtot[:] = psi_r_rtot_out[k_r].reshape(nb_r*2,*wfn.fft_grid)
    sigma_out = 0.0+0.0j
    psiprod = xp.zeros(wfn.fft_grid, dtype=xp.complex128)
    
    ##########################################
    # Loop over all (unfolded) (k-q)-points (no symmetries used for q here)
    # for each q, get M_(vrange k-q,nk)(G)
    ##########################################
    for k_l in range(int(sym.nk_tot)):
        # qvec in extended zone (= k-(k-q))
        q_ext = xp.asarray(sym.unfolded_kpts[k_r] - sym.unfolded_kpts[k_l])
        q_rounded = xp.round(q_ext)
        q_ext = xp.where(xp.abs(q_ext - q_rounded) < 1e-8, q_rounded, q_ext)
        # iq is the qvec index in sym.unfolded_kpts, but iqbar/iqbareps will be the irrBZ indices used to get vcoul/epsinv. 
        iq = find_qpoint_index(q_ext, sym, tol=1e-6) 
        iq_cpu = iq.get()

        ######################################################
        # IF W_q/V_q SYMMETRY REDUCED:
        ######################################################
        # get qbar_idx, Sq and G_Sq such that q_ext = Sq @ q_ext + G_Sq.
        iqbar = sym.irk_to_k_map[iq_cpu]
        Sq = sym.sym_mats_k[sym.irk_sym_map[iq_cpu]] # now, qbar @ Sq = q
        #G_Sq = np.round(sym.unfolded_kpts[iq_cpu] - Sq @ wfn.kpoints[iqbar]).astype(np.int32)
        G_Sq = np.round(q_ext.get() - Sq @ wfn.kpoints[iqbar]).astype(np.int32) # if we needed q outside zone, but it seems fine

        #print(f"G_Sq for k_l = {k_l} and k_r = {k_r}: {G_Sq}")
        ######################################################

        # handle the existence of eps0mat vs epsmat
        if iqbar == 0:
            eps = eps0mat
            iqbareps = xp.int32(0)
        else:
            eps = epsmat
            iqbareps = xp.int32(iqbar-1)

        if iqbar > 0 and not xp.allclose(wfn.kpoints[iqbar], eps.qpts[iqbareps]):
            print(f"q-point mismatch at index {iqbar}:")
            print(f"WFN q-point: {wfn.kpoints[iqbar]}")
            print(f"EPS q-point: {eps.qpts[iqbareps]}")

            print(f"Difference: {wfn.kpoints[iqbar] - eps.qpts[iqbareps]}")
            raise ValueError("WFN and EPS q-points do not match!")

        ######################################################
        # REORDERING OF V_qG ELEMENTS INTO EPS ORDER
        ######################################################
        # situation: W_qGG' = epsinv_qGG' * delta(G,G') v_qG
        # V_qG stored in wfn.gvecs[iq] order, not eps.components[iq] order; need V_qG[0:G_screened_cutoff] in eps[iq] order for W.
        # V_qG comps currently associated with G_q_comps:
        G_qbar_comps = xp.asarray(wfn.get_gvec_nk(iqbar), dtype=xp.int32)
        vcoul_G_q_comps_compare = xp.dot(G_qbar_comps, xp.array([1, 1000, 1000000]))

        eps_G_qbar_comps = xp.asarray(eps.unfold_eps_comps(iqbareps, sym.sym_mats_k[0], np.array([0.,0.,0.])),dtype=xp.int32)
        eps_G_qbar_comps_compare = xp.dot(eps_G_qbar_comps, xp.array([1, 1000, 1000000]))

        perm = xp.argsort(vcoul_G_q_comps_compare)
        sorted_vcoul_compare = vcoul_G_q_comps_compare[perm]
        # For each eps key, find its location in the sorted vcoul array:
        idx = xp.searchsorted(sorted_vcoul_compare, eps_G_qbar_comps_compare)
        # (Optional) Verify that every eps key is found in vcoul:
        if not xp.all(sorted_vcoul_compare[idx] == eps_G_qbar_comps_compare):
            raise ValueError("Not all eps keys were found in vcoul keys.")

        # Map back to the original indices in vcoul:
        vcoul_eps_inds = perm[idx]
        v_qG_epsorder = xp.zeros(eps_G_qbar_comps.shape[0],dtype=xp.complex128) # values are real, just use cplx dtype
        v_qG_epsorder[:] = xp.asarray(V_qG[iqbar][vcoul_eps_inds])
        #print(f"mean error in vcoul for qpt {iqbar}: {np.mean(v_qG_epsorder.get()[1:]-eps.vcoul[iqbareps,1:v_qG_epsorder.shape[0]])}")
        ######################################################

        # this is (eps-delta(GG')), which turns into W-v below.
        Wminv_qbarGG = xp.asarray(eps.get_eps_minus_delta_matrix(iqbareps),dtype=xp.complex128) 
        
        # the following replicates BGW's fixwings.f90: (since epsilon.x doesn't use minibzaverage for vcoul but sigma.x does)
        G0_idx = xp.int32(np.where(eps.gind_eps2rho[iqbareps,:100] == 0)[0][0])
        if iqbar == 0:
            # head
            Wminv_qbarGG[G0_idx,G0_idx] = wcoul0/v_qG_epsorder[G0_idx] - 1. # -1 because of delta

            # wing, wing' (the argument is: this is zeroed because it has vanishing phase space for large N_k? Baldereschi & Tosatti 1978)
            Wminv_qbarGG[G0_idx,:G0_idx] = 0.0+0.0j
            Wminv_qbarGG[G0_idx,G0_idx+1:] = 0.0+0.0j

            Wminv_qbarGG[:G0_idx,G0_idx] = 0.0+0.0j
            Wminv_qbarGG[G0_idx+1:,G0_idx] = 0.0+0.0j

        #if iqbar > 0:
            # code is difficult to interpret, but I think this is only for q0 with graphene screening.
            #fact = xp.float64(1./(sym.nk_tot*wfn.cell_volume)) # (don't know what this is)
            #q0len = xp.float64(np.dot(eps0.qpts[0], wfn.bdot @ eps0.qpts[0]))
            #Wminv_qbarGG[G0_idx,:G0_idx] *= 8.*xp.pi * fact * (xp.pi/np.sqrt(wfn.bvec[2,2])) / (q0len*v_qG_epsorder[G0_idx]) # not squared for slab trunc
            #Wminv_qbarGG[G0_idx,G0_idx+1:] *= 8.*xp.pi * fact * (xp.pi/np.sqrt(wfn.bvec[2,2])) / (q0len*v_qG_epsorder[G0_idx]) # not squared for slab trunc

        Wminv_qbarGG *= v_qG_epsorder[:,xp.newaxis]

        # Check if matrix is Hermitian (A = A^†)
        # diff = xp.abs(Wminv_qbarGG - Wminv_qbarGG.conj().T)
        # is_hermitian = xp.allclose(Wminv_qbarGG, Wminv_qbarGG.conj().T, rtol=1e-5, atol=1e-5)
        # if not is_hermitian:
        #     max_diff = xp.max(diff)
        #     max_idx = xp.unravel_index(xp.argmax(diff), diff.shape)
        #     print(f"q-point index: {iqbar}")
        #     print(f"Max difference: {max_diff:.2e} at indices: {max_idx}")
        #     print(f"Matrix elements: A[i,j]={Wminv_qbarGG[max_idx]:.2e}, A[j,i]*={Wminv_qbarGG[max_idx[1],max_idx[0]].conj():.2e}")
        #     raise ValueError("W matrix is not Hermitian!")

        eps_psiG_comps = xp.asarray(eps.unfold_eps_comps(iqbareps, Sq, G_Sq),dtype=xp.int32)
        vcoul_psiG_comps = xp.asarray(np.einsum('ij,kj->ki',Sq.astype(np.int32),wfn.get_gvec_nk(iqbar)) - G_Sq[np.newaxis,:],dtype=xp.int32)
        psiG_eps_tmp = xp.zeros(eps_G_qbar_comps.shape[0],dtype=xp.complex128)

        psi_l_rtot = psi_l_rtot.reshape(nb_l,nspinor,*wfn.fft_grid)
        psi_r_rtot = psi_r_rtot.reshape(nb_r,nspinor,*wfn.fft_grid)
            
        psi_l_rtot[:] = psi_l_rtot_out[k_l].reshape(nb_l,2,*wfn.fft_grid)

        ######################################################
        # MAIN SIGMA LOOP
        ######################################################
        # here we get:
        # M_vn(k,-q,-G) = \sum_a FFT[u_vk-q,a(r) u_nk,a(r)]
        # <nk|Sigma_SEX|nk> = \sum_GG' M_vn(k,-q,-G)^* W_q(G,G') M_vn(k,-q,-G')
        # actual computation is \sum GG'<cutoff M*(G) [W(G,G')-delta(G,G')V(G)] M(G') + \sum_G M*(G) V(G) M(G)
        # this works because for large G vectors, the interaction is approx. unscreened: W(G,G') ~ delta(G,G')V(G)
        for ib in range(psi_l_rtot.shape[0]):
            for ispinor in range(2):
                psiprod = psi_l_rtot[ib,ispinor] * psi_r_rtot[0,ispinor]
                psiprod = fftx.fft.fftn(psiprod) # CHANGING FROM FFTN, SEE BGW/SIGMA/MTXEL.F90
                psiprod *= 1./n_rtot
                # note that we use get the eps-order Gvecs from the fftbox for (W-v) and vcoul-order Gvecs for v.
                psiG_vcoul_tmp[:vcoul_psiG_comps.shape[0]] += psiprod[vcoul_psiG_comps[:,0],vcoul_psiG_comps[:,1],vcoul_psiG_comps[:,2]]
                psiG_eps_tmp[:eps_psiG_comps.shape[0]] += psiprod[eps_psiG_comps[:,0],eps_psiG_comps[:,1],eps_psiG_comps[:,2]]

            # v contribution, all G vectors
            sigma_out += xp.sum(xp.conj(psiG_vcoul_tmp) * V_qG[iqbar] * psiG_vcoul_tmp)
            # W-v contribution, |G| < screened cutoff (conj on the right side because of fortran order)
            sigma_out += xp.dot(psiG_eps_tmp,xp.matmul(Wminv_qbarGG,xp.conj(psiG_eps_tmp)))

            psiG_vcoul_tmp[:] = 0.0+0.0j
            psiG_eps_tmp[:] = 0.0+0.0j

    return -sigma_out


def find_qpoint_index(q_ext, sym, tol=1e-6):
    """Find index of q-point in unfolded k-points list.
    
    Args:
        q_ext: Vector of length 3 (crystal coordinates)
        sym: SymMaps object containing unfolded_kpts
        tol: Tolerance for floating point comparison
    
    Returns:
        Index of matching q-point, or raises ValueError if not found
    """
    # Get fractional part of q_ext
    q_frac = q_ext % 1.0
    
    # Calculate differences with all unfolded k-points
    diffs = xp.abs(xp.asarray(sym.unfolded_kpts) - q_frac[None, :])
    # Sum over coordinates and find minimum difference
    total_diffs = xp.sum(diffs, axis=1)
    min_diff = xp.min(total_diffs)
    
    if min_diff > tol:
        raise ValueError(f"No matching q-point found within tolerance {tol}")
    
    return xp.argmin(total_diffs)




if __name__ == "__main__":
    nval = 5
    ncond = 5
    nband = 30

    sys_dim = 2 # 3 for 3D, 2 for 2D

    wfn = WFNReader("WFN.h5")
    wfnq = WFNReader("WFNq.h5")
    sym = symmetry_maps.SymMaps(wfn)
    eps0 = EPSReader('eps0mat.h5')
    eps = EPSReader('epsmat.h5')
    q0 = wfnq.kpoints[0] - wfn.kpoints[0]
    
    if np.linalg.norm(q0) > 1e-6:
        print(f'Using q0 = ({q0[0]:.5f}, {q0[1]:.5f}, {q0[2]:.5f})')

    ryd2ev = 13.6056980659

    nvrange, ncrange, nsigmarange, n_fullrange, n_valrange = get_bandranges(nval, ncond, nband, wfn.nelec)

    ####################################
    # 1.) get (truncated in 2D) coulomb potential v_q(G), and wcoul0 to fix wings of eps
    ####################################
    V_qG, wcoul0 = get_V_qG(wfn, sym, eps0.qpts[0], xp, eps0.epshead, sys_dim)
    
    for i in range(21,31):
        sigma = get_sigma_sex_exact(wfn, sym, eps, eps0, 0, n_valrange, (i,i+1), V_qG, wcoul0, xp)
        #sigma = get_sigma_x_exact(wfn, sym, 0, n_valrange, (i,i+1), V_qG, cp)
        sigma *= ryd2ev
        print(f"{sigma.real:.9f}") # + {sigma.imag:.9f}j") this is 0. i checked


