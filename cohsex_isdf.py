import numpy as np
from gpu_utils import cp, xp
from wfnreader import WFNReader
from epsreader import EPSReader
import symmetry_maps
from tagged_arrays import LabeledArray, WfnArray
from get_windows import get_window_info
from w_isdf import get_chi0, get_static_w_q
from gamma_matrices import gammas_sparse
import h5py
#import matplotlib.pyplot as plt

# Using the xp alias keeps the code agnostic to NumPy/CuPy, enabling testing on
# CPUs while still targeting GPU acceleration.

# The current implementation focuses on the static COHSEX limit.  Many of the
# routines below (e.g. chi0 and sigma construction) are written in a style that
# follows the complex time shredded propagator (CTSP) formulation so that we can
# later restore full frequency dependence and iterate towards self-consistency.

# return ranges of bands necessary for \sigma_{X,SX,COH}
def get_bandranges(nv, nc, nband, nelec):
    r"""Return ranges of bands necessary for \sigma_{X,SX,COH}"""
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

def get_V_qG(wfn, sym, q0, xp, epshead, sys_dim, do_Dmunu=False):
    # first: V(q,G,G') = 4\pi/|q+G|^2 \delta_{G,G'} * trunc part in 2D, (1-exp(-zc*kxy)*cos(kz*zc))
    # (times one other factor, 1/(N_ktot * cell_volume))
    print(q0)

    # the number of photon polarizations considered in the present calc.
    # (1 long. (Coulomb) + 3 trans. (Breit))
    if do_Dmunu:
        npol = 4
    else:
        npol = 1

    bvec = xp.asarray(wfn.blat * wfn.bvec, dtype=xp.float64)
    q0xp = xp.asarray(q0, dtype=xp.float64)
    qvec = xp.array([xp.float64(0.),xp.float64(0.),xp.float64(0.)])
    zc = xp.pi/bvec[2,2] # note that the crystal z axis must align with the cartesian z axis

    #print("vqg qvec done")
    G_q_crys = xp.zeros((int(wfn.ngkmax),3), dtype=xp.float64)
    G_cart = xp.zeros((int(wfn.ngkmax),3), dtype=xp.float64)
    #print("vqg G_q_crys done")
    V_qG = xp.zeros((npol, npol, sym.nk_tot, int(wfn.ngkmax)), dtype=xp.float64)
    ngks = xp.asarray(wfn.ngk, dtype=xp.int32)
    #print("vqg all arrays done")

    # if sys dim == 3 return error not implemented
    if sys_dim == 3:
        # Future versions will extend the Coulomb truncation to 3D so that
        # layered materials and bulk systems can share the same routines.
        raise NotImplementedError("3D system calculation not yet implemented")
    #print('trying vqg')
    # get V(q,G) array for all sym-reduced q
    if sys_dim == 2:
        for iq in range(wfn.nkpts):
            qvec = xp.asarray(wfn.kpoints[iq])
            if iq == 0:
                qvec = q0xp
            Gmax_q = ngks[iq]
            G_q_crys.fill(0.)
            G_cart.fill(0.)
            # this saves memory in the case of many kpts but requires a lot of HtoD transfers. revisit.
            G_q_crys[:Gmax_q] = xp.asarray(wfn.get_gvec_nk(iq).astype(np.float64),dtype=xp.float64) # stored as int32, trying to convert efficiently
            G_cart[:Gmax_q] = xp.matmul(G_q_crys[:Gmax_q] + qvec, bvec) # @ is super slow, probably using numpy

            V_qG[0,0,iq,:Gmax_q] = xp.divide(4*xp.pi, xp.sum(G_cart*G_cart, axis=1)[:Gmax_q])
            kxy = xp.linalg.norm(G_cart[:Gmax_q,:2], axis=1)
            kz = G_cart[:Gmax_q,2]
            # NOT SURE WHY THERES AN EXTRA 2. 8PI NOT 4PI? I\neq J probably? but i compared to an epsmat.h5 file
            V_qG[0,0,iq,:Gmax_q] *= 2 * (1-xp.exp(-zc*kxy)*xp.cos(kz*zc))

            ###############################################
            # Breit interaction
            ###############################################
            if do_Dmunu:
                # normalize G_cart → shape (Gmax_q,3)
                unitG = xp.divide(G_cart, xp.linalg.norm(G_cart, axis=1, keepdims=True))
                # build projector δ_ij - \hat G_i \hat G_j for each G (→ shape (Gmax_q,3,3))
                proj = xp.eye(3)[None, :, :] - unitG[:Gmax_q, :, None] * unitG[:Gmax_q, None, :]
                # now copy into V_qG.  If your ipol/jpol indices run 1..3, slice 1:4 →
                #   proj.transpose(1,2,0) has shape (3,3,Gmax_q) matching [i,j,iq,g]
                V_qG[1:4, 1:4, iq, :Gmax_q] = proj.transpose(1, 2, 0)

        if do_Dmunu:
            # multiply Breit part by V_c(q)
            xp.multiply(V_qG[1:4,1:4,:,:], V_qG[0,0,:,:], out=V_qG[1:4,1:4,:,:])

        ################################################
        # mini-BZ voronoi monte carlo integration for V_q=0,G=0
        ################################################
        randlims = xp.matmul(bvec.T, xp.matmul(xp.diag(xp.divide(1.0, xp.asarray(wfn.kgrid))), xp.linalg.inv(bvec.T)))
        randvals = xp.random.rand(2500000,3)
        randcart = xp.einsum('ik,jk->ji', bvec.T, randvals)
        wrapped_cart = wrap_points_to_voronoi(randcart, bvec, xp, nmax=1)
        randqcart = xp.einsum('ik,jk->ji', randlims, wrapped_cart) # set of non-grid qpts closer to q=0 than any other qpt
        randqcart[:,2] = 0.0
        rand_vq = xp.divide(4*xp.pi, xp.einsum('ij,ij->i',randqcart,randqcart))
        kxy_q0 = xp.linalg.norm(randqcart[:,:2],axis=1)
        rand_vq *= 2 * (1. - xp.exp(-xp.pi/bvec[2,2] * kxy_q0) * xp.cos(randqcart[:,2] * xp.pi/bvec[2,2]))
        V_qG[0,0,0,0] = xp.mean(rand_vq)
        print(f"V_q=0,G=0 from miniBZ monte carlo: {V_qG[0,0,0,0]:.4f}")

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

        fact = xp.float64(1./wfn.cell_volume) #xp.float64(1./(sym.nk_tot*wfn.cell_volume)) # won't work if nonuniform grid
        V_qG *= fact
        wcoul0 *= fact

    return V_qG.astype(xp.complex128), wcoul0.astype(xp.complex128)

def get_small_psi_component(gvecs, kvec, bvec, psi_G, xp):
    # get alpha/2 (sigma dot (k+G)) psi_nk(G) for bispinor functionality (single k at a time).
    # possible improvements: do sigma dot v, v = p + [r,V_NL+Sigma], add the DKH4 contribution
    halfalpha = xp.complex128(0.00364867628215) # 1/2 * alpha
    sigmadotp = xp.zeros((2,2,gvecs.shape[0]), dtype=xp.complex128)

    gvecsk_cart = xp.matmul(gvecs + kvec, bvec)

    sigmadotp[0,0,:] = gvecsk_cart[:,2]
    sigmadotp[0,1,:] = gvecsk_cart[:,0] - 1j*gvecsk_cart[:,1]
    sigmadotp[1,0,:] = gvecsk_cart[:,0] + 1j*gvecsk_cart[:,1]
    sigmadotp[1,1,:] = -gvecsk_cart[:,2]

    return xp.multiply(halfalpha, xp.einsum('ijG,bjG->biG', sigmadotp, psi_G[:,0:2,:]))




def fft_bandrange(wfn, sym, bandrange, is_left, psi_rtot_out, xp=cp, bispinor=False):
    """
    Get psi_nk(r) for all k-points in the full Brillouin zone.
    (not u_nk(r)! returns psi_nk(r) = e^{ikr} u_nk(r))
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
    nspinor_wfnfile = wfn.nspinor
    nspinor = wfn.nspinor if not bispinor else 4

    # Initialize temporary arrays
    psi_rtot = xp.zeros((nb, nspinor, *wfn.fft_grid), dtype=xp.complex128)

    # Initialize exp(ikr) phase factor arrays 
    fft_nx, fft_ny, fft_nz = wfn.fft_grid
    fx = xp.arange(fft_nx, dtype=float)[None, :, None, None] / fft_nx  # Shape: (nx,1,1)
    fy = xp.arange(fft_ny, dtype=float)[None, None, :, None] / fft_ny  # Shape: (1,ny,1)
    fz = xp.arange(fft_nz, dtype=float)[None, None, None, :] / fft_nz  # Shape: (1,1,nz)

    # Pre-allocate phase arrays
    px = xp.zeros((1,fft_nx, 1, 1), dtype=xp.complex128)
    py = xp.zeros((1,1, fft_ny, 1), dtype=xp.complex128)
    pz = xp.zeros((1,1, 1, fft_nz), dtype=xp.complex128)
    
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
            psi_Gspace[ib, 0:nspinor_wfnfile, :] = xp.asarray(sym.get_cnk_fullzone(wfn,band_idx,k_idx))

        # get small psi component. probably needs performance improvement.
        if bispinor:
            psi_Gspace[:, 2:4, :] = get_small_psi_component(gvecs_k_rot, xp.asarray(sym.unfolded_kpts[k_idx], dtype=xp.float64), xp.asarray(wfn.bvec, dtype=xp.float64), psi_Gspace, xp)
        
        # FFT to real space
        psi_rtot.fill(0.+0.j)
        for ib in range(nb):
            for ispinor in range(nspinor):
                # Place G-space coefficients
                psi_rtot[ib,ispinor,gvecs_k_rot[:,0],gvecs_k_rot[:,1],gvecs_k_rot[:,2]] = psi_Gspace[ib,ispinor,:]
                # Perform FFT
                psi_rtot[ib,ispinor] = xp.fft.ifftn(psi_rtot[ib,ispinor])

        # multiply by exp(ikr) phase factor
        k_gpu = xp.asarray(sym.unfolded_kpts[k_idx], dtype=xp.float64)
        xp.exp(2j * xp.pi * k_gpu[0] * fx, out=px)
        xp.exp(2j * xp.pi * k_gpu[1] * fy, out=py)
        xp.exp(2j * xp.pi * k_gpu[2] * fz, out=pz)
        psi_rtot *= px
        psi_rtot *= py
        psi_rtot *= pz
        
        # Store results with new ordering
        if is_left:
            psi_rtot_out[k_idx] = xp.conj(psi_rtot)
        else:
            psi_rtot_out[k_idx] = psi_rtot

    psi_rtot_out *= xp.sqrt(n_rtot) # fixes to unit norm
    #print("norm of one wfn:", xp.linalg.norm(psi_rtot_out[0,0]))

def get_enk_bandrange(wfn, sym, bandrange, xp):
    nb = bandrange[1] - bandrange[0]
    en_irk = xp.asarray(wfn.energies[0,:,bandrange[0]:bandrange[1]])
    enk = LabeledArray(shape=(sym.nk_tot,nb), axes=['nk', 'nb'])
    
    # needed because WFN.h5 stores sym-reduced enk's. though real, enk's will be complex128's
    # we also use nk as the first index because other arrays use nb as the faster index.
    enk.data = en_irk[sym.irk_to_k_map,:]

    return enk

def get_WminV_qGG(wfn, iqbar, eps0mat, epsmat,xp):
    # get correct qpt index.
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




    WminV= xp.asarray(eps.get_eps_minus_delta_matrix(iqbareps),dtype=xp.complex128) 

    # the following replicates BGW's fixwings.f90: (since epsilon.x doesn't use minibzaverage for vcoul but sigma.x does)
    G0_idx = xp.int32(np.where(eps.gind_eps2rho[iqbareps,:100] == 0)[0][0])
    if iqbar == 0:
        # head
        WminV[G0_idx,G0_idx] = wcoul0/v_qG_epsorder[G0_idx] - 1. # -1 because of delta

        # wing, wing' (the argument is: this is zeroed because it has vanishing phase space for large N_k? Baldereschi & Tosatti 1978)
        WminV[G0_idx,:G0_idx] = 0.0+0.0j
        WminV[G0_idx,G0_idx+1:] = 0.0+0.0j

        WminV[:G0_idx,G0_idx] = 0.0+0.0j
        WminV[G0_idx+1:,G0_idx] = 0.0+0.0j

    WminV *= v_qG_epsorder[:,xp.newaxis]

    return WminV

def get_zeta_q_and_v_q_mu_nu(wfn, sym, centroid_indices, bandrange_l, bandrange_r, V_qG, xp, bispinor=False):
    """Find the interpolative separable density fitting representation."""
    # Get dimensions
    n_rtot = int(np.prod(wfn.fft_grid))
    n_rmu = int(centroid_indices.shape[0])
    nb_l = bandrange_l[1] - bandrange_l[0]
    nb_r = bandrange_r[1] - bandrange_r[0]
    nspinor = wfn.nspinor if not bispinor else 4
    npol = 1 if not bispinor else 4
    kgridgpu = xp.asarray(wfn.kgrid, dtype=xp.int32)

    # Initialize output arrays with (nk, nb) ordering
    psi_rtot_names = ['nk', 'nb', 'nspinor', 'rx', 'ry', 'rz']
    psi_rmu_names = ['nk', 'nb', 'nspinor', 'nrmu']
    psi_l_rtot_out = LabeledArray(shape=(sym.nk_tot, nb_l, nspinor, *wfn.fft_grid), axes=psi_rtot_names)
    psi_r_rtot_out = LabeledArray(shape=(sym.nk_tot, nb_r, nspinor, *wfn.fft_grid), axes=psi_rtot_names)
    psi_l_rmu_out = LabeledArray(shape=(sym.nk_tot, nb_l, nspinor, n_rmu), axes=psi_rmu_names)
    psi_r_rmu_out = LabeledArray(shape=(sym.nk_tot, nb_r, nspinor, n_rmu), axes=psi_rmu_names)


    # Initialize temporary arrays for processing (1 kpt, bands in bandrange) at a time
    psi_l_rtot = xp.zeros((nb_l * nspinor, *wfn.fft_grid), dtype=xp.complex128)
    psi_r_rtot = xp.zeros((nb_r * nspinor, *wfn.fft_grid), dtype=xp.complex128)
    psi_l_rmu = xp.zeros((nb_l * nspinor, n_rmu), dtype=xp.complex128)
    psi_r_rmu = xp.zeros((nb_r * nspinor, n_rmu), dtype=xp.complex128)
    psi_l_rmuT = xp.zeros((n_rmu, nb_l * nspinor), dtype=xp.complex128)
    psi_r_rmuT = xp.zeros((n_rmu, nb_r * nspinor), dtype=xp.complex128)

    # TODO: once the group's distributed GPU linear algebra backend is ready,
    # these explicit buffer allocations will be refactored to call that library
    # for improved scalability across many devices.

    P_l = xp.zeros((n_rmu, n_rtot), dtype=xp.complex128)
    P_r = xp.zeros((n_rmu, n_rtot), dtype=xp.complex128)

    ZCT = xp.zeros((n_rmu, n_rtot), dtype=xp.complex128)
    CCT = xp.zeros((n_rmu, n_rmu), dtype=xp.complex128)

    # note zeta_q only stores one q at a time.
    zeta_q = xp.zeros((n_rmu, n_rtot), dtype=xp.complex128)
    zeta_qG_mu = xp.zeros((n_rmu, int(wfn.ngkmax)), dtype=xp.complex128)

    # Initialize exp(iqr) phase factor arrays outside kq loops
    fft_nx, fft_ny, fft_nz = wfn.fft_grid
    fx = xp.arange(fft_nx, dtype=float)[None, :, None, None] / fft_nx  # Shape: (nx,1,1)
    fy = xp.arange(fft_ny, dtype=float)[None, None, :, None] / fft_ny  # Shape: (1,ny,1)
    fz = xp.arange(fft_nz, dtype=float)[None, None, None, :] / fft_nz  # Shape: (1,1,nz)

    # Pre-allocate phase arrays
    px = xp.zeros((1, fft_nx, 1, 1), dtype=xp.complex128)
    py = xp.zeros((1, 1, fft_ny, 1), dtype=xp.complex128)
    pz = xp.zeros((1, 1, 1, fft_nz), dtype=xp.complex128)

    # initialize output V_q,mu,nu array
    V_qfullG = xp.zeros((int(wfn.ngkmax)), dtype=xp.complex128)
    V_q_names = [
        'nfreq', 'npol1', 'npol2', 'nkx', 'nky', 'nkz', 'nrmu1', 'nrmu2'
    ]

    V_qmunu = LabeledArray(
        shape=(None, npol, npol, *wfn.kgrid, n_rmu, n_rmu),
        axes=V_q_names
    )

    # fill psi_l/r_rtot_out with respective psi(*)_l/r(r) for all k
    print(f"Performing FFTs for wavefunction ranges {bandrange_l} and {bandrange_r}")
    fft_bandrange(wfn, sym, bandrange_l, True, psi_l_rtot_out.data, xp=cp, bispinor=bispinor)
    fft_bandrange(wfn, sym, bandrange_r, False, psi_r_rtot_out.data, xp=cp, bispinor=bispinor)
    print("FFTs complete")
    
    # make WfnArray's that contain psi_nk(r_mu),E_nk together.
    enk_l = get_enk_bandrange(wfn, sym, bandrange_l, xp)
    enk_r = get_enk_bandrange(wfn, sym, bandrange_r, xp)

    ##########################################
    # Loop over all q-points
    # for each q, we get zeta_q via least squares, then immediately get W/v_q,mu,nu so we don't have to store all zeta_q
    ##########################################
    qvec = xp.array([0, 0, 0], dtype=xp.int32)
    for qvec_nonneg in xp.ndindex(*wfn.kgrid):
        # where qvec[i] > ceil(kgrid[i]/2), umklapp to negative (for k<->R FFTs later)
        # note: not necessary
        qvec = xp.asarray(qvec_nonneg)
        if qvec_nonneg[0] > kgridgpu[0] // 2: qvec[0] = qvec_nonneg[0] - kgridgpu[0]
        if qvec_nonneg[1] > kgridgpu[1] // 2: qvec[1] = qvec_nonneg[1] - kgridgpu[1]
        if qvec_nonneg[2] > kgridgpu[2] // 2: qvec[2] = qvec_nonneg[2] - kgridgpu[2]

        ZCT.fill(0)
        CCT.fill(0)

        # note no symmetries used here but it would be possible to do so, though it's not a bottleneck.
        for k_r in range(sym.nk_tot):
            k_l_full = xp.asarray(sym.kvecs_asints[k_r]) - qvec
            k_l_gt0 = xp.mod(k_l_full, kgridgpu)
            k_l = np.where(np.all(sym.kvecs_asints == k_l_gt0.get(), axis=1))[0][0]

            psi_l_rtot = psi_l_rtot_out.slice('nk', k_l, tagged=False).reshape(nb_l * nspinor, *wfn.fft_grid)
            psi_r_rtot = psi_r_rtot_out.slice('nk', k_r, tagged=False).reshape(nb_r * nspinor, *wfn.fft_grid)

            psi_l_rmu = psi_l_rtot[:, centroid_indices[:, 0], centroid_indices[:, 1], centroid_indices[:, 2]]
            psi_r_rmu = psi_r_rtot[:, centroid_indices[:, 0], centroid_indices[:, 1], centroid_indices[:, 2]]
            psi_l_rmuT = xp.ascontiguousarray(psi_l_rmu.T)  # memory locality in matmuls. extra mem may be a negative
            psi_r_rmuT = xp.ascontiguousarray(psi_r_rmu.T)

            # TODO: weight by sqrt(1/E_nk - E_F) for l/r
            #psi_l_rmuT = xp.multiply(psi_l_rmuT, xp.sqrt(xp.divide(1.0,xp.abs(enk_l.data[k_l]-wfn.vbm))))
            #psi_r_rmuT = xp.multiply(psi_r_rmuT, enk_r.data[k_r])

            psi_l_rtot = psi_l_rtot.reshape(nb_l * nspinor, -1)
            psi_r_rtot = psi_r_rtot.reshape(nb_r * nspinor, -1)

            # Add contribution from this k,q pair to ZC^T and CC^T
            Pmu_l = xp.matmul(psi_l_rmuT, psi_l_rmu)
            Pmu_r = xp.matmul(psi_r_rmuT, psi_r_rmu)
            CCT += xp.multiply(Pmu_l, Pmu_r)

            P_l = xp.matmul(psi_l_rmuT, psi_l_rtot)
            P_r = xp.matmul(psi_r_rmuT, psi_r_rtot)
            ZCT += xp.multiply(P_l, P_r)

        # Solve for zeta_q
        zeta_q = xp.linalg.lstsq(CCT, ZCT, rcond=-1)[0]

        # Fourier transform zeta_q to G space
        zeta_q = zeta_q.reshape(n_rmu, *wfn.fft_grid)
        # Remove phase factor from zeta_q(r) = e^{iqr} z_q(r)
        xp.exp(-2j * xp.pi * qvec[0] / kgridgpu[0] * fx, out=px)
        xp.exp(-2j * xp.pi * qvec[1] / kgridgpu[1] * fy, out=py)
        xp.exp(-2j * xp.pi * qvec[2] / kgridgpu[2] * fz, out=pz)
        zeta_q *= px
        zeta_q *= py
        zeta_q *= pz

        for mu in xp.ndindex(zeta_q.shape[0]):
            zeta_q[mu] = xp.fft.fftn(zeta_q[mu])
        #zeta_q *= xp.sqrt(1./n_rtot) # unitary FFT

        #####################################
        # now, get this V_qG from the stored V_qbarG array by remapping G components.
        # from here down is where reordering for the eps components will be necessary.
        #####################################
        qveccrys = xp.divide(xp.asarray(qvec, dtype=xp.float64), kgridgpu)
        q_rounded = xp.round(qveccrys)
        q_ext = xp.where(xp.abs(qveccrys - q_rounded) < 1e-8, q_rounded, qveccrys)
        iq = find_qpoint_index(q_ext, sym, tol=1e-6)
        iq_cpu = iq.get()

        iqbar = sym.irk_to_k_map[iq_cpu]
        Sq = sym.sym_mats_k[sym.irk_sym_map[iq_cpu]]
        G_Sq = np.round(q_ext.get() - Sq @ wfn.kpoints[iqbar]).astype(np.int32)
        vcoul_psiG_comps = xp.asarray(np.einsum('ij,kj->ki', Sq.astype(np.int32), wfn.get_gvec_nk(iqbar)) - G_Sq[np.newaxis, :], dtype=xp.int32)
        V_qfullG.fill(0.0 + 0.0j)
        V_qfullG[:vcoul_psiG_comps.shape[0]] = V_qG[0, 0, iqbar, :vcoul_psiG_comps.shape[0]]

        zeta_qG_mu.fill(0.0 + 0.0j)
        for mu in range(zeta_q.shape[0]):
            zeta_qG_mu[mu, :vcoul_psiG_comps.shape[0]] = zeta_q[mu, vcoul_psiG_comps[:, 0], vcoul_psiG_comps[:, 1], vcoul_psiG_comps[:, 2]]

        temp = xp.multiply(V_qfullG[:, None], zeta_qG_mu.T)
        V_qmunu.data[0, 0, 0, *qvec_nonneg, :, :] = xp.matmul(
            xp.conj(zeta_qG_mu), temp
        )

        #####################################
        # If desired (for debugging), read the BGW eps(0)mat.h5 file and use it to calculate E_QPs.
        #####################################
        #if read_eps == True:
        #    WminV = get_WminV_qGG(wfn,iqbar,eps0mat,epsmat,xp)

        print(f"qpoint {iq} done")

    psi_l_rmu_out.data = psi_l_rtot_out.slice_many({'rx': centroid_indices[:, 0], 'ry': centroid_indices[:, 1], 'rz': centroid_indices[:, 2]})
    psi_r_rmu_out.data = psi_r_rtot_out.slice_many({'rx': centroid_indices[:, 0], 'ry': centroid_indices[:, 1], 'rz': centroid_indices[:, 2]})

    xp.conj(psi_l_rmu_out.data, out=psi_l_rmu_out.data)

    wfn_l = WfnArray(psi_l_rmu_out, enk_l)
    wfn_r = WfnArray(psi_r_rmu_out, enk_r)

    #V_qmunu.data *= sym.nk_tot
    #V_qmunu.data *= -1.0
    V_q = V_qmunu.transpose('nfreq','nkx','nky','nkz','npol1','nrmu1','npol2','nrmu2')

    return V_q, wfn_l, wfn_r


# G_(kab)(mu,nu,t=0) = \sum_mn psi^*_mk(r_mu) * psi_nk(r_nu) (n restricted to range of psi_rmu)
# k goes over kfull
def get_G_mu_nu(wfn, psi_l, psi_r, xp, Gkij=None, return_R=False):
    # using xp to wrap cupy/numpy, calculate:
    # take the matrix psi with shape (nkpts, nbands, nspinor, nrmu) and do:
    # G_{k,a,b}(mu,nu) = \sum_mnab psi^*_mka(r_mu) * psi_nkb(r_nu) (matmul)
    kgrid = xp.asarray(wfn.kgrid)

    if Gkij is None:
        # Initialize Gkij with all variables
        Gkij = LabeledArray(
            shape=(1, *wfn.kgrid, psi_l.psi.shape('nb'), psi_r.psi.shape('nb')),
            axes=['nfreq', 'nkx', 'nky', 'nkz', 'nb1', 'nb2']
        )
        Gkij.join('nkx', 'nky', 'nkz')
        for ik in range(sym.nk_tot):
            xp.fill_diagonal(Gkij.data[0,ik], 1.0)
        

    # nspinor*nrmu
    n_spinor = psi_l.psi.shape('nspinor') 
    nrmu = psi_l.psi.shape('nrmu')
    # dims: nfreq(=0), nk, n_rmu, n_rmu
    Gk_mu_nu_0 = LabeledArray(
        shape=(1, *wfn.kgrid, n_spinor, nrmu, n_spinor, nrmu),
        axes=['nfreq', 'nkx', 'nky', 'nkz', 'nspinor1', 'nrmu1', 'nspinor2', 'nrmu2']
    )
    #Gk_mu_nu_0.join('nkx', 'nky', 'nkz')
    Gk_mu_nu_0.join('nspinor1', 'nrmu1')
    Gk_mu_nu_0.join('nspinor2', 'nrmu2')

    psi_l_tmp = xp.zeros((psi_l.psi.shape('nspinor')*psi_l.psi.shape('nrmu'),psi_l.psi.shape('nb')), dtype=xp.complex128)
    psi_r_tmp = xp.zeros((psi_r.psi.shape('nb'),psi_r.psi.shape('nspinor')*psi_r.psi.shape('nrmu')), dtype=xp.complex128)
    psi_l.psi.join('nspinor','nrmu')
    if psi_l is not psi_r:
        psi_r.psi.join('nspinor','nrmu')

    for kpt in xp.ndindex(*wfn.kgrid):
        k_idx = kpt[0] * wfn.kgrid[1] * wfn.kgrid[2] + kpt[1] * wfn.kgrid[2] + kpt[2]
        
        psi_l_tmp = psi_l.psi.slice('nk', k_idx).T
        psi_r_tmp = xp.conj(psi_r.psi.slice('nk', k_idx))
        Gk_mu_nu_0.data[0,*kpt] = xp.matmul(xp.matmul(psi_l_tmp, Gkij.slice_many({'nfreq':0,'nkx*nky*nkz':k_idx})), psi_r_tmp)

    Gk_mu_nu_0.unjoin('nspinor1', 'nrmu1')
    Gk_mu_nu_0.unjoin('nspinor2', 'nrmu2')

    if not return_R:
        return Gk_mu_nu_0
    else:
        return get_G_R(Gk_mu_nu_0) # kgrid last


def get_G_R(Gk):
    # Reorder axes to have kgrid last (batch fft mem locality)
    Gk = Gk.kgrid_to_last()
    Gk.join('nfreq','nspinor1','nrmu1','nspinor2','nrmu2')  # shape (nfreq*nspin*nrmu*nspin*nrmu, *kgrid)
    # V is umklapped to have kpts in FFT ordering [0,...nk/2,-nk/2+1,...].
    # G doesn't need to be because G_(k+G)(r,r') = G_k(r,r') (bloch fn).
    Gk.ifft_kgrid()
    Gk.unjoin('nfreq','nspinor1','nrmu1','nspinor2','nrmu2')

    return Gk


# get the real-space sigma_\alpha\beta(r,r'(omega))
# options being X, SX, COH
def get_sigma_x_mu_nu(G_R, V_q, xp):
    # sigma_kbar,ab = \sum_(set of k_i = kbar S_i) \sum_qbar G_(k-qbar,ab)(mu,nu) V_qbar(mu,nu)
    # trying in real space! \sum_R G_R W_R. woohoo

    # bispinor case:
    # Sigma_ab = \sum_IJ gamma^I_ac gamma^J_bd G_R,cd V_R,IJ

    V_q = V_q.kgrid_to_last()
    V_q.join('nfreq','npol1','nrmu1','npol2','nrmu2')
    V_q.ifft_kgrid()
    xp.multiply(V_q.data,xp.sqrt(sym.nk_tot),out=V_q.data) # this makes V_R equal to mtxels of 1/|r-r'|
    V_q.unjoin('nfreq','npol1','nrmu1','npol2','nrmu2')

    print("G_R and V_R obtained")

    sigma_R = LabeledArray(
        shape=G_R.data.shape,
        axes=['nfreq', 'nspinor1', 'nrmu1', 'nspinor2', 'nrmu2', 'nkx', 'nky', 'nkz']
    )
    if not bispinor:
        V_q.join('npol1','nrmu1')
        V_q.join('npol2','nrmu2')
        sigma_R.data += xp.multiply(G_R.data,V_q.data[:,xp.newaxis,:,xp.newaxis,:,:,:,:]) # should rename to V_R
    # if bispinor:
    # for gamma_mu,nu in 0,3:
    # G_Rdata_tmp should be xp.einsum('ac,fcmdnxyz,bd',gamma_mu,G_R.data,gamma_nu) (note mn=spatial, cd=spinor)
    # to sigma_R should be added xp.multiply(G_Rdata_tmp,V_q.data[mu,nu])
    # TODO: check this really well
    if bispinor:
        scratch = xp.empty_like(G_R.data[:,0,:,0,:,:,:,:])

        for I, (rI, cI, vI) in enumerate(gammas_sparse):
            for J, (rJ, cJ, vJ) in enumerate(gammas_sparse):
                #target[...] = 0.0
                for p in range(len(vI)):
                    a = int(rI[p])
                    c = int(cI[p])
                    gI = vI[p]
                    for q in range(len(vJ)):
                        b = int(rJ[q])
                        d = int(cJ[q])
                        gJ = vJ[q]
                        target = sigma_R.data[:,a,:,b,:,:,:,:] #slice_many({'gamma1': I, 'gamma2': J})
                        xp.multiply(G_R.slice_many({'nspinor1': c, 'nspinor2': d}),
                                    V_q.slice_many({'npol1': I, 'npol2': J}),
                                    out=scratch)
                        xp.add(target, gI * gJ * scratch, out=target)

    sigma_R.join('nfreq','nspinor1','nrmu1','nspinor2','nrmu2')
    sigma_R.fft_kgrid()
    sigma_R.unjoin('nfreq','nspinor1','nrmu1','nspinor2','nrmu2')
    sigma_R = sigma_R.transpose('nfreq','nkx','nky','nkz','nspinor1','nrmu1','nspinor2','nrmu2')
    sigma_R.join('nkx','nky','nkz')

    sigma_R.data *= -1.0/sym.nk_tot # physical factor for sum over all kpts.
    #sigma_R.data *= -1.0
    return sigma_R


def get_sigma_x_kij(psi_l, psi_r, sigma_kbar, xp):
    r"""
    Calculate the sigma_x_kij matrix elements.
    sigma_mnkbar = \sum_rmu,rnu,s,s' exp(ik(r_nu-r_mu)) u_mk^*(r_mu,s) sigma_kbar,ss'(r_mu,r_nu) u_nk(r_nu,s')
    """
    sigma_kij = xp.zeros((sigma_kbar.shape('nkx*nky*nkz'), psi_l.psi.shape('nb'), psi_r.psi.shape('nb')), 
                         dtype=xp.complex128)  # TODO: should be a labelled array
    sigma_ktmp = xp.zeros((sigma_kbar.shape('nspinor1')*sigma_kbar.shape('nrmu1'), sigma_kbar.shape('nspinor2')*sigma_kbar.shape('nrmu2')), dtype=xp.complex128)
    psi_l_tmp = xp.zeros((psi_l.psi.shape('nb'), psi_l.psi.shape('nspinor')*psi_l.psi.shape('nrmu')), dtype=xp.complex128)
    psi_r_tmp = xp.zeros((psi_r.psi.shape('nspinor')*psi_r.psi.shape('nrmu'),psi_r.psi.shape('nb')), dtype=xp.complex128)

    sigma_kbar.join('nspinor1', 'nrmu1')
    sigma_kbar.join('nspinor2', 'nrmu2')

    psi_l.psi.join('nspinor', 'nrmu')
    if psi_l is not psi_r:
        psi_r.psi.join('nspinor', 'nrmu')

    for kpt in xp.ndindex(*wfn.kgrid):
        k_idx = kpt[0] * wfn.kgrid[1] * wfn.kgrid[2] + kpt[1] * wfn.kgrid[2] + kpt[2]

        sigma_ktmp = sigma_kbar.slice_many({'nfreq': 0, 'nkx*nky*nkz': k_idx})
        psi_l_tmp = xp.conj(psi_l.psi.slice('nk', k_idx))
        psi_r_tmp = psi_r.psi.slice('nk', k_idx).T
        sigma_kij[k_idx, :, :] = xp.matmul(xp.matmul(psi_l_tmp, sigma_ktmp), psi_r_tmp)

    return sigma_kij


def write_sigma_to_file(sigma_kij, filename="eqp0.dat"):
    print(f"sigma_kij dtype before writing: {sigma_kij.dtype}")
    nk, nbands, _ = sigma_kij.shape
    
    with open(filename, 'w') as f:
        for k in range(nk):
            f.write(f"\nk-point {k}:\n")
            f.write("-" * 40 + "\n")
            for n in range(nbands):
                real = float(sigma_kij[k,n,n].real)  # Explicit conversion to float
                imag = float(sigma_kij[k,n,n].imag)
                f.write(f"n={n:<3} {real:>15.6f} + {imag:>15.6f}i\n")

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

def write_labeled_arrays_to_h5(filename, V_qmunu, psi_l, psi_r):
    """
    Write the data of LabeledArray and WfnArray objects to an HDF5 file.
    
    Args:
        filename: Name of the HDF5 file
        V_qmunu: LabeledArray for V_qmunu
        psi_l: WfnArray for left states
        psi_r: WfnArray for right states
    """
    with h5py.File(filename, 'w') as f:
        # Access the underlying numerical data arrays
        V_qmunu_data = V_qmunu.data.get() if isinstance(V_qmunu.data, cp.ndarray) else V_qmunu.data
        
        # Handle WfnArray psi and enk data
        psi_l_data = psi_l.psi.data.get() if isinstance(psi_l.psi.data, cp.ndarray) else psi_l.psi.data
        psi_r_data = psi_r.psi.data.get() if isinstance(psi_r.psi.data, cp.ndarray) else psi_r.psi.data
        enk_l_data = psi_l.enk.data.get() if isinstance(psi_l.enk.data, cp.ndarray) else psi_l.enk.data
        enk_r_data = psi_r.enk.data.get() if isinstance(psi_r.enk.data, cp.ndarray) else psi_r.enk.data

        # Write data arrays
        f.create_dataset('V_qmunu_data', data=V_qmunu_data)
        f.create_dataset('psi_l_data', data=psi_l_data)
        f.create_dataset('psi_r_data', data=psi_r_data)
        f.create_dataset('enk_l_data', data=enk_l_data)
        f.create_dataset('enk_r_data', data=enk_r_data)

def read_labeled_arrays_from_h5(filename):
    """
    Read the data arrays from an HDF5 file and reconstruct LabeledArrays and WfnArrays.

    Args:
        filename (str): The name of the HDF5 file to read from.

    Returns:
        tuple: A tuple containing (V_qmunu, psi_l, psi_r) where V_qmunu is a LabeledArray
              and psi_l/psi_r are WfnArrays.
    """
    with h5py.File(filename, 'r') as f:
        # Read data arrays
        V_qmunu_data = f['V_qmunu_data'][:]
        psi_l_data = f['psi_l_data'][:]
        psi_r_data = f['psi_r_data'][:]
        enk_l_data = f['enk_l_data'][:]
        enk_r_data = f['enk_r_data'][:]

        # Convert to CuPy arrays if a CUDA device is available
        try:
            cp.cuda.runtime.getDeviceCount()
            V_qmunu_data = cp.asarray(V_qmunu_data)
            psi_l_data = cp.asarray(psi_l_data)
            psi_r_data = cp.asarray(psi_r_data)
            enk_l_data = cp.asarray(enk_l_data)
            enk_r_data = cp.asarray(enk_r_data)
        except Exception:
            pass

        # Create LabeledArray for V_qmunu
        V_qmunu = LabeledArray(
            data=V_qmunu_data,
            axes=['nfreq', 'nkx', 'nky', 'nkz', 'npol1', 'nrmu1', 'npol2', 'nrmu2']
        )
        
        # Create LabeledArrays for psi and enk
        psi_l = LabeledArray(
            data=psi_l_data,
            axes=['nk', 'nb', 'nspinor', 'nrmu']
        )
        
        psi_r = LabeledArray(
            data=psi_r_data,
            axes=['nk', 'nb', 'nspinor', 'nrmu']
        )
        
        enk_l = LabeledArray(
            data=enk_l_data,
            axes=['nk', 'nb']
        )
        
        enk_r = LabeledArray(
            data=enk_r_data,
            axes=['nk', 'nb']
        )

        # Create WfnArrays
        psi_l_wfn = WfnArray(psi_l, enk_l)
        psi_r_wfn = WfnArray(psi_r, enk_r)

        return V_qmunu, psi_l_wfn, psi_r_wfn



if __name__ == "__main__":
    # Check GPU availability
    try:
        cp.cuda.runtime.getDeviceCount()
        print(f"Using GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
        mem_info = cp.cuda.runtime.memGetInfo()
        print(f"Memory Usage: {(mem_info[1] - mem_info[0])/1024**2:.1f}MB / {mem_info[1]/1024**2:.1f}MB")
    except Exception:
        print("Using CPU (NumPy)")

    nval = 5
    ncond = 5
    nband = 110

    sys_dim = 2 # 3 for 3D, 2 for 2D (only 2D coulomb interaction implemented currently)

    ryd2ev = 13.6056980659

    wfn = WFNReader("WFNsmall.h5")
    #wfnq = WFNReader("WFNq.h5")
    #eps0 = EPSReader("eps0mat.h5")
    #eps = EPSReader("epsmat.h5")
    sym = symmetry_maps.SymMaps(wfn)
    #q0 = wfnq.kpoints[0] - wfn.kpoints[0]
    #if np.linalg.norm(q0) > 1e-6:
    #    print(f'Using q0 = ({q0[0]:.5f}, {q0[1]:.5f}, {q0[2]:.5f})')

    nvrange, ncrange, nsigmarange, n_fullrange, n_valrange = get_bandranges(nval, ncond, nband, wfn.nelec)
    nvplussigrange = (min(n_valrange),max(nsigmarange))
    ncplussigrange = (min(nsigmarange),max(n_fullrange))

    # Load centroids
    centroids_frac = np.loadtxt('centroids_frac_600.txt')
    n_rmu = int(centroids_frac.shape[0])

    try:
        cp.cuda.runtime.getDeviceCount()
        centroids_frac = cp.asarray(centroids_frac, dtype=cp.float32)
        fft_grid = cp.asarray(wfn.fft_grid, dtype=cp.int32)
    except Exception:
        pass
    centroid_indices = xp.round(centroids_frac * fft_grid).astype(int)
    # Replace any index equal to the grid size with 0 (periodic boundary)
    for i in range(3):
        centroid_indices[centroid_indices[:, i] == wfn.fft_grid[i], i] = 0


    # windows for polarizability and sigma
        # Get window information
    epsq = 0.01
    window_pairs = get_window_info(epsq, wfn)

    # Print detailed information for each window pair
    for i, pair in enumerate(window_pairs, start=1):
        val_window = pair.val_window
        cond_window = pair.cond_window
        # print(f"\nPair {i}")
        # print(f"{'Valence Emin':<15}{'Valence Emax':<15}{'Cond Emin':<15}{'Cond Emax':<15}{'z_lm':<10}")
        # print(f"{val_window.start_energy:<15.3f}{val_window.end_energy:<15.3f}{cond_window.start_energy:<15.3f}{cond_window.end_energy:<15.3f}{pair.z_lm:<10.3f}")
    print('\n')

    # restart: if True, read interp. vectors and V_qmunu from file
    restart = True
    # x_only: if True, skip calculation of Chi(RPA)/W_q, only get exchange self energy
    x_only = False
    do_screened = True
    # bispinor: if True, expand spinor wavefunctions into bispinors, do 4-component calculations
    bispinor = False

    if x_only and do_screened:
        raise ValueError("x_only and do_screened cannot both be True")

    if not restart:
    ####################################
    # 1.) get (truncated in 2D) coulomb potential v_q(G) and W_q=0(G=G'=0) element
    ####################################
        #V_qG, wcoul0 = get_V_qG(wfn, sym, q0, xp, eps0.epshead, sys_dim)
        V_qG, wcoul0 = get_V_qG(wfn, sym,(0.001,0.,0.), xp, 0.2, sys_dim, do_Dmunu=bispinor)


    ####################################
    # 2.) get interpolative separable density fitting basis functions zeta_q,mu(r) and <mu|V_q|nu>
    ####################################
        if x_only:
            V_qmunu, psi_l_rmu_out, psi_r_rmu_out = get_zeta_q_and_v_q_mu_nu(wfn, sym, centroid_indices, n_valrange, nsigmarange, V_qG, xp, bispinor=bispinor)
            write_labeled_arrays_to_h5("taggedarrays.h5", V_qmunu, psi_l_rmu_out, psi_r_rmu_out)
        else:
            V_qmunu, psi_l_rmu_out, psi_r_rmu_out = get_zeta_q_and_v_q_mu_nu(wfn, sym, centroid_indices, nvplussigrange, ncplussigrange, V_qG, xp, bispinor=bispinor)
            write_labeled_arrays_to_h5("taggedarrays.h5", V_qmunu, psi_l_rmu_out, psi_r_rmu_out)
    elif restart and not x_only:
        V_qmunu, psi_l_rmu_out, psi_r_rmu_out = read_labeled_arrays_from_h5("taggedarrays.h5")

    if not x_only:
        chi0 = get_chi0(psi_l_rmu_out, psi_r_rmu_out, window_pairs, wfn, xp)
        # hyperparameters: (1-vX)^-1 = sum_n=0,n_mult (vX)^n, block_f is how many freqs are batched for inversion
        # update: currently inverting directly; i suspect it's ill posed in the low-dim case
        V_for_w = V_qmunu
        W_q = get_static_w_q(chi0, V_for_w, wfn, sym, xp, n_mult=10, block_f=1, bispinor=bispinor)


    psi_l_rmu_out.psi = psi_l_rmu_out.psi.slice('nb',xp.s_[:wfn.nelec],tagged=True)
    psi_r_rmu_out.psi = psi_r_rmu_out.psi.slice('nb',xp.s_[:nval+ncond],tagged=True)

    #################################### 
    # 4.) get G_k(r_mu,r_nu) for valence bands
    ####################################
    G_R_val_mu_nu = get_G_mu_nu(wfn, psi_l_rmu_out, psi_l_rmu_out, xp, return_R=True)

    ####################################
    # 5.) get sigma_mnk from V_q,mu,nu and G_k(r_mu,r_nu)
    ####################################
    if do_screened:
        sigma_in = W_q
    else:
        if bispinor:
            V_qmunu = V_qmunu.transpose('nfreq','nkx','nky','nkz','npol1','nrmu1','npol2','nrmu2')

        sigma_in = V_qmunu
    #if bispinor and sigma_in is V_qmunu:
    #    sigma_in = V_qmunu.slice_many({'npol1': 0, 'npol2': 0}, tagged=True)
    sigma_x_kbar_munu = get_sigma_x_mu_nu(G_R_val_mu_nu, sigma_in, xp)
    sigma_x_kbar_ij = get_sigma_x_kij(psi_r_rmu_out, psi_r_rmu_out, sigma_x_kbar_munu, xp)


    write_sigma_to_file(ryd2ev*sigma_x_kbar_ij, "eqp0_noqsym.dat")

    # Later stages of this project will iterate this workflow so that the COHSEX
    # potential feeds back into updated wavefunctions (self-consistent COHSEX)
    # and eventually into a full quasiparticle self-consistent GW cycle.
