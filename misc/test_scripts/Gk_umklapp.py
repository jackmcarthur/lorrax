import numpy as np
from gpu_utils import cp, xp, fftx, GPU_AVAILABLE
from wfnreader import WFNReader
from epsreader import EPSReader
import symmetry_maps
import cohsex_noqsym
#import matplotlib.pyplot as plt

# return ranges of bands necessary for \sigma_{X,SX,COH}
def get_bandranges(nv,nc,nband,nelec):
    """Return ranges of bands necessary for \sigma_{X,SX,COH}"""
    nvrange = [int(nelec-nv), int(nelec)]
    ncrange = [int(nelec), int(nelec+nc)]
    nsigmarange = [int(nelec-nv), int(nelec+nc)]
    n_fullrange = [0, int(nband)]
    n_valrange = [0, int(nelec)]
    return nvrange, ncrange, nsigmarange, n_fullrange, n_valrange



if __name__ == "__main__":
    nval = 5
    ncond = 5
    nband = 30

    sys_dim = 2 # 3 for 3D, 2 for 2D

    ryd2ev = 13.6056980659

    wfn = WFNReader("WFN.h5")
    wfnq = WFNReader("WFNq.h5")
    eps0 = EPSReader("eps0mat.h5")
    eps = EPSReader("epsmat.h5")
    sym = symmetry_maps.SymMaps(wfn)
    q0 = wfnq.kpoints[0] - wfn.kpoints[0]
    if np.linalg.norm(q0) > 1e-6:
        print(f'Using q0 = ({q0[0]:.5f}, {q0[1]:.5f}, {q0[2]:.5f})')

    nvrange, ncrange, nsigmarange, n_fullrange, n_valrange = get_bandranges(nval, ncond, nband, wfn.nelec)

    # Load centroids
    centroids_frac = np.loadtxt('centroids_frac.txt')
    n_rmu = int(centroids_frac.shape[0])

    if GPU_AVAILABLE:
        centroids_frac = cp.asarray(centroids_frac, dtype=cp.float32)
        fft_grid = cp.asarray(wfn.fft_grid, dtype=cp.int32)
    centroid_indices = xp.round(centroids_frac * fft_grid).astype(int)
    # Replace any index equal to the grid size with 0 (periodic boundary)
    for i in range(3):
        centroid_indices[centroid_indices[:, i] == wfn.fft_grid[i], i] = 0

    # V_qmunu, psi_l_rmu_out, psi_r_rmu_out, zeta_qfinal_test = cohsex_noqsym.get_zeta_q_and_v_q_mu_nu(wfn, sym, centroid_indices, n_valrange, nsigmarange, 0, xp)

    # G_k_mu_nu = cohsex_noqsym.get_Gk_mu_nu(psi_l_rmu_out, psi_r_rmu_out, n_rmu, xp)
    # print(G_k_mu_nu.shape)



    # # saving a dim to include nfreq!
    # Gkij = xp.zeros((1,sym.nk_tot, psi_l_rmu_out.shape[1], psi_r_rmu_out.shape[1]), dtype=xp.complex128)
    # # Fill diagonal with ones (TODO: assuming valence bands!!!!!!!!)
    # for ik in range(sym.nk_tot):
    #     xp.fill_diagonal(Gkij[0,ik], 1.0)

    # # nspinor*nrmu
    # n_spinmu = psi_l_rmu_out.shape[2]*psi_l_rmu_out.shape[3]
    # # dims: nfreq(=0), nk, n_rmu, n_rmu
    # Gk_mu_nu_0 = xp.zeros((1,sym.nk_tot,n_spinmu,n_spinmu), dtype=xp.complex128)


    # for nk in xp.ndindex(sym.nk_tot):
    #     psi_l = xp.conj(psi_l_rmu_out[nk,:,:,:].reshape(-1,n_spinmu)).T
    #     psi_r = psi_r_rmu_out[nk,:,:,:].reshape(-1,n_spinmu)
    #     Gk_mu_nu_0[0,nk,:,:] = xp.matmul(xp.matmul(psi_l, Gkij[0,nk]), psi_r)

    # Gk_umklapp_compare = xp.zeros_like(G_k_mu_nu[0,0])

    print(wfn.atom_positions[:,2])
    print(wfn.alat * wfn.avec @ wfn.atom_positions[:,2])

