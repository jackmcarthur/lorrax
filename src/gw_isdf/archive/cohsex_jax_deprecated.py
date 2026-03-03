# Standard Library imports
import os
# Force JAX to create four CPU devices before import
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=1"
os.environ.setdefault("JAX_ENABLE_X64", "1")
import argparse
import configparser

import numpy as np
from isdf.common.gpu_utils import cp, xp
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial
#jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_platform_name", "cpu")

# Global mesh for sharding across bands
mesh_bands = Mesh(np.asarray(jax.devices()), ("bands",))
from isdf.common.wfnreader import WFNReader
from isdf.common.epsreader import EPSReader
from isdf.common import symmetry_maps
from isdf.common.tagged_arrays import LabeledArray, WfnArray
from .get_windows import get_window_info
from .w_isdf import get_chi0, get_static_w_q
from isdf.common import Meta
from isdf.common.gamma_matrices import gammas_sparse
import h5py

def read_cohsex_input(filename: str) -> dict:
    """Parse a simple INI-style input file for the COHSEX driver."""
    parser = configparser.ConfigParser()
    parser.read(filename)
    section = parser["cohsex"] if "cohsex" in parser else parser[parser.sections()[0]]

    getb = section.getboolean
    get = section.get
    geti = section.getint

    return {
        "restart": getb("restart", fallback=True),
        "x_only": getb("x_only", fallback=False),
        "do_screened": getb("do_screened", fallback=True),
        "bispinor": getb("bispinor", fallback=False),
        "wfn_file": get("wfn_file", fallback="WFN.h5"),
        "centroids_file": get("centroids_file", fallback="centroids_frac.txt"),
        "output_file": get("output_file", fallback="eqp0_noqsym.dat"),
        "nval": geti("nval", fallback=5),
        "ncond": geti("ncond", fallback=5),
        "nband": geti("nband", fallback=100),
        "sys_dim": geti("sys_dim", fallback=2),
    }


# Using the xp alias keeps the code agnostic to NumPy/CuPy, enabling testing on
# CPUs while still targeting GPU acceleration.

# The current implementation focuses on the static COHSEX limit.  Many of the
# routines below (e.g. chi0 and sigma construction) are written in a style that
# follows the complex time shredded propagator (CTSP) formulation so that we can
# later restore full frequency dependence and iterate towards self-consistency.


# return ranges of bands necessary for \sigma_{X,SX,COH}
def get_bandranges(nv, nc, nband, nelec):
    r"""Return ranges of bands necessary for \sigma_{X,SX,COH}"""
    nvrange = [int(nelec - nv), int(nelec)]
    ncrange = [int(nelec), int(nelec + nc)]
    nsigmarange = [int(nelec - nv), int(nelec + nc)]
    n_fullrange = [0, int(nband)]
    n_valrange = [0, int(nelec)]
    return nvrange, ncrange, nsigmarange, n_fullrange, n_valrange


def wrap_points_to_voronoi(randcart, bvec, xp, nmax=1):
    """
    Helper function to get test q-points for mini-BZ average with correct voronoi cell.
    """
    # 1. Generate all candidate integer translations.
    grid = xp.arange(-nmax, nmax + 1)
    # meshgrid in 3D; shape will be (3, M) with M = (2*nmax+1)**3 candidates.
    shifts = xp.stack(xp.meshgrid(grid, grid, grid, indexing="ij"), axis=-1).reshape(
        -1, 3
    )

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


def get_V_qG(wfn, sym, q0, xp, epshead, sys_dim, meta: Meta, do_Dmunu=False):
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
    qvec = xp.array([xp.float64(0.0), xp.float64(0.0), xp.float64(0.0)])
    zc = (
        xp.pi / bvec[2, 2]
    )  # note that the crystal z axis must align with the cartesian z axis

    # print("vqg qvec done")
    G_q_crys = xp.zeros((int(wfn.ngkmax), 3), dtype=xp.float64)
    G_cart = xp.zeros((int(wfn.ngkmax), 3), dtype=xp.float64)
    # print("vqg G_q_crys done")
    V_qG = xp.zeros((npol, npol, sym.nk_tot, int(wfn.ngkmax)), dtype=xp.float64)
    ngks = xp.asarray(wfn.ngk, dtype=xp.int32)
    # print("vqg all arrays done")

    # if sys dim == 3 return error not implemented
    if sys_dim == 3:
        # Future versions will extend the Coulomb truncation to 3D so that
        # layered materials and bulk systems can share the same routines.
        raise NotImplementedError("3D system calculation not yet implemented")
    # print('trying vqg')
    # get V(q,G) array for all sym-reduced q
    if sys_dim == 2:
        for iq in range(wfn.nkpts):
            qvec = xp.asarray(wfn.kpoints[iq])
            if iq == 0:
                qvec = q0xp
            Gmax_q = ngks[iq]
            G_q_crys.fill(0.0)
            G_cart.fill(0.0)
            # this saves memory in the case of many kpts but requires a lot of HtoD transfers. revisit.
            G_q_crys[:Gmax_q] = xp.asarray(
                wfn.get_gvec_nk(iq).astype(np.float64), dtype=xp.float64
            )  # stored as int32, trying to convert efficiently
            G_cart[:Gmax_q] = xp.matmul(
                G_q_crys[:Gmax_q] + qvec, bvec
            )  # @ is super slow, probably using numpy

            V_qG[0, 0, iq, :Gmax_q] = xp.divide(
                4 * xp.pi, xp.sum(G_cart * G_cart, axis=1)[:Gmax_q]
            )
            kxy = xp.linalg.norm(G_cart[:Gmax_q, :2], axis=1)
            kz = G_cart[:Gmax_q, 2]
            # NOT SURE WHY THERES AN EXTRA 2. 8PI NOT 4PI? I\neq J probably? but i compared to an epsmat.h5 file
            V_qG[0, 0, iq, :Gmax_q] *= 2 * (1 - xp.exp(-zc * kxy) * xp.cos(kz * zc))

            ###############################################
            # Breit interaction
            ###############################################
            if do_Dmunu:
                # normalize G_cart → shape (Gmax_q,3)
                unitG = xp.divide(G_cart, xp.linalg.norm(G_cart, axis=1, keepdims=True))
                # build projector δ_ij - \hat G_i \hat G_j for each G (→ shape (Gmax_q,3,3))
                proj = (
                    xp.eye(3)[None, :, :]
                    - unitG[:Gmax_q, :, None] * unitG[:Gmax_q, None, :]
                )
                # now copy into V_qG.  If your ipol/jpol indices run 1..3, slice 1:4 →
                #   proj.transpose(1,2,0) has shape (3,3,Gmax_q) matching [i,j,iq,g]
                V_qG[1:4, 1:4, iq, :Gmax_q] = proj.transpose(1, 2, 0)

        if do_Dmunu:
            # multiply Breit part by V_c(q)
            xp.multiply(
                V_qG[1:4, 1:4, :, :], V_qG[0, 0, :, :], out=V_qG[1:4, 1:4, :, :]
            )

        ################################################
        # mini-BZ voronoi monte carlo integration for V_q=0,G=0
        ################################################
        randlims = xp.matmul(
            bvec.T,
            xp.matmul(
                xp.diag(xp.divide(1.0, xp.asarray((meta.nkx, meta.nky, meta.nkz)))),
                xp.linalg.inv(bvec.T),
            ),
        )
        randvals = xp.random.rand(2500000, 3)
        randcart = xp.einsum("ik,jk->ji", bvec.T, randvals)
        wrapped_cart = wrap_points_to_voronoi(randcart, bvec, xp, nmax=1)
        randqcart = xp.einsum(
            "ik,jk->ji", randlims, wrapped_cart
        )  # set of non-grid qpts closer to q=0 than any other qpt
        randqcart[:, 2] = 0.0
        rand_vq = xp.divide(4 * xp.pi, xp.einsum("ij,ij->i", randqcart, randqcart))
        kxy_q0 = xp.linalg.norm(randqcart[:, :2], axis=1)
        rand_vq *= 2 * (
            1.0
            - xp.exp(-xp.pi / bvec[2, 2] * kxy_q0)
            * xp.cos(randqcart[:, 2] * xp.pi / bvec[2, 2])
        )
        V_qG[0, 0, 0, 0] = xp.mean(rand_vq)
        print(f"V_q=0,G=0 from miniBZ monte carlo: {V_qG[0,0,0,0]:.4f}")

        ##############################################################
        # this is wcoul0 used in BGW/Common/fixwing.f90 (generated in minibzaverage.f90)
        # equations here are: (Ismail-Beigi PRB 2006)
        # W(q,G=G'=0) = epsinv(q,G=G'=0) * vc(q)
        # 1/epsinv(q,G=G'=0) = 1 + vc(q)*f(q)
        # f(q) = gamma |q|^2 exp(-a|q|) (a=0 in minibzaverage.f90)

        q0len = xp.linalg.norm(xp.matmul(q0xp, bvec))
        vc_qtozero = (1.0 - xp.exp(-q0len * zc)) / q0len**2
        gamma = xp.float64(
            (1.0 / xp.asarray(epshead.real, dtype=xp.float64) - 1.0)
            / (q0len**2 * vc_qtozero)
        )
        alpha = xp.float64(0.0)

        rand_wq = (1.0 - xp.exp(-kxy_q0 * zc)) / (kxy_q0**2)  # actually vc(q)
        rand_wq = xp.divide(
            rand_wq, (1.0 + rand_wq * kxy_q0**2 * gamma * xp.exp(-alpha * kxy_q0))
        )
        wcoul0 = 8 * xp.pi * xp.mean(rand_wq)

        print(f"W_q=0(G=G'=0) from miniBZ monte carlo: {wcoul0:.4f}")

        fact = xp.float64(
            1.0 / wfn.cell_volume
        )  # xp.float64(1./(sym.nk_tot*wfn.cell_volume)) # won't work if nonuniform grid
        V_qG *= fact
        wcoul0 *= fact

    return V_qG.astype(xp.complex128), wcoul0.astype(xp.complex128)


def get_small_psi_component(gvecs, kvec, bvec, psi_G, xp):
    # get alpha/2 (sigma dot (k+G)) psi_nk(G) for bispinor functionality (single k at a time).
    # possible improvements: do sigma dot v, v = p + [r,V_NL+Sigma], add the DKH4 contribution
    halfalpha = xp.complex128(0.00364867628215)  # 1/2 * alpha
    sigmadotp = xp.zeros((2, 2, gvecs.shape[0]), dtype=xp.complex128)

    gvecsk_cart = xp.matmul(gvecs + kvec, bvec)

    sigmadotp[0, 0, :] = gvecsk_cart[:, 2]
    sigmadotp[0, 1, :] = gvecsk_cart[:, 0] - 1j * gvecsk_cart[:, 1]
    sigmadotp[1, 0, :] = gvecsk_cart[:, 0] + 1j * gvecsk_cart[:, 1]
    sigmadotp[1, 1, :] = -gvecsk_cart[:, 2]

    return xp.multiply(
        halfalpha, xp.einsum("ijG,bjG->biG", sigmadotp, psi_G[:, 0:2, :])
    )


# this is the worst function in the code because it is nontrivial to read
# different parts of the .h5 file for each task.
def fft_bandrange(
    wfn, sym, bandrange, is_left, meta: Meta, bispinor=False
):
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

    # Create 2D mesh with host and device dimensions
    mesh = Mesh(
        np.array(jax.devices()).reshape(jax.process_count(), jax.local_device_count()), 
        ['host', 'dev']
    )
    
    # Calculate distribution across all devices
    total_devices = jax.process_count() * jax.local_device_count()
    devices_per_host = jax.local_device_count()
    host_id = jax.process_index()
    
    # Bands per device (rounded up)
    bands_per_device = (nb + total_devices - 1) // total_devices
    total_bands_padded = total_devices * bands_per_device

    # Initialize exp(ikr) phase factor arrays
    fx = jnp.arange(meta.fft_grid[0], dtype=float)[None, :, None, None] / meta.fft_grid[0]
    fy = jnp.arange(meta.fft_grid[1], dtype=float)[None, None, :, None] / meta.fft_grid[1]
    fz = jnp.arange(meta.fft_grid[2], dtype=float)[None, None, None, :] / meta.fft_grid[2]

    # Pre-allocate phase arrays
    px = jnp.zeros((1, meta.fft_grid[0], 1, 1), dtype=jnp.complex128)
    py = jnp.zeros((1, 1, meta.fft_grid[1], 1), dtype=jnp.complex128)
    pz = jnp.zeros((1, 1, 1, meta.fft_grid[2]), dtype=jnp.complex128)

    @partial(jax.jit)
    def fft_psi_jax(psi_Gspace, psi_Gtot, gvecs_k_rot):
        psi_Gtot = psi_Gtot.at[
            :,:,gvecs_k_rot[:,0],gvecs_k_rot[:,1],gvecs_k_rot[:,2]
        ].set(psi_Gspace)
        return jnp.fft.ifftn(psi_Gtot, axes=(-3, -2, -1))

    # Process each local device separately for memory efficiency
    local_shards = []
    for local_dev_idx in range(devices_per_host):
        device = jax.local_devices()[local_dev_idx]
        global_dev_idx = host_id * devices_per_host + local_dev_idx
        
        # Calculate band range for this specific device
        dev_start = global_dev_idx * bands_per_device + bandrange[0]
        dev_end = min((global_dev_idx + 1) * bands_per_device + bandrange[0], bandrange[1])
        
        # Skip if this device has no bands to process
        if dev_start >= bandrange[1]:
            dev_start = bandrange[1]
            dev_end = bandrange[1]
        if dev_end > bandrange[1]:
            dev_end = bandrange[1]
            
        dev_nb = dev_end - dev_start
        
        # Allocate arrays for this device only
        psi_Gtot = jnp.zeros((bands_per_device, meta.nspinor, *meta.fft_grid), dtype=jnp.complex128)
        psi_rtot_dev = jnp.zeros((meta.nk_tot, bands_per_device, meta.nspinor, *meta.fft_grid), dtype=jnp.complex128)

        # Loop over all k-points for this device
        for k_idx in range(sym.nk_tot):
            k_red = sym.irk_to_k_map[k_idx]
            psi_Gspace = jnp.zeros((bands_per_device, meta.nspinor, wfn.ngk[k_red]), dtype=jnp.complex128)

            # Get G-vectors and rotate them
            gvecs_k_rot = jnp.asarray(sym.get_gvecs_kfull(wfn, k_idx))
            
            # Load wavefunction coefficients for this device's band range
            for ib, band_idx in enumerate(range(dev_start, dev_end)):
                psi_Gspace = psi_Gspace.at[ib, 0:meta.nspinor_wfnfile, :].set(
                    jnp.asarray(sym.get_cnk_fullzone(wfn, band_idx, k_idx))
                )

            # get small psi component
            if bispinor:
                psi_Gspace = psi_Gspace.at[:dev_nb, 2:4, :].set(
                    get_small_psi_component(
                        gvecs_k_rot,
                        jnp.asarray(sym.unfolded_kpts[k_idx], dtype=jnp.float64),
                        jnp.asarray(wfn.bvec, dtype=jnp.float64),
                        psi_Gspace[:dev_nb],
                        jnp,
                    )
                )

            # FFT to real space (only for actual bands)
            if dev_nb > 0:
                fft_result = fft_psi_jax(psi_Gspace[:dev_nb], psi_Gtot[:dev_nb], gvecs_k_rot)
                psi_rtot_dev = psi_rtot_dev.at[k_idx, :dev_nb, :].set(fft_result)
            
                # multiply by exp(ikr) phase factor
                k_gpu = jnp.asarray(sym.unfolded_kpts[k_idx], dtype=jnp.float64)
                px = jnp.exp(2j * jnp.pi * k_gpu[0] * fx)
                py = jnp.exp(2j * jnp.pi * k_gpu[1] * fy)
                pz = jnp.exp(2j * jnp.pi * k_gpu[2] * fz)
                psi_rtot_dev = psi_rtot_dev.at[k_idx, :dev_nb].set(psi_rtot_dev[k_idx, :dev_nb] * px * py * pz)

        # Keep shard on CPU for concatenation, will device_put the final result
        local_shards.append(psi_rtot_dev)

    # Combine all local shards into a host-local array (all on CPU)
    host_array = jnp.concatenate(local_shards, axis=1)
    
    # Create the global array using make_array_from_process_local_data
    global_shape = (meta.nk_tot, total_bands_padded, meta.nspinor, *meta.fft_grid)
    band_sharding = NamedSharding(mesh, P(None, 'host', None, None, None, None))
    global_psi_rtot = jax.make_array_from_process_local_data(band_sharding, host_array, global_shape)

    # Apply final transformations but keep padded shape
    if is_left:
        global_psi_rtot = jnp.conj(global_psi_rtot)
    global_psi_rtot = global_psi_rtot * jnp.sqrt(meta.n_rtot)
    
    # Return padded arrays - trimming will be handled later when needed
    return global_psi_rtot


def get_enk_bandrange(wfn, sym, bandrange, sigma_bandrange, xp):
    nb = bandrange[1] - bandrange[0]
    en_irk = xp.asarray(wfn.energies[0, :, bandrange[0] : bandrange[1]])
    enk = LabeledArray(shape=(sym.nk_tot, nb), axes=["nk", "nb"])

    # needed because WFN.h5 stores sym-reduced enk's. though real, enk's will be complex128's
    # we also use nk as the first index because other arrays use nb as the faster index.
    enk.data = en_irk[sym.irk_to_k_map, :]

    # LSTSQ WEIGHTS
    # best chi obtained not when residuals of the full M(cvkq) are minimized,
    # but when residuals of M(cvkq)/sqrt(E_ck-q - E_vk) are minimized.
    # we can nearly account for this by:
    # picking the smallest energy in the sigma bandrange, giving cond states weights 1/sqrt(E_ck - E_min)
    # picking the largest energy in the sigma bandrange, giving val states weights 1/sqrt(E_max - E_vk)
    # set all weights in the sigma bandrange to the max conduction and valence weights, then
    # normalize such that those weights are 1.0.
    # this produces a very large (order of magnitude) improvement in SX energies.
    sigma_start, sigma_end = sigma_bandrange
    enk_sigma_start = max([sigma_start - bandrange[0], 0])
    enk_sigma_end = min([sigma_end - bandrange[0], nb])
    energies_sym = xp.asarray(wfn.energies[0, :, :])  # (nk_sym, nband_total)
    energies_full = energies_sym[sym.irk_to_k_map, :]  # (nk_full, nband_total)
    energies_sigma = energies_full[:, sigma_start:sigma_end]  # (nk_full, n_sigma)
    # Reference energies per k-point
    E_min = xp.min(energies_sigma)  # (nk_full,)
    E_max = xp.max(energies_sigma)  # (nk_full,)
    # Determine valence vs conduction bands within sigma range
    # band_idxs = xp.arange(sigma_start, sigma_end)[None, :]  # (1, n_sigma)
    mask_val = enk.data <= wfn.efermi  # (1, n_sigma) broadcast to (nk_full, n_sigma)
    # Compute weights
    val_weights = 1.0 / xp.sqrt((E_max - enk.data))
    cond_weights = 1.0 / xp.sqrt((enk.data - E_min))
    weights_full = xp.where(mask_val, val_weights, cond_weights)
    # Normalize so the maximum weight across the sigma range is 1.0
    wmax = xp.max(weights_full)
    weights_full = weights_full / wmax
    weights_full[:, enk_sigma_start:enk_sigma_end] = 1.0
    # TODO: in bispinor case repeats should = 4
    return enk, xp.repeat(weights_full, repeats=2, axis=1)

def get_zeta_q_and_v_q_mu_nu(
    wfn,
    sym,
    centroid_indices,
    bandrange_l,
    bandrange_r,
    V_qG,
    meta: Meta,
    xp,
    bispinor=False,
):
    """Find the interpolative separable density fitting representation."""
    # Get dimensions with padding for distributed computation
    nb_l = bandrange_l[1] - bandrange_l[0]
    nb_r = bandrange_r[1] - bandrange_r[0]
    total_devices = jax.process_count() * jax.local_device_count()
    bands_per_device_l = (nb_l + total_devices - 1) // total_devices
    bands_per_device_r = (nb_r + total_devices - 1) // total_devices
    nb_l_padded = total_devices * bands_per_device_l
    nb_r_padded = total_devices * bands_per_device_r
    kgridgpu = xp.asarray((meta.nkx, meta.nky, meta.nkz), dtype=xp.int32)

    # Initialize output arrays with (nk, nb) ordering - use original sizes for output
    psi_rtot_names = ["nk", "nb", "nspinor", "rx", "ry", "rz"]
    psi_rmu_names = ["nk", "nb", "nspinor", "nrmu"]
    psi_l_rtot_out = LabeledArray(
        shape=(meta.nk_tot, nb_l, meta.nspinor, *meta.fft_grid), axes=psi_rtot_names
    )
    psi_r_rtot_out = LabeledArray(
        shape=(meta.nk_tot, nb_r, meta.nspinor, *meta.fft_grid), axes=psi_rtot_names
    )
    psi_l_rmu_out = LabeledArray(
        shape=(meta.nk_tot, nb_l, meta.nspinor, meta.n_rmu), axes=psi_rmu_names
    )
    psi_r_rmu_out = LabeledArray(
        shape=(meta.nk_tot, nb_r, meta.nspinor, meta.n_rmu), axes=psi_rmu_names
    )

    # initialize output V_q,mu,nu array
    V_qfullG = xp.zeros((int(wfn.ngkmax)), dtype=xp.complex128)
    V_q_names = ["nfreq", "npol1", "npol2", "nkx", "nky", "nkz", "nrmu1", "nrmu2"]

    V_qmunu = LabeledArray(
        shape=(None, meta.npol, meta.npol, meta.nkx, meta.nky, meta.nkz, meta.n_rmu, meta.n_rmu),
        axes=V_q_names,
    )

    # fill psi_l/r_rtot_out with respective psi(*)_l/r(r) for all k
    print(f"Performing FFTs for wavefunction ranges {bandrange_l} and {bandrange_r}")
    psi_l_rtot_padded = fft_bandrange(
        wfn, sym, bandrange_l, False, meta, bispinor=bispinor
    )
    psi_r_rtot_padded = fft_bandrange(
        wfn, sym, bandrange_r, False, meta, bispinor=bispinor
    )
    # Debug: show how the arrays are sharded across devices
    print("psi_l_rtot_padded sharding:", psi_l_rtot_padded.sharding)
    print("psi_r_rtot_padded sharding:", psi_r_rtot_padded.sharding)
    
    # Record original band counts for trimming later
    nb_l_original = nb_l
    nb_r_original = nb_r
    
    # Compute 2D processor grid for optimal distributed linear algebra
    total_procs = jax.process_count() * jax.local_device_count()
    
    # Check if processor grid is specified in meta, otherwise compute near-square
    if hasattr(meta, 'proc_grid_x') and hasattr(meta, 'proc_grid_y'):
        grid_x, grid_y = meta.proc_grid_x, meta.proc_grid_y
        if grid_x * grid_y != total_procs:
            raise ValueError(f"Processor grid {grid_x}x{grid_y}={grid_x*grid_y} doesn't match total procs {total_procs}")
    else:
        # Default: near-square grid with X < Y
        grid_x = int(np.sqrt(total_procs))
        while total_procs % grid_x != 0:
            grid_x -= 1
        grid_y = total_procs // grid_x
        if grid_x > grid_y:
            grid_x, grid_y = grid_y, grid_x  # Ensure X < Y
    
    print(f"Using {grid_x}x{grid_y} processor grid for distributed linear algebra")
    
    # Create 2D mesh for distributed computation  
    devices_2d = np.array(jax.devices()).reshape(grid_x, grid_y)
    mesh_2d = jax.sharding.Mesh(devices_2d, ['proc_x', 'proc_y'])
    
    # Keep original 1D mesh for band-sharded wavefunctions
    mesh_1d = jax.sharding.Mesh(np.array(jax.devices()).reshape(jax.process_count(), jax.local_device_count()), ['host', 'dev'])
    
    # Compute padding for mu and rtot dimensions to be divisible by grid
    n_rmu_padded = ((meta.n_rmu + grid_y - 1) // grid_y) * grid_y
    n_rtot_padded = ((meta.n_rtot + grid_y - 1) // grid_y) * grid_y
    mu_pad_amount = n_rmu_padded - meta.n_rmu
    rtot_pad_amount = n_rtot_padded - meta.n_rtot
    
    print(f"Padding: n_rmu {meta.n_rmu} -> {n_rmu_padded} (+{mu_pad_amount})")
    print(f"Padding: n_rtot {meta.n_rtot} -> {n_rtot_padded} (+{rtot_pad_amount})")
    
    # Clean sharding setup for optimal distributed computation
    print("Setting up clean 2D sharding for CCT/ZCT computation...")
    
    # Convert centroid_indices to JAX array
    centroid_indices_cpu = centroid_indices.get() if hasattr(centroid_indices, 'get') else centroid_indices
    centroids = jnp.asarray(centroid_indices_cpu)
    
    @partial(jax.jit) 
    def extract_centroids(psi_rtot, centroid_indices):
        return psi_rtot[:, centroid_indices[:, 0], centroid_indices[:, 1], centroid_indices[:, 2]]
    
    # 1. Extract centroids while still band-sharded
    def extract_centroids_simple(psi_rtot, centroids):
        psi_rmu = jnp.zeros((meta.nk_tot, psi_rtot.shape[1], psi_rtot.shape[2], meta.n_rmu), dtype=jnp.complex128)
        for k in range(meta.nk_tot):
            psi_k_flat = psi_rtot[k].reshape(-1, *meta.fft_grid)
            psi_k_rmu = extract_centroids(psi_k_flat, centroids)
            psi_rmu = psi_rmu.at[k].set(psi_k_rmu.reshape(psi_rtot.shape[1], psi_rtot.shape[2], meta.n_rmu))
        return psi_rmu
    
    psi_l_rmu = extract_centroids_simple(psi_l_rtot_padded, centroids)
    psi_r_rmu = extract_centroids_simple(psi_r_rtot_padded, centroids)
    
    # 2. Reshard mu/tot arrays over Y dimension (fixing band padding)
    y_sharding = NamedSharding(mesh_2d, P(None, None, None, 'proc_y'))  # shard last dim over Y
    psi_l_rmu = jax.lax.with_sharding_constraint(psi_l_rmu, y_sharding)     # (k, nb, nspinor, n_rmu) Y-sharded
    psi_r_rmu = jax.lax.with_sharding_constraint(psi_r_rmu, y_sharding)
    psi_l_rtot = jax.lax.with_sharding_constraint(psi_l_rtot_padded.reshape(meta.nk_tot, -1, meta.nspinor, meta.n_rtot), y_sharding)  # (k, nb, nspinor, n_rtot) Y-sharded  
    psi_r_rtot = jax.lax.with_sharding_constraint(psi_r_rtot_padded.reshape(meta.nk_tot, -1, meta.nspinor, meta.n_rtot), y_sharding)
    
    print("Arrays properly sharded for no-communication distributed computation")
    
    # 3. Convert weights and other arrays to JAX
    kgrid = jnp.asarray((meta.nkx, meta.nky, meta.nkz), dtype=jnp.int32)
    
    # Get band ranges and weights in JAX
    enk_l, weights_l = get_enk_bandrange(
        wfn, sym, bandrange_l, (bandrange_r[0], bandrange_l[1]), xp
    )
    enk_r, weights_r = get_enk_bandrange(
        wfn, sym, bandrange_r, (bandrange_r[0], bandrange_l[1]), xp
    )
    # Convert weights to numpy before JAX conversion and pad to match distributed arrays
    weights_l_cpu = weights_l.get() if hasattr(weights_l, 'get') else weights_l
    weights_r_cpu = weights_r.get() if hasattr(weights_r, 'get') else weights_r
    
    # Pad weights to match the padded band dimensions
    weights_l_padded = np.pad(weights_l_cpu, [(0, 0), (0, (nb_l_padded * meta.nspinor) - weights_l_cpu.shape[1])], mode='constant')
    weights_r_padded = np.pad(weights_r_cpu, [(0, 0), (0, (nb_r_padded * meta.nspinor) - weights_r_cpu.shape[1])], mode='constant')
    
    weights_l = jnp.asarray(weights_l_padded)
    weights_r = jnp.asarray(weights_r_padded)
    
    ##########################################
    # Precompute ALL q-point and k-point mappings outside JIT
    ##########################################
    print("Precomputing all q-point and k-point mappings...")
    
    # Collect all q-vectors and their mappings
    all_qvecs_nonneg = []
    all_qvecs_wrapped = []
    all_k_l_indices = []
    all_k_r_indices = []
    all_V_qfullG = []
    all_vcoul_comps = []
    all_iq_indices = []
    
    for qvec_nonneg in xp.ndindex(meta.nkx, meta.nky, meta.nkz):  # Match original: *wfn.kgrid
        # Handle umklapp for qvec (exact match to original)
        qvec = jnp.asarray(qvec_nonneg)
        qvec = jnp.where(qvec > kgrid // 2, qvec - kgrid, qvec)
        
        # Precompute k_l indices for this q
        k_l_indices_q = []
        k_r_indices_q = []
        
        for k_r in range(sym.nk_tot):
            k_l_full = jnp.asarray(sym.kvecs_asints[k_r]) - qvec
            k_l_wrapped = k_l_full % kgrid
            k_l_wrapped_cpu = np.array(k_l_wrapped)
            k_l = np.where(np.all(sym.kvecs_asints == k_l_wrapped_cpu, axis=1))[0][0]
            
            k_l_indices_q.append(k_l)
            k_r_indices_q.append(k_r)
        
        # Precompute V_qG extraction for this q
        qvec_cpu = np.array(qvec)
        qveccrys = qvec_cpu.astype(np.float64) / np.array(kgrid)
        q_rounded = np.round(qveccrys)
        q_ext = np.where(np.abs(qveccrys - q_rounded) < 1e-8, q_rounded, qveccrys)
        
        # Fix the find_qpoint_index function to handle mixed array types
        iq = find_qpoint_index(q_ext, sym, tol=1e-6)
        iq_cpu = iq.get() if hasattr(iq, "get") else int(iq)

        iqbar = sym.irk_to_k_map[iq_cpu]
        Sq = sym.sym_mats_k[sym.irk_sym_map[iq_cpu]]
        q_ext_cpu = q_ext if not hasattr(q_ext, "get") else q_ext
        G_Sq = np.round(q_ext_cpu - Sq @ wfn.kpoints[iqbar]).astype(np.int32)
        vcoul_psiG_comps = np.einsum("ij,kj->ki", Sq.astype(np.int32), wfn.get_gvec_nk(iqbar)) - G_Sq[np.newaxis, :]
        
        # Get V_qG for this q-point
        V_qfullG_np = np.zeros(int(wfn.ngkmax), dtype=np.complex128)
        V_qG_slice = V_qG[0, 0, iqbar, :vcoul_psiG_comps.shape[0]]
        V_qG_slice_np = V_qG_slice.get() if hasattr(V_qG_slice, 'get') else np.asarray(V_qG_slice)
        V_qfullG_np[:vcoul_psiG_comps.shape[0]] = V_qG_slice_np
        
        # Store all data for this q-point
        all_qvecs_nonneg.append(qvec_nonneg)
        all_qvecs_wrapped.append(np.array(qvec))
        all_k_l_indices.append(k_l_indices_q)
        all_k_r_indices.append(k_r_indices_q)
        all_V_qfullG.append(V_qfullG_np)
        all_vcoul_comps.append(vcoul_psiG_comps)
        all_iq_indices.append(iq_cpu)
    
    # Convert to JAX arrays with proper padding for consistent shapes
    n_q_points = len(all_qvecs_nonneg)
    max_G = max(comps.shape[0] for comps in all_vcoul_comps)
    
    # Pad and convert to JAX arrays
    all_k_l_indices = jnp.array(all_k_l_indices)  # (n_q, n_k)
    all_k_r_indices = jnp.array(all_k_r_indices)  # (n_q, n_k)
    all_qvecs_wrapped = jnp.array(all_qvecs_wrapped)  # (n_q, 3)
    
    # Pad V_qfullG and vcoul_comps to consistent shapes
    V_qfullG_padded = np.zeros((n_q_points, int(wfn.ngkmax)), dtype=np.complex128)
    vcoul_comps_padded = np.zeros((n_q_points, max_G, 3), dtype=np.int32)
    n_G_per_q = np.zeros(n_q_points, dtype=np.int32)
    
    for i, (V_q, comps) in enumerate(zip(all_V_qfullG, all_vcoul_comps)):
        V_qfullG_padded[i] = V_q
        n_G_actual = comps.shape[0] 
        vcoul_comps_padded[i, :n_G_actual] = comps
        n_G_per_q[i] = n_G_actual
    
    all_V_qfullG = jnp.array(V_qfullG_padded)
    all_vcoul_comps = jnp.array(vcoul_comps_padded)
    n_G_per_q = jnp.array(n_G_per_q)
    
    print(f"Precomputed data for {n_q_points} q-points")

    ##########################################
    # Clean idiomatic distributed computation
    ##########################################
    
    # Compute weighted wavefunctions for all k-points
    # @partial(jax.jit)
    # def compute_weighted_psi(psi_rmu, psi_rtot, weights):
    #     # Apply weights: sqrt(weights) to each wavefunction
    #     # weights shape: (nk, nb_padded*nspinor) → need to reshape to (nk, nb_local, nspinor)
    #     nk, nb_local, nspinor = psi_rmu.shape[:3]
    #     weights_reshaped = weights[:, :nb_local*nspinor].reshape(nk, nb_local, nspinor)  # (nk, nb_local, nspinor)
    #     weights_expanded = weights_reshaped[..., None]  # (nk, nb_local, nspinor, 1)
        
    #     weighted_psi_rmu = jnp.sqrt(weights_expanded) * psi_rmu    # (nk, nb_local, nspinor, n_rmu)
    #     weighted_psi_rtot = jnp.sqrt(weights_expanded) * psi_rtot  # (nk, nb_local, nspinor, n_rtot)
        
    #     return weighted_psi_rmu, weighted_psi_rtot
    
    # # Compute weighted arrays for l and r
    # weighted_psi_l_rmu, weighted_psi_l_rtot = compute_weighted_psi(psi_l_rmu, psi_l_rtot, weights_l)
    # weighted_psi_r_rmu, weighted_psi_r_rtot = compute_weighted_psi(psi_r_rmu, psi_r_rtot, weights_r)
    
    # Simplify to match original cohsex_isdf.py implementation
    print("Processing q-points with original ISDF physics...")
    
    @partial(jax.jit) 
    def compute_CCT_ZCT_for_q(k_l_indices, k_r_indices):
        """Exact match to original cohsex_isdf.py physics - direct accumulation"""
        
        def accumulate_k_pair(carry, i):
            CCT_acc, ZCT_acc = carry
            k_l, k_r = k_l_indices[i], k_r_indices[i]
            
            # Extract wavefunctions for this k-point pair
            psi_l_rmu_k = psi_l_rmu[k_l].reshape(-1, meta.n_rmu)      # (nb*nspinor, n_rmu)
            psi_r_rmu_k = psi_r_rmu[k_r].reshape(-1, meta.n_rmu)      # (nb*nspinor, n_rmu)
            psi_l_rtot_k = psi_l_rtot[k_l].reshape(-1, meta.n_rtot)   # (nb*nspinor, n_rtot)
            psi_r_rtot_k = psi_r_rtot[k_r].reshape(-1, meta.n_rtot)   # (nb*nspinor, n_rtot)
            
            # Original: psi_l_rmuT = xp.ascontiguousarray(psi_l_rmu.T)
            psi_l_rmuT_k = jnp.conj(psi_l_rmu_k.T)    # (n_rmu, nb*nspinor)
            psi_r_rmuT_k = jnp.conj(psi_r_rmu_k.T)    # (n_rmu, nb*nspinor)
            
            # Original ISDF physics:
            # Pmu_l = xp.matmul(psi_l_rmuT, psi_l_rmu)
            # Pmu_r = xp.matmul(psi_r_rmuT, psi_r_rmu)  
            # CCT += xp.multiply(Pmu_l, Pmu_r)
            Pmu_l = psi_l_rmuT_k @ psi_l_rmu_k  # (n_rmu, n_rmu)
            Pmu_r = psi_r_rmuT_k @ psi_r_rmu_k  # (n_rmu, n_rmu)
            CCT_acc = CCT_acc + jnp.conj(Pmu_l) * Pmu_r   # Direct accumulation!
            
            # P_l = xp.matmul(psi_l_rmuT, psi_l_rtot)
            # P_r = xp.matmul(psi_r_rmuT, psi_r_rtot)
            # ZCT += xp.multiply(P_l, P_r)
            P_l = psi_l_rmuT_k @ psi_l_rtot_k   # (n_rmu, n_rtot)
            P_r = psi_r_rmuT_k @ psi_r_rtot_k   # (n_rmu, n_rtot)
            ZCT_acc = ZCT_acc + jnp.conj(P_l) * P_r       # Direct accumulation!
            
            return (CCT_acc, ZCT_acc), None
        
        # Initialize accumulators
        CCT_init = jnp.zeros((meta.n_rmu, meta.n_rmu), dtype=jnp.complex128)
        ZCT_init = jnp.zeros((meta.n_rmu, meta.n_rtot), dtype=jnp.complex128)
        
        # Use lax.scan for memory-efficient accumulation (no huge intermediate arrays!)
        k_indices = jnp.arange(len(k_l_indices))
        (CCT, ZCT), _ = jax.lax.scan(accumulate_k_pair, (CCT_init, ZCT_init), k_indices)
        
        return CCT, ZCT
    
    # Main q-point loop
    for q_idx, (qvec_nonneg, iq_cpu) in enumerate(zip(all_qvecs_nonneg, all_iq_indices)):
        print(f"Processing q-point {iq_cpu}...")
        
        # Get data for this q
        k_l_indices = all_k_l_indices[q_idx]
        k_r_indices = all_k_r_indices[q_idx]
        qvec = all_qvecs_wrapped[q_idx]
        V_qfullG = all_V_qfullG[q_idx]
        vcoul_comps = all_vcoul_comps[q_idx]
        n_G_q = n_G_per_q[q_idx]
        
        # Clean CCT and ZCT computation for this q-point
        CCT, ZCT = compute_CCT_ZCT_for_q(k_l_indices, k_r_indices)
        # lstsq solve with optimal sharding (Y over longer rtot dimension)
        CCT_cholesky = jax.scipy.linalg.cho_factor(CCT)
        zeta_q = jax.scipy.linalg.cho_solve(CCT_cholesky, ZCT, overwrite_b=True)
        #zeta_q = jnp.linalg.lstsq(CCT, ZCT, rcond=-1)[0]  # (n_rmu, n_rtot)
        
        # Reshard to be sharded over ALL processors in rmu dimension (1D mesh)
        # zeta_q shape: (n_rmu, n_rtot) - shard over n_rmu dimension only
        all_procs = jax.devices()
        mesh_1d = jax.sharding.Mesh(all_procs, ['all_procs'])
        rmu_1d_sharding = NamedSharding(mesh_1d, P('all_procs', None))  # Shard n_rmu over all procs
        zeta_q = jax.lax.with_sharding_constraint(zeta_q, rmu_1d_sharding)
        
        # Reshape zeta_q: (n_rmu, n_rtot) → (n_rmu, nx, ny, nz) 
        zeta_q_spatial = zeta_q.reshape(meta.n_rmu, *meta.fft_grid)
        
        # Phase removal and FFT  
        fx = jnp.arange(meta.fft_grid[0])[None, :, None, None] / meta.fft_grid[0]
        fy = jnp.arange(meta.fft_grid[1])[None, None, :, None] / meta.fft_grid[1]
        fz = jnp.arange(meta.fft_grid[2])[None, None, None, :] / meta.fft_grid[2]
        
        phase = jnp.exp(-2j * jnp.pi * (qvec[0] * fx + qvec[1] * fy + qvec[2] * fz))
        zeta_q_spatial = zeta_q_spatial * phase
        zeta_qG = jnp.fft.fftn(zeta_q_spatial, axes=(-3, -2, -1))
        
        # Extract G-components and compute V_qmunu
        zeta_qG_extracted = zeta_qG[:, vcoul_comps[:, 0], vcoul_comps[:, 1], vcoul_comps[:, 2]]
        G_mask = jnp.arange(vcoul_comps.shape[0]) < n_G_q
        zeta_qG_masked = jnp.where(G_mask[None, :], zeta_qG_extracted, 0.0)
        V_qfullG_masked = jnp.where(G_mask, V_qfullG, 0.0)
        
        V_weighted = V_qfullG_masked[None, :] * zeta_qG_masked
        V_qmunu_q = jnp.conj(zeta_qG_masked) @ V_weighted.T
        
        # Store result
        V_qmunu.data[0, 0, 0, *qvec_nonneg, :, :] = xp.asarray(np.array(V_qmunu_q))
        print(f"qpoint {iq_cpu} done")

    # Convert JAX arrays back to output format, trimming to original band counts
    psi_l_rtot_out.data = xp.asarray(np.array(psi_l_rtot_padded[:, :nb_l_original, :, :, :, :]))
    psi_r_rtot_out.data = xp.asarray(np.array(psi_r_rtot_padded[:, :nb_r_original, :, :, :, :]))
    
    # Extract centroid values efficiently using JAX operations, trimming to original sizes
    psi_l_rmu_out.data = xp.asarray(np.array(
        extract_centroids(psi_l_rtot_padded[:, :nb_l_original].reshape(meta.nk_tot * nb_l_original * meta.nspinor, *meta.fft_grid), 
                         centroids).reshape(meta.nk_tot, nb_l_original, meta.nspinor, meta.n_rmu)
    ))
    psi_r_rmu_out.data = xp.asarray(np.array(
        extract_centroids(psi_r_rtot_padded[:, :nb_r_original].reshape(meta.nk_tot * nb_r_original * meta.nspinor, *meta.fft_grid), 
                         centroids).reshape(meta.nk_tot, nb_r_original, meta.nspinor, meta.n_rmu)
    ))

    #xp.conj(psi_l_rmu_out.data, out=psi_l_rmu_out.data)

    wfn_l = WfnArray(psi_l_rmu_out, enk_l)
    wfn_r = WfnArray(psi_r_rmu_out, enk_r)

    # V_qmunu.data *= sym.nk_tot
    # V_qmunu.data *= -1.0
    V_q = V_qmunu.transpose(
        "nfreq", "nkx", "nky", "nkz", "npol1", "nrmu1", "npol2", "nrmu2"
    )

    return V_q, wfn_l, wfn_r


# G_(kab)(mu,nu,t=0) = \sum_mn psi^*_mk(r_mu) * psi_nk(r_nu) (n restricted to range of psi_rmu)
# k goes over kfull
def get_G_mu_nu(wfn, psi_l, psi_r, meta: Meta, xp, Gkij=None, return_R=False):
    # using xp to wrap cupy/numpy, calculate:
    # take the matrix psi with shape (nkpts, nbands, nspinor, nrmu) and do:
    # G_{k,a,b}(mu,nu) = \sum_mnab psi^*_mka(r_mu) * psi_nkb(r_nu) (matmul)
    kgrid = xp.asarray((meta.nkx, meta.nky, meta.nkz))

    if Gkij is None:
        # Initialize Gkij with all variables
        Gkij = LabeledArray(
            shape=(
                1,
                meta.nkx,
                meta.nky,
                meta.nkz,
                psi_l.psi.shape("nb"),
                psi_r.psi.shape("nb"),
            ),
            axes=["nfreq", "nkx", "nky", "nkz", "nb1", "nb2"],
        )
        Gkij.join("nkx", "nky", "nkz")
        for ik in range(sym.nk_tot):
            xp.fill_diagonal(Gkij.data[0, ik], 1.0)

    # nspinor*nrmu
    n_spinor = psi_l.psi.shape("nspinor")
    nrmu = psi_l.psi.shape("nrmu")
    # dims: nfreq(=0), nk, n_rmu, n_rmu
    Gk_mu_nu_0 = LabeledArray(
        shape=(1, meta.nkx, meta.nky, meta.nkz, n_spinor, nrmu, n_spinor, nrmu),
        axes=["nfreq", "nkx", "nky", "nkz", "nspinor1", "nrmu1", "nspinor2", "nrmu2"],
    )
    # Gk_mu_nu_0.join('nkx', 'nky', 'nkz')
    Gk_mu_nu_0.join("nspinor1", "nrmu1")
    Gk_mu_nu_0.join("nspinor2", "nrmu2")

    psi_l_tmp = xp.zeros(
        (psi_l.psi.shape("nspinor") * psi_l.psi.shape("nrmu"), psi_l.psi.shape("nb")),
        dtype=xp.complex128,
    )
    psi_r_tmp = xp.zeros(
        (psi_r.psi.shape("nb"), psi_r.psi.shape("nspinor") * psi_r.psi.shape("nrmu")),
        dtype=xp.complex128,
    )
    psi_l.psi.join("nspinor", "nrmu")
    if psi_l is not psi_r:
        psi_r.psi.join("nspinor", "nrmu")

    for kpt in xp.ndindex(meta.nkx, meta.nky, meta.nkz):
        k_idx = kpt[0] * meta.nky * meta.nkz + kpt[1] * meta.nkz + kpt[2]

        psi_l_tmp = psi_l.psi.slice("nk", k_idx).T
        psi_r_tmp = xp.conj(psi_r.psi.slice("nk", k_idx))
        Gk_mu_nu_0.data[0, *kpt] = xp.matmul(
            xp.matmul(psi_l_tmp, Gkij.slice_many({"nfreq": 0, "nkx*nky*nkz": k_idx})),
            psi_r_tmp,
        )

    Gk_mu_nu_0.unjoin("nspinor1", "nrmu1")
    Gk_mu_nu_0.unjoin("nspinor2", "nrmu2")

    if not return_R:
        return Gk_mu_nu_0
    else:
        return get_G_R(Gk_mu_nu_0)  # kgrid last


def get_G_R(Gk):
    # Reorder axes to have kgrid last (batch fft mem locality)
    Gk = Gk.kgrid_to_last()
    Gk.join(
        "nfreq", "nspinor1", "nrmu1", "nspinor2", "nrmu2"
    )  # shape (nfreq*nspin*nrmu*nspin*nrmu, *kgrid)
    # V is umklapped to have kpts in FFT ordering [0,...nk/2,-nk/2+1,...].
    # G doesn't need to be because G_(k+G)(r,r') = G_k(r,r') (bloch fn).
    Gk.ifft_kgrid()
    Gk.unjoin("nfreq", "nspinor1", "nrmu1", "nspinor2", "nrmu2")

    return Gk


# get the real-space sigma_\alpha\beta(r,r'(omega))
# options being X, SX, COH
def get_sigma_x_mu_nu(G_R, V_q, xp):
    # sigma_kbar,ab = \sum_(set of k_i = kbar S_i) \sum_qbar G_(k-qbar,ab)(mu,nu) V_qbar(mu,nu)
    # trying in real space! \sum_R G_R W_R. woohoo

    # bispinor case:
    # Sigma_ab = \sum_IJ gamma^I_ac gamma^J_bd G_R,cd V_R,IJ

    V_q = V_q.kgrid_to_last()
    V_q.join("nfreq", "npol1", "nrmu1", "npol2", "nrmu2")
    V_q.ifft_kgrid()
    xp.multiply(
        V_q.data, xp.sqrt(sym.nk_tot), out=V_q.data
    )  # this makes V_R equal to mtxels of 1/|r-r'|
    V_q.unjoin("nfreq", "npol1", "nrmu1", "npol2", "nrmu2")

    print("G_R and V_R obtained")

    sigma_R = LabeledArray(
        shape=G_R.data.shape,
        axes=["nfreq", "nspinor1", "nrmu1", "nspinor2", "nrmu2", "nkx", "nky", "nkz"],
    )
    if not bispinor:
        V_q.join("npol1", "nrmu1")
        V_q.join("npol2", "nrmu2")
        sigma_R.data += xp.multiply(
            G_R.data, V_q.data[:, xp.newaxis, :, xp.newaxis, :, :, :, :]
        )  # should rename to V_R
    # if bispinor:
    # for gamma_mu,nu in 0,3:
    # G_Rdata_tmp should be xp.einsum('ac,fcmdnxyz,bd',gamma_mu,G_R.data,gamma_nu) (note mn=spatial, cd=spinor)
    # to sigma_R should be added xp.multiply(G_Rdata_tmp,V_q.data[mu,nu])
    # TODO: check this really well
    if bispinor:
        scratch = xp.empty_like(G_R.data[:, 0, :, 0, :, :, :, :])

        for I, (rI, cI, vI) in enumerate(gammas_sparse):
            for J, (rJ, cJ, vJ) in enumerate(gammas_sparse):
                # target[...] = 0.0
                for p in range(len(vI)):
                    a = int(rI[p])
                    c = int(cI[p])
                    gI = vI[p]
                    for q in range(len(vJ)):
                        b = int(rJ[q])
                        d = int(cJ[q])
                        gJ = vJ[q]
                        target = sigma_R.data[
                            :, a, :, b, :, :, :, :
                        ]  # slice_many({'gamma1': I, 'gamma2': J})
                        xp.multiply(
                            G_R.slice_many({"nspinor1": c, "nspinor2": d}),
                            V_q.slice_many({"npol1": I, "npol2": J}),
                            out=scratch,
                        )
                        xp.add(target, gI * gJ * scratch, out=target)

    sigma_R.join("nfreq", "nspinor1", "nrmu1", "nspinor2", "nrmu2")
    sigma_R.fft_kgrid()
    sigma_R.unjoin("nfreq", "nspinor1", "nrmu1", "nspinor2", "nrmu2")
    sigma_R = sigma_R.transpose(
        "nfreq", "nkx", "nky", "nkz", "nspinor1", "nrmu1", "nspinor2", "nrmu2"
    )
    sigma_R.join("nkx", "nky", "nkz")

    sigma_R.data *= -1.0 / sym.nk_tot  # physical factor for sum over all kpts.
    # sigma_R.data *= -1.0
    return sigma_R


def get_sigma_x_kij(psi_l, psi_r, sigma_kbar, meta: Meta, xp):
    r"""
    Calculate the sigma_x_kij matrix elements.
    sigma_mnkbar = \sum_rmu,rnu,s,s' exp(ik(r_nu-r_mu)) u_mk^*(r_mu,s) sigma_kbar,ss'(r_mu,r_nu) u_nk(r_nu,s')
    """
    sigma_kij = xp.zeros(
        (sigma_kbar.shape("nkx*nky*nkz"), psi_l.psi.shape("nb"), psi_r.psi.shape("nb")),
        dtype=xp.complex128,
    )  # TODO: should be a labelled array
    sigma_ktmp = xp.zeros(
        (
            sigma_kbar.shape("nspinor1") * sigma_kbar.shape("nrmu1"),
            sigma_kbar.shape("nspinor2") * sigma_kbar.shape("nrmu2"),
        ),
        dtype=xp.complex128,
    )
    psi_l_tmp = xp.zeros(
        (psi_l.psi.shape("nb"), psi_l.psi.shape("nspinor") * psi_l.psi.shape("nrmu")),
        dtype=xp.complex128,
    )
    psi_r_tmp = xp.zeros(
        (psi_r.psi.shape("nspinor") * psi_r.psi.shape("nrmu"), psi_r.psi.shape("nb")),
        dtype=xp.complex128,
    )

    sigma_kbar.join("nspinor1", "nrmu1")
    sigma_kbar.join("nspinor2", "nrmu2")

    psi_l.psi.join("nspinor", "nrmu")
    if psi_l is not psi_r:
        psi_r.psi.join("nspinor", "nrmu")

    for kpt in xp.ndindex(meta.nkx, meta.nky, meta.nkz):
        k_idx = kpt[0] * meta.nky * meta.nkz + kpt[1] * meta.nkz + kpt[2]

        sigma_ktmp = sigma_kbar.slice_many({"nfreq": 0, "nkx*nky*nkz": k_idx})
        psi_l_tmp = xp.conj(psi_l.psi.slice("nk", k_idx))
        psi_r_tmp = psi_r.psi.slice("nk", k_idx).T
        sigma_kij[k_idx, :, :] = xp.matmul(xp.matmul(psi_l_tmp, sigma_ktmp), psi_r_tmp)

    return sigma_kij


def write_sigma_to_file(sigma_kij, filename="eqp0.dat"):
    print(f"sigma_kij dtype before writing: {sigma_kij.dtype}")
    nk, nbands, _ = sigma_kij.shape

    with open(filename, "w") as f:
        for k in range(nk):
            f.write(f"\nk-point {k}:\n")
            f.write("-" * 40 + "\n")
            for n in range(nbands):
                real = float(sigma_kij[k, n, n].real)  # Explicit conversion to float
                imag = float(sigma_kij[k, n, n].imag)
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
    # Ensure both arrays are in the same backend (xp)
    if hasattr(q_frac, 'get'):  # JAX array
        q_frac_np = q_frac.get()
    elif hasattr(q_frac, '__array__'):  # NumPy array
        q_frac_np = np.asarray(q_frac)
    else:  # CuPy array or other
        q_frac_np = q_frac.get() if hasattr(q_frac, 'get') else np.asarray(q_frac)
    
    # Convert both to the current backend (xp)
    unfolded_kpts_xp = xp.asarray(sym.unfolded_kpts)
    q_frac_xp = xp.asarray(q_frac_np)
    
    diffs = xp.abs(unfolded_kpts_xp - q_frac_xp[None, :])
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
    with h5py.File(filename, "w") as f:
        # Access the underlying numerical data arrays
        V_qmunu_data = (
            V_qmunu.data.get() if hasattr(V_qmunu.data, "get") else V_qmunu.data
        )

        # Handle WfnArray psi and enk data
        psi_l_data = (
            psi_l.psi.data.get() if hasattr(psi_l.psi.data, "get") else psi_l.psi.data
        )
        psi_r_data = (
            psi_r.psi.data.get() if hasattr(psi_r.psi.data, "get") else psi_r.psi.data
        )
        enk_l_data = (
            psi_l.enk.data.get() if hasattr(psi_l.enk.data, "get") else psi_l.enk.data
        )
        enk_r_data = (
            psi_r.enk.data.get() if hasattr(psi_r.enk.data, "get") else psi_r.enk.data
        )

        # Write data arrays
        f.create_dataset("V_qmunu_data", data=V_qmunu_data)
        f.create_dataset("psi_l_data", data=psi_l_data)
        f.create_dataset("psi_r_data", data=psi_r_data)
        f.create_dataset("enk_l_data", data=enk_l_data)
        f.create_dataset("enk_r_data", data=enk_r_data)

    print("wrote taggedarrays.h5")


def read_labeled_arrays_from_h5(filename):
    """
    Read the data arrays from an HDF5 file and reconstruct LabeledArrays and WfnArrays.

    Args:
        filename (str): The name of the HDF5 file to read from.

    Returns:
        tuple: A tuple containing (V_qmunu, psi_l, psi_r) where V_qmunu is a LabeledArray
              and psi_l/psi_r are WfnArrays.
    """
    with h5py.File(filename, "r") as f:
        # Read data arrays
        V_qmunu_data = f["V_qmunu_data"][:]
        psi_l_data = f["psi_l_data"][:]
        psi_r_data = f["psi_r_data"][:]
        enk_l_data = f["enk_l_data"][:]
        enk_r_data = f["enk_r_data"][:]

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
            axes=["nfreq", "nkx", "nky", "nkz", "npol1", "nrmu1", "npol2", "nrmu2"],
        )

        # Create LabeledArrays for psi and enk
        psi_l = LabeledArray(data=psi_l_data, axes=["nk", "nb", "nspinor", "nrmu"])

        psi_r = LabeledArray(data=psi_r_data, axes=["nk", "nb", "nspinor", "nrmu"])

        enk_l = LabeledArray(data=enk_l_data, axes=["nk", "nb"])

        enk_r = LabeledArray(data=enk_r_data, axes=["nk", "nb"])

        # Create WfnArrays
        psi_l_wfn = WfnArray(psi_l, enk_l)
        psi_r_wfn = WfnArray(psi_r, enk_r)

        return V_qmunu, psi_l_wfn, psi_r_wfn


def main(argv=None):
    global sym
    argp = argparse.ArgumentParser(description="COHSEX self-energy driver")
    argp.add_argument(
        "-i",
        "--input",
        default="cohsex_test.in",
        help="Input file",
    )
    args = argp.parse_args(argv)

    params = read_cohsex_input(args.input)

    # Check GPU availability
    try:
        cp.cuda.runtime.getDeviceCount()
        print(f"Using GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
        mem_info = cp.cuda.runtime.memGetInfo()
        print(
            f"Memory Usage: {(mem_info[1] - mem_info[0])/1024**2:.1f}MB / {mem_info[1]/1024**2:.1f}MB"
        )
    except Exception:
        print("Using CPU (NumPy)")

    nval = params["nval"]
    ncond = params["ncond"]
    nband = params["nband"]

    sys_dim = params["sys_dim"]  # 3 for 3D, 2 for 2D

    ryd2ev = 13.6056980659

    global wfn
    wfn = WFNReader(params["wfn_file"])
    # wfnq = WFNReader("WFNq.h5")
    # eps0 = EPSReader("eps0mat.h5")
    # eps = EPSReader("epsmat.h5")
    sym = symmetry_maps.SymMaps(wfn)
    # q0 = wfnq.kpoints[0] - wfn.kpoints[0]
    # if np.linalg.norm(q0) > 1e-6:
    #    print(f'Using q0 = ({q0[0]:.5f}, {q0[1]:.5f}, {q0[2]:.5f})')

    nvrange, ncrange, nsigmarange, n_fullrange, n_valrange = get_bandranges(
        nval, ncond, nband, wfn.nelec
    )
    nvplussigrange = (min(n_valrange), max(nsigmarange))
    ncplussigrange = (min(nsigmarange), max(n_fullrange))

    # Load centroids
    centroids_frac = np.loadtxt(params["centroids_file"])
    n_rmu = int(centroids_frac.shape[0])
    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    taggedarray_filename = os.path.join(tmp_dir, f"taggedarrays{n_rmu}.h5")

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
    print("unique centroid indices:")
    print(np.unique(centroid_indices, axis=0).shape)

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
    print("\n")

    # restart: if True, read interp. vectors and V_qmunu from file
    restart = params["restart"]
    x_only = params["x_only"]
    do_screened = params["do_screened"]
    global bispinor
    bispinor = params["bispinor"]

    meta = Meta.from_system(wfn, sym, nval, ncond, nband, n_rmu, bispinor)
    meta.rank = jax.process_index()
    meta.n_proc = jax.process_count()
    print(jax.devices())
    print(jax.process_indices())
    if jax.Device.id == 1:
        print('i am proc 1')

    if x_only and do_screened:
        raise ValueError("x_only and do_screened cannot both be True")

    if not restart:
        ####################################
        # 1.) get (truncated in 2D) coulomb potential v_q(G) and W_q=0(G=G'=0) element
        ####################################
        # V_qG, wcoul0 = get_V_qG(wfn, sym, q0, xp, eps0.epshead, sys_dim)
        V_qG, wcoul0 = get_V_qG(
            wfn, sym, (0.001, 0.0, 0.0), xp, 0.2, sys_dim, meta, do_Dmunu=bispinor
        )

        ####################################
        # 2.) get interpolative separable density fitting basis functions zeta_q,mu(r) and <mu|V_q|nu>
        ####################################
        if x_only:
            V_qmunu, psi_l_rmu_out, psi_r_rmu_out = get_zeta_q_and_v_q_mu_nu(
                wfn,
                sym,
                centroid_indices,
                n_valrange,
                nsigmarange,
                V_qG,
                meta,
                xp,
                bispinor=bispinor,
            )
            write_labeled_arrays_to_h5(
                taggedarray_filename, V_qmunu, psi_l_rmu_out, psi_r_rmu_out
            )
        else:
            V_qmunu, psi_l_rmu_out, psi_r_rmu_out = get_zeta_q_and_v_q_mu_nu(
                wfn,
                sym,
                centroid_indices,
                nvplussigrange,
                ncplussigrange,
                V_qG,
                meta,
                xp,
                bispinor=bispinor,
            )
            write_labeled_arrays_to_h5(
                taggedarray_filename, V_qmunu, psi_l_rmu_out, psi_r_rmu_out
            )
    elif restart and not x_only:
        V_qmunu, psi_l_rmu_out, psi_r_rmu_out = read_labeled_arrays_from_h5(
            taggedarray_filename
        )

    if not x_only:
        chi0 = get_chi0(psi_l_rmu_out, psi_r_rmu_out, window_pairs, meta, wfn, xp)
        # hyperparameters: (1-vX)^-1 = sum_n=0,n_mult (vX)^n, block_f is how many freqs are batched for inversion
        # update: currently inverting directly; i suspect it's ill posed in the low-dim case
        V_for_w = V_qmunu
        W_q = get_static_w_q(
            chi0, V_for_w, meta, wfn, sym, xp, n_mult=10, block_f=1, bispinor=bispinor
        )

    psi_l_rmu_out.psi = psi_l_rmu_out.psi.slice("nb", xp.s_[: wfn.nelec], tagged=True)
    psi_r_rmu_out.psi = psi_r_rmu_out.psi.slice(
        "nb", xp.s_[: nval + ncond], tagged=True
    )

    ####################################
    # 4.) get G_k(r_mu,r_nu) for valence bands
    ####################################
    G_R_val_mu_nu = get_G_mu_nu(
        wfn, psi_l_rmu_out, psi_l_rmu_out, meta, xp, return_R=True
    )

    ####################################
    # 5.) get sigma_mnk from V_q,mu,nu and G_k(r_mu,r_nu)
    ####################################
    if do_screened:
        sigma_in = W_q
    else:
        if bispinor:
            V_qmunu = V_qmunu.transpose(
                "nfreq", "nkx", "nky", "nkz", "npol1", "nrmu1", "npol2", "nrmu2"
            )

        sigma_in = V_qmunu
    # if bispinor and sigma_in is V_qmunu:
    #    sigma_in = V_qmunu.slice_many({'npol1': 0, 'npol2': 0}, tagged=True)
    sigma_x_kbar_munu = get_sigma_x_mu_nu(G_R_val_mu_nu, sigma_in, xp)
    sigma_x_kbar_ij = get_sigma_x_kij(
        psi_r_rmu_out, psi_r_rmu_out, sigma_x_kbar_munu, meta, xp
    )

    write_sigma_to_file(ryd2ev * sigma_x_kbar_ij, params["output_file"])

    # Later stages of this project will iterate this workflow so that the COHSEX
    # potential feeds back into updated wavefunctions (self-consistent COHSEX)
    # and eventually into a full quasiparticle self-consistent GW cycle.
    return 0


if __name__ == "__main__":
    jax.distributed.initialize()
    raise SystemExit(main())
