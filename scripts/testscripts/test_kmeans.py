import os

import numpy as np
import h5py as h5
from isdf.common.gpu_utils import cp
from isdf.common.wfnreader import WFNReader
from isdf.isdf_init.kmeans_isdf import weighted_kmeans_jax
from misc.test_scripts.kmeans_old import weighted_kmeans_cupy


def test_kmeans_agreement(tmp_path):
    cp.random.seed(0)
    with h5.File('examples/cohsex_test/charge_density.h5', 'r') as f:
        rho = f['charge_density'][...]
    wfn = WFNReader('examples/cohsex_test/WFNsmall.h5')
    avec = wfn.avec

    rho_cp = cp.asarray(rho, dtype=cp.float32)
    avec_cp = cp.asarray(avec, dtype=cp.float32)
    rho_jax = np.asarray(rho, dtype=np.float32)
    avec_jax = np.asarray(avec, dtype=np.float32)

    _, cent_ref, _, _ = weighted_kmeans_cupy(avec_cp, rho_cp, N_k=5)
    _, cent_jax, _, _ = weighted_kmeans_jax(avec_jax, rho_jax, N_k=5, seed=0)

    import pathlib
    tmp_path = pathlib.Path(tmp_path)
    ref_file = tmp_path/'centroids_ref.txt'
    jax_file = tmp_path/'centroids_jax.txt'
    np.savetxt(ref_file, cp.asnumpy(cent_ref) if hasattr(cp, 'asnumpy') else np.asarray(cent_ref))
    np.savetxt(jax_file, np.asarray(cent_jax))

    ref = np.loadtxt(ref_file)
    jax = np.loadtxt(jax_file)

    # Allow centroid permutations when comparing
    from scipy.optimize import linear_sum_assignment
    cost = np.linalg.norm(ref[:, None, :] - jax[None, :, :], axis=2)
    row, col = linear_sum_assignment(cost)
    jax_sorted = jax[col]
    assert np.allclose(ref, jax_sorted, atol=0.07)

test_kmeans_agreement('./')