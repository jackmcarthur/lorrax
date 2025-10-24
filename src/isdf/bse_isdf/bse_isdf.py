"""JAX based ISDF-BSE matrix vector product.

This module implements a prototype Bethe--Salpeter matrix--vector multiply using
interpolative separable density fitting (ISDF).  The Hamiltonian action is

.. math:: H = D + V - W

where ``D`` is the diagonal term from single particle energies and ``V`` and
``W`` are the direct and screened exchange interactions.  For this simplified
demo both ``V`` and ``W`` are taken from the same ``V_\mu\nu`` array stored in
``taggedarrays.h5``.  The heavy real-space dimension (``nrmu``) is sharded
across devices so the code runs on multiple CPU devices via JAX.

The ``haydock_eig`` routine below is unchanged and provides a Lanczos solver
using a user supplied ``apply_matrix_to_vector`` callback.
"""
import os
# Force 8 host devices for XLA
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import numpy as np
from ..gw_isdf.cohsex_isdf import read_labeled_arrays_from_h5

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


# Build a 1D device mesh used for sharding along the real-space (mu) axis
device_mesh = Mesh(np.array(jax.devices()[:4]), ('mu',))


# [WX](cvk) = \sum_{c'v'k'} W(cvk,c'v'k') X(c'v'k')
# [WX](cvk) = 1/Nk 
#             \sum_nu u_vk(rnu) [
#                 \sum_mu u*_ck(rmu) [
#                     \sum_k' W_munu(k-k') [
#                          \sum_c' u_ck'(rmu) [
#                              \sum_v' u*_v'k'(rnu)X(c'v'k')
#                          ]
#                      ]
#                  ]
#              ]
# the actual matrix to vector function will be the application of the BSE hamiltonian 
# (in the TDA, H_BSE(cvk,c'v'k') = D + 2V - W) to a test eigenvector A_c'v'k' with the ISDF density fitting method.
# [DX](cvk) = (E_ck-E_vk')X_cvk

# [VX](cvk) = \sum_{c'v'k'} V(cvk,c'v'k') X(c'v'k')
# [VX](cvk) = 1/Nk 
#             \sum_mu u*_ck(rmu)u_vk(rmu) [
#                 \sum_nu V_munu(q=0) [
#                     \sum_k'[
#                          \sum_c' u_ck'(rnu) [
#                              \sum_v' u*_v'k'(rnu)X(c'v'k')
#                          ]
#                      ]
#                  ]
#              ]
# shapes become: shape N_rnu * Nc * Nk', then N_rnu * Nk

def load_isdf_data(filename: str = "taggedarrays.h5"):
    """Load Vmunu and wavefunctions from an HDF5 file into JAX arrays."""
    V_qmunu, psi_l_wfn, psi_r_wfn = read_labeled_arrays_from_h5(filename)

    def _to_jax(arr):
        data = arr.data if hasattr(arr, "data") else arr
        if hasattr(data, "get"):
            data = data.get()
        return jnp.asarray(data)

    V = _to_jax(V_qmunu)[0, 0, 0, 0, 0, :, 0, :]
    psi_l = _to_jax(psi_l_wfn.psi)
    psi_r = _to_jax(psi_r_wfn.psi)
    eps_v = _to_jax(psi_l_wfn.enk)
    eps_c = _to_jax(psi_r_wfn.enk)

    mu_sharding = NamedSharding(device_mesh, PartitionSpec('mu'))
    psi_sharding = NamedSharding(device_mesh, PartitionSpec(None, None, None, 'mu'))

    V = jax.device_put(V, NamedSharding(device_mesh, PartitionSpec('mu', None)))
    psi_l = jax.device_put(psi_l, psi_sharding)
    psi_r = jax.device_put(psi_r, psi_sharding)

    return V, psi_l, psi_r, eps_v, eps_c


@jax.jit
def apply_matrix_to_vector(X, V, psi_v, psi_c, eps_v, eps_c):
    """Apply simplified ISDF-BSE Hamiltonian to X."""
    # Diagonal term
    D = (eps_c[:, None, :] - eps_v[:, :, None]) * X

    # rho(k,v,c,mu)
    rho = jnp.einsum('kvsm,kcsm->kvcm', jnp.conj(psi_v), psi_c)

    # Projection of X onto mu basis
    zeta_mu = jnp.einsum('kvcm,kvc->km', rho, X)
    zeta_mu = jnp.einsum('mn,kn->km', V, zeta_mu)

    # Direct term
    V_term = jnp.einsum('kvcm,km->kvc', rho, zeta_mu) / psi_v.shape[0]

    # Screened term (very simplified FFT-convolution)
    A = jnp.einsum('kvcm,km->kvcm', rho, zeta_mu)
    A_k = jnp.fft.fftn(A, axes=(0,))
    B_k = A_k * V[None, None, None, :]
    W_term = jnp.real(jnp.fft.ifftn(B_k, axes=(0,))).sum(-1) / psi_v.shape[0]

    return D + V_term - W_term


def haydock_eig(matvec, n, n_eig, max_iter=40):
    """Lanczos/Haydock solver using a matrix-vector callback."""
    # initial q
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    q = jax.random.normal(k1, (n,), dtype=jnp.complex128)
    q = q + 1j * jax.random.normal(k2, (n,), dtype=jnp.complex128)
    q = q / jnp.linalg.norm(q)

    # preallocate
    Q    = jnp.zeros((n, max_iter+1), dtype=q.dtype).at[:,0].set(q)
    alpha = jnp.zeros((max_iter,), dtype=q.real.dtype)
    beta  = jnp.zeros((max_iter,), dtype=q.real.dtype)
    col_idx = jnp.arange(max_iter+1)

    actual_iter = max_iter
    for j in range(max_iter):
        # iteration logging removed for JIT compatibility
        z = matvec(q)
        alpha = alpha.at[j].set(jnp.vdot(q,z).real)
        if j>0:
            z = z - beta[j-1] * Q[:, j-1]
        z = z - alpha[j]*q

        if j>0:
            # fully vectorized re-orth
            proj_full   = Q.conj().T @ z           # (max_iter+1,)
            proj_masked = proj_full * (col_idx <= j)  # zero out all > j
            z = z - Q @ proj_masked

        beta = beta.at[j].set(jnp.linalg.norm(z))
        q = z / beta[j]
        Q = Q.at[:, j+1].set(q)

    # build & diagonalize T
    T = jnp.diag(alpha[:actual_iter])
    if actual_iter > 1:
        off = beta[:actual_iter-1]
        T = T + jnp.diag(off,1) + jnp.diag(off,-1)
    evals_T, vecs_T = jnp.linalg.eigh(T)
    idx = jnp.argsort(evals_T)
    evals = evals_T[idx][:n_eig]
    vecs  = Q[:, :actual_iter] @ vecs_T[:, idx[:n_eig]]
    return evals, vecs

# JIT compile haydock_eig (functional) for GPU acceleration
haydock_eig = jax.jit(haydock_eig, static_argnums=(1,2,3))

def main():
    """Demonstrate the ISDF-BSE matrix-vector multiply."""
    V, psi_l, psi_r, eps_v, eps_c = load_isdf_data()

    nk, nv, ns, nrmu = psi_l.shape
    nc = psi_r.shape[1]

    def mv(v):
        vec = v.reshape(nk, nv, nc)
        return apply_matrix_to_vector(vec, V, psi_l, psi_r, eps_v, eps_c).reshape(-1)

    n = nk * nv * nc
    evals, _ = haydock_eig(mv, n, n_eig=2, max_iter=10)
    print("Example eigenvalues:", evals)


if __name__ == "__main__":
    print("starting JAX ISDF+BSE calculation.")
    main()
