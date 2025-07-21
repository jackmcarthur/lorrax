"""Haydock/Lanczos diagonalization demo for the Bethe--Salpeter equation.

The BSE Hamiltonian is Hermitian but far too large to form explicitly in
real calculations.  A practical solver therefore only needs a routine
that applies the matrix to a trial vector.  The Haydock recursion
produces a Krylov basis from repeated applications of this routine.  In
that basis the Hamiltonian reduces to a tridiagonal matrix whose
eigenpairs approximate those of the full operator.  By mapping the
tridiagonal eigenvectors back to the Krylov basis we obtain
approximations to the desired eigenvectors.

This file illustrates the procedure on a small dense matrix.  The code
is written with ``gpu_utils`` so that all operations run on either NumPy
or CuPy.  The only unavoidable Python loop is over the number of Krylov
iterations.  Expensive steps such as reorthogonalization are expressed as
matrix--vector products so that they execute efficiently on the GPU.
In a full implementation ``apply_matrix_to_vector`` would call a
specialised low-rank kernel.

JAX-based Haydock/Lanczos diagonalization demo for the Bethe--Salpeter equation
with explicit matrix-vector products sharded across 8 devices via pmap.
"""
import os
# Force 8 host devices for XLA
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.shard_map import shard_map

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)

# Set up a 1D device mesh for row sharding
device_mesh = Mesh(jax.local_devices(), ('i',))
# Partition specs for shard_map (PartitionSpec slices)
mat_spec = PartitionSpec('i', None)
out_spec = PartitionSpec('i',)

# Sharded matrix-vector function via shard_map
def _matvec_fn(mat_sh, v):
    return mat_sh @ v
_matvec_sharded = shard_map(_matvec_fn, device_mesh, in_specs=(mat_spec, None), out_specs=out_spec)

def shard_matvec(mat, vec):
    # Helper to shard mat/vec across devices
    devices = jax.local_devices()
    P = len(devices)
    rows, cols = mat.shape
    base = rows // P
    extra = rows % P
    rows_per_device = [base + (1 if i < extra else 0) for i in range(P)]
    max_rpd = max(rows_per_device)
    padded_rows = max_rpd * P
    # Pad and reshape into (P, max_rpd, cols)

    mat_padded = jnp.pad(mat, ((0, padded_rows - rows), (0, 0)))
    mat_sh = mat_padded.reshape(P, max_rpd, cols)

    # Use shard_map for local matrix-vector multiply
    res_sh = _matvec_sharded(mat_sh, vec)  # (P, max_rpd)
    return res_sh.reshape(-1)[:rows]


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

def apply_matrix_to_vector(mat, vec):
    """Single-device matrix-vector multiply on root only (no sharding)."""
    return mat @ vec


def haydock_eig(mat, n_eig, max_iter=40):
    n = mat.shape[0]
    # initial q
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    q = jax.random.normal(k1, (n,), dtype=mat.dtype) + 1j*jax.random.normal(k2, (n,), dtype=mat.dtype)
    q = q / jnp.linalg.norm(q)

    # preallocate
    Q    = jnp.zeros((n, max_iter+1), dtype=mat.dtype).at[:,0].set(q)
    alpha = jnp.zeros((max_iter,), dtype=mat.real.dtype)
    beta  = jnp.zeros((max_iter,), dtype=mat.real.dtype)
    col_idx = jnp.arange(max_iter+1)

    actual_iter = max_iter
    for j in range(max_iter):
        # iteration logging removed for JIT compatibility
        z = apply_matrix_to_vector(mat, q)
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
haydock_eig = jax.jit(haydock_eig, static_argnums=(1,2))

def main():
    # Example usage
    n = 100
    n_eig = 5
    # Random Hermitian matrix
    key = jax.random.PRNGKey(42)
    k1, k2 = jax.random.split(key)
    A = jax.random.normal(k1, (n, n)) + 1j * jax.random.normal(k2, (n, n))
    mat = 0.5 * (A + A.conj().T)

    evals_hay, evecs_hay = haydock_eig(mat, n_eig)
    evals_exact, evecs_exact = jnp.linalg.eigh(mat)
    idx = jnp.argsort(evals_exact)
    evals_exact = evals_exact[idx][:n_eig]
    evecs_exact = evecs_exact[:, idx[:n_eig]]

    print("Haydock eigenvalues:", evals_hay)
    print("Exact eigenvalues:", evals_exact)
    print("Max abs difference:", jnp.max(jnp.abs(evals_hay - evals_exact)))

    # Orthonormality check
    overlap = evecs_hay.conj().T @ evecs_hay - jnp.eye(n_eig)
    print("Orthonormality error:", jnp.linalg.norm(overlap))


if __name__ == "__main__":
    print("starting JAX ISDF+BSE calculation.")
    main()
