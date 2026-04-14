"""psp/kpar.py — K-point parallel DFT eigenvalue solver.

Distributes k-points across a 2D JAX mesh:
  axis 0 ('k'): k-point parallelism — each device gets nk/ngpu k-points
  axis 1 ('g'): G-vector parallelism — length 1 for now (future: distributed FFT)

Usage
-----
    mesh = kpar.make_mesh()  # (ngpu, 1) mesh with axes ('k', 'g')
    ngkmax = compute_ngkmax(kpoints, bdot, ecutwfc, fft_grid)

    # Pre-build all per-k data (CPU, fast with ngkmax padding)
    H_data = kpar.build_all_H_k(kpoints, V_scf, vnl_setup, crystal, ngkmax)

    # Diagonalize all k-points in parallel across mesh
    eigenvalues = kpar.diagonalize_all_k(H_data, psi_all, mesh)
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


# ---------------------------------------------------------------------------
# Mesh construction
# ---------------------------------------------------------------------------

def make_mesh(ngpu: int | None = None) -> Mesh:
    """Create a 2D mesh ('k', 'g') with g-axis = 1.

    Parameters
    ----------
    ngpu : number of GPUs for k-parallelism. Default: all available.
    """
    devices = jax.devices()
    if ngpu is None:
        ngpu = len(devices)
    else:
        ngpu = min(ngpu, len(devices))
    device_array = np.array(devices[:ngpu]).reshape(ngpu, 1)
    return Mesh(device_array, ('k', 'g'))


# ---------------------------------------------------------------------------
# Batched per-k Hamiltonian data
# ---------------------------------------------------------------------------

class AllKData:
    """Stacked per-k Hamiltonian arrays, ready for sharding along k-axis.

    All arrays have leading dimension nk and uniform trailing shapes
    (via ngkmax padding).
    """
    __slots__ = (
        'T_diag', 'Gx', 'Gy', 'Gz', 'vnl_Z', 'vnl_E',
        'h_diag', 'mask', 'V_scf', 'nk', 'ngkmax', 'fft_grid',
        'nG_per_k',
    )

    def __init__(self, T_diag, Gx, Gy, Gz, vnl_Z, vnl_E,
                 h_diag, mask, V_scf, nG_per_k, fft_grid):
        self.T_diag = T_diag    # (nk, ngkmax)
        self.Gx = Gx            # (nk, ngkmax) int32
        self.Gy = Gy
        self.Gz = Gz
        self.vnl_Z = vnl_Z      # (nk, total_R, ngkmax) complex128
        self.vnl_E = vnl_E      # (nk, ns, ns, total_R, total_R) complex128
        self.h_diag = h_diag    # (nk, ngkmax)
        self.mask = mask         # (nk, ngkmax) bool
        self.V_scf = V_scf      # (nx, ny, nz) — shared, not per-k
        self.nk = T_diag.shape[0]
        self.ngkmax = T_diag.shape[1]
        self.nG_per_k = nG_per_k  # (nk,) int — actual nG per k-point
        self.fft_grid = fft_grid


def build_all_H_k(
    kpoints: np.ndarray,
    V_scf: jax.Array,
    vnl_setup,
    crystal,
    ngkmax: int,
    V_loc_r: jax.Array | None = None,
) -> AllKData:
    """Pre-build per-k Hamiltonian data for all k-points.

    Uses ngkmax padding so all arrays have uniform shapes.
    CPU-side loop (fast: ~0.02s per k-point).

    Parameters
    ----------
    kpoints : (nk, 3) crystal coordinates
    V_scf : (nx, ny, nz) combined local potential
    vnl_setup : from vnl_ops.build_vnl_setup
    crystal : CrystalData or WFNReader (needs bdot, ecutwfc, fft_grid)
    ngkmax : from compute_ngkmax
    V_loc_r : (nx, ny, nz) for h_diag preconditioner. Optional.
    """
    from psp.dft_operators import build_T_diag_from_kvec, build_h_diag
    import psp.vnl_ops as vnl_ops

    kpoints = np.asarray(kpoints, dtype=np.float64)
    nk = len(kpoints)
    fft_grid = tuple(int(x) for x in crystal.fft_grid)

    # Pre-allocate on CPU
    T_all = np.zeros((nk, ngkmax), dtype=np.float64)
    Gx_all = np.zeros((nk, ngkmax), dtype=np.int32)
    Gy_all = np.zeros((nk, ngkmax), dtype=np.int32)
    Gz_all = np.zeros((nk, ngkmax), dtype=np.int32)
    mask_all = np.zeros((nk, ngkmax), dtype=bool)
    nG_per_k = np.zeros(nk, dtype=np.int32)

    # VNL arrays — need to know total_R first
    vnl_Z_list = []
    vnl_E_list = []
    h_diag_list = []

    for ik in range(nk):
        kvec = kpoints[ik]
        T_diag, Gx, Gy, Gz = build_T_diag_from_kvec(kvec, crystal)
        nG_actual = int(Gx.shape[0])
        nG_per_k[ik] = nG_actual

        # Pad G-vectors to ngkmax
        pad = ngkmax - nG_actual
        if pad > 0:
            T_pad = np.pad(np.asarray(T_diag), (0, pad), constant_values=1e10)
            Gx_pad = np.pad(np.asarray(Gx), (0, pad), constant_values=0)
            Gy_pad = np.pad(np.asarray(Gy), (0, pad), constant_values=0)
            Gz_pad = np.pad(np.asarray(Gz), (0, pad), constant_values=0)
            mask_k = np.concatenate([np.ones(nG_actual, dtype=bool),
                                     np.zeros(pad, dtype=bool)])
        else:
            T_pad = np.asarray(T_diag)
            Gx_pad = np.asarray(Gx)
            Gy_pad = np.asarray(Gy)
            Gz_pad = np.asarray(Gz)
            mask_k = np.ones(ngkmax, dtype=bool)

        T_all[ik] = T_pad
        Gx_all[ik] = Gx_pad
        Gy_all[ik] = Gy_pad
        Gz_all[ik] = Gz_pad
        mask_all[ik] = mask_k

        # Build VNL at padded size (uniform shapes)
        Gk_int_padded = np.stack([Gx_pad, Gy_pad, Gz_pad], axis=-1)
        kdata = vnl_ops.build_vnl_kdata_from_kvec(kvec, Gk_int_padded, vnl_setup)
        vnl_Z_list.append(np.asarray(kdata.Z))
        vnl_E_list.append(np.asarray(kdata.E_super))

        # h_diag
        T_j = jnp.asarray(T_pad)
        if V_loc_r is not None:
            hd = build_h_diag(T_j, V_loc_r, kdata.Z, kdata.E_super)
        else:
            hd = T_j
        h_diag_list.append(np.asarray(hd))

    return AllKData(
        T_diag=jnp.asarray(T_all),
        Gx=jnp.asarray(Gx_all),
        Gy=jnp.asarray(Gy_all),
        Gz=jnp.asarray(Gz_all),
        vnl_Z=jnp.asarray(np.stack(vnl_Z_list, axis=0)),
        vnl_E=jnp.asarray(np.stack(vnl_E_list, axis=0)),
        h_diag=jnp.asarray(np.stack(h_diag_list, axis=0)),
        mask=jnp.asarray(mask_all),
        V_scf=V_scf,
        nG_per_k=nG_per_k,
        fft_grid=fft_grid,
    )


# ---------------------------------------------------------------------------
# Batched matrix diagonalization (vmap over k, sharded across mesh)
# ---------------------------------------------------------------------------

def _build_matrix_k_single(psi_box, T_diag, V_scf, Gx, Gy, Gz,
                            vnl_Z, vnl_E, mask):
    """H matrix at one k-point — no JIT decorator, used inside vmap."""
    import psp.vnl_ops as vnl_ops

    mask_f = mask[None, None, :].astype(psi_box.dtype)
    psi_G = psi_box[:, :, Gx, Gy, Gz] * mask_f

    H_mn = jnp.einsum('msG,nsG->mn', jnp.conj(psi_G),
                       T_diag[None, None, :] * psi_G, optimize=True)

    psi_r = jnp.fft.ifftn(psi_box, axes=(-3, -2, -1), norm='ortho')
    Vpsi_G = jnp.fft.fftn(
        psi_r * V_scf, axes=(-3, -2, -1), norm='ortho'
    )[:, :, Gx, Gy, Gz] * mask_f
    H_mn = H_mn + jnp.einsum('msG,nsG->mn', jnp.conj(psi_G), Vpsi_G, optimize=True)

    H_mn = H_mn + vnl_ops.vnl_matrix(psi_G, vnl_Z, vnl_E)
    return H_mn


@jax.jit
def diagonalize_all_k(psi_all, T_diag_all, V_scf,
                       Gx_all, Gy_all, Gz_all,
                       vnl_Z_all, vnl_E_all, mask_all):
    """Build H matrices and diagonalize at all k-points (batched).

    Parameters
    ----------
    psi_all : (nk, nb, nspinor, nx, ny, nz) — wavefunctions at all k
    T_diag_all, Gx_all, ... : (nk, ngkmax) — per-k Hamiltonian data
    V_scf : (nx, ny, nz) — shared local potential

    Returns
    -------
    eigenvalues : (nk, nb)
    """
    def _single_k(psi_box, T_diag, Gx, Gy, Gz, vnl_Z, vnl_E, mask):
        H_mn = _build_matrix_k_single(psi_box, T_diag, V_scf,
                                       Gx, Gy, Gz, vnl_Z, vnl_E, mask)
        H_mn = 0.5 * (H_mn + H_mn.conj().T)
        return jnp.linalg.eigvalsh(H_mn)

    return jax.vmap(_single_k)(
        psi_all, T_diag_all, Gx_all, Gy_all, Gz_all,
        vnl_Z_all, vnl_E_all, mask_all,
    )


def solve_eigenvalues(H_data: AllKData, psi_all: jax.Array,
                      mesh: Mesh) -> jax.Array:
    """Solve eigenvalues at all k-points, sharded across mesh.

    Parameters
    ----------
    H_data : from build_all_H_k
    psi_all : (nk, nb, nspinor, nx, ny, nz) wavefunctions
    mesh : 2D Mesh with axes ('k', 'g')

    Returns
    -------
    eigenvalues : (nk, nb) float64
    """
    # Shard per-k arrays along k-axis, replicate V_scf
    k_spec = P('k', None)
    k_spec_3d = P('k', None, None)
    k_spec_5d = P('k', None, None, None, None)
    psi_spec = P('k', None, None, None, None, None)
    replicated = P()

    T_s = jax.device_put(H_data.T_diag, NamedSharding(mesh, k_spec))
    Gx_s = jax.device_put(H_data.Gx, NamedSharding(mesh, k_spec))
    Gy_s = jax.device_put(H_data.Gy, NamedSharding(mesh, k_spec))
    Gz_s = jax.device_put(H_data.Gz, NamedSharding(mesh, k_spec))
    Z_s = jax.device_put(H_data.vnl_Z, NamedSharding(mesh, k_spec_3d))
    E_s = jax.device_put(H_data.vnl_E, NamedSharding(mesh, k_spec_5d))
    mask_s = jax.device_put(H_data.mask, NamedSharding(mesh, k_spec))
    V_s = jax.device_put(H_data.V_scf, NamedSharding(mesh, replicated))
    psi_s = jax.device_put(psi_all, NamedSharding(mesh, psi_spec))

    evals = diagonalize_all_k(
        psi_s, T_s, V_s, Gx_s, Gy_s, Gz_s, Z_s, E_s, mask_s,
    )
    return evals


__all__ = [
    "AllKData",
    "build_all_H_k",
    "diagonalize_all_k",
    "make_mesh",
    "solve_eigenvalues",
]
