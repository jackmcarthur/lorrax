"""
Chunked computation of V_q(μ, ν) = Σ_G ζ̃*_μ(G) ζ̃_ν(G) from zeta stored in HDF5.

This module provides memory-efficient routines for computing the ISDF Coulomb
matrix elements when the full zeta_q(μ, r) doesn't fit in GPU memory.

Key features:
- μ-chunked FFT: Process B_μ centroids at a time
- ν-chunked contraction: Compute V blocks without caching FFT outputs
- Hermitian symmetry: Only compute upper triangle, fill lower by conjugation
- 2D sharding: Output V_q sharded P('x', 'y') for downstream use

Memory model:
- FFT workspace: O(B_μ × n_G) per chunk
- V_q output: O(n_μ²) - typically small (e.g., 2304² × 16B = 85 MB)
- Redundant FFT work: O((n_μ/B_μ)²) vs O(n_μ/B_μ) with caching

Note: For future optimization, if a single zeta_q(μ, r) fits on sqrt(P) processors,
      we could batch multiple q-points to amortize FFT setup costs.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from jax.experimental.shard_map import shard_map
from functools import partial

from ..common import timing


# ============================================================================
# FFT grid helpers (mirrors cohsex_jax.py)
# ============================================================================

def exp_ikr_fftbox(fft_nx: int, fft_ny: int, fft_nz: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return fractional coordinate grids for constructing exp(ik·r) on the FFT box."""
    fx = jnp.arange(fft_nx, dtype=jnp.float64)[None, :, None, None] / float(fft_nx)
    fy = jnp.arange(fft_ny, dtype=jnp.float64)[None, None, :, None] / float(fft_ny)
    fz = jnp.arange(fft_nz, dtype=jnp.float64)[None, None, None, :] / float(fft_nz)
    return fx, fy, fz


def fft_integer_axes(fft_nx: int, fft_ny: int, fft_nz: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return integer FFT frequency grids in numpy.fft.fftfreq order."""
    gx = (jnp.fft.fftfreq(fft_nx) * fft_nx).astype(jnp.float64).reshape(fft_nx, 1, 1)
    gy = (jnp.fft.fftfreq(fft_ny) * fft_ny).astype(jnp.float64).reshape(1, fft_ny, 1)
    gz = (jnp.fft.fftfreq(fft_nz) * fft_nz).astype(jnp.float64).reshape(1, 1, fft_nz)
    return gx, gy, gz


# ============================================================================
# Coulomb potential computation (2D truncated)
# ============================================================================

def compute_sqrt_vcoul_2d(
    qvec_wrapped: jax.Array,
    fft_nx: int,
    fft_ny: int,
    fft_nz: int,
    nkx: int,
    nky: int,
    nkz: int,
    bvec: np.ndarray,
    cell_volume: float,
) -> jax.Array:
    """
    Compute √v(q+G) for 2D truncated Coulomb on the FFT grid.
    
    Returns:
        sqrt_v: (fft_nx, fft_ny, fft_nz) array of √v(q+G) values
    """
    gx, gy, gz = fft_integer_axes(fft_nx, fft_ny, fft_nz)
    gx_b, gy_b, gz_b = jnp.broadcast_arrays(gx, gy, gz)
    gstack = jnp.stack((gx_b, gy_b, gz_b), axis=-1)
    
    bvec_j = jnp.asarray(bvec, dtype=jnp.float64)
    fact = jnp.float64(1.0 / cell_volume)
    zc = jnp.float64(np.pi / float(bvec[2, 2]))
    
    G_cart_base = jnp.einsum('...a,ab->...b', gstack, bvec_j, optimize=True)
    
    q_frac = jnp.asarray((
        qvec_wrapped[0] / float(nkx),
        qvec_wrapped[1] / float(nky),
        qvec_wrapped[2] / float(nkz),
    ), dtype=jnp.float64)
    q_cart = jnp.einsum('a,ab->b', q_frac, bvec_j, optimize=True).reshape((1, 1, 1, 3))
    G_cart = G_cart_base + q_cart
    
    denom = jnp.sum(G_cart * G_cart, axis=-1)
    denom_zero = denom < 1e-12
    denom_safe = jnp.where(denom_zero, 1.0, denom)
    
    kxy = jnp.sqrt(G_cart[..., 0]**2 + G_cart[..., 1]**2)
    kz = G_cart[..., 2]
    f2d = 1.0 - jnp.exp(-zc * kxy) * jnp.cos(kz * zc)
    
    v_reg = (8.0 * jnp.pi / denom_safe) * f2d
    v_scaled = jnp.where(denom_zero, 0.0, v_reg * fact)
    sqrt_v = jnp.where(v_scaled > 0.0, jnp.sqrt(v_scaled), 0.0).astype(jnp.complex128)
    
    return sqrt_v


def compute_phase_q(
    qvec_wrapped: jax.Array,
    fft_nx: int,
    fft_ny: int,
    fft_nz: int,
    nkx: int,
    nky: int,
    nkz: int,
) -> jax.Array:
    """
    Compute exp(-2πi q·r) phase factor for FFT.
    
    Returns:
        phase: (1, fft_nx, fft_ny, fft_nz) array for broadcasting with zeta
    """
    fx, fy, fz = exp_ikr_fftbox(fft_nx, fft_ny, fft_nz)
    phase = jnp.exp(-2j * jnp.pi * (
        qvec_wrapped[0] / float(nkx) * fx +
        qvec_wrapped[1] / float(nky) * fy +
        qvec_wrapped[2] / float(nkz) * fz
    ))
    return phase


# ============================================================================
# Kernel factory for V_q computation (caches static grid data)
# ============================================================================

_v_munu_kernel_cache = {}


def make_v_munu_chunked_kernel(
    fft_nx: int,
    fft_ny: int,
    fft_nz: int,
    nkx: int,
    nky: int,
    nkz: int,
    bvec: np.ndarray,
    cell_volume: float,
    sys_dim: int = 2,
):
    """
    Factory for jitted kernels that compute V_q blocks from zeta chunks.
    
    This creates two kernels:
    1. fft_and_weight: zeta_r(B_μ, n_rtot) → zeta_weighted(B_μ, n_G)
    2. contract_block: (zeta_μ, zeta_ν) → V_block(B_μ, B_ν)
    
    Args:
        fft_nx, fft_ny, fft_nz: FFT grid dimensions
        nkx, nky, nkz: k-grid dimensions
        bvec: Reciprocal lattice vectors (3×3)
        cell_volume: Unit cell volume
        sys_dim: System dimensionality (only 2 supported currently)
    
    Returns:
        Namespace with fft_and_weight, contract_block, get_sqrt_v, get_phase kernels
    """
    if sys_dim != 2:
        raise NotImplementedError("Chunked V_q currently supports sys_dim == 2 only")
    
    cache_key = (fft_nx, fft_ny, fft_nz, nkx, nky, nkz, tuple(bvec.flatten()), cell_volume)
    if cache_key in _v_munu_kernel_cache:
        return _v_munu_kernel_cache[cache_key]
    
    n_G = fft_nx * fft_ny * fft_nz
    
    # Precompute static grid data
    fx, fy, fz = exp_ikr_fftbox(fft_nx, fft_ny, fft_nz)
    gx, gy, gz = fft_integer_axes(fft_nx, fft_ny, fft_nz)
    gx_b, gy_b, gz_b = jnp.broadcast_arrays(gx, gy, gz)
    gstack = jnp.stack((gx_b, gy_b, gz_b), axis=-1)
    
    bvec_j = jnp.asarray(bvec, dtype=jnp.float64)
    fact = jnp.float64(1.0 / cell_volume)
    zc = jnp.float64(np.pi / float(bvec[2, 2]))
    G_cart_base = jnp.einsum('...a,ab->...b', gstack, bvec_j, optimize=True)
    
    nkx_f = jnp.float64(nkx)
    nky_f = jnp.float64(nky)
    nkz_f = jnp.float64(nkz)
    
    @jax.jit
    def get_sqrt_v_and_phase(qvec_wrapped: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute √v(q+G) and phase for a given q-point."""
        # Phase factor
        phase = jnp.exp(-2j * jnp.pi * (
            qvec_wrapped[0] / nkx_f * fx +
            qvec_wrapped[1] / nky_f * fy +
            qvec_wrapped[2] / nkz_f * fz
        ))
        
        # Coulomb potential
        q_frac = jnp.asarray((
            qvec_wrapped[0] / nkx_f,
            qvec_wrapped[1] / nky_f,
            qvec_wrapped[2] / nkz_f,
        ), dtype=jnp.float64)
        q_cart = jnp.einsum('a,ab->b', q_frac, bvec_j, optimize=True).reshape((1, 1, 1, 3))
        G_cart = G_cart_base + q_cart
        
        denom = jnp.sum(G_cart * G_cart, axis=-1)
        denom_zero = denom < 1e-12
        denom_safe = jnp.where(denom_zero, 1.0, denom)
        
        kxy = jnp.sqrt(G_cart[..., 0]**2 + G_cart[..., 1]**2)
        kz_arr = G_cart[..., 2]
        f2d = 1.0 - jnp.exp(-zc * kxy) * jnp.cos(kz_arr * zc)
        
        v_reg = (8.0 * jnp.pi / denom_safe) * f2d
        v_scaled = jnp.where(denom_zero, 0.0, v_reg * fact)
        sqrt_v = jnp.where(v_scaled > 0.0, jnp.sqrt(v_scaled), 0.0).astype(jnp.complex128)
        
        return sqrt_v, phase
    
    @jax.jit
    def fft_and_weight(
        zeta_r: jax.Array,
        sqrt_v: jax.Array,
        phase: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        FFT a chunk of zeta and weight by √v.
        
        Args:
            zeta_r: (B_μ, n_rtot) real-space zeta chunk
            sqrt_v: (fft_nx, fft_ny, fft_nz) precomputed √v(q+G)
            phase: (1, fft_nx, fft_ny, fft_nz) phase factor
        
        Returns:
            zeta_weighted: (B_μ, n_G) weighted G-space zeta
            g0_chunk: (B_μ,) unweighted G=0 component for head corrections
        """
        B_mu = zeta_r.shape[0]
        zeta_spatial = zeta_r.reshape(B_mu, fft_nx, fft_ny, fft_nz)
        zeta_phased = zeta_spatial * phase
        zeta_G = jnp.fft.fftn(zeta_phased, axes=(-3, -2, -1))
        g0_chunk = zeta_G[:, 0, 0, 0]  # Extract G=0 before weighting
        zeta_weighted = zeta_G * sqrt_v[None, :, :, :]
        return zeta_weighted.reshape(B_mu, n_G), g0_chunk
    
    @jax.jit
    def contract_block(
        zeta_mu: jax.Array,
        zeta_nu: jax.Array,
    ) -> jax.Array:
        """
        Contract two weighted zeta chunks to get V block.
        
        V[μ, ν] = Σ_G ζ̃*_μ(G) ζ̃_ν(G)
        
        Args:
            zeta_mu: (B_μ, n_G) weighted G-space zeta for μ-chunk
            zeta_nu: (B_ν, n_G) weighted G-space zeta for ν-chunk
        
        Returns:
            V_block: (B_μ, B_ν) Coulomb matrix block
        """
        return jnp.einsum('mG,nG->mn', jnp.conj(zeta_mu), zeta_nu, optimize=True)
    
    # Bundle kernels
    from types import SimpleNamespace
    kernels = SimpleNamespace(
        get_sqrt_v_and_phase=get_sqrt_v_and_phase,
        fft_and_weight=fft_and_weight,
        contract_block=contract_block,
        n_G=n_G,
        fft_shape=(fft_nx, fft_ny, fft_nz),
    )
    
    _v_munu_kernel_cache[cache_key] = kernels
    return kernels


# ============================================================================
# Main chunked V_q computation
# ============================================================================

def compute_V_q_from_zeta_h5(
    zeta_h5,
    q_idx: int,
    qvec_wrapped: jax.Array,
    fft_nx: int,
    fft_ny: int,
    fft_nz: int,
    nkx: int,
    nky: int,
    nkz: int,
    bvec: np.ndarray,
    cell_volume: float,
    mu_chunk_size: int = 128,
    mesh_xy: Mesh = None,
    sys_dim: int = 2,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute V_q(μ, ν) from zeta stored in HDF5 using μ/ν chunking.
    
    V_q(μ, ν) = Σ_G ζ̃*_μ(G) ζ̃_ν(G)
    
    where ζ̃_μ(G) = √v(q+G) × FFT[phase_q(r) × ζ_μ(r)]
    
    Uses Hermitian symmetry: only computes upper triangle, fills lower by conjugation.
    FFTs are recomputed per (μ,ν) block pair (no caching) to minimize memory.
    
    Args:
        zeta_h5: Open HDF5 file or group containing 'zeta_q' dataset
                 with shape (nqx, nqy, nqz, n_rmu, n_rtot)
        q_idx: Flat q-point index, or (qx, qy, qz) tuple
        qvec_wrapped: q-vector in wrapped crystal coordinates
        fft_nx, fft_ny, fft_nz: FFT grid dimensions
        nkx, nky, nkz: k-grid dimensions
        bvec: Reciprocal lattice vectors (3×3)
        cell_volume: Unit cell volume
        mu_chunk_size: Number of μ indices to process at once
        mesh_xy: Optional device mesh for 2D sharding of output
        sys_dim: System dimensionality (only 2 supported)
    
    Returns:
        V_q: (n_rmu, n_rmu) Coulomb matrix, optionally sharded P('x', 'y')
        g0_mu: (n_rmu,) ζ_μ(G=0) for head corrections
    """
    # Get kernels
    kernels = make_v_munu_chunked_kernel(
        fft_nx, fft_ny, fft_nz, nkx, nky, nkz, bvec, cell_volume, sys_dim
    )
    
    # Parse q_idx
    if isinstance(q_idx, tuple):
        qx, qy, qz = q_idx
    else:
        nqy, nqz = nky, nkz
        qx = q_idx // (nqy * nqz)
        qy = (q_idx % (nqy * nqz)) // nqz
        qz = q_idx % nqz
    
    # Get zeta shape
    zeta_dset = zeta_h5['zeta_q']
    n_rmu = zeta_dset.shape[3]
    
    n_chunks = (n_rmu + mu_chunk_size - 1) // mu_chunk_size
    
    # Precompute √v and phase for this q (JITted)
    sqrt_v, phase = kernels.get_sqrt_v_and_phase(qvec_wrapped)
    
    # Pre-allocate output as numpy, fill blocks, convert to JAX at end
    # This avoids O(n²) JAX array copies from .at[].set() in loop
    V_q_np = np.zeros((n_rmu, n_rmu), dtype=np.complex128)
    g0_mu_np = np.zeros((n_rmu,), dtype=np.complex128)
    
    # Process μ-chunks (outer loop)
    for i in range(n_chunks):
        mu_i_start = i * mu_chunk_size
        mu_i_end = min(mu_i_start + mu_chunk_size, n_rmu)
        
        # Load from HDF5 (CPU) then transfer to device
        zeta_mu_r_np = zeta_dset[qx, qy, qz, mu_i_start:mu_i_end, :]
        zeta_mu_r = jnp.asarray(zeta_mu_r_np)
        
        # FFT and weight (JITted) - also returns G=0 component
        zeta_mu_weighted, g0_chunk = kernels.fft_and_weight(zeta_mu_r, sqrt_v, phase)
        
        # Store G=0 for head corrections
        g0_mu_np[mu_i_start:mu_i_end] = np.asarray(g0_chunk)
        
        # Diagonal block: V[μ_i, μ_i] (JITted contraction)
        V_ii = kernels.contract_block(zeta_mu_weighted, zeta_mu_weighted)
        V_q_np[mu_i_start:mu_i_end, mu_i_start:mu_i_end] = np.asarray(V_ii)
        
        # Off-diagonal blocks (upper triangle only)
        for j in range(i + 1, n_chunks):
            mu_j_start = j * mu_chunk_size
            mu_j_end = min(mu_j_start + mu_chunk_size, n_rmu)
            
            # Load and FFT ν-chunk
            zeta_nu_r_np = zeta_dset[qx, qy, qz, mu_j_start:mu_j_end, :]
            zeta_nu_r = jnp.asarray(zeta_nu_r_np)
            zeta_nu_weighted, _ = kernels.fft_and_weight(zeta_nu_r, sqrt_v, phase)
            
            # Contract (JITted)
            V_ij = kernels.contract_block(zeta_mu_weighted, zeta_nu_weighted)
            V_ij_np = np.asarray(V_ij)
            
            # Set both upper and lower triangle (Hermitian)
            V_q_np[mu_i_start:mu_i_end, mu_j_start:mu_j_end] = V_ij_np
            V_q_np[mu_j_start:mu_j_end, mu_i_start:mu_i_end] = V_ij_np.conj().T
    
    # Convert to JAX array
    V_q = jnp.asarray(V_q_np)
    g0_mu_full = jnp.asarray(g0_mu_np)
    
    # Apply 2D sharding if mesh provided
    if mesh_xy is not None:
        V_shard = NamedSharding(mesh_xy, P('x', 'y'))
        V_q = jax.lax.with_sharding_constraint(V_q, V_shard)
    
    return V_q, g0_mu_full


def compute_V_q_from_zeta_array(
    zeta_q: jax.Array,
    qvec_wrapped: jax.Array,
    fft_nx: int,
    fft_ny: int,
    fft_nz: int,
    nkx: int,
    nky: int,
    nkz: int,
    bvec: np.ndarray,
    cell_volume: float,
    mu_chunk_size: int = 128,
    mesh_xy: Mesh = None,
    sys_dim: int = 2,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute V_q(μ, ν) from zeta array in memory using μ/ν chunking.
    
    Same as compute_V_q_from_zeta_h5 but takes zeta as a JAX array instead of HDF5.
    Useful for testing or when zeta is already in memory.
    
    Args:
        zeta_q: (n_rmu, n_rtot) zeta array for this q-point
        qvec_wrapped: q-vector in wrapped crystal coordinates
        ... (same as compute_V_q_from_zeta_h5)
    
    Returns:
        V_q: (n_rmu, n_rmu) Coulomb matrix
        g0_mu: (n_rmu,) ζ_μ(G=0) for head corrections
    """
    kernels = make_v_munu_chunked_kernel(
        fft_nx, fft_ny, fft_nz, nkx, nky, nkz, bvec, cell_volume, sys_dim
    )
    
    n_rmu, n_rtot = zeta_q.shape
    n_chunks = (n_rmu + mu_chunk_size - 1) // mu_chunk_size
    
    sqrt_v, phase = kernels.get_sqrt_v_and_phase(qvec_wrapped)
    
    # Pre-allocate as numpy to avoid .at[].set() overhead
    V_q_np = np.zeros((n_rmu, n_rmu), dtype=np.complex128)
    g0_mu_np = np.zeros((n_rmu,), dtype=np.complex128)
    
    for i in range(n_chunks):
        mu_i_start = i * mu_chunk_size
        mu_i_end = min(mu_i_start + mu_chunk_size, n_rmu)
        
        zeta_mu_r = zeta_q[mu_i_start:mu_i_end, :]
        zeta_mu_weighted, g0_chunk = kernels.fft_and_weight(zeta_mu_r, sqrt_v, phase)
        
        g0_mu_np[mu_i_start:mu_i_end] = np.asarray(g0_chunk)
        
        V_ii = kernels.contract_block(zeta_mu_weighted, zeta_mu_weighted)
        V_q_np[mu_i_start:mu_i_end, mu_i_start:mu_i_end] = np.asarray(V_ii)
        
        for j in range(i + 1, n_chunks):
            mu_j_start = j * mu_chunk_size
            mu_j_end = min(mu_j_start + mu_chunk_size, n_rmu)
            
            zeta_nu_r = zeta_q[mu_j_start:mu_j_end, :]
            zeta_nu_weighted, _ = kernels.fft_and_weight(zeta_nu_r, sqrt_v, phase)
            
            V_ij = kernels.contract_block(zeta_mu_weighted, zeta_nu_weighted)
            V_ij_np = np.asarray(V_ij)
            
            V_q_np[mu_i_start:mu_i_end, mu_j_start:mu_j_end] = V_ij_np
            V_q_np[mu_j_start:mu_j_end, mu_i_start:mu_i_end] = V_ij_np.conj().T
    
    V_q = jnp.asarray(V_q_np)
    g0_mu_full = jnp.asarray(g0_mu_np)
    
    if mesh_xy is not None:
        V_shard = NamedSharding(mesh_xy, P('x', 'y'))
        V_q = jax.lax.with_sharding_constraint(V_q, V_shard)
    
    return V_q, g0_mu_full


# ============================================================================
# Sharded zeta reads (distributed I/O)
# ============================================================================

def read_zeta_q_sharded(
    zeta_h5,
    qx: int,
    qy: int, 
    qz: int,
    n_rmu: int,
    n_rtot: int,
    mesh_xy: Mesh,
) -> jax.Array:
    """
    Read zeta_q from HDF5 with μ-sharding across processes.
    
    Each process reads only its portion of μ indices, then combines
    into a globally sharded array. This distributes I/O across nodes.
    
    Args:
        zeta_h5: Open HDF5 file with 'zeta_q' dataset
        qx, qy, qz: q-point indices
        n_rmu: Total number of μ points
        n_rtot: Total number of r points
        mesh_xy: Device mesh for sharding
    
    Returns:
        zeta_q: (n_rmu, n_rtot) array sharded along μ axis
    """
    zeta_dset = zeta_h5['zeta_q']
    
    # Get mesh info
    devices_2d = mesh_xy.devices
    grid_x, grid_y = devices_2d.shape
    total_devices = grid_x * grid_y
    
    # Determine which μ indices this process owns
    # Shard μ across the 'x' axis of the mesh
    local_devices = list(jax.local_devices())
    local_coords = [tuple(np.argwhere(np.asarray(devices_2d) == d)[0]) for d in local_devices]
    
    # Get unique x-coordinates (rows) owned by this process
    local_x_coords = sorted(set(coord[0] for coord in local_coords))
    
    # μ indices per x-shard
    mu_per_x = (n_rmu + grid_x - 1) // grid_x
    
    # Read only μ indices for x-coordinates this process owns
    local_zeta_chunks = []
    for x_coord in local_x_coords:
        mu_start = x_coord * mu_per_x
        mu_end = min(mu_start + mu_per_x, n_rmu)
        if mu_start < n_rmu:
            # Read this μ-chunk from HDF5
            chunk = zeta_dset[qx, qy, qz, mu_start:mu_end, :]
            # Pad if needed for uniform shard sizes
            if chunk.shape[0] < mu_per_x:
                pad_size = mu_per_x - chunk.shape[0]
                chunk = np.pad(chunk, ((0, pad_size), (0, 0)), mode='constant')
            local_zeta_chunks.append(chunk)
    
    # Stack local chunks
    if local_zeta_chunks:
        local_zeta = np.concatenate(local_zeta_chunks, axis=0)
    else:
        local_zeta = np.zeros((0, n_rtot), dtype=np.complex128)
    
    # Create globally sharded array
    # Shard along μ (axis 0) across 'x' dimension
    global_shape = (mu_per_x * grid_x, n_rtot)  # Padded shape
    mu_sharding = NamedSharding(mesh_xy, P('x', None))
    
    local_zeta_jax = jax.device_put(local_zeta)
    global_zeta = jax.make_array_from_process_local_data(
        mu_sharding, local_zeta_jax, global_shape
    )
    
    # Trim to actual size if padded
    if global_shape[0] > n_rmu:
        global_zeta = global_zeta[:n_rmu, :]
    
    return global_zeta


# ============================================================================
# Full V_q computation pipeline with all q-points
# ============================================================================

def compute_all_V_q_from_zeta_h5(
    zeta_h5,
    kgrid: tuple[int, int, int],
    fft_grid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    mu_chunk_size: int = 128,
    mesh_xy: Mesh = None,
    sys_dim: int = 2,
    verbose: bool = True,
) -> jax.Array:
    """
    Compute V_q for all q-points from zeta stored in HDF5.
    
    Loops over all q-points, computing V_q using μ-chunking for each.
    
    Args:
        zeta_h5: Open HDF5 file containing 'zeta_q' with shape (nqx, nqy, nqz, n_rmu, n_rtot)
        kgrid: (nkx, nky, nkz) k-point grid dimensions
        fft_grid: (fft_nx, fft_ny, fft_nz) FFT grid dimensions
        bvec: Reciprocal lattice vectors (3×3)
        cell_volume: Unit cell volume
        mu_chunk_size: Number of μ indices per chunk
        mesh_xy: Optional device mesh for 2D sharding
        sys_dim: System dimensionality
        verbose: Print timing breakdown
    
    Returns:
        V_qmunu: (nqx, nqy, nqz, n_rmu, n_rmu) array of Coulomb matrices
        g0_mu_all: (nqx, nqy, nqz, n_rmu) array of G=0 components
    """
    nkx, nky, nkz = kgrid
    fft_nx, fft_ny, fft_nz = fft_grid
    
    zeta_dset = zeta_h5['zeta_q']
    n_rmu = zeta_dset.shape[3]
    n_rtot = zeta_dset.shape[4]
    
    nq_total = nkx * nky * nkz
    n_chunks = (n_rmu + mu_chunk_size - 1) // mu_chunk_size
    
    # Get kernels (cached)
    kernels = make_v_munu_chunked_kernel(
        fft_nx, fft_ny, fft_nz, nkx, nky, nkz, bvec, cell_volume, sys_dim
    )
    
    # For single-chunk case, keep on GPU and batch. For multi-chunk, use numpy.
    single_chunk = (n_chunks == 1)
    
    # Determine if we should use sharded reads
    use_sharded_io = (mesh_xy is not None and jax.process_count() > 1)
    
    if single_chunk:
        # Single-chunk path: keep results on GPU, avoid CPU round-trips
        V_qmunu_list = []
        g0_mu_list = []
        
        with timing.section("compute_all_V_q"):
            q_flat = 0
            for qx in range(nkx):
                for qy in range(nky):
                    for qz in range(nkz):
                        if verbose:
                            print(f"  q-point {q_flat+1}/{nq_total}: q=({qx},{qy},{qz})")
                        
                        # Compute phase and sqrt_v for this q
                        qvec_nonneg = np.array([qx, qy, qz], dtype=np.float64)
                        kgrid_arr = np.array([nkx, nky, nkz], dtype=np.float64)
                        qvec_wrapped = np.where(
                            qvec_nonneg > kgrid_arr / 2,
                            qvec_nonneg - kgrid_arr,
                            qvec_nonneg
                        )
                        qvec_wrapped_jax = jnp.asarray(qvec_wrapped)
                        sqrt_v, phase = kernels.get_sqrt_v_and_phase(qvec_wrapped_jax)
                        
                        # Load zeta for this q
                        if use_sharded_io:
                            zeta_q = read_zeta_q_sharded(
                                zeta_h5, qx, qy, qz, n_rmu, n_rtot, mesh_xy
                            )
                        else:
                            zeta_q_np = zeta_dset[qx, qy, qz, :, :]
                            zeta_q = jnp.asarray(zeta_q_np)
                        
                        # FFT and weight (stays on GPU)
                        zeta_weighted, g0_mu = kernels.fft_and_weight(zeta_q, sqrt_v, phase)
                        
                        # Contract V = zeta^H @ zeta
                        V_q = kernels.contract_block(zeta_weighted, zeta_weighted)
                        
                        V_qmunu_list.append(V_q)
                        g0_mu_list.append(g0_mu)
                        q_flat += 1
            
            if verbose:
                print()
        
        # Stack results on GPU
        V_qmunu = jnp.stack(V_qmunu_list).reshape(nkx, nky, nkz, n_rmu, n_rmu)
        g0_mu_all = jnp.stack(g0_mu_list).reshape(nkx, nky, nkz, n_rmu)
    
    else:
        # Multi-chunk path: use numpy accumulation to avoid .at[].set() overhead
        V_qmunu_np = np.zeros((nkx, nky, nkz, n_rmu, n_rmu), dtype=np.complex128)
        g0_mu_np = np.zeros((nkx, nky, nkz, n_rmu), dtype=np.complex128)
        
        with timing.section("compute_all_V_q"):
            for qx in range(nkx):
                for qy in range(nky):
                    for qz in range(nkz):
                        q_flat = qx * nky * nkz + qy * nkz + qz
                        
                        qvec_nonneg = np.array([qx, qy, qz], dtype=np.float64)
                        kgrid_arr = np.array([nkx, nky, nkz], dtype=np.float64)
                        qvec_wrapped = np.where(
                            qvec_nonneg > kgrid_arr / 2,
                            qvec_nonneg - kgrid_arr,
                            qvec_nonneg
                        )
                        qvec_wrapped_jax = jnp.asarray(qvec_wrapped)
                        
                        if verbose:
                            print(f"  q-point {q_flat+1}/{nq_total}: q=({qx},{qy},{qz})")
                        
                        sqrt_v, phase = kernels.get_sqrt_v_and_phase(qvec_wrapped_jax)
                        V_q_local = np.zeros((n_rmu, n_rmu), dtype=np.complex128)
                        
                        for i in range(n_chunks):
                            mu_i_start = i * mu_chunk_size
                            mu_i_end = min(mu_i_start + mu_chunk_size, n_rmu)
                            
                            zeta_mu_r_np = zeta_dset[qx, qy, qz, mu_i_start:mu_i_end, :]
                            zeta_mu_r = jnp.asarray(zeta_mu_r_np)
                            zeta_mu_weighted, g0_chunk = kernels.fft_and_weight(zeta_mu_r, sqrt_v, phase)
                            
                            g0_mu_np[qx, qy, qz, mu_i_start:mu_i_end] = np.asarray(g0_chunk)
                            
                            V_ii = kernels.contract_block(zeta_mu_weighted, zeta_mu_weighted)
                            V_q_local[mu_i_start:mu_i_end, mu_i_start:mu_i_end] = np.asarray(V_ii)
                            
                            for j in range(i + 1, n_chunks):
                                mu_j_start = j * mu_chunk_size
                                mu_j_end = min(mu_j_start + mu_chunk_size, n_rmu)
                                
                                zeta_nu_r_np = zeta_dset[qx, qy, qz, mu_j_start:mu_j_end, :]
                                zeta_nu_r = jnp.asarray(zeta_nu_r_np)
                                zeta_nu_weighted, _ = kernels.fft_and_weight(zeta_nu_r, sqrt_v, phase)
                                
                                V_ij = kernels.contract_block(zeta_mu_weighted, zeta_nu_weighted)
                                V_ij_np = np.asarray(V_ij)
                                V_q_local[mu_i_start:mu_i_end, mu_j_start:mu_j_end] = V_ij_np
                                V_q_local[mu_j_start:mu_j_end, mu_i_start:mu_i_end] = V_ij_np.conj().T
                        
                        V_qmunu_np[qx, qy, qz, :, :] = V_q_local
            
            if verbose:
                print()
        
        V_qmunu = jnp.asarray(V_qmunu_np)
        g0_mu_all = jnp.asarray(g0_mu_np)
    
    # Apply sharding if mesh provided
    if mesh_xy is not None:
        V_shard = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
        V_qmunu = jax.lax.with_sharding_constraint(V_qmunu, V_shard)
    
    return V_qmunu, g0_mu_all


# ============================================================================
# Compatibility wrapper matching cohsex_jax.make_v_munu_kernel signature
# ============================================================================

def make_v_munu_kernel_chunked(
    fft_nx: int,
    fft_ny: int,
    fft_nz: int,
    nkx: int,
    nky: int,
    nkz: int,
    bvec: np.ndarray,
    cell_volume: float,
    sys_dim: int,
    mu_chunk_size: int = 128,
):
    """
    Factory for chunked V_q kernel with same signature as cohsex_jax.make_v_munu_kernel.
    
    Returns a kernel function that takes (zeta_q, qvec_wrapped) and returns (v_munu, g0_mu),
    but uses μ-chunking internally for memory efficiency.
    
    Drop-in replacement for make_v_munu_kernel when memory is constrained.
    """
    kernels = make_v_munu_chunked_kernel(
        fft_nx, fft_ny, fft_nz, nkx, nky, nkz, bvec, cell_volume, sys_dim
    )
    
    def kernel(zeta_q: jax.Array, qvec_wrapped: jax.Array) -> tuple[jax.Array, jax.Array]:
        return compute_V_q_from_zeta_array(
            zeta_q, qvec_wrapped,
            fft_nx, fft_ny, fft_nz,
            nkx, nky, nkz,
            bvec, cell_volume,
            mu_chunk_size=mu_chunk_size,
            mesh_xy=None,
            sys_dim=sys_dim,
        )
    
    return kernel

