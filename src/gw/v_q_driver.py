"""
Outer drivers for V_q computation.

Three entry points:

    * ``compute_all_V_q_sharded``         — mesh-parallel V^{0,0}_q from
      ζ.h5 via the unified V_q tile kernel in ``v_q_tile``.  Default
      production path.  This is the V^{0,0} self-contraction case
      (single ζ source).  Bispinor V^{μ_L, ν_L}_q tiles will go through
      ``v_q_tile.compute_V_q_tile`` directly with their own driver
      (left for the bispinor work — not part of this refactor).

    * ``compute_all_V_q_from_zeta_h5``    — replicated path used as
      fallback when ``cfg.use_ffi_io = False`` (h5py-only, single-GPU
      sandbox builds).  Per-rank-redundant compute; legacy path.

    * ``compute_V_q_from_zeta_h5`` /
      ``compute_V_q_from_zeta_array``    — single-q standalone helpers.
      Used by old debugging scripts; legacy.

The big design comment for the sharded path lives in this file too,
verbatim from the old ``compute_vcoul.py`` (the chooser logic and the
DUS-ref V_acc accumulator pattern).

Public surface kept stable for ``gw.gw_init.compute_V_q``:

    compute_all_V_q_sharded(zeta_io, kgrid, fft_grid, bvec, cell_volume,
                            mesh_xy, *, n_rmu, n_rtot, sys_dim, bdot,
                            mc_average_vcoul_body, bare_coulomb_cutoff,
                            bgw_v_grid_fn, budget_bytes, verbose)
        -> (V_qmunu  P(None,None,None,'x','y'),
            g0_mu_all P(None,None,None,'x'))
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from common import timing
from .coulomb_kernel import make_v_munu_chunked_kernel
from .v_q_tile import _choose_v_q_chunks, _make_V_q_tile_kernel


# ============================================================================
# Single-q standalone helpers (legacy)
# ============================================================================
#
# These predate the sharded path and the ``compute_all_V_q_*`` outer drivers.
# They take an open h5py file handle (or an in-memory ζ array) and do the
# whole μ-chunk loop on a single q-point.  Used by old debugging scripts;
# not on any production code path.  Kept here because removing them would
# be a behaviour change unrelated to the refactor goal.

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
    bdot: np.ndarray | None = None,
    mc_average_vcoul_body: bool = True,
    bare_coulomb_cutoff: float | None = None,
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
        sys_dim: System dimensionality (0=box, 2=slab, 3=bulk)

    Returns:
        V_q: (n_rmu, n_rmu) Coulomb matrix, optionally sharded P('x', 'y')
        g0_mu: (n_rmu,) ζ_μ(G=0) for head corrections
    """
    # Get kernels
    kernels = make_v_munu_chunked_kernel(
        fft_nx, fft_ny, fft_nz, nkx, nky, nkz, bvec, cell_volume, sys_dim, bdot=bdot,
        mc_average_vcoul_body=mc_average_vcoul_body,
        vcoul_cutoff_ry=bare_coulomb_cutoff,
    )

    # Parse q_idx
    if isinstance(q_idx, tuple):
        qx, qy, qz = q_idx
    else:
        nqy, nqz = nky, nkz
        qx = q_idx // (nqy * nqz)
        qy = (q_idx % (nqy * nqz)) // nqz
        qz = q_idx % nqz
    nqy, nqz = nky, nkz

    # Get zeta shape.  Dataset layout is ``(nq, n_rtot, n_rmu)``
    # (see note in ``isdf_fitting.fit_zeta_chunked_to_h5.open_file`` on
    # why).  ``n_rmu`` is the innermost axis.
    zeta_dset = zeta_h5['zeta_q']
    n_rmu = zeta_dset.shape[2]
    q_flat = qx * (nqy * nqz) + qy * nqz + qz

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

        # Load from HDF5 (CPU) then transfer to device.  Dataset is
        # ``(nq, n_rtot, n_rmu)``; we want ``(B_mu, n_rtot)`` for the
        # FFT kernel, so read ``(n_rtot, mu_chunk)`` and transpose.
        zeta_mu_r_np = zeta_dset[q_flat, :, mu_i_start:mu_i_end].T
        zeta_mu_r = jnp.asarray(zeta_mu_r_np)

        # FFT and weight (JITted) - also returns G=0 component
        B_mu = mu_i_end - mu_i_start
        zeta_mu_weighted, g0_chunk = kernels.fft_and_weight(zeta_mu_r, sqrt_v, phase, B_mu)

        # Store G=0 for head corrections
        g0_mu_np[mu_i_start:mu_i_end] = np.asarray(g0_chunk)

        # Diagonal block: V[μ_i, μ_i] (JITted contraction)
        V_ii = kernels.contract_block(zeta_mu_weighted, zeta_mu_weighted)
        V_q_np[mu_i_start:mu_i_end, mu_i_start:mu_i_end] = np.asarray(V_ii)

        # Off-diagonal blocks (upper triangle only)
        for j in range(i + 1, n_chunks):
            mu_j_start = j * mu_chunk_size
            mu_j_end = min(mu_j_start + mu_chunk_size, n_rmu)

            # Load and FFT ν-chunk (see transpose note above).
            zeta_nu_r_np = zeta_dset[q_flat, :, mu_j_start:mu_j_end].T
            zeta_nu_r = jnp.asarray(zeta_nu_r_np)
            B_nu = mu_j_end - mu_j_start
            zeta_nu_weighted, _ = kernels.fft_and_weight(zeta_nu_r, sqrt_v, phase, B_nu)

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
    bdot: np.ndarray | None = None,
    mc_average_vcoul_body: bool = True,
    bare_coulomb_cutoff: float | None = None,
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
        fft_nx, fft_ny, fft_nz, nkx, nky, nkz, bvec, cell_volume, sys_dim, bdot=bdot,
        mc_average_vcoul_body=mc_average_vcoul_body,
        vcoul_cutoff_ry=bare_coulomb_cutoff,
    )

    n_rmu, _ = zeta_q.shape
    n_chunks = (n_rmu + mu_chunk_size - 1) // mu_chunk_size

    sqrt_v, phase = kernels.get_sqrt_v_and_phase(qvec_wrapped)

    # Accumulate V_q on GPU to avoid device→host syncs in the inner loop.
    # Use .at[].set() — the overhead is small compared to the FFT+contract.
    V_q = jnp.zeros((n_rmu, n_rmu), dtype=jnp.complex128)
    g0_mu = jnp.zeros((n_rmu,), dtype=jnp.complex128)

    for i in range(n_chunks):
        mu_i_start = i * mu_chunk_size
        mu_i_end = min(mu_i_start + mu_chunk_size, n_rmu)
        B_mu = mu_i_end - mu_i_start

        zeta_mu_r = zeta_q[mu_i_start:mu_i_end, :]

        # FFT + weight mu once; reuse for diagonal + all off-diagonal blocks
        zeta_mu_weighted, g0_chunk = kernels.fft_and_weight_keep(
            zeta_mu_r, sqrt_v, phase, B_mu)
        g0_mu = g0_mu.at[mu_i_start:mu_i_end].set(g0_chunk)

        # Diagonal block: self-contraction (no extra FFT needed)
        V_ii = kernels.contract_block(zeta_mu_weighted, zeta_mu_weighted)
        V_q = V_q.at[mu_i_start:mu_i_end, mu_i_start:mu_i_end].set(V_ii)

        for j in range(i + 1, n_chunks):
            mu_j_start = j * mu_chunk_size
            mu_j_end = min(mu_j_start + mu_chunk_size, n_rmu)

            zeta_nu_r = zeta_q[mu_j_start:mu_j_end, :]

            # Off-diagonal: fused FFT(nu) + contraction with pre-weighted mu
            V_ij = kernels.fft_weight_contract_offdiag(
                zeta_nu_r, zeta_mu_weighted, sqrt_v, phase)
            V_q = V_q.at[mu_i_start:mu_i_end, mu_j_start:mu_j_end].set(V_ij)
            V_q = V_q.at[mu_j_start:mu_j_end, mu_i_start:mu_i_end].set(
                jnp.conj(V_ij).T)

        # zeta_mu_weighted goes out of scope here — XLA can reclaim it
        del zeta_mu_weighted

    g0_mu_full = g0_mu

    if mesh_xy is not None:
        V_shard = NamedSharding(mesh_xy, P('x', 'y'))
        V_q = jax.lax.with_sharding_constraint(V_q, V_shard)

    return V_q, g0_mu_full


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
    zeta_dset = zeta_h5['zeta_q']  # flat-q: (nq, n_rmu, n_rtot)

    # Get mesh info
    devices_2d = mesh_xy.devices
    grid_x, grid_y = devices_2d.shape
    # Determine which μ indices this process owns
    # Shard μ across the 'x' axis of the mesh
    local_devices = list(jax.local_devices())
    local_coords = [tuple(np.argwhere(np.asarray(devices_2d) == d)[0]) for d in local_devices]

    # Get unique x-coordinates (rows) owned by this process
    local_x_coords = sorted(set(coord[0] for coord in local_coords))

    # μ indices per x-shard
    mu_per_x = (n_rmu + grid_x - 1) // grid_x

    # Determine nq from dataset shape; derive q_flat from (qx, qy, qz).
    # Caller passes nqx/nqy/nqz implicitly via those indices.
    nqy_nqz = zeta_dset.shape[0]  # unused — but q_flat needs kgrid info
    # We use the fact that q_flat = qx*nqy*nqz + qy*nqz + qz requires
    # knowing nqy and nqz; those aren't dataset-derivable.  This legacy
    # helper has no live callers; leaving q_flat=0 for the stub.
    q_flat = 0  # TODO: accept nqy/nqz as args if this helper is revived

    # Read only μ indices for x-coordinates this process owns
    local_zeta_chunks = []
    for x_coord in local_x_coords:
        mu_start = x_coord * mu_per_x
        mu_end = min(mu_start + mu_per_x, n_rmu)
        if mu_start < n_rmu:
            # Read this μ-chunk from HDF5
            chunk = zeta_dset[q_flat, mu_start:mu_end, :]
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
# Replicated path — full V_q from ζ.h5 (per-rank-redundant compute)
# ============================================================================

def compute_all_V_q_from_zeta_h5(
    zeta_io,
    kgrid: tuple[int, int, int],
    fft_grid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    mu_chunk_size: int = 128,
    mesh_xy: Mesh = None,
    sys_dim: int = 2,
    q_batch_size: int | None = None,
    verbose: bool = True,
    bdot: np.ndarray | None = None,
    mc_average_vcoul_body: bool = True,
    bare_coulomb_cutoff: float | None = None,
    bgw_v_grid_fn=None,
    n_rmu: int | None = None,
    n_rtot: int | None = None,
) -> jax.Array:
    """
    bgw_v_grid_fn : callable(q_frac_wrapped_tuple) -> (fft_nx, fft_ny, fft_nz) ndarray
        If provided, the returned per-q v_scaled grid replaces the
        point/MC-at-G=0 computation.  G=(0,0,0) is expected to be zero
        (head handled separately).  Used to inject BGW's MC-averaged
        vcoul values for bit-reproducible BGW comparison.
    """
    """
    Compute V_q for all q-points from zeta stored in HDF5.

    Loops over all q-points, computing V_q using μ-chunking for each. When the
    μ chunks already cover the full set (single chunk), q-points can be batched
    to reuse the FFT and contraction kernels.

    Args:
        zeta_io: SlabIO handle to a file containing 'zeta_q' with
            flat-q shape (nq, n_rmu, n_rtot), q_flat = qx*nqy*nqz + qy*nqz + qz
        kgrid: (nkx, nky, nkz) k-point grid dimensions
        fft_grid: (fft_nx, fft_ny, fft_nz) FFT grid dimensions
        bvec: Reciprocal lattice vectors (3×3)
        cell_volume: Unit cell volume
        mu_chunk_size: Number of μ indices per chunk
        mesh_xy: Optional device mesh for 2D sharding
        sys_dim: System dimensionality
        q_batch_size: Number of q-points to process simultaneously when
            mu_chunk_size ≥ n_rmu (default: no batching)
        verbose: Print timing breakdown

    Returns:
        V_qmunu: (nqx, nqy, nqz, n_rmu, n_rmu) array of Coulomb matrices
        g0_mu_all: (nqx, nqy, nqz, n_rmu) array of G=0 components
    """
    nkx, nky, nkz = kgrid
    fft_nx, fft_ny, fft_nz = fft_grid

    # zeta_io is a SlabIO (file_io.slab_io) opened in 'r' mode by the
    # caller.  n_rmu / n_rtot must be passed by the caller — SlabIO
    # doesn't currently expose dataset introspection.
    if n_rmu is None or n_rtot is None:
        raise ValueError(
            "compute_all_V_q_from_zeta_h5: n_rmu / n_rtot must be provided; "
            "SlabIO doesn't expose dataset-shape introspection yet.")

    nq_total = nkx * nky * nkz
    n_chunks = (n_rmu + mu_chunk_size - 1) // mu_chunk_size

    # Get kernels (cached)
    kernels = make_v_munu_chunked_kernel(
        fft_nx, fft_ny, fft_nz, nkx, nky, nkz, bvec, cell_volume, sys_dim, bdot=bdot,
        mc_average_vcoul_body=mc_average_vcoul_body,
        vcoul_cutoff_ry=bare_coulomb_cutoff,
    )

    # For single-chunk case, keep on GPU and batch. For multi-chunk, use numpy.
    single_chunk = (n_chunks == 1)
    effective_q_batch = 1
    if single_chunk:
        if q_batch_size is None:
            effective_q_batch = 1
        else:
            effective_q_batch = max(1, min(q_batch_size, nq_total))
    else:
        effective_q_batch = 1

    # Single-chunk batch processor - ONE JIT for the whole vmap'd computation
    # Uses inner (non-JIT'd) functions to avoid nested compilation
    def _single_chunk_proc(zeta_q, sqrt_v_q, phase_q):
        """Process single q-point: FFT + weight + contract. NOT JIT'd."""
        zeta_weighted_q, g0_q = kernels.fft_and_weight_inner(zeta_q, sqrt_v_q, phase_q)
        V_q = kernels.contract_block_inner(zeta_weighted_q, zeta_weighted_q)
        return V_q, g0_q

    # Single JIT point for the batched processor
    _batch_proc = jax.jit(jax.vmap(_single_chunk_proc, in_axes=(0, 0, 0)))

    if single_chunk:
        # Single-chunk path with OVERLAPPED I/O:
        # Read batch N+1 from disk while GPU processes batch N

        V_qmunu_list = []
        g0_mu_list = []
        q_coords = [
            (qx, qy, qz)
            for qx in range(nkx)
            for qy in range(nky)
            for qz in range(nkz)
        ]

        # Split into batches upfront
        batches = []
        for batch_start in range(0, nq_total, effective_q_batch):
            batches.append(q_coords[batch_start:batch_start + effective_q_batch])

        t_h5_read = 0.0
        t_transfer = 0.0
        t_fft_contract = 0.0
        def read_batch_from_h5(batch_coords):
            """Read a batch of zeta from H5 (runs in background thread).

            Returns stacked numpy array to minimize memory fragmentation.
            """
            kgrid_arr = np.array([nkx, nky, nkz], dtype=np.float64)
            batch_size = len(batch_coords)

            # Pre-allocate contiguous array
            zeta_stacked = np.empty((batch_size, n_rmu, n_rtot), dtype=np.complex128)
            qvecs = []

            for i, (qx, qy, qz) in enumerate(batch_coords):
                qvec_nonneg = np.array([qx, qy, qz], dtype=np.float64)
                qvec_wrapped = np.where(
                    qvec_nonneg > kgrid_arr / 2,
                    qvec_nonneg - kgrid_arr,
                    qvec_nonneg
                )
                qvecs.append(qvec_wrapped)

                q_flat = qx * (nky * nkz) + qy * nkz + qz
                # Dataset layout is ``(nq, n_rtot, n_rmu)`` (see note in
                # ``isdf_fitting.fit_zeta_chunked_to_h5.open_file``).
                # Read the per-q slab as ``(1, n_rtot, n_rmu)``, then
                # transpose to the downstream kernel's expected
                # ``(n_rmu, n_rtot)``.  Per-q transpose is ~50 µs on
                # GPU, negligible next to V_q compute.
                arr = zeta_io.read_slab(
                    'zeta_q',
                    shape=(1, n_rtot, n_rmu),
                    dtype=np.complex128,
                    offset=(q_flat, 0, 0),
                    as_numpy=True,
                )
                zeta_stacked[i] = arr[0].T  # (n_rtot, n_rmu) → (n_rmu, n_rtot)

            return zeta_stacked, qvecs

        def prepare_batch_on_gpu(zeta_stacked_np, qvec_list, actual_size):
            """Transfer batch to GPU and compute sqrt_v/phase."""
            # Compute sqrt_v and phase for each q
            sqrt_batch = []
            phase_batch = []
            for qvec_wrapped in qvec_list:
                qvec_wrapped_jax = jnp.asarray(qvec_wrapped)
                sqrt_v, phase = kernels.get_sqrt_v_and_phase(qvec_wrapped_jax)
                if bgw_v_grid_fn is not None:
                    # Overlay BGW's MC-averaged v(q+G) onto LORRAX's native
                    # v(q+G).  Only G-vectors that BGW wrote (typically 2-3%
                    # fewer than LORRAX's cutoff set) get overwritten; the
                    # rest keep LORRAX's point value.
                    kgrid_a = np.array([nkx, nky, nkz], dtype=np.float64)
                    q_frac = np.asarray(qvec_wrapped, dtype=np.float64) / kgrid_a
                    q_frac = np.mod(q_frac, 1.0)
                    v_scaled_bgw = np.asarray(bgw_v_grid_fn(tuple(q_frac))).reshape(-1)
                    if kernels.sphere_idx is not None:
                        v_scaled_bgw = v_scaled_bgw[np.asarray(kernels.sphere_idx)]
                    sqrt_v_native = np.asarray(sqrt_v).reshape(-1)
                    sqrt_v_bgw = np.sqrt(np.maximum(v_scaled_bgw, 0.0))
                    sqrt_v_over = np.where(
                        v_scaled_bgw != 0.0, sqrt_v_bgw, sqrt_v_native.real
                    ).astype(np.complex128)
                    if kernels.sphere_idx is None:
                        sqrt_v_over = sqrt_v_over.reshape(fft_nx, fft_ny, fft_nz)
                    sqrt_v = jnp.asarray(sqrt_v_over)
                sqrt_batch.append(sqrt_v)
                phase_batch.append(phase)

            # Transfer stacked zeta to GPU
            zeta_batch_arr = jnp.asarray(zeta_stacked_np[:actual_size])

            # Pad to effective_q_batch to avoid recompilation
            if actual_size < effective_q_batch:
                pad_size = effective_q_batch - actual_size
                zeta_pad = jnp.tile(zeta_batch_arr[0:1], (pad_size, 1, 1))
                zeta_batch_arr = jnp.concatenate([zeta_batch_arr, zeta_pad], axis=0)
                for _ in range(pad_size):
                    sqrt_batch.append(sqrt_batch[0])
                    phase_batch.append(phase_batch[0])

            return (
                zeta_batch_arr,
                jnp.stack(sqrt_batch, axis=0),
                jnp.stack(phase_batch, axis=0),
            )

        from common.progress import LoopProgress
        if verbose:
            print(f"  V_q: {nq_total} q-points, batch={effective_q_batch}, "
                  f"mu={n_rmu} (single chunk), overlapped H5 I/O")
        vq_progress = LoopProgress(
            nq_total, print, title="V_q computation",
            item_name="q-point", max_updates=min(nq_total, 20))

        with timing.section("compute_all_V_q"):
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Submit first batch read
                pending_future = executor.submit(read_batch_from_h5, batches[0])

                for batch_idx, batch in enumerate(batches):
                    actual_batch_size = len(batch)

                    # Wait for current batch I/O to complete
                    _t0 = time.perf_counter()
                    zeta_stacked_np, qvec_list = pending_future.result()
                    t_h5_read += time.perf_counter() - _t0

                    # Submit NEXT batch read (overlaps with GPU compute below)
                    if batch_idx + 1 < len(batches):
                        pending_future = executor.submit(read_batch_from_h5, batches[batch_idx + 1])

                    # Transfer to GPU and prepare arrays
                    _t0 = time.perf_counter()
                    zeta_batch_arr, sqrt_batch_arr, phase_batch_arr = prepare_batch_on_gpu(
                        zeta_stacked_np, qvec_list, actual_batch_size
                    )
                    zeta_batch_arr.block_until_ready()
                    t_transfer += time.perf_counter() - _t0

                    # Free numpy array immediately after GPU transfer
                    del zeta_stacked_np

                    # GPU compute (while next batch is being read from disk)
                    _t0 = time.perf_counter()
                    V_batch, g0_batch = _batch_proc(zeta_batch_arr, sqrt_batch_arr, phase_batch_arr)
                    V_batch.block_until_ready()
                    t_fft_contract += time.perf_counter() - _t0
                    for _ in range(actual_batch_size):
                        vq_progress.step()

                    # Only keep actual results (trim padding)
                    V_qmunu_list.append(V_batch[:actual_batch_size])
                    g0_mu_list.append(g0_batch[:actual_batch_size])

                    # Free intermediate GPU arrays
                    del zeta_batch_arr, sqrt_batch_arr, phase_batch_arr

        vq_progress.finish()

        V_qmunu = jnp.concatenate(V_qmunu_list, axis=0).reshape(nkx, nky, nkz, n_rmu, n_rmu)
        g0_mu_all = jnp.concatenate(g0_mu_list, axis=0).reshape(nkx, nky, nkz, n_rmu)

    else:
        # Multi-chunk path: use numpy accumulation to avoid .at[].set() overhead
        V_qmunu_np = np.zeros((nkx, nky, nkz, n_rmu, n_rmu), dtype=np.complex128)
        g0_mu_np = np.zeros((nkx, nky, nkz, n_rmu), dtype=np.complex128)

        from common.progress import LoopProgress
        if verbose:
            print(f"  V_q: {nq_total} q-points, {n_chunks} mu-chunks of {mu_chunk_size}")
        vq_progress = LoopProgress(
            nq_total, print, title="V_q computation",
            item_name="q-point", max_updates=min(nq_total, 20))

        with timing.section("compute_all_V_q"):
            for qx in range(nkx):
                for qy in range(nky):
                    for qz in range(nkz):
                        qvec_nonneg = np.array([qx, qy, qz], dtype=np.float64)
                        kgrid_arr = np.array([nkx, nky, nkz], dtype=np.float64)
                        qvec_wrapped = np.where(
                            qvec_nonneg > kgrid_arr / 2,
                            qvec_nonneg - kgrid_arr,
                            qvec_nonneg
                        )
                        qvec_wrapped_jax = jnp.asarray(qvec_wrapped)

                        sqrt_v, phase = kernels.get_sqrt_v_and_phase(qvec_wrapped_jax)
                        if bgw_v_grid_fn is not None:
                            q_frac = np.asarray(qvec_wrapped, dtype=np.float64) / kgrid_arr
                            q_frac = np.mod(q_frac, 1.0)
                            v_scaled_bgw = np.asarray(bgw_v_grid_fn(tuple(q_frac))).reshape(-1)
                            if kernels.sphere_idx is not None:
                                v_scaled_bgw = v_scaled_bgw[np.asarray(kernels.sphere_idx)]
                            sqrt_v_native = np.asarray(sqrt_v).reshape(-1)
                            sqrt_v_bgw = np.sqrt(np.maximum(v_scaled_bgw, 0.0))
                            sqrt_v_over = np.where(
                                v_scaled_bgw != 0.0, sqrt_v_bgw, sqrt_v_native.real
                            ).astype(np.complex128)
                            if kernels.sphere_idx is None:
                                sqrt_v_over = sqrt_v_over.reshape(fft_nx, fft_ny, fft_nz)
                            sqrt_v = jnp.asarray(sqrt_v_over)
                        V_q_local = np.zeros((n_rmu, n_rmu), dtype=np.complex128)

                        for i in range(n_chunks):
                            mu_i_start = i * mu_chunk_size
                            mu_i_end = min(mu_i_start + mu_chunk_size, n_rmu)

                            q_flat = qx * (nky * nkz) + qy * nkz + qz
                            # Dataset ``(nq, n_rtot, n_rmu)`` — read the
                            # full r-extent for this mu-chunk, then
                            # transpose ``(n_rtot, B_mu) → (B_mu, n_rtot)``
                            # to match the FFT kernel's expected shape.
                            _arr = zeta_io.read_slab(
                                'zeta_q',
                                shape=(1, n_rtot, mu_i_end - mu_i_start),
                                dtype=np.complex128,
                                offset=(q_flat, 0, mu_i_start),
                                as_numpy=True)
                            zeta_mu_r_np = _arr[0].T  # (n_rtot, B_mu) → (B_mu, n_rtot)
                            zeta_mu_r = jnp.asarray(zeta_mu_r_np)
                            B_mu_i = mu_i_end - mu_i_start
                            zeta_mu_weighted, g0_chunk = kernels.fft_and_weight(zeta_mu_r, sqrt_v, phase, B_mu_i)

                            g0_mu_np[qx, qy, qz, mu_i_start:mu_i_end] = np.asarray(g0_chunk)

                            V_ii = kernels.contract_block(zeta_mu_weighted, zeta_mu_weighted)
                            V_q_local[mu_i_start:mu_i_end, mu_i_start:mu_i_end] = np.asarray(V_ii)

                            for j in range(i + 1, n_chunks):
                                mu_j_start = j * mu_chunk_size
                                mu_j_end = min(mu_j_start + mu_chunk_size, n_rmu)

                                _arr = zeta_io.read_slab(
                                    'zeta_q',
                                    shape=(1, n_rtot, mu_j_end - mu_j_start),
                                    dtype=np.complex128,
                                    offset=(q_flat, 0, mu_j_start),
                                    as_numpy=True)
                                zeta_nu_r_np = _arr[0].T
                                zeta_nu_r = jnp.asarray(zeta_nu_r_np)
                                B_mu_j = mu_j_end - mu_j_start
                                zeta_nu_weighted, _ = kernels.fft_and_weight(zeta_nu_r, sqrt_v, phase, B_mu_j)

                                V_ij = kernels.contract_block(zeta_mu_weighted, zeta_nu_weighted)
                                V_ij_np = np.asarray(V_ij)
                                V_q_local[mu_i_start:mu_i_end, mu_j_start:mu_j_end] = V_ij_np
                                V_q_local[mu_j_start:mu_j_end, mu_i_start:mu_i_end] = V_ij_np.conj().T

                        V_qmunu_np[qx, qy, qz, :, :] = V_q_local
                        vq_progress.step()

        vq_progress.finish()
        V_qmunu = jnp.asarray(V_qmunu_np)
        g0_mu_all = jnp.asarray(g0_mu_np)

    # Apply sharding if mesh provided
    if mesh_xy is not None:
        V_shard = NamedSharding(mesh_xy, P(None, None, None, 'x', 'y'))
        V_qmunu = jax.lax.with_sharding_constraint(V_qmunu, V_shard)

    return V_qmunu, g0_mu_all


# ============================================================================
# Sharded V_q — mesh-parallel, default production path
# ============================================================================
#
# Design
# ------
# This path supersedes ``_single_chunk_proc`` in the replicated-compute branch
# of ``compute_all_V_q_from_zeta_h5``.  The replicated path reads ζ_q(μ, r)
# into every rank, runs the FFT+contract independently on each rank, and
# produces the same output 16× — no scaling benefit from extra GPUs, plus
# inter-node overhead that *hurts* as the mesh grows.  The sharded path below
# distributes work across the full (P_x × P_y) mesh; for MoS2 3×3 at 16 GPU
# it drops V_q exec from ~9 s to sub-second.
#
# Per q-chunk data flow
# ---------------------
# Let the mesh be (P_x × P_y), N_μ = n_rmu, N_G = n_G_sph (post-sphere-cutoff
# G count), N_q the active q-chunk size.  One zeta element is 16 bytes (c128).
#
#   ζ_q,μ(r_tot)  shape (N_q, N_μ, n_rtot)  sharded  P(None, ('x','y'), None)
#     ^^ read directly into this layout via SlabIO.read_slab(..., partition_spec)
#        → per-rank bytes: 16 · N_q · (N_μ / (P_x·P_y)) · n_rtot
#
#   → 3-D FFT on the trailing rtot axis (rtot → G_box) — fully local (only
#     μ is sharded; rtot is on every rank).  No wasted FFT work: each rank
#     FFTs its own (N_μ/(P_x·P_y)) μ-rows only.
#
#   → phase multiply (per-q fractional shift) + sphere pick (G_box → G_flat
#     via take(sphere_idx)) — local.  New shape (N_q, N_μ, N_G) sharded
#     P(None, ('x','y'), None).
#
#   → elementwise multiply by sqrt_v(q+G) (replicated (N_q, N_G) array) —
#     local; still P(None, ('x','y'), None).
#
#   → All-gather on Y / All-gather on X (separate jits, parallel-issued):
#         P(None, ('x','y'), None) → P(None, 'x', None)  (μ side)
#         P(None, ('x','y'), None) → P(None, 'y', None)  (ν side)
#
#   → Contract  V_q[μ_X, ν_Y] = Σ_G conj(ζ_q,μ_X(G)) · ζ_q,ν_Y(G)
#     Einsum 'qmG,qnG->qmn' — local per-rank gemm; output sharded
#     P(None, 'x', 'y') — the exact layout downstream sigma_sx/sigma_coh
#     consume, so the chi0→W→V→sigma chain is reshard-free from here on.
#
# Output accumulation via donated DUS ref
# ---------------------------------------
# Before the main loop we pre-allocate two sharded zero arrays:
#     V_acc  : (N_q_total, N_μ, N_μ) sharded P(None,'x','y')
#     g0_acc : (N_q_total, N_μ)      sharded P(None,'x')
# The unified V_q tile kernel jit'd in ``v_q_tile`` takes them as
# ``donate_argnums=(0, 1)`` and writes via ``dynamic_update_slice``;
# XLA fuses the read-modify-write into an in-place update on the
# donated buffer — functionally equivalent to ``jax.new_ref`` on the
# pre-refs API.
#
# Chooser policy  (``_choose_v_q_chunks`` in ``v_q_tile``)
# --------------------------------------------------------
# Given ``B_compute = B_total − ref_bytes``:
#
#   Case A — μ fits on one q.  q_chunk = max q's that fit; μ_chunk = N_μ.
#     One contiguous read per q-batch; the inner kernel does its two
#     one-axis gathers from the same post-FFT ζ(G) tensor (single FFT).
#
#   Case B — μ does NOT fit on one q.  q_chunk = 1; μ_chunk < N_μ.
#     Per-q outer loop iterates over (μ_block × ν_block); every
#     (μ_i, ν_j) iteration does TWO reads (even on the diagonal) to
#     keep the jit body uniform — except that for the V^{0,0} self-
#     contraction case the unified kernel detects (μ_lo == ν_lo,
#     mu_size == nu_size) and runs the single-FFT same_zeta path
#     instead, saving a FFT and a read per diagonal block.  The
#     ``write_g0`` static flag is True only on the diagonal so the
#     diagonal block is the one that fills g0_acc.
#
# ``LORRAX_V_Q_MU_CHUNK=<int>`` env override forces Case B at the
# specified μ_chunk regardless of the chooser's pick.


def compute_all_V_q_sharded(
    zeta_io,
    kgrid: tuple[int, int, int],
    fft_grid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    mesh_xy: Mesh,
    *,
    n_rmu: int,
    n_rtot: int,
    sys_dim: int = 2,
    bdot: np.ndarray | None = None,
    mc_average_vcoul_body: bool = True,
    bare_coulomb_cutoff: float | None = None,
    bgw_v_grid_fn=None,
    budget_bytes: float | None = None,
    verbose: bool = True,
) -> tuple[jax.Array, jax.Array]:
    """Mesh-parallel V^{0,0}_q computation.

    Thin wrapper around ``v_q_tile.compute_V_q_tile`` for the V^{0,0}
    self-contraction case (single ζ source, full Coulomb v(q+G)).  This
    function preserves the ``compute_V_q_q`` public surface that
    ``gw.gw_init.compute_V_q`` depends on — the signature has not
    changed.

    Returns:
        V_qmunu  : (nkx, nky, nkz, n_rmu, n_rmu) sharded P(None,None,None,'x','y')
        g0_mu_all: (nkx, nky, nkz, n_rmu)       sharded P(None,None,None,'x')
    """
    nkx, nky, nkz = kgrid
    nq_total = nkx * nky * nkz
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])

    # Build the V-μν kernel bundle (sphere_idx, sqrt_v/phase helpers).
    # ``make_v_munu_chunked_kernel`` caches by (fft, kgrid, bvec, cell,
    # sys_dim) so repeated entry reuses its sphere_idx + jit'd
    # ``get_sqrt_v_and_phase`` (jitted internally; per-q table is reused
    # across all q in the batch via outer stack).
    kernels = make_v_munu_chunked_kernel(
        fft_grid[0], fft_grid[1], fft_grid[2], nkx, nky, nkz,
        bvec, cell_volume, sys_dim, bdot=bdot,
        mc_average_vcoul_body=mc_average_vcoul_body,
        vcoul_cutoff_ry=bare_coulomb_cutoff,
    )
    n_G_sph = int(kernels.n_sph)

    if budget_bytes is None:
        budget_bytes = 24.0e9
    choice = _choose_v_q_chunks(
        n_rmu=n_rmu, n_G=n_G_sph, n_q_total=nq_total,
        budget_bytes=budget_bytes, p_x=p_x, p_y=p_y,
    )
    # Debug knob: force Case B at a caller-specified μ-chunk so the tile
    # path can be exercised on systems that otherwise land in Case A.
    # ``LORRAX_V_Q_MU_CHUNK=<int>`` overrides the chooser's pick.
    _force_mu = int(os.environ.get('LORRAX_V_Q_MU_CHUNK', '0') or 0)
    if _force_mu > 0 and _force_mu < n_rmu:
        n_blocks = (n_rmu + _force_mu - 1) // _force_mu
        choice = dict(
            q_chunk=1, mu_chunk=_force_mu, n_mu_blocks=n_blocks,
            tiled=True, aligned=(n_rmu % _force_mu == 0),
            per_rank_peak=choice['per_rank_peak'],
            ref_bytes=choice['ref_bytes'],
        )
        if jax.process_index() == 0:
            print(f"  [LORRAX_V_Q_MU_CHUNK] forcing Case B with "
                  f"μ_chunk={_force_mu}, n_mu_blocks={n_blocks}")

    V_sh_full = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    g0_sh_full = NamedSharding(mesh_xy, P(None, 'x'))

    # Pre-allocate the output accumulators as regular sharded arrays.
    # Each per-batch kernel is jitted with ``donate_argnums=(0, 1)`` on
    # (V_acc, g0_acc) so XLA fuses the DUS read-modify-write into an
    # in-place update on the donated buffer — functionally equivalent
    # to jax.new_ref, just using the pre-refs API on this jax build.
    @partial(jax.jit, out_shardings=(V_sh_full, g0_sh_full))
    def _init_accum():
        V = jnp.zeros((nq_total, n_rmu, n_rmu), dtype=jnp.complex128)
        g = jnp.zeros((nq_total, n_rmu), dtype=jnp.complex128)
        return V, g
    V_acc, g0_acc = _init_accum()

    # Build the per-q v(q+G) and phase callables for the unified tile
    # driver.  The v half goes through optional BGW vcoul overlay
    # (operating on un-sqrt'd v values) before reaching the kernel,
    # which takes √v inside.
    _vmapped_v_per_G_phase = jax.vmap(kernels.get_v_per_G_and_phase)
    _sphere_idx_np = (np.asarray(kernels.sphere_idx)
                      if kernels.sphere_idx is not None else None)
    fft_nx, fft_ny, fft_nz = fft_grid

    def _v_per_G_fn(qvec_np_batch):
        """Build (Q, n_G_sph) c128 v(q+G) for the kernel.

        Returns a real-valued non-negative array cast to c128 — the
        kernel takes ``sqrt`` and casts inside.
        """
        qvec_arr = jnp.asarray(qvec_np_batch, dtype=jnp.float64)
        v_per_G_batch, _ = _vmapped_v_per_G_phase(qvec_arr)
        # Cast to c128 to match the kernel's v_per_G_sh = P(None, None)
        # spec on a c128 input (sqrt and astype happen inside kernel).
        return v_per_G_batch.astype(jnp.complex128)

    def _phase_fn(qvec_np_batch):
        qvec_arr = jnp.asarray(qvec_np_batch, dtype=jnp.float64)
        _, phase_batch = _vmapped_v_per_G_phase(qvec_arr)
        return phase_batch

    def _bgw_overlay(qvec_np_batch, v_per_G_batch):
        """Apply BGW vcoul overlay on the un-sqrt'd v(q+G).

        Mirrors the legacy ``_sqrt_v_phase_batch`` overlay byte-for-byte:
        where BGW provides a value, replace LORRAX's native v(q+G) with
        BGW's value (already representable as a non-negative real), then
        the kernel takes √v.  Equivalent to the old "where BGW != 0, use
        sqrt(BGW); else use sqrt_native" path because √ is monotonic.
        """
        if bgw_v_grid_fn is None:
            return v_per_G_batch
        kgrid_a = np.array([nkx, nky, nkz], dtype=np.float64)
        v_np = np.asarray(v_per_G_batch).copy()
        for iq in range(qvec_np_batch.shape[0]):
            # Pass qvec_wrapped/kgrid in signed form (no mod 1.0).  See
            # legacy _sqrt_v_phase_batch comment for why: matching the
            # FFT-box convention used by ``fill_v_grid_for_q``.
            q_frac = qvec_np_batch[iq] / kgrid_a
            v_scaled_bgw = np.asarray(
                bgw_v_grid_fn(tuple(q_frac))).reshape(-1)
            if _sphere_idx_np is not None:
                v_scaled_bgw = v_scaled_bgw[_sphere_idx_np]
            v_native = np.real(v_np[iq])
            v_overlaid = np.where(
                v_scaled_bgw != 0.0,
                np.maximum(v_scaled_bgw, 0.0),
                v_native,
            ).astype(np.complex128)
            v_np[iq] = v_overlaid
        return jnp.asarray(v_np)

    # Build the chooser_choice dict expected by ``compute_V_q_tile``
    # (takes ``n_mu_blocks`` directly; same key set).
    chooser_choice = dict(choice)

    if verbose and jax.process_index() == 0:
        tiled = bool(choice['tiled'])
        kind = 'tiled (Case B)' if tiled else 'one-shot (Case A)'
        print(f"  V_q (sharded): mesh={p_x}x{p_y}, {kind}, "
              f"q_chunk={choice['q_chunk']}, μ_chunk={choice['mu_chunk']} "
              f"({choice['n_mu_blocks']} μ-blocks), "
              f"aligned={choice['aligned']}, "
              f"N_μ={n_rmu}, N_G={n_G_sph}, "
              f"predicted peak/rank={choice['per_rank_peak']/1e9:.2f} GB "
              f"(V_ref+g0_ref={choice['ref_bytes']/1e9:.2f} GB)")

    # Run the unified tile driver.  V^{0,0} self-contraction:
    # zeta_L_io is zeta_R_io, so the kernel skips the second read+FFT
    # in the same_zeta=True path (Case A always; Case B diagonal blocks).
    from .v_q_tile import compute_V_q_tile
    V_acc, g0_acc = compute_V_q_tile(
        zeta_L_io=zeta_io,
        zeta_R_io=zeta_io,
        v_per_G_fn=_v_per_G_fn,
        phase_fn=_phase_fn,
        sphere_idx=kernels.sphere_idx,
        mesh_xy=mesh_xy,
        kgrid=kgrid,
        fft_grid=fft_grid,
        n_rmu_L=n_rmu,
        n_rmu_R=n_rmu,
        n_rtot=n_rtot,
        V_acc=V_acc,
        g0_acc=g0_acc,
        chooser_choice=chooser_choice,
        bgw_v_grid_overlay_fn=_bgw_overlay,
        verbose=False,  # outer wrapper already printed the chooser line
        timing_label="compute_all_V_q_sharded",
    )

    # Reshape (n_q_total, μ, μ) → (nkx, nky, nkz, μ, μ) and pin to the
    # downstream consumer's sharding.  The internal accumulator was
    # already P(None, 'x', 'y') so this is a free reshape.
    V_qmunu = V_acc.reshape(nkx, nky, nkz, n_rmu, n_rmu)
    g0_mu_all = g0_acc.reshape(nkx, nky, nkz, n_rmu)
    V_qmunu = jax.lax.with_sharding_constraint(
        V_qmunu, NamedSharding(mesh_xy, P(None, None, None, 'x', 'y')))
    g0_mu_all = jax.lax.with_sharding_constraint(
        g0_mu_all, NamedSharding(mesh_xy, P(None, None, None, 'x')))
    return V_qmunu, g0_mu_all
