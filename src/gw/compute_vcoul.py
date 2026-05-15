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

import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial

from common import timing
from common.fft_helpers import make_sharded_fftn_3d


# ============================================================================
# FFT grid helpers (mirrors gw_jax.py)
# ============================================================================

def exp_ikr_fftbox(fft_nx: int, fft_ny: int, fft_nz: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return fractional coordinate grids for constructing exp(ik·r) on the FFT box.

    ─ NOTE TO FUTURE EDITORS — THE numpy USAGE BELOW IS INTENTIONAL ─
    Tiny host-side shape builders (fft_n* is typically 20-200).  Using
    ``jnp.arange`` + ``jnp.reshape`` fired 3 standalone pjit compiles
    per call with zero runtime benefit.  Commit bbff26f (2026-04-18)
    switched to numpy; promoted to ``jax.Array`` only at return.
    DO NOT "fix" back to ``jnp``.
    """
    fx = np.arange(fft_nx, dtype=np.float64)[None, :, None, None] / float(fft_nx)
    fy = np.arange(fft_ny, dtype=np.float64)[None, None, :, None] / float(fft_ny)
    fz = np.arange(fft_nz, dtype=np.float64)[None, None, None, :] / float(fft_nz)
    return jnp.asarray(fx), jnp.asarray(fy), jnp.asarray(fz)


def fft_integer_axes(fft_nx: int, fft_ny: int, fft_nz: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return integer FFT frequency grids in numpy.fft.fftfreq order.

    ─ NOTE TO FUTURE EDITORS — THE numpy USAGE BELOW IS INTENTIONAL ─
    Host-side tiny-grid shape helper.  ``jnp.fft.fftfreq`` + reshape +
    astype each fire their own pjit at trace time — ~7 cache misses per
    call.  Commit bbff26f (2026-04-18) converted to numpy with the JAX
    cast deferred to return.  DO NOT "fix" back to ``jnp``.
    """
    gx = (np.fft.fftfreq(fft_nx) * fft_nx).astype(np.float64).reshape(fft_nx, 1, 1)
    gy = (np.fft.fftfreq(fft_ny) * fft_ny).astype(np.float64).reshape(1, fft_ny, 1)
    gz = (np.fft.fftfreq(fft_nz) * fft_nz).astype(np.float64).reshape(1, 1, fft_nz)
    return jnp.asarray(gx), jnp.asarray(gy), jnp.asarray(gz)


# ============================================================================
# Coulomb potential computation (2D truncated)
# ============================================================================

def compute_sqrt_vcoul_0d(
    fft_nx: int,
    fft_ny: int,
    fft_nz: int,
    bdot: np.ndarray,
    cell_volume: float,
) -> jax.Array:
    """
    Compute √v(G) for cell-box-truncated Coulomb (0D/molecule) on the FFT grid.

    Unlike 2D slab truncation (analytic formula), box truncation requires a
    numerical approach: build 1/r on a denser real-space grid truncated to the
    Wigner-Seitz cell, FFT to G-space, then sample at the WFN G-vectors.
    See compute_vcoul_0d.py for the standalone version and detailed derivation.

    Only q=0 is supported (box truncation is undefined for q≠0).
    The G=0 component is finite (integral of 1/r over the WS cell).

    Parameters
    ----------
    fft_nx, fft_ny, fft_nz : int
        WFN FFT grid dimensions.
    bdot : (3, 3) float64
        Reciprocal-space metric matrix in Bohr^-2 (as stored in WFN).
    cell_volume : float
        Unit cell volume in Bohr^3.

    Returns
    -------
    sqrt_v : (fft_nx, fft_ny, fft_nz) complex128
        √(v(G) / cell_volume) on the FFT grid, for use in V_μν contraction.
    """
    from .compute_vcoul_0d import compute_vcoul_box, _round_up_fft_size, N_IN_BOX, NCELL, TRUNC_SHIFT

    fft_grid = np.array([fft_nx, fft_ny, fft_nz], dtype=int)
    dkmax = fft_grid * N_IN_BOX
    dNfft = np.array([_round_up_fft_size(int(g)) for g in dkmax])

    # --- Build real-space metric (same as compute_vcoul_box) ---
    bdot = np.asarray(bdot, dtype=np.float64)
    adot = np.linalg.inv(bdot) * (4.0 * np.pi**2)
    for i in range(3):
        for j in range(3):
            adot[i, j] /= dNfft[i] * dNfft[j]
    scale = 2.0 * np.sqrt(np.linalg.det(adot))

    # --- Build V_trunc(r) on dense grid with WS minimum-image truncation ---
    replica_offsets = []
    for l3 in range(-NCELL + 1, NCELL + 1):
        for l2 in range(-NCELL + 1, NCELL + 1):
            for l1 in range(-NCELL + 1, NCELL + 1):
                replica_offsets.append(np.array([
                    l1 * dNfft[0], l2 * dNfft[1], l3 * dNfft[2],
                ], dtype=np.float64))

    i1 = np.arange(dNfft[0], dtype=np.float64) + TRUNC_SHIFT
    i2 = np.arange(dNfft[1], dtype=np.float64) + TRUNC_SHIFT
    i3 = np.arange(dNfft[2], dtype=np.float64) + TRUNC_SHIFT
    rr1, rr2, rr3 = np.meshgrid(i1, i2, i3, indexing='ij')

    r_len_sq = np.full(tuple(dNfft), np.inf, dtype=np.float64)
    for offset in replica_offsets:
        tt1 = rr1 - offset[0]
        tt2 = rr2 - offset[1]
        tt3 = rr3 - offset[2]
        d_sq = (
            adot[0, 0] * tt1**2 + adot[1, 1] * tt2**2 + adot[2, 2] * tt3**2
            + 2.0 * adot[0, 1] * tt1 * tt2
            + 2.0 * adot[0, 2] * tt1 * tt3
            + 2.0 * adot[1, 2] * tt2 * tt3
        )
        r_len_sq = np.minimum(r_len_sq, d_sq)

    fftbox_r = (scale / np.sqrt(r_len_sq)).astype(np.complex128)

    # --- FFT to G-space (unnormalized, matching BGW convention) ---
    fftbox_G = np.fft.fftn(fftbox_r)

    # --- Extract onto WFN FFT grid with phase correction ---
    # The dense grid has dNfft points; the WFN grid has fft_n* points.
    # For each G in the WFN grid (range [-N/2, N/2-1]), map to the dense
    # grid index and apply phase to undo the trunc_shift.
    vcoul_grid = np.zeros((fft_nx, fft_ny, fft_nz), dtype=np.float64)

    for j1_idx in range(fft_nx):
        j1 = j1_idx if j1_idx <= fft_nx // 2 else j1_idx - fft_nx
        di1 = j1 if j1 >= 0 else dNfft[0] + j1
        for j2_idx in range(fft_ny):
            j2 = j2_idx if j2_idx <= fft_ny // 2 else j2_idx - fft_ny
            di2 = j2 if j2 >= 0 else dNfft[1] + j2
            for j3_idx in range(fft_nz):
                j3 = j3_idx if j3_idx <= fft_nz // 2 else j3_idx - fft_nz
                di3 = j3 if j3 >= 0 else dNfft[2] + j3

                phase = 2.0 * np.pi * (
                    j1 * TRUNC_SHIFT / dNfft[0]
                    + j2 * TRUNC_SHIFT / dNfft[1]
                    + j3 * TRUNC_SHIFT / dNfft[2]
                )
                vtemp = fftbox_G[di1, di2, di3] * complex(np.cos(phase), -np.sin(phase))
                vcoul_grid[j1_idx, j2_idx, j3_idx] = vtemp.real

    # vcoul_grid is v(G) in Rydberg (same units as 8π/|G|² for untruncated).
    # The downstream code expects sqrt(v(G) / cell_volume).
    fact = 1.0 / cell_volume
    v_scaled = vcoul_grid * fact
    sqrt_v = np.where(v_scaled > 0.0, np.sqrt(v_scaled), 0.0)
    return jnp.asarray(sqrt_v, dtype=jnp.complex128)


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
    mesh_xy: Mesh,
    *,
    sys_dim: int = 2,
    bdot: np.ndarray | None = None,
    mc_average_vcoul_body: bool = True,
    vcoul_cutoff_ry: float | None = None,
):
    """
    Factory for jitted kernels that compute V_q blocks from zeta chunks.

    This creates two kernels:

    vcoul_cutoff_ry: If set, zero v(q+G) for |q+G|² > cutoff (in Ry).
        Use to match BGW's bare_coulomb_cutoff (default: ecutwfc).
    1. fft_and_weight: zeta_r(B_μ, n_rtot) → zeta_weighted(B_μ, n_G)
    2. contract_block: (zeta_μ, zeta_ν) → V_block(B_μ, B_ν)

    Args:
        fft_nx, fft_ny, fft_nz: FFT grid dimensions
        nkx, nky, nkz: k-grid dimensions
        bvec: Reciprocal lattice vectors (3×3)
        cell_volume: Unit cell volume
        sys_dim: System dimensionality (0=molecule/box, 2=slab)
        bdot: Reciprocal metric (3×3) in Bohr^-2; required for sys_dim=0

    Returns:
        Namespace with fft_and_weight, contract_block, get_sqrt_v, get_phase kernels
    """
    if sys_dim not in (0, 2, 3):
        raise NotImplementedError(f"Chunked V_q supports sys_dim=0 (box), 2 (slab), or 3 (bulk), got {sys_dim}")

    cache_key = (fft_nx, fft_ny, fft_nz, nkx, nky, nkz, tuple(bvec.flatten()), cell_volume, sys_dim)
    if cache_key in _v_munu_kernel_cache:
        return _v_munu_kernel_cache[cache_key]

    n_G = fft_nx * fft_ny * fft_nz

    # Local 3-D FFT helper for the ``(B_μ, fft_nx, fft_ny, fft_nz)`` box.
    # ``make_sharded_fftn_3d`` runs cuFFT per-rank on the μ shard via
    # shard_map so XLA never sees the un-sharded box.  Callers pass a
    # 1×1 mesh in single-device contexts — same code path.
    _box_spec = P(('x', 'y'), None, None, None)
    _local_fftn = make_sharded_fftn_3d(
        mesh_xy, _box_spec, _box_spec, norm=None, axes=(-3, -2, -1))

    # Precompute static grid data
    fx, fy, fz = exp_ikr_fftbox(fft_nx, fft_ny, fft_nz)

    # V_q sphere gather: when ``vcoul_cutoff_ry`` is set we gather ζ̃ to a
    # single conservative sphere right after the FFT (full-box √v mask would
    # otherwise zero out ~(1-π/6) of entries in the contract).  Radius is
    # enlarged by ``|q_max|_cart`` so one q=0-centered sphere covers every
    # per-q ball {G : |q+G|² ≤ cutoff}: G outside it has |q+G| > √cutoff at
    # every q, so sqrt_v = 0 there and the full-box contract is recovered
    # exactly.  Single source of truth for the construction lives in
    # :func:`common.coulomb_sphere.compute_bare_coulomb_sphere_idx` —
    # the G-flat ζ writer (``common.isdf_fitting``) calls the same helper
    # so writer-side and consumer-side spheres match bit-for-bit.
    from common.coulomb_sphere import compute_bare_coulomb_sphere_idx
    _sphere_np, n_sph = compute_bare_coulomb_sphere_idx(
        fft_grid=(fft_nx, fft_ny, fft_nz),
        bvec=np.asarray(bvec, dtype=np.float64),
        kgrid=(nkx, nky, nkz),
        vcoul_cutoff_ry=vcoul_cutoff_ry,
        sys_dim=sys_dim,
    )
    sphere_idx = (jnp.asarray(_sphere_np, dtype=jnp.int32)
                   if _sphere_np is not None else None)

    if sys_dim == 0:
        # 0D cell box truncation: precompute √v on the full FFT grid via
        # real-space Wigner-Seitz truncation + FFT (no closed-form formula).
        # Only q=0 is valid; the result is a static array.
        if bdot is None:
            raise ValueError("bdot is required for sys_dim=0 (box truncation)")
        sqrt_v_0d = compute_sqrt_vcoul_0d(fft_nx, fft_ny, fft_nz, bdot, cell_volume)
        # Flatten to 1-D (n_rtot,) so the per-q vmap output is (Q, n_rtot),
        # matching the kernel's ``sqrt_v_batch[:, None, :]`` broadcast onto
        # ``zeta_G`` which is (Q, μ, n_rtot) when ``sphere_idx is None``.
        # Same convention as the 3-D sphere path, which calls
        # ``sqrt_v.reshape(-1)[sphere_idx]`` to land at 1-D pre-vmap.
        sqrt_v_0d_flat = sqrt_v_0d.reshape(-1)

        @jax.jit
        def get_sqrt_v_and_phase(qvec_wrapped: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Return precomputed √v(G) and trivial phase (q must be 0)."""
            # Phase is 1.0 for q=0
            phase = jnp.ones((1, fft_nx, fft_ny, fft_nz), dtype=jnp.complex128)
            return sqrt_v_0d_flat, phase
    else:
        # 2D slab or 3D bulk: analytic formula, computed per q-point
        gx, gy, gz = fft_integer_axes(fft_nx, fft_ny, fft_nz)
        gx_b, gy_b, gz_b = jnp.broadcast_arrays(gx, gy, gz)
        gstack = jnp.stack((gx_b, gy_b, gz_b), axis=-1)

        bvec_j = jnp.asarray(bvec, dtype=jnp.float64)
        fact = jnp.float64(1.0 / cell_volume)
        if sys_dim == 2:
            zc = jnp.float64(np.pi / float(bvec[2, 2]))
        G_cart_base = jnp.einsum('...a,ab->...b', gstack, bvec_j, optimize=True)

    nkx_f = jnp.float64(nkx)
    nky_f = jnp.float64(nky)
    nkz_f = jnp.float64(nkz)

    if sys_dim == 3:
        # Precompute mini-BZ averaged v(q,G=0) for all q-points on the grid.
        # For 3D bulk, the bare Coulomb 8π/|q|² varies rapidly near Gamma,
        # so the point value at q differs from the average over the mini-BZ
        # Voronoi cell. BGW handles this via MC averaging at every q-point
        # (vcoul_generator with avgcut=∞ for 3D semiconductors).
        # We replace v(q, G=0) with <v(q+δq, G=0)>_miniBZ for all q≠0.
        from .vcoul import wrap_points_to_voronoi
        _nmc = 2**18
        _rng = np.random.RandomState(42)
        _randvals = _rng.uniform(0, 1, (_nmc, 3))
        _randcart = (_randvals @ bvec.T)
        _wrapped = np.asarray(wrap_points_to_voronoi(
            jnp.asarray(_randcart), bvec_j, nmax=1))
        _kgrid_arr = np.array([nkx, nky, nkz], dtype=np.float64)
        _randlims = bvec.T @ (np.diag(1.0 / _kgrid_arr) @ np.linalg.inv(bvec.T))
        _dq_cart = (_randlims @ _wrapped.T).T  # (nmc, 3) mini-BZ offsets in Cartesian

        _v_head_avg = np.zeros((nkx, nky, nkz), dtype=np.float64)
        for qx in range(nkx):
            for qy in range(nky):
                for qz in range(nkz):
                    qw = np.array([qx, qy, qz], dtype=np.float64)
                    qw = np.where(qw > _kgrid_arr / 2, qw - _kgrid_arr, qw)
                    q_frac = qw / _kgrid_arr
                    q_cart = q_frac @ bvec
                    if np.dot(q_cart, q_cart) < 1e-12:
                        _v_head_avg[qx, qy, qz] = 0.0  # q=0 head handled separately
                    else:
                        shifted = q_cart[None, :] + _dq_cart  # (nmc, 3)
                        denom = np.sum(shifted**2, axis=1)
                        _v_head_avg[qx, qy, qz] = np.mean(8.0 * np.pi / denom)
        _v_head_avg_j = jnp.asarray(_v_head_avg * (1.0 / cell_volume))

        @jax.jit
        def get_sqrt_v_and_phase(qvec_wrapped: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Compute √v(q+G) and phase for 3D bulk (untruncated Coulomb).

            For G≠0: v = 8π/|q+G|² (point value).
            For G=0, q≠0: v = <8π/|q+δq|²>_miniBZ (MC averaged over Voronoi cell).
            For G=0, q=0: v = 0 (head injected separately via rank-1 correction).
            """
            # Phase factor
            phase = jnp.exp(-2j * jnp.pi * (
                qvec_wrapped[0] / nkx_f * fx +
                qvec_wrapped[1] / nky_f * fy +
                qvec_wrapped[2] / nkz_f * fz
            ))

            # Body: v(q+G) = 8π/|q+G|² for all G
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

            v_reg = 8.0 * jnp.pi / denom_safe  # Rydberg units
            v_scaled = jnp.where(denom_zero, 0.0, v_reg * fact)

            # Replace G=0 with mini-BZ averaged value for q≠0
            if mc_average_vcoul_body:
                qx_idx = jnp.round(qvec_wrapped[0]).astype(jnp.int32) % nkx
                qy_idx = jnp.round(qvec_wrapped[1]).astype(jnp.int32) % nky
                qz_idx = jnp.round(qvec_wrapped[2]).astype(jnp.int32) % nkz
                v_head_mc = _v_head_avg_j[qx_idx, qy_idx, qz_idx]
                v_scaled = v_scaled.at[0, 0, 0].set(v_head_mc)

            # Optional G-vector cutoff (match BGW bare_coulomb_cutoff)
            if vcoul_cutoff_ry is not None:
                v_scaled = jnp.where(denom > vcoul_cutoff_ry, 0.0, v_scaled)

            sqrt_v = jnp.where(v_scaled > 0.0, jnp.sqrt(v_scaled), 0.0).astype(jnp.complex128)
            if sphere_idx is not None:
                sqrt_v = sqrt_v.reshape(-1)[sphere_idx]  # 1-D for fft_and_weight sphere path
            return sqrt_v, phase

    elif sys_dim == 2:
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
            # Optional G-vector cutoff (match BGW bare_coulomb_cutoff)
            if vcoul_cutoff_ry is not None:
                v_scaled = jnp.where(denom > vcoul_cutoff_ry, 0.0, v_scaled)
            sqrt_v = jnp.where(v_scaled > 0.0, jnp.sqrt(v_scaled), 0.0).astype(jnp.complex128)
            if sphere_idx is not None:
                sqrt_v = sqrt_v.reshape(-1)[sphere_idx]  # 1-D for fft_and_weight sphere path
            return sqrt_v, phase

    # NOTE: These are NOT JIT'd - they're meant to be called from an outer JIT
    # to avoid nested JIT compilation overhead. The outer JIT (_batch_proc or 
    # the chunked loop) compiles everything together.
    
    def fft_and_weight_inner(
        zeta_r: jax.Array,
        sqrt_v: jax.Array,
        phase: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """FFT zeta and weight by √v.  Output trailing axis is ``n_G`` when
        ``sphere_idx is None`` else ``n_sph``.  NOT JIT'd - outer JIT fuses."""
        B_mu = zeta_r.shape[0]
        zeta_G_flat = _local_fftn(
            zeta_r.reshape(B_mu, fft_nx, fft_ny, fft_nz) * phase
        ).reshape(B_mu, n_G)
        g0_chunk = zeta_G_flat[:, 0]  # G=(0,0,0) is flat-index 0; sphere_idx[0]==0.
        if sphere_idx is None:
            return zeta_G_flat * sqrt_v.reshape(1, -1), g0_chunk
        return jnp.take(zeta_G_flat, sphere_idx, axis=-1) * sqrt_v[None, :], g0_chunk
    
    def contract_block_inner(
        zeta_mu: jax.Array,
        zeta_nu: jax.Array,
    ) -> jax.Array:
        """
        Contract two weighted zeta chunks: V[μ,ν] = Σ_G ζ̃*_μ(G) ζ̃_ν(G)
        NOT JIT'd - call from within an outer JIT.
        """
        return jnp.einsum('mG,nG->mn', jnp.conj(zeta_mu), zeta_nu, optimize=True)
    
    # JIT'd versions for standalone use (chunked path)
    @partial(jax.jit, static_argnums=(3,))
    def fft_and_weight(zeta_r, sqrt_v, phase, B_mu: int):
        """JIT'd wrapper for standalone use."""
        return fft_and_weight_inner(zeta_r, sqrt_v, phase)

    @jax.jit
    def contract_block(zeta_mu, zeta_nu):
        """JIT'd wrapper for standalone use."""
        return contract_block_inner(zeta_mu, zeta_nu)

    @jax.jit
    def fft_weight_contract(zeta_mu_r, zeta_nu_r, sqrt_v, phase):
        """Fused FFT + weight + contraction for a (mu, nu) block pair.
        XLA compiles as one kernel — no sync between FFT and contract."""
        zeta_mu_w, g0_mu = fft_and_weight_inner(zeta_mu_r, sqrt_v, phase)
        zeta_nu_w, g0_nu = fft_and_weight_inner(zeta_nu_r, sqrt_v, phase)
        V_ij = contract_block_inner(zeta_mu_w, zeta_nu_w)
        return V_ij, g0_mu, g0_nu

    @jax.jit
    def fft_weight_contract_diag(zeta_mu_r, sqrt_v, phase):
        """Fused FFT + weight + self-contraction for a diagonal block.
        Single FFT, single contraction — minimal memory."""
        zeta_mu_w, g0_mu = fft_and_weight_inner(zeta_mu_r, sqrt_v, phase)
        V_ii = contract_block_inner(zeta_mu_w, zeta_mu_w)
        return V_ii, g0_mu

    @jax.jit
    def fft_weight_contract_offdiag(zeta_nu_r, zeta_mu_weighted, sqrt_v, phase):
        """Fused FFT(nu) + contraction with pre-weighted mu. Avoids re-FFT of mu."""
        zeta_nu_w, _ = fft_and_weight_inner(zeta_nu_r, sqrt_v, phase)
        V_ij = contract_block_inner(zeta_mu_weighted, zeta_nu_w)
        return V_ij

    @partial(jax.jit, static_argnums=(3,))
    def fft_and_weight_keep(zeta_r, sqrt_v, phase, B_mu: int):
        """FFT + weight, returning the weighted zeta for reuse in off-diagonal blocks."""
        return fft_and_weight_inner(zeta_r, sqrt_v, phase)

    # Bundle kernels
    from types import SimpleNamespace
    kernels = SimpleNamespace(
        get_sqrt_v_and_phase=get_sqrt_v_and_phase,
        fft_and_weight=fft_and_weight,  # JIT'd for standalone/chunked use
        fft_and_weight_inner=fft_and_weight_inner,  # non-JIT'd for nested use
        contract_block=contract_block,  # JIT'd for standalone/chunked use
        contract_block_inner=contract_block_inner,  # non-JIT'd for nested use
        fft_weight_contract=fft_weight_contract,  # fused off-diagonal
        fft_weight_contract_diag=fft_weight_contract_diag,  # fused diagonal
        fft_weight_contract_offdiag=fft_weight_contract_offdiag,  # fused with pre-weighted mu
        fft_and_weight_keep=fft_and_weight_keep,  # FFT+weight, keep weighted for reuse
        n_G=n_G,
        n_sph=n_sph,            # == n_G when sphere inactive
        sphere_idx=sphere_idx,  # jnp int32 (n_sph,) | None
        fft_shape=(fft_nx, fft_ny, fft_nz),
    )
    
    _v_munu_kernel_cache[cache_key] = kernels
    return kernels



# ============================================================================
# V_q dispatcher — thin wrapper around the unified V_q tile driver
# ============================================================================
#
# The actual V_q tile logic lives in :mod:`gw.v_q_tile`.  This file now only
# provides:
#
#   * ``make_v_munu_chunked_kernel``  — Coulomb factory: sphere_idx, sqrt_v(q+G)
#                                       and per-q phase callable.  Used both
#                                       here (for ``compute_all_V_q``) and by
#                                       any caller that wants v(q+G) on its
#                                       own G mesh (e.g. minimax_screening).
#   * ``compute_all_V_q``            — thin wrapper that builds the v_per_G_fn
#                                       and phase_fn closures from the
#                                       ``make_v_munu_chunked_kernel`` bundle
#                                       and dispatches to
#                                       :func:`gw.v_q_tile.compute_V_q_tile`.
#   * ``_choose_v_q_chunks``         — re-exported from ``v_q_tile`` for
#                                       backward compatibility.
#
# The two old kernels (``_make_V_q_caseA_kernel`` /
# ``_make_V_q_caseB_kernel``) and their two outer drivers
# (``_compute_all_V_q_sharded`` / ``_compute_all_V_q_replicated``) are
# replaced by a single unified kernel + driver in :mod:`gw.v_q_tile`.

from .v_q_tile import (
    compute_V_q_tile as _compute_V_q_tile,
    _choose_v_q_chunks,
    _unfold_g0_ibz_to_full,
)
from common.symmetry_maps import unfold_v_q


# ============================================================================
# Per-q, G-chunked V_q kernel (consumes G-flat zeta on the per-q sphere)
# ============================================================================
#
# This is the new V_q contraction:
#
#     V_q[μ,ν] = Σ_G  conj(ζ̃_q,μ(G)) · v(q+G) · ζ̃_q,ν(G)
#
# computed *one q at a time* with the G-axis chunked into Gchunk-sized
# slices.  Compared with the legacy ``fft_and_weight → contract_block``
# path (compute_vcoul.py:419-454) it has three properties the user
# asked for:
#
# 1. No FFT-and-gather inside V_q.  ζ̃ comes in already on the per-q
#    WFN.h5-style sphere produced by ``isdf_fitting.fit_zeta_to_h5``
#    in G-flat mode — the writer's per-q ``ngk[q]`` G-list is the
#    contract domain, padded uniformly to ``ngkmax`` with ζ̃ = 0 at
#    pad slots (so the pad contribution to V_q is exactly zero).
# 2. Contiguous G access.  Each Gchunk contracts via a single
#    GEMM-shaped einsum ``'mG,nG->mn'`` on small (n_rmu × G_chunk)
#    blocks — far better cache behaviour than the legacy whole-sphere
#    contract at once.
# 3. Independent of the shared-sphere convention.  No ``sphere_idx``
#    plumbing; the per-q G-list IS the sphere, and v(q+G) is evaluated
#    at those Miller indices.
#
# Sharding: ``zeta_q_L`` / ``zeta_q_R`` arrive sharded along μ
# (P('x', None) or product P(('x','y'), None)); ``v_q`` is replicated;
# the output ``V`` is sharded P('x', 'y').  XLA collapses the per-chunk
# einsum into one reduction per Gchunk; the trailing accumulator add
# is in-place under jit (donated).
#
# Outer q loop (and the disk reads producing ζ_q from
# ``ZetaLoader.load(layout='G_flat')``) live in the driver — this
# kernel is one q.  Q-batching could be re-introduced in a follow-up;
# the comment in the loop marks the natural seam.


@partial(jax.jit, donate_argnums=(0,), static_argnums=(4,))
def _v_q_per_q_g_chunked_jit(
    V_acc: jax.Array,            # (n_rmu_L, n_rmu_R) c128 — donated
    zeta_q_L: jax.Array,         # (n_rmu_L, ngkmax)  c128
    zeta_q_R: jax.Array,         # (n_rmu_R, ngkmax)  c128
    v_q: jax.Array,              # (ngkmax,) c128
    g_chunk: int,                # static
) -> jax.Array:
    """Inner kernel: V += conj(ζ_L) · v · ζ_Rᵀ, G-chunked.

    See module-level docstring above for the full math + memory model.
    """
    ngkmax = int(zeta_q_L.shape[-1])

    def body(start, V):
        # jax.lax.dynamic_slice keeps a single compile shape regardless
        # of where the chunk lands; the trailing chunk is sized to fit
        # by the outer python loop (no dynamic chunk size).
        L_chunk = jax.lax.dynamic_slice_in_dim(
            zeta_q_L, start, g_chunk, axis=-1)        # (n_rmu_L, g_chunk)
        R_chunk = jax.lax.dynamic_slice_in_dim(
            zeta_q_R, start, g_chunk, axis=-1)        # (n_rmu_R, g_chunk)
        v_chunk = jax.lax.dynamic_slice_in_dim(
            v_q, start, g_chunk, axis=0)              # (g_chunk,)
        # Pre-multiply v into L (real ≥ 0 for bare Coulomb; signed /
        # complex for bispinor transverse — both fine).
        L_weighted = jnp.conj(L_chunk) * v_chunk[None, :]
        return V + L_weighted @ R_chunk.T            # (n_rmu_L, n_rmu_R)

    # Whole-Gchunk strides (the python caller pads ngkmax to a multiple
    # of g_chunk via zeta's writer-side pad slots ζ=0, so no remainder
    # branch is needed inside the jit).
    n_chunks = ngkmax // g_chunk
    V = V_acc
    for i in range(n_chunks):
        V = body(i * g_chunk, V)
    return V


def compute_v_q_per_q_g_chunked(
    zeta_q_L: jax.Array,
    zeta_q_R: jax.Array,
    v_q: jax.Array,
    *,
    g_chunk: int = 4096,
    V_acc: jax.Array | None = None,
) -> jax.Array:
    """V_q[μν] = Σ_G ζ̃*_μ(G) v(G) ζ̃_ν(G) for a single q, G-chunked.

    Parameters
    ----------
    zeta_q_L : ``(n_rmu_L, ngkmax)`` c128
        ζ̃ on the per-q WFN.h5-style sphere (writer's G-flat layout).
        Pad slots ``j ≥ ngk[q]`` carry ζ̃ = 0 — they contribute zero
        to the contract regardless of what ``v_q`` does there.
    zeta_q_R : ``(n_rmu_R, ngkmax)`` c128
        Right operand.  Pass the same array as ``zeta_q_L`` for the
        diagonal V^{0,0} case (single-ζ); pass a separate buffer for
        bispinor off-diagonal tiles V^{μ_L, ν_L}.
    v_q : ``(ngkmax,)`` c128
        Per-G Coulomb weight.  Bare Coulomb is real ≥ 0; bispinor
        transverse projector tiles are signed / complex.  Pad slots
        are don't-care (ζ̃ = 0 there).
    g_chunk : int
        Number of G's per inner contraction.  Caller pads ``ngkmax``
        up to a multiple of ``g_chunk`` (typical: pad with the
        WFN.h5 sentinel slot, ζ̃ = 0).  Memory-wise: this kernel's
        peak intermediate is ``n_rmu_L · g_chunk`` complex128 plus
        the (n_rmu_L, n_rmu_R) accumulator.  TODO(q-batch): an outer
        ``vmap`` over a small q-chunk would amortize the kernel's
        launch overhead; currently the driver runs one q at a time.
    V_acc : ``(n_rmu_L, n_rmu_R)`` c128 or None
        Donated accumulator.  Pass ``None`` to allocate a fresh
        zero-init buffer matching the sharding of ``zeta_q_L`` /
        ``zeta_q_R`` (P('x', 'y') if both are P(('x','y'), None)).

    Returns
    -------
    V : ``(n_rmu_L, n_rmu_R)`` c128
        ``V_acc + Σ_G  conj(ζ̃_L(G)) · v(G) · ζ̃_R(G)``.

    Sharding & dtype notes
    ----------------------
    All inputs are c128.  v_q can be passed as float64 and is cast
    by the caller before reaching this function.  No internal
    sharding constraints — the jit inherits the caller's input
    shardings and emits the matching output sharding.
    """
    n_rmu_L = int(zeta_q_L.shape[0])
    n_rmu_R = int(zeta_q_R.shape[0])
    ngkmax = int(zeta_q_L.shape[-1])
    if int(zeta_q_R.shape[-1]) != ngkmax:
        raise ValueError(
            f"compute_v_q_per_q_g_chunked: zeta_q_L.ngkmax={ngkmax} ≠ "
            f"zeta_q_R.ngkmax={int(zeta_q_R.shape[-1])}")
    if int(v_q.shape[0]) != ngkmax:
        raise ValueError(
            f"compute_v_q_per_q_g_chunked: v_q.ngkmax={int(v_q.shape[0])} "
            f"≠ ζ.ngkmax={ngkmax}")
    g_chunk = int(g_chunk)
    if ngkmax % g_chunk != 0:
        raise ValueError(
            f"compute_v_q_per_q_g_chunked: ngkmax ({ngkmax}) must be a "
            f"multiple of g_chunk ({g_chunk}); pad the trailing slots "
            f"with ζ̃ = 0 to satisfy this (the writer's per-q sentinel "
            f"pad already does this when ngkmax is chosen by the "
            f"caller as a multiple of g_chunk).")

    if V_acc is None:
        V_acc = jnp.zeros(
            (n_rmu_L, n_rmu_R), dtype=zeta_q_L.dtype)
    if v_q.dtype != zeta_q_L.dtype:
        v_q = v_q.astype(zeta_q_L.dtype)

    return _v_q_per_q_g_chunked_jit(V_acc, zeta_q_L, zeta_q_R, v_q, g_chunk)


def compute_v_q_per_G(
    q_irr_frac: np.ndarray,
    gvec_components: np.ndarray,
    *,
    bvec: np.ndarray,
    cell_volume: float,
    sys_dim: int,
    vcoul_cutoff_ry: float | None = None,
    bdot: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``v(q+G)`` at the per-q WFN.h5-style G-list.

    Mirrors the formula inside ``make_v_munu_chunked_kernel.get_sqrt_v_and_phase``
    but operates on a per-q ``gvec_components`` table (instead of the
    full FFT grid) — the writer's ``isdf_header/gvec_components`` is
    exactly the input the consumer needs.  Returns one ``(ngkmax,)``
    row of ``v(q+G)`` per q in ``q_irr_frac``.

    Pad slots in ``gvec_components`` (sentinel Miller index
    ``(-nx/2, -ny/2, -nz/2)``) get whatever ``v`` is at that
    position — caller need not zero them because the contract uses
    ζ̃ = 0 at those slots.

    Parameters
    ----------
    q_irr_frac : ``(n_q, 3)`` float64
        Fractional q-vectors in BGW-wrap convention (already divided
        by kgrid).
    gvec_components : ``(n_q, 3, ngkmax)`` int32
        Per-q Miller indices from ``isdf_header.gvec_components``.
    bvec, cell_volume, sys_dim, vcoul_cutoff_ry, bdot
        Same conventions as ``make_v_munu_chunked_kernel``.
        ``vcoul_cutoff_ry`` zeroes ``v`` at G's with |q+G|² past
        the cutoff (== V_q's bare-Coulomb cutoff; may be < the
        ζ-sphere cutoff that built ``gvec_components``).

    Returns
    -------
    v_q_per_G : ``(n_q, ngkmax)`` float64
        ``v(q+G)`` evaluated at every (q, G) in the components table.

    Notes
    -----
    This is a *host-side* helper — the per-q ``v(q+G)`` is built once
    at consumer setup and pushed to device.  Not jitted; not sharded.
    For very large ngkmax this could be vectorised across q on device,
    but it's a one-shot cost per V_q run.
    """
    if sys_dim not in (0, 2, 3):
        raise NotImplementedError(
            f"compute_v_q_per_G: sys_dim must be 0 / 2 / 3; got {sys_dim}")
    q_irr_frac = np.asarray(q_irr_frac, dtype=np.float64).reshape(-1, 3)
    gvec = np.asarray(gvec_components, dtype=np.float64)         # (n_q, 3, ngkmax)
    if gvec.ndim != 3 or gvec.shape[1] != 3:
        raise ValueError(
            f"gvec_components must be (n_q, 3, ngkmax); got {gvec.shape}")
    n_q, _, ngkmax = gvec.shape
    bvec_f = np.asarray(bvec, dtype=np.float64)
    fact = 1.0 / float(cell_volume)

    if sys_dim == 2:
        zc = float(np.pi / float(bvec_f[2, 2]))
    out = np.zeros((n_q, ngkmax), dtype=np.float64)

    for qi in range(n_q):
        qf = q_irr_frac[qi]
        # gvec[qi]: (3, ngkmax) -> per-G Miller; (q + G) in fractional.
        qG_frac = qf[:, None] + gvec[qi]                          # (3, ngkmax)
        qG_cart = bvec_f.T @ qG_frac                              # (3, ngkmax)
        denom = np.sum(qG_cart * qG_cart, axis=0)                 # (ngkmax,)
        denom_zero = denom < 1e-12
        denom_safe = np.where(denom_zero, 1.0, denom)
        if sys_dim == 3:
            v_reg = 8.0 * np.pi / denom_safe
            v = np.where(denom_zero, 0.0, v_reg * fact)
        elif sys_dim == 2:
            kxy = np.sqrt(qG_cart[0]**2 + qG_cart[1]**2)
            kz = qG_cart[2]
            f2d = 1.0 - np.exp(-zc * kxy) * np.cos(kz * zc)
            v_reg = (8.0 * np.pi / denom_safe) * f2d
            v = np.where(denom_zero, 0.0, v_reg * fact)
        else:
            # sys_dim == 0: caller passes ``bdot`` and we'd build the
            # FFT-grid sqrt_v0d here; not yet wired to per-q lookup.
            raise NotImplementedError(
                "compute_v_q_per_G: sys_dim=0 path not wired — the 0-D "
                "box truncation builds v on the full FFT grid via "
                "compute_sqrt_vcoul_0d; the per-q gather would map "
                "components → flat-FFT index → v(G).  Plumb when "
                "needed.")
        if vcoul_cutoff_ry is not None:
            v = np.where(denom > float(vcoul_cutoff_ry), 0.0, v)
        out[qi] = v
    return out


def compute_all_V_q(
    zeta_io,
    *,
    kgrid: tuple[int, int, int],
    fft_grid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    mesh_xy: Mesh,
    n_rmu: int,
    n_rtot: int,
    sys_dim: int = 2,
    bdot: np.ndarray | None = None,
    mc_average_vcoul_body: bool = True,
    bare_coulomb_cutoff: float | None = None,
    bgw_v_grid_fn=None,
    mu_chunk_size: int | None = None,   # legacy arg (allgather path); ignored
    q_batch_size: int | None = None,    # legacy arg (allgather path); ignored
    budget_bytes: float | None = None,
    verbose: bool = True,
    sym=None,
    centroid_indices: np.ndarray | None = None,
    use_g_flat_zeta: bool = False,
    g_chunk_size: int = 0,              # 0 = auto _pick_g_chunk(ngkmax)
) -> tuple[jax.Array, jax.Array]:
    """Compute V_qmunu(q,μ,ν) and g0_μ(q) at q=0 from a sharded ζ HDF5.

    Dispatcher with two paths:

    * On-disk ``ζ`` in **G-flat** layout (the only thing
      :func:`fit_zeta_to_h5` writes): routes to
      :func:`gw.v_q_g_flat.compute_all_V_q_g_flat` — per-q, G-chunked
      contract on the writer's per-q WFN.h5-style sphere.  No FFT,
      no shared-sphere conversion, no μ × ν tiling; the chooser
      collapses to "pick g_chunk".

    * On-disk ``ζ`` in **r-space** layout: routes to
      :func:`gw.v_q_tile.compute_V_q_tile` — the older driver that
      handles FFT + sphere gather + μ × ν tiling inline.  Reachable
      only for legacy r-space ζ files; not exercised by the current
      writer.

    Parameters mirror the old API; ``mu_chunk_size`` / ``q_batch_size``
    are kept for back-compat with r-space callers but ignored on the
    G-flat path (where the chooser is essentially trivial).
    """
    # G-flat dispatch — when the loader carries the new per-q sphere
    # components, hand off to the rewritten orchestrator.  Async
    # prefetch defaults to OFF here: the prefetcher's collective read
    # on the PHDF5 FFI backend has to interleave correctly with the
    # per-q kernel's NCCL collectives, and a simple sync loop is
    # easier to debug.  Re-enable via env once the kernel is profile-
    # validated.
    if getattr(zeta_io, 'zeta_layout', None) == 'G_flat':
        from .v_q_g_flat import compute_all_V_q_g_flat
        _async = bool(int(os.environ.get(
            'LORRAX_V_Q_G_FLAT_ASYNC_PREFETCH', '0')))
        return compute_all_V_q_g_flat(
            zeta_io,
            kgrid=kgrid, fft_grid=fft_grid,
            bvec=bvec, cell_volume=cell_volume,
            mesh_xy=mesh_xy, n_rmu=n_rmu,
            sys_dim=sys_dim, bdot=bdot,
            bare_coulomb_cutoff_ry=bare_coulomb_cutoff,
            bgw_v_grid_fn=bgw_v_grid_fn,
            g_chunk=(int(g_chunk_size) if g_chunk_size > 0 else None),
            verbose=verbose, sym=sym,
            centroid_indices=centroid_indices,
            async_prefetch=_async,
        )

    nkx, nky, nkz = kgrid
    nq_full = nkx * nky * nkz

    # IBZ orchestration: when ``sym`` is provided AND the centroid set
    # is orbit-closed under the WFN sym group, compute V_q only at IBZ
    # q's and unfold to full BZ via the centroid-double-permute (no
    # τ-phase — V_q is bilinear in ζ).  Without ``sym`` (or when
    # closure fails), fall back to full-BZ iteration so legacy callers
    # and centroid files produced with ``kmeans_cli --no-orbit`` keep
    # working.
    use_ibz = False
    q_irr_kgrid_int = None
    q_full_to_irr_idx = None
    q_full_to_irr_sym = None
    sym_perm = None
    L_table = None
    nq_total = nq_full
    q_list_for_tile = None

    if sym is not None and centroid_indices is not None:
        from centroid.orbit_syms import compute_centroid_sym_perm
        n_tran = int(np.asarray(sym.sym_matrices).shape[0])
        centroid_idx_np = np.asarray(centroid_indices, dtype=np.int32)
        try:
            # ``extend_trs=True`` so ``sym_perm`` has shape
            # ``(2·n_tran, n_rmu)`` — the second half duplicates the
            # spatial rows and is indexed by the TRS-augmented sym
            # values returned by ``sym.irr_idx_q/sym_idx_q``.  See
            # ``compute_centroid_sym_perm`` docstring and the audit
            # report (``reports/trs_sym_audit_2026-05-14``).
            sym_perm, L_table = compute_centroid_sym_perm(
                centroid_idx_np,
                sym_matrices=np.asarray(sym.sym_matrices[:n_tran]),
                translations=np.asarray(sym.translations[:n_tran]),
                fft_grid=np.asarray(fft_grid, dtype=np.int32),
                extend_trs=True,
            )
        except RuntimeError as exc:
            if verbose and jax.process_index() == 0:
                print(f"  V_q tile: centroid orbit closure failed — "
                      f"falling back to full-BZ iteration.  Reason: "
                      f"{exc.args[0].splitlines()[0] if exc.args else exc}")
            sym_perm = None
            L_table = None

        if sym_perm is not None:
            q_irr_kgrid_int = sym.q_irr_kgrid_int
            q_full_to_irr_idx = sym.irr_idx_q
            q_full_to_irr_sym = sym.sym_idx_q
            nq_total = int(q_irr_kgrid_int.shape[0])
            q_list_for_tile = q_irr_kgrid_int
            use_ibz = True
            if verbose and jax.process_index() == 0:
                print(f"  V_q tile: iterating IBZ q ({nq_total} / "
                      f"{nq_full} full-BZ); post-loop unfold via "
                      f"centroid perm.")

    # Build sphere_idx + per-q (sqrt_v, phase) callable from the Coulomb
    # factory; this is the only piece that depends on Coulomb
    # dimensionality (3D/2D slab/0D box).
    kernels = make_v_munu_chunked_kernel(
        fft_grid[0], fft_grid[1], fft_grid[2], nkx, nky, nkz,
        bvec, cell_volume, mesh_xy,
        sys_dim=sys_dim, bdot=bdot,
        mc_average_vcoul_body=mc_average_vcoul_body,
        vcoul_cutoff_ry=bare_coulomb_cutoff,
    )
    sphere_idx = kernels.sphere_idx
    n_G_sph = int(kernels.n_sph)
    if sphere_idx is not None:
        _sphere_idx_np = np.asarray(sphere_idx)
    else:
        _sphere_idx_np = None

    _vmapped_sqrt_v_phase = jax.vmap(kernels.get_sqrt_v_and_phase)

    def v_per_G_fn(qvec_np_batch):
        """Pre-stack qvec batch → vmap(get_sqrt_v_and_phase) → (Q, n_G_sph) c128.

        Returns the FULL per-G weight ``v(q+G)`` (not √v) since the unified
        kernel applies it ONCE on the L side of the contraction:
            V[μ,ν] = Σ_G  conj(ζ_L(G)) · v(q+G) · ζ_R(G)
        For real, non-negative v(q+G) (V^{0,0}) this is mathematically
        identical to the old symmetric-√v form ``conj(√v ζ_L) · √v ζ_R``.
        """
        qvec_arr = jnp.asarray(qvec_np_batch, dtype=jnp.float64)
        sqrt_v_batch, _ = _vmapped_sqrt_v_phase(qvec_arr)
        # (Q, n_G_sph) c128 already: get_sqrt_v_and_phase reshape's via
        # sphere_idx, so no extra reshape needed here.  Square to recover v
        # (sqrt_v is non-negative real cast to c128).
        return sqrt_v_batch * sqrt_v_batch

    # ``phase_fn`` is gone — the V_q kernel now builds the per-q FFT-box
    # phase ``exp(-2πi q·r)`` separably inside ``_zeta_disk_to_G``/
    # ``zeta_reader._do_disk_to_G`` via
    # ``common.wfn_transforms.apply_bloch_phase`` (one source of truth).
    # The qvec batch is threaded through directly as a tiny (Q, 3)
    # array; no more 4D ``(Q, 1, nx, ny, nz)`` phase materialisation.

    # Optional BGW vcoul overlay — host-side replacement of native v(q+G)
    # with BGW's MC-averaged values for byte-reproducible comparison.
    bgw_overlay_fn = None
    if bgw_v_grid_fn is not None:
        kgrid_a = np.array([nkx, nky, nkz], dtype=np.float64)

        def bgw_overlay_fn(qvec_np_batch, v_per_G_native):  # noqa: F811
            v_per_G_np = np.asarray(v_per_G_native).copy()
            for iq in range(qvec_np_batch.shape[0]):
                q_frac = qvec_np_batch[iq] / kgrid_a
                v_scaled_bgw = np.asarray(
                    bgw_v_grid_fn(tuple(q_frac))).reshape(-1)
                if _sphere_idx_np is not None:
                    v_scaled_bgw = v_scaled_bgw[_sphere_idx_np]
                v_native_real = v_per_G_np[iq].real
                v_per_G_np[iq] = np.where(
                    v_scaled_bgw != 0.0, v_scaled_bgw, v_native_real
                ).astype(np.complex128)
            return jnp.asarray(v_per_G_np)

    # LORRAX_V_Q_MU_CHUNK env override — debug knob to force Case B at
    # caller-specified μ_chunk.
    chooser_choice = None
    _force_mu = int(os.environ.get('LORRAX_V_Q_MU_CHUNK', '0') or 0)
    if _force_mu > 0 and _force_mu < n_rmu:
        n_blocks = (n_rmu + _force_mu - 1) // _force_mu
        chooser_choice = dict(
            q_chunk=1, mu_chunk=_force_mu, n_mu_blocks=n_blocks,
            tiled=True, aligned=(n_rmu % _force_mu == 0),
            per_rank_peak=0.0, ref_bytes=0.0,
        )
        if jax.process_index() == 0:
            print(f"  [LORRAX_V_Q_MU_CHUNK] forcing Case B with "
                  f"μ_chunk={_force_mu}, n_mu_blocks={n_blocks}")

    if budget_bytes is None:
        budget_bytes = 24.0e9

    V_acc, g0_acc = _compute_V_q_tile(
        zeta_L_io=zeta_io,
        zeta_R_io=None,                  # same_zeta=True (V^{0,0})
        v_per_G_fn=v_per_G_fn,
        sphere_idx=sphere_idx,
        fft_grid=fft_grid,
        mesh_xy=mesh_xy,
        kgrid=kgrid,
        n_rmu_L=n_rmu,
        n_rmu_R=n_rmu,
        V_acc=None,                      # let driver allocate
        g0_acc=None,                     # let driver allocate (wants_g0=True)
        chooser_choice=chooser_choice,
        budget_bytes=budget_bytes,
        bgw_v_grid_overlay_fn=bgw_overlay_fn,
        verbose=verbose,
        timing_label="compute_all_V_q_sharded",
        q_list_kgrid_int=q_list_for_tile,
        use_g_flat_zeta=use_g_flat_zeta,
    )

    # IBZ orchestration: post-loop unfold V_q_ibz → V_q_full via the
    # centroid-double-permute (eq. 3 of the report).  g0 is unfolded
    # for completeness, but only the Γ slot is consumed downstream.
    if use_ibz:
        # ``sym_perm`` was built with ``extend_trs=True`` → shape
        # (2·ntran, n_rmu); the unfold helper applies a TRS-row conj
        # when ``full_to_irr_sym ≥ n_sym_spatial``, AND applies the
        # per-centroid umklapp phase from ``L_table``.
        n_sym_spatial = int(np.asarray(sym_perm).shape[0]) // 2
        # Need q_irr_frac (fractional reciprocal coords of IBZ q-list)
        # for the umklapp phase.  q_irr_kgrid_int is in kgrid-integer
        # units; wrap to (−0.5, 0.5] then divide by kgrid (same convention
        # used by v_q_g_flat._resolve_ibz_q_list).
        _kg_arr = np.asarray(sym.kgrid, dtype=np.float64)
        _q_int = np.asarray(q_irr_kgrid_int, dtype=np.float64)
        _q_wrap = np.where(_q_int > _kg_arr / 2, _q_int - _kg_arr, _q_int)
        q_irr_frac_for_phase = _q_wrap / _kg_arr
        V_acc = unfold_v_q(
            V_acc,
            irr_idx=q_full_to_irr_idx,
            sym_idx=q_full_to_irr_sym,
            sym_perm=sym_perm,
            L_table=L_table,
            q_irr_frac=q_irr_frac_for_phase,
            mesh_xy=mesh_xy,
            n_sym_spatial=n_sym_spatial,
        )
        g0_acc = _unfold_g0_ibz_to_full(
            g0_acc,
            full_to_irr_idx=q_full_to_irr_idx,
            full_to_irr_sym=q_full_to_irr_sym,
            sym_perm=sym_perm,
            mesh_xy=mesh_xy,
            n_sym_spatial=n_sym_spatial,
        )

    # Flat-q convention: keep the q axis 1-D throughout.  Downstream
    # callers that need the 3-D-k form reshape inside an FFT helper
    # (see ``common.fft_helpers.make_flat_k_fft``).  Sharding stays on
    # the trailing μ × μ axes.
    V_qmunu = jax.lax.with_sharding_constraint(
        V_acc, NamedSharding(mesh_xy, P(None, 'x', 'y')))
    g0_mu_all = jax.lax.with_sharding_constraint(
        g0_acc, NamedSharding(mesh_xy, P(None, 'x')))
    return V_qmunu, g0_mu_all
