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

import time

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from functools import partial

from common import timing


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

    # Precompute static grid data
    fx, fy, fz = exp_ikr_fftbox(fft_nx, fft_ny, fft_nz)

    # V_q sphere gather: when ``vcoul_cutoff_ry`` is set we gather ζ̃ to a
    # single conservative sphere right after the FFT (full-box √v mask would
    # otherwise zero out ~(1-π/6) of entries in the contract).  Radius is
    # enlarged by ``|q_max|_cart`` so one q=0-centered sphere covers every
    # per-q ball {G : |q+G|² ≤ cutoff}: G outside it has |q+G| > √cutoff at
    # every q, so sqrt_v = 0 there and the full-box contract is recovered
    # exactly.  Numpy construction (consistent with the FFT-grid helpers
    # above — avoid extra pjit compiles at factory time).
    if vcoul_cutoff_ry is not None and sys_dim in (2, 3):
        import itertools
        _corners_frac = (np.array(list(itertools.product([-0.5, 0.5], repeat=3)))
                         / np.array([nkx, nky, nkz], dtype=np.float64))
        _q_max_cart = float(np.max(np.linalg.norm(
            _corners_frac @ np.asarray(bvec, dtype=np.float64), axis=1)))
        _sphere_r2 = (float(np.sqrt(vcoul_cutoff_ry)) + _q_max_cart) ** 2
        _gx = (np.fft.fftfreq(fft_nx) * fft_nx).astype(np.float64)
        _gy = (np.fft.fftfreq(fft_ny) * fft_ny).astype(np.float64)
        _gz = (np.fft.fftfreq(fft_nz) * fft_nz).astype(np.float64)
        _Gc = np.stack(np.meshgrid(_gx, _gy, _gz, indexing='ij'), axis=-1) \
            @ np.asarray(bvec, dtype=np.float64)
        _sph_mask = np.sum(_Gc * _Gc, axis=-1).reshape(-1) <= _sphere_r2
        # G=(0,0,0) is flat-index 0 in fftfreq order; sphere_idx[0] must
        # equal 0 so ``g0_chunk = ζ̃_flat[:, 0]`` stays valid below.
        assert bool(_sph_mask[0])
        sphere_idx = jnp.asarray(np.nonzero(_sph_mask)[0].astype(np.int32))
        n_sph = int(sphere_idx.size)
    else:
        sphere_idx = None
        n_sph = n_G

    if sys_dim == 0:
        # 0D cell box truncation: precompute √v on the full FFT grid via
        # real-space Wigner-Seitz truncation + FFT (no closed-form formula).
        # Only q=0 is valid; the result is a static array.
        if bdot is None:
            raise ValueError("bdot is required for sys_dim=0 (box truncation)")
        sqrt_v_0d = compute_sqrt_vcoul_0d(fft_nx, fft_ny, fft_nz, bdot, cell_volume)

        @jax.jit
        def get_sqrt_v_and_phase(qvec_wrapped: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Return precomputed √v(G) and trivial phase (q must be 0)."""
            # Phase is 1.0 for q=0
            phase = jnp.ones((1, fft_nx, fft_ny, fft_nz), dtype=jnp.complex128)
            return sqrt_v_0d, phase
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
        zeta_G_flat = jnp.fft.fftn(
            zeta_r.reshape(B_mu, fft_nx, fft_ny, fft_nz) * phase,
            axes=(-3, -2, -1)).reshape(B_mu, n_G)
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
# Full V_q computation pipeline with all q-points
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
        from concurrent.futures import ThreadPoolExecutor
        
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
# Sharded V_q — mesh-parallel computation via one-axis gathers
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
#   --- free the μ_XY sharded post-multiply array after the two gathers
#       capture their own copies (XLA's aliasing/SSA gets to decide when
#       exactly but a Python `del` after the two jit calls is the hint). ---
#
#   → All-gather on Y   (separate jit, explicit in/out shardings):
#         P(None, ('x','y'), None)  →  P(None, 'x', None)
#     Result: ζ_q,μ_X(G), μ only X-sharded.  Per-rank bytes:
#         16 · N_q · (N_μ / P_x) · N_G
#
#   → All-gather on X   (separate jit, parallel-issued with the Y gather):
#         P(None, ('x','y'), None)  →  P(None, 'y', None)
#     Result: ζ_q,ν_Y(G), ν only Y-sharded.  Per-rank bytes equal to the
#         Y-gather when P_x = P_y.
#
#   → Contract  V_q[μ_X, ν_Y] = Σ_G conj(ζ_q,μ_X(G)) · ζ_q,ν_Y(G)
#     Einsum 'qmG,qnG->qmn' on (μ_X, G) × (ν_Y, G) → (μ_X, ν_Y).
#     Both inputs have the contraction axis (G) fully replicated and the
#     non-contraction axes disjointly sharded, so the product is local per
#     rank — no collective during the gemm.  Output lands natively in
#     P(None, 'x', 'y') — the exact layout downstream sigma_sx/sigma_coh
#     consume, so the chi0→W→V→sigma chain is reshard-free from here on.
#
#   → `del` both μ_X and μ_Y operands immediately after the gemm.
#
#   → phdf5 async write of V_q (or the full V_qmunu batch) — overlaps the
#     next q-chunk's ζ read + FFT.  Same pattern as fit_zeta_chunked_to_h5.
#
# Memory model (per-rank, bytes)
# ------------------------------
# Let P_min = min(P_x, P_y).  The dominant live set during compute is:
#
#     peak ≈ 3 · N_zeta · N_q · N_μ · N_G / P_min      (two gathered copies
#                                                       + one workspace buffer
#                                                       for the contract temp)
#           + N_zeta · N_q · N_G                       (sqrt_v(q+G) table,
#                                                       replicated per rank)
#           + N_zeta · N_q · N_μ²                      (output V_q, sharded
#                                                       P(x,y) so typically
#                                                       negligible per rank)
#
# With N_zeta = 16 and N_μ² / (P_x·P_y) small compared to the gathered μ·G
# tiles, the first term dominates.
#
# Chooser policy  (``_choose_v_q_chunks`` below)
# -----------------------------------------------
# The caller has a per-device budget ``B`` (bytes) minus space reserved for
# input buffers, I/O scratch, and the output accumulator.  Let
#
#     per_q_bytes = 3 · N_zeta · N_μ · N_G / P_min      (the "fits-one-q" cost)
#
# and ``max_zeta_elems`` the implied slack for the fit-sizing vocabulary the
# user uses: ``N_zeta-that-fits := (B − N_zeta·N_G) / (3 · N_zeta / P_min)``.
#
#   Case A — μ fits on one q (N_zeta_fit ≥ N_μ):
#       Q_max = floor((B − N_zeta·N_G · N_q_total) / per_q_bytes)
#       q_chunk = min(Q_max, N_q_total)
#       μ_chunk = N_μ  (no μ tiling needed)
#       ν_chunk = N_μ
#     The (N_zeta · N_G) sqrt_v table is reused for every q in the chunk
#     so an early-rank-dense-batch wins: batching many q at once amortises
#     sphere pick / phase compute / memory plan construction, but each extra
#     q adds per_q_bytes/N_q × N_q to the peak.  Cap Q_max by budget and by
#     ``N_q_total`` (the actual number of distinct q-points the caller has).
#
#   Case B — μ does NOT fit on one q (N_zeta_fit < N_μ):
#       q_chunk = 1  (single q per call, non-negotiable at this budget)
#       Then tile μ × ν into blocks (μ_chunk, ν_chunk) such that
#            3 · N_zeta · (μ_chunk + ν_chunk) · N_G / P_min ≤ B − N_zeta·N_G
#       Typically pick μ_chunk = ν_chunk for a square tile, giving
#            2 · N_zeta · μ_chunk · N_G / P_min  (only two μ-slabs live
#                                                 at once — the μ-X and
#                                                 ν-Y gathered copies —
#                                                 plus a small contract
#                                                 workspace).
#       The loop is
#            for (μ_i, ν_j) in diagonal-first order:
#                zeta_μ = read(μ_i ranges);  all-gather Y  →  μ_X-only
#                if (μ_i == ν_j):
#                    zeta_ν = zeta_μ       # same block; no second read
#                else:
#                    zeta_ν = read(ν_j ranges)
#                all-gather X  → μ_Y-only
#                V[μ_i, ν_j] = contract; write via async slab write
#       Reading the same block twice would be pointless; having the *option*
#       to read two blocks is gated purely on ``μ_i != ν_j`` at planning time,
#       i.e., known statically.
#
# Static-time knobs summary
# -------------------------
#     (q_chunk, μ_chunk, ν_chunk, read_ν_separately)
# all are Python-level ints / bools decided from (B, N_μ, N_G, N_q_total, P).
# This keeps the jit cache tight: two compile shapes maximum (full-sized and
# last-partial q-chunk; same story for μ-tile if Case B).


def _choose_v_q_chunks(
    *,
    n_rmu: int,
    n_G: int,
    n_q_total: int,
    budget_bytes: float,
    p_x: int,
    p_y: int,
) -> dict:
    """Pick (q_chunk, μ_chunk, ν_chunk) per the memory model above.

    Returns a dict with keys:
        q_chunk       : int — number of q per sharded-compute call
        mu_chunk      : int — μ block size (== n_rmu when no tile)
        nu_chunk      : int — ν block size (== n_rmu when no tile)
        tiled         : bool — True ⟺ Case B (μ×ν tiling)
        per_rank_peak : float — predicted per-rank peak bytes at this choice
    """
    N_zeta = 16.0  # c128
    p_min = float(min(int(p_x), int(p_y)))
    # sqrt_v(q+G) table lives replicated on each rank, size depends on q_chunk.
    # Since we want q_chunk static at trace time, pick q_chunk first assuming
    # the v-table scales linearly with it.

    # Case A check: can we hold *one* q's worth of gathered μ×G?
    one_q_bytes = 3.0 * N_zeta * n_rmu * n_G / p_min
    v_per_q_bytes = N_zeta * n_G
    slack_after_one_q = budget_bytes - (one_q_bytes + v_per_q_bytes)
    if slack_after_one_q < 0:
        # Case B — single q, tile μ×ν.
        # Two concurrent gathered slabs of (μ_chunk × N_G) plus a small
        # contract workspace; solve for μ_chunk with a 2× factor:
        #     2 · N_zeta · μ_chunk · N_G / p_min ≤ budget − v_per_q
        mu_chunk_max = max(
            1, int((budget_bytes - v_per_q_bytes) * p_min /
                   (2.0 * N_zeta * n_G)))
        mu_chunk = min(n_rmu, mu_chunk_max)
        # Snap to a μ divisor of p_x, p_y so each rank owns a whole number
        # of μ rows in the gathered layout.
        mu_chunk -= mu_chunk % (int(p_x) * int(p_y) or 1)
        mu_chunk = max(int(p_x) * int(p_y), mu_chunk)
        peak = 2.0 * N_zeta * mu_chunk * n_G / p_min + v_per_q_bytes
        return dict(
            q_chunk=1, mu_chunk=mu_chunk, nu_chunk=mu_chunk,
            tiled=True, per_rank_peak=peak,
        )

    # Case A — can fit at least one q with full μ.  Fit as many as possible.
    q_max = int((budget_bytes - v_per_q_bytes * max(1, n_q_total)) /
                one_q_bytes) if one_q_bytes > 0 else n_q_total
    q_chunk = max(1, min(n_q_total, q_max))
    peak = q_chunk * one_q_bytes + q_chunk * v_per_q_bytes
    return dict(
        q_chunk=q_chunk, mu_chunk=n_rmu, nu_chunk=n_rmu,
        tiled=False, per_rank_peak=peak,
    )


_v_q_batch_kernel_cache: dict = {}


def _make_V_q_batch_kernel(
    *,
    q_chunk: int,
    n_rmu: int,
    n_G_sph: int,
    fft_shape: tuple[int, int, int],
    sphere_idx: jax.Array | None,
    mesh_xy: Mesh,
):
    """Factory — ``fit_one_rchunk``-style.  Returns ONE jitted kernel that
    fuses the full per-q-chunk body:

        (ζ_q,μ,rtot  on  P(None,('x','y'),None)) →
            local transpose & FFT (r_tot → G_box) →
            phase-shift + sphere pick (G_box → G_flat) →
            √v(q+G) multiply  →
            with_sharding_constraint to  μ-X-only          (one-axis all-gather)
            with_sharding_constraint to  μ-Y-only          (one-axis all-gather)
            einsum 'qmG,qnG->qmn'                          (local gemm)
        →  V_q  on  P(None,'x','y'),  g0 on P(None,'x')

    All inside one jit so XLA can overlap the two gathers, keep the
    intermediate (μ_X, G) slab in registers/cache where possible, and
    schedule the gemm on the same stream as the subsequent FFI write.
    Follows the ``fit_one_rchunk`` convention: explicit in_shardings /
    out_shardings on the outer jit; ``with_sharding_constraint`` at each
    reshard boundary inside.  No inner ``@jax.jit`` boundaries.

    Cache key covers (mesh identity, q_chunk, n_rmu, n_G_sph, fft_shape,
    sphere_idx identity), so the (full-size + last-partial-chunk) pair
    compiles once each.
    """
    cache_key = (id(mesh_xy), q_chunk, n_rmu, n_G_sph, tuple(fft_shape),
                 id(sphere_idx))
    hit = _v_q_batch_kernel_cache.get(cache_key)
    if hit is not None:
        return hit

    nx, ny, nz = fft_shape
    n_rtot = nx * ny * nz

    mu_xy_sh   = NamedSharding(mesh_xy, P(None, ('x', 'y'), None))
    mu_x_sh    = NamedSharding(mesh_xy, P(None, 'x', None))
    mu_y_sh    = NamedSharding(mesh_xy, P(None, 'y', None))
    V_sh       = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    g0_sh      = NamedSharding(mesh_xy, P(None, 'x'))
    phase_sh   = NamedSharding(mesh_xy, P(None, None, None, None))
    sqrtv_sh   = NamedSharding(mesh_xy, P(None, None))
    # ζ comes off disk with layout (Q, n_rtot, n_rmu) — μ on ('x','y').
    # Local transpose at the top of the jit lands it in (Q, μ, rtot)
    # layout without a collective.
    zeta_rtot_mu_sh = NamedSharding(mesh_xy, P(None, None, ('x', 'y')))

    @partial(jax.jit,
             in_shardings=(zeta_rtot_mu_sh, sqrtv_sh, phase_sh),
             out_shardings=(V_sh, g0_sh))
    def _kernel(zeta_rtot_mu, sqrt_v_batch, phase_batch):
        # 1. Local transpose (Q, n_rtot, n_rmu) → (Q, n_rmu, n_rtot).
        #    μ stays on ('x','y'); no collective.
        zeta_mu_r = jax.lax.with_sharding_constraint(
            jnp.transpose(zeta_rtot_mu, (0, 2, 1)), mu_xy_sh)

        # 2. 3-D FFT (r_tot → G_box) + phase shift, all local per rank.
        Q, mu_per_rank, _ = zeta_mu_r.shape
        zeta_box = jnp.fft.fftn(
            zeta_mu_r.reshape(Q, mu_per_rank, nx, ny, nz)
                * phase_batch[:, None, :, :, :],
            axes=(-3, -2, -1),
        ).reshape(Q, mu_per_rank, n_rtot)

        # 3. G=0 column for head-handling downstream (kept sharded on μ).
        g0 = zeta_box[:, :, 0]

        # 4. Sphere pick (G_box → G_flat) + √v multiply — still μ-XY-sharded.
        if sphere_idx is not None:
            zeta_G = jnp.take(zeta_box, sphere_idx, axis=-1)
        else:
            zeta_G = zeta_box
        zeta_G = jax.lax.with_sharding_constraint(
            zeta_G * sqrt_v_batch[:, None, :], mu_xy_sh)

        # 5. Two one-axis gathers via with_sharding_constraint.  Each is a
        #    single-axis all-gather (small collective, balanced sizes on
        #    P_x = P_y); XLA can pipeline them on independent NCCL streams.
        zeta_mu_X = jax.lax.with_sharding_constraint(zeta_G, mu_x_sh)
        zeta_nu_Y = jax.lax.with_sharding_constraint(zeta_G, mu_y_sh)

        # 6. Contract  V[q, μ_X, ν_Y] = Σ_G conj(ζ_μ_X(G)) · ζ_ν_Y(G).
        #    G fully replicated on both operands; μ and ν disjointly sharded
        #    on the non-contraction axes, so no collective in the gemm.
        V_batch = jnp.einsum('qmG,qnG->qmn',
                              jnp.conj(zeta_mu_X), zeta_nu_Y, optimize=True)
        return V_batch, g0

    _kernel.mu_xy_sh = mu_xy_sh
    _kernel.zeta_rtot_mu_sh = zeta_rtot_mu_sh
    _kernel.V_sh = V_sh
    _kernel.g0_sh = g0_sh
    _v_q_batch_kernel_cache[cache_key] = _kernel
    return _kernel


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
    """Mesh-parallel V_q computation — the algorithm documented above.
    Works for any mesh, including 1×1 (the single-axis gathers degenerate
    to no-ops and the FFT/contract stay local).  Follows the
    ``fit_one_rchunk`` pattern: one big jit per q-batch, FFI reads /
    writes happen on the jax stream outside the jit so the Python loop
    schedules read + compute + write async-concurrently.

    Returns:
        V_qmunu  : (nkx, nky, nkz, n_rmu, n_rmu) sharded P(None,None,None,'x','y')
        g0_mu_all: (nkx, nky, nkz, n_rmu)       sharded P(None,None,None,'x')
    """
    from common.progress import LoopProgress

    nkx, nky, nkz = kgrid
    nq_total = nkx * nky * nkz
    p_x = int(mesh_xy.shape['x'])
    p_y = int(mesh_xy.shape['y'])

    # Build the V-μν kernel bundle (sphere_idx, sqrt_v/phase helpers, etc.).
    kernels = make_v_munu_chunked_kernel(
        fft_grid[0], fft_grid[1], fft_grid[2], nkx, nky, nkz,
        bvec, cell_volume, sys_dim, bdot=bdot,
        mc_average_vcoul_body=mc_average_vcoul_body,
        vcoul_cutoff_ry=bare_coulomb_cutoff,
    )
    n_G_sph = int(kernels.n_sph)

    # --- Chooser ---
    if budget_bytes is None:
        budget_bytes = 24.0e9
    choice = _choose_v_q_chunks(
        n_rmu=n_rmu, n_G=n_G_sph, n_q_total=nq_total,
        budget_bytes=budget_bytes, p_x=p_x, p_y=p_y,
    )
    if choice['tiled']:
        raise NotImplementedError(
            "compute_all_V_q_sharded: μ×ν tiled case (budget too tight "
            "for full μ at one q) — see Case B in the algorithm comment. "
            "Planned: always read TWO μ-blocks per tile, accept the "
            "1/n_chunks redundancy when μ_i == ν_j, keep the jit body "
            "uniform (no runtime conditionals).")
    q_chunk = int(choice['q_chunk'])

    if verbose and jax.process_index() == 0:
        print(f"  V_q (sharded): mesh={p_x}x{p_y}, q_chunk={q_chunk}, "
              f"N_μ={n_rmu}, N_G={n_G_sph}, "
              f"predicted peak/rank={choice['per_rank_peak']/1e9:.2f} GB")

    # The one big per-q-batch jit.
    kernel = _make_V_q_batch_kernel(
        q_chunk=q_chunk, n_rmu=n_rmu, n_G_sph=n_G_sph,
        fft_shape=fft_grid, sphere_idx=kernels.sphere_idx, mesh_xy=mesh_xy,
    )

    @partial(jax.jit,
             in_shardings=kernel.zeta_rtot_mu_sh,
             out_shardings=kernel.zeta_rtot_mu_sh)
    def _pad_zeta_to_q_chunk(zeta):
        """Pad the q axis up to q_chunk via DUS into zeros; preserves
        μ-on-('x','y') sharding on the rtot-first layout.  Called only
        on the last partial batch so the compiled kernel stays shape-
        stable (two compile shapes: full-size and last-partial)."""
        pad_shape = (q_chunk, zeta.shape[1], zeta.shape[2])
        padded = jax.lax.with_sharding_constraint(
            jnp.zeros(pad_shape, dtype=zeta.dtype), kernel.zeta_rtot_mu_sh)
        z0 = jnp.int32(0)
        return jax.lax.dynamic_update_slice(padded, zeta, (z0, z0, z0))

    # On-disk layout is (nq, n_rtot, n_rmu); partition_spec shards n_rmu
    # across ('x','y') so each rank only pulls its μ-shard off disk.
    # SlabIO.read_slab (FFI backend) dispatches ffi_read_call as a jax op
    # → async on the device stream, overlapping with the previous batch's
    # compute + write.  allgather fallback still works but replicates.
    _read_spec = P(None, None, ('x', 'y'))
    kgrid_arr = np.array([nkx, nky, nkz], dtype=np.float64)

    def _read_batch(batch_coords):
        """Sharded FFI read for one q-batch + wrapped-q-vec tuple list."""
        qvecs, q_flats = [], []
        for (qx, qy, qz) in batch_coords:
            qvec_nn = np.array([qx, qy, qz], dtype=np.float64)
            qvecs.append(np.where(qvec_nn > kgrid_arr / 2,
                                   qvec_nn - kgrid_arr, qvec_nn))
            q_flats.append(qx * (nky * nkz) + qy * nkz + qz)
        Q = len(batch_coords)
        contiguous = all(q_flats[i] == q_flats[0] + i for i in range(Q))
        if contiguous:
            zeta = zeta_io.read_slab(
                'zeta_q', shape=(Q, n_rtot, n_rmu), dtype=np.complex128,
                offset=(q_flats[0], 0, 0),
                mesh=mesh_xy, partition_spec=_read_spec)
        else:
            slabs = [zeta_io.read_slab(
                'zeta_q', shape=(1, n_rtot, n_rmu), dtype=np.complex128,
                offset=(qf, 0, 0),
                mesh=mesh_xy, partition_spec=_read_spec) for qf in q_flats]
            zeta = jnp.concatenate(slabs, axis=0)
            del slabs
        return zeta, qvecs

    def _prepare_sqrt_phase(qvec_list, actual_batch):
        sqrt_list, phase_list = [], []
        for qvec_wrapped in qvec_list:
            sv, ph = kernels.get_sqrt_v_and_phase(jnp.asarray(qvec_wrapped))
            sqrt_list.append(sv); phase_list.append(ph)
        if actual_batch < q_chunk:
            for _ in range(q_chunk - actual_batch):
                sqrt_list.append(sqrt_list[0]); phase_list.append(phase_list[0])
        return jnp.stack(sqrt_list, axis=0), jnp.stack(phase_list, axis=0)

    # Split q-coords into batches in flat-q order.
    q_coords = [(qx, qy, qz) for qx in range(nkx)
                for qy in range(nky) for qz in range(nkz)]
    batches = [q_coords[i:i + q_chunk] for i in range(0, nq_total, q_chunk)]

    if verbose and jax.process_index() == 0:
        print(f"  V_q (sharded): {nq_total} q-points in {len(batches)} "
              f"batches of up to {q_chunk}", flush=True)
    vq_progress = LoopProgress(
        nq_total, print, title="V_q computation",
        item_name="q-point", max_updates=min(nq_total, 20))

    V_pieces, g0_pieces = [], []

    with timing.section("compute_all_V_q_sharded"):
        for batch in batches:
            actual = len(batch)
            # Read (FFI async dispatch) — overlap with prior batch's kernel
            # + its downstream consumers on the jax stream.
            zeta, qvec_list = _read_batch(batch)
            if actual < q_chunk:
                zeta = _pad_zeta_to_q_chunk(zeta)
            sqrt_v_batch, phase_batch = _prepare_sqrt_phase(qvec_list, actual)

            # Compute — one big jit: transpose + FFT + √v + 2 gathers + gemm.
            V_batch, g0_batch = kernel(zeta, sqrt_v_batch, phase_batch)
            del zeta

            V_pieces.append(V_batch[:actual])
            g0_pieces.append(g0_batch[:actual])
            for _ in range(actual):
                vq_progress.step()

        vq_progress.finish()

    # Concatenate → reshape to final (nkx, nky, nkz, μ, μ) / (nkx,nky,nkz,μ).
    V_flat = jnp.concatenate(V_pieces, axis=0)
    g0_flat = jnp.concatenate(g0_pieces, axis=0)
    del V_pieces, g0_pieces
    V_qmunu = V_flat.reshape(nkx, nky, nkz, n_rmu, n_rmu)
    g0_mu_all = g0_flat.reshape(nkx, nky, nkz, n_rmu)
    V_qmunu = jax.lax.with_sharding_constraint(
        V_qmunu, NamedSharding(mesh_xy, P(None, None, None, 'x', 'y')))
    g0_mu_all = jax.lax.with_sharding_constraint(
        g0_mu_all, NamedSharding(mesh_xy, P(None, None, None, 'x')))
    return V_qmunu, g0_mu_all
