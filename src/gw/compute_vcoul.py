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

            return sqrt_v, phase

    # NOTE: These are NOT JIT'd - they're meant to be called from an outer JIT
    # to avoid nested JIT compilation overhead. The outer JIT (_batch_proc or 
    # the chunked loop) compiles everything together.
    
    def fft_and_weight_inner(
        zeta_r: jax.Array,
        sqrt_v: jax.Array,
        phase: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        FFT zeta and weight by √v. Shape inferred from input.
        NOT JIT'd - call from within an outer JIT.
        """
        B_mu = zeta_r.shape[0]
        zeta_spatial = zeta_r.reshape(B_mu, fft_nx, fft_ny, fft_nz)
        zeta_phased = zeta_spatial * phase
        zeta_G = jnp.fft.fftn(zeta_phased, axes=(-3, -2, -1))
        g0_chunk = zeta_G[:, 0, 0, 0]
        zeta_weighted = zeta_G * sqrt_v[None, :, :, :]
        return zeta_weighted.reshape(B_mu, n_G), g0_chunk
    
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
            
            # Load and FFT ν-chunk
            zeta_nu_r_np = zeta_dset[qx, qy, qz, mu_j_start:mu_j_end, :]
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
    zeta_dset = zeta_h5['zeta_q']
    
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
    q_batch_size: int | None = None,
    verbose: bool = True,
    bdot: np.ndarray | None = None,
    mc_average_vcoul_body: bool = True,
    bare_coulomb_cutoff: float | None = None,
) -> jax.Array:
    """
    Compute V_q for all q-points from zeta stored in HDF5.

    Loops over all q-points, computing V_q using μ-chunking for each. When the
    μ chunks already cover the full set (single chunk), q-points can be batched
    to reuse the FFT and contraction kernels.
    
    Args:
        zeta_h5: Open HDF5 file containing 'zeta_q' with shape (nqx, nqy, nqz, n_rmu, n_rtot)
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
    
    zeta_dset = zeta_h5['zeta_q']
    n_rmu = zeta_dset.shape[3]
    n_rtot = zeta_dset.shape[4]
    
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
                
                zeta_stacked[i] = zeta_dset[qx, qy, qz, :, :]
            
            return zeta_stacked, qvecs
        
        def prepare_batch_on_gpu(zeta_stacked_np, qvec_list, actual_size):
            """Transfer batch to GPU and compute sqrt_v/phase."""
            # Compute sqrt_v and phase for each q
            sqrt_batch = []
            phase_batch = []
            for qvec_wrapped in qvec_list:
                qvec_wrapped_jax = jnp.asarray(qvec_wrapped)
                sqrt_v, phase = kernels.get_sqrt_v_and_phase(qvec_wrapped_jax)
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
                        V_q_local = np.zeros((n_rmu, n_rmu), dtype=np.complex128)
                        
                        for i in range(n_chunks):
                            mu_i_start = i * mu_chunk_size
                            mu_i_end = min(mu_i_start + mu_chunk_size, n_rmu)
                            
                            zeta_mu_r_np = zeta_dset[qx, qy, qz, mu_i_start:mu_i_end, :]
                            zeta_mu_r = jnp.asarray(zeta_mu_r_np)
                            B_mu_i = mu_i_end - mu_i_start
                            zeta_mu_weighted, g0_chunk = kernels.fft_and_weight(zeta_mu_r, sqrt_v, phase, B_mu_i)
                            
                            g0_mu_np[qx, qy, qz, mu_i_start:mu_i_end] = np.asarray(g0_chunk)
                            
                            V_ii = kernels.contract_block(zeta_mu_weighted, zeta_mu_weighted)
                            V_q_local[mu_i_start:mu_i_end, mu_i_start:mu_i_end] = np.asarray(V_ii)
                            
                            for j in range(i + 1, n_chunks):
                                mu_j_start = j * mu_chunk_size
                                mu_j_end = min(mu_j_start + mu_chunk_size, n_rmu)
                                
                                zeta_nu_r_np = zeta_dset[qx, qy, qz, mu_j_start:mu_j_end, :]
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
