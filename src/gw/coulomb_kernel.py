"""
(q+G)-space Coulomb potential helpers and the per-q ``v(q+G)`` factory.

This module is the source of √v(q+G) and the FFT/sphere/√v primitives used by
the V_q tile kernel.  It is independent of ζ — the consumer (``v_q_tile``)
plugs the ``get_sqrt_v_and_phase`` callable into its FFT pipeline.

Contents:
    - ``exp_ikr_fftbox``, ``fft_integer_axes``  : tiny FFT-grid shape helpers
    - ``compute_sqrt_vcoul_0d``                  : numerical √v on FFT box for
                                                   0-D cell-box truncation
    - ``make_v_munu_chunked_kernel``             : factory returning a bundle
                                                   of jit'd primitives
                                                   (``get_sqrt_v_and_phase``,
                                                   ``fft_and_weight``,
                                                   ``contract_block``, …)

The factory caches by (FFT grid, k-grid, bvec, cell, sys_dim) so repeated
entry from different drivers shares one compiled set of kernels.

The MC mini-BZ averaging at ``q+G=0`` (3-D) and the BGW vcoul overlay are
implemented here (the overlay is wired into the driver — see
``v_q_driver._sqrt_v_phase_batch``).
"""

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


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
# Coulomb potential computation (0D box truncation)
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
        # Flatten to 1-D (n_rtot,) so the per-q vmap output is (Q, n_rtot),
        # matching the kernel's ``sqrt_v_batch[:, None, :]`` broadcast onto
        # ``zeta_G`` which is (Q, μ, n_rtot) when ``sphere_idx is None``.
        # Same convention as the 3-D sphere path, which calls
        # ``sqrt_v.reshape(-1)[sphere_idx]`` to land at 1-D pre-vmap.
        sqrt_v_0d_flat = sqrt_v_0d.reshape(-1)
        # ``v_per_G_0d`` = sqrt_v_0d ** 2 (un-sqrt'd v on the FFT box).
        # Used by the unified V_q tile kernel which takes √v inside.
        # Float64 (real, non-negative) so ``jnp.sqrt`` in the kernel is
        # well-defined (PSD assumption).
        v_per_G_0d_flat = (jnp.real(sqrt_v_0d_flat) ** 2).astype(jnp.float64)

        @jax.jit
        def get_sqrt_v_and_phase(qvec_wrapped: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Return precomputed √v(G) and trivial phase (q must be 0)."""
            # Phase is 1.0 for q=0
            phase = jnp.ones((1, fft_nx, fft_ny, fft_nz), dtype=jnp.complex128)
            return sqrt_v_0d_flat, phase

        @jax.jit
        def get_v_per_G_and_phase(qvec_wrapped: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Return v(G) (un-sqrt'd, ≥0) and trivial phase (q must be 0).

            Companion to ``get_sqrt_v_and_phase`` for the unified V_q
            tile kernel (``v_q_tile.compute_V_q_tile``) which takes √v
            inside.  Always real, non-negative.
            """
            phase = jnp.ones((1, fft_nx, fft_ny, fft_nz), dtype=jnp.complex128)
            return v_per_G_0d_flat, phase
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

        def _v_scaled_3d(qvec_wrapped):
            """Build the un-sqrt'd v(q+G) on the full FFT box (3-D bulk).

            Same masking convention as the legacy ``get_sqrt_v_and_phase``
            (real, ≥ 0, with G=0 → MC-averaged value or 0 for q=0).  The
            two factories below wrap this and either return v directly
            (``get_v_per_G_and_phase``) or its sqrt cast to c128
            (``get_sqrt_v_and_phase``).  Splitting the helper guarantees
            both surfaces produce bit-identical v_scaled, so √v_scaled
            in either factory can be applied symmetrically by the
            unified V_q tile kernel.
            """
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
            return v_scaled

        def _phase_3d(qvec_wrapped):
            return jnp.exp(-2j * jnp.pi * (
                qvec_wrapped[0] / nkx_f * fx +
                qvec_wrapped[1] / nky_f * fy +
                qvec_wrapped[2] / nkz_f * fz
            ))

        @jax.jit
        def get_sqrt_v_and_phase(qvec_wrapped: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Compute √v(q+G) and phase for 3D bulk (untruncated Coulomb).

            For G≠0: v = 8π/|q+G|² (point value).
            For G=0, q≠0: v = <8π/|q+δq|²>_miniBZ (MC averaged over Voronoi cell).
            For G=0, q=0: v = 0 (head injected separately via rank-1 correction).
            """
            phase = _phase_3d(qvec_wrapped)
            v_scaled = _v_scaled_3d(qvec_wrapped)
            sqrt_v = jnp.where(v_scaled > 0.0, jnp.sqrt(v_scaled), 0.0).astype(jnp.complex128)
            if sphere_idx is not None:
                sqrt_v = sqrt_v.reshape(-1)[sphere_idx]  # 1-D for fft_and_weight sphere path
            return sqrt_v, phase

        @jax.jit
        def get_v_per_G_and_phase(qvec_wrapped: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Companion of ``get_sqrt_v_and_phase``: returns v(q+G) un-sqrt'd.

            Used by the unified V_q tile kernel (``v_q_tile``) which
            takes ``sqrt`` inside.  Always real, non-negative.  Same
            masking and MC-averaging as ``get_sqrt_v_and_phase``.
            """
            phase = _phase_3d(qvec_wrapped)
            v_scaled = _v_scaled_3d(qvec_wrapped)
            if sphere_idx is not None:
                v_scaled = v_scaled.reshape(-1)[sphere_idx]
            return v_scaled, phase

    elif sys_dim == 2:

        def _v_scaled_2d(qvec_wrapped):
            """Build the un-sqrt'd v(q+G) on the full FFT box (2-D slab)."""
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
            return v_scaled

        def _phase_2d(qvec_wrapped):
            return jnp.exp(-2j * jnp.pi * (
                qvec_wrapped[0] / nkx_f * fx +
                qvec_wrapped[1] / nky_f * fy +
                qvec_wrapped[2] / nkz_f * fz
            ))

        @jax.jit
        def get_sqrt_v_and_phase(qvec_wrapped: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Compute √v(q+G) and phase for a given q-point."""
            phase = _phase_2d(qvec_wrapped)
            v_scaled = _v_scaled_2d(qvec_wrapped)
            sqrt_v = jnp.where(v_scaled > 0.0, jnp.sqrt(v_scaled), 0.0).astype(jnp.complex128)
            if sphere_idx is not None:
                sqrt_v = sqrt_v.reshape(-1)[sphere_idx]  # 1-D for fft_and_weight sphere path
            return sqrt_v, phase

        @jax.jit
        def get_v_per_G_and_phase(qvec_wrapped: jax.Array) -> tuple[jax.Array, jax.Array]:
            """Companion of ``get_sqrt_v_and_phase`` for the unified V_q
            tile kernel: returns v(q+G) un-sqrt'd (real, ≥0), same
            masking as ``get_sqrt_v_and_phase``.
            """
            phase = _phase_2d(qvec_wrapped)
            v_scaled = _v_scaled_2d(qvec_wrapped)
            if sphere_idx is not None:
                v_scaled = v_scaled.reshape(-1)[sphere_idx]
            return v_scaled, phase

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
        get_v_per_G_and_phase=get_v_per_G_and_phase,  # un-sqrt'd v + phase, used by v_q_tile
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
