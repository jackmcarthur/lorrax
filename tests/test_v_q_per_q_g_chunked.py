"""Tests for the per-q, G-chunked V_q kernel (commit 2 of the G-flat
ζ migration).

Coverage:

1. ``compute_v_q_per_q_g_chunked`` matches a one-shot einsum reference
   for both single-ζ (V^{0,0}) and bispinor off-diagonal (signed v)
   cases.
2. Pad slots with ζ̃ = 0 contribute exactly zero to the contract
   (the writer's invariant — checked by appending zero-padded G's
   to a logical sphere and verifying the result is identical).
3. ``compute_v_q_per_G`` agrees with the legacy V_q kernel's
   per-FFT-grid ``v(q+G)`` at the per-q sphere positions, for both
   3-D bulk and 2-D slab.  This ties the writer's components table
   to the same v-formula the consumer uses.
"""
from __future__ import annotations

import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from common.coulomb_sphere import compute_per_q_bare_coulomb_components
from gw.compute_vcoul import (
    compute_v_q_per_q_g_chunked,
    compute_v_q_per_G,
    make_v_munu_chunked_kernel,
)


# ---------------------------------------------------------------------------
# 1. Contract correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("g_chunk", [4, 8, 16])
def test_v_q_per_q_g_chunked_matches_einsum(g_chunk):
    """Single-q kernel against the closed-form einsum reference."""
    n_rmu, ngkmax = 5, 32
    rng = np.random.default_rng(0xC0DE)
    zeta = jnp.asarray(
        rng.standard_normal((n_rmu, ngkmax))
        + 1j * rng.standard_normal((n_rmu, ngkmax)),
        dtype=jnp.complex128)
    v = jnp.asarray(rng.uniform(0.1, 1.0, ngkmax), dtype=jnp.complex128)

    V_ref = np.einsum('mG,G,nG->mn',
                       np.conj(np.asarray(zeta)),
                       np.asarray(v),
                       np.asarray(zeta),
                       optimize=True)
    V_got = compute_v_q_per_q_g_chunked(zeta, zeta, v, g_chunk=g_chunk)
    np.testing.assert_allclose(
        np.asarray(V_got), V_ref, atol=1e-12, rtol=1e-12)


def test_v_q_per_q_g_chunked_bispinor_offdiag():
    """L ≠ R with a signed/complex v (bispinor transverse projector)."""
    rng = np.random.default_rng(0xFADE)
    n_rmu_L, n_rmu_R, ngkmax = 4, 6, 16
    zL = jnp.asarray(
        rng.standard_normal((n_rmu_L, ngkmax))
        + 1j * rng.standard_normal((n_rmu_L, ngkmax)),
        dtype=jnp.complex128)
    zR = jnp.asarray(
        rng.standard_normal((n_rmu_R, ngkmax))
        + 1j * rng.standard_normal((n_rmu_R, ngkmax)),
        dtype=jnp.complex128)
    v_sgn = jnp.asarray(
        rng.standard_normal(ngkmax) + 1j * rng.standard_normal(ngkmax),
        dtype=jnp.complex128)
    V_ref = np.einsum('mG,G,nG->mn',
                       np.conj(np.asarray(zL)),
                       np.asarray(v_sgn),
                       np.asarray(zR),
                       optimize=True)
    V_got = compute_v_q_per_q_g_chunked(zL, zR, v_sgn, g_chunk=4)
    np.testing.assert_allclose(
        np.asarray(V_got), V_ref, atol=1e-12, rtol=1e-12)


def test_v_q_per_q_g_chunked_pad_slots_dont_contribute():
    """Pad slots with ζ̃ = 0 must vanish from the sum regardless of v(G)."""
    n_rmu, ngk_logical, ngkmax = 3, 12, 16
    rng = np.random.default_rng(0xBADD)
    z_logical = jnp.asarray(
        rng.standard_normal((n_rmu, ngk_logical))
        + 1j * rng.standard_normal((n_rmu, ngk_logical)),
        dtype=jnp.complex128)
    v_logical = jnp.asarray(rng.uniform(0.1, 1.0, ngk_logical),
                             dtype=jnp.complex128)

    z_padded = jnp.concatenate(
        [z_logical, jnp.zeros((n_rmu, ngkmax - ngk_logical),
                                dtype=jnp.complex128)], axis=-1)
    # Garbage v at pad positions — should not matter.
    v_padded = jnp.concatenate(
        [v_logical, jnp.asarray(
            rng.standard_normal(ngkmax - ngk_logical)
            + 1j * rng.standard_normal(ngkmax - ngk_logical),
            dtype=jnp.complex128)], axis=0)

    V_pad = compute_v_q_per_q_g_chunked(
        z_padded, z_padded, v_padded, g_chunk=4)
    V_logical_ref = np.einsum(
        'mG,G,nG->mn',
        np.conj(np.asarray(z_logical)),
        np.asarray(v_logical),
        np.asarray(z_logical),
        optimize=True)
    np.testing.assert_allclose(
        np.asarray(V_pad), V_logical_ref, atol=1e-12, rtol=1e-12)


def test_v_q_per_q_g_chunked_donates_accumulator():
    """Multiple calls into the donated accumulator sum, not overwrite."""
    n_rmu, ngkmax = 2, 8
    z = jnp.ones((n_rmu, ngkmax), dtype=jnp.complex128)
    v = jnp.ones(ngkmax, dtype=jnp.complex128)
    V = jnp.zeros((n_rmu, n_rmu), dtype=jnp.complex128)
    V = compute_v_q_per_q_g_chunked(z, z, v, g_chunk=4, V_acc=V)
    V = compute_v_q_per_q_g_chunked(z, z, v, g_chunk=4, V_acc=V)
    # ζ*ζv = 1 * 1 * 1 summed over ngkmax = ngkmax per (μ,ν).  Two passes ⇒ 2·ngkmax.
    np.testing.assert_allclose(
        np.asarray(V),
        2 * ngkmax * np.ones((n_rmu, n_rmu), dtype=np.complex128),
        atol=1e-12, rtol=1e-12)


def test_v_q_per_q_g_chunked_rejects_misaligned_ngkmax():
    z = jnp.zeros((2, 10), dtype=jnp.complex128)
    v = jnp.zeros(10, dtype=jnp.complex128)
    with pytest.raises(ValueError, match="multiple of g_chunk"):
        compute_v_q_per_q_g_chunked(z, z, v, g_chunk=4)


# ---------------------------------------------------------------------------
# 2. compute_v_q_per_G matches the legacy kernel's full-FFT v(q+G)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("sys_dim", [2, 3])
def test_v_q_per_G_matches_legacy_kernel(sys_dim):
    """v(q+G) on per-q components matches the legacy per-FFT-grid kernel.

    Builds a per-q sphere, evaluates ``v(q+G)`` via ``compute_v_q_per_G``,
    and compares to the legacy ``get_sqrt_v_and_phase`` output gathered
    at the same Miller indices.  Sanity-checks that the writer's
    components table is consumed by the new V_q with the same v-formula
    the legacy kernel uses.
    """
    fft_grid = (6, 6, 8) if sys_dim == 2 else (6, 6, 6)
    kgrid = (2, 2, 1) if sys_dim == 2 else (2, 2, 2)
    bvec = np.diag([0.7, 0.7, 0.3])
    cutoff = 8.0
    cell_volume = 100.0

    # Per-q sphere.
    q_int = np.array(
        [(qx, qy, qz) for qx in range(kgrid[0])
         for qy in range(kgrid[1]) for qz in range(kgrid[2])],
        dtype=np.float64)
    kg = np.asarray(kgrid, dtype=np.float64)
    q_int_w = np.where(q_int > kg / 2, q_int - kg, q_int)
    q_frac = q_int_w / kg

    pkg = compute_per_q_bare_coulomb_components(
        fft_grid, bvec, q_frac, cutoff, sys_dim=sys_dim)
    gvec_components = pkg['gvec_components_padded']
    sphere_idx_padded = pkg['sphere_idx_padded']
    ngk = pkg['ngk_per_q']

    # New per-q v_per_G builder.
    v_new = compute_v_q_per_G(
        q_frac, gvec_components,
        bvec=bvec, cell_volume=cell_volume,
        sys_dim=sys_dim, vcoul_cutoff_ry=cutoff,
    )

    # Legacy v_per_G via the kernel factory.
    kernels = make_v_munu_chunked_kernel(
        fft_grid[0], fft_grid[1], fft_grid[2],
        kgrid[0], kgrid[1], kgrid[2],
        bvec=bvec, cell_volume=cell_volume,
        sys_dim=sys_dim, mc_average_vcoul_body=False,
        vcoul_cutoff_ry=cutoff,
    )
    # get_sqrt_v_and_phase returns sqrt_v and phase; sqrt_v lives on the
    # shared sphere (with sphere_idx) or the full FFT box.  Square it
    # to get v itself, and gather at per-q sphere positions.  We use
    # the full-FFT path by walking the legacy kernel without a sphere
    # to keep this test independent of the legacy sphere logic.
    kernels_full = make_v_munu_chunked_kernel(
        fft_grid[0], fft_grid[1], fft_grid[2],
        kgrid[0], kgrid[1], kgrid[2],
        bvec=bvec, cell_volume=cell_volume,
        sys_dim=sys_dim, mc_average_vcoul_body=False,
        vcoul_cutoff_ry=None,                 # full FFT box, no narrowing
    )

    for qi in range(q_frac.shape[0]):
        sqrt_v, _ = kernels_full.get_sqrt_v_and_phase(
            jnp.asarray(q_int_w[qi], dtype=jnp.float64))
        sqrt_v_np = np.asarray(sqrt_v).reshape(-1)        # full FFT box
        # Reconstruct v from sqrt_v (drop the cutoff, then apply ours).
        v_full = sqrt_v_np * np.conj(sqrt_v_np)
        v_full = v_full.real                              # bare Coulomb is real
        # Apply the same cutoff the new function used.
        # Build per-FFT-grid |q+G|² to gate.
        nx, ny, nz = fft_grid
        gx = np.rint(np.fft.fftfreq(nx) * nx).astype(int)
        gy = np.rint(np.fft.fftfreq(ny) * ny).astype(int)
        gz = np.rint(np.fft.fftfreq(nz) * nz).astype(int)
        G_full = np.stack(np.meshgrid(gx, gy, gz, indexing='ij'),
                            axis=-1).reshape(-1, 3).astype(np.float64)
        qG_full_cart = (q_frac[qi] + G_full) @ bvec
        denom_full = np.sum(qG_full_cart * qG_full_cart, axis=-1)
        v_full = np.where(denom_full > cutoff, 0.0, v_full)

        # Compare at the per-q sphere positions (logical slots only).
        nk = int(ngk[qi])
        sphere_q = sphere_idx_padded[qi, :nk]
        v_legacy_at_sphere = v_full[sphere_q]
        v_new_at_sphere = v_new[qi, :nk]
        np.testing.assert_allclose(
            v_new_at_sphere, v_legacy_at_sphere,
            atol=1e-12, rtol=1e-12,
            err_msg=f"sys_dim={sys_dim}, q={qi}: v mismatch")
