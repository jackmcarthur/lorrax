"""Tests for the rchunk ↔ G_flat transform pair (Phase A of
``docs/PLAN_zeta_g_flat_migration.md``).

Three properties under test:

1. ``apply_bloch_phase_on_slice`` matches ``apply_bloch_phase`` on the
   same slab — phase-on-slice and phase-on-full-then-slice are
   mathematically equivalent, so the per-cell values must agree to
   float-roundoff.
2. ``to_rchunk`` with the new phase-after-slice path matches the
   reference (phase-then-slice) closed-form value.
3. ``accumulate_rchunk_to_gflat`` is the inverse of
   ``to_rchunk`` on a full r-axis cover — running ``to_rchunk`` over
   all r-chunks and accumulating G_flat reproduces a direct
   forward-FFT-then-gather.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import pytest

jax.config.update("jax_enable_x64", True)

from common.wfn_transforms import (
    apply_bloch_phase, apply_bloch_phase_on_slice,
    to_rchunk, accumulate_rchunk_to_gflat,
)


@pytest.fixture
def single_device_mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))


def _make_psi_and_gindex(n_k=2, n_band=3, n_spinor=2, fft_grid=(4, 4, 5),
                         ngkmax=20, seed=0xABCD):
    """Synthetic ψ + g_index suitable for to_rchunk."""
    rng = np.random.default_rng(seed)
    psi = (rng.standard_normal((n_k, n_band, n_spinor, ngkmax))
           + 1j * rng.standard_normal((n_k, n_band, n_spinor, ngkmax))
          ).astype(np.complex128)
    nx, ny, nz = fft_grid
    n_rtot = nx * ny * nz
    # g_index[k, x, y, z] picks a G-row in [0, ngkmax].  Use ngkmax as
    # the sentinel "zero" slot (the kernel pads psi with a zero column
    # at column ngkmax).
    g_index = rng.integers(0, ngkmax + 1, size=(n_k, nx, ny, nz)).astype(np.int32)
    # Sprinkle the sentinel value to test zero-fill.
    g_index[g_index >= ngkmax - 1] = ngkmax
    return jnp.asarray(psi), jnp.asarray(g_index), fft_grid


# ---------------------------------------------------------------------------

def test_phase_on_slice_matches_full_phase_then_slice(single_device_mesh):
    """apply_bloch_phase_on_slice equals apply_bloch_phase followed by
    slicing — on the same r-cells, both formulas yield the same value."""
    fft_grid = (4, 4, 5)
    n_rtot = 80
    n_k = 3

    rng = np.random.default_rng(0xDEAF)
    kvecs = jnp.asarray(rng.uniform(-0.5, 0.5, size=(n_k, 3)),
                         dtype=jnp.float64)
    box = jnp.asarray(
        rng.standard_normal((n_k, *fft_grid))
        + 1j * rng.standard_normal((n_k, *fft_grid)),
        dtype=jnp.complex128,
    )

    # Reference: phase on the full box, then flatten + slice.
    box_phased = apply_bloch_phase(box, kvecs, fft_grid, sign=+1)
    ref = box_phased.reshape(n_k, n_rtot)

    # Test: build the slab (slice of the un-phased box) and apply
    # phase-on-slice across a few r0 windows.
    for r0, r_len in [(0, 5), (10, 20), (60, 20), (n_rtot - 7, 7)]:
        slab = box.reshape(n_k, n_rtot)[:, r0:r0 + r_len]
        slab_phased = apply_bloch_phase_on_slice(
            slab, kvecs, fft_grid, r0, r_len, sign=+1)
        np.testing.assert_allclose(
            np.asarray(slab_phased),
            np.asarray(ref[:, r0:r0 + r_len]),
            atol=1e-12, rtol=1e-12,
            err_msg=f"r0={r0}, r_len={r_len}")


# ---------------------------------------------------------------------------

def test_to_rchunk_phase_after_slice_matches_direct_formula():
    """to_rchunk(phase-after-slice) value = (IFFT(box) on the slab) times
    the Bloch phase evaluated at those flat-r cells."""
    psi, g_index, fft_grid = _make_psi_and_gindex()
    nx, ny, nz = fft_grid
    n_k = int(psi.shape[0])
    n_rtot = nx * ny * nz

    rng = np.random.default_rng(0xFADE)
    kvecs = jnp.asarray(rng.uniform(-0.5, 0.5, size=(n_k, 3)),
                         dtype=jnp.float64)

    # Reference: full pipeline (phase on full box, then slice).
    from common.wfn_transforms import _box_kernel
    ngkmax = int(psi.shape[-1])
    box = _box_kernel(psi, g_index, ngkmax=ngkmax)        # (n_k, nb, ns, nx, ny, nz)
    rb = jnp.fft.ifftn(box, axes=(-3, -2, -1), norm='ortho')
    rb = apply_bloch_phase(rb, kvecs, fft_grid, sign=+1)
    rb_flat = rb.reshape(*rb.shape[:3], n_rtot)

    for r0, r_len in [(0, 10), (20, 30), (n_rtot - 15, 15)]:
        ref = rb_flat[..., r0:r0 + r_len]
        got = to_rchunk(psi, g_index, fft_grid, r0, r_len,
                          norm='ortho', kvecs_frac=kvecs)
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(ref),
            atol=1e-11, rtol=1e-11,
            err_msg=f"r0={r0}, r_len={r_len}")


# ---------------------------------------------------------------------------

def test_accumulate_rchunk_to_gflat_round_trip():
    """Walk r-axis in chunks via to_rchunk; accumulate back into G_flat
    via accumulate_rchunk_to_gflat; verify the sum equals a direct
    forward FFT of the IFFT (i.e. the original G-flat input)."""
    psi, g_index, fft_grid = _make_psi_and_gindex(
        n_k=2, n_band=3, n_spinor=2, fft_grid=(4, 4, 6))
    nx, ny, nz = fft_grid
    n_rtot = nx * ny * nz
    n_k, nb, ns, ngkmax = psi.shape

    # Reference: IFFT then FFT round-trip (with no phase) returns the
    # input box up to numerical noise.  We test the no-phase path so
    # we can compare contributions cleanly.
    from common.wfn_transforms import _box_kernel
    box_ref = _box_kernel(psi, g_index, ngkmax=ngkmax)    # (n_k, nb, ns, nx, ny, nz)
    box_flat_ref = box_ref.reshape(n_k, nb, ns, n_rtot)

    # Walk r in chunks; for each, IFFT-and-slice via to_rchunk, then
    # forward-FFT-and-add-to-Gflat via accumulate_rchunk_to_gflat.
    n_G_sph = n_rtot           # keep the whole flat-FFT axis for the test
    gflat_acc = jnp.zeros((n_k, nb, ns, n_G_sph), dtype=jnp.complex128)
    r_chunk = 8
    for r0 in range(0, n_rtot, r_chunk):
        r_len = min(r_chunk, n_rtot - r0)
        rchunk = to_rchunk(psi, g_index, fft_grid, r0, r_len, norm='ortho')
        # Need n_q == n_k for the accumulate signature; treat the
        # leading axis of rchunk as the "q" axis here.
        gflat_acc = accumulate_rchunk_to_gflat(
            rchunk, gflat_acc,
            fft_grid=fft_grid, r0=r0, sphere_idx=None,
            qvec_frac=None, norm='ortho',
        )

    # Compare element-wise.  Round-trip of IFFT then FFT is identity
    # up to numerical roundoff.
    np.testing.assert_allclose(
        np.asarray(gflat_acc),
        np.asarray(box_flat_ref),
        atol=1e-10, rtol=1e-10,
    )


@pytest.mark.parametrize("fft_batch_chunks", [1, 2, 3])
def test_accumulate_rchunk_to_gflat_chunked_matches_one_shot(fft_batch_chunks):
    """fft_batch_chunks > 1 (scan over the μ axis) must be bit-equal to
    the one-shot path.  Chunks the FFT batch axis to bound the working
    set on CrI3-scale runs where the full (n_q, n_rmu, nx*ny*nz) box
    OOMs.  ``n_rmu = 6`` is divisible by every parametrised chunk count
    so the divisor check on n_mu_local (= n_rmu / p_prod = 6 here)
    passes for all of {1, 2, 3}."""
    fft_grid = (4, 4, 6)
    n_q, n_rmu, n_rtot = 6, 6, int(np.prod(fft_grid))
    rng = np.random.default_rng(0xAA)
    sphere_per_q = np.zeros((n_q, 5), dtype=np.int32)
    for q in range(n_q):
        sphere_per_q[q] = (np.arange(5) * (q + 1)) % n_rtot
    kvecs = jnp.asarray(
        rng.uniform(-0.5, 0.5, (n_q, 3)), dtype=jnp.float64)

    rch = jnp.asarray(
        rng.standard_normal((n_q, n_rmu, n_rtot))
        + 1j * rng.standard_normal((n_q, n_rmu, n_rtot)),
        dtype=jnp.complex128)
    acc_ref = jnp.zeros((n_q, n_rmu, 5), dtype=jnp.complex128)
    acc_ref = accumulate_rchunk_to_gflat(
        rchunk=rch, gflat_acc=acc_ref,
        fft_grid=fft_grid, r0=0, sphere_idx=sphere_per_q,
        qvec_frac=kvecs, norm='backward', fft_batch_chunks=1)

    acc_chk = jnp.zeros((n_q, n_rmu, 5), dtype=jnp.complex128)
    acc_chk = accumulate_rchunk_to_gflat(
        rchunk=rch, gflat_acc=acc_chk,
        fft_grid=fft_grid, r0=0, sphere_idx=sphere_per_q,
        qvec_frac=kvecs, norm='backward',
        fft_batch_chunks=fft_batch_chunks)
    np.testing.assert_allclose(
        np.asarray(acc_chk), np.asarray(acc_ref),
        atol=1e-12, rtol=1e-12,
        err_msg=f"fft_batch_chunks={fft_batch_chunks}")


def test_accumulate_rchunk_to_gflat_chunked_rejects_indivisible(
        single_device_mesh):
    """fft_batch_chunks must divide n_mu_local = n_mu_padded / p_prod
    (so each chunk's per-rank shard stays integer-sized)."""
    fft_grid = (4, 4, 4)
    # p_prod = 1 (single device) → n_mu_local = n_mu_padded = 2.
    # fft_batch_chunks=3 doesn't divide 2.
    with single_device_mesh:
        rch = jax.device_put(
            jnp.zeros((5, 2, 8), dtype=jnp.complex128),
            NamedSharding(single_device_mesh, P(None, ('x', 'y'), None)))
        acc = jax.device_put(
            jnp.zeros((5, 2, 3), dtype=jnp.complex128),
            NamedSharding(single_device_mesh, P(None, ('x', 'y'), None)))
        with pytest.raises(ValueError, match="must divide n_mu_local"):
            accumulate_rchunk_to_gflat(
                rchunk=rch, gflat_acc=acc, fft_grid=fft_grid, r0=0,
                sphere_idx=np.array([0, 1, 2], dtype=np.int32),
                qvec_frac=None, fft_batch_chunks=3)


def test_accumulate_rchunk_to_gflat_sphere_subset():
    """sphere_idx gathers a subset of the full G axis."""
    psi, g_index, fft_grid = _make_psi_and_gindex(
        n_k=1, n_band=1, n_spinor=1, fft_grid=(4, 4, 4))
    nx, ny, nz = fft_grid
    n_rtot = nx * ny * nz
    n_k, nb, ns, ngkmax = psi.shape

    rng = np.random.default_rng(0xCAFE)
    sphere_idx = rng.choice(n_rtot, size=20, replace=False).astype(np.int32)
    sphere_idx.sort()

    from common.wfn_transforms import _box_kernel
    box_ref = _box_kernel(psi, g_index, ngkmax=ngkmax)
    ref = box_ref.reshape(n_k, nb, ns, n_rtot)[..., sphere_idx]

    gflat_acc = jnp.zeros((n_k, nb, ns, 20), dtype=jnp.complex128)
    for r0 in range(0, n_rtot, 16):
        r_len = min(16, n_rtot - r0)
        rchunk = to_rchunk(psi, g_index, fft_grid, r0, r_len, norm='ortho')
        gflat_acc = accumulate_rchunk_to_gflat(
            rchunk, gflat_acc,
            fft_grid=fft_grid, r0=r0, sphere_idx=sphere_idx,
            norm='ortho',
        )

    np.testing.assert_allclose(
        np.asarray(gflat_acc), np.asarray(ref),
        atol=1e-10, rtol=1e-10,
    )
