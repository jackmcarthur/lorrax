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


# Convenience module-level mesh for tests that don't already take the
# ``single_device_mesh`` fixture as a parameter.  Public transforms
# require ``mesh`` — for single-device runs the 1×1 trivial mesh is the
# right thing to pass.
MESH = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
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
                          mesh=MESH, norm='ortho', kvecs_frac=kvecs)
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(ref),
            atol=1e-11, rtol=1e-11,
            err_msg=f"r0={r0}, r_len={r_len}")


# ---------------------------------------------------------------------------

def test_accumulate_rchunk_to_gflat_round_trip():
    """Walk r-axis in chunks via to_rchunk; accumulate back into G_flat
    via accumulate_rchunk_to_gflat; verify the sum equals a direct
    forward FFT of the IFFT (i.e. the original G-flat input).

    ``accumulate_rchunk_to_gflat`` is strict ``(n_q, n_rmu, r_len)``; we
    fold (n_band, n_spinor) into a single μ axis to match.  Identity
    per-q sphere keeps the whole flat-FFT axis so we can compare
    against the direct ``_box_kernel`` reference element-wise."""
    psi, g_index, fft_grid = _make_psi_and_gindex(
        n_k=2, n_band=3, n_spinor=2, fft_grid=(4, 4, 6))
    nx, ny, nz = fft_grid
    n_rtot = nx * ny * nz
    n_k, nb, ns, ngkmax = psi.shape
    n_mu = nb * ns

    # Reference: IFFT then FFT round-trip (with no phase) returns the
    # input box up to numerical noise.  We test the no-phase path so
    # we can compare contributions cleanly.
    from common.wfn_transforms import _box_kernel
    box_flat_ref = _box_kernel(psi, g_index, ngkmax=ngkmax).reshape(
        n_k, n_mu, n_rtot)

    # Identity sphere: sphere[q, i] = i, ngkmax = n_rtot.  Same indices
    # per q, but stored in the (n_q, ngkmax) shape the new API requires.
    sphere_id = np.tile(np.arange(n_rtot, dtype=np.int32)[None, :], (n_k, 1))

    # Walk r in chunks; for each, IFFT-and-slice via to_rchunk, then
    # forward-FFT-and-add-to-Gflat via accumulate_rchunk_to_gflat.
    gflat_acc = jnp.zeros((n_k, n_mu, n_rtot), dtype=jnp.complex128)
    r_chunk = 8
    for r0 in range(0, n_rtot, r_chunk):
        r_len = min(r_chunk, n_rtot - r0)
        rchunk_4d = to_rchunk(psi, g_index, fft_grid, r0, r_len,
                               mesh=MESH, norm='ortho')   # (n_k, nb, ns, r_len)
        rchunk = rchunk_4d.reshape(n_k, n_mu, r_len)
        gflat_acc = accumulate_rchunk_to_gflat(
            rchunk, gflat_acc, mesh=MESH,
            fft_grid=fft_grid, r0=r0, sphere_idx=sphere_id,
            qvec_frac=None, norm='ortho',
        )

    np.testing.assert_allclose(
        np.asarray(gflat_acc),
        np.asarray(box_flat_ref),
        atol=1e-10, rtol=1e-10,
    )


# chunk_size ∈ {None one-shot, divisors of N=36, a non-divisor (7) that
# exercises the zero-pad path}.
@pytest.mark.parametrize("chunk_size", [None, 12, 9, 7, 4])
def test_accumulate_rchunk_to_gflat_chunked_matches_one_shot(chunk_size):
    """``chunk_size != None`` must be bit-equal to one-shot.  The
    flat-axis chunker zero-pads ``N = n_q · n_mu_local`` up to a
    multiple of chunk_size, so ANY positive integer chunk size is
    valid (no divisibility constraint on n_q or n_mu_local)."""
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
    acc_ref = accumulate_rchunk_to_gflat(
        rchunk=rch, gflat_acc=jnp.zeros((n_q, n_rmu, 5), dtype=jnp.complex128),
        mesh=MESH, fft_grid=fft_grid, r0=0,
        sphere_idx=sphere_per_q, qvec_frac=kvecs, norm='backward',
        chunk_size=None)
    acc_chk = accumulate_rchunk_to_gflat(
        rchunk=rch, gflat_acc=jnp.zeros((n_q, n_rmu, 5), dtype=jnp.complex128),
        mesh=MESH, fft_grid=fft_grid, r0=0,
        sphere_idx=sphere_per_q, qvec_frac=kvecs, norm='backward',
        chunk_size=chunk_size)
    np.testing.assert_allclose(
        np.asarray(acc_chk), np.asarray(acc_ref),
        atol=1e-12, rtol=1e-12,
        err_msg=f"chunk_size={chunk_size}")


def test_accumulate_rchunk_to_gflat_sphere_subset():
    """Per-q sphere_idx gathers a subset of the full G axis."""
    psi, g_index, fft_grid = _make_psi_and_gindex(
        n_k=1, n_band=1, n_spinor=1, fft_grid=(4, 4, 4))
    nx, ny, nz = fft_grid
    n_rtot = nx * ny * nz
    n_k, nb, ns, ngkmax = psi.shape
    n_mu = nb * ns

    rng = np.random.default_rng(0xCAFE)
    # Same indices for every q, but stored 2-D as the new API requires.
    sphere_1d = np.sort(rng.choice(n_rtot, size=20, replace=False).astype(np.int32))
    sphere_per_q = np.tile(sphere_1d[None, :], (n_k, 1))

    from common.wfn_transforms import _box_kernel
    box_ref = _box_kernel(psi, g_index, ngkmax=ngkmax).reshape(
        n_k, n_mu, n_rtot)
    ref = box_ref[..., sphere_1d]

    gflat_acc = jnp.zeros((n_k, n_mu, 20), dtype=jnp.complex128)
    for r0 in range(0, n_rtot, 16):
        r_len = min(16, n_rtot - r0)
        rchunk = to_rchunk(psi, g_index, fft_grid, r0, r_len,
                            mesh=MESH, norm='ortho').reshape(
                                n_k, n_mu, r_len)
        gflat_acc = accumulate_rchunk_to_gflat(
            rchunk, gflat_acc, mesh=MESH,
            fft_grid=fft_grid, r0=r0, sphere_idx=sphere_per_q,
            norm='ortho',
        )

    np.testing.assert_allclose(
        np.asarray(gflat_acc), np.asarray(ref),
        atol=1e-10, rtol=1e-10,
    )
