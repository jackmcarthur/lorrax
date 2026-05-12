"""Tests for the per-q WFN.h5-style sphere helper + writer/header
round-trip.

Covers:

1. ``compute_per_q_bare_coulomb_components`` — Miller indices match
   ``|q+G|² ≤ cutoff`` exactly; pad slots carry the sentinel
   ``(-nx/2, -ny/2, -nz/2)``; ``ngk[q]`` counts agree with the
   live mask.
2. ``accumulate_rchunk_to_gflat`` with per-q sphere — matches a
   reference forward FFT + per-q gather, with pad slots zeroed by
   the caller post-loop.
3. ``IsdfHeader`` round-trip — write a G-flat header to HDF5, read it
   back, verify ``gvec_components`` / ``ngk_per_q`` /
   ``zeta_cutoff_ry`` are bit-exact.
"""
from __future__ import annotations

import h5py
import numpy as np
import pytest
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from common.coulomb_sphere import (
    compute_bare_coulomb_sphere_idx,
    compute_per_q_bare_coulomb_components,
)
from common.wfn_transforms import accumulate_rchunk_to_gflat
from file_io.isdf_header import (
    IsdfHeader, write_isdf_header, read_isdf_header,
)


# ---------------------------------------------------------------------------
# Helper: closed-form per-q sphere via direct (q+G) enumeration.
# ---------------------------------------------------------------------------

def _ref_sphere_per_q(fft_grid, bvec, q_frac, cutoff_ry):
    nx, ny, nz = fft_grid
    gx = np.rint(np.fft.fftfreq(nx) * nx).astype(np.int32)
    gy = np.rint(np.fft.fftfreq(ny) * ny).astype(np.int32)
    gz = np.rint(np.fft.fftfreq(nz) * nz).astype(np.int32)
    G_int = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
    qG = q_frac[None, :] + G_int.astype(np.float64)
    qG_cart = qG @ np.asarray(bvec, dtype=np.float64)
    return (np.sum(qG_cart * qG_cart, axis=-1) <= cutoff_ry), G_int


# ---------------------------------------------------------------------------
# 1. Helper correctness
# ---------------------------------------------------------------------------

def test_per_q_sphere_matches_direct_enumeration():
    fft_grid = (8, 8, 12)
    bvec = np.diag([0.7, 0.7, 0.3])
    kgrid = (2, 2, 1)
    cutoff = 10.0

    # IBZ q's (BGW wrap, divided by kgrid).
    q_int = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], dtype=int)
    q_int_wrapped = np.where(
        q_int > np.asarray(kgrid) / 2, q_int - np.asarray(kgrid), q_int)
    q_frac = q_int_wrapped / np.asarray(kgrid)

    out = compute_per_q_bare_coulomb_components(
        fft_grid, bvec, q_frac, cutoff, sys_dim=2)

    sphere_padded = out["sphere_idx_padded"]
    components = out["gvec_components_padded"]
    ngk = out["ngk_per_q"]
    ngkmax = out["ngkmax"]

    nx, ny, nz = fft_grid
    sentinel = np.array(
        [np.rint(np.fft.fftfreq(nx) * nx)[nx // 2],
         np.rint(np.fft.fftfreq(ny) * ny)[ny // 2],
         np.rint(np.fft.fftfreq(nz) * nz)[nz // 2]], dtype=np.int32)

    for q_i, qf in enumerate(q_frac):
        mask, G_int = _ref_sphere_per_q(fft_grid, bvec, qf, cutoff)
        ref_idx = np.nonzero(mask)[0].astype(np.int32)
        ref_idx.sort()
        nk = int(ngk[q_i])
        assert nk == ref_idx.size, (
            f"q={q_i}: ngk={nk} ≠ ref={ref_idx.size}")
        np.testing.assert_array_equal(sphere_padded[q_i, :nk], ref_idx)
        # Components for logical slots match Miller(ref_idx).
        np.testing.assert_array_equal(
            components[q_i, :, :nk], G_int[ref_idx].T)
        # Pad slots all carry the sentinel.
        for j in range(nk, ngkmax):
            np.testing.assert_array_equal(components[q_i, :, j], sentinel)


def test_shared_sphere_supersets_every_per_q_sphere():
    """The shared single sphere (radius √cut + |q_max|) must contain
    every flat-FFT index in the per-q spheres."""
    fft_grid = (8, 8, 12)
    bvec = np.diag([0.7, 0.7, 0.3])
    kgrid = (2, 2, 1)
    cutoff = 10.0
    q_int = np.array([(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)], dtype=int)
    q_int_wrapped = np.where(
        q_int > np.asarray(kgrid) / 2, q_int - np.asarray(kgrid), q_int)
    q_frac = q_int_wrapped / np.asarray(kgrid)

    per_q = compute_per_q_bare_coulomb_components(
        fft_grid, bvec, q_frac, cutoff, sys_dim=2)
    shared_idx, _ = compute_bare_coulomb_sphere_idx(
        fft_grid, bvec, kgrid, cutoff, sys_dim=2)

    shared_set = set(int(i) for i in shared_idx)
    for q_i in range(len(q_frac)):
        nk = int(per_q["ngk_per_q"][q_i])
        per_q_set = set(int(i) for i in per_q["sphere_idx_padded"][q_i, :nk])
        assert per_q_set.issubset(shared_set), (
            f"q={q_i} per-q sphere not a subset of shared sphere")


# ---------------------------------------------------------------------------
# 2. Per-q accumulate matches forward FFT + per-q gather.
# ---------------------------------------------------------------------------

def test_accumulate_per_q_sphere_matches_reference():
    """Walk r in chunks via to_rchunk-equivalent path; accumulate via
    per-q sphere; verify equals direct FFT + per-q gather."""
    fft_grid = (4, 4, 6)
    nx, ny, nz = fft_grid
    n_rtot = nx * ny * nz
    n_q, n_rmu = 3, 2
    ngkmax = 9
    rng = np.random.default_rng(0xC0DE)
    rch_full = jnp.asarray(
        rng.standard_normal((n_q, n_rmu, n_rtot))
        + 1j * rng.standard_normal((n_q, n_rmu, n_rtot)),
        dtype=jnp.complex128,
    )
    # Per-q sphere of size ngkmax with distinct indices per q.
    sphere_per_q = np.zeros((n_q, ngkmax), dtype=np.int32)
    for q in range(n_q):
        sphere_per_q[q] = (np.arange(ngkmax) * (q + 1)) % n_rtot

    # Walk r in chunks and accumulate.
    from jax.sharding import Mesh
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                axis_names=('x', 'y'))
    acc = jnp.zeros((n_q, n_rmu, ngkmax), dtype=jnp.complex128)
    chunk = 8
    for r0 in range(0, n_rtot, chunk):
        r1 = min(r0 + chunk, n_rtot)
        acc = accumulate_rchunk_to_gflat(
            rchunk=rch_full[:, :, r0:r1], gflat_acc=acc,
            mesh=mesh, fft_grid=fft_grid, r0=r0,
            sphere_idx=sphere_per_q,
            qvec_frac=None, norm='backward',
        )

    # Reference: full FFT then per-q gather on flattened axis.
    G_full = np.asarray(jnp.fft.fftn(
        rch_full.reshape(n_q, n_rmu, nx, ny, nz),
        axes=(-3, -2, -1), norm='backward')).reshape(n_q, n_rmu, n_rtot)
    ref = np.take_along_axis(G_full, sphere_per_q[:, None, :], axis=-1)
    np.testing.assert_allclose(
        np.asarray(acc), ref, atol=1e-10, rtol=1e-10)


# ---------------------------------------------------------------------------
# 3. Header round-trip.
# ---------------------------------------------------------------------------

def test_isdf_header_g_flat_round_trip(tmp_path):
    path = tmp_path / "header.h5"
    gv = np.zeros((4, 3, 7), dtype=np.int32)
    gv[..., 0] = 0
    gv[..., -1] = -8                            # sentinel-ish
    ngk = np.array([7, 6, 5, 7], dtype=np.int32)

    hdr = IsdfHeader.build(
        r_mu_fft_idx=np.arange(12, dtype=np.int32).reshape(4, 3) % 8,
        fft_grid=(8, 8, 8),
        density='scalar', vertex_mu_L=0,
        zeta_is_done=True, zeta_layout='G_flat',
        gvec_components=gv, ngk_per_q=ngk,
        zeta_cutoff_ry=42.0,
    )
    with h5py.File(path, 'w') as f:
        pass
    write_isdf_header(str(path), hdr, mode='a')
    rb = read_isdf_header(str(path))
    assert rb.zeta_layout == 'G_flat'
    assert rb.zeta_cutoff_ry == 42.0
    np.testing.assert_array_equal(rb.gvec_components, gv)
    np.testing.assert_array_equal(rb.ngk_per_q, ngk)
    assert rb.ngkmax == 7


def test_isdf_header_g_flat_requires_metadata():
    with pytest.raises(ValueError, match="requires gvec_components"):
        IsdfHeader.build(
            r_mu_fft_idx=np.zeros((1, 3), dtype=np.int32),
            fft_grid=(4, 4, 4), density='scalar', vertex_mu_L=0,
            zeta_layout='G_flat',
        )


def test_isdf_header_r_space_drops_metadata():
    """r-space build leaves G-flat fields as None even if passed
    (defensive: caller is expected to omit them; round-trip stays
    clean)."""
    hdr = IsdfHeader.build(
        r_mu_fft_idx=np.zeros((1, 3), dtype=np.int32),
        fft_grid=(4, 4, 4), density='scalar', vertex_mu_L=0,
        zeta_layout='r_space',
    )
    assert hdr.gvec_components is None
    assert hdr.ngk_per_q is None
    assert hdr.zeta_cutoff_ry is None
    assert hdr.ngkmax is None
