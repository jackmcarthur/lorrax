"""Tests for ``common.wfn_transforms``.

Verifies each transform against an independent numpy reference built
from the loader's G-flat output, exercises the zero-sentinel-gather
contract for empty FFT-box cells, and confirms band-axis sharding is
preserved through every output rank.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

import common.wfn_transforms as _wfn_transforms
from common.wfn_transforms import (
    to_box, to_rbox, to_rmu, to_rchunk,
    to_rchunk_inner, to_rmu_inner,
    gflat_to_rmu, load_centroids_band_chunked)
from common.meta import Meta
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                                 # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

from wfn_loader import IBZRows, WfnLoader                           # noqa: E402

# The synthetic-WFN builder moved WITH the loader (charter wave 1): it
# writes a ``WFN.h5``, so it belongs to the service that reads one, and the
# service suite has to own it to run from a standalone install.  This is a
# lorrax test of a lorrax module (``common.wfn_transforms`` is registered
# OUT of the service — it is a CONSUMER of the loader, survey Q3), so it
# reaches across for the builder rather than keeping a second copy of a
# 60-line HDF5 layout that would drift the first time the format moved.
#
# The path is inserted HERE rather than relied on from pytest: pytest does
# put a non-package test directory on ``sys.path``, but only once it has
# started collecting that directory, which makes the import order-dependent
# — green when the service suite is collected first and an ImportError when
# it is not.
_SVC_TESTS = str(Path(__file__).resolve().parents[1]
                 / "services" / "wfn_loader" / "tests")
if _SVC_TESTS not in sys.path:
    sys.path.insert(0, _SVC_TESTS)
from test_wfn_loader_contract import _synth_wfn, _MOS2_WFN   # noqa: E402


# Every public transform in this module takes a ``mesh`` kwarg.  For
# the single-device test bench we use a 1×1 mesh — same code path as
# multi-rank production, no None branches anywhere.
MESH = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
             axis_names=('x', 'y'))
_GNPPM_WFN = (Path(__file__).resolve().parent
               / "regression" / "gnppm_debug" / "WFN.h5")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_loader(tmp_path):
    path = _synth_wfn(tmp_path)
    with WfnLoader(path) as loader:
        yield loader


@pytest.fixture
def synth_loader_with_mesh(tmp_path):
    path = _synth_wfn(tmp_path)
    with WfnLoader(path, mesh=MESH, backend="eager") as loader:
        yield loader


@pytest.fixture
def mos2_loader():
    if not os.path.exists(_MOS2_WFN):
        pytest.skip("MoS2 3x3 WFN not present")
    with WfnLoader(_MOS2_WFN) as loader:
        yield loader


# ---------------------------------------------------------------------------
# Independent reference: numpy scatter
# ---------------------------------------------------------------------------

def _np_scatter_to_box(
    psi: np.ndarray,
    gvecs: np.ndarray,
    ngk_valid: np.ndarray,
    fft_grid: tuple[int, int, int],
) -> np.ndarray:
    """Direct (slow) reference scatter: write each valid (k, g) entry
    into its (nx, ny, nz) FFT-box cell.  Output is ``(n_k, nb, ns, nx,
    ny, nz)``.  Pad rows of psi are zero by contract so we can scatter
    only the valid prefix per k."""
    n_k, nb, ns, _ = psi.shape
    nx, ny, nz = fft_grid
    out = np.zeros((n_k, nb, ns, nx, ny, nz), dtype=np.complex128)
    fft_grid_np = np.asarray(fft_grid, dtype=np.int64)
    # NumPy fancy indexing with three integer-array indices in the
    # trailing positions moves the broadcast axis to the front, so we
    # write the per-k box one G-vector at a time to keep axis order
    # explicit.
    for k in range(n_k):
        n = int(ngk_valid[k])
        gv = gvecs[k, :n] % fft_grid_np[None, :]
        for g in range(n):
            out[k, :, :, gv[g, 0], gv[g, 1], gv[g, 2]] = psi[k, :, :, g]
    return out


# ---------------------------------------------------------------------------
# to_box
# ---------------------------------------------------------------------------

def _check_to_box(loader, k_spec):
    b_hi = min(4, int(loader.nbands))
    psi = loader.load(bands=(0, b_hi), k=k_spec)
    g_index = loader.box_index(k=k_spec)
    psi_box = np.asarray(to_box(psi, g_index, loader.fft_grid, mesh=MESH))

    psi_ref = _np_scatter_to_box(
        np.asarray(psi),
        loader.gvecs(k=k_spec),
        loader.ngk_valid(k=k_spec),
        tuple(int(s) for s in loader.fft_grid),
    )
    np.testing.assert_array_equal(psi_box, psi_ref)


def test_to_box_ibz_synth(synth_loader):
    _check_to_box(synth_loader, k_spec="ibz")


def test_to_box_full_bz_synth(synth_loader):
    _check_to_box(synth_loader, k_spec="full_bz")


def test_to_box_ibz_mos2(mos2_loader):
    _check_to_box(mos2_loader, k_spec="ibz")


def test_to_box_full_bz_mos2(mos2_loader):
    _check_to_box(mos2_loader, k_spec="full_bz")


def test_to_box_empty_cells_are_zero(synth_loader):
    """FFT-box cells outside the G-sphere must be exactly zero."""
    b_hi = min(3, int(synth_loader.nbands))
    psi = synth_loader.load(bands=(0, b_hi), k="ibz")
    g_index = synth_loader.box_index(k="ibz")
    psi_box = np.asarray(to_box(psi, g_index, synth_loader.fft_grid, mesh=MESH))
    ngkmax = int(synth_loader.ngkmax)

    # An FFT-box cell is empty iff g_index[k, x, y, z] == ngkmax.
    sentinel_mask = (np.asarray(g_index) == ngkmax)
    # Broadcast to (n_k, nb, ns, nx, ny, nz).
    sentinel_mask_b = sentinel_mask[:, None, None, :, :, :]
    assert np.all(np.abs(psi_box[np.broadcast_to(
        sentinel_mask_b, psi_box.shape)]) == 0)


# ---------------------------------------------------------------------------
# to_rbox = IFFT(to_box)
# ---------------------------------------------------------------------------

def test_to_rbox_matches_ifft_of_to_box(synth_loader):
    psi = synth_loader.load(bands=(0, 3), k="full_bz")
    g_index = synth_loader.box_index(k="full_bz")
    psi_box = np.asarray(to_box(psi, g_index, synth_loader.fft_grid, mesh=MESH))
    psi_r_box = np.asarray(to_rbox(psi, g_index, synth_loader.fft_grid, mesh=MESH))
    expected = np.fft.ifftn(psi_box, axes=(-3, -2, -1))
    np.testing.assert_allclose(psi_r_box, expected, atol=1e-13, rtol=0)


# ---------------------------------------------------------------------------
# to_rmu vs index of to_rbox
# ---------------------------------------------------------------------------

def test_to_rmu_matches_rbox_take(synth_loader):
    psi = synth_loader.load(bands=(0, 3), k="full_bz")
    g_index = synth_loader.box_index(k="full_bz")
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)

    rng = np.random.default_rng(1)
    n_rmu = 5
    r_mu = np.stack([
        rng.integers(0, nx, size=n_rmu),
        rng.integers(0, ny, size=n_rmu),
        rng.integers(0, nz, size=n_rmu),
    ], axis=-1).astype(np.int32)

    psi_rmu = np.asarray(to_rmu(psi, g_index, synth_loader.fft_grid, r_mu, mesh=MESH))
    psi_r_box = np.asarray(to_rbox(psi, g_index, synth_loader.fft_grid, mesh=MESH))
    expected = psi_r_box[:, :, :, r_mu[:, 0], r_mu[:, 1], r_mu[:, 2]]
    np.testing.assert_allclose(psi_rmu, expected, atol=1e-14, rtol=0)


# ---------------------------------------------------------------------------
# to_rchunk vs flat-r slice of to_rbox
# ---------------------------------------------------------------------------

def test_to_rchunk_matches_rbox_flat_slab(synth_loader):
    psi = synth_loader.load(bands=(0, 3), k="full_bz")
    g_index = synth_loader.box_index(k="full_bz")
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)
    n_rtot = nx * ny * nz

    r0, r_len = nx * ny + 2, 12  # arbitrary slab
    psi_rchunk = np.asarray(to_rchunk(
        psi, g_index, synth_loader.fft_grid, r0, r_len, mesh=MESH))

    psi_r_box = np.asarray(to_rbox(psi, g_index, synth_loader.fft_grid, mesh=MESH))
    expected = psi_r_box.reshape(*psi_r_box.shape[:3], n_rtot)[
        :, :, :, r0:r0 + r_len]
    np.testing.assert_allclose(psi_rchunk, expected, atol=1e-14, rtol=0)


def test_to_rchunk_rejects_out_of_bounds(synth_loader):
    psi = synth_loader.load(bands=(0, 2), k="ibz")
    g_index = synth_loader.box_index(k="ibz")
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)
    with pytest.raises(ValueError):
        to_rchunk(psi, g_index, synth_loader.fft_grid, nx * ny * nz - 2, 10, mesh=MESH)


# ---------------------------------------------------------------------------
# to_rchunk_inner (Path D §4b scaffolding) — must match to_rchunk
# numerically when called on the same per-rank-local inputs.  Tested
# both without and with the Bloch phase, since the two paths take
# different branches inside to_rchunk's shard_map body.
# ---------------------------------------------------------------------------

def test_to_rchunk_inner_matches_to_rchunk_no_phase(synth_loader):
    """to_rchunk_inner is the shard_map-less body of to_rchunk.  On a
    1×1 mesh the wrapper is trivial, so per-rank-local output must
    agree to floating point."""
    psi = synth_loader.load(bands=(0, 3), k="full_bz")
    g_index = synth_loader.box_index(k="full_bz")
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)
    r0, r_len = nx * ny + 2, 12

    psi_inner = np.asarray(to_rchunk_inner(
        psi, jnp.asarray(g_index, dtype=jnp.int32),
        synth_loader.fft_grid, r0, r_len, norm="backward"))
    psi_wrapped = np.asarray(to_rchunk(
        psi, g_index, synth_loader.fft_grid, r0, r_len, mesh=MESH,
        norm="backward"))
    np.testing.assert_allclose(psi_inner, psi_wrapped, atol=1e-14, rtol=0)


def test_to_rchunk_inner_matches_to_rchunk_with_phase(synth_loader):
    """Bloch-phase branch — apply_bloch_phase_on_slice should run
    identically inside vs outside the shard_map wrapper."""
    psi = synth_loader.load(bands=(0, 3), k="full_bz")
    g_index = synth_loader.box_index(k="full_bz")
    nk = int(psi.shape[0])
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)
    r0, r_len = nx * ny + 2, 12

    rng = np.random.default_rng(0)
    kvecs_frac = rng.uniform(-0.5, 0.5, size=(nk, 3)).astype(np.float64)

    psi_inner = np.asarray(to_rchunk_inner(
        psi, jnp.asarray(g_index, dtype=jnp.int32),
        synth_loader.fft_grid, r0, r_len,
        kvecs_frac=jnp.asarray(kvecs_frac, dtype=jnp.float64),
        norm="backward"))
    psi_wrapped = np.asarray(to_rchunk(
        psi, g_index, synth_loader.fft_grid, r0, r_len, mesh=MESH,
        kvecs_frac=kvecs_frac, norm="backward"))
    np.testing.assert_allclose(psi_inner, psi_wrapped, atol=1e-14, rtol=0)


def test_to_rchunk_inner_traced_r0(synth_loader):
    """The r0 arg must accept a traced scalar — the eventual Path D
    consumer will pass ``r_start_dyn`` (a jit input) here."""
    psi = synth_loader.load(bands=(0, 3), k="full_bz")
    g_index = synth_loader.box_index(k="full_bz")
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)
    r_len = 12

    @jax.jit
    def fn(psi_, g_index_, r0_):
        return to_rchunk_inner(psi_, g_index_, synth_loader.fft_grid,
                                r0_, r_len, norm="backward")

    r0_val = nx * ny + 2
    r0 = jnp.int32(r0_val)
    out = np.asarray(fn(psi, jnp.asarray(g_index, dtype=jnp.int32), r0))
    expected = np.asarray(to_rchunk(
        psi, g_index, synth_loader.fft_grid, r0_val, r_len, mesh=MESH,
        norm="backward"))
    np.testing.assert_allclose(out, expected, atol=1e-14, rtol=0)


# ---------------------------------------------------------------------------
# to_rmu_inner (Defect 3 mirror scaffolding) — pure-jax body of to_rmu,
# must match the shard_map-wrapped version on a 1×1 mesh.
# ---------------------------------------------------------------------------

def test_to_rmu_inner_matches_to_rmu_no_phase(synth_loader):
    """to_rmu_inner is the shard_map-less body of to_rmu.  On a 1×1
    mesh the wrapper is a no-op, so per-rank-local output must agree
    to floating point."""
    psi = synth_loader.load(bands=(0, 3), k="full_bz")
    g_index = synth_loader.box_index(k="full_bz")
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)

    rng = np.random.default_rng(11)
    n_rmu = 7
    r_mu = np.stack([
        rng.integers(0, nx, size=n_rmu),
        rng.integers(0, ny, size=n_rmu),
        rng.integers(0, nz, size=n_rmu),
    ], axis=1).astype(np.int32)

    psi_inner = np.asarray(to_rmu_inner(
        psi, jnp.asarray(g_index, dtype=jnp.int32),
        synth_loader.fft_grid, jnp.asarray(r_mu, dtype=jnp.int32),
        norm="backward"))
    psi_wrapped = np.asarray(to_rmu(
        psi, g_index, synth_loader.fft_grid, r_mu, mesh=MESH,
        norm="backward"))
    np.testing.assert_allclose(psi_inner, psi_wrapped, atol=1e-14, rtol=0)


def test_to_rmu_inner_matches_to_rmu_with_phase(synth_loader):
    """Bloch-phase branch — applied to the full FFT box before the
    centroid gather; should run identically inside vs outside the
    shard_map wrapper."""
    psi = synth_loader.load(bands=(0, 3), k="full_bz")
    g_index = synth_loader.box_index(k="full_bz")
    nk = int(psi.shape[0])
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)

    rng = np.random.default_rng(12)
    kvecs_frac = rng.uniform(-0.5, 0.5, size=(nk, 3)).astype(np.float64)
    n_rmu = 9
    r_mu = np.stack([
        rng.integers(0, nx, size=n_rmu),
        rng.integers(0, ny, size=n_rmu),
        rng.integers(0, nz, size=n_rmu),
    ], axis=1).astype(np.int32)

    psi_inner = np.asarray(to_rmu_inner(
        psi, jnp.asarray(g_index, dtype=jnp.int32),
        synth_loader.fft_grid, jnp.asarray(r_mu, dtype=jnp.int32),
        kvecs_frac=jnp.asarray(kvecs_frac, dtype=jnp.float64),
        norm="backward"))
    psi_wrapped = np.asarray(to_rmu(
        psi, g_index, synth_loader.fft_grid, r_mu, mesh=MESH,
        kvecs_frac=kvecs_frac, norm="backward"))
    np.testing.assert_allclose(psi_inner, psi_wrapped, atol=1e-14, rtol=0)


# ---------------------------------------------------------------------------
# gflat_to_rmu (Defect 3 structural fix) — must match the bc-loop +
# concatenate path that lives in load_centroids_band_chunked today.
# Same three-flavour pattern as gflat_to_rchunk: no-phase / with-phase /
# chunked-vs-oneshot.
# ---------------------------------------------------------------------------


def _gflat_to_rmu_reference(
    psi_G, g_index, fft_grid, r_mu, *,
    band_chunks, kvecs_frac=None, norm="ortho",
):
    """Reference path: per-bc ``to_rmu`` calls then ``jnp.concatenate``
    along the band axis — what ``load_centroids_band_chunked`` does
    today (modulo the optional inner k-chunk loop)."""
    parts = []
    for (b_lo, b_hi) in band_chunks:
        parts.append(to_rmu(
            psi_G[:, b_lo:b_hi, :, :], g_index, fft_grid, r_mu,
            mesh=MESH, kvecs_frac=kvecs_frac, norm=norm))
    return jnp.concatenate(parts, axis=1)


def test_gflat_to_rmu_no_phase(synth_loader):
    """One shard_map+scan call equals the bc-loop + concatenate path
    (no Bloch phase) to floating-point precision."""
    nb = min(8, int(synth_loader.nbands))
    psi = synth_loader.load(bands=(0, nb), k="full_bz")
    g_index = synth_loader.box_index(k="full_bz")
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)

    rng = np.random.default_rng(13)
    n_rmu = 13
    r_mu = np.stack([
        rng.integers(0, nx, size=n_rmu),
        rng.integers(0, ny, size=n_rmu),
        rng.integers(0, nz, size=n_rmu),
    ], axis=1).astype(np.int32)

    band_chunks = [(0, 4), (4, nb)]
    ref = np.asarray(_gflat_to_rmu_reference(
        psi, g_index, synth_loader.fft_grid, r_mu,
        band_chunks=band_chunks, kvecs_frac=None, norm="ortho"))

    out = np.asarray(gflat_to_rmu(
        psi, g_index, r_mu, mesh=MESH, fft_grid=synth_loader.fft_grid,
        kvecs_frac=None, norm="ortho", chunk_size=None))
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-12)


def test_gflat_to_rmu_with_phase(synth_loader):
    """Bloch-phase branch — apply_bloch_phase semantics (sign=+1) on
    the centroid-sampled cells must match identically inside the scan
    body."""
    nb = min(8, int(synth_loader.nbands))
    psi = synth_loader.load(bands=(0, nb), k="full_bz")
    g_index = synth_loader.box_index(k="full_bz")
    nk = int(psi.shape[0])
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)

    rng = np.random.default_rng(14)
    kvecs_frac = rng.uniform(-0.5, 0.5, size=(nk, 3)).astype(np.float64)
    n_rmu = 15
    r_mu = np.stack([
        rng.integers(0, nx, size=n_rmu),
        rng.integers(0, ny, size=n_rmu),
        rng.integers(0, nz, size=n_rmu),
    ], axis=1).astype(np.int32)

    band_chunks = [(0, 3), (3, 6), (6, nb)]
    ref = np.asarray(_gflat_to_rmu_reference(
        psi, g_index, synth_loader.fft_grid, r_mu,
        band_chunks=band_chunks, kvecs_frac=kvecs_frac, norm="ortho"))

    out = np.asarray(gflat_to_rmu(
        psi, g_index, r_mu, mesh=MESH, fft_grid=synth_loader.fft_grid,
        kvecs_frac=kvecs_frac, norm="ortho", chunk_size=None))
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-12)


def test_gflat_to_rmu_chunked_matches_oneshot(synth_loader):
    """chunk_size sweep — every choice (incl. one that triggers
    zero-padding) must produce the same output to ULP precision."""
    nb = min(6, int(synth_loader.nbands))
    psi = synth_loader.load(bands=(0, nb), k="full_bz")
    g_index = synth_loader.box_index(k="full_bz")
    nk = int(psi.shape[0])
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)

    rng = np.random.default_rng(15)
    kvecs_frac = rng.uniform(-0.5, 0.5, size=(nk, 3)).astype(np.float64)
    n_rmu = 11
    r_mu = np.stack([
        rng.integers(0, nx, size=n_rmu),
        rng.integers(0, ny, size=n_rmu),
        rng.integers(0, nz, size=n_rmu),
    ], axis=1).astype(np.int32)

    # 1×1 mesh ⇒ nb_local = nb; flat axis N = nk · nb.
    nb_local = nb
    N = nk * nb_local
    one_shot = np.asarray(gflat_to_rmu(
        psi, g_index, r_mu, mesh=MESH, fft_grid=synth_loader.fft_grid,
        kvecs_frac=kvecs_frac, norm="ortho", chunk_size=None))

    for cs in (1, 3, N, N + 1):
        chunked = np.asarray(gflat_to_rmu(
            psi, g_index, r_mu, mesh=MESH, fft_grid=synth_loader.fft_grid,
            kvecs_frac=kvecs_frac, norm="ortho", chunk_size=cs))
        np.testing.assert_allclose(
            chunked, one_shot, rtol=1e-10, atol=1e-12,
            err_msg=f"chunk_size={cs} disagrees with one-shot")


def test_gflat_to_rmu_runtime_k_and_centroid_operands_share_one_family(
        synth_loader):
    """Same-shaped k and centroid values vary without one JIT per tile."""
    for key in list(_wfn_transforms._KERNEL_CACHE):
        if key[0] == "gflat_to_rmu":
            del _wfn_transforms._KERNEL_CACHE[key]

    nb = min(4, int(synth_loader.nbands))
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)
    r_mu_base = np.asarray([
        [0, 0, 0],
        [min(1, nx - 1), min(2, ny - 1), min(3, nz - 1)],
    ], dtype=np.int32)
    kvecs = np.asarray(synth_loader.kvecs(k="full_bz"), dtype=np.float64)
    tile_sharding = NamedSharding(MESH, P(None, ('x', 'y'), None, None))

    for ik in (0, 1):
        r_mu = r_mu_base.copy()
        r_mu[:, 0] = (r_mu[:, 0] + ik) % nx
        r_mu[:, 1] = (r_mu[:, 1] + 2 * ik) % ny
        psi = jax.device_put(
            synth_loader.load(bands=(0, nb), k=[ik]), tile_sharding)
        g_index = synth_loader.box_index_dev(k=[ik], mesh=MESH)
        out = np.asarray(gflat_to_rmu(
            psi, g_index, r_mu, mesh=MESH,
            fft_grid=synth_loader.fft_grid,
            kvecs_frac=jnp.asarray(kvecs[ik:ik + 1]),
            norm="ortho", chunk_size=2))
        ref = np.asarray(to_rmu(
            psi, g_index, synth_loader.fft_grid, r_mu, mesh=MESH,
            kvecs_frac=kvecs[ik:ik + 1], norm="ortho"))
        np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-12)

    families = [
        key for key in _wfn_transforms._KERNEL_CACHE
        if key[0] == "gflat_to_rmu"
    ]
    assert len(families) == 1, [
        (i, values) for i, values in enumerate(zip(*families))
        if len(set(values)) != 1
    ]


def test_streamed_centroid_transfer_matches_bulk(
        synth_loader_with_mesh, monkeypatch):
    """Two-dimensional WFN tiles preserve the public centroid faces."""
    synth_loader = synth_loader_with_mesh
    sym = synth_loader.symmetry()
    nb = min(7, int(synth_loader.nbands))
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)
    r_mu = jnp.asarray([
        [0, 0, 0],
        [min(1, nx - 1), min(2, ny - 1), min(3, nz - 1)],
        [min(2, nx - 1), min(1, ny - 1), min(4, nz - 1)],
    ], dtype=jnp.int32)
    meta = Meta.from_system(
        synth_loader, sym, nval=2, ncond=1, nband=nb,
        n_rmu=int(r_mu.shape[0]), bispinor=False)
    meta.memory_per_device_gb = 1000.0

    bulk_y, bulk_x = load_centroids_band_chunked(
        synth_loader, sym, meta, r_mu, False, MESH, (0, nb),
        band_chunk_size=nb)
    preloaded = synth_loader.load(
        bands=(0, nb), k="full_bz",
        sharding=P(None, ('x', 'y'), None, None))
    reused_y, reused_x = load_centroids_band_chunked(
        synth_loader, sym, meta, r_mu, False, MESH, (0, nb),
        band_chunk_size=4, k_chunk_size=1, psi_G_flat=preloaded)

    np.testing.assert_allclose(
        np.asarray(reused_y), np.asarray(bulk_y), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(reused_x), np.asarray(bulk_x), rtol=1e-10, atol=1e-12)

    for key in list(_wfn_transforms._KERNEL_CACHE):
        if key[0] == "gflat_to_rmu":
            del _wfn_transforms._KERNEL_CACHE[key]
    singleton_groups = synth_loader.full_k_parent_groups()
    assert singleton_groups and all(
        len(children) == 1 for _, children in singleton_groups), (
        "the synthetic deck must pin the no-reuse singleton path")
    singleton_requests = []
    original_load = synth_loader.load

    def _counted_singleton_load(*args, **kwargs):
        singleton_requests.append(kwargs.get("k"))
        return original_load(*args, **kwargs)

    def _forbid_parent_unfold(*_args, **_kwargs):
        raise AssertionError("singleton star dispatched parent unfold")

    monkeypatch.setattr(synth_loader, "load", _counted_singleton_load)
    monkeypatch.setattr(
        synth_loader, "unfold_parent_to_full_k", _forbid_parent_unfold)
    stream_y, stream_x = load_centroids_band_chunked(
        synth_loader, sym, meta, r_mu, False, MESH, (0, nb),
        band_chunk_size=4, k_chunk_size=1)
    np.testing.assert_allclose(
        np.asarray(stream_y), np.asarray(bulk_y), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(stream_x), np.asarray(bulk_x), rtol=1e-10, atol=1e-12)
    families = [
        key for key in _wfn_transforms._KERNEL_CACHE
        if key[0] == "gflat_to_rmu"
    ]
    assert len(families) == 1
    # One direct full-child request per singleton group: no IBZRows raw
    # parent carrier and no separate parent unfold for a star of size one.
    # The production loop is band-major then parent-major.
    expected_singletons = [
        [int(children[0])]
        for _b_rel in range(0, nb, 4)
        for _, children in singleton_groups
    ]
    assert singleton_requests == expected_singletons
    assert all(not isinstance(req, IBZRows) for req in singleton_requests)


def test_k1_stream_reuses_real_multichild_parents_and_keeps_one_k_fft(
        monkeypatch):
    """The production schedule reduces real WFN reads, not FFT count.

    ``gnppm`` has non-singleton full-k stars.  The strict inequalities and
    the asserted raw-IBZ request vocabulary make this a non-tautological
    parent-reuse test: a loop that merely renames every full child a parent
    still performs ``nk_full`` reads and fails.  Conversely, preserving one
    FFT call per child pins the one-k workspace and the deliberately
    conservative transform path.
    """
    from types import SimpleNamespace

    if not _GNPPM_WFN.exists():
        pytest.skip("checked-in gnppm WFN absent")
    with WfnLoader(
            str(_GNPPM_WFN), mesh=MESH, backend="eager") as loader:
        sym = loader.symmetry()
        groups = loader.full_k_parent_groups()
        nk_full = int(sym.nk_tot)
        assert sum(len(children) for _, children in groups) == nk_full
        assert len(groups) < nk_full
        assert any(len(children) > 1 for _, children in groups)

        nb = 2
        nx, ny, nz = (int(v) for v in loader.fft_grid)
        r_mu = jnp.asarray([
            [0, 0, 0],
            [min(1, nx - 1), min(2, ny - 1), min(3, nz - 1)],
        ], dtype=jnp.int32)
        meta = SimpleNamespace(
            nk_tot=nk_full,
            nspinor=int(loader.nspinor),
            fft_grid=tuple(int(v) for v in loader.fft_grid),
            memory_per_device_gb=1000.0,
            b_id_4_user=nb,
        )

        # Independent full-k carrier through the established bulk path.
        preloaded = loader.load(
            bands=(0, nb), k="full_bz",
            sharding=P(None, ("x", "y"), None, None))
        ref_y, ref_x = load_centroids_band_chunked(
            loader, sym, meta, r_mu, False, MESH, (0, nb),
            band_chunk_size=nb, psi_G_flat=preloaded)

        load_requests = []
        fft_k_extents = []
        phase_requests = []
        original_load = loader.load
        original_fft = _wfn_transforms.gflat_to_rmu
        original_phase_rows = loader._host_phase_rows_for_full_k

        def _counted_load(*args, **kwargs):
            load_requests.append(kwargs.get("k"))
            return original_load(*args, **kwargs)

        def _counted_fft(*args, **kwargs):
            fft_k_extents.append(int(args[0].shape[0]))
            return original_fft(*args, **kwargs)

        def _counted_phase_rows(rows):
            phase_requests.append(tuple(int(v) for v in np.asarray(rows)))
            return original_phase_rows(rows)

        monkeypatch.setattr(loader, "load", _counted_load)
        monkeypatch.setattr(
            _wfn_transforms, "gflat_to_rmu", _counted_fft)
        monkeypatch.setattr(
            loader, "_host_phase_rows_for_full_k", _counted_phase_rows)
        got_y, got_x = load_centroids_band_chunked(
            loader, sym, meta, r_mu, False, MESH, (0, nb),
            band_chunk_size=nb, k_chunk_size=1)

        assert "phase_per_full" not in loader._ensure_phdf5_static()
        assert len(load_requests) == len(groups)
        for request, (parent, children) in zip(load_requests, groups):
            if len(children) == 1:
                assert request == [int(children[0])]
                assert not isinstance(request, IBZRows)
            else:
                assert isinstance(request, IBZRows)
                assert request.rows == (int(parent),)
        raw_parent_rows = [
            request.rows for request in load_requests
            if isinstance(request, IBZRows)]
        assert len(raw_parent_rows) == len(set(raw_parent_rows))
        assert len(fft_k_extents) == nk_full
        assert fft_k_extents == [1] * nk_full
        expected_phase_children = [
            int(child)
            for _, children in groups if len(children) > 1
            for child in children
        ]
        assert phase_requests == [(child,) for child in expected_phase_children]
        np.testing.assert_allclose(
            np.asarray(got_y), np.asarray(ref_y), rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(got_x), np.asarray(ref_x), rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Pad-row hygiene: pad ψ rows of zero must NOT corrupt the box
# (the gather contract guarantees this — pad indices in g_index would
# only ever point at sentinel + zero slot)
# ---------------------------------------------------------------------------

def test_pad_rows_dont_leak_into_box(synth_loader):
    """Force the band-pad case (replicated, no mesh → no band pad), and
    G-pad case (ngk[k] < ngkmax for each k); confirm the FFT-box
    contents at G-positions corresponding to valid coefficients agree
    with the raw IBZ slab, and pad columns are inert."""
    psi = synth_loader.load(bands=(0, 4), k="ibz")
    g_index = synth_loader.box_index(k="ibz")
    psi_box = np.asarray(to_box(psi, g_index, synth_loader.fft_grid, mesh=MESH))

    # Reconstruct the box from raw IBZ coefficients (bypassing loader's
    # padding logic).
    gvecs_per_k = [synth_loader.get_gvec_nk(ik)
                    for ik in range(int(synth_loader.nkpts))]
    ref = _np_scatter_to_box(
        np.asarray(psi)[:, :, :, : int(synth_loader.ngkmax)],
        synth_loader.gvecs(k="ibz"),
        synth_loader.ngk_valid(k="ibz"),
        tuple(int(s) for s in synth_loader.fft_grid),
    )
    np.testing.assert_array_equal(psi_box, ref)


# ---------------------------------------------------------------------------
# Shape sanity for non-trivial nspinor and uneven k counts
# ---------------------------------------------------------------------------

def test_apply_bloch_phase_matches_4d_reference():
    """The separable 1D-factor application of exp(2πi k·r) must
    byte-match an explicit 4D-construction reference."""
    from common.wfn_transforms import apply_bloch_phase

    rng = np.random.default_rng(7)
    n_k, nb, ns = 3, 2, 2
    nx, ny, nz = 5, 7, 4
    psi_r_box = (rng.standard_normal((n_k, nb, ns, nx, ny, nz))
                 + 1j * rng.standard_normal((n_k, nb, ns, nx, ny, nz)))
    kvecs = rng.standard_normal((n_k, 3))

    # Reference: explicit 4D phase (the implementation we removed).
    fx = np.arange(nx) / nx
    fy = np.arange(ny) / ny
    fz = np.arange(nz) / nz
    phase4d = np.exp(
        2j * np.pi * (
            kvecs[:, 0, None, None, None] * fx[None, :, None, None]
            + kvecs[:, 1, None, None, None] * fy[None, None, :, None]
            + kvecs[:, 2, None, None, None] * fz[None, None, None, :]
        )
    )
    expected = psi_r_box * phase4d[:, None, None, :, :, :]

    got = np.asarray(apply_bloch_phase(
        jnp.asarray(psi_r_box), jnp.asarray(kvecs), (nx, ny, nz)))
    np.testing.assert_allclose(got, expected, atol=1e-13, rtol=0)


def test_to_box_shape(synth_loader):
    psi = synth_loader.load(bands=(0, 3), k="full_bz")
    g_index = synth_loader.box_index(k="full_bz")
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)
    out = to_box(psi, g_index, synth_loader.fft_grid, mesh=MESH)
    assert out.shape == (psi.shape[0], psi.shape[1], psi.shape[2], nx, ny, nz)
    assert out.dtype == jnp.complex128


# ---------------------------------------------------------------------------
# get_enk_bandrange: the nspinor default comes from the file, not from 2
# ---------------------------------------------------------------------------

def test_get_enk_bandrange_default_nspinor_is_read_from_the_wfn():
    """``nspinor=2`` as a hardcoded DEFAULT was silent-wrong on an
    nspinor=1 file: nspinor widths the WEIGHTS axis only (one
    ``np.repeat``), so an omitted argument mis-widthed the weights while
    ``enk`` — which every current defaulting caller keeps — never moved.
    Pinned here: the default equals the explicit ``wfn.nspinor`` call
    exactly, an explicit override still wins (the pre-fix behaviour, so
    the GW callers passing ``meta.nspinor`` are untouched), and ``enk``
    is nspinor-independent.
    """
    from types import SimpleNamespace

    from common.wfn_transforms import get_enk_bandrange

    en = np.array([[[0.0, 1.0, 2.0, 3.0]]])       # (nspin=1, nk_irr=1, nb=4)
    wfn = SimpleNamespace(energies=en, efermi=1.5, nspinor=1)
    sym = SimpleNamespace(irr_idx_k=np.array([0, 0]))

    enk_d, w_d = get_enk_bandrange(wfn, sym, (0, 4), (1, 3))
    enk_1, w_1 = get_enk_bandrange(wfn, sym, (0, 4), (1, 3), nspinor=1)
    assert w_d.shape == (2, 4 * 1), (
        f"default weights width {w_d.shape}: the default did not follow "
        f"wfn.nspinor=1")
    np.testing.assert_array_equal(np.asarray(w_d), np.asarray(w_1))
    np.testing.assert_array_equal(np.asarray(enk_d), np.asarray(enk_1))

    # Explicit override beats the file (bispinor callers pass 4).
    enk_4, w_4 = get_enk_bandrange(wfn, sym, (0, 4), (1, 3), nspinor=4)
    assert w_4.shape == (2, 4 * 4)
    # RED-TWIN arm: enk must be identical across nspinor, or the cells
    # above would be pinning the wrong claim.
    np.testing.assert_array_equal(np.asarray(enk_d), np.asarray(enk_4))
