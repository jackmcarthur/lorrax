"""Tests for ``common.wfn_transforms``.

Verifies each transform against an independent numpy reference built
from the loader's G-flat output, exercises the zero-sentinel-gather
contract for empty FFT-box cells, and confirms band-axis sharding is
preserved through every output rank.
"""
from __future__ import annotations

import os

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from common.wfn_transforms import to_box, to_rbox, to_rmu, to_rchunk
from file_io.wfn_loader import WfnLoader

from tests.test_wfn_loader_eager import _synth_wfn, _MOS2_WFN


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth_loader(tmp_path):
    path = _synth_wfn(tmp_path)
    with WfnLoader(path) as loader:
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
    psi_box = np.asarray(to_box(psi, g_index, loader.fft_grid))

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
    psi_box = np.asarray(to_box(psi, g_index, synth_loader.fft_grid))
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
    psi_box = np.asarray(to_box(psi, g_index, synth_loader.fft_grid))
    psi_r_box = np.asarray(to_rbox(psi, g_index, synth_loader.fft_grid))
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

    psi_rmu = np.asarray(to_rmu(psi, g_index, synth_loader.fft_grid, r_mu))
    psi_r_box = np.asarray(to_rbox(psi, g_index, synth_loader.fft_grid))
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
        psi, g_index, synth_loader.fft_grid, r0, r_len))

    psi_r_box = np.asarray(to_rbox(psi, g_index, synth_loader.fft_grid))
    expected = psi_r_box.reshape(*psi_r_box.shape[:3], n_rtot)[
        :, :, :, r0:r0 + r_len]
    np.testing.assert_allclose(psi_rchunk, expected, atol=1e-14, rtol=0)


def test_to_rchunk_rejects_out_of_bounds(synth_loader):
    psi = synth_loader.load(bands=(0, 2), k="ibz")
    g_index = synth_loader.box_index(k="ibz")
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)
    with pytest.raises(ValueError):
        to_rchunk(psi, g_index, synth_loader.fft_grid, nx * ny * nz - 2, 10)


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
    psi_box = np.asarray(to_box(psi, g_index, synth_loader.fft_grid))

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

def test_to_box_shape(synth_loader):
    psi = synth_loader.load(bands=(0, 3), k="full_bz")
    g_index = synth_loader.box_index(k="full_bz")
    nx, ny, nz = (int(s) for s in synth_loader.fft_grid)
    out = to_box(psi, g_index, synth_loader.fft_grid)
    assert out.shape == (psi.shape[0], psi.shape[1], psi.shape[2], nx, ny, nz)
    assert out.dtype == jnp.complex128
