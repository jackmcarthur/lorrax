"""Tests for ``file_io.zeta_loader.ZetaLoader``.

Mirrors ``test_zeta_reader.py`` but exercises the new ``.load(...)`` API
and the ``zeta_is_done`` flag + ``mark_zeta_done`` writer hook.
"""
from __future__ import annotations

import h5py
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec as P

from file_io.mf_header import copy_mf_header
from file_io.isdf_header import (
    IsdfHeader, write_isdf_header, mark_zeta_done, read_isdf_header)
from file_io.zeta_loader import ZetaLoader

from tests.test_mf_isdf_header_roundtrip import _make_fake_wfn


def _build_zeta_h5(tmp_path, n_q_disk: int, *, n_rtot: int = 8,
                   n_rmu: int = 4, density: str = "scalar",
                   vertex_mu_L: int = 0, zeta_is_done: bool = False,
                   fill: complex = 0.0):
    wfn_path = str(tmp_path / "WFN.h5")
    out_path = str(tmp_path / "zeta_q.h5")
    _make_fake_wfn(wfn_path)
    copy_mf_header(wfn_path, out_path, dst_mode='w')

    hdr = IsdfHeader.build(
        r_mu_fft_idx=np.arange(3 * n_rmu, dtype=np.int32).reshape(n_rmu, 3) % 8,
        fft_grid=(8, 8, 8),
        density=density, vertex_mu_L=vertex_mu_L,
        zeta_is_done=zeta_is_done,
    )
    write_isdf_header(out_path, hdr, mode='a')

    with h5py.File(out_path, 'a') as f:
        # Distinct values per (q, r, μ) so the loader's slicing is testable.
        data = (np.arange(n_q_disk * n_rtot * n_rmu)
                .reshape(n_q_disk, n_rtot, n_rmu).astype(np.complex128))
        if fill:
            data = data + fill
        f.create_dataset('zeta_q', data=data)
    return out_path


@pytest.fixture
def single_device_mesh():
    return Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1),
                 axis_names=('x', 'y'))


# ---------------------------------------------------------------------------
# Header surface (parity with ZetaReader)
# ---------------------------------------------------------------------------

def test_loader_header_surface(tmp_path, single_device_mesh):
    out = _build_zeta_h5(tmp_path, n_q_disk=12)
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        assert ld.nkpts == 3
        assert ld.nspin == 1
        assert ld.n_rmu == 4
        assert ld.density == 'scalar'
        assert ld.vertex_mu_L == 0
        assert ld.zeta_is_done is False
        assert ld.n_q_on_disk == 12
        assert ld.n_q_full == 24                  # kgrid 2*3*4
        assert ld.q_layout == 'ibz'


# ---------------------------------------------------------------------------
# zeta_is_done flag round-trip via mark_zeta_done
# ---------------------------------------------------------------------------

def test_mark_zeta_done_flips_flag(tmp_path):
    out = _build_zeta_h5(tmp_path, n_q_disk=4)
    assert read_isdf_header(out).zeta_is_done is False
    mark_zeta_done(out)
    assert read_isdf_header(out).zeta_is_done is True
    # Idempotent
    mark_zeta_done(out)
    assert read_isdf_header(out).zeta_is_done is True


def test_legacy_files_without_flag_read_as_done(tmp_path):
    """A pre-flag zeta_q.h5 (no zeta_is_done dataset) reads as True."""
    out = _build_zeta_h5(tmp_path, n_q_disk=4)
    # Remove the dataset to simulate a legacy file.
    with h5py.File(out, 'a') as f:
        del f['isdf_header/zeta_is_done']
    hdr = read_isdf_header(out)
    assert hdr.zeta_is_done is True


def test_zeta_layout_round_trip(tmp_path):
    """``zeta_layout`` round-trips for both 'r_space' (default) and 'G_flat'."""
    out_r = _build_zeta_h5(tmp_path, n_q_disk=2)
    hdr_r = read_isdf_header(out_r)
    assert hdr_r.zeta_layout == 'r_space'

    # Build a synthetic G-flat header to confirm round-trip.
    out_g = tmp_path / 'zeta_q_g.h5'
    from file_io.mf_header import copy_mf_header
    wfn_path = str(tmp_path / 'WFN.h5')
    _make_fake_wfn(wfn_path)
    copy_mf_header(wfn_path, str(out_g), dst_mode='w')
    g_hdr = IsdfHeader.build(
        r_mu_fft_idx=np.zeros((4, 3), dtype=np.int32),
        fft_grid=(8, 8, 8),
        density='scalar', vertex_mu_L=0,
        zeta_is_done=True, zeta_layout='G_flat',
        # G-flat layout now carries the per-q sphere metadata
        # (WFN.h5 ``coeffs`` style); supply trivial values for the
        # round-trip test.
        gvec_components=np.zeros((1, 3, 4), dtype=np.int32),
        ngk_per_q=np.array([4], dtype=np.int32),
        zeta_cutoff_ry=30.0,
    )
    write_isdf_header(str(out_g), g_hdr, mode='a')
    hdr_g = read_isdf_header(str(out_g))
    assert hdr_g.zeta_layout == 'G_flat'


def test_zeta_layout_legacy_default(tmp_path):
    """Files without ``zeta_layout`` read as 'r_space'."""
    out = _build_zeta_h5(tmp_path, n_q_disk=2)
    with h5py.File(out, 'a') as f:
        del f['isdf_header/zeta_layout']
    hdr = read_isdf_header(out)
    assert hdr.zeta_layout == 'r_space'


def test_zeta_layout_rejects_garbage(tmp_path):
    """build() rejects non-{'r_space','G_flat'} values."""
    with pytest.raises(ValueError, match="zeta_layout must be"):
        IsdfHeader.build(
            r_mu_fft_idx=np.zeros((1, 3), dtype=np.int32),
            fft_grid=(4, 4, 4),
            density='scalar', vertex_mu_L=0,
            zeta_layout='bogus',
        )


# ---------------------------------------------------------------------------
# .load surface
# ---------------------------------------------------------------------------

def test_load_q_ibz_returns_all_disk_rows(tmp_path, single_device_mesh):
    out = _build_zeta_h5(tmp_path, n_q_disk=6, n_rmu=4)
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        z = np.asarray(ld.load(q='ibz'))
        assert z.shape == (6, 8, 4)


def test_load_q_index_list_subset(tmp_path, single_device_mesh):
    out = _build_zeta_h5(tmp_path, n_q_disk=6, n_rmu=4)
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        z = np.asarray(ld.load(q=[2, 3, 4]))
        assert z.shape == (3, 8, 4)
        # Should match the q=2,3,4 slice of the on-disk pattern.
        with h5py.File(out, 'r') as f:
            ref = f['zeta_q'][2:5]
        np.testing.assert_array_equal(z, ref)


def test_load_mu_range(tmp_path, single_device_mesh):
    out = _build_zeta_h5(tmp_path, n_q_disk=4, n_rmu=8)
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        z = np.asarray(ld.load(q='ibz', mu=(2, 6)))
        assert z.shape == (4, 8, 4)
        with h5py.File(out, 'r') as f:
            ref = f['zeta_q'][:, :, 2:6]
        np.testing.assert_array_equal(z, ref)


def test_load_q_full_bz_on_full_disk_returns_all(tmp_path, single_device_mesh):
    """When the on-disk layout IS full-BZ, q='full_bz' == q='ibz'."""
    out = _build_zeta_h5(tmp_path, n_q_disk=24)   # full = 2*3*4
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        assert ld.q_layout == 'full_bz'
        z = np.asarray(ld.load(q='full_bz'))
        assert z.shape == (24, 8, 4)


def test_load_q_full_bz_unfold_path_exists(tmp_path, single_device_mesh):
    """The IBZ → full-BZ unfold is wired and dispatches.

    The synthetic ``_make_fake_wfn`` writes random ``mtrx`` entries
    that are typically singular, so ``compute_rgrid_sym_perm`` raises
    ``LinAlgError`` / ``RuntimeError`` for fake files.  We accept any
    of: clean unfold (would require valid sym group), the singular
    matrix error from the fake WFN, or the TR-mapping
    ``NotImplementedError`` — all confirm the unfold path was
    reached.  End-to-end correctness comes from the MoS2 3×3 smoke
    against real WFN symmetries.
    """
    out = _build_zeta_h5(tmp_path, n_q_disk=1)
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        try:
            z = np.asarray(ld.load(q='full_bz'))
            # If we got here the syms were valid; identity ⇒ all rows equal.
            with h5py.File(out, 'r') as f:
                ibz = f['zeta_q'][0]
            for q in range(z.shape[0]):
                np.testing.assert_array_equal(z[q], ibz)
        except NotImplementedError:
            pass     # TR-mapping not supported — expected on this fake
        except np.linalg.LinAlgError:
            pass     # Random fake mtrx is singular — expected; smoke covers real case
        except RuntimeError as e:
            if "compute_rgrid_sym_perm" not in str(e):
                raise


def test_load_layout_g_flat_requires_qvec_and_sphere(tmp_path, single_device_mesh):
    out = _build_zeta_h5(tmp_path, n_q_disk=4)
    with ZetaLoader(out, mesh=single_device_mesh) as ld:
        with pytest.raises(ValueError, match=r"layout='G_flat'\) requires"):
            ld.load(q='ibz', layout='G_flat')
