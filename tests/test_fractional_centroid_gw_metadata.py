"""GW reuse compares the consumed fractional basis without integer coercion."""

import h5py
import numpy as np
import pytest

from file_io.isdf_header import IsdfHeader, write_isdf_header
from gw.gw_init import _check_zeta_h5_matches_basis, _zeta_reuse_ok


def _zeta(tmp_path, positions, kind='fractional', extent=None):
    path = tmp_path / 'zeta.h5'
    coordinates = (dict(r_mu_crystal=positions) if kind == 'fractional'
                   else dict(r_mu_fft_idx=positions))
    header = IsdfHeader.build(
        **coordinates, coordinate_kind=kind, fft_grid=(50, 50, 50),
        density='scalar', vertex_mu_L=0, zeta_is_done=True,
        fit_provenance='{}')
    write_isdf_header(path, header, mode='w')
    with h5py.File(path, 'a') as stream:
        stream.create_dataset('zeta_q_G', shape=(1, len(positions) if extent is None else extent, 3),
                              dtype=np.complex128)
    return path


def test_reuse_binds_subgrid_positions_and_unwrapped_images(tmp_path):
    positions = np.array([[-.13, .27, 1.31], [.630001, .77, .81]])
    path = _zeta(tmp_path, positions)
    kwargs = dict(coordinate_kind='fractional', n_rmu_expected=2,
                  print_fn=lambda *_: None)
    assert _zeta_reuse_ok(path, '{}', positions, **kwargs)
    for delta in (.000001, 1.):
        changed = positions.copy()
        changed[0, 0] += delta
        assert not _zeta_reuse_ok(path, '{}', changed, **kwargs)


def test_reuse_requires_same_explicit_kind_and_preserves_legacy(tmp_path):
    indices = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    path = _zeta(tmp_path, indices, kind='fft_indices')
    assert _zeta_reuse_ok(path, '{}', indices, print_fn=lambda *_: None)
    assert not _zeta_reuse_ok(path, '{}', indices.astype(float),
                             coordinate_kind='fractional', print_fn=lambda *_: None)
    path = _zeta(tmp_path, indices.astype(float), kind='fractional')
    assert not _zeta_reuse_ok(path, '{}', indices, print_fn=lambda *_: None)


def test_guard_checks_fractional_header_extent_without_fft_bounds(tmp_path):
    positions = np.array([[-2.13, .27, 52.31], [.630001, .77, .81]])
    path = _zeta(tmp_path, positions)
    _check_zeta_h5_matches_basis(path, 2, fft_grid=(50, 50, 50), print_fn=lambda *_: None)
    path = _zeta(tmp_path, positions, extent=3)
    with pytest.raises(ValueError, match='header and the ζ dataset'):
        _check_zeta_h5_matches_basis(path, 3, fft_grid=(50, 50, 50), print_fn=lambda *_: None)
