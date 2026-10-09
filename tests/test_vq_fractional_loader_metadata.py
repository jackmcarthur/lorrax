"""Grouped V routing reads coordinate metadata from its actual first loader."""

from types import SimpleNamespace

import numpy as np
import pytest

from gw import v_q_g_flat


@pytest.mark.parametrize('metadata, expected', [
    ({}, 'fft_indices'),
    ({'coordinate_kind': 'fractional'}, 'fractional'),
    ({'mu_basis': SimpleNamespace(coordinate_kind='fft_indices')}, 'fft_indices'),
    ({'mu_basis': SimpleNamespace(coordinate_kind='fractional'),
      'coordinate_kind': 'fft_indices'}, 'fractional'),
])
def test_grouped_v_q_uses_first_loader_coordinate_kind(monkeypatch, metadata, expected):
    loader = SimpleNamespace(zeta_layout='G_flat', **metadata)
    observed = {}

    class ReachedCanonicalQResolver(Exception):
        pass

    def resolve(**kwargs):
        observed.update(kwargs)
        raise ReachedCanonicalQResolver

    monkeypatch.setattr(v_q_g_flat, '_resolve_ibz_q_list', resolve)
    points = np.array([[-.13, .27, 1.31], [.630001, .77, .81]])
    with pytest.raises(ReachedCanonicalQResolver):
        v_q_g_flat._compute_V_q_g_flat_tiles(
            [dict(L=loader, R=None, timing_label='metadata regression')],
            kgrid=(1, 1, 1), fft_grid=(50, 50, 50), mesh_xy=None,
            g_chunk=None, sym=None, centroid_indices=points, verbose=False)
    assert observed['coordinate_kind'] == expected
    assert observed['centroid_indices'] is points
