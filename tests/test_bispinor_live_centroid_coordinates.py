"""Live charge/current handoffs preserve physical centroid coordinates."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('family', ['charge', 'current'])
@pytest.mark.parametrize('kind', ['fft_indices', 'fractional'])
def test_live_tile_handoff_preserves_coordinates_and_closure(monkeypatch, family, kind):
    from gw import gw_init, v_q_bispinor
    from symmetry_maps import centroid_source_map_and_wrap

    points = (np.asarray([[1, 2, 3], [5, 2, 3]], dtype=np.int64)
              if kind == 'fft_indices' else
              np.asarray([[.071, .113, .173], [.571, .113, .173]]))
    meta = SimpleNamespace(mu_basis=SimpleNamespace(coordinate_kind=kind),
                           kgrid=(2, 1, 1), fft_grid=(8, 8, 8),
                           cell_volume=1., sys_dim=3, n_rmu=2,
                           current_basis_rows=np.eye(3))
    cfg = SimpleNamespace(memory=SimpleNamespace(vq_g_chunk_size=0),
                          head=SimpleNamespace(mc_average_vcoul_body=False))
    monkeypatch.setattr(gw_init, '_vcoul_bvec_and_cutoff', lambda *args: (np.eye(3), 2.))
    monkeypatch.setattr(gw_init, '_bispinor_tt_head', lambda cfg: False)
    seen = []

    def receive(*args, **kwargs):
        received = kwargs['centroid_C_idx' if family == 'charge' else 'centroid_T_idx']
        np.testing.assert_array_equal(received, points)
        assert received.dtype == (np.float64 if kind == 'fractional' else np.int32)
        perm, wraps = centroid_source_map_and_wrap(
            received, sym_matrices=np.repeat(np.eye(3, dtype=int)[None], 2, axis=0),
            translations=np.asarray([[0., 0., 0.], [np.pi, 0., 0.]]),
            fft_grid=np.asarray(meta.fft_grid), coordinate_kind=kind)
        np.testing.assert_array_equal(perm, [[0, 1], [1, 0]])
        np.testing.assert_array_equal(wraps[1], [[-1, 0, 0], [0, 0, 0]])
        seen.append(received)
        return 'received exact physical coordinates'

    closed = []
    zeta = SimpleNamespace(close=lambda: closed.append(True))
    if family == 'charge':
        monkeypatch.setattr(v_q_bispinor, 'compute_bispinor_cc_tile', receive)
        result = gw_init._bispinor_charge_tile(
            zeta, cfg=cfg, meta=meta, wfn=object(), sym=object(),
            centroid_indices=points, mesh_xy=None)
        assert closed == [True]
    else:
        monkeypatch.setattr(v_q_bispinor, 'compute_bispinor_tt_tiles', receive)
        result = gw_init._bispinor_current_tiles(
            [zeta] * 3, cfg=cfg, meta_T=meta, wfn=object(), sym=object(),
            centroid_T_idx=points, mesh_xy=None)
        assert closed == []
    assert result == 'received exact physical coordinates' and len(seen) == 1
