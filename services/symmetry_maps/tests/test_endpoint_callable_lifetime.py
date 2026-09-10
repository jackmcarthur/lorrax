"""Eager endpoint calls reuse code without freezing mutable metadata."""
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from symmetry_maps.maps import unfold_endpoint_panel, _endpoint_panel_kernel


def test_endpoint_tables_are_data_not_cached_constants():
    devices = np.array(jax.devices()[:4])
    mesh = Mesh(devices.reshape(2, 2), ('x', 'y'))
    tile = NamedSharding(mesh, P(None, 'x', None, None))
    x = (np.arange(8).reshape(1, 4, 1, 2) + 1j).astype(complex)
    kw = dict(irr_idx=np.array([0]), sym_idx=np.array([0]),
              q_irr_frac=np.zeros((1, 3)), source_perm=np.arange(4)[None, :],
              L_table=np.zeros((1, 4, 3)), spin_action_full=np.ones((1, 1, 1)),
              n_sym_spatial=1, active_mask=np.ones(4, bool), mesh=mesh,
              mesh_axis='x', max_live_bytes=1000000)
    _endpoint_panel_kernel.cache_clear()
    for scale in (1., 2.):
        kw['spin_action_full'][...] = scale
        got, _ = unfold_endpoint_panel(jax.device_put(x, tile), **kw)
        np.testing.assert_array_equal(got, scale*x)
    fn = _endpoint_panel_kernel(mesh, 'x', True, 1)
    assert fn._cache_size() == 1
    same = Mesh(devices.copy().reshape(2, 2), ('x', 'y'))
    assert fn is _endpoint_panel_kernel(same, 'x', True, 1)
    for args in ((mesh, 'y', True, 1), (mesh, 'x', False, 1),
                 (mesh, 'x', True, 2),
                 (Mesh(devices[::-1].reshape(2, 2), ('x', 'y')), 'x', True, 1)):
        assert fn is not _endpoint_panel_kernel(*args)
