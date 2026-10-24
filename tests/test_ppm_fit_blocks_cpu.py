"""The GN point fit survives fixed q windows without a second full output set."""
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
import pytest


@pytest.mark.parametrize('side,ordered', [(2, False), (2, True), (3, True)])
def test_q_windows_preserve_point_fit_and_weighted_census(side, ordered):
    from common.collectives import xy_tile_mesh
    from gw import minimax_screening as M
    if jax.device_count() < side * side:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.array(jax.devices()[:side*side]).reshape(side, side), ('x', 'y'))
    sh = NamedSharding(mesh, P(None, 'x', 'y'))
    nq, mu, logical = 5, 24, 19
    r = np.arange(mu)
    active = np.ones(mu, bool); active[[1, 6, 12, 17, 23]] = False
    wc = -.03 / (1 + (r[:, None] - r[None, :])**2)
    om = .5 + (r[:, None] + r[None, :]) / 20
    w0 = np.broadcast_to(wc, (nq, mu, mu)).copy().astype(complex)
    wi = np.broadcast_to(wc*om**2/(om**2+4), w0.shape).copy().astype(complex)
    wi[0, :3, :3] = 2*w0[0, :3, :3]  # invalid modes are kept in the fit census
    if ordered:
        wi += 1j * (.002/(1+(r[:, None]-r[None, :])**2))[None]
    put = lambda a: jax.device_put(a, sh)
    w0, wi = put(w0), put(wi)
    xy = xy_tile_mesh(wi)
    z, fb = jnp.asarray(2j), jnp.asarray(2.)
    point = M._gn_ppm_fit_kernel_ordered if ordered else M._gn_ppm_fit_kernel
    pa = (z, fb, logical, active) + ((xy,) if ordered else ())
    ref = point(w0, wi, *pa)
    stars = np.array([1, 2, 1, 3, 2], np.int64)
    wanted_good = int(np.sum(np.asarray(ref[2])*stars[:, None, None]))
    for qb in (1, 2, 3, 5):
        got = M._gn_ppm_fit_blocks(w0, wi, z, fb, active, jnp.asarray(stars),
            q_block=qb, n_log=logical, ordered=ordered, xy_mesh=xy)
        for i in range(3):
            np.testing.assert_array_equal(np.asarray(got[i]), np.asarray(ref[i]))
        if ordered:
            np.testing.assert_array_equal(np.asarray(got[3]), np.asarray(ref[8]))
        else:
            assert got[3] is None
        assert float(got[4]) == wanted_good
        assert float(got[5]) == logical**2*stars.sum()
        for i in range(3):
            assert float(got[6+i]) == float(ref[5+i])


def test_q_scan_temporary_size_does_not_duplicate_the_full_outputs():
    """Doubling q doubles the output, while a one-q work arena stays bounded."""
    from gw import minimax_screening as M
    if jax.device_count() < 4:
        pytest.skip("needs 4 devices")
    mesh = Mesh(np.array(jax.devices()[:4]).reshape(2, 2), ('x', 'y'))
    sh = NamedSharding(mesh, P(None, 'x', 'y'))
    records = []
    for nq in (7, 14):
        a = jax.ShapeDtypeStruct((nq, 96, 96), jnp.complex128, sharding=sh)
        prog = M._gn_ppm_fit_blocks.lower(a, a, jnp.asarray(2j), jnp.asarray(2.),
            None, jnp.ones(nq, jnp.int64), q_block=1, n_log=91, ordered=True,
            xy_mesh=mesh).compile()
        ma = prog.memory_analysis()
        records.append((ma.output_size_in_bytes, ma.temp_size_in_bytes))
    assert records[1][0] > 1.9*records[0][0]
    # Scalar metadata can grow with q; another full field set cannot.
    assert records[1][1] - records[0][1] < (records[1][0]-records[0][0])/8
