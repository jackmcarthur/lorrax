"""The V handoff owns only consumed carries; grouped cross-Grams keep all shells."""
from types import SimpleNamespace

import numpy as np
import pytest


def _mesh(side):
    jax = pytest.importorskip('jax')
    from jax.sharding import Mesh
    jax.config.update('jax_enable_x64', True)
    if len(jax.devices('cpu')) < side * side:
        pytest.skip('requires nine virtual CPU devices')
    return Mesh(np.array(jax.devices('cpu')[:side * side]).reshape(side, side), ('x', 'y'))


@pytest.mark.parametrize('side', [2, 3])
def test_live_fit_handoff_does_not_allocate_a_discarded_v(monkeypatch, side):
    import jax
    from gw import v_q_g_flat as vq
    from jax.sharding import NamedSharding, PartitionSpec as P
    mesh = _mesh(side)
    q, mu, g = 2, 36, 4
    expected = np.arange(q * mu * mu).reshape(q, mu, mu).astype(complex)
    shell = jax.device_put(np.full((q, mu, 1), 2 + .3j),
                          NamedSharding(mesh, P(None, ('x', 'y'), None)))
    result = jax.device_put(expected, NamedSharding(mesh, P(None, 'x', 'y')))
    calls, allocations = [], []
    original_zeros = vq.jnp.zeros

    def zeros(shape, *args, **kwargs):
        allocations.append(tuple(shape))
        return original_zeros(shape, *args, **kwargs)

    def contracted(*_args, **_kwargs):
        calls.append('live-fit')
        assert (q, mu, mu) not in allocations
        return result

    loader = SimpleNamespace(zeta_layout='G_flat', n_rmu=mu,
        gvec_components=np.zeros((q, 3, g), np.int32), contract_v=contracted,
        shell=shell)
    spec = dict(L=loader, R=None, timing_label='owned-output', write_g0=True,
        is_charge_cc=False, v_per_G_builder=lambda *_a: np.ones((q, g), complex))
    monkeypatch.setattr(vq, '_resolve_ibz_q_list', lambda **_k:
        (None, np.zeros((q, 3)), None, None, None, None, False))
    monkeypatch.setattr(vq.jnp, 'zeros', zeros)
    got, g0 = vq._compute_V_q_g_flat_tiles([spec], kgrid=(q, 1, 1),
        fft_grid=(2, 2, 2), mesh_xy=mesh, g_chunk=None, sym=None,
        centroid_indices=None, verbose=False)[0]
    assert calls == ['live-fit']
    np.testing.assert_array_equal(np.asarray(got), expected)
    np.testing.assert_array_equal(np.asarray(g0), np.asarray(shell)[:, :, 0])


@pytest.mark.parametrize('side,solver', [(2, 'lu'), (3, 'inverse')])
def test_grouped_cross_grams_and_shells_allocate_only_output_v(monkeypatch, side, solver):
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from isdf import zeta_mubatch as zm
    from common import gpu_utils
    from runtime.padding import padded_axis
    mesh = _mesh(side)
    world = side * side
    q, logical, mu, g_tile, ng = world + 1, 13, (-(-13 // world)) * world, 6, 11
    qpad = -(-q // world) * world
    ngk = ng - np.arange(q) % 3
    rng = np.random.default_rng(76140 + side)
    raw = rng.normal(size=(3, q, mu, 12)) + 1j * rng.normal(size=(3, q, mu, 12))
    raw[:, :, logical:] = 0
    for iq, count in enumerate(ngk):
        raw[:, iq, :, count:] = 0
    diagonal = np.empty((3, q, mu), complex)
    diagonal[:] = 1
    diagonal[:, :, :logical] = 1 + rng.uniform(.1, .8, (3, q, logical))
    diagonal[1] *= -1
    expected_zeta = raw / diagonal[:, :, :, None]
    keep = np.stack([np.zeros(q, np.int32), ngk - 1], axis=1)
    pairs = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    weights = rng.normal(size=(6, q, ng)) + .2j * rng.normal(size=(6, q, ng))
    allocations = []
    original_zeros = zm.jnp.zeros

    def zeros(shape, *args, **kwargs):
        if tuple(shape) == (qpad, mu, mu):
            allocations.append(tuple(shape))
        return original_zeros(shape, *args, **kwargs)

    zetas = []
    for ic in range(3):
        if solver == 'lu':
            lu = np.zeros((qpad, mu, mu), complex)
            lu[:] = np.eye(mu)
            for iq in range(q):
                lu[iq] = np.diag(diagonal[ic, iq])
            factor = jax.device_put(lu, NamedSharding(mesh, P(('x', 'y'), None, None)))
            piv = jax.device_put(np.tile(np.arange(logical, dtype=np.int32), (qpad, 1)),
                                 NamedSharding(mesh, P(('x', 'y'), None)))
        else:
            factor = jax.device_put(np.stack([np.diag(1 / row) for row in diagonal[ic]]),
                                   NamedSharding(mesh, P(None, 'x', 'y')))
            piv = None
        source = np.pad(raw[ic], ((0, qpad-q), (0, 0), (0, 0)))

        def read_tile(t, source=source):
            return jax.device_put(source[:, :, t*g_tile:(t+1)*g_tile],
                                 NamedSharding(mesh, P(('x', 'y'), None, None)))

        store = SimpleNamespace(Q=q, Q_pad=qpad, mu_pad=mu, g_tile=g_tile,
            n_Gt=2, q_axis=padded_axis(q, world, name='q'),
            g_axis=padded_axis(ng, g_tile, name='G'), read_tile=read_tile, t_read=0.)
        zetas.append(zm.ZetaG(store, mesh=mesh, L_q=factor, lu_piv=piv,
            solver_kind=solver, batched_route='auto', n_rmu_solve=logical,
            n_rmu=logical, mu_basis=None, ngk_per_q=ngk,
            gvec_components=np.zeros((q, 3, ng), np.int32), path='unused'))
    monkeypatch.setattr(gpu_utils, 'shared_device_room', lambda _s: 1 << 30)
    monkeypatch.setattr(zm.jnp, 'zeros', zeros)
    result = zm.contract_v_group(zetas, pairs, weights, keep=keep)
    jax.block_until_ready((result, [z.shell for z in zetas]))
    assert len(allocations) == (6 if solver == 'lu' else 0)
    for i, (a, b) in enumerate(pairs):
        expected = np.einsum('qmg,qg,qng->qmn', expected_zeta[a, :, :, :ng].conj(),
                             weights[i], expected_zeta[b, :, :, :ng])
        np.testing.assert_allclose(np.asarray(result[i]), expected, atol=3e-13, rtol=2e-12)
        assert result[i].sharding.spec == P(None, 'x', 'y')
    for ic, zeta in enumerate(zetas):
        expected = np.take_along_axis(expected_zeta[ic], keep[:, None, :], axis=2)
        np.testing.assert_allclose(np.asarray(zeta.shell), expected, atol=3e-14, rtol=2e-12)
        assert zeta.shell.sharding.is_equivalent_to(
            NamedSharding(mesh, P(None, ('x', 'y'), None)), 3)
