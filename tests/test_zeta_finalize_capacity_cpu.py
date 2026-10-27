"""Actual finalizers admit padded reads, retained group carries and terminal outputs."""
import numpy as np
import pytest


@pytest.mark.parametrize('side,face,channels,wide', [(2, False, 1, False), (2, True, 3, False),
    (3, False, 3, False), (3, True, 1, False), (2, None, 3, False), (3, None, 3, True)])
def test_real_finalize_admits_before_carries_and_reads(monkeypatch, side, face, channels, wide):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from isdf import zeta_mubatch as owner
    from common import gpu_utils
    from lxkit import memory
    from runtime.padding import padded_axis
    jax.config.update('jax_enable_x64', True)
    if len(jax.devices('cpu')) < side * side:
        pytest.skip('requires nine virtual CPU devices')
    mesh = Mesh(np.array(jax.devices('cpu')[:side * side]).reshape(side, side), ('x', 'y'))
    world, mu, tile = side * side, side * side * 2, 6
    q = world + 1 if wide else 1
    qpad = -(-q // world) * world
    # Two batches with holes force a wider read carrier than the returned tile.
    slots = 2 * mu
    pfs = 2 * np.arange(mu, dtype=np.int32)
    rng = np.random.default_rng(481 + side + channels)
    raw = rng.normal(size=(channels, q, mu, 12)) + 1j * rng.normal(size=(channels, q, mu, 12))
    raw[..., 11:] = 0
    stores, zetas = [], []
    for i in range(channels):
        st = owner.ZStore(mesh=mesh, q_axis=padded_axis(q, world, name='q'),
            mu_pad=mu, g_axis=padded_axis(11, tile, name='G'), b=mu,
            n_batch=2, packed_from_slot=pfs, placement='host')
        source = np.zeros((q, slots, 12), complex)
        source[:, pfs] = raw[i]
        for beta in range(2):
            st.write_batch(beta, jax.device_put(source[:, beta*mu:(beta+1)*mu],
                NamedSharding(mesh, P(None, ('x', 'y'), None))))
        diagonal = 2 + i
        use_face = face or (face is None and i % 2 == 1)
        if use_face:
            factor = jax.device_put(np.broadcast_to(np.eye(mu, dtype=complex) / diagonal, (q, mu, mu)).copy(),
                                   NamedSharding(mesh, P(None, 'x', 'y')))
            piv = None
        else:
            factor = jax.device_put(np.broadcast_to(np.eye(mu, dtype=complex) * diagonal, (qpad, mu, mu)).copy(),
                                   NamedSharding(mesh, P(('x', 'y'), None, None)))
            piv = jax.device_put(np.tile(np.arange(mu, dtype=np.int32), (qpad, 1)),
                                NamedSharding(mesh, P(('x', 'y'), None)))
        zetas.append(owner.ZetaG(st, mesh=mesh, L_q=factor, lu_piv=piv,
            solver_kind='inverse' if use_face else 'lu', batched_route='auto',
            n_rmu_solve=mu, n_rmu=mu, mu_basis=None, ngk_per_q=np.full(q, 11),
            gvec_components=np.zeros((q, 3, 11), np.int32), path='unused'))
        stores.append(st)
    prior = jax.device_put(np.ones((world * 32, world * 32)),
                           NamedSharding(mesh, P('x', 'y')))
    prior.block_until_ready()
    memory.set_budget(1 << 30)
    monkeypatch.setattr(gpu_utils, 'shared_device_room', lambda _stage: 1 << 30)
    pairs = ((0, 0),) if channels == 1 else ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
    weights = np.ones((len(pairs), q, 11), complex)
    keep = np.tile(np.array([[0, 10]], np.int32), (q, 1))
    events, quotes = [], []
    real_admit, real_zero = memory.admit, owner._zero_program

    def admit(stage, price, room, **kwargs):
        if stage.startswith('zeta '):
            assert events == []
            assert room == memory.room(bucketed=False)
            quotes.append(price.bytes())
            events.append('admit')
        return real_admit(stage, price, room, **kwargs)

    class Zero:
        def __init__(self, program):
            self.program = program
        def __getattr__(self, name):
            return getattr(self.program, name)
        def __call__(self):
            assert events == ['admit'], 'V/S allocated before admission'
            events.append('carries')
            return self.program()

    monkeypatch.setattr(memory, 'admit', admit)
    monkeypatch.setattr(owner, '_zero_program', lambda shapes: Zero(real_zero(shapes)))
    for st in stores:
        read = st.read_tile
        def observed(t, read=read):
            assert events[:2] == ['admit', 'carries'], 'read before complete pass admission'
            events.append('read')
            return read(t)
        st.read_tile = observed

    def run():
        if channels == 1:
            return [zetas[0].contract_v(weights[0], keep=keep)]
        return owner.contract_v_group(zetas, pairs, weights, keep=keep)

    cached = None
    for _ in range(2):
        events.clear()
        values = run()
        jax.block_until_ready((values, [z.shell for z in zetas]))
        for value, (a, b) in zip(values, pairs):
            left, right = raw[a, ..., :11] / (2+a), raw[b, ..., :11] / (2+b)
            expected = np.einsum('qmg,qng->qmn', left.conj(), right)
            np.testing.assert_allclose(np.asarray(value), expected, rtol=2e-12, atol=3e-13)
        for i, z in enumerate(zetas):
            np.testing.assert_allclose(np.asarray(z.shell), raw[i, :, :, [0, 10]].transpose(1, 2, 0)/(2+i),
                                       rtol=2e-12, atol=3e-13)
        current = {k: id(v) for k, v in owner._kernel_cache.items()
                   if k[0] in ('compiled_phase', 'zero_program')}
        if cached is not None:
            assert current == cached  # executable reuse, while admission still repeats
        cached = current
    assert len(quotes) == 2  # cache reuse cannot reuse the prior room/admission
    _, _, final_tile, read_peak = owner._read_programs(stores[0])
    assert read_peak > owner._local_bytes(final_tile)
    assert quotes[-1] >= read_peak
    np.testing.assert_array_equal(np.asarray(prior), 1)
    # Exhausted room preserves the shared warning-and-run contract.
    events.clear()
    memory.set_budget(1)
    with pytest.warns(RuntimeWarning, match='priced over the limit'):
        values = run()
    jax.block_until_ready(values)
    assert len(quotes) == 3 and all(np.isfinite(np.asarray(a)).all() for a in values)


def test_store_read_prices_allocated_slots_before_crop():
    import jax
    from jax.sharding import Mesh
    from isdf import zeta_mubatch as owner
    from runtime.padding import padded_axis
    if len(jax.devices('cpu')) < 4:
        pytest.skip('requires four virtual CPU devices')
    mesh = Mesh(np.array(jax.devices('cpu')[:4]).reshape(2, 2), ('x', 'y'))
    st = owner.ZStore(mesh=mesh, q_axis=padded_axis(1, 4, name='q'), mu_pad=4,
        g_axis=padded_axis(8, 8, name='G'), b=16, n_batch=2,
        packed_from_slot=np.array([0, 10, 20, 30], np.int32), placement='host')
    move, pack, result, peak = owner._read_programs(st)
    # Actual MA belongs to the same two executables read_tile dispatches.
    assert move.memory_analysis().output_size_in_bytes == 32 * 8 * 16
    assert pack.memory_analysis().output_size_in_bytes == 4 * 8 * 16
    assert peak >= 32 * 8 * 16 > 2 * owner._local_bytes(result)
