"""Actual parent transport must admit new storage before reads and on reuse."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize('ns', (1, 2, 4))
@pytest.mark.parametrize('read_stop', (15, 23))
def test_parent_loader_prices_before_read_and_rechecks_cached_programs(monkeypatch, ns, read_stop):
    import jax
    from jax.sharding import Mesh, NamedSharding
    from common import psi_G_store as G, wfn_transforms as W
    from lxkit import memory

    jax.config.update('jax_enable_x64', True)
    if len(jax.devices('cpu')) < 4:
        pytest.skip('needs four virtual CPU devices')
    mesh = Mesh(np.array(jax.devices('cpu')[:4]).reshape(2, 2), ('x', 'y'))
    monkeypatch.setattr(memory, '_RUN', dict(memory._RUN))
    memory.set_budget(1 << 28)
    events, programs = [], []
    admit, priced = memory.admit, G._priced_executable

    def observe_admit(stage, price, room, **kwargs):
        if stage.startswith('psi_G_store.gslot.') or stage == 'load_centroids.finish_faces':
            events.append(('admit', stage, room))
        return admit(stage, price, room, **kwargs)

    def observe_program(kernel, args, stage, **kwargs):
        executable = priced(kernel, args, stage, **kwargs)
        programs.append((stage, executable, kwargs.get('extra', 0)))
        return executable

    monkeypatch.setattr(memory, 'admit', observe_admit)
    monkeypatch.setattr(G, '_priced_executable', observe_program)
    monkeypatch.setattr(W, '_priced_executable', observe_program)
    grid, nk, ng = (4, 4, 4), 2, 11
    coords = np.stack(np.unravel_index(np.arange(5) * 3, grid), axis=-1)
    meta = SimpleNamespace(nspinor=ns, nk_tot=nk, fft_grid=grid,
                           b_id_4_user=13, memory_per_device_gb=.1)

    def values(k, b, s, g):
        return np.where((b < 17) & (g < ng),
                        k + 2*b + 3*s + 1j*g, 0).astype(np.complex128)

    class Loader:
        nbands, nkpts, nspinor, ngkmax = 17, nk, ns, ng

        def __init__(self):
            self.reads = []
            self.widths = set()

        def box_index(self, *, k):
            return np.broadcast_to(np.arange(ng, dtype=np.int32), (nk, ng)).copy()

        def kvecs(self, *, k):
            return np.zeros((nk, 3))

        def load(self, *, bands, k, sharding, **kwargs):
            # Both full/tail consumer shapes must be admitted before loading.
            stages = [e[1] for e in events if e[0] == 'admit']
            self.widths.add(min(8, read_stop-bands[0]))
            assert stages.count('psi_G_store.gslot.a2a_faces') == len(self.widths)
            assert stages[:2] == ['psi_G_store.gslot.zero_faces',
                                  'psi_G_store.gslot.zero_store']
            events.append(('read', bands))
            self.reads.append(bands)
            lo, hi = bands
            shape = (nk, 4 * ((hi-lo+3)//4), ns, ng)

            def owned(index):
                axes = [np.arange(*sl.indices(n)) for sl, n in zip(index, shape)]
                ka, ba, sa, ga = np.ix_(*axes)
                return np.where(ba+lo < hi, values(ka, ba+lo, sa, ga), 0)
            return jax.make_array_from_callback(shape, NamedSharding(mesh, sharding), owned)

    for repeat in range(2):
        events.clear()
        programs.clear()
        loader = Loader()
        if repeat:
            # Real existing residents remain live; admission must reread room
            # despite reusing the same geometry's compiled programs.
            memory.set_budget(memory._in_use() + 64)
        def load():
            return G.load_parent_psi_G(wfn=loader, mesh_xy=mesh, meta=meta,
                band_range=(3, read_stop), band_chunk=8, centroid_indices=coords,
                store_band_range=(5, 13), print_fn=lambda *_a: None)
        if repeat:
            with pytest.warns(RuntimeWarning, match='memory'):
                product = load()
            assert len(W._EXECUTABLES) == cache_size
        else:
            product = load()
            cache_size = len(W._EXECUTABLES)
        assert loader.reads == [(3, 11), (11, min(read_stop, 17))]
        stages = [s for s, _, _ in programs]
        assert stages.count('psi_G_store.gslot.zero_faces') == 1
        assert stages.count('psi_G_store.gslot.zero_store') == 1
        widths = (8, 4) if read_stop == 15 else (8,)
        assert stages.count('psi_G_store.gslot.a2a_faces') == len(widths)
        assert stages.count('psi_G_store.gslot.keep') == 2
        assert stages.count('load_centroids.finish_faces') == 1
        quoted = [(ex, extra) for s, ex, extra in programs if s.endswith('a2a_faces')]
        assert [extra for _, extra in quoted] == [nk*w*ns*12*16//4 for w in widths]
        nb = read_stop - 3
        assert all(ex.memory_analysis().alias_size_in_bytes == 2*nk*nb*ns*8*16//4
                   for ex, _ in quoted)
        if repeat:
            assert all(e[2] <= 64 for e in events if e[0] == 'admit')
        ka, ba, sa, ga = np.ix_(np.arange(nk), np.arange(5, 13),
                                np.arange(ns), np.arange(12))
        np.testing.assert_array_equal(np.asarray(product.psi_G), values(ka, ba, sa, ga))
        g = np.arange(ng)
        gvec = np.stack(np.unravel_index(g, grid), axis=-1)
        phase = np.exp(2j*np.pi*(gvec @ (coords/np.asarray(grid)).T)) / 8
        ka, ba, sa, ga = np.ix_(np.arange(nk), np.arange(3, read_stop),
                                np.arange(ns), g)
        expected = np.zeros((nk, nb, ns, 8), complex)
        # File bands above the requested logical stop contain nonzero values;
        # the terminal program, rather than the fixture, must remove them.
        expected[..., :5] = np.einsum('kbsg,gm->kbsm',
            np.where(ba < 13, values(ka, ba, sa, ga), 0), phase)
        np.testing.assert_allclose(np.asarray(product.faces[0]), expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(np.asarray(product.faces[1]),
            expected.conj().transpose(0, 3, 1, 2), rtol=2e-12, atol=2e-12)


def test_generic_finish_admits_crop_and_nonzero_band_mask_without_retaining_meta(monkeypatch):
    import gc
    import weakref
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common import wfn_transforms as W
    from lxkit import memory
    jax.config.update('jax_enable_x64', True)
    if len(jax.devices('cpu')) < 4:
        pytest.skip('needs four virtual CPU devices')
    mesh = Mesh(np.array(jax.devices('cpu')[:4]).reshape(2, 2), ('x', 'y'))
    sy = NamedSharding(mesh, P(None, None, None, 'y'))
    sx = NamedSharding(mesh, P(None, 'x', None, None))
    class Meta:
        b_id_4_user = 2
    meta = Meta()
    reference = weakref.ref(meta)
    _, finish = W._centroid_face_kernels(0, meta, None, 5, 8, 3, sx, sy,
        NamedSharding(mesh, P(None, 'x', None, None)),
        NamedSharding(mesh, P(None, 'y', None, None)))
    del meta
    gc.collect()
    assert reference() is None, 'the geometry cache retained the caller bundle'
    data = np.arange(3*4*2*8).reshape(3, 4, 2, 8).astype(complex) * (1+.25j)
    data[..., 5:] = 0
    y, x = (jax.device_put(a, sh) for a, sh in
            ((data, sy), (data.conj().transpose(0, 3, 1, 2), sx)))
    jax.block_until_ready((y, x))
    monkeypatch.setattr(memory, '_RUN', dict(memory._RUN))
    memory.set_budget(memory._in_use() + 1)
    with pytest.warns(RuntimeWarning, match='memory'):
        actual = W._finish_centroid_faces(finish, (y, x), 2)
    expected = data[:2, :3].copy()
    expected[:, 2:] = 0
    np.testing.assert_array_equal(np.asarray(actual[0]), expected)
    np.testing.assert_array_equal(np.asarray(actual[1]), expected.conj().transpose(0, 3, 1, 2))
    assert not y.is_deleted() and not x.is_deleted(), 'terminal input donation changed'


def test_axis_program_sizing_does_not_extrapolate_a_full_mesh_spread(monkeypatch):
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from common import wfn_transforms as W
    from lxkit import memory
    from runtime.aot_memory import compiled_new_bytes
    jax.config.update('jax_enable_x64', True)
    if len(jax.devices('cpu')) < 4:
        pytest.skip('needs four virtual CPU devices')
    mesh = Mesh(np.array(jax.devices('cpu')[:4]).reshape(2, 2), ('x', 'y'))
    monkeypatch.setattr(memory, '_RUN', dict(memory._RUN))
    memory.set_budget(1 << 28)
    prices = {}
    real_admit = memory.admit
    def observe(stage, price, room, **kwargs):
        prices[stage] = price
        return real_admit(stage, price, room, **kwargs)
    monkeypatch.setattr(memory, 'admit', observe)
    for label, spec in (('axis', P(None, 'y')), ('face', P('x', 'y'))):
        sharding = NamedSharding(mesh, spec)
        kernel = jax.jit(lambda x: x+x, out_shardings=sharding)
        abstract = jax.ShapeDtypeStruct((32, 32), jnp.float64, sharding=sharding)
        ex = W._priced_executable(kernel, (abstract,), label)
        assert prices[label].bytes() == compiled_new_bytes(ex)
    # These are forecasts of the measured current-mesh quote, not recompiled
    # capacities. Only a program on both axes may assume a 1/P reduction.
    assert prices['axis'].bytes(16) == prices['axis'].bytes(4)
    assert prices['face'].bytes(16) < prices['face'].bytes(4)
