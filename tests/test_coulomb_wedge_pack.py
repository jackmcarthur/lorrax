"""P4 independent canonical→q-wedge→packed Coulomb seam and all-q checks."""
import runtime
import numpy as np


def main():
    mesh = runtime.initialize_communicator_stack().mesh
    import jax
    from types import SimpleNamespace
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import device_put_process_local, gather_to_host
    from common.centroid_basis import PackedCentroidBasis
    from common.grouped_layout import build_square_grouped_shard_layout
    from runtime.padding import padded_mu_extent
    from lxkit import memory
    from common import sanity
    import gw.gw_init as owner
    import gw.v_q_g_flat as vq
    assert mesh.size == 4
    rng = np.random.default_rng(20261023)
    n, nq = 37, 7
    labels = np.array([0]*25 + [1]*6 + [2]*6)
    layout = build_square_grouped_shard_layout(labels, tuple(mesh.shape[a] for a in ('x', 'y')))
    basis = PackedCentroidBasis(mesh, layout,
        np.stack((np.arange(n), np.zeros(n), np.zeros(n)), axis=1).astype(np.int32),
        int(padded_mu_extent(n, mesh)))
    assert not basis.is_identity and basis.n_canonical != basis.n_packed
    canonical = np.zeros((nq, basis.n_canonical, basis.n_canonical), np.complex128)
    canonical[:, :n, :n] = rng.normal(size=(nq, n, n)) + 1j*rng.normal(size=(nq, n, n))
    rows = np.array([0, 2, 5], np.int32)
    perm = np.tile(np.arange(n, dtype=np.int32), (4, 1))
    wraps = rng.integers(-2, 3, (4, n, 3))
    pperm, pwrap = basis.pack_tables(perm, wraps)
    tables = dict(irr_idx=np.array([0, 1, 1, 2, 2, 2, 0], np.int32),
        sym_idx=np.array([0, 3, 0, 1, 2, 0, 3], np.int32),
        sym_perm=pperm, L_table=pwrap, q_irr_frac=rng.uniform(-.5, .5, (3, 3)),
        n_sym_spatial=2, full_rows=rows)
    sh = NamedSharding(mesh, P(None, 'x', 'y'))
    dev = device_put_process_local(canonical, sh)
    meta = SimpleNamespace(mu_basis=basis, kgrid=(7, 1, 1))
    original, admit = vq.q_wedge, memory.admit
    receipts = []
    def record(stage, routes, room, **kwargs):
        price = routes[0][1]
        assert price.ranks == mesh.size and price.bytes() >= 0
        receipts.append((price.bytes(), room))
        return admit(stage, routes, room, **kwargs)
    memory.admit = record
    try:
        for reduced in (False, True):
            vq.q_wedge = lambda **kw: (tables, None) if reduced else None
            result = owner._coulomb_on_wedge(dev, sym=None, centroid_indices=None, meta=meta)
            selected = canonical[rows] if reduced else canonical
            expected = basis.pack_host(basis.pack_host(selected, axis=-2), axis=-1)
            np.testing.assert_array_equal(np.asarray(gather_to_host(result.values)), expected)
            assert result.is_whole_zone() == (not reduced)
            if reduced:
                for key, value in tables.items():
                    np.testing.assert_array_equal(getattr(result, key), value)
                full = np.asarray(gather_to_host(result.unfold(mesh)))
                literal = []
                for parent, action in zip(tables['irr_idx'], tables['sym_idx']):
                    phase = np.exp(2j*np.pi*(pwrap[action] @ tables['q_irr_frac'][parent]))
                    x = phase[:, None]*expected[parent]*phase.conj()[None, :]
                    literal.append(x.conj() if action >= 2 else x)
                np.testing.assert_allclose(full, np.asarray(literal), rtol=2e-13, atol=2e-13)
        # Finite checks must inspect nonrepresentative rows too, before packing.
        bad = canonical.copy(); bad[6, 0, 0] = np.nan
        baddev = device_put_process_local(bad, sh)
        g0 = device_put_process_local(np.ones((1, basis.n_canonical), np.complex128),
                                     NamedSharding(mesh, P(None, 'x')))
        try:
            with sanity.forced_level('strict'):
                owner._finalize_vq_views(g0, baddev, None, meta, None, lambda *a: None)
        except (ValueError, RuntimeError, AssertionError) as exc:
            assert 'finite' in str(exc).lower() or 'nan' in str(exc).lower(), str(exc)
        else:
            raise AssertionError('nonrepresentative nonfinite canonical q escaped validation')
    finally:
        vq.q_wedge, memory.admit = original, admit
    print('COULOMB_WEDGE_PACK PASS canonical/ragged/full/reduced/TRS/wrap/all-q checks', receipts, flush=True)


if __name__ == '__main__':
    main()
