"""Run directly under MPI to check consumed candidate-Gram buffer lifetimes."""


def main():
    from types import SimpleNamespace
    from unittest.mock import patch

    import runtime
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from centroid import pivoted_cholesky as pc

    stack = runtime.initialize_communicator_stack()
    mesh = stack.mesh
    m = 3 * int(mesh.size)
    weights = np.array([0.07, 0.11, 0.19, 0.23, 0.40])
    row = np.arange(m)[:, None]
    feature = np.arange(4)[None, :]
    features = [np.sin((row + 1) * (feature + 2) * (k + 1) / 29)
                + 1j * np.cos((row + 3) * (feature + 1) * (k + 2) / 31)
                for k in range(5)]
    # Deliberately non-Hermitian raw blocks make the terminal fold observable.
    # The separate Fourier reference covers the physical candidate builder.
    gram_k = [z @ z.conj().T + (k + 1) * (row + 1j * row.T) / 97
              for k, z in enumerate(features)]
    wfn = SimpleNamespace(
        nspinor=1,
        full_k_parent_groups=lambda rows: [(int(k), np.array([k])) for k in rows],
    )
    sym = SimpleNamespace(nk_tot=5)
    sharding = NamedSharding(mesh, P('x', 'y'))

    for limit in (2, 5):
        blocks = []
        expected_blocks = []

        def build_block(*args, full_k_rows=None, k_weights=None, **kwargs):
            assert kwargs['hermitian_fold'] is False
            if len(blocks) > 1:
                assert blocks[-1].is_deleted(), (
                    'consumed Gram block remained live when the next WFN batch began')
            rows = range(5) if full_k_rows is None else full_k_rows
            value = sum(float(w) * gram_k[int(k)]
                        for k, w in zip(rows, k_weights))
            expected_blocks.append(value)
            block = jax.device_put(value, sharding)
            blocks.append(block)
            return block

        with patch('runtime.tiles.tile_units', return_value=limit), \
             patch.object(pc, '_build_gram_q0_kbatch', side_effect=build_block):
            actual = pc.build_gram_q0_via_loadwfns(
                wfn, sym, np.zeros((m, 3), dtype=np.int64), mesh_xy=mesh,
                band_range_left=(0, 3), band_range_right=(0, 3),
                k_weights=weights, verbose=False,
            )
        expected = expected_blocks[0].copy()
        for block in expected_blocks[1:]:
            expected += block
        expected = 0.5 * (expected + expected.conj().T)
        independent = sum(w * g for w, g in zip(weights, gram_k))
        independent = 0.5 * (independent + independent.conj().T)
        for shard in actual.addressable_shards:
            np.testing.assert_array_equal(np.asarray(shard.data), expected[shard.index])
            np.testing.assert_allclose(
                np.asarray(shard.data), independent[shard.index],
                rtol=2e-15, atol=2e-15)
        assert all(block.is_deleted() for block in blocks)
        assert not actual.is_deleted()
        assert actual.sharding.spec == P('x', 'y')
        actual.delete()

    print('candidate Gram accumulation preserves values and releases consumed blocks: PASS', flush=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip('run this module directly with the configured MPI/GPU launcher')


if __name__ == '__main__':
    main()
