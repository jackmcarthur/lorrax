"""Direct compute-node gate for file-wedge versus star-wedge quadrature."""


def main():
    from types import SimpleNamespace
    import runtime
    runtime.initialize_communicator_stack()
    import numpy as np
    from centroid.sampling_metric import _quadrature_tables

    def fixture(parents, file_rows, weights):
        sym = SimpleNamespace(
            nk_tot=len(parents), irr_idx_k=np.asarray(parents, np.int32),
            sym_idx_k=np.zeros(len(parents), np.int32),
            sym_mats_k=np.stack((np.eye(3), -np.eye(3))),
            kirr_fullids=np.asarray(file_rows))
        wfn = SimpleNamespace(nkpts=len(weights), kweights=np.asarray(weights, float))
        return wfn, sym

    # Six-point grid under TR. The file stores both raw +/-1/6 rows,
    # while canonical parent 4 owns that complete two-member star.
    parents = [0, 4, 2, 3, 2, 4]
    file_rows = [0, 1, 2, 3, 5]
    file_weights = [1, 1, 2, 1, 1]
    wfn, sym = fixture(parents, file_rows, file_weights)
    used, plan, full = _quadrature_tables(wfn, sym)
    np.testing.assert_array_equal(used, [0, 2, 3, 4])
    np.testing.assert_allclose(full, np.full(6, 1/6), rtol=0, atol=1e-16)
    np.testing.assert_allclose([plan[int(p)][1].sum() for p in used],
                               np.array([1, 2, 1, 2])/6, rtol=0, atol=1e-16)

    # Independent file-weight sum of TR-even scalar feature fields. This
    # covers both charge and current-squared fields after their spin trace.
    fields = np.arange(7, dtype=float)[None] * np.array([2, 1, 3, 4, 1])[:, None]
    raw = np.einsum('k,kr->r', np.array(file_weights)/6, fields)
    expanded = np.einsum('k,kr->r', full, fields[np.asarray(parents)])
    np.testing.assert_allclose(raw, expanded, rtol=1e-15, atol=1e-15)

    minimal_wfn, minimal_sym = fixture([0, 1, 2, 3, 2, 1], [0, 1, 2, 3], [1, 2, 2, 1])
    np.testing.assert_array_equal(_quadrature_tables(minimal_wfn, minimal_sym)[2], full)
    full_wfn, full_sym = fixture([0, 1, 2, 3, 2, 1], np.arange(6), np.arange(1, 7))
    np.testing.assert_array_equal(_quadrature_tables(full_wfn, full_sym)[2], np.arange(1, 7)/21)

    refusals = 0
    for bad in ([0, 1, 2, 3, 7], [0, 1, 2, 3, 3], [0., 1., 2., 3., 5.],
                [0, 1, 3, 2, 5]):
        _, broken = fixture(parents, bad, file_weights)
        try:
            _quadrature_tables(wfn, broken)
        except ValueError:
            refusals += 1
        else:
            raise AssertionError('inconsistent exact file/parent map was accepted')
    for bad in ([1, -1, 2, 1, 1], [1, np.nan, 2, 1, 1], [0, 0, 0, 0, 0]):
        broken, _ = fixture(parents, file_rows, bad)
        try:
            _quadrature_tables(broken, sym)
        except ValueError:
            refusals += 1
        else:
            raise AssertionError('invalid WFN weights were accepted')
    assert refusals == 7
    print('FEATURE_METRIC_QUADRATURE PASS: minimal/nonminimal/full and seven refusals', flush=True)
    runtime.finalize_process(0)


def test_requires_compute_launcher():
    import pytest
    pytest.skip('run this module directly under the configured compute launcher')


if __name__ == '__main__':
    main()
