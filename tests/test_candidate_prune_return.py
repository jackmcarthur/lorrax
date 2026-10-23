"""Ragged point/group pruning returns only geometry and linear diagnostics.

Compute-node script under the production MPI launcher. Both wrapper return
branches run with real WFN data and a candidate count indivisible by the
process count, so returning a sliced candidate Gram violates the contract.
"""


def main():
    from pathlib import Path
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    import numpy as np
    from centroid.pivoted_cholesky import prune_candidates_by_pivoted_cholesky
    from wfn_loader import WfnLoader
    mesh = stack.mesh
    n = 2 * int(mesh.size) + 3
    rng = np.random.default_rng(258)
    rows = []
    with WfnLoader(str(Path(__file__).parent / 'hsuite/fixture_na/WFN.h5'), mesh=mesh) as wfn:
        grid = tuple(int(v) for v in wfn.fft_grid)
        indices = np.asarray(np.unravel_index(rng.choice(np.prod(grid), n, replace=False), grid)).T
        for grouped in (False, True):
            result = prune_candidates_by_pivoted_cholesky(
                wfn, wfn.symmetry(), indices, 8, mesh,
                band_range_left=(0, 5), band_range_right=(0, 5),
                orbit_id=(np.arange(n, dtype=np.int32)//2 if grouped else None),
                group_block=grouped, n_point_budget=(8 if grouped else None),
                verbose=False)
            assert len(result) == 7, len(result)
            kept, rank, final, taken, trace, psd, unpicked = result
            assert 0 < len(kept) <= 8 and kept.shape[1] == 3
            assert final.shape == (n,) and taken.ndim == trace.ndim == 1
            # Every diagnostic is host scalar/vector data; the capacity-size
            # candidate Gram must die inside the wrapper on both branches.
            assert not any(isinstance(v, jax.Array) for v in result)
            assert all(np.asarray(v).ndim <= 1 for v in result[1:])
            if grouped:
                kept_flat = np.ravel_multi_index(kept.T, grid)
                original_flat = np.ravel_multi_index(indices.T, grid)
                picked = np.flatnonzero(np.isin(original_flat, kept_flat))
                labels = np.arange(n)//2
                for label in np.unique(labels[picked]):
                    assert np.all(np.isin(np.flatnonzero(labels == label), picked))
            rows.append((grouped, len(kept), rank, float(unpicked)))
    if jax.process_index() == 0:
        print('CANDIDATE_PRUNE_RETURN PASS ' + repr(rows), flush=True)
    runtime.finalize_process(0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
