"""Group progress is optional, globally agreed, and numerically neutral.

Run directly under the production compute-node launcher with
LORRAX_DEBUG_PRINT=0. A rank-skewed debug setting must refuse before any
WFN or selection work; both agreed modes then select the same real-WFN
points and diagnostics bit for bit.
"""


def main():
    from pathlib import Path
    import runtime
    stack = runtime.initialize_communicator_stack()
    import jax
    import numpy as np
    from centroid import pivoted_cholesky as pc
    from wfn_loader import WfnLoader
    mesh = stack.mesh
    n = 2 * int(mesh.size) + 3
    if jax.process_count() > 1:
        runtime.set_verbose(jax.process_index() == 0)
        try:
            pc.prune_candidates_by_pivoted_cholesky(
                None, None, np.zeros((n, 3), np.int64), 8, mesh,
                orbit_id=np.arange(n)//2, group_block=True, n_point_budget=8)
        except ValueError as exc:
            assert 'verbosity differs across ranks' in str(exc), str(exc)
        else:
            raise AssertionError('rank-skewed progress effects must refuse')
    runtime.set_verbose(False)
    recorded = []
    make = pc._make_sharded_block_select
    def observed(*args, **kwargs):
        recorded.append(kwargs['progress_fn'] is not None)
        return make(*args, **kwargs)
    pc._make_sharded_block_select = observed
    previous = None
    try:
        with WfnLoader(str(Path(__file__).parent/'hsuite/fixture_na/WFN.h5'), mesh=mesh) as wfn:
            grid = tuple(map(int, wfn.fft_grid))
            indices = np.asarray(np.unravel_index(
                np.random.default_rng(258).choice(np.prod(grid), n, replace=False), grid)).T
            for enabled in (False, True):
                runtime.set_verbose(enabled)
                result = pc.prune_candidates_by_pivoted_cholesky(
                    wfn, wfn.symmetry(), indices, 8, mesh,
                    band_range_left=(0, 5), band_range_right=(0, 5),
                    orbit_id=np.arange(n)//2, group_block=True, n_point_budget=8,
                    verbose=False)
                assert recorded[-1] is enabled, recorded
                if previous is not None:
                    assert all(np.array_equal(a, b) for a, b in zip(previous, result))
                previous = result
        assert recorded == [False, True], recorded
    finally:
        pc._make_sharded_block_select = make
        runtime.set_verbose(False)
    if jax.process_index() == 0:
        print('CANDIDATE_PROGRESS PASS: rank-skew guard; INFO/debug numerical parity', flush=True)
    runtime.finalize_process(0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
