"""The candidate sketch reads each k batch as one k tile, whatever its batching.

Compute-node script for P4 (``srun -n 4``); the pytest entry skips without the
launcher.  On the hsuite H2- fixture (25 full-zone k over several raw parents,
9 bands: one band tile) the q=0 candidate sketch
(:func:`centroid.sketch_select.build_sketch_q0_via_loadwfns`) is built twice:
with the fixed tile, which admits every k row in one batch, and with a tile
that admits one k row per batch.

(a) The two batchings build the same sketch ``Y``.
(b) The WFN is read once per band tile and k batch: one collective read for
    the single batch, whatever the number of raw parents it holds, and one per
    batch for the k-row batching.  The one-batch build used to stream one raw
    parent per read (k tile 1), which multiplied the reads by the parent count
    and stalled the KNiF3 sketch on CFS (64 parents: 832 reads instead of 13).
"""


def main():
    import runtime
    stack = runtime.initialize_communicator_stack()
    from pathlib import Path
    import jax
    import numpy as np
    from common.collectives import gather_to_host
    from wfn_loader import WfnLoader
    import centroid.sketch_select as sketch
    from centroid.sampling_metric import full_k_quadrature_weights
    from runtime.tiles import TILE_BYTES

    mesh = stack.mesh
    assert mesh.size == 4, mesh
    fixture = Path(__file__).parent / 'hsuite/fixture/WFN.h5'
    with WfnLoader(fixture, mesh=mesh) as wfn:
        sym = wfn.symmetry()
        n_parents = len(wfn.full_k_parent_groups())
        assert n_parents > 1, n_parents
        rng = np.random.default_rng(20261120)
        grid = np.asarray(wfn.fft_grid)
        flat = rng.choice(int(np.prod(grid)), size=37, replace=False)
        cand_idx = np.stack(np.unravel_index(flat, tuple(grid)), axis=1).astype(np.int64)
        weights = full_k_quadrature_weights(wfn, sym)
        right = (0, int(wfn.nbands))
        left = (0, 3)

        reads = []
        load = wfn.load

        def counted(*args, **kwargs):
            reads.append(kwargs.get('k'))
            return load(*args, **kwargs)
        wfn.load = counted

        def build():
            reads.clear()
            Y = sketch.build_sketch_q0_via_loadwfns(
                wfn, sym, cand_idx, mesh, n_rows=48, seed=5,
                band_range_left=left, band_range_right=right, k_weights=weights)
            return np.asarray(gather_to_host(Y)), len(reads)

        Y_one, reads_one = build()
        schedule = sketch.candidate_k_schedule

        def one_row_per_batch(wfn_, sym_, k_weights, per_k):
            return schedule(wfn_, sym_, k_weights, float(TILE_BYTES))
        sketch.candidate_k_schedule = one_row_per_batch
        try:
            n_batches = one_row_per_batch(wfn, sym, weights, 0.0)[5]
            Y_many, reads_many = build()
        finally:
            sketch.candidate_k_schedule = schedule
            wfn.load = load

    nk = int(np.count_nonzero(weights > 0))
    assert n_batches == nk > 1, (n_batches, nk)
    # One band tile (9 bands <= the 64-band chunk): reads = batches.
    assert reads_one == 1, (reads_one, n_parents)
    assert reads_many == n_batches, (reads_many, n_batches)
    scale = float(np.abs(Y_one).max())
    assert scale > 0
    np.testing.assert_allclose(Y_many, Y_one, rtol=0, atol=1e-12 * scale)
    print(f'SKETCH_K_BATCHES PASS rank={jax.process_index()} parents={n_parents} '
          f'nk={nk} reads one batch={reads_one} k-row batches={reads_many} '
          f'max|dY|/max|Y|={float(np.abs(Y_many - Y_one).max()) / scale:.1e}', flush=True)
    runtime.finalize_process(0)


def test_requires_distributed_launcher():
    import pytest
    pytest.skip('run directly under the configured MPI launcher')


if __name__ == '__main__':
    main()
