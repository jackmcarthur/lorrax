"""P4 regression for restoring a factor whose active width is not Py aligned."""

import json
import os
from pathlib import Path
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from runtime import initialize_communicator_stack, finalize_process

    initialize_communicator_stack()
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host, resolve_mesh
    from gw.shared_pole_local import batch_to_face, face_rows

    mesh = resolve_mesh()
    assert mesh.size == 4 and int(mesh.shape['y']) == 2
    source = np.arange(4 * 8 * 7, dtype=np.float64).reshape(4, 8, 7).astype(np.complex128)
    batched = jax.make_array_from_callback(
        source.shape, NamedSharding(mesh, P(('x', 'y'))), lambda index: source[index])
    restored = batch_to_face(mesh)(batched)
    got = np.asarray(gather_to_host(restored))
    assert got.shape == (4, 8, 8), got.shape
    np.testing.assert_array_equal(got[..., :7], source)
    np.testing.assert_array_equal(got[..., 7], 0)
    rows = face_rows(mesh, (0, 2), width=8)(restored)
    np.testing.assert_array_equal(np.asarray(gather_to_host(rows)), got[[0, 2]])
    # The storage handoff has the padded carrier; its active count and
    # on-disk extent are seven. The extra factor/pole column is inert.
    from gw.shared_pole_local import canonical_factors
    from file_io import shared_pole_store as store
    from file_io.slab_io import SlabIO
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from test_shared_pole_store import _fixture
    meta, tables, recipe, identity = _fixture(mesh)
    public = canonical_factors(mesh, (0, 1, 2))(face_rows(mesh, (0, 1, 2), width=8)(restored))
    poles = np.pad(np.tile(np.arange(1, 8, dtype=np.float64), (3, 1)),
                   ((0, 0), (0, 1)), constant_values=1.0)
    pole_array = jax.make_array_from_callback(
        poles.shape, NamedSharding(mesh, P()), lambda index: poles[index])
    header = store.write_shared_pole_model(
        args.output, public, pole_array, np.full(3, 7, np.int64), q_span=(0, 3),
        meta=meta, tables=tables, recipe=recipe,
        receipts={'identity': identity, 'scope': 'factor-carrier-regression'})
    assert header['finalized'] and header['Kmax'] == 7
    validated = store.validate_shared_pole_model(
        args.output, expected_identity=identity, mesh_xy=mesh,
        capacity=meta.shared_pole_capacity)
    assert validated['digest'] == header['digest']
    with SlabIO(args.output, mode='r', mesh=mesh) as io:
        stored_poles, counts = store.read_shared_pole_census(
            io, header=validated, capacity=meta.shared_pole_capacity)
    np.testing.assert_array_equal(np.asarray(stored_poles), poles[:, :7])
    np.testing.assert_array_equal(np.asarray(counts), np.full(3, 7))
    if jax.process_index() == 0:
        import h5py
        with h5py.File(args.output, 'r') as h5:
            assert h5['factor'].shape == (3, 7, 1, 7)
            assert h5['poles2_ry2'].shape == (3, 7)
        print(json.dumps(dict(status='PASS', source_shape=list(source.shape),
                              restored_shape=list(got.shape),
                              final_Kmax=header['Kmax'],
                              job=os.environ.get('SLURM_JOB_ID'),
                              step=os.environ.get('SLURM_STEP_ID'))), flush=True)
    finalize_process()


if __name__ == '__main__':
    main()
