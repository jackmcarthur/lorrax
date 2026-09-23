"""P4 regression for restoring a factor whose active width is not Py aligned."""

import json
import os


def main():
    from runtime import initialize_communicator_stack, finalize_process

    initialize_communicator_stack()
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import gather_to_host, resolve_mesh
    from gw.shared_pole_local import batch_to_face, face_rows

    mesh = resolve_mesh()
    assert mesh.size == 4 and int(mesh.shape['y']) == 2
    source = np.arange(4 * 8 * 7, dtype=np.float64).reshape(4, 8, 7)
    batched = jax.make_array_from_callback(
        source.shape, NamedSharding(mesh, P(('x', 'y'))), lambda index: source[index])
    restored = batch_to_face(mesh)(batched)
    got = np.asarray(gather_to_host(restored))
    assert got.shape == (4, 8, 8), got.shape
    np.testing.assert_array_equal(got[..., :7], source)
    np.testing.assert_array_equal(got[..., 7], 0)
    rows = face_rows(mesh, (0, 2), width=8)(restored)
    np.testing.assert_array_equal(np.asarray(gather_to_host(rows)), got[[0, 2]])
    if jax.process_index() == 0:
        print(json.dumps(dict(status='PASS', source_shape=list(source.shape),
                              restored_shape=list(got.shape),
                              job=os.environ.get('SLURM_JOB_ID'),
                              step=os.environ.get('SLURM_STEP_ID'))), flush=True)
    finalize_process()


if __name__ == '__main__':
    main()
