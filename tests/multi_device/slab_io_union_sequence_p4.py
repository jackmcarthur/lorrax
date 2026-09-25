"""P4: a union read followed by HDF5 work on the same open handle."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from file_io.slab_io import SlabIO
    from ffi.common.ffi_loader import loaded_lib_path

    assert jax.process_count() == 4
    mesh = RUNTIME.mesh
    args.output.parent.mkdir(parents=True, exist_ok=True)
    path = args.output.parent / "union_sequence.h5"
    sharding = NamedSharding(mesh, P("x", "y"))
    rows, cols = 4096, 256
    values = jax.make_array_from_callback(
        (2 * rows, cols), sharding,
        lambda index: np.full(
            tuple(s.stop - s.start for s in index),
            7.0 if index[0].start < rows else 11.0, dtype=np.float64))

    with SlabIO(str(path), mode="w", mesh=mesh) as io:
        io.write_slab("source", values)

    with SlabIO(str(path), mode="a", mesh=mesh) as io:
        for off, valid, reason in (
                ((0, 0), (rows + 1, cols), "valid_shape exceeds slab shape"),
                ((2 * rows - 1, 0), (2, cols), "valid slab exceeds dataset extent")):
            try:
                io.read_slabs(
                    "source", shape=(rows, cols), offsets=[off],
                    valid_shapes=[valid], partition_spec=P("x", "y"),
                    window_axis=0)
            except ValueError as exc:
                assert reason in str(exc), str(exc)
            else:
                raise AssertionError(f"read_slabs accepted {reason}")
        got = io.read_slabs(
            "source", shape=(rows, cols),
            offsets=[(0, 0), (rows, 0), (rows, 0)],
            valid_shapes=[(rows, cols), (rows, cols), (0, cols)],
            partition_spec=P("x", "y"), window_axis=0)
        # This metadata operation is the first HDF5 door after the async
        # union read.  The lane must wait for that read's native worker.
        io.create_dataset("next", shape=(2 * rows, cols), dtype=np.float64)
        io.write_slab("next", values)

    expected = jnp.asarray([7.0, 11.0, 0.0])[:, None, None]
    assert bool(jax.jit(lambda x: (x == expected).all())(got))
    with SlabIO(str(path), mode="r", mesh=mesh) as io:
        after = io.read_slab("next", shape=(2 * rows, cols),
                             partition_spec=P("x", "y"))
    assert bool(jax.jit(lambda x, y: (x == y).all())(after, values))
    if jax.process_index() == 0:
        args.output.write_text(json.dumps({
            "status": "PASS", "job": os.environ.get("SLURM_JOB_ID"),
            "step": os.environ.get("SLURM_STEP_ID"),
            "source_root": str(Path(__file__).resolve().parents[2]),
            "provider": loaded_lib_path("CUDA"),
            "scope": "P4 union window admission, same-handle read then "
                     "metadata and write; empty window zero-fill and exact readback",
        }, indent=2) + "\n")
    print(f"rank {jax.process_index()}: union sequence PASS", flush=True)


if __name__ == "__main__":
    from runtime import initialize_communicator_stack, run_main_and_finalize
    RUNTIME = initialize_communicator_stack(platform="gpu")
    run_main_and_finalize(main)
