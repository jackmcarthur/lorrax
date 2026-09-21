"""P16 SlabIO host-control acceptance with a post-failure NCCL receipt.

Run on four nodes with one rank per GPU.  The successful leg covers a sharded
write/read round trip and the read-only close fast path.  The failure leg
raises on rank one only after its real H5Dwrite completes; every rank must
receive the same host-control error.  A final tiny process gather proves the
device collective plane remains usable after the host receipt exchange.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from runtime import finalize_process, initialize_communicator_stack
    runtime = initialize_communicator_stack()
    import h5py
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import all_gather_processes, rank0_transaction
    from file_io.commit_state import COMMIT_STATE
    from file_io.slab_io import SlabIO

    mesh = runtime.mesh
    rank = int(jax.process_index())
    assert jax.process_count() == 16 and tuple(mesh.devices.shape) == (4, 4)
    rank0_transaction(
        args.directory, stage="io_host_control.prepare",
        write=lambda: args.directory.mkdir(parents=True, exist_ok=True))

    started = time.monotonic()
    shape = (32, 32)
    sharding = NamedSharding(mesh, P("x", "y"))
    data = jax.make_array_from_callback(
        shape, sharding,
        lambda index: np.arange(np.prod(shape), dtype=np.float64)
        .reshape(shape)[index])

    good = args.directory / "roundtrip.h5"
    with SlabIO(good, mode="w", mesh=mesh) as handle:
        handle.create_dataset(
            "A", shape=shape, dtype=np.float64,
            attrs={"contract": "host-control-p16"})
        handle.write_slab("A", data)
    with SlabIO(good, mode="r", mesh=mesh) as handle:
        restored = handle.read_slab("A", partition_spec=P("x", "y"))
        max_error = float(np.asarray(jax.device_get(
            jnp.max(jnp.abs(restored - data)))))
    assert max_error == 0.0

    failed = args.directory / "rank1_failure.h5"
    error = None
    try:
        with SlabIO(failed, mode="w", mesh=mesh) as handle:
            handle.create_dataset("A", shape=shape, dtype=np.float64)
            submit_real = handle._backend._dispatcher.submit

            def submit(task):
                def injected():
                    task()
                    if rank == 1:
                        raise OSError("injected rank-one queued write failure")
                submit_real(injected)

            handle._backend._dispatcher.submit = submit
            handle.write_slab("A", data)
        raise AssertionError("rank-one queued write failure was not agreed")
    except RuntimeError as exc:
        error = str(exc)
    assert "stage=SlabIO.data_close; failing rank=1" in error, error
    assert "injected rank-one queued write failure" in error, error

    gathered = np.asarray(
        all_gather_processes(np.asarray([rank], dtype=np.int32))).reshape(-1)
    np.testing.assert_array_equal(gathered, np.arange(16, dtype=np.int32))

    def validate_and_publish():
        with h5py.File(good, "r") as handle:
            assert int(handle[COMMIT_STATE][0]) == 1
            assert handle["A"].attrs["contract"] == "host-control-p16"
        args.output.write_text(json.dumps({
            "status": "PASS",
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True).strip(),
            "job": os.environ.get("SLURM_JOB_ID"),
            "step": os.environ.get("SLURM_STEP_ID"),
            "processes": 16,
            "mesh": [4, 4],
            "max_roundtrip_error": max_error,
            "agreed_failure": error,
            "elapsed_seconds": time.monotonic() - started,
            "scope": (
                "P16/four-node sharded SlabIO write/read, read-only close, "
                "one-rank queued-write failure agreement, and post-failure "
                "tiny NCCL process gather; no SC physics claim"),
        }, indent=2) + "\n")

    rank0_transaction(
        args.output, stage="io_host_control.publish",
        write=validate_and_publish)
    if rank == 0:
        print("IO_HOST_CONTROL_P16_PASS", flush=True)
    finalize_process()


if __name__ == "__main__":
    main()
