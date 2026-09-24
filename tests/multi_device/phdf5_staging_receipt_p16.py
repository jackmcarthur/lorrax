"""Exercise PHDF5 staging growth while the process receipt is sampled.

This is the target-runtime check for the atomic ``pinned_capacity`` and
``read_capacity`` fields.  Each rank polls the native registry from one host
thread while the SlabIO worker grows its write buffer and the XLA read path
grows its read buffer.  The gate also proves that the requested freshly built
provider was loaded and that every distributed round trip is exact.

Run with one rank per GPU on four nodes::

    lx run -N 4 -G 4 -n 16 -- env LORRAX_FFI_SO=/abs/liblorrax_ffi.so \
      PYTHONPATH=/abs/checkout/src python3 -u \
      tests/multi_device/phdf5_staging_receipt_p16.py \
      --provider /abs/liblorrax_ffi.so --output /abs/receipt.json
"""

import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from runtime import initialize_communicator_stack, finalize_process
    runtime = initialize_communicator_stack()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import all_gather_processes, rank0_transaction
    from ffi.common.ffi_loader import loaded_lib_path
    from ffi.io import staging_totals
    from file_io.slab_io import SlabIO

    mesh = runtime.mesh
    assert jax.process_count() == 16 and mesh.size == 16, (
        jax.process_count(), mesh.size)
    assert tuple(mesh.devices.shape) == (4, 4), mesh.devices.shape
    sharding = NamedSharding(mesh, P("x", "y"))

    # Load before starting the sampling thread: this gate targets concurrent
    # registry/capacity access, not concurrent ctypes library discovery.
    initial = staging_totals(platform="CUDA")
    assert initial == (0, 0), initial
    loaded = os.path.realpath(loaded_lib_path("CUDA"))
    requested = os.path.realpath(args.provider)
    assert loaded == requested, (loaded, requested)
    provider_sha256 = _sha256(loaded)
    rank0_transaction(
        args.output.parent, stage="phdf5_receipt.prepare",
        write=lambda: args.output.parent.mkdir(parents=True, exist_ok=True))

    def sample_during(action, *, expect_monotonic):
        stop = threading.Event()
        samples = []
        errors = []

        def poll():
            previous = 0
            try:
                while not stop.is_set():
                    value = staging_totals(platform="CUDA")
                    if value is None:
                        raise AssertionError(
                            "fresh provider has no staging_totals accessor")
                    live, staged = (int(value[0]), int(value[1]))
                    if live < 0 or staged < 0:
                        raise AssertionError((live, staged))
                    if expect_monotonic and live > 0 and staged < previous:
                        raise AssertionError(
                            f"live staging receipt decreased {previous}->{staged}")
                    if live > 0:
                        previous = staged
                    samples.append((live, staged))
                    time.sleep(0.0005)
            except BaseException as exc:
                errors.append(repr(exc))

        thread = threading.Thread(target=poll, name="phdf5-receipt-poll")
        thread.start()
        try:
            result = action()
        finally:
            stop.set()
            thread.join(timeout=30.0)
        assert not thread.is_alive(), "staging receipt poll thread did not exit"
        assert not errors, errors
        live_samples = [value for value in samples if value[0] > 0]
        assert len(live_samples) >= 10, len(live_samples)
        assert max(value[1] for value in live_samples) > 0, live_samples[-10:]
        return result, samples

    sizes = (512, 1024, 1536, 2048)
    arrays = {}
    data_path = args.output.parent / "phdf5_staging_receipt.h5"

    def write_phase():
        with SlabIO(data_path, mode="w", mesh=mesh) as handle:
            for size in sizes:
                def tile(index, n=size):
                    rows = np.arange(index[0].start, index[0].stop)[:, None]
                    cols = np.arange(index[1].start, index[1].stop)[None, :]
                    return (rows * n + cols).astype(np.float64)
                array = jax.make_array_from_callback(
                    (size, size), sharding, tile)
                arrays[size] = array
                handle.write_slab(f"square_{size}", array)
                handle.sync_writes()

    _, write_samples = sample_during(write_phase, expect_monotonic=True)
    assert staging_totals(platform="CUDA") == (0, 0)

    def read_phase():
        errors = []
        with SlabIO(data_path, mode="r", mesh=mesh) as handle:
            for size in sizes:
                value = handle.read_slab(
                    f"square_{size}", shape=(size, size),
                    partition_spec=P("x", "y"))
                errors.append(float(np.asarray(
                    jax.device_get(jnp.max(jnp.abs(value - arrays[size]))))))
        return errors

    roundtrip_errors, read_samples = sample_during(
        read_phase, expect_monotonic=False)
    assert staging_totals(platform="CUDA") == (0, 0)
    assert max(roundtrip_errors) == 0.0, roundtrip_errors

    local = np.asarray([
        len(write_samples), len(read_samples),
        max(staged for live, staged in write_samples if live > 0),
        max(staged for live, staged in read_samples if live > 0),
    ], dtype=np.int64)
    all_receipts = np.asarray(all_gather_processes(local), dtype=np.int64)
    result = {
        "status": "PASS",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parents[2]),
             "rev-parse", "HEAD"], text=True).strip(),
        "provider": loaded,
        "provider_sha256": provider_sha256,
        "job": os.environ.get("SLURM_JOB_ID"),
        "step": os.environ.get("SLURM_STEP_ID"),
        "processes": int(jax.process_count()),
        "mesh": list(mesh.devices.shape),
        "sizes": list(sizes),
        "max_roundtrip_error": max(roundtrip_errors),
        "per_rank_poll_and_peak_bytes": all_receipts.tolist(),
        "scope": (
            "P16/four-node concurrent native staging receipt sampling "
            "during four increasing SlabIO writes and reads; exact "
            "distributed round trip; no SC physics or performance claim"),
    }
    if jax.process_index() == 0:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finalize_process()


if __name__ == "__main__":
    main()
