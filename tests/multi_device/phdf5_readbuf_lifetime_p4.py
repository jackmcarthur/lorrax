"""P4 PHDF5 read staging lifetime and exact SlabIO round trip.

Run with one rank per GPU and a private provider built from this checkout.
Four read-only file contexts stay open while each reads a 48 MiB/rank slab.
The receipt must return to zero after every completed H2D, rather than
retaining one slab per file. A pair of small reads checks cached reuse.
"""

import argparse
import hashlib
import json
import os
import threading
import time
from contextlib import ExitStack
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path,
                        help="Read an immutable four-file fixture from a prior run")
    parser.add_argument("--small-only", action="store_true",
                        help="Check two in-flight cached reads without large slabs")
    args = parser.parse_args()

    from runtime import initialize_communicator_stack, finalize_process
    runtime = initialize_communicator_stack()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import all_gather_processes
    from ffi.common.ffi_loader import loaded_lib_path
    from ffi.io import staging_totals
    from file_io.slab_io import SlabIO

    mesh = runtime.mesh
    assert jax.process_count() == 4 and tuple(mesh.devices.shape) == (2, 2)
    sharding = NamedSharding(mesh, P("x", "y"))
    shape = (4096, 6144)  # 48 MiB/rank, above the 32 MiB retention cap.
    local_bytes = int(np.prod(shape) * 8 // 4)
    assert local_bytes == 48 * 1024 * 1024

    args.output.parent.mkdir(parents=True, exist_ok=True)
    initial = staging_totals(platform="CUDA")
    assert initial == (0, 0), initial
    provider = os.path.realpath(loaded_lib_path("CUDA"))
    assert provider == os.path.realpath(args.provider), (provider, args.provider)
    data_dir = args.input_dir or args.output.parent
    paths = [data_dir / f"readbuf_{i}.h5" for i in range(4)]

    if args.input_dir is None:
        for i, path in enumerate(paths):
            def tile(index, fill=float(i + 1)):
                return np.full(
                    tuple(s.stop - s.start for s in index), fill, dtype=np.float64)
            value = jax.make_array_from_callback(shape, sharding, tile)
            with SlabIO(path, mode="w", mesh=mesh) as handle:
                handle.write_slab("large", value, global_shape=shape)
                if i == 0:
                    small = jax.make_array_from_callback(
                        (16, 16), sharding,
                        lambda index: np.full(
                            tuple(s.stop - s.start for s in index),
                            7.0, dtype=np.float64))
                    handle.write_slab("small", small, global_shape=(16, 16))
                handle.sync_writes()
            assert staging_totals(platform="CUDA") == (0, 0)

    observations = []
    with ExitStack() as stack:
        handles = [stack.enter_context(SlabIO(path, mode="r", mesh=mesh))
                   for path in paths]
        assert staging_totals(platform="CUDA") == (4, 0)
        # Issue both reads before consuming either result: the second
        # H5Dread may otherwise overwrite the first in-flight H2D source.
        small_pair = (
            handles[0].read_slab(
                "small", shape=(16, 16), partition_spec=P("x", "y")),
            handles[0].read_slab(
                "large", shape=(16, 16), offset=(0, 0),
                partition_spec=P("x", "y")),
        )
        for small, expected in zip(small_pair, (7.0, 1.0)):
            assert float(jax.device_get(
                jnp.max(jnp.abs(small - expected)))) == 0.0
        small_staging = staging_totals(platform="CUDA")
        assert small_staging[0] == 4 and 0 < small_staging[1] <= 2 * 1024 * 1024
        for i, handle in enumerate(() if args.small_only else handles):
            stop = threading.Event()
            samples = []
            def sample():
                while not stop.is_set():
                    samples.append(staging_totals(platform="CUDA"))
                    time.sleep(0.005)
            sampler = threading.Thread(target=sample, name="readbuf-receipt")
            sampler.start()
            started = time.perf_counter()
            try:
                large = handle.read_slab(
                    "large", shape=shape, partition_spec=P("x", "y"))
                dispatch_s = time.perf_counter() - started
                error = float(jax.device_get(
                    jnp.max(jnp.abs(large - float(i + 1)))))
                ready_s = time.perf_counter() - started
            finally:
                stop.set()
                sampler.join(timeout=30.0)
            assert not sampler.is_alive()
            assert samples
            assert error == 0.0, (i, error)
            peak_after_dispatch = staging_totals(platform="CUDA")
            deadline = time.monotonic() + 30.0
            while True:
                settled = staging_totals(platform="CUDA")
                if settled == (4, 0):
                    break
                assert time.monotonic() < deadline, (i, settled)
                time.sleep(0.01)
            observations.append({
                "file": i, "dispatch_s": dispatch_s, "ready_s": ready_s,
                "peak_staging_bytes": max(n for _, n in samples),
                "staging_after_dispatch": peak_after_dispatch[1],
                "staging_after_dma": settled[1], "error": error,
            })
    assert staging_totals(platform="CUDA") == (0, 0)

    local = np.asarray(
        [round(1000 * max((x["dispatch_s"] for x in observations), default=0)),
         round(1000 * max((x["ready_s"] for x in observations), default=0)),
         max((x["peak_staging_bytes"] for x in observations), default=0)],
        dtype=np.int64)
    all_receipts = np.asarray(all_gather_processes(local), dtype=np.int64)
    result = {
        "status": "PASS", "job": os.environ.get("SLURM_JOB_ID"),
        "step": os.environ.get("SLURM_STEP_ID"),
        "provider": provider,
        "provider_sha256": hashlib.sha256(Path(provider).read_bytes()).hexdigest(),
        "source_root": str(Path(__file__).resolve().parents[2]),
        "processes": int(jax.process_count()), "mesh": list(mesh.devices.shape),
        "large_local_read_bytes": 0 if args.small_only else local_bytes,
        "small_local_read_bytes": 512,
        "small_staging_bytes": small_staging[1],
        "small_read_expected_values": [7.0, 1.0],
        "small_only": args.small_only,
        "input_dir": str(data_dir),
        "observations_rank0": observations if jax.process_index() == 0 else None,
        "per_rank_max_dispatch_ms_ready_ms_peak_bytes": all_receipts.tolist(),
        "scope": (
            "P4 same-context 7/1 small-read reuse before either result is "
            "consumed; four contexts open; no large slabs"
            if args.small_only else
            "P4 exact SlabIO reads, four open contexts, native large-staging "
            "release and distinct-value cached reuse"),
    }
    if jax.process_index() == 0:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finalize_process()


if __name__ == "__main__":
    main()
