"""P4 regression: two SlabIO readers share one live PHDF5 context."""

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from runtime import initialize_communicator_stack, finalize_process
    runtime = initialize_communicator_stack()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.collectives import all_gather_processes
    from ffi.common.ffi_loader import loaded_lib_path
    from ffi.io import _DS_ID_MEMO, staging_totals
    from file_io.slab_io import SlabIO

    mesh = runtime.mesh
    assert jax.process_count() == 4 and tuple(mesh.devices.shape) == (2, 2)
    sharding = NamedSharding(mesh, P("x", "y"))
    path = args.output.parent / "same_path_owners.h5"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assert staging_totals(platform="CUDA") == (0, 0)

    alpha = jax.make_array_from_callback(
        (16, 16), sharding,
        lambda index: np.full(tuple(s.stop - s.start for s in index),
                              1.0, dtype=np.float64))
    beta = jax.make_array_from_callback(
        (16, 16), sharding,
        lambda index: np.full(tuple(s.stop - s.start for s in index),
                              2.0 if index[0].start < 8 else 3.0,
                              dtype=np.float64))
    with SlabIO(path, mode="w", mesh=mesh) as writer:
        writer.write_slab("alpha", alpha, global_shape=(16, 16))
        writer.write_slab("beta", beta, global_shape=(16, 16))
        writer.sync_writes()
    assert staging_totals(platform="CUDA") == (0, 0)

    first = SlabIO(path, mode="r", mesh=mesh)
    second = SlabIO(path, mode="r", mesh=mesh)
    try:
        assert first._backend.fh == second._backend.fh
        ctx = int(first._backend.fh)
        assert staging_totals(platform="CUDA")[0] == 1
        try:
            with SlabIO(path, mode="w", mesh=mesh):
                pass
        except RuntimeError as exc:
            assert "live context" in str(exc), str(exc)
        else:
            raise AssertionError("duplicate writer was accepted")
        assert path.exists()
        one = first.read_slab("alpha", shape=(16, 16),
                              partition_spec=P("x", "y"))
        two = second.read_slab("beta", shape=(16, 16),
                               partition_spec=P("x", "y"))
        assert float(jax.device_get(jnp.max(jnp.abs(one - 1.0)))) == 0.0
        expected_beta = jnp.where(jnp.arange(16)[:, None] < 8, 2.0, 3.0)
        assert float(jax.device_get(jnp.max(jnp.abs(two - expected_beta)))) == 0.0

        offsets = [(0, 0), (8, 0)]
        valid = [(8, 16), (8, 16)]
        packed = second.read_slabs(
            "beta", shape=(8, 16), offsets=offsets, valid_shapes=valid,
            partition_spec=P("x", "y"), window_axis=0)
        assert float(jax.device_get(jnp.max(jnp.abs(
            packed - jnp.asarray([2.0, 3.0])[:, None, None])))) == 0.0
        memo_before = {k: v for k, v in _DS_ID_MEMO.items() if k[0] == ctx}
        assert memo_before

        first.close()
        assert staging_totals(platform="CUDA")[0] == 1
        assert {k: v for k, v in _DS_ID_MEMO.items() if k[0] == ctx} == memo_before
        after = second.read_slabs(
            "beta", shape=(8, 16), offsets=offsets, valid_shapes=valid,
            partition_spec=P("x", "y"), window_axis=0)
        assert float(jax.device_get(jnp.max(jnp.abs(
            after - jnp.asarray([2.0, 3.0])[:, None, None])))) == 0.0
    finally:
        first.close()
        second.close()
    assert staging_totals(platform="CUDA") == (0, 0)
    assert not any(k[0] == ctx for k in _DS_ID_MEMO)

    local = np.asarray([ctx != 0, len(memo_before)], dtype=np.int64)
    gathered = np.asarray(all_gather_processes(local), dtype=np.int64)
    result = {
        "status": "PASS", "job": os.environ.get("SLURM_JOB_ID"),
        "step": os.environ.get("SLURM_STEP_ID"),
        "source_root": str(Path(__file__).resolve().parents[2]),
        "provider": loaded_lib_path("CUDA"),
        "processes": int(jax.process_count()), "mesh": list(mesh.devices.shape),
        "path": str(path), "per_rank_nonzero_handle_and_memo_count": gathered.tolist(),
        "native_contexts_after_two_opens": 1,
        "native_contexts_after_first_close": 1,
        "native_contexts_after_final_close": 0,
        "scope": "P4 same-path read-only SlabIO owners, duplicate-writer "
                 "preflight, distinct slab reads, kchunk dataset memo "
                 "retained after first close, final cleanup",
    }
    if jax.process_index() == 0:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finalize_process()


if __name__ == "__main__":
    main()
