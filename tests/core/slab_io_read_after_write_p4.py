"""P4: a write that fails on ONE rank surfaces at the next read on that handle.

Rank 1's second queued write raises AFTER its real collective H5Dwrite
returned (the async writer-error channel, not a death inside MPI).  The read
that follows on the same handle must raise the same ``SlabIO.
read_after_write`` refusal on every rank; before the fix every rank read the
dataset silently and the error surfaced only at close.  Each rank exits 0
only if its own read raised that refusal and its close completed.
"""
import sys
from pathlib import Path


def main(argv=None):
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P
    from file_io.slab_io import SlabIO

    rank = jax.process_index()
    assert jax.process_count() == 4, jax.process_count()
    out = Path((argv or sys.argv[1:])[0])
    out.mkdir(parents=True, exist_ok=True)
    mesh = RUNTIME.mesh
    sharding = NamedSharding(mesh, P("x", "y"))

    def block(t):
        host = np.arange(64, dtype=np.float64).reshape(8, 8) + 100.0 * t
        return jax.make_array_from_callback((8, 8), sharding,
                                            lambda index: host[index])

    read_error = close_error = None
    try:
        with SlabIO(str(out / "read_after_write.h5"), mode="w", mesh=mesh) as io:
            io.create_dataset("a", shape=(16, 8), dtype=np.float64)
            dispatcher = io._backend._dispatcher
            submit, count = dispatcher.submit, [0]

            def failing_submit(task):
                count[0] += 1
                k = count[0]

                def queued():
                    task()
                    if rank == 1 and k == 2:
                        raise OSError("injected rank-1 write failure after "
                                      "its collective H5Dwrite")
                submit(queued)

            dispatcher.submit = failing_submit
            io.write_slab("a", block(0), offset=(0, 0))
            io.write_slab("a", block(1), offset=(8, 0))
            try:
                io.read_slab("a", shape=(8, 8), offset=(8, 0),
                             partition_spec=P("x", "y"))
            except RuntimeError as exc:
                read_error = str(exc)
    except RuntimeError as exc:
        close_error = str(exc)
    print(f"rank {rank} read_error={read_error!r} close_error={close_error!r}",
          flush=True)
    assert read_error is not None, "the read after a failed write did not raise"
    assert "stage=SlabIO.read_after_write" in read_error, read_error
    assert "failing rank=1" in read_error, read_error
    assert close_error is not None and "SlabIO.data_close" in close_error, close_error
    print("done: read_after_write refused on every rank", flush=True)
    return 0


if __name__ == "__main__":
    from runtime import initialize_communicator_stack, run_main_and_finalize
    RUNTIME = initialize_communicator_stack(platform="gpu")
    run_main_and_finalize(main)
