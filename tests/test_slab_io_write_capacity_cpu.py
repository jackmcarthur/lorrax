"""The real bounded writer retains previous inputs separately from its producer."""
import threading
from types import SimpleNamespace
import weakref

import pytest


def test_blocked_collective_lane_capacity_matches_live_inputs():
    from file_io._slab_io_ffi import _CollectiveLane, _FfiBackend
    from file_io._slab_io_serial import _SerialBackend
    lane = _CollectiveLane()
    slot = SimpleNamespace(_run=lambda task: task)
    worker_started, release_worker = threading.Event(), threading.Event()
    producer_started, producer_done = threading.Event(), threading.Event()
    refs = []

    class Input:
        pass

    def submit(block=False):
        value = Input()
        refs.append(weakref.ref(value))
        def task(value=value):
            if block:
                worker_started.set()
                assert release_worker.wait(5)
        lane.submit(slot, task)

    def producer():
        producer_started.set()
        submit()
        producer_done.set()

    try:
        submit(True)
        assert worker_started.wait(2)
        for _ in range(lane.queued_write_inputs):
            submit()
        expected = _FfiBackend.__new__(_FfiBackend).retained_previous_write_inputs
        assert sum(r() is not None for r in refs) == expected
        thread = threading.Thread(target=producer)
        thread.start()
        assert producer_started.wait(2)
        # The worker is deliberately stopped; queue.put must apply backpressure.
        assert not producer_done.wait(.05)
        assert sum(r() is not None for r in refs) == expected + 1
        release_worker.set()
        thread.join(3)
        assert not thread.is_alive() and producer_done.is_set()
        lane._worker().drain()
        assert not any(r() is not None for r in refs)
        assert _SerialBackend.__new__(_SerialBackend).retained_previous_write_inputs == 0
    finally:
        release_worker.set()
        lane._worker().close()


def test_finalize_price_includes_writer_queue_and_current_producer():
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    import numpy as np
    from file_io._slab_io_ffi import _FfiBackend
    from isdf import zeta_mubatch as owner
    jax.config.update('jax_enable_x64', True)
    mesh = Mesh(np.array(jax.devices('cpu')[:1]).reshape(1, 1), ('x', 'y'))
    shape = jax.ShapeDtypeStruct((1, 8, 8), jnp.complex128,
                               sharding=NamedSharding(mesh, P(None, ('x', 'y'), None)))
    program = jax.jit(lambda x: x).lower(shape).compile()
    io = SimpleNamespace(
        retained_previous_write_inputs=_FfiBackend.__new__(_FfiBackend).retained_previous_write_inputs,
        write_device_scratch=lambda _name, _shape: 0)
    tile = owner._local_bytes(shape)
    peak = owner._single_pass_peak(tile, 0, 0, 0, (program, shape), io)
    assert peak >= (io.retained_previous_write_inputs + 1) * tile

