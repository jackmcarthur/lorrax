"""Focused host-schedule tests for the private coupled transverse Zq route."""

import threading

import jax.numpy as jnp
import numpy as np

from gw.gw_init import _CoupledMu123ZqCoordinator


class _DummyStack:
    def __init__(self, events):
        self._events = events

    def block_until_ready(self):
        self._events.append(("ready",))
        return self

    def __getitem__(self, index):
        return index + 1


def test_coupled_coordinator_builds_once_and_orders_consumers_and_finalizers():
    coordinator = _CoupledMu123ZqCoordinator()
    events = []
    event_lock = threading.Lock()

    def record(*event):
        with event_lock:
            events.append(event)

    def worker(mu):
        coordinator.channel_prepared(mu)
        for chunk in range(2):
            def build(mu=mu, chunk=chunk):
                record("build", mu, chunk)
                return _DummyStack(events)

            value = coordinator.acquire_channel_Z_q(mu, chunk, build)
            record("consume", mu, chunk, value)
            coordinator.finish_chunk(mu, chunk)
        coordinator.wait_finalize(mu)
        record("finalize", mu)
        coordinator.finish_channel(mu)

    threads = []
    for mu in (1, 2, 3):
        thread = threading.Thread(target=worker, args=(mu,))
        thread.start()
        threads.append(thread)
        coordinator.wait_channel_prepared(mu)
    coordinator.release_channels()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert [event for event in events if event[0] == "build"] == [
        ("build", 1, 0), ("build", 1, 1)]
    assert [event[:3] for event in events if event[0] == "consume"] == [
        ("consume", 1, 0), ("consume", 2, 0), ("consume", 3, 0),
        ("consume", 1, 1), ("consume", 2, 1), ("consume", 3, 1)]
    assert [event for event in events if event[0] == "finalize"] == [
        ("finalize", 1), ("finalize", 2), ("finalize", 3)]


def test_final_write_turn_waits_for_all_channel_loops():
    coordinator = _CoupledMu123ZqCoordinator()
    entered = {mu: threading.Event() for mu in (1, 2, 3)}
    release = {mu: threading.Event() for mu in (1, 2, 3)}

    def worker(mu):
        coordinator.wait_finalize(mu)
        entered[mu].set()
        assert release[mu].wait(timeout=5)
        coordinator.finish_channel(mu)

    first = threading.Thread(target=worker, args=(1,))
    first.start()
    # μ=1 owns the first write turn, but it must not enter that turn until
    # μ=2 and μ=3 have both reported that their r-chunk loops are complete.
    assert not entered[1].wait(timeout=0.1)

    second = threading.Thread(target=worker, args=(2,))
    third = threading.Thread(target=worker, args=(3,))
    second.start()
    third.start()
    assert entered[1].wait(timeout=5)
    assert not entered[2].is_set()
    assert not entered[3].is_set()
    release[1].set()
    assert entered[2].wait(timeout=5)
    assert not entered[3].is_set()
    release[2].set()
    assert entered[3].wait(timeout=5)
    release[3].set()
    for thread in (first, second, third):
        thread.join(timeout=5)
        assert not thread.is_alive()


def test_prepared_channels_publish_one_mu_major_solve_stack():
    coordinator = _CoupledMu123ZqCoordinator()
    seen = {}

    def worker(mu):
        factor = jnp.full((2, 1, 1), mu, dtype=jnp.complex128)
        trace = jnp.full((2,), 10 * mu, dtype=jnp.complex128)
        coordinator.channel_prepared(mu, solve_inputs=(factor, trace))
        seen[mu] = coordinator.stacked_solve_inputs()

    threads = []
    for mu in (1, 2, 3):
        thread = threading.Thread(target=worker, args=(mu,))
        thread.start()
        threads.append(thread)
        coordinator.wait_channel_prepared(mu)
    coordinator.release_channels()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    factor, trace = seen[1]
    np.testing.assert_array_equal(
        np.asarray(factor[:, 0, 0]), [1, 1, 2, 2, 3, 3])
    np.testing.assert_array_equal(
        np.asarray(trace), [10, 10, 20, 20, 30, 30])
    assert all(seen[mu][0] is factor for mu in (1, 2, 3))
    assert coordinator._solve_inputs == {}
