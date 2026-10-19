"""Timing-only waits and the host barrier (CPU, seconds).

Owner rule: no host-device synchronization in a hot path.  A wait whose only purpose is
an honest stage clock runs under ``LORRAX_STAGE_SYNC=1`` alone (``common.timing.stage_sync``);
``collectives.barrier`` orders hosts through the coordination service, not a device
all-gather, and gives every use of a name its own barrier id.
"""
import jax
import jax.numpy as jnp


class _Probe:
    def __init__(self):
        self.waited = 0

    def block_until_ready(self):
        self.waited += 1
        return self


def _with_stage_sync(monkeypatch, value):
    from common import timing
    monkeypatch.setattr(timing, "_STAGE_SYNC", None)
    if value is None:
        monkeypatch.delenv("LORRAX_STAGE_SYNC", raising=False)
    else:
        monkeypatch.setenv("LORRAX_STAGE_SYNC", value)
    return timing


def test_watchers_and_stage_waits_run_only_under_stage_sync(monkeypatch):
    for value, expected in ((None, 0), ("0", 0), ("1", 1)):
        timing = _with_stage_sync(monkeypatch, value)
        assert timing.stage_sync() is bool(expected)
        probe = _Probe()
        with timing.section("stage_sync_probe") as sec:
            sec.watch(probe)
        assert probe.waited == expected
        timing.stage_wait(jnp.ones(3))          # an array is accepted either way


def test_progress_milestone_waits_only_under_stage_sync(monkeypatch):
    from common.progress import LoopProgress
    for value, expected in ((None, 0), ("1", 1)):
        timing = _with_stage_sync(monkeypatch, value)
        calls = []
        monkeypatch.setattr(jax, "block_until_ready", lambda x: calls.append(x) or x)
        bar = LoopProgress(1, lambda *_: None, title="probe", item_name="step", max_updates=1)
        bar.start()
        bar.step(wait=jnp.ones(2))
        assert len(calls) == expected, (value, calls)


def test_barrier_uses_the_coordination_service_with_one_id_per_use(monkeypatch):
    from common import collectives
    from jax._src import distributed

    class Client:
        def __init__(self):
            self.ids = []

        def wait_at_barrier(self, barrier_id, timeout_in_ms):
            self.ids.append((barrier_id, timeout_in_ms))

    client = Client()
    monkeypatch.setattr(distributed.global_state, "client", client)
    monkeypatch.setattr(collectives, "_BARRIER_SERIAL", {})

    class NoDevice:
        @staticmethod
        def sync_global_devices(name):
            raise AssertionError("device barrier used with a coordination client")
    for _ in range(2):
        collectives._coordination_barrier("probe", NoDevice)
    collectives._coordination_barrier("other", NoDevice)
    assert [i for i, _ in client.ids] == ["lorrax.barrier:probe:1", "lorrax.barrier:probe:2",
                                          "lorrax.barrier:other:1"]
    assert all(t == collectives.BARRIER_TIMEOUT_S * 1000 for _, t in client.ids)

    monkeypatch.setattr(distributed.global_state, "client", None)
    used = []

    class Device:
        @staticmethod
        def sync_global_devices(name):
            used.append(name)
    collectives._coordination_barrier("fallback", Device)
    assert used == ["fallback"]
    assert collectives.barrier("single_process_is_a_no_op") is False


def test_large_read_heartbeat_never_waits_on_the_caller(tmp_path, capsys):
    """A >= 1 GiB file's read is watched on a thread (is_ready polling), not waited for."""
    import time
    from file_io import _slab_io_ffi as ffi
    big = tmp_path / "big.h5"
    with open(big, "wb") as fh:
        fh.truncate(ffi._LARGE_READ_LAYOUT_BYTES)        # sparse: no bytes written

    class Late:
        def __init__(self, after):
            self.t = time.monotonic() + after

        def is_ready(self):
            return time.monotonic() >= self.t

        def block_until_ready(self):
            raise AssertionError("the heartbeat waited on the caller's thread")
    t0 = time.monotonic()
    with ffi._large_read_progress(str(big), "x", global_bytes=8, local_bytes=8) as watch:
        watch(Late(0.3))
    assert time.monotonic() - t0 < 0.2                  # returned at dispatch
    deadline, seen = time.monotonic() + 5.0, ""
    while "COMPLETE" not in seen and time.monotonic() < deadline:
        time.sleep(0.05)
        seen += capsys.readouterr().out
    assert "START" in seen or "COMPLETE" in seen
    assert "COMPLETE" in seen, seen
