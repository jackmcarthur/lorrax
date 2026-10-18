"""The hsuite first-collective watchdog (tests/hsuite/watchdog.py).  Stdlib only."""
import os
import threading

from tests.hsuite import watchdog

# --- the hsuite first-collective watchdog ------------------------------------------

def test_watchdog_fires_once_names_rank_node_gpu_coordinator_and_exits_1():
    read_fd, write_fd = os.pipe()
    seen, exited = [], threading.Event()
    codes = []

    def fake_exit(code):
        codes.append(code)
        exited.set()

    guard = watchdog.FirstCollectiveWatchdog(
        stage="kmeans", rank=3, count=16, coordinator="frontier00079:62146", timeout=0.05,
        fds=(write_fd,), writers=(seen.append,), exit=fake_exit,
        environ={"ROCR_VISIBLE_DEVICES": "5"}).start()
    assert exited.wait(5)
    os.close(write_fd)
    text = os.read(read_fd, 65536).decode()
    os.close(read_fd)
    assert codes == [1] and guard.fired and seen == [text]
    for part in ("rank 3/16", "ROCR_VISIBLE_DEVICES=5", "stage kmeans", "frontier00079:62146",
                 "faulty node or GPU"):
        assert part in text
    assert watchdog.placement()["host"] in text


def test_watchdog_disarmed_in_time_never_fires():
    codes = []
    guard = watchdog.FirstCollectiveWatchdog(stage="kmeans", rank=0, count=4, coordinator="x",
                                             timeout=0.5, exit=codes.append).start()
    guard.done()
    guard._thread.join(2)
    assert not guard._thread.is_alive() and codes == [] and not guard.fired


def test_watchdog_timeout_default_and_override():
    assert watchdog.timeout_s({}) == watchdog.DEFAULT_TIMEOUT_S
    assert watchdog.timeout_s({watchdog.TIMEOUT_ENV: "2.5"}) == 2.5


def test_rank_map_line_lists_every_rank():
    line = watchdog.rank_map_line(2, "h0:61500", [{"host": "h0", "gpu": "g=0"},
                                                  {"host": "h1", "gpu": "g=1"}])
    assert line == "hsuite: P=2, jax.distributed coordinator h0:61500; ranks 0: h0 g=0; 1: h1 g=1"


def test_terminal_writer_outside_pytest_is_none():
    assert watchdog.terminal_writer(None) is None
