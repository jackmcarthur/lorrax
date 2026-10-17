"""The ordered end's stderr hygiene when a harness moves fd 2 under the PJRT
notice filter (the hsuite chain does, per stage).  Plain Python: no jax, no GPU,
no MPI.

``runtime.pjrt_log_filter.ExactPjrtNoticeFilter`` takes an injectable target
descriptor, so these tests never touch pytest's own fd 2.
"""
import os

from runtime.pjrt_log_filter import ExactPjrtNoticeFilter

RECEIPT = (b"W1016 21:03:55.186522 2899984 pjrt_client.cc:1604] WatchJobStateAsync failed "
           b"for task 1: CANCELLED: CANCELLED\n"
           b"Additional GRPC error information from remote target coordination_service "
           b"while calling /tensorflow.CoordinationService/WatchJobState:\n")
PLAIN = b"an ordinary stderr line\n"


def _file_fd(tmp_path, name):
    return os.open(tmp_path / name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)


def _same(fd_a, fd_b):
    a, b = os.fstat(fd_a), os.fstat(fd_b)
    return (a.st_dev, a.st_ino) == (b.st_dev, b.st_ino)


def test_clean_shutdown_reattaches_after_a_harness_moved_the_target(tmp_path):
    """The hsuite shape: fd 2 -> stage log while the filter starts, then back."""
    console = _file_fd(tmp_path, "console")
    stage = _file_fd(tmp_path, "stage")
    target = os.dup(console)                  # stands for fd 2
    saved = os.dup(target)                    # the harness saves fd 2 ...
    os.dup2(stage, target)                    # ... points it at the stage log
    flt = ExactPjrtNoticeFilter(target_fd=target)
    flt.start()                               # first driver start-up: filter on the stage log
    os.write(target, PLAIN)
    os.dup2(saved, target)                    # stage ends: the harness restores fd 2
    flt.begin_clean_shutdown()                # finalize_process, after the barrier
    assert flt._attached()
    os.write(target, RECEIPT + PLAIN)         # jax's coordinator-close receipts
    flt.stop()
    assert _same(target, console)
    assert (tmp_path / "console").read_bytes() == PLAIN
    assert (tmp_path / "stage").read_bytes() == PLAIN
    assert flt.suppressed == 2
    for fd in (console, stage, target, saved):
        os.close(fd)


def test_stop_never_overwrites_a_descriptor_the_harness_installed(tmp_path):
    first = _file_fd(tmp_path, "first")
    other = _file_fd(tmp_path, "other")
    target = os.dup(first)
    flt = ExactPjrtNoticeFilter(target_fd=target)
    flt.start()
    os.dup2(other, target)                    # moved under the filter
    flt.stop()
    assert _same(target, other)               # not the stale saved "first"
    for fd in (first, other, target):
        os.close(fd)


def test_an_attached_filter_is_unchanged_by_start_and_clean_shutdown(tmp_path):
    console = _file_fd(tmp_path, "console")
    target = os.dup(console)
    flt = ExactPjrtNoticeFilter(target_fd=target)
    flt.start()
    pipe = flt._pipe_id
    flt.start()
    flt.begin_clean_shutdown()
    assert flt._pipe_id == pipe
    os.write(target, RECEIPT + PLAIN)
    flt.stop()
    assert _same(target, console)
    assert (tmp_path / "console").read_bytes() == PLAIN
    for fd in (console, target):
        os.close(fd)


def test_receipts_during_the_run_are_still_forwarded(tmp_path):
    console = _file_fd(tmp_path, "console")
    target = os.dup(console)
    flt = ExactPjrtNoticeFilter(target_fd=target)
    flt.start()
    os.write(target, RECEIPT)                 # no clean shutdown yet: a real failure
    flt.stop()
    assert (tmp_path / "console").read_bytes() == RECEIPT
    for fd in (console, target):
        os.close(fd)
