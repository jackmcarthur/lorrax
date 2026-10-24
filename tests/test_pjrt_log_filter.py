"""The clean-shutdown receipt filter (runtime.pjrt_log_filter) on JAX 0.12's receipts.

The receipts are copied from a Perlmutter P = 4 hsuite run on JAX 0.12.0.1
(2026-10-23): every rank's PJRT client prints one ``CANCELLED`` block when rank
zero closes the coordination service, then an ``UNAVAILABLE ... Connection
refused`` block on every retry until ``os._exit``.
"""
import os
import tempfile

from runtime.pjrt_log_filter import ExactPjrtNoticeFilter, clean_shutdown_step

_TRACE = (b"=== Source Location Trace: === \n"
          b"external/xla+/xla/tsl/distributed_runtime/rpc/grpc_util.h:89\n"
          b"\n")
_CONTEXT = (b"Additional GRPC error information from remote target coordination_service "
            b"while calling /xla.coordination.CoordinationService/WatchTasks:\n")
_REFUSED = (b"failed to connect to all addresses; last error: UNKNOWN: "
            b"ipv4:128.55.69.36:63993: Failed to connect to remote host: Connection refused")
CANCELLED = (b"W1023 20:04:22.947271  987344 pjrt_client.cc:1620] WatchTasksAsync failed for "
             b"task 0: CANCELLED: CANCELLED\n" + _CONTEXT
             + b":CANCELLED:CANCELLED [type.googleapis.com/tensorflow.DerivedStatus='']\n" + _TRACE)
REFUSED = (b"W1023 20:04:23.948099  987343 pjrt_client.cc:1620] WatchTasksAsync failed for "
           b"task 1: UNAVAILABLE: " + _REFUSED + b"\n" + _CONTEXT
           + b":UNAVAILABLE:" + _REFUSED + b"\n" + _TRACE)


def _run(before, after):
    """Feed ``before``, open the shutdown gate, feed ``after``; return what got through."""
    with tempfile.TemporaryFile() as sink:
        f = ExactPjrtNoticeFilter(target_fd=sink.fileno())
        f.start()
        os.write(sink.fileno(), before)
        f.stop()                       # drain ``before`` through the closed gate
        f.start()
        f.begin_clean_shutdown()
        os.write(sink.fileno(), after)
        f.stop()
        sink.seek(0)
        return sink.read(), f.suppressed


def test_every_line_of_both_receipts_is_recognized():
    for block in (CANCELLED, REFUSED):
        state = False
        for line in block.splitlines(keepends=True):
            suppress, state = clean_shutdown_step(line, state)
            assert suppress, line
        assert not state


def test_receipts_after_the_gate_are_dropped_and_nothing_else():
    tail = b"rank-0 report line\n\nnext line\n"
    out, dropped = _run(b"", CANCELLED + REFUSED * 3 + tail)
    assert out == tail
    assert dropped == len((CANCELLED + REFUSED * 3).splitlines())


def test_receipts_before_the_gate_are_forwarded():
    out, dropped = _run(CANCELLED, b"")
    assert out == CANCELLED and dropped == 0


def test_trace_and_blank_lines_outside_a_receipt_are_forwarded():
    stray = _TRACE + b"\n" + _CONTEXT
    out, _ = _run(b"", stray)
    assert out == stray
