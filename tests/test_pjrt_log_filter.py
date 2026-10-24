"""The clean-shutdown receipt filter on the supported JAX 0.9.1/0.12 formats.

The receipts are copied from a Perlmutter P = 4 hsuite run on JAX 0.12.0.1
(2026-10-23): every rank's PJRT client prints one ``CANCELLED`` block when rank
zero closes the coordination service, then an ``UNAVAILABLE ... Connection
refused`` block on every retry until ``os._exit``.
The legacy three-line format is captured from Frontier JAX 0.9.1 (2026-10-24).
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
_LEGACY_CONTEXT = (b"Additional GRPC error information from remote target coordination_service "
                   b"while calling /tensorflow.CoordinationService/WatchJobState:\n")
LEGACY_CANCELLED = (
    b"W1024 06:24:22.656981  678478 pjrt_client.cc:1604] WatchJobStateAsync failed for "
    b"task 1: CANCELLED: CANCELLED\n" + _LEGACY_CONTEXT
    + b':UNKNOWN:Error received from peer  {grpc_message:"CANCELLED", grpc_status:1}\n')
LEGACY_REFUSED = (
    b"W1024 06:24:23.656981  678478 pjrt_client.cc:1604] WatchJobStateAsync failed for "
    b"task 1: UNAVAILABLE: " + _REFUSED + b"\n" + _LEGACY_CONTEXT
    + b':UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"' + _REFUSED + b'"}\n')
RECEIPTS = (CANCELLED, REFUSED, LEGACY_CANCELLED, LEGACY_REFUSED)


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


def test_every_line_of_supported_receipts_is_recognized():
    for block in RECEIPTS:
        state = False
        for line in block.splitlines(keepends=True):
            suppress, state = clean_shutdown_step(line, state)
            assert suppress, line
        assert not state


def test_receipts_after_the_gate_are_dropped_and_nothing_else():
    tail = b"rank-0 report line\n\nnext line\n"
    receipts = b"".join(RECEIPTS) * 3
    out, dropped = _run(b"", receipts + tail)
    assert out == tail
    assert dropped == len(receipts.splitlines())


def test_receipts_before_the_gate_are_forwarded():
    receipts = b"".join(RECEIPTS)
    out, dropped = _run(receipts, b"")
    assert out == receipts and dropped == 0


def test_trace_and_blank_lines_outside_a_receipt_are_forwarded():
    stray = _TRACE + b"\n" + _CONTEXT + _LEGACY_CONTEXT
    out, _ = _run(b"", stray)
    assert out == stray


def test_real_errors_and_orphan_details_remain_visible_after_the_gate():
    # A genuine unhealthy-peer error seen on Frontier must survive, even
    # during finalization; a cancellation detail needs a recognized header.
    reset = b"recvmsg:Connection reset by peer"
    real_errors = REFUSED.replace(_REFUSED, reset) + LEGACY_REFUSED.replace(_REFUSED, reset)
    orphan_details = b"".join(block.splitlines(keepends=True)[2] for block in RECEIPTS)
    out, dropped = _run(b"", real_errors + orphan_details)
    assert out == real_errors + orphan_details and dropped == 0
    # A legacy receipt ends on its detail, so an unrelated trace is preserved.
    out, _ = _run(b"", LEGACY_CANCELLED + _TRACE)
    assert out == _TRACE
