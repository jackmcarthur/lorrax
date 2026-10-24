"""The clean-shutdown receipt filter on the supported JAX 0.9.1/0.12 formats.

The receipts are copied from a Perlmutter P = 4 hsuite run on JAX 0.12.0.1
(2026-10-23): every rank's PJRT client prints one ``CANCELLED`` block when rank
zero closes the coordination service, then an ``UNAVAILABLE ... Connection
refused`` block on every retry until ``os._exit``.
The legacy three-line format is captured from Frontier JAX 0.9.1 (2026-10-24).
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

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
    + b":UNKNOWN:Error received from peer  {grpc_message:\"CANCELLED\", grpc_status:1}"
    + b" [type.googleapis.com/tensorflow.DerivedStatus='']\n")
LEGACY_REFUSED = (
    b"W1024 06:24:23.656981  678478 pjrt_client.cc:1604] WatchJobStateAsync failed for "
    b"task 1: UNAVAILABLE: " + _REFUSED + b"\n" + _LEGACY_CONTEXT
    + b':UNKNOWN:Error received from peer  {grpc_status:14, grpc_message:"' + _REFUSED + b'"}\n')
RECEIPTS = (CANCELLED, REFUSED, LEGACY_CANCELLED, LEGACY_REFUSED,
            LEGACY_CANCELLED.replace(b" [type.googleapis.com/tensorflow.DerivedStatus='']", b""))


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


def test_finalize_keeps_filter_through_exit_hooks_and_requires_a_passed_barrier():
    # Real finalizer/atexit/fd filter; only the GPU and collective are stand-ins.
    # The early hook would run AFTER the filter's later-registered stop hook.
    code = textwrap.dedent("""
        import atexit, os, sys, types
        import runtime
        from runtime.pjrt_log_filter import install_pjrt_log_filter
        receipt = bytes.fromhex(sys.argv[1])
        passed = sys.argv[2] == '1'
        def late_hook():
            os.write(2, receipt + b'UNHEALTHY_PEER_ERROR\\n')
            sys.stderr.write('buffered Python diagnostic')
        atexit.register(late_hook)
        install_pjrt_log_filter()
        jax = types.ModuleType('jax')
        jax.effects_barrier = lambda: None
        jax.process_index = lambda: 0
        jax.distributed = types.SimpleNamespace(shutdown=lambda: None)
        api = types.ModuleType('jax._src.api')
        api.clean_up = lambda: None
        src = types.ModuleType('jax._src')
        src.api = api
        collective = types.ModuleType('common.collectives')
        collective.barrier = lambda name: passed
        sys.modules.update({'jax':jax, 'jax._src':src, 'jax._src.api':api,
                            'common.collectives':collective})
        hard_exit = os._exit
        def check_exit(rc):
            try:
                os.fstat(2)
            except OSError:
                os.write(1, b'STDERR_CLOSED_AT_HARD_EXIT\\n')
            hard_exit(rc)
        os._exit = check_exit
        runtime.finalize_process(7)
    """)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(root / 'src'))
    for receipt in (REFUSED, LEGACY_CANCELLED):
        for passed in (False, True):
            result = subprocess.run([sys.executable, '-c', code, receipt.hex(), str(int(passed))],
                env=env, capture_output=True, timeout=10)
            assert result.returncode == 7, result.stderr
            assert b'UNHEALTHY_PEER_ERROR\n' in result.stderr
            assert b'buffered Python diagnostic' in result.stderr
            assert b'STDERR_CLOSED_AT_HARD_EXIT\n' in result.stdout
            assert (receipt in result.stderr) is not passed, result.stderr


def test_hard_stop_drains_owned_pipe_and_preserves_a_foreign_descriptor():
    with tempfile.TemporaryFile() as sink, tempfile.TemporaryFile() as foreign:
        target = os.dup(sink.fileno())
        f = ExactPjrtNoticeFilter(target_fd=target)
        f.start()
        os.write(target, b'pending diagnostic\n')
        f.stop(restore=False)
        try:
            os.fstat(target)
        except OSError:
            pass
        else:
            raise AssertionError('hard stop restored its owned descriptor')
        sink.seek(0)
        assert sink.read() == b'pending diagnostic\n'

        target = os.dup(sink.fileno())
        f = ExactPjrtNoticeFilter(target_fd=target)
        f.start()
        os.dup2(foreign.fileno(), target)  # A harness has replaced the owned pipe.
        f.stop(restore=False)
        os.write(target, b'foreign descriptor survives\n')
        os.close(target)
        foreign.seek(0)
        assert foreign.read() == b'foreign descriptor survives\n'
