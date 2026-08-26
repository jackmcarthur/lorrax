"""The PJRT noise filter drops one exact C++ line and nothing else."""
from __future__ import annotations

import os

from runtime.pjrt_log_filter import (ExactPjrtNoticeFilter,
                                     is_clean_shutdown_notice,
                                     is_spurious_pjrt_version_notice)


NOTICE = (
    b"W0818 15:31:29.533178  537785 pjrt_executable.cc:638] "
    b"Assume version compatibility. PjRt-IFRT does not track XLA "
    b"executable versions.\n"
)

CLEAN_SHUTDOWN_NOTICE = (
    b"W0826 01:39:27.598081 2064395 pjrt_client.cc:1604] "
    b"WatchJobStateAsync failed for task 0: CANCELLED: CANCELLED\n"
    b"Additional GRPC error information from remote target "
    b"coordination_service while calling "
    b"/tensorflow.CoordinationService/WatchJobState:\n"
    b":UNKNOWN:Error received from peer  {grpc_status:1, "
    b"grpc_message:\"CANCELLED\"} "
    b"[type.googleapis.com/tensorflow.DerivedStatus='']\n"
)

CLEAN_SHUTDOWN_UNAVAILABLE = (
    b"W0826 01:43:18.920461 2065781 pjrt_client.cc:1604] "
    b"WatchJobStateAsync failed for task 0: UNAVAILABLE: failed to connect "
    b"to all addresses; last error: UNKNOWN: ipv4:128.55.69.199:22013: "
    b"Failed to connect to remote host: Connection refused\n"
    b"Additional GRPC error information from remote target "
    b"coordination_service while calling "
    b"/tensorflow.CoordinationService/WatchJobState:\n"
    b":UNKNOWN:Error received from peer  {grpc_message:\"failed to connect "
    b"to all addresses; last error: UNKNOWN: ipv4:128.55.69.199:22013: "
    b"Failed to connect to remote host: Connection refused\", "
    b"grpc_status:14}\n"
)

CLEAN_SHUTDOWN_CANCELLING = (
    b"W0826 01:49:14.197663 2069810 pjrt_client.cc:1604] "
    b"WatchJobStateAsync failed for task 0: UNAVAILABLE: Cancelling all calls\n"
    b"Additional GRPC error information from remote target "
    b"coordination_service while calling "
    b"/tensorflow.CoordinationService/WatchJobState:\n"
    b":UNKNOWN:Error received from peer  {grpc_status:14, "
    b"grpc_message:\"Cancelling all calls\"}\n"
)


def _roundtrip(writes: list[bytes]) -> tuple[bytes, int]:
    sink_read, sink_write = os.pipe()
    target_fd = os.dup(sink_write)
    os.close(sink_write)
    filt = ExactPjrtNoticeFilter(target_fd=target_fd)
    try:
        filt.start()
        for payload in writes:
            os.write(target_fd, payload)
        count = filt.stop()
        os.close(target_fd)
        output = bytearray()
        while True:
            chunk = os.read(sink_read, 4096)
            if not chunk:
                break
            output.extend(chunk)
        return bytes(output), count
    finally:
        try:
            filt.stop()
        finally:
            for fd in (target_fd, sink_read):
                try:
                    os.close(fd)
                except OSError:
                    pass


def test_matcher_requires_the_complete_absl_pjrt_notice():
    assert is_spurious_pjrt_version_notice(NOTICE)
    assert is_spurious_pjrt_version_notice(NOTICE.rstrip(b"\n"))
    assert not is_spurious_pjrt_version_notice(
        NOTICE.replace(b"Assume version compatibility", b"Version mismatch"))
    assert not is_spurious_pjrt_version_notice(
        b"Assume version compatibility. PjRt-IFRT does not track XLA "
        b"executable versions.\n")


def test_filter_drops_only_the_exact_notice_and_counts_it():
    other = b"W0818 15:31:30.000000  537785 pjrt_client.cc:10] useful warning\n"
    near = NOTICE.replace(b"versions.", b"versions!")
    output, count = _roundtrip([other, NOTICE, near, NOTICE])
    assert output == other + near
    assert count == 2


def test_filter_reassembles_fragmented_lines_and_flushes_partial_tail():
    prefix = b"ordinary stderr before\n"
    partial_tail = b"ordinary stderr without newline"
    output, count = _roundtrip(
        [prefix + NOTICE[:17], NOTICE[17:81], NOTICE[81:], partial_tail]
    )
    assert output == prefix + partial_tail
    assert count == 1


def test_clean_shutdown_receipt_is_gated_and_exact():
    notices = (CLEAN_SHUTDOWN_NOTICE + CLEAN_SHUTDOWN_UNAVAILABLE
               + CLEAN_SHUTDOWN_CANCELLING)
    lines = notices.splitlines(keepends=True)
    assert all(is_clean_shutdown_notice(line) for line in lines)
    assert not is_clean_shutdown_notice(
        lines[0].replace(b"CANCELLED: CANCELLED", b"INTERNAL: data loss"))

    output, count = _roundtrip([CLEAN_SHUTDOWN_NOTICE])
    assert output == CLEAN_SHUTDOWN_NOTICE
    assert count == 0

    sink_read, sink_write = os.pipe()
    target_fd = os.dup(sink_write)
    os.close(sink_write)
    filt = ExactPjrtNoticeFilter(target_fd=target_fd)
    try:
        filt.start()
        filt.begin_clean_shutdown()
        os.write(target_fd, notices)
        assert filt.stop() == 9
        os.close(target_fd)
        assert os.read(sink_read, 4096) == b""
    finally:
        for fd in (target_fd, sink_read):
            try:
                os.close(fd)
            except OSError:
                pass


def test_start_and_stop_are_idempotent():
    sink_read, sink_write = os.pipe()
    target_fd = os.dup(sink_write)
    os.close(sink_write)
    filt = ExactPjrtNoticeFilter(target_fd=target_fd)
    try:
        filt.start()
        filt.start()
        os.write(target_fd, NOTICE)
        assert filt.stop() == 1
        assert filt.stop() == 1
    finally:
        for fd in (target_fd, sink_read):
            try:
                os.close(fd)
            except OSError:
                pass
