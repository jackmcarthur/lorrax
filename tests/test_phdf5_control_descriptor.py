"""PHDF5 slab-control width and CUDA-stream lifetime contract.

The intermittent ``valid_shape`` failures that motivated this test exposed
the int32 words ``(0, 1, 2, 5, 6, 7)`` as three int64 values.  Those words
are the beginning of the Si ``kirr_to_kfull`` table: the native handler read
stale allocator contents before XLA had produced its signed-64-bit control
operand.  The Python and C++ widths were already both int64; the defect was
the legacy-default-stream copy in ``copy_index_to_host``.

These are source-level ratchets because exercising the race requires a CUDA
FFI rebuild and a favourable allocator schedule.  The negative control is the
pre-fix helper body, so the structural check is known to discriminate.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


_REPO = Path(__file__).resolve().parents[1]
_SEAM = _REPO / "src/ffi/cpp/phdf5/platform_seam.h"
_READ = _REPO / "src/ffi/cpp/phdf5/read_ffi.cc"
_WRITE = _REPO / "src/ffi/cpp/phdf5/write_ffi.cc"
_SLAB = _REPO / "src/file_io/_slab_io_ffi.py"


def _control_copy_violations(source: str) -> list[str]:
    """Return defects in the CUDA control-buffer copy implementation."""
    bad = []
    copy = re.search(
        r"cudaMemcpyAsync\s*\(\s*dst\s*,\s*src\s*,\s*nbytes\s*,\s*"
        r"cudaMemcpyDeviceToHost\s*,\s*xla_stream\s*\)",
        source,
    )
    sync = re.search(r"cudaStreamSynchronize\s*\(\s*xla_stream\s*\)", source)
    if copy is None:
        bad.append("control copy is not enqueued on xla_stream")
    if sync is None:
        bad.append("host may read the control buffer before the copy lands")
    if copy is not None and sync is not None and sync.start() < copy.end():
        bad.append("xla_stream is synchronized before the control copy")
    if re.search(
        r"cudaMemcpy\s*\(\s*dst\s*,\s*src\s*,\s*nbytes\s*,\s*"
        r"cudaMemcpyDeviceToHost\s*\)",
        source,
    ):
        bad.append("legacy-default-stream cudaMemcpy is still reachable")
    return bad


def test_control_copy_is_ordered_after_its_xla_producer():
    assert _control_copy_violations(_SEAM.read_text()) == []


def test_the_pre_fix_default_stream_copy_fails_the_auditor():
    legacy = """
        static inline bool copy_index_to_host(
            void* dst, const void* src, size_t nbytes, std::string* err) {
          cudaError_t ce = cudaMemcpy(
              dst, src, nbytes, cudaMemcpyDeviceToHost);
          return ce == cudaSuccess;
        }
    """
    assert _control_copy_violations(legacy) == [
        "control copy is not enqueued on xla_stream",
        "host may read the control buffer before the copy lands",
        "legacy-default-stream cudaMemcpy is still reachable",
    ]


def test_every_read_and_write_control_copy_receives_the_xla_stream():
    """No new copy site may silently fall back to the default stream."""
    for path, expected in ((_READ, 8), (_WRITE, 3)):
        source = path.read_text()
        calls = re.findall(r"\bcopy_index_to_host\s*\(", source)
        streamed = re.findall(
            r"\bcopy_index_to_host\s*\(\s*LRX_STREAM_ARG", source)
        assert len(calls) == expected, (path, len(calls), expected)
        assert len(streamed) == len(calls), (path, len(streamed), len(calls))


def test_slab_control_vectors_are_packed_as_owned_int64_arrays():
    """The Python/JAX half constructs a fresh signed-64-bit buffer per op."""
    source = _SLAB.read_text()
    body = source.split("def _replicated_i64_vector", 1)[1].split(
        "def _normalize_slab_request", 1)[0]
    assert "np.asarray(tuple(int(v) for v in values), dtype=np.int64)" in body

    words = np.asarray((0, 1, 2, 5, 6, 7), dtype="<i4")
    misread = np.frombuffer(words.tobytes(), dtype="<i8")
    assert misread.tolist() == [2**32, 5 * 2**32 + 2, 7 * 2**32 + 6]

