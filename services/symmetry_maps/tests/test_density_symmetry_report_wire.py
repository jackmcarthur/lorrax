"""Focused gates for the small report codec and byte transport.

These are intentionally transport-only.  The real four-process gate lives in
``test_symmetry_maps_multiproc.py``; a fake client here makes every failure
arm deterministic without requiring a distributed runtime.
"""

from __future__ import annotations

from dataclasses import fields
import json
import math

import numpy as np
import pytest

from symmetry_maps._collectives import broadcast_root_bytes
from symmetry_maps.density_symmetry_check import (
    DensitySymmetryReport,
    _decode_density_symmetry_report,
    _decode_distributed_outcome,
    _encode_density_symmetry_report,
    _encode_distributed_outcome,
)


def _report() -> DensitySymmetryReport:
    return DensitySymmetryReport(
        trs_holds=True,
        trs_basis="raw-subspace",
        m_rel=2.5e-9,
        m_rel_total=None,
        trs_coverage=0.75,
        tol_trs=1.0e-6,
        trs_implied_by_mesh=True,
        spatial_ops_ok=True,
        spatial_residual=np.asarray([0.0, 1.5e-12], dtype=np.float64),
        spatial_untested=(3,),
        spatial_failed=(),
        tol_spatial=float("nan"),
        charge=12.0,
        charge_expected=12.0,
        charge_rel_err=0.0,
        rho_min_rel=-float("inf"),
        invariants_ok=True,
        manifold_gap=float("inf"),
        path="/shared/WFN.h5",
        nocc=6,
        nspin=1,
        nspinor=2,
        n_k_used=3,
        n_k_total=4,
        subsampled=True,
        fft_grid=(12, 10, 8),
        seconds=4.25,
        seconds_io=3.0,
        seconds_quad=1.0,
        messages=("one", "two"),
        method="occupied-density-subspace",
        subspace_residual=2.5e-9,
        min_overlap_singular_value=0.999999,
        evidence_counts=(("raw-pair", 1), ("spatial-pair", 1), ("trim", 0)),
        conclusive=True,
    )


def _assert_report_equal(left, right):
    assert type(left) is type(right) is DensitySymmetryReport
    for item in fields(DensitySymmetryReport):
        a, b = getattr(left, item.name), getattr(right, item.name)
        if isinstance(a, np.ndarray):
            assert isinstance(b, np.ndarray)
            assert a.dtype == b.dtype and a.shape == b.shape
            assert np.array_equal(a, b, equal_nan=True)
        elif isinstance(a, float) and math.isnan(a):
            assert isinstance(b, float) and math.isnan(b)
        else:
            assert a == b, item.name


def test_report_wire_round_trip_preserves_every_field_and_nonfinite_float():
    original = _report()
    payload = _encode_density_symmetry_report(original)
    assert b"NaN" not in payload and b"Infinity" not in payload
    _assert_report_equal(original, _decode_density_symmetry_report(payload))


def test_report_wire_refuses_schema_drift_and_bad_array_size():
    raw = json.loads(_encode_density_symmetry_report(_report()))
    raw["fields"]["new_unread_field"] = 1
    with pytest.raises(ValueError, match="wire fields differ"):
        _decode_density_symmetry_report(json.dumps(raw).encode())

    raw = json.loads(_encode_density_symmetry_report(_report()))
    raw["fields"]["spatial_residual"]["shape"] = [99]
    with pytest.raises(ValueError, match="wire size mismatch"):
        _decode_density_symmetry_report(json.dumps(raw).encode())


class _FakeClient:
    def __init__(self):
        self.values = {}
        self.gets = []
        self.barriers = []

    def key_value_set_bytes(self, key, value):
        self.values[key] = bytes(value)

    def blocking_key_value_get_bytes(self, key, timeout_ms):
        self.gets.append((key, timeout_ms))
        return self.values[key]

    def wait_at_barrier(self, key, *, timeout_in_ms):
        self.barriers.append((key, timeout_in_ms))


def test_fake_transport_publishes_once_and_returns_exact_bytes_to_every_rank():
    client = _FakeClient()
    payload = bytes(range(256))
    got0 = broadcast_root_bytes(
        payload, key="receipt/7", client=client,
        process_index=0, process_count=4, timeout_ms=17)
    peers = [broadcast_root_bytes(
        None, key="receipt/7", client=client,
        process_index=rank, process_count=4, timeout_ms=17)
        for rank in (1, 2, 3)]
    assert got0 == payload
    assert peers == [payload, payload, payload]
    assert client.gets == [("receipt/7", 17)] * 3
    assert client.barriers == [("receipt/7/commit", 17)] * 4


def test_fake_transport_refuses_wrong_rank_roles_and_missing_byte_api():
    client = _FakeClient()
    with pytest.raises(TypeError, match="rank 0 must supply"):
        broadcast_root_bytes(
            None, key="k", client=client, process_index=0, process_count=2)
    with pytest.raises(TypeError, match="non-root processes"):
        broadcast_root_bytes(
            b"wrong", key="k", client=client,
            process_index=1, process_count=2)
    with pytest.raises(RuntimeError, match="byte-exact KV API"):
        broadcast_root_bytes(
            b"x", key="k", client=object(),
            process_index=0, process_count=2)


def test_outcome_codec_carries_report_none_and_root_error():
    request = {"path": "/shared/WFN.h5", "mode": "strict"}
    req, kind, value = _decode_distributed_outcome(
        _encode_distributed_outcome(request, report=_report()))
    assert req == request and kind == "report"
    _assert_report_equal(_report(), value)

    assert _decode_distributed_outcome(
        _encode_distributed_outcome(request)) == (request, "none", None)
    req, kind, value = _decode_distributed_outcome(
        _encode_distributed_outcome(
            request, error=RuntimeError("strict root refusal")))
    assert req == request and kind == "error"
    assert value == ("RuntimeError", "strict root refusal")
