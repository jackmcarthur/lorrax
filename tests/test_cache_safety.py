"""On-disk cache publication: whole-or-nothing, host-unique temporaries,
validated reads (``lxkit.atomic_write_bytes`` and the 2c-TRS stamps)."""
from __future__ import annotations

import dataclasses
import json
import multiprocessing as mp
import os

import numpy as np
import pytest

from lxkit import atomic_write_bytes
from symmetry_maps import density_symmetry_check as dsc


def test_atomic_write_publishes_whole_file_and_leaves_no_temporary(tmp_path):
    target = tmp_path / "sub" / "entry.bin"
    atomic_write_bytes(target, b"first")
    atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"second"
    assert sorted(p.name for p in target.parent.iterdir()) == ["entry.bin"]


def test_atomic_write_failure_keeps_previous_target(tmp_path, monkeypatch):
    target = tmp_path / "entry.bin"
    atomic_write_bytes(target, b"good")

    def boom(src, dst):
        raise OSError("simulated rename failure")
    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        atomic_write_bytes(target, b"bad")
    assert target.read_bytes() == b"good"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["entry.bin"]


def _writer(path, tag, n):
    payload = bytes([tag]) * (1 << 18)
    for _ in range(n):
        atomic_write_bytes(path, payload)


def test_concurrent_writers_never_expose_a_partial_file(tmp_path):
    target = tmp_path / "shared.bin"
    atomic_write_bytes(target, bytes([0]) * (1 << 18))
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_writer, args=(str(target), t, 40)) for t in (1, 2, 3, 4)]
    for p in procs:
        p.start()
    seen = set()
    while any(p.is_alive() for p in procs):
        blob = target.read_bytes()
        assert len(blob) == 1 << 18 and blob.count(blob[:1]) == len(blob), "torn read"
        seen.add(blob[0])
    for p in procs:
        p.join()
        assert p.exitcode == 0
    assert not [p for p in tmp_path.iterdir() if p.name != "shared.bin"]


def _measured_report(path="/x/WFN.h5"):
    base = dsc._empty_report(None, basis="qe-magnetic", message="m", tol_trs=1e-6,
                             nspin=1, nspinor=2, conclusive=True, trs_holds=False)
    return dataclasses.replace(base, path=path, trs_coverage=1.0)


def _identity(**over):
    ident = {"qe_trs_off": "magnetic", "schema": dsc._STAMP_SCHEMA,
             "algorithm": dsc._ALGORITHM_VERSION, "code": dsc._code_key(),
             "path": "/x/WFN.h5", "size": 1, "mtime_ns": 2, "inode": 3,
             "content_sha256": "c", "tol_trs": "1e-06", "max_k": 12, "nocc": 3}
    ident.update(over)
    return ident


@pytest.fixture
def stamp_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    return tmp_path


def test_trs_stamp_round_trip_hits(stamp_root):
    ident = _identity()
    assert dsc._write_stamp(ident, _measured_report()) is not None
    hit = dsc._read_stamp(ident)
    assert hit is not None and hit[0].trs_holds is False and hit[0].nspinor == 2


def test_trs_stamp_truncated_or_tampered_is_a_miss(stamp_root):
    ident = _identity()
    path = dsc._write_stamp(ident, _measured_report())
    record = json.loads(open(path).read())
    record["report"]["trs_holds"] = True            # payload no longer matches its digest
    open(path, "w").write(json.dumps(record))
    assert dsc._read_stamp(ident) is None
    dsc._write_stamp(ident, _measured_report())
    blob = open(path).read()
    open(path, "w").write(blob[: len(blob) // 2])   # a partial file
    assert dsc._read_stamp(ident) is None


def test_trs_stamp_out_of_range_payload_is_a_miss(stamp_root):
    ident = _identity()
    bad = dataclasses.replace(_measured_report(), trs_coverage=float("nan"))
    dsc._write_stamp(ident, bad)
    assert dsc._read_stamp(ident) is None


def test_trs_stamp_from_other_code_is_a_miss(stamp_root):
    dsc._write_stamp(_identity(code="0000000000000000"), _measured_report())
    assert dsc._read_stamp(_identity()) is None
    assert "code" in _identity() and dsc._code_key() != "0000000000000000"
