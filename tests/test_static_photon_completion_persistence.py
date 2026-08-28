"""Focused FULL/OFF persistence contract for packed static-photon GW."""

import pytest

from file_io import static_gauge_head
from gw.gw_config import HeadCorrection
from gw.gw_jax import _persist_static_photon_head_completion


def test_full_requires_and_writes_exactly_one_completion_receipt(
        tmp_path, monkeypatch):
    completion = object()
    calls = []

    def fake_writer(path, value, *, mesh):
        calls.append((path, value, mesh))
        return {"schema_version": 1}

    monkeypatch.setattr(
        static_gauge_head, "write_static_photon_head_completion_receipt_h5",
        fake_writer)
    metadata, path = _persist_static_photon_head_completion(
        tmp_path, completion, head_correction=HeadCorrection.FULL,
        mesh="mesh")
    assert metadata == {"schema_version": 1}
    assert path == str(tmp_path / "static_slab_photon_head_completion.h5")
    assert calls == [(path, completion, "mesh")]

    with pytest.raises(RuntimeError, match="returned no q=0 completion"):
        _persist_static_photon_head_completion(
            tmp_path, None, head_correction=HeadCorrection.FULL, mesh="mesh")


def test_off_requires_no_completion_and_never_calls_the_writer(
        tmp_path, monkeypatch):
    def forbidden_writer(*args, **kwargs):
        raise AssertionError("head_correction=off called the receipt writer")

    monkeypatch.setattr(
        static_gauge_head, "write_static_photon_head_completion_receipt_h5",
        forbidden_writer)
    assert _persist_static_photon_head_completion(
        tmp_path, None, head_correction=HeadCorrection.OFF,
        mesh="mesh") == (None, None)

    with pytest.raises(RuntimeError, match="unexpected q=0 completion"):
        _persist_static_photon_head_completion(
            tmp_path, object(), head_correction=HeadCorrection.OFF,
            mesh="mesh")
