"""``open_h5_settled``: only a transiently short size view is waited out."""
import h5py
import numpy as np
import pytest

from file_io import commit_state
from file_io.commit_state import open_h5_settled

_TRUNC = ("Unable to synchronously open file (truncated file: eof = 3269459968, "
          "sblock->base_addr = 0, stored_eof = 3280207872)")


def _flaky(monkeypatch, failures, message=_TRUNC):
    real, calls = h5py.File, []

    def fake(path, mode="r", **kwargs):
        calls.append(mode)
        if len(calls) <= failures:
            raise OSError(message)
        return real(path, mode, **kwargs)

    monkeypatch.setattr(h5py, "File", fake)
    return calls


def test_short_view_is_retried_until_the_file_opens(tmp_path, monkeypatch, capsys):
    path = tmp_path / "bank.h5"
    with h5py.File(path, "w") as f:
        f["x"] = np.arange(3)
    calls = _flaky(monkeypatch, failures=3)
    with open_h5_settled(path, "r") as f:
        assert f["x"][()].tolist() == [0, 1, 2]
    assert calls == ["r"] * 4
    assert "settled after 3 refused open(s)" in capsys.readouterr().err


def test_other_open_errors_are_not_retried(tmp_path, monkeypatch):
    calls = _flaky(monkeypatch, failures=5, message="Unable to open file (file signature not found)")
    with pytest.raises(OSError, match="signature"):
        open_h5_settled(tmp_path / "bank.h5", "a")
    assert calls == ["a"]


def test_a_file_that_stays_short_raises_the_original_refusal(tmp_path, monkeypatch):
    monkeypatch.setattr(commit_state, "SETTLE_TIMEOUT_S", 0.05)
    _flaky(monkeypatch, failures=10**6)
    with pytest.raises(OSError, match="truncated file"):
        open_h5_settled(tmp_path / "bank.h5", "r")
