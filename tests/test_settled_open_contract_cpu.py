"""The contract of ``file_io.commit_state.open_h5_settled`` beyond the happy path.

The settled open retries ``h5py.File`` while HDF5 refuses it as "truncated
file" (its own check: client size + user block below the superblock's stored
end of file).  These cases pin what it must and must not do:

* guard and resume readers (``settle_s=GUARD_SETTLE_S``) and the ζ probe
  give up on a file that really is truncated within seconds, not a minute;
* the ζ probe keeps its promise of no jax and no LORRAX (``src/`` off the
  path) and still never raises;
* collective (``driver="mpio"``) and ``swmr=True`` opens are refused by name;
* modes ``a`` and ``r+`` wait out a short view and then write normally;
* an empty, non-HDF5, missing file or a directory is refused at once with the
  original error (no wait, no settled-open message).
"""
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_ZETA_SRC = _REPO / "services/zeta_loader/src"


def _fixture(path):
    """A small contiguous HDF5 file and its bytes."""
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.arange(1 << 17, dtype=np.float64))
        f.attrs["tag"] = "fixture"
    return path.read_bytes()


def _cut(path, data):
    cut = (len(data) // 2) // 4096 * 4096
    os.truncate(path, cut)
    return cut


def _restore_later(path, data, cut, delay=3.0):
    def put_back():
        time.sleep(delay)
        with open(path, "r+b") as f:
            f.seek(cut)
            f.write(data[cut:])
    t = threading.Thread(target=put_back)
    t.start()
    return t


def test_guard_readers_give_up_within_seconds(tmp_path):
    from file_io import commit_state
    from file_io.restart_bundle import read_coulomb_policy_from_h5
    from zeta_loader import probe_zeta_file

    p = tmp_path / "zeta_q.h5"
    _cut(p, _fixture(p))
    assert commit_state.SETTLE_TIMEOUT_S == 60.0
    t0 = time.monotonic()
    probe = probe_zeta_file(p)
    t_probe = time.monotonic() - t0
    assert probe.exists and not probe.readable
    assert "truncated file" in probe.error
    t0 = time.monotonic()
    assert read_coulomb_policy_from_h5(str(p)) is None
    t_guard = time.monotonic() - t0
    # Bounded by the guard wait, not the 60 s default.
    assert 4.0 < t_probe < commit_state.GUARD_SETTLE_S + 5.0, t_probe
    assert 4.0 < t_guard < commit_state.GUARD_SETTLE_S + 5.0, t_guard
    with pytest.raises(OSError, match="open_h5_settled"):
        commit_state.open_h5_settled(p, "r", settle_s=0.5)
