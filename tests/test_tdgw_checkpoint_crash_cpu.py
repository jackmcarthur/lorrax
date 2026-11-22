"""TD-aGW checkpoint file size and SIGKILL safety (``tdagw.tdgw_io``).

(1) Fifty checkpoints with a 10 MB carry: ``tdgw.h5`` stays at the size of two
    carry slots plus the header; the writer before 2026-11-22 (reproduced
    below, for the record) is measured beside it.
(2) A subprocess writes checkpoints in a loop and is killed with SIGKILL at a
    random moment (100 trials, at least one committed checkpoint first): the
    file opens, and ``read_checkpoint`` returns one consistent checkpoint (the
    record's index, step, time and every leaf name the same checkpoint).
"""
import os
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _main_write_checkpoint(path, *, it_next, tchkp, lessG, carry=None):
    """``tdgw_io.write_checkpoint`` as on main before 2026-11-22."""
    with h5py.File(path, "a") as f:
        c = f["tdgw_data/checkpoint"]
        c["it_next"][...] = it_next
        c["tchkp"][...] = tchkp
        c["lessG_knn"][...] = lessG
        if carry is not None:
            if "lorrax" in c:
                del c["lorrax"]
            g = c.create_group("lorrax")
            for i, leaf in enumerate(carry["leaves"]):
                g.create_dataset(f"leaf_{i:03d}", data=np.asarray(leaf))
            g.attrs["n_leaves"] = len(carry["leaves"])
            g.attrs["step"] = int(carry["step"])
            g.attrs["fingerprint"] = carry["fingerprint"]


def _staged_write_checkpoint(path, *, it_next, tchkp, lessG, carry=None):
    """The unlanded 2026-11-22 staged writer (claude/io-hazards-20261122-r2), for the record."""
    with h5py.File(path, "a") as f:
        c = f["tdgw_data/checkpoint"]
        if carry is not None:
            if "lorrax_staged" in c:
                del c["lorrax_staged"]
            g = c.create_group("lorrax_staged")
            for i, leaf in enumerate(carry["leaves"]):
                g.create_dataset(f"leaf_{i:03d}", data=np.asarray(leaf))
            g.attrs["n_leaves"] = len(carry["leaves"])
            g.attrs["step"] = int(carry["step"])
            g.attrs["fingerprint"] = carry["fingerprint"]
            g.attrs["it_next"] = int(it_next)
            g.attrs["tchkp"] = float(tchkp)
            g.attrs["complete"] = True
            f.flush()
            if "lorrax" in c:
                del c["lorrax"]
            c.move("lorrax_staged", "lorrax")
            f.flush()
        c["it_next"][...] = it_next
        c["tchkp"][...] = tchkp
        c["lessG_knn"][...] = lessG
        f.flush()


def _blank(path):
    with h5py.File(path, "w") as f:
        c = f.create_group("tdgw_data/checkpoint")
        c.create_dataset("it_next", data=-1, dtype=np.int64)
        c.create_dataset("tchkp", data=0.0, dtype=np.float64)
        c.create_dataset("lessG_knn", shape=(2, 2, 2), dtype=np.complex128)


def _carry(i, n):
    return dict(leaves=[np.full(n, float(i)), np.full((4, 4), 10.0 * i)], step=10 * i,
                fingerprint="fp")


def test_fifty_checkpoints_do_not_grow_the_file(tmp_path):
    from tdagw import tdgw_io
    n = 10 * 2 ** 20 // 8                                  # 10 MiB leaf
    sizes = {}
    for name, write in (("slotted", tdgw_io.write_checkpoint), ("before", _main_write_checkpoint),
                        ("staged", _staged_write_checkpoint)):
        path = tmp_path / f"{name}.h5"
        _blank(path)
        for i in range(1, 51):
            write(path, it_next=i, tchkp=0.5 * i, lessG=np.full((2, 2, 2), 1j * i),
                  carry=_carry(i, n))
        sizes[name] = os.path.getsize(path)
    rec = tdgw_io.read_checkpoint(tmp_path / "slotted.h5")
    assert rec["it_next"] == 50 and rec["step"] == 500 and float(rec["leaves"][0][0]) == 50.0
    print(f"tdgw.h5 after 50 checkpoints of a 10 MiB carry: slotted {sizes['slotted']} B, "
          f"before (main) {sizes['before']} B, staged (unlanded r2) {sizes['staged']} B")
    assert sizes["slotted"] <= 2 * 8 * n + 2 ** 20, sizes


_WRITER = r"""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from tdagw import tdgw_io
path, n = sys.argv[2], int(sys.argv[3])
i = 0
while True:
    i += 1
    tdgw_io.write_checkpoint(path, it_next=i, tchkp=0.5 * i, lessG=np.full((2, 2, 2), 1j * i),
                             carry=dict(leaves=[np.full(n, float(i)), np.full((4, 4), 10.0 * i)],
                                        step=10 * i, fingerprint="fp"))
    print(i, flush=True)
"""


def test_sigkill_anywhere_leaves_one_checkpoint(tmp_path):
    from tdagw import tdgw_io
    rng = np.random.default_rng(20261122)
    n = 2 ** 20 // 8                                       # 1 MiB leaf
    bad, done = [], []
    for trial in range(100):
        path = str(tmp_path / f"k{trial}.h5")
        _blank(path)
        p = subprocess.Popen([sys.executable, "-c", _WRITER, SRC, path, str(n)],
                             stdout=subprocess.PIPE, text=True,
                             env={**os.environ, "HDF5_USE_FILE_LOCKING": "FALSE"})
        first = p.stdout.readline()                        # one checkpoint committed
        assert first.strip(), "writer died before its first checkpoint"
        time.sleep(float(rng.uniform(0.0, 0.05)))
        p.send_signal(signal.SIGKILL)
        p.wait()
        p.stdout.close()
        try:
            rec = tdgw_io.read_checkpoint(path)
            got = (rec["it_next"], rec["tchkp"] / 0.5, rec["step"] / 10,
                   float(rec["leaves"][0][0]), float(rec["leaves"][0][-1]),
                   float(rec["leaves"][1][0, 0]) / 10)
            if len(set(got)) != 1:
                bad.append((trial, got))
            done.append(rec["it_next"])
        except Exception as exc:                           # noqa: BLE001
            bad.append((trial, repr(exc)[:200]))
        os.remove(path)
    print(f"SIGKILL trials: 100, consistent {100 - len(bad)}, resumed indices "
          f"{min(done) if done else None}..{max(done) if done else None}")
    assert not bad, bad[:5]
