"""A TD-aGW checkpoint stopped part way resumes from a consistent state.

``tdagw.tdgw_io.write_checkpoint`` writes TDGW's ``it_next`` / ``tchkp`` /
``lessG_knn`` and LORRAX's integrator carry (``checkpoint/lorrax``).  Before
2026-11-22 it wrote ``it_next`` first and then deleted and recreated the
carry, so a writer stopped in between left the new ``it_next`` beside the old
carry, and ``read_checkpoint`` returned that pair: a restart then continued
the old state from the new measurement index, silently.

Here a second checkpoint is stopped at every HDF5 mutation in turn (an
exception raised by the n-th dataset write, attribute write, group create,
delete, move or flush; the file is then closed as an exception closes it).
Every stop must read back as the first checkpoint or the second, never a mix.
``_write_chunk`` is stopped the same way with two trials: the resumed
``it_next`` never runs past rows that are on disk in every trial's file.

Limitation: an exception closes the file through HDF5's normal path; a
SIGKILL does not, and HDF5 gives no crash consistency for an unclosed file.
The flushes in ``write_checkpoint`` order what reaches the disk.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")


class _Stop(RuntimeError):
    pass


def _mutators():
    from h5py._hl import attrs, dataset, files, group
    return [(dataset.Dataset, "__setitem__"), (group.Group, "__delitem__"),
            (group.Group, "create_group"), (group.Group, "create_dataset"),
            (group.Group, "move"), (attrs.AttributeManager, "__setitem__"),
            (files.File, "flush")]


class _StopAt:
    """Raise :class:`_Stop` at the n-th HDF5 mutation (n = None: count only)."""

    def __init__(self, monkeypatch, n):
        self.n, self.count = n, 0
        for cls, name in _mutators():
            orig = getattr(cls, name)
            monkeypatch.setattr(cls, name, self._wrap(orig))

    def _wrap(self, orig):
        def call(*a, **kw):
            self.count += 1
            if self.n is not None and self.count == self.n:
                raise _Stop(f"stopped at mutation {self.n}")
            return orig(*a, **kw)
        return call


def _blank(path, nk=2, nb=2, measurements=()):
    with h5py.File(path, "w") as f:
        c = f.create_group("tdgw_data/checkpoint")
        c.create_dataset("it_next", data=-1, dtype=np.int64)
        c.create_dataset("tchkp", data=0.0, dtype=np.float64)
        c.create_dataset("lessG_knn", shape=(nk, nb, nb), dtype=np.complex128)
        td = f.create_group("tdgw_data/time_domain")
        for name, n in measurements:
            m = td.create_group(name)
            m.create_dataset("times", data=np.arange(n, dtype=np.float64))
            m.create_dataset("data", shape=(n, 3), dtype=np.float64)


def _carry(i):
    """A carry whose every field names the measurement index it goes with."""
    return dict(leaves=[np.full(3, float(i)), np.full((2, 2), 10.0 * i)], step=10 * i,
                fingerprint="fp")


def _checkpoint(tdgw_io, path, i):
    tdgw_io.write_checkpoint(path, it_next=i, tchkp=0.5 * i,
                             lessG=np.full((2, 2, 2), 1j * i), carry=_carry(i))


def _pair(rec):
    """(it_next, tchkp, step, leaf value): one index if the record is consistent."""
    return (rec["it_next"], rec["tchkp"] / 0.5, rec["step"] / 10,
            float(rec["leaves"][0][0]), float(rec["leaves"][1][0, 0]) / 10)


def test_checkpoint_stopped_anywhere_reads_one_checkpoint(tmp_path, monkeypatch):
    from tdagw import tdgw_io
    with pytest.MonkeyPatch.context() as mp:
        probe = _StopAt(mp, None)
        _blank(tmp_path / "count.h5")
        _checkpoint(tdgw_io, tmp_path / "count.h5", 10)
        start = probe.count
        _checkpoint(tdgw_io, tmp_path / "count.h5", 20)
        n_ops = probe.count - start
    assert n_ops >= 5, n_ops
    seen = set()
    for n in range(1, n_ops + 1):
        path = tmp_path / f"stop_{n}.h5"
        _blank(path)
        _checkpoint(tdgw_io, path, 10)
        with pytest.MonkeyPatch.context() as mp:
            stopper = _StopAt(mp, None)
            stopper.n = n
            with pytest.raises(_Stop):
                _checkpoint(tdgw_io, path, 20)
        rec = tdgw_io.read_checkpoint(path)
        got = _pair(rec)
        assert len(set(got)) == 1 and got[0] in (10, 20), (
            f"stop at mutation {n} of {n_ops}: it_next, tchkp, step and carry disagree: {got}")
        seen.add(got[0])
    path = tmp_path / "whole.h5"
    _blank(path)
    _checkpoint(tdgw_io, path, 10)
    _checkpoint(tdgw_io, path, 20)
    rec = tdgw_io.read_checkpoint(path)
    assert set(_pair(rec)) == {20} and not rec.get("torn", False)
    assert seen == {10, 20}, seen


def test_first_checkpoint_stopped_anywhere_reads_none_or_it(tmp_path):
    """The first carry: a stop before its commit reads as no carry, never a partial one."""
    from tdagw import tdgw_io
    with pytest.MonkeyPatch.context() as mp:
        probe = _StopAt(mp, None)
        _blank(tmp_path / "count.h5")
        start = probe.count
        _checkpoint(tdgw_io, tmp_path / "count.h5", 10)
        n_ops = probe.count - start
    outcomes = set()
    for n in range(1, n_ops + 1):
        path = tmp_path / f"first_{n}.h5"
        _blank(path)
        with pytest.MonkeyPatch.context() as mp:
            stopper = _StopAt(mp, None)
            stopper.n = n
            with pytest.raises(_Stop):
                _checkpoint(tdgw_io, path, 10)
        rec = tdgw_io.read_checkpoint(path)
        if rec["leaves"] is None:
            outcomes.add("none")
            continue
        got = _pair(rec)
        assert len(set(got)) == 1 and got[0] == 10, (n, got)
        outcomes.add(10)
    assert outcomes == {"none", 10}, outcomes


def test_write_chunk_rows_precede_the_carry(tmp_path, monkeypatch):
    """Two trials: the resumed it_next never runs past rows on disk in either file."""
    stages = pytest.importorskip("tdagw.stages")
    from tdagw import tdgw_io
    monkeypatch.setattr(stages, "process_rank", lambda: 0)
    n_meas = 8
    paths = tuple(str(tmp_path / f"trial{t}.h5") for t in range(2))
    meas = (("obs", "scalar", {}, range(0, n_meas)),)
    outs = types.SimpleNamespace(paths=paths, meas=meas)
    system = types.SimpleNamespace(nk=2)

    def chunk(i0, i1):
        samples = {"obs": np.stack([np.full((2, 3), float(i)) + np.arange(2)[:, None]
                                    for i in range(i0, i1)])}
        rho = np.zeros((2, 2, 2, 2), dtype=np.complex128)
        stages._write_chunk(outs, system, i0, i1, samples,
                            lambda: (rho, _carry(i1)["leaves"], 10 * i1), 0.5 * i1, "fp")

    def fresh():
        for p in paths:
            _blank(p, measurements=(("obs", n_meas),))
        chunk(0, 4)

    with pytest.MonkeyPatch.context() as mp:
        fresh()
        probe = _StopAt(mp, None)
        chunk(4, 8)
        n_ops = probe.count
    for n in range(1, n_ops + 1):
        fresh()
        with pytest.MonkeyPatch.context() as mp:
            stopper = _StopAt(mp, None)
            stopper.n = n
            with pytest.raises(_Stop):
                chunk(4, 8)
        rec = tdgw_io.read_checkpoint(paths[0])
        got = _pair(rec)
        assert len(set(got)) == 1 and got[0] in (4, 8), (n, got)
        for t, p in enumerate(paths):
            with h5py.File(p, "r") as f:
                rows = f["tdgw_data/time_domain/obs/data"][: rec["it_next"], 0]
            want = np.arange(rec["it_next"], dtype=np.float64) + t
            assert np.array_equal(rows, want), (
                f"stop at mutation {n} of {n_ops}: resume at it_next {rec['it_next']} but "
                f"trial {t} holds rows {rows}")
