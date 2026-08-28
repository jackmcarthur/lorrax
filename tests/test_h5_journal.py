"""The HDF5 operation journal — the black box, at the file_io seam.

WHAT THIS SUITE IS ABOUT.  All three HDF5 failure signatures in
``reports/metal_mpa_plan_2026-08-15/SLAB_IO_ROOT_CAUSE_AUDIT.md`` are
NATIVE events: a SIGSEGV after ``SlabIO.close``, a phantom "already open
for write", a garbage ``offset_base`` refused inside C++.  Python sees
none of the state that would explain them, so §C of that audit asked for
the one instrument that survives them — a line-buffered per-rank text
file written BEFORE each HDF5 call.  This suite pins the three properties
that make it evidence rather than noise:

* **a line per op, through both HDF5 library instances**, in one stream;
* **the crash ring lands on disk** when the registry refuses or when an
  exception crosses a ``SlabIO`` method — the 256 lines before the event;
* **``sync`` mode really is synchronous**, because the mode exists for
  the case where the process dies between the write and the flush.

FAKE TRANSPORT, REAL SEAM (the ``test_hdf5_one_owner`` template).  The
phdf5 FFI is not available in every environment this suite runs in, so
the ffi leg drives the REAL ``SlabIO`` methods over a stand-in backend
and the h5py leg is a real ``h5py`` file declared through the real
registry door.  What is under test is the journal and the hooks, both of
which are pure Python; the transport's own leg is a run log.
"""

import os
import re

import numpy as np
import pytest

from file_io import h5_journal as J
from file_io import hdf5_owner as HO
from file_io import slab_io as SIO

#: The frozen key order (audit §C).  A change here is a format change and
#: every consumer of a journal file has to be told about it.
KEYS = ["t", "rank", "stack", "op", "path", "handle", "ds", "off", "cnt",
        "mode", "owner", "rc"]


#: The format, as a matcher: every field but ``rc`` is whitespace-free by
#: construction, so anchoring on the key names is exact — and a line that
#: reorders, renames, drops or space-pollutes a field simply will not
#: match, which is what "frozen" has to mean to be testable.
LINE_RE = re.compile(
    " ".join(f"{k}=(?P<{k}>\\S*)" for k in KEYS[:-1]) + " rc=(?P<rc>.*)$")


@pytest.fixture(autouse=True)
def journal_dir(tmp_path, monkeypatch):
    """A private journal directory and a clean registry for every cell."""
    HO.reset_for_test()
    J.reset_for_test()
    # The production default is deliberately quiet.  This suite tests the
    # opt-in instrument, so enable it explicitly except in the default-mode
    # cell below.
    monkeypatch.setenv(J.MODE_ENV, "1")
    monkeypatch.delenv(HO.POLICY_ENV, raising=False)
    monkeypatch.setenv(J.DIR_ENV, str(tmp_path))
    yield tmp_path
    J.reset_for_test()
    HO.reset_for_test()


def parse(line):
    """One journal line -> dict, asserting the frozen format as it goes."""
    got = LINE_RE.match(line)
    assert got, f"line does not carry the frozen fields in order: {line}"
    return got.groupdict()


def lines():
    """Every journal line written by this process, from DISK."""
    path = J.journal_path()
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [ln for ln in fh.read().splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# A stand-in for the collective transport
# ---------------------------------------------------------------------------

class FakeBackend:
    """Enough ``_FfiBackend`` for ``SlabIO`` to drive: no MPI, no HDF5.

    It DECLARES ITS OPEN to the registry through the same ``note_open``
    the real backend calls, because that claim is what the journal's
    ``owner`` field reads and what the h5py side collides with — faking
    the transport must not fake the ownership, or the two-stack stream
    this suite is about stops being two-stack.

    For the same reason it declares ``journal_stack``: since 2026-08-27
    ``SlabIO`` stamps its journal lines with the backend's library name
    rather than a module constant of its own (there are two backends now,
    and the constant became a lie for one of them).  A stand-in that
    omitted it would either crash the door or need a default there — and a
    silently defaulted library name is exactly the defect that change
    removed.
    """

    #: This fake stands in for the collective transport, so it names that
    #: transport's library.
    journal_stack = "ffi"

    def __init__(self, path, mesh=None, mode="w"):
        self.path, self.mesh, self.mode = path, mesh, mode
        self._token = HO.note_open(path, HO.STACK_FFI, mode,
                                   where="test: FakeBackend")
        self.fh = 0x7F1234000000            # a plausible ctx pointer
        self._deferred_ds_attrs = []
        self.calls = []

    def close(self):
        self.calls.append("close")
        self.fh = 0
        HO.note_close(self.path, self._token)

    def create_dataset(self, name, *, shape, dtype, chunks=None, attrs=None):
        self.calls.append(("create", name, tuple(shape)))

    def write_attr(self, name, value):
        self.calls.append(("attr", name))

    def write_slab(self, name, A, **kw):
        self.calls.append(("write", name))

    def read_slab(self, name, *, shape=None, **kw):
        self.calls.append(("read", name))
        return np.zeros(shape or (2, 2))

    def padded_shape_for(self, name, *, mesh, partition_spec):
        return (2, 2)


class Boom(RuntimeError):
    """A transport failure that must cross ``SlabIO`` and be journaled."""


def _one_by_one_mesh():
    """A real 1x1 ``('x','y')`` mesh, where an ``object()`` used to do.

    ``SlabIO`` reads the mesh at construction now — ``common.collectives.
    mesh_is_emulated`` picks the tier from it — so a sentinel is refused by
    name (``expected a jax.sharding.Mesh, got object``).  That refusal is
    the point: a duck-typed mesh that silently answered "not emulated"
    would route to the collective transport on a geometry nobody checked.
    1x1 is the geometry this suite means anyway — it keeps every cell on
    the FFI door, which is the backend ``FakeBackend`` stands in for.
    """
    import jax
    import numpy as np
    from jax.sharding import Mesh
    return Mesh(np.asarray(jax.devices("cpu")[:1]).reshape(1, 1), ("x", "y"))


def slabio(monkeypatch, path, *, mode="w", backend=FakeBackend):
    monkeypatch.setattr(SIO, "_FfiBackend", backend)
    return SIO.SlabIO(path, mode=mode, mesh=_one_by_one_mesh())


# ---------------------------------------------------------------------------
# One open/write/read/close cycle, through both stacks
# ---------------------------------------------------------------------------

def test_one_cycle_through_both_stacks_lands_in_one_journal(monkeypatch,
                                                            tmp_path):
    """The deliverable: both HDF5 library instances, one file, in order.

    The ffi leg is a real ``SlabIO`` open/create/write/read/close over a
    stand-in transport.  The h5py leg is a real ``h5py`` file declared
    through the registry door — and its payload ops are deliberately NOT
    journaled, because h5py's own reads and writes pass no LORRAX choke
    point: what the seam sees, and all it can honestly claim to see, is
    each h5py open and its close, carrying the mode that says whether it
    could write.  That is exactly the fact audit A1 needs (who held this
    path, in which library, able to write, and when did they let go).
    """
    h5py = pytest.importorskip("h5py")
    path = str(tmp_path / "store.h5")

    # --- the h5py stack: a write cycle and a read cycle -----------------
    with HO.open_scope(path, HO.STACK_H5PY, "w", where="test: h5py writer"):
        with h5py.File(path, "w") as f:
            f.create_dataset("A", data=np.arange(4.0))
    with HO.open_scope(path, HO.STACK_H5PY, "r", where="test: h5py reader"):
        with h5py.File(path, "r") as f:
            assert f["A"].shape == (4,)

    # --- the ffi stack: the whole SlabIO surface ------------------------
    io = slabio(monkeypatch, path, mode="a")
    io.create_dataset("A", shape=(2, 2), dtype=np.float64)
    io.write_slab("A", np.zeros((2, 2)))
    io.read_slab("A", shape=(2, 2))
    io.close()

    rows = [parse(ln) for ln in lines()]
    assert rows, "the journal must exist on disk without any flush call"
    assert {r["path"] for r in rows} == {os.path.abspath(path)}
    assert all(r["rc"] == "ok" for r in rows), rows

    h5 = [r for r in rows if r["stack"] == "h5py"]
    ffi = [r for r in rows if r["stack"] == "ffi"]
    assert [r["op"] for r in h5] == ["open", "close", "open", "close"]
    assert [r["mode"] for r in h5] == ["w", "w", "r", "r"]
    # open (the registry claim) + open (SlabIO's completion line, the only
    # one that can carry the ctx handle) + create + write + read +
    # close (SlabIO, at issue) + close (the registry release).
    assert [r["op"] for r in ffi] == [
        "open", "open", "create", "write", "read", "close", "close"]

    opens = [r for r in ffi if r["op"] == "open"]
    assert opens[0]["handle"].startswith("tok"), "registry token pairs open/close"
    assert opens[0]["owner"] == "free", "what the FFI open walked into"
    assert opens[1]["handle"] == hex(0x7F1234000000), "the ctx pointer"
    assert opens[1]["owner"] == "ffi:1w", "the registry's live verdict"
    closes = [r for r in ffi if r["op"] == "close"]
    assert closes[-1]["handle"] == opens[0]["handle"], "open/close pair"
    create = next(r for r in ffi if r["op"] == "create")
    assert create["ds"] == "A" and create["cnt"] == "(2,2)"
    assert next(r for r in ffi if r["op"] == "read")["cnt"] == "(2,2)"
    assert float(rows[0]["t"]) <= float(rows[-1]["t"])
    assert {int(r["rank"]) for r in rows} == {int(rows[0]["rank"])}


def test_the_line_format_is_frozen(monkeypatch, tmp_path):
    """Twelve keys, that order, no spaces except inside ``rc``'s tail."""
    J.record("write", tmp_path / "f.h5", stack="ffi", handle=0x10, ds="A",
             off=(0, 0), cnt=(4, 8), mode="w")
    (line,) = lines()
    got = parse(line)
    assert got["stack"] == "ffi" and got["op"] == "write"
    assert got["handle"] == "0x10" and got["ds"] == "A"
    assert got["off"] == "(0,0)" and got["cnt"] == "(4,8)"
    assert got["mode"] == "w" and got["rc"] == "ok"
    assert got["owner"] == "free", "nothing holds the path in the registry"
    assert line.startswith("t="), "the monotonic clock leads every line"
    assert "  " not in line, "single-space separated fields"
    assert float(got["t"]) > 0 and got["rank"].isdigit()


# ---------------------------------------------------------------------------
# The crash ring
# ---------------------------------------------------------------------------

def test_the_crash_ring_dumps_on_a_registry_refusal(monkeypatch, tmp_path):
    """(i) of the crash hook: the refusal the segfault used to be.

    The FFI holds the store open for writing and serial h5py walks in —
    the one-owner refusal.  The journal must carry the refusal WITH its
    reason, and the ring must be on disk with the lines that led to it,
    because the next thing this shape does in production is die natively.
    """
    path = str(tmp_path / "fit.h5")
    token = HO.note_open(path, HO.STACK_FFI, "a", where="test: SlabIO(a)")
    with pytest.raises(RuntimeError, match="one-owner-per-file"):
        HO.note_open(path, HO.STACK_H5PY, "r", where="test: h5py reader")
    HO.note_close(path, token)

    rows = [parse(ln) for ln in lines()]
    refused = [r for r in rows if r["rc"] != "ok"]
    assert len(refused) == 1, rows
    assert refused[0]["stack"] == "h5py" and refused[0]["op"] == "open"
    assert refused[0]["rc"].startswith("refused:LORRAX HDF5 one-owner")
    assert len(refused[0]["rc"]) <= len("refused:") + J.REFUSAL_CHARS
    # The verdict field says what the refused open walked INTO.
    assert refused[0]["owner"] == "ffi:1w"

    dump = J.crash_path()
    assert os.path.exists(dump), "the ring must be dumped by the refusal"
    text = open(dump).read()
    assert "crash ring" in text and "RuntimeError" in text
    assert "one-owner-per-file" in text
    # ... and it carries the HISTORY, not just the failing line.
    assert text.count("op=open") >= 2


def test_an_exception_crossing_slabio_dumps_the_ring(monkeypatch, tmp_path):
    """(ii) of the crash hook: any exception crossing a SlabIO method."""
    class Exploding(FakeBackend):
        def write_slab(self, name, A, **kw):
            raise Boom("the collective write died")

    path = str(tmp_path / "cube.h5")
    io = slabio(monkeypatch, path, backend=Exploding)
    io.create_dataset("A", shape=(2, 2), dtype=np.float64)
    with pytest.raises(Boom):
        io.write_slab("A", np.zeros((2, 2)))

    rows = [parse(ln) for ln in lines()]
    writes = [r for r in rows if r["op"] == "write"]
    assert [r["rc"] for r in writes] == [
        "ok", "refused:the collective write died"], (
        "the issue-time line, then the refusal")
    assert os.path.exists(J.crash_path())
    text = open(J.crash_path()).read()
    assert "Boom" in text and "op=create" in text


def test_the_ring_keeps_the_last_256_lines(monkeypatch, tmp_path):
    for i in range(J.RING_LINES + 40):
        J.record("read", tmp_path / "f.h5", stack="ffi", ds=f"d{i}")
    ring = J.ring()
    assert len(ring) == J.RING_LINES
    assert "ds=d39" not in "\n".join(ring)
    assert f"ds=d{J.RING_LINES + 39}" in ring[-1]
    # The FILE keeps everything; only the ring is bounded.
    assert len(lines()) == J.RING_LINES + 40


def test_the_atexit_hook_dumps_a_refusal_nobody_dumped(monkeypatch, tmp_path):
    """(iii): a refusal the caller caught still leaves its ring behind."""
    J.record("close", tmp_path / "f.h5", stack="ffi", rc=Boom("caught"))
    assert not os.path.exists(J.crash_path())
    J._atexit_dump()
    assert os.path.exists(J.crash_path())
    assert "atexit" in open(J.crash_path()).read()


# ---------------------------------------------------------------------------
# sync mode
# ---------------------------------------------------------------------------

def test_sync_mode_fsyncs_every_line_in_order(monkeypatch, tmp_path):
    """``sync`` is the segfault-grade mode: durable BEFORE the next op.

    Line buffering already puts each line in the page cache as it is
    written (asserted by every other cell here reading the file with no
    flush).  ``sync`` adds the fsync, which is the difference between
    surviving a process death and surviving a node death — and it must
    happen per LINE, not once at the end, or the last op before the
    crash is exactly the one that is missing.
    """
    seen = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (seen.append(fd),
                                                 real_fsync(fd))[1])
    monkeypatch.setenv(J.MODE_ENV, "sync")
    assert J.mode() == J.MODE_SYNC and J.enabled()
    for i in range(3):
        J.record("write", tmp_path / "f.h5", stack="ffi", ds=f"d{i}")
        # Durable already, before the NEXT record — the ordering property.
        assert len(lines()) == i + 1, "sync mode must not batch"
    assert len(seen) == 3, "one fsync per line, not one per file"
    assert [parse(ln)["ds"] for ln in lines()] == ["d0", "d1", "d2"]


def test_the_toggle_defaults_off_and_one_enables_it(monkeypatch, tmp_path):
    monkeypatch.delenv(J.MODE_ENV, raising=False)
    assert J.mode() == J.MODE_OFF, "production default creates no sidecar"
    assert not J.enabled()
    assert J.record("open", tmp_path / "f.h5", stack="ffi", mode="w") is None
    assert lines() == []
    monkeypatch.setenv(J.MODE_ENV, "1")
    assert J.mode() == J.MODE_ON and J.enabled()
    J.record("open", tmp_path / "f.h5", stack="ffi", mode="w")
    assert len(lines()) == 1
    monkeypatch.setenv(J.MODE_ENV, "0")
    assert not J.enabled()
    assert J.record("close", tmp_path / "f.h5", stack="ffi") is None
    assert len(lines()) == 1, "nothing written while off"
    monkeypatch.setenv(J.MODE_ENV, "on")
    J.record("close", tmp_path / "f.h5", stack="ffi")
    assert len(lines()) == 2


def test_an_unknown_toggle_refuses_by_name(monkeypatch):
    monkeypatch.setenv(J.MODE_ENV, "verbose")
    with pytest.raises(ValueError, match="LORRAX_H5_JOURNAL"):
        J.mode()


def test_an_unknown_op_or_stack_refuses(tmp_path):
    """The vocabulary is frozen with the format; a typo is unrecoverable."""
    with pytest.raises(ValueError, match="unknown op"):
        J.record("flush", tmp_path / "f.h5", stack="ffi")
    with pytest.raises(ValueError, match="unknown stack"):
        J.record("open", tmp_path / "f.h5", stack="hdf5")


# ---------------------------------------------------------------------------
# The journal must never be the thing that kills a run
# ---------------------------------------------------------------------------

def test_the_journal_never_journals_itself(monkeypatch, tmp_path):
    """One op is one line even when the owner lookup logs (audit §C)."""
    def reentrant(path):
        J.record("read", path, stack="ffi", ds="recursion")
        return "free"

    monkeypatch.setattr(HO, "live_verdict", reentrant)
    J.record("write", tmp_path / "f.h5", stack="ffi", ds="A")
    got = lines()
    assert len(got) == 1 and "ds=A" in got[0], got


def test_an_unwritable_directory_disables_the_journal_without_raising(
        monkeypatch, tmp_path):
    """A log that kills a 40-node job is worse than no log."""
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("I am a file where the journal wants a directory\n")
    monkeypatch.setenv(J.DIR_ENV, str(blocker / "deeper"))
    with pytest.warns(UserWarning, match="journal disabled"):
        J.record("open", tmp_path / "f.h5", stack="ffi", mode="w")
    assert not J.enabled(), "disabled for the rest of the process"
    # And the caller's next op still returns normally.
    assert J.record("close", tmp_path / "f.h5", stack="ffi") is None
    assert J.crash_dump("nothing to write it to") is None


def test_a_slabio_open_that_fails_is_journaled_and_dumped(monkeypatch,
                                                          tmp_path):
    """The open path's own failure is evidence too — S2's shape."""
    class Refusing(FakeBackend):
        def __init__(self, path, mesh=None, mode="w"):
            raise Boom("already open for write")

    with pytest.raises(Boom):
        slabio(monkeypatch, str(tmp_path / "f.h5"), backend=Refusing)
    rows = [parse(ln) for ln in lines()]
    assert [r["op"] for r in rows] == ["open"]
    assert rows[0]["rc"] == "refused:already open for write"
    assert os.path.exists(J.crash_path())
