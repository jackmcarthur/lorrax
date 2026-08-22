"""The status_flags OSError must arrive with its cause attached.

THE FAILURE THIS IS ABOUT.  At 16 ranks on 4 nodes the twelve ranks NOT on
rank 0's node die reading the MPA fit ledger — every one of them, at the
same line, after all of an iteration's blocks are on disk — with

    OSError: Unable to synchronously open file (file is already open for
    write (may use <h5clear file> to clear file consistency flags))

The message is HDF5's and it is accurate; what it cannot say is WHY, and
the why has been re-derived by hand at least twice.  It is the SUPERBLOCK's
``status_flags`` write-open bit: the FFI's cray ``libhdf5_parallel`` wrote
and closed the store, h5py's bundled libhdf5 reads the superblock, and the
clear has not become visible to that Lustre client.  Three things it is
NOT — a POSIX lock, this module's registry, a regression from the A1
work — each ruled out by its own measurement (register 2026-08-15/16;
A/B at ``bf57701b``).

``mpa_store._superblock_flag_diagnosis`` attaches that.  It repairs
nothing, and the docstring says so: the repair is to stop reading a
phdf5-written store with a second HDF5 library.  What it buys is that the
NEXT twelve identical tracebacks name their own mechanism.

WHY IT NEEDS A TEST.  A message-decorating wrapper has exactly two ways to
be wrong, and both are silent: it can fail to fire on the message it is
for, and it can fire on messages it is not for — swallowing an unrelated
``OSError`` (a missing file, a permission error) into a paragraph about
Lustre coherence, which is strictly worse than the bare error.
"""
from __future__ import annotations

import pytest

from file_io import mpa_store


#: HDF5's phrase, verbatim from the failing runs (JID 57038615 and the
#: 4-rank single-node sighting of 2026-08-16).
_REAL = (
    "Unable to synchronously open file (file is already open for write "
    "(may use <h5clear file> to clear file consistency flags))")


def test_the_status_flags_error_is_diagnosed():
    got = mpa_store._superblock_flag_diagnosis(
        OSError(_REAL), "/scratch/mpa_fit_sc_0000.h5", "r",
        "mpa_store.fit_completion_ledger")
    text = str(got)
    assert isinstance(got, OSError)
    # The original message survives — a diagnosis that REPLACES the
    # library's own text loses the string every existing log grep uses.
    assert _REAL in text
    assert "status_flags" in text
    assert "/scratch/mpa_fit_sc_0000.h5" in text
    assert "mpa_store.fit_completion_ledger" in text
    # The three exonerations, each of which cost a session to establish.
    assert "HDF5_USE_FILE_LOCKING" in text
    assert "hdf5_owner" in text
    assert "bf57701b" in text
    # And a repair, not just a story.
    assert "LORRAX_HDF5_ONE_OWNER=strict" in text


@pytest.mark.parametrize("message", [
    "No such file or directory",
    "Permission denied",
    "Unable to synchronously open file (bad superblock version number)",
    "unable to lock file, errno = 11",
])
def test_an_unrelated_oserror_passes_through_untouched(message):
    """The red twin, and the one that matters more.

    A wrapper that decorates every OSError would bury a missing file under
    a paragraph about Lustre cache coherence, and the operator would go
    looking for a race that is not there.  Identity, not equality: the
    caller must get back the SAME object, so nothing about the traceback
    or the errno moves.
    """
    exc = OSError(message)
    assert mpa_store._superblock_flag_diagnosis(
        exc, "/scratch/x.h5", "a", "mpa_store.whatever") is exc


def test_the_door_still_declares_and_releases_on_a_failed_open(tmp_path):
    """The diagnosis must not leak a registry claim.

    ``_h5`` declares the open to ``file_io.hdf5_owner`` BEFORE h5py runs,
    so an exception raised out of ``QirrDest.__enter__`` has to leave the
    claim released — otherwise the first failed open would make every
    later legitimate open on that path refuse, and the run would report a
    one-owner violation that never happened.
    """
    from file_io import hdf5_owner
    missing = tmp_path / "not_there.h5"
    with pytest.raises(OSError):
        with mpa_store._h5(str(missing), "r"):
            pass                                    # pragma: no cover
    assert hdf5_owner.live_verdict(str(missing)) == "free", (
        "a failed open left a live claim behind; the next open on this "
        "path would be refused for a handle that does not exist.")
