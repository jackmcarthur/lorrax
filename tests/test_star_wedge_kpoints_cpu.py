"""Star-wedge files carry their k lists, and the run's readers check them.

``kin_ion.h5`` and ``sigma_mnk.h5`` store one row per symmetry star, in the
star order of ``symmetry_maps.star_wedge_rows``, which on a reduced cell is a
permutation of the WFN's k order (bulk CrCl3 4³: stored row 6 = WFN k 13).
The writers file both k lists in crystal coordinates
(``file_io.kin_ion.star_wedge_kpoints``); here, on the bcc Na hsuite fixture
(27 full k over 4 stars):

* every full-BZ k is the symmetry image, under its own ``sym_idx_k`` row, of
  the stored k its ``irr_idx`` row points at (and a reversed stored list is
  not), so the stored list labels the rows the tables unfold;
* the queue refuses k lists of the wrong extent;
* ``restart_bundle.check_kpoints_against_run`` accepts the run's own list,
  refuses a shifted one by name, and reads a file without the lists
  (the committed fixture's ``kin_ion.h5``) unchecked.

Host-side file metadata, no device code: a CPU unit cell, exempt from the
four-GPU rule under its unit-and-CPU-cell clause.
"""
import os

import h5py
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "hsuite", "fixture_na")


def _images_match(sym, stored, take):
    """Is every full-BZ k the image of its stored parent k (mod G)?"""
    full = np.asarray(sym.unfolded_kpts, dtype=np.float64)
    mats = np.asarray(sym.sym_mats_k, dtype=np.float64)
    rows = np.asarray(sym.sym_idx_k)
    mapped = np.einsum("kij,kj->ki", mats[rows], stored[np.asarray(take)])
    d = mapped - full
    return bool(np.all(np.abs(d - np.rint(d)) < 1e-8))


def test_star_wedge_kpoints(tmp_path):
    from ffi import _services
    _services.ensure_on_path()
    import symmetry_maps
    from wfn_loader import WfnLoader
    from file_io.kin_ion import (
        KPOINTS_FULL_DATASET, KPOINTS_STORED_DATASET, queue_star_kpoints,
        star_wedge_kpoints)
    from file_io.restart_bundle import check_kpoints_against_run

    wfn = WfnLoader(os.path.join(FIXTURE, "WFN.h5"))
    sym = wfn.symmetry()
    stored, full = star_wedge_kpoints(sym)
    take, sidx, nss = symmetry_maps.star_wedge_tables(sym)
    assert stored.shape == (int(np.max(take)) + 1, 3) == (4, 3)
    assert full.shape == (int(sym.nk_tot), 3)
    # The stored list is the WFN's own k (mod G), in star order.
    labels, _ = symmetry_maps.star_wedge_rows(sym)
    d = stored - np.asarray(wfn.kpoints)[labels]
    assert np.max(np.abs(d - np.rint(d))) < 1e-12
    assert _images_match(sym, stored, take)
    assert not _images_match(sym, stored[::-1], take)

    class _IO:
        def __init__(self):
            self.queued = {}

        def write_attr(self, name, value):
            self.queued[name] = np.asarray(value)

    io = _IO()
    queue_star_kpoints(io, (stored, full), n_rows=4, nk_full=full.shape[0],
                       where="test")
    assert set(io.queued) == {KPOINTS_STORED_DATASET, KPOINTS_FULL_DATASET}
    with pytest.raises(ValueError, match="do not match"):
        queue_star_kpoints(io, (stored[:3], full), n_rows=4,
                           nk_full=full.shape[0], where="test")
    with pytest.raises(ValueError, match="without their k lists"):
        queue_star_kpoints(io, None, n_rows=4, nk_full=full.shape[0],
                           where="test")

    path = tmp_path / "star.h5"
    with h5py.File(path, "w") as f:
        for name, value in io.queued.items():
            f[name] = value
    notes = []
    assert check_kpoints_against_run(str(path), wfn.kvecs(), print_fn=notes.append)
    kirr_fullids = np.asarray(sym.kirr_fullids)
    assert check_kpoints_against_run(str(path), np.asarray(wfn.kpoints) + 1.0,
                                     rows=kirr_fullids, print_fn=notes.append)
    shifted = wfn.kvecs().copy()
    shifted[5] += 0.25
    with pytest.raises(ValueError, match="GATE k_list_mismatch"):
        check_kpoints_against_run(str(path), shifted, print_fn=notes.append)
    with pytest.raises(ValueError, match="GATE k_list_mismatch"):
        check_kpoints_against_run(str(path), wfn.kvecs()[:-1], print_fn=notes.append)
    assert notes == []
    # A file written before the lists existed is read unchecked, with a NOTE.
    assert not check_kpoints_against_run(
        os.path.join(FIXTURE, "kin_ion.h5"), wfn.kvecs(), print_fn=notes.append)
    assert len(notes) == 1 and "NOTE" in notes[0]
