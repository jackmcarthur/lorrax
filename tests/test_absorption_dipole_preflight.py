"""davidson_absorption: the dipole file is a PRODUCED input (2026-08-08).

RED TWIN recorded in FIX_driver_blockers.md.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

# ===========================================================================
# 1. davidson_absorption's dipole is a PRODUCED input, and says so
# ===========================================================================
def test_missing_dipole_names_its_producer(tmp_path):
    """A missing dipole file must refuse with the command that builds it.

    RED TWIN: drop the ``Path(path).is_file()`` preflight in
    ``absorption_common.load_dipole_h5`` and this raises h5py's
    ``FileNotFoundError: Unable to synchronously open file`` instead — true,
    and useless: it names neither the producer nor ``--skip-vnl``.
    """
    from bse.absorption_common import load_dipole_h5

    missing = tmp_path / "dipole_p_only.h5"
    with pytest.raises(FileNotFoundError) as ei:
        load_dipole_h5(missing)
    msg = str(ei.value)
    assert "get_dipole_mtxels" in msg, msg
    assert "--skip-vnl" in msg, msg
    assert "dipole_p_only.h5" in msg, msg


def test_present_dipole_still_loads(tmp_path):
    """The preflight must not stand between a real file and its reader."""
    h5py = pytest.importorskip("h5py")
    from bse.absorption_common import load_dipole_h5

    p = tmp_path / "d.h5"
    nk, nb = 2, 3
    dip = (np.arange(3 * nk * nb * nb).reshape(3, nk, nb, nb)
           .astype(np.complex128))
    de = np.ones((nk, nb, nb))
    with h5py.File(p, "w") as f:
        f.create_dataset("dipole_cart", data=dip)
        f.create_dataset("deltaE", data=de)
        f.attrs["nbands"] = nb
        f.attrs["nk"] = nk
    got, gde, attrs = load_dipole_h5(p)
    assert got.shape == (3, nk, nb, nb)
    assert attrs["nbands"] == nb and attrs["nk"] == nk
    assert np.array_equal(gde, de)


