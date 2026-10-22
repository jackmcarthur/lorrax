"""BerkeleyGW-named BSE input: the behaviour no driver test reaches, on CPU in seconds.

1. Old names refuse with one line ("`old` is now `new`"), never aliased.
2. BerkeleyGW line syntax: bare flags and ``key value`` lines read as ``key = value``.
3. Spin model: spinor (weight 1) by default on a spinor WFN; singlet (2) by default and
   triplet (0) on a non-spinor WFN; singlet/triplet refuse on spinors; RPA only by
   ``local_fields``.
4. ``polarization p1 p2 p3`` is along b1 b2 b3 and normalized as a Cartesian vector.
5. ``kgrid_fine_shift`` follows kgrid.x: k = (i + dk)/N; a half-step N grid is the odd
   points of the 2N grid.
"""
import h5py
import numpy as np
import pytest

from bandstructure.bse_setup import uniform_kgrid_frac
from bse import bse_deck, bse_preconditioner
from bse.absorption_common import polarization_directions
from gw.gw_config import _bgw_line
from runtime import renames


def test_old_names_refuse_with_one_line():
    with pytest.raises(ValueError, match=r"`bse_k_grid` is now `kgrid_fine`"):
        renames.refuse_old_deck_keys("gw", {"bse_k_grid": "4 4 4"})
    lines = renames.old_cli_flags(["--n-val", "4"], program="bse.bse_jax")
    assert lines == ["`--n-val` is now deck key `number_val_bands_fine`"]


@pytest.mark.parametrize("line, want", [
    ("use_momentum\n", "use_momentum = true\n"),
    ("number_val_bands_fine 4\n", "number_val_bands_fine = 4\n"),
    ("kgrid_fine_shift 0.47 0.37 0.31  # WFN_fi\n", "kgrid_fine_shift = 0.47 0.37 0.31  # WFN_fi\n"),
    ("nval = 4\n", "nval = 4\n"),
    ("# comment\n", "# comment\n"),
    ("[cohsex]\n", "[cohsex]\n"),
])
def test_bgw_line_syntax(line, want):
    assert _bgw_line(line) == want


@pytest.fixture
def spin(monkeypatch, tmp_path):
    wfn = tmp_path / "WFN.h5"
    wfn.write_bytes(b"")
    def run(nspinor, **keys):
        monkeypatch.setattr(bse_deck, "wfn_spin", lambda path: (nspinor, 1))
        return bse_deck.spin_kernel(keys, set(keys), str(wfn))
    yield run
    bse_preconditioner.set_exchange_channel("singlet")


def test_spin_model(spin):
    k = spin(2)
    assert (k["include_w"], k["channel"], k["nspinor"]) == (True, "singlet", 2)
    assert bse_preconditioner.exchange_spin_weight(2) == 1.0
    assert spin(1)["channel"] == "singlet"
    bse_preconditioner.set_exchange_channel("singlet")
    assert bse_preconditioner.exchange_spin_weight(1) == 2.0
    assert spin(1, spin_triplet="true")["channel"] == "triplet"
    bse_preconditioner.set_exchange_channel("triplet")
    assert bse_preconditioner.exchange_spin_weight(1) == 0.0
    assert bse_preconditioner.exchange_spin_weight(2) == 1.0
    assert spin(1, local_fields="true")["include_w"] is False
    with pytest.raises(bse_deck.DeckError, match="non-spinor"):
        spin(2, spin_singlet="true")


def test_polarization_is_cartesian_normalized(tmp_path):
    bvec = np.array([[1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.0, 0.0, 2.0]])
    wfn = tmp_path / "WFN.h5"
    with h5py.File(wfn, "w") as f:
        f["mf_header/crystal/bvec"] = bvec
    d, labels = polarization_directions(None, str(wfn))
    assert labels == ("b1", "b2", "b3") and np.allclose(d, np.eye(3))
    d, labels = polarization_directions([1, 1, 0], str(wfn))
    e = bvec[0] + bvec[1]
    assert labels == ("e",) and np.allclose(d[0], e / np.linalg.norm(e))
    assert np.isclose(np.linalg.norm(d[0]), 1.0)


def test_kgrid_fine_shift_follows_kgrid_x():
    k = uniform_kgrid_frac((4, 3, 2), (0.47, 0.37, 0.31))
    assert np.allclose(k[0], [0.47 / 4, 0.37 / 3, 0.31 / 2])
    half = uniform_kgrid_frac((3, 3, 3), (0.5, 0.5, 0.5))
    fine = uniform_kgrid_frac((6, 6, 6), (0.0, 0.0, 0.0)).reshape(6, 6, 6, 3)[1::2, 1::2, 1::2]
    assert np.allclose(half, fine.reshape(-1, 3))
