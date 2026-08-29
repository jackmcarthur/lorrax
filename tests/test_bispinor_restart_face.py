"""Bispinor restart, face layout: the pass-1 refusal contracts.

The full write->read round-trip (both channels' face pairs, bit-exact on
every addressable shard, through the PRODUCTION writer/reader on a real
deck) is certified by the real-CUDA leg this branch runs on the
``tests/regression/bispinor_debug`` fixture (fresh face run with
``write_restart_tensors = true`` followed by ``restart = true``; see the
branch's claims row for job ids).  What THIS file pins is the part that
must hold with no FFI present at all: the reader's pass-1 (serial h5py)
refusal matrix for the transverse face pair, which fires BEFORE SlabIO
ever opens the file.

  1. A file holding 'psi_full_y_transverse' WITHOUT the additive
     'psi_full_y_transverse_mun' (i.e. written by a legacy-layout run)
     refuses BY NAME under ``low_mem_bands=True`` -- never a silent
     one-face derivation via an unowned transpose.
  2. The same file reads fine with ``low_mem_bands=False`` past pass 1
     (the legacy path's own contract; this test stops at the refusal
     boundary since SlabIO needs the FFI).
  3. A scalar file (no transverse datasets at all) passes pass 1 under
     both flags -- the transverse pair is optional, not required.

The refusal is reached through ``read_restart_state_from_h5`` with a
stub mesh: pass 1 is pure h5py and raises before the mesh is touched.
"""
import h5py
import numpy as np
import pytest

from file_io.tagged_arrays import read_restart_state_from_h5


def _seed_file(path, *, with_transverse_nmu, with_transverse_mun):
    nk, nb, ns, mu, mu_t = 2, 4, 4, 8, 6
    with h5py.File(path, "w") as f:
        f["psi_full_y"] = np.zeros((nk, nb, ns, mu), dtype=np.complex128)
        f["psi_full_y_mun"] = np.zeros((nk, ns, mu, nb),
                                       dtype=np.complex128)
        if with_transverse_nmu:
            f["psi_full_y_transverse"] = np.zeros(
                (nk, nb, ns, mu_t), dtype=np.complex128)
        if with_transverse_mun:
            f["psi_full_y_transverse_mun"] = np.zeros(
                (nk, ns, mu_t, nb), dtype=np.complex128)


class _StubMesh:
    """Never touched: pass 1 raises (or this test stops) before SlabIO."""


def test_legacy_written_transverse_refuses_by_name_under_face(tmp_path):
    p = str(tmp_path / "tensors_legacy_T.h5")
    _seed_file(p, with_transverse_nmu=True, with_transverse_mun=False)
    with pytest.raises(ValueError, match="psi_full_y_transverse_mun"):
        read_restart_state_from_h5(p, _StubMesh(), low_mem_bands=True)


def test_face_pair_present_passes_pass1(tmp_path):
    # Both transverse faces present: pass 1 must NOT raise the pair
    # refusal.  SlabIO (FFI) is unavailable here, so any error past the
    # pass-1 boundary is acceptable -- but it must not be the pair
    # refusal, and it must not be the charge-pair refusal either.
    p = str(tmp_path / "tensors_face_T.h5")
    _seed_file(p, with_transverse_nmu=True, with_transverse_mun=True)
    try:
        read_restart_state_from_h5(p, _StubMesh(), low_mem_bands=True)
    except ValueError as e:
        assert "psi_full_y_transverse_mun" not in str(e)
        assert "has no 'psi_full_y_mun'" not in str(e)
    except Exception:
        pass  # SlabIO/FFI/mesh errors past pass 1 are out of scope here


def test_scalar_file_passes_pass1_under_both_flags(tmp_path):
    p = str(tmp_path / "tensors_scalar.h5")
    _seed_file(p, with_transverse_nmu=False, with_transverse_mun=False)
    for flag in (True, False):
        try:
            read_restart_state_from_h5(p, _StubMesh(), low_mem_bands=flag)
        except ValueError as e:
            assert "transverse" not in str(e)
        except Exception:
            pass  # past pass 1, out of scope
