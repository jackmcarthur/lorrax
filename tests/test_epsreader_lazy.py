"""``EPSReader`` reads slices, not the whole ``mats/matrix``.

THE DEFECT.  The constructor used to do ``self.matrix =
self._file['mats/matrix'][:]`` — every q, every frequency, the whole
``nmtx_max x nmtx_max`` block, ON EVERY RANK, before the caller had asked
for anything.  Its one in-tree consumer,
``gw.head_correction.resolve_head_sample``'s ``epshead`` branch, then used
six numbers out of it.  At a production ``epsmat.h5`` that is tens of GB
per rank for a complex scalar, and it is the ``file_io/epsreader.py:67-80``
row of the defect register.

WHY THE FIXTURE IS BIGGER THAN THE SLICE, and why that is the whole test.
A reader that copies everything and a reader that slices return the SAME
NUMBERS.  Byte-equality therefore discriminates nothing, which is why the
old behaviour survived a rewrite of the surrounding module.  What DOES
discriminate is whether the object holds a resident array: build a file
whose ``mats/matrix`` is materially larger than any slice taken, then
assert (a) the attribute is an h5py handle and not an ``ndarray``, and (b)
the accessors still return exactly what a full read would have.

No jax, no GPU, no FFI.  ``eps0mat.h5`` is a BerkeleyGW artifact and this
reader is serial h5py by design — see the module docstring for why that is
not the one-owner hazard.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

# LOADED BY PATH, not as ``file_io.epsreader``.  ``file_io/__init__.py``
# re-exports the whole shelf, which pulls ``wfn_loader`` and therefore jax;
# this reader imports h5py and numpy and nothing else, so importing it
# through the package would make an h5py-only test need a GPU module.  The
# file under test is the same file either way, and the indirection also
# pins that ``epsreader`` acquires no heavy import of its own.
_EPSREADER_PY = (pathlib.Path(__file__).resolve().parents[1]
                 / "src" / "file_io" / "epsreader.py")
_spec = importlib.util.spec_from_file_location("_epsreader_ut", _EPSREADER_PY)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EPSReader = _mod.EPSReader


#: Deliberately non-square in every axis that has a meaning, so a
#: transposed or mis-ordered slice cannot pass by symmetry.
_NQ, _NMATRIX, _NFREQ, _NMTX_MAX = 3, 2, 4, 9
#: Every q keeps a DIFFERENT number of G, and none of them is nmtx_max —
#: the ragged case is the one the accessors' ``:nmtx_q`` bound is for.
_NMTX = np.array([5, 7, 4], dtype=np.int32)


def _write_fixture(path):
    rng = np.random.default_rng(20260822)
    matrix = rng.standard_normal(
        (_NQ, _NMATRIX, _NFREQ, _NMTX_MAX, _NMTX_MAX, 2))
    diagonal = rng.standard_normal((2, _NMTX_MAX, _NQ))
    with h5py.File(path, "w") as f:
        p = f.create_group("eps_header/params")
        p["matrix_type"] = np.int32(1)
        p["has_advanced"] = np.int32(0)
        p["nmatrix"] = np.int32(_NMATRIX)
        p["matrix_flavor"] = np.int32(2)
        p["icutv"] = np.int32(0)
        p["ecuts"] = np.float64(10.0)
        p["nband"] = np.int32(8)
        p["efermi"] = np.float64(0.0)
        f["eps_header/versionnumber"] = np.int32(3)
        f["eps_header/flavor"] = np.int32(2)
        q = f.create_group("eps_header/qpoints")
        q["nq"] = np.int32(_NQ)
        q["qpts"] = np.zeros((_NQ, 3))
        q["qgrid"] = np.array([1, 1, 1], dtype=np.int32)
        q["qpt_done"] = np.ones(_NQ, dtype=np.int32)
        fr = f.create_group("eps_header/freqs")
        fr["freq_dep"] = np.int32(0)
        fr["nfreq"] = np.int32(_NFREQ)
        fr["nfreq_imag"] = np.int32(0)
        fr["freqs"] = np.zeros((_NFREQ, 2))
        g = f.create_group("eps_header/gspace")
        g["nmtx"] = _NMTX
        g["nmtx_max"] = np.int32(_NMTX_MAX)
        g["ekin"] = np.zeros(_NMTX_MAX)
        # +1: the reader subtracts one for Fortran indexing.
        g["gind_eps2rho"] = np.arange(1, _NMTX_MAX + 1, dtype=np.int32)
        g["gind_rho2eps"] = np.arange(1, _NMTX_MAX + 1, dtype=np.int32)
        g["vcoul"] = np.ones(_NMTX_MAX)
        f["mf_header/gspace/components"] = np.zeros(
            (_NMTX_MAX, 3), dtype=np.int32)
        f["mats/matrix"] = matrix
        f["mats/matrix-diagonal"] = diagonal
    return matrix, diagonal


def test_the_matrix_attributes_are_handles_not_copies(tmp_path):
    path = tmp_path / "eps0mat.h5"
    _write_fixture(path)
    # NO ``with`` here, deliberately: this is the one cell that has to fail
    # for the RIGHT reason on a reader that copies.  Going through the
    # context manager would fail on the missing ``__enter__`` first and
    # report a shape of API, not the residency this test is about.
    eps = EPSReader(str(path))
    try:
        assert isinstance(eps.matrix, h5py.Dataset), (
            "EPSReader.matrix is a resident ndarray again — the constructor "
            "is copying mats/matrix on every rank.  It must stay an h5py "
            "handle so the accessors read hyperslabs.")
        assert isinstance(eps.matrix_diagonal, h5py.Dataset)
    finally:
        getattr(eps, "close", lambda: None)()


def test_the_accessors_return_what_a_full_read_would_have(tmp_path):
    """Value parity against the copy this reader no longer makes."""
    path = tmp_path / "eps0mat.h5"
    matrix, diagonal = _write_fixture(path)
    with EPSReader(str(path)) as eps:
        for iq in range(_NQ):
            n = int(_NMTX[iq])
            for imat in range(_NMATRIX):
                for ifreq in range(_NFREQ):
                    got = eps.get_eps_matrix(iq, ifreq=ifreq, imatrix=imat)
                    want = (matrix[iq, imat, ifreq, :n, :n, 0]
                            + 1j * matrix[iq, imat, ifreq, :n, :n, 1])
                    assert got.shape == (n, n)
                    assert np.array_equal(got, want)

            minus = eps.get_eps_minus_delta_matrix(iq)
            want = (matrix[iq, 0, 0, :n, :n, 0]
                    + 1j * matrix[iq, 0, 0, :n, :n, 1])
            want = want - np.eye(n)
            assert np.allclose(minus, want, rtol=0, atol=0)

            diag = eps.get_eps_diagonal(iq)
            assert np.array_equal(
                diag, diagonal[0, :n, iq] + 1j * diagonal[1, :n, iq])


def test_epshead_is_the_q0_head(tmp_path):
    path = tmp_path / "eps0mat.h5"
    matrix, _ = _write_fixture(path)
    with EPSReader(str(path)) as eps:
        assert eps.epshead == complex(
            matrix[0, 0, 0, 0, 0, 0], matrix[0, 0, 0, 0, 0, 1])


def test_close_is_idempotent_and_ends_the_handles(tmp_path):
    """The lifetime contract the handles create, stated as a test.

    A caller that closes and then slices must get an error, not stale
    memory — that is the one behaviour change the laziness costs, and it
    is worth pinning because it is the thing a future caller will trip on.
    """
    path = tmp_path / "eps0mat.h5"
    _write_fixture(path)
    eps = EPSReader(str(path))
    head = eps.epshead                       # read in the constructor
    eps.close()
    eps.close()                              # idempotent
    assert head == eps.epshead               # a plain scalar, still valid
    with pytest.raises(Exception):
        eps.get_eps_matrix(0)
