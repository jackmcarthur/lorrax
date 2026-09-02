"""``run_nscf``'s pseudobands-only WFN.h5 read, gated at nk>1.

The read this file gates used to be a hand-rolled ``h5py`` block in
``psp.run_nscf`` that carried a comment claiming ``wfns/coeffs`` was
``(nbnd, nspinor, ngk_max, 2)``.  The repo's own writer says otherwise:
:class:`file_io.wfn_writer.WFNWriter` allocates
``shape=(nbands, nspinor, ngktot, 2)`` with ``ngktot = ngk.sum()`` and
``write_k`` writes the slice ``[:, :, off:off+ngk[ik], :]``.  The G axis is
CONCATENATED over k.  So ``coeffs[ib]`` is every k-point's ψ end to end,
and the old block poured it into k-slot 0 of an ``ngkmax``-wide buffer.

Because per-k ``ngk`` is within a few percent of ``ngkmax``, Σ_k ngk[k]
always overruns that buffer for nk>1 — the failure is a broadcast
``ValueError``, not a quiet wrong answer.  That failure is the red twin
below (:func:`_read_the_old_way`, kept verbatim), and it is what makes the
green half of this file mean something.  On the 4-k fixture here, where
``ngk = [57, 44, 44, 48]`` and the grid's ``ngkmax`` is 57, it reads::

    ValueError: could not broadcast input array from shape (1,193)
                into shape (1,57)

The two gates:

* **nk>1 round-trip.**  A synthetic 4-k WFN.h5 written through the repo's
  own ``WFNWriter``, read back through the new service path, must return
  ψ bit-identical to what was written, per k, with the columns past
  ``ngk[ik]`` exactly zero.  The old path raises on the same file.
* **nk=1 bit-identity.**  On a single-k file the two paths must agree
  bit-for-bit, because the whole point of the fix is that nothing about
  the case that used to work changes.

Nothing here needs a GPU, a mesh, or a QE ``.save``: the crystal is a
namespace with the dozen fields the writer and the G-sphere builders read.

Gate 3 (2026-08-28) rides on the same fixtures: the writer's ``n_occ`` /
``ifmax`` must be an occupied-BAND count for BOTH crystal shapes it is
handed (QE ``CrystalData``: nelec = electron count; ``WfnLoader`` via
``qp_wfn``: nelec = band count, ``num_electrons`` = electron count).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from file_io import WFNWriter
from psp.gvec_utils import build_master_gvec_list, select_gvecs_for_k
from psp.run_nscf import _load_deterministic_bands, _setup_kgrid

# The loader runs ``symmetry_maps``'s density-invariant check on every file
# it opens, and these ψ are Gaussian noise rather than eigenstates, so ∫ρ is
# meaningless by construction.  The check is doing its job; silencing it here
# keeps a genuine warning from being trained away elsewhere in the suite.
pytestmark = pytest.mark.filterwarnings(
    "ignore:WFN density symmetry check FAILED:RuntimeWarning")

ALAT = 10.0                 # bohr, simple cubic
ECUTWFC = 2.0               # Ry — gives a ~57-G sphere on an 8³ box
NBANDS = 6
NSPINOR = 1


def _cubic_crystal():
    """A simple-cubic stand-in carrying only what the writer + G-builders read."""
    avec = np.eye(3, dtype=np.float64)                     # units of alat
    bvec = np.eye(3, dtype=np.float64)                     # units of blat
    blat = 2.0 * np.pi / ALAT
    bdot = (blat ** 2) * np.eye(3, dtype=np.float64)
    mtrx = np.zeros((48, 3, 3), dtype=np.int32)
    mtrx[0] = np.eye(3, dtype=np.int32)
    return SimpleNamespace(
        nspin=1, nspinor=NSPINOR, nelec=4, nat=1,
        ecutwfc=ECUTWFC, ecutrho=4.0 * ECUTWFC,
        fft_grid=(8, 8, 8),
        alat=ALAT, blat=blat, cell_volume=ALAT ** 3,
        avec=avec, bvec=bvec, bdot=bdot,
        atom_crys=np.zeros((1, 3), dtype=np.float64),
        atom_types=np.array([1], dtype=np.int32),
        ntran=1, sym_matrices=mtrx,
        translations=np.zeros((48, 3), dtype=np.float64),
        assume_isolated=None,
    )


def _write_synthetic_wfn(path, kpoints):
    """Write a WFN.h5 with ragged per-k G-spheres through the REPO's writer.

    Returns ``(gvecs_per_k, eigenvalues, coeffs_per_k)`` — the ground truth
    the round-trip is compared against.
    """
    crystal = _cubic_crystal()
    kpoints = np.asarray(kpoints, dtype=np.float64)
    weights = np.full((kpoints.shape[0],), 1.0 / kpoints.shape[0])

    G_master, _ = build_master_gvec_list(crystal)
    gvecs_per_k = [select_gvecs_for_k(kpoints[ik], G_master, crystal.bdot,
                                      crystal.ecutwfc)[0]
                   for ik in range(kpoints.shape[0])]

    rng = np.random.default_rng(20260808)
    eigenvalues = np.sort(rng.uniform(-1.0, 3.0,
                                      size=(kpoints.shape[0], NBANDS)), axis=1)
    coeffs_per_k = []
    writer = WFNWriter(str(path), crystal, kpoints, weights, (2, 2, 1),
                       NBANDS, gvecs_per_k, nosym=True)
    for ik, gk in enumerate(gvecs_per_k):
        ng_k = gk.shape[0]
        c = (rng.standard_normal((NBANDS, NSPINOR, ng_k))
             + 1j * rng.standard_normal((NBANDS, NSPINOR, ng_k)))
        coeffs_per_k.append(c)
        writer.write_k(ik, eigenvalues[ik], c)
    writer.close()
    return gvecs_per_k, eigenvalues, coeffs_per_k


def _read_the_old_way(path, nspinor, ngkmax):
    """``run_nscf.py:307-330`` at 22d99b5e, verbatim.  THE RED TWIN.

    Kept as-is (including the comment that is wrong about the layout) so
    the ``ValueError`` it raises at nk>1 is this repository's own text and
    not a paraphrase.
    """
    import h5py

    with h5py.File(str(path), "r") as f:
        nk_file = int(f["mf_header/kpoints/nrk"][()])
        nbnd_file = int(f["mf_header/kpoints/mnband"][()])

        # Read wavefunction coefficients
        coeffs = f["wfns/coeffs"]  # (nbnd, nspinor, ngk_max, 2)
        all_evecs = np.zeros((nk_file, nbnd_file, nspinor, ngkmax),
                             dtype=np.complex128)
        for ib in range(nbnd_file):
            c = coeffs[ib]  # (nspinor, ngk, 2)
            c_complex = c[:, :, 0] + 1j * c[:, :, 1]
            # Pad to ngkmax if needed
            ngk_file = c_complex.shape[1]
            all_evecs[0, ib, :, :ngk_file] = c_complex
    return all_evecs


def _grid_ngkmax(kpoints):
    """The ``ngkmax`` ``_setup_kgrid`` derives for these k-points."""
    _, _, ngkmax, _ = _setup_kgrid(_cubic_crystal(), (2, 2, 1), True,
                                   np.asarray(kpoints, dtype=np.float64),
                                   np.full((len(kpoints),), 1.0 / len(kpoints)),
                                   False)
    return int(ngkmax)


MULTI_K = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0],
           [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]]


# ---------------------------------------------------------------------------
# Gate 1 — nk>1: the old path dies, the new one round-trips
# ---------------------------------------------------------------------------

def test_the_old_hand_rolled_read_dies_at_nk_gt_1(tmp_path):
    """RED TWIN.  Σ_k ngk[k] does not fit an ngkmax-wide k-slot."""
    path = tmp_path / "WFN_multik.h5"
    gvecs_per_k, _, _ = _write_synthetic_wfn(path, MULTI_K)
    ngk = np.array([g.shape[0] for g in gvecs_per_k])
    assert len(MULTI_K) > 1 and int(ngk.sum()) > int(ngk.max()), (
        "fixture must actually have a concatenated G axis")

    with pytest.raises(ValueError) as excinfo:
        _read_the_old_way(path, NSPINOR, _grid_ngkmax(MULTI_K))
    assert "shape" in str(excinfo.value) or "broadcast" in str(excinfo.value)


def test_the_service_read_round_trips_a_multi_k_wfn(tmp_path):
    path = tmp_path / "WFN_multik.h5"
    gvecs_per_k, eigenvalues, coeffs_per_k = _write_synthetic_wfn(path, MULTI_K)

    (nbnd, nk, kpoints, weights, ngkmax, gvecs_read,
     evals_read, all_evecs) = _load_deterministic_bands(
        str(path), _cubic_crystal(), (2, 2, 1), True, NSPINOR, False)

    assert (nbnd, nk) == (NBANDS, len(MULTI_K))
    assert np.array_equal(kpoints, np.asarray(MULTI_K, dtype=np.float64))
    assert np.allclose(weights, 1.0 / len(MULTI_K))
    assert np.array_equal(evals_read, eigenvalues)
    assert all_evecs.shape == (len(MULTI_K), NBANDS, NSPINOR, ngkmax)

    for ik, gk in enumerate(gvecs_per_k):
        ng_k = gk.shape[0]
        assert np.array_equal(gvecs_read[ik], gk)
        assert np.array_equal(all_evecs[ik, :, :, :ng_k], coeffs_per_k[ik]), (
            f"ψ at k={ik} is not what the writer wrote")
        # ngk_valid is not decoration: past the sphere there is nothing.
        assert not np.any(all_evecs[ik, :, :, ng_k:]), (
            f"pad columns at k={ik} are not zero")

    # The k-slots are DISTINCT — the defect's signature was every k but 0
    # staying zero, which an all-zeros ψ would have hidden.
    for ik in range(1, len(MULTI_K)):
        assert np.any(all_evecs[ik]), f"k={ik} came back empty"


# ---------------------------------------------------------------------------
# Gate 2 — nk=1: nothing that used to work changes
# ---------------------------------------------------------------------------

def test_the_single_k_path_is_bit_identical(tmp_path):
    single = [[0.0, 0.0, 0.0]]
    path = tmp_path / "WFN_gamma.h5"
    _write_synthetic_wfn(path, single)

    ngkmax = _grid_ngkmax(single)
    old = _read_the_old_way(path, NSPINOR, ngkmax)
    new = _load_deterministic_bands(
        str(path), _cubic_crystal(), (2, 2, 1), True, NSPINOR, False)[7]

    assert new.shape == old.shape
    assert np.array_equal(new, old), (
        "the nk=1 case must be bit-for-bit what the hand-rolled block gave")


# ---------------------------------------------------------------------------
# The reconciliation refuses rather than reading against the wrong sphere
# ---------------------------------------------------------------------------

def test_a_mismatched_g_sphere_is_refused(tmp_path):
    """A file whose ngk disagrees with this run's ecutwfc is not readable."""
    path = tmp_path / "WFN_multik.h5"
    _write_synthetic_wfn(path, MULTI_K)

    crystal = _cubic_crystal()
    crystal.ecutwfc = 0.5 * ECUTWFC          # smaller sphere than the file's
    with pytest.raises(ValueError, match="disagrees with the NSCF G-sphere"):
        _load_deterministic_bands(str(path), crystal, (2, 2, 1), True,
                                  NSPINOR, False)


# ---------------------------------------------------------------------------
# Gate 3 — the writer's n_occ is an occupied-BAND count for BOTH input shapes
# ---------------------------------------------------------------------------
# ``WFNWriter`` receives two crystal shapes whose ``nelec`` MEAN DIFFERENT
# THINGS: a QE ``CrystalData`` (nelec = physical electron count, no
# ``num_electrons`` attr) and, via ``file_io.qp_wfn``, a ``WfnLoader``
# (nelec = max(ifmax), already an occupied-band count; ``num_electrons`` =
# the physical count).  n_occ = electrons·nspin·nspinor/2 converts the
# physical count to bands in both conventions; halving a loader's nelec
# AGAIN was the qp_wfn double-halving defect at nspinor=1.


def _ns_with(base, **over):
    """A copy of a SimpleNamespace crystal with fields overridden."""
    return SimpleNamespace(**{**vars(base), **over})


def _written_ifmax_and_occ(tmp_path, crystal, fname):
    """Write a header-only Γ WFN through the writer; return (ifmax, occ[k=0])."""
    import h5py

    path = tmp_path / fname
    kpoints = np.zeros((1, 3), dtype=np.float64)
    G_master, _ = build_master_gvec_list(crystal)
    gk = select_gvecs_for_k(kpoints[0], G_master, crystal.bdot,
                            crystal.ecutwfc)[0]
    w = WFNWriter(str(path), crystal, kpoints, np.ones(1), (1, 1, 1),
                  NBANDS, [gk], nosym=True)
    w.close()
    with h5py.File(str(path), "r") as f:
        return (int(f["mf_header/kpoints/ifmax"][0, 0]),
                np.asarray(f["mf_header/kpoints/occ"][0, 0, :]))


def test_crystal_shaped_nelec_keeps_todays_ifmax_at_both_spinor_counts(tmp_path):
    """The QE-CrystalData arithmetic is PINNED: 4 e- → 2 bands scalar, 4 FR."""
    for nspinor, want in ((1, 2), (2, 4)):
        crystal = _ns_with(_cubic_crystal(), nspinor=nspinor)      # nelec=4 e-
        ifmax, occ = _written_ifmax_and_occ(
            tmp_path, crystal, f"WFN_crystal_ns{nspinor}.h5")
        assert ifmax == want, (
            f"nspinor={nspinor}: ifmax={ifmax}, want {want} occupied bands "
            f"for {crystal.nelec} electrons")
        assert np.array_equal(
            occ, [1.0] * want + [0.0] * (NBANDS - want)), (
            f"nspinor={nspinor}: occ row disagrees with ifmax={want}")


def test_loader_shaped_input_is_not_double_halved_at_nspinor1(tmp_path):
    """THE BROKEN CASE.  A WfnLoader's nelec is already a band count.

    4 electrons at nspinor=1 fill 2 bands, so the loader-shaped stub has
    nelec=2 (= max(ifmax)) and num_electrons=4.0.  The pre-fix expression
    ``int(crystal.nelec) // 2`` halved the band count AGAIN → ifmax=1; the
    precondition assert keeps this cell from ever passing vacuously.
    """
    crystal = _ns_with(_cubic_crystal(), nspinor=1,
                       nelec=2, num_electrons=4.0)
    assert int(crystal.nelec) // 2 != 2, (
        "fixture numbers no longer discriminate: the pre-fix arithmetic "
        "agrees with the expected value, so this cell proves nothing")
    ifmax, occ = _written_ifmax_and_occ(tmp_path, crystal, "WFN_loader_ns1.h5")
    assert ifmax == 2, (
        f"ifmax={ifmax}: the loader-shaped nspinor=1 input was re-halved "
        f"(want 2 = the band count the loader already computed)")
    assert np.array_equal(occ, [1.0, 1.0] + [0.0] * (NBANDS - 2))


def test_loader_shaped_input_at_nspinor2_is_unchanged(tmp_path):
    """nspinor=2 bit-identity: 4 e- → nelec=4 spinor bands → ifmax=4."""
    crystal = _ns_with(_cubic_crystal(), nspinor=2,
                       nelec=4, num_electrons=4.0)
    ifmax, occ = _written_ifmax_and_occ(tmp_path, crystal, "WFN_loader_ns2.h5")
    assert ifmax == 4
    assert np.array_equal(occ, [1.0] * 4 + [0.0] * (NBANDS - 4))
