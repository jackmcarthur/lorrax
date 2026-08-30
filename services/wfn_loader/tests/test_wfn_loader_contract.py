"""Layer L-a: the ``wfn_loader`` door's contract, in one process.

WHAT MOVED HERE.  The twelve collected cells of
``tests/test_wfn_loader_eager.py`` — the padding contract, the band
iterator, the bispinor lift, the four backend-resolution doors and the
phdf5 unfold kernel's single-process parity — arrived VERBATIM with the
extraction (SERVICE_FORM: a service owns its own suite).  That file is now
a pointer with zero cells; see its docstring for the census.

WHAT IS NEW HERE, and why each gap was worth closing (survey
w1_wfn_loader, §2 and §8):

* ``adopt_mesh``'s FOUR narrowing conditions.  Survey concept 2 — a
  documented MAY-RAISE with four early returns and, before this file,
  zero executing tests.  Each condition gets a cell that fails if the
  condition is deleted, and the positive arm proves the narrowing did NOT
  fire, so the four negatives are not four tautologies.
* ``bands()``.  One caller in the whole tree and, before this file, one
  cell asserting the CHUNK BOUNDARIES and nothing about the ψ it yields.
* ``load`` / ``load_process_local`` band-range refusals, and the
  no-padding contract that is the entire reason ``load_process_local``
  exists as a second primitive.
* The constructor/env refusal surface, exercised CONSTRUCTIBLY — every
  refusal reached through the public door on a real file, not through a
  hand-built ``__new__`` stub, because a refusal that only fires on a stub
  is a refusal nobody has seen fire.
* The structural cell for the 2026-08-07 fold: the loader must not have
  grown a second copy of the per-rank band clamp.  (The clip's own
  regression cells live next to the clip, in
  ``tests/test_slab_io_hostile_geometry.py`` §4; what belongs HERE is the
  claim that this service does not reimplement it.)

TIERS ABOVE THIS ONE.  A mesh with more than one device is L-b
(``test_wfn_loader_emulated_mesh.py``); four REAL ranks and the phdf5
backend are L-c (``test_wfn_loader_multiproc.py``, whose ``check_*``
bodies are also the cluster legs).

SKIPS.  Cells that need a checked-in deck skip with the reason in
``conftest.NO_DECK``, which names the covering leg — and on a machine
whose profile promises the fixtures, ``test_wfn_loader_skip_honesty.py``
turns that skip into a FAILURE.
"""
from __future__ import annotations

import inspect
import os

import h5py
import numpy as np
import pytest

from wfn_loader import WfnLoader, read_wfn_provenance


# ---------------------------------------------------------------------------
# The synthetic deck (moved verbatim) — a laptop-sized WFN with ntran=1
# ---------------------------------------------------------------------------

def _synth_wfn(tmp_path) -> str:
    """Tiny synthetic WFN with ntran=1 (no real sym) for laptop tests."""
    out = str(tmp_path / "WFN.h5")
    rng = np.random.default_rng(0xC0FFEE)
    nspin, nspinor, nrk, mnband = 1, 2, 2, 6
    ngk = np.array([7, 9], dtype=np.int32)
    ngkmax = int(ngk.max())
    ngktot = int(ngk.sum())
    fft_grid = np.array([8, 8, 8], dtype=np.int32)

    with h5py.File(out, "w") as f:
        g = f.create_group("mf_header")
        g.create_dataset("versionnumber", data=np.int32(3))
        g.create_dataset("flavor", data=np.int32(2))
        kp = g.create_group("kpoints")
        kp.create_dataset("nspin", data=np.int32(nspin))
        kp.create_dataset("nspinor", data=np.int32(nspinor))
        kp.create_dataset("nrk", data=np.int32(nrk))
        kp.create_dataset("mnband", data=np.int32(mnband))
        kp.create_dataset("ngkmax", data=np.int32(ngkmax))
        kp.create_dataset("ecutwfc", data=np.float64(20.0))
        kp.create_dataset("kgrid", data=np.array([1, 1, 2], dtype=np.int32))
        kp.create_dataset("shift", data=np.zeros(3, dtype=np.float64))
        kp.create_dataset("ngk", data=ngk)
        kp.create_dataset("ifmin", data=np.ones((nspin, nrk), dtype=np.int32))
        kp.create_dataset("ifmax", data=np.full((nspin, nrk), 3, dtype=np.int32))
        kp.create_dataset("w", data=np.full(nrk, 1.0 / nrk, dtype=np.float64))
        kp.create_dataset("rk", data=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
                                              dtype=np.float64))
        kp.create_dataset("el", data=rng.random((nspin, nrk, mnband)))
        kp.create_dataset("occ", data=np.ones((nspin, nrk, mnband), dtype=np.float64))
        gs = g.create_group("gspace")
        gs.create_dataset("ng", data=np.int32(100))
        gs.create_dataset("ecutrho", data=np.float64(80.0))
        gs.create_dataset("FFTgrid", data=fft_grid)
        sym = g.create_group("symmetry")
        sym.create_dataset("ntran", data=np.int32(1))
        sym.create_dataset("cell_symmetry", data=np.int32(0))
        sym.create_dataset(
            "mtrx",
            data=np.tile(np.eye(3, dtype=np.int32)[None], (48, 1, 1)))
        sym.create_dataset("tnp", data=np.zeros((48, 3), dtype=np.float64))
        cr = g.create_group("crystal")
        cr.create_dataset("celvol", data=np.float64(100.0))
        cr.create_dataset("recvol", data=np.float64(0.62))
        cr.create_dataset("alat", data=np.float64(10.0))
        cr.create_dataset("blat", data=np.float64(0.628))
        cr.create_dataset("nat", data=np.int32(2))
        cr.create_dataset("avec", data=np.eye(3, dtype=np.float64) * 10.0)
        cr.create_dataset("bvec", data=np.eye(3, dtype=np.float64) * 0.628)
        cr.create_dataset("adot", data=np.eye(3, dtype=np.float64) * 100.0)
        cr.create_dataset("bdot", data=np.eye(3, dtype=np.float64) * 0.39)
        cr.create_dataset("atyp", data=np.array([1, 2], dtype=np.int32))
        cr.create_dataset("apos", data=rng.random((2, 3)))

        # wfns/* — coeffs packed re/im, gvecs concatenated across k.
        wf = f.create_group("wfns")
        # G-vectors: first 7 at k=0, then 9 at k=1.
        gvecs = rng.integers(-3, 4, size=(ngktot, 3)).astype(np.int32)
        wf.create_dataset("gvecs", data=gvecs)
        coeffs = rng.random((mnband, nspinor, ngktot, 2)) * 2 - 1
        wf.create_dataset("coeffs", data=coeffs)
    return out


@pytest.fixture
def synth_wfn_path(tmp_path):
    return _synth_wfn(tmp_path)


def test_physical_density_occupations_keep_signed_tails_and_explicit_k_map(
        tmp_path, monkeypatch):
    path = _synth_wfn(tmp_path)
    occs = np.asarray([[[1.0, 0.8, 0.2, -0.05, 0.0, 0.0],
                        [1.0, 0.6, 0.4, -0.02, 0.01, 0.0]]])
    with h5py.File(path, "r+") as h5:
        h5["mf_header/kpoints/occ"][...] = occs
    monkeypatch.setenv("LORRAX_TRS_CHECK", "0")
    loader = WfnLoader(path)
    assert not loader.occupations_are_exact_integer
    assert loader.physical_density_band_stop == 5
    assert loader.occupation_state_capacity == 1.0
    assert loader.num_electrons == pytest.approx(
        0.5 * float(np.sum(occs[0, 0]) + np.sum(occs[0, 1])))

    from types import SimpleNamespace
    loader._sym = SimpleNamespace(
        nk_tot=4, irr_idx_k=np.asarray([1, 0, 1, 0], dtype=np.int32))
    got = loader.physical_density_occupations(
        k="full_bz", unit_as_none=True)
    assert np.array_equal(got, occs[0, [1, 0, 1, 0], :5])
    assert float(got[:, 3:].min()) < 0.0
    loader.close()


def test_path_authentication_uses_header_view_not_full_loader(
        tmp_path, monkeypatch):
    """Post-hoc provenance must not initialize G/psi or the FFT diagnostic."""
    from common.parallel_transport import (
        WFN_FINGERPRINT_SCHEME, wfn_fingerprint)
    from file_io.kin_ion import authenticate_kin_ion_hartree_wfn_receipt

    wfn_path = _synth_wfn(tmp_path)
    occs = np.asarray([[[1.0, 0.8, 0.2, -0.05, 0.0, 0.0],
                        [1.0, 0.6, 0.4, -0.02, 0.01, 0.0]]])
    with h5py.File(wfn_path, "r+") as h5:
        h5["mf_header/kpoints/occ"][...] = occs

    provenance = read_wfn_provenance(wfn_path)
    assert provenance.physical_density_band_stop == 5
    assert not provenance.occupations_are_exact_integer
    fingerprint = wfn_fingerprint(provenance)

    kin_path = str(tmp_path / "kin_ion.h5")
    with h5py.File(kin_path, "w") as h5:
        ds = h5.create_dataset(
            "kin_ion", data=np.zeros((2, 6, 6), dtype=np.complex128))
        ds.attrs["bispinor"] = False
        ds.attrs["wfn_fingerprint_scheme"] = WFN_FINGERPRINT_SCHEME
        ds.attrs["wfn_fingerprint"] = fingerprint

    def _forbid_full_loader(*_args, **_kwargs):
        raise AssertionError("post-hoc authentication constructed WfnLoader")

    monkeypatch.setattr(WfnLoader, "__init__", _forbid_full_loader)
    monkeypatch.setattr(
        WfnLoader, "_run_density_symmetry_check", _forbid_full_loader)
    attrs = authenticate_kin_ion_hartree_wfn_receipt(
        kin_path, wfn_path, selected_hartree_source="isdf",
        band_stop=6)
    assert attrs["wfn_fingerprint"] == fingerprint


#: The ``/pscratch`` deck the moved cells used to name.  Its machine is
#: gone; the survey (§6.4) established the in-repo ``gnppm`` fixture is the
#: SAME FILE (byte-size identical, and its header re-read on Perlmutter:
#: nrk 9, mnband 82, nspinor 2, ngkmax 1963, ntran 2).  Kept as an explicit
#: override so a machine that still has it can measure against it.
_MOS2_WFN = os.environ.get(
    "LORRAX_WFN_TEST_MOS2",
    "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/"
    "qe/nscf/WFN.h5")


@pytest.fixture(params=["synth", "real"])
def wfn_path(request, tmp_path, gnppm_wfn):
    """The moved ``wfn_path`` fixture, with its dead arm repointed.

    THE ONE DELIBERATE CHANGE to a moved cell's inputs, and it is a
    coverage INCREASE, declared rather than smuggled.  The ``mos2`` param
    resolved a ``/pscratch`` path on a machine that no longer exists, so
    on every machine anybody runs today it produced ``skip: MoS2 3x3 WFN
    not present`` — a param that skips everywhere is evaporated coverage,
    and this suite's own gate would call that skip a failure on Perlmutter.
    The in-repo ``gnppm`` deck IS that file (survey §6.4), so the arm now
    points there and RUNS, with ``LORRAX_WFN_TEST_MOS2`` kept for the
    machine that still has the original.
    """
    if request.param == "real":
        return _MOS2_WFN if os.path.exists(_MOS2_WFN) else gnppm_wfn
    return _synth_wfn(tmp_path)


def _mesh_1x1():
    import jax
    from jax.sharding import Mesh
    devs = np.asarray(jax.devices()[:1]).reshape(1, 1)
    return Mesh(devs, ("x", "y"))


# ===========================================================================
#  MOVED VERBATIM from tests/test_wfn_loader_eager.py
# ===========================================================================
# The earlier bit-equality tests compared the loader to legacy classes
# (``WFNReader.get_cnk_batch`` / ``SymMaps.get_cnk_fullzone[_batch]`` /
# ``get_gvecs_kfull``) that no longer exist.  After P5 the loader is the
# contract; the parity surface that survived is
# ``test_bispinor_lift_matches_legacy`` below, the L-c backend parity
# (``test_wfn_loader_multiproc.py``), and the MoS2 3x3 xonly GW smoke.
# ---------------------------------------------------------------------------
# Padding contract
# ---------------------------------------------------------------------------

def test_band_pad_rows_are_zero(synth_wfn_path):
    with WfnLoader(synth_wfn_path) as loader:
        psi = np.asarray(loader.load(bands=(0, 3), k="ibz"))
        # No mesh → no pad; nb_padded == nb_logical == 3.
        assert psi.shape[1] == 3


def test_g_pad_columns_are_zero(synth_wfn_path):
    """Past ngk_valid[k], the ψ slab is zero."""
    with WfnLoader(synth_wfn_path) as loader:
        psi = np.asarray(loader.load(bands=(0, 3), k="ibz"))
        ngk_valid = loader.ngk_valid(k="ibz")
        for j in range(psi.shape[0]):
            n = int(ngk_valid[j])
            assert np.all(psi[j, :, :, n:] == 0), f"k={j} not zero past ngk={n}"


def test_iterator_chunks_band_axis(synth_wfn_path):
    with WfnLoader(synth_wfn_path) as loader:
        b_hi = 6
        chunks = list(loader.bands(0, b_hi, chunk=4, k="ibz"))
        # 6 bands at chunk=4 → (0,4) and (4,6).
        assert [bc for bc, _ in chunks] == [(0, 4), (4, 6)]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------

def test_bispinor_lift_uses_cartesian_momentum_with_blat(wfn_path):
    """The production lift is ``(α/2) σ·[(k+G) @ (blat*bvec)] L``.

    The reference is written independently in NumPy and every arm has
    ``blat != 1``.  Omitting ``blat`` therefore fails this cell instead of
    agreeing with a second copy of the same defect.
    """
    from common.bispinor_init import HALFALPHA

    with WfnLoader(wfn_path) as loader:
        if int(loader.nspinor) != 2:
            pytest.skip("test requires 2-spinor WFN")
        b_hi = min(4, int(loader.nbands))
        # Use full-BZ unfold path so the gvecs and kvecs go through the
        # symmetry-unfolded full-BZ tables (the harder case).
        psi_2 = np.asarray(loader.load(bands=(0, b_hi), k="full_bz"))
        psi_4 = np.asarray(loader.load(bands=(0, b_hi), k="full_bz",
                                        bispinor=True))

        gvecs_full = loader.gvecs(k="full_bz")
        ngk_v = loader.ngk_valid(k="full_bz")
        sym = loader._ensure_sym()
        unfolded_kpts = np.asarray(sym.unfolded_kpts, dtype=np.float64)
        bvec_cart_bohr = (
            float(loader.blat) * np.asarray(loader.bvec, dtype=np.float64))
        n_k = psi_2.shape[0]

        for nk in range(n_k):
            n = int(ngk_v[nk])
            gvecs_k = gvecs_full[nk, :n].astype(np.float64)
            psi_L = psi_2[nk, :, :, :n]                          # (nb, 2, n)
            p = (gvecs_k + unfolded_kpts[nk][None, :]) @ bvec_cart_bohr
            px, py, pz = p.T
            psi_S_ref = np.empty_like(psi_L)
            psi_S_ref[:, 0, :] = HALFALPHA * (
                pz[None, :] * psi_L[:, 0, :]
                + (px - 1j * py)[None, :] * psi_L[:, 1, :])
            psi_S_ref[:, 1, :] = HALFALPHA * (
                (px + 1j * py)[None, :] * psi_L[:, 0, :]
                - pz[None, :] * psi_L[:, 1, :])
            np.testing.assert_allclose(
                psi_4[nk, :, 2:4, :n], psi_S_ref,
                atol=1e-13, rtol=0,
                err_msg=f"nk={nk}")
            if np.linalg.norm(psi_S_ref) > 0.0:
                wrong_without_blat = psi_S_ref / float(loader.blat)
                assert not np.allclose(
                    psi_4[nk, :, 2:4, :n], wrong_without_blat,
                    atol=1e-13, rtol=1e-12), (
                        f"nk={nk}: test failed to discriminate omitted blat")
            # Upper components preserved.
            np.testing.assert_array_equal(
                psi_4[nk, :, 0:2, :], psi_2[nk, :, 0:2, :],
                err_msg=f"upper components nk={nk}")
            # Pad columns beyond ngk: small components are zero too.
            assert np.all(psi_4[nk, :, 2:4, n:] == 0)


def test_phdf5_backend_requires_mesh(synth_wfn_path):
    with pytest.raises(ValueError, match="requires a Mesh"):
        WfnLoader(synth_wfn_path, backend="phdf5")


def test_auto_backend_resolves_to_eager_on_single_process(synth_wfn_path):
    # Single-rank pytest: even with a mesh, auto should pick eager.
    with WfnLoader(synth_wfn_path, backend="auto") as loader:
        assert loader.backend == "eager"


# ===========================================================================
#  The three publics DESIGN DECISION 3 promoted (step 3, the replumb's door).
#
#  Each of ``path`` / ``symmetry()`` / ``kpt_starts`` replaced a PRIVATE that
#  consumers outside this service were already reading (survey §2.2: nine
#  load-bearing privates).  A promotion with no cell is a rename nobody has
#  watched work, so each cell below states the claim in a form that FAILS if
#  the promotion is deleted, mis-wired, or made to return a placeholder.
# ===========================================================================

def test_path_property_is_the_constructor_argument(tmp_path, synth_wfn_path):
    """``path`` == what was passed in, normalised to ``str``.

    Falsifiable three ways: against the literal argument, against a
    ``pathlib.Path`` argument (the property must not leak a Path object —
    ``qp_wfn``/``centroid`` feed it straight back to ``h5py.File`` and to
    ``WfnLoader`` itself), and against the ``_filename`` compat attribute,
    which the sibling wave-1 branches still read and which must therefore
    stay the SAME string.
    """
    from pathlib import Path

    with WfnLoader(synth_wfn_path) as loader:
        assert loader.path == synth_wfn_path
        assert isinstance(loader.path, str)
        # Compat attribute survives and does not drift from the property.
        assert loader._filename == loader.path

    # A Path argument is normalised, not passed through.
    with WfnLoader(Path(synth_wfn_path)) as loader:
        assert loader.path == str(synth_wfn_path)
        assert type(loader.path) is str

    # Read-only: the consumers that re-open off this string must not be
    # able to retarget a live loader's identity behind its own file handle.
    with WfnLoader(synth_wfn_path) as loader:
        with pytest.raises(AttributeError):
            loader.path = str(tmp_path / "somewhere_else.h5")
        assert loader.path == synth_wfn_path


def test_symmetry_is_lazy_idempotent_and_the_ensure_sym_alias(synth_wfn_path):
    """``symmetry()`` builds once and returns the SAME object thereafter.

    The two external consumers (``common/wfn_transforms.py``,
    ``common/psi_G_store.py``) hand the result to kernels keyed on its
    identity, so "equal" is not the claim — ``is`` is.

    Anti-tautology: the cache is asserted EMPTY before the first call, so a
    property that returned a fresh object every time, or one that had been
    pre-populated in ``__init__`` (which would change construction cost for
    every consumer in the tree), both fail here rather than pass silently.
    """
    with WfnLoader(synth_wfn_path) as loader:
        assert loader._sym is None, (
            "symmetry() must stay LAZY: building SymMaps in __init__ puts "
            "the symmetry cost on every consumer that only wanted headers")
        first = loader.symmetry()
        assert first is not None
        assert loader._sym is first          # the call is what populated it
        assert loader.symmetry() is first    # idempotent
        # ``_ensure_sym`` is an alias, not a second implementation.
        assert loader._ensure_sym() is first
        assert WfnLoader._ensure_sym is WfnLoader.symmetry


def test_kpt_starts_property_matches_the_mf_header_prefix_sum(synth_wfn_path):
    """``kpt_starts`` == exclusive prefix sum of ``ngk``.

    Checked against ``mf_header.kpt_starts`` (the function the loader calls
    — the seam ``common/density_symmetry_check.py`` depends on) AND against
    an independently written cumsum, because agreeing with the very
    function under test is close to a tautology on its own.

    Non-vacuity: the synthetic deck is asserted RAGGED (ngk differs across
    k), so an implementation returning zeros, ``arange``, or the inclusive
    prefix sum all fail.
    """
    from file_io.mf_header import kpt_starts as mf_kpt_starts

    with WfnLoader(synth_wfn_path) as loader:
        ngk = np.asarray(loader.ngk)
        assert ngk.size >= 2 and len(set(ngk.tolist())) > 1, (
            "the deck must be ragged or this cell cannot see the "
            "difference between a prefix sum and a constant")

        got = loader.kpt_starts
        np.testing.assert_array_equal(got, mf_kpt_starts(ngk))
        # Independent statement of the same arithmetic.
        expected = np.concatenate(([0], np.cumsum(ngk)[:-1]))
        np.testing.assert_array_equal(np.asarray(got), expected)
        assert int(got[0]) == 0
        assert int(got[-1]) + int(ngk[-1]) == int(ngk.sum())
        # Same object the private carried — density_symmetry_check reads
        # one of the two and must not get a copy that drifts.
        assert got is loader._kpt_starts

        with pytest.raises(AttributeError):
            loader.kpt_starts = np.zeros_like(got)


# ===========================================================================
#  The phdf5 collective read's on-device unfold tail, vs eager.
#
#  ``_phdf5_unfold_and_shard`` is the only part of the phdf5 backend that
#  runs without an FFI ``.so``: it takes the re/im-packed IBZ union buffer
#  the C++ read would have produced and turns it into G-flat ψ.  Feeding
#  it a union buffer built here with plain h5py pins that kernel against
#  ``_eager_build`` on a 1x1 mesh (single addressable shard).  Untested
#  here: the collective read itself, and the multi-rank band split —
#  those are ``test_wfn_loader_multiproc.py`` on hardware.
#
#  (This test used to run through the ``phdf5_host`` BACKEND, deleted
#  2026-08-06 — it was the eager backend's own POSIX h5py transport with
#  this kernel bolted on, i.e. a duplicate compute path auto-selected by
#  a missing .so.  The kernel it covered is real, so the coverage stays
#  and only the tier goes.)
# ===========================================================================
def _union_buffer_for_ibz(loader, b_lo, b_hi, ibz_unique_sorted, ngkmax):
    """The ``(nb, ns, n_reads, ngkmax, 2)`` f64 buffer the C++ union read
    delivers, built here with the plain h5py handle."""
    ns = int(loader.nspinor)
    n_reads = len(ibz_unique_sorted)
    buf = np.zeros((b_hi - b_lo, ns, n_reads, ngkmax, 2), dtype=np.float64)
    for r, ibz in enumerate(ibz_unique_sorted):
        start = int(loader._kpt_starts[int(ibz)])
        ngk = int(loader.ngk[int(ibz)])
        buf[:, :, r, :ngk, :] = loader._coeffs_ds[
            b_lo:b_hi, :, start:start + ngk, :]
    return buf


@pytest.mark.parametrize("bands", [(0, 6), (2, 5)])
def test_phdf5_unfold_kernel_matches_eager_ibz(synth_wfn_path, bands):
    """Raw IBZ parity: the on-device unfold tail == the eager h5py read.

    Restricted to ``k='ibz'``.  The tiny synth WFN builds a *degenerate*
    ``SymMaps`` whose ``sym_mats_k`` is length ``ntran`` rather than the
    ``2·ntran`` real WFNs carry, which trips an ``unfold_psi`` edge case
    (``n_sym_spatial = len//2 = 0`` ⇒ every row read as TRS).  So
    ``full_bz`` unfold parity is covered by the real-symmetry WFNsmall
    htransform regression, not this fixture.
    """
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    mesh = _mesh_1x1()
    shard = P(None, ("x", "y"), None, None)
    b_lo, b_hi = bands
    with WfnLoader(synth_wfn_path, mesh=mesh, backend="eager") as le:
        ref = np.asarray(le.load(bands=bands, k="ibz", sharding=shard))

        k_idxs, unfold = le._resolve_k("ibz")
        assert not unfold
        ibz_unique_sorted, n_reads, position_in_reads, n_k = le._kplan(
            k_idxs, unfold)
        ngkmax = int(le.ngkmax)
        buf = _union_buffer_for_ibz(
            le, b_lo, b_hi, ibz_unique_sorted, ngkmax)
        cnk_at_ibz = jax.device_put(
            buf, NamedSharding(mesh, P(("x", "y"), None, None, None, None)))
        got = np.asarray(le._phdf5_unfold_and_shard(
            cnk_at_ibz, k_idxs=k_idxs, unfold=unfold, n_reads=n_reads,
            n_k=n_k, bands_per_rank=b_hi - b_lo, ns=int(le.nspinor),
            ngkmax=ngkmax, position_in_reads=position_in_reads,
            out_sharding=NamedSharding(mesh, shard)))
    assert got.shape == ref.shape
    # Same numpy bytes + identical layout ⇒ bit-identical.
    np.testing.assert_allclose(got, ref, rtol=0, atol=0)


def test_env_forces_backend(synth_wfn_path, monkeypatch):
    monkeypatch.setenv("LORRAX_WFN_BACKEND", "eager")
    mesh = _mesh_1x1()
    with WfnLoader(synth_wfn_path, mesh=mesh, backend="auto") as loader:
        assert loader.backend == "eager"


def test_no_ffi_at_P_gt_1_refuses_and_names_both_libraries(monkeypatch):
    """The tier that WAS here is now a refusal, per decisions.md 2026-08-01.

    Runs the resolver with ``jax.process_count`` monkeypatched to 4 so it
    reaches the terminal arm, and asserts the refusal quotes the read
    door's reason for BOTH platforms and names the escape hatch: a
    refusal that does not say what to do is a different defect.

    THE SECOND DECLARED CHANGE to a moved cell (see the module docstring
    for the first).  As written, this cell reached the terminal arm
    because "this checkout has no ``.so``" — true on a laptop and FALSE
    on Perlmutter, where BUILD_NOTES pins both libraries and the resolver
    would return ``'phdf5'`` and the cell would report DID NOT RAISE.  A
    cell that only fires on machines WITHOUT the library is exactly the
    evaporated coverage this suite's skip-honesty gate exists to catch,
    and it is the cluster leg that most needs the refusal text to be
    right.  So the probe is now an INPUT: both platforms report absent
    with their own reason string, on every machine, and what is asserted
    is the message — which is the cell's actual claim.  On a machine with
    no ``.so`` the real probe produces the identical arm.

    NOT covered here: that the phdf5 backend then works when a library IS
    present — that needs an FFI build (leg L-c).
    """
    import jax
    from file_io import slab_io
    stub = WfnLoader.__new__(WfnLoader)
    stub._mesh = _mesh_1x1()
    monkeypatch.delenv("LORRAX_WFN_BACKEND", raising=False)
    monkeypatch.setattr(jax, "process_count", lambda: 4)
    monkeypatch.setattr(
        slab_io, "probe_read_availability",
        lambda plat=None: (False, (
            "Could not locate liblorrax_ffi.so (platform=CUDA)"
            if plat == "CUDA"
            else "Could not locate liblorrax_ffi_host.so (platform=cpu)")))
    with pytest.raises(RuntimeError) as ei:
        stub._auto_pick_backend()
    msg = str(ei.value)
    assert "liblorrax_ffi.so" in msg and "liblorrax_ffi_host.so" in msg
    assert "LORRAX_WFN_BACKEND=eager" in msg
    assert "phdf5_host" in msg          # says what was removed, not just that
    # ...and it asked BOTH platforms before refusing, in that order: a
    # ladder that stopped at the first no would refuse on a CUDA-less
    # node that has a perfectly good host library.
    assert msg.index("CUDA") < msg.index("cpu:")


def test_deleted_phdf5_host_tier_refuses_rather_than_resolving_elsewhere():
    """A deleted backend spelling must not silently become another one.

    Both doors: the constructor argument and the env escape hatch.  If
    either resolved to 'eager' or 'phdf5' instead, an operator's A/B
    would measure the arm they did not ask for — which is the failure
    the `screening_method = ctsp` ruling names (decisions.md 2026-08-06).
    """
    import os
    with pytest.raises(ValueError, match="deleted"):
        WfnLoader("/nonexistent/WFN.h5", backend="phdf5_host")

    # The env door, exercised on the pure resolver (no file needed).
    stub = WfnLoader.__new__(WfnLoader)
    stub._mesh = _mesh_1x1()
    old = os.environ.get("LORRAX_WFN_BACKEND")
    os.environ["LORRAX_WFN_BACKEND"] = "phdf5_host"
    try:
        with pytest.raises(ValueError, match="deleted"):
            stub._auto_pick_backend()
    finally:
        if old is None:
            os.environ.pop("LORRAX_WFN_BACKEND", None)
        else:
            os.environ["LORRAX_WFN_BACKEND"] = old


# ===========================================================================
#  NEW — the constructor / env refusal surface, reached CONSTRUCTIBLY
# ===========================================================================

@pytest.mark.parametrize("spelling", ["phdf5_HOST", "Phdf5", "posix", "",
                                      "auto2", "eager "])
def test_an_unknown_backend_spelling_refuses_and_lists_the_vocabulary(
        synth_wfn_path, spelling):
    """A backend name the door does not know must NAME the three it does.

    Six spellings, and they are the near-misses rather than random
    strings: a case variant of the DELETED tier (which must not reach the
    deleted-tier message, since that would tell an operator the wrong
    story), a case variant of a live tier, a plausible-but-absent
    transport, the empty string (what a shell ``$UNSET`` expands to), a
    typo, and a trailing space.  ``__init__`` compares EXACTLY — only the
    env door lowercases and strips — so all six land on the same refusal,
    and that is the claim.
    """
    with pytest.raises(ValueError, match="unknown backend") as ei:
        WfnLoader(synth_wfn_path, backend=spelling)
    msg = str(ei.value)
    for known in ("auto", "eager", "phdf5"):
        assert repr(known) in msg or f"'{known}'" in msg, (
            f"the refusal for {spelling!r} does not list {known!r}: {msg}")


def test_the_env_door_normalizes_case_and_space_but_the_ctor_does_not(
        synth_wfn_path, monkeypatch):
    """The two doors have DIFFERENT grammars, and that is deliberate.

    ``_auto_pick_backend`` reads ``LORRAX_WFN_BACKEND`` with
    ``.strip().lower()`` — an operator exporting a shell variable gets
    the benefit of the doubt.  ``__init__``'s ``backend=`` is a Python
    argument and is compared exactly.  Nobody had written that down; a
    future "tidy-up" that lowercased both, or neither, would silently
    change what an A/B measures.  Both halves asserted, in one cell,
    because the claim is the DIFFERENCE.
    """
    mesh = _mesh_1x1()
    monkeypatch.setenv("LORRAX_WFN_BACKEND", "  EAGER  ")
    with WfnLoader(synth_wfn_path, mesh=mesh, backend="auto") as loader:
        assert loader.backend == "eager"
    with pytest.raises(ValueError, match="unknown backend"):
        WfnLoader(synth_wfn_path, backend="  EAGER  ")


def test_a_mesh_less_loader_ignores_the_env_door_entirely(
        synth_wfn_path, monkeypatch):
    """``LORRAX_WFN_BACKEND=phdf5`` must not raise on a mesh-less loader.

    The mesh check comes FIRST in ``_auto_pick_backend``, before the env
    read, and the comment there says why: htransform builds a
    metadata-only loader before its mesh exists, and an operator who
    exported the variable for the ψ loads must not have that construction
    explode.  Asserted rather than trusted — the ordering is one line and
    a refactor that hoisted the env read would pass every other cell in
    this file.
    """
    monkeypatch.setenv("LORRAX_WFN_BACKEND", "phdf5")
    with WfnLoader(synth_wfn_path, backend="auto") as loader:
        assert loader.backend == "eager"


def test_the_deleted_tier_refusal_beats_the_unknown_backend_refusal(
        synth_wfn_path):
    """Order matters: ``phdf5_host`` must get the STORY, not the list.

    Both refusals are ``ValueError``.  If the ``backend not in
    ('eager','phdf5')`` check ran first, an operator who exported the
    deleted spelling would be told "unknown backend" — true, useless, and
    it would read as a typo rather than as a removed transport whose
    replacement is named.  The cell that pins WHICH refusal fires is the
    only thing standing between those two orderings.
    """
    with pytest.raises(ValueError) as ei:
        WfnLoader(synth_wfn_path, backend="phdf5_host")
    msg = str(ei.value)
    assert "deleted" in msg and "2026-08-06" in msg
    assert "unknown backend" not in msg


# ===========================================================================
#  NEW — adopt_mesh: FOUR narrowing conditions (survey concept 2)
# ===========================================================================
#  ``adopt_mesh`` is the late-mesh-binding door: kmeans sizes its mesh from
#  the FFT grid the WFN declares, so the loader is necessarily built
#  mesh-less and lands on the per-rank eager read even though every ψ load
#  it will do is mesh-wide (scorecard BD.2).  It is DELIBERATELY NARROW —
#  four early returns — and before this file every one of them was
#  untested.  A narrowing condition nobody exercises is a condition that
#  can be deleted without a red test, which for THIS method means a loader
#  silently switching transports mid-life in a run that asked for a
#  specific one.
#
#  The four negatives share a positive control below; without it, four
#  cells asserting "the backend did not change" would all pass on a method
#  whose body was ``return self.backend``.
# ===========================================================================

def _p_gt_1(monkeypatch, n=4):
    """Make ``jax.process_count()`` report ``n`` for the duration.

    The loader calls ``jax.process_count`` through the ``jax`` module
    object it imported at module scope, so patching the attribute on that
    module is what the code under test actually reads.
    """
    import jax
    monkeypatch.setattr(jax, "process_count", lambda: n)


def test_adopt_mesh_narrowing_1_a_none_mesh_is_a_no_op(
        synth_wfn_path, monkeypatch):
    """Condition 1a: ``mesh is None``.  Nothing to adopt, nothing changes."""
    _p_gt_1(monkeypatch)
    with WfnLoader(synth_wfn_path, backend="auto") as loader:
        assert loader.backend == "eager" and loader._mesh is None
        assert loader.adopt_mesh(None) == "eager"
        assert loader._mesh is None


def test_adopt_mesh_narrowing_1_b_a_loader_that_already_has_a_mesh_keeps_it(
        synth_wfn_path, monkeypatch):
    """Condition 1b: ``self._mesh is not None``.

    The loader was CONSTRUCTED with a mesh, so its callers already know
    which one; swapping it underneath them would move every cached device
    buffer (``_gvecs_dev_cache`` is keyed on ``id(mesh)``) onto a mesh
    nobody asked for.
    """
    first = _mesh_1x1()
    # CONSTRUCT at P=1 and patch afterwards: with the mesh already in hand
    # a P>1 construction would run the whole auto pick in ``__init__`` and
    # refuse there (no .so), which is a different cell's claim.
    with WfnLoader(synth_wfn_path, mesh=first, backend="auto") as loader:
        _p_gt_1(monkeypatch)
        assert loader.adopt_mesh(_mesh_1x1()) == loader.backend
        assert loader._mesh is first


def test_adopt_mesh_narrowing_2_an_explicit_backend_is_never_overridden(
        synth_wfn_path, monkeypatch):
    """Condition 2a: ``not self._backend_was_auto``.

    THE A/B RULE.  An operator who wrote ``backend='eager'`` is measuring
    the eager arm; a late mesh must not promote them onto the collective
    read halfway through, because the number they publish would then be a
    number for a transport they did not request.
    """
    _p_gt_1(monkeypatch)
    with WfnLoader(synth_wfn_path, backend="eager") as loader:
        assert loader._backend_was_auto is False
        assert loader.adopt_mesh(_mesh_1x1()) == "eager"
        assert loader._mesh is None, (
            "an explicit-backend loader adopted the mesh anyway; the "
            "narrowing returned before the assignment, or should have")


def test_adopt_mesh_narrowing_3_a_loader_already_sharded_keeps_its_mesh(
        synth_wfn_path, monkeypatch):
    """Condition 2b: ``self.backend != 'eager'``.

    A loader already on the collective backend has an open ``SlabIO``
    keyed to a mesh; re-picking would strand it.  Reached here by moving
    an auto-resolved loader onto ``phdf5`` by hand, which is the only way
    to build the state on a machine with no ``.so``.
    """
    _p_gt_1(monkeypatch)
    with WfnLoader(synth_wfn_path, backend="auto") as loader:
        loader.backend = "phdf5"        # the state, without needing the .so
        assert loader.adopt_mesh(_mesh_1x1()) == "phdf5"
        assert loader._mesh is None


def test_adopt_mesh_narrowing_4_a_single_process_run_keeps_the_mesh_less_contract(
        synth_wfn_path):
    """Condition 3: ``jax.process_count() <= 1``.

    NOT patched — this really is a single-process pytest.  The contract
    the callers were built against is that a P=1 run gets the mesh-less
    replicated load, so no band-axis mesh padding appears that was not
    there before.  Asserted on the LOAD as well as on the backend name,
    because "the name did not change" would still pass on a loader that
    had quietly taken the mesh.
    """
    import jax
    assert int(jax.process_count()) == 1, (
        "this cell is about the single-process arm and this session has "
        f"{jax.process_count()} processes; the P>1 arm is leg L-c")
    with WfnLoader(synth_wfn_path, backend="auto") as loader:
        assert loader.adopt_mesh(_mesh_1x1()) == "eager"
        assert loader._mesh is None
        psi = np.asarray(loader.load(bands=(0, 3), k="ibz"))
        assert psi.shape[1] == 3, "a mesh got adopted: the band axis padded"


def test_adopt_mesh_positive_arm_reaches_the_re_pick(
        synth_wfn_path, monkeypatch):
    """THE CONTROL for the four cells above.

    At P>1, on an ``auto``-resolved eager loader with no mesh, adoption
    must actually HAPPEN: the mesh is taken and ``_auto_pick_backend``
    re-runs.  On a machine with a phdf5-capable ``.so`` the re-pick
    returns ``'phdf5'``; on a machine without one it RAISES, which is the
    documented MAY-RAISE (2026-08-06: a missing FFI library is a refusal
    at startup, not a demotion).  Either outcome proves the narrowing did
    not fire — and the mesh is asserted ADOPTED in both, because that
    assignment happens before the re-pick and is the observable that
    separates "reached the re-pick" from "returned early".

    Without this cell the four negatives above would all pass on
    ``def adopt_mesh(self, mesh): return self.backend``.
    """
    _p_gt_1(monkeypatch)
    mesh = _mesh_1x1()
    with WfnLoader(synth_wfn_path, backend="auto") as loader:
        assert loader._mesh is None and loader.backend == "eager"
        try:
            picked = loader.adopt_mesh(mesh)
        except RuntimeError as exc:
            # The MAY-RAISE arm: no FFI library can serve the read here.
            assert "collective WFN read" in str(exc), str(exc)
            assert "LORRAX_WFN_BACKEND=eager" in str(exc)
        else:
            assert picked == "phdf5", (
                f"the re-pick ran and returned {picked!r}; at P>1 with a "
                f"mesh the only non-raising answer is 'phdf5'")
        assert loader._mesh is mesh, (
            "adopt_mesh returned without taking the mesh, so one of the "
            "four narrowing conditions fired on the arm that must not "
            "narrow — the four negative cells above are now tautologies")


# ===========================================================================
#  NEW — bands(): the iterator, not just its chunk boundaries
# ===========================================================================

def test_bands_yields_the_same_psi_the_matching_load_calls_return(
        synth_wfn_path):
    """The iterator's ψ, value for value, against ``load`` on each chunk.

    ``bands`` has ONE caller in the tree and had ONE cell, which asserted
    the ``(bc_lo, bc_hi)`` pairs and threw the arrays away.  A body that
    yielded the RIGHT boundaries with the WRONG window — ``bands=(b_lo,
    bc_hi)`` instead of ``(bc_lo, bc_hi)``, a one-token slip — passed
    that cell and every other cell in this file.  Bit-identity is the bar:
    the iterator is meant to be a loop around ``load``, not an
    approximation of one.
    """
    with WfnLoader(synth_wfn_path) as loader:
        chunks = list(loader.bands(1, 6, chunk=2, k="ibz"))
        assert [bc for bc, _ in chunks] == [(1, 3), (3, 5), (5, 6)]
        # The ragged tail is the interesting one and it is present above.
        assert chunks[-1][0][1] - chunks[-1][0][0] == 1
        for (bc_lo, bc_hi), psi in chunks:
            ref = np.asarray(loader.load(bands=(bc_lo, bc_hi), k="ibz"))
            got = np.asarray(psi)
            assert got.shape == ref.shape, f"{(bc_lo, bc_hi)}: {got.shape}"
            assert np.array_equal(got, ref), (
                f"bands() chunk {(bc_lo, bc_hi)} differs from the matching "
                f"load(); max|Δ| = {np.abs(got - ref).max():.3e}")
        # ...and the chunks CONCATENATE to the whole window, which is the
        # property a driver loop actually relies on.
        whole = np.asarray(loader.load(bands=(1, 6), k="ibz"))
        stitched = np.concatenate([np.asarray(p) for _, p in chunks], axis=1)
        assert np.array_equal(stitched, whole)


def test_bands_refuses_a_non_positive_chunk(synth_wfn_path):
    """``chunk=0`` is an infinite loop and ``chunk=-1`` is an empty sweep
    that reports success; both must refuse, and the refusal must be
    reachable BEFORE the generator is consumed is NOT the contract — it
    is a generator, so the check fires on the first ``next``.  Pinned as
    it is rather than as one might wish it were."""
    with WfnLoader(synth_wfn_path) as loader:
        for bad in (0, -1):
            with pytest.raises(ValueError, match="chunk must be positive"):
                list(loader.bands(0, 4, chunk=bad, k="ibz"))


# ===========================================================================
#  NEW — load / load_process_local band-range refusals and the no-pad rule
# ===========================================================================

@pytest.mark.parametrize("bands,match", [
    ((3, 3), "empty band range"),
    ((4, 2), "empty band range"),
    ((-1, 3), "out of"),
    ((0, 7), "out of"),          # synth mnband is 6
    ((6, 8), "out of"),
])
def test_load_refuses_a_band_range_it_cannot_serve(synth_wfn_path, bands, match):
    """Five ways to ask for bands that are not there.

    The over-file arms matter more than they look: ``_phdf5_build`` clips
    against ``mnband`` and zero-fills the tail, so an unrefused
    ``b_hi > mnband`` would come back as ψ with silent zero bands rather
    than an error — the failure mode is a WRONG NUMBER, not a crash.
    """
    with WfnLoader(synth_wfn_path) as loader:
        with pytest.raises(ValueError, match=match):
            loader.load(bands=bands, k="ibz")


@pytest.mark.parametrize("bands,match", [
    ((3, 3), "empty band range"),
    ((-1, 3), "out of"),
    ((0, 7), "out of"),
])
def test_load_process_local_refuses_the_same_band_ranges(
        synth_wfn_path, bands, match):
    """The SECOND primitive owes the same refusals as the first.

    They are separate code paths with separately written checks, which is
    exactly how one of them acquires a hole.
    """
    with WfnLoader(synth_wfn_path) as loader:
        with pytest.raises(ValueError, match=match):
            loader.load_process_local(bands=bands, k="ibz")


def test_load_process_local_does_not_pad_the_band_axis(synth_wfn_path):
    """THE contract that makes ``load_process_local`` a second primitive.

    ``load`` returns a GLOBAL array: every rank must ask for the same
    window, and the band axis is rounded up to the mesh's band divisor so
    the shards divide.  ``load_process_local`` returns an array addressable
    by THIS process only, so nothing about it is global and
    ``nb == b_hi - b_lo`` EXACTLY — that is what lets ``gw.kin_ion_io``
    give each rank a different ``(bands, k)`` window.

    Asserted on a mesh-carrying loader, because a mesh-less one would give
    ``nb == b_hi - b_lo`` from either method and the cell would prove
    nothing.  The band divisor for a 1x1 mesh is 1, so the ANTI-TAUTOLOGY
    half — a window whose length the divisor would round up — is L-b
    (``test_wfn_loader_emulated_mesh.py``); what is pinned here is the
    single-device commitment and the exact-length rule.
    """
    import jax
    mesh = _mesh_1x1()
    with WfnLoader(synth_wfn_path, mesh=mesh, backend="eager") as loader:
        psi = loader.load_process_local(bands=(1, 4), k="ibz")
        assert psi.shape[1] == 3, f"padded to {psi.shape[1]}, want 3"
        assert len(psi.addressable_shards) == 1
        assert psi.sharding.num_devices == 1, (
            "load_process_local returned an array over more than one "
            "device; its whole contract is that it is process-local")
        assert psi.devices() == {jax.local_devices()[0]}
        # ...and the VALUES are the mesh-less load's, bit for bit.
        ref = np.asarray(loader.load(bands=(1, 4), k="ibz", sharding=None))
        assert np.array_equal(np.asarray(psi), ref)


def test_load_process_local_serves_a_different_k_per_call(gnppm_wfn):
    """The kin_ion pattern at P=1: one explicit k-index per call.

    ``gw.kin_ion_io``'s ρ sweep hands rank *r* the k-points nobody else is
    holding and calls ``load_process_local(bands, k=[ik])`` for each. Two
    claims: the single-k window is the corresponding ROW of the whole
    full-BZ load (so a k-list is a selection and not a different
    computation), and ``load_process_local`` agrees with ``load`` on the
    same k-list bit for bit.

    On the REAL deck, not the synth one: the synth WFN's ``SymMaps`` is
    degenerate (``sym_mats_k`` is length ``ntran`` rather than the
    ``2·ntran`` real files carry), so its full-BZ unfold reads every row
    as time-reversed and a k-list comparison there would be measuring the
    edge case rather than the contract.  The REAL per-rank-DIFFERENT-window
    version is leg L-c.
    """
    with WfnLoader(gnppm_wfn) as loader:
        full = np.asarray(loader.load(bands=(0, 4), k="full_bz"))
        for ik in (0, int(full.shape[0]) // 2, int(full.shape[0]) - 1):
            one = np.asarray(loader.load_process_local(bands=(0, 4), k=[ik]))
            assert one.shape == (1,) + full.shape[1:]
            assert np.array_equal(one[0], full[ik]), f"ik={ik}"
            glob = np.asarray(loader.load(bands=(0, 4), k=[ik]))
            assert np.array_equal(one, glob), (
                f"ik={ik}: load_process_local and load disagree on the same "
                f"single-k window")


def test_an_unknown_k_spec_refuses_by_name(synth_wfn_path):
    """``k='full'``/``'bz'``/``'IBZ'`` are the near-misses, and a k-spec
    that silently resolved to the wrong set would change which physical
    object was loaded with no shape change to notice it by."""
    with WfnLoader(synth_wfn_path) as loader:
        for bad in ("full", "bz", "IBZ", "full-bz"):
            with pytest.raises(ValueError, match="unknown k-spec"):
                loader.load(bands=(0, 2), k=bad)


def test_box_index_dev_refuses_without_a_mesh(synth_wfn_path):
    """A sharding-aware accessor has no sensible single-device fallback,
    and the refusal says which of the two ways to supply a mesh to take."""
    with WfnLoader(synth_wfn_path) as loader:
        with pytest.raises(ValueError, match="box_index_dev") as ei:
            loader.box_index_dev(k="ibz")
        assert "mesh=" in str(ei.value)


def test_bispinor_refuses_a_file_that_is_not_two_spinor(tmp_path, monkeypatch):
    """Both load primitives owe this refusal, and they raise it separately.

    Reached by lying about ``nspinor`` on an open loader rather than by
    building a one-spinor WFN: the check reads ``self.nspinor`` and the
    claim is about the check, not about the file format.
    """
    with WfnLoader(_synth_wfn(tmp_path)) as loader:
        monkeypatch.setattr(loader, "nspinor", 1, raising=False)
        with pytest.raises(ValueError, match="requires a 2-spinor WFN"):
            loader.load(bands=(0, 2), k="ibz", bispinor=True)
        with pytest.raises(ValueError, match="requires a 2-spinor WFN"):
            loader.load_process_local(bands=(0, 2), k="ibz", bispinor=True)


# ===========================================================================
#  NEW — the checked-in hostile decks, and the structural cell for the fold
# ===========================================================================

def test_the_deck_table_is_true_of_the_file_on_disk(any_deck):
    """``conftest.DECKS`` re-measured from the header it claims to describe.

    The table is what the L-b and L-c tiers pick their geometry from, and
    it was transcribed from a survey.  A transcription nobody re-measures
    is how a leg ends up asserting ``(b_hi-b_lo) % world != 0`` about a
    band count the file does not have.  Also asserts the two properties
    that make every one of these decks HOSTILE — ``mnband % 4 == 2`` and
    ``min(ngk) < ngkmax`` — so a future deck added to the table without
    them cannot pass as one.
    """
    name, path, nrk, mnband, ngkmax, ngk_min = any_deck
    with h5py.File(path, "r") as f:
        kp = f["mf_header/kpoints"]
        got = (int(kp["nrk"][()]), int(kp["mnband"][()]),
               int(kp["ngkmax"][()]), int(np.asarray(kp["ngk"]).min()))
    assert got == (nrk, mnband, ngkmax, ngk_min), (
        f"deck {name!r} on disk is {got}, table says "
        f"{(nrk, mnband, ngkmax, ngk_min)}")
    assert mnband % 4 == 2, (
        f"deck {name!r} has mnband={mnband}, which DIVIDES a 4-rank band "
        f"split — it cannot serve as a hostile deck")
    assert ngk_min < ngkmax, (
        f"deck {name!r} is not ragged (min ngk == ngkmax == {ngkmax}), so "
        f"it exercises no G-axis pad")


def test_the_gnppm_deck_loads_and_its_pad_slots_are_the_sentinel(gnppm_wfn):
    """The L-c subject, opened once at P=1: the pairing, single-rank.

    This is the SAME conjunction leg L-c asserts on a sharded multi-rank
    load — ψ zero AND ``gvecs`` at the sentinel, for every slot past
    ``ngk_valid`` — run here so a break in the pairing is caught on a
    laptop instead of only on four ranks.  What it CANNOT cover is the
    band axis (a 1x1 mesh pads nothing) and the collective read; those are
    exactly what leg L-c adds.
    """
    from common.gvec_fft_box import fft_box_pad_sentinel
    with WfnLoader(gnppm_wfn) as loader:
        psi = np.asarray(loader.load(bands=(0, 10), k="ibz"))
        gvecs = loader.gvecs(k="ibz")
        nv = loader.ngk_valid(k="ibz")
        sentinel, _flat = fft_box_pad_sentinel(
            tuple(int(s) for s in loader.fft_grid))
        pad_slots = int(psi.shape[0] * psi.shape[3] - int(nv.sum()))
        assert pad_slots > 0, (
            f"no pad slots at this geometry (ngkmax={psi.shape[3]}, "
            f"ngk={list(map(int, nv))}), so the conjunction below is vacuous")
        for kk in range(psi.shape[0]):
            n = int(nv[kk])
            assert np.all(psi[kk, :, :, n:] == 0), f"k={kk}: ψ pad not zero"
            assert np.array_equal(
                gvecs[kk, n:],
                np.broadcast_to(sentinel, (gvecs.shape[1] - n, 3))), (
                f"k={kk}: gvecs pad rows are not the sentinel {sentinel}")


# ---------------------------------------------------------------------------
#  nspinor = 1 — the scalar WFN this loader could not read until 2026-08-09
# ---------------------------------------------------------------------------

def _scalar_wfn_from(spinor_path, dst):
    """A genuine ``nspinor = 1`` WFN.h5, built in tmp from a spinor deck.

    EVERY CHECKED-IN FIXTURE IS nspinor=2, which is exactly why the defect
    below survived: the deck suite structurally cannot reach the scalar
    path.  Rather than add a seventh fixture (the owner's QE scalar deck
    is a separate, owed piece of work — see ``tests/KNOWN_FAILURES.md``),
    this makes one for the duration of a single test: keep spinor
    component 0, renormalize each (band, k) block to unit norm — slicing a
    normalized 2-spinor otherwise leaves a sub-unit scalar that the
    density invariant would rightly reject — and stamp ``nspinor = 1``.

    What this file IS: a scalar WFN with a real symmetry table, real
    G-lists, ragged ``ngk``, and TRS-augmented rows that the full-BZ
    unfold actually uses.  That is all the loader path under test reads.
    What it is NOT: a self-consistent DFT solution.  Half of an SOC
    density does not respect the parent deck's σ_z mirror, so the
    file-level density-symmetry check would (correctly) object to the
    FILE; the cell disables it, because the subject here is the loader's
    unfold and not the provenance of a fixture that lives for one test.
    """
    import shutil
    shutil.copyfile(spinor_path, dst)
    os.chmod(dst, 0o644)
    with h5py.File(dst, "r+") as f:
        ngk = f["mf_header/kpoints/ngk"][:]
        starts = np.concatenate([[0], np.cumsum(ngk)])
        c = f["wfns/coeffs"][:, :1, :, :]
        z = c[..., 0] + 1j * c[..., 1]
        for i in range(len(ngk)):
            s, e = int(starts[i]), int(starts[i + 1])
            n = np.linalg.norm(z[:, 0, s:e], axis=1, keepdims=True)
            z[:, 0, s:e] /= np.where(n > 0.0, n, 1.0)
        del f["wfns/coeffs"]
        f.create_dataset("wfns/coeffs",
                         data=np.stack([z.real, z.imag], axis=-1))
        f["mf_header/kpoints/nspinor"][()] = 1
    return dst


def test_a_scalar_wfn_unfolds_to_the_full_bz_with_no_spinor_factor(
        gnppm_wfn, tmp_path, monkeypatch):
    """The registered nspinor=1 defect, end to end, on a real HDF5 file.

    UNTIL 2026-08-09 THIS RAISED.  ``symmetry_maps.unfold_psi`` asked for
    a spinor rotation without saying how many components ψ had, got the
    2x2 back unconditionally, and numpy's einsum BROADCAST the size-1
    spinor axis instead of raising — so the unfold returned a
    2-COMPONENT block and ``_eager_build``'s slab write died with

        ValueError: could not broadcast input array from shape (8,2,1947)
                    into shape (8,1,1947)

    which says nothing about spinors and sent two readers to the slab
    arithmetic.  (The registered report has 1457 where this deck has
    1947; same defect, different ``ngk``.)

    What is asserted is the scalar rule itself, not merely that nothing
    raised.  ``gnppm`` has τ = 0 and uses sym rows {0, 2} with ntran = 2,
    so row 2 is TIME REVERSAL — and for a scalar field time reversal is
    Θ = K with no iσ_y.  The whole unfold therefore collapses to
    "identity, or plain conjugation", exactly, with no tolerance:
    ``unfold_psi`` returns on the IBZ G-axis, so the comparison is
    element-for-element against the IBZ block it came from.
    """
    monkeypatch.setenv("LORRAX_TRS_CHECK", "0")   # see _scalar_wfn_from
    path = _scalar_wfn_from(gnppm_wfn, str(tmp_path / "WFN_ns1.h5"))

    with WfnLoader(path, backend="eager") as loader:
        assert int(loader.nspinor) == 1, "the fixture surgery did not take"
        assert not np.any(np.abs(loader.translations) > 1e-12), (
            "PRECONDITION: this deck must be symmorphic for the exact "
            "comparison below (a live τ-phase would need a tolerance)")
        sym = loader._ensure_sym()
        n_tran = int(sym.sym_matrices.shape[0])
        rows = sorted({int(s) for s in sym.sym_idx_k})
        assert any(r >= n_tran for r in rows), (
            f"PRECONDITION: no TRS row in {rows} with ntran={n_tran}, so "
            f"the Θ = K half of the scalar rule would go untested")

        full = np.asarray(loader.load(bands=(0, 8), k="full_bz"))
        ibz = np.asarray(loader.load(bands=(0, 8), k="ibz"))

        assert full.shape[2] == 1, (
            f"the spinor axis came back at width {full.shape[2]}; a scalar "
            f"ψ must stay scalar through the unfold — width 2 here IS the "
            f"defect, and the slab write is only where it surfaces")
        assert full.shape == (int(sym.nk_tot), 8, 1, int(loader.ngkmax))

        for nk in range(full.shape[0]):
            s = int(sym.sym_idx_k[nk])
            src = ibz[int(sym.irr_idx_k[nk])]
            expect = np.conj(src) if s >= n_tran else src
            np.testing.assert_array_equal(
                full[nk], expect,
                err_msg=f"full-BZ k={nk} (sym row {s}, "
                        f"{'TRS' if s >= n_tran else 'spatial'}) is not the "
                        f"scalar rule applied to IBZ k={int(sym.irr_idx_k[nk])}")


def test_the_device_spinor_table_is_1x1_on_a_scalar_wfn(
        gnppm_wfn, tmp_path, monkeypatch):
    """The OTHER consumer — the phdf5 path — carries the same defect.

    ``_ensure_phdf5_static`` builds one spinor matrix per full-BZ k and
    the jitted unfold contracts it as ``einsum("kac,bckg->bakg", ...)``.
    jax broadcasts a size-1 labelled axis exactly as numpy does, so a 2x2
    table there is the same bug on the collective path — and that path has
    no cheap slab-write to trip over it.  The table build is pure numpy
    (its docstring says so: "Touches NO FFI"), so it is reachable at P=1
    with a 1-device mesh and no ``.so``.
    """
    import jax
    from jax.sharding import Mesh

    monkeypatch.setenv("LORRAX_TRS_CHECK", "0")
    path = _scalar_wfn_from(gnppm_wfn, str(tmp_path / "WFN_ns1_dev.h5"))
    mesh = Mesh(np.asarray(jax.devices()[:1]).reshape(1, 1), ("x", "y"))

    with WfnLoader(path, backend="eager", mesh=mesh) as loader:
        static = loader._ensure_phdf5_static()
        U = np.asarray(static["U_per_full"])
        assert U.shape[1:] == (1, 1), (
            f"device spinor table is {U.shape[1:]} per k on an nspinor=1 "
            f"file; the unfold kernel's einsum would broadcast it")
        np.testing.assert_array_equal(U, np.ones_like(U))
        # Non-vacuity: the same deck at nspinor=2 must give 2x2, or this
        # cell would pass on a table that is 1x1 for the wrong reason.
        with WfnLoader(gnppm_wfn, backend="eager", mesh=mesh) as spinor:
            assert np.asarray(
                spinor._ensure_phdf5_static()["U_per_full"]).shape[1:] == (2, 2)


# ---------------------------------------------------------------------------
#  Header layouts the coeffs slicing cannot serve — refused by NAME
# ---------------------------------------------------------------------------

def test_a_real_flavor_wfn_is_refused_by_name_not_by_indexerror(tmp_path):
    """flavor=1 (real coefficients) must refuse at construction, by name.

    BGW sizes the trailing axis of ``wfns/coeffs`` by iflavor
    (``Common/wfn_io_hdf5.F90`` ``setup_hdf5_wfn_file``: ``a3(1)=iflavor``);
    this loader hardcodes that re/im axis as 2 at every coeffs slice.  The
    surgery below builds the HONEST iflavor=1 layout — trailing axis extent
    1, not just a stamped header — which is what makes the cell
    discriminate ORDER as well as presence: the density-symmetry check at
    the END of ``__init__`` slices ``raw[..., 1]`` off the raw dataset, so
    a refusal placed after it (or deleted) surfaces as an IndexError that
    says nothing about flavor, and the ``match=`` below rejects exactly
    that.

    The twin arm first: the same builder's file at flavor=2 constructs,
    so the refusal is keyed on the header value, not fired unconditionally.
    """
    path = _synth_wfn(tmp_path)
    with WfnLoader(path) as loader:
        assert int(loader.flavor) == 2, "the builder no longer writes flavor=2"
    with h5py.File(path, "r+") as f:
        c = f["wfns/coeffs"][..., :1]          # honest iflavor=1 layout
        del f["wfns/coeffs"]
        f.create_dataset("wfns/coeffs", data=c)
        f["mf_header/flavor"][...] = 1
    with pytest.raises(ValueError, match=r"got mf_header/flavor=1, want 2"):
        WfnLoader(path)


def test_a_collinear_nspin2_wfn_is_refused_not_read_as_spin_up_only(tmp_path):
    """nspin=2 must refuse at construction — the silent-wrong alternative.

    BGW packs coeffs axis 1 as ``nspin*nspinor`` (same F90 routine:
    ``a3(3)``); the loader treats that axis as the SPINOR axis alone.  The
    surgery makes a LAYOUT-HONEST collinear file — nspin=2, nspinor=1,
    coeffs axis 1 reinterpreted as the two spin channels, and el/occ/
    ifmin/ifmax duplicated along their spin axis — so it passes every
    OTHER ``__init__`` check (the occs-shape check compares against
    ``(nspin, nkpts, nbands)`` and would be satisfied): absent this
    refusal the loader constructs and every coeffs slice reads channel 0
    only, which is the spin-up-only silent read, not an error.
    ``symmetry_maps/density_symmetry_check.py``'s nspin==2 arm documents
    the same gap but stays permissive, so it is no backstop.

    Twin arm first: the untouched nspin=1 file constructs.
    """
    path = _synth_wfn(tmp_path)
    with WfnLoader(path) as loader:
        assert int(loader.nspin) == 1, "the builder no longer writes nspin=1"
    with h5py.File(path, "r+") as f:
        kp = f["mf_header/kpoints"]
        kp["nspin"][...] = 2
        kp["nspinor"][...] = 1     # coeffs axis 1 (extent 2) = nspin*nspinor
        for name in ("el", "occ", "ifmin", "ifmax"):
            d = kp[name][:]
            del kp[name]
            kp.create_dataset(name, data=np.concatenate([d, d], axis=0))
    with pytest.raises(ValueError, match=r"nspin=2, want 1"):
        WfnLoader(path)


def test_the_loader_kept_no_second_copy_of_the_per_rank_band_clamp():
    """STRUCTURAL, and the only structural cell this service keeps.

    The 2026-08-07 fold moved the per-rank clamped-counts table
    (``_build_phdf5_clamped_counts``) behind the slab_io door, into
    ``file_io._slab_io_ffi._derive_window_counts`` — the same file as
    ``_derive_valid_shape``, which is what makes the 22049c3 band-bound
    divergence structurally impossible rather than merely test-detectable.
    The clip's own regression cells moved WITH it (a clip's cells belong
    next to the clip, ``tests/test_slab_io_hostile_geometry.py`` §4).

    What belongs HERE is the other half of that ruling: this service must
    not grow the copy back.  A loader that reimplemented the clamp would
    pass every value cell in this suite AND every cell over there, right
    up until the two copies disagreed — which is the failure that took
    months.  So: the loader's source states extents and delegates, and
    names no FFI target and no clamp of its own.
    """
    from wfn_loader import loader as _loader_mod
    src = inspect.getsource(_loader_mod)
    for gone in ("_build_phdf5_clamped_counts", "ffi.phdf5",
                 "lorrax_phdf5_read", "bands_per_rank_for_rank"):
        assert gone not in src, (
            f"{gone!r} is back in wfn_loader.loader: the per-rank band "
            f"clamp / FFI knowledge belongs behind the slab_io door "
            f"(DESIGN.md DECISION 1), and a second copy of it is the "
            f"22049c3 divergence class")
    # ...and the delegation it replaced them with is really there, so the
    # absence above is a MOVE and not a deletion.
    assert "read_slabs" in src and "valid_shapes" in src, (
        "the loader no longer states its window extents to the door; the "
        "absences asserted above would then be vacuous")


def test_the_structural_cell_can_fail():
    """RED TWIN for the cell above.

    A source string that DOES carry the forbidden spelling must trip the
    same check.  Without this, a typo in the ``gone`` tuple (or a
    ``getsource`` that quietly returned an empty string on a compiled
    module) would make the structural cell pass forever.
    """
    src = "def _phdf5_build(self):\n    _build_phdf5_clamped_counts(...)\n"
    hits = [g for g in ("_build_phdf5_clamped_counts", "ffi.phdf5")
            if g in src]
    assert hits == ["_build_phdf5_clamped_counts"], hits
    # And the real module is not empty, which is the other way the cell
    # above could pass while measuring nothing.
    from wfn_loader import loader as _loader_mod
    assert len(inspect.getsource(_loader_mod)) > 10_000
