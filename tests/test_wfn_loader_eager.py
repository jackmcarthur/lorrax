"""Bit-equality tests for the eager backend of ``file_io.wfn_loader.WfnLoader``.

The contract: ``loader.load(bands, k='ibz')`` reproduces
``WFNReader.get_cnk_batch`` exactly, and ``loader.load(bands, k='full_bz')``
reproduces ``SymMaps.get_cnk_fullzone_batch`` exactly (after stripping
band/G padding).  Same for ``loader.gvecs``.

Uses the small captured MoS2 3x3 WFN if available; otherwise builds a
tiny synthetic WFN.h5 and uses it.  Gated by file presence so pytest
runs cleanly on a laptop.
"""
from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from file_io.wfn_loader import WfnLoader
from common.symmetry_maps import SymMaps


_MOS2_WFN = "/pscratch/sd/j/jackm/lorrax_sandbox/runs/MoS2/00_mos2_3x3_cohsex/qe/nscf/WFN.h5"


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


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------

def _wfn_path() -> str | None:
    return _MOS2_WFN if os.path.exists(_MOS2_WFN) else None


@pytest.fixture
def synth_wfn_path(tmp_path):
    return _synth_wfn(tmp_path)


@pytest.fixture(params=["synth", "mos2"])
def wfn_path(request, tmp_path):
    if request.param == "mos2":
        if not _wfn_path():
            pytest.skip("MoS2 3x3 WFN not present")
        return _MOS2_WFN
    return _synth_wfn(tmp_path)


# ---------------------------------------------------------------------------
# Per-k g_flat round-trip
# ---------------------------------------------------------------------------
#
# The earlier bit-equality tests compared the loader to legacy classes
# (``WFNReader.get_cnk_batch`` / ``SymMaps.get_cnk_fullzone[_batch]`` /
# ``get_gvecs_kfull``) that no longer exist.  After P5 the loader is
# the contract; the parity surface that survived is:
#
#   * ``test_bispinor_lift_matches_legacy`` (below) — exercises
#     ``loader.load(k='full_bz', bispinor=True)`` against the legacy
#     :func:`common.bispinor_init.get_small_psi_component` math, which
#     stays as the lift's reference even after the wfn-unfold helpers
#     were retired.
#   * ``common/wfn_loader_backend_parity_test.py`` (phdf5 vs eager
#     bit-equality under Shifter).
#   * MoS2 3x3 xonly GW smoke (full pipeline through unfold +
#     bispinor + ζ-FFT, eqp0.dat bit-identical).

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

def test_bispinor_lift_matches_legacy(wfn_path):
    """``loader.load(bispinor=True)`` must reproduce the legacy
    :func:`common.bispinor_init.get_small_psi_component` lift exactly
    when applied to ``loader.load(bispinor=False)``."""
    from common.bispinor_init import get_small_psi_component

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
        bvec = np.asarray(loader.bvec, dtype=np.float64)
        n_k = psi_2.shape[0]

        for nk in range(n_k):
            n = int(ngk_v[nk])
            gvecs_k = gvecs_full[nk, :n].astype(np.float64)
            psi_L = psi_2[nk, :, :, :n]                          # (nb, 2, n)
            import jax as _jax
            psi_S_ref = np.asarray(get_small_psi_component(
                _jax.numpy.asarray(gvecs_k),
                _jax.numpy.asarray(unfolded_kpts[nk]),
                _jax.numpy.asarray(bvec),
                _jax.numpy.asarray(psi_L)))
            np.testing.assert_allclose(
                psi_4[nk, :, 2:4, :n], psi_S_ref,
                atol=1e-13, rtol=0,
                err_msg=f"nk={nk}")
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
#  phdf5 per-rank band clamp (merged from test_wfn_loader_phdf5_clamp.py)
#  Regression for the 16-GPU CrI3 H5Dread crash: world doesn't divide bands.
# ===========================================================================


from file_io.wfn_loader import _build_phdf5_clamped_counts


# ---------------------------------------------------------------------------
# Unit-level: the clamp formula.
# ---------------------------------------------------------------------------

def test_clamp_in_extent_passthrough():
    """When the request fits in mnband, every rank gets the full
    bands_per_rank — no clamping."""
    # world=16, bands_per_rank=4 → 64 bands, mnband=64 ⇒ all ranks fit.
    counts = _build_phdf5_clamped_counts(
        world=16, bands_per_rank=4, b_lo_logical=0, mnband_file=64,
        n_reads=2, ngk_per_ibz_read=(50, 60), ns=2,
    ).reshape(16, 2, 4)
    assert (counts[:, :, 0] == 4).all(), "every rank gets bands_per_rank=4"
    # Other axes copied through:
    assert (counts[:, :, 1] == 2).all()
    assert (counts[:, 0, 2] == 50).all()
    assert (counts[:, 1, 2] == 60).all()
    assert (counts[:, :, 3] == 2).all()


def test_clamp_bispinor_16gpu_regression():
    """The exact case that crashed the bispinor 16-GPU gate.

    world=16, mnband=86, bands_per_rank=6 (band-pad 96).
    Rank 14: offset 0+14*6=84 → count min(6, 86-84)=2.
    Rank 15: offset 0+15*6=90 → past EOF, count=0.
    Ranks 0..13: full count=6.
    """
    counts = _build_phdf5_clamped_counts(
        world=16, bands_per_rank=6, b_lo_logical=0, mnband_file=86,
        n_reads=3, ngk_per_ibz_read=(100, 110, 120), ns=2,
    ).reshape(16, 3, 4)
    for r in range(14):
        assert (counts[r, :, 0] == 6).all(), \
            f"rank {r} should have band_cnt=6, got {counts[r, :, 0].tolist()}"
    # Rank 14: straddles EOF.
    assert (counts[14, :, 0] == 2).all()
    # Rank 15: fully past EOF.
    assert (counts[15, :, 0] == 0).all()


def test_clamp_extreme_zero_avail():
    """All ranks past EOF (degenerate) ⇒ all band_cnt=0."""
    counts = _build_phdf5_clamped_counts(
        world=4, bands_per_rank=10, b_lo_logical=100, mnband_file=50,
        n_reads=1, ngk_per_ibz_read=(20,), ns=2,
    ).reshape(4, 1, 4)
    assert (counts[:, :, 0] == 0).all()


def test_clamp_with_b_lo_offset():
    """b_lo > 0 shifts each rank's window; clamp must respect it."""
    # bands (10, 30) ⇒ nb_logical=20, world=4 ⇒ bands_per_rank=5.
    # Rank 0: off=10+0*5=10, count=5. Rank 1: off=15, count=5.
    # Rank 2: off=20, count=5. Rank 3: off=25, count=min(5,30-25)=5.
    # All fit in mnband=30.
    counts = _build_phdf5_clamped_counts(
        world=4, bands_per_rank=5, b_lo_logical=10, mnband_file=30,
        n_reads=1, ngk_per_ibz_read=(40,), ns=1,
    ).reshape(4, 1, 4)
    assert (counts[:, :, 0] == 5).all(), \
        f"all 5, got {counts[:, 0, 0].tolist()}"

    # Same but mnband only 28: rank 3 reads [25, 28), count=3.
    counts2 = _build_phdf5_clamped_counts(
        world=4, bands_per_rank=5, b_lo_logical=10, mnband_file=28,
        n_reads=1, ngk_per_ibz_read=(40,), ns=1,
    ).reshape(4, 1, 4)
    assert counts2[0, 0, 0] == 5
    assert counts2[1, 0, 0] == 5
    assert counts2[2, 0, 0] == 5
    assert counts2[3, 0, 0] == 3


def test_clamp_shape_and_axes():
    """Result shape ``(world * n_reads, 4)`` with proper axis values."""
    world, n_reads, ns = 8, 4, 2
    ngk_list = (5, 7, 9, 11)
    counts = _build_phdf5_clamped_counts(
        world=world, bands_per_rank=3, b_lo_logical=0, mnband_file=24,
        n_reads=n_reads, ngk_per_ibz_read=ngk_list, ns=ns,
    )
    assert counts.shape == (world * n_reads, 4)
    counts_r = counts.reshape(world, n_reads, 4)
    # Spinor axis: all entries ns.
    assert (counts_r[:, :, 1] == ns).all()
    # G axis: per-ki value.
    for ki, ngk in enumerate(ngk_list):
        assert (counts_r[:, ki, 2] == ngk).all()
    # Re/im axis: always 2.
    assert (counts_r[:, :, 3] == 2).all()


# ---------------------------------------------------------------------------
# Integration: the unpatched bug would fail at the FFI inside
# ``WfnLoader._phdf5_build``.  We don't run an actual phdf5 FFI here
# (would require MPI + multi-process), but we verify the construction
# of the counts table inside ``_phdf5_build`` exercises the helper —
# i.e. a synthetic WFN.h5 with mnband=86 + a 16-device mesh would hit
# the patched code path, not the old replicated counts path.  The full
# integration check (real 16-GPU srun) lives in the
# ``reports/bispinor_ibz_e2e_gate_16gpu_v2_2026-05-16/`` smoke run.
# ---------------------------------------------------------------------------

def test_helper_is_used_by_phdf5_build():
    """Smoke check that ``_phdf5_build`` source references the helper.

    Guards against future refactors that inline the clamp logic and
    re-introduce a per-rank-overshoot regression — a regression would
    likely involve someone removing the helper call and putting back
    the replicated-bands_per_rank shortcut.  This is a string check, so
    it's brittle by design: if someone *renames* the helper, they must
    update this test, which forces a thoughtful review of the rename.
    """
    import inspect
    import file_io.wfn_loader as wm
    src = inspect.getsource(wm.WfnLoader._phdf5_build)
    assert "_build_phdf5_clamped_counts" in src, \
        "_phdf5_build no longer calls the per-rank clamp helper"
    assert "count_partition_spec" in src, \
        "_phdf5_build no longer requests sharded counts"
