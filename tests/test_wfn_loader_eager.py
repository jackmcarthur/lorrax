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
#  The phdf5 collective read's on-device unfold tail, vs eager.
#
#  ``_phdf5_unfold_and_shard`` is the only part of the phdf5 backend that
#  runs without an FFI ``.so``: it takes the re/im-packed IBZ union buffer
#  the C++ read would have produced and turns it into G-flat ψ.  Feeding
#  it a union buffer built here with plain h5py pins that kernel against
#  ``_eager_build`` on a 1x1 mesh (single addressable shard).  Untested
#  here: the collective read itself, and the multi-rank band split —
#  those are `tests/bench/wfn_loader_backend_parity_test.py` on hardware.
#
#  (This test used to run through the ``phdf5_host`` BACKEND, deleted
#  2026-08-06 — it was the eager backend's own POSIX h5py transport with
#  this kernel bolted on, i.e. a duplicate compute path auto-selected by
#  a missing .so.  The kernel it covered is real, so the coverage stays
#  and only the tier goes.)
# ===========================================================================
def _mesh_1x1():
    import jax
    from jax.sharding import Mesh
    devs = np.asarray(jax.devices()[:1]).reshape(1, 1)
    return Mesh(devs, ("x", "y"))


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

    Runs the resolver with ``jax.process_count`` monkeypatched to 4 on a
    tree with no ``.so`` (which is this checkout — see the paths in the
    message), so it reaches the terminal arm.  Asserts the refusal quotes
    ``probe_target``'s reason for BOTH platforms and names the escape
    hatch: a refusal that does not say what to do is a different defect.

    NOT covered here: that the phdf5 backend then works when a library
    IS present — that needs an FFI build (`tests/bench/…parity_test.py`).
    """
    import jax
    from file_io.wfn_loader import WfnLoader as _WL
    stub = _WL.__new__(_WL)
    stub._mesh = _mesh_1x1()
    monkeypatch.delenv("LORRAX_WFN_BACKEND", raising=False)
    monkeypatch.setattr(jax, "process_count", lambda: 4)
    with pytest.raises(RuntimeError) as ei:
        stub._auto_pick_backend()
    msg = str(ei.value)
    assert "liblorrax_ffi.so" in msg and "liblorrax_ffi_host.so" in msg
    assert "LORRAX_WFN_BACKEND=eager" in msg
    assert "phdf5_host" in msg          # says what was removed, not just that


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
    from file_io.wfn_loader import WfnLoader as _WL
    stub = _WL.__new__(_WL)
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
#  The per-rank band clamp for the collective read MOVED, 2026-08-07.
#
#  The table it pins (``_build_phdf5_clamped_counts``) was the loader's copy
#  of SlabIO's padded-slab clip, applied once per (rank, window); it is now
#  ``file_io._slab_io_ffi._derive_window_counts``, in the same file as
#  ``_derive_valid_shape`` — which is what makes the 22049c3 band-bound
#  divergence structurally impossible rather than test-detectable.  Its NINE
#  cells / twelve collected ids (five clamp cases, the four-row band-pad
#  parametrization with its declared negative control, the all-four-dims
#  value check and both inspect.getsource structural cells) moved WITH it,
#  verbatim, to ``tests/test_slab_io_hostile_geometry.py`` §4.  Nothing was
#  dropped: a clip's regression cells belong next to the clip.
# ===========================================================================
