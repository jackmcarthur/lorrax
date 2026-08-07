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
#  phdf5 per-rank band clamp (merged from test_wfn_loader_phdf5_clamp.py)
#  Regression for the 16-GPU CrI3 H5Dread crash: world doesn't divide bands.
# ===========================================================================


from file_io.wfn_loader import _build_phdf5_clamped_counts


# ---------------------------------------------------------------------------
# Unit-level: the clamp formula.
# ---------------------------------------------------------------------------

def _kstarts(ngk_per_read):
    """Exclusive prefix sum + total, as ``_phdf5_build`` passes them."""
    starts, acc = [], 0
    for n in ngk_per_read:
        starts.append(acc)
        acc += int(n)
    return tuple(starts), acc


def _counts(*, world, bands_per_rank, b_lo_logical, band_extent,
            ngk_per_ibz_read, ns, ngktot=None):
    starts, tot = _kstarts(ngk_per_ibz_read)
    return _build_phdf5_clamped_counts(
        world=world, bands_per_rank=bands_per_rank,
        b_lo_logical=b_lo_logical, band_extent=band_extent,
        n_reads=len(ngk_per_ibz_read),
        ngk_per_ibz_read=tuple(ngk_per_ibz_read),
        kchunk_start_per_read=starts,
        ngktot=tot if ngktot is None else ngktot,
        ns=ns,
    ).reshape(world, len(ngk_per_ibz_read), 4)


def test_clamp_in_extent_passthrough():
    """When the request fills the window exactly, every rank gets the full
    bands_per_rank — no clamping."""
    # world=16, bands_per_rank=4 → 64 bands, window ends at 64 ⇒ all fit.
    counts = _counts(world=16, bands_per_rank=4, b_lo_logical=0,
                     band_extent=64, ngk_per_ibz_read=(50, 60), ns=2)
    assert (counts[:, :, 0] == 4).all(), "every rank gets bands_per_rank=4"
    # Other axes copied through:
    assert (counts[:, :, 1] == 2).all()
    assert (counts[:, 0, 2] == 50).all()
    assert (counts[:, 1, 2] == 60).all()
    assert (counts[:, :, 3] == 2).all()


def test_clamp_bispinor_16gpu_regression():
    """The exact case that crashed the bispinor 16-GPU gate.

    world=16, window end 86, bands_per_rank=6 (band-pad 96).
    Rank 14: offset 0+14*6=84 → count min(6, 86-84)=2.
    Rank 15: offset 0+15*6=90 → past the window, count=0.
    Ranks 0..13: full count=6.
    """
    counts = _counts(world=16, bands_per_rank=6, b_lo_logical=0,
                     band_extent=86, ngk_per_ibz_read=(100, 110, 120), ns=2)
    for r in range(14):
        assert (counts[r, :, 0] == 6).all(), \
            f"rank {r} should have band_cnt=6, got {counts[r, :, 0].tolist()}"
    # Rank 14: straddles the end.
    assert (counts[14, :, 0] == 2).all()
    # Rank 15: fully past it.
    assert (counts[15, :, 0] == 0).all()


def test_clamp_extreme_zero_avail():
    """All ranks past the window (degenerate) ⇒ all band_cnt=0."""
    counts = _counts(world=4, bands_per_rank=10, b_lo_logical=100,
                     band_extent=50, ngk_per_ibz_read=(20,), ns=2)
    assert (counts[:, :, 0] == 0).all()


def test_clamp_with_b_lo_offset():
    """b_lo > 0 shifts each rank's window; the clip must respect it."""
    # bands (10, 30) ⇒ nb_logical=20, world=4 ⇒ bands_per_rank=5.
    # Rank 0: off=10+0*5=10, count=5. Rank 1: off=15, count=5.
    # Rank 2: off=20, count=5. Rank 3: off=25, count=min(5,30-25)=5.
    counts = _counts(world=4, bands_per_rank=5, b_lo_logical=10,
                     band_extent=30, ngk_per_ibz_read=(40,), ns=1)
    assert (counts[:, :, 0] == 5).all(), \
        f"all 5, got {counts[:, 0, 0].tolist()}"

    # Same but the window ends at 28: rank 3 reads [25, 28), count=3.
    counts2 = _counts(world=4, bands_per_rank=5, b_lo_logical=10,
                      band_extent=28, ngk_per_ibz_read=(40,), ns=1)
    assert counts2[0, 0, 0] == 5
    assert counts2[1, 0, 0] == 5
    assert counts2[2, 0, 0] == 5
    assert counts2[3, 0, 0] == 3


def test_clamp_shape_and_axes():
    """Result shape ``(world * n_reads, 4)`` with proper axis values."""
    world, ns = 8, 2
    ngk_list = (5, 7, 9, 11)
    counts_r = _counts(world=world, bands_per_rank=3, b_lo_logical=0,
                       band_extent=24, ngk_per_ibz_read=ngk_list, ns=ns)
    assert counts_r.reshape(-1, 4).shape == (world * len(ngk_list), 4)
    # Spinor axis: all entries ns.
    assert (counts_r[:, :, 1] == ns).all()
    # G axis: per-ki value.
    for ki, ngk in enumerate(ngk_list):
        assert (counts_r[:, ki, 2] == ngk).all()
    # Re/im axis: always 2.
    assert (counts_r[:, :, 3] == 2).all()


# ---------------------------------------------------------------------------
# The divergence the duplicated clamp was hiding.
# ---------------------------------------------------------------------------
#
# ``_build_phdf5_clamped_counts`` used to clip the band axis to ``mnband``
# (the FILE extent) while ``_eager_build`` / ``_eager_build_process_local``
# clip to ``b_hi`` (the LOGICAL window) and zero the rest.  Since
# ``WfnLoader.load`` refuses ``b_hi > mnband``, the file clip is never the
# tighter of the two: it fires only past EOF, and NOT on the band-pad rows
# between ``b_hi`` and ``b_lo + nb_padded``.  Those rows therefore came
# back holding real file bands on the phdf5 backend and zeros on the eager
# one — a break of both the documented padding contract ("Band axis pad
# rows are zero-filled") and the P2 byte-identity contract, invisible to a
# parity run whose band count happens to divide the world size.

@pytest.mark.parametrize(
    "world, bpr, b_lo, b_hi, mnband",
    [
        (4, 3, 0, 10, 20),     # nb_padded 12, pad rows 10,11 well inside file
        (16, 1, 0, 4, 64),     # the parity harness's default --bands 0,4 @ 4x4
        (4, 3, 5, 15, 40),     # b_lo offset, nb_padded 12, pad rows 10,11
        (2, 4, 0, 8, 8),       # exactly divisible: no pad rows, nothing clipped
    ],
)
def test_band_pad_rows_get_no_file_data(world, bpr, b_lo, b_hi, mnband):
    """Per-rank band counts must sum to the LOGICAL band count.

    Positive control, RUN: replaying the old ``min(bpr, mnband - off)``
    clip over these four rows gives sums 12, 16, 12 and 8 against the
    logical 10, 4, 10 and 8 — so the first three rows return False under
    the pre-fix helper and only the exactly-divisible row (the geometry
    the parity harness tends to use, which is why this went unseen)
    agrees either way.
    """
    counts = _counts(world=world, bands_per_rank=bpr, b_lo_logical=b_lo,
                     band_extent=min(b_hi, mnband),
                     ngk_per_ibz_read=(11, 13), ns=2)
    # counts[r, ki, 0] is the same for every ki; take read 0.
    assert int(counts[:, 0, 0].sum()) == b_hi - b_lo
    # ...and no rank that reads anything may reach past the window.  A
    # rank whose whole block is past it gets count 0 and selects nothing,
    # which is the pad-row semantics, not an overrun.
    for r in range(world):
        cnt = int(counts[r, 0, 0])
        if cnt:
            assert b_lo + r * bpr + cnt <= b_hi


def test_clamp_agrees_with_the_slab_io_clip_on_every_dim():
    """Every counts row equals ``_derive_valid_shape`` of that rank's
    slab/offset/logical triple — on ALL FOUR dims, not just the band one.

    SCOPE, stated because it is narrower than the name suggests: this is
    a VALUE-equivalence check, not a no-second-copy check.  Re-inlining
    an *equivalent* clip here passes it (measured: it does).  What it
    catches is a re-inline that DIVERGES — which is what happened, on the
    band axis, for as long as the two spellings coexisted.  The
    structural half is ``test_helper_delegates_the_clip``.
    """
    from file_io._slab_io_ffi import _derive_valid_shape
    world, bpr, b_lo, band_extent, ns = 5, 3, 2, 11, 2
    ngk = (11, 13, 17)
    starts, ngktot = _kstarts(ngk)
    counts = _counts(world=world, bands_per_rank=bpr, b_lo_logical=b_lo,
                     band_extent=band_extent, ngk_per_ibz_read=ngk, ns=ns)
    for r in range(world):
        for ki in range(len(ngk)):
            assert tuple(int(v) for v in counts[r, ki]) == _derive_valid_shape(
                (bpr, ns, ngk[ki], 2),
                (b_lo + r * bpr, 0, starts[ki], 0),
                (band_extent, ns, ngktot, 2))


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


def test_helper_delegates_the_clip():
    """The clip ARITHMETIC must not be respelled in this module.

    The structural half of the pair above: a value check cannot tell an
    equivalent re-inline from the shared helper, and an equivalent
    re-inline is how the two spellings started before they diverged.
    String check, brittle by design — renaming ``_derive_valid_shape``
    must force a look at this call site.
    """
    import inspect
    import file_io.wfn_loader as wm
    src = inspect.getsource(wm._build_phdf5_clamped_counts)
    assert "_derive_valid_shape" in src, \
        ("_build_phdf5_clamped_counts no longer delegates to SlabIO's "
         "_derive_valid_shape — the padded-slab clip has been respelled "
         "here, which is exactly how the band bound drifted from b_hi to "
         "mnband last time")
