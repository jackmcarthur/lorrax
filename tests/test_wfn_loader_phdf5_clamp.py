"""Per-rank clamp tests for the phdf5 backend of ``WfnLoader``.

Surface: ``_build_phdf5_clamped_counts`` builds the per-rank ``counts``
table for the kchunk-union phdf5 read so each rank's hyperslab stays
inside the file's [0, mnband) band extent — even when the band-pad
inflates the request past the on-disk extent.

Trigger: bispinor 16-GPU CrI3 6×6 30Ry gate, ``bands=(0, 86)`` across
``world=16`` ranks ⇒ band-pad of 96 ⇒ ``bands_per_rank=6`` ⇒ ranks
14/15 would read [84,90)/[90,96) of an 86-band file — H5Dread fails
with "selection + offset not within extent".  See
``reports/bispinor_ibz_e2e_gate_16gpu_v2_2026-05-16/``.
"""
from __future__ import annotations

import numpy as np
import pytest

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
