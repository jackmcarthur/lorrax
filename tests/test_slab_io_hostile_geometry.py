"""CONSTRUCTED-hostile SlabIO geometry + MPI-world regressions.

Every geometry here is built to be non-divisible, ragged or empty-tiled
on purpose (TASTE 11) rather than merely tolerated: an extent that
happens to divide teaches nothing about the padded path, which is the
only path the ζ / V_q producers ever take.

The arithmetic under test is pure, so it pins the contract without a
launcher.  The hardware runs that produced these numbers are named on
each test; the reproducer for the MPI-world guard is in its docstring
because it CANNOT be expressed in-process — it needs a real srun with
the wrong PMI flavour.
"""
from __future__ import annotations

import numpy as np
import pytest

from file_io._slab_io_ffi import (
    _derive_valid_shape,
    _derive_window_counts,
    _mpi_world_verdict,
    _normalize_valid_shape,
    _normalize_window_tables,
    _rank_block_offsets,
    _validate_block_divisible,
)


# ---------------------------------------------------------------------------
# 1. The padded path, CONSTRUCTED non-divisible
# ---------------------------------------------------------------------------
#
# Measured bit-exact end-to-end on Perlmutter job 56389339, 4 nodes / 16
# ranks, 4x4 mesh, backend=PHDF5_FFI forced, each case written from a
# padded sharded operand and compared against a single-rank reference
# plus an explicit zero-check of the pad region.

@pytest.mark.parametrize(
    "logical, padded, expect_valid",
    [
        # prime extents against a 4x4 mesh: remainder 1 on dim 0
        ((17, 23), (20, 24), (17, 23)),
        # remainder 1 on BOTH axes
        ((13, 17), (16, 20), (13, 17)),
        # non-divisible on one axis only
        ((17, 16), (20, 16), (17, 16)),
        # more ranks than slices: 15 of 16 tiles are wholly pad
        ((1, 1), (4, 4), (1, 1)),
        # empty tiles on one axis
        ((2, 16), (4, 16), (2, 16)),
    ],
)
def test_padded_slab_clips_to_the_logical_extent(logical, padded,
                                                 expect_valid):
    """A padded operand writes its logical prefix and nothing else."""
    got = _derive_valid_shape(padded, (0,) * len(padded), logical)
    assert got == expect_valid


def test_slab_starting_past_the_dataset_is_an_empty_no_op_not_a_refusal():
    """Offset past the end yields a 0 extent — a legitimate rendezvous.

    Measured (job 56389339, case B6): every rank selects nothing, the
    collective H5Dwrite still happens, no hang, no error.
    """
    assert _derive_valid_shape((16, 16), (64, 0), (16, 16)) == (0, 16)


def test_offset_overrunning_the_dataset_clips_silently():
    """Documented clip (slab_io.write_slab): min(A.shape, ds - offset).

    Job 56389339 case B5 read the result back and confirmed the clipped
    write lands exactly where the clip says, with no garbage past it.
    """
    assert _derive_valid_shape((16, 16), (8, 0), (16, 16)) == (8, 16)


def test_valid_shape_override_that_overruns_the_dataset_refuses():
    """valid_shape is an assertion SlabIO has no licence to shrink."""
    with pytest.raises(ValueError, match="valid slab exceeds dataset extent"):
        _normalize_valid_shape(
            op="write_slab", name="D", valid_shape=(16, 16),
            slab_shape=(16, 16), offset=(8, 0), ds_shape=(16, 16))


# ---------------------------------------------------------------------------
# 2. Non-divisibility is a REFUSAL, on both backends — the docstring that
#    promised otherwise was wrong
# ---------------------------------------------------------------------------

def test_nondivisible_sharded_extent_refuses_naming_the_padded_shape():
    """``read_slab(shape=(17,16), P('x','y'))`` on a 4x4 mesh REFUSES.

    ``SlabIO.read_slab``'s docstring used to promise "it need not be
    mesh-divisible under partition_spec either — SlabIO reads the
    rounded-up extent and trims".  That was never true of any backend:
    measured job 56389339, PHDF5_FFI raises this ValueError and
    H5PY_ALLGATHER raises JAX's own IndivisibleError, because the return
    value is a jax.Array of exactly ``shape`` sharded by
    ``partition_spec`` and JAX will not build one at 17/4.  The refusal
    must name the shape the caller should ask for instead.
    """
    with pytest.raises(ValueError, match="not divisible") as ei:
        _validate_block_divisible(
            op="read_slab", name="D", shape=(17, 16),
            axis_count_per_dim=(1, 1), axis_flat=(0, 1),
            mesh_shape=(4, 4))
    # the rounded-up extent is the actionable part of the message
    assert "20" in str(ei.value)


def test_padded_extent_that_is_itself_not_divisible_refuses():
    """A caller who pads to 18 on a 4-wide mesh axis is still wrong."""
    with pytest.raises(ValueError, match="dimension 0 size 18"):
        _validate_block_divisible(
            op="write_slab", name="D", shape=(18, 24),
            axis_count_per_dim=(1, 1), axis_flat=(0, 1),
            mesh_shape=(4, 4))


def test_replicated_dims_are_never_divisibility_checked():
    """A dim no mesh axis shards has divisor 1, so ANY extent is legal.

    The A6 geometry (job 56389339): logical (3, 17, 23) under
    P(None,'x','y') is written from the padded operand (3, 20, 24) — the
    leading 3 is replicated and stays 3, indivisible by 4 and irrelevant.
    """
    _validate_block_divisible(
        op="write_slab", name="D", shape=(3, 20, 24),
        axis_count_per_dim=(0, 1, 1), axis_flat=(0, 1),
        mesh_shape=(4, 4))
    # ...and the sharded dims of that same operand ARE checked
    with pytest.raises(ValueError, match="dimension 1 size 17"):
        _validate_block_divisible(
            op="write_slab", name="D", shape=(3, 17, 23),
            axis_count_per_dim=(0, 1, 1), axis_flat=(0, 1),
            mesh_shape=(4, 4))


def test_one_dim_sharded_by_both_mesh_axes_uses_the_axis_PRODUCT():
    """P(('x','y'), None) on 4x4 needs dim 0 divisible by 16, not 4."""
    with pytest.raises(ValueError, match="mesh-axis product 16"):
        _validate_block_divisible(
            op="write_slab", name="D", shape=(20, 5),
            axis_count_per_dim=(2, 0), axis_flat=(0, 1),
            mesh_shape=(4, 4))
    _validate_block_divisible(
        op="write_slab", name="D", shape=(32, 5),
        axis_count_per_dim=(2, 0), axis_flat=(0, 1),
        mesh_shape=(4, 4))


# ---------------------------------------------------------------------------
# 3. The MPI-world guard — the trap that a working collective write did
#    NOT close
# ---------------------------------------------------------------------------
#
# REPRODUCER (cannot be expressed in-process; needs a real launcher):
#
#   srun --jobid=<J> --overlap --mpi=pmi2 -N 4 -n 16 ... \
#        python -m <any driver with slab_io=phdf5_ffi>
#
# Measured job 56389339, 2026-08-06.  With --mpi=pmi2 (the wrong PMI
# flavour for Cray MPICH) every rank gets a private singleton
# MPI_COMM_WORLD: MPI_Comm_size()==1 while jax.process_count()==16.
# BEFORE this guard, and with the independent-I/O settings that
# stage/phdf5_stage_cray.sh documents as the remedy for the
# ad_cray_write_coll.c:669 OOM that the collective path dies with under
# the same launch, SlabIO wrote and read back all eight hostile
# geometries BIT-EXACT with rc=0 and a correctly striped file: a
# completely broken MPI producing a perfect-looking artifact.  Neither
# ffi.io.open_file's ``p*q == jax.process_count()`` nor shard_index.h's
# ``prod(mesh_shape) == ctx->world_size`` can catch it, because
# ctx->world_size IS jax.process_count() — both compare JAX to JAX.

def test_singleton_mpi_world_under_a_16_rank_launch_is_refused():
    """The measured case: MPI says 1, JAX says 16."""
    verdict, msg = _mpi_world_verdict(1, 16, "MPICH ABI", require=True)
    assert verdict == "refuse"
    assert "MPI_Comm_size(MPI_COMM_WORLD)=1" in msg
    assert "jax.process_count()=16" in msg
    # the message must name the actionable cause, not just the symptom
    assert "cray_shasta" in msg


def test_matching_world_sizes_pass_and_a_single_process_run_is_fine():
    assert _mpi_world_verdict(16, 16, "MPICH ABI", require=True)[0] == "ok"
    assert _mpi_world_verdict(1, 1, "MPICH ABI", require=True)[0] == "ok"


def test_an_unprobeable_mpi_warns_by_default_and_refuses_when_required():
    """Not finding libmpi must not break a working run by itself...

    ...but it must be escalatable, because "we could not check" is
    exactly the state the silent-corruption run was in.
    """
    assert _mpi_world_verdict(
        None, 16, "no libmpi found", require=False)[0] == "unprobed"
    assert _mpi_world_verdict(
        None, 16, "no libmpi found", require=True)[0] == "refuse"


def test_the_verdict_is_rank_invariant_by_construction():
    """Every rank compares its own MPI_Comm_size to a replicated
    jax.process_count(), so a broken bootstrap refuses on ALL ranks.

    Confirmed on hardware: 16/16 ranks printed the refusal (job
    56389339, --mpi=pmi2).  A refusal on a proper subset of ranks inside
    a collective is a hang, not an error.
    """
    verdicts = {_mpi_world_verdict(1, 16, "MPICH ABI", require=True)[0]
                for _rank in range(16)}
    assert verdicts == {"refuse"}


# ===========================================================================
# 4. The SAME clip, per rank, for a MULTI-WINDOW read (``read_slabs``)
#
# MIGRATED HERE 2026-08-07 from tests/test_wfn_loader_eager.py:372-589, with
# the table they pin: it was ``wfn_loader._build_phdf5_clamped_counts``, one
# import away from the clip it was applying, and it is now
# ``_derive_window_counts`` in the same file as ``_derive_valid_shape``.  The
# cells are the ones that caught the 16-GPU CrI3 H5Dread crash and the
# 22049c3 band-bound divergence, and they read the same as they did there —
# only the adapter below changed vocabulary, because the door takes LOGICAL
# per-window extents where the loader took absolute file bounds.
# ===========================================================================


def _kstarts(ngk_per_read):
    """Exclusive prefix sum + total, as ``_phdf5_build`` passes them."""
    starts, acc = [], 0
    for n in ngk_per_read:
        starts.append(acc)
        acc += int(n)
    return tuple(starts), acc


def _counts(*, world, bands_per_rank, b_lo_logical, band_extent,
            ngk_per_ibz_read, ns, ngktot=None):
    """The ``(world, n_windows, 4)`` counts table for one psi-shaped request.

    The helper these cells were written against took the LOADER's
    vocabulary — ``b_lo_logical`` plus an absolute ``band_extent``, a
    ``kchunk_start`` per window, and the file's ``ngktot``.  The door takes
    per-window LOGICAL extents instead, so the identical request is

      per-rank slab      ``(bands_per_rank, ns, ngkmax, 2)``
      window valid shape ``(band_extent - b_lo, ns, ngk[w], 2)``

    i.e. the same arithmetic with the window origin subtracted out.
    ``kchunk_start``/``ngktot`` fall out because a window's G extent IS
    ``ngk[w]``; the file bound they used to carry is the caller's to apply
    before it asks (``band_extent = min(b_hi, mnband)``).  The cell
    ``test_clamp_agrees_with_the_slab_io_clip_on_every_dim`` below is the
    proof that the two spellings agree on all four dims — it still builds
    the OLD triple by hand and compares.

    ``mesh_shape=(world, 1)`` with the band dim sharded by both axes is the
    ``P(('x','y'), None, None, None)`` production layout at a world these
    cells can state directly (16, 5, 8 …) without needing a square mesh.
    """
    del ngktot                      # the door derives no bound from it
    ngkmax = max(int(n) for n in ngk_per_ibz_read)
    per_rank_shape = (int(bands_per_rank), int(ns), ngkmax, 2)
    valid_shapes = np.asarray(
        [[int(band_extent) - int(b_lo_logical), int(ns), int(n), 2]
         for n in ngk_per_ibz_read], dtype=np.int64)
    rank_offsets = _rank_block_offsets(
        per_rank_shape=per_rank_shape,
        axis_count_per_dim=(2, 0, 0, 0), axis_flat=(0, 1),
        mesh_shape=(int(world), 1))
    return _derive_window_counts(
        per_rank_shape=per_rank_shape, rank_offsets=rank_offsets,
        valid_shapes=valid_shapes,
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
# The loader's copy of this table used to clip the band axis to ``mnband``
# (the FILE extent) while its serial reader clips to ``b_hi`` (the LOGICAL
# window) and zeros the rest.  Since ``WfnLoader.load`` refuses
# ``b_hi > mnband``, the file clip is never the tighter of the two: it fires
# only past EOF, and NOT on the band-pad rows between ``b_hi`` and
# ``b_lo + nb_padded``.  Those rows therefore came back holding real file
# bands on the collective backend and zeros on the serial one — a break of
# both the documented padding contract ("Band axis pad rows are zero-filled")
# and the byte-identity contract, invisible to a parity run whose band count
# happens to divide the world size.

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

    It doubles as the migration's own proof: the triple built by hand
    below is the ABSOLUTE-bounds spelling the loader used, and the table
    on the left is the door's origin-relative one.
    """
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
# Integration: the unpatched bug would fail at the FFI inside the collective
# read.  We don't run an actual phdf5 FFI here (would require MPI +
# multi-process), but we verify that the door's read path is what builds the
# counts table — i.e. a synthetic WFN.h5 with mnband=86 + a 16-device mesh
# would hit the patched code path, not a replicated-counts shortcut.  The
# full integration check (real srun) lives in the wfn_loader service's
# cluster legs and in ``reports/bispinor_ibz_e2e_gate_16gpu_v2_2026-05-16/``.
# ---------------------------------------------------------------------------

def test_helper_is_used_by_read_slabs():
    """Smoke check that ``read_slabs`` source references the helper.

    Guards against future refactors that inline the clamp logic and
    re-introduce a per-rank-overshoot regression — a regression would
    likely involve someone removing the helper call and putting back
    the replicated-bands_per_rank shortcut.  This is a string check, so
    it's brittle by design: if someone *renames* the helper, they must
    update this test, which forces a thoughtful review of the rename.
    """
    import inspect
    from file_io._slab_io_ffi import _FfiBackend
    src = inspect.getsource(_FfiBackend.read_slabs)
    assert "_derive_window_counts" in src, \
        "read_slabs no longer calls the per-rank clamp helper"
    assert "count_partition_spec" in src, \
        "read_slabs no longer requests sharded counts"


def test_helper_delegates_the_clip():
    """The clip ARITHMETIC must not be respelled in the table builder.

    The structural half of the pair above: a value check cannot tell an
    equivalent re-inline from the shared helper, and an equivalent
    re-inline is how the two spellings started before they diverged.
    String check, brittle by design — renaming ``_derive_valid_shape``
    must force a look at this call site.
    """
    import inspect
    src = inspect.getsource(_derive_window_counts)
    assert "_derive_valid_shape" in src, \
        ("_derive_window_counts no longer delegates to the padded-slab clip "
         "— it has been respelled here, which is exactly how the band bound "
         "drifted from b_hi to mnband last time")


# ===========================================================================
# 5. The window tables are a REQUEST, so a malformed one refuses at the door
# ===========================================================================

@pytest.mark.parametrize(
    "offsets, valid_shapes, ndim, match",
    [
        # row counts disagree: three windows' offsets, two windows' extents
        ([[0, 0], [4, 0], [8, 0]], [[4, 7], [4, 7]], 2,
         "different window counts"),
        # ndim disagrees with the slab shape
        ([[0, 0, 0], [4, 0, 0]], [[4, 7, 2], [4, 7, 2]], 2,
         "window tables have ndim"),
        # ndim disagrees BETWEEN the two tables
        ([[0, 0], [4, 0]], [[4, 7, 2], [4, 7, 2]], 2,
         "window tables have ndim"),
        # a 1-D table is not a table of windows at all
        ([0, 4, 8], [[4, 7], [4, 7], [4, 7]], 2, "must be a 2-D"),
        # negative extents would reach H5Sselect_hyperslab as huge unsigneds
        ([[0, 0], [4, 0]], [[4, 7], [-1, 7]], 2, "negative entry"),
    ],
)
def test_inconsistent_window_tables_refuse(offsets, valid_shapes, ndim, match):
    """``read_slabs`` takes TWO tables describing ONE list of windows.

    Neither mismatch has a downstream symptom: the tables are dispatched
    into a compiled shard_map (whose complaint would name a traced buffer),
    and a table at the wrong ndim is read row-major by the C++ handler as a
    different set of windows entirely — the wrong hyperslab, rc=0.  So the
    disagreement is caught where the caller can still see both shapes.
    """
    with pytest.raises(ValueError, match=match):
        _normalize_window_tables(
            name="wfns/coeffs", offsets=offsets,
            valid_shapes=valid_shapes, ndim=ndim)


def test_the_window_table_check_passes_a_well_formed_request():
    """RED TWIN for the refusals above: the check is not unconditional.

    A refusal that fires on everything is not a check, and the four rows
    above would all pass if ``_normalize_window_tables`` simply raised.
    This is the production shape — three ragged G windows of a
    ``(nb, ns, ngkmax, 2)`` psi slab — coming back as int64 tables.
    """
    off, val = _normalize_window_tables(
        name="wfns/coeffs",
        offsets=[[0, 0, 0, 0], [0, 0, 11, 0], [0, 0, 24, 0]],
        valid_shapes=[[10, 2, 11, 2], [10, 2, 13, 2], [10, 2, 17, 2]],
        ndim=4)
    assert off.shape == (3, 4) and val.shape == (3, 4)
    assert off.dtype == np.int64 and val.dtype == np.int64
