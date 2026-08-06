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

import pytest

from file_io._slab_io_ffi import (
    _derive_valid_shape,
    _mpi_world_verdict,
    _normalize_valid_shape,
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
