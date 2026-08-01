"""Centroid-specific distribution policy — a thin layer over the service.

``common.collectives`` is THE device layer.  This module owns only what is
specific to selecting ISDF centroids and would be wrong to generalise:

* :func:`build_mesh` — the k-means **latency-floor policy**.  Mesh
  CONSTRUCTION is ``collectives.resolve_mesh`` / ``single_device_mesh``;
  what lives here is the measured decision of when sharding the real-space
  grid is worth its allreduce (see ``_P_PER_SHARD_MIN``), which is a
  property of the Lloyd step's arithmetic intensity, not of the machine.
* :func:`grid_parallel_loop` / :func:`sum_over_grid` — the Lloyd
  ``shard_map`` pattern: real-space grid distributed in, centroids
  replicated out, two psums per iteration in the body.
* :func:`place`, :func:`require_axes`, :func:`n_shards` — spelling
  adapters over the service, kept so the physics files never name
  ``NamedSharding`` / ``PartitionSpec``.  Requests 1-3 in this
  workstream's report propose moving them into ``common.collectives``,
  at which point this module keeps only the two items above.

There is deliberately NO ``single_device_mesh`` / ``process_local_mesh``
here: both are ``common.collectives.single_device_mesh``.  An earlier
draft of this file defined its own, with the OPPOSITE meaning (global
device 0, refusing at P>1) — a name collision the orchestrator ruled on;
it is deleted rather than renamed, because there was never a second thing
to name.
"""
from __future__ import annotations

import jax
from jax import lax, config
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.experimental.shard_map import shard_map

from common.collectives import (
    device_put_process_local,
    process_count,
    prepare_mesh,
    resolve_mesh,
    single_device_mesh,
)

# The centroid geometry is fp64 throughout: a fractional coordinate has to
# resolve 1/N_fft of a lattice vector, and the Lloyd movement tolerance is
# 5e-3 Angstrom.  The drivers' startup call already exports JAX_ENABLE_X64=1
# for every driver; this covers a bare ``import centroid.<...>`` that did
# not go through it.
config.update("jax_enable_x64", True)


MESH_AXES = ("x", "y")
"""Axis names of every centroid mesh — the same pair ``gw_jax`` uses, so a
centroid set and the GW run that consumes it agree on device layout."""

_P_PER_SHARD_MIN = 100_000
"""NCCL-latency floor: shard only when each device sees >= this many grid
points (measured on Si 4x4x4: P=110k / 4 GPUs ran SLOWER than one device
because the allreduce dominated the 1 ms of local compute)."""


# ─────────────────────────────────────────────────────────────────────────
# Mesh policy
# ─────────────────────────────────────────────────────────────────────────

def build_mesh(n_points: int, *, shard: bool = True,
               force_shard: bool = False, print_fn=print) -> Mesh:
    """The device mesh for a k-means over ``n_points`` real-space points.

    Returns the run's square 2-D ``('x','y')`` mesh, or the 1x1
    process-local one when ``shard`` is off, when there is a single device,
    or when the per-device share of the grid is below the latency floor
    (the only decision this function actually makes — construction and the
    addressability check are ``common.collectives``').  The single-device
    case is still 2-D so the downstream pipeline
    (``load_centroids_band_chunked`` and friends) has one code path.

    ``force_shard`` overrides the latency floor (benchmarking).  A
    multi-process run NEVER falls back to one device: the other ranks
    would sit on collectives against a mesh they are not in and the
    distributed shutdown barrier would hang for minutes.
    """
    multi_host = process_count() > 1

    if not shard:
        if multi_host:
            raise ValueError(
                f"build_mesh: --no-shard on {process_count()} processes. The "
                f"k-means and the pivoted-Cholesky prune both run collectives "
                f"over this mesh, so a per-rank 1x1 mesh would have each rank "
                f"reducing against a communicator the others are not in and "
                f"the shutdown barrier would hang. Drop --no-shard, or run on "
                f"one process."
            )
        return single_device_mesh()

    # prepare_mesh, not resolve_mesh: without the clique warm-up the first
    # Lloyd collective fires from an XLA pool thread and jaxlib refuses
    # communicator creation at P>1 (job 7884867, 16/16 ranks).
    mesh = prepare_mesh(axis_names=MESH_AXES)
    nx, ny = (int(mesh.shape[a]) for a in MESH_AXES)
    n_dev = nx * ny
    if n_dev < 2:
        return mesh
    per_shard = n_points // n_dev

    if per_shard < _P_PER_SHARD_MIN and not force_shard and not multi_host:
        print_fn(f"P/{n_dev} = {per_shard} < {_P_PER_SHARD_MIN} points per "
                 "shard; falling back to single-device. Pass --force-shard to "
                 "override.")
        return single_device_mesh()
    if multi_host and per_shard < _P_PER_SHARD_MIN:
        print_fn(f"P/{n_dev} = {per_shard} < {_P_PER_SHARD_MIN}; sharding "
                 "anyway (multi-host: single-device fallback would deadlock).")

    print_fn(f"Sharded mesh: ('x'={nx}, 'y'={ny}) over {n_dev} devices")
    return mesh


def require_axes(mesh: Mesh, axes, who: str) -> tuple[str, ...]:
    """Check ``axes`` are all present in ``mesh``; return them as a tuple."""
    names = (axes,) if isinstance(axes, str) else tuple(axes)
    missing = [a for a in names if a not in mesh.axis_names]
    if missing:
        raise ValueError(
            f"{who}: mesh axis {missing} not in mesh {mesh.axis_names}. Build "
            f"the mesh with centroid.distribution.build_mesh (axes "
            f"{MESH_AXES})."
        )
    return names


def n_shards(mesh: Mesh, axes) -> int:
    """Number of shards along ``axes`` of ``mesh``."""
    n = 1
    for a in ((axes,) if isinstance(axes, str) else tuple(axes)):
        n *= int(mesh.shape[a])
    return n


# ─────────────────────────────────────────────────────────────────────────
# Placement
# ─────────────────────────────────────────────────────────────────────────

def place(host_array, mesh: Mesh, *axes):
    """Put a host array onto ``mesh`` with layout ``axes``, no collective.

    ``axes`` is a ``PartitionSpec`` spelled positionally: ``place(x, mesh)``
    replicates a scalar/matrix, ``place(x, mesh, grid_axis)`` splits a
    1-D grid quantity, ``place(x, mesh, grid_axis, None)`` splits the rows
    of a ``(P, 3)`` position table.

    NOT ``jax.device_put``: on a multi-process mesh that runs a hidden
    ``multihost_utils.assert_equal``, a P-linear all-gather of the operand
    (scorecard AA.1) -- and the operands here are the full FFT grid, the
    largest host arrays the k-means driver touches.  The precondition it
    was asserting (identical on every rank) holds by construction: every
    input is a pure function of the WFN density plus the seed.
    ``LORRAX_CHECK_REPLICA=1`` restores the assertion for a debugging run.
    """
    return device_put_process_local(
        host_array, NamedSharding(mesh, PartitionSpec(*axes)))


# ─────────────────────────────────────────────────────────────────────────
# Grid-parallel iteration
# ─────────────────────────────────────────────────────────────────────────

def sum_over_grid(x, grid_axis):
    """Sum a per-centroid accumulator over every rank's slice of the grid.

    Callable only from inside a :func:`grid_parallel_loop` body.  This is
    the one collective the Lloyd iteration needs: the assignment step is
    embarrassingly parallel over grid points, and the update step needs
    each centroid's total weight and weighted displacement over ALL points.
    """
    return lax.psum(x, axis_name=grid_axis)


def grid_parallel_loop(mesh: Mesh, iterate, *, grid_axis=MESH_AXES,
                       who: str = "grid_parallel_loop"):
    """Compile a converge-until-done loop whose parallel axis is the grid.

    ``iterate(positions, centroids, rho, metric)`` is called ONCE, inside a
    ``shard_map``, and sees this rank's slice of ``positions`` / ``rho``
    plus the full replicated centroid list.  It must return
    ``(centroids, steps, max_movement_sq)`` -- i.e. it owns the whole
    convergence loop, so the iteration never returns to the host and the
    Lloyd sweep costs zero syncs per step.

    Distribution contract, in and out::

        positions (P, 3)  split on grid_axis     centroids (C, 3) replicated
        centroids (C, 3)  replicated             steps     ()     replicated
        rho       (P,)    split on grid_axis     max_mv_sq ()     replicated
        metric    (3, 3)  replicated
    """
    grid_axis = require_axes(mesh, grid_axis, who)
    grid_axis = grid_axis[0] if len(grid_axis) == 1 else grid_axis

    in_specs = (
        PartitionSpec(grid_axis, None),   # positions
        PartitionSpec(None, None),        # centroids
        PartitionSpec(grid_axis),         # rho
        PartitionSpec(None, None),        # metric
    )
    out_specs = (
        PartitionSpec(None, None),        # centroids
        PartitionSpec(),                  # steps
        PartitionSpec(),                  # max movement^2
    )

    @jax.jit
    def run(positions, centroids, rho, metric):
        return shard_map(
            iterate, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
            check_rep=False,
        )(positions, centroids, rho, metric)
    return run


# ─────────────────────────────────────────────────────────────────────────
# Run banner (topology comes from the service; only the wording is local)
# ─────────────────────────────────────────────────────────────────────────

def device_summary() -> str:
    """One line naming the devices this run has, for the driver's banner."""
    from common.collectives import device_count, process_rank_world
    rank, world = process_rank_world()
    return (f"{device_count()} device(s) "
            f"(local: {len(jax.local_devices())}, proc {rank}/{world})")
