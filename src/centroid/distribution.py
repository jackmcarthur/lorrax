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
* :func:`place_grid`, :func:`place_points`, :func:`grid_argmax`,
  :func:`grid_points` — the grid-sharded seed and Lloyd inputs and small
  selected-point collectives. A seed
  broadcasts its chosen point without gathering the full grid.

There is deliberately NO ``single_device_mesh`` / ``process_local_mesh``
here: both are ``common.collectives.single_device_mesh``.  An earlier
draft of this file defined its own, with the OPPOSITE meaning (global
device 0, refusing at P>1) — a name collision the orchestrator ruled on;
it is deleted rather than renamed, because there was never a second thing
to name.
"""
from __future__ import annotations

from functools import partial

import jax
from lxkit import ordered_psum
from jax import lax, config
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax import shard_map

from common.collectives import (
    device_put_process_local,
    device_put_process_tiles,
    process_count,
    prepare_mesh,
    resolve_mesh,
    single_device_mesh,
)
from common.log import get_logger
from runtime.padding import spec_divisor

log = get_logger(__name__)

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

def build_mesh(n_points: int) -> Mesh:
    """The device mesh for a k-means over ``n_points`` real-space points.

    Returns the run's square 2-D ``('x','y')`` mesh, or the 1x1
    process-local one when there is a single device, or when one process
    drives several devices and the per-device share of the grid is below
    the latency floor (the only decision this function actually makes —
    construction and the addressability check are ``common.collectives``').
    The single-device case is still 2-D so the downstream pipeline
    (``load_centroids_band_chunked`` and friends) has one code path.

    A multi-process run ALWAYS shards: on one device per process the other
    ranks would sit on collectives against a mesh they are not in and the
    distributed shutdown barrier would hang for minutes.  (The
    ``--no-shard`` / ``--force-shard`` overrides were removed 2026-10-21:
    neither touched a one-device run or a multi-process one, and the floor
    they bypassed applies only to one process driving several devices.)
    """
    multi_host = process_count() > 1

    # prepare_mesh, not resolve_mesh: without the clique warm-up the first
    # Lloyd collective fires from an XLA pool thread and jaxlib refuses
    # communicator creation at P>1 (job 7884867, 16/16 ranks).
    mesh = prepare_mesh(axis_names=MESH_AXES)
    nx, ny = (int(mesh.shape[a]) for a in MESH_AXES)
    n_dev = nx * ny
    if n_dev < 2:
        return mesh
    per_shard = n_points // n_dev

    if per_shard < _P_PER_SHARD_MIN and not multi_host:
        log.info(f"P/{n_dev} = {per_shard} < {_P_PER_SHARD_MIN} points per "
                 "shard; falling back to single-device.")
        return single_device_mesh()
    if multi_host and per_shard < _P_PER_SHARD_MIN:
        log.info(f"P/{n_dev} = {per_shard} < {_P_PER_SHARD_MIN}; sharding "
                 "anyway (multi-host: single-device fallback would deadlock).")

    log.debug(f"Sharded mesh: ('x'={nx}, 'y'={ny}) over {n_dev} devices")
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
    entry = axes if isinstance(axes, str) else tuple(axes)
    return spec_divisor(mesh, (entry,), axis=0)


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


def place_grid(rho, mesh: Mesh, grid_axis=MESH_AXES):
    """Build fractional grid positions and density directly on grid shards.

    Only each process's density slice reaches its device. Fractional
    positions use the same linspace values as the logical FFT grid, with
    zero positions and weights beyond its padded extent.
    """
    import numpy as np
    import jax.numpy as jnp
    from runtime.padding import padded_axis

    axes = require_axes(mesh, grid_axis, 'place_grid')
    axis = axes[0] if len(axes) == 1 else axes
    nx, ny, nz = map(int, rho.shape)
    tag = padded_axis(nx * ny * nz, n_shards(mesh, axes), name='k-means grid')
    positions_sharding = NamedSharding(mesh, PartitionSpec(axis, None))

    @partial(jax.jit, out_shardings=positions_sharding)
    def positions():
        index = jnp.arange(tag.carrier)
        return jnp.where((index < tag.logical)[:, None],
                         _grid_fractions(index, (nx, ny, nz)), 0.)

    density = device_put_process_tiles(
        (tag.carrier,), NamedSharding(mesh, PartitionSpec(axis)),
        _padded_tile(np.asarray(rho, dtype=np.float64).reshape(-1), tag))
    return positions(), density, tag


def place_points(flat_index, weights, grid_shape, mesh: Mesh,
                 grid_axis=MESH_AXES):
    """:func:`place_grid` for a subset of the grid: the fractional
    positions of the logical points ``flat_index`` (C order over
    ``grid_shape``) and their ``weights``, on grid shards.  Padding rows
    carry zero position and zero weight."""
    import numpy as np
    import jax.numpy as jnp
    from runtime.padding import padded_axis

    axes = require_axes(mesh, grid_axis, 'place_points')
    axis = axes[0] if len(axes) == 1 else axes
    shape = tuple(int(n) for n in grid_shape)
    tag = padded_axis(len(flat_index), n_shards(mesh, axes),
                      name='k-means points')
    rows = NamedSharding(mesh, PartitionSpec(axis))
    index = device_put_process_tiles(
        (tag.carrier,), rows,
        _padded_tile(np.asarray(flat_index, dtype=np.int64), tag))
    density = device_put_process_tiles(
        (tag.carrier,), rows,
        _padded_tile(np.asarray(weights, dtype=np.float64), tag))

    @partial(jax.jit, out_shardings=NamedSharding(mesh, PartitionSpec(axis, None)))
    def positions(index):
        valid = jnp.arange(tag.carrier) < tag.logical
        return jnp.where(valid[:, None], _grid_fractions(index, shape), 0.)

    return positions(index), density, tag


def _grid_fractions(index, shape):
    """(n, 3) fractional coordinates of flat C-order grid indices, from the
    same ``linspace`` values for every placement (``index`` past the grid
    is clamped; callers mask it)."""
    import jax.numpy as jnp
    nx, ny, nz = shape
    fx, fy, fz = (jnp.linspace(0, 1, n, endpoint=False, dtype=jnp.float64)
                  for n in (nx, ny, nz))
    return jnp.stack((fx[jnp.minimum(index // (ny*nz), nx-1)],
                      fy[(index // nz) % ny], fz[index % nz]), axis=-1)


def _padded_tile(values, tag):
    """``device_put_process_tiles`` callback: this shard's slice of the
    logical ``values``, zero past ``tag.logical``."""
    import numpy as np

    def tile(index):
        begin, end = index[0].start or 0, index[0].stop or tag.carrier
        out = np.zeros(end-begin, dtype=values.dtype)
        valid_end = min(end, tag.logical)
        if begin < valid_end:
            out[:valid_end-begin] = values[begin:valid_end]
        return out
    return tile


def grid_points(positions, index, mesh: Mesh, grid_axis=MESH_AXES):
    """Return selected fractional points without gathering the grid.

    Exactly one grid shard owns each index. Other ranks contribute zeros,
    so the collective moves only the selected rows and preserves their bits.
    """
    import jax.numpy as jnp
    axes = require_axes(mesh, grid_axis, 'grid_points')
    axis = axes[0] if len(axes) == 1 else axes

    @partial(shard_map, mesh=mesh,
             in_specs=(PartitionSpec(axis, None), PartitionSpec()),
             out_specs=PartitionSpec(), check_vma=False)
    def selected(points, chosen):
        local = chosen - lax.axis_index(axis) * points.shape[0]
        row = points[jnp.clip(local, 0, points.shape[0]-1)]
        valid = (local >= 0) & (local < points.shape[0])
        row = jnp.where(valid[..., None], row, 0.)
        return lax.psum(row, axis)  # exact-reduce: one owner supplies the point, other ranks zero

    return selected(positions, index)


def grid_argmax(positions, scores, mesh: Mesh, grid_axis=MESH_AXES):
    """Stable global score maximum and its point in one small collective.

    Each rank contributes its local best score, index and coordinates.
    The index uses two exactly representable 32-bit words in the FP64
    message; ties retain the lowest global index, as ``argmax`` does.
    """
    import jax.numpy as jnp
    axes = require_axes(mesh, grid_axis, 'grid_argmax')
    axis = axes[0] if len(axes) == 1 else axes

    @partial(shard_map, mesh=mesh,
             in_specs=(PartitionSpec(axis, None), PartitionSpec(axis)),
             out_specs=PartitionSpec(), check_vma=False)
    def selected(points, values):
        local = jnp.argmax(values)
        index = local + jnp.int64(lax.axis_index(axis)) * points.shape[0]
        word = jnp.int64(2**32)
        candidate = jnp.concatenate((values[local, None],
            jnp.asarray([index // word, index % word], jnp.float64), points[local]))
        candidates = lax.all_gather(candidate, axis, axis=0, tiled=False)
        best = candidates[:, 0] == jnp.max(candidates[:, 0])
        high = jnp.min(jnp.where(best, candidates[:, 1], jnp.inf))
        best &= candidates[:, 1] == high
        low = jnp.min(jnp.where(best, candidates[:, 2], jnp.inf))
        winner = jnp.argmax(best & (candidates[:, 2] == low))
        return candidates[winner, 3:]

    return selected(positions, scores)


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
    return ordered_psum(x, axis_name=grid_axis)


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
            check_vma=False,
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
