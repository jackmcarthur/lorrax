"""A node rule evaluated sub-tile by sub-tile: the one loop of the direct χ₀ stream.

The stream integrates ``A[o, q, μ, ν] += Σ_n Σ_s w[s, o, n] P_n^s(q)[μ, ν]``:
``P_n`` is one rule node's correlation (two parent operands met by a k
convolution door, its q rows kept), ``s`` its forward and reverse
orientation, ``o`` the outputs (value and slope of every sample).  The
loop is inverted against the tile: for each row pass of the rank's
``(μ_X, ν_Y)`` tile, every node is evaluated on that pass only, and its
rows go straight into the carry.  So

- the operands are built from band-complete factor rows (the ``axis``
  carrier: ψ rows for this rank's μ block and ν block), with one local GEMM
  per node and no exchange inside the node loop;
- an operand and a correlation exist only as one pass's sub-tile;
- a pass is a union of whole centroid orbits on every X shard
  (:func:`orbit_cuts`), so the door's typed unfold reads only the pass's
  own rows;
- the node-to-output weights act in chunks of nodes through the in-place
  block accumulator (``ffi.contour.contour_block_accumulate_local``); its
  terms add in order, so a chunk gives the bytes of one accumulate per node;
- a node may return several planes, each landing in its own block of the
  carry (:class:`Block`): the four-current stream's channel planes of one
  family pair go to their blocks of the packed photon layout.

Sizes come from :data:`runtime.tiles.TILE_BYTES` and the shapes alone
(:func:`plan_rows`, :func:`plan_passes`); the budget never enters, so no
result depends on it.  The χ₀ operands and doors are ``gw.w_isdf``'s (the
response owner), for the charge stream and for the four-current stream
(Dirac-half quadrant Greens and the mode-11 vertex door per family pair):
nodes in chunks into a carry (:func:`stream_passes`).  The Σ G⋆W convolution
(``gw.ppm_tau_kernel``, the Σ owner) runs the same passes inside each τ
node: G(τ) on the pass's ψ rows, W(τ) on the pass's rows of its q parents
through the mode-9 load (:func:`pass_load`) and mode 7, and the band
projection, which is linear in the rows, summed over the passes
(:func:`fold_passes`).
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


def orbit_cuts(lsrc, side, ns):
    """The local centroid rows a row pass may start at: no row on one side reads the other.

    ``lsrc`` ``(nk, side * rows * ns)`` are a plan's X-shard-local merged sources
    (``symmetry_maps.unfold_load_tables``).  A cut ``b`` is admissible when, on
    every k and every X shard, rows below ``b`` read only rows below ``b`` and
    rows at or above it only rows at or above it: a union of whole centroid
    orbits.  Returns the sorted admissible cuts in ``(0, rows)``.
    """
    src = np.asarray(lsrc).reshape(np.asarray(lsrc).shape[0], int(side), -1, int(ns))
    rows = src.shape[2]
    row = np.where(src >= 0, src // int(ns), -1)
    hi = row.max(axis=(0, 1, 3))                                  # per local row
    lo = np.where(row >= 0, row, rows).min(axis=(0, 1, 3))
    below = np.maximum.accumulate(hi)                             # max source of rows < b+1
    above = np.minimum.accumulate(lo[::-1])[::-1]                 # min source of rows >= b
    return tuple(int(b) for b in range(1, rows) if below[b - 1] < b and above[b] >= b)


def row_passes(n_pass, local_rows, cuts):
    """``((x0, xr), ...)``: ``n_pass`` near-equal row passes, each boundary the nearest orbit cut.

    ``None`` when more than one pass is asked and no cut exists.
    """
    n_pass, local_rows = int(n_pass), int(local_rows)
    if n_pass <= 1:
        return ((0, local_rows),)
    if not cuts:
        return None
    bounds = sorted({min(cuts, key=lambda c: abs(c - i * local_rows / n_pass))
                     for i in range(1, n_pass)})
    edges = [0, *bounds, local_rows]
    return tuple((a, b - a) for a, b in zip(edges[:-1], edges[1:]))


def pass_tables(tables, x0, xr, side, ns):
    """A plan's load tables cut to the local centroid rows ``[x0, x0 + xr)`` of every X shard."""
    lsrc = np.asarray(tables.lsrc)
    width = lsrc.shape[1] // int(side)
    cols = np.concatenate([s * width + np.arange(x0 * ns, (x0 + xr) * ns) for s in range(int(side))])
    cut = lsrc[:, cols]
    moved = np.where(cut >= 0, cut - x0 * ns, -1)
    if np.any((cut >= 0) & ((moved < 0) | (moved >= xr * ns))):
        raise ValueError("pass_tables: a row pass reads outside its own rows (not an orbit cut)")
    return tables._replace(lsrc=moved.astype(np.int32), mph=np.asarray(tables.mph)[:, cols])


@dataclass(frozen=True)
class Block:
    """Where one plane of a node's pass rows lands in each rank's carry tile.

    ``m0``/``n0`` are the plane's local row and column offsets (a pass adds
    its ``x0`` to ``m0``).  ``extents`` ``(la, wa, lb, wb)`` are the block's
    logical row/column extents and per-shard widths, so its pad rows and
    columns are not touched; ``None``: every row of the pass and every column
    of the carry tile is live.
    """
    m0: int = 0
    n0: int = 0
    extents: tuple | None = None


@dataclass(frozen=True)
class PassPlan:
    """Row passes ``((x0, xr), ...)`` of every X shard's local rows, nodes per accumulate,
    and the carry blocks of a node's planes (one whole-tile block by default)."""
    passes: tuple
    chunk: int
    blocks: tuple = (Block(),)


def plan_rows(local_rows, row_bytes, cuts):
    """Row passes of ``local_rows`` whose per-row live set ``row_bytes`` fits one tile.

    ``cuts()`` gives the admissible orbit cuts (:func:`orbit_cuts`; the
    intersection over every table the passes' doors read); it is called only
    when more than one pass is needed.  No cut: one pass.
    """
    from runtime.tiles import tile_units
    local_rows = int(local_rows)
    rows = tile_units(row_bytes, local_rows)
    n_pass = -(-local_rows // rows)
    passes = row_passes(n_pass, local_rows, tuple(cuts()) if n_pass > 1 else ())
    return ((0, local_rows),) if passes is None else passes


def plan_passes(tables, mesh, *, ns, row_bytes, chunk_bytes, n_nodes, blocks=(Block(),)):
    """Row passes and the node chunk from :data:`runtime.tiles.TILE_BYTES` and shapes.

    ``chunk_bytes`` are one node's kept rows on the whole local tile, so a
    chunk of nodes fits one tile; ``row_bytes(chunk)`` is one local row's
    live set at that chunk (the two operands, the door's R-space output and
    its transform, the chunk's kept rows), so a pass fits one tile.  Passes
    split at the nearest orbit cuts (:func:`row_passes`); a rank whose rows
    are one orbit keeps one pass.  ``blocks`` (:class:`Block`) place a
    node's planes in the carry.
    """
    from runtime.tiles import tile_units
    side = int(mesh.shape["x"])
    local_rows = int(np.asarray(tables.lsrc).shape[1]) // (side * int(ns))
    chunk = tile_units(chunk_bytes, n_nodes)
    passes = plan_rows(local_rows, row_bytes(chunk), lambda: orbit_cuts(tables.lsrc, side, ns))
    return PassPlan(passes=passes, chunk=int(chunk), blocks=tuple(blocks))


def pass_load(load, mesh, x0, xr, ns=1):
    """Device load tables (``symmetry_maps.DeviceLoadTables``) cut to every X shard's rows ``[x0, x0 + xr)``.

    The device twin of :func:`pass_tables` for a door that reads its tables as
    operands (mathdx mode 9): the left sources and phases are cut and moved
    to the pass's own rows.  The caller cuts at an orbit cut of the host
    tables, so no source leaves the pass.
    """
    c0, c1 = int(x0) * int(ns), (int(x0) + int(xr)) * int(ns)

    def cut(lsrc, mph):
        rows = lsrc[:, c0:c1]
        return jnp.where(rows >= 0, rows - c0, -1).astype(lsrc.dtype), mph[:, c0:c1]
    spec = P(None, "x")
    lsrc, mph = jax.shard_map(cut, mesh=mesh, in_specs=(spec, spec), out_specs=(spec, spec),
                              check_vma=False)(load.lsrc, load.mph)
    return load._replace(lsrc=lsrc, mph=mph)


def fold_passes(passes, step, carry, operands):
    """``carry = step(p, x0, xr, carry, operands)`` over every pass, one pass's temporaries at a time.

    Each later pass reads ``operands`` (the whole-tile inputs it slices)
    from behind an optimization barrier with the running carry, so its
    work cannot be scheduled beside the previous pass's.
    """
    for p, (x0, xr) in enumerate(passes):
        if p:
            carry, operands = jax.lax.optimization_barrier((carry, operands))
        carry = step(p, x0, xr, carry, operands)
    return carry


def band_complete(psi_mun, psi_nmu, mesh):
    """The ``axis`` carrier: ψ rows with every band for this rank's μ block and ν block.

    Face operands are gathered over their band axis once per dispatch, outside
    any node loop; axis operands pass through.
    """
    rows = jax.lax.with_sharding_constraint(psi_mun, NamedSharding(mesh, P(None, None, "x", None)))
    cols = jax.lax.with_sharding_constraint(psi_nmu, NamedSharding(mesh, P(None, None, None, "y")))
    return rows, cols


def projection_complete(psi_left, psi_right, mesh):
    """The band projection's ``axis`` operands (``common.contract_bands``'s axis projector).

    ``psi_left`` ``(nk, m, s, μ)`` with every band and μ on X, ``psi_right``
    ``(nk, s', ν, n)`` with every band and ν on Y: a pass projects its own μ
    rows with no exchange, and one band-block reduce-scatter ends the sum.
    """
    left = jax.lax.with_sharding_constraint(psi_left, NamedSharding(mesh, P(None, None, None, "x")))
    right = jax.lax.with_sharding_constraint(psi_right, NamedSharding(mesh, P(None, None, "y", None)))
    return left, right


def pass_rows(a, mesh, x0, xr, axis, spec=None):
    """Every X shard's local rows ``[x0, x0 + xr)`` of ``a`` on ``axis`` (its X-sharded axis).

    ``spec`` is ``a``'s placement (default: ``axis`` on X, the rest whole), so
    no axis is gathered by the slice.
    """
    if spec is None:
        spec = [None] * a.ndim
        spec[axis] = "x"
        spec = P(*spec)
    return jax.shard_map(lambda t: jax.lax.slice_in_dim(t, x0, x0 + xr, axis=axis),
                         mesh=mesh, in_specs=spec, out_specs=spec, check_vma=False)(a)


def stream_passes(carry, *, mesh, plan, weights, count, node_rows):
    """``carry[o, q, μ, ν] += Σ_n Σ_s weights[s, o, n] node_rows(p, n)[s]`` over every pass ``p``.

    ``carry`` ``[n_out, q, μ, ν]`` at ``P(None, None, 'x', 'y')`` is updated in
    place; ``weights`` ``[2, n_out, n_cap]``; ``count`` the live node prefix
    (traced).  ``node_rows(p, n)`` returns pass ``p``'s ``[n_blocks, 2, q,
    px*xr, ν_b]`` rows of node ``n`` at ``P(None, None, None, 'x', 'y')``,
    plane ``b`` landing in ``plan.blocks[b]``.  Nodes run in chunks of
    ``plan.chunk``; a chunk's nodes past ``count`` are not evaluated and add
    exact zeros.
    """
    from ffi.contour import contour_block_accumulate_local
    chunk = int(plan.chunk)
    n_cap = int(weights.shape[-1])
    n_chunks_cap = -(-n_cap // chunk)
    weights = jnp.pad(weights, ((0, 0), (0, 0), (0, n_chunks_cap * chunk - n_cap)))
    n_chunks = ((count + chunk - 1) // chunk).astype(jnp.int32)
    spec_c = P(None, None, "x", "y")
    for p, (x0, xr) in enumerate(plan.passes):
        shape = jax.eval_shape(lambda n, p=p: node_rows(p, n), jnp.zeros((), jnp.int32))

        def skipped(shape=shape):
            return jax.lax.with_sharding_constraint(jnp.zeros(shape.shape, shape.dtype),
                                                    NamedSharding(mesh, P(None, None, None, "x", "y")))

        def block_add(acc, rows, projection, block, x0=x0, xr=xr):
            def local(a, r, w):
                if block.extents is None:
                    valid = jnp.asarray([xr, a.shape[3]], jnp.int32)
                else:
                    la, wa, lb, wb = block.extents
                    valid = jnp.stack((
                        jnp.clip(la - jax.lax.axis_index("x") * wa - x0, 0, xr),
                        jnp.clip(lb - jax.lax.axis_index("y") * wb, 0, wb))).astype(jnp.int32)
                return contour_block_accumulate_local(a, r, w, valid, m0=block.m0 + x0,
                                                      n0=block.n0, mesh=mesh)
            return jax.shard_map(local, mesh=mesh, in_specs=(spec_c, spec_c, P()),
                                 out_specs=spec_c, check_vma=False)(acc, rows, projection)

        def chunk_body(state, p=p, skipped=skipped):
            c, acc = state

            def one(_, j):
                n = c * chunk + j
                return None, jax.lax.cond(n < count, lambda: node_rows(p, n), skipped)
            _, rows = jax.lax.scan(one, None, jnp.arange(chunk, dtype=jnp.int32), unroll=1)
            # Read behind a barrier: a counter-indexed slice must not be
            # rematerialized after the counter's in-place increment.
            w = jax.lax.optimization_barrier(
                jax.lax.dynamic_slice_in_dim(weights, c * chunk, chunk, axis=2))
            projection = jnp.transpose(w, (2, 0, 1)).reshape(2 * chunk, -1)
            for b, block in enumerate(plan.blocks):
                # [chunk, b, 2, q, m, n] -> terms in node order, forward then reverse.
                plane = rows[:, b].reshape((2 * chunk,) + rows.shape[3:])
                acc = block_add(acc, plane, projection, block)
            return c + 1, acc

        _, carry = jax.lax.while_loop(lambda s: s[0] < n_chunks, chunk_body,
                                      (jnp.zeros((), jnp.int32), carry))
    return carry
