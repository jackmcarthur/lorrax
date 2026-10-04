"""A node rule evaluated sub-tile by sub-tile: the one loop of the direct χ₀ stream.

The stream integrates ``A[o, q, μ, ν] += Σ_n Σ_s w[s, o, n] P_n^s(q)[μ, ν]``:
``P_n`` is one rule node's correlation (two parent operands met by a
kconv call, its q rows kept), ``s`` its forward and reverse
orientation, ``o`` the outputs (value and slope of every sample).  The
loop is inverted against the tile: for each row pass of the rank's
``(μ_X, ν_Y)`` tile, every node is evaluated on that pass only, and its
rows go straight into the carry.  So

- the operands are built from band-complete factor rows (the ``axis``
  carrier: ψ rows for this rank's μ block and ν block), with one local GEMM
  per node and no exchange inside the node loop;
- an operand and a correlation exist only as one pass's sub-tile;
- a pass is a union of whole centroid orbits on every X shard
  (:func:`orbit_cuts`), so the kconv call's typed unfold reads only the pass's
  own rows;
- the node-to-output weights act in chunks of nodes through the in-place
  block accumulator (``ffi.contour.contour_block_accumulate_local``); its
  terms add in order, so a chunk gives the bytes of one accumulate per node;
- a node may return several planes, each landing in its own block of the
  carry (:class:`Block`): the four-current stream's channel planes of one
  family pair go to their blocks of the packed photon layout.

Sizes come from :data:`runtime.tiles.TILE_BYTES` and the shapes alone
(:func:`plan_windows`, :func:`plan_passes`); the budget never enters, so no
result depends on it.  The χ₀ operands and kconv calls are ``gw.w_isdf``'s (the
response owner), for the charge stream and for the four-current stream
(Dirac-half quadrant Greens and the mode-11 vertex kconv call per family pair):
nodes in chunks into a carry (:func:`stream_passes`).  The Σ G⋆W convolution
(``gw.ppm_tau_kernel``, the Σ owner) runs the same passes inside each τ
node: G(τ) on the pass's ψ rows, W(τ) on the pass's rows of its q parents
through the mode-9 load and mode 7, and the band projection, which is
linear in the rows, summed over the passes.  Those passes are equal
orbit-aligned windows (:func:`plan_windows`) run as one ``lax.scan``
(:func:`scan_passes`): each pass slices its window by a traced offset
(:func:`window_rows`), and its tables are cut from the placed ones on the
device (:func:`window_load`), so the program does not grow with the pass count.  The static Σ (exchange G(0⁻)⋆V, static SX and COH)
is the same node at τ = 0 with the static interaction in place of W(τ).
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
    the window rows, the windows and the carry blocks of a node's planes.

    Every pass is one window of ``rows`` rows (:func:`plan_windows`):
    ``windows`` ``((s, lo, hi), ...)``, ``passes`` their live spans, so one
    program serves every pass (:func:`stream_passes`).
    """
    passes: tuple
    chunk: int
    rows: int
    windows: tuple
    blocks: tuple = (Block(),)


def plan_passes(tables, mesh, *, ns, row_bytes, chunk_bytes, n_nodes, blocks=(Block(),)):
    """Row-pass windows and the node chunk from :data:`runtime.tiles.TILE_BYTES` and shapes.

    ``chunk_bytes`` are one node's kept rows on the whole local tile, so a
    chunk of nodes fits one tile; ``row_bytes(chunk)`` is one local row's
    live set at that chunk (the two operands, the kconv call's R-space output and
    its transform, the chunk's kept rows), so a window fits one tile.  The
    passes are equal orbit-aligned windows (:func:`plan_windows`); a rank
    whose rows are one orbit keeps one pass.  ``blocks`` (:class:`Block`)
    place a node's planes in the carry.
    """
    from runtime.tiles import tile_units
    side = int(mesh.shape["x"])
    local_rows = int(np.asarray(tables.lsrc).shape[1]) // (side * int(ns))
    chunk = tile_units(chunk_bytes, n_nodes)
    cuts = lambda: orbit_cuts(tables.lsrc, side, ns)
    R, win = plan_windows(local_rows, row_bytes(chunk), cuts)
    return PassPlan(passes=tuple((s + lo, hi - lo) for s, lo, hi in win), chunk=int(chunk),
                    rows=int(R), windows=win, blocks=tuple(blocks))


def plan_windows(local_rows, row_bytes, cuts):
    """Equal orbit-aligned windows of a scanned pass loop: ``(R, windows)``.

    ``g`` is the orbit block, the most common spacing of the admissible cuts
    (``cuts()``, :func:`orbit_cuts`; called only when one tile does not hold
    every row), and ``g·max(1, round(tile_rows / g))`` the orbit-aligned size
    nearest the tile (:data:`runtime.tiles.TILE_BYTES` over ``row_bytes``).
    Spans of at most that size ending on cuts (a longer one where no cut
    allows less) fix the pass count; the window ``R`` is then the shortest
    longest span that keeps it, so every window is as full as the cuts allow
    (a stored segment carries ``R`` rows, its dead rows ``passes·R − L``).
    Pass ``p`` reads the window ``[s, s + R)`` of every X
    shard's local rows, ``s = min(x0, local_rows - R)``, and its live rows are
    ``[lo, hi)`` of the window: ``windows`` is ``((s, lo, hi), ...)``.  One
    window holding every row: ``(local_rows, ((0, 0, local_rows),))``.
    """
    from runtime.tiles import tile_units
    L = int(local_rows)
    rows = tile_units(row_bytes, L)
    whole = (L, ((0, 0, L),))
    if rows >= L:
        return whole
    ends = sorted(set(int(c) for c in cuts()) | {L})
    if len(ends) == 1:
        return whole
    g = int(np.bincount(np.diff([0, *ends])).argmax())

    def spans_within(target):
        spans, start = [], 0
        while start < L:
            fit = [c for c in ends if start < c <= start + target]
            end = max(fit) if fit else min(c for c in ends if c > start)
            spans.append((start, end - start))
            start = end
        return spans
    spans = spans_within(g * max(1, int(round(rows / g))))
    if len(spans) == 1:
        return whole
    # The shortest span bound that keeps the pass count (bisection: the count
    # only falls as the bound grows).
    lo, hi = -(-L // len(spans)), max(xr for _, xr in spans)
    while lo < hi:
        mid = (lo + hi) // 2
        trial = spans_within(mid)
        if len(trial) <= len(spans) and max(xr for _, xr in trial) <= mid:
            hi = mid
        else:
            lo = mid + 1
    spans = spans_within(hi)
    R = max(xr for _, xr in spans)
    return R, tuple((min(x0, L - R), x0 - min(x0, L - R), x0 - min(x0, L - R) + xr)
                    for x0, xr in spans)


def window_tables(tables, R, side, ns):
    """Host load tables of an ``R``-row window: the shapes a kernel factory reads.

    Every left source is -1 and the left phases are the first window's; the
    content each pass reads comes from the placed tables cut by
    :func:`window_load`.
    """
    lsrc = np.asarray(tables.lsrc)
    width = lsrc.shape[1] // int(side)
    cols = np.concatenate([s * width + np.arange(int(R) * int(ns)) for s in range(int(side))])
    return tables._replace(lsrc=np.full((lsrc.shape[0], cols.size), -1, np.int32),
                           mph=np.asarray(tables.mph)[:, cols])


def window_load(load, mesh, s, lo, hi, R, ns=1):
    """Placed load tables (``symmetry_maps.DeviceLoadTables``) cut to every X shard's window ``[s, s + R)``.

    ``s``, ``lo`` and ``hi`` are traced.  The left sources move to the window
    and are -1 outside its live rows ``[lo, hi)`` (read as exact zeros; the
    k-convolution also takes ``live = [lo, hi)`` and skips those rows); a pass
    is a union of whole orbits, so no live source leaves the window.
    """
    width = int(R) * int(ns)

    def cut(lsrc, mph, s, lo, hi):
        rows = jax.lax.dynamic_slice_in_dim(lsrc, s * ns, width, axis=1)
        live = (jnp.arange(width) // ns >= lo) & (jnp.arange(width) // ns < hi)
        rows = jnp.where((rows >= 0) & live[None, :], rows - s * ns, -1).astype(lsrc.dtype)
        return rows, jax.lax.dynamic_slice_in_dim(mph, s * ns, width, axis=1)
    spec = P(None, "x")
    lsrc, mph = jax.shard_map(cut, mesh=mesh, in_specs=(spec, spec, P(), P(), P()),
                              out_specs=(spec, spec), check_vma=False)(load.lsrc, load.mph, s, lo, hi)
    return load._replace(lsrc=lsrc, mph=mph)


def window_rows(a, mesh, s, R, axis, spec=None, live=None):
    """Every X shard's local rows ``[s, s + R)`` of ``a`` on ``axis`` (its X-sharded axis), ``s`` traced.

    ``spec`` is ``a``'s placement (default: ``axis`` on X, the rest whole).
    ``live`` ``(lo, hi)``: rows of the window outside it are zeroed.
    """
    if spec is None:
        spec = [None] * a.ndim
        spec[axis] = "x"
        spec = P(*spec)

    def cut(t, s, lo, hi):
        t = jax.lax.dynamic_slice_in_dim(t, s, int(R), axis=axis)
        if live is None:
            return t
        keep = (jnp.arange(int(R)) >= lo) & (jnp.arange(int(R)) < hi)
        shape = [1] * t.ndim
        shape[axis] = int(R)
        return jnp.where(keep.reshape(shape), t, jnp.zeros((), t.dtype))
    lo, hi = (jnp.int32(0), jnp.int32(R)) if live is None else live
    return jax.shard_map(cut, mesh=mesh, in_specs=(spec, P(), P(), P()), out_specs=spec,
                         check_vma=False)(a, s, lo, hi)


#: The placement of :func:`green_rows`: μ on X, every spin and band local.
GREEN_ROWS_SPEC = P(None, "x", None, None)


def green_rows(psi_mun, mesh):
    """A row-pass Green's band-complete ψ rows, μ-major ``(nk, μ, s, n)`` (placed once per call).

    A window of μ rows is then one contiguous block, sliced in place by a
    scanned pass (:func:`window_green_rows`).  Spin-major rows ``(nk, s, μ, n)``
    make XLA lay the scan's loop state out μ-major, a transpose of the whole
    operand per call (1.38 GB per rank at the Fe/Ni 20³ P64-local Σ tile).
    """
    return jax.lax.with_sharding_constraint(jnp.transpose(psi_mun, (0, 2, 1, 3)),
                                            NamedSharding(mesh, GREEN_ROWS_SPEC))


def window_green_rows(rows, mesh, s=None, R=None):
    """:func:`green_rows`' window ``[s, s + R)`` (every row when ``s`` is None) in the Green
    build's order ``(nk, s, μ, n)``: the transpose cancels against the build's
    centroid-major merge (``common.contract_bands.merge_spin_centroid``)."""
    if s is not None:
        rows = window_rows(rows, mesh, s, R, axis=1, spec=GREEN_ROWS_SPEC)
    return jnp.transpose(rows, (0, 2, 1, 3))


def scan_passes(windows, step, carry):
    """``carry = step(s, lo, hi, carry)`` over every window (:func:`plan_windows`) in one ``lax.scan``.

    The windows' ``(s, lo, hi)`` rows are the scanned operand, so the program
    is one body whatever the pass count, and one window's temporaries are live
    at a time.
    """
    table = jnp.asarray(np.asarray(windows, np.int32))

    def body(c, w):
        # Read behind a barrier: a counter-indexed slice must not be
        # rematerialized after the counter's in-place increment (R82).
        s, lo, hi = jax.lax.optimization_barrier((w[0], w[1], w[2]))
        return step(s, lo, hi, c), None
    carry, _ = jax.lax.scan(body, carry, table, unroll=1)
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


def segment_blocks(plan, p, cols):
    """Pass ``p`` of ``plan`` as one compact segment of the carry tile (the streamed bank).

    The pass's planes keep their order: each distinct block row offset gets a
    band of the pass's ``xr`` rows, each distinct column offset a band of its
    block's width (``cols``, the tile's local columns, for a whole-tile block).
    Returns ``(blocks, rows, width, rects)``: the planes' :class:`Block`
    placements in the ``[rows, width]`` local segment, and the rectangles
    ``(r0, c0, R0, C0, nr, nc)`` that put segment ``[r0:r0+nr, c0:c0+nc]`` back
    at ``[R0:R0+nr, C0:C0+nc]`` of the local tile.
    """
    x0, xr = plan.passes[p]
    # Every segment has one shape: the window's R rows, its live rows
    # [lo, hi) the ones that go back to the tile.
    R, lo = int(plan.rows), int(plan.windows[p][1])
    width = {b.n0: int(cols) if b.extents is None else int(b.extents[3]) for b in plan.blocks}
    row = {m: i * R for i, m in enumerate(sorted({b.m0 for b in plan.blocks}))}
    col, c = {}, 0
    for n0 in sorted(width):
        col[n0], c = c, c + width[n0]
    blocks = tuple(Block(m0=row[b.m0], n0=col[b.n0], extents=b.extents) for b in plan.blocks)
    rects = tuple((row[b.m0] + lo, col[b.n0], b.m0 + x0, b.n0, xr, width[b.n0])
                  for b in plan.blocks)
    return blocks, len(row) * R, c, rects


def stream_passes(carry, *, mesh, plan, weights, count, node_rows, prepare, only=None):
    """``carry[o, q, μ, ν] += Σ_n Σ_s weights[s, o, n] P_p(n)[s]`` over every pass ``p``.

    ``carry`` ``[n_out, q, μ, ν]`` at ``P(None, None, 'x', 'y')`` is updated in
    place; ``weights`` ``[n_sets, n_out, n_cap]`` (the direct stream's two
    orientations, or any fixed set of rows per node); ``count`` the live node
    prefix (traced).  Every pass runs through one body: ``prepare(window)``
    forms a window's operands once, outside the node loop (``window`` is
    ``(s, lo, hi)`` traced, or ``None`` for the one whole-tile window), and
    ``node_rows(operands, n)`` returns node ``n``'s ``[n_blocks, n_sets, q,
    px·R, ν_b]`` window rows at ``P(None, None, None, 'x', 'y')``, plane ``b``
    landing in ``plan.blocks[b]``.  Nodes run in chunks of ``plan.chunk``; a
    chunk's nodes past ``count`` are not evaluated and add exact zeros.

    ``only`` (a traced pass index): ``carry`` is that pass's segment
    (:func:`segment_blocks`) and the body runs once on its window; every
    element gets the same terms in the same order, so its bytes are the whole
    tile's (the streamed bank, ``file_io.slab_io.StreamedBank``).  Otherwise
    the carry is the whole tile: one window holding every row runs as is,
    more run as one ``lax.scan`` over the windows, each window's planes added
    at its traced row offset (the contour accumulate's runtime origin).  A
    window's rows outside its live rows carry exact zeros (their sources are
    -1; the k-convolution also skips them), so adding a whole window adds the
    pass.
    """
    from ffi.contour import contour_block_accumulate_local
    chunk, R = int(plan.chunk), int(plan.rows)
    n_sets, n_cap = int(weights.shape[0]), int(weights.shape[-1])
    n_chunks_cap = -(-n_cap // chunk)
    weights = jnp.pad(weights, ((0, 0), (0, 0), (0, n_chunks_cap * chunk - n_cap)))
    n_chunks = ((count + chunk - 1) // chunk).astype(jnp.int32)
    spec_c = P(None, None, "x", "y")
    table = jnp.asarray(np.asarray(plan.windows, np.int32))

    def window_body(acc, window, segment):
        blocks = segment_blocks(plan, 0, 0)[0] if segment else plan.blocks
        ops = prepare(window)
        s = 0 if window is None or segment else window[0]
        shape = jax.eval_shape(lambda n: node_rows(ops, n), jnp.zeros((), jnp.int32))

        def skipped():
            return jax.lax.with_sharding_constraint(jnp.zeros(shape.shape, shape.dtype),
                                                    NamedSharding(mesh, P(None, None, None, "x", "y")))

        def block_add(acc, rows, projection, block):
            def local(a, r, w):
                if block.extents is None:
                    valid = jnp.asarray([R, a.shape[3]], jnp.int32)
                else:
                    la, wa, lb, wb = block.extents
                    row0 = 0 if window is None else window[0]
                    valid = jnp.stack((
                        jnp.clip(la - jax.lax.axis_index("x") * wa - row0, 0, R),
                        jnp.clip(lb - jax.lax.axis_index("y") * wb, 0, wb))).astype(jnp.int32)
                return contour_block_accumulate_local(a, r, w, valid, m0=block.m0 + s,
                                                      n0=block.n0, mesh=mesh)
            return jax.shard_map(local, mesh=mesh, in_specs=(spec_c, spec_c, P()),
                                 out_specs=spec_c, check_vma=False)(acc, rows, projection)

        def chunk_body(state):
            c, acc = state

            def one(_, j):
                n = c * chunk + j
                return None, jax.lax.cond(n < count, lambda: node_rows(ops, n), skipped)
            _, rows = jax.lax.scan(one, None, jnp.arange(chunk, dtype=jnp.int32), unroll=1)
            # Read behind a barrier: a counter-indexed slice must not be
            # rematerialized after the counter's in-place increment.
            w = jax.lax.optimization_barrier(
                jax.lax.dynamic_slice_in_dim(weights, c * chunk, chunk, axis=2))
            projection = jnp.transpose(w, (2, 0, 1)).reshape(n_sets * chunk, -1)
            for b, block in enumerate(blocks):
                plane = rows[:, b].reshape((n_sets * chunk,) + rows.shape[3:])
                acc = block_add(acc, plane, projection, block)
            return c + 1, acc

        _, acc = jax.lax.while_loop(lambda st: st[0] < n_chunks, chunk_body,
                                    (jnp.zeros((), jnp.int32), acc))
        return acc

    if only is not None:
        if len(plan.windows) == 1:
            return window_body(carry, None, True)
        w = jax.lax.optimization_barrier(jax.lax.dynamic_index_in_dim(table, only, keepdims=False))
        return window_body(carry, (w[0], w[1], w[2]), True)
    if len(plan.windows) == 1:
        return window_body(carry, None, False)

    def scan_body(acc, w):
        s, lo, hi = jax.lax.optimization_barrier((w[0], w[1], w[2]))
        return window_body(acc, (s, lo, hi), False), None
    carry, _ = jax.lax.scan(scan_body, carry, table, unroll=1)
    return carry
