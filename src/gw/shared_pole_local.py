"""Rounds of shared-pole parents, one parent per mesh rank (batch layout).

The round schedule, the column tables and the one round program that packs,
assembles, reduces, gates and sorts every parent of a round on its own rank
with local dense kernels, and the restore of round results to the face.
"""
from functools import lru_cache, partial

# Batch layout: rank x*Py + y owns whole parents; every other axis is unsharded.
BATCH = ('x', 'y')


def parent_rounds(parents, width):
    """Fixed-width rounds of parent slots in canonical order: the one schedule of every route.

    ``parents`` is a parent count or a sequence of parent ids; ``width`` the
    slots of every round (P on the local route, the admitted batch on the
    face). Every parent's minus-q actions are already in its own bank panels,
    so no round needs another parent. A short last round repeats its last
    real parent in the synthetic slots, so every round program sees one
    shape; consumers read the leading ``real`` slots (gates, receipts, the
    writer), and a local synthetic slot is never solved.

    Returns ``[(ids, real, slots)]``: ``ids`` the ``width`` parent ids, ``real``
    the number of leading real slots, ``slots`` the slot index of each slot.
    """
    import numpy as np

    ids = (list(range(int(parents))) if isinstance(parents, (int, np.integer))
           else [int(q) for q in parents])
    width = int(width)
    out = []
    for q0 in range(0, len(ids), width):
        chunk = ids[q0:q0 + width]
        out.append((chunk + [chunk[-1]] * (width - len(chunk)), len(chunk),
                    np.arange(width, dtype=np.int64)))
    return out


@lru_cache(maxsize=None)
def batch_to_face(mesh_xy):
    """[B, m, r] at P(('x','y'), None, None) -> the face P(None, 'x', 'y'): y then x all_to_all.

    The reduced factor width need not be divisible by Py. Zero-pad its
    carrier before splitting; callers keep only their admitted active width.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from jax import shard_map
    from runtime.padding import padded_axis
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])

    def restore(a):
        if py > 1:
            pad = padded_axis(a.shape[2], py, name='shared_pole_factor_width').pad
            if pad:
                a = jnp.pad(a, ((0, 0), (0, 0), (0, pad)))
            a = jax.lax.all_to_all(a, 'y', split_axis=2, concat_axis=0, tiled=True)
        if px > 1:
            a = jax.lax.all_to_all(a, 'x', split_axis=1, concat_axis=0, tiled=True)
        return a
    return jax.jit(shard_map(restore, mesh=mesh_xy, in_specs=P(BATCH), out_specs=P(None, 'x', 'y'),
                             check_vma=False))


@lru_cache(maxsize=None)
def batch_stack_to_face(mesh_xy, nq):
    """Fields [B, m, r_f] in batch layout -> one face stack [nq, F, m_X, r_Y].

    The same y-then-x exchange as ``batch_to_face``, applied once to the
    stacked fields (each padded to its own Py-divisible width first) instead
    of once per field; the result's rows are the first ``nq`` batch rows.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jax import shard_map
    from runtime.padding import padded_axis
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])

    def restore(a):
        if py > 1:
            a = jax.lax.all_to_all(a, 'y', split_axis=3, concat_axis=0, tiled=True)
        if px > 1:
            a = jax.lax.all_to_all(a, 'x', split_axis=2, concat_axis=0, tiled=True)
        return a
    move = shard_map(restore, mesh=mesh_xy, in_specs=P(BATCH), out_specs=P(None, None, 'x', 'y'),
                     check_vma=False)

    def pad(a):
        width = padded_axis(a.shape[2], py, name='shared_pole_factor_width').pad if py > 1 else 0
        return jnp.pad(a, ((0, 0), (0, 0), (0, width))) if width else a

    return jax.jit(lambda *fields: move(jnp.stack([pad(f) for f in fields], axis=1))[:nq],
                   out_shardings=NamedSharding(mesh_xy, P(None, None, 'x', 'y')))


@lru_cache(maxsize=None)
def face_rows(mesh_xy, rows, width=None):
    """Select parent rows of face stacks [B, m_X, r_Y] on their replicated leading axis.

    Several stacks are joined along that axis first; ``width`` keeps the leading
    columns, and a ``width`` past the stacks' own carrier (a held writer
    carrier) appends zero columns. Rows move nothing between ranks.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    index = tuple(int(r) for r in rows)

    def select(*parts):
        a = jnp.concatenate(parts)[jnp.asarray(index)][..., :width]
        if width is not None and a.shape[-1] < width:
            a = jnp.pad(a, ((0, 0),) * (a.ndim - 1) + ((0, width - a.shape[-1]),))
        return a
    return jax.jit(select, out_shardings=NamedSharding(mesh_xy, P(None, 'x', 'y')))


@lru_cache(maxsize=None)
def canonical_factors(mesh_xy, order, components=1):
    """Store handoff [nq, mu_X, component, K_Y] from flattened endpoint rows.

    Blocks are padded with zero columns to the widest, joined in round order and
    placed in canonical parent order (``order[i]`` is the joined row of parent i).
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    index = tuple(int(i) for i in order)

    def stack(*parts):
        width = max(part.shape[-1] for part in parts)
        whole = jnp.concatenate([jnp.pad(part, ((0, 0), (0, 0), (0, width - part.shape[-1]))) for part in parts])
        whole = whole[jnp.asarray(index)]
        return whole.reshape(whole.shape[0], whole.shape[1] // components, components, width)
    return jax.jit(stack, out_shardings=NamedSharding(mesh_xy, P(None, 'x', None, 'y')))


def carrier_history(meta):
    """One model's held writer widths (``held_writer_width``), shared by its SC maps."""
    history = getattr(meta, "_shared_pole_carrier_history", None)
    if history is None:
        history = {}
        meta._shared_pole_carrier_history = history
    return history


@lru_cache(maxsize=None)
def _pad_columns(sharding, shape, width):
    import jax
    import jax.numpy as jnp
    pad = ((0, 0),) * (len(shape) - 1) + ((0, width - shape[-1]),)
    return jax.jit(lambda a: jnp.pad(a, pad), out_shardings=sharding)


def recipe_panel_widths(roles, states, recipe, *, column_extent, logical_n):
    """Each state's panel carrier from the recipe alone, known before round 1.

    An imaginary support (and its partner, on Q's carrier) carries the recipe's
    imaginary width; a line support (and its conjugate) the line cap; a mirror
    its original's. ``roles`` are one slot's role records (``_roles``), one per
    state. A selection closed over a boundary multiplet can exceed its width;
    such a panel keeps its own, wider carrier.
    """
    widths = []
    for record, state in zip(roles, states):
        kind = record["role"].split(":", 1)[0]
        width = recipe["imaginary_width"] if kind == "imaginary" else (recipe.get("line_direction_cap") or logical_n)
        widths.append(max(column_extent(min(int(logical_n), max(1, int(width)))), int(state[1].shape[-1])))
    return widths


def recipe_infinity_width(infinity, recipe, *, column_extent, logical_n):
    """The infinity block's carrier: the recipe's infinity width, or the block's own width when wider.

    The leading directions of M1 close over a boundary multiplet, so a block
    can exceed the recipe width (Na, a metal); as a state panel keeps its
    wider carrier (``recipe_panel_widths``), so does the block. Decided once,
    before the tables, and never below the block.
    """
    return max(column_extent(min(int(logical_n), max(1, int(recipe["infinity_width"])))),
               int(infinity[0].shape[-1]))


def pad_states(states, widths, infinity, infinity_width):
    """Zero-pad every state panel and the infinity block to their carriers (inert columns).

    A carrier is a recipe bound or the array's own width when wider
    (``recipe_panel_widths``, ``recipe_infinity_width``); it is never below
    the array, so no panel is ever cut.
    """
    def pad(a, w):
        if int(a.shape[-1]) > int(w):
            raise ValueError(f"GATE shared_pole_carrier: got: a panel of {int(a.shape[-1])} columns on a "
                             f"carrier of {int(w)}; want: carrier >= panel; why: a carrier is the recipe "
                             "bound or the panel's own width, never narrower")
        return a if int(a.shape[-1]) == int(w) else _pad_columns(a.sharding, a.shape, int(w))(a)
    return ([(st[0], *(pad(a, w) for a in st[1:])) for st, w in zip(states, widths)],
            tuple(pad(a, infinity_width) for a in infinity))


def round_tables(counts, widths, nodes, infinity_counts, infinity_width, *, column_extent, ordered,
                 odd_moments, key=None, history=None):
    """Host column tables of one round: each slot's states packed into its pencil columns.

    ``counts`` int [P, A] retained widths per slot and state, ``widths`` the A
    panel widths, ``nodes`` the A state nodes (s, or z on the ordered route),
    ``infinity_counts`` [P] retained infinity widths of the [P, n,
    ``infinity_width``] infinity panels. A state keeps its carrier
    ``column_extent(count)`` with the inert tail inside, as a per-parent pack
    does; each slot is compacted in state order and padded with zero columns to
    the round extent. An ordered round packs originals (the first A/2 states)
    and mirrors (the last A/2) as two halves of one extent, so every slot stays
    in the paired layout [X(z); X(-z)].

    The extent is the carrier of the largest selection, at most the panels'
    capacity. With ``key`` and ``history`` (``carrier_history``, in the SC
    session) it is grow-only across the model's rounds and SC maps: the partner
    (TRS-odd) counts have no bound below the capacity, and the capacity costs
    the pencil eigh (capacity / high water)^3 on every round of every map (CrI3
    24x24 at P64: CC 20800 against 17472, TT 32000 against 24832), so the extent
    is discovered in map 0 and held from then on. The hold is keyed by the model
    alone, not by its panel count: the line sites, and so the panels, change
    from map to map, while the round programs take the packed columns
    (``pack_panels``), whose shape is this extent only. A held extent past this
    round's capacity is more inert zero columns.

    Returns a dict: ``order`` int32 [P, F] (index sum(widths) is the zero
    column), ``points`` complex128 [P, F], ``active`` bool [P, side] (finite
    columns, then the infinity block; k0 and k1 on an odd-moment ordered round,
    none on a finite-state ordered bank), and ``own`` [P], each slot's Gram
    spectrum length at its own extent, for the receipt; ``extents`` [P, 2]
    carries each finite half and infinity block before round padding.
    """
    import numpy as np

    counts = np.asarray(counts, np.int64)
    ranks, states = counts.shape
    offsets = np.concatenate(([0], np.cumsum(widths))).astype(np.int64)
    carriers = np.asarray([[column_extent(int(c)) for c in row] for row in counts], np.int64)
    halves = (range(states // 2), range(states // 2, states)) if ordered else (range(states),)
    # The state panels sit on their recipe carriers (``recipe_panel_widths``),
    # so a round program's inputs have one shape; only this extent, the pencil
    # side, follows the selection. The padding is inert: zero columns that
    # the zero-row-safe eigensolver keeps out of every spectrum.
    if np.any(carriers > np.asarray(widths, np.int64)[None, :]):
        raise ValueError("GATE shared_pole_round_tables: got: a state carrier wider than its "
                         "panel; want: column_extent(count) <= panel width; why: its columns "
                         "would index the next state's panel")
    capacity = max(sum(int(widths[a]) for a in half) for half in halves)
    selected = max(int(carriers[:, list(half)].sum(axis=1).max()) for half in halves)
    # Never below the selection: an extent function may saturate on a sum.
    extent = min(capacity, max(selected, column_extent(selected)))
    # The extent tiles the mesh as every carrier does (a face column block per y rank):
    # rounded up to the smallest carrier, a no-op whenever the panels sit on carriers.
    grain = max(1, int(column_extent(1)))
    extent = -(-extent // grain) * grain
    if key is not None:
        if history is None:
            raise ValueError("round_tables: a grow-only extent needs the model's carrier history")
        name = (key, "extent", len(halves))
        extent = max(extent, int(history.get(name, (0,))[0]))
        history[name] = (extent,)
    order = np.full((ranks, extent * len(halves)), offsets[-1], np.int32)
    points = np.zeros(order.shape, np.complex128)
    live = np.zeros(order.shape, bool)
    for r in range(ranks):
        for k, half in enumerate(halves):
            at = k * extent
            for a in half:
                c = int(carriers[r, a])
                order[r, at:at + c] = offsets[a] + np.arange(c)
                points[r, at:at + c] = nodes[a]
                live[r, at:at + int(counts[r, a])] = True
                at += c
    blocks = (2 if odd_moments else 0) if ordered else 1
    tail = np.arange(infinity_width)[None, :] < np.asarray(infinity_counts)[:, None]
    finite = carriers[:, list(halves[0])].sum(axis=1)
    infinity = np.asarray([column_extent(int(c)) if blocks else 0 for c in infinity_counts], np.int64)
    return dict(order=order, points=points, active=np.concatenate((live,) + (tail,) * blocks, axis=1),
                own=finite + infinity, extents=np.column_stack((finite, infinity)))


def pack_panels(fields, order, widths, *, mesh_xy, layout):
    """Each slot's pencil columns, placed from the round's state panels one panel at a time.

    ``fields`` are K lists of the round's S state panels ``[B, n_k, r_s]`` (Q,
    WQ, dWQ, or a cross action), ``order`` the host ``[B, F]`` table of
    ``round_tables`` (column ``order[b, f]`` of the joined panels;
    ``sum(widths)`` is the zero column) and ``widths`` the S panel widths.
    Returns K arrays ``[B, n_k, F]`` with ``out[b, :, f] = panel_s[b, :, j]``
    where ``order[b, f] = offset_s + j`` and zero where it names the zero
    column, in the panels' ``layout``: ``'batch'`` (whole parents per rank,
    ``P(BATCH)``) or ``'face'`` (``P(None, 'x', 'y')``).

    One program per (layout, panel width, F) places one panel, and the panel
    count is only the number of calls: the line sites change it from map to
    map, and no program's shape follows it. The host inverts ``order`` per
    panel, ``dest[b, j]`` the packed column of panel column j or F where the
    round does not take it (the scatter drops it); a source column appears at
    most once in a slot's table, so the scatter has one writer per column.
    """
    import numpy as np
    order = np.asarray(order)
    offsets = np.concatenate(([0], np.cumsum(widths))).astype(np.int64)
    shapes = tuple((int(panels[0].shape[1]), np.dtype(panels[0].dtype).name) for panels in fields)
    key = (mesh_xy, layout, int(order.shape[0]), int(order.shape[1]), shapes)
    accs = _pack_start(*key)()
    for s, width in enumerate(int(w) for w in widths):
        # dest[b, j]: where column j of panel s lands in slot b's pencil, F (dropped) if nowhere.
        taken = (order >= offsets[s]) & (order < offsets[s] + width)
        dest = np.full((order.shape[0], width), order.shape[1], np.int32)
        b, f = np.nonzero(taken)
        dest[b, order[b, f] - offsets[s]] = f
        dest = _batch_put(mesh_xy, dest) if layout == 'batch' else dest
        accs = _pack_place(*key, width)(accs, tuple(panels[s] for panels in fields), dest)
    return _pack_finish(*key)(accs)


def _slab_rows(mesh_xy, rows):
    """Rows per rank of a face matrix's slab: its x-tile's rows, padded to Py, split over y."""
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])
    return -(-(int(rows) // px) // py)


@lru_cache(maxsize=None)
def _pack_start(mesh_xy, layout, slots, extent, shapes):
    """The K zero accumulators of ``pack_panels`` in their placement layout.

    Batch: ``[B, n_k, F]`` at ``P(BATCH)``. Face: the slab ``[B, P * s_k, F]`` at
    ``P(None, ('x', 'y'), None)``, rank x*Py+y holding s_k = ``_slab_rows`` whole
    pencil rows (its x-tile's rows, padded to a multiple of Py, block y of them),
    so a panel's columns land in any packed column without a second exchange.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    ranks = int(mesh_xy.size)
    if layout == 'batch':
        spec, rows = P(BATCH), [n for n, _ in shapes]
    else:
        spec, rows = P(None, BATCH, None), [ranks * _slab_rows(mesh_xy, n) for n, _ in shapes]
    sharding = NamedSharding(mesh_xy, spec)
    return jax.jit(lambda: tuple(jnp.zeros((slots, r, extent), dt) for r, (_, dt) in zip(rows, shapes)),
                   out_shardings=(sharding,) * len(shapes))


@lru_cache(maxsize=None)
def _pack_place(mesh_xy, layout, slots, extent, shapes, width):
    """``accs[k][b, :, dest[b, j]] = panel_k[b, :, j]`` for one panel width, accumulators donated.

    Face: one all_to_all over y moves the panel's tile ``[B, m, r/Py]`` to its
    slab ``[B, s, r]`` (row block y of the x-tile, every column of the panel in
    global order), the same bytes as the tile; the scatter is then local.
    Batch: whole parents per rank, no exchange.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import PartitionSpec as P
    from jax import shard_map
    py = int(mesh_xy.shape['y'])

    def scatter(acc, panel, dest):
        return jax.vmap(lambda a, p, d: a.at[:, d].set(p, mode='drop'))(acc, panel, dest)

    if layout == 'batch':
        def body(accs, panels, dest):
            return tuple(scatter(a, p, dest) for a, p in zip(accs, panels))
        specs = ((P(BATCH),) * len(shapes), (P(BATCH),) * len(shapes), P(BATCH))
        out = (P(BATCH),) * len(shapes)
    else:
        def body(accs, panels, dest):
            moved = []
            for acc, panel in zip(accs, panels):
                pad = py * int(acc.shape[1]) - int(panel.shape[1])
                if pad:
                    panel = jnp.pad(panel, ((0, 0), (0, pad), (0, 0)))
                # Face tile -> slab: split the rows over y, join the column blocks in y order.
                if py > 1:
                    panel = jax.lax.all_to_all(panel, 'y', split_axis=1, concat_axis=2, tiled=True)
                moved.append(scatter(acc, panel, dest))
            return tuple(moved)
        slab = P(None, BATCH, None)
        specs = ((slab,) * len(shapes), (P(None, 'x', 'y'),) * len(shapes), P())
        out = (slab,) * len(shapes)
    return jax.jit(shard_map(body, mesh=mesh_xy, in_specs=specs, out_specs=out, check_vma=False),
                   donate_argnums=0)


@lru_cache(maxsize=None)
def _pack_finish(mesh_xy, layout, slots, extent, shapes):
    """The packed accumulators in the panels' layout: batch as placed; face by the inverse
    all_to_all (slab -> tile, one per field), its row padding dropped."""
    import jax
    from jax.sharding import PartitionSpec as P
    from jax import shard_map
    if layout == 'batch':
        return lambda accs: accs
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])
    if extent % py:
        # round_tables rounds every extent to the smallest carrier, which tiles the mesh.
        raise ValueError(f"pack_panels: the extent {extent} must tile the {py} y ranks of the face; "
                         "build the tables with the face's column_extent (round_tables)")

    def body(accs):
        out = []
        for acc, (rows, _) in zip(accs, shapes):
            if py > 1:
                acc = jax.lax.all_to_all(acc, 'y', split_axis=2, concat_axis=1, tiled=True)
            out.append(acc[:, :rows // px])
        return tuple(out)
    slab = P(None, BATCH, None)
    return jax.jit(shard_map(body, mesh=mesh_xy, in_specs=((slab,) * len(shapes),),
                             out_specs=(P(None, 'x', 'y'),) * len(shapes), check_vma=False))


def own_extent_receipts(reduction, own):
    """Per-slot receipt rows of a round reduction at each parent's own extent.

    A parent's own ascending spectrum carries exact-zero capacity padding.
    Its receipt drops that many exact zeros, reads ``gram_min_relative``, and
    normalizes the metric residual by its own side. ``reduction`` holds host
    arrays [P, ...]; returns one dict of [1, ...] arrays per entry of ``own``.
    """
    import numpy as np

    rows = []
    for slot, side in enumerate(int(v) for v in own):
        row = {key: value[slot:slot + 1] for key, value in reduction.items()}
        spectrum = row["gram_spectrum_relative"][0]
        # Capacity padding contributes exact zeros at their ascending place;
        # drop that many exact zeros to recover the own-extent spectrum.
        pad = spectrum.size - side
        zeros = np.flatnonzero(spectrum == 0)
        kept = np.delete(spectrum, zeros[len(zeros) - pad:] if pad > 0 else [])[:side]
        row["gram_spectrum_relative"] = kept[None]
        row["gram_min_relative"] = kept[:1]
        row["metric_inverse_root_residual_relative"] = row["metric_inverse_root_residual_fro"] / np.sqrt(side)
        rows.append(row)
    return rows


def _batch_put(mesh_xy, a):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.make_array_from_callback(a.shape, NamedSharding(mesh_xy, P(BATCH)), lambda idx: a[idx])


def reduce_round(states, infinity, tables, *, real, mesh_xy, native_eigh, ordered, odd_moments, keep_budget,
                 retain_span=False, gram_keep=None):
    """Run ``round_program`` on one round: host tables in, round-order results out.

    An ordered round solves its kept span on the carrier of its pole budget
    (the keep cut retains at most ``keep_budget`` directions), as the face
    round does on ``face_ritz_carrier``: the carrier is known before the
    first round, so one program serves every round and SC map, and only
    exact-zero columns ever leave the solve.
    """
    import numpy as np

    put = lambda a: _batch_put(mesh_xy, np.asarray(a))
    live = np.arange(len(tables["own"])) < int(real)
    q, o, d = pack_panels(tuple([st[k] for st in states] for k in (1, 2, 3)), tables["order"],
                          [int(st[1].shape[-1]) for st in states], mesh_xy=mesh_xy, layout='batch')
    args = (put(live), put(tables["points"]), put(tables["active"]), q, o, d, tuple(infinity))
    budget = None if keep_budget is None else int(keep_budget)
    return round_program(mesh_xy, native_eigh, bool(ordered), bool(odd_moments), budget,
                         bool(retain_span), gram_keep, budget)(*args)


def solve_parent_pencil(points, q, o, d, infinity, active, *, eigh, matmul,
                        gates, ordered, odd_moments, keep_budget, retain_span=False, matrix_sharding=None,
                        gram_keep=None, carrier=None):
    """One equation owner for local and whole-mesh parent execution.

    Inputs carry one or more independent parents. Execution adapters supply
    service/local products and eigensolve; this function owns the pencil,
    reduction, zero policy and original/retained moment identities.
    """
    import jax.numpy as jnp
    from gw.shared_pole_gates import (apply_shared_pole_zero_policy, ordered_moment_identity,
                                     retained_moment_identity)
    from gw.shared_pole_pencil import assemble_ordered_shared_pole_pencil, assemble_shared_pole_pencil
    from gw.shared_pole_reduction import reduce_ordered_shared_pole_pencil, reduce_shared_pole_pencil
    finite = [(points, q, o, d)]
    # Support geometry of the even route's Gram validity floor (shared_pole_reduction.gram_rounding_floor).
    rounding = dict(points=points, derivative_norm=jnp.linalg.norm(d, axis=-2), rows=int(q.shape[-2]))
    if ordered:
        pencil = assemble_ordered_shared_pole_pencil(finite, infinity if odd_moments else None,
                                                     matmul=matmul, matrix_sharding=matrix_sharding)
        reduced = reduce_ordered_shared_pole_pencil(
            pencil, active, eigh=eigh, matmul=matmul, gates=gates, keep_budget=keep_budget,
            retain_span=retain_span, matrix_sharding=matrix_sharding, gram_keep=gram_keep, carrier=carrier)
        model, signed, reduction = reduced[:3]
        if retain_span:
            coefficients = reduced[3]
        retained = ordered_moment_identity(signed, infinity, matmul=matmul) if odd_moments else {}
        model, zero = apply_shared_pole_zero_policy(model, gates=gates)
        zero["zero_policy"] = zero["zero_policy"] & reduction["infinite_weight_ok"]
    else:
        pencil = assemble_shared_pole_pencil(finite, infinity, matmul=matmul)
        model, reduction, coefficients = reduce_shared_pole_pencil(
            pencil, active, eigh=eigh, matmul=matmul, gates=gates, keep_budget=keep_budget,
            rounding=rounding)
        model, zero = apply_shared_pole_zero_policy(model, gates=gates)
        # E selects the infinity block, the last columns of X.
        side, width = pencil[0].shape[-1], infinity[0].shape[-1]
        selector = (jnp.arange(side)[:, None] == jnp.arange(side - width, side)[None, :])
        # One selector per parent: the whole-mesh service matmul does not broadcast batches.
        selector = jnp.broadcast_to(selector, (pencil[0].shape[0],) + selector.shape)
        retained = retained_moment_identity(pencil, coefficients, model, selector.astype(jnp.complex128),
                                            matmul=matmul)
        signed = ()
    result = model, signed, (reduction, zero, retained)
    if retain_span:
        return (*result, coefficients)
    return result


@lru_cache(maxsize=None)
def round_program(mesh_xy, native_eigh, ordered, odd_moments, keep_budget, retain_span=False,
                  gram_keep=None, carrier=None):
    """Assemble, reduce, gate and sort a round of parents, each on its own rank.

    One program over batch layout: rank r takes slot r's packed Q, WQ, dWQ
    columns (``pack_panels``, at the round's extent; no panel count enters the
    program), assembles the even or ordered pencil, reduces it with the
    local eigensolver ``native_eigh`` (zero-row safe in the service), applies the zero
    policy, forms the retained (even) or original-infinity (ordered, odd
    moments) moment identity and sorts the poles. Every slot solves at the
    round's laddered extent (``round_tables``); its inert columns are exact
    zeros that the eigensolver wrapper keeps out of every spectrum, so rounds
    and SC maps share the program's shapes. A synthetic slot (``live`` False) skips all of it through
    ``lax.cond``. No array crosses ranks: the models stay in batch layout for
    ``round_checks``; only vectors are gathered.

    Arguments ``(live [P], points, active, Q, WQ, dWQ [P, n, F], infinity)``, all
    in batch layout. Returns ``(model, signed, vectors, (reduction, zero,
    retained, permutation))`` in round order: model (b [P, n, side], poles2,
    active) and signed (c, mu, retained; ordered route, else ``()``) in batch
    layout, ``vectors`` = (poles2, active) and the diagnostics replicated.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jax import shard_map
    from gw.shared_pole_gates import sort_shared_pole_columns
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1, shared_real_pole_gates_v1_r3b

    gates = shared_real_pole_gates_ordered_v1 if ordered else shared_real_pole_gates_v1_r3b
    batch, replicated = P(BATCH), NamedSharding(mesh_xy, P())
    # The service eigh deflates exact-zero rows (distrib_la._eigh_safe).
    eigh = native_eigh

    def solve(points, q, o, d, infinity, active):
        return solve_parent_pencil(points, q, o, d, infinity, active,
            eigh=eigh, matmul=_mm, gates=gates, ordered=ordered,
            odd_moments=odd_moments, keep_budget=keep_budget, retain_span=retain_span, gram_keep=gram_keep,
            carrier=carrier)

    def body(live, points, active, q, o, d, infinity):
        args = (points, q, o, d, infinity, active)

        def work(args):
            return solve(*args)
        def skip(args):
            return jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), jax.eval_shape(work, args))
        reduced = jax.lax.cond(live[0], work, skip, args)
        model, signed, diagnostics = reduced[:3]
        model, permutation = sort_shared_pole_columns(model)
        result = model, signed, (*diagnostics, permutation)
        return (*result,reduced[3]) if retain_span else result

    mapped = shard_map(body, mesh=mesh_xy, in_specs=(batch,) * 7, out_specs=batch, check_vma=False)

    @jax.jit
    def execute(live, points, active, q, o, d, infinity):
        reduced = mapped(live, points, active, q, o, d, infinity)
        model, signed, diagnostics = reduced[:3]
        result = model, signed, _gather(replicated, model[1:]), _gather(replicated, diagnostics)
        if retain_span:
            return (*result, reduced[3])
        return result
    return execute


def _gather(replicated, tree):
    import jax
    return jax.tree.map(lambda a: jax.lax.with_sharding_constraint(a, replicated), tree)


def _mm(a, b, *, transa='N', transb='N'):
    """The rank-local GEMM of the round programs, with the service's transa/transb flags."""
    import jax.numpy as jnp

    def op(value, trans):
        if trans == 'N':
            return value
        value = jnp.swapaxes(value, -1, -2)
        return jnp.conj(value) if trans == 'C' else value
    return jnp.matmul(op(a, transa), op(b, transb))


def check_round(model, signed, inverse_coulomb_sqrt, held, moments, infinity_directions, *, real, nodes, eta_ry,
                mesh_xy, eigh_plan, ordered, room=None):
    """Run ``round_checks`` through the plan matching the arrays' layout.

    A face round given ``room`` decides its eighs against ``room(price)``,
    the room beside the check program's price for one parent
    (``face_check_bytes``)."""
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    from gw.shared_pole_execution import is_face
    if is_face(model[0]):
        # A face round holds ``real`` physical parents in its leading slots;
        # the gate equations are per parent, so each is checked on its own
        # leading row (no data moves) and the replicated rows are stacked.
        from gw.shared_pole_execution import face_check_bytes, face_eigh, face_round_check_program
        if eigh_plan.n not in (None, int(model[0].shape[-2])):
            raise ValueError('whole-mesh shared-pole checks need the n x n eigh plan')
        program = lambda plan: face_round_check_program(mesh_xy, bool(ordered), plan)
        n, rows = int(model[0].shape[-2]), []
        for slot in range(int(real)):
            pick = partial(_leading_row, slot=slot)
            args = (jax.tree.map(pick, model), jax.tree.map(pick, signed),
                    pick(inverse_coulomb_sqrt), *(pick(h) for h in held),
                    np.asarray(nodes, np.complex128), np.float64(eta_ry),
                    *(pick(m) for m in moments), pick(infinity_directions))
            if callable(room):
                eigh_plan, room = face_eigh(mesh_xy, n, room(face_check_bytes(mesh_xy, n, int(model[0].shape[-1])))), None
            rows.append(jax.tree.map(np.asarray, program(eigh_plan)(*args)))
        return jax.tree.map(lambda *v: np.concatenate(v, axis=0), *rows)

    replicated = NamedSharding(mesh_xy, P())
    # Batch-layout equations run inside shard_map and therefore require the
    # plan's public trace-safe native callable. Face arrays were dispatched
    # above and use the plan's eager ``batched`` surface in their own program.
    program = round_checks(mesh_xy, eigh_plan.native_fn, bool(ordered))
    live = _batch_put(mesh_xy, np.arange(int(model[0].shape[0])) < int(real))
    out = program(live, model, signed, inverse_coulomb_sqrt, *held,
                  jax.device_put(np.asarray(nodes, np.complex128), replicated),
                  jax.device_put(np.float64(eta_ry), replicated), *moments, infinity_directions)
    return jax.tree.map(np.asarray, out)


def _leading_row(array, *, slot):
    """Parent ``slot`` of a stack whose leading parent axis is unsharded."""
    return _leading_row_program(array.sharding)(array, slot)


@lru_cache(maxsize=None)
def _leading_row_program(sharding):
    import jax
    return jax.jit(lambda a, s: jax.lax.dynamic_slice_in_dim(a, s, 1, axis=0),
                   out_shardings=sharding)


def _round_check_equations(model, signed, inverse, wc, dw, z, eta, m1, m3, qi,
                           *, matmul, eigh, gates, ordered):
    """Shared scalar gate equations for local-parent and whole-mesh adapters."""
    import jax
    import jax.numpy as jnp
    from gw.shared_pole_directions import _model_diagnostics
    from gw.shared_pole_gates import (shared_pole_passivity,
                                      shared_pole_reciprocity,
                                      signed_shared_pole_passivity)

    factor, poles, active = model
    if ordered:
        factor, mu, kept = signed
        passive = signed_shared_pole_passivity(
            signed, inverse, eta_ry=eta, matmul=matmul, eigh=eigh, gates=gates)
        node = z[:, None]
        weights = jnp.where(kept, 1 / (node * mu - 1), 0)
        slopes = jnp.where(kept, -mu / (node * mu - 1) ** 2 / (2 * node), 0)
        scale = jnp.where(kept, 1 / jnp.where(kept, jnp.abs(mu), 1), 0)
        moment_model = (factor * scale[:, None, :], scale ** 2, kept)
    else:
        passive = shared_pole_passivity(
            model, inverse, eta_ry=eta, matmul=matmul, eigh=eigh, gates=gates)
        weights = jnp.where(active, 1 / ((z ** 2)[:, None] - poles), 0)
        slopes = -weights ** 2
        moment_model = model

    def sample(args):
        weight, slope, w, d = args
        errors, reciprocity = [], []
        for coefficient, exact in ((weight, w), (slope, d)):
            value = matmul(factor * coefficient[None, None, :],
                           factor, transb='C')[0]
            errors.append(jnp.linalg.norm(value - exact)
                          / jnp.maximum(jnp.linalg.norm(exact),
                                        jnp.finfo(jnp.float64).tiny))
            reciprocity.append({} if ordered else
                shared_pole_reciprocity(value, exact, gates=gates))
        return jnp.stack(errors), jax.tree.map(lambda *v: jnp.stack(v), *reciprocity)
    errors, reciprocity = jax.lax.map(sample, (weights, slopes, wc[0], dw[0]))
    rows = lambda a: jnp.swapaxes(a, 0, 1)[None]
    defects = _model_diagnostics(moment_model, {'M1': m1, 'M3': m3}, qi,
                                 matmul=matmul)
    return passive, rows(errors), jax.tree.map(rows, reciprocity), defects


@lru_cache(maxsize=None)
def round_checks(mesh_xy, native_eigh, ordered):
    """Passivity, held W and dW/ds, and moment defects of a round's models, each parent on its own rank.

    Arguments ``(live [P], model, signed, V^-1/2 [P, n, n], Wc and dWc/ds [P, S, n, n]
    at the held nodes z [S], eta, M1, M3 [P, n, n], Q_inf [P, n, r])``, all in
    batch layout except z and eta (replicated). The ordered route checks the
    signed model: -Wc(i eta) = sum c c^H/(1 - i eta mu), Wc(z) = c diag(1/(z mu - 1)) c^H,
    dWc/ds = c diag(-mu/(z mu - 1)^2/(2z)) c^H, moments of (c/|mu|, mu^-2). The
    even route checks W(s) = b (s - Lambda)^-1 b^H, dW/ds = -b (s - Lambda)^-2 b^H at
    s = z^2 and its reciprocity. Held samples are compared one at a time.
    Returns replicated ``(passive, held [P, 2, S], reciprocity [P, 2, S] rows,
    moment_defects)``; rows 0 and 1 of the held axis are W and dW/ds. A synthetic
    slot is skipped.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from jax import shard_map
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1, shared_real_pole_gates_v1_r3b

    gates = shared_real_pole_gates_ordered_v1 if ordered else shared_real_pole_gates_v1_r3b
    batch, rep = P(BATCH), P()

    def check(model, signed, inverse, wc, dw, z, eta, m1, m3, qi):
        return _round_check_equations(
            model, signed, inverse, wc, dw, z, eta, m1, m3, qi,
            matmul=_mm, eigh=native_eigh, gates=gates, ordered=ordered)

    def body(live, model, signed, inverse, wc, dw, z, eta, m1, m3, qi):
        args = (model, signed, inverse, wc, dw, z, eta, m1, m3, qi)

        def skip(args):
            return jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), jax.eval_shape(check, *args))
        return jax.lax.cond(live[0], lambda args: check(*args), skip, args)

    mapped = shard_map(body, mesh=mesh_xy, in_specs=(batch,) * 6 + (rep, rep) + (batch,) * 3, out_specs=batch,
                       check_vma=False)
    replicated = NamedSharding(mesh_xy, rep)
    return jax.jit(lambda *args: _gather(replicated, mapped(*args)))
