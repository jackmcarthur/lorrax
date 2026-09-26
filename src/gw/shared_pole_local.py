"""Rounds of shared-pole parents, whole parents per mesh rank (batch layout).

The round schedule, the column tables and the one round program that packs,
assembles, reduces, gates and sorts every parent of a round on its own rank
with local dense kernels, and the restore of round results to the face.
"""
from functools import lru_cache, partial

# Batch layout: rank x*Py + y owns whole parents; every other axis is unsharded.
BATCH = ('x', 'y')


def parent_rounds(nq, ranks, depth=1):
    """Rounds of ``ranks * depth`` parent slots, ``depth`` parents per rank, in canonical order.

    A round is the next contiguous run of parents; every parent's minus-q
    actions are already in its own bank panels, so no round needs another
    parent. Rank r owns slots ``[r*depth, (r+1)*depth)`` (batch layout). A
    short round repeats its last real parent in the synthetic slots, whose
    results are exact zeros. ``depth`` is the caller's ledger-admitted count.

    Returns ``[(ids, real, slots)]``: ``ids`` the ``ranks * depth`` parent
    ids, ``real`` the number of leading real slots, ``slots`` the slot index
    of each slot.
    """
    import numpy as np

    width = int(ranks) * int(depth)
    if width < 1:
        raise ValueError("parent rounds need at least one slot per rank")
    out = []
    for q0 in range(0, int(nq), width):
        ids = list(range(q0, min(q0 + width, int(nq))))
        real = len(ids)
        out.append((ids + [ids[-1]] * (width - real), real, np.arange(width, dtype=np.int64)))
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
    from common.shard_map import shard_map
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
    from common.shard_map import shard_map
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
    columns. Rows move nothing between ranks.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    index = tuple(int(r) for r in rows)
    return jax.jit(lambda *parts: jnp.concatenate(parts)[jnp.asarray(index)][..., :width],
                   out_shardings=NamedSharding(mesh_xy, P(None, 'x', 'y')))


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
    """One model's optional executable-reuse widths, shared by its SC maps."""
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


def grow_round(key, states, infinity, *, history, preview, admit):
    """Admit optional high-water carriers before allocating their padding.

    ``preview`` checks the candidate via ConstructorCapacity.  If reuse does
    not fit, ``admit`` prices the actual current widths and propagates a real
    capacity refusal.  The host table and its admitted side come back from
    ``admit``; every rank makes the same decision from the same receipt.
    """
    pad = lambda a, w: a if a.shape[-1] == w else _pad_columns(a.sharding, a.shape, w)(a)
    current = tuple(int(st[1].shape[-1]) for st in states)
    current_infinity = int(infinity[0].shape[-1])
    state_key, infinity_key = (key, "states", len(states)), (key, "infinity")
    old_states = history.get(state_key, current)
    old_infinity = history.get(infinity_key, (current_infinity,))[0]
    widths = tuple(max(a, b) for a, b in zip(current, old_states))
    width = max(current_infinity, old_infinity)
    try:
        reuse = bool(preview(widths, width, True))
    except (ValueError, MemoryError, RuntimeError):
        # A historical width may exceed this run's budget or native route.
        # The actual current round is still admitted below, where any real
        # failure propagates rather than being mistaken for optional reuse.
        reuse = False
    if not reuse:
        widths, width = current, current_infinity
    admitted = admit(widths, width, reuse)
    if reuse:
        history[state_key] = widths
        history[infinity_key] = (width,)
    states = [(st[0], *(pad(a, w) for a in st[1:])) for st, w in zip(states, widths)]
    return states, tuple(pad(a, width) for a in infinity), admitted


def round_tables(counts, widths, nodes, infinity_counts, infinity_width, *, column_extent, ordered,
                 odd_moments, key=None, history=None):
    """Host column tables of one round: each slot's states packed into its pencil columns.

    ``counts`` int [P, A] retained widths per slot and state, ``widths`` the A
    panel widths, ``nodes`` the A state nodes (s, or z on the ordered route),
    ``infinity_counts`` [P] retained infinity widths of the [P, n,
    ``infinity_width``] infinity panels. A state keeps its carrier
    ``column_extent(count)`` with the inert tail inside, as a per-parent pack
    does; each slot is compacted in state order and padded with zero columns to
    the round extent. An ordered round packs originals (the first A/2 states) and
    mirrors (the last A/2) as two halves of one extent, so every slot stays in the
    paired layout [X(z); X(-z)].

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
    # The round extent is the carrier (``column_extent``, on the ladder of
    # ``runtime.padding.ladder_extent`` in the constructors) of the largest
    # selection, at most every state's full panel, so rounds and SC maps
    # share round executables. The padding is inert: zero columns that the
    # zero-row-safe eigensolver keeps out of every spectrum.
    if np.any(carriers > np.asarray(widths, np.int64)[None, :]):
        raise ValueError("GATE shared_pole_round_tables: got: a state carrier wider than its "
                         "panel; want: column_extent(count) <= panel width; why: its columns "
                         "would index the next state's panel")
    capacity = max(sum(int(widths[a]) for a in half) for half in halves)
    selected = max(int(carriers[:, list(half)].sum(axis=1).max()) for half in halves)
    # Never below the selection: an extent function may saturate on a sum.
    extent = min(capacity, max(selected, column_extent(selected)))
    if key is not None:
        if history is None:
            raise ValueError("round_tables high water requires a model-local history")
        # Grow-only with the panels (grow_round); a round with a different
        # state count can have less capacity, which only costs a compile.
        old = history.get((key, "extent", len(halves), len(widths)), (extent,))[0]
        extent = min(capacity, max(extent, old))
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
    """Run ``round_program`` on one round: host tables in, round-order results out."""
    import numpy as np

    put = lambda a: _batch_put(mesh_xy, np.asarray(a))
    program = round_program(mesh_xy, native_eigh, bool(ordered), bool(odd_moments),
                            None if keep_budget is None else int(keep_budget), bool(retain_span), gram_keep)
    live = np.arange(len(tables["own"])) < int(real)
    return program(put(live), put(tables["points"]), put(tables["order"]), put(tables["active"]),
                   tuple(st[1] for st in states), tuple(st[2] for st in states),
                   tuple(st[3] for st in states), tuple(infinity))


def solve_parent_pencil(points, q, o, d, infinity, active, *, eigh, matmul,
                        gates, ordered, odd_moments, keep_budget, retain_span=False, matrix_sharding=None,
                        gram_keep=None):
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
    if ordered:
        pencil = assemble_ordered_shared_pole_pencil(finite, infinity if odd_moments else None,
                                                     matmul=matmul, matrix_sharding=matrix_sharding)
        reduced = reduce_ordered_shared_pole_pencil(
            pencil, active, eigh=eigh, matmul=matmul, gates=gates, keep_budget=keep_budget,
            retain_span=retain_span, matrix_sharding=matrix_sharding, gram_keep=gram_keep)
        model, signed, reduction = reduced[:3]
        if retain_span:
            coefficients = reduced[3]
        retained = ordered_moment_identity(signed, infinity, matmul=matmul) if odd_moments else {}
        model, zero = apply_shared_pole_zero_policy(model, gates=gates)
        zero["zero_policy"] = zero["zero_policy"] & reduction["infinite_weight_ok"]
    else:
        pencil = assemble_shared_pole_pencil(finite, infinity, matmul=matmul)
        model, reduction, coefficients = reduce_shared_pole_pencil(
            pencil, active, eigh=eigh, matmul=matmul, gates=gates, keep_budget=keep_budget)
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


def zero_row_safe_eigh(eigh):
    """Wrap a Hermitian eigensolver so exact zero rows cannot break it.

    Capacity padding and unselected columns reach the eigensolver as exact
    zero rows/columns; a large zero block made the native eigensolver return
    nonfinite values (Na P16, 1460 of 2584 rows). Those rows are decoupled, so
    they are replaced by distinct diagonal sentinels below the Gershgorin
    bound of the rest, solved, and reported back as exact zero eigenvalues with
    their unit eigenvectors, in ascending order: the spectrum and vectors of
    the zero-padded matrix, without a zero block inside the solver.
    """
    import jax.numpy as jnp

    def solve(a):
        n = a.shape[-1]
        dead = jnp.all(a == 0, axis=-1)
        # Every live eigenvalue lies in [-bound, bound] (Gershgorin); the
        # sentinels sit at or below -(2 bound + 1), a relative gap of one bound.
        bound = jnp.max(jnp.sum(jnp.abs(a), axis=-1), axis=-1)
        sentinel = -(2 * bound + 1)[..., None] * (1 + jnp.arange(n, dtype=bound.dtype) / n)
        diagonal = jnp.where(dead, sentinel, 0).astype(a.dtype)
        values, vectors = eigh(a + diagonal[..., :, None] * jnp.eye(n, dtype=a.dtype))
        values = jnp.where(values < -(1.5 * bound[..., None] + 0.5), 0, values)
        order = jnp.argsort(values, axis=-1, stable=True)
        return (jnp.take_along_axis(values, order, axis=-1),
                jnp.take_along_axis(vectors, order[..., None, :], axis=-1))
    return solve


@lru_cache(maxsize=None)
def round_program(mesh_xy, native_eigh, ordered, odd_moments, keep_budget, retain_span=False,
                  gram_keep=None):
    """Pack, assemble, reduce, gate and sort a round of parents, each on its own rank.

    One program over batch layout: each rank packs its slots' Q, WQ, dWQ
    panels by their own ``round_tables`` rows, assembles the even or ordered
    pencils, reduces them with the local eigensolver ``native_eigh`` (made
    zero-row safe), applies the zero policy, forms the retained (even) or
    original-infinity (ordered, odd moments) moment identity and sorts the
    poles; a rank's slots are one batch of independent parents. Every slot
    solves at the round's laddered extent (``round_tables``); its inert
    columns are exact zeros that the eigensolver wrapper keeps out of every
    spectrum, so rounds and SC maps share the program's shapes. A synthetic
    slot (``live`` False) holds only inert columns; its results are selected
    to exact zeros. No array crosses ranks: the models stay in batch layout
    for ``round_checks``; only vectors are gathered.

    Arguments ``(live [P], points, order, active, Qs, WQs, dWQs, infinity)``, all
    in batch layout. Returns ``(model, signed, vectors, (reduction, zero,
    retained, permutation))`` in round order: model (b [P, n, side], poles2,
    active) and signed (c, mu, retained; ordered route, else ``()``) in batch
    layout, ``vectors`` = (poles2, active) and the diagnostics replicated.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.shard_map import shard_map
    from gw.shared_pole_gates import sort_shared_pole_columns
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1, shared_real_pole_gates_v1_r3b

    gates = shared_real_pole_gates_ordered_v1 if ordered else shared_real_pole_gates_v1_r3b
    batch, replicated = P(BATCH), NamedSharding(mesh_xy, P())
    eigh = zero_row_safe_eigh(native_eigh)

    def solve(points, q, o, d, infinity, active):
        return solve_parent_pencil(points, q, o, d, infinity, active,
            eigh=eigh, matmul=_mm, gates=gates, ordered=ordered,
            odd_moments=odd_moments, keep_budget=keep_budget, retain_span=retain_span, gram_keep=gram_keep)

    def body(live, points, order, active, qs, os, ds, infinity):
        def pack(panels):
            # Each slot's own column order; index sum(widths) is the zero column.
            return jnp.take_along_axis(jnp.concatenate(panels, axis=-1), order[:, None, :], axis=-1,
                                       mode='fill', fill_value=0)
        reduced = solve(points, pack(qs), pack(os), pack(ds), infinity, active)
        # Synthetic slots of a rank's batch are selected to exact zeros.
        keep = lambda a: jnp.where(live.reshape(live.shape + (1,) * (a.ndim - 1)), a, jnp.zeros((), a.dtype))
        reduced = jax.tree.map(keep, reduced)
        model, signed, diagnostics = reduced[:3]
        model, permutation = sort_shared_pole_columns(model)
        result = model, signed, (*diagnostics, permutation)
        return (*result,reduced[3]) if retain_span else result

    mapped = shard_map(body, mesh=mesh_xy, in_specs=(batch,) * 8, out_specs=batch, check_vma=False)

    @jax.jit
    def execute(live, points, order, active, qs, os, ds, infinity):
        reduced = mapped(live, points, order, active, qs, os, ds, infinity)
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
                mesh_xy, eigh_plan, ordered):
    """Run ``round_checks`` through the plan matching the arrays' layout."""
    import jax
    import numpy as np
    from jax.sharding import NamedSharding, PartitionSpec as P

    from gw.shared_pole_execution import is_face
    if is_face(model[0]):
        # A face round holds ``real`` physical parents on the leading axis;
        # the gate equations are per parent, so each is checked on its own
        # leading row (no data moves) and the replicated rows are stacked.
        from gw.shared_pole_execution import face_round_check_program
        if int(model[0].shape[0]) != int(real):
            raise ValueError('whole-mesh shared-pole checks take physical parents only')
        program = face_round_check_program(mesh_xy, bool(ordered), model[0].shape[-2])
        rows = []
        for slot in range(int(real)):
            pick = partial(_leading_row, slot=slot)
            out = program(jax.tree.map(pick, model), jax.tree.map(pick, signed),
                          pick(inverse_coulomb_sqrt), *(pick(h) for h in held),
                          np.asarray(nodes, np.complex128), np.float64(eta_ry),
                          *(pick(m) for m in moments), pick(infinity_directions))
            rows.append(jax.tree.map(np.asarray, out))
        return jax.tree.map(lambda *v: np.concatenate(v, axis=0), *rows)

    replicated = NamedSharding(mesh_xy, P())
    if int(model[0].shape[0]) != int(mesh_xy.size):
        raise ValueError('batch-layout shared-pole checks take one parent per rank')
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
    from common.shard_map import shard_map
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
