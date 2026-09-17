"""Rounds of shared-pole parents, one parent per mesh rank (batch layout).

The round schedule and its partner table, the packed -q realization, the
column tables and the one round program that packs, assembles, reduces, gates
and sorts every parent of a round on its own rank with local dense kernels,
and the restore of round results to the face.
"""
from functools import lru_cache

# Batch layout: rank x*Py + y owns whole parents; every other axis is unsharded.
BATCH = ('x', 'y')


def parent_rounds(nq, ranks, partner=None):
    """Rounds of ``ranks`` parent slots, one parent per rank, in canonical order.

    Without ``partner`` a round is the next contiguous run of parents. With
    ``partner`` (int [nq], an involution: the raw parent that carries each
    parent's -q) a round is partner-closed: a parent enters with its partner,
    so the mirror exchange of a round never leaves it. A short round repeats its
    last real parent in the synthetic slots, which are never solved.

    Returns ``[(ids, real, slots)]``: ``ids`` the ``ranks`` parent ids, ``real``
    the number of leading real slots, ``slots[r]`` the slot of slot r's
    partner (synthetic slots pair with themselves).
    """
    import numpy as np

    if partner is None:
        groups = [[q] for q in range(nq)]
    else:
        partner = [int(v) for v in partner]
        if any(partner[partner[q]] != q for q in range(nq)):
            raise ValueError("GATE minus_q_partner: got: a partner table that is not an involution; want: "
                             "partner[partner[p]] == p; why: the round exchange pairs slots")
        groups, seen = [], set()
        for q in range(nq):
            if q not in seen:
                group = [q] if partner[q] == q else [q, partner[q]]
                seen.update(group)
                groups.append(group)
    rounds, current = [], []
    for group in groups:
        if len(group) > ranks:
            raise ValueError(f"GATE shared_pole_round: got: a partner pair on a mesh of {ranks} rank; "
                             "want: at least two ranks for an ordered deck with -q != q; why: one parent per rank")
        if len(current) + len(group) > ranks:
            rounds.append(current)
            current = []
        current += group
    if current:
        rounds.append(current)
    out = []
    for ids in rounds:
        real = len(ids)
        slots = [ids.index(partner[q]) if partner is not None else r for r, q in enumerate(ids)]
        ids = ids + [ids[-1]] * (ranks - real)
        slots = slots + list(range(real, ranks))
        out.append((ids, real, np.asarray(slots, np.int64)))
    return out


@lru_cache(maxsize=None)
def batch_to_face(mesh_xy):
    """[B, m, r] at P(('x','y'), None, None) -> the face P(None, 'x', 'y'): y then x all_to_all."""
    import jax
    from jax.sharding import PartitionSpec as P
    from common.shard_map import shard_map
    px, py = int(mesh_xy.shape['x']), int(mesh_xy.shape['y'])

    def restore(a):
        if py > 1:
            a = jax.lax.all_to_all(a, 'y', split_axis=2, concat_axis=0, tiled=True)
        if px > 1:
            a = jax.lax.all_to_all(a, 'x', split_axis=1, concat_axis=0, tiled=True)
        return a
    return jax.jit(shard_map(restore, mesh=mesh_xy, in_specs=P(BATCH), out_specs=P(None, 'x', 'y'),
                             check_vma=False))


@lru_cache(maxsize=None)
def face_rows(mesh_xy, rows):
    """Select parent rows of a face stack [B, m_X, r_Y] on its replicated leading axis."""
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    index = tuple(int(r) for r in rows)
    return jax.jit(lambda a: a[jnp.asarray(index)], out_shardings=NamedSharding(mesh_xy, P(None, 'x', 'y')))


def partner_realization(meta, header, ids, partner_parent, partner_row, *, mesh_xy):
    """Per-slot packed action of each parent's -q partner row, in batch layout.

    For slot r (parent ids[r], partner p', row s): ``alpha[r]`` the packed source
    permutation of s, ``inverse[r]`` its inverse, ``phase[r]`` = exp(2 pi i
    q(p') . L_s) on the packed rows, so R_s[W] = Phi Pi W Pi^T Phi^*. Host tables
    only; each rank receives its own slot's rows.
    """
    import numpy as np
    from gw.qgrid_symmetry import shared_pole_packed_action

    packed, wraps, _ = shared_pole_packed_action(meta, header, mesh_xy=mesh_xy)
    q_frac = np.asarray(header["qirr"]["q_irr_frac"], dtype=np.float64)
    rows = [int(partner_row[q]) for q in ids]
    parents = [int(partner_parent[q]) for q in ids]
    alpha = np.asarray([packed[s] for s in rows], np.int32)
    inverse = np.argsort(alpha, axis=1).astype(np.int32)
    phase = np.exp(2j * np.pi * np.einsum('ri,rmi->rm', q_frac[parents],
                                          np.asarray([wraps[s] for s in rows], np.float64)))
    return (_batch_put(mesh_xy, alpha), _batch_put(mesh_xy, inverse),
            _batch_put(mesh_xy, phase.astype(np.complex128)))


def round_tables(counts, widths, nodes, infinity_counts, infinity_width, *, column_extent, ordered,
                 odd_moments):
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
    spectrum length at its own extent, for the receipt.
    """
    import numpy as np

    counts = np.asarray(counts, np.int64)
    ranks, states = counts.shape
    offsets = np.concatenate(([0], np.cumsum(widths))).astype(np.int64)
    carriers = np.asarray([[column_extent(int(c)) for c in row] for row in counts], np.int64)
    halves = (range(states // 2), range(states // 2, states)) if ordered else (range(states),)
    extent = max(int(carriers[:, list(half)].sum(axis=1).max()) for half in halves)
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
    own = (carriers[:, list(halves[0])].sum(axis=1)
           + np.asarray([column_extent(int(c)) if blocks else 0 for c in infinity_counts], np.int64))
    return dict(order=order, points=points, active=np.concatenate((live,) + (tail,) * blocks, axis=1), own=own)


def own_extent_receipts(reduction, own):
    """Per-slot receipt rows of a round reduction at each parent's own extent.

    A parent solved at the round extent carries the padding's exact zeros in its
    ascending normalized Gram spectrum. Its receipt drops that many entries of
    smallest magnitude, reads ``gram_min_relative`` from what remains, and
    normalizes the metric residual by its own side. ``reduction`` holds host
    arrays [P, ...]; returns one dict of [1, ...] arrays per entry of ``own``.
    """
    import numpy as np

    rows = []
    for slot, side in enumerate(int(v) for v in own):
        row = {key: value[slot:slot + 1] for key, value in reduction.items()}
        spectrum = row["gram_spectrum_relative"][0]
        kept = np.delete(spectrum, np.argsort(np.abs(spectrum), kind="stable")[:spectrum.size - side])
        row["gram_spectrum_relative"] = kept[None]
        row["gram_min_relative"] = kept[:1]
        row["metric_inverse_root_residual_relative"] = row["metric_inverse_root_residual_fro"] / np.sqrt(side)
        rows.append(row)
    return rows


def _batch_put(mesh_xy, a):
    import jax
    from jax.sharding import NamedSharding, PartitionSpec as P
    return jax.make_array_from_callback(a.shape, NamedSharding(mesh_xy, P(BATCH)), lambda idx: a[idx])


def reduce_round(states, infinity, tables, *, real, mesh_xy, native_eigh, ordered, odd_moments, keep_budget):
    """Run ``round_program`` on one round: host tables in, round-order results out."""
    import numpy as np

    put = lambda a: _batch_put(mesh_xy, np.asarray(a))
    program = round_program(mesh_xy, native_eigh, bool(ordered), bool(odd_moments),
                            None if keep_budget is None else int(keep_budget))
    live = np.arange(len(tables["own"])) < int(real)
    return program(put(live), put(tables["points"]), put(tables["order"]), put(tables["active"]),
                   tuple(st[1] for st in states), tuple(st[2] for st in states),
                   tuple(st[3] for st in states), tuple(infinity))


@lru_cache(maxsize=None)
def round_program(mesh_xy, native_eigh, ordered, odd_moments, keep_budget):
    """Pack, assemble, reduce, gate and sort a round of parents, each on its own rank.

    One program over batch layout: rank r packs slot r's Q, WQ, dWQ panels by
    ``round_tables``, assembles the even or ordered pencil, reduces it with the
    local eigensolver ``native_eigh``, applies the zero policy, forms the
    retained (even) or original-infinity (ordered, odd moments) moment identity
    and sorts the poles. A synthetic slot (``live`` False) skips all of it through
    ``lax.cond``. No array crosses ranks until the end: the factors return to the
    face through the two all_to_alls of ``batch_to_face``, the vectors
    replicated.

    Arguments ``(live [P], points, order, active, Qs, WQs, dWQs, infinity)``, all
    in batch layout. Returns ``(model, signed, (reduction, zero, retained,
    permutation))`` in round order: model (b face [P, n, side], poles2, active),
    signed (c face, mu, retained) on the ordered route else ``()``.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.shard_map import shard_map
    from gw.shared_pole_gates import (apply_shared_pole_zero_policy, ordered_moment_identity,
                                      retained_moment_identity, sort_shared_pole_columns)
    from gw.shared_pole_pencil import assemble_ordered_shared_pole_pencil, assemble_shared_pole_pencil
    from gw.shared_pole_reduction import reduce_ordered_shared_pole_pencil, reduce_shared_pole_pencil
    from gw.shared_pole_recipe import shared_real_pole_gates_ordered_v1, shared_real_pole_gates_v1_r3b

    gates = shared_real_pole_gates_ordered_v1 if ordered else shared_real_pole_gates_v1_r3b
    batch, replicated = P(BATCH), NamedSharding(mesh_xy, P())
    to_face = batch_to_face(mesh_xy)

    def mm(a, b, *, transa='N', transb='N'):
        def op(value, trans):
            if trans == 'N':
                return value
            value = jnp.swapaxes(value, -1, -2)
            return jnp.conj(value) if trans == 'C' else value
        return jnp.matmul(op(a, transa), op(b, transb))

    def solve(points, q, o, d, infinity, active):
        finite = [(points, q, o, d)]
        if ordered:
            pencil = assemble_ordered_shared_pole_pencil(finite, infinity if odd_moments else None, matmul=mm)
            model, signed, reduction = reduce_ordered_shared_pole_pencil(
                pencil, active, eigh=native_eigh, matmul=mm, gates=gates, keep_budget=keep_budget)
            retained = ordered_moment_identity(signed, infinity, matmul=mm) if odd_moments else {}
            model, zero = apply_shared_pole_zero_policy(model, gates=gates)
            zero["zero_policy"] = zero["zero_policy"] & reduction["infinite_weight_ok"]
        else:
            pencil = assemble_shared_pole_pencil(finite, infinity, matmul=mm)
            model, reduction, coefficients = reduce_shared_pole_pencil(
                pencil, active, eigh=native_eigh, matmul=mm, gates=gates, keep_budget=keep_budget)
            model, zero = apply_shared_pole_zero_policy(model, gates=gates)
            # E selects the infinity block, the last columns of X.
            side, width = pencil[0].shape[-1], infinity[0].shape[-1]
            selector = (jnp.arange(side)[:, None] == jnp.arange(side - width, side)[None, :])[None]
            retained = retained_moment_identity(pencil, coefficients, model, selector.astype(jnp.complex128),
                                                matmul=mm)
            signed = ()
        model, permutation = sort_shared_pole_columns(model)
        return model, signed, (reduction, zero, retained, permutation)

    def body(live, points, order, active, qs, os, ds, infinity):
        def pack(panels):
            return jnp.take(jnp.concatenate(panels, axis=-1), order[0], axis=-1, mode='fill', fill_value=0)
        args = (points, pack(qs), pack(os), pack(ds), infinity, active)

        def skip(args):
            return jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), jax.eval_shape(solve, *args))
        return jax.lax.cond(live[0], lambda args: solve(*args), skip, args)

    mapped = shard_map(body, mesh=mesh_xy, in_specs=(batch,) * 8, out_specs=batch, check_vma=False)

    @jax.jit
    def execute(live, points, order, active, qs, os, ds, infinity):
        model, signed, diagnostics = mapped(live, points, order, active, qs, os, ds, infinity)
        gather = lambda tree: jax.tree.map(lambda a: jax.lax.with_sharding_constraint(a, replicated), tree)
        model = (to_face(model[0]), *gather(model[1:]))
        signed = (to_face(signed[0]), *gather(signed[1:])) if ordered else ()
        return model, signed, gather(diagnostics)
    return execute


@lru_cache(maxsize=None)
def local_model_checks(mesh_xy, native_eigh):
    """Check held W/dW and V-whitened passivity with independent local parents.

    Model faces are [b,n,K], inverse Coulomb faces [b,n,n], held samples
    [b,s,n,n], and squared Ry supports [s]. Complete parents move once to
    ranks; all held-point products stay inside the local mapped body. Only
    small diagnostic arrays return replicated. Callers admit padded batches.
    """
    import jax
    import jax.numpy as jnp
    from jax.sharding import NamedSharding, PartitionSpec as P
    from common.shard_map import shard_map
    from common.staged_reshard import face_to_batch_reshard
    from gw.shared_pole_gates import shared_pole_passivity, shared_pole_reciprocity
    from gw.shared_pole_recipe import shared_real_pole_gates_v1_r3b as gates

    to_batch = face_to_batch_reshard(mesh_xy)
    qspec = P(('x', 'y'))
    def mm(a, b, *, transb='N'):
        return a @ (jnp.conj(jnp.swapaxes(b, -1, -2)) if transb == 'C' else b)

    def one(args, supports, eta):
        b, poles, mask, inverse, wc, dw = args
        model = (b[None], poles[None], mask[None])
        passive = shared_pole_passivity(model, inverse[None], eta_ry=eta,
                                        matmul=mm, eigh=native_eigh, gates=gates)
        # W(s)=b(s-Lambda)^-1 b.H; dW/ds=-b(s-Lambda)^-2 b.H.
        weights = jnp.where(mask[None], 1/(supports[:, None]-poles[None]), 0)
        factors = b[None] * jnp.stack((weights, -weights**2))[:, :, None, :]
        values = factors @ jnp.conj(b.T)
        exact = jnp.stack((wc, dw))
        errors = jnp.linalg.norm(values-exact, axis=(-2, -1)) / jnp.maximum(
            jnp.linalg.norm(exact, axis=(-2, -1)), jnp.finfo(jnp.float64).tiny)
        reciprocity = shared_pole_reciprocity(values, exact, gates=gates)
        return jax.tree.map(lambda a: a[0], passive), errors, reciprocity

    mapped = shard_map(
        lambda b, p, m, inv, w, dw, s, eta: jax.lax.map(
            lambda row: one(row, s, eta), (b, p, m, inv, w, dw)),
        mesh=mesh_xy, in_specs=(qspec,)*6+(P(), P()),
        out_specs=(qspec, qspec, qspec), check_vma=False)

    @jax.jit
    def execute(model, inverse, wc, dw, supports, eta):
        b, poles, mask = model
        batch = b.shape[0]
        def pad(a):
            return jnp.concatenate((a, jnp.repeat(a[-1:], batch-a.shape[0], axis=0)), axis=0)
        def sample_move(a):
            a = pad(a)
            b, s, n, _ = a.shape
            # Keep spatial x tiles contiguous while folding the replicated
            # sample axis into M for the canonical volume-preserving move.
            face = jnp.transpose(a, (0, 2, 1, 3)).reshape(b, n*s, n)
            local = to_batch(face).reshape(b, n, s, n)
            return jnp.transpose(local, (0, 2, 1, 3))
        scalar = NamedSharding(mesh_xy, qspec)
        out = mapped(to_batch(b), jax.lax.with_sharding_constraint(poles, scalar),
                     jax.lax.with_sharding_constraint(mask, scalar),
                     to_batch(pad(inverse)), sample_move(wc), sample_move(dw), supports, eta)
        return jax.tree.map(lambda a: jax.lax.with_sharding_constraint(
            a, NamedSharding(mesh_xy, P())), out)
    return execute
