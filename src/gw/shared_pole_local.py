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


def partner_realization(meta, header, ids, partner_parent, partner_row, *, mesh_xy, components=1):
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
    if components != 1:
        alpha = (alpha[:, :, None] * components + np.arange(components)).reshape(len(ids), -1).astype(np.int32)
        inverse = np.argsort(alpha, axis=1).astype(np.int32)
        phase = np.repeat(phase, components, axis=1)
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
    spectrum length at its own extent, for the receipt; ``extents`` [P, 2]
    carries each finite half and infinity block before round padding.
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
    finite = carriers[:, list(halves[0])].sum(axis=1)
    infinity = np.asarray([column_extent(int(c)) if blocks else 0 for c in infinity_counts], np.int64)
    return dict(order=order, points=points, active=np.concatenate((live,) + (tail,) * blocks, axis=1),
                own=finite + infinity, extents=np.column_stack((finite, infinity)))


def own_extent_receipts(reduction, own):
    """Per-slot receipt rows of a round reduction at each parent's own extent.

    A parent's own ascending spectrum is followed by exact-zero round padding.
    Its receipt strips that output padding, reads ``gram_min_relative``, and
    normalizes the metric residual by its own side. ``reduction`` holds host
    arrays [P, ...]; returns one dict of [1, ...] arrays per entry of ``own``.
    """
    import numpy as np

    rows = []
    for slot, side in enumerate(int(v) for v in own):
        row = {key: value[slot:slot + 1] for key, value in reduction.items()}
        kept = row["gram_spectrum_relative"][0, :side]
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
                 retain_span=False):
    """Run ``round_program`` on one round: host tables in, round-order results out."""
    import numpy as np

    put = lambda a: _batch_put(mesh_xy, np.asarray(a))
    extents = [tuple(int(v) for v in row) for row in tables["extents"][:real]]
    sizes = tuple(sorted(set(extents)))
    program = round_program(mesh_xy, native_eigh, bool(ordered), bool(odd_moments),
                            None if keep_budget is None else int(keep_budget), sizes, bool(retain_span))
    live = np.arange(len(tables["own"])) < int(real)
    dispatch = np.zeros(len(live), np.int32)
    dispatch[:real] = [sizes.index(size) for size in extents]
    return program(put(live), put(dispatch), put(tables["points"]), put(tables["order"]), put(tables["active"]),
                   tuple(st[1] for st in states), tuple(st[2] for st in states),
                   tuple(st[3] for st in states), tuple(infinity))


def finalize_ordered_parent_pencil(reduced, infinity, *, matmul, gates,
                                   odd_moments, retain_span=False):
    """Apply the ordered parent gates after its three Ritz equations."""
    from gw.shared_pole_gates import apply_shared_pole_zero_policy, ordered_moment_identity

    model, signed, reduction = reduced[:3]
    retained = (ordered_moment_identity(signed, infinity, matmul=matmul)
                if odd_moments else {})
    model, zero = apply_shared_pole_zero_policy(model, gates=gates)
    zero["zero_policy"] = zero["zero_policy"] & reduction["infinite_weight_ok"]
    result = model, signed, (reduction, zero, retained)
    if retain_span:
        return (*result, reduced[3])
    return result


def solve_parent_pencil(points, q, o, d, infinity, active, *, eigh, matmul,
                        gates, ordered, odd_moments, keep_budget, retain_span=False, matrix_sharding=None):
    """One equation owner for local and whole-mesh parent execution.

    Inputs carry one or more independent parents. Execution adapters supply
    service/local products and eigensolve; this function owns the pencil,
    reduction, zero policy and original/retained moment identities.
    """
    import jax.numpy as jnp
    from gw.shared_pole_gates import apply_shared_pole_zero_policy, retained_moment_identity
    from gw.shared_pole_pencil import assemble_ordered_shared_pole_pencil, assemble_shared_pole_pencil
    from gw.shared_pole_reduction import reduce_ordered_shared_pole_pencil, reduce_shared_pole_pencil
    finite = [(points, q, o, d)]
    if ordered:
        pencil = assemble_ordered_shared_pole_pencil(finite, infinity if odd_moments else None,
                                                     matmul=matmul, matrix_sharding=matrix_sharding)
        reduced = reduce_ordered_shared_pole_pencil(
            pencil, active, eigh=eigh, matmul=matmul, gates=gates, keep_budget=keep_budget,
            retain_span=retain_span, matrix_sharding=matrix_sharding)
        return finalize_ordered_parent_pencil(
            reduced, infinity, matmul=matmul, gates=gates,
            odd_moments=odd_moments, retain_span=retain_span)
    else:
        pencil = assemble_shared_pole_pencil(finite, infinity, matmul=matmul)
        model, reduction, coefficients = reduce_shared_pole_pencil(
            pencil, active, eigh=eigh, matmul=matmul, gates=gates, keep_budget=keep_budget)
        model, zero = apply_shared_pole_zero_policy(model, gates=gates)
        # E selects the infinity block, the last columns of X.
        side, width = pencil[0].shape[-1], infinity[0].shape[-1]
        selector = (jnp.arange(side)[:, None] == jnp.arange(side - width, side)[None, :])[None]
        retained = retained_moment_identity(pencil, coefficients, model, selector.astype(jnp.complex128),
                                            matmul=matmul)
        signed = ()
    result = model, signed, (reduction, zero, retained)
    if retain_span:
        return (*result, coefficients)
    return result


@lru_cache(maxsize=None)
def round_program(mesh_xy, native_eigh, ordered, odd_moments, keep_budget, sizes, retain_span=False):
    """Pack, assemble, reduce, gate and sort a round of parents, each on its own rank.

    One program over batch layout: rank r packs slot r's Q, WQ, dWQ panels by
    ``round_tables``, assembles the even or ordered pencil, reduces it with the
    local eigensolver ``native_eigh``, applies the zero policy, forms the
    retained (even) or original-infinity (ordered, odd moments) moment identity
    and sorts the poles. Each parent solves at its own selected extents: large
    artificial zero blocks can make the native eigensolver return NaNs on a
    finite Gram. Only outputs acquire round padding. ``sizes`` contains the
    distinct (finite-half, infinity) extents; the slot dispatch is a runtime
    operand, so reordering parents reuses the same program. A synthetic slot (``live`` False) skips all of it through
    ``lax.cond``. No array crosses ranks: the models stay in batch layout for
    ``round_checks``; only vectors are gathered.

    Arguments ``(live [P], dispatch [P], points, order, active, Qs, WQs, dWQs, infinity)``, all
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

    def solve(points, q, o, d, infinity, active):
        return solve_parent_pencil(points, q, o, d, infinity, active,
            eigh=native_eigh, matmul=_mm, gates=gates, ordered=ordered,
            odd_moments=odd_moments, keep_budget=keep_budget, retain_span=retain_span)

    def body(live, dispatch, points, order, active, qs, os, ds, infinity):
        def pack(panels):
            return jnp.take(jnp.concatenate(panels, axis=-1), order[0], axis=-1, mode='fill', fill_value=0)
        args = (points, pack(qs), pack(os), pack(ds), infinity, active)

        halves = 2 if ordered else 1
        blocks = (2 if odd_moments else 0) if ordered else 1
        finite_width = points.shape[-1] // halves
        infinity_width = infinity[0].shape[-1]
        side = active.shape[-1]

        def branch(nf, ni):
            def run(args):
                points, q, o, d, infinity, active = args
                finite = lambda a: jnp.concatenate(
                    [a[..., k * finite_width:k * finite_width + nf] for k in range(halves)], axis=-1)
                mask = jnp.concatenate((finite(active[..., :halves * finite_width]), *[
                    active[..., halves * finite_width + k * infinity_width:
                           halves * finite_width + k * infinity_width + ni] for k in range(blocks)]), axis=-1)
                reduced = solve(finite(points), finite(q), finite(o), finite(d),
                                tuple(a[..., :ni] for a in infinity), mask)
                model, signed, diagnostics = reduced[:3]
                def padded(model, sentinel):
                    b, poles, kept = model
                    pad = side - poles.shape[-1]
                    return (jnp.pad(b, ((0, 0), (0, 0), (0, pad))),
                            jnp.pad(poles, ((0, 0), (0, pad)), constant_values=sentinel),
                            jnp.pad(kept, ((0, 0), (0, pad))))
                model = padded(model, 1)
                if ordered:
                    signed = padded(signed, 0)
                reduction, zero, retained = diagnostics
                reduction['gram_spectrum_relative'] = jnp.pad(
                    reduction['gram_spectrum_relative'], ((0, 0), (0, side // halves - nf - ni)))
                result = model, signed, (reduction, zero, retained)
                if retain_span:
                    # Original round-pencil rows have gaps between finite
                    # halves and infinity blocks. Trailing padding alone
                    # would attach coefficients to the wrong CT states.
                    rows = ([k*finite_width+i for k in range(halves) for i in range(nf)]
                            + [halves*finite_width+k*infinity_width+i
                               for k in range(blocks) for i in range(ni)])
                    retained_side = reduced[3].shape[-1]
                    coefficients = jnp.zeros((1,side,side),reduced[3].dtype)
                    coefficients = coefficients.at[:,jnp.asarray(rows),:retained_side].set(reduced[3])
                    return (*result,coefficients)
                return result
            return run

        branches = tuple(branch(nf, ni) for nf, ni in sizes)
        def work(args):
            return jax.lax.switch(dispatch[0], branches, args)
        def skip(args):
            return jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), jax.eval_shape(branches[0], args))
        reduced = jax.lax.cond(live[0], work, skip, args)
        model, signed, diagnostics = reduced[:3]
        model, permutation = sort_shared_pole_columns(model)
        result = model, signed, (*diagnostics, permutation)
        return (*result,reduced[3]) if retain_span else result

    mapped = shard_map(body, mesh=mesh_xy, in_specs=(batch,) * 9, out_specs=batch, check_vma=False)

    @jax.jit
    def execute(live, dispatch, points, order, active, qs, os, ds, infinity):
        reduced = mapped(live, dispatch, points, order, active, qs, os, ds, infinity)
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
        if real != 1:
            raise ValueError('whole-mesh shared-pole checks require one physical parent')
        from gw.shared_pole_execution import face_round_check_program
        out = face_round_check_program(mesh_xy, bool(ordered), model[0].shape[-2])(
            model, signed, inverse_coulomb_sqrt, *held,
            np.asarray(nodes, np.complex128), np.float64(eta_ry),
            *moments, infinity_directions)
        return jax.tree.map(np.asarray, out)

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
