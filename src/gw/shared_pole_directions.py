"""Direction selection and the per-parent state panels the pencil consumes.

Line, imaginary and infinity supports (SP, section 3), the ordered partner
directions and their dedupe (SP 13), the retained per-parent callables, and the
model diagnostics evaluated against the bank moments.
"""

from __future__ import annotations

from functools import lru_cache

import distrib_la
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P


def _sample_point(recipe, sample_id):
    """Read a deduplicated physical point from the canonical role arrays."""
    rows = np.flatnonzero(np.asarray(recipe["distinct_id"]) == sample_id)
    points = [complex(value["real"], value["imag"]) if isinstance(value, dict)
              else complex(value) for value in recipe["z_ry"]]
    values = np.asarray(points)[rows]
    if not len(rows) or not np.all(values == values[0]):
        raise ValueError("GATE shared_pole_sample_identity: got: absent/inconsistent point; want: one z per distinct_id; why: roles share bank evaluations only")
    return complex(values[0])


@lru_cache(maxsize=None)
def _round_kernels(mesh):
    """Rank-local programs on batch-layout round stacks [P, ...] (one parent per rank).

    Every array argument carries its parent axis over ('x','y'); scalars are
    replicated. No program moves a whole matrix between ranks; the mirror
    exchange moves only [n, r] panels between partner ranks.
    """
    from types import SimpleNamespace
    from common.shard_map import shard_map

    batch, rep = P(('x', 'y')), P()
    adjoint = lambda a: jnp.conj(jnp.swapaxes(a, -1, -2))

    def program(fn, specs, out):
        return jax.jit(shard_map(fn, mesh=mesh, in_specs=specs, out_specs=out, check_vma=False))

    def sample(stack, j):
        return jax.lax.dynamic_index_in_dim(stack, j, axis=1, keepdims=False)

    @lru_cache(maxsize=None)
    def take(indices):
        return program(lambda w: jnp.take(w, jnp.asarray(indices), axis=1), (batch,), batch)

    @lru_cache(maxsize=None)
    def act(conjugate):
        # Output and action of state x at sample j: W x, dW x (adjoints for a conjugate state).
        def body(w, dw, j, x, scale):
            a, d = sample(w, j), sample(dw, j)
            if conjugate:
                a, d = adjoint(a), adjoint(d)
            return a @ x, (d @ x) * scale
        return program(body, (batch, batch, rep, batch, rep), (batch, batch))

    @lru_cache(maxsize=None)
    def exchange(flags, perm):
        """Mirror products with the partner rank's samples, R_s realized on the panels.

        ``xs`` [P, k, n, r]: the states of one sample, ``flags[k]`` True for a
        conjugate state. Slot r sends Pi^T Phi x to its partner, which returns
        W^T y (original) or conj(W) y (conjugate) and the same for dW; slot r
        applies Phi^* Pi: (R_s W)^T x and conj(R_s W) x, R_s W = Phi Pi W Pi^T Phi^*.
        """
        def body(w, dw, xs, alpha, inverse, phase, j, scales):
            y = jnp.take_along_axis(phase[:, None, :, None] * xs, inverse[:, None, :, None], axis=-2)
            y = jax.lax.ppermute(y, BATCH_AXES, perm)
            a, d = sample(w, j), sample(dw, j)
            outs = []
            for k, conjugate in enumerate(flags):
                op_a = jnp.conj(a) if conjugate else jnp.swapaxes(a, -1, -2)
                op_d = jnp.conj(d) if conjugate else jnp.swapaxes(d, -1, -2)
                outs += [op_a @ y[:, k], op_d @ y[:, k]]
            back = jax.lax.ppermute(jnp.stack(outs, axis=1), BATCH_AXES, perm)
            back = jnp.conj(phase)[:, None, :, None] * jnp.take_along_axis(back, alpha[:, None, :, None], axis=-2)
            return tuple(back[:, i] * (scales[i // 2] if i % 2 else 1) for i in range(2 * len(flags)))
        return program(body, (batch,) * 6 + (rep, rep), (batch,) * (2 * len(flags)))

    def dedupe(q, o):
        # O W-output of the direction set Q: the part of O outside span(Q), and O O^H for its scale.
        rest = o - q @ (adjoint(q) @ o)
        herm = lambda a: (a + adjoint(a)) / 2
        return herm(o @ adjoint(o)), herm(rest @ adjoint(rest))

    column = program(lambda q, e: jax.lax.dynamic_index_in_dim(q, e, axis=1, keepdims=False), (batch, rep), batch)
    apply = program(lambda m, q: m @ q, (batch, batch), batch)
    negative_hermitian = program(lambda a: -(a + adjoint(a)) / 2, (batch,), batch)
    return SimpleNamespace(
        take=take, column=column, negative_hermitian=negative_hermitian, exchange=exchange, act=act, apply=apply,
        dedupe=program(dedupe, (batch, batch), (batch, batch)))


BATCH_AXES = ('x', 'y')


def select_round_states(samples, recipe, *, sample_lo, real, mesh_xy, eigh_plan, svd_plan,
                        column_extent, logical_n, ordered=False, exchange=None):
    """Directions, outputs and actions of one round of parents, batched per role (SP 3, SP 13).

    ``samples`` holds ``Wc``/``dWc_ds`` [P, S, n, n] in batch layout (rank r owns
    round slot r), sample ``sample_lo + j`` at index j; slots ``>= real`` are
    synthetic. Line supports select right singular vectors (cutoff, per-support
    cap), imaginary supports leading eigenvectors of -Herm W, each role in ONE
    batched call over the round's [slot x sample] stack; only spectra cross the
    host. States follow the per-sample order of the paired layout: per sample
    the role state and, when its node is off the real s axis, the conjugate
    state on O = W Q (ordered imaginary or Re z = 0 roles: the partner directions
    of O orthogonal to Q, per-slot widths, 0 allowed); after all originals the
    ordered mirrors X(-node) on the same directions. A mirror of a Re z = 0
    sample uses its own sample; any other mirror uses the partner parent's
    sample through ``exchange = (slots, alpha, inverse, phase)``
    (``shared_pole_local.partner_realization``): W_q(-conj z) = conj(R_s[W_q(p')](z)).

    Returns ``(states, counts, roles)``: panels [P, n, r] in batch layout, counts
    int [P, A], and per-slot role records.
    """
    from gw.shared_pole_recipe import ROLE_CODES

    k = _round_kernels(eigh_plan.mesh)
    W, dW = samples["Wc"], samples["dWc_ds"]
    ranks = int(W.shape[0])
    names = {code: name for name, code in ROLE_CODES.items()}
    fit_ids = [int(i) for i in recipe["fit_ids"]]
    entries = [(int(sample), names[int(role)] + f":{i}")
               for i, (sample, role, held) in enumerate(zip(recipe["distinct_id"], recipe["role"], recipe["held"]))
               if not held]
    entries = [(sid, label) for sid in fit_ids for sample, label in entries if sample == sid]
    kinds = {kind: [(sid, label) for sid, label in entries if label.split(":", 1)[0] == kind]
             for kind in ("line", "imaginary")}
    if len(kinds["line"]) + len(kinds["imaginary"]) != len(entries):
        raise ValueError("GATE shared_pole_role: got: a fitted role other than line or imaginary; want: line or imaginary fitted role; why: unknown tangent semantics")
    tol = recipe["multiplet_relative_tolerance"]
    put = lambda value: jax.device_put(value, NamedSharding(mesh_xy, P()))
    selected = {}
    if kinds["line"]:
        stack = k.take(tuple(sid - sample_lo for sid, _ in kinds["line"]))(W)
        selected["line"] = distrib_la.right_singular_vectors(
            stack, recipe["direction_cutoff"], eigh_plan=svd_plan, column_extent=column_extent,
            multiplet_tol=tol, real_rows=real, max_rank=recipe.get("line_direction_cap"))
        del stack
    if kinds["imaginary"]:
        stack = k.negative_hermitian(k.take(tuple(sid - sample_lo for sid, _ in kinds["imaginary"]))(W))
        width = min(logical_n, max(1, int(recipe["imaginary_width"])))
        selected["imaginary"] = distrib_la.leading_eigenvectors(
            stack, width, eigh_plan=eigh_plan, column_extent=column_extent, multiplet_tol=tol, real_rows=real)
        del stack

    def widths_of(values):
        return tuple(int(np.asarray(v).size) for v in values)

    states, counts, flags = [], [], []
    roles = [[] for _ in range(ranks)]
    mirror_roles = [[] for _ in range(ranks)]
    mirrors, mirror_counts = [], []
    for sid in fit_ids:
        first = len(states)
        z = _sample_point(recipe, sid)
        j = put(np.int32(sid - sample_lo))
        for kind in ("line", "imaginary"):
            for e, (esid, label) in enumerate(kinds[kind]):
                if esid != sid:
                    continue
                q_all, values = selected[kind]
                direction = k.column(q_all, put(np.int32(e)))
                widths = widths_of(tuple(row[e] for row in values))
                if any(w < 1 or w > logical_n for w in widths[:real]):
                    raise ValueError(f"GATE shared_pole_directions: got: ranks {widths[:real]}; want: 1..{logical_n}; why: empty or padded physical direction set")
                s = z if ordered else z ** 2
                for conjugate in ((False, True) if (ordered or kind == "line") and s.imag != 0 else (False,)):
                    state_widths = widths
                    if conjugate and ordered and (kind == "imaginary" or s.real == 0):
                        direction, state_widths = _round_partner_directions(
                            states[-1][1], states[-1][2], recipe["direction_cutoff"], real=real,
                            kernels=k, eigh_plan=eigh_plan, column_extent=column_extent, tol=tol)
                        if direction is None:
                            continue
                    elif conjugate:
                        direction = states[-1][2]
                    scale = put(np.complex128(2 * (s.conjugate() if conjugate else s) if ordered else 1.0))
                    output, action = k.act(bool(conjugate))(W, dW, j, direction, scale)
                    states.append((s.conjugate() if conjugate else s, direction, output, action))
                    flags.append(conjugate)
                    counts.append(state_widths)
                    for r, width in enumerate(state_widths):
                        roles[r].append({"sample_id": sid, "role": label, "conjugate": conjugate,
                                         "width": int(width), "carrier_width": column_extent(int(width))})
        if ordered and len(states) > first:
            # W(-z) = W(-conj z)^H on Q; W(-conj z) on the partner's O. dW/dz at the
            # mirror node is -2 node dW/ds(-conj z).
            span = range(first, len(states))
            if z.real == 0:
                results = []
                for index in span:
                    node = states[index][0]
                    results.append(k.act(not flags[index])(W, dW, j, states[index][1],
                                                           put(np.complex128(-2 * node))))
            else:
                slots, alpha, inverse, phase = exchange
                program = k.exchange(tuple(flags[index] for index in span),
                                     tuple((int(r), int(slots[r])) for r in range(ranks)))
                xs = jnp.stack([states[index][1] for index in span], axis=1)
                flat = program(W, dW, xs, alpha, inverse, phase, j,
                               put(np.asarray([-2 * states[index][0] for index in span], np.complex128)))
                results = [(flat[2 * i], flat[2 * i + 1]) for i in range(len(span))]
                del xs
            for index, (output, action) in zip(span, results):
                mirrors.append((-states[index][0], states[index][1], output, action))
                mirror_counts.append(counts[index])
                for r in range(ranks):
                    mirror_roles[r].append(dict(roles[r][index], mirror=True))
    if ordered:
        states, counts = states + mirrors, counts + mirror_counts
        roles = [row + mirror_row for row, mirror_row in zip(roles, mirror_roles)]
    return states, np.asarray(counts, dtype=np.int64).T, roles


def _round_partner_directions(q, output, cutoff, *, real, kernels, eigh_plan, column_extent, tol):
    """Partner directions of O = W Q orthogonal to Q above the direction cutoff, per slot.

    For a role whose conjugate node brings no new tangent under time reversal
    (imaginary z, or Re z = 0), O lies in span(Q) on time-reversal-symmetric data.
    The component of O outside span(Q) is rank-revealed against the largest
    singular value of O with the line cutoff; on a magnet the survivors carry the
    odd channel. Each slot keeps its own count (0 allowed); only spectra cross the
    host. Returns (directions [P, n, r], widths) or (None, None) when no slot has any.
    """
    top, perp = kernels.dedupe(q, output)
    m = int(perp.shape[-1])
    _, largest = distrib_la.leading_eigenvectors(top, 1, eigh_plan=eigh_plan, column_extent=column_extent,
                                                 multiplet_tol=tol, real_rows=real)
    _, spectrum = distrib_la.leading_eigenvectors(perp, m, eigh_plan=eigh_plan, column_extent=column_extent,
                                                  multiplet_tol=tol, real_rows=real)
    counts = tuple(int(np.sum(np.asarray(values) > float(cutoff) ** 2 * float(np.asarray(t)[0])))
                   if i < real else 0 for i, (values, t) in enumerate(zip(spectrum, largest)))
    if max(counts) == 0:
        return None, None
    directions, kept = distrib_la.leading_eigenvectors(perp, counts, eigh_plan=eigh_plan, column_extent=column_extent,
                                                       multiplet_tol=tol, real_rows=real)
    return directions, tuple(int(np.asarray(v).size) for v in kept)


def _model_diagnostics(model, moments, infinity_directions, *, matmul):
    """Compare M1=bb.H/2 and M3=b Lambda b.H/2 with physical moments.

    Factors are [b,n,K], moments [b,n,n], and infinity directions [b,n,r],
    in their existing face layouts. The loop selects from the two resident
    moments without stacking dense matrices; only [2,b] scalar defects are
    stacked. The caller compiles this stage with its accounted service GEMM.
    """
    b, poles, _ = model

    def moment_defect(third):
        target = jax.lax.cond(third, lambda: moments["M3"],
                              lambda: moments["M1"])
        weighted = jax.lax.cond(third, lambda: b * poles[:, None, :],
                                lambda: b)
        value = matmul(weighted, b, transb="C") / 2
        defect = target - value
        projected = matmul(infinity_directions,
                           matmul(defect, infinity_directions), transa="C")
        projected_target = matmul(infinity_directions,
                                  matmul(target, infinity_directions), transa="C")
        norm = jnp.linalg.norm(target, axis=(-2, -1))
        projected_norm = jnp.linalg.norm(projected_target, axis=(-2, -1))
        return (jnp.linalg.norm(defect, axis=(-2, -1)) / norm,
                jnp.linalg.norm(projected, axis=(-2, -1)) / projected_norm)

    full, projected = jax.lax.map(moment_defect, jnp.asarray([False, True]))
    return {
        "M1": {"full_relative": full[0], "original_infinity_relative": projected[0]},
        "M3": {"full_relative": full[1], "original_infinity_relative": projected[1]},
    }
