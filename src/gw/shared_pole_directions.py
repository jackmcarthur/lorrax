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
from distrib_la import hermitian_part
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


def _fit_roles(recipe):
    """Ephemeral constructor record view; only flat arrays are serialized."""
    from gw.shared_pole_recipe import ROLE_CODES
    names = {code: name for name, code in ROLE_CODES.items()}
    return [{"sample_id": int(sample), "role": f"{names[int(role)]}:{i}",
             "held": bool(held)}
            for i, (sample, role, held) in enumerate(zip(
                recipe["distinct_id"], recipe["role"], recipe["held"]))]


@lru_cache(maxsize=256)
def _parent_panel_slice(mesh_xy, width):
    """Slice [b,n,r] direction/action faces without a host or replicated seam."""
    return jax.jit(
        lambda arrays, parent: jax.tree.map(
            lambda a: jax.lax.dynamic_slice_in_dim(a, parent, 1, axis=0)[:, :, :width], arrays),
        out_shardings=NamedSharding(mesh_xy, P(None, 'x', 'y')))



@lru_cache(maxsize=None)
def _parent_result_slice(mesh_xy):
    """Slice a parent's padded factor and scalar receipts in one executable."""
    face = NamedSharding(mesh_xy, P(None, 'x', 'y'))
    scalar = NamedSharding(mesh_xy, P())
    return jax.jit(
        lambda result, parent: jax.tree.map(
            lambda a: jax.lax.dynamic_slice_in_dim(a, parent, 1, axis=0), result),
        out_shardings=((face, scalar, scalar), scalar, scalar, scalar))


@lru_cache(maxsize=None)
def _hermitian_part_kernel(mesh):
    """Reuse Hermitian projection on current [b,n,n] response faces."""
    return jax.jit(hermitian_part, out_shardings=NamedSharding(mesh, P(None, 'x', 'y')))


@lru_cache(maxsize=None)
def _public_factor_kernel(mesh):
    """Insert the scalar-spin axis without recreating the executable."""
    return jax.jit(lambda value: value[:, :, None, :],
                   out_shardings=NamedSharding(mesh, P(None, 'x', None, 'y')))


@lru_cache(maxsize=None)
def _stack_model_kernel(mesh):
    """Stack the current admitted factor/pole/count batch on named layouts."""
    return jax.jit(
        lambda parts: tuple(jnp.concatenate([row[i] for row in parts], axis=0)
                            for i in range(3)),
        out_shardings=(NamedSharding(mesh, P(None, 'x', None, 'y')),
                       NamedSharding(mesh, P()), NamedSharding(mesh, P())))


def _odd_partner_directions(q, output, cutoff, *, eigh_plan, matmul, column_extent, multiplet_tol):
    """Partner directions of O = W Q orthogonal to Q above the recipe direction cutoff.

    For a role whose conjugate node brings no new tangent under time reversal
    (imaginary z, or a line role with Re z = 0), W Q lies in span(Q) on
    time-reversal-symmetric data and the partner would only duplicate Q. The
    component of O orthogonal to Q is rank-revealed against the largest
    singular value of O with the same relative cutoff used for line directions;
    on a magnet the surviving directions carry the odd channel. Only spectra
    cross to the host. Returns (orthonormal directions [b,n,r], widths) or
    (None, None) when nothing survives.
    """
    hermitian_kernel = _hermitian_part_kernel(eigh_plan.mesh)
    remainder = output - matmul(q, matmul(q, output, transa="C"))
    top = np.asarray(eigh_plan.batched(hermitian_kernel(matmul(output, output, transb="C")))[0])[..., -1]
    perp = hermitian_kernel(matmul(remainder, remainder, transb="C"))
    values = np.asarray(eigh_plan.batched(perp)[0])[..., ::-1]
    counts = [int(np.sum(row > float(cutoff) ** 2 * float(t)))
              for row, t in zip(np.atleast_2d(values), np.atleast_1d(top))]
    if max(counts) == 0:
        return None, None
    directions, kept = distrib_la.leading_eigenvectors(
        perp, max(counts), eigh_plan=eigh_plan, column_extent=column_extent, multiplet_tol=multiplet_tol)
    return directions, tuple(int(v.shape[-1]) for v in kept)


def _direction_states(read_sample, recipe, *, eigh_plan, svd_plan, matmul,
                      column_extent, logical_n, admit, infinity_carrier, ordered=False,
                      read_mirror=None):
    """Select each q row independently from a bounded batch of fitted samples.

    ``read_sample(sample_id)`` returns W/dW [b,n,n] faces. Only spectra cross the host;
    Output states keep their batch axis. Counts and role receipts are small
    host metadata; the packer compacts each parent's original port carriers.

    Ordered data pair every state X(z) on direction Q with its particle-hole
    mirror X(-z) on the SAME Q, the layout ``reduce_ordered_shared_pole_pencil``
    cuts in. The mirror needs W(-conj z) and dW/ds(-conj z): for purely
    imaginary z it is the sample itself; otherwise ``read_mirror(id)``
    (production: the conjugated sample of the parent of -q) or, when None, the
    one fitted recipe id at -conj(z), which then selects no directions of its
    own. Mirrors follow all original states in the same order.
    """
    hermitian_kernel = _hermitian_part_kernel(eigh_plan.mesh)
    fit_roles = _fit_roles(recipe)
    fit_ids = [int(i) for i in recipe["fit_ids"]]
    mirror_source = {}
    if ordered and read_mirror is None:
        points = {i: _sample_point(recipe, i) for i in fit_ids}
        for j in fit_ids:
            # The first fitted id of a pair is the original; its -conj(z) partner is the mirror.
            if points[j].real == 0 or j in mirror_source.values():
                continue
            matches = [k for k in fit_ids if points[k] == -points[j].conjugate()]
            if len(matches) != 1:
                raise ValueError(f"GATE shared_pole_orientation_pair: got: {len(matches)} fitted supports at -conj(z) for sample {j}; want: exactly one mirror sample or read_mirror; why: the ordered cut acts on paired particle-hole states")
            mirror_source[j] = matches[0]
    skip = set(mirror_source.values())
    states, counts, roles, conjugates = [], [], [], []
    mirrors, mirror_counts, mirror_roles = [], [], []
    def largest_side():
        return infinity_carrier + sum(st[1].shape[-1] for st in (*states, *mirrors))
    for sample_id in fit_ids:
        if sample_id in skip:
            continue
        admit(largest_side())
        w, derivative = read_sample(int(sample_id))
        if not roles:
            roles = [[] for _ in range(w.shape[0])]
            mirror_roles = [[] for _ in range(w.shape[0])]
        first = len(states)
        for role in fit_roles:
            if role["held"] or int(role["sample_id"]) != int(sample_id):
                continue
            kind = role["role"].split(":", 1)[0]
            if kind == "line":
                q_batch, values = distrib_la.right_singular_vectors(
                    w, recipe["direction_cutoff"], eigh_plan=svd_plan,
                    column_extent=column_extent,
                    multiplet_tol=recipe["multiplet_relative_tolerance"],
                    max_rank=recipe.get("line_direction_cap"))
            elif kind == "imaginary":
                width = min(logical_n, max(1, int(recipe["imaginary_width"])))
                q_batch, values = distrib_la.leading_eigenvectors(
                    hermitian_kernel(-w), width, eigh_plan=eigh_plan,
                    column_extent=column_extent,
                    multiplet_tol=recipe["multiplet_relative_tolerance"])
            else:
                raise ValueError(f"GATE shared_pole_role: got: {kind}; want: line or imaginary fitted role; why: unknown tangent semantics")
            widths = tuple(int(v.shape[-1]) for v in values)
            if any(width < 1 or width > logical_n for width in widths):
                raise ValueError(f"GATE shared_pole_directions: got: ranks {widths}; want: 1..{logical_n}; why: empty or padded physical direction set")
            s = (_sample_point(recipe, int(sample_id)) if ordered
                 else _sample_point(recipe, int(sample_id)) ** 2)
            # W(s*)=W(s).H: its right singular space is the LEFT space
            # of W(s). Use WQ=U sigma without another selection or transport.
            # Column equilibration removes sigma; this also preserves the
            # conjugate space when the underlying response is real symmetric.
            # Ordered data: nodes are z; every role takes its conjugate partner
            # (Wc(conj z) = Wc(z)^H holds without time reversal) and actions
            # use dW/dz = 2 z dW/ds.
            for conjugate in ((False, True) if (ordered or kind == "line") and s.imag != 0 else (False,)):
                admit(largest_side() + q_batch.shape[-1])
                transa = "C" if conjugate else "N"
                state_widths = widths
                if conjugate and ordered and (kind == "imaginary" or s.real == 0):
                    # Dedupe before the cut: only partner directions orthogonal to Q survive.
                    direction, state_widths = _odd_partner_directions(
                        states[-1][1], states[-1][2], recipe["direction_cutoff"], eigh_plan=eigh_plan,
                        matmul=matmul, column_extent=column_extent,
                        multiplet_tol=recipe["multiplet_relative_tolerance"])
                    if direction is None:
                        continue
                else:
                    direction = states[-1][2] if conjugate else q_batch
                output = matmul(w, direction, transa=transa)
                action = matmul(derivative, direction, transa=transa)
                if ordered:
                    action = action * (2 * (s.conjugate() if conjugate else s))
                states.append((s.conjugate() if conjugate else s, direction, output, action))
                conjugates.append(conjugate)
                counts.append(state_widths)
                for i, width in enumerate(state_widths):
                    roles[i].append({"sample_id": int(sample_id), "role": role["role"],
                                     "conjugate": conjugate, "width": width,
                                     "carrier_width": column_extent(width)})
            del q_batch, direction, output, action
        if ordered and len(states) > first:
            if _sample_point(recipe, int(sample_id)).real == 0:
                w_m, d_m = w, derivative
            elif read_mirror is not None:
                w_m, d_m = read_mirror(int(sample_id))
            else:
                w_m, d_m = read_sample(mirror_source[int(sample_id)])
            # W(-z) = W(-conj z)^H on Q; W(-conj z) on the partner's O; in both
            # cases dW/dz at the mirror node is -2 node dW/ds(-conj z) (adjoint on Q).
            for index in range(first, len(states)):
                node, direction = states[index][0], states[index][1]
                admit(largest_side() + direction.shape[-1])
                transa = "N" if conjugates[index] else "C"
                mirrors.append((-node, direction, matmul(w_m, direction, transa=transa),
                                matmul(d_m, direction, transa=transa) * (-2 * node)))
                mirror_counts.append(counts[index])
                for i in range(len(roles)):
                    mirror_roles[i].append(dict(roles[i][index], mirror=True))
            del w_m, d_m
        del w, derivative
    if ordered:
        states, counts = states + mirrors, counts + mirror_counts
        roles = [row + mirror_row for row, mirror_row in zip(roles, mirror_roles)]
    return states, np.asarray(counts, dtype=np.int64).T, roles


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
