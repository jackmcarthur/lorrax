"""Direction selection and the per-parent state panels the pencil consumes.

Line, imaginary and infinity supports (W 14), the ordered partner
directions and their dedupe (W 29), and the model diagnostics evaluated
against the bank moments. A line support's directions depend on its own
sample only, so the producer selects them and stores the states
(``line_sample_states``, ``line_panels``); the constructor reads them back
(``line_panel_states``) and selects the dense on-axis supports itself
(``select_round_states``). Both use the same state equations.
"""

from __future__ import annotations

from functools import lru_cache

import distrib_la
import jax
import jax.numpy as jnp
import numpy as np
from distrib_la import hermitian_part
from jax.sharding import NamedSharding, PartitionSpec as P


def replicated(mesh_xy):
    """Place a small host scalar or table replicated on the mesh (sample indices, node scales)."""
    return lambda value: jax.device_put(value, NamedSharding(mesh_xy, P()))


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


@lru_cache(maxsize=None)
def _round_kernels(mesh, layout="batch"):
    """Selection programs for batch or whole-mesh face stacks of parents.

    Batch arrays carry their parent axis over ('x','y'). Face arrays carry
    their parents on an unsharded leading axis at P(None,'x','y'). Scalars
    are replicated.
    """
    from types import SimpleNamespace
    from common.shard_map import shard_map

    from gw.shared_pole_execution import face_program, face_matmul
    from gw.shared_pole_local import _mm as _local_product
    batch, rep = P(('x', 'y')), P()
    mm = face_matmul(mesh) if layout == 'face' else lambda a,b,**kw: _local_product(a,b,**kw)
    adjoint = lambda a: jnp.conj(jnp.swapaxes(a, -1, -2))

    def program(fn, specs, out):
        if layout == 'face':
            return face_program(fn,mesh)
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
            return mm(a,x), mm(d,x) * scale
        return program(body, (batch, batch, rep, batch, rep), (batch, batch))

    @lru_cache(maxsize=None)
    def minus_q_partner(flags):
        """Act with the stored minus-q partner W_q(-conj z) on this parent's original directions.

        The original state at z needs the adjoint to reach -z. Its conjugate
        state at conj z needs the stored value directly at -conj z. Each
        derivative is with respect to s at -conj z; ``scales`` converts it to
        the derivative with respect to z there.
        """
        def body(w, dw, xs, j, scales):
            a, d = sample(w, j), sample(dw, j)
            outputs = []
            for k, conjugate in enumerate(flags):
                op_a, op_d = ((a, d) if conjugate else (adjoint(a), adjoint(d)))
                outputs.extend((mm(op_a,xs[:,k]), mm(op_d,xs[:,k]) * scales[k]))
            return tuple(outputs)
        return program(body, (batch, batch, batch, rep, rep),
                       (batch,) * (2 * len(flags)))

    def dedupe(q, o):
        # O W-output of the direction set Q: the part of O outside span(Q), and O O^H for its scale.
        rest = o - mm(q, mm(q,o,transa="C"))
        herm = lambda a: (a + adjoint(a)) / 2
        return herm(mm(o,o,transb="C")), herm(mm(rest,rest,transb="C"))

    column = program(lambda q, e: jax.lax.dynamic_index_in_dim(q, e, axis=1, keepdims=False), (batch, rep), batch)

    @lru_cache(maxsize=None)
    def columns(count):
        # Every field of a stored panel stack [.., F, n, r] in one program, not one call per field.
        return program(lambda q: tuple(q[:, i] for i in range(count)), (batch,), (batch,) * count)
    apply = program(lambda m, q: mm(m,q), (batch, batch), batch)
    negative_hermitian = program(lambda a: -(a + adjoint(a)) / 2, (batch,), batch)
    return SimpleNamespace(
        take=take, column=column, columns=columns, negative_hermitian=negative_hermitian,
        minus_q_partner=minus_q_partner, act=act, apply=apply,
        stack=program(lambda *a:jnp.stack(a,axis=1),batch,batch),
        dedupe=program(dedupe, (batch, batch), (batch, batch)))


def port_extent(mesh_xy):
    """Direction carrier of a retained width: the extent ladder, tiling the mesh.

    Direction ranks move between rounds, samples and maps; their carriers sit on
    ``runtime.padding.ladder_extent`` so selection and round programs repeat. No
    rank cap: the same function sizes round sums and pole budgets, which exceed n.
    """
    from runtime.padding import ladder_extent, padded_axis
    return lambda width: padded_axis(ladder_extent(width), mesh_xy, name="shared_pole_port",
                                     specs=((P("x", "y"), 0), (P("x", "y"), 1))).carrier


def line_width_bounds(meta, *, mesh_xy, photon_bases=None):
    """Admission width of each family's line panels: the carrier of its line cap.

    The recipe's cap (ceil(n/16) in production; the family's logical rows when
    the tier sets none) on the extent ladder, the constructor's own bound for a
    line support; a multiplet closed past the cap stores its actual width.
    """
    from gw.shared_pole_sectors import sector_recipe
    extent = port_extent(mesh_xy)
    recipe = meta.shared_pole_recipe
    if photon_bases is None:
        families = {"charge": int(meta.n_rmu)}
    else:
        families = {name: (3 if f else 1) * int(b.n_logical)
                    for f, (name, b) in enumerate(zip(("C", "T"), photon_bases))}
    widths = {}
    for name, n in families.items():
        cap = (recipe if photon_bases is None else sector_recipe(recipe, n)).get("line_direction_cap")
        widths[name] = int(extent(max(1, n if cap is None else min(int(cap), n))))
    return widths


def leading_response_directions(matrix, width, **kwargs):
    """Select within the PSD response's resolved support before Gram equilibration.

    A requested width can exceed the response rank. Exclude eigenvalues below
    the dense eigensystem's n*eps relative resolution, not weak Gram modes or
    negative poles; all downstream physical gates remain unchanged.
    """
    return distrib_la.leading_eigenvectors(
        matrix, width, rcond=matrix.shape[-1] * np.finfo(np.float64).eps, **kwargs)


def _role_entries(recipe):
    """Fitted (sample id, role label) pairs in fit order, labelled as the receipts name them."""
    from gw.shared_pole_recipe import ROLE_CODES
    names = {code: name for name, code in ROLE_CODES.items()}
    entries = [(int(sample), names[int(role)] + f":{i}")
               for i, (sample, role, held) in enumerate(zip(recipe["distinct_id"], recipe["role"], recipe["held"]))
               if not held]
    entries = [(sid, label) for sid in (int(i) for i in recipe["fit_ids"])
               for sample, label in entries if sample == sid]
    if any(label.split(":", 1)[0] not in ("line", "imaginary") for _, label in entries):
        raise ValueError("GATE shared_pole_role: got: a fitted role other than line or imaginary; want: line or imaginary fitted role; why: unknown tangent semantics")
    return entries


def _select(k, W, entries, kind, index, recipe, *, real, eigh_plan, svd_plan, column_extent, logical_n):
    """Directions of every ``kind`` entry, one batched service call over [parent x sample].

    Line supports take the right singular vectors of W above the relative cutoff,
    at most the line cap, closed over the boundary multiplet; imaginary supports
    the leading eigenvectors of -Herm W. Each row's count and multiplet closure
    read that row's own spectrum only. Returns ``(Q [.., e, n, r], spectra)``.
    """
    spectral_rows = None if is_face_stack(W) else real
    tol = recipe["multiplet_relative_tolerance"]
    stack = k.take(tuple(index[sid] for sid, _ in entries))(W)
    if kind == "line":
        return distrib_la.right_singular_vectors(
            stack, recipe["direction_cutoff"], eigh_plan=svd_plan, column_extent=column_extent,
            multiplet_tol=tol, real_rows=spectral_rows, max_rank=recipe.get("line_direction_cap"))
    width = min(logical_n, max(1, int(recipe["imaginary_width"])))
    return leading_response_directions(
        k.negative_hermitian(stack), width, eigh_plan=eigh_plan, column_extent=column_extent,
        multiplet_tol=tol, real_rows=spectral_rows)


def is_face_stack(array):
    from gw.shared_pole_execution import is_face
    return is_face(array)


def _sample_states(k, W, dW, j, z, kind, direction, widths, recipe, *, ordered, real, eigh_plan,
                   column_extent, put):
    """The states one fitted role of one sample contributes, before its mirrors.

    ``direction`` [.., n, r] is the role's selection. The state at the node
    acts with W and dW there; a line support off the real s axis (and every
    ordered support) adds the conjugate state: on O = W Q for Re z != 0, else
    (ordered, Re z = 0) on the part of O outside span(Q) (W 29), per-slot widths,
    0 allowed. Returns ``(states, flags, counts)``: states (node, direction,
    output, action) with the action dW/dz (ordered) or dW/ds (TRS).
    """
    states, flags, counts = [], [], []
    s = z if ordered else z ** 2
    for conjugate in ((False, True) if (ordered or kind == "line") and s.imag != 0 else (False,)):
        state_widths = widths
        if conjugate and ordered and (kind == "imaginary" or s.real == 0):
            direction, state_widths = _round_partner_directions(
                states[-1][1], states[-1][2], recipe["direction_cutoff"], real=real,
                kernels=k, eigh_plan=eigh_plan, column_extent=column_extent,
                tol=recipe["multiplet_relative_tolerance"])
            if direction is None:
                continue
        elif conjugate:
            direction = states[-1][2]
        scale = put(np.complex128(2 * (s.conjugate() if conjugate else s) if ordered else 1.0))
        output, action = k.act(bool(conjugate))(W, dW, j, direction, scale)
        states.append((s.conjugate() if conjugate else s, direction, output, action))
        flags.append(conjugate)
        counts.append(state_widths)
    return states, flags, counts


def _mirror_states(k, states, flags, z, *, W=None, dW=None, j=None, partner=None, put):
    """Ordered states X(-node) on each state's directions (W 25).

    They act with W_q(-conj z) = conj(W_{-q}(z)): the original state at z needs
    its adjoint to reach -z, the conjugate state at conj z needs it directly.
    At Re z = 0, -conj z = z and the sample is its own partner; otherwise
    ``partner`` is the (value, derivative) pair of the minus-q partner W_q(-conj z)
    in the stack layout of W. dW/dz at the mirror node is -2 node dW/ds(-conj z).
    """
    if z.real != 0:
        if partner is None:
            raise ValueError("GATE shared_pole_minus_q_partner: a line support off the imaginary axis "
                             "needs W_q(-conj z) on the ordered route")
        xs = k.stack(*[st[1] for st in states])
        flat = k.minus_q_partner(tuple(flags))(
            partner[0], partner[1], xs, put(np.int32(0)),
            put(np.asarray([-2 * st[0] for st in states], np.complex128)))
        results = [(flat[2 * i], flat[2 * i + 1]) for i in range(len(states))]
        del xs
    else:
        # An imaginary support's optional partner direction can have a
        # different carrier width, so act on each state separately.
        results = [k.act(not flag)(W, dW, j, st[1], put(np.complex128(-2 * st[0])))
                   for st, flag in zip(states, flags)]
    return [(-st[0], st[1], output, action) for st, (output, action) in zip(states, results)]


def _roles(entries, counts, ranks, column_extent, **extra):
    """Per-slot role records of consecutive states (width and carrier per slot)."""
    rows = [[] for _ in range(ranks)]
    for (sid, label, conjugate), widths in zip(entries, counts):
        for r, width in enumerate(widths):
            rows[r].append({"sample_id": sid, "role": label, "conjugate": conjugate,
                            "width": int(width), "carrier_width": column_extent(int(width)), **extra})
    return rows


def line_sample_states(W, dW, recipe, *, sid, ordered, real, mesh_xy, eigh_plan, svd_plan,
                       column_extent, logical_n):
    """Directions and the states at z (and conj z) of one fitted line sample, Re z != 0.

    The producer's half of the constructor's selection (``select_round_states``):
    ``W``/``dW`` hold this one sample's value and s-derivative for a stack of
    parents, ``[B,1,n,n]`` in batch layout (rows ``>= real`` synthetic) or
    ``[B,1,n_X,n_Y]`` on the face. The directions are the right singular vectors
    of W above the cutoff, at most the line cap, whole multiplets, each parent's
    count read from its own spectrum; the states and actions follow
    ``_sample_states``. Returns ``dict(states, flags, counts)`` with counts
    int [B]; ordered mirrors follow from ``line_sample_mirrors`` once the minus-q
    partner exists.
    """
    k = _round_kernels(eigh_plan.mesh, "face" if is_face_stack(W) else "batch")
    entries = [(s, label) for s, label in _role_entries(recipe) if s == sid]
    if not entries or any(not label.startswith("line:") for _, label in entries):
        raise ValueError(f"GATE shared_pole_line_panel: sample {sid} is not a fitted line support")
    z = _sample_point(recipe, sid)
    if z.real == 0:
        raise ValueError(f"GATE shared_pole_line_panel: sample {sid} lies on the imaginary axis; it stays dense")
    put = replicated(mesh_xy)
    q_all, values = _select(k, W, entries[:1], "line", {sid: 0}, recipe, real=real, eigh_plan=eigh_plan,
                            svd_plan=svd_plan, column_extent=column_extent, logical_n=logical_n)
    direction = k.column(q_all, put(np.int32(0)))
    widths = tuple(int(np.asarray(row[0]).size) for row in values)
    if any(w < 1 or w > logical_n for w in widths[:real]):
        raise ValueError(f"GATE shared_pole_directions: got: ranks {widths[:real]}; want: 1..{logical_n}; why: empty or padded physical direction set")
    states, flags, counts = _sample_states(k, W, dW, put(np.int32(0)), z, "line", direction, widths, recipe,
                                           ordered=ordered, real=real, eigh_plan=eigh_plan,
                                           column_extent=column_extent, put=put)
    return dict(states=states, flags=flags, counts=np.asarray(widths, np.int64))


def line_sample_mirrors(line, partner, recipe, *, sid, mesh_xy):
    """Ordered mirror states of one line sample on its stored directions (``_mirror_states``)."""
    k = _round_kernels(mesh_xy, "face" if is_face_stack(partner[0]) else "batch")
    put = replicated(mesh_xy)
    return _mirror_states(k, line["states"], line["flags"], _sample_point(recipe, sid),
                          partner=partner, put=put)


@lru_cache(maxsize=None)
def selection_layout(mesh, execution, nq):
    """Programs between the producer's face stacks and the selection layout.

    ``move`` takes a face stack [nq, m, n] to the selection layout: whole
    parents per rank in batch layout [B, m, n] (B = nq padded to P, the
    padding rows synthetic), or leaves the face; ``axis`` adds the one-sample
    axis [B, 1, m, n]. ``to_face`` takes panel fields [B, m, r] back to the
    face and stacks them [nq, F, m_X, r_Y].
    """
    from gw.shared_pole_local import batch_stack_to_face
    face4 = NamedSharding(mesh, P(None, None, "x", "y"))
    stack = jax.jit(lambda *fields: jnp.stack(fields, axis=1)[:nq], out_shardings=face4)
    axis = jax.jit(lambda a: a[:, None], out_shardings=(
        face4 if execution == "face" else NamedSharding(mesh, P(("x", "y"), None, None, None))))
    if execution == "face":
        return (lambda a: a), axis, stack
    # One exchange for the whole panel set: the fields are stacked in batch
    # layout and moved together, not one batch_to_face per field.
    return ((lambda a: distrib_la.batch_layout(a, mesh)), axis, batch_stack_to_face(mesh, nq))


class LineSelection:
    """The producer's direction selection at the fitted line samples (Re z != 0).

    A line support's directions and every action the pencil reads from it
    depend on that one sample (``line_sample_states``), so the producer
    selects them while W(z), dW/ds and the minus-q partner are in hand and the
    bank stores only the panels. ``families`` lists each endpoint family as
    ``dict(name, index, recipe, logical_n, rows, cross)``; ``block(value, (f, g))``
    returns the (f, g) endpoint block of a face-tiled operator [nq, d, d] as a
    selection stack [B, 1, n_f, n_g], in the order and with the padding the
    constructor reads it. A family with ``cross`` also stores the actions of
    the rectangles (g, f) and (f, g), g the other family, on its directions.
    ``execution`` is 'local' (whole parents per rank, the q-local kernels) or
    'face'.
    """

    def __init__(self, families, block, *, mesh_xy, ordered, execution, nq):
        from gw.shared_pole_capacity import constructor_eigenplan
        self.families, self.block, self.mesh = families, block, mesh_xy
        self.ordered, self.execution, self.nq = bool(ordered), execution, int(nq)
        self.extent = port_extent(mesh_xy)
        self.to_face = selection_layout(mesh_xy, execution, self.nq)[2]
        self.plans = {f["name"]: (constructor_eigenplan(mesh_xy, f["rows"], execution),
                                  constructor_eigenplan(mesh_xy, 2 * f["rows"], execution))
                      for f in families}

    def _rectangles(self, value, slope):
        """The off-diagonal endpoint blocks of one operator pair, cut once for every family.

        Family f's cross actions read (g, f) and (f, g) of the value and the
        slope, g the other family, so the two families share these four blocks.
        """
        cut = {e: (self.block(value, e), self.block(slope, e)) for e in ((1, 0), (0, 1))}
        return {(f, g): (cut[(g, f)][0], cut[(f, g)][0], cut[(g, f)][1], cut[(f, g)][1])
                for f, g in ((0, 1), (1, 0))}

    def _cross(self, family, rectangles, states, *, mirror):
        from gw.shared_pole_sectors import _local_cross_action_program
        from gw.shared_pole_execution import cross_action_program
        f = family["index"]
        rectangles = rectangles[(f, 1 - f)]
        # A mirror state acts with the minus-q partner's rectangles.
        panels = (None,) * 4 + rectangles if mirror else rectangles
        sample = jnp.asarray(0, jnp.int32)
        out = []
        for conjugate, state in zip((False, True), states):
            program = (cross_action_program(self.mesh, mirror, False, conjugate) if self.execution == "face"
                       else _local_cross_action_program(self.mesh, mirror, False, conjugate))
            out.extend(program(panels, state[1], jnp.asarray(state[0]), sample))
        return out

    def select(self, sid, value, slope):
        """Directions and the states at z and conj z of every family, from W(z) and dW/ds."""
        lines = {}
        for family in self.families:
            eig, svd = self.plans[family["name"]]
            f = family["index"]
            W, dW = self.block(value, (f, f)), self.block(slope, (f, f))
            lines[family["name"]] = line_sample_states(
                W, dW, family["recipe"], sid=sid, ordered=self.ordered,
                real=self.nq, mesh_xy=self.mesh, eigh_plan=eig, svd_plan=svd,
                column_extent=self.extent, logical_n=family["logical_n"])
            del W, dW
        # The cross actions follow every diagonal selection, so the shared
        # rectangles are live only while they act.
        if any(family["cross"] for family in self.families):
            rectangles = self._rectangles(value, slope)
            for family in self.families:
                if family["cross"]:
                    line = lines[family["name"]]
                    line["cross"] = self._cross(family, rectangles, line["states"], mirror=False)
            del rectangles
        return lines

    def mirror(self, sid, lines, value, slope):
        """The states at -z and -conj z from the minus-q partner W_q(-conj z) and its dW/ds."""
        for family in self.families:
            line = lines[family["name"]]
            f = family["index"]
            partner = (self.block(value, (f, f)), self.block(slope, (f, f)))
            line["mirrors"] = line_sample_mirrors(line, partner, family["recipe"], sid=sid, mesh_xy=self.mesh)
            del partner
        if any(family["cross"] for family in self.families):
            rectangles = self._rectangles(value, slope)
            for family in self.families:
                if family["cross"]:
                    line = lines[family["name"]]
                    line["cross"] += self._cross(family, rectangles, line["mirrors"], mirror=True)
            del rectangles

    def panels(self, sid, lines):
        """The bank write of one line sample: every family's face panels and counts."""
        out = dict(sample=int(sid), panels={}, counts={}, cross={})
        for family in self.families:
            line = lines[family["name"]]
            states = line["states"] + list(line.get("mirrors", ()))
            out["panels"][family["name"]] = self.to_face(states[0][1], *[a for st in states for a in st[2:]])
            out["counts"][family["name"]] = np.asarray(line["counts"][:self.nq], np.int64)
            if family["cross"]:
                out["cross"][family["name"]] = self.to_face(*line["cross"])
        if not out["cross"]:
            out.pop("cross")
        return out


def charge_line_selection(meta, *, mesh_xy, ordered, execution, nq):
    """The charge bank's one family: the packed centroid operator, its padding zeroed.

    Zeroing the padded centroid rows and columns is the reader's round trip
    pack(unpack(W)) of the stored sample, so selection sees the constructor's bits.
    """
    basis = meta.mu_basis
    mask = _padding_mask(mesh_xy, tuple(bool(v) for v in np.asarray(basis.active_mask)))
    move, axis, _ = selection_layout(mesh_xy, execution, int(nq))
    return LineSelection([dict(name="charge", index=0, recipe=meta.shared_pole_recipe,
                               logical_n=int(meta.n_rmu), rows=int(basis.n_packed), cross=False)],
                         lambda value, _: axis(move(mask(value))),
                         mesh_xy=mesh_xy, ordered=ordered, execution=execution, nq=nq)


@lru_cache(maxsize=None)
def _padding_mask(mesh, active):
    keep = np.asarray(active)
    return jax.jit(lambda a: jnp.where(jnp.asarray(keep)[:, None] & jnp.asarray(keep)[None, :], a, 0),
                   out_shardings=NamedSharding(mesh, P(None, "x", "y")))


def line_panel_states(panels, counts, recipe, *, sid, ordered, mesh_xy):
    """States of one stored line sample, in ``select_round_states`` order.

    ``panels`` [B, 1+2S, n, r] (``line_panels``), ``counts`` int [B]. Returns
    ``(originals, mirrors, widths)``: originals at z (s = z^2 on the TRS route)
    and conj z, the conj z state's direction being the same panel as the z
    state's output O; mirrors at -z and -conj z on the same directions.
    """
    k = _round_kernels(mesh_xy, "face" if is_face_stack(panels) else "batch")
    field = k.columns(int(panels.shape[1]))(panels)
    z = _sample_point(recipe, sid)
    s = z if ordered else z ** 2
    q, o = field[0], field[1]
    originals = [(s, q, o, field[2]), (s.conjugate(), o, field[3], field[4])]
    mirrors = ([(-s, q, field[5], field[6]), (-s.conjugate(), o, field[7], field[8])]
               if ordered else [])
    return originals, mirrors, tuple(int(c) for c in counts)


def select_round_states(samples, recipe, *, sample_ids, real, mesh_xy, eigh_plan, svd_plan,
                        column_extent, logical_n, ordered=False, line_states=None):
    """Directions, outputs and actions of one round of parents, batched per role (SP 3, SP 13).

    ``samples`` holds ``Wc``/``dWc_ds`` of the dense fitted samples ``sample_ids``
    as [P,S,n,n] in local batch layout (rank r owns round slot r), or
    [b,S,n_X,n_Y] in face layout. Local slots ``>= real`` are synthetic.
    ``line_states`` maps every line-panel sample id (Re z != 0) to its stored
    states (``line_panel_states``): the producer selected them from that
    sample alone with the rule below. The dense samples select here: line
    supports on the imaginary axis the right singular vectors (cutoff,
    per-support cap), imaginary supports the leading eigenvectors of -Herm W,
    each role in ONE batched call over the round's [slot x sample] stack; only
    spectra cross the host. States follow the per-sample order of the paired
    layout (``_sample_states``); after all originals the ordered mirror states
    X(-node) on the same directions (``_mirror_states``).

    Returns ``(states, counts, roles)``: panels [P,n,r] in local batch layout
    or [b,n_X,r_Y] in face layout, replicated counts int [P,A], and per-slot
    role records.
    """
    W, dW = samples["Wc"], samples["dWc_ds"]
    k = _round_kernels(eigh_plan.mesh, "face" if is_face_stack(W) else "batch")
    ranks = int(W.shape[0])
    line_states = {} if line_states is None else line_states
    index = {int(sid): i for i, sid in enumerate(sample_ids)}
    entries = _role_entries(recipe)
    if any(sid not in index and sid not in line_states for sid, _ in entries):
        raise ValueError("GATE shared_pole_sample_identity: a fitted sample is neither dense nor a stored line sample")
    put = replicated(mesh_xy)
    kinds = {kind: [(sid, label) for sid, label in entries
                    if sid not in line_states and label.split(":", 1)[0] == kind]
             for kind in ("line", "imaginary")}
    selected = {kind: _select(k, W, kinds[kind], kind, index, recipe, real=real, eigh_plan=eigh_plan,
                              svd_plan=svd_plan, column_extent=column_extent, logical_n=logical_n)
                for kind in kinds if kinds[kind]}
    states, counts, records, mirrors, mirror_counts, mirror_records = [], [], [], [], [], []
    for sid in (int(i) for i in recipe["fit_ids"]):
        labels = [label for s, label in entries if s == sid]
        if sid in line_states:
            originals, mirrored, widths = line_states[sid]
            for label in labels:
                states += originals
                counts += [widths] * len(originals)
                records += [(sid, label, i == 1) for i in range(len(originals))]
                mirrors += mirrored
                mirror_counts += [widths] * len(mirrored)
                mirror_records += [(sid, label, i == 1) for i in range(len(mirrored))]
            continue
        z, j = _sample_point(recipe, sid), put(np.int32(index[sid]))
        sample_states, sample_flags, sample_counts, sample_records = [], [], [], []
        for kind in ("line", "imaginary"):
            for e, (esid, label) in enumerate(kinds[kind]):
                if esid != sid:
                    continue
                q_all, values = selected[kind]
                direction = k.column(q_all, put(np.int32(e)))
                widths = tuple(int(np.asarray(row[e]).size) for row in values)
                if any(w < 1 or w > logical_n for w in widths[:real]):
                    raise ValueError(f"GATE shared_pole_directions: got: ranks {widths[:real]}; want: 1..{logical_n}; why: empty or padded physical direction set")
                new, flags, new_counts = _sample_states(
                    k, W, dW, j, z, kind, direction, widths, recipe, ordered=ordered, real=real,
                    eigh_plan=eigh_plan, column_extent=column_extent, put=put)
                sample_states += new
                sample_flags += flags
                sample_counts += new_counts
                sample_records += [(sid, label, flag) for flag in flags]
        states += sample_states
        counts += sample_counts
        records += sample_records
        if ordered and sample_states:
            mirrors += _mirror_states(k, sample_states, sample_flags, z, W=W, dW=dW, j=j, put=put)
            mirror_counts += sample_counts
            mirror_records += sample_records
    roles = _roles(records, counts, ranks, column_extent)
    if ordered:
        mirror_roles = _roles(mirror_records, mirror_counts, ranks, column_extent, mirror=True)
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
    from gw.shared_pole_execution import is_face
    spectral_rows = None if is_face(q) else real
    top, perp = kernels.dedupe(q, output)
    m = int(perp.shape[-1])
    _, largest = distrib_la.leading_eigenvectors(top, 1, eigh_plan=eigh_plan, column_extent=column_extent,
                                                 multiplet_tol=tol, real_rows=spectral_rows)
    directions, spectrum = distrib_la.leading_eigenvectors(
        perp, m, eigh_plan=eigh_plan, column_extent=column_extent,
        multiplet_tol=tol, real_rows=spectral_rows)
    counts = tuple(int(np.sum(np.asarray(values) > float(cutoff) ** 2 * float(np.asarray(t)[0])))
                   if i < real else 0 for i, (values, t) in enumerate(zip(spectrum, largest)))
    if max(counts) == 0:
        return None, None
    directions, kept = distrib_la.retain_leading_eigenvectors(
        directions, spectrum, counts, mesh=eigh_plan.mesh,
        column_extent=column_extent, multiplet_tol=tol, real_rows=spectral_rows)
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
