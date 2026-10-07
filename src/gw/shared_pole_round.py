"""One bounded shared-pole round, shared by production and selected endpoints.

Providers supply physical Wc/dW_ds (Ry/Ry^-1), moments M_k=C_(k+1)/2,
and the supported physical Coulomb inverse root. This owner selects, reduces,
and measures the round; it never invents a bank schema or an electronic mesh.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Protocol

import jax
import numpy as np

from common.units import RYD_TO_EV
from common.staged_reshard import face_to_batch_reshard
from gw.shared_pole_directions import (_round_kernels, _sample_point, line_panel_states,
                                       select_round_states, infinity_directions)
from gw.shared_pole_local import (batch_to_face, check_round, face_rows,
    own_extent_receipts, reduce_round, round_tables, recipe_panel_widths,
    recipe_infinity_width, pad_states)
from gw.shared_pole_capacity import round_padding_output_bytes, face_eigh_room
from gw.shared_pole_reduction import ORIENTATION_PAIR_REFUSAL
from gw.shared_pole_recipe import build_construction_row, construction_receipt


@dataclass(frozen=True)
class PoleEndpointGeometry:
    """Endpoint geometry, independent of the full physical electronic k mesh.

    ``kind`` declares the coordinates; it never changes normalization. The
    provider's authenticated ``identity`` separately binds basis, E/f/valid,
    source parent and sample recipe. No fake CentroidBasis/Meta is required.
    """
    kind: str
    logical_size: int
    carrier_size: int
    nspinor: int
    k_grid: tuple[int, int, int]
    physical_k_count: int

    def __post_init__(self):
        if self.kind not in ('packed-centroid-charge', 'gamma-real-charge'):
            raise ValueError('Shared-pole round requires an explicit charge endpoint kind')
        values = (self.logical_size, self.carrier_size, self.nspinor,
                  self.physical_k_count, *self.k_grid)
        if (len(self.k_grid) != 3 or any(isinstance(v, (bool, np.bool_)) or
                not isinstance(v, (int, np.integer)) or int(v) <= 0 for v in values)
                or self.logical_size > self.carrier_size
                or self.nspinor not in (1, 2, 4)
                or int(np.prod(self.k_grid, dtype=object)) != self.physical_k_count):
            raise ValueError('Shared-pole round endpoint and actual electronic k-grid census disagree')


class PoleRoundProvider(Protocol):
    """Bounded source-parent reads; caller authenticates payloads before entry.

    Each method returns the requested rows in slot order, including repeated
    inert tail slots. Selection reads return dense sample-major fields and
    source-selected line panels. All physical matrix faces stay on all P;
    local production reads use the incumbent whole-parent batch layout.
    ``identity`` is the authenticated physical recipe/basis/state token.
    """
    identity: object

    def moments(self, q_ids, *, fields): ...
    def selection(self, q_ids, *, sample_ids, line_span): ...
    def samples(self, q_ids, *, sample_span): ...
    def coulomb_inverse(self, q_span): ...


@dataclass(frozen=True)
class SharedPoleRoundPlan:
    """Resolved policy/geometry of a round; arrays belong to its provider.

    Native plans and capacity admissions are resolved by their existing
    service/ledger owners. ``face_*`` carry the admitted whole-mesh reduction
    row and retained-output bound, not a second memory policy.
    """
    endpoint: PoleEndpointGeometry
    recipe: dict
    gates: dict
    mesh_xy: object
    execution: str
    ordered: bool
    odd_moments: bool
    charge4: bool
    detail: bool
    identity: object
    ledger: object
    upstream: tuple
    conservative_side: int
    selection_faces: int
    column_extent: object
    eigh_plan: object
    dense_fit: tuple
    line_span: tuple
    moment_fields: tuple
    history: dict
    face_reduction: object = None
    face_retained_bound: object = None
    face_carrier: object = None


@dataclass(frozen=True)
class SharedPoleRoundResult:
    """Selected-parent result; this is not a finalized all-q model bank.

    Factor is the canonical all-P face, poles2 are Ry², counts are actual
    whole-tie realized K. Production alone restores the complete parent order
    and calls the canonical store writer after every round has passed.
    """
    factor: object
    poles2: np.ndarray
    counts: np.ndarray
    q_ids: tuple
    receipts: dict
    capacity_entry_end: int


def construct_shared_pole_round(provider: PoleRoundProvider, *, plan: SharedPoleRoundPlan,
                                q_ids, real_rows, budget, retained_factors=(),
                                capacity_entry_start=0, phase=None):
    """Select/reduce/check one bounded round with the production equations.

    The exact gate table is supplied by the current recipe owner. The round
    never repairs a negative metric, zero pole, null Coulomb mode, missing
    held support or unmatched particle/hole orientation. A Γ-only provider
    retains Nk and state identity from its actual electronic mesh.
    """
    from common.collectives import require_full_mesh
    endpoint = plan.endpoint
    mesh_xy = require_full_mesh(plan.mesh_xy, origin='shared_pole_round')
    if (int(mesh_xy.shape['x']) != int(mesh_xy.shape['y']) or
            any(endpoint.carrier_size % int(mesh_xy.shape[a]) for a in ('x', 'y'))):
        raise ValueError('Shared-pole round requires a square full mesh and divisible endpoint carrier')
    raw = np.asarray(q_ids)
    if (raw.ndim != 1 or raw.dtype.kind not in 'iu' or
            isinstance(real_rows, (bool, np.bool_)) or
            not isinstance(real_rows, (int, np.integer)) or
            not 0 < int(real_rows) <= len(raw) or np.any(raw < 0)):
        raise ValueError('Shared-pole round requires typed parent slots and a positive real prefix')
    ids, real = raw.tolist(), int(real_rows)
    if (len(set(ids[:real])) != real or
            any(q != ids[real - 1] for q in ids[real:]) or
            max(ids) >= endpoint.physical_k_count):
        raise ValueError('Shared-pole round parent census or repeated tail slots disagree')
    if provider.identity != plan.identity:
        raise ValueError('Shared-pole round provider physical identity differs from the plan')
    recipe, gates, identity = plan.recipe, plan.gates, plan.identity
    if endpoint.kind == 'gamma-real-charge' and identity.get('recipe_hash') != recipe.get('recipe_hash'):
        raise ValueError('Shared-pole round physical recipe identity differs')
    if plan.execution not in ('local', 'face'):
        raise ValueError('Shared-pole round execution must be resolved by the capacity owner')
    from gw.shared_pole_recipe import table_hash
    if recipe['gate_hash'] != table_hash(gates):
        raise ValueError('Shared-pole round recipe and immutable gate table disagree')
    held_ids = [int(i) for i in recipe['held_ids']]
    if not held_ids:
        raise ValueError('GATE shared_pole_held: round requires actual held value and slope supports')
    held_lo, held_hi = min(held_ids), max(held_ids) + 1
    n, logical_n = endpoint.carrier_size, endpoint.logical_size
    execution, ordered, odd_moments = plan.execution, plan.ordered, plan.odd_moments
    charge4, detail = plan.charge4, plan.detail
    ledger, upstream, history = plan.ledger, plan.upstream, plan.history
    conservative_side, selection_faces = plan.conservative_side, plan.selection_faces
    column_extent, eig = plan.column_extent, plan.eigh_plan
    dense_fit, moment_fields = plan.dense_fit, plan.moment_fields
    line_lo, line_hi = plan.line_span
    face_reduction, face_retained_bound, face_carrier = (plan.face_reduction,
        plan.face_retained_bound, plan.face_carrier)
    keep_budget = recipe.get('pole_budget')
    receipt_entry_start = capacity_entry_start
    phase = phase or (lambda name: nullcontext())
    kernels = _round_kernels(mesh_xy, 'batch' if execution == 'local' else 'face')
    to_face, to_batch = batch_to_face(mesh_xy), face_to_batch_reshard(mesh_xy)
    receipts = {}
    with phase("batch_admission"):
        budget.batch_width = len(ids)
        budget.retained_panels = tuple(retained_factors)
        budget.plan(
            conservative_side, phase="selection",
            sample_batch=len(dense_fit), selection_faces=selection_faces)
        budget.live(())
    with phase("scratch_read"):
        exact = provider.moments(ids, fields=moment_fields)
    with phase("infinity_selection"):
        width = min(logical_n, max(1, int(recipe["infinity_width"])))
        qi, round_infinity_values = infinity_directions(
            kernels, exact["M1"], width, eigh_plan=eig, column_extent=column_extent,
            multiplet_tol=recipe["multiplet_relative_tolerance"],
            real_rows=real if execution == 'local' else None)
        infinity = (qi, *(kernels.apply(exact[name], qi)
                          for name in moment_fields))
        del exact
    with phase("sample_batch_read"):
        budget.live(infinity)
        samples, line = provider.selection(ids, sample_ids=dense_fit,
                                           line_span=(line_lo, line_hi))
        # The store admits this complete bounded scratch batch before
        # allocation. Charge it while directions/actions are selected;
        # release it before admitting the dense pencil.
    with phase("direction_selection"):
        # Synthetic local slots select nothing, as a dense selection's do.
        line_states = {sid: line_panel_states(panels, np.where(np.arange(len(counts)) < real, counts, 0),
                                              recipe, sid=sid, ordered=ordered, mesh_xy=mesh_xy)
                       for sid, (panels, counts) in line.items()}
        round_states, round_counts, round_roles = select_round_states(
            samples, recipe, sample_ids=dense_fit, real=real, mesh_xy=mesh_xy, eigh_plan=eig,
            column_extent=column_extent, logical_n=logical_n, ordered=ordered,
            line_states=line_states)
        del samples, line, line_states, qi
    with phase("reduction_admission"):
        infinity_counts = [int(v.shape[-1]) for v in round_infinity_values]
        budget.retained_panels = tuple(retained_factors)
        budget.batch_width = len(ids)
        # Every state panel is padded to its recipe carrier, so the round
        # program's inputs have one shape; the pencil extent is grow-only
        # over this model's rounds and SC maps (round_tables): discovered
        # in map 0, held from map 1. The table is host metadata, known
        # before any panel is allocated.
        widths = recipe_panel_widths(round_roles[0], round_states, recipe,
                                     column_extent=column_extent, logical_n=logical_n)
        infinity_width = recipe_infinity_width(infinity, recipe, column_extent=column_extent,
                                               logical_n=logical_n)
        tables = round_tables(
            round_counts, widths, [st[0] for st in round_states], infinity_counts, infinity_width,
            column_extent=column_extent, ordered=ordered, odd_moments=odd_moments,
            key=("scalar", logical_n, ordered, odd_moments), history=history)
        side = int(tables["active"].shape[-1])
        # Resolve before either reduction program is traced; the ledger
        # warns when the route price is over the budget.
        local_eigh = budget.eigenplan(side)
        budget.plan(side, phase="reduction", padding_output_bytes_per_rank=round_padding_output_bytes(
            round_states, infinity, widths, infinity_width))
        round_states, infinity = pad_states(round_states, widths, infinity, infinity_width)
    with phase("gram_reduction"):
        if execution == 'face':
            from gw.shared_pole_execution import face_reduce_round
            # The kept span on the budget's carrier, as the local round
            # solves it on its Ritz carrier: the Schur and final eigh run
            # at about 2 x budget instead of the pencil side.
            # The eigh room beside this round's own program: the batch row
            # with the round's price at its actual side for the batch's.
            row = face_reduction
            room = lambda price: face_eigh_room(dict(row, aggregate_bytes_per_rank=(
                row['aggregate_bytes_per_rank'] - budget.program_bytes + price)), face_retained_bound)
            round_model, round_signed, vectors, round_diagnostics = face_reduce_round(
                round_states, infinity, tables, mesh=mesh_xy,
                budget=budget, ordered=ordered, odd_moments=odd_moments,
                keep_budget=keep_budget, admit=False, room=room,
                carrier=face_carrier)
        else:
            round_model, round_signed, vectors, round_diagnostics = reduce_round(
                round_states, infinity, tables, real=real, mesh_xy=mesh_xy,
                native_eigh=local_eigh.native_fn, ordered=ordered,
                odd_moments=odd_moments, keep_budget=recipe.get("pole_budget"))
        qi = infinity[0]
        del round_states, infinity
        round_reduction, round_zero, round_retained, round_permutation = jax.tree.map(np.asarray, round_diagnostics)
        poles, active = (np.asarray(a) for a in vectors)
        budget.retained_panels = tuple(retained_factors)
    with phase("gates"):
        for slot, q in enumerate(ids[:real]):
            if ordered and not round_reduction["orientation_paired"][slot]:
                raise ValueError(ORIENTATION_PAIR_REFUSAL + f" (q={q})")
            for name in ("gram_diagonal_positive", "gram_valid", "retained_metric_positive"):
                if not round_reduction[name][slot]:
                    raise ValueError(
                        f"GATE shared_pole_{name}: got: failed at q={q}, "
                        f"Gram min/max={round_reduction['gram_min_relative'][slot]}, "
                        f"rounding floor/max={round_reduction['gram_floor_relative'][slot]}, "
                        f"paired Schur S min/max={round_reduction['paired_min_relative'][slot] if ordered else 'n/a'}, "
                        f"metric infinity norm={round_reduction['metric_initial_infinity_norm'][slot]}, "
                        f"inverse-root residual={round_reduction['metric_inverse_root_residual_relative'][slot]}; "
                        f"want: Gram min >= -floor (normalized_gram_validity; even route: the larger of it and gram_rounding_validity) "
                        "and valid diagonal/retained metric; why: no PSD repair")
            if not round_zero["zero_policy"][slot]:
                raise ValueError(f"GATE shared_pole_zero_ritz: got: failed at q={q}; want: finite positive response within dropped-weight budget; why: no pole clipping")
            # The ordered identity is on the ORIGINAL infinity directions: exact only for the
            # full Galerkin span, projection accuracy after the keep/retention cuts. It is
            # reported beside the full_m1/full_m3 diagnostic bands, as the TRS route reports
            # its original-direction defects; the retained Ritz algebra is the metric gate above.
            if not ordered and not all(value[slot] <= gates["retained_subspace_moments"]["threshold"]
                                       for value in round_retained.values()):
                raise ValueError(f"GATE shared_pole_retained_moments: got: failed at q={q}; want: projected latent moment identity <=1e-10; why: corrected Ritz algebra")
        reductions = own_extent_receipts(round_reduction, tables["own"][:real])
    with phase("coulomb"):
        budget.batch_width = len(ids)
        model_row = budget.plan(side, phase="model", sample_batch=len(held_ids))
        budget.live((*round_model, *round_signed, qi))
        # V^-1/2 of the round's parents: one owner call per contiguous run of ids, rows in slot order.
        runs = []
        for q in sorted(ids[:real]):
            if runs and q == runs[-1][1]:
                runs[-1][1] += 1
            else:
                runs.append([q, q + 1])
        parts, coulomb = [], {}
        for lo, hi in runs:
            part, receipt = provider.coulomb_inverse((lo, hi))
            parts.append(part)
            # One parent's support rank and the resource hash; the resource
            # itself (path, q order) is the bank's, stated once.
            coulomb.update({q: dict(sha256=receipt["coulomb_identity"].get("sha256"),
                                    support_ranks=receipt["support_ranks"][q - lo:q - lo + 1])
                            for q in range(lo, hi)})
        inverse_sqrt = face_rows(mesh_xy, tuple(sorted(ids[:real]).index(q) for q in ids))(*parts)
        if execution == 'local':
            inverse_sqrt = to_batch(inverse_sqrt)
        del parts
        # Callee I/O admission needs the actual arrays that survive the
        # Coulomb call, not the earlier pre-call live set.
        budget.live((*round_model, *round_signed, qi, inverse_sqrt))
    with phase("sample_batch_read"):
        held = provider.samples(ids, sample_span=(held_lo, held_hi))
        pick = kernels.take(tuple(i - held_lo for i in held_ids))
        held = tuple(pick(held[name]) for name in ("Wc", "dWc_ds"))
        budget.live((*round_model, *round_signed, qi, inverse_sqrt, *held))
    with phase("moment_read"):
        exact = provider.moments(ids, fields=("M1", "M3"))
        budget.live((*round_model, *round_signed, qi, inverse_sqrt, *held,
                     *exact.values()))
    with phase("passivity_held"):
        passive, held_errors, reciprocity, moment_defects = check_round(
            round_model, round_signed, inverse_sqrt, held, (exact["M1"], exact["M3"]), qi,
            real=real, nodes=[_sample_point(recipe, i) for i in held_ids], eta_ry=recipe["eta_ev"] / RYD_TO_EV,
            mesh_xy=mesh_xy, eigh_plan=eig if execution == 'face' else local_eigh, ordered=ordered,
            # A face check decides its eighs beside the model phase and its own program.
            room=lambda price: face_eigh_room(dict(model_row, aggregate_bytes_per_rank=(
                model_row['aggregate_bytes_per_rank'] + price))))
        del inverse_sqrt, held, exact
    with phase("gates"):
        for slot, q in enumerate(ids[:real]):
            if not passive["passivity"][slot]:
                raise ValueError(f"GATE shared_pole_passivity: got: failed at q={q}; want: 0 <= V-whitened -W(i eta) <= I; why: passive screening")
            if not ordered and not np.all(reciprocity["passed"][slot]):
                raise ValueError(f"GATE shared_pole_model_reciprocity: got: { {k: v[slot].tolist() for k, v in reciprocity.items()} } at q={q}; want: model preserves transpose symmetry of symmetric held data; why: conjugate-port closure must survive reduction")
    with phase("receipts"):
        price = budget.plan(side)
        counts = active.sum(axis=-1, dtype=np.int64)
        for slot, q in enumerate(ids[:real]):
            row_of = lambda tree: {key: value[slot:slot + 1] for key, value in tree.items()}
            row, measurements = build_construction_row(
                (None, poles[slot:slot + 1], active[slot:slot + 1]), counts[slot:slot + 1],
                dict(reduction=reductions[slot], zero=row_of(round_zero), passive=row_of(passive),
                     retained=row_of(round_retained),
                     moment_defects={name: row_of(value) for name, value in moment_defects.items()},
                     held=[{"sample_id": sample_id, "Wc": float(held_errors[slot, 0, i]),
                            "dWc_ds": float(held_errors[slot, 1, i])} for i, sample_id in enumerate(held_ids)],
                     # Preserve the receipt's sample-major W,dW rows, each with one parent.
                     reciprocity={key: value[slot].T.reshape(-1, 1).tolist()
                                  for key, value in reciprocity.items()},
                     permutation=round_permutation[slot:slot + 1]),
                span=(q, q + 1), roles=round_roles[slot], price=price, coulomb=coulomb[q],
                native_queries=budget.native_queries, identity=identity,
                gates=gates, nspinor=endpoint.nspinor, logical_n=logical_n,
                ordered=ordered, odd_moments=odd_moments,
                detail=detail)
            receipt = construction_receipt(
                measurements, capacity=ledger,
                capacity_entry_start=receipt_entry_start, ordered=ordered,
                charge4=charge4)
            receipt.update(identity=identity, constructor=row)
            receipts[q] = receipt
        receipt_entry_start = len(ledger.entries)
    with phase("export_prepare"):
        # The sorted model is an active prefix: keep the round's widest K, then restore to the face.
        width = column_extent(int(counts[:real].max()))
        factor = (face_rows(mesh_xy, tuple(range(real)), width)(round_model[0]) if execution == 'face' else
                  face_rows(mesh_xy, tuple(range(real)), width)(to_face(round_model[0])))
        del round_model, round_signed, qi, reductions, vectors
        ledger.live_stages = upstream
        budget.retained_panels = (*retained_factors, factor)
    return SharedPoleRoundResult(factor, poles[:real, :width], counts[:real],
                                 tuple(ids[:real]), receipts, receipt_entry_start)
