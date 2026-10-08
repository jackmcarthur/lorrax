"""Tangential Hermite/Ritz construction of physical shared real-pole W.

The physical convention is W(s) = b (s - Lambda)^-1 b.H, s = z_Ry**2;
S_m = 2 M_(2m+1).  Thus b has units Ry**(3/2), Lambda Ry**2, and
M1/M3 are physical moments in Ry**3/Ry**5, never bare chi coefficients.

Sample matrices are consumed in bounded batches.  Only narrow direction,
output and derivative-action panels survive to the pencil assembly.  Dense
products and eigensolves are supplied by the resolved distrib_la operations;
there is no local vendor or alternative eigensolver in this physics owner.
"""

from __future__ import annotations

from contextlib import contextmanager
import time

import jax
from common import timing
from gw.shared_pole_capacity import ConstructorCapacity
from gw.shared_pole_directions import (_round_kernels, _sample_point, line_panel_states, port_extent,
                                       select_round_states, infinity_directions)
from gw.shared_pole_reduction import ORIENTATION_PAIR_REFUSAL


def _cap_replay_admit(row, enabled, phase):
    """A cap replay must stop on hardware refusal before its next allocation.

    Scaling WARNs within the device budget remain admissible. The default
    constructor's policy is unchanged; this door authorizes only a guarded
    cap contrast against an immutable bank.
    """
    if enabled and row.get('device_budget_status') != 'PASS':
        raise MemoryError(f'GATE shared_pole_cap_replay_capacity: {phase} exceeds '
                          f'current device capacity before allocation: {row}')
    return row


def constructor_route(meta, config, recipe, *, mesh_xy, ledger, upstream, ordered,
                      odd_moments, nq):
    """Resolve local or whole-mesh parent execution for one map's bank.

    The single admission the constructor applies before its first bank read,
    also consulted by the map owner before the bank exists (bank residence).
    Returns ``(execution, receipt, column_extent, selection_faces,
    moment_fields)``; ``upstream`` names the accepted live reservations.
    The selection holds the dense fitted samples (W, dW/ds), the moments and
    every line sample's stored panels (``selection_face_count``).
    """
    from gw.gw_config import linalg_resolution
    from gw.shared_pole_execution import constructor_execution, line_panel_count, selection_face_count
    from file_io.shared_pole_store import line_panel_geometry

    column_extent = port_extent(mesh_xy)
    moment_fields = ("M0", "M1", "M2", "M3") if odd_moments else ("M1", "M3")
    families, states = line_panel_geometry(meta, ordered=ordered)
    faces = selection_face_count(recipe, n=int(meta.n_rmu_padded), logical_n=int(meta.n_rmu),
        states=states, rows=int(meta.n_rmu_padded), dense_fields=2, moment_fields=len(moment_fields),
        column_extent=column_extent)
    execution, receipt = constructor_execution(
        meta, linalg_resolution({"linalg": config.backend.linalg}), recipe,
        mesh=mesh_xy, ledger=ledger, upstream=upstream, ordered=ordered,
        odd_moments=odd_moments, selection_faces=faces,
        sample_batch=len(recipe["fit_ids"]) - line_panel_count(recipe), parent_count=int(nq),
        column_extent=column_extent,
        ritz_budget=recipe.get("pole_budget") if ordered else None)
    return execution, receipt, column_extent, faces, moment_fields


def construct_shared_poles(bank, moments, meta, config, *, mesh_xy, output, residence=None,
                           cap_only_replay=False, model_identity=None):
    """Construct and write a current-state, bounded-batch real-pole model.

    Parameters
    ----------
    bank : mapping
        Authenticated resource descriptor: ``path``, ``identity``, ``tables``,
        ``coulomb`` (response-owner authenticated Coulomb resource). Dense
        workspace is queried from the resolved service plans. A producer
        certificate is carried as ``rule_receipt``. No dense bank is passed
        as a jit argument. The recipe is read only from meta.
    moments : mapping
        ``path`` to committed physical M1/M3 in the same scratch schema.
    meta : Meta
        Current packed centroid basis, scalar representation and the once-
        resolved ``shared_pole_recipe`` with current state identities.
    config : LorraxConfig
        Cached dense policy is read once, before any per-q plans.
    mesh_xy : Mesh
        Supplied named x/y mesh, never reconstructed by this driver.
    output : path-like
        New immutable compact model file, written by the store owner.
    residence : callable, optional
        The map owner's model residence rule, ``residence(nq, width)`` ->
        ``(ResidentSectorModel or None, receipt)``, asked once the model's
        column width is known and before it is written. A resident target
        replaces ``output`` and its stage stays reserved for the caller; the
        receipt is returned as ``model_residence``.
    cap_only_replay : bool, optional
        Explicit production scalar one-shot cap-only replay. Authenticate
        unchanged bank state, supports and geometry before using a different
        source-resolved cap. The bank and its headers remain immutable.
    model_identity : mapping, optional
        Current source-resolved identity, required only for cap-only replay;
        the bank descriptor retains its original identity for every read.

    Returns
    -------
    dict
        Versioned construction receipt with all gate rows, per-q diagnostics,
        capacity prices and the store header/digest. Structural failures raise
        before the affected q is written; a partial file is never finalized.
    """
    walls = {}

    @contextmanager
    def phase(name):
        """One named constructor stage; its wall adds to ``walls[name]`` (the per-map summary)."""
        started = time.monotonic()
        with timing.section("spole." + name):
            yield
        walls[name] = walls.get(name, 0.0) + time.monotonic() - started

    with phase("entry"):
        import numpy as np
        from jax.sharding import NamedSharding, PartitionSpec as P
        from file_io.shared_pole_store import (
            charge_representation, validate_shared_pole_bank, open_shared_pole_bank,
            read_line_panels, read_shared_pole_bank, write_shared_pole_model,
        )
        from gw.gw_config import linalg_resolution
        from common.staged_reshard import face_to_batch_reshard
        from gw.shared_pole_local import (batch_to_face, canonical_factors, check_round, face_rows,
                                          own_extent_receipts, reduce_round, round_tables,
                                          recipe_panel_widths, recipe_infinity_width, pad_states,
                                          carrier_history)
        from gw.shared_pole_capacity import round_padding_output_bytes
        from gw.shared_pole_recipe import (
            build_construction_row, charge4_gates, construction_receipt,
            shared_real_pole_gates_v1_r3b as gates,
            shared_real_pole_gates_ordered_v1,
        )
        from common.units import RYD_TO_EV
        from gw.w_isdf import response_coulomb_powers

    with phase("setup"):
        recipe = meta.shared_pole_recipe
        # Time-reversal-broken scalar states take the ordered particle-hole route.
        ordered = not bool(bank["tables"]["sym"].trs_allowed)
        # Every charge carrier contracts to one mu x mu CC operator.
        if not charge_representation(meta):
            raise ValueError("GATE shared_pole_representation: expected an authenticated charge carrier")
        charge4 = recipe.get("charge_operator") == "four-component-spin-traced-v1"
        if (int(meta.nspinor) == 4) != charge4:
            raise ValueError("GATE shared_pole_representation: charge4 recipe and carrier disagree")
        if charge4:
            gates = charge4_gates(ordered)
        elif ordered:
            gates = shared_real_pole_gates_ordered_v1
        from gw.shared_pole_recipe import table_hash
        if recipe["gate_hash"] != table_hash(gates):
            raise ValueError("GATE shared_pole_representation: recipe gate table differs from charge carrier")
        resolution = linalg_resolution({"linalg": config.backend.linalg})
        identity = bank["identity"]
        if type(cap_only_replay) is not bool or (not cap_only_replay and model_identity is not None):
            raise ValueError('GATE shared_pole_cap_replay: explicit bool and separate model identity required')
        cap_replay = None
        ledger = meta.shared_pole_capacity
        upstream = ledger.live_stages
        n = int(meta.n_rmu_padded)
        # Every device byte this construction admits is priced here; the driver
        # below keeps only `budget.retained_panels` and `budget.batch_width`
        # current as it moves from selection to reduction to the model checks.
        header = validate_shared_pole_bank(bank["path"], expected_identity=identity,
                                           mesh_xy=mesh_xy, require_complete=True)
        moment_header = validate_shared_pole_bank(moments["path"], expected_identity=identity,
                                                  mesh_xy=mesh_xy, require_complete=cap_only_replay)
        # The stored plan must bind the current physical points and role census.
        stored_recipe = header["recipe"]
        if cap_only_replay:
            from gw.shared_pole_recipe import authenticate_cap_only_replay
            from file_io.shared_pole_store import authenticate_bank_geometry, _json
            if ordered or charge4 or int(meta.nspinor) != 1 or int(meta.nspinor_wfnfile) != 1:
                raise ValueError('GATE shared_pole_cap_replay: only one-component scalar TRS input supported')
            cap_replay = authenticate_cap_only_replay(stored_recipe, recipe, identity, model_identity)
            if (_json(moment_header['recipe']) != _json(stored_recipe)
                    or moment_header['bank_plan_digest'] != header['bank_plan_digest']):
                raise ValueError('GATE shared_pole_cap_replay: bank and moments sampled plans differ')
            for preserved in (header, moment_header):
                authenticate_bank_geometry(preserved, meta=meta, tables=bank['tables'])
                if not preserved.get('final_commit'):
                    raise ValueError('GATE shared_pole_cap_replay: missing preserved final commit')
            cap_replay.update(bank_final_commit=header['final_commit'],
                              moments_final_commit=moment_header['final_commit'],
                              bank_plan_digest=header['bank_plan_digest'],
                              bank_header_recipe_sha256=header['recipe_hash'])
            identity = dict(model_identity)
        for name in ("recipe_hash", "gate_hash", "fit_ids", "held_ids", "role", "role_codes",
                     "distinct_id", "held", "support_pair", "census"):
            if cap_only_replay and name == 'recipe_hash':
                continue  # canonical cap hashes and every other field authenticated above
            current = recipe[name]
            previous = stored_recipe[name]
            if isinstance(current, np.ndarray):
                equal = np.array_equal(current, previous)
            else:
                equal = current == previous
            if not equal:
                raise ValueError(f"GATE shared_pole_bank_state: got: stale {name}; want: current resolved recipe; why: no frozen SC inputs")
        for sample_id in (*recipe["fit_ids"], *recipe["held_ids"]):
            if _sample_point(recipe, sample_id) != _sample_point(stored_recipe, sample_id):
                raise ValueError("GATE shared_pole_bank_state: got: changed z; want: current physical sample point; why: recipe hashes alone do not bind resolved points")
        if bool(header.get("ordered", False)) != ordered:
            raise ValueError(f"GATE shared_pole_representation: got: bank ordered={bool(header.get('ordered', False))} with trs_allowed={not ordered}; want: an ordered bank exactly when time reversal is broken (both orientations, -q as its own parent); why: particle-hole pairing across parents")
        # Odd z-moments M0/M2 certify the ordered infinity block; a finite-state
        # ordered bank builds without it and records the moments NOT_MEASURED.
        odd_moments = ordered and bool(header.get("odd_moments", False))
        logical_n = int(meta.n_rmu)
        from gw.shared_pole_execution import constructor_side_upper_bound
        execution, execution_receipt, column_extent, selection_faces, moment_fields = constructor_route(
            meta, config, recipe, mesh_xy=mesh_xy, ledger=ledger, upstream=upstream,
            ordered=ordered, odd_moments=odd_moments,
            nq=int(header['bank_shape']['nq']))
        # Line supports off the imaginary axis arrive as their stored states;
        # the dense fitted samples (imaginary axis) are selected per round.
        line_lo, line_hi = (int(v) for v in header["line_panels"]["sample_span"])
        dense_fit = [int(i) for i in recipe["fit_ids"] if not line_lo <= int(i) < line_hi]
        budget = ConstructorCapacity(meta, resolution, mesh_xy=mesh_xy,
                                     ledger=ledger, upstream=upstream,
                                     execution=execution)
        budget.ritz_budget = recipe.get("pole_budget") if ordered and execution == 'local' else None
        # A face constructor keeps the Coulomb eigensolve on the same complete
        # mesh even when the deck's default dense policy is local.
        coulomb_config = ({'linalg': 'distributed'}
                          if execution == 'face' else config)
        conservative_side = constructor_side_upper_bound(
            recipe, ordered=ordered, odd_moments=odd_moments,
            logical_n=logical_n, column_extent=column_extent)
        eig = budget.eigenplan(n)
        receipts, factors, store_poles, store_counts, placed = {}, [], [], [], []
        receipt_entry_start = 0
        held_ids = [int(i) for i in recipe["held_ids"]]
        if not held_ids:
            raise ValueError("GATE shared_pole_held: got: no held samples; want: at least one held support; why: the model checks compare W and dW/ds there")
        held_lo, held_hi = min(held_ids), max(held_ids) + 1
        nq = int(header["bank_shape"]["nq"])
        # A local round runs one parent per rank from its sample read to its
        # sorted model. A face round runs a budget-sized batch of parents over
        # all ranks. Both take fixed-width rounds from the one schedule
        # (``parent_rounds``); a short last round repeats its last parent.
        face_batch = 1
        if execution == 'face':
            from gw.shared_pole_capacity import face_eigh_room
            from gw.shared_pole_execution import face_batch_width, face_reduction_bytes, face_ritz_carrier
            keep_budget = recipe.get("pole_budget")
            sizing = dict(rows=n, side=conservative_side,
                          carrier=face_ritz_carrier(mesh_xy, keep_budget) if ordered else None)
            face_batch, execution_receipt['face_batch'] = face_batch_width(
                meta, resolution, mesh=mesh_xy, ledger=ledger, upstream=upstream,
                side=conservative_side, nq=nq,
                selection=dict(sample_batch=len(dense_fit), selection_faces=selection_faces),
                program_bytes=lambda width: face_reduction_bytes(mesh_xy, width, **sizing))
            # Every face round's reduction is priced at this size.
            budget.program_bytes = execution_receipt['face_batch']['program_bytes_per_rank']
            # distrib_la decides each eigh stack against the room beside the
            # admitted batch and the retained factors: whole matrices per rank
            # where they fit, else the whole mesh.
            retained_bound = execution_receipt['retained_output_upper_bound_bytes_per_rank']
            budget.face_room = face_eigh_room(execution_receipt['face_batch']['selection'], retained_bound)
            execution_receipt['face_eigh_room_bytes_per_rank'] = dict(selection=budget.face_room)
            eig = budget.eigenplan(n)
        from gw.shared_pole_local import parent_rounds
        # One fixed width per route; a short last round repeats its last real parent.
        trim_parents = set()
        if (execution == 'face' and not ordered and not charge4 and int(meta.nspinor) == 1
                and int(meta.nspinor_wfnfile) == 1
                and bool(bank['tables']['sym'].trs_allowed) and recipe.get('pole_budget') is not None):
            from symmetry_maps import bgw_integer_q_to_fractional, self_negative_q_mask
            qfrac = np.asarray(bgw_integer_q_to_fractional(bank['tables']['sym'].q_irr_kgrid_int,
                (int(meta.nkx), int(meta.nky), int(meta.nkz))))
            if qfrac.shape != (nq, 3):
                raise ValueError('GATE shared_pole_real_trim: typed actual q coordinates differ from bank parents')
            full_rows = np.asarray(bank['tables']['q_irr_full_idx'])
            if full_rows.shape != (nq,):
                raise ValueError('GATE shared_pole_real_trim: actual canonical parent rows differ from bank')
            trim_parents = set(np.flatnonzero(self_negative_q_mask(full_rows,
                kgrid=(int(meta.nkx),int(meta.nky),int(meta.nkz)))).tolist())
        # In this source's full-Bloch charge chart ordinary conjugation acts
        # within each self-negative parent (response_bank reciprocity owner).
        # Generic q belongs to its minus-q partner and remains unchanged.
        width = mesh_xy.size if execution == 'local' else face_batch
        rounds = (parent_rounds(sorted(trim_parents), 1)
                  + parent_rounds([q for q in range(nq) if q not in trim_parents], width)
                  if trim_parents else parent_rounds(nq, width))
        execution_receipt['real_trim_parents'] = sorted(trim_parents)
        batch_spec = P(("x", "y"))
        read_spec = batch_spec if execution == 'local' else None
        kernels = _round_kernels(mesh_xy, 'batch' if execution == 'local' else 'face')
        to_face, to_batch = batch_to_face(mesh_xy), face_to_batch_reshard(mesh_xy)
    for ids, real, slots in rounds:
        with phase("batch_admission"):
            budget.batch_width = len(ids)
            budget.retained_panels = tuple(factors)
            _cap_replay_admit(budget.plan(
                conservative_side, phase="selection",
                sample_batch=len(dense_fit), selection_faces=selection_faces),
                cap_only_replay, 'selection')
            budget.live(())
        with phase("scratch_read"):
            with open_shared_pole_bank(moments["path"], mesh_xy=mesh_xy) as moment_io:
                exact = read_shared_pole_bank(moment_io, meta=meta, header=moment_header,
                                              q_ids=ids, partition_spec=read_spec,
                                              fields=moment_fields)
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
            with open_shared_pole_bank(bank["path"], mesh_xy=mesh_xy) as bank_io:
                samples = read_shared_pole_bank(
                    bank_io, meta=meta, header=header, q_ids=ids,
                    partition_spec=read_spec, sample_ids=dense_fit, fields=("Wc", "dWc_ds"))
                line = {sid: read_line_panels(bank_io, meta=meta, header=header, family="charge",
                                              sample=sid, q_ids=ids, partition_spec=read_spec)
                        for sid in range(line_lo, line_hi)}
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
            budget.retained_panels = tuple(factors)
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
                key=("scalar", logical_n, ordered, odd_moments), history=carrier_history(meta))
            side = int(tables["active"].shape[-1])
            real_trim = real == 1 and ids[0] in trim_parents
            if execution == 'face':
                # The selected side and this round's actual parent count
                # are now known, so price their public executable geometry.
                budget.program_bytes = face_reduction_bytes(mesh_xy, len(ids), rows=n, side=side,
                    carrier=sizing['carrier'])
            # Resolve before either reduction program is traced; the ledger
            # warns when the route price is over the budget.
            local_eigh = budget.eigenplan(side)
            reduction_row = budget.plan(side, phase="reduction", padding_output_bytes_per_rank=round_padding_output_bytes(
                round_states, infinity, widths, infinity_width))
            _cap_replay_admit(reduction_row, cap_only_replay, 'actual pencil plus padding')
            if real_trim:
                import distrib_la
                from gw.shared_pole_execution import face_eigh
                from gw.shared_pole_capacity import real_trim_projection_bytes
                if reduction_row['device_budget_status'] != 'PASS':
                    raise MemoryError('First ordinary TRIM reduction exceeds explicit full-mesh device capacity')
                span = min(int(recipe['pole_budget']), side)
                aug = 2 * span
                real_plan = face_eigh(mesh_xy, aug, room=0)
                workspace = distrib_la.workspace_bytes_per_rank(real_plan, 'eigh', ((aug, aug),), np.float64)
                # These are successive stages of the same source program.
                # The new stage retains the original matrices/Y/actions;
                # first-reducer workspace is dead before mixed K/L assembly.
                trim_terms = real_trim_projection_bytes(mesh_xy=mesh_xy, parents=len(ids), rows=n,
                    side=side, keep_budget=recipe['pole_budget'], retained_panels=budget.retained_panels)
                trim_price = ledger.reserve(f'constructor.trim-real-projection.q{ids[0]}',
                    resident_bytes_per_rank=trim_terms['resident_bytes_per_rank'],
                    workspace_bytes_per_rank=int(workspace), concurrent_with=upstream)
                trim_price['staged_terms'] = trim_terms
                if trim_price['device_budget_status'] != 'PASS':
                    raise MemoryError('TRIM real projection exceeds explicit full-mesh device capacity')
            round_states, infinity = pad_states(round_states, widths, infinity, infinity_width)
        with phase("gram_reduction"):
            if execution == 'face':
                from gw.shared_pole_execution import face_reduce_round
                # The kept span on the budget's carrier, as the local round
                # solves it on its Ritz carrier: the Schur and final eigh run
                # at about 2 x budget instead of the pencil side.
                # The eigh room beside this round's own program: the batch row
                # with the round's price at its actual side for the batch's.
                row = execution_receipt['face_batch']['reduction']
                room = lambda price: face_eigh_room(dict(row, aggregate_bytes_per_rank=(
                    row['aggregate_bytes_per_rank'] - budget.program_bytes + price)), retained_bound)
                round_model, round_signed, vectors, round_diagnostics = face_reduce_round(
                    round_states, infinity, tables, mesh=mesh_xy,
                    budget=budget, ordered=ordered, odd_moments=odd_moments,
                    keep_budget=keep_budget, admit=False, room=room,
                    carrier=sizing['carrier'], real_trim=real_trim)
            else:
                round_model, round_signed, vectors, round_diagnostics = reduce_round(
                    round_states, infinity, tables, real=real, mesh_xy=mesh_xy,
                    native_eigh=local_eigh.native_fn, ordered=ordered,
                    odd_moments=odd_moments, keep_budget=recipe.get("pole_budget"))
            qi = infinity[0]
            del round_states, infinity
            host_diagnostics = jax.tree.map(np.asarray, round_diagnostics)
            real_method = None
            if real_trim:
                round_reduction, round_zero, round_retained, real_method, round_permutation = host_diagnostics
            else:
                round_reduction, round_zero, round_retained, round_permutation = host_diagnostics
            poles, active = (np.asarray(a) for a in vectors)
            budget.retained_panels = tuple(factors)
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
                if real_method is not None:
                    for name in ("gram_diagonal_positive", "gram_valid", "retained_metric_positive"):
                        if not real_method['reduction'][name][slot]:
                            raise ValueError(f"GATE shared_pole_real_trim_{name}: failed at q={q}; unchanged Gram/metric gate")
                    if not real_method['zero']['zero_policy'][slot]:
                        raise ValueError(f"GATE shared_pole_real_trim_zero: failed at q={q}; unchanged dropped-weight budget")
                    if not all(value[slot] <= gates['retained_subspace_moments']['threshold']
                               for value in real_method['first_retained'].values()):
                        raise ValueError(f"GATE shared_pole_first_retained_moments: failed at q={q}; first compression unchanged")
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
            _cap_replay_admit(model_row, cap_only_replay, 'model/held/passivity')
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
                coulomb_sqrt, part, receipt = response_coulomb_powers(
                    meta, coulomb_config, mesh_xy=mesh_xy, bank_io=bank,
                    q_span=(lo, hi))
                del coulomb_sqrt
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
            with open_shared_pole_bank(bank["path"], mesh_xy=mesh_xy) as bank_io:
                held = read_shared_pole_bank(bank_io, meta=meta, header=header, q_ids=ids, partition_spec=read_spec,
                                             sample_span=(held_lo, held_hi), fields=("Wc", "dWc_ds"))
            pick = kernels.take(tuple(i - held_lo for i in held_ids))
            held = tuple(pick(held[name]) for name in ("Wc", "dWc_ds"))
            budget.live((*round_model, *round_signed, qi, inverse_sqrt, *held))
        with phase("moment_read"):
            with open_shared_pole_bank(moments["path"], mesh_xy=mesh_xy) as moment_io:
                exact = read_shared_pole_bank(moment_io, meta=meta, header=moment_header, q_ids=ids,
                                              partition_spec=read_spec, fields=("M1", "M3"))
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
                    gates=gates, nspinor=int(meta.nspinor), logical_n=logical_n,
                    ordered=ordered, odd_moments=odd_moments,
                    detail=bool(config.debug.sigma_freq_debug_output))
                receipt = construction_receipt(
                    measurements, capacity=ledger,
                    capacity_entry_start=receipt_entry_start, ordered=ordered,
                    charge4=charge4)
                receipt.update(identity=identity, constructor=row)
                if real_method is not None:
                    receipt['real_trim_projection'] = dict(
                        method='phase-balanced real Galerkin input columns', original_pencil_side=side,
                        augmented_side=2*min(int(recipe['pole_budget']), side), keep_budget=int(recipe['pole_budget']),
                        q_parent=int(q), q_full=int(bank['tables']['q_irr_full_idx'][q]),
                        q_fractional=qfrac[q].tolist(),
                        eligibility=dict(route='face',nspinor=int(meta.nspinor),nspinor_wfnfile=int(meta.nspinor_wfnfile),
                            trs_allowed=bool(bank['tables']['sym'].trs_allowed),ordered=ordered,charge4=charge4,
                            predicate='public self_negative_q_mask on authenticated full-parent rows',
                            chart='source full-Bloch charge; G maps to -G-2q at TRIM'),
                        coordinate_scope='new real input columns; no original-pencil Y returned',
                        moment_scope='fresh original Qi anchor projection into new final Ritz span; first compression preserved separately',
                        first_retained={k:v[slot:slot+1].tolist() for k,v in real_method['first_retained'].items()},
                        reduction={k:v[slot:slot+1].tolist() for k,v in real_method['reduction'].items()},
                        zero={k:v[slot:slot+1].tolist() for k,v in real_method['zero'].items()},
                        fresh_retained={k:v[slot:slot+1].tolist() for k,v in real_method['fresh_retained'].items()},
                        capacity=trim_price)
                receipts[q] = receipt
            receipt_entry_start = len(ledger.entries)
        with phase("export_prepare"):
            # The sorted model is an active prefix: keep the round's widest K, then restore to the face.
            width = column_extent(int(counts[:real].max()))
            factors.append(face_rows(mesh_xy, tuple(range(real)), width)(round_model[0]) if execution == 'face' else
                           face_rows(mesh_xy, tuple(range(real)), width)(to_face(round_model[0])))
            store_poles.append(poles[:real, :width])
            store_counts.append(counts[:real])
            placed += list(ids[:real])
            del round_model, round_signed, qi, reductions, vectors
            ledger.live_stages = upstream
            budget.retained_panels = tuple(factors)
    with phase("writer_stack"):
        if sorted(placed) != list(range(nq)):
            raise ValueError(f"GATE shared_pole_rounds: got: parents {sorted(placed)}; want: each of {nq} parents once; why: one canonical store")
        order = np.argsort(placed)
        width = max(block.shape[-1] for block in factors)
        if cap_only_replay:
            budget.batch_width = nq
            budget.retained_panels = tuple(factors)
            _cap_replay_admit(budget.plan(width, phase='model'), True, 'writer stack')
        public_b = canonical_factors(mesh_xy, tuple(int(i) for i in order))(*factors)
        store_poles = np.concatenate([np.pad(block, ((0, 0), (0, width - block.shape[-1])), constant_values=1.0)
                                      for block in store_poles])[order]
        budget.batch_width = nq
        budget.retained_panels = ()
        budget.plan(width, phase="model")
        budget.live((public_b,))
        del factors
    with phase("writer"):
        target, model_residence = output, dict(residence="file")
        if residence is not None:
            resident, model_residence = residence(nq, width)
            if resident is not None:
                target = resident
                ledger.live_stages = (*ledger.live_stages, model_residence["stage"])
        store_header = write_shared_pole_model(
            target, public_b, jax.device_put(store_poles, NamedSharding(mesh_xy, P())),
            np.concatenate(store_counts)[order], q_span=(0, nq), meta=meta, tables=bank["tables"], recipe=recipe,
            receipts={"identity": identity, "q_receipts": [receipts[q] for q in range(nq)],
                      **({"cap_only_replay": cap_replay} if cap_replay is not None else {})}, ordered=ordered)
        ledger.live_stages = upstream
        del public_b
    with phase("return"):
        # Stage walls summed over the rounds (host clock; a stage that only
        # dispatches lends its device time to the next stage that reads back).
        seconds = {name: round(value, 3) for name, value in walls.items()}
        seconds["rounds"] = len(rounds)
        return {"q_receipts": [receipts[q] for q in range(nq)], "model_header": store_header,
                "model": target if model_residence.get("stage") else str(output),
                "model_residence": model_residence,
                "capacity": ledger.receipt(),
                "identity": identity, "status": "CONSTRUCTED", "seconds": seconds,
                "execution": dict(mode=execution, **execution_receipt),
                **({"cap_only_replay": cap_replay} if cap_replay is not None else {})}
