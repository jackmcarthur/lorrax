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

import jax
from common import timing
from gw.shared_pole_capacity import ConstructorCapacity
from gw.shared_pole_directions import _round_kernels, _sample_point, select_round_states, leading_response_directions
from gw.shared_pole_reduction import ORIENTATION_PAIR_REFUSAL


def construct_shared_poles(bank, moments, meta, config, *, mesh_xy, output):
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

    Returns
    -------
    dict
        Versioned construction receipt with all gate rows, per-q diagnostics,
        capacity prices and the store header/digest. Structural failures raise
        before the affected q is written; a partial file is never finalized.
    """
    with timing.fenced_section("spole.entry"):
        import numpy as np
        from jax.sharding import NamedSharding, PartitionSpec as P
        import distrib_la
        from runtime.padding import mesh_divisor, padded_axis
        from file_io.slab_io import SlabIO
        from file_io.shared_pole_store import (
            charge_representation, validate_shared_pole_bank,
            read_shared_pole_bank, write_shared_pole_model,
        )
        from gw.gw_config import linalg_resolution
        from common.staged_reshard import face_to_batch_reshard
        from gw.shared_pole_local import (batch_to_face, canonical_factors, check_round, face_rows,
                                          own_extent_receipts, parent_rounds, partner_realization,
                                          reduce_round, round_tables)
        from gw.shared_pole_recipe import (
            build_construction_row, construction_receipt,
            shared_real_pole_gates_v1_r3b as gates,
            shared_real_pole_gates_ordered_v1,
        )
        from common.units import RYD_TO_EV
        from gw.w_isdf import response_coulomb_powers

    with timing.fenced_section("spole.setup"):
        recipe = meta.shared_pole_recipe
        # Time-reversal-broken scalar states take the ordered particle-hole route.
        ordered = not bool(bank["tables"]["sym"].trs_allowed)
        # Scalar and two-component decks share one mu x mu charge operator.
        if not charge_representation(meta):
            raise ValueError("GATE shared_pole_representation: got: bispinor or unsupported state; want: scalar or two-component charge operator (TRS-broken states take the ordered route); why: both-endpoint spin action is not yet supported")
        if ordered:
            gates = shared_real_pole_gates_ordered_v1
        resolution = linalg_resolution({"linalg": config.backend.linalg})
        identity = bank["identity"]
        ledger = meta.shared_pole_capacity
        upstream = ledger.live_stages
        n = int(meta.n_rmu_padded)
        # Every device byte this construction admits is priced here; the driver
        # below keeps only `budget.retained_panels` and `budget.batch_width`
        # current as it moves from selection to reduction to the model checks.
        budget = ConstructorCapacity(meta, resolution, mesh_xy=mesh_xy,
                                     ledger=ledger, upstream=upstream)
        header = validate_shared_pole_bank(bank["path"], expected_identity=identity,
                                           mesh_xy=mesh_xy, require_complete=True)
        moment_header = validate_shared_pole_bank(moments["path"], expected_identity=identity,
                                                  mesh_xy=mesh_xy)
        # The stored plan must bind the current physical points and role census.
        stored_recipe = header["recipe"]
        for name in ("recipe_hash", "gate_hash", "fit_ids", "held_ids", "role", "role_codes",
                     "distinct_id", "held", "support_pair", "census"):
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
        if ordered:
            # The mirror of parent p reads the -q parent p' through a unitary row s,
            # S_s q(p') = -q(p) (symmetry service; antiunitary-only routes refuse there).
            from symmetry_maps import minus_q_parent_partners
            qt, operations = header["qirr"], header["operations"]
            partner_parent, partner_row = minus_q_parent_partners(
                header["q_irr_full_idx"], qt["irr_idx_q"], qt["sym_idx_q"], kgrid=header["grid"],
                sym_mats_k=np.asarray(operations["rotation"]),
                antiunitary=np.asarray(operations["antiunitary"], dtype=bool),
                authorized_rows=operations["authorized_rows"])

        budget.plan(0)
        logical_n = int(meta.n_rmu)
        column_extent = lambda width: padded_axis(
            width, mesh_xy, name="shared_pole_port",
            specs=((P("x", "y"), 0), (P("x", "y"), 1))).carrier
        eig, svd = budget.eigenplan(n), budget.eigenplan(2*n)
        receipts, factors, store_poles, store_counts, placed = {}, [], [], [], []
        receipt_entry_start = 0
        held_ids = [int(i) for i in recipe["held_ids"]]
        if not held_ids:
            raise ValueError("GATE shared_pole_held: got: no held samples; want: at least one held support; why: the model checks compare W and dW/ds there")
        held_lo, held_hi = min(held_ids), max(held_ids) + 1
        nq = int(header["bank_shape"]["nq"])
        # A round of parents runs one parent per rank (batch layout) from its sample read to
        # its sorted model; ordered rounds are partner-closed so the mirror exchange stays inside.
        # A parent reduces on its own rank whatever the linalg dial says (capacity refuses below).
        ranks = mesh_divisor(mesh_xy)
        rounds = parent_rounds(nq, ranks, partner_parent if ordered else None)
        batch_spec = P(("x", "y"))
        kernels = _round_kernels(mesh_xy)
        to_face, to_batch = batch_to_face(mesh_xy), face_to_batch_reshard(mesh_xy)
        fit_lo = min(int(i) for i in recipe["fit_ids"])
        fit_hi = max(int(i) for i in recipe["fit_ids"]) + 1
    for ids, real, slots in rounds:
        with timing.fenced_section("spole.batch_admission"):
            budget.batch_width = ranks
            budget.plan(0, phase="selection")
            budget.live(())
        with timing.fenced_section("spole.scratch_read"):
            with SlabIO(moments["path"], mode="r", mesh=mesh_xy) as moment_io:
                exact = read_shared_pole_bank(moment_io, meta=meta, header=moment_header,
                                              q_ids=ids, partition_spec=batch_spec,
                                              fields=("M0", "M1", "M2", "M3") if odd_moments else ("M1", "M3"))
        with timing.fenced_section("spole.infinity_selection"):
            width = min(logical_n, max(1, int(recipe["infinity_width"])))
            qi, round_infinity_values = leading_response_directions(
                exact["M1"], width, eigh_plan=eig, column_extent=column_extent,
                multiplet_tol=recipe["multiplet_relative_tolerance"], real_rows=real)
            budget.plan(qi.shape[-1], phase="selection")
            infinity = (qi, *(kernels.apply(exact[name], qi)
                              for name in (("M0", "M1", "M2", "M3") if odd_moments else ("M1", "M3"))))
            del exact
        with timing.fenced_section("spole.sample_batch_read"):
            budget.live(infinity)
            with SlabIO(bank["path"], mode="r", mesh=mesh_xy) as bank_io:
                sample_fields = (
                    ("Wc", "dWc_ds", "Wc_mirror", "dWc_mirror_ds")
                    if header.get("mirror_mode") is not None
                    else ("Wc", "dWc_ds"))
                samples = read_shared_pole_bank(
                    bank_io, meta=meta, header=header, q_ids=ids,
                    partition_spec=batch_spec, sample_span=(fit_lo, fit_hi),
                    fields=sample_fields)
            # The store admits this complete bounded scratch batch before
            # allocation. Charge it while directions/actions are selected;
            # release it before admitting the dense pencil.
            budget.retained_panels = tuple(samples.values())
        with timing.fenced_section("spole.direction_selection"):
            exchange = ((slots, *partner_realization(meta, header, ids, partner_parent, partner_row,
                                                      mesh_xy=mesh_xy)) if ordered else None)
            round_states, round_counts, round_roles = select_round_states(
                samples, recipe, sample_lo=fit_lo, real=real, mesh_xy=mesh_xy, eigh_plan=eig,
                svd_plan=svd, column_extent=column_extent, logical_n=logical_n, ordered=ordered,
                exchange=exchange)
            del samples, exchange, qi
        with timing.fenced_section("spole.reduction_admission"):
            infinity_counts = [int(v.shape[-1]) for v in round_infinity_values]
            tables = round_tables(round_counts, [st[1].shape[-1] for st in round_states],
                                  [st[0] for st in round_states], infinity_counts, infinity[0].shape[-1],
                                  column_extent=column_extent, ordered=ordered, odd_moments=odd_moments)
            side = int(tables["active"].shape[-1])
            # A pencil reduces on one device: its eight [side, side] blocks and the native eigh workspace.
            local_eigh = budget.eigenplan(side)
            if not distrib_la.fits_local(local_eigh, "eigh", ((1, side, side),) * 8, np.complex128,
                                         ledger.device_budget_bytes_per_rank):
                raise ValueError(f"GATE shared_pole_round_capacity: got: pencil side {side} at parents {ids[:real]}; want: eight [side, side] complex blocks and the eigh workspace within {ledger.device_budget_bytes_per_rank} bytes on one device; why: every parent reduces on its own rank")
            budget.retained_panels = (*infinity, *(v for st in round_states for v in st[1:]))
            budget.batch_width = ranks
            budget.plan(side, phase="reduction")
        with timing.fenced_section("spole.gram_reduction"):
            round_model, round_signed, vectors, round_diagnostics = reduce_round(
                round_states, infinity, tables, real=real, mesh_xy=mesh_xy, native_eigh=local_eigh.native_fn,
                ordered=ordered, odd_moments=odd_moments, keep_budget=recipe.get("pole_budget"))
            qi = infinity[0]
            del round_states, infinity
            round_reduction, round_zero, round_retained, round_permutation = jax.tree.map(np.asarray, round_diagnostics)
            poles, active = (np.asarray(a) for a in vectors)
            budget.retained_panels = (*round_model, *round_signed, qi, *factors)
        with timing.fenced_section("spole.gates"):
            for slot, q in enumerate(ids[:real]):
                if ordered and not round_reduction["orientation_paired"][slot]:
                    raise ValueError(ORIENTATION_PAIR_REFUSAL + f" (q={q})")
                for name in ("gram_diagonal_positive", "gram_valid", "retained_metric_positive"):
                    if not round_reduction[name][slot]:
                        raise ValueError(
                            f"GATE shared_pole_{name}: got: failed at q={q}, "
                            f"Gram min/max={round_reduction['gram_min_relative'][slot]}, "
                            f"paired H_r min/max={round_reduction['paired_min_relative'][slot] if ordered else 'n/a'}, "
                            f"metric infinity norm={round_reduction['metric_initial_infinity_norm'][slot]}, "
                            f"inverse-root residual={round_reduction['metric_inverse_root_residual_relative'][slot]}; "
                            f"want: Gram min/max >= {gates['normalized_gram_validity']['threshold']} "
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
        with timing.fenced_section("spole.coulomb"):
            budget.batch_width = ranks
            budget.plan(side, phase="model", sample_batch=len(held_ids))
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
                    meta, config, mesh_xy=mesh_xy, bank_io=bank, q_span=(lo, hi))
                del coulomb_sqrt
                parts.append(part)
                coulomb.update({q: dict(receipt, support_ranks=receipt["support_ranks"][q - lo:q - lo + 1])
                                for q in range(lo, hi)})
            inverse_sqrt = to_batch(face_rows(mesh_xy, tuple(sorted(ids[:real]).index(q) for q in ids))(*parts))
            del parts
        with timing.fenced_section("spole.sample_batch_read"):
            with SlabIO(bank["path"], mode="r", mesh=mesh_xy) as bank_io:
                held = read_shared_pole_bank(bank_io, meta=meta, header=header, q_ids=ids, partition_spec=batch_spec,
                                             sample_span=(held_lo, held_hi), fields=("Wc", "dWc_ds"))
            pick = kernels.take(tuple(i - held_lo for i in held_ids))
            held = tuple(pick(held[name]) for name in ("Wc", "dWc_ds"))
        with timing.fenced_section("spole.moment_read"):
            with SlabIO(moments["path"], mode="r", mesh=mesh_xy) as moment_io:
                exact = read_shared_pole_bank(moment_io, meta=meta, header=moment_header, q_ids=ids,
                                              partition_spec=batch_spec, fields=("M1", "M3"))
        with timing.fenced_section("spole.passivity_held"):
            passive, held_errors, reciprocity, moment_defects = check_round(
                round_model, round_signed, inverse_sqrt, held, (exact["M1"], exact["M3"]), qi,
                real=real, nodes=[_sample_point(recipe, i) for i in held_ids], eta_ry=recipe["eta_ev"] / RYD_TO_EV,
                mesh_xy=mesh_xy, native_eigh=local_eigh.native_fn, ordered=ordered)
            del inverse_sqrt, held, exact
        with timing.fenced_section("spole.gates"):
            for slot, q in enumerate(ids[:real]):
                if not passive["passivity"][slot]:
                    raise ValueError(f"GATE shared_pole_passivity: got: failed at q={q}; want: 0 <= V-whitened -W(i eta) <= I; why: passive screening")
                if not ordered and not np.all(reciprocity["passed"][slot]):
                    raise ValueError(f"GATE shared_pole_model_reciprocity: got: { {k: v[slot].tolist() for k, v in reciprocity.items()} } at q={q}; want: model preserves transpose symmetry of symmetric held data; why: conjugate-port closure must survive reduction")
        with timing.fenced_section("spole.receipts"):
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
                    ordered=ordered, odd_moments=odd_moments)
                receipt = construction_receipt(
                    measurements, capacity=ledger,
                    capacity_entry_start=receipt_entry_start, ordered=ordered)
                receipt.update(identity=identity, constructor=row)
                receipts[q] = receipt
            receipt_entry_start = len(ledger.entries)
        with timing.fenced_section("spole.export_prepare"):
            # The sorted model is an active prefix: keep the round's widest K, then restore to the face.
            width = column_extent(int(counts[:real].max()))
            factors.append(face_rows(mesh_xy, tuple(range(real)), width)(to_face(round_model[0])))
            store_poles.append(poles[:real, :width])
            store_counts.append(counts[:real])
            placed += list(ids[:real])
            del round_model, round_signed, qi, reductions, vectors
            ledger.live_stages = upstream
            budget.retained_panels = tuple(factors)
    with timing.fenced_section("spole.writer_stack"):
        if sorted(placed) != list(range(nq)):
            raise ValueError(f"GATE shared_pole_rounds: got: parents {sorted(placed)}; want: each of {nq} parents once; why: one canonical store")
        order = np.argsort(placed)
        width = max(block.shape[-1] for block in factors)
        public_b = canonical_factors(mesh_xy, tuple(int(i) for i in order))(*factors)
        store_poles = np.concatenate([np.pad(block, ((0, 0), (0, width - block.shape[-1])), constant_values=1.0)
                                      for block in store_poles])[order]
        budget.batch_width = nq
        budget.retained_panels = ()
        budget.plan(width, phase="model")
        budget.live((public_b,))
        del factors
    with timing.fenced_section("spole.writer"):
        store_header = write_shared_pole_model(
            output, public_b, jax.device_put(store_poles, NamedSharding(mesh_xy, P())),
            np.concatenate(store_counts)[order], q_span=(0, nq), meta=meta, tables=bank["tables"], recipe=recipe,
            receipts={"identity": identity, "q_receipts": [receipts[q] for q in range(nq)]}, ordered=ordered)
        ledger.live_stages = upstream
        del public_b
    with timing.fenced_section("spole.return"):
        return {"q_receipts": [receipts[q] for q in range(nq)], "model_header": store_header,
                "capacity": ledger.receipt(),
                "identity": identity, "status": "CONSTRUCTED"}
