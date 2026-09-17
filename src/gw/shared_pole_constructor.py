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

from functools import partial

import jax
import jax.numpy as jnp
from common import timing
from gw.shared_pole_capacity import ConstructorCapacity
from gw.shared_pole_directions import (_model_diagnostics, _public_factor_kernel,
                                       _round_kernels, _sample_point, select_round_states)
from gw.shared_pole_gates import (shared_pole_passivity,
                                  shared_pole_reciprocity,
                                  signed_shared_pole_passivity)
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
        from gw.shared_pole_local import (batch_to_face, face_rows, own_extent_receipts, parent_rounds,
                                          partner_realization, reduce_round, round_tables)
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
        mm = budget.matmul
        eig, svd = budget.eigenplan(n), budget.eigenplan(2*n)
        # Bind once for this construction, outside both parent and support
        # loops. The GEMM seam owns native workspace accounting; keeping
        # this jit local also avoids retaining the map's capacity ledger in
        # a global callable cache across self-consistent reconstructions.
        model_diagnostics = jax.jit(partial(_model_diagnostics, matmul=mm))
        receipts = []
        receipt_entry_start = 0
        nq = int(header["bank_shape"]["nq"])
        # A round of parents runs one parent per rank (batch layout) from its sample read to
        # its sorted model; ordered rounds are partner-closed so the mirror exchange stays inside.
        # A parent reduces on its own rank whatever the linalg dial says (capacity refuses below).
        ranks = mesh_divisor(mesh_xy)
        rounds = parent_rounds(nq, ranks, partner_parent if ordered else None)
        batch_spec = P(("x", "y"))
        kernels = _round_kernels(mesh_xy)
        to_face = batch_to_face(mesh_xy)
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
            qi, round_infinity_values = distrib_la.leading_eigenvectors(
                exact["M1"], width, eigh_plan=eig, column_extent=column_extent,
                multiplet_tol=recipe["multiplet_relative_tolerance"], real_rows=real)
            budget.plan(qi.shape[-1], phase="selection")
            infinity = (qi, *(kernels.apply(exact[name], qi)
                              for name in (("M0", "M1", "M2", "M3") if odd_moments else ("M1", "M3"))))
            del exact
        with timing.fenced_section("spole.sample_batch_read"):
            budget.live(infinity)
            with SlabIO(bank["path"], mode="r", mesh=mesh_xy) as bank_io:
                samples = read_shared_pole_bank(
                    bank_io, meta=meta, header=header, q_ids=ids, partition_spec=batch_spec,
                    sample_span=(fit_lo, fit_hi))
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
            local_eigh = distrib_la.plan("eigh", mesh_xy, n=side, backend="off", batched_route="batch_reshard")
            if not distrib_la.fits_local(local_eigh, "eigh", ((1, side, side),) * 8, np.complex128,
                                         ledger.device_budget_bytes_per_rank):
                raise ValueError(f"GATE shared_pole_round_capacity: got: pencil side {side} at parents {ids[:real]}; want: eight [side, side] complex blocks and the eigh workspace within {ledger.device_budget_bytes_per_rank} bytes on one device; why: every parent reduces on its own rank")
            budget.retained_panels = (*infinity, *(v for st in round_states for v in st[1:]))
            budget.batch_width = ranks
            price = budget.plan(side, phase="reduction")
        with timing.fenced_section("spole.gram_reduction"):
            round_model, round_signed, round_diagnostics = reduce_round(
                round_states, infinity, tables, real=real, mesh_xy=mesh_xy, native_eigh=local_eigh.native_fn,
                ordered=ordered, odd_moments=odd_moments, keep_budget=recipe.get("pole_budget"))
            round_qi = to_face(infinity[0])
            del round_states, infinity
            round_reduction, round_zero, round_retained, round_permutation = jax.tree.map(np.asarray, round_diagnostics)
            budget.retained_panels = (*round_model, *round_signed, round_qi)
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
            vectors = jax.tree.map(np.asarray, (round_model[1:], round_signed[1:]))
        replicated = NamedSharding(mesh_xy, P())
        for slot, q in enumerate(ids[:real]):
            span = (q, q + 1)
            take = face_rows(mesh_xy, (slot,))
            model_vectors, signed_vectors = jax.tree.map(
                lambda a: jax.device_put(a[slot:slot + 1], replicated), vectors)
            model = (take(round_model[0]), *model_vectors)
            if ordered:
                signed = (take(round_signed[0]), *signed_vectors)
            qi = take(round_qi)
            reduction, roles = reductions[slot], round_roles[slot]
            zero = {key: value[slot:slot + 1] for key, value in round_zero.items()}
            retained = {key: value[slot:slot + 1] for key, value in round_retained.items()}
            permutation = round_permutation[slot:slot + 1]
            r = model[0].shape[-1]
            budget.batch_width = 1
            budget.plan(r, phase="model")
            with timing.fenced_section("spole.coulomb"):
                budget.query_workspace("gemm", ((1, n, n), (1, n, n)))
                budget.plan()
                budget.live((*model, qi))
                coulomb_sqrt, inverse_sqrt, coulomb_receipt = response_coulomb_powers(
                    meta, config, mesh_xy=mesh_xy, bank_io=bank, q_span=span)
            with timing.fenced_section("spole.passivity"):
                passive = (signed_shared_pole_passivity(
                               signed, inverse_sqrt, eta_ry=recipe["eta_ev"] / RYD_TO_EV,
                               matmul=mm, eigh=eig.batched, gates=gates) if ordered else
                           shared_pole_passivity(model, inverse_sqrt,
                                               eta_ry=recipe["eta_ev"] / RYD_TO_EV,
                                               matmul=mm, eigh=eig.batched, gates=gates))
                del coulomb_sqrt, inverse_sqrt
            with timing.fenced_section("spole.gates"):
                if not bool(jnp.all(passive["passivity"])):
                    raise ValueError(f"GATE shared_pole_passivity: got: failed at q={q}; want: 0 <= V-whitened -W(i eta) <= I; why: passive screening")
                budget.live((*model, qi))
            with timing.fenced_section("spole.moment_read"):
                with SlabIO(moments["path"], mode="r", mesh=mesh_xy) as moment_io:
                    exact = read_shared_pole_bank(moment_io, span, meta=meta,
                                                  header=moment_header, fields=("M1", "M3"))
            with timing.fenced_section("spole.moment_diagnostics"):
                if ordered:
                    # Signed model moments: M1 = sum c c^H mu^-2 / 2, M3 = sum c c^H mu^-4 / 2.
                    c_signed, mu_signed, kept = signed
                    inverse = jnp.where(kept, 1 / jnp.where(kept, jnp.abs(mu_signed), 1), 0)
                    moment_defects = model_diagnostics(
                        (c_signed * inverse[:, None, :], inverse**2, kept), exact, qi)
                    del c_signed, mu_signed, kept, inverse
                else:
                    moment_defects = model_diagnostics(model, exact, qi)
                del exact, qi
            with timing.fenced_section("spole.held"):
                held = []
                reciprocity = []
                with SlabIO(bank["path"], mode="r", mesh=mesh_xy) as bank_io:
                    for sample_id in recipe["held_ids"]:
                        budget.live(model)
                        samples = read_shared_pole_bank(bank_io, span, meta=meta, header=header,
                                                       sample_span=(int(sample_id), int(sample_id)+1),
                                                       fields=("Wc", "dWc_ds"))
                        b, poles, mask = model
                        s = _sample_point(recipe, int(sample_id)) ** 2
                        weights = jnp.where(mask, 1 / (s-poles), 0)
                        if ordered:
                            # Signed particle-hole model at z; dW/ds = (dW/dz)/(2z).
                            b, mu_signed, mask = signed
                            z = _sample_point(recipe, int(sample_id))
                            weights = jnp.where(mask, 1 / (z * mu_signed - 1), 0)
                            derivative = jnp.where(mask, -mu_signed / (z * mu_signed - 1)**2 / (2 * z), 0)
                        diagnostic = {"sample_id": int(sample_id)}
                        for field, weight in (("Wc", weights), ("dWc_ds", derivative if ordered else -weights**2)):
                            sample = samples[field][:, 0]
                            value = mm(b * weight[:, None, :], b, transb="C")
                            diagnostic[field] = float(jnp.linalg.norm(value-sample) /
                                                      jnp.maximum(jnp.linalg.norm(sample), jnp.finfo(jnp.float64).tiny))
                            if not ordered:
                                reciprocity.append(shared_pole_reciprocity(value, sample, gates=gates))
                        held.append(diagnostic)
                        del samples, sample, value
                reciprocity = ({key: np.asarray([row[key] for row in reciprocity]).tolist()
                                for key in reciprocity[0]} if not ordered else {})
                if not ordered and not np.all(reciprocity["passed"]):
                    raise ValueError(f"GATE shared_pole_model_reciprocity: got: {reciprocity} at q={q}; want: model preserves transpose symmetry of symmetric held data; why: conjugate-port closure must survive reduction")
            with timing.fenced_section("spole.receipts"):
                if ordered:
                    del signed
                b, poles, mask = model
                counts = jnp.sum(mask, axis=-1, dtype=jnp.int64)
                # All scalar reductions precede rank-selective store formatting.
                price = budget.plan(r)
                row, measurements = build_construction_row(
                    model, counts,
                    dict(reduction=reduction, zero=zero, passive=passive,
                         retained=retained, moment_defects=moment_defects,
                         held=held, reciprocity=reciprocity, permutation=permutation),
                    span=span, roles=roles, price=price, coulomb=coulomb_receipt,
                    native_queries=budget.native_queries, identity=identity,
                    gates=gates, nspinor=int(meta.nspinor), logical_n=logical_n,
                    ordered=ordered, odd_moments=odd_moments)
                receipt = construction_receipt(
                    measurements, capacity=ledger,
                    capacity_entry_start=receipt_entry_start, ordered=ordered)
                receipt_entry_start = len(ledger.entries)
                receipt.update(identity=identity, constructor=row)
                receipts.append(receipt)
            with timing.fenced_section("spole.writer"):
                store_header = write_shared_pole_model(
                    output, _public_factor_kernel(mesh_xy)(b), poles, counts, q_span=span, meta=meta,
                    tables=bank["tables"], recipe=recipe,
                    receipts={"identity": identity, "q_receipts": [receipt]}, ordered=ordered)
                del model, b, poles, mask, counts
        with timing.fenced_section("spole.cleanup"):
            ledger.live_stages = upstream
            del round_model, round_signed, round_qi, reductions, vectors
            budget.retained_panels = ()
    with timing.fenced_section("spole.return"):
        return {"q_receipts": receipts, "model_header": store_header,
                "capacity": ledger.receipt(),
                "identity": identity, "status": "CONSTRUCTED"}
