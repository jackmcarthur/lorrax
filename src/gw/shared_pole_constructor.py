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
from gw.shared_pole_directions import (_direction_states, _model_diagnostics,
                                       _parent_panel_slice, _parent_result_slice,
                                       _public_factor_kernel, _sample_point,
                                       _stack_model_kernel)
from gw.shared_pole_gates import (apply_shared_pole_zero_policy,
                                  ordered_moment_identity,
                                  retained_moment_identity,
                                  shared_pole_passivity,
                                  shared_pole_reciprocity,
                                  signed_shared_pole_passivity,
                                  sort_shared_pole_columns)
from gw.shared_pole_pencil import (assemble_ordered_shared_pole_pencil,
                                   assemble_shared_pole_pencil)
from gw.shared_pole_reduction import (reduce_ordered_shared_pole_pencil,
                                      reduce_shared_pole_pencil)


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
        from gw.shared_pole_local import pack_parent_panels, local_parent_reducer
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
        pending = []
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
            source, wraps = np.asarray(qt["sym_perm"]), np.asarray(qt["L_table"])
            realized = [p for p, row in enumerate(partner_row)
                        if not np.array_equal(source[row], np.arange(source.shape[1])) or np.any(wraps[row])]
            if realized:
                raise ValueError(f"GATE minus_q_partner: got: parents {realized} reach -q through rows {partner_row[realized].tolist()} with a centroid permutation or umklapp wrap; want: -q read as a raw parent; why: this constructor reads the partner's raw samples, and the realized partner R_s[W_q(p')] is not implemented here")

        budget.plan(0)
        logical_n = int(meta.n_rmu)
        face = NamedSharding(mesh_xy, P(None, "x", "y"))
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
        stack_models = _stack_model_kernel(mesh_xy)
        # Local dense algebra assigns independent parents to mesh ranks. The
        # distributed plan keeps its one-parent face-tiled execution schedule.
        # The ordered route runs the one-parent distributed schedule.
        local_layout = resolution.layout == "local" and not ordered
        batch_limit = mesh_divisor(mesh_xy) if local_layout else 1
        nq = int(header["bank_shape"]["nq"])
    for q_start in range(0, nq, batch_limit):
        with timing.fenced_section("spole.batch_admission"):
            q_stop = min(q_start + batch_limit, nq)
            span = (q_start, q_stop)
            budget.batch_width = q_stop - q_start
            budget.plan(0, phase="selection")
            budget.live(())
        with timing.fenced_section("spole.scratch_read"):
            with SlabIO(moments["path"], mode="r", mesh=mesh_xy) as moment_io:
                exact = read_shared_pole_bank(moment_io, span, meta=meta,
                                              header=moment_header,
                                              fields=("M0", "M1", "M2", "M3") if odd_moments else ("M1", "M3"))
        with timing.fenced_section("spole.infinity_selection"):
            width = min(logical_n, max(1, int(recipe["infinity_width"])))
            qi, infinity_values = distrib_la.leading_eigenvectors(
                exact["M1"], width, eigh_plan=eig, column_extent=column_extent,
                multiplet_tol=recipe["multiplet_relative_tolerance"])
            budget.plan(qi.shape[-1], phase="selection")
            infinity = ((qi, mm(exact["M0"], qi), mm(exact["M1"], qi), mm(exact["M2"], qi), mm(exact["M3"], qi))
                        if odd_moments else (qi, mm(exact["M1"], qi), mm(exact["M3"], qi)))
            del exact
        with SlabIO(bank["path"], mode="r", mesh=mesh_xy) as bank_io:
            with timing.fenced_section("spole.sample_batch_read"):
                sample_lo = min(int(i) for i in recipe["fit_ids"])
                sample_hi = max(int(i) for i in recipe["fit_ids"]) + 1
                budget.live(infinity)
                samples = read_shared_pole_bank(
                    bank_io, span, meta=meta, header=header,
                    sample_span=(sample_lo, sample_hi))
                # The store admits this complete bounded scratch batch before
                # allocation. Charge it while directions/actions are selected;
                # release it before admitting the dense pencil.
                budget.retained_panels = tuple(samples.values())
                def read_sample(sample_id):
                    index = sample_id - sample_lo
                    return samples["Wc"][:, index], samples["dWc_ds"][:, index]

                read_mirror = None
                if ordered:
                    partner = int(partner_parent[q_start])

                    def read_mirror(sample_id):
                        # W_q(-conj z) = conj W_-q(z), and dW/ds likewise: one sample of the -q parent.
                        part = read_shared_pole_bank(
                            bank_io, (partner, partner + 1), meta=meta, header=header,
                            sample_span=(int(sample_id), int(sample_id) + 1))
                        return jnp.conj(part["Wc"][:, 0]), jnp.conj(part["dWc_ds"][:, 0])

            with timing.fenced_section("spole.direction_selection"):
                states, counts, roles = _direction_states(
                    read_sample, recipe, eigh_plan=eig, svd_plan=svd, matmul=mm,
                    column_extent=column_extent, logical_n=logical_n, admit=budget.plan,
                    infinity_carrier=qi.shape[-1], ordered=ordered, read_mirror=read_mirror)
        with timing.fenced_section("spole.direction_pack_and_drain"):
            del samples
            jax.block_until_ready((states, infinity))
            budget.retained_panels = (*infinity, *(v for st in states for v in st[1:]))
            finite_width = max(sum(row['carrier_width'] for row in parent) for parent in roles)
            infinity_width = infinity[0].shape[-1]
            budget.batch_width = mesh_divisor(mesh_xy) if local_layout else 1
            # The permutation temporarily has the sum of the batched port
            # carriers, before compaction to the largest original parent side.
        with timing.fenced_section("spole.reduction_admission"):
            price = budget.plan(sum(st[1].shape[-1] for st in states) + infinity_width,
                                phase="reduction")
        with timing.fenced_section("spole.panel_pack"):
            packed, extents = pack_parent_panels(
                states, infinity, counts, [v.shape[-1] for v in infinity_values],
                mesh_xy=mesh_xy, parent_batch=budget.batch_width,
                layout=resolution.layout if not ordered else "distributed")
            for i, values in enumerate(infinity_values):
                ri = column_extent(values.shape[-1])
                parent_infinity = _parent_panel_slice(mesh_xy, ri)(infinity, np.int32(i))
                pending.append((q_start+i, None, None, None, parent_infinity[0], roles[i]))
            del states, infinity, qi, infinity_values, counts, roles, parent_infinity
            budget.retained_panels = (*jax.tree.leaves(packed), *(item[4] for item in pending))
            batch_results = None
        if local_layout:
            with timing.fenced_section("spole.reduction_admission"):
                price = budget.plan(finite_width + infinity_width, phase="reduction")
                reduce_eigh = budget.eigenplan(finite_width + infinity_width)
                extents += (extents[-1],) * (budget.batch_width - len(extents))
            with timing.fenced_section("spole.gram_reduction"):
                batch_results = local_parent_reducer(
                    mesh_xy, reduce_eigh.native_fn, extents)(*packed)
                jax.block_until_ready(batch_results)
                del packed
                # Drop selected action panels after the fused boundary. Model
                # slices below may coexist with the full padded result buffer.
                budget.retained_panels = (*jax.tree.leaves(batch_results),
                                          *(item[4] for item in pending))
        else:
            with timing.fenced_section("spole.panel_pack"):
                # The distributed schedule admits one parent; its packed panels
                # already have the original finite and infinity extents.
                iq, _, _, _, directions, row_roles = pending[0]
                finite, infinity, active, _columns = jax.tree.map(lambda a: a[:1], packed)
                if ordered:
                    # k0 and k1 double the infinity columns; a finite-state bank has none.
                    rf = active.shape[-1] - infinity[0].shape[-1]
                    active = (jnp.concatenate((active, active[:, rf:]), axis=-1)
                              if odd_moments else active[:, :rf])
                    infinity = infinity if odd_moments else None
                pending = [(iq, [finite], infinity, active, directions, row_roles)]
                del packed
        batch_checks = None
        if batch_results is not None:
            with timing.fenced_section("spole.reduction_admission"):
                from gw.shared_pole_local import local_model_checks
                check_span = (pending[0][0], pending[-1][0]+1)
                held_ids = tuple(int(i) for i in recipe["held_ids"])
                sample_lo, sample_hi = min(held_ids), max(held_ids)+1
                budget.plan(batch_results[0][0].shape[-1], phase="model",
                            sample_batch=sample_hi-sample_lo)
                budget.live(batch_results[0])
            with timing.fenced_section("spole.coulomb"):
                coulomb_sqrt, inverse_sqrt, batch_coulomb = response_coulomb_powers(
                    meta, config, mesh_xy=mesh_xy, bank_io=bank, q_span=check_span)
                del coulomb_sqrt
            with timing.fenced_section("spole.sample_batch_read"):
                budget.live((*batch_results[0], inverse_sqrt))
                with SlabIO(bank["path"], mode="r", mesh=mesh_xy) as bank_io:
                    held_samples = read_shared_pole_bank(
                        bank_io, check_span, meta=meta, header=header,
                        sample_span=(sample_lo, sample_hi), fields=("Wc", "dWc_ds"))
            with timing.fenced_section("spole.passivity_held"):
                indices = jnp.asarray([i-sample_lo for i in held_ids])
                supports = jnp.asarray([_sample_point(recipe, i)**2 for i in held_ids])
                batch_checks = local_model_checks(mesh_xy, eig.native_fn)(
                    batch_results[0], inverse_sqrt,
                    held_samples["Wc"][:, indices], held_samples["dWc_ds"][:, indices],
                    supports, jnp.asarray(recipe["eta_ev"] / RYD_TO_EV))
                batch_checks = jax.tree.map(np.asarray, batch_checks)
                del inverse_sqrt, held_samples, indices, supports
        budget.batch_width = 1
        selected = pending
        pending = []
        ready_models = []
        for slot, (q, states, infinity, active_columns, qi, roles) in enumerate(selected):
            with timing.fenced_section("spole.gram_reduction"):
                span = (q, q + 1)
                if batch_results is None:
                    price = budget.plan(active_columns.shape[-1], phase="reduction")
                    if ordered:
                        pencil = assemble_ordered_shared_pole_pencil(states, infinity, matmul=mm)
                        reduce_eigh = budget.eigenplan(pencil[0].shape[-1])
                        model, signed, reduction = reduce_ordered_shared_pole_pencil(
                            pencil, active_columns, eigh=reduce_eigh.batched, matmul=mm, gates=gates)
                        ordered_retained = (ordered_moment_identity(signed, infinity, matmul=mm)
                                            if odd_moments else {})
                    else:
                        pencil = assemble_shared_pole_pencil(states, infinity, matmul=mm)
                        reduce_eigh = budget.eigenplan(pencil[0].shape[-1])
                        model, reduction, coefficients = reduce_shared_pole_pencil(
                            pencil, active_columns, eigh=reduce_eigh.batched, matmul=mm, gates=gates)
                else:
                    model, reduction, zero, retained = _parent_result_slice(mesh_xy)(
                        batch_results, np.int32(slot))
                del states, infinity
            with timing.fenced_section("spole.gates"):
                for name in ("gram_diagonal_positive", "gram_valid", "retained_metric_positive"):
                    if not bool(jnp.all(reduction[name])):
                        raise ValueError(
                            f"GATE shared_pole_{name}: got: failed at q={q}, "
                            f"Gram min/max={reduction['gram_min_relative']}, "
                            f"metric infinity norm={reduction['metric_initial_infinity_norm']}, "
                            f"inverse-root residual={reduction['metric_inverse_root_residual_relative']}; "
                            f"want: Gram min/max >= {gates['normalized_gram_validity']['threshold']} "
                            "and valid diagonal/retained metric; why: no PSD repair")
                if batch_results is None:
                    model, zero = apply_shared_pole_zero_policy(model, gates=gates)
                    if ordered:
                        zero["zero_policy"] = zero["zero_policy"] & reduction["infinite_weight_ok"]
                if not bool(jnp.all(zero["zero_policy"])):
                    raise ValueError(f"GATE shared_pole_zero_ritz: got: failed at q={q}; want: finite positive response within dropped-weight budget; why: no pole clipping")
                if batch_results is None and ordered:
                    retained = ordered_retained
                elif batch_results is None:
                    # E selects the last infinity block of X. Build it as a face array;
                    # only the small row/column coordinate vectors are replicated.
                    r, ri = pencil[0].shape[-1], qi.shape[-1]
                    selector = jax.jit(lambda: (jnp.arange(r)[:, None] ==
                                               jnp.arange(r-ri, r)[None, :])[None].astype(jnp.complex128),
                                       out_shardings=face)()
                    retained = retained_moment_identity(pencil, coefficients, model, selector, matmul=mm)
                # The ordered identity is on the ORIGINAL infinity directions: exact only for the
                # full Galerkin span, projection accuracy after the keep/retention cuts. It is
                # reported beside the full_m1/full_m3 diagnostic bands, as the TRS route reports
                # its original-direction defects; the retained Ritz algebra is the metric gate above.
                if not ordered and not all(bool(jnp.all(value <= gates["retained_subspace_moments"]["threshold"]))
                                           for value in retained.values()):
                    raise ValueError(f"GATE shared_pole_retained_moments: got: failed at q={q}; want: projected latent moment identity <=1e-10; why: corrected Ritz algebra")
                if batch_results is None and not ordered:
                    del pencil, coefficients, selector
                elif ordered:
                    del pencil
                r = model[0].shape[-1]
                del active_columns
                budget.plan(model[0].shape[-1], phase="model")
            with timing.fenced_section("spole.sort"):
                model, permutation = sort_shared_pole_columns(model, mesh_xy=mesh_xy)
            if batch_checks is None:
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
            else:
                with timing.fenced_section("spole.passivity"):
                    passive = {key: value[slot:slot+1] for key, value in batch_checks[0].items()}
                    coulomb_receipt = dict(batch_coulomb)
                    coulomb_receipt["support_ranks"] = batch_coulomb["support_ranks"][slot:slot+1]
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
                if batch_checks is None:
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
                else:
                    held = [{"sample_id": sample_id,
                             "Wc": float(batch_checks[1][slot, 0, i]),
                             "dWc_ds": float(batch_checks[1][slot, 1, i])}
                            for i, sample_id in enumerate(held_ids)]
                    reciprocity = {key: value[slot].tolist() for key, value in batch_checks[2].items()}
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
                    pencil_side=r, ordered=ordered, odd_moments=odd_moments)
                receipt = construction_receipt(
                    measurements, capacity=ledger,
                    capacity_entry_start=receipt_entry_start, ordered=ordered)
                receipt_entry_start = len(ledger.entries)
                receipt.update(identity=identity, constructor=row)
            with timing.fenced_section("spole.export_prepare"):
                public_b = _public_factor_kernel(mesh_xy)(b)
                del model, b, mask
                ready_models.append((public_b, poles, counts))
                budget.retained_panels = (*budget.retained_panels, public_b, poles, counts)
                receipts.append(receipt)
                del public_b, poles, counts
        with timing.fenced_section("spole.reduction_admission"):
            budget.batch_width = len(selected)
            budget.plan(ready_models[0][0].shape[-1], phase="model")
        with timing.fenced_section("spole.writer_stack"):
            public_b, poles, counts = stack_models(tuple(ready_models))
            budget.live((public_b, poles, counts))
            span = (selected[0][0], selected[-1][0] + 1)
            batch_receipt = {"identity": identity,
                             "q_receipts": receipts[-len(selected):]}
        with timing.fenced_section("spole.writer"):
            store_header = write_shared_pole_model(
                output, public_b, poles, counts, q_span=span, meta=meta,
                tables=bank["tables"], recipe=recipe, receipts=batch_receipt, ordered=ordered)
        with timing.fenced_section("spole.cleanup"):
            del public_b, poles, counts, ready_models
            ledger.live_stages = upstream
            del selected, batch_results
            budget.retained_panels = ()
    with timing.fenced_section("spole.return"):
        return {"q_receipts": receipts, "model_header": store_header,
                "capacity": ledger.receipt(),
                "identity": identity, "status": "CONSTRUCTED"}
