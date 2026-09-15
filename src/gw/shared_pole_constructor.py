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

from functools import lru_cache, partial

import jax
import jax.numpy as jnp
from distrib_la import (diagonal_like, face_sharding, hermitian_block, hermitian_part,
                        join_columns, on_face)
from common import timing


def shared_pole_byte_terms(meta, *, mesh_xy, resolution, pencil_side,
                           parent_batch, sample_batch, phase="reduction"):
    """Price constructor carriers; the map CapacityLedger owns admission.

    Selection holds samples, current narrow actions and the n/2n direction
    solve; it has no R-by-R pencil. Reduction holds the actual selected
    pencil. Model checks hold factors and bounded samples, with no pencil.
    Native workspace is separately supplied by the service. No threshold
    or independent capacity policy lives in this constructor helper.
    """
    import math

    p = int(mesh_xy.shape["x"]) * int(mesh_xy.shape["y"])
    # Constructor carriers are mu x mu charge operators on every admitted deck.
    packed = int(meta.n_rmu_padded)
    b, a, r = int(parent_batch), int(sample_batch), int(pencil_side)
    if min(packed, b, a) <= 0 or r < 0:
        raise ValueError("GATE shared_pole_capacity: got: invalid extents; want: positive basis/batches and nonnegative pencil; why: live-set pricing")
    dense_copies = math.ceil(b / p) if resolution.layout == "local" else b / p
    if phase == "selection":
        dense = 24 * packed**2
        sample_faces = max(2*a, 2)
    elif phase == "reduction":
        dense = 14 * r*r + 12 * packed * r
        sample_faces = 0
    elif phase == "model":
        dense = 8 * packed**2 + 4 * packed * r
        sample_faces = max(2*a, 2)
    else:
        raise ValueError(f"unknown shared-pole capacity phase: {phase}")
    terms = {
        "sample_or_moment_batch": math.ceil(16*b*sample_faces*packed**2/p),
        "narrow_actions": math.ceil(16*b*3*packed*r/p),
        "replicated_scalars": 8*b*(12*r+4*packed),
        "phase_dense_temporaries": math.ceil(16*dense_copies*dense),
    }
    return {"terms_bytes_per_rank": terms,
            "resident_bytes_per_rank": sum(terms.values()),
            "layout": resolution.layout, "phase": phase, "pencil_side": r,
            "parent_batch": b, "sample_batch": a}




# The construction is split by concern; this module stays the door its callers name.
from gw.shared_pole_pencil import (  # noqa: F401  (re-exported for existing callers)
    _adjoint,
    _finite_column_g,
    _scale_rows,
    _subtract,
    assemble_ordered_shared_pole_pencil,
    assemble_shared_pole_pencil,
    finite_pencil_column,
    infinity_pencil_column,
    ordered_infinity_pencil_column,
)
from gw.shared_pole_reduction import (  # noqa: F401  (re-exported for existing callers)
    _metric_inverse_root,
    _paired_member,
    _paired_output,
    _restricted_block,
    reduce_ordered_shared_pole_pencil,
    reduce_shared_pole_pencil,
)
from gw.shared_pole_gates import (  # noqa: F401  (re-exported for existing callers)
    _factor_column_permutation,
    _passivity_response_checks,
    apply_shared_pole_zero_policy,
    ordered_moment_identity,
    ordered_pole_bound_ry,
    ordered_shared_pole_value,
    retained_moment_identity,
    shared_pole_operator_passivity,
    shared_pole_passivity,
    shared_pole_reciprocity,
    signed_shared_pole_passivity,
    sort_shared_pole_columns,
)
from gw.shared_pole_directions import (  # noqa: F401  (re-exported for existing callers)
    _direction_states,
    _fit_roles,
    _hermitian_part_kernel,
    _model_diagnostics,
    _odd_partner_directions,
    _parent_panel_slice,
    _parent_result_slice,
    _public_factor_kernel,
    _sample_point,
    _stack_model_kernel,
)

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
    timing.fence("spole.entry")
    with timing.section("spole.entry"):
        import numpy as np
        import jax
        from jax.sharding import NamedSharding, PartitionSpec as P
        import distrib_la
        from runtime.padding import padded_axis
        from file_io.slab_io import SlabIO
        from file_io.shared_pole_store import (
            validate_shared_pole_bank, read_shared_pole_bank, write_shared_pole_model,
        )
        from gw.gw_config import linalg_resolution
        from gw.shared_pole_recipe import (
            representation_row_passed, retained_moment_row_passed,
            construction_receipt, shared_real_pole_gates_v1_r3b as gates,
            shared_real_pole_gates_ordered_v1,
        )
        from common.units import RYD_TO_EV
        from gw.w_isdf import response_coulomb_powers

    timing.fence("spole.setup")
    with timing.section("spole.setup"):
        recipe = meta.shared_pole_recipe
        # Time-reversal-broken scalar states take the ordered particle-hole route.
        ordered = not bool(bank["tables"]["sym"].trs_allowed)
        from file_io.shared_pole_store import charge_representation
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
        native_queries = {}
        native_maxima = {"eigh": 0, "gemm": 0}
        workspace = 0
        current_side = 0
        current_phase = "selection"
        pending = []
        retained_panels = ()
        batch_width = 1
        plans = {}

        def eigenplan(side):
            if side not in plans:
                plans[side] = distrib_la.plan(
                    "eigh", mesh_xy, n=side, backend=resolution.eigh_backend,
                    batched_route=resolution.batched_route)
            return plans[side]

        def query_workspace(op, shapes, plan):
            nonlocal workspace
            key = (op, shapes)
            if key not in native_queries:
                size = (distrib_la.matmul_workspace_bytes_per_rank(
                    mesh_xy, shapes, np.complex128, backend="auto",
                    batched_route=resolution.batched_route) if op == "gemm" else
                    distrib_la.workspace_bytes_per_rank(plan, op, shapes, np.complex128))
                native_queries[key] = size
            if op == "gemm":
                native_maxima[op] = max(native_maxima[op], native_queries[key])
                workspace = sum(native_maxima.values())
            return native_queries[key]
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

        def capacity(side, *, phase=None, sample_batch=1, transpose_staging=0):
            nonlocal current_side, current_phase, workspace
            if phase is not None:
                current_phase = phase
            current_side = side
            extents = {n, 2*n} if current_phase == "selection" else (
                {side} if current_phase == "reduction" else {n})
            # Eigh scratch is transient: replace it at each phase boundary.
            # Only the actually used GEMM context workspace persists.
            native_maxima["eigh"] = max(query_workspace(
                "eigh", ((batch_width, extent, extent),), eigenplan(extent))
                for extent in sorted(extents))
            workspace = sum(native_maxima.values())
            price = shared_pole_byte_terms(
                meta, mesh_xy=mesh_xy, resolution=resolution, pencil_side=side,
                parent_batch=batch_width, sample_batch=sample_batch, phase=current_phase)
            # Other parents' narrow inputs survive selection and each model's
            # checks; they are additional live storage, never hidden in a limit.
            extra = sum(int(np.prod(a.sharding.shard_shape(a.shape))) * a.dtype.itemsize
                        for a in {id(a): a for a in retained_panels}.values())
            price["terms_bytes_per_rank"]["retained_parent_panels"] = extra
            price["terms_bytes_per_rank"]["gemm_transpose_staging"] = transpose_staging
            price["resident_bytes_per_rank"] += extra + transpose_staging
            row = ledger.reserve(f"constructor.plan.{len(ledger.entries)}",
                                 resident_bytes_per_rank=price["resident_bytes_per_rank"],
                                 workspace_bytes_per_rank=workspace,
                                 concurrent_with=upstream)
            return dict(row, price=price, native_workspace=dict(native_maxima))

        def expose_live(arrays):
            # Callees price only their additional allocations. Supply their
            # exact current inputs instead of the future dense-phase envelope,
            # so a store read does not count its own returned arrays twice.
            unique = {id(array): array for array in (*arrays, *retained_panels)}
            resident = sum(int(np.prod(array.sharding.shard_shape(array.shape)))
                           * array.dtype.itemsize for array in unique.values())
            row = ledger.reserve(f"constructor.live.{len(ledger.entries)}",
                                 resident_bytes_per_rank=resident,
                                 workspace_bytes_per_rank=workspace,
                                 concurrent_with=upstream)
            ledger.live_stages = (*upstream, row["stage"])

        capacity(0)
        logical_n = int(meta.n_rmu)
        face = NamedSharding(mesh_xy, P(None, "x", "y"))
        public_factor = NamedSharding(mesh_xy, P(None, "x", None, "y"))
        column_extent = lambda width: padded_axis(
            width, mesh_xy, name="shared_pole_port",
            specs=((P("x", "y"), 0), (P("x", "y"), 1))).carrier
        def mm(a, b, **kwargs):
            # Workspace belongs to the matmul route, not the eigh backend.
            # Its query excludes operand-sized endpoint transpose staging;
            # charge those transient faces separately before execution.
            shapes = tuple(value.shape[:-2] + (value.shape[-2:][::-1]
                           if kwargs.get(trans, "N") != "N" else value.shape[-2:])
                           for value, trans in ((a, "transa"), (b, "transb")))
            previous = workspace
            query_workspace("gemm", shapes, None)
            staging = (sum(int(np.prod(value.shape)) * value.dtype.itemsize
                           // int(mesh_xy.size)
                           for value, trans in ((a, "transa"), (b, "transb"))
                           if kwargs.get(trans, "N") != "N")
                       if resolution.batched_route != "batch_reshard" else 0)
            if workspace != previous or staging:
                capacity(current_side, transpose_staging=staging)
            return distrib_la.matmul(a, b, mesh=mesh_xy, backend="auto",
                                     batched_route=resolution.batched_route, **kwargs)

        eig, svd = eigenplan(n), eigenplan(2*n)
        # Bind once for this construction, outside both parent and support
        # loops. The GEMM closure owns native workspace accounting; keeping
        # this jit local also avoids retaining the map's capacity ledger in
        # a global callable cache across self-consistent reconstructions.
        model_diagnostics = jax.jit(partial(_model_diagnostics, matmul=mm))
        receipts = []
        receipt_entry_start = 0
        stack_models = _stack_model_kernel(mesh_xy)
        from runtime.padding import mesh_divisor
        from gw.shared_pole_local import pack_parent_panels, local_parent_reducer
        # Local dense algebra assigns independent parents to mesh ranks. The
        # distributed plan keeps its one-parent face-tiled execution schedule.
        # The ordered route runs the one-parent distributed schedule.
        local_layout = resolution.layout == "local" and not ordered
        batch_limit = mesh_divisor(mesh_xy) if local_layout else 1
        nq = int(header["bank_shape"]["nq"])
    for q_start in range(0, nq, batch_limit):
        timing.fence("spole.batch_admission")
        with timing.section("spole.batch_admission"):
            q_stop = min(q_start + batch_limit, nq)
            span = (q_start, q_stop)
            batch_width = q_stop - q_start
            capacity(0, phase="selection")
            expose_live(())
        timing.fence("spole.scratch_read")
        with timing.section("spole.scratch_read"):
            with SlabIO(moments["path"], mode="r", mesh=mesh_xy) as moment_io:
                exact = read_shared_pole_bank(moment_io, span, meta=meta,
                                              header=moment_header,
                                              fields=("M0", "M1", "M2", "M3") if odd_moments else ("M1", "M3"))
        timing.fence("spole.infinity_selection")
        with timing.section("spole.infinity_selection"):
            width = min(logical_n, max(1, int(recipe["infinity_width"])))
            qi, infinity_values = distrib_la.leading_eigenvectors(
                exact["M1"], width, eigh_plan=eig, column_extent=column_extent,
                multiplet_tol=recipe["multiplet_relative_tolerance"])
            capacity(qi.shape[-1], phase="selection")
            infinity = ((qi, mm(exact["M0"], qi), mm(exact["M1"], qi), mm(exact["M2"], qi), mm(exact["M3"], qi))
                        if odd_moments else (qi, mm(exact["M1"], qi), mm(exact["M3"], qi)))
            del exact
        with SlabIO(bank["path"], mode="r", mesh=mesh_xy) as bank_io:
            timing.fence("spole.sample_batch_read")
            with timing.section("spole.sample_batch_read"):
                sample_lo = min(int(i) for i in recipe["fit_ids"])
                sample_hi = max(int(i) for i in recipe["fit_ids"]) + 1
                expose_live(infinity)
                samples = read_shared_pole_bank(
                    bank_io, span, meta=meta, header=header,
                    sample_span=(sample_lo, sample_hi))
                # The store admits this complete bounded scratch batch before
                # allocation. Charge it while directions/actions are selected;
                # release it before admitting the dense pencil.
                retained_panels = tuple(samples.values())
                def read_sample(sample_id, retained_states):
                    index = sample_id - sample_lo
                    return samples["Wc"][:, index], samples["dWc_ds"][:, index]

                read_mirror = None
                if ordered:
                    from gw.mpa.sigma import shared_pole_minus_q_index
                    parents_full = [int(v) for v in header["q_irr_full_idx"]]
                    minus_full = shared_pole_minus_q_index(tuple(int(v) for v in header["grid"]))
                    partner = parents_full.index(int(minus_full[parents_full[q_start]]))

                    def read_mirror(sample_id):
                        # W_q(-conj z) = conj W_-q(z), and dW/ds likewise: one sample of the -q parent.
                        part = read_shared_pole_bank(
                            bank_io, (partner, partner + 1), meta=meta, header=header,
                            sample_span=(int(sample_id), int(sample_id) + 1))
                        return jnp.conj(part["Wc"][:, 0]), jnp.conj(part["dWc_ds"][:, 0])

            timing.fence("spole.direction_selection")
            with timing.section("spole.direction_selection"):
                states, counts, roles = _direction_states(
                    read_sample, recipe, eigh_plan=eig, svd_plan=svd, matmul=mm,
                    column_extent=column_extent, logical_n=logical_n, admit=capacity,
                    infinity_carrier=qi.shape[-1], ordered=ordered, read_mirror=read_mirror)
        timing.fence("spole.direction_pack_and_drain")
        with timing.section("spole.direction_pack_and_drain"):
            del samples
            jax.block_until_ready((states, infinity))
            retained_panels = (*infinity, *(v for st in states for v in st[1:]))
            finite_width = max(sum(row['carrier_width'] for row in parent) for parent in roles)
            infinity_width = infinity[0].shape[-1]
            batch_width = mesh_divisor(mesh_xy) if local_layout else 1
            # The permutation temporarily has the sum of the batched port
            # carriers, before compaction to the largest original parent side.
        timing.fence("spole.reduction_admission")
        with timing.section("spole.reduction_admission"):
            price = capacity(sum(st[1].shape[-1] for st in states) + infinity_width,
                             phase="reduction")
        timing.fence("spole.panel_pack")
        with timing.section("spole.panel_pack"):
            packed, extents = pack_parent_panels(
                states, infinity, counts, [v.shape[-1] for v in infinity_values],
                mesh_xy=mesh_xy, parent_batch=batch_width,
                layout=resolution.layout if not ordered else "distributed")
            for i, values in enumerate(infinity_values):
                ri = column_extent(values.shape[-1])
                parent_infinity = _parent_panel_slice(mesh_xy, ri)(infinity, np.int32(i))
                pending.append((q_start+i, None, None, None, parent_infinity[0], roles[i]))
            del states, infinity, qi, infinity_values, counts, roles, parent_infinity
            retained_panels = (*jax.tree.leaves(packed), *(item[4] for item in pending))
            batch_results = None
        if local_layout:
            timing.fence("spole.reduction_admission")
            with timing.section("spole.reduction_admission"):
                price = capacity(finite_width + infinity_width, phase="reduction")
                reduce_eigh = eigenplan(finite_width + infinity_width)
                extents += (extents[-1],) * (batch_width - len(extents))
            timing.fence("spole.gram_reduction")
            with timing.section("spole.gram_reduction"):
                batch_results = local_parent_reducer(
                    mesh_xy, reduce_eigh.native_fn, extents)(*packed)
                jax.block_until_ready(batch_results)
                del packed
                # Drop selected action panels after the fused boundary. Model
                # slices below may coexist with the full padded result buffer.
                retained_panels = (*jax.tree.leaves(batch_results),
                                   *(item[4] for item in pending))
        else:
            timing.fence("spole.panel_pack")
            with timing.section("spole.panel_pack"):
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
            timing.fence("spole.reduction_admission")
            with timing.section("spole.reduction_admission"):
                from gw.shared_pole_local import local_model_checks
                check_span = (pending[0][0], pending[-1][0]+1)
                held_ids = tuple(int(i) for i in recipe["held_ids"])
                sample_lo, sample_hi = min(held_ids), max(held_ids)+1
                capacity(batch_results[0][0].shape[-1], phase="model",
                         sample_batch=sample_hi-sample_lo)
                expose_live(batch_results[0])
            timing.fence("spole.coulomb")
            with timing.section("spole.coulomb"):
                coulomb_sqrt, inverse_sqrt, batch_coulomb = response_coulomb_powers(
                    meta, config, mesh_xy=mesh_xy, bank_io=bank, q_span=check_span)
                del coulomb_sqrt
            timing.fence("spole.sample_batch_read")
            with timing.section("spole.sample_batch_read"):
                expose_live((*batch_results[0], inverse_sqrt))
                with SlabIO(bank["path"], mode="r", mesh=mesh_xy) as bank_io:
                    held_samples = read_shared_pole_bank(
                        bank_io, check_span, meta=meta, header=header,
                        sample_span=(sample_lo, sample_hi), fields=("Wc", "dWc_ds"))
            timing.fence("spole.passivity_held")
            with timing.section("spole.passivity_held"):
                indices = jnp.asarray([i-sample_lo for i in held_ids])
                supports = jnp.asarray([_sample_point(recipe, i)**2 for i in held_ids])
                batch_checks = local_model_checks(mesh_xy, eig.native_fn)(
                    batch_results[0], inverse_sqrt,
                    held_samples["Wc"][:, indices], held_samples["dWc_ds"][:, indices],
                    supports, jnp.asarray(recipe["eta_ev"] / RYD_TO_EV))
                batch_checks = jax.tree.map(np.asarray, batch_checks)
                del inverse_sqrt, held_samples, indices, supports
        batch_width = 1
        selected = pending
        pending = []
        ready_models = []
        for slot, (q, states, infinity, active_columns, qi, roles) in enumerate(selected):
            timing.fence("spole.gram_reduction")
            with timing.section("spole.gram_reduction"):
                span = (q, q + 1)
                if batch_results is None:
                    price = capacity(active_columns.shape[-1], phase="reduction")
                    if ordered:
                        pencil = assemble_ordered_shared_pole_pencil(states, infinity, matmul=mm)
                        reduce_eigh = eigenplan(pencil[0].shape[-1])
                        model, signed, reduction = reduce_ordered_shared_pole_pencil(
                            pencil, active_columns, eigh=reduce_eigh.batched, matmul=mm, gates=gates)
                        ordered_retained = (ordered_moment_identity(signed, infinity, matmul=mm)
                                            if odd_moments else {})
                    else:
                        pencil = assemble_shared_pole_pencil(states, infinity, matmul=mm)
                        reduce_eigh = eigenplan(pencil[0].shape[-1])
                        model, reduction, coefficients = reduce_shared_pole_pencil(
                            pencil, active_columns, eigh=reduce_eigh.batched, matmul=mm, gates=gates)
                else:
                    model, reduction, zero, retained = _parent_result_slice(mesh_xy)(
                        batch_results, np.int32(slot))
                del states, infinity
            timing.fence("spole.gates")
            with timing.section("spole.gates"):
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
                capacity(model[0].shape[-1], phase="model")
            timing.fence("spole.sort")
            with timing.section("spole.sort"):
                model, permutation = sort_shared_pole_columns(model, mesh_xy=mesh_xy)
            if batch_checks is None:
                timing.fence("spole.coulomb")
                with timing.section("spole.coulomb"):
                    query_workspace("gemm", ((1, n, n), (1, n, n)), eig)
                    capacity(current_side)
                    expose_live((*model, qi))
                    coulomb_sqrt, inverse_sqrt, coulomb_receipt = response_coulomb_powers(
                        meta, config, mesh_xy=mesh_xy, bank_io=bank, q_span=span)
                timing.fence("spole.passivity")
                with timing.section("spole.passivity"):
                    passive = (signed_shared_pole_passivity(
                                   signed, inverse_sqrt, eta_ry=recipe["eta_ev"] / RYD_TO_EV,
                                   matmul=mm, eigh=eig.batched, gates=gates) if ordered else
                               shared_pole_passivity(model, inverse_sqrt,
                                                   eta_ry=recipe["eta_ev"] / RYD_TO_EV,
                                                   matmul=mm, eigh=eig.batched, gates=gates))
                    del coulomb_sqrt, inverse_sqrt
            else:
                timing.fence("spole.passivity")
                with timing.section("spole.passivity"):
                    passive = {key: value[slot:slot+1] for key, value in batch_checks[0].items()}
                    coulomb_receipt = dict(batch_coulomb)
                    coulomb_receipt["support_ranks"] = batch_coulomb["support_ranks"][slot:slot+1]
            timing.fence("spole.gates")
            with timing.section("spole.gates"):
                if not bool(jnp.all(passive["passivity"])):
                    raise ValueError(f"GATE shared_pole_passivity: got: failed at q={q}; want: 0 <= V-whitened -W(i eta) <= I; why: passive screening")
                expose_live((*model, qi))
            timing.fence("spole.moment_read")
            with timing.section("spole.moment_read"):
                with SlabIO(moments["path"], mode="r", mesh=mesh_xy) as moment_io:
                    exact = read_shared_pole_bank(moment_io, span, meta=meta,
                                                  header=moment_header, fields=("M1", "M3"))
            timing.fence("spole.moment_diagnostics")
            with timing.section("spole.moment_diagnostics"):
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
            timing.fence("spole.held")
            with timing.section("spole.held"):
                if batch_checks is None:
                    held = []
                    reciprocity = []
                    with SlabIO(bank["path"], mode="r", mesh=mesh_xy) as bank_io:
                        for sample_id in recipe["held_ids"]:
                            expose_live(model)
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
            timing.fence("spole.receipts")
            with timing.section("spole.receipts"):
                if ordered:
                    del signed
                b, poles, mask = model
                counts = jnp.sum(mask, axis=-1, dtype=jnp.int64)
                # All scalar reductions precede rank-selective store formatting.
                price = capacity(r)
                row = {"q_span": list(span), "roles": roles,
                       "diagnostic_operator": "raw-latent-pole-model",
                       "K": np.asarray(counts).tolist(), "J": int(np.unique(np.asarray(poles)[np.asarray(mask)]).size),
                       "damping_fraction": 0.0, "capacity": price, "coulomb": coulomb_receipt,
                       "condition": np.asarray(reduction["gram_condition"]).tolist(),
                       "normalized_gram_spectrum": np.asarray(reduction["gram_spectrum_relative"])[..., :int(reduction.get("pencil_side", [r])[0])].tolist(),
                       "native_workspace_queries": [dict(op=op, shapes=shapes, bytes_per_rank=value)
                                                     for (op, shapes), value in native_queries.items()],
                       "retained_moment_relative": {k: np.asarray(v).tolist() for k, v in retained.items()},
                       "moment_defects": {k: {a: np.asarray(value).tolist() for a, value in v.items()}
                                          for k, v in moment_defects.items()},
                       "held_W": held, "permutation": np.asarray(permutation).tolist(),
                       "storage_bytes": int(counts[0]) * (16*logical_n + 8)}
                if ordered:
                    row["ordered"] = {key: np.asarray(reduction[key]).tolist() for key in (
                        "positive_count", "negative_count", "infinite_weight_fraction",
                        "paired_rank", "paired_min_relative")}
                    row["ordered"]["odd_moments"] = odd_moments
                row["metric_inverse_root"] = {
                    name: np.asarray(reduction[name]).tolist() for name in (
                        "metric_initial_infinity_norm", "metric_inverse_root_iterations",
                        "metric_inverse_root_residual_fro", "metric_inverse_root_residual_relative")}
                representation_value = ({"nspinor": int(meta.nspinor), "trs_allowed": False, "ordered": True}
                                       if ordered else {"nspinor": int(meta.nspinor), "trs_allowed": True})
                measurements = {
                    "normalized_gram_keep": dict(value=int(reduction["retained_rank"][0]), passed=True, reason="normalized Gram cut, current q"),
                    "normalized_gram_validity": dict(value=float(reduction["gram_min_relative"][0]), passed=True, reason="normalized Gram spectrum"),
                    "zero_ritz_policy": dict(value=float(zero["dropped_factor_weight_fraction"][0]), passed=True, reason="physical factor weight, sentinels excluded"),
                    "finite_factors_poles": dict(value=True, passed=True, reason="zero policy, active prefix and exact inert sentinels"),
                    "passivity": dict(value={k: np.asarray(v).tolist() for k, v in passive.items() if k != "passivity"}, passed=True, reason=("signed particle-hole model, Hermitian part at i eta; anti-Hermitian part is the odd channel, reported" if ordered else "raw latent model; authenticated inverse Coulomb square root at current eta; projected operator not measured")),
                    "retained_subspace_moments": dict(value=row["retained_moment_relative"], passed=retained_moment_row_passed(row["retained_moment_relative"], gates["retained_subspace_moments"]), reason=(("signed model z-moments m0..m3 on the original infinity directions, each order against its own norm; projection-accuracy diagnostic beside full_m1/full_m3, not a refusal" if odd_moments else "finite-state ordered bank without odd moments: infinity block uncertified") if ordered else "raw latent Ritz identity: A=Y†GE, B=YA; pencil B†(G,H)B/2 versus model A†(I,Lambda)A/2")),
                    "held_w": dict(value=held, passed=True, reason="raw latent W and dW/ds diagnostics; projected operator not measured; no universal acceptance threshold"),
                    "model_reciprocity": (dict(value=None, passed=None, reason="not applicable: time-reversal-broken samples carry no transpose symmetry") if ordered else dict(value=reciprocity, passed=True, reason="raw latent model sampled W/dW transpose symmetry, conditional on symmetric reference; projected operator not measured; applicability recorded per sample")),
                    "full_m1_defect": dict(value=float(moment_defects["M1"]["full_relative"][0]), passed=bool(moment_defects["M1"]["full_relative"][0] <= gates["full_m1_defect"]["threshold"]), reason="raw latent model versus physical full M1; projected moment not measured; CD8 diagnostic band, never a refusal"),
                    "full_m3_defect": dict(value=float(moment_defects["M3"]["full_relative"][0]), passed=bool(moment_defects["M3"]["full_relative"][0] <= gates["full_m3_defect"]["threshold"]), reason="raw latent model versus physical full M3; projected moment not measured; CD8 diagnostic band, never a refusal"),
                    "representation": dict(value=representation_value, passed=representation_row_passed(representation_value, gates["representation"]["threshold"]), reason="current typed symmetry capability against the gate threshold"),
                    "capacity": dict(value=price, passed=True, reason="conservative aggregate constructor live-set price"),
                    "sc_rebuild": dict(value=identity, passed=True, reason="current recipe/census authenticated; directions and Ritz model rebuilt"),
                }
                receipt = construction_receipt(
                    measurements, capacity=ledger,
                    capacity_entry_start=receipt_entry_start, ordered=ordered)
                receipt_entry_start = len(ledger.entries)
                receipt.update(identity=identity, constructor=row)
            timing.fence("spole.export_prepare")
            with timing.section("spole.export_prepare"):
                public_b = _public_factor_kernel(mesh_xy)(b)
                del model, b, mask
                ready_models.append((public_b, poles, counts))
                retained_panels = (*retained_panels, public_b, poles, counts)
                receipts.append(receipt)
                del public_b, poles, counts
        timing.fence("spole.reduction_admission")
        with timing.section("spole.reduction_admission"):
            batch_width = len(selected)
            capacity(ready_models[0][0].shape[-1], phase="model")
        timing.fence("spole.writer_stack")
        with timing.section("spole.writer_stack"):
            public_b, poles, counts = stack_models(tuple(ready_models))
            expose_live((public_b, poles, counts))
            span = (selected[0][0], selected[-1][0] + 1)
            batch_receipt = {"identity": identity,
                             "q_receipts": receipts[-len(selected):]}
        timing.fence("spole.writer")
        with timing.section("spole.writer"):
            store_header = write_shared_pole_model(
                output, public_b, poles, counts, q_span=span, meta=meta,
                tables=bank["tables"], recipe=recipe, receipts=batch_receipt, ordered=ordered)
        timing.fence("spole.cleanup")
        with timing.section("spole.cleanup"):
            del public_b, poles, counts, ready_models
            ledger.live_stages = upstream
            del selected, batch_results
            retained_panels = ()
    timing.fence("spole.return")
    with timing.section("spole.return"):
        return {"q_receipts": receipts, "model_header": store_header,
                "capacity": ledger.receipt(),
                "identity": identity, "status": "CONSTRUCTED"}
