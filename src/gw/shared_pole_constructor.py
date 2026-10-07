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
from gw.shared_pole_directions import _sample_point, port_extent


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


class _StoredPoleRoundProvider:
    """The canonical store adapter; numerical round ownership is shared."""
    def __init__(self, bank, moments, meta, *, mesh_xy, header, moment_header,
                 read_spec, coulomb_config):
        self.bank, self.moment_bank, self.meta = bank, moments, meta
        self.mesh_xy, self.header, self.moment_header = mesh_xy, header, moment_header
        self.read_spec, self.coulomb_config = read_spec, coulomb_config
        self.identity = bank['identity']

    def moments(self, q_ids, *, fields):
        from file_io.shared_pole_store import open_shared_pole_bank, read_shared_pole_bank
        with open_shared_pole_bank(self.moment_bank['path'], mesh_xy=self.mesh_xy) as io:
            return read_shared_pole_bank(io, meta=self.meta, header=self.moment_header,
                q_ids=q_ids, partition_spec=self.read_spec, fields=fields)

    def selection(self, q_ids, *, sample_ids, line_span):
        from file_io.shared_pole_store import open_shared_pole_bank, read_shared_pole_bank, read_line_panels
        with open_shared_pole_bank(self.bank['path'], mesh_xy=self.mesh_xy) as io:
            samples = read_shared_pole_bank(io, meta=self.meta, header=self.header,
                q_ids=q_ids, partition_spec=self.read_spec, sample_ids=sample_ids,
                fields=('Wc', 'dWc_ds'))
            line = {sid: read_line_panels(io, meta=self.meta, header=self.header,
                family='charge', sample=sid, q_ids=q_ids, partition_spec=self.read_spec)
                for sid in range(*line_span)}
        return samples, line

    def samples(self, q_ids, *, sample_span):
        from file_io.shared_pole_store import open_shared_pole_bank, read_shared_pole_bank
        with open_shared_pole_bank(self.bank['path'], mesh_xy=self.mesh_xy) as io:
            return read_shared_pole_bank(io, meta=self.meta, header=self.header,
                q_ids=q_ids, partition_spec=self.read_spec, sample_span=sample_span,
                fields=('Wc', 'dWc_ds'))

    def coulomb_inverse(self, q_span):
        from gw.w_isdf import response_coulomb_powers
        root, inverse, receipt = response_coulomb_powers(self.meta, self.coulomb_config,
            mesh_xy=self.mesh_xy, bank_io=self.bank, q_span=q_span)
        del root
        return inverse, receipt


def construct_shared_poles(bank, moments, meta, config, *, mesh_xy, output, residence=None):
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
            charge_representation, validate_shared_pole_bank,
            write_shared_pole_model,
        )
        from gw.gw_config import linalg_resolution
        from gw.shared_pole_local import canonical_factors, carrier_history
        from gw.shared_pole_recipe import (
            charge4_gates,
            shared_real_pole_gates_v1_r3b as gates,
            shared_real_pole_gates_ordered_v1,
        )

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
        ledger = meta.shared_pole_capacity
        upstream = ledger.live_stages
        n = int(meta.n_rmu_padded)
        # Every device byte this construction admits is priced here; the driver
        # below keeps only `budget.retained_panels` and `budget.batch_width`
        # current as it moves from selection to reduction to the model checks.
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
        from gw.shared_pole_execution import sector_round_schedule
        rounds = [row[:3] for row in sector_round_schedule(
            bank, header, meta, config, mesh_xy, execution=execution, batch_width=face_batch)]
        batch_spec = P(("x", "y"))
        read_spec = batch_spec if execution == 'local' else None
    from gw.shared_pole_round import (PoleEndpointGeometry, SharedPoleRoundPlan,
                                      construct_shared_pole_round)
    provider = _StoredPoleRoundProvider(bank, moments, meta, mesh_xy=mesh_xy,
        header=header, moment_header=moment_header, read_spec=read_spec,
        coulomb_config=coulomb_config)
    endpoint = PoleEndpointGeometry('packed-centroid-charge', logical_n, n,
        int(meta.nspinor), (int(meta.nkx), int(meta.nky), int(meta.nkz)), int(meta.nk_tot))
    round_plan = SharedPoleRoundPlan(endpoint, recipe, gates, mesh_xy, execution,
        ordered, odd_moments, charge4, bool(config.debug.sigma_freq_debug_output),
        identity, ledger, upstream, conservative_side, selection_faces, column_extent,
        eig, tuple(dense_fit), (line_lo, line_hi), moment_fields, carrier_history(meta),
        execution_receipt['face_batch']['reduction'] if execution == 'face' else None,
        retained_bound if execution == 'face' else None,
        sizing['carrier'] if execution == 'face' else None)
    for ids, real, slots in rounds:
        result = construct_shared_pole_round(provider, plan=round_plan,
            q_ids=ids, real_rows=real, budget=budget, retained_factors=tuple(factors),
            capacity_entry_start=receipt_entry_start, phase=phase)
        factors.append(result.factor)
        store_poles.append(result.poles2)
        store_counts.append(result.counts)
        placed += list(result.q_ids)
        receipts.update(result.receipts)
        receipt_entry_start = result.capacity_entry_end
    with phase("writer_stack"):
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
            receipts={"identity": identity, "q_receipts": [receipts[q] for q in range(nq)]}, ordered=ordered)
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
                "execution": dict(mode=execution, **execution_receipt)}
