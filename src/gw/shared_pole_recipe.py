"""Shared real-pole input recipe and gate vocabulary (DESIGN §§5–6).

Host metadata only: no response matrices, device allocation, backend selection,
chemical-potential solve, or numerical gate implementation lives here. Physical
sample coordinates are Ry; reporting coordinates are eV. Consumers import these
tables instead of copying the thresholds into bank/constructor/store/Sigma code.
"""
from __future__ import annotations

import hashlib
import json
import math

RECIPE_VERSION = "shared_real_pole_v2_r1"
GATE_VERSION = "shared_real_pole_gates_v1_r3b"
RECEIPT_SCHEMA = "lorrax.shared-real-pole.receipt.v1"
ROLE_CODES = {"line": 0, "imaginary": 1, "infinity": 2, "held_line": 3, "held_imaginary": 4}

# SCALE-FREE BY CONSTRUCTION.  Every dial below is a multiple of eta, of the
# plasma frequency of the screening-active electrons, of the band energies, or
# of the Sigma omega-box, or a dimensionless fraction.  There is no absolute
# energy in this table, so the same recipe resolves sensibly for a material
# with any bandwidth and any plasmon.  v1_r3b's four absolute constants
# (line_break_ev 12, plasma_margin_ev 3.5, imaginary_floor_max_ev 16,
# active_depth_ev 15) each encoded one of the two campaign decks: the 12 eV
# break fell BELOW Si's 16.6 eV plasmon and ABOVE Na's 6.05 eV one, so Si
# sampled its own plasmon on the coarse branch, and the 15 eV depth sat 0.04 eV
# from Na's 2p semicore.
shared_real_pole_v2_r1 = {
    "version": RECIPE_VERSION,
    "height_eta_factor": 4.0,
    # An electron screens at a frequency only if it is bound by less than that
    # frequency: the active set is the fixed point of depth <= factor*omega_p
    # over the set including the candidate band.  Factor 1 needs no calibration.
    "active_plasma_factor": 1.0,
    "borderline_plasma_factor": 2.0,
    # Fine spacing 2*eta on [0, omega_fine], then geometric growth: the
    # structure (e-h continuum and the collective pole) lives at and below
    # omega_fine, and above it W_c is a smooth tail whose only scale is omega.
    "line_step_eta_factor": 2.0,
    # MEASURED, not assumed. 0.15 was tried on Si because the held-W trigger
    # fired (the second held line point at 23.206 eV, inside that deck's
    # 30.325 eV Sigma reach, at 15.10x the incumbent's dWc/ds). It works as a
    # model-fidelity dial -- the tail held point improves 6.18x -- but it moves
    # the DELIVERED Sigma by at most 0.98 meV with no row over 2 meV, for
    # +8.3% sum K, +10.8% Kmax and +24 MB of W storage (job 58217047 Si arms
    # .8 vs xi15). Below the campaign's 2-5 meV accuracy scale and its 2x cost
    # scale, so the default stays 0.25 and 0.15 is recorded as the measured
    # alternative in algorithm_guide.md section 7.
    "line_growth_fraction": 0.25,
    "line_spacing_rule": ("2*eta to omega_fine=max(omega_p, E_g+active depth), "
                          "then step*(1+0.25) per interval; no absolute energy, "
                          "no material branch"),
    # top clears BOTH the plasmon and the frequencies Sigma actually samples.
    # 2.25*omega_fine: |W_c| has fallen to ~1/2.25^2 of its static scale, where
    # M1/M3 carry it.  1.25*(Sigma box extent + active occupied depth): the W
    # frequency Sigma needs at evaluation energy E is |E - E_n'|, so the top of
    # that window is the box edge plus the depth of the screening manifold.
    "plasma_top_factor": 2.25,
    "sigma_window_top_factor": 1.25,
    # -W_c(i u) is in its 1/u^2 tail by ~2 omega_p, so the LAST node belongs
    # just past that, not at it.  2.5 is within 6% of the only measured-good
    # u_max (Na 16.0 eV = 2.65 omega_p; u_max at the line top, 1.58 omega_p,
    # was 2.9x worse) and changes neither reference deck's node count: the
    # Zolotarev count turns over at kappa = 16.096, and 2.5*omega_fine leaves
    # Na at 3 nodes (kappa 15.12) and Si at 4 (kappa 41.5).
    "imaginary_top_factor": 2.5,
    # The bank's remote Laplace cells expand about their lowest transition
    # edge, and that expansion has a convergence domain the SAMPLE PLAN must
    # respect: a support above it is refused by response_laplace_rule with an
    # instruction to repartition that the bank owner never acts on (ARECIPE,
    # Si P4, job 58217047.5/.6).  The ceiling is not a dial here -- it is read
    # from the bank's own order budget through minimax.response_remote_max_*,
    # so there is exactly one owner of the convergence math.
    "remote_domain_rule": "every emitted |z| <= minimax.response_remote_max_abs_z(lowest cell [delta_lo, delta_hi], h, bank domain pad)",
    "imaginary_count_epsilon": 1.0e-3,
    "imaginary_min_count": 2,
    "imaginary_count_rule": "max(2, round(log(16*(u_max/u_min)^2)*log(4000)/(2*pi^2)))",
    # One held line point per spacing law, at that law's own midpoint.
    "held_line_fine_fraction": 0.5,
    "held_line_rule": "midpoints nearest 0.5*omega_fine and sqrt(omega_fine*top)",
    "multiplet_relative_tolerance": 1.0e-6,
    "bank_rule_tolerance": 1.0e-8,
    "moment_convention": "S_m = 2 M_(2m+1); physical M1 and M3 only",
    "operator_realization": "little-group-reynolds-v1",
    "production": {"direction_cutoff": 1.0e-3, "imaginary_width_fraction": 0.25,
                   "infinity_width_fraction": 0.125, "sigma_tolerance": 1.0e-4},
    # A tier is COARSER DIALS, not a different shape: a linspace over a
    # relative top would step 7.4 eV across a Si-like plasmon.
    "relaxed": {"direction_cutoff": 1.0e-2, "imaginary_width_fraction": 0.125,
                "infinity_width_fraction": 0.0625, "sigma_tolerance": 1.0e-3,
                "line_step_eta_factor": 4.0, "line_growth_fraction": 0.5,
                "imaginary_count": 2},
}

# Each entry is (predicate description, threshold); the public table adds name
# and version. Composite checks retain their individual dimensional thresholds.
_GATE_ROWS = {
    "normalized_gram_keep": ("retain gamma/gamma_max strictly above cut", 1.0e-8),
    "normalized_gram_validity": ("gamma_min/gamma_max >= threshold", -1.0e-7),
    "zero_ritz_policy": ("drop lambda <= cutoff only within factor-weight budget",
                         {"lambda_cutoff_ry2": 1.0e-6, "max_dropped_weight_fraction": 1.0e-6}),
    # Legacy C denotes b: preserve this hashed predicate for stored identities.
    "finite_factors_poles": ("finite complex128 C; finite positive float64 active poles2; int64 K; exact-zero inactive C and positive sentinel", True),
    "passivity": ("V-whitened -Wc(i eta) spectrum in bounds and relative anti-Hermitian part within tolerance",
                  {"eigenvalue_min": -1.0e-10, "eigenvalue_max": 1.0 + 1.0e-8,
                   "antihermitian_relative_max": 1.0e-10}),
    "retained_subspace_moments": ("relative M1/M3 identity defect in retained Ritz infinity states P_R x_inf, with P_R Gram-metric orthogonal on span(OZ), after cut and zero policy <= threshold; original q_inf defect is diagnostic", 1.0e-10),
    "held_w": ("held W value/derivative relative defects with coordinates and receipt paths; diagnostic, no universal threshold", None),
    "model_reciprocity": ("at held W/dW samples that are transpose symmetric, the evaluated model preserves transpose symmetry; generic complex Hermitian residues are not required to be real",
                          {"reference_relative_max": 1.0e-12, "model_relative_max": 1.0e-10}),
    "full_m1_defect": ("maximum over q of relative full M1 defect after cut and zero policy; PASS within diagnostic band, WARN outside, never refuse", 2.0e-4),
    "full_m3_defect": ("maximum over q of relative full M3 defect after cut and zero policy; PASS within diagnostic band, WARN outside, never refuse", 2.0e-3),
    "representation": ("scalar N_spinor=1 and authenticated TRS allowed", {"nspinor": 1, "trs_allowed": True}),
    "capacity": ("aggregate live device bytes per rank of new shared-pole objects including workspace <= threshold * U", 3.0),
    "stream_peak": ("inherited response stream peak <= threshold * incumbent MPA stream peak on the same deck and processor geometry, using the same measurement method", 1.05),
    "sigma_peak": ("inherited Sigma peak including one incumbent-shaped W <= threshold * incumbent MPA Sigma peak on the same deck, processor geometry and window plan, using the same measurement method", 1.05),
    "rule_validity": ("bank and Sigma certificates cover current domains at resolved tolerances", True),
    "sc_rebuild": ("physical samples, directions, poles and ranks rebuilt at current bands and occupations; reused quadrature certified for current domains", True),
}
shared_real_pole_gates_v1_r3b = {
    name: {"name": name, "predicate": predicate, "threshold": threshold,
           "version": GATE_VERSION}
    for name, (predicate, threshold) in _GATE_ROWS.items()
}

# The SC quadrature contract changed independently of all numerical gates.
shared_real_pole_gates_v1_r3b["sc_rebuild"]["version"] = "sc_quadrature_recertification_20260910"


for _name, _range in (("full_m1_defect", (2.2e-6, 1.9e-5)),
                      ("full_m3_defect", (3.6e-5, 1.7e-4))):
    shared_real_pole_gates_v1_r3b[_name].update({
        "version": "cd8_58061895.50", "diagnostic": True,
        "calibration_range": _range,
        "source": "CD8 construction 58061895.50; coordinator ruling5, 2026-09-09",
    })


def table_hash(table):
    """SHA256 of a small JSON table with canonical ordering and finite numbers."""
    return hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


RECIPE_HASH = table_hash(shared_real_pole_v2_r1)
GATE_HASH = table_hash(shared_real_pole_gates_v1_r3b)


def gate_receipt(name, value=None, *, passed=None, reason):
    """Record a consumer-measured gate; missing data can never produce PASS.

    Parameters
    ----------
    name : str
        Key in the canonical gate table.
    value : JSON-compatible scalar, list or dict, optional
        Small measured summaries only; no matrices. Nonfinite numbers refuse.
    passed : bool or None
        Consumer's evaluation of the named predicate, or unmeasured/diagnostic.
    reason : str
        Measurement scope/evidence, or the concrete missing dependency.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("shared-pole receipt requires a nonempty reason")
    if passed is not None and type(passed) is not bool:
        raise TypeError("shared-pole receipt passed must be bool or None")
    json.dumps(value, allow_nan=False)
    def missing(item):
        if isinstance(item, dict):
            return not item or any(missing(v) for v in item.values())
        if isinstance(item, (list, tuple)):
            return not item or any(missing(v) for v in item)
        return item is None
    row = shared_real_pole_gates_v1_r3b[name]
    status = "NOT_MEASURED" if missing(value) or passed is None else (
        "PASS" if passed else ("WARN" if row.get("diagnostic") else "FAIL"))
    return {"predicate": row["predicate"], "name": name, "version": row["version"],
            "value": value, "threshold": row["threshold"], "status": status,
            "reason": reason}


class CapacityLedger:
    """Plan-time aggregate capacity for one SC map (DESIGN §§0,4).

    Parameters
    ----------
    meta : Meta
        Logical ``nk_tot``, ``nspinor`` and ``n_rmu`` from the current deck.
    mesh_xy : Mesh
        Named processor axes x/y. U=16*Q*(spin*mu)^2/(Px*Py) bytes/rank.
    device_budget_bytes : int, optional
        Resolved per-device budget in bytes. Production supplies the deck
        budget; standalone synthetic callers default conservatively to 3U.

    Each reservation owns disjoint device resident/workspace bytes computed by its
    caller for its ACTUAL batch sizes, including packing and native workspace.
    ``concurrent_with`` names earlier reservations simultaneously live with it;
    each named footprint is counted once. Historical concurrency is not carried
    forward: callers name all allocations live in the current phase.
    Sequential stages omit predecessors. Stage names must be unique (include
    batch/phase identifiers when necessary). A refusal is recorded but does not
    create a usable reservation. The ledger owns no arrays or memory allocator.
    Host I/O staging is reported separately by the store, which refuses a host
    copy larger than its device panel (coordinator ruling 11).
    """

    def __init__(self, meta, *, mesh_xy, device_budget_bytes=None):
        import operator
        geometry = dict(nq=meta.nk_tot, nspinor=meta.nspinor, nmu=meta.n_rmu,
                        px=mesh_xy.shape['x'], py=mesh_xy.shape['y'])
        self.geometry = {}
        for key, value in geometry.items():
            if isinstance(value, bool):
                raise ValueError(f"capacity geometry {key} must be a positive integer")
            value = operator.index(value)
            if value <= 0:
                raise ValueError(f"capacity geometry {key} must be a positive integer")
            self.geometry[key] = value
        g = self.geometry
        self._global_unit_bytes = 16 * g['nq'] * (g['nspinor'] * g['nmu'])**2
        self.U_bytes_per_rank = self._global_unit_bytes / (g['px'] * g['py'])
        self.limit_bytes_per_rank = (shared_real_pole_gates_v1_r3b['capacity']['threshold']
                                     * self.U_bytes_per_rank)
        # Production supplies the resolved deck budget. Standalone synthetic
        # callers retain their conservative 3U budget until they supply one.
        self.device_budget_bytes_per_rank = (
            int(self.limit_bytes_per_rank) if device_budget_bytes is None
            else self._bytes(device_budget_bytes))
        if self.device_budget_bytes_per_rank <= 0:
            raise ValueError("capacity device budget must be positive")
        self.entries = []
        self._accepted = {}
        self._live_stages = None
        self.measured_peak = gate_receipt('capacity', reason='measured peak not supplied')
        self.stream_peak = gate_receipt('stream_peak', reason='same-deck incumbent comparison not supplied')
        self.sigma_peak = gate_receipt('sigma_peak', reason='same-deck/window incumbent comparison not supplied')

    @property
    def live_stages(self):
        """Caller-bound ambient reservations for callees without lifetime args.

        The caller sets an explicit tuple before I/O; () means no upstream
        allocations are live. Unbound is unknown and refuses. Callees pass the
        tuple as ``concurrent_with`` to reserve their own disjoint footprint.
        """
        if self._live_stages is None:
            raise ValueError("GATE shared_pole_capacity_lifetimes: got: unbound caller lifetimes; want: explicitly set ledger.live_stages before callee admission; why: unknown upstream bytes are not zero")
        return self._live_stages

    @live_stages.setter
    def live_stages(self, stages):
        if isinstance(stages, str):
            raise ValueError("live_stages must contain accepted stage names, not a string")
        names = tuple(dict.fromkeys(stages))
        for name in names:
            if name not in self._accepted:
                raise ValueError(f"capacity live stage {name!r} has no accepted reservation")
        self._live_stages = names

    @staticmethod
    def _bytes(value):
        import operator
        if isinstance(value, bool):
            raise ValueError("capacity bytes must be nonnegative integers")
        try:
            result = operator.index(value)
        except TypeError as exc:
            raise ValueError("capacity bytes must be nonnegative integers") from exc
        if result < 0:
            raise ValueError("capacity bytes must be nonnegative integers")
        return result

    def reserve(self, stage, *, resident_bytes_per_rank,
                workspace_bytes_per_rank, concurrent_with=()):
        """Admit actual-batch bytes before allocation, or record FAIL and refuse.

        Returns a detached JSON row. ``concurrent_with`` is an iterable of
        accepted stage names, not their byte totals. Caller-live allocations
        must be charged here OR in a named concurrent reservation, never both.
        No runtime peak is inferred from a successful analytical admission.
        """
        import copy
        if not isinstance(stage, str) or not stage.strip() or stage in self._accepted:
            raise ValueError(f"capacity stage must be a new nonempty name; got {stage!r}")
        if isinstance(concurrent_with, str):
            raise ValueError("concurrent_with must contain stage names, not a string")
        live = set()
        for name in concurrent_with:
            if name not in self._accepted:
                raise ValueError(f"capacity concurrent stage {name!r} has no accepted reservation")
            live.add(name)
        resident = self._bytes(resident_bytes_per_rank)
        workspace = self._bytes(workspace_bytes_per_rank)
        total = resident + workspace + sum(
            self._accepted[name]['resident_bytes_per_rank']
            + self._accepted[name]['workspace_bytes_per_rank'] for name in live)
        scaling_passed = total <= self.limit_bytes_per_rank
        # Stream and Sigma are sequential inherited phases, not simultaneous
        # allocations. Retain the larger recorded peak conservatively.
        inherited = max((row.get('value', {}).get('shared_bytes_per_rank') or 0
                         for row in (self.stream_peak, self.sigma_peak)
                         if isinstance(row.get('value'), dict)), default=0)
        available = max(0, self.device_budget_bytes_per_rank - inherited)
        passed = total <= available
        g = self.geometry
        max_ranks = math.floor(self.limit_bytes_per_rank * g['px'] * g['py'] / total) if total else None
        reason = ('actual-batch analytical admission; peak not measured' if passed else
                  f"aggregate {total} B/rank exceeds available device budget {available} B/rank; "
                  f"geometry Q={g['nq']}, spin={g['nspinor']}, mu={g['nmu']}, "
                  f"Px={g['px']}, Py={g['py']}; at these fixed reservation bytes "
                  f"want Px*Py <= {max_ranks}; reprice actual batches/workspaces "
                  "for any changed geometry, or reduce concurrent live bytes")
        row = gate_receipt('capacity', total / self.U_bytes_per_rank,
                           passed=scaling_passed, reason=reason)
        row['status'] = 'FAIL' if not passed else ('PASS' if scaling_passed else 'WARN')
        if passed and not scaling_passed:
            row['reason'] = 'above 3U scaling target; admitted within device budget (coordinator ruling24); peak not measured'
        row.update(stage=stage, resident_bytes_per_rank=resident,
                   workspace_bytes_per_rank=workspace,
                   aggregate_bytes_per_rank=total,
                   limit_bytes_per_rank=self.limit_bytes_per_rank,
                   device_budget_bytes_per_rank=self.device_budget_bytes_per_rank,
                   inherited_peak_bytes_per_rank=inherited,
                   available_device_bytes_per_rank=available,
                   device_budget_status='PASS' if passed else 'FAIL',
                   concurrent_with=sorted(live), live_stages=sorted(live | {stage}),
                   geometry=dict(g), max_mesh_ranks_at_fixed_bytes=max_ranks)
        self.entries.append(row)
        if not passed:
            raise MemoryError(f"GATE shared_pole_capacity: stage={stage}; got: {reason}; "
                              "why: aggregate live allocation must not exceed the remaining device budget")
        self._accepted[stage] = row
        return copy.deepcopy(row)

    def record_measured_peak(self, bytes_per_rank, *, reason):
        """Record the new-object maximum over ranks, excluding inherited stream."""
        peak = self._bytes(bytes_per_rank)
        if peak < self.measured_peak.get('bytes_per_rank', 0):
            return self.receipt()['measured_peak']
        self.measured_peak = gate_receipt(
            'capacity', peak / self.U_bytes_per_rank,
            passed=peak <= self.limit_bytes_per_rank, reason=reason)
        self.measured_peak['bytes_per_rank'] = peak
        return self.receipt()['measured_peak']

    def record_stream_peak(self, shared_bytes_per_rank, incumbent_bytes_per_rank, *, reason):
        """Record the inherited stream comparison (coordinator ruling 9).

        Both byte counts must use the same deck, mesh and measurement method;
        ``reason`` names that scope and both evidence paths/job.steps. Missing
        counts stay NOT_MEASURED. The inherited stream is not a reservation and
        cannot be named in ``concurrent_with``. New bank outputs/batches still
        enter ``reserve('bank_outputs', ...)`` and obey 3U.
        """
        return self._record_inherited_peak('stream_peak', shared_bytes_per_rank,
                                           incumbent_bytes_per_rank, reason=reason)

    def record_sigma_peak(self, shared_bytes_per_rank, incumbent_bytes_per_rank, *, reason):
        """Record matched inherited Sigma footprint (coordinator ruling 12).

        ``reason`` names the same deck, mesh, window plan and compile-only
        measurement method with both evidence paths/job.steps. One W replacing
        the incumbent W is inherited. Faces, weights, reader/routed panels,
        unfold scratch and any simultaneous second W remain new reservations.
        """
        return self._record_inherited_peak('sigma_peak', shared_bytes_per_rank,
                                           incumbent_bytes_per_rank, reason=reason)

    def _record_inherited_peak(self, name, shared_bytes_per_rank, incumbent_bytes_per_rank, *, reason):
        if getattr(self, name)['status'] != 'NOT_MEASURED':
            raise ValueError(f"inherited {name} comparison already recorded for this map")
        shared = None if shared_bytes_per_rank is None else self._bytes(shared_bytes_per_rank)
        incumbent = None if incumbent_bytes_per_rank is None else self._bytes(incumbent_bytes_per_rank)
        if incumbent == 0:
            raise ValueError("incumbent peak must be positive")
        threshold = shared_real_pole_gates_v1_r3b[name]['threshold']
        passed = None if shared is None or incumbent is None else shared <= threshold * incumbent
        row = gate_receipt(
            name, {'shared_bytes_per_rank': shared,
                   'incumbent_bytes_per_rank': incumbent},
            passed=passed, reason=reason)
        row.update(stage=name, geometry=dict(self.geometry))
        setattr(self, name, row)
        if passed is False:
            raise MemoryError(f"GATE shared_pole_{name}: shared={shared} B/rank; "
                              f"incumbent={incumbent} B/rank; limit={threshold} * incumbent; "
                              f"geometry={self.geometry}; why: inherited {name} regressed")
        return self.receipt()[name]

    def receipt(self, *, entry_start=None):
        """Snapshot stage rows and peaks, optionally as an indexed ledger segment.

        A segment retains absolute entry indices so ordered constructor receipts
        reconstruct the prefix without duplicating every earlier reservation.
        The default remains the complete ledger snapshot.
        """
        import operator
        if entry_start is not None:
            if isinstance(entry_start, bool):
                raise ValueError("capacity entry_start must be an integer index")
            entry_start = operator.index(entry_start)
            if not 0 <= entry_start <= len(self.entries):
                raise ValueError("capacity entry_start lies outside the ledger")
        import copy
        snapshot = copy.deepcopy(dict(geometry=self.geometry,
                                  U_bytes_per_rank=self.U_bytes_per_rank,
                                  limit_bytes_per_rank=self.limit_bytes_per_rank,
                                  device_budget_bytes_per_rank=self.device_budget_bytes_per_rank,
                                  entries=self.entries[entry_start or 0:], live_stages=self._live_stages,
                                  measured_peak=self.measured_peak,
                                  stream_peak=self.stream_peak,
                                  sigma_peak=self.sigma_peak))
        if entry_start is not None:
            snapshot["entry_span"] = [entry_start, len(self.entries)]
        return snapshot


def construction_receipt(measurements=None, *, capacity=None, capacity_entry_start=None):
    """Complete receipt skeleton, explicitly marking every absent gate unmeasured.

    ``measurements`` maps names to ``gate_receipt`` keyword dictionaries. Dense
    consumers supply measurements; creating this skeleton certifies no physics.
    """
    measurements = {} if measurements is None else measurements
    unknown = set(measurements) - set(shared_real_pole_gates_v1_r3b)
    if unknown:
        raise ValueError(f"unknown shared-pole receipt predicates: {sorted(unknown)}")
    result = {"schema": RECEIPT_SCHEMA, "recipe_version": RECIPE_VERSION,
            "recipe_hash": RECIPE_HASH, "gate_version": GATE_VERSION,
            "gate_hash": GATE_HASH,
            "gates": [gate_receipt(name, **measurements.get(
                name, {"reason": "measurement not supplied"}))
                for name in shared_real_pole_gates_v1_r3b]}
    if capacity is not None:
        if not isinstance(capacity, CapacityLedger):
            raise TypeError("construction receipt capacity must be the map's CapacityLedger")
        result['capacity'] = capacity.receipt(entry_start=capacity_entry_start)
        result['gates'] = [result['capacity'][r['name']] if r['name'] in ('stream_peak', 'sigma_peak')
                           else r for r in result['gates']]
        # The verdict always covers the full prefix, even for a compact segment.
        rows = capacity.entries
        measured = result['capacity']['measured_peak']
        values = [r['value'] for r in rows]
        if measured['value'] is not None:
            values.append(measured['value'])
        aggregate = max(values, default=None)
        result['gates'] = [r for r in result['gates'] if r['name'] != 'capacity']
        result['gates'].append(gate_receipt(
            'capacity', aggregate, passed=(all(r['status'] == 'PASS' for r in rows)
                                           and measured['status'] != 'FAIL'),
            reason='plan-time ledger rows; independent measured peak recorded separately'))
        if (result['gates'][-1]['status'] == 'FAIL'
                and all(r['status'] != 'FAIL' for r in rows)
                and measured['status'] != 'FAIL'):
            result['gates'][-1]['status'] = 'WARN'
            result['gates'][-1]['reason'] = '3U scaling target exceeded; device-budget admissions passed (coordinator ruling24)'
    return result


def bind_shared_pole_sc_identity(meta, state, *, occupation_state, print_fn):
    """Label current SC scratch, without claiming QP provenance (ruling 22).

    ``state`` supplies the current iteration. Reuse the carried occupation
    digest or the already-bound insulating census digest; compute no new hash.
    These labels MUST NOT authenticate restart membership or skip construction.
    """
    import operator

    iteration = operator.index(state.iteration)
    if isinstance(state.iteration, bool) or iteration < 0:
        raise ValueError("GATE shared_pole_sc_identity: iteration must be a nonnegative integer")
    recipe = meta.shared_pole_recipe
    occ_hash = (getattr(occupation_state, 'occ_hash', None)
                if occupation_state is not None else
                meta.shared_pole_census.get('occupation_sha256'))
    if not isinstance(occ_hash, str) or not occ_hash:
        raise ValueError("GATE shared_pole_sc_identity: current occupation label is missing")
    identity = dict(hamiltonian=f'sc_map_{iteration}:{occ_hash}',
                    wavefunctions='qp_rotation_unreceipted',
                    recipe_hash=recipe['recipe_hash'], gate_hash=recipe['gate_hash'],
                    authentication='NON-AUTHENTICATING')
    meta.shared_pole_state_identity = identity
    print_fn(f"shared-pole SC identity NON-AUTHENTICATING: {identity}; "
             "scratch only; rebuild every map; no restart reuse or publication")
    return dict(identity)


def shared_pole_restart_handle(restart_path, *, expected_identity, meta,
                               mesh_xy, print_fn):
    """Authenticate one current-map restart member through the store owner.

    Parameters
    ----------
    restart_path : path-like
        Existing committed ISDF bundle; all ranks call this on compute nodes.
    expected_identity : dict
        Current identity supplied by the screening owner, never rebuilt here.
    meta : Meta
        Current resolved recipe and capacity ledger with bound live_stages.
    mesh_xy : Mesh
        Current named processor mesh passed unchanged to the store validator.
    print_fn : callable
        Existing rank-safe startup/progress printer.

    Returns
    -------
    dict or None
        Small one-shot path/identity/digest/K handle, or None ONLY for a typed missing
        member in a committed bundle. Stale/corrupt/partial members refuse.
        The authenticated header comes from the same payload validation.
    """
    from pathlib import Path

    if (str(expected_identity.get('iteration_id', '')).startswith('sc_')
            or expected_identity.get('wavefunctions') == 'qp_rotation_unreceipted'
            or expected_identity.get('authentication') == 'NON-AUTHENTICATING'
            or str(expected_identity.get('hamiltonian', '')).startswith('sc_map_')):
        raise ValueError("GATE shared_pole_sc_restart: SC models are scratch-only and "
                         "NON-AUTHENTICATING; rebuild A/B/C on every map")
    from file_io.tagged_arrays import (
        read_shared_pole_restart_member, SharedPoleMemberMissing,
        SharedPoleMemberRefused,
    )

    recipe = getattr(meta, 'shared_pole_recipe', None)
    capacity = getattr(meta, 'shared_pole_capacity', None)
    keys = ('recipe_version', 'recipe_hash', 'gate_version', 'gate_hash',
            'accuracy', 'eta_ev', 'n')
    if not isinstance(recipe, dict) or any(recipe.get(key) is None for key in keys):
        raise ValueError("GATE shared_pole_restart: current resolved recipe is missing")
    if not isinstance(capacity, CapacityLedger):
        raise ValueError("GATE shared_pole_restart: current map capacity ledger is missing")
    capacity.live_stages  # Unknown upstream residency is not zero.
    try:
        member, header = read_shared_pole_restart_member(
            restart_path, expected_identity=expected_identity, mesh_xy=mesh_xy,
            capacity=capacity, return_header=True)
    except SharedPoleMemberMissing as exc:
        print_fn(f"shared-pole restart: {exc}; build and register a new member")
        return None
    stored = header.get('recipe', {})
    # Table hashes bind the deterministic recipe; state identity binds its
    # energy/occupation/WFN/centroid inputs. Tier, eta and n distinguish the
    # physical choices without binding mesh-dependent planning bytes into W.
    for key in keys:
        if stored.get(key) != recipe[key]:
            raise SharedPoleMemberRefused(
                f"GATE shared_pole_restart: recipe {key} mismatch; "
                f"got {stored.get(key)!r}, want {recipe[key]!r}")
    path = (Path(restart_path).resolve().parent / member['path']).resolve()
    print_fn(f"shared-pole restart: authenticated {path}; digest={member['digest']}; "
             "skip bank, moments and construction")
    return dict(path=str(path), identity=dict(header['identity']),
                digest=member['digest'], K=list(header['K']))


# A resolvable ladder never reaches this; a census or deck error does.
_LINE_COUNT_CEILING = 4096


def _sigma_extent_ev(config):
    """The largest |omega| the Sigma grid evaluates, in eV.

    Patches, when set, ARE the grid, so they replace the contiguous box rather
    than extending it; `parsed_omega_patches_ev` stays the single owner of that
    spelling. Out-of-grid states are clamped to an endpoint by the Sigma
    consumer, so this extent bounds every frequency Sigma asks W for.
    """
    patches = config.sigma.parsed_omega_patches_ev()
    edges = ([e for patch in patches for e in patch] if patches
             else [config.sigma.omega_min_ev, config.sigma.omega_max_ev])
    return max(abs(float(e)) for e in edges)


def _remote_domain_cap(wfns, meta, census, eta_ry, rel_tol, domain_pad_ry=0.0,
                       session=None):
    """Largest sample |z| in eV the bank's remote cells admit, and that edge.

    ONE PARTITIONER. ``response_windows`` is the bank's own near/remote split
    and needs only bands, occupations and mu -- exactly what this resolver
    already has -- so it is called, not reimplemented; ``minimax`` owns the
    convergence predicate and answers the radius. Returns ``(None, None)``
    when the deck has no remote cell, in which case no sample can be refused.

    THE PUBLISHED DOMAIN IS THE RULE'S OWN.  ``response_remote_max_abs_z``
    evaluates ``response_laplace_rule``'s full three-stage acceptance at the
    radius it returns, through the rule's own padding, so the number here is
    admissible by construction rather than a model of one stage of it.
    Under self-consistency the bank hands ``response_laplace_rule`` a
    ``domain_pad_ry`` (``response_bank.RESPONSE_DOMAIN_PAD_RY``) so a rule
    reused across maps stays valid while the spectrum drifts, and the rule
    then expands about ``max(lo - pad, lo/2)`` instead of ``lo``.  A lower
    edge means a LARGER geometric ratio at the same ``|z|``, so a cap taken
    on the unpadded edge admits samples the padded expansion cannot certify:
    the first SC map refuses with 'remote Taylor order budget exceeded' on a
    support the one-shot cap was happy with (measured on run 49's Si SC deck,
    AREBASE 2026-09-12).  Capping on the padded edge is self-consistent --
    every emitted ``|z|`` then lies inside the padded domain, which is
    exactly the predicate the rule uses to decide the pad applies at all.
    One-shot passes ``domain_pad_ry = 0`` and is unchanged.
    """
    from common.units import RYD_TO_EV
    from minimax import response_remote_cap_receipt
    from .response_bank import response_weights, response_windows
    energy, f, u, _reference, _receipt = response_weights(wfns, meta)
    _masks, _ft, _ut, cells, _window_receipt = response_windows(
        energy, f, u, chemical_potential_ry=census['mu_ry'])
    if not cells:
        return None, None, None, None
    # The binding cell is the one with the lowest edge; its own upper edge
    # goes with it, because the rule's NNLS rows are fitted on [lo, hi] and
    # the admissible radius depends on both.
    cell = min(cells, key=lambda c: float(c['delta_min_ry']))
    delta_lo, delta_hi = float(cell['delta_min_ry']), float(cell['delta_max_ry'])
    # The SERVICE applies the pad and answers with a radius its own rule
    # accepts; nothing here models the expansion.  That answer is FOUND, by
    # evaluating the rule -- minutes on a wide cell -- so a self-consistent
    # loop retains it in the support session it already uses for every other
    # scalar support bound, keyed on the geometry the answer depends on.  A
    # map whose cell edges move gets a fresh answer; one whose edges do not
    # pays once.
    key = ("remote_cap", round(delta_lo, 12), round(delta_hi, 12),
           round(float(eta_ry), 12), float(rel_tol), round(float(domain_pad_ry), 12))
    if session is not None and key in session:
        receipt = session[key]
    else:
        receipt = response_remote_cap_receipt(
            delta_lo, delta_hi, eta_ry, rel_tol,
            domain_pad_ry=float(domain_pad_ry))
        if session is not None:
            session[key] = receipt
    cap = receipt["cap_ry"]
    effective_lo = max(delta_lo - float(domain_pad_ry), delta_lo / 2.0)
    return cap * RYD_TO_EV, effective_lo * RYD_TO_EV, delta_hi, receipt


def _support_report(r, tier, census):
    """The human-readable resolved support geometry, for the run log.

    This is the user-facing surface of the recipe: it states which electrons
    were judged to screen, where the structure ends, how far Sigma reaches, and
    which of the two floors set the top support -- so a user looking at a long
    bank can see WHY it is long, and a user on an unfamiliar material can see
    what the recipe decided for it. The full key/rule dump follows it.
    """
    from common.units import RYD_TO_EV
    recipe = shared_real_pole_v2_r1
    plasmon_term = recipe['plasma_top_factor'] * r['omega_fine_ev']
    window_term = recipe['sigma_window_top_factor'] * r['sigma_window_ev']
    bound = ('the PLASMON floor' if r['top_bound_by'] == 'plasmon'
             else 'the SIGMA WINDOW')
    lines = [
        "",
        "  ==========================================================",
        f"  SHARED-POLE SUPPORT ({RECIPE_VERSION}, {tier})",
        f"  Screening electrons : {census['active_electrons']:.3f} in "
        f"{len(census['active_bands'])} active band(s), omega_p = {r['plasma_ev']:.3f} eV",
        f"                        active set = depth <= "
        f"{census['active_threshold_ev']:.3f} eV (fixed point); deepest active "
        f"{r['active_depth_ev']:.3f} eV",
        f"  Structure scale     : omega_fine = {r['omega_fine_ev']:.3f} eV = "
        f"max(omega_p {r['plasma_ev']:.3f}, E_g {census['gap_ev']:.3f} + depth "
        f"{r['active_depth_ev']:.3f})",
        f"  Sigma asks W up to  : {r['sigma_window_ev']:.3f} eV = grid extent "
        f"{r['sigma_extent_ev']:.3f} + active depth {r['active_depth_ev']:.3f}",
        f"  Top support         : {r['top_ev']:.3f} eV, set by {bound}",
        f"                        max(2.25*omega_fine = {plasmon_term:.3f}, "
        f"1.25*Sigma window = {window_term:.3f})"
        + ("" if r.get('remote_cap_ev') is None else
           f", capped at {r['remote_cap_ev']:.3f}"),
        f"  Line supports       : {r['line_count']} -- step {r['line_step_ev']:.3f} eV "
        f"(2*eta) to {r['omega_fine_ev']:.3f} eV, then x"
        f"{1.0 + r['line_growth_fraction']:.2f} per step to {r['top_ev']:.3f} eV",
        f"  Imaginary supports  : {r['imaginary_count']} -- {r['u_min_ev']:.3f} to "
        f"{r['u_max_ev']:.3f} eV, log-spaced (kappa = {r['kappa']:.1f}), "
        f"u_max set by the {r.get('u_max_bound_by', 'zolotarev')} rule"
        + ("" if r.get('u_max_bound_by') != 'remote_cap' else
           f" (Zolotarev would have asked {r['u_max_uncapped_ev']:.3f})"),
        f"  Sample height       : {r['height_ev']:.3f} eV = 4*eta",
        ("  Bank remote domain  : |z| <= %.3f eV, from the lowest remote cell "
         "edge %.3f eV%s" % (
             r['remote_cap_ev'], r['remote_delta_lo_ev'],
             "" if not r.get('remote_domain_pad_ev') else
             " (already lowered by the SC rule session's %.3f eV pad, which is "
             "the edge the bank's own expansion will use)"
             % r['remote_domain_pad_ev'])
         + ("" if not r.get('remote_cap_search') else
            "; found by %d rule evaluation(s) in %.1f s, to %.0f%% of the "
            "orientation-free radius %.3f eV"
            % (r['remote_cap_search']['evaluations'],
               r['remote_cap_search']['seconds'],
               100.0*r['remote_cap_search']['relative_precision'],
               r['remote_cap_search']['stage_one_ry']*RYD_TO_EV))
         if r.get('remote_cap_ev') is not None else
         "  Bank remote domain  : no remote cell; no sample ceiling"),
        "  Bank cost grows with the top support and as 1/height: widening",
        "  sigma_omega_min_ev/max_ev, or adding a sigma_omega_patches_ev",
        "  window over a semicore state, raises the top support with it.",
    ]
    envelope = r.get('support_envelope')
    if envelope is not None and envelope['status'] != 'initial_reference':
        lines.append(f"  SC enclosure {envelope['status']} at epoch {envelope['epoch']}: "
                     "the retained bounds above may exceed this map's own.")
    lines.append("  ==========================================================")
    return "\n".join(lines)


def _plasma_ev(electrons, volume_bohr3):
    """omega_p = 2 sqrt(4 pi n_e) Ry in eV, for a charge in one cell.

    The single owner of the plasma equation: the census uses it to decide which
    bands screen and the resolver uses it to place the supports, so the two can
    never disagree about what omega_p means.
    """
    from common.units import RYD_TO_EV
    return 2.0 * math.sqrt(4.0 * math.pi * float(electrons)
                           / float(volume_bohr3)) * RYD_TO_EV


def bind_shared_pole_census(wfns, meta, *, occupation_state, trs_allowed, state_capacity, kweights):
    """Bind the current physical charge census to the existing metadata bundle.

    Parameters
    ----------
    wfns : Wavefunctions
        Replicated energies (nk_full, nb_carrier), Ry, and normalized occupations;
        only ``slices.b4_logical`` bands enter the census. No psi is read.
    meta : Meta
        Logical centroids, spin counts, full k count and cell volume in bohr^3.
    occupation_state : OccupationState or None
        The authoritative current metallic state; None uses the insulating step
        occupations and logical VBM/CBM midpoint. No chemical potential is solved.
    state_capacity : float
        WfnLoader.occupation_state_capacity, the canonical spin normalization.
    kweights : array_like
        Normalized physical full-BZ weights [nk_full] from the existing weight owner.
    trs_allowed : bool
        Authenticated result from the symmetry service, never a guessed default.

    Notes
    -----
    The plasma equation is omega_p = sqrt(4*pi*n_e) Ha, with n_e obtained from
    the capacity-weighted occupations of the SCREENING-ACTIVE bands: the fixed
    point of ``depth <= active_plasma_factor * omega_p(set)``, grown greedily
    from the shallowest occupied band. No absolute depth is involved, so a
    semicore manifold is excluded because it is deep relative to the plasmon
    its own charge would produce, and included when it is not.
    All occupations of each active band are retained, including tails above mu. The bundle
    uses the supplied authenticated full-BZ quadrature weights. This census
    must be rebound at every SC map, after that map's occupation solve.
    """
    import numpy as np
    from common.units import RYD_TO_EV

    stop = wfns.slices.b4_logical - wfns.slices.b0
    energies = np.asarray(wfns.enk, dtype=np.float64)[:, :stop]
    occupations = np.asarray(wfns.occ if occupation_state is None else
                             occupation_state.f_kn, dtype=np.float64)[:, :stop]
    if (energies.shape != occupations.shape or energies.ndim != 2
            or energies.shape[0] != meta.nk_tot or not energies.size
            or not np.all(np.isfinite(energies))
            or not np.all(np.isfinite(occupations))):
        raise ValueError("GATE shared_pole_census: got: inconsistent/nonfinite energy or occupation table; want: current full-BZ logical tables; why: physical charge needs authenticated weights")
    if int(meta.nspinor) != 1 or int(meta.nspin) != 1 or not trs_allowed:
        raise ValueError("GATE shared_pole_representation: got: non-scalar or TRS-broken census; want: nspin=nspinor=1 and TRS allowed; why: both-endpoint spin action is not yet supported")
    capacity = float(state_capacity)
    if capacity != 2.0:
        raise ValueError("GATE shared_pole_census: got: scalar state capacity other than 2; want: authenticated spin-restricted scalar capacity; why: charge normalization")
    val = energies[:, wfns.slices.val]
    cond = energies[:, wfns.slices.cond_all_logical]
    if not val.size or not cond.size:
        raise ValueError("GATE shared_pole_gap: got: empty logical valence/conduction window; want: both nonempty; why: support geometry needs a physical gap")
    vbm, cbm = float(np.max(val)), float(np.min(cond))
    mu = (0.5 * (vbm + cbm) if occupation_state is None
          else float(occupation_state.mu_ry))
    gap_ev = max(0.0, (cbm - vbm) * RYD_TO_EV)
    partial = (occupations > 0.0) & (occupations < 1.0)
    weights = np.asarray(kweights, dtype=np.float64)
    if (weights.shape != (meta.nk_tot,) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("GATE shared_pole_kweights: got: invalid physical weights; want: finite nonnegative full-BZ weights summing to one; why: plasma charge normalization")
    # A partially occupied BAND must cross mu across the k census; smearing
    # tails in a gapped band do not authorize metallic line spacing.
    crossing = (np.min(energies, axis=0) <= mu) & (np.max(energies, axis=0) >= mu)
    partial_at_mu = bool(np.any(np.any(partial, axis=0) & crossing))
    depth = (mu - np.max(energies, axis=0)) * RYD_TO_EV
    recipe = shared_real_pole_v2_r1
    volume = float(meta.cell_volume)
    # WHICH ELECTRONS SCREEN.  An electron participates in the collective mode
    # only if its binding energy is below that mode's frequency, so the active
    # set is the fixed point of `depth <= factor * omega_p(set)`.  Grown
    # greedily from the shallowest occupied band, inclusively (a band is tested
    # against the plasma frequency of the set that CONTAINS it), the ascent is
    # monotone in both depth and omega_p and terminates at the band list.  No
    # absolute depth: a semicore state is excluded because it is deep compared
    # with the plasmon its own inclusion would produce, not because of a
    # hard-coded eV.
    charge = capacity * (weights[:, None] * occupations).sum(axis=0)
    candidates = [b for b in np.argsort(depth, kind='stable') if charge[b] > 0.0]
    if not candidates or not (math.isfinite(volume) and volume > 0):
        raise ValueError("GATE shared_pole_plasma: got: nonpositive/nonfinite active charge or cell volume; want: positive finite electrons and bohr^3; why: omega_p requires positive density")
    active_idx, electrons = [candidates[0]], float(charge[candidates[0]])
    for b in candidates[1:]:
        trial = electrons + float(charge[b])
        if depth[b] > recipe['active_plasma_factor'] * _plasma_ev(trial, volume):
            break
        active_idx.append(int(b))
        electrons = trial
    threshold_ev = recipe['active_plasma_factor'] * _plasma_ev(electrons, volume)
    active = np.zeros(depth.shape, dtype=bool)
    active[np.asarray(active_idx, dtype=int)] = True
    borderline = (~active) & (charge > 0.0) & (
        depth <= recipe['borderline_plasma_factor'] * threshold_ev)
    active_depth_ev = float(mu * RYD_TO_EV - np.min(energies[:, active]) * RYD_TO_EV)
    if not (math.isfinite(mu) and math.isfinite(electrons) and electrons > 0
            and math.isfinite(active_depth_ev)):
        raise ValueError("GATE shared_pole_plasma: got: nonpositive/nonfinite active charge or cell volume; want: positive finite electrons and bohr^3; why: omega_p requires positive density")
    meta.shared_pole_census = {
        "mu_ry": mu, "gap_ev": gap_ev, "partial_at_mu": partial_at_mu,
        "energy_span_ry": float(energies.max()-energies.min()),
        "active_electrons": electrons, "cell_volume_bohr3": volume,
        "active_manifold_depth_ev": active_depth_ev,
        "active_threshold_ev": threshold_ev,
        "active_rule": "fixed point of band depth <= active_plasma_factor*omega_p(set)",
        "state_capacity": capacity, "k_weights": weights.tolist(),
        "k_weight_sum": float(weights.sum()), "k_weight_rule": "authenticated full-BZ quadrature weights",
        "occupation_source": ("logical insulating step" if occupation_state is None
                              else f"current {occupation_state.smearing_family} OccupationState"),
        "active_bands": (np.flatnonzero(active) + wfns.slices.b0).tolist(),
        "borderline_bands": (np.flatnonzero(borderline) + wfns.slices.b0).tolist(),
        "energy_sha256": hashlib.sha256(energies.tobytes()).hexdigest(),
        "occupation_sha256": hashlib.sha256(occupations.tobytes()).hexdigest(),
        "trs_allowed": bool(trs_allowed), "logical_band_count": stop,
    }


def _support_envelope(required, key, session):
    """Keep a run-local scalar enclosure; never retain samples or a census.

    The reference map uses its current interval without retaining it. The
    first interacting map seeds the enclosure: an initial DFT gap can be
    substantially smaller and select a different imaginary support count.
    Increasing L/u_min is conservative for the imaginary-count heuristic.
    This is a sampling-geometry enclosure, not an interpolation-error bound;
    the bank still certifies its current energies at every supplied frequency.
    """
    version = "sc_interacting_support_enclosure_20260911"
    scope = "sampling geometry only; interpolation accuracy not certified"
    if not session.get("reference_complete", False):
        session["reference_complete"] = True
        return dict(version=version, status="initial_reference", epoch=-1,
                    required=dict(required), retained=dict(required), scope=scope)
    previous = session.get("envelope")
    same_policy = previous is not None and session.get("key") == key
    envelope = dict(required)
    if same_policy:
        # omega_fine joins the enclosure because it is now a support BOUNDARY,
        # not only a reported scale: the uniform region ends there, so an
        # enclosure that did not retain it would regenerate a different ladder
        # whenever omega_p moved between SC maps.
        envelope = {
            "line_top_ev": max(previous["line_top_ev"], required["line_top_ev"]),
            "omega_fine_ev": max(previous["omega_fine_ev"], required["omega_fine_ev"]),
            "u_min_ev": min(previous["u_min_ev"], required["u_min_ev"]),
            "u_max_ev": max(previous["u_max_ev"], required["u_max_ev"]),
        }
    changed = not same_policy or envelope != previous
    status = ("initial" if previous is None else
              "policy_changed" if not same_policy else
              "expanded" if changed else "hit")
    epoch = session.get("epoch", -1) + int(changed)
    session.update(key=key, envelope=envelope, epoch=epoch)
    return dict(version=version, status=status,
                epoch=epoch, required=dict(required), retained=dict(envelope),
                scope=scope)


def resolve_shared_pole_recipe(config, wfns, meta, *, mesh_xy, print_fn,
                              support_session=None):
    """Resolve DESIGN §5 from current metadata into scalars and small arrays.

    ``bind_shared_pole_census`` must have consumed this map's occupation state.
    Returns a plain dict with equal-length ``z_ry`` complex128, ``role`` int8,
    ``distinct_id`` int64 and ``held`` bool arrays [role]. Repeated physical
    points share a distinct_id; fitting and held IDs are disjoint. Infinity's
    role code is reserved (moments need no bank evaluation). Conjugates are
    constructor states, never bank calls. ``support_pair`` int64 [role,2]
    binds held endpoints; [-1,-1] means not a held midpoint.
    ``support_session`` optionally retains only scalar support bounds and a
    policy/basis key across SC maps, after one unretained reference map.
    Enclosed current intervals regenerate the same points and roles, while
    the census and capacity ledger remain fresh.
    Expanding intervals enlarge the envelope; policy changes start a new one.
    All ranks execute the metadata work; only ``print_fn`` may filter by rank.
    """
    import numpy as np
    from common.units import RYD_TO_EV

    if config.sigma.w_model != "shared_pole":
        return None
    census = getattr(meta, "shared_pole_census", None)
    if census is None:
        raise ValueError("GATE shared_pole_census: got: absent current census; want: bind_shared_pole_census after current occupations; why: no guessed plasma charge or frozen recipe")
    stop = wfns.slices.b4_logical - wfns.slices.b0
    energies = np.asarray(wfns.enk, dtype=np.float64)[:, :stop]
    if hashlib.sha256(energies.tobytes()).hexdigest() != census['energy_sha256']:
        raise ValueError("GATE shared_pole_census: got: stale energies; want: census rebound at current bands; why: SC must rebuild geometry")
    meta.shared_pole_capacity = CapacityLedger(
        meta, mesh_xy=mesh_xy,
        device_budget_bytes=int(config.memory.per_device_gb * 2**30))
    recipe = shared_real_pole_v2_r1
    tier = config.sigma.w_accuracy
    policy = recipe[tier]
    eta = float(config.sigma.regularization_ev)
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("GATE shared_pole_eta: got: invalid eta; want: finite positive sigma_regularization_ev; why: causal sampling height")
    height = recipe['height_eta_factor'] * eta
    plasma_ev = _plasma_ev(census['active_electrons'], census['cell_volume_bohr3'])
    plasma_ry = plasma_ev / RYD_TO_EV
    depth_act = float(census['active_manifold_depth_ev'])
    # omega_fine: where the STRUCTURE stops.  The collective pole sits at
    # omega_p; the first interband continuum runs to E_g + the depth of the
    # screening manifold.  omega_p alone is the wrong anchor for a dilute
    # system (a molecule in a large cell has a small omega_p and transitions
    # well above it), so take the larger.
    fine = max(plasma_ev, census['gap_ev'] + depth_act)
    # The W frequency Sigma samples at evaluation energy E is |E - E_n'|, so
    # the top of that window is the box edge plus the active occupied depth.
    # Patches, when set, are the box.
    sigma_extent = _sigma_extent_ev(config)
    sigma_window = sigma_extent + depth_act
    top_terms = {'plasmon': recipe['plasma_top_factor'] * fine,
                 'sigma_window': recipe['sigma_window_top_factor'] * sigma_window}
    top_bound_by = max(top_terms, key=top_terms.get)
    top = top_terms[top_bound_by]
    step = policy.get('line_step_eta_factor', recipe['line_step_eta_factor']) * eta
    growth = policy.get('line_growth_fraction', recipe['line_growth_fraction'])
    umin = max(height, census['gap_ev'])
    umax_zolotarev = recipe['imaginary_top_factor'] * fine
    # THE BANK'S DOMAIN IS A CEILING ON EVERY SAMPLE.  Reuse the bank's own
    # partitioner -- response_windows needs only bands, occupations and mu, the
    # same inputs this resolver already uses -- so there is one near/remote
    # partition in the tree, not a copy of one here; then ask minimax for the
    # largest |z| its remote Taylor rule can certify for the lowest cell.
    from .response_bank import RESPONSE_DOMAIN_PAD_RY
    (remote_cap_ev, remote_delta_lo_ev, remote_delta_hi_ry,
     remote_cap_receipt) = _remote_domain_cap(
        wfns, meta, census, height / RYD_TO_EV, recipe['bank_rule_tolerance'],
        # The bank pads its remote cells exactly when it keeps a rule session,
        # which is exactly when this resolver is given a support session.
        domain_pad_ry=(0.0 if support_session is None
                       else RESPONSE_DOMAIN_PAD_RY),
        session=support_session)
    top_uncapped, umax_uncapped = top, umax_zolotarev
    if remote_cap_ev is not None:
        # The ceiling is on |z|. An imaginary sample has |z| = u exactly, but a
        # line sample sits at E + i*h, so the admissible REAL part is
        # sqrt(cap^2 - h^2) -- capping E at the radius would put the last
        # support just outside the domain it was capped to.
        line_cap_ev = math.sqrt(max(remote_cap_ev**2 - height**2, 0.0))
        if line_cap_ev < top:
            top, top_bound_by = line_cap_ev, 'remote_cap'
        if remote_cap_ev < umax_zolotarev:
            umax_zolotarev = remote_cap_ev
    umax = umax_zolotarev
    u_max_bound_by = ('remote_cap' if remote_cap_ev is not None
                      and umax <= remote_cap_ev * (1 + 1e-12)
                      and umax < recipe['imaginary_top_factor'] * fine * (1 - 1e-12)
                      else 'zolotarev')
    if remote_cap_ev is not None and remote_cap_ev < sigma_window:
        # NOT something the recipe may paper over: the deck is asking Sigma for
        # W at frequencies the bank cannot serve at all, so the near window has
        # to grow (bank owner) or the Sigma grid has to shrink (deck).
        print_fn(
            "\n  ==========================================================\n"
            "  WARNING: the Sigma grid reaches W at "
            f"{sigma_window:.3f} eV, ABOVE the bank's remote Taylor domain\n"
            f"  ({remote_cap_ev:.3f} eV, set by the lowest remote cell edge "
            f"{remote_delta_lo_ev:.3f} eV).\n"
            "  The shared-pole support is capped there, so W above it is\n"
            "  carried by M1/M3 alone and the Sigma box samples an\n"
            "  unconstrained region.  Fix by growing the bank's near window\n"
            "  (repartition, KNOWN_LORRAX_ISSUES) or narrowing\n"
            "  sigma_omega_min_ev/max_ev; the recipe cannot resolve it.\n"
            "  ==========================================================")
    if umin >= umax:
        raise ValueError(f"GATE shared_pole_interval: got: u_min={umin} >= u_max={umax} eV; want: u_min < u_max; why: imaginary support interval is unresolved")
    support_receipt = None
    if support_session is not None:
        key = (RECIPE_HASH, tier, eta, int(meta.nspinor), int(meta.n_rmu),
               census['logical_band_count'])
        support_receipt = _support_envelope(
            dict(line_top_ev=top, omega_fine_ev=fine, u_min_ev=umin, u_max_ev=umax),
            key, support_session)
        retained = support_receipt['retained']
        top, fine, umin, umax = (retained['line_top_ev'], retained['omega_fine_ev'],
                                 retained['u_min_ev'], retained['u_max_ev'])
    # SELF-CONSISTENCY, not a user gate: `top` is constructed as a max that
    # INCLUDES the Sigma-window term and the envelope only ever raises it, so
    # this is unreachable by construction.  It names both numbers if it fires,
    # because the failure it guards against -- Sigma evaluating W above the
    # fit's pointwise support, where the model is unconstrained and the
    # accuracy cliff lives -- is silent in every other receipt (ASIMOM 2026-09-11).
    required_top = recipe['sigma_window_top_factor'] * sigma_window
    capped = remote_cap_ev is not None and top <= remote_cap_ev * (1.0 + 1.0e-12)
    if capped and top < required_top * (1.0 - 1.0e-12) and top >= sigma_window:
        # The bank's domain trimmed the design margin but still covers every W
        # frequency Sigma asks for. A note, not a refusal: the support is
        # sufficient, only the headroom is smaller than the recipe wanted.
        print_fn(f"  [shared-pole recipe {RECIPE_VERSION}] note: the bank's remote "
                 f"domain trimmed the top support to {top:.3f} eV, so the Sigma "
                 f"window margin is {top/sigma_window:.3f}x rather than "
                 f"{recipe['sigma_window_top_factor']}x; the window "
                 f"({sigma_window:.3f} eV) is still covered")
    elif not capped and top < required_top * (1.0 - 1.0e-12):
        raise ValueError(
            f"GATE shared_pole_support_window: got: line top {top:.6g} eV below "
            f"{recipe['sigma_window_top_factor']}*(Sigma box extent "
            f"{sigma_extent:.6g} eV + active occupied depth {depth_act:.6g} eV) "
            f"= {required_top:.6g} eV; want: pointwise support covering every W "
            "frequency Sigma samples; why: above the top support the model is "
            "pinned only by M1/M3 and its error rises by orders of magnitude")
    # ONE LADDER, TWO LAWS.  Uniform 2*eta while the structure lasts, then a
    # step that grows by (1+growth) per interval, so the count beyond the
    # plasmon is logarithmic in `top` and a deep-band material pays a handful
    # of supports, not hundreds.  The step is continuous at omega_fine: the
    # first geometric step IS 2*eta, so the plasmon's upper shoulder is still
    # resolved at or below the sample height.
    low = [i * step for i in range(math.ceil(fine / step))]
    if len(low) > 1 and fine - low[-1] < 0.5 * step:
        low.pop()                      # omega_fine absorbs a stub interval
    tail = []
    while True:
        # k = 0 is omega_fine itself, so the uniform region ends ON the
        # structure scale and the growth starts from there.
        nxt = fine + (step / growth) * ((1.0 + growth) ** len(tail) - 1.0)
        if not (nxt < top):
            break
        tail.append(nxt)
        if len(low) + len(tail) > _LINE_COUNT_CEILING:
            raise ValueError(f"GATE shared_pole_line_count: got: over {_LINE_COUNT_CEILING} line supports for top={top:.6g} eV at step={step:.6g} eV; want: a resolvable ladder; why: a runaway support list is a deck or census error, not a recipe")
    # The exact endpoint REPLACES the last ladder point when the gap left is
    # under half the local step; appending it there would put two samples a few
    # tens of meV apart at height h = 4*eta, a near-duplicate Hermite block.
    ladder = low + tail
    if len(ladder) > 1 and (top - ladder[-1]) < 0.5 * (ladder[-1] - ladder[-2]):
        ladder = ladder[:-1]
    line = np.asarray(ladder + [top], dtype=np.float64)
    kappa = umax / umin
    count = max(recipe['imaginary_min_count'], round(
        math.log(16 * kappa**2) * math.log(4 / recipe['imaginary_count_epsilon'])
        / (2 * math.pi**2))) if tier == 'production' else policy['imaginary_count']
    imaginary = np.geomspace(umin, umax, count)
    mids = 0.5 * (line[:-1] + line[1:])
    # One held line point per spacing law, at that law's own midpoint:
    # arithmetic inside the uniform region, geometric inside the geometric one.
    held_targets = (recipe['held_line_fine_fraction'] * fine, math.sqrt(fine * top))
    held_pairs = [int(np.argmin(abs(mids - target))) for target in held_targets]
    held_line = mids[held_pairs]
    held_imag = np.sqrt(imaginary[[0, -2]] * imaginary[[1, -1]])
    points, role_z, role_codes, distinct_ids, held_flags, support_pairs = [], [], [], [], [], []
    def add(real, imag, role, held, pair=None):
        z = complex(real, imag) / RYD_TO_EV
        if z not in points:
            points.append(z)
        role_z.append(z)
        role_codes.append(ROLE_CODES[role])
        distinct_ids.append(points.index(z))
        held_flags.append(held)
        support_pairs.append([-1, -1] if pair is None else pair)
    for i, e in enumerate(line):
        add(e, height, 'line', False)
    for i, u in enumerate(imaginary):
        add(0.0, u, 'imaginary', False)
    for i, e in enumerate(held_line):
        j = held_pairs[i]
        add(e, height, 'held_line', True, [j, j+1])
    for i, u in enumerate(held_imag):
        add(0.0, u, 'held_imaginary', True,
            [0, 1] if i == 0 else [count-2, count-1])
    fit_ids = sorted({i for i, held in zip(distinct_ids, held_flags) if not held})
    held_ids = sorted({i for i, held in zip(distinct_ids, held_flags) if held})
    if set(fit_ids) & set(held_ids):
        raise ValueError("GATE shared_pole_held_exclusion: got: held/training collision; want: disjoint physical IDs; why: held diagnostics must be independent")
    n = int(meta.nspinor) * int(meta.n_rmu)
    result = {
        'role_codes': dict(ROLE_CODES),
        'recipe_version': RECIPE_VERSION, 'recipe_hash': RECIPE_HASH,
        'gate_version': GATE_VERSION, 'gate_hash': GATE_HASH,
        'accuracy': tier, 'accuracy_status': 'NOT_MEASURED',
        'accuracy_reason': 'resolved geometry has no authenticated matching campaign receipt',
        'eta_ev': eta, 'height_ev': height, 'height_ry': height / RYD_TO_EV,
        'plasma_ev': plasma_ev, 'plasma_ry': plasma_ry,
        'omega_fine_ev': fine, 'active_depth_ev': depth_act,
        'sigma_extent_ev': sigma_extent, 'sigma_window_ev': sigma_window,
        'top_ev': top, 'top_bound_by': top_bound_by,
        'top_uncapped_ev': top_uncapped, 'u_max_uncapped_ev': umax_uncapped,
        'u_max_bound_by': u_max_bound_by,
        'remote_cap_ev': remote_cap_ev, 'remote_delta_lo_ev': remote_delta_lo_ev,
        'remote_domain_pad_ev': (0.0 if support_session is None
                                 else RESPONSE_DOMAIN_PAD_RY * RYD_TO_EV),
        'remote_delta_hi_ry': remote_delta_hi_ry,
        # The cap is FOUND by evaluating the rule, so the receipt says how
        # many evaluations it took and what they cost.
        'remote_cap_search': remote_cap_receipt,
        'line_step_ev': step, 'line_growth_fraction': growth,
        'line_ev': line, 'imaginary_ev': imaginary,
        'held_line_ev': held_line, 'held_imaginary_ev': held_imag,
        'u_min_ev': umin, 'u_max_ev': umax, 'kappa': kappa,
        'z_ry': np.asarray(role_z, dtype=np.complex128),
        'role': np.asarray(role_codes, dtype=np.int8),
        'distinct_id': np.asarray(distinct_ids, dtype=np.int64),
        'held': np.asarray(held_flags, dtype=np.bool_),
        'support_pair': np.asarray(support_pairs, dtype=np.int64),
        'fit_ids': np.asarray(fit_ids, dtype=np.int64),
        'held_ids': np.asarray(held_ids, dtype=np.int64),
        'line_count': len(line), 'imaginary_count': count,
        'unique_evaluations': len(points), 'fit_count': len(fit_ids),
        'held_count': len(held_ids), 'role_count': len(role_codes),
        'n': n, 'direction_cutoff': policy['direction_cutoff'],
        'imaginary_width': math.ceil(n * policy['imaginary_width_fraction']),
        'infinity_width': math.ceil(n * policy['infinity_width_fraction']),
        'multiplet_relative_tolerance': recipe['multiplet_relative_tolerance'],
        'bank_rule_tolerance': recipe['bank_rule_tolerance'],
        'sigma_tolerance': policy['sigma_tolerance'],
        'moment_convention': recipe['moment_convention'], 'census': dict(census),
        'operator_realization': recipe['operator_realization'],
        'U_bytes_per_rank': meta.shared_pole_capacity.U_bytes_per_rank,
    }
    if support_receipt is not None:
        result['support_envelope'] = support_receipt
    result['metadata_array_bytes'] = sum(v.nbytes for v in result.values()
                                         if isinstance(v, np.ndarray))
    # Longest prefix FIRST: the lookup below takes the first match, so
    # 'line_step_ev' must not be answered by the 'line' rule.
    rules = {
        'height': 'h=4*eta', 'eta': 'literal sigma_regularization_ev',
        'plasma': '2*sqrt(4*pi*active_electrons/volume) Ry',
        'omega_fine': 'max(omega_p, E_g + active occupied depth)',
        'active_depth': 'mu - lowest energy of the screening-active manifold',
        'sigma_extent': 'max |edge| of the Sigma omega grid (patches when set, else the box)',
        'sigma_window': 'Sigma extent + active depth: the top W frequency Sigma samples',
        'top_bound': 'which term of the top max() bound it',
        'top_uncapped': 'top before the bank remote-domain ceiling',
        'u_max_uncapped': 'u_max before the bank remote-domain ceiling',
        'u_max_bound': 'zolotarev tier rule, or the bank remote-domain cap',
        'remote_cap': 'largest |z| response_laplace_rule accepts on the lowest remote cell, padded as the bank will pad it',
        'remote_delta': 'lowest remote Laplace cell transition edge, from response_windows',
        'top': 'max(2.25*omega_fine, 1.25*sigma_window)',
        'line_step': '2*eta (tier line_step_eta_factor)',
        'line_growth': 'geometric step ratio beyond omega_fine (tier)',
        'line': 'uniform step to omega_fine, then step*(1+growth) per interval, exact top once',
        'imaginary': 'log-spaced u_min..u_max; round(log(16*(u_max/u_min)^2)*log(4000)/(2*pi^2)), min2; tier width ceil(f*n)',
        'held_line': 'adjacent-support midpoint nearest 0.5*omega_fine and sqrt(omega_fine*top); lower-index tie',
        'held_imaginary': 'geometric midpoint of first/last adjacent imaginary pair',
        'u_min': 'max(h,logical gap)', 'u_max': '2.5*omega_fine', 'kappa': 'u_max/u_min',
        'infinity': 'ceil(tier infinity fraction*n)', 'direction': 'tier relative singular cutoff',
        'multiplet': 'whole multiplets within relative 1e-6',
        'bank': 'fixed Hermite certificate tolerance 1e-8',
        'sigma': 'tier Sigma tolerance production1e-4/relaxed1e-3',
        'census': 'current full-band occupations, authenticated k weights/capacity; active set is the fixed point of depth <= omega_p(set)',
        'U_bytes': '16*nk_full*(nspinor*nmu)^2/(Px*Py), logical bytes/rank',
        'metadata': 'sum of replicated metadata array nbytes',
    }
    if support_receipt is not None:
        rules.update(top='SC high-water envelope of max(2.25*omega_fine, 1.25*sigma_window)',
                     omega_fine='SC high-water envelope of max(omega_p, E_g + active depth)',
                     u_min='SC low-water envelope of max(h,logical gap)',
                     u_max='SC high-water envelope of 2.5*omega_fine',
                     support_envelope='current required bounds and retained sampling enclosure; not an interpolation-error certificate')
    print_fn(_support_report(result, tier, census))
    for key, value in result.items():
        shown = value.tolist() if isinstance(value, np.ndarray) else value
        rule = next((v for prefix, v in rules.items() if key.startswith(prefix)),
                    'canonical role/ID census and versioned recipe; no accuracy inferred from missing evidence')
        print_fn(f'  [shared-pole recipe {RECIPE_VERSION}] {key}={shown} (rule: {rule})')
    return result
