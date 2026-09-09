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

RECIPE_VERSION = "shared_real_pole_v1_r3b"
GATE_VERSION = "shared_real_pole_gates_v1_r3b"
RECEIPT_SCHEMA = "lorrax.shared-real-pole.receipt.v1"
ROLE_CODES = {"line": 0, "imaginary": 1, "infinity": 2, "held_line": 3, "held_imaginary": 4}

shared_real_pole_v1_r3b = {
    "version": RECIPE_VERSION,
    "height_eta_factor": 4.0,
    "reference_eta_ev": 0.25,
    "active_depth_ev": 15.0,
    "borderline_depth_ev": 25.0,
    "line_break_ev": 12.0,
    "line_low_step_ev": 0.5,
    "line_spacing_rule": "2*eta below 12 eV, 4*eta above; no material branch",
    "line_high_step_ev": 1.0,
    "plasma_margin_ev": 3.5,
    "imaginary_floor_max_ev": 16.0,
    "imaginary_count_epsilon": 1.0e-3,
    "imaginary_min_count": 2,
    "imaginary_count_rule": "max(2, round(log(16*(L/u_min)^2)*log(4000)/(2*pi^2)))",
    "held_line_fractions": (0.25, 0.65),
    "multiplet_relative_tolerance": 1.0e-6,
    "bank_rule_tolerance": 1.0e-8,
    "moment_convention": "S_m = 2 M_(2m+1); physical M1 and M3 only",
    "production": {"direction_cutoff": 1.0e-3, "imaginary_width_fraction": 0.25,
                   "infinity_width_fraction": 0.125, "sigma_tolerance": 1.0e-4},
    "relaxed": {"direction_cutoff": 1.0e-2, "imaginary_width_fraction": 0.125,
                "infinity_width_fraction": 0.0625, "sigma_tolerance": 1.0e-3,
                "line_count": 8, "imaginary_count": 2},
}

# Each entry is (predicate description, threshold); the public table adds name
# and version. Composite checks retain their individual dimensional thresholds.
_GATE_ROWS = {
    "normalized_gram_keep": ("retain gamma/gamma_max strictly above cut", 1.0e-8),
    "normalized_gram_validity": ("gamma_min/gamma_max >= threshold", -1.0e-7),
    "zero_ritz_policy": ("drop lambda <= cutoff only within factor-weight budget",
                         {"lambda_cutoff_ry2": 1.0e-6, "max_dropped_weight_fraction": 1.0e-6}),
    "finite_factors_poles": ("finite complex128 C; finite positive float64 active poles2; int64 K; exact-zero inactive C and positive sentinel", True),
    "passivity": ("V-whitened -Wc(i eta) spectrum in bounds and relative anti-Hermitian part within tolerance",
                  {"eigenvalue_min": -1.0e-10, "eigenvalue_max": 1.0 + 1.0e-8,
                   "antihermitian_relative_max": 1.0e-10}),
    "retained_subspace_moments": ("relative M1/M3 identity defect in retained Ritz infinity states P_R x_inf, with P_R Gram-metric orthogonal on span(OZ), after cut and zero policy <= threshold; original q_inf defect is diagnostic", 1.0e-10),
    "held_w": ("held W value/derivative relative defects with coordinates and receipt paths; diagnostic, no universal threshold", None),
    "full_m1_defect": ("maximum over q of relative full M1 defect after cut and zero policy; PASS within diagnostic band, WARN outside, never refuse", 2.0e-4),
    "full_m3_defect": ("maximum over q of relative full M3 defect after cut and zero policy; PASS within diagnostic band, WARN outside, never refuse", 2.0e-3),
    "representation": ("scalar N_spinor=1 and authenticated TRS allowed", {"nspinor": 1, "trs_allowed": True}),
    "capacity": ("aggregate live device bytes per rank of new shared-pole objects including workspace <= threshold * U", 3.0),
    "stream_peak": ("inherited response stream peak <= threshold * incumbent MPA stream peak on the same deck and processor geometry, using the same measurement method", 1.05),
    "sigma_peak": ("inherited Sigma peak including one incumbent-shaped W <= threshold * incumbent MPA Sigma peak on the same deck, processor geometry and window plan, using the same measurement method", 1.05),
    "rule_validity": ("bank and Sigma certificates cover current domains at resolved tolerances", True),
    "sc_rebuild": ("samples, directions, poles, ranks, intervals and rules rebuilt at current bands and occupations", True),
}
shared_real_pole_gates_v1_r3b = {
    name: {"name": name, "predicate": predicate, "threshold": threshold,
           "version": GATE_VERSION}
    for name, (predicate, threshold) in _GATE_ROWS.items()
}

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


RECIPE_HASH = table_hash(shared_real_pole_v1_r3b)
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

    def receipt(self):
        """Snapshot ordered stage rows and the independently measured peak."""
        import copy
        return copy.deepcopy(dict(geometry=self.geometry,
                                  U_bytes_per_rank=self.U_bytes_per_rank,
                                  limit_bytes_per_rank=self.limit_bytes_per_rank,
                                  device_budget_bytes_per_rank=self.device_budget_bytes_per_rank,
                                  entries=self.entries, live_stages=self._live_stages,
                                  measured_peak=self.measured_peak,
                                  stream_peak=self.stream_peak,
                                  sigma_peak=self.sigma_peak))


def construction_receipt(measurements=None, *, capacity=None):
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
        result['capacity'] = capacity.receipt()
        result['gates'] = [result['capacity'][r['name']] if r['name'] in ('stream_peak', 'sigma_peak')
                           else r for r in result['gates']]
        rows = result['capacity']['entries']
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
    The plasma equation is omega_p = sqrt(4*pi*n_e) Ha, with n_e obtained
    from capacity-weighted occupations of bands with max_k E_nk >= mu-15 eV.
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
    recipe = shared_real_pole_v1_r3b
    active = depth <= recipe['active_depth_ev']
    borderline = ((depth > recipe['active_depth_ev'])
                  & (depth <= recipe['borderline_depth_ev']))
    electrons = float(capacity * np.sum(weights[:, None] * np.where(active, occupations, 0.0)))
    volume = float(meta.cell_volume)
    if not (math.isfinite(mu) and math.isfinite(volume) and volume > 0
            and math.isfinite(electrons) and electrons > 0):
        raise ValueError("GATE shared_pole_plasma: got: nonpositive/nonfinite active charge or cell volume; want: positive finite electrons and bohr^3; why: omega_p requires positive density")
    meta.shared_pole_census = {
        "mu_ry": mu, "gap_ev": gap_ev, "partial_at_mu": partial_at_mu,
        "active_electrons": electrons, "cell_volume_bohr3": volume,
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


def resolve_shared_pole_recipe(config, wfns, meta, *, mesh_xy, print_fn):
    """Resolve DESIGN §5 from current metadata into scalars and small arrays.

    ``bind_shared_pole_census`` must have consumed this map's occupation state.
    Returns a plain dict with equal-length ``z_ry`` complex128, ``role`` int8,
    ``distinct_id`` int64 and ``held`` bool arrays [role]. Repeated physical
    points share a distinct_id; fitting and held IDs are disjoint. Infinity's
    role code is reserved (moments need no bank evaluation). Conjugates are
    constructor states, never bank calls. ``support_pair`` int64 [role,2]
    binds held endpoints; [-1,-1] means not a held midpoint.
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
    recipe = shared_real_pole_v1_r3b
    tier = config.sigma.w_accuracy
    policy = recipe[tier]
    eta = float(config.sigma.regularization_ev)
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("GATE shared_pole_eta: got: invalid eta; want: finite positive sigma_regularization_ev; why: causal sampling height")
    height = recipe['height_eta_factor'] * eta
    plasma_ry = 2.0 * math.sqrt(4.0 * math.pi * census['active_electrons']
                               / census['cell_volume_bohr3'])
    top = plasma_ry * RYD_TO_EV + recipe['plasma_margin_ev']
    scale = eta / recipe['reference_eta_ev']
    low_step = recipe['line_low_step_ev'] * scale
    high_step = recipe['line_high_step_ev'] * scale
    if tier == 'relaxed':
        line = np.linspace(0.0, top, policy['line_count'])
    else:
        edge = min(recipe['line_break_ev'], top)
        # Each segment has an exact endpoint, included once; no accumulated
        # stepping error and no thinning to satisfy a historical sample count.
        low = [i * low_step for i in range(math.ceil(edge / low_step))]
        high = ([edge + i * high_step for i in range(math.ceil((top-edge)/high_step))]
                if top > edge else [])
        line = np.asarray(low + high + [top], dtype=np.float64)
    umin, umax = max(height, census['gap_ev']), max(recipe['imaginary_floor_max_ev'], top)
    if umin >= umax:
        raise ValueError(f"GATE shared_pole_interval: got: u_min={umin} >= u_max={umax} eV; want: u_min < u_max; why: imaginary support interval is unresolved")
    kappa = top / umin
    count = max(recipe['imaginary_min_count'], round(
        math.log(16 * kappa**2) * math.log(4 / recipe['imaginary_count_epsilon'])
        / (2 * math.pi**2))) if tier == 'production' else policy['imaginary_count']
    imaginary = np.geomspace(umin, umax, count)
    mids = 0.5 * (line[:-1] + line[1:])
    held_pairs = [int(np.argmin(abs(mids - fraction*top)))
                  for fraction in recipe['held_line_fractions']]
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
        'plasma_ev': plasma_ry * RYD_TO_EV, 'plasma_ry': plasma_ry,
        'top_ev': top, 'spacing_scale': scale, 'low_step_ev': low_step,
        'high_step_ev': high_step,
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
        'U_bytes_per_rank': meta.shared_pole_capacity.U_bytes_per_rank,
    }
    result['metadata_array_bytes'] = sum(v.nbytes for v in result.values()
                                         if isinstance(v, np.ndarray))
    rules = {
        'height': 'h=4*eta', 'eta': 'literal sigma_regularization_ev',
        'plasma': '2*sqrt(4*pi*active_electrons/volume) Ry',
        'top': 'L=omega_p+3.5 eV', 'spacing': 'eta/0.25',
        'low_step': '2*eta', 'high_step': '4*eta',
        'line': '2eta below 12 eV, 4eta above, exact L once; relaxed 8 endpoints',
        'imaginary': 'log-spaced u_min..u_max; round(log(16*(L/u_min)^2)*log(4000)/(2*pi^2)), min2; tier width ceil(f*n)',
        'held_line': 'adjacent-support midpoint nearest 25%/65% L; lower-index tie',
        'held_imaginary': 'geometric midpoint of first/last adjacent imaginary pair',
        'u_min': 'max(h,logical gap)', 'u_max': 'max(16 eV,L)', 'kappa': 'L/u_min',
        'infinity': 'ceil(tier infinity fraction*n)', 'direction': 'tier relative singular cutoff',
        'multiplet': 'whole multiplets within relative 1e-6',
        'bank': 'fixed Hermite certificate tolerance 1e-8',
        'sigma': 'tier Sigma tolerance production1e-4/relaxed1e-3',
        'census': 'current full-band occupations, authenticated k weights/capacity; active band top >= mu-15 eV',
        'U_bytes': '16*nk_full*(nspinor*nmu)^2/(Px*Py), logical bytes/rank',
        'metadata': 'sum of replicated metadata array nbytes',
    }
    for key, value in result.items():
        shown = value.tolist() if isinstance(value, np.ndarray) else value
        rule = next((v for prefix, v in rules.items() if key.startswith(prefix)),
                    'canonical role/ID census and versioned recipe; no accuracy inferred from missing evidence')
        print_fn(f'  [shared-pole recipe {RECIPE_VERSION}] {key}={shown} (rule: {rule})')
    return result
