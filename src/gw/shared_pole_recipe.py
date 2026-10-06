"""Shared real-pole input recipe and gate vocabulary (DESIGN §§5–6).

Host metadata only: no response matrices, device allocation, backend selection,
chemical-potential solve, or numerical gate implementation lives here. Physical
sample coordinates are Ry; reporting coordinates are eV. Consumers import these
tables instead of copying the thresholds into bank/constructor/store/Sigma code.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import operator

import numpy as np

RECIPE_VERSION = "shared_real_pole_v1_r4"
GATE_VERSION = "shared_real_pole_gates_v1_r3b"
RECEIPT_SCHEMA = "lorrax.shared-real-pole.receipt.v1"
ROLE_CODES = {"line": 0, "imaginary": 1, "infinity": 2, "held_line": 3, "held_imaginary": 4}

shared_real_pole_v1_r3b = {
    "version": RECIPE_VERSION,
    "height_eta_factor": 4.0,
    "height_floor_ev": 2.6,
    "active_depth_ev": 15.0,
    "borderline_depth_ev": 25.0,
    # L = omega_p + plasma_margin_ev tops the IMAGINARY ladder (u_max, kappa); the line
    # ladder stops at the reach of the requested reads, not at L.
    "plasma_margin_ev": 3.5,
    "imaginary_floor_max_ev": 16.0,
    "imaginary_count_epsilon": 1.0e-3,
    "imaginary_min_count": 2,
    "imaginary_count_rule": "max(2, round(log(16*(L/u_min)^2)*log(4000)/(2*pi^2)))",
    "held_line_fractions": (0.25, 0.65),
    # Production line sites (report section IV.B, the support rule): quantiles of the
    # consumer's crossing-pair density to the power alpha on [omega_lo, omega_reach].
    # The reads are the requested states' Sigma frequencies (gw.qp_support
    # .requested_reads_ev: E_in +/- this recipe's support_read_pad_ev), so the line reaches
    # every state the run asks for (WSUPPORT, claims 3282/3284; was a fixed +/-5 eV
    # window). Si Sigma optimum flat over alpha 0.25-0.75.
    "support_density_power": 0.5,
    # The line reads each requested state at +-5 eV (the Sigma plan pads only 2 eV):
    # the old rule's offsets, now for every requested state, so the line never reaches
    # less far than it did (the H2- hsuite one-shot moved 52 meV near E_F against a
    # dense ladder when its reach fell from 9.9 to 7.0 eV at +-2 eV).
    "support_read_pad_ev": 5.0,
    "multiplet_relative_tolerance": 1.0e-6,
    "moment_convention": "S_m = 2 M_(2m+1); physical M1 and M3 only",
    "operator_realization": "little-group-reynolds-v1",
    # bank_rule_tolerance is tier-owned (METAL 2026-09-16): the remote-cell certificate
    # floor on a deep-semicore metal (Fe: even rows 3.4e-9 against 1e-8/4/amp) is a
    # deck property; production keeps 1e-8 bit for bit, relaxed admits 1e-7.
    # Production sizing (owner rulings 2026-09-17, 2026-10-05): at least 18 fitted supports
    # (line + imaginary; held and the M1/M3 block are extra) and at most sample_cap bank
    # samples including the held ones; between the two, the line count is the strip law
    # of support_line_count; at most N_mu/16 right singular directions per line support;
    # at most 1.8 N_mu retained Gram directions, the pole count K per parent.
    "production": {"direction_cutoff": 1.0e-3, "imaginary_width_fraction": 0.25,
                   "infinity_width_fraction": 0.125,
                   "bank_rule_tolerance": 1.0e-8, "fitted_support_count": 18,
                   "sample_cap": 40,
                   "line_direction_cap_fraction": 0.0625, "pole_budget_fraction": 1.8},
    "relaxed": {"direction_cutoff": 1.0e-2, "imaginary_width_fraction": 0.125,
                "infinity_width_fraction": 0.0625,
                "line_count": 8, "imaginary_count": 2,
                "bank_rule_tolerance": 1.0e-7},
}

# Each entry is (predicate description, threshold); the public table adds name
# and version. Composite checks retain their individual dimensional thresholds.
_GATE_ROWS = {
    "normalized_gram_keep": ("retain gamma/gamma_max strictly above cut", 1.0e-8),
    "normalized_gram_validity": ("gamma_min/gamma_max >= threshold; every route (the even parent Gram takes the larger of this and gram_rounding_validity)", -1.0e-7),
    # The exact Gram is PSD, so a negative computed eigenvalue is rounding.
    # Weyl: gamma_min >= -||E||_2 >= -gamma_n ||D^-1/2 Sigma D^-1/2||_F with
    # Sigma the uncancelled magnitude of each divided-difference entry
    # (shared_pole_reduction.gram_rounding_floor), plus R u gamma_max for the solver.
    "gram_rounding_validity": ("even route: equilibrated gamma_min >= -max(-normalized_gram_validity * gamma_max, bound_factor * gamma_n ||D^-1/2 Sigma D^-1/2||_F + R u gamma_max); gamma_n = n u/(1-n u), Sigma the uncancelled divided-difference magnitudes; sample error not bounded",
                               {"bound_factor": 1.0, "unit_roundoff": 2.0 ** -53}),
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
                          # owner 2026-09-22: 1e-10 refused TaAs 8x8x8 roundoff (3.5e-10 at q=58, held data 6e-15)
                          {"reference_relative_max": 1.0e-12, "model_relative_max": 1.0e-8}),
    "full_m1_defect": ("maximum over q of relative full M1 defect after cut and zero policy; PASS within diagnostic band, WARN outside, never refuse", 2.0e-4),
    "full_m3_defect": ("maximum over q of relative full M3 defect after cut and zero policy; PASS within diagnostic band, WARN outside, never refuse", 2.0e-3),
    # The stored operator is the spin-traced mu x mu charge response on scalar and
    # two-component decks alike; only G carries the spinor axes.
    "representation": ("charge operator from N_spinor in (1, 2) and authenticated TRS allowed",
                       {"nspinor": (1, 2), "trs_allowed": True}),
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


RECIPE_HASH = table_hash(shared_real_pole_v1_r3b)
#: Pad (eV) of the W line reads around each requested state (``qp_support.requested_reads_ev``).
SUPPORT_READ_PAD_EV = shared_real_pole_v1_r3b["support_read_pad_ev"]
#: The Sigma quadrature is not W.  A tier's default for an omitted
#: sigma_quadrature_eps lives outside the hashed table above, whose every field
#: sets W's sampling or poles, so RECIPE_HASH (restart and bank identity) binds
#: only W and a Sigma-only edit never refuses a stored model.
SIGMA_EPS_DEFAULT = {"relaxed": 5.0e-4}
GATE_HASH = table_hash(shared_real_pole_gates_v1_r3b)

# Time-reversal-broken ordered route (particle-hole pencil in z). A separate
# table keeps the TRS table, its hash and every stored TRS identity unchanged.
ORDERED_GATE_VERSION = "shared_real_pole_gates_ordered_v1"
shared_real_pole_gates_ordered_v1 = {
    name: dict(row) for name, row in shared_real_pole_gates_v1_r3b.items()}
# CC/TT spans are normalized separately before the CT projection. Resolve
# their two Gram cuts more conservatively than the joint/inverse-pole cut.
# The ordered Fe 8^3 bank needs a 1e-7 joint cut: at 1e-8 the weak span
# produces a negative projected pencil despite a positive retained metric.
shared_real_pole_gates_ordered_v1["normalized_gram_keep"].update(
    threshold=1.0e-7, sector_threshold=1.0e-5)
for _name, (_predicate, _threshold) in {
    "representation": ("charge operator from N_spinor in (1, 2), authenticated TRS broken, ordered bank: positive poles per parent, hole side from the parent of -q transposed",
                       {"nspinor": (1, 2), "trs_allowed": False, "ordered": True}),
    "passivity": ("Hermitian part of the V-whitened -Wc(i eta) of the signed particle-hole model in bounds; the anti-Hermitian part is the odd channel, reported only",
                  {"eigenvalue_min": -1.0e-10, "eigenvalue_max": 1.0 + 1.0e-8}),
    "retained_subspace_moments": ("relative projected z-moment m0..m3 defect of the signed particle-hole model on the infinity directions <= threshold; NOT_MEASURED for a finite-state bank without odd moments", 1.0e-10),
    "model_reciprocity": ("not applicable: time-reversal-broken samples carry no transpose symmetry", None),
}.items():
    shared_real_pole_gates_ordered_v1[_name] = {
        "name": _name, "predicate": _predicate, "threshold": _threshold,
        "version": ORDERED_GATE_VERSION}
# The ordered row measures a PROJECTED z-moment identity on the original infinity
# directions: exact only for the full Galerkin span, projection accuracy after the keep
# and retention cuts. It is therefore diagnostic with a calibration band, as full_m1 and
# full_m3 are, and its verdict is computed from the measured defects -- never asserted.
# Band: measured CrI3 q=1 construction (m0 4.8e-9, m1 6.8e-11, m2 1.2e-5, m3 1.3e-7,
# claim 2357) against the hand-caught failure this row exists for (m3 own-norm 0.632 from
# spurious 100-1000 Ry poles, TRMOM 2026-09-15).
shared_real_pole_gates_ordered_v1["retained_subspace_moments"].update({
    "diagnostic": True, "calibration_range": (1.2e-5, 1.0e-3),
    "source": "CrI3 q=1 58385920 vs claim 2357; TRMOM spurious-pole case 0.632",
})

ORDERED_GATE_HASH = table_hash(shared_real_pole_gates_ordered_v1)

# A four-component charge carrier is a new representation of the same scalar
# CC operator. Keep the scalar and full photon gate hashes unchanged, so an
# existing artifact cannot silently acquire the charge4 interpretation.
CHARGE4_GATE_VERSION = "shared_real_pole_charge4_v1"
shared_real_pole_gates_charge4_trs_v1 = copy.deepcopy(shared_real_pole_gates_v1_r3b)
shared_real_pole_gates_charge4_ordered_v1 = copy.deepcopy(shared_real_pole_gates_ordered_v1)
for _ordered, _table in ((False, shared_real_pole_gates_charge4_trs_v1),
                         (True, shared_real_pole_gates_charge4_ordered_v1)):
    _table["representation"] = {
        "name": "representation",
        "predicate": ("four-component spin-traced charge operator; "
                      "ordered particle-hole bank" if _ordered else
                      "four-component spin-traced charge operator; TRS-even bank"),
        "threshold": {"nspinor": 4, "trs_allowed": not _ordered,
                      **({"ordered": True} if _ordered else {})},
        "version": CHARGE4_GATE_VERSION,
    }
CHARGE4_TRS_GATE_HASH = table_hash(shared_real_pole_gates_charge4_trs_v1)
CHARGE4_ORDERED_GATE_HASH = table_hash(shared_real_pole_gates_charge4_ordered_v1)


def charge4_gates(ordered):
    """The charge4 receipt table, separate from photon and older charge banks."""
    return (shared_real_pole_gates_charge4_ordered_v1 if ordered
            else shared_real_pole_gates_charge4_trs_v1)


def representation_row_passed(measured, threshold):
    """True when every measured representation field matches its threshold.

    ``nspinor`` may be a tuple of admitted values; every other field compares equal.
    The constructor used to write ``passed=True`` as a literal here, so a deck that
    violated the row recorded PASS.
    """
    for key, want in threshold.items():
        got = measured.get(key)
        if isinstance(want, tuple):
            matched = got in want
        elif isinstance(want, bool):
            # A JSON round trip brings a flag back as 0/1, so compare truth.
            matched = bool(got) == bool(want)
        else:
            matched = got == want
        if not matched:
            return False
    return True


def retained_moment_row_passed(measured, row):
    """Verdict for the retained-moment row from the measured per-order defects.

    ``measured`` maps order name to a value or list of values. A diagnostic row with a
    calibration band passes at the band ceiling; otherwise the row's own threshold applies.
    ``None`` (nothing measured) is not a pass.
    """
    if not measured:
        return None
    ceiling = (row.get("calibration_range") or (None, row.get("threshold")))[1]
    if ceiling is None:
        return None
    worst = max(float(np.max(np.abs(np.asarray(value, dtype=float))))
                for value in measured.values())
    return bool(worst <= float(ceiling))


def reciprocity_row_verdict(measured):
    """Verdict for the model-reciprocity row from its per-sample records.

    ``measured`` is the constructor's per-sample record with ``passed`` and
    ``applicable`` entries (scalars or nested lists over held samples and
    fields). The predicate is CONDITIONAL: it compares the model's transpose
    symmetry against the held reference's only where that reference has the
    symmetry, so on a sample whose reference does not, nothing is compared.

    Returns ``True`` only when at least one record was evaluated and every
    evaluated record passed, ``False`` when an evaluated record failed, and
    ``None`` when nothing was evaluated -- which :func:`gate_receipt` records
    as NOT_MEASURED rather than PASS.

    The constructor used to write ``passed=True`` as a literal here, so a run
    whose reference was never symmetric enough to compare recorded a PASS for a
    row that had compared nothing. Measured on the Si reference deck: at q=0 the
    held reference defect is ~5e-08 against a ``reference_relative_max`` of
    1e-12, so the row was NOT_MEASURED on every map and both accelerators while
    reading as PASS (SCGRAM-A, 2026-09-16). This is the same defect
    :func:`representation_row_passed` exists to fix, and INVARIANTS 23: an
    absent measurement is never PASS.
    """
    if not measured:
        return None
    applicable = np.asarray(measured.get("applicable", []), dtype=bool).ravel()
    passed = np.asarray(measured.get("passed", []), dtype=bool).ravel()
    if applicable.size == 0 or applicable.size != passed.size:
        return None
    if not applicable.any():
        return None
    return bool(passed[applicable].all())


def gate_receipt(name, value=None, *, passed=None, reason, table=None):
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
    row = (shared_real_pole_gates_v1_r3b if table is None else table)[name]
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
    batch/phase identifiers when necessary). A reservation over the budget is
    recorded as FAIL, warned once and admitted: the run never refuses on a
    price (owner 2026-10-01). The ledger owns no arrays or memory allocator.
    Host I/O staging is reported separately by the store, which refuses a host
    copy larger than its device panel (coordinator ruling 11).
    """

    def __init__(self, meta, *, mesh_xy, device_budget_bytes=None):
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
        # What the pool draws at run time beyond every priced and compiled
        # byte, measured per platform and P (runtime.aot_memory); it is taken
        # off the budget once here, so every admission sees it.
        from runtime.aot_memory import runtime_reserve_bytes
        self.reserve_bytes_per_rank = int(runtime_reserve_bytes(
            mesh_xy.devices.flat[0].platform, g['px'] * g['py']))
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
        """Admit actual-batch bytes before allocation; over the budget, warn and admit.

        Returns a detached JSON row. ``concurrent_with`` is an iterable of
        accepted stage names, not their byte totals. Caller-live allocations
        must be charged here OR in a named concurrent reservation, never both.
        No runtime peak is inferred from a successful analytical admission.
        A row over the available budget keeps ``device_budget_status`` FAIL
        and prints one line (``common.gpu_utils.warn_over_budget``).
        """
        if not isinstance(stage, str) or not stage.strip() or stage in self._accepted:
            raise ValueError(f"capacity stage must be a new nonempty name; got {stage!r}")
        row = self.preview(resident_bytes_per_rank=resident_bytes_per_rank,
                           workspace_bytes_per_rank=workspace_bytes_per_rank,
                           concurrent_with=concurrent_with)
        row['stage'] = stage
        row['live_stages'] = sorted(set(row['concurrent_with']) | {stage})
        self.entries.append(row)
        from common.gpu_utils import record_stage_price, warn_over_budget
        if row['device_budget_status'] != 'PASS':
            warn_over_budget(f"shared-pole ledger {stage}", row['aggregate_bytes_per_rank'],
                             row['available_device_bytes_per_rank'])
        self._accepted[stage] = row
        record_stage_price("shared-pole ledger", row['aggregate_bytes_per_rank'])
        return copy.deepcopy(row)

    def preview(self, *, resident_bytes_per_rank,
                workspace_bytes_per_rank, concurrent_with=()):
        """Price one reservation without recording it or making refusal fatal.

        Constructor execution uses this before any bank read to choose one
        complete-mesh route for an oversized parent. The subsequent actual
        phase still calls :meth:`reserve` and creates the evidence row.
        """
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
        available = max(0, self.device_budget_bytes_per_rank - inherited
                        - self.reserve_bytes_per_rank)
        passed = total <= available
        g = self.geometry
        max_ranks = math.floor(self.limit_bytes_per_rank * g['px'] * g['py'] / total) if total else None
        reason = ('actual-batch analytical admission; peak not measured' if passed else
                  f"aggregate {total} B/rank exceeds available device budget {available} B/rank; "
                  f"geometry Q={g['nq']}, spin={g['nspinor']}, mu={g['nmu']}, "
                  f"Px={g['px']}, Py={g['py']}; the 3U scaling target corresponds to "
                  f"{max_ranks} ranks at this fixed byte envelope, which is not a "
                  "device-fit recommendation; reprice actual batches/workspaces "
                  "for any changed geometry, or reduce concurrent live bytes")
        row = gate_receipt('capacity', total / self.U_bytes_per_rank,
                           passed=scaling_passed, reason=reason)
        row['status'] = 'FAIL' if not passed else ('PASS' if scaling_passed else 'WARN')
        if passed and not scaling_passed:
            row['reason'] = 'above 3U scaling target; admitted within device budget (coordinator ruling24); peak not measured'
        row.update(resident_bytes_per_rank=resident,
                   workspace_bytes_per_rank=workspace,
                   aggregate_bytes_per_rank=total,
                   limit_bytes_per_rank=self.limit_bytes_per_rank,
                   device_budget_bytes_per_rank=self.device_budget_bytes_per_rank,
                   inherited_peak_bytes_per_rank=inherited,
                   runtime_reserve_bytes_per_rank=self.reserve_bytes_per_rank,
                   available_device_bytes_per_rank=available,
                   device_budget_status='PASS' if passed else 'FAIL',
                   concurrent_with=sorted(live), live_stages=sorted(live),
                   geometry=dict(g), max_mesh_ranks_at_fixed_bytes=max_ranks)
        return copy.deepcopy(row)

    def room_bytes_per_rank(self, concurrent_with=()):
        """What a new reservation beside ``concurrent_with`` may still take: the device
        budget less the inherited peak and the named live stages (never negative).  A
        planner sizes its batch from this, then reserves the batch it chose."""
        row = self.preview(resident_bytes_per_rank=0, workspace_bytes_per_rank=0,
                           concurrent_with=concurrent_with)
        return max(0, int(row['available_device_bytes_per_rank'])
                   - int(row['aggregate_bytes_per_rank']))

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

    def receipt(self, *, entry_start=None):
        """Snapshot stage rows and peaks, optionally as an indexed ledger segment.

        A segment retains absolute entry indices so ordered constructor receipts
        reconstruct the prefix without duplicating every earlier reservation.
        The default remains the complete ledger snapshot.
        """
        if entry_start is not None:
            if isinstance(entry_start, bool):
                raise ValueError("capacity entry_start must be an integer index")
            entry_start = operator.index(entry_start)
            if not 0 <= entry_start <= len(self.entries):
                raise ValueError("capacity entry_start lies outside the ledger")
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


def construction_receipt(measurements=None, *, capacity=None, capacity_entry_start=None,
                         ordered=False, charge4=False):
    """Complete receipt skeleton, explicitly marking every absent gate unmeasured.

    ``measurements`` maps names to ``gate_receipt`` keyword dictionaries. Dense
    consumers supply measurements; creating this skeleton certifies no physics.
    """
    measurements = {} if measurements is None else measurements
    table = (charge4_gates(ordered) if charge4 else
             shared_real_pole_gates_ordered_v1 if ordered else shared_real_pole_gates_v1_r3b)
    unknown = set(measurements) - set(table)
    if unknown:
        raise ValueError(f"unknown shared-pole receipt predicates: {sorted(unknown)}")
    result = {"schema": RECEIPT_SCHEMA, "recipe_version": RECIPE_VERSION,
            "recipe_hash": RECIPE_HASH,
            "gate_version": (CHARGE4_GATE_VERSION if charge4 else
                             ORDERED_GATE_VERSION if ordered else GATE_VERSION),
            "gate_hash": (table_hash(table) if charge4 else
                          ORDERED_GATE_HASH if ordered else GATE_HASH),
            "gates": [gate_receipt(name, table=table, **measurements.get(
                name, {"reason": "measurement not supplied"}))
                for name in table]}
    if capacity is not None:
        if not isinstance(capacity, CapacityLedger):
            raise TypeError("construction receipt capacity must be the map's CapacityLedger")
        # The segment is named by its ledger indices, not copied: every parent
        # of a round shares it, and the map's ledger receipt (the constructor's
        # ``capacity``) holds each row once.  A copy per parent was 35 MB of
        # the 484 MB constructor receipt at Ni 20^3 (1062 parents, 874 rows).
        result['capacity'] = capacity.receipt(entry_start=len(capacity.entries))
        result['capacity']['entry_span'] = [int(capacity_entry_start or 0), len(capacity.entries)]
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
            'capacity', aggregate, table=table, passed=(all(r['status'] == 'PASS' for r in rows)
                                           and measured['status'] != 'FAIL'),
            reason='plan-time ledger rows; independent measured peak recorded separately'))
        if (result['gates'][-1]['status'] == 'FAIL'
                and all(r['status'] != 'FAIL' for r in rows)
                and measured['status'] != 'FAIL'):
            result['gates'][-1]['status'] = 'WARN'
            result['gates'][-1]['reason'] = '3U scaling target exceeded; device-budget admissions passed (coordinator ruling24)'
    return result


def build_construction_row(model, counts, diagnostics, *, span, roles, price,
                           coulomb, native_queries, identity, gates, nspinor,
                           logical_n, ordered, odd_moments, detail=False):
    """Host view of one constructed parent, and the gate rows it measures.

    The constructor has finished q: it holds the model, the device diagnostics
    the reduction and the checks produced, and the host lists from the held-W
    and moment stages. This turns those into the two JSON objects the receipt
    needs -- the per-q ``constructor`` block and the ``measurements`` mapping
    :func:`construction_receipt` consumes -- so that the driver above it is the
    physics and this is the bookkeeping.

    Parameters
    ----------
    model : tuple
        ``(b, poles2, active)`` as exported, after the zero policy and the sort.
    counts : array
        Active pole count per parent in this row, int64 [b].
    diagnostics : mapping
        ``reduction``, ``zero``, ``passive``, ``retained``, ``moment_defects``
        from the reduction and the gates, and the host ``held``,
        ``reciprocity`` and ``permutation`` records. ``reduction`` is at the
        parent's own extent (``gw.shared_pole_local.own_extent_receipts``).
    price, coulomb, native_queries, identity : mapping / iterable
        The capacity row for this parent, the Coulomb receipt, the constructor's
        native workspace queries, and the current state identity.
    gates : mapping
        The canonical TRS or ordered gate table, already selected by the caller.
    nspinor, logical_n : int
        Deck spin count and logical centroid count.
    ordered, odd_moments : bool
        The route and whether its bank carried the odd z-moments.
    detail : bool
        ``sigma_freq_debug_output``: keep the whole normalized Gram spectrum
        and the column sort permutation.  Otherwise the spectrum is its size,
        ends and exact-zero count, and the permutation is omitted; at Ni 20^3
        the two were 140 MB of the per-parent rows, encoded on every map.

    Returns
    -------
    (row, measurements)
    """
    _, poles, mask = model
    reduction = diagnostics["reduction"]
    zero, passive = diagnostics["zero"], diagnostics["passive"]
    retained, moment_defects = diagnostics["retained"], diagnostics["moment_defects"]
    held, reciprocity = diagnostics["held"], diagnostics["reciprocity"]
    spectrum = np.asarray(reduction["gram_spectrum_relative"])
    if not detail:
        spectrum = {"size": int(spectrum.shape[-1]), "zeros": int(np.sum(spectrum == 0)),
                    "min": float(spectrum.min(initial=np.inf)),
                    "max": float(spectrum.max(initial=-np.inf))}
    row = {"q_span": list(span), "roles": roles,
           "diagnostic_operator": "raw-latent-pole-model",
           "K": np.asarray(counts).tolist(), "J": int(np.unique(np.asarray(poles)[np.asarray(mask)]).size),
           "damping_fraction": 0.0, "capacity": price, "coulomb": coulomb,
           "condition": np.asarray(reduction["gram_condition"]).tolist(),
           "normalized_gram_spectrum": spectrum.tolist() if detail else spectrum,
           "native_workspace_queries": [dict(op=op, shapes=shapes, bytes_per_rank=value)
                                         for (op, shapes), value in native_queries.items()],
           "retained_moment_relative": {k: np.asarray(v).tolist() for k, v in retained.items()},
           "moment_defects": {k: {a: np.asarray(value).tolist() for a, value in v.items()}
                              for k, v in moment_defects.items()},
           "held_W": held, "storage_bytes": int(counts[0]) * (16*logical_n + 8)}
    if detail:
        row["permutation"] = np.asarray(diagnostics["permutation"]).tolist()
    if ordered:
        row["ordered"] = {key: np.asarray(reduction[key]).tolist() for key in (
            "positive_count", "negative_count", "infinite_weight_fraction",
            "paired_rank", "paired_min_relative")}
        row["ordered"]["odd_moments"] = odd_moments
    row["metric_inverse_root"] = {
        name: np.asarray(reduction[name]).tolist() for name in (
            "metric_initial_infinity_norm", "metric_inverse_root_iterations",
            "metric_inverse_root_residual_fro", "metric_inverse_root_residual_relative")}
    representation_value = ({"nspinor": nspinor, "trs_allowed": False, "ordered": True}
                            if ordered else {"nspinor": nspinor, "trs_allowed": True})
    # The reciprocity row compares nothing on a sample whose held reference is
    # not itself transpose symmetric, so the receipt must say how many records
    # were actually evaluated instead of asserting a literal pass (INVARIANTS 23).
    recip_applicable = np.asarray(reciprocity.get("applicable", []), dtype=bool).ravel()
    measurements = {
        "normalized_gram_keep": dict(value=int(reduction["retained_rank"][0]), passed=True, reason="normalized Gram cut, current q"),
        "normalized_gram_validity": dict(value=float(reduction["gram_min_relative"][0]), passed=True, reason="normalized Gram spectrum"),
        "gram_rounding_validity": dict(value={"gram_min_relative": float(reduction["gram_min_relative"][0]),
                                              "floor_relative": float(reduction["gram_floor_relative"][0])},
                                       passed=True, reason="propagated float64 floor of the equilibrated Gram"),
        "zero_ritz_policy": dict(value=float(zero["dropped_factor_weight_fraction"][0]), passed=True, reason="physical factor weight, sentinels excluded"),
        "finite_factors_poles": dict(value=True, passed=True, reason="zero policy, active prefix and exact inert sentinels"),
        "passivity": dict(value={k: np.asarray(v).tolist() for k, v in passive.items() if k != "passivity"}, passed=True, reason=("signed particle-hole model, Hermitian part at i eta; anti-Hermitian part is the odd channel, reported" if ordered else "raw latent model; authenticated inverse Coulomb square root at current eta; projected operator not measured")),
        "retained_subspace_moments": dict(value=row["retained_moment_relative"], passed=retained_moment_row_passed(row["retained_moment_relative"], gates["retained_subspace_moments"]), reason=(("signed model z-moments m0..m3 on the original infinity directions, each order against its own norm; projection-accuracy diagnostic beside full_m1/full_m3, not a refusal" if odd_moments else "finite-state ordered bank without odd moments: infinity block uncertified") if ordered else "raw latent Ritz identity: A=Y†GE, B=YA; pencil B†(G,H)B/2 versus model A†(I,Lambda)A/2")),
        "held_w": dict(value=held, passed=True, reason="raw latent W and dW/ds diagnostics; projected operator not measured; no universal acceptance threshold"),
        "model_reciprocity": (dict(value=None, passed=None, reason="not applicable: time-reversal-broken samples carry no transpose symmetry") if ordered else dict(value=reciprocity, passed=reciprocity_row_verdict(reciprocity), reason=f"raw latent model sampled W/dW transpose symmetry, conditional on a symmetric reference: {int(recip_applicable.sum())} of {int(recip_applicable.size)} held records evaluated at reference_relative_max={gates['model_reciprocity']['threshold']['reference_relative_max']:g}; NOT_MEASURED when none was; projected operator not measured")),
        "full_m1_defect": dict(value=float(moment_defects["M1"]["full_relative"][0]), passed=bool(moment_defects["M1"]["full_relative"][0] <= gates["full_m1_defect"]["threshold"]), reason="raw latent model versus physical full M1; projected moment not measured; CD8 diagnostic band, never a refusal"),
        "full_m3_defect": dict(value=float(moment_defects["M3"]["full_relative"][0]), passed=bool(moment_defects["M3"]["full_relative"][0] <= gates["full_m3_defect"]["threshold"]), reason="raw latent model versus physical full M3; projected moment not measured; CD8 diagnostic band, never a refusal"),
        "representation": dict(value=representation_value, passed=representation_row_passed(representation_value, gates["representation"]["threshold"]), reason="current typed symmetry capability against the gate threshold"),
        "capacity": dict(value=price, passed=True, reason="conservative aggregate constructor live-set price"),
        "sc_rebuild": dict(value=identity, passed=True, reason="current recipe/census authenticated; directions and Ritz model rebuilt"),
    }
    return row, measurements


def bind_shared_pole_sc_identity(meta, state, *, occupation_state, print_fn):
    """Label current SC scratch, without claiming QP provenance (ruling 22).

    ``state`` supplies the current iteration. Reuse the carried occupation
    digest or the already-bound insulating census digest; compute no new hash.
    These labels MUST NOT authenticate restart membership or skip construction.
    """

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


def active_band_mask(energies_kn_ry, mu_ry, recipe=None, *, valid_kn=None):
    """Bands the W model treats as dynamically active: ``max_k E_nk >= mu -
    active_depth_ev``.  Deeper bands (semicore) carry no plasma charge.

    The one owner of that rule: the shared-pole census counts the plasma
    charge with it, and the SC ``cover`` policy grows the Sigma grid only
    over identities it calls active (``sc_iteration._sc_sampled_support``).
    """
    from common.units import RYD_TO_EV
    recipe = shared_real_pole_v1_r3b if recipe is None else recipe
    energies = np.asarray(energies_kn_ry, dtype=np.float64)
    if valid_kn is not None:
        from .wavefunction_bundle import _validate_valid_kn
        _validate_valid_kn(energies, valid_kn)
        energies = np.where(np.asarray(valid_kn, bool), energies, -np.inf)
    depth = (float(mu_ry) - np.max(energies, axis=0)) * RYD_TO_EV
    return depth <= recipe['active_depth_ev']


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
    from common.units import RYD_TO_EV

    stop = wfns.slices.b4_logical - wfns.slices.b0
    energies = np.asarray(wfns.enk, dtype=np.float64)[:, :stop]
    occupations = np.asarray(wfns.occ if occupation_state is None else
                             occupation_state.f_kn, dtype=np.float64)[:, :stop]
    valid = None
    if getattr(wfns, "valid_kn", None) is not None:
        from .wavefunction_bundle import physical_band_mask
        valid = physical_band_mask(wfns, occupations=(None if occupation_state is None else occupation_state.f_kn))[:, :stop]
    if (energies.shape != occupations.shape or energies.ndim != 2
            or energies.shape[0] != meta.nk_tot or not energies.size
            or not np.all(np.isfinite(energies))
            or not np.all(np.isfinite(occupations))):
        raise ValueError("GATE shared_pole_census: got: inconsistent/nonfinite energy or occupation table; want: current full-BZ logical tables; why: physical charge needs authenticated weights")
    from file_io.shared_pole_store import charge_representation
    # trs_allowed is recorded, not gated: a measured break selects the ordered
    # bank and model, and consumers that need the even form refuse by name.
    bispinor = int(meta.nspinor) == 4 and int(meta.nspinor_wfnfile) == 2
    if not charge_representation(meta) or int(meta.nspin) != 1:
        raise ValueError("GATE shared_pole_representation: want one scalar, spinor or authenticated bispinor state")
    capacity = float(state_capacity)
    physical_spinor = int(meta.nspinor_wfnfile) if bispinor else int(meta.nspinor)
    if capacity * physical_spinor != 2.0:
        raise ValueError("GATE shared_pole_census: got: state capacity times Nspinor other than 2; want: authenticated spin-restricted capacity; why: charge normalization")
    val = energies[:, wfns.slices.val]
    cond = energies[:, wfns.slices.cond_all_logical]
    response_cond = energies[:, wfns.slices.cond]
    if valid is not None:
        val = val[valid[:, wfns.slices.val]]
        cond = cond[valid[:, wfns.slices.cond_all_logical]]
        response_cond = response_cond[valid[:, wfns.slices.cond]]
    if not val.size or not cond.size or not response_cond.size:
        raise ValueError("GATE shared_pole_gap: got: empty logical valence/conduction window; want: both nonempty; why: support geometry needs a physical gap")
    vbm, cbm = float(np.max(val)), float(np.min(cond))
    mu = (0.5 * (vbm + cbm) if occupation_state is None
          else float(occupation_state.mu_ry))
    gap_ev = max(0.0, (cbm - vbm) * RYD_TO_EV)
    partial = (occupations > 0.0) & (occupations < 1.0)
    if valid is not None:
        partial &= valid
    weights = np.asarray(kweights, dtype=np.float64)
    if (weights.shape != (meta.nk_tot,) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("GATE shared_pole_kweights: got: invalid physical weights; want: finite nonnegative full-BZ weights summing to one; why: plasma charge normalization")
    # A partially occupied BAND must cross mu across the k census; smearing
    # tails in a gapped band do not authorize metallic line spacing.
    crossing = (np.min(energies, axis=0) <= mu) & (np.max(energies, axis=0) >= mu)
    top = np.max(energies, axis=0)
    if valid is not None:
        top = np.where(valid, energies, -np.inf).max(axis=0)
        bottom = np.where(valid, energies, np.inf).min(axis=0)
        crossing = (bottom <= mu) & (top >= mu)
    partial_at_mu = bool(np.any(np.any(partial, axis=0) & crossing))
    depth = (mu - top) * RYD_TO_EV
    recipe = shared_real_pole_v1_r3b
    active = active_band_mask(energies, mu, valid_kn=valid)
    borderline = ((depth > recipe['active_depth_ev'])
                  & (depth <= recipe['borderline_depth_ev']))
    electrons = float(capacity * np.sum(weights[:, None] * np.where(active, occupations, 0.0)))
    volume = float(meta.cell_volume)
    if not (math.isfinite(mu) and math.isfinite(volume) and volume > 0
            and math.isfinite(electrons) and electrons > 0):
        raise ValueError("GATE shared_pole_plasma: got: nonpositive/nonfinite active charge or cell volume; want: positive finite electrons and bohr^3; why: omega_p requires positive density")
    meta.shared_pole_census = {
        "mu_ry": mu, "gap_ev": gap_ev, "partial_at_mu": partial_at_mu,
        "energy_span_ry": float(energies.max()-energies.min()) if valid is None else float(energies[valid].max()-energies[valid].min()),
        "response_transition_span_ry": float(response_cond.max()-val.min()),
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
    if valid is not None:
        meta.shared_pole_census.update(
            valid_kn_sha256=hashlib.sha256(np.asarray(wfns.valid_kn, bool).tobytes()).hexdigest(),
            physical_bands_by_k=valid.sum(axis=1).tolist())


def _support_envelope(required, key, session):
    """Keep a run-local scalar enclosure; never retain samples or a census.

    The reference map uses its current interval without retaining it. The
    first interacting map seeds the enclosure: an initial DFT gap can be
    substantially smaller and select a different imaginary support count.
    Increasing L/u_min is conservative for the imaginary-count heuristic.
    This is a sampling-geometry enclosure, not an interpolation-error bound;
    the bank still certifies its current energies at every supplied frequency.
    """
    version = "sc_interacting_support_enclosure_20260910"
    scope = "sampling geometry only; interpolation accuracy not certified"
    if not session.get("reference_complete", False):
        session["reference_complete"] = True
        return dict(version=version, status="initial_reference", epoch=-1,
                    required=dict(required), retained=dict(required), scope=scope)
    from gw.sigma_box_plan import snap_outward
    previous = session.get("envelope")
    same_policy = previous is not None and session.get("key") == key
    # Compared and retained on the snap grid: 1 ulp of the census charge
    # (imaginary_top) otherwise flips hit/expanded, hence the SC sample sites.
    envelope = {k: snap_outward(v, 1.0, -1 if k == "u_min_ev" else +1)
                for k, v in required.items()}
    if same_policy:
        envelope = {
            "imaginary_top_ev": max(previous["imaginary_top_ev"], envelope["imaginary_top_ev"]),
            "u_min_ev": min(previous["u_min_ev"], envelope["u_min_ev"]),
            "u_max_ev": max(previous["u_max_ev"], envelope["u_max_ev"]),
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


def _sector_treatment_ceiling(response_span_ry, session):
    """Hold the bispinor pole treatment ceiling while the current span is inside it.

    Twice the current chi transition span is an explicit approximation
    policy. It is not a collective-mode bound. A fixed-SC session holds the
    ceiling of its plan while the current span stays within the held span
    (current candidate <= held ceiling); a map whose span exceeds it re-plans
    the ceiling at twice the current span and holds that (``replanned``).
    The constructor applies the held ceiling to every current model.
    """
    required = 2.0 * float(response_span_ry)
    if not math.isfinite(required) or required <= 0.0:
        raise ValueError(
            "GATE shared_pole_treatment_ceiling: current response span must be "
            f"finite and positive, got {response_span_ry!r}")
    if session is None:
        return dict(version="sector_twice_map_span_v1", status="current_map",
                    ceiling_ry=required,
                    source_response_span_ry=float(response_span_ry),
                    current_response_span_ry=float(response_span_ry),
                    current_candidate_ceiling_ry=required,
                    scope="numerical treatment; not a physical pole bound")
    previous = session.get("sector_treatment_ceiling_ry")
    if previous is None:
        session["sector_treatment_ceiling_ry"] = required
        status = "initialized_map0"
    elif required > float(previous):
        session["sector_treatment_ceiling_ry"] = required
        status = "replanned"
    else:
        status = "reused_map0"
    return dict(version="sector_twice_map_span_v1", status=status,
                ceiling_ry=float(session["sector_treatment_ceiling_ry"]),
                source_response_span_ry=(
                    0.5 * float(session["sector_treatment_ceiling_ry"])),
                current_response_span_ry=float(response_span_ry),
                current_candidate_ceiling_ry=required,
                scope="numerical treatment; not a physical pole bound")


def parse_support_sites(text):
    """Parse ``sigma_w_support_sites_ev`` into the two explicit site ladders.

    ``""`` is the production ladder: this returns ``None`` and the resolver's
    own rule stands untouched. Any other value spells both ladders in eV as
    ``"<line list> | <imaginary list>"``, replacing the 2*eta/4*eta line rule
    and the Zolotarev imaginary geometric ladder together. Sites are strictly
    increasing; line sites are >= 0 and imaginary sites > 0, with at least two
    imaginary sites so the held geometric midpoints stay defined. The height
    (4*eta), the held fractions, the widths and every gate are unchanged, so an
    override run is the same construction on a different support set.
    """
    text = str(text).strip()
    if not text:
        return None
    parts = text.split('|')
    if len(parts) != 2:
        raise ValueError(f"GATE shared_pole_support_sites: got: {text!r}; want: '<line eV list> | <imaginary eV list>'; why: the override replaces both ladders or neither")

    def sites(raw, name, floor, minimum):
        try:
            values = [float(v) for v in raw.replace(',', ' ').split()]
        except ValueError:
            raise ValueError(f"GATE shared_pole_support_sites: got: {raw!r} for {name}; want: comma-separated eV numbers; why: the ladder is an explicit site list") from None
        if len(values) < minimum:
            raise ValueError(f"GATE shared_pole_support_sites: got: {len(values)} {name} sites; want: at least {minimum}; why: the held midpoints need adjacent pairs")
        if any(not math.isfinite(v) or v < floor for v in values):
            raise ValueError(f"GATE shared_pole_support_sites: got: {values} for {name}; want: finite sites >= {floor} eV; why: a support off the causal quadrant has no bank evaluation")
        if any(b <= a for a, b in zip(values, values[1:])):
            raise ValueError(f"GATE shared_pole_support_sites: got: {values} for {name}; want: strictly increasing; why: repeated sites collapse the pencil")
        return values

    line = sites(parts[0], 'line', 0.0, 2)
    imaginary = sites(parts[1], 'imaginary', 0.0, 2)
    if imaginary[0] <= 0.0:
        raise ValueError(f"GATE shared_pole_support_sites: got: imaginary site {imaginary[0]} eV; want: > 0; why: the imaginary ladder is geometric")
    # repr round-trips a double exactly, so the canonical text that enters
    # recipe_hash is the geometry the stream samples, to the last bit.
    canonical = (','.join(repr(v) for v in line) + '|'
                 + ','.join(repr(v) for v in imaginary))
    return {'line_ev': line, 'imaginary_ev': imaginary, 'text': canonical}


def imaginary_sample_count(kappa, tier, recipe=shared_real_pole_v1_r3b):
    """Tier-owned sample count on a log-spaced imaginary ladder of span ``kappa`` (max/min).

    Production: the rational-approximation count
    ``max(min_count, round(log(16 kappa^2) log(4/eps) / (2 pi^2)))``; relaxed:
    the tier's fixed count.  One owner for the shared-pole imaginary ladder and
    the Matsubara index set.
    """
    if tier == 'production':
        return max(recipe['imaginary_min_count'], round(
            math.log(16 * kappa**2) * math.log(4 / recipe['imaginary_count_epsilon'])
            / (2 * math.pi**2)))
    return recipe[tier]['imaginary_count']


#: Held SC line sites are replaced when the support rule would move any site
#: by more than this (eV) for the current bands.  The Fe 4^3 bispinor SC end
#: point moves about 0.3 meV per meV of held-site offset (sites frozen at map 2
#: of a DFT start sat 0.32 eV off the rule's sites at the fixed point and left
#: E_F +- 10 eV states 99 meV from a fresh plan's end point; 0.003 eV gave
#: 0.9 meV; REPLAN, runs/Fe/67-71, 2026-09-29), so 3 meV keeps it near 1 meV.
#: The hold is max(this, a tenth of the SC's last max|dE|): site moves an
#: order below the iterate's own error only make each map a different
#: function, and Anderson then mixes residuals of different maps (Fe 4^3
#: scalar parallel_transport: 10 re-plans in 15 maps, STALLED; FESTALL,
#: runs/Fe/82, 2026-09-30).
LINE_SITE_HOLD_EV = 3.0e-3


def support_line_count(reads_ev, height_ev, imaginary_count, held_count,
                       recipe=shared_real_pole_v1_r3b, tier='production'):
    """Production line-site count for the requested reads: the strip law, clamped.

    W(x + i h) is analytic in a strip of half-width h about the sampling line, so sites
    spaced by Delta alias at most e^(-pi h / Delta): spacing pi h / ln(4/eps) keeps that
    below eps/4, eps the recipe's imaginary_count_epsilon (one sampling accuracy for both
    ladders).  Over the reach Omega_R = max |read - mu| that is
    n = ceil((Omega_R - h) ln(4/eps) / (pi h)) sites, held between today's floor
    (fitted_support_count - imaginary_count) and the sample cap (sample_cap - imaginary
    - held).  Closed form, no fit (owner 2026-09-26).  Returns (count, receipt).
    """
    policy = recipe[tier]
    reach = float(np.max(np.abs(reads_ev))) if np.size(reads_ev) else 0.0
    eps, h = recipe['imaginary_count_epsilon'], float(height_ev)
    strip = math.ceil(max(reach - h, 0.0) * math.log(4.0 / eps) / (math.pi * h))
    floor = int(policy['fitted_support_count']) - int(imaginary_count)
    cap = int(policy['sample_cap']) - int(imaginary_count) - int(held_count)
    count = max(floor, min(strip, cap))
    return count, dict(reach_ev=reach, strip=int(strip), floor=floor, cap=cap, count=count)


def support_rule_line_sites(energies_ev, mu_ev, eta_ev, height_ev, reads_ev, count,
                            recipe=shared_real_pole_v1_r3b, grid=4001, *, valid_kn=None):
    """Line sites from the band structure alone: the support rule (report section IV.B).

    Sigma evaluates W on the line at the crossings |E - eps|: E a read, the energy (eV,
    relative to ``mu_ev``) at which a requested state reads Sigma
    (``qp_support.requested_reads_ev``), and eps any level, at any k (k - q spans the
    zone), strictly between mu and E. ``energies_ev`` [k, bands]. The crossing density
    rho broadens each crossing at ``eta_ev``; ``count`` sites sit at equal quantiles of
    rho**alpha on [omega_lo, omega_reach], omega_reach the largest crossing and omega_lo
    the fixed point max(height, first spacing). Crossings at or below the height count
    as none; with none rho is flat on [omega_lo, max(|reads|, 2 height)]. Distances from
    mu are binned at 0.01 eV (far below eta) and paired by one correlation per side of
    mu; sites land on a grid of ``grid`` points. Returns strictly increasing sites in eV.
    """
    energies = np.asarray(energies_ev, dtype=np.float64)
    if valid_kn is not None:
        from .wavefunction_bundle import _validate_valid_kn
        _validate_valid_kn(energies, valid_kn)
        energies = energies[np.asarray(valid_kn, bool)]
    levels = energies.ravel() - mu_ev
    evaluation, delta = np.asarray(reads_ev, dtype=np.float64).ravel(), 0.01
    pairs = np.zeros(1)
    for side in (1.0, -1.0):
        e, eps = side * evaluation, side * levels
        e, eps = e[e > 0], eps[eps > 0]
        if not e.size:
            continue
        n = int(np.rint(e.max() / delta)) + 1
        he = np.bincount(np.rint(e / delta).astype(np.int64), minlength=n)
        hs = np.bincount(np.rint(eps[eps <= e.max()] / delta).astype(np.int64), minlength=n)[:n]
        lagged = np.correlate(he, hs, 'full')[n - 1:]            # [l] = sum_i he[i] hs[i - l]
        pairs = np.pad(pairs, (0, max(0, n - pairs.size))) + np.pad(lagged, (0, max(0, pairs.size - n)))
    pairs[:int(np.floor(height_ev / delta)) + 1] = 0
    lags = np.flatnonzero(pairs)
    flat = max(float(np.max(np.abs(evaluation))) if evaluation.size else 0.0, 2.0 * float(height_ev))
    reach = float(lags[-1] * delta) if lags.size else flat
    x = np.linspace(0.0, reach, grid)
    rho = (eta_ev / math.pi * (pairs[lags] / ((x[:, None] - lags * delta) ** 2 + eta_ev ** 2)).sum(axis=1)
           ) ** recipe['support_density_power'] if lags.size else np.ones(grid)
    lo = float(height_ev)
    for _ in range(200):
        k = x >= lo - 1e-12
        c = np.concatenate([[0.0], np.cumsum(np.diff(x[k]) * 0.5 * (rho[k][1:] + rho[k][:-1]))])
        sites = np.interp(np.linspace(0.0, c[-1], int(count)), c, x[k])
        lo, previous = max(float(height_ev), float(sites[1] - sites[0])), lo
        if abs(lo - previous) < 1e-10:
            break
    return sites


def matsubara_indices(beta_ry_inv, bandwidth_ry, tier, recipe=shared_real_pole_v1_r3b):
    """Bosonic Matsubara indices at this accuracy tier: 0 plus a log-spaced ladder up to the bandwidth.

    ``nu_n = 2 pi n / beta``.  The ladder spans ``nu_1 .. nu_top`` with
    ``n_top = ceil(bandwidth beta / 2 pi)`` (frequencies above the bandwidth are
    the exact M1/M3 moment tail) and carries :func:`imaginary_sample_count`
    distinct indices for span ``kappa = n_top``.  No deck key: beta is the
    Fermi-Dirac width's inverse and the count is the tier's.
    """
    beta, width = float(beta_ry_inv), float(bandwidth_ry)
    if not (math.isfinite(beta) and beta > 0.0 and math.isfinite(width) and width > 0.0):
        raise ValueError(
            f"GATE matsubara_indices: got beta={beta_ry_inv!r}, bandwidth={bandwidth_ry!r}; "
            "want finite positive values; why: the ladder spans nu_1 .. bandwidth")
    n_top = max(1, math.ceil(width * beta / (2.0 * math.pi)))
    count = imaginary_sample_count(float(n_top), tier, recipe) if n_top > 1 else 1
    positive = np.unique(np.rint(np.geomspace(1.0, float(n_top), max(1, count))).astype(np.int64))
    return np.concatenate(([0], positive)).astype(np.int64)


def resolve_shared_pole_recipe(config, wfns, meta, *, mesh_xy, print_fn,
                              support_reads_ev, support_session=None):
    """Resolve DESIGN §5 from current metadata into scalars and small arrays.

    ``bind_shared_pole_census`` must have consumed this map's occupation state.
    Returns a plain dict with equal-length ``z_ry`` complex128, ``role`` int8,
    ``distinct_id`` int64 and ``held`` bool arrays [role]. Repeated physical
    points share a distinct_id; fitting and held IDs are disjoint. Infinity's
    role code is reserved (moments need no bank evaluation). Conjugates are
    constructor states, never bank calls. ``support_pair`` int64 [role,2]
    binds held endpoints; [-1,-1] means not a held midpoint.
    ``support_reads_ev`` are the absolute energies (eV) at which this map's requested
    states read Sigma (``qp_support.requested_reads_ev`` plus the caller's frame mu):
    the line sites are placed over them and their count follows ``support_line_count``.
    ``support_session`` optionally retains support bounds, the small line-site
    tuple and a policy/basis key across SC maps, after one unretained reference map.
    Enclosed current intervals retain the same points and roles, while
    the census and capacity ledger remain fresh.
    Expanding intervals enlarge the envelope; policy changes start a new one.
    All ranks execute the metadata work; only ``print_fn`` may filter by rank.
    """
    from common.units import RYD_TO_EV
    from .gw_config import (uses_bare_transverse_shared_pole,
                            uses_full_bispinor_shared_pole)

    if config.sigma.w_model != "shared_pole":
        return None
    charge4 = uses_bare_transverse_shared_pole(config)
    photon = uses_full_bispinor_shared_pole(config)
    if int(meta.nspinor) == 4 and not (charge4 or photon):
        raise ValueError("GATE shared_pole_representation: four-component shared-pole "
                         "response needs an explicit charge or full photon route")
    census = getattr(meta, "shared_pole_census", None)
    if census is None:
        raise ValueError("GATE shared_pole_census: got: absent current census; want: bind_shared_pole_census after current occupations; why: no guessed plasma charge or frozen recipe")
    stop = wfns.slices.b4_logical - wfns.slices.b0
    energies = np.asarray(wfns.enk, dtype=np.float64)[:, :stop]
    valid = None
    if getattr(wfns, "valid_kn", None) is not None:
        from .wavefunction_bundle import physical_band_mask
        valid = physical_band_mask(wfns)[:, :stop]
        valid_hash = hashlib.sha256(np.asarray(wfns.valid_kn, bool).tobytes()).hexdigest()
        if valid_hash != census.get('valid_kn_sha256'):
            raise ValueError("GATE shared_pole_census: stale physical-band validity; rebind the current census")
    if hashlib.sha256(energies.tobytes()).hexdigest() != census['energy_sha256']:
        raise ValueError("GATE shared_pole_census: got: stale energies; want: census rebound at current bands; why: SC must rebuild geometry")
    # The run's budget in decimal GB (was per_device_gb * 2**30: GiB, 7.4% over the deck).
    from common.gpu_utils import device_budget_bytes
    meta.shared_pole_capacity = CapacityLedger(
        meta, mesh_xy=mesh_xy, device_budget_bytes=int(device_budget_bytes()))
    recipe = shared_real_pole_v1_r3b
    tier = config.sigma.w_accuracy
    policy = recipe[tier]
    eta = float(config.sigma.regularization_ev)
    if not math.isfinite(eta) or eta <= 0:
        raise ValueError("GATE shared_pole_eta: got: invalid eta; want: finite positive sigma_regularization_ev; why: causal sampling height")
    height = max(recipe['height_floor_ev'], recipe['height_eta_factor'] * eta)
    override = parse_support_sites(getattr(config.sigma, 'w_support_sites_ev', ''))
    plasma_ry = 2.0 * math.sqrt(4.0 * math.pi * census['active_electrons']
                               / census['cell_volume_bohr3'])
    # L tops the imaginary ladder only (u_max, kappa); the line stops at the reads' reach.
    top = plasma_ry * RYD_TO_EV + recipe['plasma_margin_ev']
    umin = max(recipe['height_eta_factor'] * eta, census['gap_ev'])
    umax = max(recipe['imaginary_floor_max_ev'], top)
    reads_abs = np.asarray(support_reads_ev, dtype=np.float64).ravel()
    if not reads_abs.size or not np.isfinite(reads_abs).all():
        raise ValueError("GATE shared_pole_support_reads: got: no finite requested reads; want: qp_support.requested_reads_ev of this map's requested states; why: the line sites are placed over the energies Sigma reads")
    reads = reads_abs - census['mu_ry'] * RYD_TO_EV
    if umin >= umax:
        raise ValueError(f"GATE shared_pole_interval: got: u_min={umin} >= u_max={umax} eV; want: u_min < u_max; why: imaginary support interval is unresolved")
    support_receipt = None
    if support_session is not None:
        key = (RECIPE_HASH, tier, eta, int(meta.nspinor), int(meta.n_rmu),
               census['logical_band_count'],
               '' if override is None else override['text'])
        support_receipt = _support_envelope(
            dict(imaginary_top_ev=top, u_min_ev=umin, u_max_ev=umax), key,
            support_session)
        retained = support_receipt['retained']
        top, umin, umax = (retained['imaginary_top_ev'], retained['u_min_ev'],
                           retained['u_max_ev'])
    kappa = top / umin
    count = imaginary_sample_count(kappa, tier, recipe)
    imaginary = np.geomspace(umin, umax, count)
    line_count_rule = None
    if tier == 'relaxed':
        line = np.linspace(0.0, max(float(np.max(np.abs(reads))), 2.0 * height),
                           policy['line_count'])
    else:
        # The strip-law count over the reads' reach, placed by the support rule.
        held_count = len(recipe['held_line_fractions']) + np.unique(
            np.sqrt(imaginary[[0, -2]] * imaginary[[1, -1]])).size
        line_count, line_count_rule = support_line_count(reads, height, count, held_count, recipe, tier)
        line = support_rule_line_sites(energies * RYD_TO_EV, census['mu_ry'] * RYD_TO_EV, eta, height,
                                       reads, line_count, valid_kn=valid)
        if support_receipt is not None and support_receipt['status'] != 'initial_reference':
            previous_line = support_session.get('line_ev')
            if support_receipt['status'] == 'hit' and previous_line is not None:
                # The held sites stay valid while the support rule would place
                # the same sites for the current bands (s(E_now) = s_held).
                held = np.asarray(previous_line, dtype=np.float64)
                drift = (float(np.max(np.abs(line - held))) if line.shape == held.shape
                         else math.inf)
                support_receipt['line_site_drift_ev'] = drift
                hold = max(LINE_SITE_HOLD_EV, 0.1 * float(support_session.get('sc_residual_ev', 0.0)))
                if drift <= hold:
                    line = held
                else:
                    support_session['epoch'] += 1
                    support_receipt.update(status='expanded', epoch=support_session['epoch'])
                    worst = (int(np.argmax(np.abs(line - held)))
                             if line.shape == held.shape else -1)
                    support_receipt['line_site_replan'] = (
                        f"max site drift {drift:.4f} eV > {hold:.4f} eV "
                        f"(site {worst}: {held[worst] if worst >= 0 else float('nan'):.4f} -> "
                        f"{line[worst] if worst >= 0 else float('nan'):.4f} eV); "
                        f"epoch {support_session['epoch']}")
            # This is the existing SC sampling geometry, never W samples or a model.
            support_session['line_ev'] = tuple(float(v) for v in line)
    if override is not None:
        # Both ladders are replaced together; height, held fractions, widths,
        # zero policy and every gate stay the resolver's own. u_min/u_max/kappa
        # follow the delivered imaginary ladder so the reported scalars and the
        # sampled geometry cannot disagree.
        line = np.asarray(override['line_ev'], dtype=np.float64)
        imaginary = np.asarray(override['imaginary_ev'], dtype=np.float64)
        count = int(imaginary.size)
        umin, umax = float(imaginary[0]), float(imaginary[-1])
        kappa = top / umin
    mids = 0.5 * (line[:-1] + line[1:])
    held_pairs = [int(np.argmin(abs(mids - (line[0] + fraction*(line[-1] - line[0])))))
                  for fraction in recipe['held_line_fractions']]
    held_line = mids[held_pairs]
    held_imag = np.sqrt(imaginary[[0, -2]] * imaginary[[1, -1]])
    sector_treatment = None
    if photon:
        sector_treatment = _sector_treatment_ceiling(
            census['response_transition_span_ry'], support_session)
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
    # The operator is the spin-traced charge response [q, mu, mu] on scalar
    # and two-component decks alike (shared_pole_store.charge_representation).
    n = int(meta.n_rmu)
    # A support override is a different physical sampling geometry, so it must
    # be a different recipe identity: restart membership, the bank header and
    # the model identity all authenticate through these two fields, and an
    # unchanged hash would let a store built on one ladder be reused on another.
    version, table = RECIPE_VERSION, RECIPE_HASH
    if override is not None:
        version = RECIPE_VERSION + '+support_sites'
        table = hashlib.sha256(
            (RECIPE_HASH + '|' + override['text']).encode()).hexdigest()
    if sector_treatment is not None:
        version += '+sector_treatment'
        treatment_identity = {key: sector_treatment[key] for key in (
            'version','ceiling_ry','source_response_span_ry','scope')}
        table = hashlib.sha256((table + '|' + json.dumps(
            treatment_identity, sort_keys=True,
            separators=(',', ':'))).encode()).hexdigest()
    if charge4:
        version += '+charge4'
        table = hashlib.sha256((table + '|four-component-spin-traced-v1').encode()).hexdigest()
    gate_table = (charge4_gates(not census['trs_allowed']) if charge4 else
                  shared_real_pole_gates_v1_r3b if census['trs_allowed'] else
                  shared_real_pole_gates_ordered_v1)
    result = {
        'role_codes': dict(ROLE_CODES),
        'recipe_version': version, 'recipe_hash': table,
        'support_sites_override': '' if override is None else override['text'],
        'gate_version': (CHARGE4_GATE_VERSION if charge4 else
                         GATE_VERSION if census['trs_allowed'] else ORDERED_GATE_VERSION),
        'gate_hash': table_hash(gate_table),
        'accuracy': tier, 'accuracy_status': 'NOT_MEASURED',
        'accuracy_reason': 'resolved geometry has no authenticated matching campaign receipt',
        'eta_ev': eta, 'height_ev': height, 'height_ry': height / RYD_TO_EV,
        'plasma_ev': plasma_ry * RYD_TO_EV, 'plasma_ry': plasma_ry,
        'imaginary_top_ev': top,
        'line_ev': line, 'imaginary_ev': imaginary,
        # The Sigma planner reads these from the model header (gw.sigma_box_plan).
        'support_top_ev': float(line[-1]),
        'support_reads_ev': [float(reads_abs.min()), float(reads_abs.max())],
        'line_count_rule': line_count_rule,
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
        'line_direction_cap': (math.ceil(n * policy['line_direction_cap_fraction'])
                               if 'line_direction_cap_fraction' in policy else None),
        'pole_budget': (math.ceil(n * policy['pole_budget_fraction'])
                        if 'pole_budget_fraction' in policy else None),
        'multiplet_relative_tolerance': recipe['multiplet_relative_tolerance'],
        'bank_rule_tolerance': policy['bank_rule_tolerance'],
        'moment_convention': recipe['moment_convention'], 'census': dict(census),
        'operator_realization': recipe['operator_realization'],
        'U_bytes_per_rank': meta.shared_pole_capacity.U_bytes_per_rank,
    }
    if charge4:
        result['charge_operator'] = 'four-component-spin-traced-v1'
    if support_receipt is not None:
        result['support_envelope'] = support_receipt
    if sector_treatment is not None:
        result['sector_pole_treatment'] = sector_treatment
    result['metadata_array_bytes'] = sum(v.nbytes for v in result.values()
                                         if isinstance(v, np.ndarray))
    rules = {
        'support_sites': 'sigma_w_support_sites_ev; "" = the resolver ladder, else explicit line|imaginary eV sites folded into recipe_version/recipe_hash',
        'height': 'h=4*eta', 'eta': 'literal sigma_regularization_ev',
        'plasma': '2*sqrt(4*pi*active_electrons/volume) Ry',
        'imaginary_top': 'L=omega_p+3.5 eV (imaginary ladder only)',
        'line': 'production: strip-law count ceil((max|read-mu|-h) ln(4/eps)/(pi h)) within [18 - imaginary count, 40 - imaginary - held], quantiles of the crossing density of the requested reads; relaxed 8 sites on [0, max|read-mu|]',
        'support_top': 'largest line site, eV; read by the Sigma planner',
        'support_reads': 'absolute eV extent of the requested reads (E_in +/- the plan pad)',
        'line_direction_cap': 'production ceil(n/16) right singular directions per line support (whole multiplets); relaxed none',
        'pole_budget': 'production ceil(1.8 n) retained Gram directions per parent (largest first); relaxed none',
        'imaginary': 'log-spaced u_min..u_max; round(log(16*(L/u_min)^2)*log(4000)/(2*pi^2)), min2; tier width ceil(f*n)',
        'held_line': 'adjacent-support midpoint nearest 25%/65% of the line interval; lower-index tie',
        'held_imaginary': 'geometric midpoint of first/last adjacent imaginary pair',
        'u_min': 'max(h,logical gap)', 'u_max': 'max(16 eV,L)', 'kappa': 'L/u_min',
        'infinity': 'ceil(tier infinity fraction*n)', 'direction': 'tier relative singular cutoff',
        'multiplet': 'whole multiplets within relative 1e-6',
        'bank': 'fixed Hermite certificate tolerance 1e-8',
        'census': 'current full-band occupations, authenticated k weights/capacity; active band top >= mu-15 eV',
        'sector_pole_treatment': 'bispinor-only numerical treatment at twice the chi transition span, held while the current span stays inside it; not a physical pole bound',
        'U_bytes': '16*nk_full*(nspinor*nmu)^2/(Px*Py), logical bytes/rank',
        'metadata': 'sum of replicated metadata array nbytes',
    }
    if support_receipt is not None:
        rules.update(imaginary_top='SC high-water envelope of omega_p+3.5 eV',
                     u_min='SC low-water envelope of max(h,logical gap)',
                     u_max='SC high-water envelope of max(16 eV,L)',
                     support_envelope='current required bounds and retained sampling enclosure; not an interpolation-error certificate')
    for key, value in result.items():
        shown = value.tolist() if isinstance(value, np.ndarray) else value
        rule = next((v for prefix, v in rules.items() if key.startswith(prefix)),
                    'canonical role/ID census and versioned recipe; no accuracy inferred from missing evidence')
        print_fn(f'  [shared-pole recipe {RECIPE_VERSION}] {key}={shown} (rule: {rule})')
    return result
