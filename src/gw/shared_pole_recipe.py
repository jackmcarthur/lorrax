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
    "retained_subspace_moments": ("relative Q_inf-projected M1 and M3 defects after cut and zero policy <= threshold", 1.0e-10),
    "held_w_full_moment_defects": ("held W and full M1/M3 diagnostics with extracted CD8/CD10 values and receipt paths; missing extraction blocks landing", None),
    "representation": ("scalar N_spinor=1 and authenticated TRS allowed", {"nspinor": 1, "trs_allowed": True}),
    "capacity": ("aggregate live bytes per rank including workspace <= threshold * U", 3.0),
    "rule_validity": ("bank and Sigma certificates cover current domains at resolved tolerances", True),
    "sc_rebuild": ("samples, directions, poles, ranks, intervals and rules rebuilt at current bands and occupations", True),
}
shared_real_pole_gates_v1_r3b = {
    name: {"name": name, "predicate": predicate, "threshold": threshold,
           "version": GATE_VERSION}
    for name, (predicate, threshold) in _GATE_ROWS.items()
}


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
    row = shared_real_pole_gates_v1_r3b[name]
    status = "NOT_MEASURED" if value is None or passed is None else (
        "PASS" if passed else "FAIL")
    return {"predicate": row["predicate"], "name": name, "version": GATE_VERSION,
            "value": value, "threshold": row["threshold"], "status": status,
            "reason": reason}


def construction_receipt(measurements=None):
    """Complete receipt skeleton, explicitly marking every absent gate unmeasured.

    ``measurements`` maps names to ``gate_receipt`` keyword dictionaries. Dense
    consumers supply measurements; creating this skeleton certifies no physics.
    """
    measurements = {} if measurements is None else measurements
    unknown = set(measurements) - set(shared_real_pole_gates_v1_r3b)
    if unknown:
        raise ValueError(f"unknown shared-pole receipt predicates: {sorted(unknown)}")
    return {"schema": RECEIPT_SCHEMA, "recipe_version": RECIPE_VERSION,
            "recipe_hash": RECIPE_HASH, "gate_version": GATE_VERSION,
            "gate_hash": GATE_HASH,
            "gates": [gate_receipt(name, **measurements.get(
                name, {"reason": "measurement not supplied"}))
                for name in shared_real_pole_gates_v1_r3b]}


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
    Returns a plain dict: ``z_ry`` is complex128 [sample], ``roles`` names each
    tangent/held role and its physical sample ID, ``fit_ids``/``held_ids`` are
    disjoint physical IDs. Conjugates are constructor roles, never bank calls.
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
    points, roles = [], []
    def add(real, imag, role, held, pair=None):
        z = complex(real, imag) / RYD_TO_EV
        if z not in points:
            points.append(z)
        roles.append({'sample_id': points.index(z), 'role': role, 'held': held,
                      'value': True, 'derivative_s': True, 'support_pair': pair})
    for i, e in enumerate(line):
        add(e, height, f'line:{i}', False)
    for i, u in enumerate(imaginary):
        add(0.0, u, f'imaginary:{i}', False)
    for i, e in enumerate(held_line):
        j = held_pairs[i]
        add(e, height, f'held_line:{i}', True, [j, j+1])
    for i, u in enumerate(held_imag):
        add(0.0, u, f'held_imaginary:{i}', True,
            [0, 1] if i == 0 else [count-2, count-1])
    fit_ids = sorted({r['sample_id'] for r in roles if not r['held']})
    held_ids = sorted({r['sample_id'] for r in roles if r['held']})
    if set(fit_ids) & set(held_ids):
        raise ValueError("GATE shared_pole_held_exclusion: got: held/training collision; want: disjoint physical IDs; why: held diagnostics must be independent")
    n = int(meta.nspinor) * int(meta.n_rmu)
    result = {
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
        'z_ry': np.asarray(points, dtype=np.complex128), 'roles': roles,
        'fit_ids': np.asarray(fit_ids, dtype=np.int64),
        'held_ids': np.asarray(held_ids, dtype=np.int64),
        'line_count': len(line), 'imaginary_count': count,
        'unique_evaluations': len(points), 'fit_count': len(fit_ids),
        'held_count': len(held_ids), 'role_count': len(roles),
        'n': n, 'direction_cutoff': policy['direction_cutoff'],
        'imaginary_width': math.ceil(n * policy['imaginary_width_fraction']),
        'infinity_width': math.ceil(n * policy['infinity_width_fraction']),
        'multiplet_relative_tolerance': recipe['multiplet_relative_tolerance'],
        'bank_rule_tolerance': recipe['bank_rule_tolerance'],
        'sigma_tolerance': policy['sigma_tolerance'],
        'moment_convention': recipe['moment_convention'], 'census': dict(census),
        'U_bytes_per_rank': 16 * int(meta.nk_tot) * n*n / (
            int(mesh_xy.shape['x']) * int(mesh_xy.shape['y'])),
    }
    result['metadata_array_bytes'] = sum(v.nbytes for v in result.values()
                                         if isinstance(v, np.ndarray))
    for key, value in result.items():
        shown = value.tolist() if isinstance(value, np.ndarray) else value
        print_fn(f'  [shared-pole recipe {RECIPE_VERSION}] {key}={shown} (DESIGN §5; census/current bands, eta/tier; units in key)')
    return result
