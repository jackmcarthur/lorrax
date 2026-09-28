"""Response-bank exponential sums shared by groups of samples: derived nodes.

A group's rule represents, for every pole p of its samples (forward z and
reverse -conj z) and every real transition d in [lo, hi],

    1/(d - p) ~ sum_j c_j exp(-(d - r) T_j),   1/(d - p)^2 on the same T_j,

with one Green pair per complex time T_j. In D = p - d the targets are -1/D
and 1/D^2 on the horizontal segment {p - d : d in [lo, hi]} at height Im p:
the Sigma denominator-box problem on a thin box, T = i t. The poles of one
height form one level. The levels split into families (sign-definite:
elliptic sector rule; crossing: the bent contour of ``analytic_box``, plus a
Gauss-Legendre leg and tall-box image-phase nodes; optionally the crossing
levels whose narrow side lies within one height: sector rule), each on the
smallest box holding its levels, built at eps = tol/ln(4/tol) so the ds rows
meet tol too. The node set is the union of the families' sets. Nothing is
fitted but linear weights: one least-squares solve per level on the union
(value and ds together), and an evaluation-only certificate on each level's
segment in the pole's own currency (eta_p |error| and eta_p^3 |ds error|). A
family whose levels fail climbs its own fixed ladder. Energies are Ry, times
Ry^-1, slopes are d/d(z^2). docs/theory/response-laplace.md owns the
derivation and the node count.
"""
import math

import numpy as np
from scipy import linalg as la

from .analytic_box import _CROSSING_RUNGS, _SECTOR_RUNGS, _sector_times
from .analytic_box import corner_exponent, crossing_nodes
from .uniform_rule import _BoundaryCloud, _cexp, _edge_points, _pinned_blas_threads

RESPONSE_RULE_CAPACITY = 384
# Node slots of one rule (the stream kernel's compiled time extent; only the
# live prefix is executed).
RESPONSE_NODE_CAPACITY = 2*RESPONSE_RULE_CAPACITY
# The executor's admitted term mass eta_p * sum |c exp(-(d - r) T)|, per pole.
_RESPONSE_MAX_KAPPA = 5000.
# Ridge of the weight solve in units of the tolerance: a term-mass penalty,
# 1/200 of the error budget per unit-maximum column (the mass gate accepts).
_RIDGE = 0.005
# Growth-side times: |Re T| (hi - lo) <= 3, so no Green factor grows past e^3.
_GROWTH_CAP = 3.0
# A column below exp(_DEAD) of the tolerance on a level is not fitted there.
_DEAD = -40.0
# Error currency of the weight solve and the certificate: "relative" (both
# |D| |dQ| and |D|^2 |dQ2|), "rellsq" (relative rows in the weight solve,
# peak certificate) or "peak" (eta |dQ| and eta^2 |dQ2| in both).
_CURRENCY = "peak"
# Node tolerance of a family: "ds" (tol times min(1, 2|p|/(eta ln(4/tol))),
# the ds horizon factor relieved by the ds currency's eta/(2|p|)) or "lam"
# (tol/ln(4/tol) for every family).
_EPS_RULE = "ds"
# Metal occupation envelope in the currency, and node boxes clipped at its
# knee d = 0 (x = max Re p) as the first candidate partition.
_ENVELOPE = True


def _poles(z):
    """Forward poles z and reverse poles -conj(z), all in the upper half plane.

    One Green-pair evaluation A(t) serves both orientations: the reverse
    product at time conj(t) is conj(A(t)). An imaginary-axis sample is its own
    reverse pole, so it contributes one pole, not two.
    """
    poles = []
    for p in np.asarray(z, dtype=np.complex128):
        for q in (p, -p.conjugate()):
            if not any(abs(q - r) <= 1e-12*abs(q) for r in poles):
                poles.append(q)
    return np.asarray(poles)


def response_levels(lo, hi, poles):
    """One thin box (re_lo, re_hi, y, y) of D = p - d per pole height y, Ry."""
    return [(float(poles.real[poles.imag == y].min()) - hi,
             float(poles.real[poles.imag == y].max()) - lo, float(y), float(y))
            for y in np.unique(poles.imag)]


#: Thin-box corner exponent of the response rules: e^c amplifies the narrow
#: edge, and the executor admits a term mass of 5000, so c = 6 leaves the
#: least-squares cancellation a factor of ten (the Sigma rule's c = 4 sits
#: under its own cap of 83 the same way).
_BEND = 6.0


def _hull(levels):
    return (min(lv[0] for lv in levels), max(lv[1] for lv in levels),
            min(lv[2] for lv in levels), max(lv[3] for lv in levels))


def family_partitions(levels, eps_of, decay_rate=0., clipped=None):
    """Candidate family partitions of a group's levels, cheapest rung-0 union first.

    Each partition is a list of ``(box, crossing, heights)``:

    - sign-definite levels (Re D of one sign): one box, the elliptic sector rule;
    - crossing levels: one box, the bent contour; or, in the second
      candidate, the crossing levels whose narrow side lies within one height
      (``min(-re_lo, re_hi) < y``: a high imaginary sample on a metal) split
      off into one sector-rule box, keeping the highest such levels whose
      rung-0 times hold ``0 <= Re T <= decay_rate`` (the lowest is dropped
      until they do).

    Each box is the smallest holding its levels. ``clipped`` (a metal's
    levels cut at the occupation envelope's knee) adds the same candidates on
    those boxes; the certificate stays on the true segments. The candidates
    are ordered by the size of their rung-0 node union, a formula of the
    boxes; the builder takes the first whose ladders certify.
    """
    candidates = []
    for level_set in [levels] + ([clipped] if clipped else []):
        candidates += _partitions(level_set, eps_of, decay_rate)

    def size(partition):
        sets = [family_nodes(box, eps_of(heights), 0, crossing) for box, crossing, heights in partition]
        return np.unique(np.concatenate(sets)).size
    return sorted(candidates, key=size)


def _partitions(levels, eps_of, decay_rate):
    definite = [lv for lv in levels if not lv[0] < 0.0 < lv[1]]
    crossing = [lv for lv in levels if lv[0] < 0.0 < lv[1]]
    base = [(_hull(definite), False, {lv[2] for lv in definite})] if definite else []
    candidates = [base + ([(_hull(crossing), True, {lv[2] for lv in crossing})] if crossing else [])]
    narrow = sorted((lv for lv in crossing if min(-lv[0], lv[1]) < lv[2]), key=lambda lv: lv[2])
    while narrow:
        re_t = (1j*_sector_times(_hull(narrow), eps_of({lv[2] for lv in narrow}), 0)[0]).real
        if float(re_t.min()) >= 0.0 and (not decay_rate or float(re_t.max()) <= decay_rate):
            rest = [lv for lv in crossing if lv not in narrow]
            candidates.append(base + [(_hull(narrow), False, {lv[2] for lv in narrow})]
                              + ([(_hull(rest), True, {lv[2] for lv in rest})] if rest else []))
            break
        narrow = narrow[1:]
    return candidates


def family_nodes(box, eps, rung, crossing, bend=None):
    """Derived times of one family box at ``rung``, or None past its ladder.

    Sign-definite: the elliptic sector rule, six rungs. Crossing: the bent
    contour at ``c = bend`` capped by the tall-box leg phase, then ``c/2``,
    ``c/4`` (six rungs each), plus two derived sets the Sigma rule does not
    need at its tolerance:

    - ``ceil((c + ln(1/eps))/2)`` Gauss-Legendre nodes on the leg
      ``[0, -i c/m]``: the narrow side grows there as ``e^{c u}``, which the
      geometric grading toward 0 leaves unresolved below 1e-8;
    - on a tall box (height ratio H), ``ceil(gamma S (H - 1)/pi)``
      Gauss-Legendre nodes on the decaying image axis ``[0, -i S]``: its
      members turn their phase through ``S (H - 1)`` radians there.
    """
    if not crossing:
        return _sector_times(box, eps, rung)[0] if rung < _SECTOR_RUNGS else None
    if rung >= 3*_CROSSING_RUNGS:
        return None
    y = box[2]
    c = corner_exponent(box, _BEND if bend is None else bend)/2.0**(rung//_CROSSING_RUNGS)
    s, receipt = crossing_nodes(box, eps, rung % _CROSSING_RUNGS, c)
    L, m, B0, gamma = math.log(1.0/eps), receipt["m"], receipt["B0"], receipt["gamma"]
    parts = [s, -1j*receipt["tau_c"]*_unit_gauss(math.ceil((c + L)/2.0))]
    horizon = (L + c)/max(1.0 - m/B0, 0.05)/B0
    k = math.ceil(gamma*horizon*(box[3]/y - 1.0)/math.pi)
    if k:
        parts.append(-1j*horizon*_unit_gauss(k))
    return np.concatenate(parts)/y


def _unit_gauss(n):
    """Gauss-Legendre nodes on [0, 1]."""
    return 0.5*(np.polynomial.legendre.leggauss(n)[0] + 1.0)


def _live(level, times, eps):
    """Columns alive somewhere on the level's segment (above exp(_DEAD) eps)."""
    xa, xb, y, _ = level
    edge = np.array([xa + 1j*y, xb + 1j*y])
    log_max = np.max(-(edge[:, None]*times[None, :]).imag, axis=0)
    return log_max > math.log(eps) + _DEAD


def _envelope(level, poles, decay_rate):
    """The occupation envelope of a level as a function of D: min(1, e^{beta d})
    at d = max Re p - Re D (the level's least suppressed pole), or None."""
    if not decay_rate:
        return None
    top = float(poles.real[poles.imag == level[2]].max())
    return lambda D: np.exp(np.minimum(decay_rate*(top - np.real(D)), 0.0))


def _level_weights(level, times, eps, envelope=None):
    """Weights of 1/D and 1/D^2 on one level: two linear least-squares solves.

    Rows sample the segment at two points per half wave of the largest live
    ``|t|``, each weighted by the certificate's currency (``|D|`` and
    ``|D|^2`` relative, ``eta`` and ``eta^2`` peak); columns are scaled to
    unit maximum in log space; a ridge ``_RIDGE*eps`` prices each term's
    largest contribution in that currency.
    """
    xa, xb, y, _ = level
    live = _live(level, times, eps)
    t = times[live]
    x = _edge_points(xa, xb, lambda v: v + 1j*y, 0.0, float(np.abs(t).max()), 1e-300, 2.0, 8.0)
    d = x + 1j*y
    w = np.zeros((times.size, 2), complex)
    for k, power in enumerate((1, 2)):
        rho = np.abs(d)**power if _CURRENCY in ("relative", "rellsq") else np.full(d.size, y**power)
        if envelope is not None:
            rho = rho*np.maximum(envelope(d), 1e-300)
        log_rho = np.log(rho)
        log_max = np.max(-(d[:, None]*t[None, :]).imag + log_rho[:, None], axis=0)
        a = _cexp(1j*d[:, None]*t[None, :] + (log_rho[:, None] - log_max[None, :]))
        a = np.vstack([a, _RIDGE*eps*math.sqrt(d.size)*np.eye(t.size)])
        rhs = np.zeros(a.shape[0], complex)
        rhs[:d.size] = rho/d**power
        q, r = la.qr(a, mode="economic", check_finite=False)
        w[live, k] = la.solve_triangular(r, q.conj().T @ rhs, check_finite=False)*np.exp(-log_max)
    return w[:, 0], w[:, 1]


def _certify(level, times, weights, poles, envelope=None):
    """Evaluation-only sup of the value and ds errors on one level, and the mass.

    The cloud resolves the largest ``|t|`` of the union everywhere (a union
    of families has no single ray), and refines every sampled local maximum.
    """
    xa, xb, y, _ = level
    relative = _CURRENCY == "relative"
    cloud = _BoundaryCloud(level, 0.0, float(np.abs(times).max()), 1e-300, p=6.0, p_target=8.0,
                           weight=envelope)
    value, _, _ = cloud.sup(times, weights[0], relative)
    slope, _, _ = cloud.sup(times, weights[1], relative, power=2)
    # the executor's noise: eta times the value term mass, peak currency in both modes
    terms = _cexp(1j*cloud.d[:, None]*times[None, :])*weights[0][None, :]
    mass = y*float(np.abs(terms).sum(1).max())
    members = poles[poles.imag == y]
    # ds currency: (eta^2 or |D|^2) |error of 1/D^2| * eta/(2|p|), i.e. eta^3 |d/ds error| at the peak
    return value, slope*float((y/(2*np.abs(members))).max()), mass


def _admissible(times, lo, hi, decay_rate):
    """No Green factor grows past e^3, nor past a metal's occupation envelope."""
    re_t = (1j*times).real
    return (-float(re_t.min())*(hi - lo) <= _GROWTH_CAP*(1 + 1e-12)
            and (not decay_rate or float(re_t.max()) <= decay_rate))


def _fit(levels, poles, times, tol, decay_rate=0.):
    """Per-level weights and certificates on fixed times: {y: (weights, value, ds)}.

    On a metal the currency carries the occupation envelope min(1, e^{beta d})
    (``response_bank.response_occupation_envelope`` bounds every occupation
    product by it), in the weight solve and in the certificate."""
    out = {}
    for level in levels:
        envelope = _envelope(level, poles, decay_rate) if _ENVELOPE else None
        weights = _level_weights(level, times, tol, envelope)
        value, slope, mass = _certify(level, times, weights, poles, envelope)
        ok = (np.isfinite([value, slope, mass]).all() and value <= tol
              and slope <= tol and mass <= _RESPONSE_MAX_KAPPA)
        out[level[2]] = (weights, value, slope, ok)
    return out


def _coefficients(pole, times, reference, level):
    """(value, ds) coefficients of one pole on exp(-(d - reference) T), T = i t."""
    (w_value, w_slope), value_error, slope_error, _ = level
    phase = _cexp(1j*times*(pole - reference))
    return (np.column_stack((-w_value*phase, w_slope*phase)),
            np.array([value_error, slope_error]))


def _rule(z, times, poles, fits, reference):
    """Scatter pole coefficients into forward and reverse sample rows.

    A node T is one Green pair A(T): forward rows fit 1/(d-z) on T; reverse
    rows, evaluated at conj(T) from conj(A(T)), fit 1/(d+z) as the conjugate
    of the -conj(z) fit on T.
    """
    count = len(times)
    T = np.zeros(RESPONSE_NODE_CAPACITY, complex)
    T[:count] = 1j*times
    shape = (len(z), 2, RESPONSE_NODE_CAPACITY)
    value, derivative = np.zeros(shape, complex), np.zeros(shape, complex)
    errors, mass = np.zeros((len(z), 2, 2)), np.zeros((len(z), 2, 2))
    for j, point in enumerate(z):
        for side, pole in enumerate((point, -point.conjugate())):
            match = poles[np.flatnonzero(abs(poles - pole) <= 1e-12*abs(pole))[0]]
            coefficient, error = _coefficients(match, times, reference, fits[match.imag])
            if side:
                coefficient = np.conj(coefficient)
            value[j, side, :count] = coefficient[:, 0]
            derivative[j, side, :count] = (1 if side == 0 else -1)*coefficient[:, 1]/(2*point)
            errors[j, side] = error
        for side in (0, 1):
            mass[j, side] = [point.imag*np.sum(abs(value[j, side])),
                             point.imag**3*np.sum(abs(derivative[j, side]))]
    return dict(t=T, value=value, derivative=derivative, count=count,
                sampled_error=errors, coefficient_mass=mass)


def response_group_rules(lo_ry, hi_ry, z_ry, *, rel_tol=1e-8, previous=None,
                         decay_rate=0.):
    """Shared complex-time rules for a group of response samples.

    Every node T is ONE Green-pair evaluation A(T): forward rows use the
    exponential exp[-(d-reference_ry)*T] and fit 1/(d-z); reverse rows use
    conj(A(T)), i.e. the exponential at conj(T), and fit 1/(d+z). The nodes
    are the union of the group's family rules (module docstring); value and
    ds are certified at rel_tol/2 each on every level. A family whose levels
    fail climbs its own fixed ladder and the union is refit; the candidate
    partitions (``family_partitions``) are tried cheapest first. A positive
    decay_rate (the occupation envelope min(1, exp(decay_rate*d))) admits
    only times with Re T <= decay_rate. No group is split and nothing is
    searched: a group whose ladders run out refuses by name.

    Returns a one-element list of rules. The rule has ``members`` (indices
    into ``z_ry``), ``t[RESPONSE_NODE_CAPACITY]``, ``value``/``derivative`` of
    shape ``[members, 2 (forward, reverse), RESPONSE_NODE_CAPACITY]`` (value
    and d/d(z^2)), ``count``, ``sampled_error`` (certified sup, value and ds,
    in eta and eta^3 currency), ``coefficient_mass`` ``[members, 2, 2]``,
    ``reference_ry``, the per-family ``rungs`` and ``families``
    ``[(crossing, heights)]``. ``previous`` is a list of
    earlier rules; one whose members match is re-certified on its times and
    reused.
    """
    lo, hi = float(lo_ry), float(hi_ry)
    z = np.asarray(z_ry, dtype=np.complex128).reshape(-1)
    if (not z.size or not np.isfinite([lo, hi, rel_tol, decay_rate]).all()
            or not np.isfinite(z).all() or decay_rate < 0 or hi <= lo
            or np.any(z.imag <= 0) or not 1e-13 <= rel_tol < .1):
        raise ValueError('invalid response frequency/domain/tolerance')
    reference = 0. if decay_rate else lo
    tol = rel_tol/2
    # the ds target's time density is s exp(isD): one horizon factor Lam more
    horizon = math.log(4.0/tol)

    def eps_of(heights):
        """Node tolerance of a family: the ds target's density s exp(isD) needs one
        horizon factor ln(4/tol) more than the value, and the ds currency
        eta^3/(2|p|) relieves it by 2|p|/eta."""
        if _EPS_RULE == "lam":
            return tol/horizon
        members = poles[np.isin(poles.imag, list(heights))]
        return tol*min(1.0, float((2*np.abs(members)/members.imag).min())/horizon)
    poles = _poles(z)
    levels = response_levels(lo, hi, poles)
    members = list(range(len(z)))
    old = {tuple(rule["members"]): rule for rule in (previous or ())}
    with _pinned_blas_threads():
        warm = old.get(tuple(members))
        if warm is not None:
            times = -1j*np.asarray(warm["t"][:warm["count"]])
            if _admissible(times, lo, hi, decay_rate):
                fits = _fit(levels, poles, times, tol, decay_rate)
                if all(f[3] for f in fits.values()):
                    return [dict(_rule(z, times, poles, fits, reference), members=members,
                                 reference_ry=reference, rungs=warm.get("rungs"),
                                 families=warm.get("families"))]
        clipped = None
        if _ENVELOPE and decay_rate:
            # the envelope's knee d = 0 sits at x = max Re p of each level
            clipped = [(lv[0], min(lv[1], float(poles.real[poles.imag == lv[2]].max())), lv[2], lv[3])
                       for lv in levels]
        for boxes in family_partitions(levels, eps_of, decay_rate, clipped):
            rungs = [0]*len(boxes)
            sets = [None]*len(boxes)

            def climb(i, rung):
                """The first admissible rung at or above ``rung`` (inadmissible rungs are skipped)."""
                box, crossing, heights = boxes[i]
                while True:
                    times = family_nodes(box, eps_of(heights), rung, crossing)
                    if times is None or _admissible(times, lo, hi, decay_rate):
                        rungs[i], sets[i] = rung, times
                        return times is not None
                    rung += 1

            alive = all([climb(i, 0) for i in range(len(boxes))])
            while alive:
                times = np.unique(np.concatenate(sets))
                if times.size > RESPONSE_NODE_CAPACITY:
                    break
                fits = _fit(levels, poles, times, tol, decay_rate)
                failing = sorted({i for i, (_b, _c, heights) in enumerate(boxes)
                                  for level in levels if level[2] in heights and not fits[level[2]][3]})
                if not failing:
                    return [dict(_rule(z, times, poles, fits, reference), members=members,
                                 reference_ry=reference, rungs=list(rungs),
                                 families=[(bool(c), sorted(h)) for _b, c, h in boxes])]
                alive = all([climb(i, rungs[i] + 1) for i in failing])
    raise ValueError(
        f'GATE response_rule_certificate: the derived level rules of {len(levels)} pole '
        f'heights do not certify at tolerance {tol:.3e} within {RESPONSE_NODE_CAPACITY} nodes '
        f'and their ladders (interval [{lo}, {hi}] Ry, samples {z.tolist()}, decay_rate '
        f'{decay_rate})')
