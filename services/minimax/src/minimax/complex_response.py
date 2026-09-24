"""Response-bank exponential sums shared by groups of samples.

Scalar construction only: a stacked Hankel shift pencil chooses complex
Laplace times shared by every forward (z) and reverse (-conj z) pole of a
sample group; linear projection fits 1/(d-p) and 1/(d-p)^2 on those nodes.
No continuum certificate is claimed. Energies are Ry, times Ry^-1, slopes are
d/d(z^2).
"""
import numpy as np
from scipy import linalg as la

RESPONSE_RULE_CAPACITY = 192
# Node slots of one rule: a shared set uses at most one pencil's capacity; a
# single sample whose forward and reverse poles need separate sets uses two.
RESPONSE_NODE_CAPACITY = 2*RESPONSE_RULE_CAPACITY
_RESPONSE_MAX_KAPPA = 5000.
# A group's pencil geometry is abandoned after this many more exponentials
# without a decade of accuracy; the next geometry, then a smaller group, is
# tried. A single sample keeps the full scan, so it never refuses earlier.
_STAGNATION_NODES = 24


def _project(lo, hi, pole, times, decay_rate=0.):
    eta = abs(pole.imag)
    span, zp = (hi-lo)/eta, (pole-lo)/eta
    x = np.unique(np.r_[0., span,
        span/2*(1-np.cos(np.pi*(np.arange(1400)+.5)/1400)),
        zp.real+np.linspace(-5, 5, 500)])
    x = x[(x >= 0) & (x <= span)]
    f = 1/(x-zp)
    origin = 0. if decay_rate else lo
    def basis(xx):
        d = lo + eta*xx
        weight = np.exp(np.minimum(decay_rate*d, 0.))
        return np.exp(np.minimum(decay_rate*d, 0.)[:, None]
                      - (d[:, None]-origin)*times), weight
    a, weight = basis(x)
    a = a/abs(f[:, None])
    scale = la.norm(a, axis=0)
    if np.any(scale == 0) or not np.isfinite(scale).all():
        return None
    coeff = la.lstsq(a/scale, weight[:, None]*np.column_stack((f, f*f))/abs(f[:, None]),
                    cond=1e-14, lapack_driver='gelsd', check_finite=False)[0]/scale[:, None]
    probe = np.unique(np.r_[x, np.linspace(0, span, 2501)])
    error = np.zeros(2)
    for xx in np.array_split(probe, 16):
        exact = 1/(xx-zp)
        a, weight = basis(xx)
        error = np.maximum(error, np.max(abs(a@coeff
                       - weight[:, None]*np.column_stack((exact, exact*exact))), axis=0))
    mass = np.sum(abs(coeff), axis=0)
    return coeff / np.array([eta, eta*eta]), error, mass


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


def _fits(lo, hi, poles, times, tol, decay_rate, order=None):
    """Project every pole on shared times; stop at the first rejected pole.

    ``order`` is a caller-owned list of pole indices; a rejected pole moves to
    its front, so the next candidate is tested on the hardest pole first.
    """
    order = list(range(len(poles))) if order is None else order
    fits = [None]*len(poles)
    worst = 0.
    for position, index in enumerate(list(order)):
        pole = poles[index]
        fit = _project(lo, hi, pole, times, decay_rate)
        error = np.inf if fit is None else float(np.max(fit[1]*[1., pole.imag/(2*abs(pole))]))
        mass_rejected = fit is not None and error <= tol and fit[2][0] > _RESPONSE_MAX_KAPPA
        if fit is None or not error <= tol or not np.isfinite(fit[2]).all() or mass_rejected:
            order.insert(0, order.pop(position))
            return None, error, bool(mass_rejected)
        worst = max(worst, error)
        fits[index] = fit
    return fits, worst, False


def _shared_times(lo, hi, poles, tol, previous=None, decay_rate=0., patience=None):
    """Complex times shared by every pole, or None.

    A stacked (multi-channel) Hankel shift pencil: each pole contributes the
    Hankel matrix of 1/(x - z_p) sampled at the group's finest scale; their
    common row space holds the shared exponentials. Coefficients are then the
    per-pole linear projection of the single-pole construction, with the same
    sampled-error and coefficient-mass acceptance.
    """
    order = list(range(len(poles)))
    if previous is not None and len(previous):
        fits, _, _ = _fits(lo, hi, poles, previous, tol, decay_rate, order)
        if fits is not None:
            return previous, fits
    eta = float(poles.imag.min())
    origin = 0. if decay_rate else lo
    span = max(abs(lo), abs(hi))/eta if decay_rate else (hi-lo)/eta
    zps = (poles-origin)/eta
    geometry = max(span, 8*max(float(zps.real.max()), 0.))
    # Resolve the reciprocal and its square before taking the shift pencil:
    # principal-log modes stop at pi/step; their tails decay as t*exp(-t).
    horizon = -np.log(tol) + np.log(-np.log(tol)) + 6.
    size = max(800, int(np.ceil((geometry + 16.)*horizon/(2*np.pi))))
    ids = np.arange(size)
    for padding, flatten in ((0., 1.), (2., .9), (4., .9), (8., 1.), (10., 1.), (16., .9)):
        # A short interval ending near the resonance gives growing modes.
        # Extend only the proposal geometry; fit/check the physical interval.
        step = (geometry+padding)/(2*size)
        base = -padding+(ids[:, None]+ids[None, :])*step
        blocks, shifted = [], []
        # The shared exponentials live in the ROW space (index i). A column
        # subset dense near j=0, where 1/(x-z_p) varies fastest, and geometric
        # beyond spans it; the stack stays ~2*size wide for any group.
        width = max(32, 2*size//len(zps))
        columns = np.unique(np.r_[ids[:16], np.round(np.geomspace(16, size-1, width)).astype(int)])
        for zp in zps:
            block = 1/(base[:, columns]-zp)
            scale = 1/la.norm(block)
            blocks.append(scale*block)
            shifted.append(scale/(base[:, columns]+step-zp))
        # The stack is size x (poles*size). Factor its tall adjoint once,
        # H = R^H Q^H, and take the SVD of the small R^H: the same singular
        # triplets as a direct SVD of the wide stack, at QR cost.
        q, r = la.qr(np.hstack(blocks).conj().T, mode="economic", check_finite=False)
        u, s, wh = la.svd(r.conj().T, check_finite=False)
        rank = min(RESPONSE_RULE_CAPACITY, len(s))
        shift = u[:, :rank].conj().T@(np.hstack(shifted)@q)@wh[:rank].conj().T
        best, best_n = np.inf, 0
        for n in range(8, rank + 1, 4):
            roots = la.eigvals(shift[:n, :n]/np.sqrt(s[:n, None]*s[None, :n]),
                              check_finite=False)
            with np.errstate(divide='ignore', invalid='ignore'):
                t = -np.log(roots)/step
            if not np.isfinite(t).all() or np.any(t.real < 0):
                continue
            times = (flatten*t.real+1j*t.imag)/eta
            if decay_rate:
                times = times[times.real <= decay_rate]
            if not len(times):
                continue
            fits, error, mass_rejected = _fits(lo, hi, poles, times, tol, decay_rate, order)
            if fits is not None:
                return times, fits
            if mass_rejected:
                break  # Try the next padding/flattening geometry, not more cancelling terms.
            if error < best/10:
                best, best_n = error, n
            elif patience is not None and n - best_n >= patience:
                break  # No decade of accuracy in this many more exponentials.
    return None


def _rule(z, sets):
    """Scatter pole fits on node sets into forward and reverse sample rows.

    ``sets`` is a list of (times, poles, fits). A node t is one Green pair
    A(t): forward rows fit 1/(d-z) on t; reverse rows, evaluated at conj(t)
    from conj(A(t)), fit 1/(d+z) as the conjugate of the -conj(z) fit on t.
    A set without a sample's pole contributes zero weight to that row.
    """
    times = np.concatenate([t for t, _, _ in sets])
    count = len(times)
    t = np.zeros(RESPONSE_NODE_CAPACITY, complex)
    t[:count] = times
    shape = (len(z), 2, RESPONSE_NODE_CAPACITY)
    value, derivative = np.zeros(shape, complex), np.zeros(shape, complex)
    errors, mass = np.zeros((len(z), 2, 2)), np.zeros((len(z), 2, 2))
    for j, point in enumerate(z):
        start = 0
        for set_times, poles, fits in sets:
            stop = start + len(set_times)
            for side, pole in enumerate((point, -point.conjugate())):
                match = np.flatnonzero(abs(poles - pole) <= 1e-12*abs(pole))
                if not match.size:
                    continue
                coefficient, fit_error = fits[int(match[0])][:2]
                if side:
                    coefficient = np.conj(coefficient)
                value[j, side, start:stop] = coefficient[:, 0]
                derivative[j, side, start:stop] = (1 if side == 0 else -1)*coefficient[:, 1]/(2*point)
                errors[j, side] = [fit_error[0], fit_error[1]*abs(point.imag/(2*point))]
            start = stop
        for side in (0, 1):
            mass[j, side] = [point.imag*np.sum(abs(value[j, side])),
                             point.imag**3*np.sum(abs(derivative[j, side]))]
    return dict(t=t, value=value, derivative=derivative, count=count,
                sampled_error=errors, coefficient_mass=mass)


def response_group_rules(lo_ry, hi_ry, z_ry, *, rel_tol=1e-8, previous=None,
                         decay_rate=0.):
    """Shared complex-time rules for a group of response samples.

    Every node t is ONE Green-pair evaluation A(t): forward rows use the
    exponential exp[-(d-reference_ry)*t] and fit 1/(d-z); reverse rows use
    conj(A(t)), i.e. the exponential at conj(t), and fit 1/(d+z). A group
    whose shared fit fails is split in halves, down to single samples, so
    no sample is ever evaluated with more nodes than its own rule needs.

    Returns a list of rules. Each has ``members`` (indices into ``z_ry``),
    ``t[RESPONSE_NODE_CAPACITY]``, ``value``/``derivative`` of shape
    ``[members, 2 (forward, reverse), RESPONSE_NODE_CAPACITY]`` (value and
    d/d(z^2)), ``count``, ``sampled_error`` and ``coefficient_mass``
    ``[members, 2, 2]``, and ``reference_ry``. ``previous`` is a list of
    earlier rules; one whose members match is tried first. A positive
    decay_rate bounds occupation products by min(1,exp(decay_rate*d)); errors
    then use that envelope and 0<=Re(t)<=decay_rate. Bounds are sampled, not
    proven.
    """
    lo, hi = float(lo_ry), float(hi_ry)
    z = np.asarray(z_ry, dtype=np.complex128).reshape(-1)
    if (not z.size or not np.isfinite([lo, hi, rel_tol, decay_rate]).all()
            or not np.isfinite(z).all() or decay_rate < 0 or hi <= lo
            or np.any(z.imag <= 0) or not 1e-13 <= rel_tol < .1):
        raise ValueError('invalid response frequency/domain/tolerance')
    reference = 0. if decay_rate else lo
    old = {tuple(rule["members"]): rule["t"][:rule["count"]]
           for rule in (previous or ())}

    def build(members):
        poles = _poles(z[members])
        got = _shared_times(lo, hi, poles, rel_tol/2, old.get(tuple(members)), decay_rate,
                            patience=None if len(members) == 1 else _STAGNATION_NODES)
        if got is not None:
            rule = _rule(z[members], [(*got[:1], poles, got[1])])
            return [dict(rule, members=list(members), reference_ry=reference)]
        if len(members) == 1:
            # Forward and reverse poles on separate node sets: each node then
            # serves one orientation, exactly as separate per-pole rules do.
            sets = []
            for pole in poles:
                one = np.asarray([pole])
                got = _shared_times(lo, hi, one, rel_tol/2, None, decay_rate)
                if got is None:
                    raise ValueError(f'response exponential fit failed: interval={lo, hi}, '
                                     f'sample={z[members[0]]}, pole={pole}, tolerance={rel_tol}')
                sets.append((got[0], one, got[1]))
            rule = _rule(z[members], sets)
            return [dict(rule, members=list(members), reference_ry=reference)]
        half = len(members)//2
        return build(members[:half]) + build(members[half:])

    return build(list(range(len(z))))
