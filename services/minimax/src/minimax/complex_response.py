"""Frequency-specific exponential sums, with sampled value/slope accuracy.

Scalar construction only: Hankel shift pencils choose complex Laplace times;
linear projection fits 1/(d-p) and 1/(d-p)^2 on the same nodes. No continuum
certificate is claimed. Energies are Ry, times Ry^-1, slopes are d/d(z^2).
"""
import numpy as np
from scipy import linalg as la

RESPONSE_RULE_CAPACITY = 192
_RESPONSE_MAX_KAPPA = 5000.


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


def _primitive(lo, hi, pole, tol, previous=None, decay_rate=0.):
    if pole.imag < 0:
        t, c, error = _primitive(lo, hi, pole.conjugate(), tol,
                                 None if previous is None else previous.conj(), decay_rate)
        return t.conj(), c.conj(), error
    eta = pole.imag
    origin = 0. if decay_rate else lo
    span = max(abs(lo), abs(hi))/eta if decay_rate else (hi-lo)/eta
    zp = (pole-origin)/eta
    geometry = max(span, 8*max(zp.real, 0.))
    # Resolve the reciprocal and its square before taking the shift pencil:
    # principal-log modes stop at pi/step; their tails decay as t*exp(-t).
    horizon = -np.log(tol) + np.log(-np.log(tol)) + 6.
    size = max(800, int(np.ceil((geometry + 16.)*horizon/(2*np.pi))))
    best = [float("inf"), None]
    def accept(t):
        if decay_rate and np.any(t.real > decay_rate):
            return None, False
        fit = _project(lo, hi, pole, t, decay_rate)
        if fit is not None and np.max(fit[1]) < best[0]:
            best[:] = [float(np.max(fit[1])), (len(t), fit[1], fit[2])]
        accurate = fit is not None and np.max(fit[1]) <= tol and np.isfinite(fit[2]).all()
        if accurate and fit[2][0] <= _RESPONSE_MAX_KAPPA:
            return (t, fit[0], fit[1]), False
        return None, accurate
    if previous is not None:
        got, _ = accept(previous)
        if got is not None:
            return got
    ids = np.arange(size)
    for padding, flatten in ((0., 1.), (2., .9), (4., .9), (8., 1.), (10., 1.), (16., .9)):
        # A short interval ending near the resonance gives growing modes.
        # Extend only the proposal geometry; fit/check the physical interval.
        step = (geometry+padding)/(2*size)
        center = -padding+(ids[:, None]+ids[None, :])*step-zp
        u, s, vh = la.svd(1/center, full_matrices=False, check_finite=False)
        shift = u[:, :RESPONSE_RULE_CAPACITY].conj().T@(1/(center+step))@vh[:RESPONSE_RULE_CAPACITY].conj().T
        for n in range(8, RESPONSE_RULE_CAPACITY + 1, 4):
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
            got, mass_rejected = accept(times)
            if got is not None:
                return got
            if mass_rejected:
                break  # Try the next padding/flattening geometry, not more cancelling terms.
    raise ValueError(f'response exponential fit failed: interval={lo, hi}, pole={pole}, tolerance={tol}, best={best}')


def response_frequency_rule(lo_ry, hi_ry, z_ry, *, rel_tol=1e-8, previous=None,
                            decay_rate=0.):
    """Return independent forward/backward nodes and value/ds coefficients.

    Arrays have fixed capacity [2,RESPONSE_RULE_CAPACITY]; zero coefficient entries do no spatial
    work. Each exponential is exp[-(d-reference_ry)*t]. A positive decay_rate
    bounds occupation products by min(1,exp(decay_rate*d)); errors then use
    that envelope and 0<=Re(t)<=decay_rate. This adds no occupation discard
    to the caller's support policy.
    Bounds are sampled, not proven.
    """
    lo, hi, z = float(lo_ry), float(hi_ry), complex(z_ry)
    if not np.isfinite([lo, hi, z.real, z.imag, rel_tol, decay_rate]).all() or decay_rate < 0 or hi <= lo or z.imag <= 0 or not 1e-13 <= rel_tol < .1:
        raise ValueError('invalid response frequency/domain/tolerance')
    times = np.zeros((2, RESPONSE_RULE_CAPACITY), complex)
    value, derivative = np.zeros_like(times), np.zeros_like(times)
    errors, counts = [], []
    for i, pole in enumerate((z, -z)):
        old = None if previous is None else previous[i][np.abs(previous[i]) > 0]
        t, c, err = _primitive(lo, hi, pole, rel_tol/2, old, decay_rate)
        counts.append(len(t))
        times[i, :len(t)] = t
        value[i, :len(t)] = c[:, 0]
        derivative[i, :len(t)] = (1 if i == 0 else -1)*c[:, 1]/(2*z)
        errors.append([err[0], err[1]*abs(z.imag/(2*z))])
    return dict(t=times, value=value, derivative=derivative,
                reference_ry=0. if decay_rate else lo,
                counts=np.asarray(counts), sampled_error=np.asarray(errors))
