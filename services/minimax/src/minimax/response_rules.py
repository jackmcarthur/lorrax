"""Scalar response-bank quadratures; Ry frequencies and d/ds, s=z**2.

Real-time and noncrossing Laplace rules share Gaussian ellipse bounds and
analytic derivative/tail control. No GW implementation is imported.
"""
from functools import lru_cache
import hashlib
import math

import numpy as np
from numpy.polynomial.legendre import leggauss


@lru_cache(maxsize=256)
def _legendre(n):
    return leggauss(n)


def _inputs(z_ry, rel_tol):
    z = np.asarray(z_ry, dtype=np.complex128)
    if (z.ndim != 1 or not z.size or not np.all(np.isfinite(z))
            or np.any(z.imag <= 0)):
        raise ValueError('response rule needs a nonempty upper-half-plane z vector')
    if not np.isfinite(rel_tol) or not 1e-13 <= rel_tol < .1:
        raise ValueError('response rule tolerance must lie in [1e-13, .1)')
    return z


def _tail_x(tol):
    lo, hi = 0., -math.log(tol)+8
    while math.exp(-hi)*(1+hi) > tol/8:
        hi *= 2
    for _ in range(80):
        mid = (lo+hi)/2
        if math.exp(-mid)*(1+mid) > tol/8:
            lo = mid
        else:
            hi = mid
    return hi


def _panel_bounds(n, left, right, z, delta_max):
    """Run212 Bernstein-ellipse bound, including the full ellipse centre."""
    width = right-left
    u = z.imag
    a = (delta_max+np.abs(z.real)+u)*width/4
    # Also bound integral plus positive quadrature directly by its envelope.
    envelope = np.exp(-u*left)
    vb = 4*width*envelope
    db = 2*width*envelope*right/np.abs(z)
    active = n > a
    aa = a[active]
    rho = (n+np.sqrt(n*n-aa*aa))/aa
    logb = (np.log(64/15*width/2)-u[active]*left
            + aa*(rho-1/rho)-2*n*np.log(rho)
            - np.log1p(-rho**(-2*n)))
    base = np.exp(np.minimum(logb, 700))
    vb[active] = np.minimum(vb[active], 2*base)
    # |t| includes the midpoint, omitted in the experimental loose estimate.
    t_ellipse = left+width/2+width/4*(rho+1/rho)
    db[active] = np.minimum(db[active], 2*base*t_ellipse/np.abs(z[active]))
    return vb, db


def _node_digest(rule):
    """Identify the integration arrays independently of current projections."""
    digest = hashlib.sha256()
    for key in ("t", "h"):
        if key in rule:
            array = np.ascontiguousarray(rule[key], dtype="<f8")
            digest.update(key.encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def _bank_bounds(z, delta, edges, orders, eta):
    """Continuum panel errors plus the complete infinite-time tail."""
    vb, db = np.zeros(z.size), np.zeros(z.size)
    for left, right, n in zip(edges[:-1], edges[1:], orders):
        v, d = _panel_bounds(n, left, right, z, delta)
        vb += v
        db += d
    stop = edges[-1]
    vb += 2*np.exp(-z.imag*stop)/z.imag
    db += np.exp(-z.imag*stop)*(1+z.imag*stop)/(np.abs(z)*z.imag**2)
    return eta*vb, eta**3*db


def _reuse_bank_rule(previous, z, delta, rel_tol):
    """Revalidate fixed positive time nodes at the current complex points."""
    cert = previous["certificate"]
    if _node_digest(previous) != previous.get("node_digest"):
        raise ValueError("response rule node digest mismatch")
    if cert["rel_tol"] != rel_tol:
        return None, "tolerance changed"
    if delta > cert["delta_max_ry"]:
        return None, "transition interval escaped"
    eta = float(z.imag.min())
    vb, db = _bank_bounds(z, cert["delta_max_ry"], cert["panel_edges"],
                          cert["orders"], eta)
    if max(vb.max(), db.max()) > rel_tol:
        return None, "current frequency certificate failed"
    t, h = previous["t"], previous["h"]
    value = h[None, :]*np.exp(1j*z[:, None]*t)
    result = dict(previous, projection_value=value,
                  projection_derivative=value*(1j*t[None, :]/(2*z[:, None])))
    result["certificate"] = dict(cert, eta_ry=eta,
        z_ry=[[v.real, v.imag] for v in z], value_bound=vb.tolist(),
        derivative_bound=db.tolist())
    return result, "current domain and frequency certificates pass"


def response_bank_rule(z_ry, delta_max_ry, *, rel_tol=1e-8,
                       previous=None, domain_pad_ry=0.0):
    """Positive shared Hermite time rule for current response samples.

    Parameters
    ----------
    z_ry : complex ndarray, shape (sample,)
        Actual upper-half-plane points in Ry, including imaginary states.
    delta_max_ry : float
        Bound on absolute transition energy in Ry.
    rel_tol : float
        Peak-scaled absolute tolerance: eta*|value error| and
        eta**3*|d/ds error|, eta=min(Im(z)). Not relative W accuracy.
    previous : dict or None
        In-memory rule from this owner. Reuse only its integration arrays;
        recompute projections and certify every current z, including the tail.
    domain_pad_ry : float
        Nonnegative extra transition extent when a new rule is necessary.

    Returns
    -------
    dict
        Positive times ``t`` [1/Ry], positive weights ``h`` [1/Ry],
        projections [sample,time], and a per-point continuum certificate.
        Value projection is h*exp(i*z*t); derivative multiplies i*t/(2*z).
        The bank owns the paired retarded correlation and its sign/factor.
    """
    z = _inputs(z_ry, rel_tol)
    delta = float(delta_max_ry)
    if not np.isfinite(delta) or delta < 0:
        raise ValueError('delta_max_ry must be finite and nonnegative')
    if not np.isfinite(domain_pad_ry) or domain_pad_ry < 0:
        raise ValueError('domain_pad_ry must be finite and nonnegative')
    reason = "initial rule"
    if previous is not None:
        reused, reason = _reuse_bank_rule(previous, z, delta, rel_tol)
        if reused is not None:
            return dict(reused, reuse_status="hit", reuse_reason=reason)
    delta += domain_pad_ry
    eta = float(z.imag.min())
    stop = _tail_x(rel_tol)/eta
    freq = max(delta+float(np.abs(z.real).max()), eta)
    count = max(1, math.ceil(stop/(16*2*np.pi/freq)))
    if count > 10000:
        raise RuntimeError('Hermite panel budget exceeded')
    width = stop/count
    early = []
    edge = 1/float(z.imag.max())
    while edge < width:
        early.append(edge)
        edge *= 4
    edges = np.r_[0., early, np.arange(1, count+1)*width]
    panel_count = len(edges)-1
    nodes, weights, orders = [], [], []
    for left, right in zip(edges[:-1], edges[1:]):
        for n in range(2, 257):
            vb, db = _panel_bounds(n, left, right, z, delta)
            if np.all(vb*eta <= rel_tol/(4*panel_count)) and np.all(
                    db*eta**3 <= rel_tol/(4*panel_count)):
                break
        else:
            raise RuntimeError('Hermite order budget exceeded')
        x, h = _legendre(n)
        nodes.append(left+(right-left)*(x+1)/2)
        weights.append((right-left)*h/2)
        orders.append(n)
    t, h = np.concatenate(nodes), np.concatenate(weights)
    vbound, dbound = _bank_bounds(z, delta, edges, orders, eta)
    if max(vbound.max(), dbound.max()) > rel_tol:
        raise RuntimeError('Hermite certificate failed')
    value = h[None, :]*np.exp(1j*z[:, None]*t)
    derivative = value*(1j*t[None, :]/(2*z[:, None]))
    result = dict(t=t, h=h, projection_value=value,
                projection_derivative=derivative,
                certificate=dict(status='PASS', scope='continuum |delta|<=delta_max; all supplied z',
                    norm='eta*absolute(value), eta**3*absolute(ds); paired branches',
                    eta_ry=eta, delta_max_ry=delta, z_ry=[[v.real, v.imag] for v in z],
                    rel_tol=rel_tol, value_bound=vbound.tolist(),
                    derivative_bound=dbound.tolist(), orders=orders,
                    panel_edges=edges.tolist(), time_max=float(stop),
                    owner='Run212 Hermite / Run307 unequal-decay panels'))
    return dict(result, node_digest=_node_digest(result),
                reuse_status="build" if previous is None else "rebuild",
                reuse_reason=reason)


def _remote_scaled_bounds(z, lo, hi, value, time_value, ordered):
    """Propagate bounds for 1/(delta +/- z) and its squared reciprocal."""
    v = value.reshape(2, -1).sum(axis=0)/2
    d = time_value.reshape(2, -1).sum(axis=0)/(4*np.abs(z))
    ends = np.asarray([lo, hi])[:, None]
    denom = ends**2 + np.abs(z)**2
    bounds = [v/np.min(ends/denom, axis=0),
              d/np.min(ends/denom**2, axis=0)]
    if ordered:
        # Certify K=1/(delta^2-z^2) and dK/ds separately: d(z*K)/ds
        # can vanish on the imaginary axis, so relative error there is undefined.
        bounds += [v/np.abs(z)*(hi*hi+np.abs(z)**2),
                   (d/np.abs(z)+v/(2*np.abs(z)**3))*(hi*hi+np.abs(z)**2)**2]
    return np.asarray(bounds)


def _remote_bounds(z, lo, hi, edges, orders, ordered, *, tail=True):
    """Continuum relative bounds, enclosing each decay scale separately.

    These scalar geometric cells share ONE time rule/Green stream. Keeping
    their envelopes and response scales together avoids charging a fast
    high-energy decay the error of the slowest low-energy decay.
    """
    reach = float(np.abs(z.real).max())
    count = max(1, math.ceil(math.log2((hi-reach)/(lo-reach))))
    delta = reach+np.geomspace(lo-reach, hi-reach, count+1)
    points = 1j*(delta[:-1, None, None]+np.array([-1, 1])[None, :, None]*z)
    span = np.broadcast_to(np.diff(delta)[:, None, None], points.shape)
    rate = points.imag
    stop = edges[-1]
    value = np.exp(-rate*stop)/rate if tail else np.zeros_like(rate)
    time_value = value*(stop+1/rate)
    for left, right, n in zip(edges[:-1], edges[1:], orders):
        v, d = _panel_bounds(n, left, right, points.ravel(), span.ravel())
        value += v.reshape(points.shape)
        time_value += (d*np.abs(points.ravel())).reshape(points.shape)
    return np.max([_remote_scaled_bounds(z, a, b, v, d, ordered)
                   for a, b, v, d in zip(delta[:-1], delta[1:], value, time_value)], axis=0)


def response_laplace_rule(delta_lo_ry, delta_hi_ry, z_ry, *, rel_tol=1e-8,
                          previous=None, domain_pad_ry=0.0, ordered=False,
                          reference_ry=None):
    """Direct positive-time quadrature of the noncrossing response.

    For delta > |Re(z)|, integrate exp(-delta*t) cosh(z*t) (even) and
    exp(-delta*t) sinh(z*t) (odd), with analytic d/ds, s=z**2. Projections
    include exp(-reference_ry*t); the consumer supplies exp(-(delta-ref)*t).
    The default reference is delta_lo, keeping both exponential branches
    bounded. Panel/tail certificates use the same Gaussian owner as the
    real-time rule; no Taylor expansion or fitted inverse-moment rows.
    """
    z = _inputs(z_ry, rel_tol)
    lo, hi = float(delta_lo_ry), float(delta_hi_ry)
    if not np.isfinite(lo+hi) or not 0 < lo <= hi:
        raise ValueError('remote Laplace interval must be finite and positive')
    if not np.isfinite(domain_pad_ry) or domain_pad_ry < 0:
        raise ValueError('domain_pad_ry must be finite and nonnegative')
    reach = float(np.abs(z.real).max())
    if lo <= reach:
        raise ValueError('remote Laplace integral does not converge; repartition in bank owner')
    ref = lo if reference_ry is None else float(reference_ry)
    if not np.isfinite(ref) or not reach < ref <= lo:
        raise ValueError('remote reference must lie above |Re(z)| and at or below delta_lo')
    reason = "initial rule"
    reused = False
    if previous is not None:
        if _node_digest(previous) != previous.get("node_digest"):
            raise ValueError("response rule node digest mismatch")
        cert = previous["certificate"]
        low, high = cert["delta_ry"]
        reason = "domain or current frequency certificate changed"
        if (cert["rel_tol"] == rel_tol and low <= lo <= hi <= high
                and low > reach):
            bounds = _remote_bounds(z, low, high, cert["panel_edges"],
                                    cert["orders"], ordered)
            if bounds.max() <= rel_tol:
                lo, hi = low, high
                t, h = previous["t"], previous["h"]
                edges, orders = cert["panel_edges"], cert["orders"]
                reused, reason = True, "current domain and frequency certificates pass"
    if not reused:
        padded_lo = max(lo-domain_pad_ry, lo/2)
        if padded_lo > reach:
            lo = padded_lo
        hi += domain_pad_ry
        rate = lo-reach
        stop = _tail_x(rel_tol)/rate
        while _remote_bounds(z, lo, hi, [stop], [], ordered).max() > rel_tol/8:
            stop *= 2
        edges = [0.]
        edge = 1/(hi+float(np.abs(z).max()))
        while edge < stop:
            edges.append(edge)
            edge *= 2
        edges.append(stop)
        nodes, weights, orders = [], [], []
        for left, right in zip(edges[:-1], edges[1:]):
            for n in range(2, 257):
                bound = _remote_bounds(z, lo, hi, [left, right], [n], ordered, tail=False)
                if bound.max() <= rel_tol/(4*(len(edges)-1)):
                    break
            else:
                raise RuntimeError('remote Gaussian order budget exceeded')
            x, w = _legendre(n)
            nodes.append(left+(right-left)*(x+1)/2)
            weights.append((right-left)*w/2)
            orders.append(n)
        t, h = np.concatenate(nodes), np.concatenate(weights)
        bounds = _remote_bounds(z, lo, hi, edges, orders, ordered)
    if bounds.max() > rel_tol:
        raise RuntimeError('remote Gaussian certificate failed')
    plus = np.exp(-(ref-z[:, None])*t)
    minus = np.exp(-(ref+z[:, None])*t)
    even, odd = h*(plus+minus)/2, h*(plus-minus)/2
    result = dict(t=t, h=h, reference_ry=ref, projection_value=even,
                  projection_derivative=odd*t/(2*z[:, None]),
                  certificate=dict(status='PASS', scope='continuum delta interval; all supplied z',
                      norm='relative even value and ds; relative odd K and dK/ds',
                      delta_ry=[lo, hi], z_ry=[[v.real, v.imag] for v in z],
                      rel_tol=rel_tol, value_bound=bounds[0].tolist(),
                      derivative_bound=bounds[1].tolist(),
                      panel_edges=list(edges), orders=list(orders), ordered=ordered,
                      owner='direct Laplace / shared Gaussian ellipse bounds'))
    if ordered:
        result.update(odd_projection_value=odd,
                      odd_projection_derivative=even*t/(2*z[:, None]))
        result['certificate'].update(odd_value_bound=bounds[2].tolist(),
                                     odd_derivative_bound=bounds[3].tolist())
    return dict(result, node_digest=_node_digest(result),
                reuse_status='hit' if reused else ('build' if previous is None else 'rebuild'),
                reuse_reason=reason)
