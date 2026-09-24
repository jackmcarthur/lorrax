"""Scalar response-bank quadratures; Ry frequencies and d/ds, s=z**2.

Real-time Gaussian panels and compact noncrossing exponential sums certify
current-frequency values and analytic ds targets separately. No GW implementation is imported.
"""
from functools import lru_cache
import hashlib
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from .complex_response import RESPONSE_NODE_CAPACITY, RESPONSE_RULE_CAPACITY, response_group_rules


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
            array = np.ascontiguousarray(rule[key], dtype="<c16" if np.iscomplexobj(rule[key]) else "<f8")
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


def response_laplace_rule(delta_lo_ry, delta_hi_ry, z_ry, *, rel_tol=1e-8,
                          previous=None, domain_pad_ry=0.0, ordered=False,
                          reference_ry=None):
    """Compact noncrossing response rule on positive real time nodes.

    Elliptic decay scales and a time-moment Ritz solve prescribe the nodes;
    linear projection fits exact value and ds targets on those same nodes.
    A continuum residual certificate covers the rounded returned arrays.
    Projections include exp(-reference_ry*t); the consumer supplies the
    remaining exp(-(delta-reference_ry)*t). See docs/theory/response-laplace.md.
    """
    from .laplace_ritz import place_times, project_response

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
    if previous is not None:
        if _node_digest(previous) != previous.get("node_digest"):
            raise ValueError("response rule node digest mismatch")
        low, high = previous['certificate']['delta_ry']
        reason = "transition interval escaped"
        if reach < low <= lo <= hi <= high:
            result = project_response(previous['t'], low, high, z, ref, rel_tol, ordered)
            reason = "current frequency certificate failed"
            if result['certificate']['status'] == 'PASS':
                return dict(result, node_digest=_node_digest(result), reuse_status='hit',
                            reuse_reason="current domain and frequency certificates pass")
    padded_lo = max(lo-domain_pad_ry, lo/2)
    if padded_lo > reach:
        lo = padded_lo
    hi += domain_pad_ry
    # Work-saving starting degree only; acceptance always uses the certificate.
    ratio = (hi+reach)/(lo-reach)
    first = min(64, max(4, math.ceil(math.log(16*ratio)*math.log(1/rel_tol)/math.pi**2)))
    best = math.inf
    for degree in range(first, 65):
        t = place_times(lo, hi, reach, degree)
        result = project_response(t, lo, hi, z, ref, rel_tol, ordered)
        cert = result['certificate']
        best = min(best, cert['maximum_bound'])
        if cert['status'] == 'PASS':
            return dict(result, node_digest=_node_digest(result),
                        reuse_status='build' if previous is None else 'rebuild', reuse_reason=reason)
    raise RuntimeError(f'remote Ritz certificate failed through 64 nodes: '
                       f'best bound {best:.3e}, tolerance {rel_tol:.3e}')
