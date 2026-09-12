"""Scalar response-bank quadratures; Ry frequencies and d/ds, s=z**2.

Hermite panel construction ports Run212 (rho/bound and derivative tail) and
Run307 (early panels for unequal decays). Remote positive NNLS rows port
Run183, with adaptive order and an interval certificate instead of its fixed
Na gate. No campaign modules or GW implementation are imported.
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
    for key in ("t", "h", "coefficient_rows"):
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


def _row_certificate(t, weights, n, anchor, hi, intervals):
    """Continuum relative-error bound using linear interpolation of ratio.

    At interval midpoints evaluate ratio second derivatives. Bound their
    variation by an absolute third-derivative bound on each positive term;
    linear interpolation then contributes h**2/8 * sup|ratio''|.
    All variables here are dimensionless, delta/delta_lo.
    """
    edges = np.geomspace(1., hi, intervals+1) if hi > 1 else np.array([1., 1.])
    maximum = 0.
    roundoff = 0.
    for start in range(0, len(edges)-1, 256):
        lo, high = edges[start:start+256], edges[start+1:start+257]
        mid, half = (lo+high)/2, (high-lo)/2
        def terms(x):
            logtarget = np.log(x)-(n+1)*np.log(x*x+anchor*anchor)
            return np.exp(-x[:, None]*t[None, :]-logtarget[:, None])*weights
        gl, gh, gm = terms(lo), terms(high), terms(mid)
        endpoint = np.maximum(np.abs(gl.sum(1)-1), np.abs(gh.sum(1)-1))
        l1 = -t[None, :]+(2*(n+1)*mid/(mid*mid+anchor*anchor)-1/mid)[:, None]
        l2 = (2*(n+1)*(anchor*anchor-mid*mid)/(mid*mid+anchor*anchor)**2+1/mid**2)
        second = np.abs(np.sum(gm*(l1*l1+l2[:, None]), axis=1))
        b1 = t[None, :]+((2*(n+1)+1)/lo)[:, None]
        b2 = (2*(n+1)+1)/lo**2
        b3 = (12*(n+1)+2)/lo**3
        gmax = gm*np.exp(b1*half[:, None])
        third = np.sum(gmax*(b1**3+3*b1*b2[:, None]+b3[:, None]), axis=1)
        rounding = 64*np.finfo(float).eps*len(t)*np.maximum(1., gmax.sum(1))
        bound = endpoint+(high-lo)**2/8*(second+half*third)+rounding
        maximum = max(maximum, float(bound.max()))
        roundoff = max(roundoff, float(rounding.max()))
    return maximum, roundoff


def _reuse_laplace_rule(previous, lo, hi, z, rel_tol):
    """Reuse certified inverse-moment rows; update the Taylor projections."""
    cert = previous["certificate"]
    if _node_digest(previous) != previous.get("node_digest"):
        raise ValueError("response rule node digest mismatch")
    a, b = cert["delta_ry"]
    if cert["rel_tol"] != rel_tol:
        return None, "tolerance changed"
    if lo < a or hi > b:
        return None, "transition interval escaped"
    order, anchor = cert["order"], cert["eta_ry"]
    x = (z*z+anchor*anchor)/(a*a)
    rho = np.abs(x)/(1+(anchor/a)**2)
    if np.any(rho >= 1):
        return None, "current Taylor domain does not converge"
    vr, dr = rho**(order+1), rho**order*((order+1)+order*rho)
    amp = (1+rho)/(1-rho)
    vb = vr+amp*max(cert["row_relative_bounds"])
    db = dr+amp**2*max(cert["row_relative_bounds"])
    if max(vb.max(), db.max()) > rel_tol:
        return None, "current frequency certificate failed"
    powers = (z*z+anchor*anchor)[:, None]**np.arange(order+1)
    dpowers = np.zeros_like(powers)
    dpowers[:, 1:] = np.arange(1, order+1)*(z*z+anchor*anchor)[:, None]**np.arange(order)
    rows = previous["coefficient_rows"]
    result = dict(previous, projection_value=powers@rows,
                  projection_derivative=dpowers@rows)
    result["certificate"] = dict(cert, z_ry=[[v.real, v.imag] for v in z],
        rho=rho.tolist(), value_taylor_bounds=vr.tolist(),
        derivative_taylor_bounds=dr.tolist(), value_bound=vb.tolist(),
        derivative_bound=db.tolist())
    return result, "current domain and frequency certificates pass"


def response_laplace_rule(delta_lo_ry, delta_hi_ry, z_ry, *, rel_tol=1e-8,
                          previous=None, domain_pad_ry=0.0):
    """Positive NNLS inverse-moment rows and remote response projections.

    Parameters
    ----------
    delta_lo_ry, delta_hi_ry : float
        Positive remote transition interval in Ry.
    z_ry : complex ndarray, shape (sample,)
        Current upper-half-plane evaluation points in Ry.
    rel_tol : float
        Relative tolerance for delta/(delta**2-z**2) and its s derivative.

    Returns
    -------
    dict
        Shared positive ``t`` [1/Ry], nonnegative coefficient rows,
        ``projection_value`` and ``projection_derivative`` [sample,time].
        Bank multiplies projections by -exp(-gap*t); paired-stream factor
        two belongs to the bank. Certificate includes continuum NNLS row
        bounds and value/derivative Taylor remainders for every supplied z.
        Refuses a nonconvergent Taylor domain or an unmet NNLS certificate.
    """
    from scipy.optimize import nnls
    z = _inputs(z_ry, rel_tol)
    lo, hi = float(delta_lo_ry), float(delta_hi_ry)
    if not np.isfinite(lo+hi) or not 0 < lo <= hi:
        raise ValueError('remote Laplace interval must be finite and positive')
    if not np.isfinite(domain_pad_ry) or domain_pad_ry < 0:
        raise ValueError('domain_pad_ry must be finite and nonnegative')
    reason = "initial rule"
    if previous is not None:
        reused, reason = _reuse_laplace_rule(previous, lo, hi, z, rel_tol)
        if reused is not None:
            return dict(reused, reuse_status="hit", reuse_reason=reason)
    eta = float(z.imag.min())
    padded_lo = max(lo-domain_pad_ry, lo/2)
    # Padding must not move a valid remote cell across its Taylor boundary.
    # Keeping the physical lower edge still admits later gap increases.
    if np.all(np.abs(z*z+eta*eta) < padded_lo*padded_lo+eta*eta):
        lo = padded_lo
    hi += domain_pad_ry
    anchor = eta/lo
    a0 = 1+anchor*anchor
    x = (z/lo)**2+anchor*anchor
    rho = np.abs(x)/a0
    if np.any(rho >= 1):
        raise ValueError('remote Taylor domain does not converge; repartition in bank owner')
    # N+1 powers for value; derivative remainder is the differentiated
    # geometric remainder, bounded relative to the exact squared resolvent.
    for order in range(1, 65):
        vr = rho**(order+1)
        dr = rho**order*((order+1)+order*rho)
        if max(vr.max(), dr.max()) <= rel_tol/4:
            break
    else:
        raise RuntimeError('remote Taylor order budget exceeded')
    # Propagate positive row-relative errors through complex Taylor powers.
    amp_v = (1+rho)/(1-rho)
    amp_d = ((1+rho)/(1-rho))**2
    row_tol = rel_tol/(4*float(max(amp_v.max(), amp_d.max())))
    train = np.geomspace(1., hi/lo, 4096)
    targets = [train/(train*train+anchor*anchor)**(n+1) for n in range(order+1)]
    last = None
    for node_count in (48, 96, 192):
        stop = max(np.log(2e12), 4*(order+1)-np.log(row_tol))
        gx, _ = _legendre(node_count)
        t = stop*(gx+1)/2
        basis = np.exp(-train[:, None]*t)
        rows = []
        fit_errors = []
        for target in targets:
            matrix = basis/target[:, None]
            scale = np.maximum(np.linalg.norm(matrix, axis=0), np.finfo(float).tiny)
            try:
                w = nnls(matrix/scale, np.ones(len(train)), maxiter=100*node_count)[0]/scale
            except RuntimeError:
                break
            rows.append(w)
            fit_errors.append(float(np.max(np.abs(matrix@w-1))))
        if len(rows) != order+1 or max(fit_errors) > row_tol/2:
            last = dict(nodes=node_count, fit_errors=fit_errors)
            continue
        errors, rounding = [], []
        for n, w in enumerate(rows):
            for intervals in (4096, 8192, 16384, 32768, 65536):
                error, rnd = _row_certificate(t, w, n, anchor, hi/lo, intervals)
                if error <= row_tol:
                    break
            errors.append(error)
            rounding.append(rnd)
        if max(errors) <= row_tol:
            break
        last = dict(nodes=node_count, interval_errors=errors)
    else:
        raise RuntimeError(f'remote NNLS certificate failed: {last}')
    rows = np.asarray(rows)
    powers = x[:, None]**np.arange(order+1)
    dpowers = np.zeros_like(powers)
    dpowers[:, 1:] = np.arange(1, order+1)*x[:, None]**np.arange(order)
    value, derivative = (powers@rows)/lo, (dpowers@rows)/lo**3
    vb = vr+amp_v*max(errors)
    db = dr+amp_d*max(errors)
    if max(vb.max(), db.max()) > rel_tol:
        raise RuntimeError('remote combined certificate failed')
    result = dict(t=t/lo, coefficient_rows=rows/lo**(2*np.arange(order+1)[:, None]+1),
                projection_value=value, projection_derivative=derivative,
                certificate=dict(status='PASS', scope='continuum delta interval; all supplied z',
                    norm='relative value and ds', delta_ry=[lo, hi],
                    z_ry=[[v.real, v.imag] for v in z], eta_ry=eta,
                    rel_tol=rel_tol, order=order, rho=rho.tolist(),
                    row_relative_bounds=errors, row_roundoff_allowance=rounding,
                    value_taylor_bounds=vr.tolist(), derivative_taylor_bounds=dr.tolist(),
                    value_bound=vb.tolist(), derivative_bound=db.tolist(),
                    owner='Run183 positive NNLS inverse-moment rows'))
    return dict(result, node_digest=_node_digest(result),
                reuse_status="build" if previous is None else "rebuild",
                reuse_reason=reason)
