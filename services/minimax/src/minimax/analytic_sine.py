"""Pole-aware analytic sine rule for 1/x on [-A,-1] U [1,A].

Keeps the earlier odd-harmonic minimax core. Replaces the smooth correction
by an interpolant at roots of V_d(y)-q*V_{d-1}(y), where V denotes the
third-kind Chebyshev polynomial. A Stieltjes representation proves

  correction_error <= h*q**(m+3/2)/(1-q).

All certificates refer to ideal exact-arithmetic coefficients. Coefficient
construction, special-function evaluation, and runtime rounding are separate.
No nonlinear fit, LP, empirical coefficient pruning, or time-node search is used.
Default eta=0.8 controls the inherited padding ONLY, not an ellipse safety factor.
This module currently certifies ABSOLUTE error only.

Dependencies: numpy, scipy. Optional mpmath for elevated-precision construction.
"""
from dataclasses import dataclass
import math
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike, NDArray
from numpy.polynomial.chebyshev import chebval
from scipy.fft import dct, dst
from scipy.linalg import eigh_tridiagonal
from scipy.integrate import quad


@dataclass
class SineRule:
    times: NDArray[np.float64]
    weights: NDArray[np.float64]
    A: float
    epsilon: float
    h: float
    alpha: float
    n_core: int
    m_correction: int
    core_bound: float
    correction_bound: float
    q_correction: float
    eta: float
    correction_weights: NDArray[np.float64]
    interpolation_points: NDArray[np.float64]

    @property
    def bound(self) -> float:
        """Uniform absolute-error bound, in exact arithmetic."""
        return self.core_bound + self.correction_bound

    @property
    def node_count(self) -> int:
        return len(self.times)

    def evaluate(self, x: ArrayLike, block: int = 512) -> NDArray[np.float64]:
        if block < 1:
            raise ValueError('block must be positive')
        xx = np.asarray(x, dtype=float)
        flat = xx.reshape(-1)
        out = np.empty_like(flat)
        for k in range(0, flat.size, block):
            out[k:k+block] = np.sin(flat[k:k+block, None]*self.times) @ self.weights
        return out.reshape(xx.shape)

    def derivative(self, x: ArrayLike, block: int = 512) -> NDArray[np.float64]:
        if block < 1:
            raise ValueError('block must be positive')
        xx = np.asarray(x, dtype=float)
        flat = xx.reshape(-1)
        out = np.empty_like(flat)
        for k in range(0, flat.size, block):
            out[k:k+block] = np.cos(flat[k:k+block, None]*self.times) @ (self.times*self.weights)
        return out.reshape(xx.shape)

    def correction(self, theta: ArrayLike) -> NDArray[np.float64]:
        tt = np.asarray(theta, dtype=float)
        j = np.arange(1, self.m_correction+2)
        return (np.sin(tt.reshape(-1, 1)*j) @ self.correction_weights).reshape(tt.shape)


def correction_function(v: ArrayLike) -> NDArray[np.float64]:
    """G(v) defined by 1/theta-csc(theta)=sin(theta)*G(sin(theta/2)^2)."""
    v = np.asarray(v, dtype=float)
    if np.any(v < 0) or np.any(v >= 1):
        raise ValueError('This real implementation requires 0 <= v < 1')
    out = np.empty_like(v)
    small = v < .04
    if np.any(small):
        K = 18
        central = np.ones(K+2, dtype=np.longdouble)
        arcsin = np.ones(K+2, dtype=np.longdouble)
        beta = np.ones(K+2, dtype=np.longdouble)
        for k in range(1, K+2):
            central[k] = central[k-1]*(2*k-1)/(2*k)
            arcsin[k] = central[k]/(2*k+1)
            beta[k] = central[k] - np.dot(arcsin[1:k+1], beta[k-1::-1])
        gc = np.asarray((beta[1:]-1)/4, dtype=float)
        out[small] = np.polynomial.polynomial.polyval(v[small], gc)
    if np.any(~small):
        vv = v[~small]
        av = np.arcsin(np.sqrt(vv))/np.sqrt(vv)
        out[~small] = (1/(np.sqrt(1-vv)*av)-1/(1-vv))/(4*vv)
    return out


def parameters(A: float, epsilon: float, eta: float = .8,
               alpha: Optional[float] = None) -> dict:
    """Closed sufficient degrees; eta changes padding, not the new proof."""
    A, epsilon, eta = float(A), float(epsilon), float(eta)
    if not math.isfinite(A) or A <= 1:
        raise ValueError('A must be finite and greater than 1')
    if not (math.isfinite(epsilon) and 0 < epsilon < 1):
        raise ValueError('epsilon must lie in (0,1)')
    if not (0 < eta < 1):
        raise ValueError('eta must lie in (0,1)')
    if alpha is None:
        alpha = math.pi*math.sqrt(eta*A)/(1+math.sqrt(eta*A))
    alpha = float(alpha)
    if not (0 < alpha <= math.pi*A/(A+1)):
        raise ValueError('Need 0 < alpha <= pi*A/(A+1)')
    h = alpha/A
    s = math.sin(h)
    kappa = math.atanh(s)
    ell = -2*math.log(math.tan(alpha/4))
    q = math.exp(-ell)
    eps_core = epsilon*ell/(ell+kappa)
    eps_corr = epsilon-eps_core
    n = max(1, math.ceil(math.log(h*(1+s)/(s*eps_core))/(2*kappa)))
    m = max(0, math.ceil(math.log(h/(eps_corr*(-math.expm1(-ell))))/ell - 1.5))
    core_bound = h*(1+s)/s*math.exp(-2*n*kappa)
    corr_bound = h*math.exp(-(m+1.5)*ell)/(-math.expm1(-ell))
    N = max(m+1, n+(m+1)//2)
    return dict(A=A, epsilon=epsilon, eta=eta, alpha=alpha, h=h,
                s=s, kappa=kappa, V=math.sin(alpha/2)**2,
                q=q, ell=ell, eps_core=eps_core, eps_corr=eps_corr,
                n=n, m=m, N=N, core_bound=core_bound,
                correction_bound=corr_bound, bound=core_bound+corr_bound)


def _core_weights(n: int, h: float) -> np.ndarray:
    s = math.sin(h)
    kappa = math.atanh(s)
    theta = math.pi*(np.arange(n+1)+.5)/(2*(n+1))
    d = s*s - np.sin(theta)**2
    Tn = np.empty_like(theta)
    Tm = np.empty_like(theta)
    mask = d >= 0
    u = 2*np.arcsinh(np.sqrt(d[mask]/(1-s*s)))
    Tn[mask], Tm[mask] = np.cosh(n*u), np.cosh((n-1)*u)
    u = 2*np.arcsin(np.sqrt(np.clip(-d[~mask]/(1-s*s), 0, 1)))
    Tn[~mask], Tm[~mask] = np.cos(n*u), np.cos((n-1)*u)
    q = math.exp(-2*kappa)
    D = .5*(-math.expm1(-4*kappa))*math.exp(2*n*kappa)
    rc = dct((Tn-q*Tm)/D, type=2)/(n+1)
    rc[0] *= .5
    return 2*h*np.cumsum(rc[:0:-1])[::-1]


def _correction_weights(p: dict, precision: int = 0):
    d, h, V, q = p['m']+1, p['h'], p['V'], p['q']
    diagonal = np.zeros(d)
    diagonal[0] += .5
    diagonal[-1] += q/2
    y = eigh_tridiagonal(diagonal, np.full(d-1, .5), eigvals_only=True)
    v = .5*V*(1+y)
    if precision:
        import mpmath as mp
        if precision < 30:
            raise ValueError('precision must be zero or at least 30 decimal digits')
        with mp.workdps(precision):
            # Refine roots of the explicitly prescribed polynomial; NOT a minimax fit.
            qq = mp.tan(mp.mpf(p['alpha'])/4)**2
            VV = mp.sin(mp.mpf(p['alpha'])/2)**2
            def P(z):
                v0 = mp.mpf(1)
                if d == 1:
                    return 2*z-1-qq
                v1 = 2*z-1
                for k in range(2,d+1):
                    v0,v1=v1,2*z*v1-v0
                return v1-qq*v0
            yy = [mp.findroot(P, mp.mpf(float(z)), solver='newton') for z in y]
            vv = [VV*(1+z)/2 for z in yy]
            gd = [(mp.sqrt(z)/(mp.sqrt(1-z)*mp.asin(mp.sqrt(z)))-1/(1-z))/(4*z) for z in vv]
            # Newton interpolation and sine conversion, at elevated precision.
            dd = gd.copy()
            for k in range(1,d):
                for j in range(d-1,k-1,-1):
                    dd[j] = (dd[j]-dd[j-1])/(vv[j]-vv[j-k])
            vals = []
            for j in range(1,d+1):
                th = mp.pi*j/(d+1)
                z = (1-mp.cos(th))/2
                val = dd[-1]
                for k in range(d-2,-1,-1):
                    val = dd[k]+(z-vv[k])*val
                vals.append(mp.mpf(h)*mp.sin(th)*val)
            wc = np.array([float(2*mp.fsum(vals[j-1]*mp.sin(mp.pi*j*k/(d+1)) for j in range(1,d+1))/(d+1)) for k in range(1,d+1)])
            return wc, np.array([float(z) for z in vv])
    # A small linear polynomial interpolation, no free-frequency optimization.
    M = np.cos(np.arccos(y)[:,None]*np.arange(d)[None,:])
    cc = np.linalg.solve(M, correction_function(v))
    th = math.pi*np.arange(1,d+1)/(d+1)
    qc = h*np.sin(th)*chebval((1-np.cos(th))/V-1, cc)
    return dst(qc, type=1)/(d+1), v


def make_rule(A: float, epsilon: float, eta: float = .8,
              alpha: Optional[float] = None, precision: int = 0) -> SineRule:
    """Construct the new rule. precision=60 optionally improves coefficient generation.

    The default implementation is float64, audited in the accompanying files
    over A=10..1000 and epsilon=1e-3..1e-9. The analytic bound is not a bound
    for all floating-point errors. Re-audit outside that tested range.
    """
    p = parameters(A,epsilon,eta=eta,alpha=alpha)
    n, m, h = p['n'], p['m'], p['h']
    core = _core_weights(n,h)
    corr, v = _correction_weights(p, precision=precision)
    degree = max(2*n-1,m+1)
    w = np.zeros(degree)
    w[:2*n-1:2] += core
    w[:m+1] += corr
    active = np.zeros(degree,dtype=bool)
    active[:2*n-1:2] = True
    active[:m+1] = True
    return SineRule(h*np.arange(1,degree+1)[active],w[active],p['A'],p['epsilon'],
                    h,p['alpha'],n,m,p['core_bound'],p['correction_bound'],
                    p['q'],p['eta'],corr,v)


def amplitude_factor(rule: SineRule, v: float) -> tuple[float,float]:
    """K_m(v) in the exact correction error formula; adaptive scalar quadrature.

    Returns factor and the quadrature routine's (not rigorous) error estimate.
    The closed parameter bound does NOT require evaluating this function.
    """
    V = math.sin(rule.alpha/2)**2
    if not (0 <= v <= V):
        raise ValueError('Need 0 <= v <= sin(alpha/2)^2')
    q, d = rule.q_correction, rule.m_correction+1
    def f(u):
        if u > 30:
            return 0.
        s = math.cosh(u)**2
        gamma = math.acosh(2*s/V-1)
        r = math.exp(-gamma)
        ratio = ((1-q)*math.exp(d*(math.log(r)-math.log(q)))*(1+r)
                 /(1-q*r+(r-q)*r**(2*d)))
        return (1-v)/(s-v)*ratio/(u*u+math.pi**2/4)
    integral, est = quad(f,0,30,epsabs=5e-14,epsrel=5e-13,limit=300)
    return 1-integral,est


def audit(rule: SineRule, points: int = 20001) -> dict:
    if points < 3:
        raise ValueError('points must be at least 3')
    x = np.unique(np.r_[np.linspace(1,rule.A,points),np.geomspace(1,rule.A,max(3,points//4))])
    err = rule.evaluate(x)-1/x
    return dict(A=rule.A,epsilon=rule.epsilon,N=rule.node_count,n=rule.n_core,
                m=rule.m_correction,analytic_bound=rule.bound,
                core_bound=rule.core_bound,correction_bound=rule.correction_bound,
                sampled_max_error=float(np.max(np.abs(err))),
                sampled_worst_x=float(x[np.argmax(np.abs(err))]),
                weight_l1=float(np.sum(np.abs(rule.weights))),
                maximum_time=float(rule.times[-1]))


if __name__ == '__main__':
    import argparse,json
    ap=argparse.ArgumentParser()
    ap.add_argument('--A',type=float,default=20)
    ap.add_argument('--epsilon',type=float,default=1e-6)
    ap.add_argument('--precision',type=int,default=0)
    ap.add_argument('--output',default='')
    args=ap.parse_args()
    r=make_rule(args.A,args.epsilon,precision=args.precision)
    print(json.dumps(audit(r),indent=2))
    if args.output:
        np.savez(args.output,times=r.times,weights=r.weights,A=r.A,
                 epsilon=r.epsilon,analytic_bound=r.bound,h=r.h)


def asymptotic_alpha(A: float, epsilon: float) -> float:
    """Analytic padding that optimizes the sqrt(A) count coefficient (fixed epsilon).

    Use make_rule(A, epsilon, alpha=asymptotic_alpha(A, epsilon)).
    This is not a finite-A integer-count optimization.
    """
    if not (math.isfinite(A) and A > 1 and 0 < epsilon < 1):
        raise ValueError('Need finite A > 1 and 0 < epsilon < 1')
    L = math.log(1/epsilon)
    delta = math.sqrt(A*(1+1/L))
    return math.pi*A/(A+delta)
