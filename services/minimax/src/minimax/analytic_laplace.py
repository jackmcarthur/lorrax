"""Positive exponential approximation to 1/x on the real interval [1,R].

Construction: elliptic interpolation abscissae -> positive multipoint Laplace
quadrature -> optional frozen-Blaschke-envelope corrections.

This is a semi-analytic numerical construction, NOT a closed-form formula for
minimax time nodes. The inner quadrature construction solves nonlinear moment
conditions at prescribed abscissae in elevated precision. Each optional outer
correction solves one explicit rational/Cauchy linear system, then reconstructs
the multipoint quadrature. No convergence theorem is claimed for these outer
corrections. Returned arrays use float64 and are independently audited.

Representation: Q(x)=sum_j strengths[j]*exp(-(x-1)*times[j]).
The degree is supplied explicitly. No analytic sufficient degree bound is
claimed for the corrected rule.

Dependencies: numpy, scipy, mpmath. Tested for R=10..10000, degree=6..19.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil
from typing import Any
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import roots_jacobi
from scipy.optimize import brentq
import mpmath as mp

@dataclass(frozen=True)
class TargetedLaplaceRule:
    R: float
    times: NDArray[np.float64]
    strengths: NDArray[np.float64]
    interpolation_points: NDArray[np.float64]
    shift: float
    corrections: int
    construction_digits: int
    history: tuple[dict[str, Any], ...]

    @property
    def degree(self) -> int:
        return len(self.times)

    def evaluate(self, x: ArrayLike):
        x = np.asarray(x)
        return np.exp(-(x[..., None]-1)*self.times) @ self.strengths

    def save(self, path: str):
        np.savez(path, R=self.R, times=self.times, strengths=self.strengths,
                 interpolation_points=self.interpolation_points,
                 shift=self.shift, corrections=self.corrections)


def elliptic_abscissae(R: float, degree: int, shift: float = 0.5,
                       digits: int = 60) -> NDArray[np.float64]:
    """2N shifted Zolotarev zeros, not the final time nodes."""
    if not np.isfinite(R) or R <= 1:
        raise ValueError('Require finite R>1.')
    if isinstance(degree, bool) or int(degree)!=degree or degree<1:
        raise ValueError('degree must be a positive integer.')
    if not np.isfinite(shift) or not 0 <= shift < 1:
        raise ValueError('Require 0 <= shift < 1.')
    with mp.workdps(digits):
        c=mp.mpf(float(shift)); b=mp.mpf(float(R))-c; a=1-c
        m=1-(a/b)**2; K=mp.ellipk(m)
        points=[c+b*mp.ellipfun('dn',(j+mp.mpf('.5'))*K/(2*degree),m)
                for j in range(2*degree)]
        return np.array(sorted(float(v) for v in points))


def _moment_solve(points, times, strengths, digits, maxiter=100):
    """Newton in log time/strength coordinates for prescribed moments.

    We cap parameter-space steps. Requiring a monotone unpreconditioned moment
    residual can stall on these very ill-conditioned moment systems.
    """
    n=len(times)
    with mp.workdps(digits):
        x=[mp.mpf(float(v))-1 for v in points]
        t=[mp.mpf(float(v)) for v in times]
        g=[mp.mpf(float(v)) for v in strengths]
        tol=mp.power(10, -(digits-18))
        for iteration in range(maxiter):
            vals=[[g[j]*mp.exp(-xx*t[j]) for j in range(n)] for xx in x]
            f=mp.matrix([sum(row)-1/(xx+1) for xx,row in zip(x,vals)])
            if mp.norm(f,mp.inf)<tol:
                ix=sorted(range(n),key=lambda j:t[j])
                return (np.array([float(t[j]) for j in ix]),
                        np.array([float(g[j]) for j in ix]))
            J=mp.matrix(2*n,2*n)
            for k,xx in enumerate(x):
                for j in range(n):
                    J[k,j]=vals[k][j]
                    J[k,n+j]=-xx*t[j]*vals[k][j]
            try:
                d=mp.lu_solve(J,-f)
            except (ZeroDivisionError, ValueError) as ex:
                raise RuntimeError('Moment Jacobian failed; increase precision.') from ex
            scale=min(mp.mpf(1),mp.mpf('.25')/mp.norm(d,mp.inf))
            g=[g[j]*mp.exp(scale*d[j]) for j in range(n)]
            t=[t[j]*mp.exp(scale*d[n+j]) for j in range(n)]
        raise RuntimeError('Prescribed-moment Newton iteration did not converge.')


def _move_moments(old, new, times, strengths, digits):
    """Geometric homotopy of ordered abscissae; no minimax optimization."""
    distance=float(np.max(np.abs(np.log(new/old))))
    pieces=max(1,ceil(distance/.05))
    t,g=times.copy(),strengths.copy()
    for a in np.linspace(0,1,pieces+1)[1:]:
        target=np.exp((1-a)*np.log(old)+a*np.log(new))
        t,g=_moment_solve(target,t,g,digits)
    # Last target uses the exact supplied float64 abscissae.
    return _moment_solve(new,t,g,digits)


def _initial_quadrature(points, digits):
    n=len(points)//2
    start=np.linspace(points[0],points[-1],2*n)
    h=(start[-1]-start[0])/(2*n-1)
    a0=start[0]
    # q=exp(-h*t): e^(-a0*t)dt = q^(a0/h-1)dq/h.
    q,w=roots_jacobi(n,0,a0/h-1)
    q=(q+1)/2
    weights=w/(h*2**(a0/h))
    t=-np.log(q)/h
    g=weights*np.exp((a0-1)*t)
    ix=np.argsort(t);t,g=t[ix],g[ix]
    t,g=_moment_solve(start,t,g,digits)
    return _move_moments(start,points,t,g,digits)


def _extrema(R,points,times,strengths,digits=60):
    """All relevant extrema for an interpolating positive exponential rule.

    Between each adjacent pair of its 2N error zeros is one stationary point.
    The final positive maximum can be before R or at R. All function evaluations
    used for root signs and final errors use mpmath. This is not interval arithmetic.
    """
    with mp.workdps(digits):
        tt=[mp.mpf(float(v)) for v in times]
        gg=[mp.mpf(float(v)) for v in strengths]
        def deriv(x):
            xx=mp.mpf(float(x))
            return float(-1/xx**2+sum(g*t*mp.exp(-(xx-1)*t) for t,g in zip(tt,gg)))
        xx=[1.0]
        for a,b in zip(points[:-1],points[1:]):
            if deriv(a)*deriv(b)>=0:
                raise RuntimeError('Derivative signs do not bracket the expected alternant.')
            xx.append(brentq(deriv,a,b,xtol=5e-15,rtol=2e-14))
        if deriv(R)<0:
            xx.append(brentq(deriv,points[-1],R,xtol=5e-15,rtol=2e-14))
        else:
            xx.append(float(R))
        ee=[]
        for v in xx:
            x=mp.mpf(float(v))
            ee.append(float(1/x-sum(g*mp.exp(-(x-1)*t) for t,g in zip(tt,gg))))
        xx=np.array(xx);ee=np.array(ee)
        if np.any(np.sign(ee)!=(-1.)**np.arange(2*len(times)+1)):
            raise RuntimeError('No full positive-start alternating error sequence.')
        return xx,ee


def _snapshot(R,points,t,g,digits,step):
    x,e=_extrema(R,points,t,g,digits)
    hi=float(np.max(np.abs(e)));lo=float(np.min(np.abs(e)))
    return dict(step=step, maximum_error=hi, alternating_lower_bound=lo,
                numerical_optimality_ratio=hi/lo, extrema=x.tolist(),
                errors_at_extrema=e.tolist(), sum_strengths=float(g.sum()),
                minimum_strength=float(g.min()), t_min=float(t.min()),t_max=float(t.max()))


def _frozen_envelope_update(R,points,t,g,shift,digits):
    mu,errors=_extrema(R,points,t,g,digits)
    count=len(points)
    with mp.workdps(digits):
        c=mp.mpf(float(shift))
        beta=[mp.mpf(float(v))-c for v in points]
        A=mp.matrix(count+1,count+1);rhs=mp.matrix(count+1,1)
        for i,v in enumerate(mu):
            y=mp.mpf(float(v))-c
            for j in range(count):
                A[i,j]=-2*y*beta[j]/(y*y-beta[j]*beta[j])
            A[i,count]=-1
            rhs[i]=-mp.log(abs(mp.mpf(float(errors[i]))))
        d=mp.lu_solve(A,rhs)
        move=np.array([float(d[j]) for j in range(count)])
    factor=1.0
    for _ in range(30):
        new=shift+(points-shift)*np.exp(factor*move)
        if new[0]>1 and new[-1]<R and np.all(np.diff(new)>0):
            return new
        factor*=.5
    raise RuntimeError('Envelope correction failed to preserve ordered interior points.')


def make_rule(R: float, degree: int, *, corrections: int=2, shift: float=.5,
              digits: int|None=None) -> TargetedLaplaceRule:
    """Build and numerically audit a fixed-degree rule.

    corrections=0: prescribed elliptic grid plus nonlinear Gaussian moment solve.
    corrections>0: additionally correct the observed envelope; this is an iterative
    numerical approximation procedure, not an exact analytical minimax formula.
    """
    if corrections<0 or int(corrections)!=corrections:
        raise ValueError('corrections must be a nonnegative integer.')
    digits=max(50,2*int(degree)+25) if digits is None else int(digits)
    if digits<35:raise ValueError('Use at least 35 decimal digits for construction.')
    points=elliptic_abscissae(R,degree,shift,digits)
    t,g=_initial_quadrature(points,digits)
    hist=[_snapshot(R,points,t,g,digits,0)]
    for step in range(1,corrections+1):
        proposed=_frozen_envelope_update(R,points,t,g,shift,digits)
        tn,gn=_move_moments(points,proposed,t,g,digits)
        snap=_snapshot(R,proposed,tn,gn,digits,step)
        # Conservative safeguard, not a convergence theorem.
        if snap['maximum_error']>hist[-1]['maximum_error']*1.001:
            raise RuntimeError('Envelope correction increased the error; reduce its step.')
        points,t,g=proposed,tn,gn
        hist.append(snap)
    return TargetedLaplaceRule(float(R),t,g,points,float(shift),int(corrections),digits,tuple(hist))


def audit(rule: TargetedLaplaceRule, points: int=10001) -> dict[str,Any]:
    out=_snapshot(rule.R,rule.interpolation_points,rule.times,rule.strengths,
                  rule.construction_digits,rule.corrections)
    grid=np.unique(np.r_[np.linspace(1,rule.R,points),np.geomspace(1,rule.R,points)])
    worst=0.;runtime=0.
    for xx in np.array_split(grid,max(1,ceil(len(grid)/2048))):
        xl=xx.astype(np.longdouble)
        accurate=np.exp(-(xl[:,None]-1)*rule.times.astype(np.longdouble))@rule.strengths.astype(np.longdouble)
        worst=max(worst,float(np.max(np.abs(1/xl-accurate))))
        runtime=max(runtime,float(np.max(np.abs(rule.evaluate(xx).astype(np.longdouble)-accurate))))
    out.update(R=rule.R,degree=rule.degree,sampled_max_abs_error=worst,
               sampled_float64_execution_error=runtime,construction_digits=rule.construction_digits,
               positive_weights=bool(np.all(rule.strengths>0)),
               comment='Extremum search and alternating lower bound evaluated in elevated precision; not an interval-arithmetic enclosure.')
    return out

if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--R',type=float,default=1000)
    p.add_argument('--degree',type=int,default=11)
    p.add_argument('--corrections',type=int,default=2)
    p.add_argument('--save')
    a=p.parse_args()
    rule=make_rule(a.R,a.degree,corrections=a.corrections)
    if a.save:rule.save(a.save)
    print(json.dumps(audit(rule),indent=2))
