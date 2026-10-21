"""Rules for 1/x = int_0^inf exp(-x t) dt on x in [1, R] (units of Delta_min).

minimax: LORRAX's static rule, minimax.levelled.noncrossing_levelled
         (services/minimax/src/minimax/levelled.py:316), called read-only.
uniform: midpoint rule t_k = (k + 1/2) h, weights h, with h chosen to minimise
         the sup error on [1, R] for the given N (the best a uniform grid can do).
gauss:   Gauss-Legendre on [0, T], T chosen the same way (a generic rule for
         smooth integrands on a finite interval).
Errors are absolute, in units of 1/Delta_min: the currency the static rule is
certified in (src/gw/minimax_screening.py:1062-1110).
"""
from functools import lru_cache

import numpy as np

from lorrax_minimax import noncrossing_levelled

XS_CACHE = {}


def xs(R, n=3000):
    key = (R, n)
    if key not in XS_CACHE:
        XS_CACHE[key] = np.geomspace(1.0, R, n)
    return XS_CACHE[key]


def evaluate(t, w, x):
    return np.exp(-np.outer(x, t)) @ w


def sup_error(t, w, R):
    x = xs(R)
    return float(np.max(np.abs(evaluate(t, w, x) - 1.0 / x)))


@lru_cache(maxsize=None)
def minimax(R, N):
    """The N-node levelled rule: with an unreachable target the solver
    returns its N_max rule and that rule's certified error."""
    t, w, n, err = noncrossing_levelled(R, 1e-300, N_max=N)
    return np.asarray(t), np.asarray(w), float(err)


def _scan(f, lo, hi):
    """Coarse log scan of a one-parameter family, then a fine scan around the best."""
    best = min((f(p), p) for p in np.geomspace(lo, hi, 160))
    return min((f(p), p) for p in np.geomspace(best[1] / 1.2, best[1] * 1.2, 60))


def uniform_rule(h, N):
    t = (np.arange(N) + 0.5) * h
    return t, np.full(N, h)


@lru_cache(maxsize=None)
def uniform_best(R, N):
    best = _scan(lambda h: sup_error(*uniform_rule(h, N), R), 1e-4, 20.0)
    t, w = uniform_rule(best[1], N)
    return t, w, best[0]


def gauss_rule(T, N):
    u, v = np.polynomial.legendre.leggauss(N)
    return 0.5 * T * (u + 1.0), 0.5 * T * v


@lru_cache(maxsize=None)
def gauss_best(R, N):
    best = _scan(lambda T: sup_error(*gauss_rule(T, N), R), 0.5, 2000.0)
    t, w = gauss_rule(best[1], N)
    return t, w, best[0]
