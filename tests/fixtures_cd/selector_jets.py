"""External-frequency selector jets for the frozen Run211 CD split.

All derivatives are with respect to ``x`` in the same units as ``eta``,
``beta``, ``u``, and ``spacing``.  Occupations are held fixed.  The caller
owns the physical-band mask and any final multiplication by eta.
"""
from __future__ import annotations

import numpy as np


def _inputs(x, occ):
    x = np.asarray(x, dtype=np.float64)
    occ = np.broadcast_to(np.asarray(occ, dtype=np.float64), x.shape)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(occ)):
        raise ValueError("selector inputs must be finite")
    if np.any((occ < 0.0) | (occ > 1.0)):
        raise ValueError("occupations must lie in [0,1]")
    return x, occ


def anchor_selector(x, occ, eta, beta, derivative=False):
    """Run211 two-pole-anchor selector, or its piecewise d/dx jet.

    At exactly x=0 this follows Run211's right-branch convention sign=+1.
    No distributional derivative of sign(x) is included.
    """
    x, f = _inputs(x, occ)
    eta, beta = float(eta), float(beta)
    sign = np.where(x >= 0.0, 1.0, -1.0)
    dv, dc = x-1j*eta, x+1j*eta
    if derivative:
        return (-f/(2.0*beta*(beta+sign*dv)**2)
                -(1.0-f)/(2.0*beta*(beta+sign*dc)**2))
    return (f*sign/(2.0*beta*(beta+sign*dv))
            +(1.0-f)*sign/(2.0*beta*(beta+sign*dc)))


def imaginary_selector(x, occ, eta, u, weight, derivative=False):
    """Run211 imaginary-axis remainder selector, or its d/dx jet."""
    x, f = _inputs(x, occ)
    eta, u, weight = float(eta), float(u), float(weight)
    dv, dc = x-1j*eta, x+1j*eta
    if derivative:
        value = (f*(u**2-dv**2)/(dv**2+u**2)**2
                 +(1.0-f)*(u**2-dc**2)/(dc**2+u**2)**2)
    else:
        value = (f*dv/(dv**2+u**2)
                 +(1.0-f)*dc/(dc**2+u**2))
    return (weight/np.pi)*value


def hermite_selectors(x, occ, spacing, derivative=False):
    """Run211 uniform-|x| Hermite selectors, optionally differentiated.

    The returned dictionary has the exact Run211 keys.  With
    ``derivative=True``, each coefficient is its external-x derivative;
    the labels still identify the value-node and derivative-node data that
    coefficient multiplies.  The x=0 branch is the Run211 right branch.
    """
    x, f = _inputs(x, occ)
    spacing = float(spacing)
    if not spacing > 0.0:
        raise ValueError("spacing must be positive")
    query = np.abs(x)
    count = max(2, int(np.ceil(float(np.max(query))/spacing))+1)
    index = np.floor(query/spacing).astype(np.int64)
    index = np.clip(index, 0, count-2)
    t = np.clip(query/spacing-index, 0.0, 1.0)
    residue_sign = np.where(x < 0.0, f, -(1.0-f))
    if derivative:
        dx = np.where(x < 0.0, -1.0, 1.0)
        h00t, h10t = 6*t**2-6*t, 3*t**2-4*t+1
        h01t, h11t = -6*t**2+6*t, 3*t**2-2*t
        return {
            "count": count,
            "index": index,
            "value_left": residue_sign*dx*h00t/spacing,
            "value_right": residue_sign*dx*h01t/spacing,
            "derivative_left": residue_sign*dx*h10t,
            "derivative_right": residue_sign*dx*h11t,
        }
    h00, h10 = 2*t**3-3*t**2+1, t**3-2*t**2+t
    h01, h11 = -2*t**3+3*t**2, t**3-t**2
    return {
        "count": count,
        "index": index,
        "value_left": residue_sign*h00,
        "value_right": residue_sign*h01,
        "derivative_left": residue_sign*spacing*h10,
        "derivative_right": residue_sign*spacing*h11,
    }
