"""Contour-deformation laws for Sigma_c in the full-matrix layout (CD production route).

The equations are those of the Fe harness of record (``ordered_cd.py``, lane TRREF; its
diagonal form ``cd_diag.py`` reproduces ``fe_cd_states.csv``) and the Run211 selector jets
(``selector_jets.py``), rewritten once for one layout and one unit system (Ry):

    a[k, m, n, l]      = -(1/nk) <k m, l| W_c,q(z) |k n, l>      (contraction with the pair's own W_q)
    at[k, m, n, l]     = -(1/nk) <k m, l| W_c,-q(z)^T |k n, l>   (read by occupied residues, x < 0)
    d, dt              = the same contractions of -dW_c/ds, s = z^2
    x[k, E, l]         = omega_E - eps_l(k - q)                  (Ry; one common omega grid)
    f[k, l]            = occupation of the intermediate state l at k - q

Every kernel is a closed form of (x, f) and is evaluated in-kernel, never tabulated over
nodes (memory plan, cd_production_design.md section 3).  ``apply(a, w)`` contracts the
intermediate band: out[E, k, m, n] = sum_l a[k, m, n, l] w[k, E, l].

    Sigma_c = anchor + sum_i imag_i - anchor_subtraction + sum_j real_j

with the two-pole anchor at z = i eta (beta = BETA_RY), the tan-mapped Gauss rule on the
imaginary axis (gw.cd.rule), and Hermite residues on the uniform knot grid x_j = j h.
Shapes are documented per function; all arrays are jax arrays inside a trace.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from common.units import RYD_TO_EV

#: Two-pole anchor energies (eV) of the Run211/TRREF split, as module constants hashed
#: into the recipe (not deck keys).
BETA_EV = np.array([0.5, 4.0])
BETA_RY = BETA_EV / float(RYD_TO_EV)


def apply(a, w):
    """out[E, k, m, n] = sum_l a[k, m, n, l] * w[k, E, l]."""
    return jnp.einsum('kmnl,kel->ekmn', a, w)


def own_apply(a, w):
    """Own-energy diagonal: out[k, m] = sum_l a[k, m, m, l] * w[k, m, l] (E bound to the bra m)."""
    diag = jnp.moveaxis(jnp.diagonal(a, axis1=1, axis2=2), -1, 1)       # [k, m, l]
    return jnp.einsum('kml,kml->km', diag, w)


def dagger(a):
    """Contraction of X -> contraction of X^dagger: conj and swap (m, n)."""
    return jnp.conj(jnp.swapaxes(a, 1, 2))


def kernels(x, f, eta):
    """f broadcast to x's shape, and the two denominators x -+ i eta ([k, E, l])."""
    fb = jnp.broadcast_to(f[:, None, :], x.shape)
    return fb, x - 1j * eta, x + 1j * eta


# --------------------------------------------------------------------------- imaginary axis
def imag_node_kernels(u, w, x, f, eta):
    """Minus/plus-branch selectors of one imaginary node (u, w) [k, E, l]."""
    fb, dv, dc = kernels(x, f, eta)
    minus = (fb / (dv - 1j * u) + (1.0 - fb) / (dc - 1j * u)) * (w / (2.0 * np.pi))
    plus = (fb / (dv + 1j * u) + (1.0 - fb) / (dc + 1j * u)) * (w / (2.0 * np.pi))
    return minus, plus


def imag_node_terms(a, u, w, x, f, eta):
    """One imaginary node as (operators [k, m, n, 2l], weights [k, E, 2l]): a and its dagger stacked on l."""
    minus, plus = imag_node_kernels(u, w, x, f, eta)
    return jnp.concatenate([a, dagger(a)], axis=-1), jnp.concatenate([minus, plus], axis=-1)


def imag_raw_node(a, u, w, x, f, eta, apply_fn=apply):
    """Anchor-free contribution of one imaginary node: [E, k, m, n] (``apply_fn = own_apply``: [k, m])."""
    return apply_fn(*imag_node_terms(a, u, w, x, f, eta))


def imag_anchor_sums(u_nodes, w_nodes, x, f, eta, beta):
    """S_-(beta), S_+(beta) for each beta: list of ([k,E,l], [k,E,l]) summed over the rule.

    One ``lax.scan`` over the nodes in rule order (the summation order of ordered_cd's loop), so the
    trace is one node body however many nodes the rule has.
    """
    u_nodes, w_nodes = jnp.asarray(u_nodes), jnp.asarray(w_nodes)
    zero = jnp.zeros(x.shape, jnp.complex128)

    def body(carry, uw):
        u, w = uw
        minus, plus = imag_node_kernels(u, w, x, f, eta)
        return tuple((sm + minus / (u * u + b * b), sp + plus / (u * u + b * b))
                     for (sm, sp), b in zip(carry, beta)), None
    out, _ = jax.lax.scan(body, tuple((zero, zero) for _ in beta), (u_nodes, w_nodes), unroll=1)
    return [tuple(pair) for pair in out]


# --------------------------------------------------------------------------- anchor at i eta
def anchor_fit(value, derivative_s, eta, beta):
    """Two-pole anchor matching value and d/ds at i*eta (decisive_cd.py:22-26)."""
    derivative = -derivative_s
    d1, d2 = eta ** 2 + beta[0] ** 2, eta ** 2 + beta[1] ** 2
    y1 = (derivative + value / d2) / (1.0 / d2 - 1.0 / d1)
    return y1 * d1, (value - y1) * d2


def odd_anchor(d, x, beta, zero_angle):
    """K = int_0^inf u du / ((u^2+d^2)(u^2+beta^2)) = ln(beta^2/d^2) / (2 (beta^2 - d^2)); x -> 0+ branch at x = 0."""
    d2 = d * d
    angle = jnp.where(x == 0, zero_angle, jnp.angle(d2))
    return (np.log(beta * beta) - (jnp.log(jnp.abs(d2)) + 1j * angle)) / (2.0 * (beta * beta - d2))


def anchor_selector(x, f, eta, beta):
    """Run211 two-pole-anchor selector [k, E, l] (x = 0 follows the right branch, sign = +1)."""
    fb, dv, dc = kernels(x, f, eta)
    sign = jnp.where(x >= 0.0, 1.0, -1.0)
    return (fb * sign / (2.0 * beta * (beta + sign * dv))
            + (1.0 - fb) * sign / (2.0 * beta * (beta + sign * dc)))


def anchor_part(value0, deriv0, x, f, eta, beta, apply_fn=apply):
    """Analytic anchor integral [E, k, m, n] (``own_apply``: [k, m]) and the anchor coefficients (cp, cm)."""
    fb, dv, dc = kernels(x, f, eta)
    cp = anchor_fit(value0, deriv0, eta, beta)
    cm = anchor_fit(dagger(value0), dagger(deriv0), eta, beta)
    total = 0
    for p, m, b in zip(cp, cm, beta):
        odd = (fb * odd_anchor(dv, x, b, -np.pi) + (1.0 - fb) * odd_anchor(dc, x, b, np.pi)) * (1j / (2.0 * np.pi))
        total = total + apply_fn(0.5 * (p + m), anchor_selector(x, f, eta, b))
        total = total + apply_fn(p - m, odd)
    return total, cp, cm


def anchor_subtraction(cp, cm, sums, apply_fn=apply):
    """The imaginary node sums' anchor part: sum_beta cp S_- + cm S_+  -> [E, k, m, n]."""
    out = 0
    for j in range(len(cp)):
        out = out + apply_fn(cp[j], sums[j][0]) + apply_fn(cm[j], sums[j][1])
    return out


# --------------------------------------------------------------------------- real-axis residues
def hermite_coefficients(x, f, h):
    """Run211 uniform-|x| cubic Hermite selectors: (index[k,E,l] int32, vl, vr, dl, dr).

    index = floor(|x|/h); t = |x|/h - index; residue sign f for x < 0, -(1 - f) for x >= 0
    (x = 0 is the right branch).  ``count`` clipping is the caller's coverage gate.
    """
    fb = jnp.broadcast_to(f[:, None, :], x.shape)
    query = jnp.abs(x)
    index = jnp.floor(query / h)
    t = jnp.clip(query / h - index, 0.0, 1.0)
    rs = jnp.where(x < 0.0, fb, -(1.0 - fb))
    h00, h10 = 2 * t ** 3 - 3 * t ** 2 + 1, t ** 3 - 2 * t ** 2 + t
    h01, h11 = -2 * t ** 3 + 3 * t ** 2, t ** 3 - t ** 2
    return index.astype(jnp.int32), rs * h00, rs * h01, rs * h * h10, rs * h * h11


def real_knot_terms(node, z, a, d, x, f, h, negative):
    """One real knot's Hermite residues on ONE branch as (operators [k, m, n, 2l], weights [k, E, 2l]).

    ``negative=False``: residues x >= 0 (unoccupied l), read a, d = contractions of -W_c,q(z),
    -dW_c,q/ds; ``negative=True``: residues x < 0 (occupied l), read the W_{-q}^T twins.  The
    derivative node datum is d/ds * 2z = d/dx.  ``node`` is the knot index of z = x_node + i eta.
    """
    index, vl, vr, dl, dr = hermite_coefficients(x, f, h)
    sel_v = jnp.where(index == node, vl, 0.0) + jnp.where(index + 1 == node, vr, 0.0)
    sel_d = jnp.where(index == node, dl, 0.0) + jnp.where(index + 1 == node, dr, 0.0)
    keep = (x < 0.0) if negative else (x >= 0.0)
    return (jnp.concatenate([a, d * (2 * z)], axis=-1),
            jnp.concatenate([jnp.where(keep, sel_v, 0.0), jnp.where(keep, sel_d, 0.0)], axis=-1))


def real_knot(node, z, a, d, at, dt, x, f, h, apply_fn=apply):
    """One real knot's Hermite residue contribution [E, k, m, n]: both branches, one apply (K = 4l)."""
    op_p, w_p = real_knot_terms(node, z, a, d, x, f, h, negative=False)
    op_n, w_n = real_knot_terms(node, z, at, dt, x, f, h, negative=True)
    return apply_fn(jnp.concatenate([op_p, op_n], axis=-1), jnp.concatenate([w_p, w_n], axis=-1))
