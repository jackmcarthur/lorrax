"""The long-range Coulomb channels of a periodic cell: where v(q+G) is not analytic in q.

A channel is one plane wave e^{i(q+G)·r} whose kernel value v(q+G) carries
the non-analytic small-q behaviour of the interaction; everything else in
v is a smooth function of q on the coarse grid.

* **Bulk (sys_dim = 3).**  One channel: the Wigner-Seitz slot
  G_ws(q) = argmin_G |q+G|, where v = 8π/|q+G_ws|² has its 1/q² pole.
* **Slab (sys_dim = 2, truncation along z).**  The in-plane WS slot with every
  out-of-plane G_z = n·b₃, |n| ≤ n_z: the Ismail-Beigi kernel
  8π/|K|² (1 − e^{−z_c|K_∥|} cos K_z z_c) is non-analytic in |q_∥| in each of
  them, with slope 8π z_c (−1)ⁿ/G_z² at q → 0, not only at G = 0.

The plane-wave basis is the one channel basis: v is diagonal in it, so the
range split V = V_b + Fᴴ diag(v_L) F leaves a positive remainder V_b, and the
channels transform under the space group as single plane waves.  Consumers:
``gw.lr_channels`` (the dressed form factors at the coarse q) and
``bse.bse_densify`` (the long-range W model at any q).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from vcoul.base import get_kernel

__all__ = ["LRChannels", "lr_channels", "lr_kernel", "ws_shift",
           "inscribed_radius"]


class LRChannels(NamedTuple):
    """The long-range channel set at a list of q.

    ``gvec``  ``(n_q, n_L, 3)`` int32 — channel G in crystal units, in the
              frame of the q it belongs to (q + G is the channel momentum).
              Slab channels are ordered n = −n_z … n_z along b₃.
    ``v``     ``(n_q, n_L)`` float64 — the bare kernel v(q+G), Ry·bohr³
              (no 1/Ω), zero where q + G = 0 (the Γ head, not in V's body).
    """
    gvec: np.ndarray
    v: np.ndarray


def _shift_box(sys_dim: int) -> np.ndarray:
    r = np.arange(-2, 3)
    if sys_dim == 2:
        return np.array([[a, b, 0] for a in r for b in r], dtype=np.int64)
    return np.array([[a, b, c] for a in r for b in r for c in r], dtype=np.int64)


def ws_shift(q_frac, bvec, sys_dim: int) -> np.ndarray:
    """``G_ws(q)``: the integer shift making ``q + G`` the shortest image.

    In-plane only for a slab (the z component of q is zero there).  Ties at
    the zone boundary go to the first shortest shift in a fixed order, so
    the choice is a function of q alone.

    q_frac : (n, 3) crystal; bvec : (3, 3) Cartesian rows, bohr⁻¹.
    Returns (n, 3) int64.
    """
    q = np.atleast_2d(np.asarray(q_frac, dtype=np.float64))
    shifts = _shift_box(int(sys_dim))
    K = (q[:, None, :] + shifts[None]) @ np.asarray(bvec, dtype=np.float64)
    d2 = np.sum(K * K, axis=-1)
    return shifts[np.argmin(d2 + 1e-12 * np.arange(len(shifts)), axis=1)]


def inscribed_radius(bvec, sys_dim: int) -> float:
    """Radius of the largest ball (disk for a slab) inside the first BZ, bohr⁻¹.

    Half the shortest nonzero reciprocal-lattice vector of the periodic
    directions; inside it the WS slot is unique.
    """
    G = _shift_box(int(sys_dim)) @ np.asarray(bvec, dtype=np.float64)
    n = np.linalg.norm(G, axis=1)
    return 0.5 * float(np.min(n[n > 1e-12]))


def lr_kernel(q_frac, gvec, *, bvec, sys_dim: int) -> np.ndarray:
    """v(q+G) at each channel, Ry·bohr³ (bare units, no 1/Ω).

    q_frac : (n_q, 3) crystal; gvec : (n_q, n_L, 3) int.  Returns (n_q, n_L).
    The kernel is the one ``vcoul.get_kernel(sys_dim)`` formula V is built
    with, so a channel value here is the value V carries in that slot.
    """
    kernel = get_kernel(int(sys_dim))
    bvec_f = np.asarray(bvec, dtype=np.float64)
    q = np.atleast_2d(np.asarray(q_frac, dtype=np.float64))
    g = np.asarray(gvec, dtype=np.float64)
    out = np.empty(g.shape[:2], dtype=np.float64)
    for i in range(q.shape[0]):
        out[i] = kernel._v_bare_per_q(q[i], g[i].T, bvec_f=bvec_f, fact=1.0)[0]
    return out


def lr_channels(q_frac, *, bvec, sys_dim: int, n_z: int = 0,
                head_fn=None) -> LRChannels:
    """The long-range channel set and its kernel at each q (module docstring).

    q_frac : (n_q, 3) crystal momenta, any representative.
    n_z : out-of-plane shells for a slab (``bse.vq_interp.lr_fit_degrees``
        decides it); ignored in bulk.
    head_fn : optional ``K (m, 3) Cartesian → (m,)`` bare-unit value of the
        head channel (G_z = 0) at q + G ≠ 0, for a V whose head slot carries
        a cell average rather than the point kernel.
    """
    sys_dim = int(sys_dim)
    if sys_dim not in (2, 3):
        raise ValueError(
            f"vcoul.lr_channels: sys_dim {sys_dim} has no long-range channel "
            "set (bulk 3 or slab 2 only).")
    q = np.atleast_2d(np.asarray(q_frac, dtype=np.float64))
    g0 = ws_shift(q, bvec, sys_dim)
    if sys_dim == 3:
        gvec = g0[:, None, :]
    else:
        n = np.arange(-int(n_z), int(n_z) + 1)
        gvec = g0[:, None, :] + n[None, :, None] * np.array([0, 0, 1])
    gvec = gvec.astype(np.int32)
    v = lr_kernel(q, gvec, bvec=bvec, sys_dim=sys_dim)
    if head_fn is not None:
        c0 = gvec.shape[1] // 2
        K = (q + gvec[:, c0]) @ np.asarray(bvec, dtype=np.float64)
        live = np.sum(K * K, axis=1) > 1e-12
        v[live, c0] = np.asarray(head_fn(K[live]), dtype=np.float64)
    return LRChannels(gvec=gvec, v=v)
