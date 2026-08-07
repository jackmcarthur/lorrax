"""Coulomb-kernel dispatcher: SysDim enum, abstract base, the v(q+G) driver.

A caller wants two things from "the Coulomb interaction" given a system
dimensionality:

1. ``v_qG(geometry, qvec, comps_qG)`` — V(q+G) on the per-q sphere, with
   q+G=0 zeroed (the head is added back as a separate rank-1 term).
2. ``q0_average(geometry, kgrid, ...)`` — the (vc0_mean, wcoul0) pair at
   q→0, typically by Monte-Carlo over the mini-BZ Voronoi cell.

Each dimension lives in its own module (:mod:`~vcoul.bulk_3d`,
:mod:`~vcoul.slab_2d`, :mod:`~vcoul.box_0d`); :func:`get_kernel` picks one
off an int or a :class:`SysDim`.  The shared mini-BZ sampler lives in
:mod:`~vcoul.minibz` so 2D and 3D don't duplicate it.

WHAT CHANGED AT THE SERVICE BOUNDARY.  These signatures used to take a
``wfn`` loader object and a lorrax ``common.Meta``, and reached into them
for ``blat``, ``bvec``, ``cell_volume``, ``bdot``, ``fft_grid``,
``nkx/nky/nkz`` and ``sys_dim``.  They take a
:class:`~vcoul.geometry.CoulombGeometry` and an explicit ``kgrid``
instead — the same numbers, named once, with ``blat * bvec`` taken in one
place rather than five.  The lorrax-side wrappers
(``gw.compute_vcoul``, ``gw.coulomb.*``, ``gw.vcoul``) do the
translation and keep their old wfn/meta-facing spellings.
"""
from __future__ import annotations

import enum
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from vcoul.geometry import CoulombGeometry

__all__ = ["SysDim", "CoulombKernel", "get_kernel", "v_qG_table",
           "v_qG_single"]


class SysDim(int, enum.Enum):
    """System dimensionality for Coulomb truncation.

    Stored as int so legacy ``cohsex.in`` values (0/2/3) keep parsing
    cleanly through ``int(...)``; comparisons against literal ints
    still work (``sys_dim == 3``).
    """
    BULK_3D = 3
    SLAB_2D = 2
    BOX_0D = 0


class CoulombKernel(Protocol):
    """One implementation per dimensionality.

    All kernels return arrays with the same q+G=0-zeroed convention
    (head is injected separately via ``gw.head_correction.HeadResolver``).
    Volume-factor convention: outputs are in Rydberg with the
    BerkeleyGW factor ``v_q(G) · (1/Ω_cell)`` already applied so the
    downstream ``ζ Vc(G) ζ†`` contraction comes out in Ry directly.

    A kernel implements exactly ONE arithmetic method,
    :meth:`_v_bare_per_q` — the dimension's bare formula at one q.
    Everything dimension-independent (the q loop, the ``vcoul_cutoff_ry``
    mask, the G=0 head-slot injection, dtype and shape validation) lives
    once in :func:`v_qG_table` and is shared by all three.  ``v_qG`` and
    the production entry point ``gw.compute_vcoul.compute_v_q_per_G``
    are both thin wrappers over that one driver, so there is a single
    implementation of ``v(q+G)`` per dimensionality.
    """
    sys_dim: SysDim

    def _v_bare_per_q(self, qf, gvec_q, *, bvec_f, fact,
                      bdot=None, fft_grid=None):
        """Bare ``v(q+G)·fact`` and ``|q+G|²`` at ONE q.

        ``qf`` — (3,) fractional q.  ``gvec_q`` — (3, nG) float64 Miller
        indices.  Returns ``(v, denom)``, both ``(nG,)`` float64:

        * ``v``     — ``v(q+G) / Ω_cell``, already zeroed where
          ``|q+G|² < 1e-12`` (the q+G=0 slot; the head is a separate
          rank-1 term).
        * ``denom`` — ``|q+G|²`` in Ry, the operand the shared
          ``vcoul_cutoff_ry`` mask compares against.

        This stays a RAW-array method rather than a geometry-taking one:
        it is called once per q inside :func:`v_qG_table`'s loop, and the
        four things it needs are already unpacked there.
        """
        ...

    def v_qG(self, geometry: CoulombGeometry, qvec_wrapped,
             comps_qG) -> jax.Array:
        """V_q(G) on the per-q G-vector list, length nG, with q+G=0 zeroed."""
        ...

    def q0_average(
        self, geometry: CoulombGeometry, kgrid, *,
        S_cart: jnp.ndarray | None = None,
        epshead: jnp.ndarray | None = None,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
        analytic_sphere: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Return ``(vc0_mean, wcoul0)`` averaged over the mini-BZ Voronoi cell.

        - ``S_cart`` (preferred):  ``wcoul0 = ⟨v(q) / (1 - v(q) qᵀSq)⟩``,
          using the same sample points as ``vc0_mean`` (anisotropic).
        - ``epshead`` (fallback): historical Ismail-Beigi gamma model.
          Less accurate; kept for back-compat with older runs.

        ``Box0D`` ignores everything except ``geometry`` (and the
        ``epshead`` slot, unused) — V(q=G=0) is finite from the cell-box
        FFT.
        """
        ...


def get_kernel(sys_dim) -> CoulombKernel:
    """Return the :class:`CoulombKernel` for the given dimensionality.

    Accepts either :class:`SysDim` or a raw int (0/2/3).  Default is 3D
    when ``sys_dim`` is None or unset; any other value is an error.
    """
    if sys_dim is None:
        sys_dim = SysDim.BULK_3D
    try:
        sd = SysDim(int(sys_dim))
    except ValueError:
        raise ValueError(
            f"sys_dim={sys_dim!r} invalid; expected 0 (box), 2 (slab), "
            f"or 3 (bulk)."
        )
    if sd is SysDim.BULK_3D:
        from vcoul.bulk_3d import Bulk3D
        return Bulk3D()
    if sd is SysDim.SLAB_2D:
        from vcoul.slab_2d import Slab2D
        return Slab2D()
    if sd is SysDim.BOX_0D:
        from vcoul.box_0d import Box0D
        return Box0D()
    raise AssertionError(f"unhandled SysDim: {sd}")  # pragma: no cover


# ---------------------------------------------------------------------------
# The single v(q+G) driver.  Everything dimension-INDEPENDENT lives here
# exactly once; each kernel contributes only ``_v_bare_per_q``.
# ---------------------------------------------------------------------------

def v_qG_table(
    kernel,
    q_irr_frac,
    gvec_components,
    *,
    geometry: CoulombGeometry,
    vcoul_cutoff_ry: float | None = None,
    v_head_miniBZ=None,
) -> np.ndarray:
    """``v(q+G)`` at the writer's per-q WFN.h5-style G-sphere, for every q.

    Returns ``(n_q, ngkmax)`` float64.  ``gvec_components`` is
    ``(n_q, 3, ngkmax)`` Miller indices (``isdf_header/gvec_components``).

    This is the ONE place the three dimension-independent capabilities
    are implemented, in this order (the order is load-bearing — the head
    slot is injected BEFORE the cutoff mask, so a head slot outside the
    bare-Coulomb cutoff is zeroed like any other G):

    1. **``v_head_miniBZ``** — G=0 head-slot injection.  When given, the
       ``(nkx, nky, nkz)`` table replaces ``v`` at the Miller-``(0,0,0)``
       slot of each q with that q's mini-BZ-averaged head.  The slot is
       selected by ``all(G == 0)``, not by ``argmin |q+G|`` — see the
       note below.  q=0 keeps ``v=0`` by construction of the table (its
       head is the separate rank-1 Σ_X term).
    2. **``vcoul_cutoff_ry``** — zero ``v`` wherever ``|q+G|² >`` cutoff.
       This is V_q's bare-Coulomb cutoff, which may be *smaller* than the
       ζ-sphere cutoff that built ``gvec_components``.
    3. dtype/shape: float64 ``(n_q, ngkmax)``.  Pad slots (sentinel
       Miller ``(nx/2, ny/2, nz/2)``) get whatever ``v`` is at that
       position — callers need not zero them because the V_q contract
       carries ζ̃ = 0 there.

    HEAD SLOT.  ``all(G == 0)`` and ``argmin |q+G|²`` disagree on 12 of 64
    q for Si and 1 of 4 for MoS2: for a q whose smallest ``|q+G|`` is at a
    nonzero umklapp G*, the Miller-(0,0,0) rule injects the head at
    G=(0,0,0) while an argmin rule would inject it at G*.  Which is
    correct is a physics question about what
    :func:`~vcoul.minibz.build_v_head_miniBZ_avg_3d` averages, and it is
    deliberately NOT settled here — this preserves the shipped
    Miller-(0,0,0) behaviour exactly.
    """
    q_irr_frac = np.asarray(q_irr_frac, dtype=np.float64).reshape(-1, 3)
    gvec = np.asarray(gvec_components, dtype=np.float64)   # (n_q, 3, ngkmax)
    if gvec.ndim != 3 or gvec.shape[1] != 3:
        raise ValueError(
            f"gvec_components must be (n_q, 3, ngkmax); got {gvec.shape}")
    n_q, _, ngkmax = gvec.shape
    bvec_f = np.asarray(geometry.bvec, dtype=np.float64)
    fact = 1.0 / float(geometry.cell_volume)
    bdot = geometry.bdot
    fft_grid = geometry.fft_grid

    head_arr = None
    if v_head_miniBZ is not None:
        head_arr = np.asarray(v_head_miniBZ, dtype=np.float64)
        if head_arr.ndim != 3:
            raise ValueError(
                f"v_head_miniBZ must be (nkx, nky, nkz); got shape "
                f"{head_arr.shape}")
        head_kgrid = np.array(head_arr.shape, dtype=np.float64)

    out = np.zeros((n_q, ngkmax), dtype=np.float64)
    for qi in range(n_q):
        qf = q_irr_frac[qi]
        v, denom = kernel._v_bare_per_q(
            qf, gvec[qi], bvec_f=bvec_f, fact=fact,
            bdot=bdot, fft_grid=fft_grid)
        if head_arr is not None:
            # Per-q grid index: round (q_frac * kgrid) and wrap modulo
            # kgrid — ``v_head_miniBZ`` is indexed by integer (qx, qy, qz).
            qx_i = int(np.round(qf[0] * head_kgrid[0])) % int(head_kgrid[0])
            qy_i = int(np.round(qf[1] * head_kgrid[1])) % int(head_kgrid[1])
            qz_i = int(np.round(qf[2] * head_kgrid[2])) % int(head_kgrid[2])
            g0_mask = np.all(gvec[qi] == 0.0, axis=0)          # (ngkmax,)
            v = np.where(g0_mask, head_arr[qx_i, qy_i, qz_i], v)
        if vcoul_cutoff_ry is not None:
            v = np.where(denom > float(vcoul_cutoff_ry), 0.0, v)
        out[qi] = v
    return out


def v_qG_single(kernel, geometry: CoulombGeometry, qvec_wrapped,
                comps_qG) -> jax.Array:
    """Single-q :meth:`CoulombKernel.v_qG`, as complex128 ``(nG,)``.

    Thin wrapper over :func:`v_qG_table` so the Protocol method and the
    production table share one arithmetic path.  ``comps_qG`` is
    ``(nG, 3)`` here (the historical single-q spelling), transposed to
    the driver's ``(1, 3, nG)``.
    """
    comps = np.asarray(comps_qG, dtype=np.float64)
    v = v_qG_table(
        kernel,
        np.asarray(qvec_wrapped, dtype=np.float64).reshape(1, 3),
        comps.T[None, :, :],
        geometry=geometry,
    )
    return jnp.asarray(v[0], dtype=jnp.complex128)
