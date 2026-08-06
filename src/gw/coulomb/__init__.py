"""Coulomb-kernel package: dimension-aware V(q+G) and q→0 head averaging.

Three implementations under one interface — see :mod:`.base`:

    SysDim.BULK_3D  → Bulk3D     (8π/|q+G|², default)
    SysDim.SLAB_2D  → Slab2D     (Ismail-Beigi truncation along c)
    SysDim.BOX_0D   → Box0D      (Wigner-Seitz FFT, no q→0 divergence)

Driver pattern::

    from gw.coulomb import get_kernel
    kernel = get_kernel(meta.sys_dim)        # raises on invalid
    v_qG = kernel.v_qG(wfn, qvec_wrapped, comps_qG)
    vc0_mean, wcoul0 = kernel.q0_average(wfn, meta, S_cart=S_omega)

The dimension-independent ``D_munu`` projector — when present — runs
*outside* this package on the kernel output; ``v_qG`` itself never
depends on it.

WHO CALLS WHAT (2026-08-06 consolidation).  Production ``v(q+G)`` enters
through :func:`gw.compute_vcoul.compute_v_q_per_G`, which is now a thin
dispatcher over :func:`.base.v_qG_table`.  That driver owns everything
dimension-INDEPENDENT — the per-q loop, the ``vcoul_cutoff_ry`` mask, the
G=0 head-slot injection, the ``(n_q, ngkmax)`` float64 contract — and each
kernel contributes exactly one arithmetic method, ``_v_bare_per_q``.
``v_qG`` is the single-q complex128 wrapper over the same driver.  So the
formula for a given dimensionality exists once in the tree, and the
capabilities that used to live only in ``compute_vcoul`` are available to
every caller of the package.

``Slab2D.v_head_minibz_avg`` is deliberately NOT part of that driver: it
is a finite-shift mini-BZ CELL AVERAGE (§16.4's missing 2D body average),
not a point evaluation of ``v``.  It is built and unwired — a capability
awaiting a caller, not a duplicate.
"""
from .base import CoulombKernel, SysDim, get_kernel
from .bulk_3d import Bulk3D
from .slab_2d import Slab2D
from .box_0d import Box0D

__all__ = [
    "CoulombKernel", "SysDim", "get_kernel",
    "Bulk3D", "Slab2D", "Box0D",
]
