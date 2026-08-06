"""Coulomb-kernel package: dimension-aware V(q+G) and q→0 head averaging.

Three implementations under one interface — see :mod:`.base`:

    SysDim.BULK_3D  → Bulk3D     (8π/|q+G|², default)
    SysDim.SLAB_2D  → Slab2D     (Ismail-Beigi truncation along c)
    SysDim.BOX_0D   → Box0D      (Wigner-Seitz FFT, no q→0 divergence)

Driver pattern::

    from gw.coulomb import get_kernel
    kernel = get_kernel(meta.sys_dim)        # raises on invalid
    vc0_mean, wcoul0 = kernel.q0_average(wfn, meta, S_cart=S_omega)

``v(q+G)`` on a per-q sphere is not this package's job — see
:func:`gw.compute_vcoul.compute_v_q_per_G` (GW) and
:func:`bse.vq_interp.v_slab_on_set` (BSE).
"""
from .base import CoulombKernel, SysDim, get_kernel
from .bulk_3d import Bulk3D
from .slab_2d import Slab2D
from .box_0d import Box0D

__all__ = [
    "CoulombKernel", "SysDim", "get_kernel",
    "Bulk3D", "Slab2D", "Box0D",
]
