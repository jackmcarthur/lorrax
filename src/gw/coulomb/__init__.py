"""COMPAT SHIM package — dimension-aware V(q+G) and q→0 head averaging.

The physics moved to the ``vcoul`` service (2026-08-07,
``services/vcoul/``).  What is left under ``gw/coulomb/`` is the
``(wfn, meta)``-facing translation layer every call site in this tree
still writes:

    SysDim.BULK_3D  → Bulk3D     (8π/|q+G|², default)
    SysDim.SLAB_2D  → Slab2D     (Ismail-Beigi truncation along c)
    SysDim.BOX_0D   → Box0D      (Wigner-Seitz FFT, no q→0 divergence)

Driver pattern (unchanged by the extraction)::

    from gw.coulomb import get_kernel
    kernel = get_kernel(meta.sys_dim)        # raises on invalid
    v_qG = kernel.v_qG(wfn, qvec_wrapped, comps_qG)
    vc0_mean, wcoul0 = kernel.q0_average(wfn, meta, S_cart=S_omega)

The three classes here SUBCLASS ``vcoul.{Bulk3D,Slab2D,Box0D}`` and
override only the two methods whose signatures mention a loader or a
deck: ``v_qG`` and ``q0_average``.  The arithmetic method each kernel
contributes, ``_v_bare_per_q``, is inherited — so the formula for a given
dimensionality still exists exactly once, and now it exists once in a
package that can be imported without lorrax.

The dimension-independent ``D_munu`` projector — when present — runs
*outside* this package on the kernel output; ``v_qG`` itself never
depends on it.

DELETING THIS PACKAGE IS THE REPLUMB'S GATE (step 3), not the
extraction's: the shims stay until every consumer says ``import vcoul``.
"""
from .base import CoulombKernel, SysDim, get_kernel
from .bulk_3d import Bulk3D
from .slab_2d import Slab2D
from .box_0d import Box0D

__all__ = [
    "CoulombKernel", "SysDim", "get_kernel",
    "Bulk3D", "Slab2D", "Box0D",
]
