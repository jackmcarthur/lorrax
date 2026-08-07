"""COMPAT SHIM — the Coulomb kernels live in the ``vcoul`` service now.

The arithmetic that used to be here moved to ``services/vcoul`` (2026-08-07):

    SysDim, CoulombKernel, get_kernel, v_qG_table, v_qG_single  → vcoul.base
    minibz_voronoi_batches, sample_minibz_qpoints,
    minibz_inscribed_sphere_r2, _minibz_kernel_bare,
    minibz_average                                              → vcoul.minibz

What is left here is a TRANSLATION LAYER, and that is the whole of its job.
The service takes a :class:`vcoul.CoulombGeometry` and an explicit
``kgrid``; these names take the ``wfn`` loader object and the lorrax
``common.Meta`` they always took, with byte-compatible signatures, and do
the ``blat * bvec`` / ``meta.nk*`` translation on the way in.  Every
number that comes out is bit-identical to the pre-extraction one — that
is the extraction's claim and it is A/B-measured, not asserted.

The docstrings for the physics live with the physics, in the service.
Read ``vcoul.base`` and ``vcoul.minibz``.

WHY THE SHIM STAYS.  Deleting it is the REPLUMB's gate (step 3), not the
extraction's: at this step every consumer still says ``from gw.coulomb
import get_kernel`` and lorrax has to stay green with no call-site edits.
The shim reaches the service through its TOP-LEVEL door only
(``import vcoul``, never ``vcoul.base``), which ``tests/test_layering.py``
rule 6 measures.
"""
from __future__ import annotations

import jax
import numpy as np

from common import Meta
from ffi import _services      # noqa: F401  (path bootstrap; dies with the
                               # owner's workspace fix -- see _services.py)

_services.ensure_on_path()

import vcoul                                               # noqa: E402
from vcoul import (                                        # noqa: E402,F401
    CoulombGeometry,
    CoulombKernel,
    SysDim,
    _minibz_kernel_bare,
    minibz_average,
    minibz_inscribed_sphere_r2,
    minibz_voronoi_batches,
)

__all__ = [
    "SysDim", "CoulombKernel", "get_kernel", "v_qG_table", "v_qG_single",
    "minibz_voronoi_batches", "sample_minibz_qpoints",
    "minibz_inscribed_sphere_r2", "_minibz_kernel_bare", "minibz_average",
]


def get_kernel(sys_dim) -> CoulombKernel:
    """Return the wfn/meta-facing :class:`CoulombKernel` for ``sys_dim``.

    Accepts either :class:`SysDim` or a raw int (0/2/3).  Default is 3D
    when ``sys_dim`` is None or unset; any other value is an error.

    NOT ``vcoul.get_kernel``: this returns the lorrax ADAPTER subclasses
    in ``gw.coulomb.{bulk_3d,slab_2d,box_0d}``, whose ``v_qG`` and
    ``q0_average`` still take ``(wfn, meta)``.  The service's own
    ``get_kernel`` returns the geometry-facing classes.  Same arithmetic,
    two call shapes, and this one is the one the tree still writes.
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
        from .bulk_3d import Bulk3D
        return Bulk3D()
    if sd is SysDim.SLAB_2D:
        from .slab_2d import Slab2D
        return Slab2D()
    if sd is SysDim.BOX_0D:
        from .box_0d import Box0D
        return Box0D()
    raise AssertionError(f"unhandled SysDim: {sd}")  # pragma: no cover


def v_qG_table(
    kernel,
    q_irr_frac,
    gvec_components,
    *,
    bvec,
    cell_volume: float,
    vcoul_cutoff_ry: float | None = None,
    v_head_miniBZ=None,
    bdot=None,
    fft_grid=None,
) -> np.ndarray:
    """``v(q+G)`` at the per-q WFN.h5-style G-sphere — see :func:`vcoul.v_qG_table`.

    Loose-argument spelling of the service driver: ``bvec``,
    ``cell_volume``, ``bdot`` and ``fft_grid`` are exactly the four fields
    of a :class:`vcoul.CoulombGeometry`, which this builds and forwards.
    Head-slot injection happens BEFORE the cutoff mask (the order is
    load-bearing); the head slot is chosen by ``all(G == 0)``, not by
    ``argmin |q+G|``.  Both rules and the reason the choice is not settled
    here are documented on the service function.
    """
    return vcoul.v_qG_table(
        kernel, q_irr_frac, gvec_components,
        geometry=CoulombGeometry(bvec=bvec, cell_volume=cell_volume,
                                 bdot=bdot, fft_grid=fft_grid),
        vcoul_cutoff_ry=vcoul_cutoff_ry,
        v_head_miniBZ=v_head_miniBZ,
    )


def v_qG_single(kernel, wfn, qvec_wrapped, comps_qG) -> jax.Array:
    """Single-q :meth:`CoulombKernel.v_qG`, as complex128 ``(nG,)``.

    ``wfn``-facing spelling of :func:`vcoul.v_qG_single`;
    ``CoulombGeometry.from_wfn`` is the one place ``blat * bvec`` is taken
    now (it used to be written out at five call sites in this package).
    """
    return vcoul.v_qG_single(kernel, CoulombGeometry.from_wfn(wfn),
                             qvec_wrapped, comps_qG)


def sample_minibz_qpoints(
    wfn, meta: Meta, *,
    nsamples: int = 2**18,
    method: str = "sobol",
    qmc_reps: int = 10,
    nmax: int = 1,
):
    """Batches of q-points in the mini-BZ Voronoi cell — ``(wfn, meta)`` form.

    Slab geometry (``meta.sys_dim == 2``) zeros the qz component.  See
    :func:`vcoul.sample_minibz_qpoints` for the rest.
    """
    return vcoul.sample_minibz_qpoints(
        CoulombGeometry.from_wfn(wfn),
        (meta.nkx, meta.nky, meta.nkz),
        nsamples=nsamples, method=method, qmc_reps=qmc_reps,
        nmax=nmax, is_2d=(int(meta.sys_dim) == 2))
