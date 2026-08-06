"""0D cell-box truncation: Wigner-Seitz real-space FFT, no q→0 divergence.

The body kernel and head share the same FFT — at q=0, ``vc0`` is just
the G=0 entry of the truncated v(G), which is finite, so no mini-BZ
sampling is needed.  See BerkeleyGW Common/trunc_cell_box.f90.

The numerical FFT routine ``compute_vcoul_box`` lives in
``gw/compute_vcoul_0d.py`` (the existing file, unchanged).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from common import Meta
from ..compute_vcoul_0d import compute_vcoul_box
from .base import SysDim


class Box0D:
    sys_dim = SysDim.BOX_0D
    q0_units = "bare"

    def q0_average(
        self, wfn, meta: Meta, *,
        S_cart=None,        # ignored (box truncation: head is finite already)
        epshead=None,       # ignored (no screening correction at the head)
        nsamples=None,
        method=None,
        qmc_reps=None,
        analytic_sphere=False,  # ignored (box head is finite; no mini-BZ avg)
    ):
        """Box: V(q=0, G=0) is finite from the WS-truncated FFT.

        BGW convention: ``wcoul0 = vc0`` for box truncation.  Screening
        enters only through the body of the dielectric matrix; the head
        is left untouched.  See BGW Common/vcoul_generator.f90:717.

        BARE units, like every other kernel (``q0_units``).  Until
        2026-08-05 this one alone divided by ``cell_volume`` while
        :class:`~gw.coulomb.bulk_3d.Bulk3D` and
        :class:`~gw.coulomb.slab_2d.Slab2D` returned bare — a factor
        Omega_cell that the shared consumer
        (:func:`gw.head_correction.resolve_head`) had no way to see.
        """
        del S_cart, epshead, nsamples, method, qmc_reps
        bdot = np.asarray(wfn.bdot, dtype=np.float64)
        fft_grid = np.asarray(wfn.fft_grid, dtype=int)
        g0 = np.array([[0, 0, 0]], dtype=int)
        vc0_raw = compute_vcoul_box(bdot, fft_grid, g0)[0]
        vc0_mean = jnp.asarray(vc0_raw, dtype=jnp.complex128)
        return vc0_mean, vc0_mean
