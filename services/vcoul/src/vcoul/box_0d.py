"""0D cell-box truncation: Wigner-Seitz real-space FFT, no q→0 divergence.

The body kernel and head share the same FFT — at q=0, ``vc0`` is just
the G=0 entry of the truncated v(G), which is finite, so no mini-BZ
sampling is needed.  See BerkeleyGW Common/trunc_cell_box.f90.

The numerical FFT routine :func:`~vcoul.box_fft.compute_vcoul_box` lives
in :mod:`vcoul.box_fft`.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from vcoul.base import SysDim, v_qG_single
from vcoul.box_fft import compute_vcoul_box
from vcoul.geometry import CoulombGeometry

__all__ = ["Box0D"]


class Box0D:
    sys_dim = SysDim.BOX_0D

    def _v_bare_per_q(self, qf, gvec_q, *, bvec_f, fact,
                      bdot=None, fft_grid=None):
        """Wigner-Seitz cell-box truncation.  See the base Protocol.

        ``compute_vcoul_box`` builds ``v(G)`` from a real-space FFT of the
        WS-truncated kernel and takes **no q** — it is a q=0 routine (see
        its docstring, "for q=0", and BGW ``Common/trunc_cell_box.f90``,
        which a 0-D deck only ever calls at Γ because a box has one
        k-point).  A q≠0 request is therefore REFUSED rather than served
        the q=0 answer: silently returning ``v(G)`` for ``v(q+G)`` would
        be wrong by ``O(|q|)`` with no symptom.

        G=0 is NOT zeroed here, unlike 3D/2D.  Box truncation makes
        ``v(G=0)`` finite, and :meth:`q0_average` returns exactly that
        value as ``vc0``; this preserves the shipped ``v_qG`` behaviour.
        Whether the G=0 slot should additionally be zeroed to avoid
        double-counting against a rank-1 head injection is UNSETTLED and
        untestable while the 0-D V_q path is unreachable — no per-q sphere
        is built for sys_dim=0 at all (``gw/isdf_fitting.py:670``) and
        ``compute_all_V_q_g_flat`` refuses sys_dim not in (2,3)
        (``gw/v_q_g_flat.py:527``).  Deliberately not decided here.
        """
        if bdot is None or fft_grid is None:
            raise ValueError(
                "Box0D._v_bare_per_q needs bdot and fft_grid (the cell-box "
                "truncation is a real-space FFT, not a closed form in bvec).")
        q = np.asarray(qf, dtype=np.float64)
        if float(np.dot(q, q)) > 1e-24:
            raise NotImplementedError(
                f"Box0D: v(q+G) requested at q={q.tolist()} != 0.  "
                f"compute_vcoul_box is a q=0 routine (real-space WS FFT of "
                f"v(G), no q argument); serving it at finite q would return "
                f"v(G) where v(q+G) was asked for.  A 0-D box deck is "
                f"Gamma-only, so this should not arise — if it does, the "
                f"k-grid is wrong, not this kernel.")
        gvecs = np.asarray(np.rint(gvec_q.T), dtype=int)       # (nG, 3)
        v_raw = compute_vcoul_box(
            np.asarray(bdot, dtype=np.float64),
            np.asarray(fft_grid, dtype=int),
            gvecs)
        # Volume convention matches 3D/2D: v(q+G) / Omega_cell.
        v = v_raw * fact
        # ``denom`` is |q+G|^2 in Ry from bvec, the SAME operand the shared
        # vcoul_cutoff_ry mask compares against in 3D/2D — the cutoff means
        # the same thing in every dimension even though v does not.
        qG_cart = bvec_f.T @ (qf[:, None] + gvec_q)            # (3, nG)
        denom = np.sum(qG_cart * qG_cart, axis=0)              # (nG,)
        return v, denom

    def v_qG(self, geometry: CoulombGeometry, qvec_wrapped,
             comps_qG) -> jax.Array:
        return v_qG_single(self, geometry, qvec_wrapped, comps_qG)

    def q0_average(
        self, geometry: CoulombGeometry, kgrid=None, *,
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

        ``kgrid`` is accepted and ignored so the three kernels share one
        call shape; a box has one k-point by construction.
        """
        del S_cart, epshead, nsamples, method, qmc_reps, kgrid
        if geometry.bdot is None or geometry.fft_grid is None:
            raise ValueError(
                "Box0D.q0_average needs a CoulombGeometry carrying bdot and "
                "fft_grid (the cell-box head is a real-space WS FFT, not a "
                "closed form in bvec).")
        bdot = np.asarray(geometry.bdot, dtype=np.float64)
        fft_grid = np.asarray(geometry.fft_grid, dtype=int)
        g0 = np.array([[0, 0, 0]], dtype=int)
        vc0_raw = compute_vcoul_box(bdot, fft_grid, g0)[0]
        vc0_mean = jnp.asarray(vc0_raw / float(geometry.cell_volume),
                               dtype=jnp.complex128)
        return vc0_mean, vc0_mean
