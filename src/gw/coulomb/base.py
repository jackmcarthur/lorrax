"""Coulomb-kernel dispatcher: SysDim enum, abstract base, sampling helper.

The driver wants two things from "the Coulomb interaction" given a system
dimensionality:

1. ``v_qG(wfn, qvec, comps_qG)`` — V(q+G) on the per-q sphere, with
   q+G=0 zeroed (the head is added back as a separate rank-1 term).
2. ``q0_average(wfn, meta, ...)`` — the (vc0_mean, wcoul0) pair at q→0,
   typically by Monte-Carlo over the mini-BZ Voronoi cell.

Each dimension lives in its own module (``bulk_3d``, ``slab_2d``,
``box_0d``); :func:`get_kernel` picks one off ``meta.sys_dim``.  The
shared mini-BZ sampler lives here so 2D and 3D don't duplicate it.
"""
from __future__ import annotations

import enum
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from common import Meta


class SysDim(int, enum.Enum):
    """System dimensionality for Coulomb truncation.

    Stored as int so legacy ``cohsex.in`` values (0/2/3) keep parsing
    cleanly through ``int(...)``; comparisons against literal ints
    still work (``meta.sys_dim == 3``).
    """
    BULK_3D = 3
    SLAB_2D = 2
    BOX_0D = 0


class CoulombKernel(Protocol):
    """One implementation per dimensionality.

    All kernels return arrays with the same q+G=0-zeroed convention
    (head is injected separately via :class:`gw.head_correction.HeadResolver`).
    Volume-factor convention: outputs are in Rydberg with the
    BerkeleyGW factor ``v_q(G) · (1/Ω_cell)`` already applied so the
    downstream ``ζ Vc(G) ζ†`` contraction comes out in Ry directly.
    """
    sys_dim: SysDim

    def v_qG(self, wfn, qvec_wrapped, comps_qG) -> jax.Array:
        """V_q(G) on the per-q G-vector list, length nG, with q+G=0 zeroed."""
        ...

    def q0_average(
        self, wfn, meta: Meta, *,
        S_cart: jnp.ndarray | None = None,
        epshead: jnp.ndarray | None = None,
        nsamples: int = 2**18,
        method: str = "sobol",
        qmc_reps: int = 10,
    ) -> tuple[jax.Array, jax.Array]:
        """Return ``(vc0_mean, wcoul0)`` averaged over the mini-BZ Voronoi cell.

        - ``S_cart`` (preferred):  ``wcoul0 = ⟨v(q) / (1 - v(q) qᵀSq)⟩``,
          using the same sample points as ``vc0_mean`` (anisotropic).
        - ``epshead`` (fallback): historical Ismail-Beigi gamma model.
          Less accurate; kept for back-compat with older runs.

        ``Box_0D`` ignores everything except ``wfn`` and the ``epshead``
        slot (both unused) — V(q=G=0) is finite from the cell-box FFT.
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
        from .bulk_3d import Bulk3D
        return Bulk3D()
    if sd is SysDim.SLAB_2D:
        from .slab_2d import Slab2D
        return Slab2D()
    if sd is SysDim.BOX_0D:
        from .box_0d import Box0D
        return Box0D()
    raise AssertionError(f"unhandled SysDim: {sd}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Mini-BZ Voronoi sampling — shared by 2D and 3D q0_average implementations.
# Box_0D doesn't need this (V(q=G=0) is finite from the FFT directly).
# ---------------------------------------------------------------------------

def sample_minibz_qpoints(
    wfn, meta: Meta, *,
    nsamples: int = 2**18,
    method: str = "sobol",
    qmc_reps: int = 10,
):
    """Yield batches of q-points sampled in the mini-BZ Voronoi cell.

    Returns a list of ``qcart`` arrays (one per Sobol replicate, or a
    single batch in the uniform fallback) in the format that
    :class:`Bulk3D` / :class:`Slab2D` consume in ``q0_average``.

    Slab geometry (``sys_dim == 2``) zeros out the qz component of the
    returned points so callers don't need their own per-dim branch.
    """
    # Reuse the production wrap helper.  Reimplementing it locally ate a
    # ``shifts @ bvec.T`` vs ``shifts @ bvec`` bug in an earlier draft.
    from ..vcoul import wrap_points_to_voronoi
    bvec = jnp.asarray(wfn.blat * wfn.bvec, dtype=jnp.float64)
    is_2d = (int(meta.sys_dim) == 2)
    randlims = bvec.T @ (
        jnp.diag(1.0 / jnp.asarray((meta.nkx, meta.nky, meta.nkz)))
        @ jnp.linalg.inv(bvec.T)
    )

    use_qmc = (str(method).lower() == "sobol")
    if use_qmc:
        try:
            from scipy.stats import qmc as _qmc
            import math as _math
            m = max(1, int(_math.floor(_math.log2(max(2, int(nsamples))))))
            batches = []
            for rep in range(max(1, int(qmc_reps))):
                sob = _qmc.Sobol(d=3, scramble=True, seed=rep)
                U = sob.random_base2(m)
                Uj = jnp.asarray(np.asarray(U, dtype=np.float64))
                randcart = (bvec.T @ Uj.T).T
                wrapped = wrap_points_to_voronoi(randcart, bvec, nmax=1)
                rq = (randlims @ wrapped.T).T
                if is_2d:
                    rq = rq.at[:, 2].set(0.0)
                batches.append(rq)
            return batches
        except Exception:
            use_qmc = False
            nsamples = max(int(nsamples), 2_500_000)

    # Uniform fallback (also the path on systems without scipy.stats.qmc)
    key = jax.random.PRNGKey(0)
    randvals = jax.random.uniform(key, (nsamples, 3), dtype=jnp.float64)
    randcart = (bvec.T @ randvals.T).T
    wrapped = wrap_points_to_voronoi(randcart, bvec, nmax=1)
    rq = (randlims @ wrapped.T).T
    if is_2d:
        rq = rq.at[:, 2].set(0.0)
    return [rq]
