"""Coulomb-kernel dispatcher: SysDim enum, abstract base, sampling helper.

The driver wants ONE thing from "the Coulomb interaction" given a system
dimensionality:

* ``q0_average(wfn, meta, ...)`` — the (vc0_mean, wcoul0) pair at q→0,
  typically by Monte-Carlo over the mini-BZ Voronoi cell.

Each dimension lives in its own module (``bulk_3d``, ``slab_2d``,
``box_0d``); :func:`get_kernel` picks one off ``meta.sys_dim``.  The
shared mini-BZ sampler lives here so 2D and 3D don't duplicate it.

``v(q+G)`` on a per-q sphere is NOT here: the live builders are
:func:`gw.compute_vcoul.compute_v_q_per_G` (GW) and
:func:`bse.vq_interp.v_slab_on_set` (BSE).  A per-dimension ``v_qG``
method existed on these classes from their creating commit (d5b5119)
until 2026-08-05 and was never called by anything; it was deleted rather
than left as a second, drifting spelling of the same formula.
"""
from __future__ import annotations

import enum
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from common import Meta

from .kernel import TOL_MC_NAN, v_qG
# minibz_inscribed_sphere_r2 / minibz_cell_average live in .sampler (the ONE
# MC path).  Re-exported here because .base is the historical import site
# for every caller and for tests/test_minibz_average.py; single
# implementation, single import, no second copy.
from .sampler import (_qmc_engine, minibz_cell_average,  # noqa: F401
                      minibz_inscribed_sphere_r2)


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
    """One implementation per dimensionality."""
    sys_dim: SysDim

    def q0_average(
        self, wfn, meta: Meta, *,
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

def minibz_voronoi_batches(
    bvec, kgrid, *,
    nsamples: int = 2**18,
    method: str = "sobol",
    qmc_reps: int = 10,
    nmax: int = 1,
    is_2d: bool = False,
    seed_offset: int = 0,
):
    """Core mini-BZ Voronoi sampler on RAW arrays (no wfn/meta wrapper).

    ``bvec`` — (3,3) Cartesian reciprocal vectors (rows, 1/bohr, incl. blat).
    ``kgrid`` — (nkx, nky, nkz) q-grid.  Returns a list of ``(N, 3)``
    Cartesian mini-BZ offset batches (δq), one per Sobol replicate (or a
    single uniform-fallback batch).  ``nmax`` is the Voronoi-fold replica
    half-width (BGW ``ncell=3``; default 1 preserves the historical q=0
    head).  ``is_2d`` zeros the qz component.  ``seed_offset`` shifts the
    Sobol scramble seed so an independent draw can be requested for a
    seed-stability check.

    Single source for both :func:`sample_minibz_qpoints` (wfn/meta wrapper,
    GW head) and the BSE per-Q head (:func:`bse.vq_interp.minibz_head_vlr`).
    """
    # Reuse the production wrap helper.  Reimplementing it locally ate a
    # ``shifts @ bvec.T`` vs ``shifts @ bvec`` bug in an earlier draft.
    from ..vcoul import wrap_points_to_voronoi
    bvec = jnp.asarray(bvec, dtype=jnp.float64)
    nkx, nky, nkz = (int(s) for s in kgrid)
    randlims = bvec.T @ (
        jnp.diag(1.0 / jnp.asarray((nkx, nky, nkz), dtype=jnp.float64))
        @ jnp.linalg.inv(bvec.T)
    )

    # ``_qmc_engine`` returns None and ANNOUNCES when scipy.stats.qmc is
    # missing.  This used to be a bare ``except Exception``: a demotion from
    # a low-discrepancy estimator to a plain one, in a production average,
    # with no trace in any log.
    _qmc = _qmc_engine() if str(method).lower() == "sobol" else None
    if _qmc is not None:
        import math as _math
        m = max(1, int(_math.floor(_math.log2(max(2, int(nsamples))))))
        batches = []
        for rep in range(max(1, int(qmc_reps))):
            sob = _qmc.Sobol(d=3, scramble=True, seed=rep + int(seed_offset))
            U = sob.random_base2(m)
            Uj = jnp.asarray(np.asarray(U, dtype=np.float64))
            randcart = (bvec.T @ Uj.T).T
            wrapped = wrap_points_to_voronoi(randcart, bvec, nmax=nmax)
            rq = (randlims @ wrapped.T).T
            if is_2d:
                rq = rq.at[:, 2].set(0.0)
            batches.append(rq)
        return batches
    if str(method).lower() == "sobol":
        # announced demotion: compensate the worse convergence with N
        nsamples = max(int(nsamples), 2_500_000)

    # Uniform fallback (also the path on systems without scipy.stats.qmc)
    key = jax.random.PRNGKey(int(seed_offset))
    randvals = jax.random.uniform(key, (nsamples, 3), dtype=jnp.float64)
    randcart = (bvec.T @ randvals.T).T
    wrapped = wrap_points_to_voronoi(randcart, bvec, nmax=nmax)
    rq = (randlims @ wrapped.T).T
    if is_2d:
        rq = rq.at[:, 2].set(0.0)
    return [rq]


def sample_minibz_qpoints(
    wfn, meta: Meta, *,
    nsamples: int = 2**18,
    method: str = "sobol",
    qmc_reps: int = 10,
    nmax: int = 1,
):
    """Yield batches of q-points sampled in the mini-BZ Voronoi cell.

    Returns a list of ``qcart`` arrays (one per Sobol replicate, or a
    single batch in the uniform fallback) in the format that
    :class:`Bulk3D` / :class:`Slab2D` consume in ``q0_average``.

    Slab geometry (``sys_dim == 2``) zeros out the qz component of the
    returned points so callers don't need their own per-dim branch.
    Thin wrapper over :func:`minibz_voronoi_batches`; ``nmax`` defaults to
    1 (the historical q=0 head — bit-identical) and is widened to 3 by the
    mini-BZ-averaging path (BGW ``ncell=3``).
    """
    bvec = jnp.asarray(wfn.blat * wfn.bvec, dtype=jnp.float64)
    return minibz_voronoi_batches(
        bvec, (meta.nkx, meta.nky, meta.nkz),
        nsamples=nsamples, method=method, qmc_reps=qmc_reps,
        nmax=nmax, is_2d=(int(meta.sys_dim) == 2))


#: MC ``kind`` label -> the (sys_dim, channel) cell of the shared kernel.
_KIND_CELL = {
    "bulk_3d":    (3, "full"),
    "bulk_3d_lr": (3, "lr"),
    "bulk_3d_sr": (3, "sr"),
    "slab":       (2, "full"),
    "slab_lr":    (2, "lr"),
    "slab_sr":    (2, "sr"),
}


def _minibz_kernel_bare(shift_cart, dq_cart, *, kind, alpha=None, zc=None):
    """Bare Coulomb kernel ``8π·[trunc]·[gauss]/|shift+δq|²`` (NO 1/celvol),
    evaluated on a batch of mini-BZ offsets ``dq_cart`` (N,3).  Returns
    (v (N,), len2 (N,)) with ``len2 = |shift+δq|²``.

    ``kind`` names a cell of the shared formula
    :func:`gw.coulomb.kernel.v_qG` — the full ``{bulk, slab} x {full, lr,
    sr}`` product, including the ``slab_sr``/``bulk_3d_sr`` ``-expm1``
    channels this local copy did not carry before 2026-08-05.  Bit-exact
    against that copy on all four kinds it did carry (measured, Frontera
    job 7890613; ``tests/test_coulomb_kernel.py`` is the committed gate).

    The guard here is :data:`~gw.coulomb.kernel.TOL_MC_NAN` (1e-24), NOT
    the 1e-12 lattice-slot tolerance: these are Monte-Carlo draws.
    """
    if kind not in _KIND_CELL:
        raise ValueError(f"_minibz_kernel_bare: unknown kind {kind!r}; "
                         f"expected one of {sorted(_KIND_CELL)}")
    sys_dim, channel = _KIND_CELL[kind]
    K = np.asarray(shift_cart, dtype=np.float64)[None, :] + np.asarray(dq_cart)
    len2 = np.sum(K * K, axis=1)
    v = v_qG(K, axis=1, sys_dim=sys_dim, channel=channel, units="bare",
             alpha=alpha, zc=zc, zero_tol=TOL_MC_NAN)
    return v, len2


def minibz_average(
    shift_cart, dq_batches, *,
    kind: str,
    celvol: float,
    n_kpts: int,
    q0sph2: float,
    alpha: float | None = None,
    zc: float | None = None,
    analytic_sphere: bool = False,
    adaptive: bool = True,
    n_coarse: int = 250_000,
) -> float:
    """Mini-BZ Voronoi CELL AVERAGE of a bare Coulomb kernel around a shift.

    Returns ``<v(shift+δq)>_mBZ`` in **bare** units (``8π·[trunc]·[gauss]/|K|²``,
    NO ``1/celvol`` — the caller applies its own volume convention: the GW
    q=0 head keeps bare and divides at injection; the BSE ``eval_vq`` head
    divides by ``celvol`` to match its stored tile).  ``shift_cart`` is the
    Cartesian ``Q + G*``; ``dq_batches`` the mini-BZ offset batches from
    :func:`minibz_voronoi_batches`.  Averaged over the mean of the replicate
    batches (a free error bar).

    Two BGW branches on ``|shift|²`` (``Common/minibzaverage.f90:35-90``):

    * ``|shift|² < TOL`` and ``analytic_sphere`` (3D only) — the true head:
      MC of the kernel over δq **outside** the inscribed sphere (divided by
      the FULL sample count N) **plus** the analytic Baldereschi-Tosatti
      term ``4·√q0sph2·celvol·N_k/π`` (= ∫_sphere 8π/q² d³q / V_mBZ,
      ``minibzaverage.f90:79-81``).  The 3D ``1/q²`` singularity is handled
      analytically inside the sphere, MC only where it is smooth.

    * else — finite ``|shift|`` (or a 2D slab head, whose ``|Q|``-cusp is
      integrable so it needs no sphere split, ``minibzaverage.f90:97-186``):
      **pure adaptive MC**, sample count ``N_Q = clamp(round(n_coarse·4·
      q0sph2/|shift|²), 1, N)`` (``minibzaverage.f90:63-75``); no analytic
      term.  ``adaptive=False`` uses the full N.
    """
    len_shift2 = float(np.dot(np.asarray(shift_cart, dtype=np.float64),
                              np.asarray(shift_cart, dtype=np.float64)))
    head_branch = (len_shift2 < 1e-12) and analytic_sphere
    per_batch = []
    for dq in dq_batches:
        dq = np.asarray(dq, dtype=np.float64)
        n_tot = dq.shape[0]
        v, len2 = _minibz_kernel_bare(shift_cart, dq, kind=kind,
                                      alpha=alpha, zc=zc)
        if head_branch:
            # Baldereschi split: MC OUTSIDE the inscribed sphere (÷ full N),
            # analytic sphere term added once.
            outside = len2 > q0sph2
            mc = float(np.sum(np.where(outside, v, 0.0))) / float(n_tot)
            analytic = 4.0 * np.sqrt(q0sph2) * float(celvol) * float(n_kpts) / np.pi
            per_batch.append(mc + analytic)
        else:
            if adaptive and len_shift2 > 1e-12:
                n_q = int(round(n_coarse * 4.0 * q0sph2 / len_shift2))
                n_q = max(1, min(n_q, n_tot))
            else:
                n_q = n_tot
            per_batch.append(float(np.mean(v[:n_q])))
    return float(np.mean(per_batch))
