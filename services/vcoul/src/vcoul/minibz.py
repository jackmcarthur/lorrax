"""Mini-BZ Voronoi sampling, cell averaging, and the 3D body-head table.

Three things live here, and they are three because BerkeleyGW's
``Common/minibzaverage.f90`` treats them as three:

1. **The draw** — :func:`wrap_points_to_voronoi` and
   :func:`minibz_voronoi_batches`: uniform (or scrambled-Sobol) points on
   the Voronoi cell of the mini-BZ, mapped to Cartesian δq offsets.
2. **The kernel** — :func:`_minibz_kernel_bare`: the bare Coulomb value at
   ``shift + δq``, four truncation flavours.
3. **The average** — :func:`minibz_average`: the two BGW branches
   (Baldereschi-Tosatti analytic sphere at ``|shift|→0``, adaptive MC
   otherwise), and :func:`build_v_head_miniBZ_avg_3d`, the per-q 3D body
   head table the G-flat V_q path injects at the Miller-(0,0,0) slot.

THE ROW CONVENTION IS THE WHOLE BALLGAME.  ``bvec`` rows are the
Cartesian reciprocal vectors, so a fractional draw ``U`` maps to
Cartesian by ``U @ bvec``.  Every site here uses that spelling; see the
comment block in :func:`build_v_head_miniBZ_avg_3d` for what the
transposed spelling costs and why silicon cannot see it.

SCIPY IS QUARANTINED HERE.  It is the service's one optional dependency,
and the only thing it provides is the scrambled-Sobol generator.  See
:func:`minibz_voronoi_batches` for the announce-or-refuse gate.
"""
from __future__ import annotations

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "wrap_points_to_voronoi",
    "minibz_voronoi_batches",
    "sample_minibz_qpoints",
    "minibz_inscribed_sphere_r2",
    "minibz_average",
    "build_v_head_miniBZ_avg_3d",
]

#: nsamples the uniform fallback is raised to when Sobol demotes.  BGW's
#: plain-MC arm needs many more points than a scrambled Sobol sequence for
#: the same head accuracy; this bump was in the code before the extraction
#: and was applied SILENTLY, which is exactly what the announcement below
#: exists to end.
_UNIFORM_FALLBACK_NSAMPLES = 2_500_000

#: Announce ONCE per process, not once per call: ``q0_average`` is called
#: per q-point per SCF iteration, and a per-call warning is a demotion
#: nobody reads.
_SOBOL_DEMOTION_ANNOUNCED = False


def _announce_sobol_demotion(exc: BaseException, nsamples_before: int) -> None:
    """The `auto` demotion, said out loud exactly once."""
    global _SOBOL_DEMOTION_ANNOUNCED
    if _SOBOL_DEMOTION_ANNOUNCED:
        return
    _SOBOL_DEMOTION_ANNOUNCED = True
    warnings.warn(
        f"vcoul.minibz: method='auto' asked for a scrambled-Sobol mini-BZ "
        f"draw and could not have one — scipy.stats.qmc is unavailable or "
        f"failed ({type(exc).__name__}: {exc}).  DEMOTED to the uniform "
        f"jax.random draw, and nsamples was raised from {nsamples_before} to "
        f"{_UNIFORM_FALLBACK_NSAMPLES} to buy back the accuracy the "
        f"low-discrepancy sequence was providing.  The numbers this run "
        f"produces are NOT bit-comparable with a Sobol run.  Install scipy, "
        f"or pass method='sobol' to make this a refusal instead of a "
        f"demotion, or method='uniform' to ask for this path on purpose.",
        RuntimeWarning, stacklevel=3)


@functools.partial(jax.jit, static_argnames=('nmax',))
def wrap_points_to_voronoi(randcart, bvec, nmax: int = 1):
    """Wrap Cartesian points onto the Voronoi cell of the ``bvec`` lattice.

    Helper function to get test q-points for mini-BZ average with correct
    Voronoi cell.  Rewritten to use JAX arrays.

    Wrapped in ``@jax.jit`` (with ``nmax`` static) so all the per-line
    primitives (meshgrid, stack, reshape, matmul, broadcast subtract,
    norm, argmin, gather, subtract) collapse into a single XLA module
    cached on (input shape x nmax).  Without the jit each call site
    emitted ~10 eager-pjit cache misses.

    ``nmax`` is the replica half-width: the candidate shifts are
    ``{-nmax..nmax}^3 @ bvec``.  1 is the historical value (and is what
    the q=0 head is pinned to); 3 is BGW's ``ncell``, which a skewed cell
    needs because the nearest lattice point can be two replicas away.
    """
    randcart_j = jnp.asarray(randcart, dtype=jnp.float64)
    bvec_j = jnp.asarray(bvec, dtype=jnp.float64)

    grid = jnp.arange(-nmax, nmax + 1)
    shifts = jnp.stack(jnp.meshgrid(grid, grid, grid, indexing="ij"),
                       axis=-1).reshape(-1, 3)
    candidate_shifts = shifts @ bvec_j  # (M, 3)

    diff = randcart_j[:, None, :] - candidate_shifts[None, :, :]  # (N, M, 3)
    dists = jnp.linalg.norm(diff, axis=2)  # (N, M)
    best_idx = jnp.argmin(dists, axis=1)  # (N,)
    wrapped = randcart_j - candidate_shifts[best_idx]
    return wrapped


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

    Single source for both :func:`sample_minibz_qpoints` (geometry wrapper,
    GW head) and the BSE per-Q head (``bse.vq_interp.minibz_head_vlr``).

    THE SCIPY GATE (``method``), announce-or-refuse:

    ``"sobol"``
        Scrambled Sobol, ``qmc_reps`` replicates, seed ``rep +
        seed_offset``.  This is the production default and the pinned
        reference path.  If ``scipy.stats.qmc`` cannot be had, this
        **REFUSES** — an explicit request never silently becomes a
        different generator.
    ``"auto"``
        Sobol if it can, otherwise the uniform fallback, ANNOUNCED once
        (including the ``nsamples`` bump, which was silent before the
        extraction).  This is what a caller that does not care should say.
    ``"uniform"``
        The ``jax.random`` fallback on purpose, with NO nsamples bump and
        no announcement.  Any other token means the same thing — that is
        the pre-extraction behaviour (``use_qmc = method.lower() ==
        "sobol"``) and it is preserved verbatim rather than promoted to a
        refusal, because a refusal here would be a behaviour change on a
        path nothing in the tree exercises.
    """
    bvec = jnp.asarray(bvec, dtype=jnp.float64)
    nkx, nky, nkz = (int(s) for s in kgrid)
    randlims = bvec.T @ (
        jnp.diag(1.0 / jnp.asarray((nkx, nky, nkz), dtype=jnp.float64))
        @ jnp.linalg.inv(bvec.T)
    )

    want = str(method).lower()
    if want in ("sobol", "auto"):
        try:
            from scipy.stats import qmc as _qmc
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
        except Exception as exc:                                # noqa: BLE001
            if want == "sobol":
                raise RuntimeError(
                    f"minibz_voronoi_batches: method='sobol' was requested "
                    f"explicitly and the scrambled-Sobol draw is not "
                    f"available ({type(exc).__name__}: {exc}).  This is a "
                    f"REFUSAL, not a fallback: the uniform draw is a "
                    f"different generator with a different sample count, so "
                    f"serving it here would silently change every head "
                    f"number in the run.  FIX: install scipy (the Sobol "
                    f"generator is scipy.stats.qmc), or pass method='auto' "
                    f"to accept an announced demotion, or method='uniform' "
                    f"to ask for the fallback on purpose."
                ) from exc
            _announce_sobol_demotion(exc, int(nsamples))
            nsamples = max(int(nsamples), _UNIFORM_FALLBACK_NSAMPLES)

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
    geometry, kgrid, *,
    nsamples: int = 2**18,
    method: str = "sobol",
    qmc_reps: int = 10,
    nmax: int = 1,
    is_2d: bool = False,
):
    """Yield batches of q-points sampled in the mini-BZ Voronoi cell.

    Returns a list of ``qcart`` arrays (one per Sobol replicate, or a
    single batch in the uniform fallback) in the format that
    :class:`~vcoul.bulk_3d.Bulk3D` / :class:`~vcoul.slab_2d.Slab2D`
    consume in ``q0_average``.

    Slab geometry (``is_2d``) zeros out the qz component of the returned
    points so callers don't need their own per-dim branch.  Thin wrapper
    over :func:`minibz_voronoi_batches`; ``nmax`` defaults to 1 (the
    historical q=0 head — bit-identical) and is widened to 3 by the
    mini-BZ-averaging path (BGW ``ncell=3``).

    ``geometry`` is a :class:`~vcoul.geometry.CoulombGeometry`; before the
    extraction this took ``(wfn, meta)`` and multiplied ``wfn.blat *
    wfn.bvec`` itself, which is the product the geometry object owns now.
    """
    return minibz_voronoi_batches(
        jnp.asarray(geometry.bvec, dtype=jnp.float64), kgrid,
        nsamples=nsamples, method=method, qmc_reps=qmc_reps,
        nmax=nmax, is_2d=is_2d)


def minibz_inscribed_sphere_r2(bvec, kgrid, *, is_2d: bool = False) -> float:
    """q0sph2 — squared radius of the largest sphere inscribed in the mini-BZ.

    The mini-BZ is the Voronoi cell of the q-grid; its reciprocal lattice
    is ``b_i / nk_i``.  The nearest Voronoi face sits at half the shortest
    mini-BZ reciprocal vector, so ``q0sph2 = min_{n≠0} |0.5·Σ_i n_i b_i/nk_i|²``
    (Cartesian).  Mirrors BGW ``vcoul_generator.f90:296-312``; the 2D slab
    variant restricts the search to the in-plane replicas (n_3 = 0,
    ``:318-331``).
    """
    bvec = np.asarray(bvec, dtype=np.float64)
    nk = np.asarray([int(s) for s in kgrid], dtype=np.float64)
    rng3 = range(0, 1) if is_2d else range(-2, 3)
    best = np.inf
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in rng3:
                if i == 0 and j == 0 and k == 0:
                    continue
                fc_frac = 0.5 * np.array([i, j, k], dtype=np.float64) / nk
                fc_cart = fc_frac @ bvec
                best = min(best, float(fc_cart @ fc_cart))
    return best


def _minibz_kernel_bare(shift_cart, dq_cart, *, kind, alpha=None, zc=None):
    """Bare Coulomb kernel ``8π·[trunc]·[gauss]/|shift+δq|²`` (NO 1/celvol),
    evaluated on a batch of mini-BZ offsets ``dq_cart`` (N,3).  Returns
    (v (N,), len2 (N,)) with ``len2 = |shift+δq|²``.

    ``kind``: ``bulk_3d`` (8π/K²), ``bulk_3d_lr`` (·e^{−K²/4α²}),
    ``slab`` (·f2d), ``slab_lr`` (·f2d·e^{−K²/4α²}); f2d = 1 −
    e^{−zc|K∥|}cos(K_z zc) is the Ismail-Beigi slab truncation.
    """
    K = np.asarray(shift_cart, dtype=np.float64)[None, :] + np.asarray(dq_cart)
    len2 = np.sum(K * K, axis=1)
    len2s = np.where(len2 < 1e-24, 1.0, len2)
    v = 8.0 * np.pi / len2s
    if kind in ("slab", "slab_lr"):
        kxy = np.linalg.norm(K[:, :2], axis=1)
        f2d = 1.0 - np.exp(-zc * kxy) * np.cos(K[:, 2] * zc)
        v = v * f2d
    if kind in ("bulk_3d_lr", "slab_lr"):
        v = v * np.exp(-len2 / (4.0 * alpha ** 2))
    v = np.where(len2 < 1e-24, 0.0, v)
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


def build_v_head_miniBZ_avg_3d(
    kgrid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    *,
    nmc: int = 2**18,
    seed: int = 42,
) -> np.ndarray:
    """Mini-BZ-averaged bare Coulomb head ``<v(q+δq, G=0)>_miniBZ`` per q.

    3D bulk only.  Returns ``(nkx, nky, nkz)`` real array of head values
    in Rydberg / cell-volume units.  q=0 returns 0 (the actual head is
    injected separately via a rank-1 correction in the Σ_X path).

    The MC integration draws ``nmc`` (default 2¹⁸) points uniformly on
    the Voronoi cell of the mini-BZ and averages ``8π/|q+δq|²``.  The
    G-flat path (``compute_v_q_per_G``, called from
    ``gw.v_q_g_flat.compute_all_V_q_g_flat``) consumes this table
    as the ``v_head_miniBZ`` argument.

    ``bvec`` is taken RAW here rather than as a
    :class:`~vcoul.geometry.CoulombGeometry` because the one production
    caller has ``(kgrid, bvec, cell_volume)`` in hand and no wfn, and
    because the frozen-reference pins (seed=42, nmc=2**18) are pins on
    THIS signature.
    """
    nkx, nky, nkz = (int(s) for s in kgrid)
    bvec_j = jnp.asarray(bvec, dtype=jnp.float64)
    rng = np.random.RandomState(seed)
    randvals = rng.uniform(0, 1, (nmc, 3))
    # Row convention: ``bvec`` rows are the Cartesian reciprocal vectors, so
    # ``randvals @ bvec`` spans the b1,b2,b3 parallelepiped — a fundamental
    # domain of the reciprocal lattice, which the Voronoi wrap below maps
    # measure-preservingly onto the Voronoi cell.  ``randvals @ bvec.T``
    # (shipped 2026-05-16..2026-08-07) spans the COLUMN parallelepiped,
    # which is not a fundamental domain: the wrapped cloud double-covers
    # part of the cell and misses part, with the same total volume, so no
    # normalisation check can see it.  Si FCC is provably blind to the
    # difference (bvec.T = P·bvec, P cyclic ⇒ pure reseed); on non-cubic
    # cells it is a bias worth ~50 % of the whole mc-average correction.
    # Pinned by tests/test_vcoul_minibz_head_draw.py; matches
    # minibz_voronoi_batches' draw and bse/vq_interp.py's draw.
    randcart = (randvals @ bvec)
    wrapped = np.asarray(wrap_points_to_voronoi(
        jnp.asarray(randcart), bvec_j, nmax=1))
    kgrid_arr = np.array([nkx, nky, nkz], dtype=np.float64)
    randlims = bvec.T @ (np.diag(1.0 / kgrid_arr) @ np.linalg.inv(bvec.T))
    dq_cart = (randlims @ wrapped.T).T  # (nmc, 3) mini-BZ offsets in Cartesian

    v_head_avg = np.zeros((nkx, nky, nkz), dtype=np.float64)
    for qx in range(nkx):
        for qy in range(nky):
            for qz in range(nkz):
                qw = np.array([qx, qy, qz], dtype=np.float64)
                qw = np.where(qw > kgrid_arr / 2, qw - kgrid_arr, qw)
                q_frac = qw / kgrid_arr
                q_cart = q_frac @ bvec
                if np.dot(q_cart, q_cart) < 1e-12:
                    v_head_avg[qx, qy, qz] = 0.0  # q=0 head handled separately
                else:
                    shifted = q_cart[None, :] + dq_cart  # (nmc, 3)
                    denom = np.sum(shifted**2, axis=1)
                    v_head_avg[qx, qy, qz] = np.mean(8.0 * np.pi / denom)
    return v_head_avg * (1.0 / float(cell_volume))
