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
   otherwise), and :func:`build_v_head_miniBZ_fn_3d`, the 3D body head
   the G-flat V_q path injects at the ``argmin |q+G|`` slots.

THE BODY HEAD IS A FUNCTION OF K, NOT A TABLE INDEXED BY q.  Before
2026-08-08 this module built an ``(nkx, nky, nkz)`` array
(``build_v_head_miniBZ_avg_3d``) and the driver dropped it at the slot
labelled Miller-(0,0,0).  That label is not equivariant under q → −q and
cost V_q 6.0e−3 of reciprocity; the fix injects at ``argmin |q+G|²``
instead, every tied slot valued from its OWN Cartesian K, so a per-q
scalar can no longer carry the answer.  See
:func:`~vcoul.base.v_qG_table`'s HEAD SLOT note for the whole argument
and its measurements.

THE ROW CONVENTION IS THE WHOLE BALLGAME.  ``bvec`` rows are the
Cartesian reciprocal vectors, so a fractional draw ``U`` maps to
Cartesian by ``U @ bvec``.  Since the 2026-08-07 consolidation that map
exists ONCE, as :func:`minibz_frac_to_cart`, and every draw in this
module routes through it (the mini-BZ affine likewise through
:func:`minibz_cell_affine`); see the docstring of
:func:`build_miniBZ_dq_cart` for what the transposed spelling cost
and why silicon cannot see it.

WHAT THE CONSOLIDATION DID AND DID NOT UNIFY.  The 3D body head
(:func:`build_v_head_miniBZ_fn_3d`) and the q→0 head
(:func:`minibz_voronoi_batches` + :func:`minibz_average`) were born as
independent implementations that diverged nine ways (survey
w1_vcoul.md §1.3).  Consolidated here — single-sourced, geometry only:
the frac→Cartesian draw map, the mini-BZ affine, and the bare-3D kernel
(the body head now evaluates through :func:`_minibz_kernel_bare`).
DELIBERATELY KEPT DIVERGENT, because each would move the frozen
``si_bse_debug`` reference beyond the 358bb0b fix and the phase allows
exactly one physics-moving commit (all four are REGISTERED owner
questions riding the same refreeze):

  =============  =========================  ==========================
  axis           body head (this file)      q→0 head (this file)
  =============  =========================  ==========================
  generator      RandomState(seed=42),      scrambled Sobol, seed=rep
                 centrosymmetrised
  estimator      plain mean over all 2·nmc  BGW adaptive N_Q + analytic
                                            sphere branch, qmc_reps=10
  Voronoi fold   nmax=1 (hard)              nmax 1 or 3 (BGW ncell=3)
  volume         × 1/celvol at return       bare; caller divides
  =============  =========================  ==========================

The body head's ``{δq} = {−δq}`` closure (2026-08-08) is NOT one of the
divergences: it is a correctness requirement of the argmin injection —
see :func:`build_miniBZ_dq_cart`.

THE FOURTH COPY OF THE DRAW GEOMETRY IS GONE (verified 2026-08-22).
``bse.vq_interp``'s rank-parallel ``fold_in`` draw used to spell the
frac→Cartesian map, the Voronoi wrap and the mini-BZ affine locally; it
now calls :func:`minibz_frac_to_cart`, :func:`wrap_points_to_voronoi` and
:func:`minibz_cell_affine` through the door (``vq_interp.py`` ~:1738).
Its own comment records the measurement that licensed the swap —
bit-identical to the local spellings over 200 random cells, both helpers
exactly equal in float64 — so the BSE fixtures did not move.  What is
still ITS own is the PRNG (``jax.random.fold_in``, rank-parallel by
construction) and the ``nmax=3`` choice, and both are deliberate.  *(This
paragraph previously said the rewiring was registered rather than done.)*

SCIPY IS QUARANTINED HERE.  It is the service's one optional dependency,
and the only thing it provides is the scrambled-Sobol generator.  See
:func:`minibz_voronoi_batches` for the announce-or-refuse gate.
"""
from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import functools
import hashlib
import itertools
from typing import NamedTuple
import warnings

import jax
import jax.numpy as jnp
import numpy as np

from vcoul.quadrature import gauss_legendre_interval

__all__ = [
    "COULOMB_GAUGE_TT_SIGN",
    "apply_transverse_projector",
    "transverse_projector",
    "wrap_points_to_voronoi",
    "minibz_frac_to_cart",
    "minibz_cell_affine",
    "minibz_voronoi_batches",
    "sample_minibz_qpoints",
    "minibz_inscribed_sphere_r2",
    "minibz_average",
    "minibz_moment_tensor",
    "minibz_transverse_head_avg",
    "MinibzPhotonReceipt",
    "minibz_photon_cubature",
    "slab_minibz_photon_cubature",
    "bulk_minibz_photon_cubature",
    "validate_minibz_photon_receipt",
    "iter_minibz_photon_samples",
    "build_miniBZ_dq_cart",
    "build_v_head_miniBZ_fn_3d",
]


#: Lorentz-metric sign of the spatial Coulomb-gauge photon block in the
#: stored ``(C, Tx, Ty, Tz)`` current basis.  The geometric transverse
#: projector remains ``P_T = I - Khat Khat``; physical bare interactions use
#: ``D_TT = COULOMB_GAUGE_TT_SIGN * v * P_T``.  This is the single sign owner
#: shared by the G-flat TT builder and streamed mini-BZ photon samples.
COULOMB_GAUGE_TT_SIGN = -1.0


_SLAB_MINIBZ_PHOTON_METHOD = (
    "true_ws_polygon_duffy_gauss_legendre_v1")
_SLAB_MINIBZ_PHOTON_ORDERS = (16, 24, 32)
_BULK_MINIBZ_PHOTON_METHOD = (
    "true_ws_polyhedron_duffy_gauss_legendre_v1")
_BULK_MINIBZ_PHOTON_ORDERS = (8, 12, 16)
_MINIBZ_PHOTON_RECEIPT_TOKEN = object()


class _MinibzPhotonChunk(NamedTuple):
    """One weighted, fixed-shape rule issued with a photon receipt."""

    order: int
    q_cart: np.ndarray
    D_raw: np.ndarray
    physical_count: int
    sample_weight: np.ndarray


@dataclass(frozen=True)
class MinibzPhotonReceipt:
    """Provider-issued exact mini-BZ cubature and its geometry facts.

    This is the one deliberately typed service result: the screened Dyson
    consumer must be able to distinguish an exact Wigner--Seitz/Duffy ladder
    from the legacy Sobol iterator without trusting caller-supplied labels.
    The private token prevents accidental construction outside the provider;
    all geometry and rule diagnostics travel with the immutable receipt.
    """

    dimension: int
    method: str
    orders: tuple[int, int, int]
    reciprocal_lattice_rows: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    kgrid: tuple[int, int, int]
    mini_lattice_rows: tuple[tuple[float, ...], ...]
    polytope_vertices: tuple[tuple[float, ...], ...]
    polytope_faces: tuple[tuple[int, ...], ...]
    minibz_measure: float
    slab_zc: float | None
    cell_volume: float
    physical_counts: tuple[int, int, int]
    padded_counts: tuple[int, int, int]
    weight_sum_defects: tuple[float, float, float]
    weighted_q_centroids: tuple[tuple[float, float, float], ...]
    chunks: tuple[_MinibzPhotonChunk, ...] = field(
        repr=False, compare=False)
    _issue_token: InitVar[object] = None
    # ``init=False`` is load-bearing: dataclasses.replace() does not copy the
    # stored issuance token or digest, and its default ``_issue_token=None``
    # is refused by __post_init__.  The ndarray write flags below are only
    # accidental-mutation friction; validation rechecks the stored token, a
    # digest, and every regenerated payload before production consumption.
    _provider_token: object = field(
        init=False, default=None, repr=False, compare=False)
    _provider_digest: str = field(
        init=False, default="", repr=False, compare=False)

    def __post_init__(self, _issue_token) -> None:
        if _issue_token is not _MINIBZ_PHOTON_RECEIPT_TOKEN:
            raise TypeError(
                "MinibzPhotonReceipt is issued only by "
                "minibz_photon_cubature")
        object.__setattr__(self, "_provider_token", _issue_token)


def minibz_frac_to_cart(U, bvec):
    """Fractional draw → Cartesian: ``U @ bvec``, and ONLY that spelling.

    ``bvec`` rows are the Cartesian reciprocal vectors, so the row-space
    product maps the unit cube onto the b1,b2,b3 parallelepiped — a
    fundamental domain of the reciprocal lattice, which
    :func:`wrap_points_to_voronoi` then maps measure-preservingly onto
    the Voronoi cell.  The transposed spelling (``U @ bvec.T``) is NOT a
    fundamental domain of that lattice and shipped as a three-month bias
    in the 3D body head (fixed 358bb0b).  Every draw in this module
    routes through this function so the convention is decidable in
    exactly one place; it is polymorphic over numpy/jax — the caller's
    array module performs the matmul, so a numpy caller's bits do not
    change because a jax caller exists.
    """
    return U @ bvec


def minibz_cell_affine(bvec, kgrid):
    """The mini-BZ affine: full-cell Voronoi points → mini-BZ δq offsets.

    ``bvec.T @ diag(1/kgrid) @ inv(bvec.T)`` — scales fractional
    coordinates by ``1/kgrid`` while staying in Cartesian, so a point
    wrapped to the FULL reciprocal Voronoi cell becomes an offset in the
    mini-BZ Voronoi cell.  numpy float64 (single implementation); jax
    callers ``jnp.asarray`` the (3,3) result.
    """
    bvec = np.asarray(bvec, dtype=np.float64)
    kg = np.asarray([int(s) for s in np.ravel(kgrid)], dtype=np.float64)
    return bvec.T @ (np.diag(1.0 / kg) @ np.linalg.inv(bvec.T))

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


def _iter_minibz_voronoi_batches(
    bvec, kgrid, *,
    nsamples: int = 2**18,
    method: str = "sobol",
    qmc_reps: int = 10,
    nmax: int = 1,
    is_2d: bool = False,
    seed_offset: int = 0,
):
    """Yield mini-BZ Voronoi draws one replicate at a time.

    This is the sole implementation of the draw.  The public list-returning
    :func:`minibz_voronoi_batches` and the streamed photon-kernel provider both
    consume it, so streaming cannot acquire a second seed, geometry, or
    Sobol-demotion policy.

    See :func:`minibz_voronoi_batches` for the public contract.
    """
    bvec_np = np.asarray(bvec, dtype=np.float64)
    bvec = jnp.asarray(bvec, dtype=jnp.float64)
    nkx, nky, nkz = (int(s) for s in kgrid)
    # Consolidated 2026-08-07: the affine is the shared numpy helper (one
    # implementation; was a jnp twin of the body head's numpy expression).
    # Last-ulp move possible (np.linalg.inv vs jnp) — measured, and no
    # frozen gate reaches this path (every pinned deck overrides the head).
    randlims = jnp.asarray(minibz_cell_affine(bvec_np, (nkx, nky, nkz)))

    def _map_unit_draw(unit_draw):
        unit_draw = jnp.asarray(unit_draw, dtype=jnp.float64)
        randcart = minibz_frac_to_cart(unit_draw, bvec)
        wrapped = wrap_points_to_voronoi(randcart, bvec, nmax=nmax)
        rq = (randlims @ wrapped.T).T
        if is_2d:
            rq = rq.at[:, 2].set(0.0)
        return rq

    want = str(method).lower()
    if want in ("sobol", "auto"):
        try:
            from scipy.stats import qmc as _qmc
            import math as _math
            m = max(1, int(_math.floor(_math.log2(max(2, int(nsamples))))))
            for rep in range(max(1, int(qmc_reps))):
                sob = _qmc.Sobol(d=3, scramble=True, seed=rep + int(seed_offset))
                yield _map_unit_draw(sob.random_base2(m))
            return
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

    # Uniform fallback (also the path on systems without scipy.stats.qmc).
    key = jax.random.PRNGKey(int(seed_offset))
    randvals = jax.random.uniform(key, (nsamples, 3), dtype=jnp.float64)
    yield _map_unit_draw(randvals)


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
    return list(_iter_minibz_voronoi_batches(
        bvec, kgrid, nsamples=nsamples, method=method,
        qmc_reps=qmc_reps, nmax=nmax, is_2d=is_2d,
        seed_offset=seed_offset))


def sample_minibz_qpoints(
    geometry, kgrid, *,
    nsamples: int = 2**18,
    method: str = "sobol",
    qmc_reps: int = 10,
    nmax: int = 1,
    is_2d: bool = False,
    stream_reps: bool = False,
):
    """Yield batches of q-points sampled in the mini-BZ Voronoi cell.

    By default, returns the historical list of ``qcart`` arrays (one per
    Sobol replicate, or a single batch in the uniform fallback).  With
    ``stream_reps=True``, returns an iterator over exactly the same arrays,
    so consumers can finish and release one replicate before drawing the
    next instead of stacking ``qmc_reps``.  Both paths consume the same sole
    sampler implementation.

    The arrays are in the format that
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
    args = dict(
        nsamples=nsamples, method=method, qmc_reps=qmc_reps,
        nmax=nmax, is_2d=is_2d)
    bvec = jnp.asarray(geometry.bvec, dtype=jnp.float64)
    if stream_reps:
        return _iter_minibz_voronoi_batches(bvec, kgrid, **args)
    return minibz_voronoi_batches(bvec, kgrid, **args)


def _sample_q0_minibz_qpoints(
    geometry, kgrid, *,
    nsamples: int,
    method: str,
    qmc_reps: int,
    analytic_sphere: bool,
    is_2d: bool,
    stream_reps: bool = False,
):
    """The one q=0 sampler policy shared by CC, TT, and packed C⊕T."""
    return sample_minibz_qpoints(
        geometry, kgrid, nsamples=nsamples, method=method,
        qmc_reps=qmc_reps, nmax=3 if analytic_sphere else 1,
        is_2d=is_2d, stream_reps=stream_reps)


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


def apply_transverse_projector(
    K_cart, vector_cart, len2, *, eps_K2: float = 1e-30,
    component_axis: int = -1,
):
    """Apply ``J - K (K·J)/K²`` without materialising a ``3×3`` tensor.

    ``K_cart`` has shape ``(...,3)``.  ``vector_cart`` may carry additional
    right-hand-side axes after the shared ``...`` prefix; ``component_axis``
    names its Cartesian axis.  At a singular direction the geometric action
    is the identity, leaving the physical consumer to own its singular-slot
    policy (periodic direct current subsequently zeros ``G=0``).
    """
    use_jax = any(isinstance(x, (jax.Array, jax.core.Tracer))
                  for x in (K_cart, vector_cart, len2))
    xp = jnp if use_jax else np
    K = xp.asarray(K_cart, dtype=xp.float64)
    vector = xp.asarray(vector_cart)
    if K.ndim < 1 or K.shape[-1] != 3:
        raise ValueError(f"K_cart must have shape (...,3); got {K.shape}.")
    axis = int(component_axis)
    if axis < 0:
        axis += vector.ndim
    if axis < 0 or axis >= vector.ndim or vector.shape[axis] != 3:
        raise ValueError(
            "vector_cart's Cartesian component axis must have extent 3; "
            f"got shape {vector.shape}, component_axis={component_axis}.")
    moved = xp.moveaxis(vector, axis, -1)
    extra = moved.ndim - K.ndim
    if extra < 0 or moved.shape[:K.ndim - 1] != K.shape[:-1]:
        raise ValueError(
            "vector_cart must share K_cart's leading grid/sample axes; "
            f"got K={K.shape}, vector={vector.shape}.")
    K_broadcast = K.reshape(K.shape[:-1] + (1,) * extra + (3,))
    len2_array = xp.asarray(len2, dtype=xp.float64)
    if len2_array.shape != K.shape[:-1]:
        raise ValueError(
            f"len2 must have shape {K.shape[:-1]}; got {len2_array.shape}.")
    len2_safe = xp.where(len2_array > eps_K2, len2_array, 1.0)
    denominator = len2_safe.reshape(
        len2_array.shape + (1,) * extra + (1,))
    longitudinal = (xp.sum(K_broadcast * moved, axis=-1, keepdims=True)
                    / denominator)
    return xp.moveaxis(moved - K_broadcast * longitudinal, -1, axis)


def transverse_projector(K_cart, len2, *, eps_K2: float = 1e-30):
    """``I - Khat Khat`` derived from the matrix-free public action.

    This is the sole transverse-projector formula shared by finite-q photon
    tiles, mini-BZ sampling, and the periodic direct-current Hartree solve.
    The zero-direction row is the identity here; each physical consumer owns
    its own singular-slot policy (periodic direct sets the entire G=0 field
    to zero, whereas an exchange head may replace that slot explicitly).
    """
    use_jax = isinstance(K_cart, (jax.Array, jax.core.Tracer)) or isinstance(
        len2, (jax.Array, jax.core.Tracer))
    xp = jnp if use_jax else np
    K = xp.asarray(K_cart, dtype=xp.float64)
    identity = xp.broadcast_to(xp.eye(3), K.shape[:-1] + (3, 3))
    return apply_transverse_projector(
        K, identity, len2, eps_K2=eps_K2, component_axis=-2)


def _analytic_sphere_bare_head(q0sph2, celvol, n_kpts) -> float:
    """Baldereschi-Tosatti bare scalar contribution inside the sphere."""
    return (4.0 * np.sqrt(q0sph2) * float(celvol) * float(n_kpts)
            / np.pi)


def _gauss_reduce_2d_lattice(a: np.ndarray, b: np.ndarray):
    """Return a Gauss-reduced basis for one nonsingular 2-D lattice."""
    u = np.asarray(a, dtype=np.float64).copy()
    v = np.asarray(b, dtype=np.float64).copy()
    if u.shape != (2,) or v.shape != (2,):
        raise ValueError("2-D lattice vectors must both have shape (2,)")
    scale = max(float(np.linalg.norm(u)), float(np.linalg.norm(v)))
    if (not np.all(np.isfinite(u)) or not np.all(np.isfinite(v))
            or scale == 0.0
            or abs(_det2(u, v))
            <= 128.0 * np.finfo(np.float64).eps * scale * scale):
        raise ValueError("slab mini-BZ in-plane lattice is singular")
    for _ in range(64):
        if float(v @ v) < float(u @ u):
            u, v = v, u
        multiple = int(np.rint(float(u @ v) / float(u @ u)))
        if multiple == 0:
            return u, v
        v = v - float(multiple) * u
    raise RuntimeError("2-D Gauss lattice reduction did not converge")


def _det2(left: np.ndarray, right: np.ndarray) -> float:
    """Oriented 2-D determinant without NumPy's deprecated 2-vector cross."""
    return float(left[0] * right[1] - left[1] * right[0])


def _polygon_signed_twice_area(polygon: np.ndarray) -> float:
    """One deterministic reduction for issuing and authenticating an area.

    Exact receipt validation compares the issued float bit-for-bit.  Using a
    Python scalar reduction while issuing and ``np.sum`` while validating can
    differ by one rounding bit on an otherwise valid oblique cell (CrI3 is a
    concrete example), so both sites must call this owner.
    """
    return float(sum(
        _det2(left, right)
        for left, right in zip(polygon, np.roll(polygon, -1, axis=0))))


def _clip_polygon_half_plane(
    polygon: np.ndarray, normal: np.ndarray, offset: float, *,
    inside_tol: float, point_tol: float,
) -> np.ndarray:
    """Sutherland--Hodgman clip against ``x.normal <= offset``."""
    if polygon.shape[0] == 0:
        return polygon
    out = []
    previous = polygon[-1]
    previous_value = float(previous @ normal) - float(offset)
    previous_inside = previous_value <= inside_tol
    for current in polygon:
        current_value = float(current @ normal) - float(offset)
        current_inside = current_value <= inside_tol
        if current_inside != previous_inside:
            denominator = previous_value - current_value
            if denominator == 0.0:
                raise RuntimeError("degenerate mini-BZ half-plane crossing")
            fraction = previous_value / denominator
            out.append(previous + fraction * (current - previous))
        if current_inside:
            out.append(current)
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    if not out:
        return np.empty((0, 2), dtype=np.float64)
    clipped = np.asarray(out, dtype=np.float64)
    keep = np.ones(clipped.shape[0], dtype=bool)
    for i in range(clipped.shape[0]):
        if np.linalg.norm(clipped[i] - clipped[i - 1]) <= point_tol:
            keep[i] = False
    return clipped[keep]


def _slab_minibz_wigner_seitz_polygon(bvec, kgrid):
    """Return the true mini-lattice WS polygon, lattice rows, and area.

    ``bvec`` has already passed the canonical :class:`vcoul.Slab2D`
    orientation check.  This geometry helper owns only the two-dimensional
    Voronoi construction and deliberately makes no second slab decision.
    """
    bvec = np.asarray(bvec, dtype=np.float64)
    kg = tuple(int(v) for v in kgrid)
    if bvec.shape != (3, 3):
        raise ValueError(f"bvec must be (3,3); got {bvec.shape}")
    if len(kg) != 3 or any(v <= 0 for v in kg):
        raise ValueError(
            f"kgrid must contain three positive integers; got {kgrid}")
    if kg[2] != 1:
        raise ValueError(
            "slab polygon cubature requires Nkz=1; got "
            f"kgrid={kg}")
    if not np.all(np.isfinite(bvec)):
        raise ValueError("slab polygon cubature requires finite bvec rows")

    # The mini reciprocal lattice itself owns the Voronoi cell.  Reducing a
    # parent-cell WS polygon and scaling it afterward is wrong on an
    # anisotropic k-grid because Voronoi construction and anisotropic scaling
    # do not commute.
    g1 = bvec[0, :2] / float(kg[0])
    g2 = bvec[1, :2] / float(kg[1])
    u, v = _gauss_reduce_2d_lattice(g1, g2)
    w = v - u if float(u @ v) >= 0.0 else v + u

    radius = 2.0 * (float(np.linalg.norm(u)) + float(np.linalg.norm(v)))
    polygon = np.asarray(
        ((-radius, -radius), (radius, -radius),
         (radius, radius), (-radius, radius)), dtype=np.float64)
    scale = max(float(np.linalg.norm(u)), float(np.linalg.norm(v)))
    length_tol = 4096.0 * np.finfo(np.float64).eps * scale
    halfplane_tol = length_tol * scale
    for normal in (u, -u, v, -v, w, -w):
        polygon = _clip_polygon_half_plane(
            polygon, normal, 0.5 * float(normal @ normal),
            inside_tol=halfplane_tol, point_tol=length_tol)
        if polygon.shape[0] < 3:
            raise RuntimeError("mini-BZ half-plane intersection became empty")

    if polygon.shape[0] not in (4, 6):
        raise RuntimeError(
            "a nonsingular 2-D lattice Voronoi cell must have four or six "
            f"vertices after clipping +/-u,+/-v,+/-w; got {polygon.shape[0]}")
    signed_twice_area = _polygon_signed_twice_area(polygon)
    if signed_twice_area < 0.0:
        polygon = polygon[::-1].copy()
        signed_twice_area = -signed_twice_area
    if signed_twice_area <= 0.0:
        raise RuntimeError("slab mini-BZ polygon has nonpositive area")

    edge_lengths = np.linalg.norm(
        np.roll(polygon, -1, axis=0) - polygon, axis=1)
    origin_edge_margins = np.asarray([
        _det2(right - left, -left)
        for left, right in zip(polygon, np.roll(polygon, -1, axis=0))])
    if (np.any(edge_lengths <= length_tol)
            or np.any(origin_edge_margins <= halfplane_tol)):
        raise RuntimeError(
            "slab mini-BZ polygon is not strictly CCW with positive edges "
            "and Gamma in its interior")

    central_error = max(
        min(float(np.linalg.norm(point + partner)) for partner in polygon)
        for point in polygon)
    if central_error > length_tol:
        raise RuntimeError(
            "slab mini-BZ polygon failed central symmetry: "
            f"max vertex-pair defect={central_error:.3e}")

    polygon_area = 0.5 * signed_twice_area
    lattice_area = abs(_det2(g1, g2))
    area_relative_error = abs(polygon_area - lattice_area) / lattice_area
    if area_relative_error > 2.0e-12:
        raise RuntimeError(
            "slab mini-BZ polygon failed its Voronoi cell-area identity: "
            f"polygon={polygon_area:.17e}, lattice={lattice_area:.17e}, "
            f"relative_error={area_relative_error:.3e}")
    return polygon, np.asarray((g1, g2), dtype=np.float64), polygon_area


def _dedupe_polyhedron_polygon(points, point_tol):
    """Remove adjacent duplicate vertices from one clipped 3-D face."""
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    polygon = np.asarray(points, dtype=np.float64)
    keep = np.ones(polygon.shape[0], dtype=bool)
    for index in range(polygon.shape[0]):
        if np.linalg.norm(polygon[index] - polygon[index - 1]) <= point_tol:
            keep[index] = False
    return polygon[keep]


def _clip_polyhedron_half_space(
    faces, normal, offset, *, inside_tol, point_tol,
):
    """Clip a convex polyhedron against ``x.normal <= offset``.

    Faces are ordered convex polygons.  The clipping plane's intersection
    points form the one new cap face; its order is fixed by the outward
    normal, so subsequent triangulation is deterministic.
    """
    clipped_faces = []
    cap_points = []
    for face in faces:
        out = []
        previous = face[-1]
        previous_value = float(previous @ normal) - float(offset)
        previous_inside = previous_value <= inside_tol
        for current in face:
            current_value = float(current @ normal) - float(offset)
            current_inside = current_value <= inside_tol
            if current_inside != previous_inside:
                denominator = previous_value - current_value
                if denominator == 0.0:
                    raise RuntimeError(
                        "degenerate bulk mini-BZ half-space crossing")
                crossing = previous + (
                    previous_value / denominator) * (current - previous)
                out.append(crossing)
                cap_points.append(crossing)
            if current_inside:
                out.append(current)
            previous = current
            previous_value = current_value
            previous_inside = current_inside
        polygon = _dedupe_polyhedron_polygon(out, point_tol)
        if polygon.shape[0] >= 3:
            clipped_faces.append(polygon)

    unique_cap = []
    for point in cap_points:
        if not any(np.linalg.norm(point - old) <= point_tol
                   for old in unique_cap):
            unique_cap.append(point)
    if len(unique_cap) >= 3:
        cap = np.asarray(unique_cap, dtype=np.float64)
        center = np.mean(cap, axis=0)
        normal_unit = normal / np.linalg.norm(normal)
        relative = cap - center
        first = int(np.argmax(np.linalg.norm(relative, axis=1)))
        axis_1 = relative[first] / np.linalg.norm(relative[first])
        axis_2 = np.cross(normal_unit, axis_1)
        angles = np.arctan2(relative @ axis_2, relative @ axis_1)
        cap = cap[np.argsort(angles)]
        cap = _dedupe_polyhedron_polygon(cap, point_tol)
        if cap.shape[0] >= 3:
            clipped_faces.append(cap)
    return clipped_faces


def _polyhedron_indexed_faces(faces, point_tol):
    """Return one canonical vertex table and face-index description."""
    vertices = []
    indexed = []
    for face in faces:
        indices = []
        for point in face:
            matches = [
                index for index, old in enumerate(vertices)
                if np.linalg.norm(point - old) <= point_tol]
            if matches:
                index = matches[0]
            else:
                index = len(vertices)
                vertices.append(np.asarray(point, dtype=np.float64))
            if not indices or indices[-1] != index:
                indices.append(index)
        if len(indices) >= 3 and indices[0] == indices[-1]:
            indices.pop()
        if len(indices) >= 3:
            indexed.append(tuple(indices))
    return np.asarray(vertices, dtype=np.float64), tuple(indexed)


def _bulk_minibz_wigner_seitz_polyhedron(bvec, kgrid):
    r"""Return the true 3-D mini-lattice Wigner--Seitz polyhedron.

    The neighbour shell is finite by construction.  A centered fundamental
    parallelepiped gives the covering-radius bound
    ``R <= 1/2 sum_i |g_i|``.  Therefore every lattice vector longer than
    ``2R`` has a redundant Voronoi half-space on the entire WS cell.  If
    ``sigma_min`` is the smallest singular value of the mini-lattice matrix,
    ``|n| <= 2R/sigma_min`` contains every nonredundant integer vector.  The
    resulting fixed shell is clipped once; there is no convergence search.
    """
    bvec = np.asarray(bvec, dtype=np.float64)
    kg = tuple(int(value) for value in kgrid)
    if bvec.shape != (3, 3) or not np.all(np.isfinite(bvec)):
        raise ValueError("bulk polyhedron cubature requires finite (3,3) bvec")
    if len(kg) != 3 or any(value <= 0 for value in kg):
        raise ValueError(
            f"kgrid must contain three positive integers; got {kgrid}")
    mini = bvec / np.asarray(kg, dtype=np.float64)[:, None]
    singular_values = np.linalg.svd(mini, compute_uv=False)
    scale = float(np.max(singular_values))
    sigma_min = float(np.min(singular_values))
    singular_tol = 256.0 * np.finfo(np.float64).eps * max(scale, 1.0)
    if sigma_min <= singular_tol:
        raise ValueError("bulk mini-BZ lattice is singular")

    covering_bound = 0.5 * float(np.sum(np.linalg.norm(mini, axis=1)))
    shell = int(np.ceil(2.0 * covering_bound / sigma_min)) + 1
    if shell > 24:
        raise ValueError(
            "GATE bulk_minibz_neighbour_shell: exact WS construction would "
            "exceed its bounded neighbour-shell budget.\n"
            f"  got: shell half-width = {shell}, sigma_min = "
            f"{sigma_min:.6e}, covering bound = {covering_bound:.6e}\n"
            "  want: shell half-width <= 24\n"
            "  why: an extremely ill-conditioned reciprocal lattice makes "
            "the certified finite clipping problem impractically large\n"
            "  fix: supply a reduced primitive reciprocal basis; the "
            "physical lattice and its WS cell are unchanged")
    cutoff = 2.0 * covering_bound
    cutoff_tol = 4096.0 * np.finfo(np.float64).eps * max(cutoff, 1.0)
    neighbours = []
    for coefficients in itertools.product(range(-shell, shell + 1), repeat=3):
        if coefficients == (0, 0, 0):
            continue
        vector = np.asarray(coefficients, dtype=np.float64) @ mini
        norm = float(np.linalg.norm(vector))
        if norm <= cutoff + cutoff_tol:
            neighbours.append((norm, coefficients, vector))
    neighbours.sort(key=lambda item: (item[0], item[1]))

    cube_radius = 2.0 * covering_bound + max(scale, 1.0) * 1.0e-12
    lo, hi = -cube_radius, cube_radius
    faces = [
        np.asarray(points, dtype=np.float64)
        for points in (
            ((lo, lo, lo), (lo, lo, hi), (lo, hi, hi), (lo, hi, lo)),
            ((hi, lo, lo), (hi, hi, lo), (hi, hi, hi), (hi, lo, hi)),
            ((lo, lo, lo), (hi, lo, lo), (hi, lo, hi), (lo, lo, hi)),
            ((lo, hi, lo), (lo, hi, hi), (hi, hi, hi), (hi, hi, lo)),
            ((lo, lo, lo), (lo, hi, lo), (hi, hi, lo), (hi, lo, lo)),
            ((lo, lo, hi), (hi, lo, hi), (hi, hi, hi), (lo, hi, hi)),
        )
    ]
    point_tol = 32768.0 * np.finfo(np.float64).eps * max(scale, 1.0)
    inside_tol = point_tol * max(scale, 1.0)
    for _, _, normal in neighbours:
        faces = _clip_polyhedron_half_space(
            faces, normal, 0.5 * float(normal @ normal),
            inside_tol=inside_tol, point_tol=point_tol)
        if len(faces) < 4:
            raise RuntimeError(
                "bulk mini-BZ half-space intersection became empty")

    vertices, indexed_faces = _polyhedron_indexed_faces(faces, point_tol)
    if vertices.shape[0] < 4 or len(indexed_faces) < 4:
        raise RuntimeError("bulk mini-BZ polyhedron is not three-dimensional")
    if np.any(np.max(np.abs(vertices), axis=1) >= cube_radius - point_tol):
        raise RuntimeError(
            "bulk mini-BZ clipping retained an artificial bounding-cube face")
    for _, _, normal in neighbours:
        excess = vertices @ normal - 0.5 * float(normal @ normal)
        if float(np.max(excess)) > 4.0 * inside_tol:
            raise RuntimeError(
                "bulk mini-BZ vertex violates a certified lattice half-space")

    central_error = max(
        min(float(np.linalg.norm(point + partner)) for partner in vertices)
        for point in vertices)
    if central_error > 8.0 * point_tol:
        raise RuntimeError(
            "bulk mini-BZ polyhedron failed central symmetry: "
            f"max vertex-pair defect={central_error:.3e}")

    volume = 0.0
    for face in indexed_faces:
        left = vertices[face[0]]
        for offset in range(1, len(face) - 1):
            middle = vertices[face[offset]]
            right = vertices[face[offset + 1]]
            volume += abs(float(np.linalg.det(
                np.stack((left, middle, right), axis=0)))) / 6.0
    lattice_volume = abs(float(np.linalg.det(mini)))
    relative_error = abs(volume - lattice_volume) / lattice_volume
    if relative_error > 5.0e-12:
        raise RuntimeError(
            "bulk mini-BZ polyhedron failed its Voronoi cell-volume identity: "
            f"polyhedron={volume:.17e}, lattice={lattice_volume:.17e}, "
            f"relative_error={relative_error:.3e}")
    return vertices, indexed_faces, mini, volume


def _slab_minibz_polygon_rule(polygon, polygon_area, order):
    r"""One normalized Gamma-to-edge Duffy--Gauss rule.

    ``q(r,s)=r*((1-s)*v_i+s*v_{i+1})`` has Jacobian
    ``r*|v_i x v_{i+1}|``.  That radial factor cancels the slab bare
    kernel's integrable ``1/|q|`` cusp before summation.  Gauss nodes are
    interior to ``(0,1)``, so the singular point itself is never evaluated.
    """
    n = int(order)
    unit_nodes, unit_weights = gauss_legendre_interval(n, 0.0, 1.0)
    r, s = np.meshgrid(unit_nodes, unit_nodes, indexing="ij")
    wr, ws = np.meshgrid(unit_weights, unit_weights, indexing="ij")
    q_parts = []
    w_parts = []
    for left, right in zip(polygon, np.roll(polygon, -1, axis=0)):
        cross = abs(_det2(left, right))
        q2 = r[..., None] * (
            (1.0 - s)[..., None] * left + s[..., None] * right)
        w2 = wr * ws * r * cross
        q_parts.append(q2.reshape(-1, 2))
        w_parts.append(w2.reshape(-1))
    qxy = np.concatenate(q_parts, axis=0)
    average_weights = (
        np.concatenate(w_parts, axis=0) / float(polygon_area))
    q_cart = np.zeros((qxy.shape[0], 3), dtype=np.float64)
    q_cart[:, :2] = qxy
    weight_sum_defect = abs(float(np.sum(average_weights)) - 1.0)
    weighted_centroid = np.sum(
        average_weights[:, None] * q_cart, axis=0)
    if weight_sum_defect > 4.0e-15:
        raise RuntimeError("polygon cubature weights do not sum to one")
    q_scale = max(float(np.max(np.linalg.norm(q_cart[:, :2], axis=1))),
                  np.finfo(np.float64).tiny)
    if float(np.linalg.norm(weighted_centroid)) > 2.0e-13 * q_scale:
        raise RuntimeError(
            "polygon cubature failed its weighted-centroid identity: "
            f"|sum(w*q)|={np.linalg.norm(weighted_centroid):.3e}")
    return q_cart, average_weights, weight_sum_defect, weighted_centroid


def _bulk_minibz_polyhedron_rule(
    vertices, faces, minibz_measure, order,
):
    r"""One normalized Gamma-to-face tetrahedral Duffy--Gauss rule.

    Every convex face is triangulated as a fan.  On the tetrahedron joining
    ``Gamma`` to triangle ``(v0,v1,v2)``, the product-cube map is

    ``p(u,t)=(1-u)v0+u((1-t)v1+t v2)``, ``q(r,u,t)=r p(u,t)``.

    Its Jacobian is ``r^2 u |det(v0,v1,v2)|``.  The radial ``r^2`` factor
    cancels the bulk ``8*pi/|q|^2`` singularity algebraically before the
    weighted sum, and interior Gauss nodes never evaluate Gamma.
    """
    n = int(order)
    nodes, weights = gauss_legendre_interval(n, 0.0, 1.0)
    r, u, t = np.meshgrid(nodes, nodes, nodes, indexing="ij")
    wr, wu, wt = np.meshgrid(weights, weights, weights, indexing="ij")
    q_parts = []
    w_parts = []
    for face in faces:
        face_vertices = vertices[np.asarray(face, dtype=np.int64)]
        v0 = np.mean(face_vertices, axis=0)
        for offset in range(len(face)):
            v1 = face_vertices[offset]
            v2 = face_vertices[(offset + 1) % len(face)]
            determinant = abs(float(np.linalg.det(
                np.stack((v0, v1, v2), axis=0))))
            if determinant <= 0.0:
                raise RuntimeError(
                    "bulk mini-BZ face triangulation has zero volume")
            angular = (
                (1.0 - u)[..., None] * v0
                + u[..., None] * (
                    (1.0 - t)[..., None] * v1 + t[..., None] * v2))
            q_parts.append((r[..., None] * angular).reshape(-1, 3))
            w_parts.append(
                (wr * wu * wt * r * r * u * determinant).reshape(-1))
    q_cart = np.concatenate(q_parts, axis=0)
    average_weights = np.concatenate(w_parts, axis=0) / float(minibz_measure)
    weight_sum_defect = abs(float(np.sum(average_weights)) - 1.0)
    weighted_centroid = np.sum(
        average_weights[:, None] * q_cart, axis=0)
    if weight_sum_defect > 8.0e-15:
        raise RuntimeError("polyhedron cubature weights do not sum to one")
    q_scale = max(float(np.max(np.linalg.norm(q_cart, axis=1))),
                  np.finfo(np.float64).tiny)
    if float(np.linalg.norm(weighted_centroid)) > 5.0e-13 * q_scale:
        raise RuntimeError(
            "polyhedron cubature failed its weighted-centroid identity: "
            f"|sum(w*q)|={np.linalg.norm(weighted_centroid):.3e}")
    return q_cart, average_weights, weight_sum_defect, weighted_centroid


def _photon_D_raw(q_cart, *, kind, zc):
    """Bare Coulomb-gauge block on logical, nonzero q rows only."""
    q_valid = np.asarray(q_cart, dtype=np.float64)
    v, q2 = _minibz_kernel_bare(
        np.zeros(3), q_valid, kind=kind, zc=zc)
    transverse = transverse_projector(q_valid, q2)
    D_raw = np.zeros((q_valid.shape[0], 4, 4), dtype=np.float64)
    D_raw[:, 0, 0] = v
    D_raw[:, 1:, 1:] = (
        COULOMB_GAUGE_TT_SIGN * v[:, None, None] * transverse)
    if kind == "bulk_3d":
        # ``tr(P_T)=2`` is an algebraic Coulomb-gauge invariant.  Own its
        # floating representative here rather than asking two independently
        # rounded large cell sums to cancel afterward.  The slab payload is
        # deliberately untouched: its authenticated byte stream predates
        # the dimension-general receipt and is a regression oracle.
        D_raw[:, 3, 3] = (
            -2.0 * v - D_raw[:, 1, 1] - D_raw[:, 2, 2])
    return D_raw, q2


def _minibz_photon_receipt_digest(receipt) -> str:
    """Digest every receipt field and payload; ndarray flags are not trust.

    The dimension-two byte stream deliberately retains the pre-generalization
    labels and field order.  This makes the slab digest a regression oracle:
    changing only the Python field names cannot move an issued slab rule.
    """
    digest = hashlib.sha256()

    def _add_array(name, value):
        array = np.asarray(value)
        digest.update(name.encode("ascii") + b"\0")
        digest.update(array.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())

    digest.update(receipt.method.encode("utf-8") + b"\0")
    _add_array("orders", receipt.orders)
    _add_array("reciprocal_lattice_rows", receipt.reciprocal_lattice_rows)
    _add_array("kgrid", receipt.kgrid)
    _add_array("mini_lattice_rows", receipt.mini_lattice_rows)
    if int(receipt.dimension) == 2:
        _add_array("polygon_vertices", receipt.polytope_vertices)
        _add_array("polygon_area", receipt.minibz_measure)
        _add_array("slab_zc", receipt.slab_zc)
    else:
        _add_array("dimension", receipt.dimension)
        _add_array("polytope_vertices", receipt.polytope_vertices)
        for index, face in enumerate(receipt.polytope_faces):
            _add_array(f"polytope_face_{index}", face)
        _add_array("minibz_measure", receipt.minibz_measure)
        _add_array("slab_zc_is_none", receipt.slab_zc is None)
    _add_array("cell_volume", receipt.cell_volume)
    _add_array("physical_counts", receipt.physical_counts)
    _add_array("padded_counts", receipt.padded_counts)
    _add_array("weight_sum_defects", receipt.weight_sum_defects)
    _add_array("weighted_q_centroids", receipt.weighted_q_centroids)
    for index, chunk in enumerate(receipt.chunks):
        _add_array(f"chunk_{index}_order", chunk.order)
        _add_array(f"chunk_{index}_physical_count", chunk.physical_count)
        _add_array(f"chunk_{index}_q_cart", chunk.q_cart)
        _add_array(f"chunk_{index}_D_raw", chunk.D_raw)
        _add_array(f"chunk_{index}_sample_weight", chunk.sample_weight)
    return digest.hexdigest()


def slab_minibz_photon_cubature(
    kernel, geometry, kgrid,
) -> MinibzPhotonReceipt:
    """Issue the fixed exact-WS slab photon cubature receipt.

    There is no method token, order dial, or caller-supplied geometry fact on
    this surface.  The service constructs the true mini-lattice Voronoi cell,
    triangulates Gamma to every edge, evaluates the fixed 16/24/32 Duffy--GL
    ladder, and binds the normalized weights and padded/physical solve counts
    into one provider-issued result.  Raw ``D`` carries no cell-volume factor.
    """
    from vcoul.slab_2d import Slab2D
    if type(kernel) is not Slab2D:
        raise TypeError(
            "slab_minibz_photon_cubature needs the exact Slab2D kernel "
            "returned by get_kernel(2)")

    bvec = np.asarray(geometry.bvec, dtype=np.float64)
    kg = tuple(int(v) for v in kgrid)
    zc = kernel.truncation_half_height(geometry)
    cell_volume = float(geometry.cell_volume)
    if not np.isfinite(cell_volume) or cell_volume <= 0.0:
        raise ValueError(
            "slab photon cubature requires a finite positive cell_volume; "
            f"got {cell_volume}")
    polygon, mini_lattice, polygon_area = (
        _slab_minibz_wigner_seitz_polygon(bvec, kg))
    padded_count = int(polygon.shape[0]) * max(
        _SLAB_MINIBZ_PHOTON_ORDERS) ** 2

    chunks = []
    physical_counts = []
    weight_defects = []
    weighted_centroids = []
    for order in _SLAB_MINIBZ_PHOTON_ORDERS:
        q_valid, weight_valid, defect, centroid = (
            _slab_minibz_polygon_rule(polygon, polygon_area, order))
        physical_count = int(q_valid.shape[0])
        D_valid, _ = _photon_D_raw(q_valid, kind="slab", zc=zc)

        q_chunk = np.zeros((padded_count, 3), dtype=np.float64)
        D_chunk = np.zeros((padded_count, 4, 4), dtype=np.float64)
        sample_weight = np.zeros(padded_count, dtype=np.float64)
        q_chunk[:physical_count] = q_valid
        D_chunk[:physical_count] = D_valid
        sample_weight[:physical_count] = weight_valid
        for array in (q_chunk, D_chunk, sample_weight):
            array.setflags(write=False)

        centroid_tuple = tuple(float(v) for v in centroid)
        chunks.append(_MinibzPhotonChunk(
            order=int(order), q_cart=q_chunk, D_raw=D_chunk,
            physical_count=physical_count, sample_weight=sample_weight))
        physical_counts.append(physical_count)
        weight_defects.append(float(defect))
        weighted_centroids.append(centroid_tuple)

    def _rows(values):
        return tuple(tuple(float(x) for x in row) for row in values)

    receipt = MinibzPhotonReceipt(
        dimension=2,
        method=_SLAB_MINIBZ_PHOTON_METHOD,
        orders=_SLAB_MINIBZ_PHOTON_ORDERS,
        reciprocal_lattice_rows=_rows(bvec),
        kgrid=kg,
        mini_lattice_rows=_rows(mini_lattice),
        polytope_vertices=_rows(polygon),
        polytope_faces=tuple(
            (index, (index + 1) % int(polygon.shape[0]))
            for index in range(int(polygon.shape[0]))),
        minibz_measure=polygon_area,
        slab_zc=zc,
        cell_volume=cell_volume,
        physical_counts=tuple(physical_counts),
        padded_counts=(padded_count,) * len(_SLAB_MINIBZ_PHOTON_ORDERS),
        weight_sum_defects=tuple(weight_defects),
        weighted_q_centroids=tuple(weighted_centroids),
        chunks=tuple(chunks),
        _issue_token=_MINIBZ_PHOTON_RECEIPT_TOKEN,
    )
    object.__setattr__(
        receipt, "_provider_digest", _minibz_photon_receipt_digest(receipt))
    return validate_minibz_photon_receipt(receipt)


def bulk_minibz_photon_cubature(
    kernel, geometry, kgrid,
) -> MinibzPhotonReceipt:
    """Issue the fixed exact-WS bulk photon cubature receipt.

    The provider constructs the true three-dimensional mini-lattice Voronoi
    polyhedron from a rigorously bounded neighbour shell, triangulates its
    faces to Gamma, and evaluates the fixed 8/12/16 tetrahedral Duffy--Gauss
    ladder.  Weights are normalized cell-average weights.  Raw ``D`` carries
    no cell-volume factor.
    """
    from vcoul.bulk_3d import Bulk3D
    if type(kernel) is not Bulk3D:
        raise TypeError(
            "bulk_minibz_photon_cubature needs the exact Bulk3D kernel "
            "returned by get_kernel(3)")

    bvec = np.asarray(geometry.bvec, dtype=np.float64)
    kg = tuple(int(value) for value in kgrid)
    cell_volume = float(geometry.cell_volume)
    if not np.isfinite(cell_volume) or cell_volume <= 0.0:
        raise ValueError(
            "bulk photon cubature requires a finite positive cell_volume; "
            f"got {cell_volume}")
    vertices, faces, mini_lattice, minibz_measure = (
        _bulk_minibz_wigner_seitz_polyhedron(bvec, kg))
    triangle_count = sum(len(face) for face in faces)
    padded_count = triangle_count * max(_BULK_MINIBZ_PHOTON_ORDERS) ** 3

    chunks = []
    physical_counts = []
    weight_defects = []
    weighted_centroids = []
    for order in _BULK_MINIBZ_PHOTON_ORDERS:
        q_valid, weight_valid, defect, centroid = (
            _bulk_minibz_polyhedron_rule(
                vertices, faces, minibz_measure, order))
        physical_count = int(q_valid.shape[0])
        D_valid, _ = _photon_D_raw(q_valid, kind="bulk_3d", zc=None)

        q_chunk = np.zeros((padded_count, 3), dtype=np.float64)
        D_chunk = np.zeros((padded_count, 4, 4), dtype=np.float64)
        sample_weight = np.zeros(padded_count, dtype=np.float64)
        q_chunk[:physical_count] = q_valid
        D_chunk[:physical_count] = D_valid
        sample_weight[:physical_count] = weight_valid
        for array in (q_chunk, D_chunk, sample_weight):
            array.setflags(write=False)
        chunks.append(_MinibzPhotonChunk(
            order=int(order), q_cart=q_chunk, D_raw=D_chunk,
            physical_count=physical_count, sample_weight=sample_weight))
        physical_counts.append(physical_count)
        weight_defects.append(float(defect))
        weighted_centroids.append(tuple(float(value) for value in centroid))

    def _rows(values):
        return tuple(tuple(float(value) for value in row) for row in values)

    receipt = MinibzPhotonReceipt(
        dimension=3,
        method=_BULK_MINIBZ_PHOTON_METHOD,
        orders=_BULK_MINIBZ_PHOTON_ORDERS,
        reciprocal_lattice_rows=_rows(bvec),
        kgrid=kg,
        mini_lattice_rows=_rows(mini_lattice),
        polytope_vertices=_rows(vertices),
        polytope_faces=tuple(tuple(int(value) for value in face)
                             for face in faces),
        minibz_measure=minibz_measure,
        slab_zc=None,
        cell_volume=cell_volume,
        physical_counts=tuple(physical_counts),
        padded_counts=(padded_count,) * len(_BULK_MINIBZ_PHOTON_ORDERS),
        weight_sum_defects=tuple(weight_defects),
        weighted_q_centroids=tuple(weighted_centroids),
        chunks=tuple(chunks),
        _issue_token=_MINIBZ_PHOTON_RECEIPT_TOKEN,
    )
    object.__setattr__(
        receipt, "_provider_digest", _minibz_photon_receipt_digest(receipt))
    return validate_minibz_photon_receipt(receipt)


def minibz_photon_cubature(kernel, geometry, kgrid) -> MinibzPhotonReceipt:
    """Issue the exact photon Gamma-cell rule selected by kernel dimension."""
    try:
        dimension = int(kernel.sys_dim)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "minibz_photon_cubature needs a vcoul kernel returned by "
            "get_kernel(sys_dim)") from exc
    if dimension == 2:
        return slab_minibz_photon_cubature(kernel, geometry, kgrid)
    if dimension == 3:
        return bulk_minibz_photon_cubature(kernel, geometry, kgrid)
    raise NotImplementedError(
        "minibz_photon_cubature supports sys_dim=2 (slab) or 3 (bulk); "
        f"got {dimension}")


def validate_minibz_photon_receipt(
    receipt,
) -> MinibzPhotonReceipt:
    """Non-virtual validation of an exact provider-issued receipt."""
    if type(receipt) is not MinibzPhotonReceipt:
        raise TypeError(
            "production photon cubature requires the exact provider "
            "receipt type MinibzPhotonReceipt; subclasses, proxies, and "
            f"caller-labelled payloads are refused (got {type(receipt).__name__})")
    if receipt._provider_token is not _MINIBZ_PHOTON_RECEIPT_TOKEN:
        raise TypeError(
            "MinibzPhotonReceipt was not issued by minibz_photon_cubature")
    if receipt.dimension == 2:
        _require_slab_minibz_photon_receipt(receipt)
    elif receipt.dimension == 3:
        _require_bulk_minibz_photon_receipt(receipt)
    else:
        raise ValueError(
            "exact photon receipt dimension must be 2 or 3; got "
            f"{receipt.dimension!r}")
    return receipt


def _require_slab_minibz_photon_receipt(receipt) -> None:
    """Validate every small fact bound into one provider-issued receipt."""
    if receipt.dimension != 2 or receipt.method != _SLAB_MINIBZ_PHOTON_METHOD:
        raise ValueError(
            "exact slab photon receipt carries the wrong cubature method")
    if receipt.orders != _SLAB_MINIBZ_PHOTON_ORDERS:
        raise ValueError(
            "exact slab photon receipt must carry the fixed 16/24/32 ladder")
    bvec = np.asarray(receipt.reciprocal_lattice_rows, dtype=np.float64)
    mini = np.asarray(receipt.mini_lattice_rows, dtype=np.float64)
    polygon = np.asarray(receipt.polytope_vertices, dtype=np.float64)
    kg = receipt.kgrid
    if (type(kg) is not tuple or len(kg) != 3
            or any(type(value) is not int or value <= 0 for value in kg)
            or kg[2] != 1
            or bvec.shape != (3, 3)
            or mini.shape != (2, 2)
            or polygon.ndim != 2 or polygon.shape[1] != 2
            or polygon.shape[0] not in (4, 6)
            or not np.all(np.isfinite(bvec))
            or not np.all(np.isfinite(mini))
            or not np.all(np.isfinite(polygon))
            or receipt.polytope_faces != tuple(
                (index, (index + 1) % int(polygon.shape[0]))
                for index in range(int(polygon.shape[0])))
            or receipt.slab_zc is None
            or not np.isfinite(receipt.slab_zc)
            or receipt.slab_zc <= 0.0
            or not np.isfinite(receipt.cell_volume)
            or receipt.cell_volume <= 0.0):
        raise ValueError(
            "exact slab photon receipt carries invalid lattice/polygon rows")
    from vcoul.geometry import CoulombGeometry
    from vcoul.slab_2d import Slab2D
    expected_zc = Slab2D.truncation_half_height(CoulombGeometry(
        bvec=bvec, cell_volume=receipt.cell_volume))
    expected_polygon, expected_mini, expected_area = (
        _slab_minibz_wigner_seitz_polygon(bvec, kg))
    if (receipt.slab_zc != expected_zc
            or not np.array_equal(mini, expected_mini)
            or not np.array_equal(polygon, expected_polygon)):
        raise ValueError(
            "exact slab photon receipt geometry differs from the service "
            "Wigner-Seitz construction")
    crosses = np.asarray([
        _det2(left, right)
        for left, right in zip(polygon, np.roll(polygon, -1, axis=0))])
    polygon_area = 0.5 * _polygon_signed_twice_area(polygon)
    lattice_area = abs(_det2(mini[0], mini[1]))
    if (np.any(crosses <= 0.0)
            or not np.isfinite(receipt.minibz_measure)
            or receipt.minibz_measure != polygon_area
            or receipt.minibz_measure != expected_area
            or abs(polygon_area - lattice_area) / lattice_area > 2.0e-12):
        raise ValueError(
            "exact slab photon receipt failed its CCW/area identity")

    n_orders = len(_SLAB_MINIBZ_PHOTON_ORDERS)
    fields = (
        receipt.physical_counts, receipt.padded_counts,
        receipt.weight_sum_defects, receipt.weighted_q_centroids,
        receipt.chunks)
    if any(len(values) != n_orders for values in fields):
        raise ValueError(
            "exact slab photon receipt does not contain three complete rules")
    for chunk in receipt.chunks:
        if (type(chunk) is not _MinibzPhotonChunk
                or type(chunk.q_cart) is not np.ndarray
                or type(chunk.D_raw) is not np.ndarray
                or type(chunk.sample_weight) is not np.ndarray):
            raise TypeError(
                "exact slab photon receipt chunks require the provider's "
                "exact chunk and ndarray payload types")
    if (type(receipt._provider_digest) is not str
            or len(receipt._provider_digest) != 64
            or receipt._provider_digest
            != _minibz_photon_receipt_digest(receipt)):
        raise ValueError(
            "exact slab photon receipt payload or metadata changed after "
            "provider issuance")
    expected_physical = tuple(
        int(polygon.shape[0]) * order * order
        for order in _SLAB_MINIBZ_PHOTON_ORDERS)
    expected_padded = (int(polygon.shape[0]) * 32 * 32,) * n_orders
    if (receipt.physical_counts != expected_physical
            or receipt.padded_counts != expected_padded):
        raise ValueError(
            "exact slab photon receipt physical/padded solve counts drifted")
    for index, chunk in enumerate(receipt.chunks):
        physical = expected_physical[index]
        padded = expected_padded[index]
        q = np.asarray(chunk.q_cart)
        D = np.asarray(chunk.D_raw)
        weight = np.asarray(chunk.sample_weight)
        expected_q, expected_weight, defect, centroid = (
            _slab_minibz_polygon_rule(
                expected_polygon, receipt.minibz_measure,
                _SLAB_MINIBZ_PHOTON_ORDERS[index]))
        expected_D, _ = _photon_D_raw(
            expected_q, kind="slab", zc=receipt.slab_zc)
        if (chunk.order != receipt.orders[index]
                or chunk.physical_count != physical
                or q.shape != (padded, 3)
                or D.shape != (padded, 4, 4)
                or weight.shape != (padded,)
                or q.dtype != np.dtype(np.float64)
                or D.dtype != np.dtype(np.float64)
                or weight.dtype != np.dtype(np.float64)
                or not np.array_equal(q[:physical], expected_q)
                or not np.array_equal(D[:physical], expected_D)
                or not np.array_equal(weight[:physical], expected_weight)
                or np.any(q[physical:] != 0.0)
                or np.any(D[physical:] != 0.0)
                or np.any(weight[physical:] != 0.0)
                or receipt.weight_sum_defects[index] != defect
                or receipt.weighted_q_centroids[index]
                != tuple(float(v) for v in centroid)
                or defect > 4.0e-15):
            raise ValueError(
                "exact slab photon receipt chunk diagnostics do not match "
                f"the bound order {receipt.orders[index]}")


def _require_bulk_minibz_photon_receipt(receipt) -> None:
    """Validate every small fact bound into one provider-issued bulk rule."""
    if receipt.method != _BULK_MINIBZ_PHOTON_METHOD:
        raise ValueError(
            "exact bulk photon receipt carries the wrong cubature method")
    if receipt.orders != _BULK_MINIBZ_PHOTON_ORDERS:
        raise ValueError(
            "exact bulk photon receipt must carry the fixed 8/12/16 ladder")
    bvec = np.asarray(receipt.reciprocal_lattice_rows, dtype=np.float64)
    mini = np.asarray(receipt.mini_lattice_rows, dtype=np.float64)
    vertices = np.asarray(receipt.polytope_vertices, dtype=np.float64)
    kg = receipt.kgrid
    if (type(kg) is not tuple or len(kg) != 3
            or any(type(value) is not int or value <= 0 for value in kg)
            or bvec.shape != (3, 3) or mini.shape != (3, 3)
            or vertices.ndim != 2 or vertices.shape[1] != 3
            or vertices.shape[0] < 4
            or not np.all(np.isfinite(bvec))
            or not np.all(np.isfinite(mini))
            or not np.all(np.isfinite(vertices))
            or receipt.slab_zc is not None
            or not np.isfinite(receipt.minibz_measure)
            or receipt.minibz_measure <= 0.0
            or not np.isfinite(receipt.cell_volume)
            or receipt.cell_volume <= 0.0):
        raise ValueError(
            "exact bulk photon receipt carries invalid lattice/polyhedron rows")
    expected_vertices, expected_faces, expected_mini, expected_measure = (
        _bulk_minibz_wigner_seitz_polyhedron(bvec, kg))
    if (not np.array_equal(mini, expected_mini)
            or not np.array_equal(vertices, expected_vertices)
            or receipt.polytope_faces != expected_faces
            or receipt.minibz_measure != expected_measure):
        raise ValueError(
            "exact bulk photon receipt geometry differs from the service "
            "Wigner-Seitz construction")

    n_orders = len(_BULK_MINIBZ_PHOTON_ORDERS)
    fields = (
        receipt.physical_counts, receipt.padded_counts,
        receipt.weight_sum_defects, receipt.weighted_q_centroids,
        receipt.chunks)
    if any(len(values) != n_orders for values in fields):
        raise ValueError(
            "exact bulk photon receipt does not contain three complete rules")
    for chunk in receipt.chunks:
        if (type(chunk) is not _MinibzPhotonChunk
                or type(chunk.q_cart) is not np.ndarray
                or type(chunk.D_raw) is not np.ndarray
                or type(chunk.sample_weight) is not np.ndarray):
            raise TypeError(
                "exact bulk photon receipt chunks require the provider's "
                "exact chunk and ndarray payload types")
    if (type(receipt._provider_digest) is not str
            or len(receipt._provider_digest) != 64
            or receipt._provider_digest
            != _minibz_photon_receipt_digest(receipt)):
        raise ValueError(
            "exact bulk photon receipt payload or metadata changed after "
            "provider issuance")
    triangle_count = sum(len(face) for face in expected_faces)
    expected_physical = tuple(
        triangle_count * order ** 3
        for order in _BULK_MINIBZ_PHOTON_ORDERS)
    expected_padded = (
        triangle_count * max(_BULK_MINIBZ_PHOTON_ORDERS) ** 3,
    ) * n_orders
    if (receipt.physical_counts != expected_physical
            or receipt.padded_counts != expected_padded):
        raise ValueError(
            "exact bulk photon receipt physical/padded solve counts drifted")
    for index, chunk in enumerate(receipt.chunks):
        physical = expected_physical[index]
        padded = expected_padded[index]
        q = np.asarray(chunk.q_cart)
        D = np.asarray(chunk.D_raw)
        weight = np.asarray(chunk.sample_weight)
        expected_q, expected_weight, defect, centroid = (
            _bulk_minibz_polyhedron_rule(
                expected_vertices, expected_faces, expected_measure,
                _BULK_MINIBZ_PHOTON_ORDERS[index]))
        expected_D, _ = _photon_D_raw(
            expected_q, kind="bulk_3d", zc=None)
        if (chunk.order != receipt.orders[index]
                or chunk.physical_count != physical
                or q.shape != (padded, 3)
                or D.shape != (padded, 4, 4)
                or weight.shape != (padded,)
                or q.dtype != np.dtype(np.float64)
                or D.dtype != np.dtype(np.float64)
                or weight.dtype != np.dtype(np.float64)
                or not np.array_equal(q[:physical], expected_q)
                or not np.array_equal(D[:physical], expected_D)
                or not np.array_equal(weight[:physical], expected_weight)
                or np.any(q[physical:] != 0.0)
                or np.any(D[physical:] != 0.0)
                or np.any(weight[physical:] != 0.0)
                or receipt.weight_sum_defects[index] != defect
                or receipt.weighted_q_centroids[index]
                != tuple(float(value) for value in centroid)
                or defect > 8.0e-15):
            raise ValueError(
                "exact bulk photon receipt chunk diagnostics do not match "
                f"the bound order {receipt.orders[index]}")


def iter_minibz_photon_samples(
    kernel,
    geometry,
    kgrid,
    *,
    nsamples: int = 2**18,
    method: str = "sobol",
    qmc_reps: int = 10,
    analytic_sphere: bool = False,
    chunk_size: int = 2**15,
):
    """Stream q and the raw bare Coulomb-gauge ``C⊕T`` kernel at q=0.

    The fixed basis is ``(C, Tx, Ty, Tz)`` and, for every valid Cartesian
    mini-BZ sample ``q``, the returned real ``D_raw`` obeys

    ``D_CC = v(q)``, ``D_TT = -v(q) (I - qhat qhat)``,
    ``D_CT = D_TC = 0``.

    ``v`` is the existing bulk/slab mini-BZ bare kernel in **bare units**:
    no ``1 / cell_volume`` is applied.  Draw geometry, seeds,
    Sobol demotion, slab ``qz=0``, and the ``nmax=1`` versus ``nmax=3``
    ``analytic_sphere`` policy all route through the sampler used by
    :meth:`Bulk3D.q0_average` and :meth:`Slab2D.q0_average`.  The projector
    is evaluated from runtime q directions with NumPy; there is no
    direction-specialized JIT family.

    Returns
    -------
    iterator
        Each item is
        ``(rep, start, stop, q_cart, D_raw, valid_count, mc_weight,
        analytic_D_raw)``.  ``q_cart`` has fixed shape ``(chunk_size, 3)``
        and ``D_raw`` fixed shape ``(chunk_size, 4, 4)``; both are host
        ``float64`` NumPy arrays.  The final chunk is zero padded and
        ``valid_count = stop - start``.  Invalid rows have zero q, zero D,
        and zero weight, and are never evaluated by the singular kernel.

        Accumulate one replicate as
        ``sum(mc_weight * completed_integrand) / sum(valid_count)``.  For the
        *linear bare-kernel parity only*, add the yielded
        ``analytic_D_raw`` values; the addend is nonzero only in the first
        chunk of a 3D ``analytic_sphere`` replicate.  It is deliberately
        separate because it is not a screened coupled solve inside the
        excised sphere.  ``rep`` lets the caller finish each completed
        integrand before averaging equally over replicates; no replicate
        stack is materialized.

    Notes
    -----
    Memory is ``O(nsamples * 3 + chunk_size * 16)`` for one replicate and is
    independent of ``qmc_reps``.  A 0-D box has no finite-q mini-BZ and
    refuses.  This surface accepts only ``sobol``, ``auto``, and ``uniform``;
    unlike the legacy sampler, an unknown token cannot silently mean uniform.
    """
    try:
        sys_dim = int(kernel.sys_dim)
    except (AttributeError, TypeError, ValueError) as exc:
        raise TypeError(
            "iter_minibz_photon_samples needs a vcoul kernel from "
            "get_kernel(sys_dim)") from exc
    if sys_dim == 0:
        raise NotImplementedError(
            "iter_minibz_photon_samples: a 0-D box is Gamma-only and has "
            "no finite-q mini-BZ sample policy")
    if sys_dim not in (2, 3):
        raise ValueError(
            f"iter_minibz_photon_samples: kernel sys_dim={sys_dim!r}; "
            "expected 2 (slab) or 3 (bulk)")

    draw_method = str(method).lower()
    if draw_method not in ("sobol", "auto", "uniform"):
        raise ValueError(
            f"iter_minibz_photon_samples: method={method!r} unsupported; "
            "expected 'sobol', 'auto', or 'uniform'")
    if int(nsamples) <= 0:
        raise ValueError("iter_minibz_photon_samples: nsamples must be positive")
    if int(qmc_reps) <= 0:
        raise ValueError("iter_minibz_photon_samples: qmc_reps must be positive")
    if int(chunk_size) <= 0:
        raise ValueError("iter_minibz_photon_samples: chunk_size must be positive")
    kg = tuple(int(s) for s in kgrid)
    if len(kg) != 3 or any(s <= 0 for s in kg):
        raise ValueError(
            f"iter_minibz_photon_samples: kgrid={kgrid!r}; expected three "
            "positive integers")

    is_2d = sys_dim == 2
    bvec = np.asarray(geometry.bvec, dtype=np.float64)
    if bvec.shape != (3, 3):
        raise ValueError(
            f"iter_minibz_photon_samples: geometry.bvec shape={bvec.shape}; "
            "expected (3, 3)")
    zc = kernel.truncation_half_height(geometry) if is_2d else None
    kind = "slab" if is_2d else "bulk_3d"
    q0sph2 = minibz_inscribed_sphere_r2(bvec, kg, is_2d=is_2d)

    analytic_D = np.zeros((4, 4), dtype=np.float64)
    if analytic_sphere and not is_2d:
        analytic = _analytic_sphere_bare_head(
            q0sph2, geometry.cell_volume, int(np.prod(kg)))
        analytic_D[0, 0] = analytic
        analytic_D[1:, 1:] = (
            COULOMB_GAUGE_TT_SIGN * np.eye(3)
            * ((2.0 / 3.0) * analytic))
    zero_analytic_D = np.zeros_like(analytic_D)

    def _chunks():
        q_reps = _sample_q0_minibz_qpoints(
            geometry, kg, nsamples=int(nsamples), method=draw_method,
            qmc_reps=int(qmc_reps), analytic_sphere=bool(analytic_sphere),
            is_2d=is_2d, stream_reps=True)
        for rep, q_rep in enumerate(q_reps):
            q_rep = np.asarray(q_rep, dtype=np.float64)
            n_rep = int(q_rep.shape[0])
            for start in range(0, n_rep, int(chunk_size)):
                stop = min(start + int(chunk_size), n_rep)
                valid_count = stop - start
                q_valid = q_rep[start:stop]

                # Evaluate only the logical rows.  Padding is appended after
                # the singular bare-kernel call and is therefore inert.
                D_valid, q2 = _photon_D_raw(q_valid, kind=kind, zc=zc)

                q_chunk = np.zeros((int(chunk_size), 3), dtype=np.float64)
                D_chunk = np.zeros(
                    (int(chunk_size), 4, 4), dtype=np.float64)
                mc_weight = np.zeros(int(chunk_size), dtype=np.float64)
                q_chunk[:valid_count] = q_valid
                D_chunk[:valid_count] = D_valid
                if analytic_sphere and not is_2d:
                    mc_weight[:valid_count] = (q2 > q0sph2)
                else:
                    mc_weight[:valid_count] = 1.0

                yield (
                    rep, start, stop, q_chunk, D_chunk, valid_count,
                    mc_weight,
                    analytic_D.copy() if start == 0 else zero_analytic_D.copy(),
                )

    return _chunks()


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
            analytic = _analytic_sphere_bare_head(q0sph2, celvol, n_kpts)
            per_batch.append(mc + analytic)
        else:
            if adaptive and len_shift2 > 1e-12:
                n_q = int(round(n_coarse * 4.0 * q0sph2 / len_shift2))
                n_q = max(1, min(n_q, n_tot))
            else:
                n_q = n_tot
            per_batch.append(float(np.mean(v[:n_q])))
    return float(np.mean(per_batch))


def minibz_moment_tensor(
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
) -> np.ndarray:
    """Mini-BZ Voronoi CELL AVERAGE of ``v(q) q_a q_b`` — the (3,3) moment.

    The second moment of the Coulomb kernel over the same cell, on the same
    draws, in the same **bare** units as :func:`minibz_average` (no
    ``1/celvol``; the caller applies its own volume convention).  ``q`` is
    the FULL Cartesian momentum ``shift_cart + δq``, not the offset — the
    tensor is the coefficient of a dipole bilinear at the cell's own
    location, so it must carry the cell's absolute position.

    WHY A TENSOR AND NOT A SCALAR.  The BSE exchange head at momentum Q is
    ``v(q)·|q·d|²`` with ``d`` the transition dipole, and the cell average
    of that is not ``⟨v⟩·|Q·d|²``.  Because ``d`` is a property of the
    transition and not of the integration variable, the average factorises
    exactly:

        ⟨ v(q) |q·d|² ⟩_cell  =  conj(d_a) · M_ab · d_b ,
        M_ab = ⟨ v(q) q_a q_b ⟩_cell ,

    with no small-q expansion.  Replacing ``M_ab`` by ``⟨v⟩ q̂_a q̂_b |Q|²``
    — which is what a scalar average amounts to — throws away every
    direction in the cell but the one sampled point's, AND weights the
    radius at the sample rather than by ``v``.  The two errors are
    independent (one angular, one radial), so neither is fixable alone.
    See ``LT_HEAD_PROBLEM.md`` §2.2 and §3.

    THE TRACE IS A FREE, EXACT DIAGNOSTIC.  In 3D bulk ``v(q) q² = 8π``
    identically, so

        tr M = 8π   (bare)   =   8π/Ω   after the caller's 1/celvol,

    for any cell shape and any shift.  On the plain-MC branch that holds
    pointwise and only checks the algebra; on the Baldereschi
    analytic-sphere branch it is a real test of the estimator, because the
    analytic term has to supply exactly the trace the MC drops by skipping
    the inscribed sphere.  Under the 2D slab truncation the same trace
    becomes ``⟨8π·f2d⟩ → 8π z_c ⟨|q∥|⟩``, which vanishes LINEARLY with the
    cell, and ``M_zz`` is identically zero for the ``q_z = 0`` slab — the
    rank-two structure a scalar cannot represent.  No dimensional branch:
    the geometry arrives entirely through ``kind`` and the draws.

    ESTIMATOR.  Deliberately the same two BGW branches as
    :func:`minibz_average`, sample for sample, so the tensor and the scalar
    are the same average of different integrands and can be compared:

    * ``|shift|² < TOL`` and ``analytic_sphere`` (3D only) — MC of
      ``v·q_a q_b`` OUTSIDE the inscribed sphere divided by the FULL sample
      count, plus the closed-form sphere term.  The scalar's
      ``4·√q0sph2·celvol·N_k/π`` has an isotropic tensor twin,

          ∫_{|q|<q0} (8π/q²) q_a q_b d³q / V_mBZ
              = δ_ab · (4/9π) · q0³ · celvol · N_k ,

      whose trace is exactly the sphere's share of ``8π``, which is why the
      trace diagnostic closes on this branch.
    * else — pure adaptive MC on the first ``n_q`` draws, ``n_q`` by the
      same clamp.

    Returns a real ``(3, 3)`` symmetric ``numpy`` array, meaned over the
    replicate batches (the same free error bar as the scalar).
    """
    shift = np.asarray(shift_cart, dtype=np.float64)
    len_shift2 = float(np.dot(shift, shift))
    head_branch = (len_shift2 < 1e-12) and analytic_sphere
    per_batch = []
    for dq in dq_batches:
        dq = np.asarray(dq, dtype=np.float64)
        n_tot = dq.shape[0]
        v, len2 = _minibz_kernel_bare(shift, dq, kind=kind,
                                      alpha=alpha, zc=zc)
        K = shift[None, :] + dq                       # (N, 3) full momentum
        if head_branch:
            w = np.where(len2 > q0sph2, v, 0.0)
            M = (K * w[:, None]).T @ K / float(n_tot)
            # Baldereschi tensor twin: isotropic, trace = the sphere's share
            # of 8π, so tr M stays 8π across the split.
            M = M + np.eye(3) * (4.0 / (9.0 * np.pi)) * q0sph2 ** 1.5 \
                * float(celvol) * float(n_kpts)
        else:
            if adaptive and len_shift2 > 1e-12:
                n_q = int(round(n_coarse * 4.0 * q0sph2 / len_shift2))
                n_q = max(1, min(n_q, n_tot))
            else:
                n_q = n_tot
            Kq = K[:n_q]
            M = (Kq * v[:n_q, None]).T @ Kq / float(n_q)
        per_batch.append(M)
    return np.mean(np.stack(per_batch, axis=0), axis=0)


def minibz_transverse_head_avg(
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
    eps_K2: float = 1e-30,
) -> np.ndarray:
    """Mini-BZ Voronoi CELL AVERAGE of ``v(q) t_ab(q̂)`` — the bare
    Coulomb-gauge transverse-projector head, ``t_ab(q̂) = δ_ab − q̂_a q̂_b``.

    This is the current-current (bispinor TT) analogue of
    :func:`minibz_average`'s scalar charge head ``⟨v(q)⟩``: the CHARGE
    structure factor obeys ``M_mn(q→0) → δ_mn`` so its q=0 exchange slot
    needs only the bare cell average ``⟨v⟩``, but the CURRENT structure
    factor ``⟨m|α^i|n⟩`` is finite and generically non-diagonal, and the
    q=0 slot of the bare transverse propagator carries the DIRECTION-
    DEPENDENT projector ``t_ab(q̂)`` rather than the identity.  A single
    grid point cannot represent that — ``t_ab`` has no limit as ``q→0`` —
    so the correct discrete-BZ-sum replacement for the zeroed q=0 slot is
    this cell average, exactly as ``⟨v⟩`` replaces the zeroed CC slot.

    For an isotropic 3D cell the closed form is
    ``⟨t_ab⟩_angle = (2/3) δ_ab`` (the projector's trace is 2 in every
    direction); for the in-plane mini-BZ of a slab it is
    ``diag(1/2, 1/2, 1)`` — the measured LORRAX reference value
    (``docs/BISPINOR_DHFB_DESIGN.md`` §11, bi4 deck: 0.4993, 0.5007,
    1.0000).  Neither closed form is assumed here; both fall out of the
    same Monte-Carlo estimator ``minibz_average`` already uses, weighted
    by ``t_ab`` instead of ``1``.

    ``K2_safe`` guards the SAME single degenerate sample any batch can
    contain — ``shift_cart = 0`` and ``δq = 0`` exactly never occurs on a
    continuous Sobol/uniform draw, so this only matters for a
    pathological/test batch; production draws never hit it.

    ESTIMATOR — same two BGW branches as :func:`minibz_average`, sample
    for sample:

    * ``|shift|² < TOL`` and ``analytic_sphere`` (3D only) — MC of
      ``v·t_ab`` OUTSIDE the inscribed sphere (÷ full sample count) plus
      the closed-form isotropic sphere term.  The angular average of
      ``t_ab`` over a full sphere is ``(2/3)δ_ab`` for ANY radius, so the
      sphere's analytic contribution is the scalar Baldereschi-Tosatti
      term (:func:`minibz_average`) times ``(2/3)δ_ab`` — the same
      factorisation :func:`minibz_moment_tensor` uses for its own
      isotropic sphere twin.
    * else — pure adaptive MC on the first ``n_q`` draws (2D slab heads,
      whose ``|Q|`` cusp is integrable, always take this branch).

    Returns a real ``(3, 3)`` array, meaned over the replicate batches
    (the same free error bar as :func:`minibz_average`).  Bare units —
    NO ``1/celvol`` — matching :func:`minibz_average`'s convention; the
    caller applies its own volume factor at injection.
    """
    shift = np.asarray(shift_cart, dtype=np.float64)
    len_shift2 = float(np.dot(shift, shift))
    head_branch = (len_shift2 < 1e-12) and analytic_sphere
    per_batch = []
    for dq in dq_batches:
        dq = np.asarray(dq, dtype=np.float64)
        n_tot = dq.shape[0]
        v, len2 = _minibz_kernel_bare(shift, dq, kind=kind,
                                      alpha=alpha, zc=zc)
        K = shift[None, :] + dq                       # (N, 3) full momentum
        t = transverse_projector(K, len2, eps_K2=eps_K2)
        if head_branch:
            outside = len2 > q0sph2
            w = np.where(outside, v, 0.0)
            T = np.einsum('n,nab->ab', w, t) / float(n_tot)
            analytic = _analytic_sphere_bare_head(q0sph2, celvol, n_kpts)
            T = T + np.eye(3) * (2.0 / 3.0) * analytic
        else:
            if adaptive and len_shift2 > 1e-12:
                n_q = int(round(n_coarse * 4.0 * q0sph2 / len_shift2))
                n_q = max(1, min(n_q, n_tot))
            else:
                n_q = n_tot
            T = np.einsum('n,nab->ab', v[:n_q], t[:n_q]) / float(n_q)
        per_batch.append(T)
    return np.mean(np.stack(per_batch, axis=0), axis=0)


def build_miniBZ_dq_cart(
    kgrid: tuple[int, int, int],
    bvec: np.ndarray,
    *,
    nmc: int = 2**18,
    seed: int = 42,
) -> np.ndarray:
    """CENTROSYMMETRIC Monte-Carlo sample of the mini-BZ, Cartesian.

    Returns ``(2 * nmc, 3)`` offsets δq filling the Voronoi cell of the
    mini-BZ, as the union of an ``nmc``-point draw with its own negation.

    The centrosymmetry is LOAD-BEARING, not cosmetic.  The head injected
    at a slot is ``⟨v(K + δq)⟩``, and ``V_q = conj(V_{−q})`` requires the
    injected value to satisfy ``f(K) = f(−K)``.  Since
    ``⟨v(−K + δq)⟩ = ⟨v(K − δq)⟩``, that identity holds EXACTLY iff the
    finite sample set obeys ``{δq} = {−δq}``.  The historical one-sided
    draw was symmetric only in the ``nmc → ∞`` limit, so it left an
    MC-noise-sized residual in reciprocity even once the slot selection
    was right.  The Voronoi cell is centrosymmetric as a SET, so negating
    the draw keeps every point inside it and the estimator unbiased.

    Row convention: ``bvec`` rows are the Cartesian reciprocal vectors, so
    the draw goes through :func:`minibz_frac_to_cart` (``U @ bvec``) — the
    single place that convention is decidable.  The transposed spelling
    (shipped 2026-05-16..2026-08-07, fixed 358bb0b) spans the COLUMN
    parallelepiped, which is not a fundamental domain of the row lattice:
    the wrapped cloud double-covers part of the cell and misses part, with
    the same total volume, so no normalisation check can see it.  Si FCC is
    provably blind to the difference (``bvec.T = P·bvec``, P cyclic ⇒ pure
    reseed); on non-cubic cells it is a bias worth ~50 % of the whole
    mc-average correction.  Pinned by
    ``tests/test_vcoul_minibz_head_draw.py``.
    """
    bvec = np.asarray(bvec, dtype=np.float64)
    bvec_j = jnp.asarray(bvec, dtype=jnp.float64)
    nkx, nky, nkz = (int(s) for s in kgrid)
    rng = np.random.RandomState(seed)
    randvals = rng.uniform(0, 1, (int(nmc), 3))
    randcart = minibz_frac_to_cart(randvals, bvec)      # np: randvals @ bvec
    wrapped = np.asarray(wrap_points_to_voronoi(
        jnp.asarray(randcart), bvec_j, nmax=1))
    randlims = minibz_cell_affine(bvec, (nkx, nky, nkz))
    dq = (randlims @ wrapped.T).T   # (nmc, 3) mini-BZ offsets in Cartesian
    return np.concatenate([dq, -dq], axis=0)


def build_v_head_miniBZ_fn_3d(
    kgrid: tuple[int, int, int],
    bvec: np.ndarray,
    cell_volume: float,
    *,
    nmc: int = 2**18,
    seed: int = 42,
):
    """Mini-BZ-averaged Coulomb head as a FUNCTION of Cartesian ``q+G``.

    3D bulk only.  Returns ``head_fn(K_cart) -> (m,) float64``, where
    ``K_cart`` is ``(m, 3)`` Cartesian ``q+G`` and the value is
    ``⟨8π/|K + δq|²⟩_miniBZ / Ω_cell`` in Rydberg.  This is the
    ``v_head_fn`` argument of :func:`~vcoul.base.v_qG_table`, which
    evaluates it at every slot attaining ``argmin |q+G|²`` — see that
    function's HEAD SLOT note for why the selection is by argmin and why
    ALL of the argmin.

    This REPLACES ``build_v_head_miniBZ_avg_3d``, which returned an
    ``(nkx, nky, nkz)`` table indexed by the BGW-wrapped q index.  That
    shape was unfixable rather than merely inconvenient: a per-q value is
    attached to ``q_frac @ bvec``, which is NOT the argmin ``q+G`` on 12
    of Si's 64 q, and it carries no per-slot resolution at all — the
    caller must evaluate the head at each tied slot's own K before it can
    average over the tied set.

    ``f(K) = f(−K)`` holds to machine precision here because
    :func:`build_miniBZ_dq_cart` is centrosymmetric; Γ is not
    special-cased (the caller skips it — the q→0 head is the separate
    rank-1 Σ_X term).

    ``bvec`` is taken RAW rather than as a
    :class:`~vcoul.geometry.CoulombGeometry` because the one production
    caller has ``(kgrid, bvec, cell_volume)`` in hand and no wfn, and
    because the frozen-reference pins (seed=42, nmc=2**18) are pins on
    THIS signature.
    """
    dq_cart = build_miniBZ_dq_cart(kgrid, bvec, nmc=nmc, seed=seed)
    fact = 1.0 / float(cell_volume)

    def head_fn(K_cart) -> np.ndarray:
        K = np.asarray(K_cart, dtype=np.float64).reshape(-1, 3)
        out = np.empty(K.shape[0], dtype=np.float64)
        for i in range(K.shape[0]):
            # Shared bare-3D kernel (2026-08-07 consolidation).  For
            # |K+δq|² ≥ 1e-24 the arithmetic is bit-identical to the
            # historical inline 8π/Σ(shifted²); the kernel's guard
            # additionally maps a pathological exact-hit to 0 instead of
            # inf.  Estimator unchanged: plain mean over the whole draw.
            v, _len2 = _minibz_kernel_bare(K[i], dq_cart, kind="bulk_3d")
            out[i] = np.mean(v)
        return out * fact

    return head_fn
