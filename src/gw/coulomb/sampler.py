"""The ONE mini-BZ Monte-Carlo cell average.

``<v(shift + dq)>_miniBZ`` — the average of the bare Coulomb kernel over the
Voronoi cell of the q-grid — had three independent implementations:

* ``gw.compute_vcoul.build_v_head_miniBZ_avg_3d`` — the 3D body-head table.
  Serial, ``RandomState(42)``, ``nmax=1``, no adaptive sample count, and it
  sampled ``randvals @ bvec.T``: the COLUMNS of ``bvec`` are not a lattice
  period, so the Voronoi wrap was not measure-preserving and the estimator
  was BIASED (0% on a symmetric ``bvec``, up to 64% per-q on fcc
  ``ibrav=2``; Frontera job 7890612).  Deleted 2026-08-05.
* ``gw.coulomb.base.minibz_voronoi_batches`` + ``minibz_average`` — correct
  sampling (``U @ bvec``), scrambled Sobol, adaptive N, the
  Baldereschi-Tosatti sphere split.  Serial and replicated on every rank.
* ``bse.vq_interp.minibz_head_vlr`` — correct sampling AND rank-parallel:
  disjoint rank slabs over a global kept-sample index, per-sample
  randomness from ``jax.random.fold_in(key, global_slot)``, summed with
  ``process_allgather``.

:func:`minibz_cell_average` is the third one generalised: it keeps the
rank-parallel structure (which is BIT-IDENTICAL ACROSS RANK COUNTS, because
a sample is a pure function of its global slot index — the scorecard
records head = 27.38911... at nproc 1/2/4), and gains the Sobol draw, the
adaptive count and the analytic sphere term from the second.  Because it is
rank-invariant, ``distribute=True`` can be the default: distributing costs
nothing in reproducibility.

The kernel it evaluates is :func:`gw.coulomb.kernel.v_qG` — the same
formula the per-sphere builders use, not a private copy.

Every cloud in this module is filled AND folded by :func:`_fill_and_wrap`,
which refuses when the two use different bases
(:func:`assert_fill_matches_wrap`).  That is the 2026-04-05 defect's whole
class asserted rather than measured; :func:`_selftest_fill_wrap_guard` runs
at import so the guard is never silently vacuous.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .kernel import TOL_MC_NAN, TOL_QG_ZERO, v_qG

# Announced once per process when the Sobol engine is unavailable.
_QMC_DEMOTION_ANNOUNCED = False


def _qmc_engine():
    """``scipy.stats.qmc`` or None, announcing the demotion exactly once.

    Was a bare ``except Exception`` that silently fell through to uniform
    sampling — so a missing scipy turned a low-discrepancy estimator into a
    plain one with no trace in any log.  A demotion that changes the
    convergence rate of a production estimator has to say so.
    """
    global _QMC_DEMOTION_ANNOUNCED
    try:
        from scipy.stats import qmc as _qmc
        return _qmc
    except ImportError as exc:
        if not _QMC_DEMOTION_ANNOUNCED:
            _QMC_DEMOTION_ANNOUNCED = True
            print(f"[coulomb.sampler] DEMOTION: scipy.stats.qmc unavailable "
                  f"({exc}); the mini-BZ average falls back to plain uniform "
                  f"sampling.  Same estimator, worse convergence "
                  f"(N^-1/2 instead of ~N^-1) — raise nsamples or install "
                  f"scipy if this average feeds a pinned number.", flush=True)
        return None


def minibz_inscribed_sphere_r2(bvec, kgrid, *, is_2d: bool = False) -> float:
    """q0sph2 — squared radius of the largest sphere inscribed in the mini-BZ.

    The mini-BZ is the Voronoi cell of the q-grid; its reciprocal lattice
    is ``b_i / nk_i``.  The nearest Voronoi face sits at half the shortest
    mini-BZ reciprocal vector, so ``q0sph2 = min_{n!=0} |0.5*sum_i n_i b_i/nk_i|^2``
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


def _randlims(bvec, kgrid):
    """Cartesian map that shrinks the full BZ Voronoi cell to the mini-BZ.

    ``B^T diag(1/nk) B^-T``: in fractional coordinates it is just "divide
    each b_i component by nk_i", written as a Cartesian operator so it can
    be applied to wrapped Cartesian points.

    REGISTERED DEBT: ``base.minibz_voronoi_batches`` still builds this same
    3x3 with ``jnp.linalg.inv`` inline.  It is not routed here because the
    last bits of ``jnp.linalg.inv`` and ``np.linalg.inv`` need not agree,
    and that path feeds numbers pinned before 2026-08-05
    (``tests/test_minibz_average.py``, the BSE head scorecard).  Collapse
    the two when those pins are next re-cut.
    """
    bvec = np.asarray(bvec, dtype=np.float64)
    return bvec.T @ (np.diag(1.0 / np.asarray([int(s) for s in kgrid],
                                              dtype=np.float64))
                     @ np.linalg.inv(bvec.T))


def _wrap(randcart, bvec, nmax):
    from ..vcoul import wrap_points_to_voronoi
    return np.asarray(wrap_points_to_voronoi(
        jnp.asarray(np.asarray(randcart, dtype=np.float64)),
        jnp.asarray(np.asarray(bvec, dtype=np.float64)), nmax=nmax),
        dtype=np.float64)


# ---------------------------------------------------------------------------
# The fill/wrap convention guard.
#
# The 2026-04-05 defect's CLASS, stated once: *the cloud was generated from
# one set of basis vectors and folded against another*.  ``randvals @
# bvec.T`` filled the parallelepiped spanned by the COLUMNS of ``bvec``;
# :func:`gw.vcoul.wrap_points_to_voronoi` folds against ``shifts @ bvec``,
# the lattice generated by its ROWS.  Two bases, one fold, no complaint.
#
# This is asserted, not measured: three 3x3 operations on the fill path, no
# sampling, no tolerance on a physical quantity.  A reintroduction refuses on
# the first draw instead of biasing a table nobody re-derives.
# ---------------------------------------------------------------------------

#: Dimensionless; ``M`` below is a ratio of bases, so this is scale-free.
_FILL_WRAP_TOL = 1.0e-9


class MiniBZBasisMismatch(ValueError):
    """A mini-BZ cloud was filled from one basis and folded against another."""


def assert_fill_matches_wrap(fill_basis, wrap_basis):
    """Refuse unless the filled parallelepiped IS the wrap lattice's cell.

    ``fill_basis`` — rows span the parallelepiped the ``U in [0,1)^3`` cloud
    fills.  ``wrap_basis`` — rows generate the lattice the Voronoi fold uses
    as candidate shifts.  ``M = fill @ inv(wrap)`` must be the identity.

    WHY IDENTITY AND NOT MERELY "A FUNDAMENTAL DOMAIN".  The weaker
    *sufficient* condition is that ``M`` be an INTEGER matrix with
    ``|det M| = 1`` — then the fill parallelepiped is *a* fundamental domain
    of the wrap lattice, the fold is measure-preserving, and even ``bvec.T``
    is harmless.  That predicate is the real one, and it is WIDER than the
    "signed row-permutation" phrasing this branch shipped on 2026-08-05.
    Measured (Frontera job 7890705, 63 nonzero q of a 4x4x4 grid, against a
    volume-validated rejection ground truth; ``nmax=3`` throughout so the
    fold width is not confounded with the fill):

        cell            M integer & |det|=1 ?   deleted ``bvec.T`` deviates
        cubic                   yes  (M = I)              3.9e-03
        fcc, 2pi inv(A).T       yes  (M != I)             5.0e-03
        hexagonal               no                        1.5e-01
        rhombohedral            no                        8.5e-02
        triclinic               no                        4.3e-02

    (MC self-noise on that ground truth: 2.3e-03 to 4.8e-03.)  So the fcc
    cell is in the benign class *because M is unimodular*, not because it is
    a row permutation — its ``M`` is ``[[0,0,1],[1,1,1],[-1,0,0]]``.

    The guard still demands the identity.  A sampler that fills a *different*
    fundamental domain than it folds against is a coincidence waiting to stop
    being one when someone changes a cell, and the strict check is the one
    that cannot itself be wrong.  The diagnostic below reports which class a
    violation is in, so a deliberate widening is a one-line decision rather
    than a re-derivation.
    """
    f = np.asarray(fill_basis, dtype=np.float64)
    w = np.asarray(wrap_basis, dtype=np.float64)
    if f.shape != (3, 3) or w.shape != (3, 3):
        raise MiniBZBasisMismatch(
            f"mini-BZ fill/wrap bases must both be (3,3); got fill{f.shape} "
            f"wrap{w.shape}")
    M = f @ np.linalg.inv(w)
    dev = float(np.max(np.abs(M - np.eye(3))))
    if dev <= _FILL_WRAP_TOL:
        return
    integral = float(np.max(np.abs(M - np.round(M)))) <= _FILL_WRAP_TOL
    unimodular = integral and abs(abs(float(np.linalg.det(M))) - 1.0) <= 1e-9
    verdict = ("a DIFFERENT fundamental domain of the same lattice (M is "
               "integer unimodular): measure-preserving, so unbiased, but "
               "not this module's convention"
               if unimodular else
               "NOT a fundamental domain of the wrap lattice (M is not "
               "integer unimodular): the Voronoi fold is NOT "
               "measure-preserving and the mini-BZ average IS BIASED")
    raise MiniBZBasisMismatch(
        f"mini-BZ cloud filled from one basis and folded against another.\n"
        f"  M = fill @ inv(wrap), should be I, max|M - I| = {dev:.3e}\n"
        f"  M =\n{M}\n"
        f"  the fill parallelepiped is {verdict}.\n"
        f"  The fill must span the ROWS of the same `bvec` the fold uses as "
        f"candidate shifts (`shifts @ bvec`, gw.vcoul.wrap_points_to_voronoi)."
        f"  `randvals @ bvec.T` is the 2026-04-05 defect; do not resurrect it.")


def probe_fill_basis(fill_fn):
    """The basis a fill EXPRESSION actually uses, recovered by evaluating it.

    ``fill_fn`` maps ``U (N,3)`` linearly to Cartesian points; feed it the
    identity and the rows of the answer ARE the parallelepiped edges.  One
    3x3 evaluation.

    Probing beats reading: ``(bvec.T @ U.T).T``, ``U @ bvec`` and
    ``U @ bvec.T`` are one transpose apart in a place no reviewer looks, and
    the comment next to them is not evidence.  Call sites that fill and fold
    must pass the SAME callable to this and to the fill itself, so the two
    cannot drift.
    """
    return np.asarray(fill_fn(np.eye(3, dtype=np.float64)), dtype=np.float64)


def _fill_and_wrap(U, bvec, nmax, *, fill_fn=None):
    """Fill the parallelepiped from ``U`` and fold it — against ONE basis.

    Every mini-BZ cloud in this module goes through here, so there is exactly
    one place where the fill basis and the wrap basis can disagree, and it
    refuses.  ``fill_fn`` is not a production knob: it exists so the guard
    has something to guard (the tests hand it ``U @ bvec.T`` and get a
    refusal).
    """
    if fill_fn is None:
        def fill_fn(X):
            return np.asarray(X, dtype=np.float64) @ np.asarray(
                bvec, dtype=np.float64)
    assert_fill_matches_wrap(probe_fill_basis(fill_fn), bvec)
    return _wrap(fill_fn(U), bvec, nmax)


def _selftest_fill_wrap_guard():
    """Run at import: prove the guard is not vacuous.  Two 3x3 inverses.

    A guard that never fires is indistinguishable from no guard, and this
    module exists because a silent bias survived from 2026-04-05 to
    2026-08-05.  The basis below is deliberately triclinic — on a symmetric
    one the two spellings are the same draw bitwise and the check would pass
    for the wrong reason.
    """
    b = np.array([[1.0, 0.00, 0.00],
                  [0.31, 0.95, 0.00],
                  [0.17, 0.23, 0.87]], dtype=np.float64)
    assert_fill_matches_wrap(b, b)              # the shipped spelling passes
    try:
        assert_fill_matches_wrap(b.T, b)        # the deleted spelling refuses
    except MiniBZBasisMismatch:
        return
    raise AssertionError(
        "gw.coulomb.sampler: the fill/wrap convention guard did not fire on "
        "`bvec.T` for a triclinic basis — it no longer guards anything.")


_selftest_fill_wrap_guard()


def _unit_draws_uniform(gidx, seed_offset):
    """U(0,1)^3 for a set of GLOBAL draw indices, rank-invariant.

    The randomness of a sample depends ONLY on its global index, via
    ``fold_in(PRNGKey(seed), i)`` — the sharded threefry idiom.  No rank
    ever reseeds, so which rank draws a given sample cannot change it.
    """
    base_key = jax.random.PRNGKey(int(seed_offset))

    @jax.jit
    def _draw(idx):
        def one(i):
            return jax.random.uniform(jax.random.fold_in(base_key, i),
                                      (3,), dtype=jnp.float64)
        return jax.vmap(one)(idx)

    return np.asarray(_draw(jnp.asarray(gidx, dtype=jnp.uint32)),
                      dtype=np.float64)


def _unit_draws_sobol(nsamples, qmc_reps, seed_offset):
    """Scrambled Sobol U(0,1)^3, ``qmc_reps`` replicate batches of 2^m.

    Returns ``(qmc_reps, 2**m, 3)`` or None when the engine is unavailable.
    The block is identical on every rank (it is a deterministic function of
    the seed), so rank slabs index into it without communicating.
    """
    qmc = _qmc_engine()
    if qmc is None:
        return None
    m = max(1, int(math.floor(math.log2(max(2, int(nsamples))))))
    out = []
    for rep in range(max(1, int(qmc_reps))):
        sob = qmc.Sobol(d=3, scramble=True, seed=rep + int(seed_offset))
        out.append(np.asarray(sob.random_base2(m), dtype=np.float64))
    return np.stack(out)


def minibz_offsets(bvec, kgrid, *, sys_dim, nsamples=2 ** 18, qmc_reps=10,
                   method="sobol", nmax=3, seed_offset=0):
    """The mini-BZ offset block ``(reps, N, 3)`` used by every average.

    Draw + Voronoi-wrap + mini-BZ affine map, done ONCE.  A table over a
    whole q-grid (:func:`gw.compute_vcoul.build_v_head_minibz_table_3d`)
    reuses one block for every q and only varies how many of its samples
    the adaptive rule keeps — which is exactly the semantics of
    ``base.minibz_average``'s ``v[:n_q]`` slice, and orders of magnitude
    cheaper than re-wrapping per q.
    """
    reps = max(1, int(qmc_reps))
    sob = (_unit_draws_sobol(nsamples, reps, seed_offset)
           if str(method).lower() == "sobol" else None)
    if sob is None:
        n = int(nsamples)
        gidx = (np.arange(reps, dtype=np.int64)[:, None] * n
                + np.arange(n, dtype=np.int64)[None, :])
        sob = _unit_draws_uniform(gidx.reshape(-1), seed_offset
                                  ).reshape(reps, n, 3)
    bvec = np.asarray(bvec, dtype=np.float64)
    # fill (ROWS of bvec) + Voronoi fold, guarded: see _fill_and_wrap.
    wrapped = _fill_and_wrap(sob.reshape(-1, 3), bvec, nmax)
    dq = (_randlims(bvec, kgrid) @ wrapped.T).T
    if int(sys_dim) == 2:
        dq[:, 2] = 0.0
    return dq.reshape(sob.shape)


def minibz_cell_average(
    shift_cart, *,
    bvec, kgrid, sys_dim, units, celvol,
    channel: str = "full",
    n_kpts: int | None = None,
    alpha: float | None = None,
    zc: float | None = None,
    nsamples: int = 2 ** 18,
    qmc_reps: int = 10,
    method: str = "sobol",
    nmax: int = 3,
    seed_offset: int = 0,
    analytic_sphere: bool = False,
    adaptive: bool = True,
    n_coarse: int = 250_000,
    distribute: bool = True,
    dq_batches=None,
) -> float:
    """``<v(shift + dq)>`` over the mini-BZ Voronoi cell.  Rank-invariant.

    Parameters
    ----------
    shift_cart : (3,) float
        Cartesian ``Q + G*`` the cell is centred on.  For the GW body head
        this is ``q + G*``; for the q->0 head it is 0.
    bvec, kgrid : (3,3) float, 3-tuple int
        Cartesian reciprocal vectors (ROWS are the ``b_i``, 1/bohr, blat
        included) and the q-grid.  ``bvec`` rows, not columns: sampling the
        parallelepiped spanned by the COLUMNS is the 2026-04-05 bug this
        module exists to end.
    sys_dim, channel, units, celvol, alpha, zc
        Passed to :func:`gw.coulomb.kernel.v_qG`.  ``units`` is required.
        ``sys_dim == 2`` zeroes the ``dq_z`` component (in-plane cell).
    n_kpts : int, optional
        Required when ``analytic_sphere`` — the Baldereschi-Tosatti term
        scales with the number of q-points.
    method : {'sobol', 'uniform'}
        ``sobol`` = scrambled Sobol (low-discrepancy, ~N^-1); ``uniform`` =
        per-sample ``fold_in`` draws (N^-1/2, but needs no scipy and is the
        BSE head's historical path).  A missing ``scipy.stats.qmc`` demotes
        ``sobol`` to ``uniform`` with an ANNOUNCEMENT, never silently.
    nmax : int
        Voronoi-fold replica half-width (BGW ``ncell=3``).  ``nmax=1`` is
        too narrow for skewed cells; 3 is the default here.
    analytic_sphere : bool
        At ``|shift| = 0`` in 3D, split off the ``1/q^2`` pole analytically
        (``4 sqrt(q0sph2) celvol N_k / pi``, ``minibzaverage.f90:79-81``) and
        MC only the smooth remainder outside the inscribed sphere.  Makes
        the q->0 head seed-independent.  Ignored at finite shift and in 2D
        (the slab head is a ``|Q|`` cusp, integrable, no split needed).
    adaptive : bool
        BGW's per-shift sample count ``N = clamp(round(n_coarse * 4 *
        q0sph2 / |shift|^2), 1, nsamples)`` (``minibzaverage.f90:63-75``):
        far from the pole the integrand is nearly constant and a handful of
        samples is already converged.
    distribute : bool
        Split the kept samples into disjoint contiguous rank slabs and
        all-reduce the partial sums.  DEFAULT TRUE, and that is safe
        precisely because the estimator is rank-count invariant: a sample
        is a pure function of its global slot index, so the answer does not
        depend on who drew it (bit-equal up to float summation reorder,
        ~1e-13).

    Returns
    -------
    float
        The cell average, in ``units``.
    """
    if units not in ("bare", "per_volume"):
        raise ValueError(f"minibz_cell_average: units must be 'bare' or "
                         f"'per_volume'; got {units!r}")
    bvec = np.asarray(bvec, dtype=np.float64)
    kgrid = tuple(int(s) for s in kgrid)
    shift_cart = np.asarray(shift_cart, dtype=np.float64).reshape(3)
    is_2d = (int(sys_dim) == 2)
    len_shift2 = float(shift_cart @ shift_cart)
    q0sph2 = minibz_inscribed_sphere_r2(bvec, kgrid, is_2d=is_2d)
    if analytic_sphere and is_2d:
        analytic_sphere = False
    head_branch = analytic_sphere and (len_shift2 < TOL_QG_ZERO)
    if head_branch and n_kpts is None:
        raise ValueError("minibz_cell_average: analytic_sphere at |shift|=0 "
                         "needs n_kpts for the Baldereschi-Tosatti term")

    reps = max(1, int(qmc_reps))
    if dq_batches is not None:
        # pre-drawn offsets (minibz_offsets) — a q-grid table draws once
        dq_all = np.asarray(dq_batches, dtype=np.float64)
        reps, n_avail = int(dq_all.shape[0]), int(dq_all.shape[1])
        sob = None
    else:
        dq_all = None
        sob = (_unit_draws_sobol(nsamples, reps, seed_offset)
               if str(method).lower() == "sobol" else None)
        # Sobol emits 2**m <= nsamples points per replicate; the kept count
        # can never exceed what was drawn.
        n_avail = int(nsamples) if sob is None else int(sob.shape[1])

    # per-replicate kept-sample count (BGW adaptive rule)
    if head_branch or not adaptive or len_shift2 <= TOL_QG_ZERO:
        n_q = n_avail
    else:
        n_q = int(round(n_coarse * 4.0 * q0sph2 / len_shift2))
        n_q = max(1, min(n_q, n_avail))

    n_kept = reps * n_q

    if distribute:
        rank = int(jax.process_index())
        nranks = int(jax.process_count())
    else:
        rank, nranks = 0, 1
    lo = rank * n_kept // nranks
    hi = (rank + 1) * n_kept // nranks

    if hi > lo:
        slots = np.arange(lo, hi, dtype=np.int64)
        rep = slots // n_q                    # which replicate batch
        loc = slots % n_q                     # position inside it
        if dq_all is not None:
            dq = dq_all[rep, loc]
        else:
            if sob is not None:
                U = sob[rep, loc]
            else:
                # global draw index: the SAME sample for a given (rep, loc)
                # regardless of how the slabs are cut.
                U = _unit_draws_uniform(rep * np.int64(int(nsamples)) + loc,
                                        seed_offset)
            # fill (ROWS of bvec) + fold, guarded: see _fill_and_wrap.
            wrapped = _fill_and_wrap(U, bvec, nmax)
            dq = (_randlims(bvec, kgrid) @ wrapped.T).T
            if is_2d:
                dq[:, 2] = 0.0
        K = shift_cart[None, :] + dq
        v = v_qG(K, axis=1, sys_dim=int(sys_dim), channel=channel,
                 units="bare", celvol=celvol, alpha=alpha, zc=zc,
                 zero_tol=TOL_MC_NAN)
        if head_branch:
            # MC only OUTSIDE the inscribed sphere; the denominator stays
            # the FULL kept count (minibzaverage.f90:79-81).
            len2 = np.sum(K * K, axis=1)
            v = np.where(len2 > q0sph2, v, 0.0)
        local_sum = float(np.sum(v))
        local_cnt = float(v.shape[0])
    else:
        local_sum = local_cnt = 0.0

    if nranks > 1:
        from jax.experimental import multihost_utils
        g = np.asarray(multihost_utils.process_allgather(
            np.asarray([local_sum, local_cnt], dtype=np.float64), tiled=False))
        tot_sum, tot_cnt = float(g[:, 0].sum()), float(g[:, 1].sum())
    else:
        tot_sum, tot_cnt = local_sum, local_cnt

    avg = tot_sum / tot_cnt
    if head_branch:
        avg = avg + 4.0 * np.sqrt(q0sph2) * float(celvol) * float(n_kpts) \
            / np.pi
    if units == "per_volume":
        avg = avg / float(celvol)
    return float(avg)


__all__ = ["minibz_cell_average", "minibz_inscribed_sphere_r2",
           "minibz_offsets", "assert_fill_matches_wrap", "probe_fill_basis",
           "MiniBZBasisMismatch", "TOL_QG_ZERO", "TOL_MC_NAN"]
