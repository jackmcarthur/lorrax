"""Committed guard: the mini-BZ sampler is measure-preserving.

`gw/compute_vcoul.py::build_v_head_miniBZ_avg_3d` drew mini-BZ samples as
``randvals @ bvec.T`` from 2026-04-05 to 2026-08-05.  The lattice period is
the ROWS of ``bvec`` (`q_cart = q_frac @ bvec` everywhere else in the tree);
its COLUMNS need not be one, so the parallelepiped being sampled need not be
a fundamental domain, the Voronoi wrap that follows is then not
measure-preserving, and the Monte-Carlo mini-BZ average is biased.  How
biased depends on the cell, and that dependence is the whole story below.

Why it survived four months: the estimator is UNBIASED whenever ``bvec.T`` is
a signed row-permutation of ``bvec`` — then the two draws are different points
from the same distribution on the same parallelepiped.  A symmetric ``bvec``
is the trivial case of that, and so, as it happens, is the fcc cell pw2bgw
wrote into ``tests/regression/si_cohsex_debug`` (permutation (2,0,1),
measured).  So every cell the tree actually ran was in the benign class.
``allclose(bvec, bvec.T)`` is NOT the predicate; the row-permutation one is.
A 3D HEXAGONAL or TRICLINIC deck would have been wrong, and nothing in the
suite would have said so.

This module does not compare the sampler against another copy of itself.  It
compares it against a REJECTION sampler — draw uniformly in a box containing
the mini-BZ, keep the points whose nearest mini-BZ lattice site is the origin
— which shares no code with the production path and is uniform on the Voronoi
cell by construction.  That is the ground truth.

Everything is swept over the 63 nonzero q of a 4x4x4 grid, not one shift: the
bias is strongly q-dependent (0.2% at some q, 65% at others), so a single-q
check can pass while the table it stands for is badly wrong.

MEASURED, Frontera jobs 7890645 and 7890650, max / mean relative deviation
from the rejection ground truth over those 63 q:

    cell         row-perm?  estimator                    max        mean
    fcc(2pi/A)   no         rejection, other seed      2.19e-03   5.99e-04
    fcc(2pi/A)   no         DELETED `bvec.T`           6.43e-01   8.43e-02
    fcc(2pi/A)   no         the one-line fix           3.69e-03   1.27e-03
    fcc(2pi/A)   no         shipped cell average       3.15e-03   8.40e-04
    hexagonal    no         DELETED `bvec.T`           1.66e-01   6.53e-02
    triclinic    no         DELETED `bvec.T`           2.28e-02   7.44e-03
    Si fixture   YES        DELETED `bvec.T`           3.70e-03   1.03e-03
    cubic        YES        DELETED `bvec.T`           3.00e-03   6.57e-04

CORRECTION, 2026-08-05 (Frontera job 7890705).  Two claims above are wrong
and are kept only because the code below still depends on them; the
replacement measurement lives in
``tests/test_minibz_sampler_lattice_classes.py``.

1. The predicate is UNIMODULARITY, not "signed row-permutation".  The
   columns of ``bvec`` span a fundamental domain of its row lattice iff
   ``M = bvec.T @ inv(bvec)`` is an integer matrix with ``|det M| = 1``.
   The ``BVEC_FCC`` built below satisfies that with
   ``M = [[0,0,1],[1,1,1],[-1,0,0]]`` — integer, unimodular, and NOT a
   permutation.  So this fcc cell is in the BENIGN class, contrary to the
   table above.
2. The 6.43e-01 attributed to ``bvec.T`` on that cell is not attributable to
   ``bvec.T``.  ``deleted_buggy_offsets`` reproduces the deleted routine
   with BOTH of its defects — the wrong parallelepiped and ``nmax=1``, a
   fold one replica shell wide.  Re-measured with ``nmax=3`` so the fill is
   isolated, the deleted spelling deviates 5.0e-03 on this cell, at the
   3.4e-03 self-noise: unbiased.  The 64% was the narrow fold.
   ``test_the_deleted_spelling_is_biased_on_a_skewed_cell`` below therefore
   passes for a reason other than the one its name gives.  It is left
   standing (it is a true statement about the deleted ROUTINE) and the
   ``bvec.T`` claim is gated properly on hexagonal / rhombohedral /
   triclinic cells in the sibling module, where the fill alone deviates
   1.5e-01 / 8.5e-02 / 6.5e-02 at ``nmax=3``.

Pure numpy + the production sampler; no fixture, no GPU (~a minute).
"""
import numpy as np
import pytest

from gw.coulomb.sampler import minibz_cell_average

BVEC_CUBIC = (2.0 * np.pi / 10.26) * np.eye(3)


def _bvec_fcc(a=10.26):
    """fcc from ``2pi inv(A).T``.  NOT the fcc `bvec` in the Si fixture — that
    one is a signed row-permutation of its own transpose and therefore in the
    benign class.  This construction is not, which is what makes it a useful
    adversary here."""
    avec = 0.5 * a * np.array([[-1.0, 0.0, 1.0],
                               [0.0, 1.0, 1.0],
                               [-1.0, 1.0, 0.0]])
    return 2.0 * np.pi * np.linalg.inv(avec).T


BVEC_FCC = _bvec_fcc()
KGRID = (4, 4, 4)
NQ = int(np.prod(KGRID))

# Ceilings from the table above, with headroom over the MC self-noise.
REL_TO_GROUND_TRUTH = 1.0e-2      # shipped sampler; measured 3.2e-3
REL_BUG_IS_VISIBLE = 1.0e-1       # deleted spelling on fcc; measured 6.5e-1


def _celvol(bvec):
    return (2.0 * np.pi) ** 3 / abs(np.linalg.det(bvec))


# --- ground truth ----------------------------------------------------------
def rejection_minibz_offsets(bvec, kgrid, n, seed=0):
    """Uniform on the mini-BZ Voronoi cell, by rejection.  Shares no code
    with the production sampler.

    The mini-BZ reciprocal lattice is ``b_i / nk_i``.  Draw uniformly in a
    box comfortably containing its Voronoi cell and keep a point iff the
    nearest lattice site is the origin — that IS the Voronoi cell, and
    uniform-in-a-box conditioned on a subset is uniform on the subset.
    """
    B = np.asarray(bvec, dtype=np.float64) / np.asarray(kgrid, float)[:, None]
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1)
                       for k in (-1, 0, 1)], dtype=np.float64) @ B
    half = np.abs(B).sum(axis=0)                      # bounds the cell
    rng = np.random.RandomState(seed)
    out, got = [], 0
    while got < n:
        p = rng.uniform(-half, half, (4 * n, 3))
        d = np.linalg.norm(p[:, None, :] - shifts[None, :, :], axis=2)
        keep = p[np.argmin(d, axis=1) == 13]          # index of shift (0,0,0)
        out.append(keep)
        got += keep.shape[0]
    return np.concatenate(out)[:n]


def head_from_offsets(shift_cart, dq, celvol):
    """<8pi/|q+dq|^2>/celvol on an explicit offset cloud."""
    K = np.asarray(shift_cart, float)[None, :] + dq
    return float(np.mean(8.0 * np.pi / np.sum(K * K, axis=1))) / celvol


# --- the DELETED spelling, frozen verbatim so the guard can have teeth -----
def deleted_buggy_offsets(bvec, kgrid, nmc=2 ** 16, seed=42):
    """`build_v_head_miniBZ_avg_3d` lines 55-63, verbatim (deleted 2026-08-05).

    Kept ONLY so this test can demonstrate the bias it caused.  Do not
    resurrect: `randvals @ bvec.T` is the defect.
    """
    import jax.numpy as jnp
    from gw.vcoul import wrap_points_to_voronoi
    bvec = np.asarray(bvec, dtype=np.float64)
    rng = np.random.RandomState(seed)
    randvals = rng.uniform(0, 1, (nmc, 3))
    randcart = (randvals @ bvec.T)                    # <-- THE BUG
    wrapped = np.asarray(wrap_points_to_voronoi(
        jnp.asarray(randcart), jnp.asarray(bvec), nmax=1))
    kg = np.asarray(kgrid, dtype=np.float64)
    randlims = bvec.T @ (np.diag(1.0 / kg) @ np.linalg.inv(bvec.T))
    return (randlims @ wrapped.T).T


def q_shifts(bvec, kgrid):
    """The 63 nonzero q of the grid, BGW-wrapped, Cartesian — the exact set
    the body-head table is built on."""
    kg = np.asarray(kgrid, dtype=np.float64)
    out = []
    for i in range(int(kg[0])):
        for j in range(int(kg[1])):
            for k in range(int(kg[2])):
                qw = np.array([i, j, k], dtype=np.float64)
                qw = np.where(qw > kg / 2, qw - kg, qw)
                qc = (qw / kg) @ np.asarray(bvec, dtype=np.float64)
                if float(qc @ qc) > 1e-12:
                    out.append(qc)
    return out


@pytest.mark.parametrize("bvec,name", [(BVEC_FCC, "fcc"),
                                       (BVEC_CUBIC, "cubic")])
def test_shipped_sampler_matches_the_rejection_ground_truth(bvec, name):
    """The shipped sampler reproduces a uniform-on-Voronoi draw at every q."""
    cv = _celvol(bvec)
    dq_ref = rejection_minibz_offsets(bvec, KGRID, 200_000, seed=0)
    worst = 0.0
    for qc in q_shifts(bvec, KGRID):
        ref = head_from_offsets(qc, dq_ref, cv)
        got = minibz_cell_average(
            qc, bvec=bvec, kgrid=KGRID, sys_dim=3, channel="full",
            units="per_volume", celvol=cv, n_kpts=NQ, nsamples=2 ** 16,
            qmc_reps=4, nmax=3, adaptive=False, distribute=False)
        worst = max(worst, abs(got - ref) / abs(ref))
    assert worst <= REL_TO_GROUND_TRUTH, (
        f"{name}: shipped mini-BZ average deviates {worst:.3%} from a "
        f"uniform-on-Voronoi ground truth — the draw is not measure-"
        f"preserving")


def test_the_deleted_spelling_is_biased_on_a_skewed_cell():
    """The bug was real and this is its size.

    A regression guard with teeth: if `randvals @ bvec.T` ever comes back,
    the fcc case of the test above goes red.  This one proves that guard can
    fire, by showing the wrong spelling misses the ground truth by >10% at
    the worst q on the grid (measured 65%).  Note it is only 0.2% at the
    BEST q — checking one shift is how this bug hides.
    """
    cv = _celvol(BVEC_FCC)
    dq_ref = rejection_minibz_offsets(BVEC_FCC, KGRID, 200_000, seed=0)
    dq_bad = deleted_buggy_offsets(BVEC_FCC, KGRID)
    worst = 0.0
    for qc in q_shifts(BVEC_FCC, KGRID):
        ref = head_from_offsets(qc, dq_ref, cv)
        worst = max(worst, abs(head_from_offsets(qc, dq_bad, cv) - ref)
                    / abs(ref))
    assert worst > REL_BUG_IS_VISIBLE, (
        f"the deleted `randvals @ bvec.T` spelling came within {worst:.3%} of "
        f"the ground truth at every q on an fcc cell — if that is now true, "
        f"this guard no longer distinguishes the bug and needs a sharper cell")


def test_the_bug_was_invisible_on_a_symmetric_bvec():
    """WHY it survived from 2026-04-05 to 2026-08-05.

    On a symmetric `bvec` the rows and the columns are the same vectors, so
    `randvals @ bvec.T` and `randvals @ bvec` are the SAME DRAW — bitwise.
    A unit test built on a cubic cell is blind to this class of bug; the fcc
    case above is the one that has to exist.
    """
    assert np.allclose(BVEC_CUBIC, BVEC_CUBIC.T)
    assert not np.allclose(BVEC_FCC, BVEC_FCC.T)
    rng = np.random.RandomState(3)
    u = rng.uniform(0, 1, (1000, 3))
    assert np.array_equal(u @ BVEC_CUBIC.T, u @ BVEC_CUBIC)
    assert not np.array_equal(u @ BVEC_FCC.T, u @ BVEC_FCC)


if __name__ == "__main__":
    test_shipped_sampler_matches_the_rejection_ground_truth(BVEC_FCC, "fcc")
    test_the_deleted_spelling_is_biased_on_a_skewed_cell()
    test_the_bug_was_invisible_on_a_symmetric_bvec()
    print("[minibz-bias] guards pass")
