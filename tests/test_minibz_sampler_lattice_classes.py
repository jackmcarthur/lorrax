"""The mini-BZ sampler is unbiased on cells that are NOT a fundamental
domain of themselves under transposition.

Companion to ``tests/test_minibz_fill_wrap_convention.py`` (which asserts the
defect's class structurally) and to ``tests/test_minibz_sampler_bias.py``
(which covers the fcc/cubic pair).  This module is the numerical half, on the
cells where the 2026-04-05 defect actually bit.

WHAT THE DEFECT WAS.  ``gw.compute_vcoul.build_v_head_miniBZ_avg_3d`` filled
its mini-BZ cloud with ``randvals @ bvec.T`` — the parallelepiped spanned by
the COLUMNS of ``bvec`` — and then folded it with
``gw.vcoul.wrap_points_to_voronoi``, whose candidate lattice is
``shifts @ bvec``, the ROWS.  A fold is measure-preserving only on a
fundamental domain of the lattice it folds against, so the Monte-Carlo
mini-BZ average was biased.

WHICH CELLS.  Write ``M = bvec.T @ inv(bvec)``.  The columns span a
fundamental domain of the row lattice iff ``M`` is an INTEGER matrix with
``|det M| = 1``.  That — unimodularity, not "signed row permutation" — is
the predicate.  Cubic cells have ``M = I``; the ``2*pi*inv(A).T`` fcc cell
has ``M = [[0,0,1],[1,1,1],[-1,0,0]]``, integer and unimodular but NOT a
permutation.  Both are benign.  Hexagonal, rhombohedral and triclinic cells
are not, and none of them existed in any deck in the tree, which is exactly
why the bug survived from 2026-04-05 to 2026-08-05.

WHAT THIS COMPARES AGAINST.  Not another copy of the production path — that
proves only that the code equals itself.  The reference is a REJECTION
sampler: draw uniformly in a Cartesian box that contains the mini-BZ Voronoi
cell, keep the points whose nearest mini-BZ lattice site is the origin.
Uniform-in-a-box conditioned on a subset is uniform on the subset, and no
line of it is shared with the production draw/fold/map.

That reference VALIDATES ITSELF, which matters more than it sounds: an
earlier draft of this measurement used a box of half-width
``0.5 * sum_i |b_i|`` per axis, which does NOT contain the Voronoi cell —
it clipped 4-10% of it on the skewed cells and manufactured an apparent
3e-02 "residual bias" in the *fixed* sampler (Frontera job 7890705).  The
acceptance rate times the box volume must equal ``|det B_miniBZ|``; that
identity catches a clipped box (ratio < 1) and a too-narrow candidate-shift
window (ratio > 1) in one number, and it is asserted below before any head
is compared.

TOLERANCE.  Not guessed — measured, every run, from a SECOND independent
rejection cloud.  MEASURED (Frontera job 7890705, 63 nonzero q of a 4x4x4
grid, 150k accepted points per cloud, production draw 4 x 2^16 scrambled
Sobol at ``nmax=3``):

    cell            M integer?   self-noise   shipped     deleted `bvec.T`
    cubic (control)   yes (I)     3.0e-03     2.9e-03        3.9e-03
    hexagonal         no          2.3e-03     3.1e-03        1.5e-01
    rhombohedral      no          2.5e-03     3.1e-03        8.5e-02
    triclinic         no          4.4e-03     3.8e-03        6.5e-02

(max over the 63 q, relative.)  The shipped sampler sits ON the self-noise
on every cell; the deleted spelling is 15-60x above it on the three that are
not fundamental domains, and at the self-noise on the control.

COST.  ~40 s, pure numpy plus one production draw; no fixture, no
wavefunctions, no pseudopotentials, no GPU.  It belongs in the normal suite.
"""
import numpy as np
import pytest

from gw.coulomb.sampler import minibz_cell_average, minibz_offsets

KGRID = (4, 4, 4)
NQ = int(np.prod(KGRID))

#: Accepted rejection samples per cloud.  Sets the self-noise (~3e-03 max
#: over 63 q at this N); two clouds are drawn per cell, so 4x this is the
#: whole numerical cost.
N_REJECT = 150_000

#: Production draw.  Same shape as tests/test_minibz_sampler_bias.py uses, so
#: the XLA module for the Voronoi fold is compiled once per pytest session
#: rather than once per file.
N_SOBOL, QMC_REPS, NMAX = 2 ** 16, 4, 3

#: Rejection box: ``BOX_SCALE * 0.5 * sum_i |b_i|`` per axis.  1.0 CLIPS the
#: Voronoi cell (measured 0.90-0.96 of its volume on skewed cells); 1.5
#: contains it with the volume identity satisfied to <0.3%.  The identity is
#: asserted, so this constant is checked rather than trusted.
BOX_SCALE = 1.5
#: Candidate-shift half-width for the reference's nearest-site test.  +-1
#: already suffices at BOX_SCALE=1.5 (measured identical to +-3); +-2 is
#: margin, and the volume identity would catch it being too small.
REJ_WINDOW = 2
#: Ground truth is trustworthy only if acceptance * box volume == cell volume.
VOL_IDENTITY_TOL = 0.02

#: Floor on the pass tolerance.  ~2x the largest shipped-sampler deviation
#: measured (4.6e-03) and ~2x the largest self-noise (4.8e-03).
TOL_FLOOR = 8.0e-3
#: The tolerance may float up with the measured self-noise, but not without
#: limit — a tolerance that tracks its own noise can neuter itself.
TOL_SELF_NOISE_FACTOR = 3.0
TOL_CEILING = 2.5 * TOL_FLOOR
#: The deleted spelling must clear the floor by this much, or this module no
#: longer distinguishes the bug it exists for.  Measured ratios: 8.1 (hex),
#: 4.7 (rhombohedral), 3.6 (triclinic) at 4x -> passes with >=2x to spare.
BUG_VISIBILITY_FACTOR = 4.0


# --------------------------------------------------------------------------
# cells
# --------------------------------------------------------------------------
def avec_from_lengths_angles(a, b, c, alpha, beta, gamma):
    """Rows = a1,a2,a3, from lengths (bohr) and angles (degrees)."""
    al, be, ga = np.radians([alpha, beta, gamma])
    a1 = np.array([a, 0.0, 0.0])
    a2 = np.array([b * np.cos(ga), b * np.sin(ga), 0.0])
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    a3 = np.array([cx, cy, np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))])
    return np.stack([a1, a2, a3])


def bvec_of(avec):
    """``2*pi*inv(A).T`` — rows are the b_i, the convention everywhere in the
    tree (``q_cart = q_frac @ bvec``)."""
    return 2.0 * np.pi * np.linalg.inv(np.asarray(avec, np.float64)).T


BVEC = {
    "cubic":        bvec_of(10.26 * np.eye(3)),
    "hexagonal":    bvec_of(avec_from_lengths_angles(5.9, 5.9, 9.4,
                                                     90, 90, 120)),
    "rhombohedral": bvec_of(avec_from_lengths_angles(6.4, 6.4, 6.4,
                                                     68, 68, 68)),
    "triclinic":    bvec_of(avec_from_lengths_angles(4.6, 6.1, 8.3,
                                                     62, 78, 104)),
}
#: cells whose COLUMNS do not span a fundamental domain of their ROW lattice
ADVERSARIAL = ["hexagonal", "rhombohedral", "triclinic"]


def transpose_is_a_fundamental_domain(bvec, tol=1e-9):
    """Is ``M = bvec.T @ inv(bvec)`` integer with ``|det M| = 1``?

    True <=> the parallelepiped spanned by the COLUMNS of ``bvec`` is a
    fundamental domain of the lattice spanned by its ROWS <=> the deleted
    ``randvals @ bvec.T`` draw was measure-preserving on this cell after all.
    """
    M = np.asarray(bvec, np.float64).T @ np.linalg.inv(
        np.asarray(bvec, np.float64))
    return (float(np.max(np.abs(M - np.round(M)))) <= tol
            and abs(abs(float(np.linalg.det(M))) - 1.0) <= tol)


# --------------------------------------------------------------------------
# the independent reference
# --------------------------------------------------------------------------
def rejection_minibz_offsets(bvec, kgrid, n, seed):
    """Uniform on the mini-BZ Voronoi cell, by rejection.  Self-validating.

    Returns ``(offsets, volume_ratio)`` where ``volume_ratio`` is
    ``acceptance * box_volume / |det B_miniBZ|`` and must be 1: below 1 the
    box clipped the cell, above 1 the candidate-shift window was too narrow
    and points outside the cell were accepted.  Shares no code with
    ``gw.coulomb.sampler``.
    """
    B = np.asarray(bvec, np.float64) / np.asarray(kgrid, np.float64)[:, None]
    r = range(-REJ_WINDOW, REJ_WINDOW + 1)
    shifts = np.array([[i, j, k] for i in r for j in r for k in r],
                      np.float64) @ B
    origin = (shifts.shape[0] - 1) // 2
    half = BOX_SCALE * 0.5 * np.abs(B).sum(axis=0)
    rng = np.random.RandomState(seed)
    kept, got, tried = [], 0, 0
    while got < n:
        p = rng.uniform(-half, half, (max(n, 50_000), 3))
        tried += p.shape[0]
        d = p[:, None, :] - shifts[None, :, :]
        keep = p[np.argmin(np.einsum("nmi,nmi->nm", d, d), axis=1) == origin]
        kept.append(keep)
        got += keep.shape[0]
    vol_ratio = ((got / float(tried)) * float(np.prod(2.0 * half))
                 / abs(float(np.linalg.det(B))))
    return np.concatenate(kept)[:n], vol_ratio


def head_from_offsets(shift_cart, dq, celvol):
    """``<8pi/|q+dq|^2>/Omega`` on an explicit offset cloud."""
    K = np.asarray(dq, np.float64) + np.asarray(shift_cart, np.float64)[None]
    return float(np.mean(8.0 * np.pi / np.sum(K * K, axis=1))) / celvol


def deleted_bvec_T_offsets(bvec, kgrid, nmc=2 ** 16, seed=42):
    """The deleted ``randvals @ bvec.T`` fill, folded at ``nmax=3``.

    Kept only so this module can show the bias it caused.  ``nmax=3`` (not
    the historical ``nmax=1``) is deliberate: it isolates the WRONG
    PARALLELEPIPED from the too-narrow fold, which was a separate defect
    of the same deleted routine.  Everything reported here is therefore
    attributable to ``bvec.T`` alone.  Do not resurrect it.
    """
    import jax.numpy as jnp
    from gw.vcoul import wrap_points_to_voronoi
    bvec = np.asarray(bvec, np.float64)
    rng = np.random.RandomState(seed)
    randcart = rng.uniform(0, 1, (nmc, 3)) @ bvec.T          # <-- THE DEFECT
    wrapped = np.asarray(wrap_points_to_voronoi(
        jnp.asarray(randcart), jnp.asarray(bvec), nmax=3))
    kg = np.asarray(kgrid, np.float64)
    randlims = bvec.T @ (np.diag(1.0 / kg) @ np.linalg.inv(bvec.T))
    return (randlims @ wrapped.T).T


def q_shifts(bvec, kgrid):
    """The 63 nonzero q of the grid, BGW-wrapped, Cartesian — the exact set
    the 3D body-head table is built on.  The bias is strongly q-dependent
    (near zero at some q, >10% at others), so one shift proves nothing."""
    kg = np.asarray(kgrid, np.float64)
    out = []
    for i in range(int(kg[0])):
        for j in range(int(kg[1])):
            for k in range(int(kg[2])):
                qw = np.array([i, j, k], np.float64)
                qw = np.where(qw > kg / 2, qw - kg, qw)
                qc = (qw / kg) @ np.asarray(bvec, np.float64)
                if float(qc @ qc) > 1e-12:
                    out.append(qc)
    return out


def celvol_of(bvec):
    return (2.0 * np.pi) ** 3 / abs(float(np.linalg.det(bvec)))


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(BVEC))
def test_cells_are_in_the_class_this_module_claims(name):
    """The cell set covers both classes, and says which is which.

    If a future edit swaps in a cell whose transpose IS a fundamental domain,
    the numerical test below would pass for the wrong reason — silently, the
    same way the whole deck inventory did for four months.
    """
    benign = transpose_is_a_fundamental_domain(BVEC[name])
    assert benign == (name not in ADVERSARIAL), (
        f"{name}: `bvec.T` spans a fundamental domain of the row lattice = "
        f"{benign}, but this module files it under "
        f"{'adversarial' if name in ADVERSARIAL else 'benign'}")


@pytest.mark.parametrize("name", list(BVEC))
def test_shipped_sampler_matches_an_independent_estimator(name):
    """The shipped mini-BZ average reproduces a uniform-on-Voronoi draw at
    every q, to the Monte-Carlo self-noise measured in the same run.

    ``cubic`` is the negative control: its ``M`` is the identity, the
    deleted spelling was the same draw bitwise, and it passes trivially.
    That it passes is not evidence about the fix; the other three are.
    """
    bvec = BVEC[name]
    cv = celvol_of(bvec)
    qs = q_shifts(bvec, KGRID)
    assert len(qs) == NQ - 1

    ref, vr_ref = rejection_minibz_offsets(bvec, KGRID, N_REJECT, seed=0)
    alt, vr_alt = rejection_minibz_offsets(bvec, KGRID, N_REJECT, seed=1)
    for tag, vr in (("seed0", vr_ref), ("seed1", vr_alt)):
        assert abs(vr - 1.0) <= VOL_IDENTITY_TOL, (
            f"{name}/{tag}: the rejection reference is not uniform on the "
            f"mini-BZ Voronoi cell — acceptance x box volume / |det B| = "
            f"{vr:.4f}, should be 1.  <1 means BOX_SCALE={BOX_SCALE} clips "
            f"the cell; >1 means REJ_WINDOW={REJ_WINDOW} is too narrow and "
            f"points outside it were accepted.  Fix the reference before "
            f"reading anything below it.")

    dq = minibz_offsets(bvec, KGRID, sys_dim=3, nsamples=N_SOBOL,
                        qmc_reps=QMC_REPS, nmax=NMAX, seed_offset=0)
    dq_flat = dq.reshape(-1, 3)
    bad = deleted_bvec_T_offsets(bvec, KGRID)

    self_noise = prod = prod_avg = bug = 0.0
    for qc in qs:
        r = head_from_offsets(qc, ref, cv)
        self_noise = max(self_noise, abs(head_from_offsets(qc, alt, cv) - r)
                         / abs(r))
        prod = max(prod, abs(head_from_offsets(qc, dq_flat, cv) - r) / abs(r))
        bug = max(bug, abs(head_from_offsets(qc, bad, cv) - r) / abs(r))
        got = minibz_cell_average(
            qc, bvec=bvec, kgrid=KGRID, sys_dim=3, channel="full",
            units="per_volume", celvol=cv, n_kpts=NQ, adaptive=False,
            distribute=False, dq_batches=dq)
        prod_avg = max(prod_avg, abs(got - r) / abs(r))

    tol = max(TOL_FLOOR, TOL_SELF_NOISE_FACTOR * self_noise)
    assert tol <= TOL_CEILING, (
        f"{name}: measured MC self-noise {self_noise:.3e} would open the "
        f"tolerance to {tol:.3e}, past the {TOL_CEILING:.3e} ceiling — the "
        f"reference clouds are too noisy to gate anything.  Raise N_REJECT.")

    assert prod <= tol, (
        f"{name}: the mini-BZ offset cloud deviates {prod:.3e} from a "
        f"uniform-on-Voronoi ground truth at the worst of {len(qs)} q "
        f"(tolerance {tol:.3e}, self-noise {self_noise:.3e}).  The draw is "
        f"not measure-preserving on this cell.")
    assert prod_avg <= tol, (
        f"{name}: minibz_cell_average deviates {prod_avg:.3e} from the "
        f"ground truth (tolerance {tol:.3e}) even though the raw offset "
        f"cloud is within {prod:.3e} — the bias is downstream of the draw.")

    if name in ADVERSARIAL:
        assert bug > BUG_VISIBILITY_FACTOR * TOL_FLOOR, (
            f"{name}: the deleted `randvals @ bvec.T` fill came within "
            f"{bug:.3e} of the ground truth at every q, under the "
            f"{BUG_VISIBILITY_FACTOR * TOL_FLOOR:.3e} this module needs to "
            f"tell the bug from the fix.  Either the reference stopped being "
            f"the reference or this cell is no longer adversarial.")
    else:
        assert bug <= tol, (
            f"{name} is the negative control: `bvec.T` spans a fundamental "
            f"domain here, so the deleted spelling must be unbiased too, but "
            f"it deviates {bug:.3e} > {tol:.3e}")


def test_minibz_cell_average_draws_its_own_cloud_correctly():
    """The internal draw path, not just the pre-drawn ``dq_batches`` one.

    ``build_v_head_minibz_table_3d`` passes ``dq_batches``; the q=0 head and
    every single-shift caller do not, and take the fill inside
    ``minibz_cell_average`` instead.  Three q on the worst cell — enough to
    show the two paths agree, cheap because the fold is already compiled.
    """
    bvec = BVEC["hexagonal"]
    cv = celvol_of(bvec)
    ref, vr = rejection_minibz_offsets(bvec, KGRID, N_REJECT, seed=0)
    assert abs(vr - 1.0) <= VOL_IDENTITY_TOL
    for qc in q_shifts(bvec, KGRID)[:3]:
        want = head_from_offsets(qc, ref, cv)
        got = minibz_cell_average(
            qc, bvec=bvec, kgrid=KGRID, sys_dim=3, channel="full",
            units="per_volume", celvol=cv, n_kpts=NQ, nsamples=N_SOBOL,
            qmc_reps=QMC_REPS, nmax=NMAX, adaptive=False, distribute=False)
        assert abs(got - want) / abs(want) <= TOL_CEILING, (
            f"hexagonal, q={qc}: minibz_cell_average's own draw gives "
            f"{got:.6e} against a ground truth of {want:.6e}")
