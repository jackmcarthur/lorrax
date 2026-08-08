"""Committed guard for the 3D mini-BZ body-head DRAW (gw.compute_vcoul).

``build_v_head_miniBZ_avg_3d`` draws nmc points "uniformly on the Voronoi
cell of the mini-BZ" by sampling U ~ U[0,1)^3 in fractional coordinates,
mapping to Cartesian, and wrapping to the Voronoi cell.  The map has exactly
one correct spelling: ``U @ bvec`` (the parallelepiped spanned by the ROWS
b1,b2,b3 — a fundamental domain of the reciprocal lattice, so the wrap is a
measure-preserving bijection).  The transposed spelling ``U @ bvec.T`` spans
the parallelepiped of the COLUMNS, which is NOT a fundamental domain of that
lattice: the wrap then covers parts of the Voronoi cell twice and parts not
at all, with the same total volume (det is transpose-invariant), so no
volume or normalisation check can see it.  ``compute_vcoul.py:62`` shipped
the transposed spelling from its birth (2b8c1d9, 2026-05-16) until the
2026-08-07 fix this file rides with.

WHY SILICON CANNOT TEST THIS (and this file can): for the Si FCC fixture,
``bvec.T == P @ bvec`` with P a cyclic permutation matrix, so
``U @ bvec.T == (U @ P) @ bvec`` — and a column permutation of an i.i.d.
uniform draw is distributionally identical to the draw.  On Si the bug is a
RESEED (Monte-Carlo noise, ~1e-3 relative at nmc=2**18); on any cell without
such a P it is a BIAS (measured 2026-08-07, nmc=65536, 12 seeds: hexagonal
MoS2-class 3D bulk 6.49e-2 mean / 1.63e-1 max relative head error = 49.8 %
of the whole mc-average correction, z = 376 standard errors; monoclinic
beta=105 z = 58; triclinic z = 71).  Every gate fixture in this repository
is either Si (cyclic-degenerate) or sys_dim=2 (head table gated off), so
this synthetic-lattice suite is the ONLY coverage the draw has.  Pure
numpy + the production function itself; no fixture I/O on the main paths;
~seconds.

Layout — four independent arms, each falsifiable:

1. the Si-immunity ALGEBRA (P exists, exactly) + its red twin (no P for the
   hexagonal cell — the immunity does not generalise);
2. two-sided DISCRIMINATION on the hexagonal cell: production output must
   sit tightly on an independent numpy mirror of the correct rule and far
   from the mirror of the shipped (transposed) rule.  This is the line that
   goes red if the transpose ever returns;
3. the bias-vs-reseed SEED-BAND method: the shipped rule's seed-averaged
   table sits hundreds of standard errors outside the production seed band
   on the hexagonal cell, and INSIDE it on Si (the arm where the same
   statistic must NOT fire — the falsification case);
4. convention-free GROUND TRUTH: a rejection-sampled uniform-on-the-
   Voronoi-cell head table (no wrap, no frac->cart map at all) agrees with
   production and disagrees with the shipped rule, so arms 2-3 cannot both
   be wrong the same way.
"""
import itertools

import numpy as np
import pytest

from gw.compute_vcoul import build_v_head_miniBZ_avg_3d

# ---------------------------------------------------------------------------
# Lattices.
# ---------------------------------------------------------------------------

# The si_cohsex_debug fixture's blat * bvec (rows, 1/bohr), verified against
# tests/regression/si_cohsex_debug/WFN.h5 by test_si_literal_matches_fixture.
_SI_C = 0.612323844648
SI_BVEC = _SI_C * np.array([[1.0, 1.0, -1.0],
                            [-1.0, 1.0, 1.0],
                            [1.0, -1.0, 1.0]])
SI_KGRID = (4, 4, 4)

# Hexagonal MoS2-class 3D bulk (a = 3.16 A, c = 12.3 A, in bohr) — the
# smallest realistic cell class on which the two draw rules diverge.
_A_HEX = 3.16 / 0.529177
_C_HEX = 12.3 / 0.529177
HEX_BVEC = 2.0 * np.pi * np.array([
    [1.0 / _A_HEX, 1.0 / (np.sqrt(3.0) * _A_HEX), 0.0],
    [0.0, 2.0 / (np.sqrt(3.0) * _A_HEX), 0.0],
    [0.0, 0.0, 1.0 / _C_HEX]])
HEX_KGRID = (4, 4, 2)


def _celvol(bvec):
    """Real-space cell volume from the reciprocal rows: (2pi)^3 / det(b)."""
    return abs(8.0 * np.pi ** 3 / np.linalg.det(bvec))


# ---------------------------------------------------------------------------
# Independent numpy mirror of build_v_head_miniBZ_avg_3d (no jax), with the
# draw rule as an argument.  ``transpose=True`` is the pre-fix shipped rule,
# kept here as the RED TWIN — the counterfactual arm 2 discriminates against.
# ---------------------------------------------------------------------------

def _np_wrap_to_voronoi(randcart, bvec, nmax=1):
    grid = np.arange(-nmax, nmax + 1)
    shifts = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"),
                      axis=-1).reshape(-1, 3).astype(np.float64)
    cand = shifts @ bvec
    d = np.linalg.norm(randcart[:, None, :] - cand[None, :, :], axis=2)
    return randcart - cand[np.argmin(d, axis=1)]


def _head_from_offsets(dq_cart, kgrid, bvec, cell_volume):
    """The q-loop of build_v_head_miniBZ_avg_3d, verbatim, in numpy."""
    nkx, nky, nkz = (int(s) for s in kgrid)
    kgrid_arr = np.array([nkx, nky, nkz], dtype=np.float64)
    out = np.zeros((nkx, nky, nkz), dtype=np.float64)
    for qx in range(nkx):
        for qy in range(nky):
            for qz in range(nkz):
                qw = np.array([qx, qy, qz], dtype=np.float64)
                qw = np.where(qw > kgrid_arr / 2, qw - kgrid_arr, qw)
                q_cart = (qw / kgrid_arr) @ bvec
                if np.dot(q_cart, q_cart) < 1e-12:
                    continue
                shifted = q_cart[None, :] + dq_cart
                out[qx, qy, qz] = np.mean(
                    8.0 * np.pi / np.sum(shifted ** 2, axis=1))
    return out * (1.0 / float(cell_volume))


def _mirror_head(bvec, kgrid, *, nmc, seed, transpose):
    rng = np.random.RandomState(seed)
    randvals = rng.uniform(0, 1, (nmc, 3))
    randcart = randvals @ (bvec.T if transpose else bvec)
    wrapped = _np_wrap_to_voronoi(randcart, bvec, nmax=1)
    kgrid_arr = np.asarray(kgrid, dtype=np.float64)
    randlims = bvec.T @ (np.diag(1.0 / kgrid_arr) @ np.linalg.inv(bvec.T))
    dq_cart = (randlims @ wrapped.T).T
    return _head_from_offsets(dq_cart, kgrid, bvec, _celvol(bvec))


def _rel(a, b, mask):
    return np.abs(a - b)[mask] / np.abs(b)[mask]


def _nonzero_mask(kgrid):
    m = np.ones(tuple(int(s) for s in kgrid), dtype=bool)
    m[0, 0, 0] = False
    return m


# ---------------------------------------------------------------------------
# Arm 1 — the Si-immunity algebra, and its red twin.
# ---------------------------------------------------------------------------

def _signed_perm_with(bvec, tol=1e-12):
    """P over the 48 signed permutations with bvec.T == P @ bvec, or None."""
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            P = np.zeros((3, 3))
            for i, j in enumerate(perm):
                P[i, j] = signs[i]
            if np.max(np.abs(P @ bvec - bvec.T)) < tol:
                return P
    return None


def test_si_transpose_is_a_signed_permutation_hexagonal_is_not():
    """bvec.T = P.bvec holds EXACTLY for Si FCC (so U@bvec.T is a reseed of
    U@bvec there) and for NO signed permutation on the hexagonal cell (so
    nothing about Si's immunity generalises)."""
    P = _signed_perm_with(SI_BVEC)
    assert P is not None, (
        "Si FCC lost its cyclic-permutation degeneracy?!  The immunity "
        "argument in the 2026-08-07 fix commit would be wrong.")
    # The relation is exact for this bvec (entries are +-c), not approximate.
    assert np.max(np.abs(P @ SI_BVEC - SI_BVEC.T)) == 0.0
    # It is the cyclic permutation the fix commit quotes.
    assert np.array_equal(P, np.array([[0., 0., 1.],
                                       [1., 0., 0.],
                                       [0., 1., 0.]]))
    # Red twin: the hexagonal cell admits no such P, so the transposed draw
    # is NOT distributionally equivalent there — the bug is live on it.
    assert _signed_perm_with(HEX_BVEC) is None


def test_si_literal_matches_fixture():
    """Pin the SI_BVEC literal to the actual si_cohsex_debug WFN header."""
    h5py = pytest.importorskip("h5py")
    from pathlib import Path
    wfn = (Path(__file__).resolve().parent / "regression" / "si_cohsex_debug"
           / "WFN.h5")
    if not wfn.exists():
        pytest.skip("si_cohsex_debug/WFN.h5 not present")
    with h5py.File(wfn, "r") as f:
        bvec = np.asarray(f["mf_header/crystal/bvec"][:], dtype=np.float64)
        blat = float(np.asarray(f["mf_header/crystal/blat"])[()])
        kgrid = tuple(int(x) for x in f["mf_header/kpoints/kgrid"][:])
    np.testing.assert_allclose(blat * bvec, SI_BVEC, rtol=0.0, atol=1e-12)
    assert kgrid == SI_KGRID


# ---------------------------------------------------------------------------
# Arm 2 — two-sided discrimination on the hexagonal cell.  THE regression
# pin: this is the test that goes red if the transpose ever comes back.
# ---------------------------------------------------------------------------

def test_production_head_uses_the_row_convention_on_noncubic():
    nmc, seed = 16384, 42
    prod = build_v_head_miniBZ_avg_3d(
        HEX_KGRID, HEX_BVEC, _celvol(HEX_BVEC), nmc=nmc, seed=seed)
    correct = _mirror_head(HEX_BVEC, HEX_KGRID, nmc=nmc, seed=seed,
                           transpose=False)
    shipped = _mirror_head(HEX_BVEC, HEX_KGRID, nmc=nmc, seed=seed,
                           transpose=True)
    nz = _nonzero_mask(HEX_KGRID)
    # q=0 stays exactly 0 in all three (head handled as a separate term).
    assert prod[0, 0, 0] == 0.0
    # The two RULES are far apart on this cell (anti-tautology: if they were
    # close, the discrimination below would be vacuous).  Measured 6.5e-2
    # mean at nmc=65536; one seed at 16384 keeps the same scale.
    assert _rel(shipped, correct, nz).mean() > 1e-2
    # Production == the correct rule, to cross-implementation (jax-vs-numpy
    # wrap) precision — many orders tighter than the rule separation.
    np.testing.assert_allclose(prod[nz], correct[nz], rtol=1e-10)
    # ... and NOT the shipped (transposed) rule.
    assert _rel(prod, shipped, nz).mean() > 1e-2, (
        "build_v_head_miniBZ_avg_3d matches the TRANSPOSED draw "
        "(randvals @ bvec.T) — the 2026-05-16 bug is back.  See this "
        "file's docstring; Si gates CANNOT catch this, only this test can.")


# ---------------------------------------------------------------------------
# Arm 3 — the seed-band bias test, plus the Si arm where it must NOT fire.
# ---------------------------------------------------------------------------

def _seed_band_z(bvec, kgrid, *, nmc, seeds):
    """Max PAIRED z: per-seed (shipped − production) head-table differences,
    |mean| over seeds in units of their standard error, max over q.  Paired
    on the seed so the null (a pure reseed, as on Si) is a ~t_{n-1} per q —
    a bias shifts the mean of the differences, a reseed only widens them."""
    nz = _nonzero_mask(kgrid)
    prod = np.stack([
        build_v_head_miniBZ_avg_3d(kgrid, bvec, _celvol(bvec),
                                   nmc=nmc, seed=s)
        for s in seeds])
    ship = np.stack([
        _mirror_head(bvec, kgrid, nmc=nmc, seed=s, transpose=True)
        for s in seeds])
    d = (ship - prod)[:, nz]                       # (nseed, nq)
    se = np.maximum(d.std(axis=0, ddof=1) / np.sqrt(len(seeds)), 1e-300)
    return float(np.max(np.abs(d.mean(axis=0)) / se))


_SEEDS = tuple(42 + 1000 * s for s in range(12))


def test_shipped_rule_is_bias_on_hexagonal_reseed_on_si():
    """Bias survives seed averaging; a reseed does not.  MEASURED at this
    test's scale (nmc=8192, 12 seeds, paired): z_hex = 80.0, z_si = 2.43 —
    thresholds carry >3x margin each side.  (Unpaired survey scale,
    nmc=65536 x 12: z_hex = 376.)  The Si arm is the same statistic on the
    cell where it must NOT fire — the falsification case."""
    z_hex = _seed_band_z(HEX_BVEC, HEX_KGRID, nmc=8192, seeds=_SEEDS)
    z_si = _seed_band_z(SI_BVEC, SI_KGRID, nmc=8192, seeds=_SEEDS)
    assert z_hex > 25.0, (
        f"the transposed draw no longer registers as a BIAS on the "
        f"hexagonal cell (z={z_hex:.1f}) — the seed-band method lost its "
        f"discriminating power; do not weaken it, find out why.")
    assert z_si < 10.0, (
        f"the transposed draw registers as a bias on Si (z={z_si:.1f}) — "
        f"but bvec.T = P.bvec makes it a pure reseed there.  Either the "
        f"statistic broke or SI_BVEC changed.")


# ---------------------------------------------------------------------------
# Arm 4 — convention-free ground truth by rejection sampling.
# ---------------------------------------------------------------------------

def test_production_head_matches_rejection_ground_truth_on_hexagonal():
    """Uniform-on-the-Voronoi-cell by REJECTION (nearest-lattice-point test
    only; no wrap, no frac->cart draw map) — a ground truth that shares no
    convention with either rule.  Production must sit at MC-noise distance;
    the shipped rule must sit several times farther."""
    nmc, seed = 8192, 42
    bvec, kgrid = HEX_BVEC, HEX_KGRID
    rng = np.random.RandomState(7)
    kept = []
    got = 0
    # Sample a box that strictly contains the Voronoi cell: fractional
    # coords in [-1, 1)^3 cover it with room (the cell fits in one lattice
    # fundamental domain centred at 0).
    while got < nmc:
        pts = rng.uniform(-1.0, 1.0, (4 * nmc, 3)) @ bvec
        grid = np.arange(-2, 3)
        shifts = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"),
                          axis=-1).reshape(-1, 3).astype(np.float64)
        cand = shifts @ bvec
        d = np.linalg.norm(pts[:, None, :] - cand[None, :, :], axis=2)
        zero_row = int(np.argmin(np.linalg.norm(cand, axis=1)))
        m = np.argmin(d, axis=1) == zero_row
        kept.append(pts[m])
        got += int(m.sum())
    voronoi_pts = np.concatenate(kept, axis=0)[:nmc]
    kgrid_arr = np.asarray(kgrid, dtype=np.float64)
    randlims = bvec.T @ (np.diag(1.0 / kgrid_arr) @ np.linalg.inv(bvec.T))
    truth = _head_from_offsets((randlims @ voronoi_pts.T).T, kgrid, bvec,
                               _celvol(bvec))

    prod = build_v_head_miniBZ_avg_3d(kgrid, bvec, _celvol(bvec),
                                      nmc=nmc, seed=seed)
    shipped = _mirror_head(bvec, kgrid, nmc=nmc, seed=seed, transpose=True)
    nz = _nonzero_mask(kgrid)
    d_prod = _rel(prod, truth, nz).mean()
    d_ship = _rel(shipped, truth, nz).mean()
    assert d_prod < d_ship / 3.0, (
        f"production head is not closer to the rejection ground truth than "
        f"the transposed rule (|prod-truth| {d_prod:.3e} vs |shipped-truth| "
        f"{d_ship:.3e}) — the fix direction itself is in question.")
    # And production sits at MC-noise scale from the truth (two independent
    # 8192-point estimates; generous bound, red if systematically off).
    assert d_prod < 3e-2
