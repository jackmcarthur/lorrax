"""Committed guard for the mini-BZ Coulomb-head cell average (vcoul.minibz).

The subtle piece is the q->0 head: BGW splits the 3D 1/q^2 singularity into an
analytic Baldereschi-Tosatti sphere term + MC of the smooth remainder outside
the inscribed sphere (Common/minibzaverage.f90:79-81).  LORRAX used to omit the
analytic term (pure Sobol), which makes the head seed-dependent.  This test
locks in the fix: the analytic-sphere head is stable across MC seeds while the
pure-Sobol head is not.  Pure-numpy, no fixture, no GPU (~seconds).

Full physical validation (point-vs-cell-avg vs arbitrary_q_bse.md §16.4b, eval_vq
bit-identity, exciton flag on/off) lives at
reports/minibz_head_averaging_2026-07-20/validate_minibz.py (needs the 6x6 fixture).
"""
import numpy as np
import vcoul as cb  # noqa: E402  — raw-array service names, no translation

# synthetic cubic reciprocal cell: bvec rows = 2pi * e_i  =>  real cell vol = 1
BVEC = 2.0 * np.pi * np.eye(3)
CELVOL = 1.0
KGRID = (4, 4, 4)
NK = int(np.prod(KGRID))


def _head3d(seed, analytic):
    """3D bulk head <v>_mBZ at shift=0 for a given MC seed."""
    dq = cb.minibz_voronoi_batches(BVEC, KGRID, nsamples=2**14, qmc_reps=1,
                                   nmax=3 if analytic else 1, is_2d=False,
                                   seed_offset=seed)
    q0sph2 = cb.minibz_inscribed_sphere_r2(BVEC, KGRID, is_2d=False)
    return cb.minibz_average(np.zeros(3), dq, kind="bulk_3d", celvol=CELVOL,
                             n_kpts=NK, q0sph2=q0sph2,
                             analytic_sphere=analytic, adaptive=not analytic)


def test_analytic_sphere_head_is_seed_independent():
    a0, a1 = _head3d(0, True), _head3d(777, True)
    p0, p1 = _head3d(0, False), _head3d(777, False)
    an = abs(a0 - a1) / abs(0.5 * (a0 + a1))       # analytic-sphere rel spread
    pu = abs(p0 - p1) / abs(0.5 * (p0 + p1))       # pure-Sobol rel spread
    assert a0 > 0.0 and np.isfinite(a0), "analytic head must be finite & positive"
    assert an < 5e-3, f"analytic-sphere head seed-dependent: rel-spread {an:.2e}"
    assert an < pu, f"analytic term did not stabilize the head: {an:.2e} !< {pu:.2e}"


def test_analytic_term_matches_baldereschi_closed_form():
    # <v>_mBZ - (MC outside) must equal the closed-form sphere term
    # 4*sqrt(q0sph2)*celvol*Nk/pi  (minibzaverage.f90:79-81); the analytic piece
    # dominates the head, so the head must be >= that term (MC remainder >= 0).
    q0sph2 = cb.minibz_inscribed_sphere_r2(BVEC, KGRID, is_2d=False)
    analytic = 4.0 * np.sqrt(q0sph2) * CELVOL * NK / np.pi
    head = _head3d(0, True)
    assert head >= analytic * (1 - 1e-9), f"head {head:.5g} < sphere term {analytic:.5g}"
    assert head < analytic * 2.0, f"MC remainder implausibly large: {head:.5g} vs {analytic:.5g}"


if __name__ == "__main__":
    test_analytic_sphere_head_is_seed_independent()
    test_analytic_term_matches_baldereschi_closed_form()
    print("[minibz] both guards pass")
