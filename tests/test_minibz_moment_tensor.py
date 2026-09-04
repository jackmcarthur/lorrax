"""The mini-BZ second moment ``M_ab = <v(q) q_a q_b>_cell`` (vcoul.minibz).

The BSE exchange head's cell average is not ``<v>`` times anything: it is
``conj(d_a) M_ab d_b`` with ``d`` the transition dipole, exactly and with no
small-q expansion (``LT_HEAD_PROBLEM.md`` §3).  Six numbers, on the draws the
scalar average already makes.

What makes this gateable for free is that the moment has closed forms the
scalar does not.  In 3D bulk ``v(q) q² = 8π`` identically, so ``tr M = 8π/Ω``
for any cell shape and any shift — exact, and on the Baldereschi branch it is a
real test of the estimator, because the analytic sphere term has to supply
precisely the trace the MC drops.  Under slab truncation the same trace
vanishes LINEARLY with the cell and ``M_zz`` is identically zero, which is the
rank-two structure no scalar can carry.

Pure numpy/jax, no fixture files, no GPU; the two lattices are the checked-in
Si and hBN cell geometries transcribed as constants so the cells cost nothing
to read.  Seconds.

CENSUS-CLASS: convention/identity gates on a shared estimator.  They should
carry the ``census`` pytest marker as soon as it exists — it does not exist on
``main`` as of 2026-08-09 (a parallel lane is introducing it).
"""
import numpy as np
import pytest

import vcoul as cb  # noqa: E402  — raw-array service names, no translation

# tests/regression/si_cohsex_debug/WFN.h5 — FCC Si, blat*bvec, 1/bohr
SI_B = np.array([[0.61232384,  0.61232384, -0.61232384],
                 [-0.61232384, 0.61232384,  0.61232384],
                 [0.61232384, -0.61232384,  0.61232384]])
SI_OMEGA = 270.107161

# tests/regression/hbn_cohsex_debug/WFN.h5 — hBN slab, blat*bvec, 1/bohr
HBN_B = np.array([[1.32784284, 0.76663028, 0.0],
                  [0.0,        1.53326057, 0.0],
                  [0.0,        0.0,        0.49916206]])
HBN_OMEGA = 244.081798
HBN_ZC = float(np.pi / HBN_B[2, 2])

NS, REPS = 2 ** 16, 2


def _draw(bvec, kgrid, *, is_2d):
    return cb.minibz_voronoi_batches(bvec, kgrid, nsamples=NS, method="sobol",
                                     qmc_reps=REPS, nmax=3, is_2d=is_2d)


def test_trace_is_the_bare_coulomb_constant_on_both_bgw_branches():
    """tr M = 8π/Ω — LT_HEAD_PROBLEM.md §3.2, and the Si worked value.

    Two branches, and they are diagnostics of different strength.  Plain MC
    holds pointwise (``v q² = 8π`` sample by sample), so it certifies the
    contraction, the shift handling and the volume convention.  The
    Baldereschi analytic-sphere branch does NOT hold pointwise: the MC skips
    the inscribed sphere and the closed-form term has to put back exactly the
    trace that removed, so agreement there is a genuine test of the estimator
    and of the tensor twin of the sphere integral.
    """
    kg, nk = (4, 4, 4), 64
    q0sph2 = cb.minibz_inscribed_sphere_r2(SI_B, kg, is_2d=False)
    batches = _draw(SI_B, kg, is_2d=False)
    exact = 8.0 * np.pi / SI_OMEGA          # 0.09304730 Ry/bohr², §7 pin

    M = cb.minibz_moment_tensor(np.zeros(3), batches, kind="bulk_3d",
                                celvol=SI_OMEGA, n_kpts=nk, q0sph2=q0sph2,
                                analytic_sphere=False, adaptive=False)
    M = M / SI_OMEGA
    assert np.isclose(np.trace(M), exact, rtol=1e-12), (
        f"plain-MC tr M = {np.trace(M):.10f} != 8pi/Omega = {exact:.10f}")
    assert np.isclose(np.trace(M), 0.09304730, atol=5e-9)

    # cubic Si: isotropic to ~1e-4, off-diagonals at MC noise (§3.2)
    ratio = np.diag(M) / (np.trace(M) / 3.0)
    assert np.allclose(ratio, 1.0, atol=2e-3), f"not isotropic: {ratio}"
    offdiag = np.max(np.abs(M - np.diag(np.diag(M)))) / np.trace(M)
    assert offdiag < 5e-3, f"off-diagonal leak {offdiag:.2e}"

    M_as = cb.minibz_moment_tensor(np.zeros(3), batches, kind="bulk_3d",
                                   celvol=SI_OMEGA, n_kpts=nk, q0sph2=q0sph2,
                                   analytic_sphere=True) / SI_OMEGA
    rel = abs(float(np.trace(M_as)) - exact) / exact
    assert rel < 5e-4, (
        f"analytic-sphere tr M off by {rel:.2e} — the closed-form tensor term "
        f"and the outside-sphere MC do not add up to 8pi/Omega")

    # the shift is carried: any cell, anywhere, still traces to 8pi/Omega
    shift = np.array([0.19, -0.07, 0.31])
    M_sh = cb.minibz_moment_tensor(shift, batches, kind="bulk_3d",
                                   celvol=SI_OMEGA, n_kpts=nk, q0sph2=q0sph2,
                                   analytic_sphere=False, adaptive=False)
    assert np.isclose(np.trace(M_sh) / SI_OMEGA, exact, rtol=1e-12)


def test_slab_moment_is_rank_two_and_vanishes_linearly_with_the_cell():
    """2D: M_zz ≡ 0 and tr M ∝ Δ — LT_HEAD_PROBLEM.md §3.4.

    Both are things a scalar ``<v>`` gets wrong, and in opposite ways: it has
    no representation for "no out-of-plane component", and it does not vanish
    with the cell, so it over-weights the head on a fine grid.  The tensor
    needs no dimensional branch to get either right — the geometry arrives
    through ``kind`` and the draws.
    """
    asym = 8.0 * np.pi * HBN_ZC / HBN_OMEGA        # 0.648056
    rows = []
    for n in (3, 6, 12):
        kg = (n, n, 1)
        q0sph2 = cb.minibz_inscribed_sphere_r2(HBN_B, kg, is_2d=True)
        batches = _draw(HBN_B, kg, is_2d=True)
        M = cb.minibz_moment_tensor(np.zeros(3), batches, kind="slab",
                                    celvol=HBN_OMEGA, n_kpts=n * n,
                                    q0sph2=q0sph2, zc=HBN_ZC,
                                    analytic_sphere=False,
                                    adaptive=False) / HBN_OMEGA
        qpar = float(np.mean([
            np.mean(np.linalg.norm(np.asarray(b)[:, :2], axis=1))
            for b in batches]))
        assert np.max(np.abs(M[2, :])) == 0.0, (
            f"{n}x{n}: M has an out-of-plane component {M[2, :]} — the q_z=0 "
            f"slab moment must be exactly rank two")
        rows.append((n, qpar, float(np.trace(M))))

    (_, q3, t3), (_, q6, t6), (_, q12, t12) = rows
    # <|q_par|> halves exactly with the grid (it is a pure cell-size scale)
    assert np.isclose(q3 / q6, 2.0, rtol=2e-3) and np.isclose(q6 / q12, 2.0,
                                                              rtol=2e-3)
    # the trace follows it, sub-linearly at coarse grids and approaching the
    # 8*pi*zc/Omega asymptote from below as the cusp linearises
    assert 1.3 < t3 / t6 < 1.7, f"3x3 -> 6x6 trace ratio {t3/t6:.3f}"
    assert 1.5 < t6 / t12 < 1.9, f"6x6 -> 12x12 trace ratio {t6/t12:.3f}"
    ratios = [t3 / q3, t6 / q6, t12 / q12]
    assert ratios[0] < ratios[1] < ratios[2] < asym, (
        f"tr M / <|q_par|> must climb monotonically toward "
        f"8*pi*zc/Omega = {asym:.4f}, got {ratios}")
    assert ratios[2] > 0.5 * asym, f"12x12 ratio {ratios[2]:.4f} too far off"


@pytest.mark.parametrize("shift", [np.zeros(3), np.array([0.21, 0.0, 0.0])])
def test_red_twin_averaging_the_offset_instead_of_the_momentum(shift):
    """The estimator must weight the FULL momentum, not the cell offset.

    ``M_ab`` is the coefficient of a dipole bilinear at the cell's own
    location, so ``q`` is ``shift + δq`` and not ``δq``.  The offset-only
    spelling is the natural typo, it is invisible at Γ, and it destroys the
    trace identity everywhere else — which is exactly what makes the trace
    worth gating rather than merely reporting.
    """
    kg, nk = (4, 4, 4), 64
    q0sph2 = cb.minibz_inscribed_sphere_r2(SI_B, kg, is_2d=False)
    batches = _draw(SI_B, kg, is_2d=False)
    exact = 8.0 * np.pi / SI_OMEGA

    good = cb.minibz_moment_tensor(shift, batches, kind="bulk_3d",
                                   celvol=SI_OMEGA, n_kpts=nk, q0sph2=q0sph2,
                                   analytic_sphere=False,
                                   adaptive=False) / SI_OMEGA
    assert np.isclose(np.trace(good), exact, rtol=1e-12)

    # the twin, spelled out here rather than reached for in the module
    twin = np.zeros((3, 3))
    for dq in batches:
        dq = np.asarray(dq)
        v, _ = cb._minibz_kernel_bare(shift, dq, kind="bulk_3d")
        twin += (dq * v[:, None]).T @ dq / dq.shape[0]
    twin /= len(batches) * SI_OMEGA

    if np.allclose(shift, 0.0):
        assert np.allclose(twin, good, rtol=1e-10), (
            "at Gamma the two spellings coincide — that is why the defect "
            "survives a Gamma-only check")
    else:
        assert not np.isclose(np.trace(twin), exact, rtol=1e-3), (
            "the offset-only twin passed the trace gate; the gate cannot see "
            "the defect it exists for")
