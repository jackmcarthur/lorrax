"""Gates for the MPA-W fit kernel (``src/gw/mpa``).

CPU only, ``JAX_PLATFORMS=cpu``; nothing here touches a cluster.

The suite is organised as the theory plan's own claims, one test per
claim:

* exact recovery -- synthesize ``W_c`` from a known pole set, sample it
  on the protocol grid, fit, and require the poles and residues back;
* one RED TWIN per guard -- a synthesized pole set the guard must catch,
  paired with the guard-disabled run that gets it wrong, plus the proof
  that the mandatory residue refit actually ran;
* noise robustness at a documented perturbation amplitude;
* ``vmap`` over a batch equals a loop of single fits, bit-identically;
* the nested-partition property.

PRECISION NOTE.  "Near machine precision" for this kernel means
``cond(A) * eps``, not ``eps``.  The Pade-in-z^2 system at ``n_p = 8``
over a Si-like span has a condition number of order ``5e7`` even after
the z_max rescaling and row equilibration -- that is a property of the
monomial Vandermonde the published method prescribes, not of this
implementation, and it is exactly the theory plan's ranked risk 6.  The
measured recovery below is ``|dOmega| ~ 3e-9`` and ``|dB| ~ 1e-8``,
which is the conditioning limit; the asserted tolerances sit one decade
above the measurement so the test reports a regression rather than
weather.
"""

import time

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from gw.mpa import diagnostics, pade_fit, sampling  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _si_like_poles(n_p, *, omega_lo=0.3, omega_hi=3.5, seed=0):
    """A well-separated, time-ordered pole set spanning a Si-like range.

    Widths grow with the pole energy, which is the physics the theory
    plan describes ("the fit assigns broad envelopes to high-lying pole
    bundles"), and every pole satisfies ``Gamma < a`` so it sits inside
    the admissible box without any guard having to fire.
    """

    rng = np.random.default_rng(seed)
    a = np.linspace(omega_lo, omega_hi, n_p)
    gamma = 0.04 + 0.05 * np.arange(n_p) / max(n_p - 1, 1)
    Omega = (a - 1j * gamma).astype(np.complex128)
    mag = 0.4 + 0.8 * rng.random(n_p)
    phase = 2.0 * np.pi * rng.random(n_p)
    B = (mag * np.exp(1j * phase)).astype(np.complex128)
    return Omega, B


def _sorted_like_fit(Omega, B):
    """Reorder a reference pole set into the kernel's canonical order."""

    order = np.lexsort((Omega.imag, Omega.real))
    return Omega[order], B[order]


def _grid(n_p=8, omega_m=4.0, **kw):
    return sampling.double_parallel_grid(n_p, omega_m, **kw)


# ---------------------------------------------------------------------------
# (a) EXACT RECOVERY
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_p", [4, 6, 8])
def test_exact_recovery_well_separated(n_p):
    """Known poles in, same poles out, at the conditioning limit."""

    z = _grid(n_p)
    Omega_t, B_t = _si_like_poles(n_p)
    W = pade_fit.synthesize_w_samples(Omega_t, B_t, z)

    Omega, B, diag = pade_fit.fit_mpa_poles(W, z, n_p)
    Omega = np.asarray(Omega)
    B = np.asarray(B)
    ref_O, ref_B = _sorted_like_fit(Omega_t, B_t)

    # No guard should fire on a physical, well-separated, in-range set.
    assert int(diag["n_reflected"]) == 0
    assert int(diag["n_time_order_flipped"]) == 0
    assert int(diag["n_pruned_coincident"]) == 0
    assert int(diag["n_pruned_out_of_range"]) == 0
    assert int(diag["n_valid"]) == n_p
    assert not bool(diag["any_correction"])

    assert np.max(np.abs(Omega - ref_O)) < 1.0e-7
    assert np.max(np.abs(B - ref_B)) < 1.0e-6
    assert float(diag["max_abs_residual"]) < 1.0e-7


def test_exact_recovery_si_np8_reports_precision(capsys):
    """The Si schedule (n_p = 8), with the achieved numbers printed."""

    n_p = sampling.POLE_SCHEDULE["Si"]
    assert n_p == 8
    z = _grid(n_p, omega_m=4.0)
    Omega_t, B_t = _si_like_poles(n_p)
    W = pade_fit.synthesize_w_samples(Omega_t, B_t, z)

    Omega, B, diag = pade_fit.fit_mpa_poles(W, z, n_p)
    ref_O, ref_B = _sorted_like_fit(Omega_t, B_t)
    d_omega = float(np.max(np.abs(np.asarray(Omega) - ref_O)))
    d_b = float(np.max(np.abs(np.asarray(B) - ref_B)))
    with capsys.disabled():
        print(f"\n[exact recovery n_p=8] max|dOmega|={d_omega:.3e} "
              f"max|dB|={d_b:.3e} cond={float(diag['cond_pade']):.3e} "
              f"resid={float(diag['max_abs_residual']):.3e}")
    assert d_omega < 1.0e-7
    assert d_b < 1.0e-6


def test_exact_recovery_metal_grid_alpha2():
    """The metal protocol: origin shifted to i*1e-5 Ha, alpha = 2."""

    n_p = 6
    z = sampling.double_parallel_grid(
        n_p, 5.0, material_class="metal", alpha=2)
    assert z[0].real == 0.0
    assert z[0].imag == pytest.approx(1.0e-5)
    Omega_t, B_t = _si_like_poles(n_p, omega_lo=0.2, omega_hi=3.0)
    W = pade_fit.synthesize_w_samples(Omega_t, B_t, z)
    Omega, B, diag = pade_fit.fit_mpa_poles(W, z, n_p)
    ref_O, ref_B = _sorted_like_fit(Omega_t, B_t)
    assert np.max(np.abs(np.asarray(Omega) - ref_O)) < 1.0e-6
    assert int(diag["n_valid"]) == n_p


# ---------------------------------------------------------------------------
# (b) RED TWINS, one per guard, each with the refit proof
# ---------------------------------------------------------------------------

_NO_REFLECT = {"reflection": False, "prune_out_of_range": False}


def test_red_twin_reflection_wrong_branch():
    """A pole with ``Gamma > a`` -- ``Re[Omega**2] < 0``, SI Eq. (S18).

    Unguarded the fit returns the overdamped pole verbatim: a mode
    further from the real axis than from the imaginary one, which the
    papers call nonphysical and which the Sigma stage's width buckets and
    Laplace complements are not certified for.  The reflection guard maps
    it to ``-conj(Omega**2)``, which swaps ``a`` and ``Gamma`` -- keeping
    ``Im[Omega**2]``, hence the time ordering -- and the residues are
    refit against the moved pole.
    """

    n_p = 4
    z = _grid(n_p, omega_m=4.0)
    Omega_t = np.array(
        [0.30 - 0.60j, 1.20 - 0.10j, 2.10 - 0.15j, 3.00 - 0.20j],
        dtype=np.complex128)
    B_t = np.array([0.7 + 0.2j, 0.9 - 0.3j, 0.5 + 0.1j, 0.6 + 0.4j],
                   dtype=np.complex128)
    W = pade_fit.synthesize_w_samples(Omega_t, B_t, z)

    O_bad, B_bad, d_bad = pade_fit.fit_mpa_poles(W, z, n_p, guards=_NO_REFLECT)
    O_ok, B_ok, d_ok = pade_fit.fit_mpa_poles(W, z, n_p)

    # THE UNGUARDED FIT GETS IT WRONG: it emits a pole outside the
    # physical sector, Re[Omega**2] < 0.
    assert int(d_bad["n_reflected"]) == 0
    assert np.any(np.real(np.asarray(O_bad) ** 2) < 0.0)
    assert np.any(np.abs(np.imag(np.asarray(O_bad)))
                  > np.real(np.asarray(O_bad)))

    # THE GUARD CATCHES IT: exactly one pole reflected, and every live
    # pole now sits in the admissible sector.
    assert int(d_ok["n_reflected"]) == 1
    live = np.asarray(d_ok["valid"])
    assert np.all(np.real(np.asarray(O_ok)[live] ** 2) >= 0.0)
    assert np.all(np.abs(np.imag(np.asarray(O_ok)[live]))
                  <= np.real(np.asarray(O_ok)[live]) + 1e-12)

    # The reflection is the a <-> Gamma swap, exactly.
    reflected = np.asarray(O_ok)[np.argmin(np.abs(np.asarray(O_ok) - 0.6))]
    assert reflected == pytest.approx(0.60 - 0.30j, abs=1e-6)

    # THE REFIT RAN, and it mattered.
    assert bool(d_ok["any_correction"])
    assert bool(d_ok["refit_performed"])
    O_norefit, B_norefit, d_norefit = pade_fit.fit_mpa_poles(
        W, z, n_p, refit_after_guards=False)
    assert not bool(d_norefit["refit_performed"])
    np.testing.assert_allclose(np.asarray(O_norefit), np.asarray(O_ok),
                               rtol=0, atol=0)
    assert np.max(np.abs(np.asarray(B_norefit) - np.asarray(B_ok))) > 1.0e-6


def test_red_twin_time_ordering():
    """An anti-time-ordered pole (``Im Omega > 0``) must be conjugated.

    Left alone it enters the tau stage as ``exp(-i Omega tau)`` with
    ``Im Omega > 0``, i.e. ``exp(+|Im Omega| tau)`` -- an exponential
    blow-up, which is why the theory plan calls the refit after this
    correction mandatory rather than advisory.
    """

    n_p = 4
    z = _grid(n_p, omega_m=4.0)
    Omega_t = np.array(
        [0.80 + 0.10j, 1.60 - 0.10j, 2.40 - 0.15j, 3.20 - 0.20j],
        dtype=np.complex128)
    B_t = np.array([0.8 + 0.1j, 0.7 - 0.2j, 0.6 + 0.3j, 0.5 - 0.1j],
                   dtype=np.complex128)
    W = pade_fit.synthesize_w_samples(Omega_t, B_t, z)

    guards_off = {"time_order": False, "prune_out_of_range": False}
    O_bad, _, d_bad = pade_fit.fit_mpa_poles(W, z, n_p, guards=guards_off)
    O_ok, B_ok, d_ok = pade_fit.fit_mpa_poles(W, z, n_p)

    assert int(d_bad["n_time_order_flipped"]) == 0
    assert np.any(np.imag(np.asarray(O_bad)) > 0.0)
    # The unguarded pole blows up in imaginary time; the guarded one decays.
    tau = 10.0
    env_bad = np.max(np.abs(np.exp(-1j * np.asarray(O_bad) * tau)))
    env_ok = np.max(np.abs(np.exp(-1j * np.asarray(O_ok) * tau)))
    assert env_bad > 1.0
    assert env_ok <= 1.0

    assert int(d_ok["n_time_order_flipped"]) == 1
    assert np.all(np.imag(np.asarray(O_ok)[np.asarray(d_ok["valid"])]) <= 0.0)
    assert bool(d_ok["refit_performed"])

    _, B_norefit, d_norefit = pade_fit.fit_mpa_poles(
        W, z, n_p, refit_after_guards=False)
    assert not bool(d_norefit["refit_performed"])
    assert np.max(np.abs(np.asarray(B_norefit) - np.asarray(B_ok))) > 1.0e-6


def test_red_twin_coincident_poles():
    """Two poles inside the coincidence tolerance collapse to one.

    The synthesis puts two genuinely distinct poles 0.01 apart and the
    tolerance is raised so that separation counts as coincident.  Raising
    the tolerance rather than shrinking the separation is deliberate: it
    keeps the underlying Pade solve well conditioned, so the test
    measures THE GUARD and not the roundoff of an unresolvable double
    root.
    """

    n_p = 4
    z = _grid(n_p, omega_m=4.0)
    Omega_t = np.array(
        [1.20 - 0.10j, 1.21 - 0.10j, 2.40 - 0.15j, 3.20 - 0.20j],
        dtype=np.complex128)
    B_t = np.array([0.6 + 0.2j, 0.5 - 0.1j, 0.7 + 0.3j, 0.4 - 0.2j],
                   dtype=np.complex128)
    W = pade_fit.synthesize_w_samples(Omega_t, B_t, z)

    scale = float(np.max(np.abs(z)))
    tol = 0.02 / scale        # 0.02 absolute -> the 0.01 pair is coincident
    tight = {"coincident_tol": 1.0e-9}
    loose = {"coincident_tol": tol}

    O_all, _, d_all = pade_fit.fit_mpa_poles(W, z, n_p, guards=tight)
    O_col, B_col, d_col = pade_fit.fit_mpa_poles(W, z, n_p, guards=loose)

    # Un-collapsed: both near-degenerate poles survive.
    assert int(d_all["n_pruned_coincident"]) == 0
    assert int(d_all["n_valid"]) == n_p
    near = np.asarray(O_all)[np.abs(np.real(np.asarray(O_all)) - 1.2) < 0.05]
    assert near.size == 2

    # Collapsed: one of the pair is dropped, its residue forced to zero,
    # and the survivor's residue refit to carry the pair's weight.
    assert int(d_col["n_pruned_coincident"]) == 1
    assert int(d_col["n_valid"]) == n_p - 1
    dead = ~np.asarray(d_col["valid"])
    assert np.all(np.asarray(B_col)[dead] == 0)
    survivor = np.asarray(B_col)[
        np.asarray(d_col["valid"])
        & (np.abs(np.real(np.asarray(O_col)) - 1.2) < 0.05)]
    assert survivor.size == 1
    assert abs(survivor[0] - (B_t[0] + B_t[1])) < 0.2 * abs(B_t[0] + B_t[1])

    # THE REFIT RAN.
    assert bool(d_col["refit_performed"])
    _, B_norefit, d_norefit = pade_fit.fit_mpa_poles(
        W, z, n_p, guards=loose, refit_after_guards=False)
    assert not bool(d_norefit["refit_performed"])
    assert np.max(np.abs(np.asarray(B_norefit) - np.asarray(B_col))) > 1.0e-6


def test_red_twin_out_of_range_pole():
    """A pole far above the sampled span carries no support and is dropped."""

    n_p = 4
    omega_m = 4.0
    z = _grid(n_p, omega_m=omega_m)
    scale = float(np.max(np.abs(z)))
    far = 12.0 * scale        # well beyond range_factor_hi = 2 * scale
    Omega_t = np.array(
        [1.20 - 0.10j, 2.40 - 0.15j, 3.20 - 0.20j, far - 0.9j],
        dtype=np.complex128)
    B_t = np.array([0.8 + 0.2j, 0.6 - 0.1j, 0.5 + 0.3j, 0.02 + 0.0j],
                   dtype=np.complex128)
    W = pade_fit.synthesize_w_samples(Omega_t, B_t, z)

    keep_all = {"prune_out_of_range": False}
    O_all, _, d_all = pade_fit.fit_mpa_poles(W, z, n_p, guards=keep_all)
    O_cut, B_cut, d_cut = pade_fit.fit_mpa_poles(W, z, n_p)

    # Unguarded the extrapolating pole is emitted verbatim.
    assert int(d_all["n_pruned_out_of_range"]) == 0
    assert np.max(np.real(np.asarray(O_all))) > 2.0 * scale

    # Guarded it is dropped, with its residue zeroed.
    assert int(d_cut["n_pruned_out_of_range"]) == 1
    assert int(d_cut["n_valid"]) == n_p - 1
    live = np.asarray(d_cut["valid"])
    assert np.all(np.real(np.asarray(O_cut))[live] <= 2.0 * scale)
    assert np.all(np.asarray(B_cut)[~live] == 0)

    # THE REFIT RAN, and the surviving three poles still describe the
    # sampled data -- the guard costs accuracy, it does not destroy it.
    assert bool(d_cut["refit_performed"])
    assert float(d_cut["rel_rms_residual"]) < 1.0e-2
    _, B_norefit, d_norefit = pade_fit.fit_mpa_poles(
        W, z, n_p, refit_after_guards=False)
    assert not bool(d_norefit["refit_performed"])
    assert np.max(np.abs(np.asarray(B_norefit) - np.asarray(B_cut))) > 1.0e-9


# ---------------------------------------------------------------------------
# (c) NOISE ROBUSTNESS
# ---------------------------------------------------------------------------

def test_noise_robustness_bounded_pole_movement(capsys):
    """Pole movement stays bounded under a documented sample perturbation.

    AMPLITUDE (documented, and the only thing this test asserts about the
    physics): a relative complex perturbation of ``1e-8`` of
    ``max_j |W_c(z_j)|``, applied independently to every sample.  That is
    a stand-in for the certified error vector of the fit-stage chi0
    quadrature, whose real budget line is 2 meV (theory plan section H);
    the meV threshold lives downstream in QP energy and cannot be
    evaluated here.

    Two claims, both of which a conditioning failure would break:

    1. BOUNDED.  At ``eps_rel = 1e-8`` no pole moves by more than
       ``1e-3`` in the sampling energy unit, and no pole dies.  The
       measured movement is ~``8e-5``, so the threshold carries a decade
       of margin.
    2. LINEAR.  Scaling the perturbation by ten scales the movement by
       roughly ten (asserted loosely, 3x-30x).  A fit whose poles respond
       chaotically to a perturbation this small is the theory plan's
       ranked risk 6 regardless of how small the movement happened to be
       at one amplitude.
    """

    n_p = 8
    z = _grid(n_p, omega_m=4.0)
    Omega_t, B_t = _si_like_poles(n_p)
    W = pade_fit.synthesize_w_samples(Omega_t, B_t, z)

    eps_rel = 1.0e-8
    rng = np.random.default_rng(11)
    unit = (rng.standard_normal(2 * n_p)
            + 1j * rng.standard_normal(2 * n_p)) * np.max(np.abs(W))

    out = diagnostics.perturbation_refit(W, z, n_p, eps_rel * unit)
    big = diagnostics.perturbation_refit(W, z, n_p, 10.0 * eps_rel * unit)

    max_d = float(out["max_d_omega"])
    cond = float(pade_fit.fit_mpa_poles(W, z, n_p)[2]["cond_pade"])
    ratio = float(big["max_d_omega"]) / max(max_d, 1.0e-300)
    with capsys.disabled():
        print(f"\n[noise eps_rel={eps_rel:.0e}] max|dOmega|={max_d:.3e} "
              f"max|dGamma|={float(out['max_d_gamma']):.3e} "
              f"cond={cond:.3e} 10x-ratio={ratio:.2f}")

    assert int(out["valid_count_change"]) == 0
    assert int(big["valid_count_change"]) == 0
    assert max_d < 1.0e-3
    assert float(out["max_d_gamma"]) < 1.0e-3
    assert 3.0 < ratio < 30.0


def test_diagnostics_conditioning_and_backward_error():
    """A correct solve of an ill-conditioned system reports BOTH facts."""

    n_p = 8
    z = _grid(n_p, omega_m=4.0)
    Omega_t, B_t = _si_like_poles(n_p)
    W = pade_fit.synthesize_w_samples(Omega_t, B_t, z)

    cnd = diagnostics.solve_conditioning(W, z, n_p)
    # Tiny backward error: the linear algebra was done right.
    assert float(cnd["backward_error"]) < 1.0e-12
    # Large condition number: the ANSWER is still only good to cond*eps.
    # This pairing is the whole point of reporting both -- a small
    # backward error alone would look like a clean bill of health.
    assert float(cnd["cond"]) > 1.0e6
    assert float(cnd["forward_residual"]) < 1.0e-7
    assert int(cnd["n_valid"]) == n_p


def test_diagnostics_holdout_discriminates():
    """The held-out residual separates representable from under-parameterised.

    Holding two samples out leaves ``2*n_p - 2`` points, which is the
    exact sample support of an ``n_p - 1`` pole fit.  So the held-out
    residual answers precisely one question: can ``n_p - 1`` poles carry
    this element?  The test pins both answers.
    """

    # REPRESENTABLE: 6 true poles, n_p = 7, so the reduced fit has
    # exactly the poles it needs.  Held-out residual at the conditioning
    # floor.
    n_p = 7
    z = _grid(n_p, omega_m=4.0)
    Omega_t, B_t = _si_like_poles(n_p - 1)
    W = pade_fit.synthesize_w_samples(Omega_t, B_t, z)
    ok = diagnostics.holdout_residual(W, z, n_p)
    assert tuple(np.asarray(ok["holdout_indices"])) == (3, 10)
    assert float(ok["max_rel_error"]) < 1.0e-6

    # UNDER-PARAMETERISED: 8 true poles, n_p = 8, so the reduced fit is
    # one pole short and the held-out points are where that shows.  The
    # diagnostic must NOT report this as fine.
    n_p = 8
    z = _grid(n_p, omega_m=4.0)
    Omega_t, B_t = _si_like_poles(n_p)
    W = pade_fit.synthesize_w_samples(Omega_t, B_t, z)
    short = diagnostics.holdout_residual(W, z, n_p)
    assert float(short["max_rel_error"]) > 1.0e-3
    assert (float(short["max_rel_error"])
            > 1.0e3 * float(ok["max_rel_error"]))


def test_holdout_refuses_np_below_two():
    with pytest.raises(ValueError, match="GATE holdout_support"):
        diagnostics.default_holdout_indices(1)


# ---------------------------------------------------------------------------
# (d) VMAP == LOOP, BIT-IDENTICALLY
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_elements", [1, 5, 32])
def test_vmap_batch_equals_loop_bit_identical(n_elements):
    n_p = 6
    z = _grid(n_p, omega_m=4.0)
    rng = np.random.default_rng(7)
    rows = []
    for e in range(n_elements):
        Omega_t, B_t = _si_like_poles(n_p, seed=100 + e)
        rows.append(pade_fit.synthesize_w_samples(Omega_t, B_t, z))
    tile = np.stack(rows).astype(np.complex128)

    O_b, B_b, d_b = pade_fit.fit_mpa_poles_batched(tile, z, n_p)
    O_b = np.asarray(O_b)
    B_b = np.asarray(B_b)
    for e in range(n_elements):
        O_s, B_s, d_s = pade_fit.fit_mpa_poles(tile[e], z, n_p)
        np.testing.assert_array_equal(O_b[e], np.asarray(O_s))
        np.testing.assert_array_equal(B_b[e], np.asarray(B_s))
        np.testing.assert_array_equal(
            np.asarray(d_b["valid"])[e], np.asarray(d_s["valid"]))
        assert (np.asarray(d_b["cond_pade"])[e]
                == np.asarray(d_s["cond_pade"]))
    del rng


def test_vmap_batch_timing_sanity(capsys):
    """Batched throughput, reported not asserted (beyond not-catastrophic).

    The production shape is column tiles of ``W_q(mu, nu)``: thousands of
    independent element fits per tile.  This measures that the vmapped
    kernel runs at a per-element cost in the same order as the single
    fit, i.e. that nothing in the guard chain forces a per-element host
    round trip.
    """

    n_p = 8
    n_elements = 2048
    z = _grid(n_p, omega_m=4.0)
    Omega_t, B_t = _si_like_poles(n_p)
    base = pade_fit.synthesize_w_samples(Omega_t, B_t, z)
    rng = np.random.default_rng(3)
    tile = (base[None, :] * (1.0 + 1.0e-3 * rng.standard_normal(
        (n_elements, 2 * n_p)))).astype(np.complex128)

    fn = jax.jit(lambda t: pade_fit.fit_mpa_poles_batched(t, z, n_p))
    out = fn(tile)
    jax.block_until_ready(out)
    t0 = time.perf_counter()
    for _ in range(3):
        out = fn(tile)
        jax.block_until_ready(out)
    dt = (time.perf_counter() - t0) / 3.0

    with capsys.disabled():
        print(f"\n[vmap timing] n_elements={n_elements} n_p={n_p} "
              f"{dt * 1e3:.1f} ms/batch  "
              f"{n_elements / dt:,.0f} elements/s  "
              f"{dt / n_elements * 1e6:.1f} us/element")
    assert np.asarray(out[0]).shape == (n_elements, n_p)
    assert dt < 60.0


# ---------------------------------------------------------------------------
# (e) NESTED PARTITIONS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_p", list(range(1, 8)))
def test_partition_matches_published_table(n_p):
    """Metals-paper Eq. (11), verbatim, for every tabulated n_p."""

    assert (sampling.partition_fractions(n_p)
            == sampling.PUBLISHED_PARTITION_FRACTIONS[n_p])


@pytest.mark.parametrize("n_p", list(range(1, 16)))
@pytest.mark.parametrize("alpha", [1, 2])
def test_nested_partition_adds_never_moves(n_p, alpha):
    """``n_p -> n_p + 1`` contains the old grid EXACTLY (bit-identical)."""

    a = sampling.partition_omegas(n_p, 4.0, alpha=alpha)
    b = sampling.partition_omegas(n_p + 1, 4.0, alpha=alpha)
    # Every old sample survives bit-for-bit, and exactly one is added.
    assert b.size == a.size + 1
    for value in a:
        assert np.any(b == value), f"omega={value!r} moved or vanished"


def test_np1_insulator_grid_is_the_gn_probe_pair():
    """``n_p = 1`` reproduces the GN-PPM anchor's two-point grid exactly.

    The theory plan (section A) keeps the ``n_p = 1`` Godby-Needs path
    verbatim as the compatibility mode that ties everything to the Si
    parity baseline.  It samples ``z = 0`` and ``z = i * varpi_2``, and
    ``varpi_2 = 1 Ha = 2 Ry`` is exactly the GN probe this tree already
    uses -- so the anchor and the MPA family share one grid rather than
    two grids that happen to agree.
    """

    g_ha = sampling.double_parallel_grid(1, 4.0)
    np.testing.assert_array_equal(
        g_ha, np.array([0.0 + 0.0j, 0.0 + 1.0j], dtype=np.complex128))
    g_ry = sampling.double_parallel_grid(1, 8.0, energy_unit="Ry")
    np.testing.assert_array_equal(
        g_ry, np.array([0.0 + 0.0j, 0.0 + 2.0j], dtype=np.complex128))


@pytest.mark.parametrize("material", ["insulator", "metal"])
def test_nested_double_parallel_grid(material):
    """The full 2*n_p complex grid is nested, one point added per line."""

    for n_p in range(1, 13):
        g0 = sampling.double_parallel_grid(
            n_p, 4.0, material_class=material)
        g1 = sampling.double_parallel_grid(
            n_p + 1, 4.0, material_class=material)
        assert g1.size == g0.size + 2
        for value in g0:
            assert np.any(g1 == value), (
                f"z={value!r} moved between n_p={n_p} and {n_p + 1}")


# ---------------------------------------------------------------------------
# GATES -- every refusal loud and by name
# ---------------------------------------------------------------------------

def test_gate_sample_support():
    z = _grid(4)
    W = np.ones(2 * 4, dtype=np.complex128)
    with pytest.raises(ValueError, match="GATE sample_support"):
        pade_fit.fit_mpa_poles(W, z, 5)
    with pytest.raises(ValueError, match="GATE sample_support"):
        pade_fit.fit_mpa_poles(W[:6], z[:6], 4)
    # FALSE case: matched support fits without complaint.
    pade_fit.fit_mpa_poles(W, z, 4)


def test_gate_rank_and_batching():
    z = _grid(4)
    W = np.ones((3, 8), dtype=np.complex128)
    with pytest.raises(ValueError, match="GATE W_samples_rank"):
        pade_fit.fit_mpa_poles(W, z, 4)
    with pytest.raises(ValueError, match="GATE W_tile_rank"):
        pade_fit.fit_mpa_poles_batched(W[0], z, 4)
    # FALSE case: the 2-D tile goes through the batched door.
    pade_fit.fit_mpa_poles_batched(W, z, 4)


def test_gate_guard_keys_known():
    z = _grid(4)
    W = np.ones(8, dtype=np.complex128)
    with pytest.raises(ValueError, match="GATE guard_keys_known"):
        pade_fit.fit_mpa_poles(W, z, 4, guards={"reflexion": False})
    # FALSE case: the correctly spelled key is accepted.
    pade_fit.fit_mpa_poles(W, z, 4, guards={"reflection": False})


def test_gate_sampling_refusals():
    with pytest.raises(ValueError, match="GATE n_p_positive"):
        sampling.double_parallel_grid(0, 4.0)
    with pytest.raises(ValueError, match="GATE omega_m_positive"):
        sampling.double_parallel_grid(4, 0.0)
    with pytest.raises(ValueError, match="GATE alpha_supported"):
        sampling.double_parallel_grid(4, 4.0, alpha=3)
    with pytest.raises(ValueError, match="GATE material_class_known"):
        sampling.double_parallel_grid(4, 4.0, material_class="semimetal")
    with pytest.raises(ValueError, match="GATE energy_unit_known"):
        sampling.double_parallel_grid(4, 4.0, energy_unit="eV")
    with pytest.raises(ValueError, match="GATE varpi_ordering"):
        sampling.double_parallel_grid(4, 4.0, varpi_near=1.0, varpi_far=0.1)
    # FALSE case for each: the scheduled Si call is accepted.
    assert sampling.double_parallel_grid(8, 4.0).shape == (16,)


def test_gate_error_vector_shape():
    n_p = 4
    z = _grid(n_p)
    Omega_t, B_t = _si_like_poles(n_p)
    W = pade_fit.synthesize_w_samples(Omega_t, B_t, z)
    with pytest.raises(ValueError, match="GATE error_vector_shape"):
        diagnostics.perturbation_refit(W, z, n_p, np.zeros(3, np.complex128))
    # FALSE case: a correctly sized certified error vector propagates.
    out = diagnostics.perturbation_refit(
        W, z, n_p, np.zeros(2 * n_p, np.complex128))
    assert float(out["max_d_omega"]) == 0.0


def test_gate_synthesis_off_pole():
    z = np.array([1.0 + 0.0j, 2.0 + 0.1j], dtype=np.complex128)
    with pytest.raises(ValueError, match="GATE synthesis_off_pole"):
        pade_fit.synthesize_w_samples(
            np.array([1.0 + 0.0j]), np.array([1.0 + 0.0j]), z)
    # FALSE case: a pole with a finite width is off every sample.
    pade_fit.synthesize_w_samples(
        np.array([1.0 - 0.1j]), np.array([1.0 + 0.0j]), z)
