"""The complex-frequency W evaluator and the 2x2 sampling object.

Three groups, in the order the pieces depend on each other:

1. THE KERNEL.  That the closed form ``K_z(Delta) = -2 Delta /
   (Delta**2 - z**2)`` really is theory-plan section B's damped-tau
   integral (checked against an independent numerical integration, not
   against the module's own rule), and that it reduces to each of the
   three shipped families' targets at their cell of the table -- which
   is the claim that makes the four cells one object.
2. THE RULE.  That the positive composite quadrature reproduces the
   closed form across the protocol's z values and the plan's
   bandwidths, at the tolerance it was asked for, with the node counts
   printed as a table.
3. THE PLAN, AND EVALUATE->FIT.  That each cell dispatches to its own
   route and an off-table sample refuses by name; and the first
   end-to-end that plants a spectrum, EVALUATES W on the protocol
   grid, and fits it -- as opposed to synthesizing W from a pole set,
   which is what every earlier MPA test does.

Tolerances here are read off the rule's convergence, never tuned: the
damped rule splits its budget half to the truncation and half to the
panels, so its measured error is ``~rel_tol`` times the kernel's
``1/varpi`` scale, and the assertions allow a factor of 2 on that.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from gw.mpa import evaluator, pade_fit, sample_plan, sampling  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _toy_spectrum(n_delta=8, lo=0.6, hi=7.0, n_chan=3, seed=3):
    """A toy chi0-like transition spectrum, in Rydberg.

    ``n_delta`` transitions with positive oscillator weights.  Fed
    through ``K_z`` this is EXACTLY an ``n_delta``-pole MPA model with
    ``Omega_p = Delta_p`` and ``B_p = g_p``, because

        sum_j g_j K_z(Delta_j) = sum_j 2 Delta_j g_j / (z^2 - Delta_j^2)

    is the model of ``pade_fit`` term for term.  That identity is what
    lets the smoke test below name the poles it planted -- and it is
    the physics, not a convenience: a discrete transition spectrum has
    real poles of zero width, and the widths the fit reports on real
    data come from the continuum it is summarising.
    """

    rng = np.random.default_rng(seed)
    delta = np.linspace(lo, hi, n_delta)
    weight = 0.2 + 0.6 * rng.random((n_delta, n_chan))
    return delta, weight


def _closed_form_samples(z, delta, weight):
    """``W_c(z_j)`` from the closed form: the oracle, in plain numpy."""

    k = evaluator.damped_kernel(
        np.asarray(z)[:, None], np.asarray(delta)[None, :])
    return np.einsum("zj,j...->z...", k, np.asarray(weight))


def _rule_error(rule, z_values, delta):
    """Worst |quadrature - closed form| over points and transitions."""

    worst = 0.0
    d = np.asarray(delta, dtype=np.float64)
    for z in z_values:
        proj = np.asarray(evaluator.damped_line_projection(rule, z))
        got = np.sin(np.outer(d, rule["t"])) @ proj
        worst = max(worst, float(np.max(np.abs(
            got - evaluator.damped_kernel(z, d)))))
    return worst


# ---------------------------------------------------------------------------
# 1. The kernel
# ---------------------------------------------------------------------------

def test_closed_form_is_the_damped_tau_integral():
    """The derivation, checked against a brute-force time integral.

    Independent of everything in ``evaluator`` except the formula
    itself: a fine uniform trapezoid over ``[0, t_max]`` of
    ``e^{i z t} sin(Delta t)``, with ``t_max`` far enough out that the
    envelope has died.  Slow and dumb on purpose -- if the closed form
    were wrong, a smarter check might be wrong the same way.

    The tolerance is the TRAPEZOID's, not the kernel's: at 4e6 points
    over ``[0, 60/varpi]`` the step is ~2e-4 wavelengths in the
    fastest case here, and ``(2 pi h / lambda)**2 / 12`` is ~1e-8
    relative.  Asking for more would be asking the reference to be
    better than it is.
    """

    z = 1.7 + 0.35j
    for delta in (0.4, 2.9, 6.1):
        t_max = 60.0 / z.imag
        t = np.linspace(0.0, t_max, 4_000_001)
        f = np.exp(1j * z * t) * np.sin(delta * t)
        brute = -2.0 * np.trapezoid(f, t)
        exact = evaluator.damped_kernel(z, delta)
        assert abs(brute - exact) < 5.0e-8 * abs(exact), (
            f"closed form {exact!r} vs trapezoid {brute!r} at "
            f"Delta={delta}")


def test_closed_form_reduces_to_each_shipped_family_target():
    """The unification claim, as three identities.

    ``sample_plan.FAMILIES`` says the three live cells approximate
    ``1/Delta``, ``Delta/(Delta^2+varpi^2)`` and
    ``Delta/(Delta^2-omega^2)``.  Each is ``K_z / KERNEL_FACTOR`` at
    that cell's ``z``.  If this test fails, the 2x2 table is a
    coincidence rather than a decomposition.
    """

    d = np.linspace(0.3, 6.0, 40)
    f = sample_plan.KERNEL_FACTOR
    varpi, omega = 0.7, 11.0

    static = evaluator.damped_kernel(0.0 + 0.0j, d) / f
    assert np.allclose(static, 1.0 / d, rtol=0, atol=1e-14)

    imag = evaluator.damped_kernel(1j * varpi, d) / f
    assert np.allclose(imag, d / (d ** 2 + varpi ** 2), rtol=0,
                       atol=1e-14)

    real = evaluator.damped_kernel(complex(omega, 0.0), d) / f
    assert np.allclose(real, d / (d ** 2 - omega ** 2), rtol=0,
                       atol=1e-14)


def test_closed_form_is_the_resonant_antiresonant_pair():
    """``K_z = 1/(z-Delta) - 1/(z+Delta)`` -- the chi0 spectral kernel."""

    z = np.array([0.0 + 0.4j, 2.5 + 0.2j, 1.0 + 3.0j])[:, None]
    d = np.array([0.5, 3.3, 8.0])[None, :]
    assert np.allclose(evaluator.damped_kernel(z, d),
                       1.0 / (z - d) - 1.0 / (z + d),
                       rtol=1e-14, atol=0)


def test_kernel_refuses_on_its_own_pole():
    with pytest.raises(ValueError, match="GATE off_kernel_pole"):
        evaluator.damped_kernel(complex(2.0, 0.0), 2.0)


# ---------------------------------------------------------------------------
# 2. The rule
# ---------------------------------------------------------------------------

#: ``A = (max|omega| + Delta_max) / varpi``.  The theory plan quotes
#: A ~ 80 for Si-like spans and ~200 for Cu-like; 20 is the far line.
_BANDWIDTHS = (20.0, 80.0, 200.0)
_DELTA_MAX = 8.0


def _line_case(a_dim):
    """``(varpi, freq_max, z_values, delta_grid)`` at bandwidth ``A``.

    The protocol's own geometry: ``omega_m`` is the top transition, so
    the widest point on the line carries ``freq_max = 2*Delta_max``,
    and the sampled ``omega`` are the partition of ``[0, omega_m]``.
    """

    freq_max = 2.0 * _DELTA_MAX
    varpi = freq_max / a_dim
    omegas = sampling.partition_omegas(8, _DELTA_MAX)
    z_values = [complex(om, varpi) for om in omegas if om > 0.0]
    delta = np.linspace(1.0e-3, _DELTA_MAX, 600)
    return varpi, freq_max, z_values, delta


def test_quadrature_matches_closed_form_over_the_protocol_and_A():
    """The agreement table, printed and asserted.

    ASSERTION, and where its number comes from.  The rule gives half
    its absolute budget ``rel_tol/varpi`` to the truncation
    (``|tail| <= 2 e^{-varpi t_max}/varpi``) and half to the panels,
    so the measured worst error is ``~rel_tol/varpi`` by construction
    -- not by tuning.  The test allows twice that.  It also asserts the
    error MOVES with the tolerance, which a rule that had silently
    saturated on something else would fail.
    """

    rows = []
    measured = {}
    for rel_tol in (1e-6, 1e-8, 1e-10, 1e-12):
        for a_dim in _BANDWIDTHS:
            varpi, freq_max, z_values, delta = _line_case(a_dim)
            rule = evaluator.damped_line_rule(
                varpi, freq_max, rel_tol=rel_tol)
            err = _rule_error(rule, z_values, delta)
            budget = rel_tol / varpi
            measured[(rel_tol, a_dim)] = err
            rows.append(
                f"  rel_tol={rel_tol:.0e}  A={a_dim:5.0f}  "
                f"nodes={rule['n_nodes']:5d} ({rule['n_nodes']/a_dim:5.2f}"
                f"*A)  panels={rule['n_panels']:3d}  GL order "
                f"{min(rule['orders']):3d}-{max(rule['orders']):3d}  "
                f"kappa0={rule['kappa0']:.4f}  err={err:.3e}  "
                f"err/budget={err/budget:.2f}")
            assert rule["n_nodes"] == rule["t"].size
            assert err < 2.0 * budget, (
                f"A={a_dim} rel_tol={rel_tol}: {err:.3e} exceeds twice "
                f"the requested {budget:.3e}")
    print("\n[mpa evaluator] closed form vs positive composite "
          "quadrature\n" + "\n".join(rows))

    # The tolerance is the knob it claims to be: four decades of
    # request buy at least three and a half decades of error.
    for a_dim in _BANDWIDTHS:
        ratio = measured[(1e-6, a_dim)] / measured[(1e-10, a_dim)]
        assert ratio > 3.0e3, f"A={a_dim}: error barely moved ({ratio})"


def test_rule_is_positive_and_bounded_in_amplification():
    """Positivity is the stability argument; kappa0 is the certificate.

    ``sum_l |w_l(z)| = 2 sum_l h_l e^{-varpi t_l} <= 2/varpi`` for
    every point on the line, independent of the node count.  The
    weights are the Gauss-Legendre ones, so they are positive and sum
    to ``t_max`` exactly, and ``kappa0`` approaches 1 from below as the
    truncation moves out.
    """

    for a_dim in _BANDWIDTHS:
        varpi, freq_max, z_values, _ = _line_case(a_dim)
        rule = evaluator.damped_line_rule(varpi, freq_max)
        assert np.all(rule["h"] > 0.0)
        assert np.all(rule["t"] >= 0.0)
        assert abs(rule["h"].sum() - rule["t_max"]) < 1e-12 * rule["t_max"]
        assert 0.99 < rule["kappa0"] <= 1.0
        for z in z_values:
            proj = np.asarray(evaluator.damped_line_projection(rule, z))
            assert np.sum(np.abs(proj)) <= 2.0 / varpi * (1 + 1e-12)


def test_rule_bandwidth_uses_omega_plus_delta_not_delta_alone():
    """A rule sized on the transitions alone under-resolves the line.

    The red twin for ``freq_max``: ``e^{i omega t} sin(Delta t)`` beats
    at ``omega +/- Delta``, so a rule built with ``freq_max =
    Delta_max`` is short by up to a factor two of oscillation and
    misses its own tolerance at the line's outer points.
    """

    varpi = 0.2
    delta = np.linspace(1e-3, _DELTA_MAX, 400)
    z_far = complex(_DELTA_MAX, varpi)
    right = evaluator.damped_line_rule(varpi, 2.0 * _DELTA_MAX,
                                       rel_tol=1e-8)
    wrong = evaluator.damped_line_rule(varpi, _DELTA_MAX, rel_tol=1e-8)
    budget = 1e-8 / varpi
    assert _rule_error(right, [z_far], delta) < 2.0 * budget
    assert _rule_error(wrong, [z_far], delta) > 10.0 * budget


def test_rule_refusals():
    with pytest.raises(ValueError,
                       match="GATE damped_line_positive_varpi"):
        evaluator.damped_line_rule(0.0, 16.0)
    with pytest.raises(ValueError,
                       match="GATE damped_line_positive_bandwidth"):
        evaluator.damped_line_rule(0.2, 0.0)
    with pytest.raises(ValueError, match="GATE damped_line_tolerance"):
        evaluator.damped_line_rule(0.2, 16.0, rel_tol=2.0)
    with pytest.raises(ValueError, match="GATE damped_line_panel_order"):
        evaluator.damped_line_rule(0.2, 16.0, rel_tol=1e-12,
                                   wavelengths_per_panel=400.0,
                                   max_order=64)
    rule = evaluator.damped_line_rule(0.2, 16.0)
    with pytest.raises(ValueError, match="GATE projection_on_line"):
        evaluator.damped_line_projection(rule, complex(1.0, 0.9))


# ---------------------------------------------------------------------------
# 3a. The 2x2 table
# ---------------------------------------------------------------------------

def test_the_four_cells_dispatch_by_analytic_character():
    """R4's table, cell by cell, as the sampling object reports it."""

    expect = {
        0.0 + 0.0j: ("static", "exponential_sum", "existing-kernel"),
        0.0 + 0.7j: ("imag", "exponential_sum_imag", "existing-kernel"),
        3.0 + 0.0j: ("real", "sine_sum", "existing-kernel"),
        3.0 + 0.7j: ("strip", "damped_line", "damped-tau"),
    }
    for z, (character, family, route) in expect.items():
        assert sample_plan.sample_character(z) == character
        assert sample_plan.family_for(z) == family
        assert sample_plan.route_for(z) == route
        assert sample_plan.FAMILIES[family]["character"] == character

    # The census the design brief quotes, carried as data so the hole
    # in the bottom-left cell stays visible.
    assert sample_plan.FAMILIES["exponential_sum"]["shipped_tables"] == 26
    assert sample_plan.FAMILIES["sine_sum"]["shipped_tables"] == 5
    assert sample_plan.FAMILIES[
        "exponential_sum_imag"]["shipped_tables"] == 0


def test_off_table_sample_refuses_by_name():
    """The red twin: a sample on the wrong sheet, named in the message.

    ``varpi < 0`` is not an empty cell of the table -- it is the lower
    half plane, where ``W_c`` is not analytic and the damped-tau
    integral diverges.  The refusal has to quote the ``z``.
    """

    with pytest.raises(ValueError) as exc:
        sample_plan.sample_character(complex(1.5, -0.25))
    assert "GATE sample_upper_half_plane" in str(exc.value)
    assert "-0.25" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        sample_plan.sample_point(complex(np.inf, 1.0), "bad")
    assert "GATE sample_finite" in str(exc.value)


def test_real_axis_sample_inside_the_band_refuses_by_name():
    """The other red twin: a live cell whose PRECONDITION fails.

    The ``sine_sum`` cell ships, but only above the band: its route
    splits the target into two ``1/y`` minimaxes and needs both
    branches positive.  A real-axis sample among the transitions has a
    real pole and no quadrature at all, and the plan must say so before
    any physics runs.
    """

    plan = sample_plan.sampling_plan(
        (sample_plan.sample_point(complex(3.0, 0.0), "hl_probe"),),
        label="hl-probe-inside-band")
    sample_plan.refuse_unsupported(plan, delta_max=2.0)      # fine
    with pytest.raises(ValueError) as exc:
        sample_plan.refuse_unsupported(plan, delta_max=7.0)
    assert "GATE sine_sum_above_band" in str(exc.value)
    assert "hl_probe" in str(exc.value)


def test_plan_refuses_duplicate_roles():
    with pytest.raises(ValueError, match="GATE distinct_roles"):
        sample_plan.sampling_plan(
            (sample_plan.sample_point(0.0 + 0.5j, "probe"),
             sample_plan.sample_point(0.0 + 1.5j, "probe")),
            label="clash")


def test_existing_families_are_cases_not_parallel_paths():
    """``static_plan`` / ``gn_ppm_plan`` reproduce today's requests.

    ``DESIGN_minimax.md`` section 4 asks that
    ``screening_requests_for`` become a projection of the sampling
    object rather than a sibling of it, and makes the bit-identity of
    the ``(omega_ry, role)`` pairs the gate on that refactor.  The
    refactor is not this branch's, but the identity is assertable now,
    and if it ever stops holding the projection is no longer available.
    """

    from types import SimpleNamespace

    from gw.gw_config import ComputeMode
    from gw.screening import screening_requests_for

    config = SimpleNamespace(ppm=SimpleNamespace(omega_p=1.234))

    for mode, plan in ((ComputeMode.COHSEX, sample_plan.static_plan()),
                       (ComputeMode.GN_PPM,
                        sample_plan.gn_ppm_plan(1.234))):
        want = [(complex(r.omega_ry), r.role)
                for r in screening_requests_for(mode, config)]
        got = [(p["z"], p["role"]) for p in sample_plan.plan_points(plan)]
        assert got == want, f"{mode}: {got} != {want}"


def test_mpa_plan_is_the_protocol_grid_with_cells_attached():
    """The plan adds the 2x2 to the grid and changes nothing else."""

    for material_class in ("insulator", "metal"):
        for n_p in (4, 8, 11):
            plan = sample_plan.mpa_plan(
                n_p, 7.0, material_class=material_class,
                energy_unit="Ry")
            grid = sampling.double_parallel_grid(
                n_p, 7.0, material_class=material_class,
                energy_unit="Ry")
            assert np.array_equal(sample_plan.plan_z(plan), grid)
            assert len(sample_plan.plan_points(plan)) == 2 * n_p

    plan = sample_plan.mpa_plan(8, 7.0, energy_unit="Ry")
    routes = sample_plan.plan_routes(plan)
    # Theory plan section B: z = 0 and the far line's omega = 0 point
    # take the shipped kernels; every sample with a real part is strip.
    assert [p["role"] for p in routes["existing"]] == ["near_00",
                                                       "far_00"]
    assert [p["family"] for p in routes["existing"]] == [
        "exponential_sum", "exponential_sum_imag"]
    assert [v for v, _ in routes["lines"]] == [0.2, 2.0]
    assert [len(pts) for _, pts in routes["lines"]] == [7, 7]

    # A metal's near-origin sample is IMAG, not static -- the 1e-5 Ha
    # shift is a real line height and the table must not round it away.
    metal = sample_plan.mpa_plan(8, 7.0, material_class="metal",
                                 energy_unit="Ry")
    first = sample_plan.plan_points(metal)[0]
    assert first["character"] == "imag"
    assert first["family"] == "exponential_sum_imag"
    assert first["varpi"] == 2.0e-5


def test_n_p_one_plan_is_the_gn_probe_pair():
    """The fit-kernel branch's build note 3, now on the plan object.

    ``mpa_plan(n_p=1)`` and ``gn_ppm_plan(varpi_2)`` are the same two
    points, so the GN compatibility anchor and the MPA family share one
    grid rather than two that coincide.
    """

    mpa = sample_plan.plan_z(sample_plan.mpa_plan(1, 7.0,
                                                  energy_unit="Ry"))
    gn = sample_plan.plan_z(sample_plan.gn_ppm_plan(2.0))
    assert np.array_equal(mpa, gn)


# ---------------------------------------------------------------------------
# 3b. Every cell evaluates, through one evaluator
# ---------------------------------------------------------------------------

def test_every_cell_evaluates_through_the_same_call():
    """One plan holding all four cells, evaluated in one call.

    The strip point is held to the damped rule's tolerance; the three
    shipped points are held to their own family's reported
    ``max_error`` times the spectrum's ``L1`` mass, which is the
    honest bound on an exponential-sum representation of the target.
    """

    delta, weight = _toy_spectrum(n_chan=1)
    delta_max = float(delta.max())
    varpi = 0.5
    plan = sample_plan.sampling_plan(
        (sample_plan.sample_point(0.0 + 0.0j, "cell_static"),
         sample_plan.sample_point(1j * varpi, "cell_imag"),
         sample_plan.sample_point(complex(3.0 * delta_max, 0.0),
                                  "cell_real"),
         sample_plan.sample_point(complex(2.0, varpi), "cell_strip")),
        label="four-cells")

    values, cost = evaluator.evaluate_samples(
        plan, delta, weight, rel_tol=1e-10, kernel_target_error=1e-10)
    exact = _closed_form_samples(sample_plan.plan_z(plan), delta, weight)
    err = np.abs(np.asarray(values) - exact)[:, 0]

    l1 = 2.0 * float(np.abs(weight).sum())
    shipped_budget = l1 * cost["kernel_max_error"]
    print(f"\n[mpa evaluator] four-cell errors {err} "
          f"(shipped budget {shipped_budget:.2e}, strip budget "
          f"{1e-10 / varpi:.2e})")
    assert np.all(err[:3] < shipped_budget)
    assert err[3] < 2.0 * (1e-10 / varpi)
    assert cost["kernel_points"] == 3 and cost["damped_points"] == 1


def test_shipped_imag_cell_and_damped_rule_agree_on_the_same_point():
    """The cross-route check that makes the table one object.

    ``z = i*varpi`` is served by ``exponential_sum_imag`` under the
    theory plan's routing, and by the damped rule as the ``omega = 0``
    edge of the strip.  They are two machines for one number, and they
    must agree to the looser of the two tolerances -- if they did not,
    "the existing families are cases of the sampling object" would be
    false in the only way that matters.
    """

    delta, weight = _toy_spectrum(n_chan=1)
    varpi = 1.4
    point = sample_plan.sample_point(1j * varpi, "probe")
    rule = evaluator.existing_kernel_rule(
        point, delta_min=float(delta.min()),
        delta_max=float(delta.max()), target_error=1e-10)
    shipped = sample_plan.KERNEL_FACTOR * np.einsum(
        "l,lj,jc->c", rule["h"],
        np.exp(-np.outer(rule["t"], delta)), weight)

    damped_rule = evaluator.damped_line_rule(
        varpi, float(delta.max()), rel_tol=1e-10)
    proj = np.asarray(evaluator.damped_line_projection(
        damped_rule, 1j * varpi))
    damped = np.einsum("l,jl,jc->c", proj,
                       np.sin(np.outer(delta, damped_rule["t"])), weight)

    exact = _closed_form_samples(np.array([1j * varpi]), delta,
                                 weight)[0]
    budget = max(2.0 * float(np.abs(weight).sum()) * rule["max_error"],
                 2.0e-10 / varpi)
    print(f"\n[mpa evaluator] iv={varpi}: shipped err "
          f"{abs(shipped[0] - exact[0]):.3e}, damped err "
          f"{abs(damped[0] - exact[0]):.3e}, routes differ by "
          f"{abs(shipped[0] - damped[0]):.3e} (budget {budget:.3e})")
    assert abs(shipped[0] - damped[0]) < budget


def test_per_point_and_per_line_batching_agree():
    """The fallback and the line-batched arithmetic are one evaluator.

    Theory-plan section B makes independent per-point sweeps the named
    correctness fallback for the line-batched design.  Both consume the
    identical plan; only the cost record differs -- which is exactly
    ``DESIGN_minimax.md`` section 5.1's reason for keeping batching out
    of the plan object.
    """

    delta, weight = _toy_spectrum()
    plan = sample_plan.mpa_plan(8, 7.0, energy_unit="Ry")
    per_point, cost_pp = evaluator.evaluate_samples(
        plan, delta, weight, batching="per-point", rel_tol=1e-10)
    per_line, cost_pl = evaluator.evaluate_samples(
        plan, delta, weight, batching="per-line", rel_tol=1e-10)

    worst_budget = max(2.0e-10 / row["varpi"] for row in cost_pp["lines"])
    assert np.max(np.abs(np.asarray(per_point) - np.asarray(per_line))) \
        < worst_budget
    assert cost_pp["damped_sweeps"] == 14
    assert cost_pl["damped_sweeps"] == 2
    # Batching saves sweeps, never outputs: the plan's demand that a
    # line-batched calculation not be called "one build".
    assert cost_pp["rq_transforms"] == cost_pl["rq_transforms"] == 16
    assert cost_pl["node_dispatches"] < cost_pp["node_dispatches"]

    with pytest.raises(ValueError, match="GATE batching_known"):
        evaluator.evaluate_samples(plan, delta, weight, batching="auto")


def test_the_sweep_is_a_replaceable_seam():
    """The signature survives the sweep becoming something else.

    ``evaluate_samples`` takes the two per-node sweeps as arguments so
    that the production path -- where the sine sweep is a chi0 build at
    time node ``t_l`` -- is a change of argument and not of caller. The
    stub here computes the same quantity a different way
    (``Im e^{i Delta t}``), and the values must be unchanged.
    """

    delta, weight = _toy_spectrum(n_chan=2)
    plan = sample_plan.mpa_plan(4, 7.0, energy_unit="Ry")

    def _sine_via_exp(t, d, g):
        basis = np.imag(np.exp(1j * np.outer(np.asarray(t),
                                             np.asarray(d))))
        return np.tensordot(basis, np.asarray(g), axes=(1, 0))

    base, _ = evaluator.evaluate_samples(plan, delta, weight)
    swapped, _ = evaluator.evaluate_samples(
        plan, delta, weight, sine_sweep_fn=_sine_via_exp)
    assert np.max(np.abs(np.asarray(base) - np.asarray(swapped))) < 1e-13


# ---------------------------------------------------------------------------
# 3c. evaluate -> fit, end to end
# ---------------------------------------------------------------------------

def test_evaluate_then_fit_recovers_the_planted_spectrum():
    """The first end-to-end that goes through EVALUATION, not synthesis.

    Every earlier MPA test plants ``(Omega, B)`` and synthesizes
    ``W_c(z_j)`` from the fit's own model.  This one plants a toy
    chi0-like transition spectrum, EVALUATES ``W_c`` on the full
    ``n_p = 8`` protocol grid through the sampling object -- shipped
    kernels at ``z = 0`` and ``i*varpi_2``, the damped-tau rule at the
    other fourteen -- and asks the fit kernel to name the transitions
    back.

    WHERE THE TOLERANCE COMES FROM, and what actually limits it.  The
    recovery is compared against the SAME fit run on closed-form
    samples, so the fit's own conditioning (measured ~6.6e8 here) is
    divided out of the claim: what is asserted is that evaluating
    instead of synthesizing costs less than a factor of a hundred in
    pole position.  The measured sample error is NOT the damped rule's
    -- at ``rel_tol = 1e-12`` that is 2e-12 -- but the imaginary-axis
    family's, which has ZERO shipped tables and solves at runtime; at
    the 1e-10 target used here it contributes ~1.5e-10, and tightening
    it to 1e-12 was measured at 96 s of runtime solve for a first-time
    cache miss.  That is R1's structural hole showing up as the
    accuracy floor of the MPA fit stage, and it is the reason the
    generator campaign of R4 point 2 is worth its cost.
    """

    n_p = 8
    delta, weight = _toy_spectrum(n_delta=n_p, n_chan=3)
    plan = sample_plan.mpa_plan(n_p, float(delta.max()),
                                energy_unit="Ry")
    z = sample_plan.plan_z(plan)

    values, cost = evaluator.evaluate_samples(
        plan, delta, weight, rel_tol=1e-12, kernel_target_error=1e-10)
    values = np.asarray(values)
    exact = _closed_form_samples(z, delta, weight)
    sample_error = float(np.max(np.abs(values - exact)))

    worst_eval, worst_synth, cond = 0.0, 0.0, 0.0
    for channel in range(weight.shape[1]):
        omega_e, b_e, diag = pade_fit.fit_mpa_poles(
            values[:, channel], z, n_p)
        omega_s, b_s, _ = pade_fit.fit_mpa_poles(
            exact[:, channel], z, n_p)
        worst_eval = max(
            worst_eval,
            float(np.max(np.abs(np.asarray(omega_e) - delta))),
            float(np.max(np.abs(np.asarray(b_e) - weight[:, channel]))))
        worst_synth = max(
            worst_synth,
            float(np.max(np.abs(np.asarray(omega_s) - delta))),
            float(np.max(np.abs(np.asarray(b_s) - weight[:, channel]))))
        cond = max(cond, float(diag["cond_pade"]))

    print(evaluator.format_evaluator_cost_report(cost))
    print(f"[mpa evaluate->fit] n_p={n_p} {weight.shape[1]} channels: "
          f"sample error {sample_error:.3e}, Pade condition "
          f"{cond:.3e}\n  planted spectrum recovered to "
          f"{worst_eval:.3e} through EVALUATION, against "
          f"{worst_synth:.3e} through synthesis "
          f"(ratio {worst_eval / worst_synth:.1f})")

    assert sample_error < 1.0e-8
    assert worst_synth < 1.0e-7
    assert worst_eval < 1.0e-5
    assert worst_eval < 100.0 * worst_synth


def test_cost_report_states_what_the_plan_demands():
    """Nodes, sweeps, GN-sweep units, and per-output work, all present."""

    delta, weight = _toy_spectrum(n_chan=1)
    plan = sample_plan.mpa_plan(8, float(delta.max()), energy_unit="Ry")
    _, cost = evaluator.evaluate_samples(plan, delta, weight,
                                         batching="per-line")
    text = evaluator.format_evaluator_cost_report(cost)
    print(text)

    assert cost["logical_outputs"] == 16
    assert cost["node_dispatches"] == (cost["kernel_nodes"]
                                       + cost["damped_nodes"])
    assert cost["gn_sweep_nodes"] > 0
    assert abs(cost["gn_sweeps"]
               - cost["node_dispatches"] / cost["gn_sweep_nodes"]) < 1e-9
    assert cost["rq_transforms"] == cost["dyson_solves"] == 16
    for token in ("logical outputs", "node dispatches", "GN sweeps",
                  "R->q transforms", "Dyson solves", "kappa0"):
        assert token in text
