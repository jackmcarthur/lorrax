"""The MP1 occupation table's far tail is snapped to exact 0 / 1.

``occupation_clamp_tol`` answers a DIFFERENT question from
``occupation_window_threshold``, and this file exists partly to keep them
from being collapsed into one another.  The threshold decides which bands
are worth putting in a Green's-function branch; the clamp decides whether a
meaningless value exists in the table at all.  Both are live; neither
subsumes the other (``test_the_two_rules_are_not_the_same_rule``).

The argument for the clamp is a number.  Unclamped, the MP1 table's support
edge sits at ``x = (E-mu)/(2W) = 27.2971``, which is
``sqrt(-ln(5e-324)) = 27.2844`` to four figures -- the radius at which
``exp(-x^2)`` stops being REPRESENTABLE in float64.  That is not a physical
scale.  It puts weight on states ``54.59 * W`` from mu, i.e. 7.43 eV at
``W = 0.01 Ry`` -- 5.4x room-temperature kT -- and it moves if a backend
flushes subnormals.  ``tol = 1e-8`` puts the support at ``8.6167 * W``
(1.17 eV at the same width) and makes it backend-independent.

Three constraints this file pins, each of which a plausible wrong
implementation breaks:

1. **The near-Fermi overshoot is untouched, provably.**  MP1 overshoots
   [0, 1] near the Fermi surface and that overshoot is configured
   quadrature (closed-form extremum -0.0354579 at ``x = sqrt(3/2)``;
   measured -0.0316 to 1.0022 on the Na bcc SOC deck).  The guard is the
   ACCEPTED RANGE, capped 35x below the extremum, not the default happening
   to be small.
2. **The clamp is inside the fixed-N root.**  It perturbs the electron
   count, so mu must be solved for the clamped table or
   ``assert_fixed_n`` fails on the state's own invariant.  The
   counterfactual is executed here, not assumed.
3. **An insulating table is bit-identical.**  Proved by counting that the
   clamp changes nothing, not inferred from equal outputs.

Scope: this file is a unit file.  It exercises the clamp, the solver
placement and the deck plumbing.  It runs no driver and says nothing about
QP energies -- those come from the deck-level A/B.
"""

import numpy as np
import pytest

import jax.numpy as jnp

from gw.efermi import (MP1_LOBE_EXTREMUM,
                       OCCUPATION_CLAMP_TOL_DEFAULT,
                       OCCUPATION_CLAMP_TOL_MAX,
                       OCCUPATION_WINDOW_THRESHOLD_DEFAULT,
                       OccupationState,
                       band_in_occupation_window,
                       clamp_occupation_tail,
                       mp1_occupations,
                       occupation_clamp_tol,
                       occupation_weight_floor,
                       occupied_band_count,
                       solve_mp1_occupations)


def _mp1_closed_form(x):
    """``f(x)`` on the host, in numpy, independent of the module under test."""
    from scipy.special import erf
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 - erf(x)) - x * np.exp(-x * x) / (2.0 * np.sqrt(np.pi))


def _metallic_table(width=0.01, nb=241, span=0.6):
    """A one-k MP1 table that carries BOTH lobes and both far tails."""
    energies = np.linspace(-span, span, nb)
    return energies, np.asarray(mp1_occupations(
        jnp.asarray(energies[None, :]), 0.0, width, 0.0))[0]


# --------------------------------------------------------------------------
# The helper itself
# --------------------------------------------------------------------------

def test_the_rule_is_exactly_what_the_owner_asked_for():
    """``abs(f) < tol -> 0.0``; ``abs(1-f) < tol -> 1.0``; nothing else."""
    f = np.array([[0.0, 1e-12, -1e-12, 1e-9, 0.5, 1.0 - 1e-9,
                   1.0, 1.0 + 1e-12, 0.25]])
    out = np.asarray(clamp_occupation_tail(jnp.asarray(f), 1e-8))
    expect = np.array([[0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.25]])
    assert np.array_equal(out, expect), out


def test_the_boundary_is_strict_so_a_value_at_tol_survives():
    """``<``, not ``<=``: ``f = tol`` is outside the clamp, by construction."""
    tol = 1e-8
    f = jnp.asarray([[tol, np.nextafter(tol, 0.0)]])
    out = np.asarray(clamp_occupation_tail(f, tol))[0]
    assert out[0] == tol
    assert out[1] == 0.0


def test_no_element_moves_by_more_than_the_tolerance():
    """The bound that makes constraint 1 an argument rather than a hope."""
    _, f = _metallic_table()
    for tol in (0.0, 1e-12, OCCUPATION_CLAMP_TOL_DEFAULT,
                OCCUPATION_CLAMP_TOL_MAX):
        moved = np.abs(np.asarray(clamp_occupation_tail(
            jnp.asarray(f[None, :]), tol))[0] - f)
        assert moved.max() <= tol, (tol, moved.max())


def test_zero_tolerance_is_the_bit_for_bit_escape_hatch():
    _, f = _metallic_table()
    out = np.asarray(clamp_occupation_tail(jnp.asarray(f[None, :]), 0.0))[0]
    assert out.tobytes() == f.tobytes()


# --------------------------------------------------------------------------
# CONSTRAINT 1 -- the near-Fermi overshoot is untouched, provably
# --------------------------------------------------------------------------

def test_the_lobe_extremum_constant_is_the_closed_form():
    """``f'(x) = 0`` at ``2x^2 = 3``; the guard's whole basis is this number."""
    x = np.sqrt(1.5)
    assert MP1_LOBE_EXTREMUM == pytest.approx(
        abs(float(_mp1_closed_form(x))), rel=0.0, abs=1e-13)


def test_the_accepted_range_cannot_reach_the_overshoot():
    """The GUARD, not the default, is what makes constraint 1 hold.

    The largest change the clamp can make to any element is ``tol``
    (``test_no_element_moves_by_more_than_the_tolerance``); the smallest
    change that would damage the lobe is ``MP1_LOBE_EXTREMUM``.  The
    ceiling keeps the first at least 35x below the second, so no ADMISSIBLE
    setting -- not merely the default -- can flatten the overshoot.
    """
    assert OCCUPATION_CLAMP_TOL_MAX < MP1_LOBE_EXTREMUM / 30.0
    assert OCCUPATION_CLAMP_TOL_DEFAULT <= OCCUPATION_CLAMP_TOL_MAX


@pytest.mark.parametrize("bad", [-1e-12, 1e-2, MP1_LOBE_EXTREMUM, 1.0,
                                 float("nan"), float("inf")])
def test_a_tolerance_that_could_reach_the_lobe_refuses_by_name(bad):
    with pytest.raises(ValueError, match="occupation_clamp_tol"):
        occupation_clamp_tol(bad)


def test_a_synthetic_table_with_both_an_overshoot_and_a_tail():
    """The table the owner asked for: overshoot survives, tail becomes 0/1.

    Built by hand rather than sampled, so the overshoot is present at
    exactly the measured Na values and the tail at exactly the underflow
    scale -- neither depends on where a grid happened to land.
    """
    tail_hi = np.array([1e-9, 1e-12, 5e-324, 0.0])
    tail_lo = 1.0 - tail_hi
    overshoot = np.array([-0.0316, -0.0355, 1.0022, 1.0355])
    ordinary = np.array([0.25, 0.5, 0.75])
    f = np.concatenate([tail_lo, ordinary, overshoot, tail_hi])[None, :]

    out = np.asarray(clamp_occupation_tail(
        jnp.asarray(f), OCCUPATION_CLAMP_TOL_DEFAULT))[0]

    n = tail_hi.size
    assert np.array_equal(out[:n], np.ones(n)), out[:n]
    assert np.array_equal(out[-n:], np.zeros(n)), out[-n:]
    # Everything else is untouched BIT FOR BIT, the overshoot included.
    kept = f[0][n:-n]
    assert out[n:-n].tobytes() == kept.tobytes()


def test_the_solved_table_keeps_its_overshoot():
    """End to end through the real solver, not just the helper."""
    energies, f_raw = _metallic_table()
    f_clamped = np.asarray(mp1_occupations(
        jnp.asarray(energies[None, :]), 0.0, 0.01,
        OCCUPATION_CLAMP_TOL_DEFAULT))[0]
    assert f_raw.min() < -0.03, f_raw.min()
    assert f_raw.max() > 1.03, f_raw.max()
    assert f_clamped.min() == f_raw.min()
    assert f_clamped.max() == f_raw.max()


# --------------------------------------------------------------------------
# The headline: the support edge stops being a float64 artifact
# --------------------------------------------------------------------------

def _support_edge_in_x(width, clamp_tol):
    """Largest ``abs(x)`` at which the table is neither exactly 0 nor 1."""
    energies = np.linspace(-0.8, 0.8, 400001)
    f = np.asarray(mp1_occupations(
        jnp.asarray(energies[None, :]), 0.0, width, clamp_tol))[0]
    live = (f != 0.0) & (f != 1.0)
    return float(np.max(np.abs(energies[live]))) / (2.0 * width)


def test_the_support_edge_moves_off_the_float64_underflow_radius():
    """27.30 is ``sqrt(-ln(5e-324))``, not a physical scale.  4.31 is one."""
    unclamped = _support_edge_in_x(0.01, 0.0)
    clamped = _support_edge_in_x(0.01, OCCUPATION_CLAMP_TOL_DEFAULT)
    assert unclamped == pytest.approx(27.2971, rel=2e-4), unclamped
    assert unclamped == pytest.approx(
        float(np.sqrt(-np.log(5e-324))), rel=1e-3)
    assert clamped == pytest.approx(4.30834, rel=2e-4), clamped


def test_the_clamped_support_is_a_fixed_number_of_smearing_widths():
    """Backend-independent AND width-independent, unlike the underflow edge."""
    for width in (0.005, 0.01, 0.02):
        edge = _support_edge_in_x(width, OCCUPATION_CLAMP_TOL_DEFAULT)
        assert edge == pytest.approx(4.30834, rel=1e-3), (width, edge)


# --------------------------------------------------------------------------
# CONSTRAINT 2 -- the clamp is INSIDE the fixed-N root
# --------------------------------------------------------------------------

def _metal_spectrum(nk=6, nb=24, seed=0):
    rng = np.random.default_rng(seed)
    E = np.sort(rng.uniform(-0.5, 0.5, size=(nk, nb)), axis=1)
    w = np.full(nk, 1.0 / nk)
    return E, w


def test_the_solved_state_satisfies_fixed_n_with_the_clamp_applied():
    E, w = _metal_spectrum()
    st = OccupationState.solve_mp1(E, w, 12.0, 0.02, state_capacity=1.0,
                                   clamp_tol=OCCUPATION_CLAMP_TOL_DEFAULT)
    realized = occupied_band_count(st.f_kn, w)
    assert abs(realized - 12.0) < 1e-10, realized
    f = np.asarray(st.f_kn)
    assert np.any(f == 0.0) or np.any(f == 1.0), "no tail in the fixture"


def test_clamping_after_the_solve_would_break_the_invariant():
    """The counterfactual, EXECUTED: this is why the placement is load-bearing.

    Solve unclamped, then clamp the finished table -- exactly the wrong
    order -- and the realised count no longer equals the target, so
    ``OccupationState``'s own ``assert_fixed_n`` refuses.  AT THE SHIPPING
    DEFAULT ``tol = 1e-8``, not at some inflated value: the residual the
    wrong order leaves is 3.9e-10, and ``assert_fixed_n``'s atol is 1e-10.
    """
    E, w = _metal_spectrum(nk=4, nb=64)
    tol = OCCUPATION_CLAMP_TOL_DEFAULT
    mu, f_raw = solve_mp1_occupations(E, w, 32.0, 0.01, state_capacity=1.0,
                                      clamp_tol=0.0)
    exact = occupied_band_count(f_raw, w)
    after = occupied_band_count(clamp_occupation_tail(f_raw, tol), w)
    assert abs(exact - 32.0) < 1e-13, exact
    assert abs(after - 32.0) > 1e-10, (
        f"fixture carries no clampable tail (residual {after - 32.0:.3e}); "
        "the counterfactual proves nothing")

    with pytest.raises(ValueError, match="fixed-N invariant violated"):
        from gw.efermi import assert_fixed_n
        assert_fixed_n(
            OccupationState(f_kn=clamp_occupation_tail(f_raw, tol),
                            mu_ry=float(mu), smearing_family="mp1",
                            smearing_width_ry=0.01, n_electrons=32.0),
            w, state_capacity=1.0)

    # ...and the RIGHT order leaves the invariant intact on the same fixture.
    inside = occupied_band_count(
        solve_mp1_occupations(E, w, 32.0, 0.01, state_capacity=1.0,
                              clamp_tol=tol)[1], w)
    assert abs(inside - 32.0) < 1e-13, inside


def test_the_root_and_the_table_use_the_same_clamped_values():
    """mu moves with the tolerance -- i.e. the count saw the clamp too."""
    E, w = _metal_spectrum(nk=4, nb=64)
    mu_off, _ = solve_mp1_occupations(E, w, 32.0, 0.01, state_capacity=1.0,
                                      clamp_tol=0.0)
    mu_on, f_on = solve_mp1_occupations(E, w, 32.0, 0.01, state_capacity=1.0,
                                        clamp_tol=OCCUPATION_CLAMP_TOL_MAX)
    assert float(mu_off) != float(mu_on)
    # ...and the clamped table still hits the target exactly.
    assert abs(occupied_band_count(f_on, w) - 32.0) < 1e-13


def test_the_occ_hash_describes_the_clamped_table():
    """``occ_hash`` binds the bytes consumers receive, so a stale MPA fit
    store built from an unclamped table refuses by name rather than being
    silently reused."""
    E, w = _metal_spectrum()
    a = OccupationState.solve_mp1(E, w, 12.0, 0.02, state_capacity=1.0,
                                  clamp_tol=0.0)
    b = OccupationState.solve_mp1(E, w, 12.0, 0.02, state_capacity=1.0,
                                  clamp_tol=OCCUPATION_CLAMP_TOL_DEFAULT)
    assert a.occ_hash != b.occ_hash


# --------------------------------------------------------------------------
# CONSTRAINT 3 -- an insulating table is bit-identical, proved by execution
# --------------------------------------------------------------------------

@pytest.mark.parametrize("tol", [0.0, 1e-12, OCCUPATION_CLAMP_TOL_DEFAULT,
                                 OCCUPATION_CLAMP_TOL_MAX])
def test_the_clamp_changes_nothing_on_a_zero_one_table(tol):
    """COUNTED, not inferred from equal outputs."""
    rng = np.random.default_rng(7)
    f = (rng.random((9, 40)) < 0.4).astype(np.float64)
    out = np.asarray(clamp_occupation_tail(jnp.asarray(f), tol))
    changed = int(np.count_nonzero(out != f))
    assert changed == 0, f"{changed} of {f.size} entries moved at tol={tol}"
    assert out.tobytes() == f.tobytes()


def test_a_gapped_step_state_is_untouched_by_any_admissible_clamp():
    """The insulating path, end to end and COUNTED.

    ``OccupationState.step`` builds a table of exact 0.0/1.0, so the clamp
    is the identity on it at every admissible tolerance -- by construction,
    not by the tolerance happening to be small.
    """
    nk, nb = 5, 8
    E = np.tile(np.array([-0.9, -0.8, -0.7, -0.6, 0.4, 0.5, 0.6, 0.7]),
                (nk, 1))
    E += np.linspace(0.0, 0.01, nk)[:, None]
    w = np.full(nk, 1.0 / nk)
    st = OccupationState.step(E, w, 4.0, state_capacity=2.0)
    f = np.asarray(st.f_kn)
    assert set(np.unique(f)) <= {0.0, 1.0}, np.unique(f)

    for tol in (0.0, OCCUPATION_CLAMP_TOL_DEFAULT, OCCUPATION_CLAMP_TOL_MAX):
        out = np.asarray(clamp_occupation_tail(jnp.asarray(f), tol))
        changed = int(np.count_nonzero(out != f))
        assert changed == 0, f"{changed} entries moved at tol={tol}"
        assert out.tobytes() == f.tobytes()


# --------------------------------------------------------------------------
# The two rules are additive, not alternatives
# --------------------------------------------------------------------------

def test_the_two_rules_are_not_the_same_rule():
    """They cut at different radii and they answer different questions.

    The threshold's floor bites at ``x = 2.14``; the clamp's support edge is
    at ``x = 4.31``.  The threshold is the binding cut everywhere it applies,
    which is why the clamp changes no band-set decision at the default
    threshold (next test) -- and why neither can be deleted in favour of the
    other.
    """
    floor = occupation_weight_floor(OCCUPATION_WINDOW_THRESHOLD_DEFAULT)
    assert floor == pytest.approx(0.005)
    assert OCCUPATION_CLAMP_TOL_DEFAULT < floor
    edge_thresh = float(
        max(x for x in np.linspace(0.0, 10.0, 200001)
            if abs(_mp1_closed_form(x)) > floor))
    edge_clamp = _support_edge_in_x(0.01, OCCUPATION_CLAMP_TOL_DEFAULT)
    assert edge_thresh == pytest.approx(2.1375, rel=1e-3), edge_thresh
    assert edge_clamp > edge_thresh


def test_the_threshold_sites_decide_identically_with_and_without_the_clamp():
    """Every band the clamp zeroes was already outside the window.

    The four consumers of ``band_in_occupation_window`` cut on
    ``abs(weight) > 0.005``; the clamp only moves values whose magnitude is
    below 1e-8.  So on both branch weights, at every default-threshold
    consumer, the band set is unchanged -- the clamp makes some of the
    threshold's work redundant on paper without changing a single result.
    Executed over a real MP1 table on both branches.
    """
    energies, f_raw = _metallic_table(nb=20001)
    f_clamped = np.asarray(mp1_occupations(
        jnp.asarray(energies[None, :]), 0.0, 0.01,
        OCCUPATION_CLAMP_TOL_DEFAULT))[0]
    assert np.count_nonzero(f_clamped != f_raw) > 0, "no tail in the fixture"

    floor = occupation_weight_floor(OCCUPATION_WINDOW_THRESHOLD_DEFAULT)
    for raw, clamped in ((f_raw, f_clamped), (1.0 - f_raw, 1.0 - f_clamped)):
        assert np.array_equal(band_in_occupation_window(raw, floor),
                              band_in_occupation_window(clamped, floor))


def test_the_exact_rule_escape_hatch_is_where_the_clamp_does_show_up():
    """``threshold = 1.0`` (floor 0) is the one consumer setting the clamp
    moves -- it drags the exact rule's edge in from x=27.30 to x=4.31, which
    is the entire point of the key."""
    energies, f_raw = _metallic_table(nb=20001, span=0.6)
    f_clamped = np.asarray(mp1_occupations(
        jnp.asarray(energies[None, :]), 0.0, 0.01,
        OCCUPATION_CLAMP_TOL_DEFAULT))[0]
    exact_floor = occupation_weight_floor(1.0)
    assert exact_floor == 0.0
    kept_raw = int(np.count_nonzero(
        band_in_occupation_window(f_raw, exact_floor)))
    kept_clamped = int(np.count_nonzero(
        band_in_occupation_window(f_clamped, exact_floor)))
    assert kept_clamped < kept_raw, (kept_raw, kept_clamped)


def test_the_fermi_surface_weight_is_deliberately_not_clamped():
    """``-df/dE`` is a different quantity; this key says nothing about it."""
    import inspect
    from gw.efermi import _mp1_negative_derivative_values
    assert "clamp" not in inspect.getsource(_mp1_negative_derivative_values)


# --------------------------------------------------------------------------
# Deck plumbing
# --------------------------------------------------------------------------

def test_the_key_is_a_deck_key_with_the_owners_default():
    from gw.gw_config import _DEFAULTS
    assert _DEFAULTS["occupation_clamp_tol"] == 1e-8
    assert _DEFAULTS["occupation_clamp_tol"] == OCCUPATION_CLAMP_TOL_DEFAULT


def test_the_key_reaches_the_config_and_the_solver_entry_points():
    import inspect
    from gw.gw_config import LorraxConfig

    assert "occupation_clamp_tol" in LorraxConfig.__dataclass_fields__
    for fn in (solve_mp1_occupations, mp1_occupations,
               OccupationState.solve_mp1):
        assert "clamp_tol" in inspect.signature(fn).parameters, fn.__name__

    # Every MP1 solve in the SC driver reads the deck value; none may fall
    # back to the module default silently.
    from gw import sc_iteration
    src = inspect.getsource(sc_iteration)
    n_solves = src.count("solve_mp1_occupations(\n") + src.count(
        "solve_mp1(\n")
    assert src.count("clamp_tol=") >= n_solves, (n_solves,
                                                 src.count("clamp_tol="))


def test_the_key_has_a_row_in_the_input_reference():
    from pathlib import Path
    doc = (Path(__file__).resolve().parents[1]
           / "docs" / "input_reference.md").read_text()
    rows = [ln for ln in doc.splitlines()
            if ln.startswith("| `occupation_clamp_tol`")]
    assert len(rows) == 1, "expected exactly one reference row"
    assert "`1e-8`" in rows[0]
    # The three facts a future reader must not have to rediscover.
    assert "27.2971" in rows[0]
    assert "fixed-N" in rows[0]
    assert "occupation_window_threshold" in rows[0]
