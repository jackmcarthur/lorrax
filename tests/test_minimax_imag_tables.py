"""The ``complex_laplace`` catalog, and the certificate that admits it.

WHAT A CERTIFICATE IS FOR, AND WHY IT IS TESTED THIS WAY.  The request
census measured the same host producing two different answers to the
same uncertified solve four months apart, under one cache key, and the
older one is what a frozen gate is currently consuming.  A generator
that merely reports its own numbers reproduces that failure with extra
steps.  So every check below is re-derived HERE, from the shipped bytes,
without asking the generator what it thinks it produced -- and every
check is paired with a red twin: a deliberately mis-certified entry that
the same check must refuse.  A certificate nothing can fail is not a
certificate.

The six checks, and what each one is actually guarding:

``held_out_error``
    Maximum complex-modulus error on a half-cell-offset log grid the
    solver never sampled, plus both endpoints.  Guards fitting the
    grid instead of the function.
``real_part_error``
    The same, on ``Re`` alone, because ``alpha.real`` is what the
    ``noncrossing_imag`` consumer takes.  ``|Re E| <= |E|`` makes this
    implied, so it is really a statement that the two consumers share
    one payload -- see the alias test.
``kappa0``
    ``max_u u * sum_l |w_l| e^{-t_l u}`` -- the amplification against
    the pure-damping envelope, the same functional
    ``gw.mpa.evaluator.damped_line_rule`` reports and the same one
    every shipped ``noncrossing`` table measures between 1.0000 and
    1.0215 under.  Guards the near-cancellation the census found at
    ``sum|w| = 8.8e4``.
``moment``
    ``|int_1^R (fit - target) du|`` with both sides in closed form.
    The only check that touches no grid at all, so it is the one that
    cannot be gamed by any choice of sample points.
``rescale``
    That the catalog's physical-error convention -- absolute error
    scales as ``max_error / x_min`` -- is true over eleven decades of
    ``x_min``.  Nothing has ever checked it.
``payload_sha256``
    Byte identity of the TABLE.  The cross-platform claim is about the
    shipped bytes, never about re-running the solve; the census is the
    argument for that distinction.
"""

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_ASSETS = (_REPO / "services" / "minimax" / "src" / "minimax"
           / "minimax_assets")
_CATALOG = _ASSETS / "catalog_complex_laplace.json"


def _load_generator():
    name = "lorrax_tools_generate_imag_minimax_assets"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(str(_REPO), "tools",
                           "generate_imag_minimax_assets.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load_generator()

pytestmark = pytest.mark.skipif(
    not _CATALOG.exists(),
    reason="complex_laplace catalog not staged in this checkout")


def _catalog():
    return json.loads(_CATALOG.read_text(encoding="utf-8"))


def _entries():
    return _catalog()["tables"]


def _load(entry):
    return gen.read_table(_ASSETS / entry["file"])


def _ids(entries):
    return [f"R{e['range_max']:.0f}_b{e['beta']:.3f}_{e['rule']}"
            for e in entries]


ENTRIES = _entries() if _CATALOG.exists() else []


# ---------------------------------------------------------------------------
# 1. The catalog's own shape
# ---------------------------------------------------------------------------

def test_the_catalog_covers_the_census_request_table():
    """Three R buckets, two beta grids, two tiers -- and why each exists.

    ``MINIMAX_REQUEST_CENSUS.md`` sec 3.1 sizes the arming hole at three
    R buckets by six omega-hats: the R axis rounds up, so three buckets
    off the existing half-decade ladder cover all seven in-tree deck
    spans.  The beta axis turned out to need TWO grids and the reason is
    the campaign's subject.  The census DISPLAYS its six omega-hats to
    three decimals, and the first sweep took the display for the
    specification; the beta selector then scored the decks at
    ``2 Ry / x_min`` and four of seven missed their nearest band.  Both
    grids are kept: the rounded one because it is certified and other
    cells pin behaviour against it, the full-precision one because it is
    what a request actually carries.

    And two tiers, which are two different consumers: 1e-6 is
    production's -- C sec 2.6 measured that no in-tree request has ever
    asked for another -- and 1e-12 is the MPA fit stage's alone.
    """

    doc = _catalog()
    assert doc["schema_version"] == 2
    assert doc["family"] == "complex_laplace"
    got_r = sorted({e["range_max"] for e in doc["tables"]})
    got_b = sorted({e["beta"] for e in doc["tables"]})
    tiers = sorted({e["error_bound"] for e in doc["tables"]})
    assert got_r == sorted(gen.CENSUS_R_BUCKETS)
    assert tiers == sorted({1.0e-6, gen.FIT_STAGE_ERROR_BOUND})
    # Both beta grids are present in full.
    assert set(gen.CENSUS_OMEGA_HAT) <= set(got_b)
    assert set(gen.FULL_PRECISION_OMEGA_HAT) <= set(got_b)
    # Every deck's own omega-hat is tabulated at every R bucket, at both
    # tiers -- which is what makes the R axis's rounding usable.
    for beta in gen.DECK_OMEGA_HAT.values():
        for tier in tiers:
            covered = {e["range_max"] for e in doc["tables"]
                       if e["beta"] == beta and e["error_bound"] == tier}
            assert covered == set(gen.CENSUS_R_BUCKETS), (beta, tier)
    assert all(e["certified"] for e in doc["tables"])


def test_every_entry_declares_a_known_rule_and_a_shipping_tier():
    for entry in _entries():
        assert entry["rule"] in ("btv_minimax", "positive_composite")
        assert entry["kappa0_tier"] in ("normal", "versioned_exception")
        assert entry["kappa0"] <= gen.KAPPA_EXCEPTION
        assert entry["beta_axis"] in ("exact_entry_only",
                                      "exact_phase_on_fixed_nodes")


# ---------------------------------------------------------------------------
# 2. The six checks, re-derived from the shipped bytes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry", ENTRIES, ids=_ids(ENTRIES))
def test_shipped_entry_recertifies_from_its_own_bytes(entry):
    """Every stamped number, recomputed off disk and compared."""

    tau, alpha, stamped_error, stamped_kappa = _load(entry)
    cert = gen.certify(tau, alpha, entry["range_max"], entry["beta"],
                       entry["error_bound"],
                       kappa_max=gen.KAPPA_EXCEPTION)
    assert cert["passes"], cert["failures"]
    assert cert["node_count"] == entry["node_count"]
    assert cert["payload_sha256"] == entry["payload_sha256"]
    assert cert["held_out_max_error"] == pytest.approx(
        entry["max_error"], rel=1e-12)
    assert cert["kappa0"] == pytest.approx(entry["kappa0"], rel=1e-12)
    assert stamped_error == pytest.approx(entry["max_error"], rel=1e-12)
    assert stamped_kappa == pytest.approx(entry["kappa0"], rel=1e-12)
    assert cert["held_out_max_error"] <= entry["error_bound"]
    assert cert["kappa0"] <= float(entry["kappa0_bound"]) + 1e-9


def test_the_amplification_metric_agrees_with_the_shipped_catalog():
    """kappa0 is not a metric invented for this family.

    Every certified ``noncrossing`` table at the 1e-6 tier is scored
    with the identical functional; they land between 1.0000 and
    1.0215, which is what makes ``kappa0 <= 2`` a shipping rule rather
    than a number.  If this ever moves, the new family's certificate
    has silently changed meaning.
    """

    catalog = json.loads((_ASSETS / "catalog.json").read_text())
    worst = 0.0
    for entry in catalog["tables"]:
        if entry["family"] != "noncrossing":
            continue
        with np.load(_ASSETS / entry["file"]) as data:
            tau = np.asarray(data["tau"], dtype=np.float64)
            alpha = np.asarray(data["alpha"], dtype=np.float64)
        assert np.all(alpha > 0.0), (
            f"{entry['file']} was expected to have positive weights -- "
            "that positivity is why the static family self-certifies")
        stats = gen.measure(tau, alpha.astype(np.complex128),
                            entry["range_max"], 0.0)
        worst = max(worst, stats["kappa0"])
    assert 1.0 <= worst <= 1.05


# ---------------------------------------------------------------------------
# 3. The red twins.  One per check.
# ---------------------------------------------------------------------------

def _first(rule=None):
    for entry in _entries():
        if rule is None or entry["rule"] == rule:
            return entry
    pytest.skip(f"no {rule} entry in the catalog")


def test_red_twin_inflated_weight_fails_the_amplification_check():
    """A weight pair that cancels on the target still amplifies.

    Adding ``+c`` and ``-c`` at two nearby nodes barely moves the fit
    but doubles ``sum|w|`` there -- which is precisely the shape of the
    near-cancellation the census measured, so it is the right twin.
    """

    entry = _first()
    tau, alpha, _, _ = _load(entry)
    bad_t = np.concatenate([tau, tau[:1] * 1.0001, tau[:1] * 1.0002])
    bump = 40.0 * max(abs(alpha[0]), 1.0)
    bad_w = np.concatenate([alpha, [bump], [-bump]])
    cert = gen.certify(bad_t, bad_w, entry["range_max"], entry["beta"],
                       entry["error_bound"], kappa_max=gen.KAPPA_EXCEPTION)
    assert not cert["passes"]
    assert "kappa0" in cert["failures"]
    assert cert["kappa0"] > gen.KAPPA_EXCEPTION


def test_red_twin_moved_node_fails_the_held_out_error_check():
    entry = _first()
    tau, alpha, _, _ = _load(entry)
    bad_t = tau.copy()
    bad_t[len(bad_t) // 2] *= 1.05
    cert = gen.certify(bad_t, alpha, entry["range_max"], entry["beta"],
                       entry["error_bound"], kappa_max=gen.KAPPA_EXCEPTION)
    assert not cert["passes"]
    assert "held_out_error" in cert["failures"]


def test_red_twin_wrong_beta_fails_because_the_target_moved():
    """The omega-hat axis does not round up for a minimax entry.

    This is the census's sec 3.1 question, settled as an executable
    fact: score a ``btv_minimax`` entry against its NEIGHBOUR's beta and
    the error goes up by orders, because ``1/(u - i beta)`` is a
    different function at every beta.  It is why the schema-v2
    selection rule demands an exact beta match for that rule.
    """

    entry = _first("btv_minimax")
    # A NEIGHBOUR, NOT MERELY A DIFFERENT NUMBER.  The catalog now holds
    # the census's rounded grid and the decks' full-precision omega-hats
    # side by side, and those pairs differ by less than a band -- which
    # is the whole reason the band is measured.  The twin has to move
    # beta by more than any band to be testing what it claims.
    betas = sorted({e["beta"] for e in _entries()})
    other = next(b for b in betas
                 if abs(b - entry["beta"]) > 100.0 * entry["beta_tolerance"])
    tau, alpha, _, _ = _load(entry)
    cert = gen.certify(tau, alpha, entry["range_max"], other,
                       entry["error_bound"], kappa_max=gen.KAPPA_EXCEPTION)
    assert not cert["passes"]
    assert "held_out_error" in cert["failures"]
    assert (cert["held_out_max_error"]
            > 100.0 * entry["max_error"])


#: The two requests C sec 2.2 measured refusing, at the precision the
#: shim recorded them: (label, R, omega_hat).  ``gnppm``'s omega-hat is
#: the census's own logged value; ``bispinor``'s is ``2 Ry / x_min``
#: with the census's x_min, which is the best precision that document
#: carries.
CENSUS_REFUSALS = (
    ("gnppm", 42.6814, 16.006056518),
    ("bispinor", 38.7749, 2.0 / 0.122161),
)


@pytest.mark.parametrize("label,R,omega_hat", CENSUS_REFUSALS,
                         ids=[r[0] for r in CENSUS_REFUSALS])
def test_the_catalog_serves_the_two_requests_that_refuse_today(
        label, R, omega_hat):
    """The arming claim, executable.

    C sec 3.3 puts the whole stage-2 gap at these two rows.  Serving
    them means the R bucket rounds up (which it does, safely) AND the
    beta lands inside the entry's stamped tolerance band -- and the
    band is narrow: it is about ``error_bound * (1 + beta**2)``, which
    at beta = 16 and the 1e-6 tier is 2.6e-4.  The census displays its
    six omega-hat values to three decimals; three decimals happens to
    fit inside that band here, and the margin on ``bispinor`` is a few
    percent of it.  That is a fact to record, not a design to lean on:
    the grid is a specification in the deck's own omega-hat and the
    displayed rounding is not it.
    """

    served = [e for e in _entries()
              if e["range_max"] >= R
              and abs(e["beta"] - omega_hat) <= e["beta_tolerance"]]
    assert served, (
        f"{label}: no entry covers R={R} at omega_hat={omega_hat!r}; "
        f"nearest betas "
        f"{sorted({e['beta'] for e in _entries()})}")
    entry = min(served, key=lambda e: e["range_max"])
    tau, alpha, _, _ = _load(entry)
    x = gen.held_out_grid(R, 8001)
    err = float(np.max(np.abs(1.0 / (x - 1j * omega_hat)
                              - np.exp(-np.outer(x, tau)) @ alpha)))
    frac = abs(entry["beta"] - omega_hat) / entry["beta_tolerance"]
    print(f"[imag tables] {label}: R={R} -> bucket "
          f"{entry['range_max']:.4f}, omega_hat={omega_hat:.9f} -> "
          f"beta={entry['beta']} (delta is {frac:.1%} of the stamped "
          f"band {entry['beta_tolerance']:.3e}), N={entry['node_count']}"
          f", achieved err {err:.3e}, kappa0 {entry['kappa0']:.4f}")
    assert err <= entry["error_bound"]


def test_the_beta_band_is_measured_and_refuses_outside_itself():
    """The band is a certificate, so it has a FALSE case too."""

    entry = _first("btv_minimax")
    tau, alpha, _, _ = _load(entry)
    band = entry["beta_tolerance"]
    assert band > 0.0
    inside = gen.certify(tau, alpha, entry["range_max"],
                         entry["beta"] + 0.9 * band,
                         entry["error_bound"])
    assert "held_out_error" not in inside["failures"]
    outside = gen.certify(tau, alpha, entry["range_max"],
                          entry["beta"] + 4.0 * band,
                          entry["error_bound"])
    assert "held_out_error" in outside["failures"]
    # And the band tracks the derivative it should: d/dbeta of the
    # target has modulus 1/(1 + beta**2) at the hard end of the
    # interval, so the band is that suppression times the slack.
    predicted = entry["error_bound"] * (1.0 + entry["beta"] ** 2)
    assert 0.05 * predicted <= band <= predicted


def test_red_twin_flipped_byte_fails_the_digest():
    entry = _first()
    tau, alpha, _, _ = _load(entry)
    assert gen.payload_digest(tau, alpha) == entry["payload_sha256"]
    bad = alpha.copy()
    bad[0] = np.nextafter(bad[0].real, np.inf) + 1j * bad[0].imag
    assert gen.payload_digest(tau, bad) != entry["payload_sha256"]
    # And the digest is over BOTH arrays, not just the weights.
    bad_t = tau.copy()
    bad_t[-1] = np.nextafter(bad_t[-1], np.inf)
    assert gen.payload_digest(bad_t, alpha) != entry["payload_sha256"]


def test_red_twin_constant_offset_fails_the_moment_identity():
    """The grid-free check catches what a grid-blind rule hides.

    A node at ``t -> 0`` contributes an almost constant ``epsilon``
    across the whole interval: small enough in the sup norm to slip
    under the error bound, but it integrates to ``epsilon * (R - 1)``,
    which is the one thing the closed-form moment sees and no sample
    grid does.
    """

    entry = _first()
    tau, alpha, _, _ = _load(entry)
    eps = 0.4 * entry["error_bound"]
    bad_t = np.concatenate([[1.0e-9], tau])
    bad_w = np.concatenate([[eps + 0j], alpha])
    cert = gen.certify(bad_t, bad_w, entry["range_max"], entry["beta"],
                       entry["error_bound"], kappa_max=gen.KAPPA_EXCEPTION)
    assert "moment" in cert["failures"]
    assert not cert["passes"]
    # Specificity: the offset is small enough that the sup-norm checks
    # still pass, so the moment identity is the ONLY thing refusing it.
    assert "held_out_error" not in cert["failures"]
    assert "kappa0" not in cert["failures"]


def test_red_twin_wrong_rescaling_convention_fails_the_rescale_check():
    """The convention the catalog states, against the one it does not.

    ``tau/x_min`` with ``alpha`` LEFT ALONE is the plausible mistake --
    it is dimensionally wrong by exactly one factor of ``x_min`` and it
    is invisible at ``x_min = 1``.  The check refuses it away from
    unity, which is the whole point of drawing ``x_min`` over eleven
    decades.
    """

    entry = _first()
    tau, alpha, _, _ = _load(entry)
    R, beta = entry["range_max"], entry["beta"]
    base = gen.measure(tau, alpha, R, beta)["held_out_max_error"]
    x_min = 1.0e-3
    x = gen.held_out_grid(R, 4001) * x_min
    wrong = np.exp(-np.outer(x, tau / x_min)) @ alpha      # alpha unscaled
    got = float(np.max(np.abs(1.0 / (x - 1j * beta * x_min) - wrong)))
    assert got / (base / x_min) > 100.0
    # ...and the convention the catalog does state passes.
    right = np.exp(-np.outer(x, tau / x_min)) @ (alpha / x_min)
    ok = float(np.max(np.abs(1.0 / (x - 1j * beta * x_min) - right)))
    assert ok / (base / x_min) <= 1.05


def test_certify_refuses_a_negative_node():
    entry = _first()
    tau, alpha, _, _ = _load(entry)
    bad_t = tau.copy()
    bad_t[0] = -bad_t[0]
    cert = gen.certify(bad_t, alpha, entry["range_max"], entry["beta"],
                       entry["error_bound"], kappa_max=gen.KAPPA_EXCEPTION)
    assert "positive_nodes" in cert["failures"]


# ---------------------------------------------------------------------------
# 4. The two consumers, one payload
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry", ENTRIES, ids=_ids(ENTRIES))
def test_the_real_part_is_the_noncrossing_imag_target(entry):
    """``alpha.real`` fits ``minimax._imag_target`` at this beta.

    R4's "one campaign" claim, discharged against the production
    module's own target function rather than against a restatement of
    it.  ``|Re E| <= |E|``, so the complex certificate dominates -- the
    assertion is that the real consumer inherits the same bound and not
    merely a similar one.
    """

    import minimax as production                        # noqa: PLC0415

    tau, alpha, _, _ = _load(entry)
    x = gen.held_out_grid(entry["range_max"], 4001)
    want = production._imag_target(x, entry["beta"])
    got = production.evaluate_noncrossing_imag(x, tau, np.real(alpha))
    err = float(np.max(np.abs(want - got)))
    assert err <= entry["error_bound"]
    assert err == pytest.approx(entry["real_part_max_error"], rel=0.05)


def test_the_composite_route_rounds_the_beta_axis_and_minimax_does_not():
    """Gate 0's structural finding, as a test.

    A ``positive_composite`` rule's beta dependence is the exact
    unit-modulus phase ``e^{i beta t_l}`` on nodes chosen without
    reference to beta, so re-phasing the SAME nodes to any smaller beta
    is not an approximation -- it is the rule that route would have
    built there, and it certifies.  A ``btv_minimax`` rule has no such
    structure.  That asymmetry is the entire reason the schema-v2
    selection rule has two beta clauses.
    """

    R, beta_max, tol = 21.544346900318832, 40.214, 1.0e-6
    t, w, info = gen.composite_rule(R, beta_max, tol, beta_min=0.0)
    assert info["kappa0_bound"] == 1.0
    assert info["beta_axis"] == "exact_phase_on_fixed_nodes"
    h = np.abs(w)
    assert np.all(h > 0.0)
    assert np.sum(h) == pytest.approx(info["t_max"], rel=1e-9)
    for beta in (0.0, 1.0, 5.836, 16.006, beta_max):
        w_beta = h * np.exp(1j * beta * t)
        cert = gen.certify(t, w_beta, R, beta, tol,
                           kappa_max=gen.KAPPA_EXCEPTION)
        assert cert["passes"], (beta, cert["failures"])
        assert cert["kappa0"] == pytest.approx(1.0, abs=1e-6)


def test_the_composite_envelope_costs_a_longer_tail_and_says_so():
    """The one place the beta envelope is not free, stated as a number.

    The truncation error carries a ``1/|x - i beta|``, so the tail is
    EASIER at large beta and hardest at the bottom of the band.  An
    entry that means to serve down to beta = 0 must pay for the tail
    there.  Sizing it at its own beta instead is the mistake this
    parameter exists to prevent, and it is invisible until someone
    rounds.
    """

    R, beta, tol = 21.544346900318832, 40.214, 1.0e-6
    narrow = gen.composite_rule(R, beta, tol)
    wide = gen.composite_rule(R, beta, tol, beta_min=0.0)
    assert narrow[2]["beta_axis"] == "exact_entry_only"
    assert wide[2]["t_max"] > narrow[2]["t_max"]
    assert len(wide[0]) > len(narrow[0])
    # The narrow rule certifies where it was built and nowhere below.
    assert gen.certify(*narrow[:2], R, beta, tol)["passes"]
    assert not gen.certify(narrow[0], np.abs(narrow[1]), R, 0.0,
                           tol)["passes"]


def test_positive_composite_amplification_is_one_by_construction():
    """And a red twin: break positivity and the bound goes with it."""

    R, beta, tol = 46.41588833612777, 16.006, 1.0e-6
    t, w, info = gen.composite_rule(R, beta, tol)
    stats = gen.measure(t, w, R, beta)
    assert stats["kappa0"] == pytest.approx(1.0, abs=1e-6)
    assert stats["sum_abs_w"] == pytest.approx(info["sum_h"], rel=1e-12)

    # The cancelling pair again: it barely moves the fit and doubles
    # the absolute-value sum, which is the failure mode positivity was
    # protecting against in the first place.
    bump = 40.0 * float(np.max(np.abs(w)))
    bad_t = np.concatenate([t, t[:1] * 1.0001, t[:1] * 1.0002])
    bad_w = np.concatenate([w, [bump], [-bump]])
    twin = gen.certify(bad_t, bad_w, R, beta, tol,
                       kappa_max=gen.KAPPA_EXCEPTION)
    assert not twin["passes"]
    assert "kappa0" in twin["failures"]


# ---------------------------------------------------------------------------
# 5. Against the status quo
# ---------------------------------------------------------------------------

def test_the_shipped_entry_beats_the_runtime_solve_on_amplification():
    """The measurement that motivates the whole campaign.

    ``noncrossing_imag_grids`` is what production calls today when no
    table exists.  It is scored here with the identical functional, on
    the identical request, with the disk cache off so that what is
    compared is a solve and not an artefact from April.
    """

    import minimax as production                        # noqa: PLC0415

    os.environ["LORRAX_DISABLE_MINIMAX_DISK_CACHE"] = "1"
    worst_ratio = 0.0
    for entry in _entries()[:4]:
        R, beta = entry["range_max"], entry["beta"]
        t_rt, w_rt, _n, _e = production.noncrossing_imag_grids(
            R, beta, entry["error_bound"], N_start=2, N_max=64)
        runtime = gen.measure(t_rt, w_rt.astype(np.complex128), R, beta)
        print(f"[imag tables] R={R:8.3f} beta={beta:7.3f}: "
              f"table N={entry['node_count']:4d} kappa0="
              f"{entry['kappa0']:.4f} | runtime N={len(t_rt):3d} "
              f"kappa0={runtime['kappa0']:.4e} "
              f"(x{runtime['kappa0'] / entry['kappa0']:.3e})")
        worst_ratio = max(worst_ratio,
                          runtime["kappa0"] / entry["kappa0"])
    assert worst_ratio > 10.0


# ---------------------------------------------------------------------------
# 6. The fit stage's accuracy floor, with a table in the imaginary cell
# ---------------------------------------------------------------------------

def test_the_fit_stage_floor_is_no_longer_a_quadrature():
    """R1's hole was the MPA fit stage's accuracy floor.  It is not now.

    Build note V measured evaluate->fit recovery at 5.3e-08 against a
    synthesis baseline of 3.8e-09 and named
    ``exponential_sum_imag``'s runtime solve as the cause.  Separating
    the cells sharpens that: the worst SAMPLE error at build note V's
    setting belongs to the static ``z = 0`` cell (1.46e-10, against the
    imaginary cell's 5.34e-11), but the RECOVERY is imag-limited
    anyway, because the Pade fit weights its sample points very
    unevenly.  Holding every other cell at 1e-12 and moving only the
    imaginary cell:

        runtime  @1e-10 -> recovery 4.71e-08
        runtime  @1e-12 -> recovery 4.85e-09
        table    @1e-12 -> recovery 2.22e-09   <- below the baseline
        table    @1e-14 -> recovery 4.82e-09   <- bouncing on the floor

    This cell asserts the third row.  What it is really asserting is
    that the recovery has reached the fit's own conditioning (~6.6e8
    here) and stopped being a statement about quadrature at all --
    which is why the assertion is against ``worst_synth`` and not
    against an absolute number that would drift with the fixture.

    AND UNTIL THIS COMMIT IT ALSO CARRIED AN ABSOLUTE 1e-8 THAT DID NOT
    MEAN THAT, which is the registered open row.  The absolute bar asked
    the sampled fit to beat, by nearly an order of magnitude, a floor the
    EXACT fit does not reach on this host: `worst_synth` is 8.65e-08
    here, so `worst_eval < 1e-8` was unsatisfiable no matter how good the
    quadrature got.  It is removed, not widened.  The relative bar that
    was already beside it is the whole assertion now, and it says what
    the paragraph above always claimed it said.

    THE TWO-HOST MEASUREMENT IS WHY THIS IS THE RIGHT SHAPE AND NOT JUST
    THE CONVENIENT ONE.  The synthesis baseline is set by the Pade
    fixture's conditioning, which is a property of the host's LAPACK, and
    it moves by more than a decade between hosts.  Measured:

        this host   recovery 9.98e-08   baseline 8.65e-08   ratio 1.15
        the tables campaign's host
                    recovery 3.33e-09   baseline 3.81e-09   ratio 0.87

    The baselines are 23x apart; the RATIOS agree to within 30%, and the
    absolute 1e-8 bar happens to fall between the two baselines -- which
    is exactly why the same code passed that assertion on one host and
    failed it on the other.  A bar on the ratio is a bar on the thing the
    cell is about: whether the sampled fit has reached the exact fit's
    floor.  A bar on the absolute error is a bar on whose machine it ran.
    """

    from gw.mpa import evaluator, pade_fit, sample_plan   # noqa: PLC0415

    n_p, tol = 8, 1.0e-12
    rng = np.random.default_rng(3)
    delta = np.linspace(0.6, 7.0, n_p)
    weight = 0.2 + 0.6 * rng.random((n_p, 3))
    plan = sample_plan.mpa_plan(n_p, float(delta.max()), energy_unit="Ry")
    z = sample_plan.plan_z(plan)
    original = evaluator.existing_kernel_rule
    built = {}

    def from_table(point, *, delta_min, delta_max, target_error,
                   max_nodes=64, use_shipped_tables=True):
        if point["family"] != "exponential_sum_imag":
            return original(point, delta_min=delta_min,
                            delta_max=delta_max,
                            target_error=target_error,
                            max_nodes=max_nodes,
                            use_shipped_tables=use_shipped_tables)
        R = float(delta_max) / float(delta_min)
        beta = float(point["varpi"]) / float(delta_min)
        t, w, _info = gen.composite_rule(R, beta, tol)
        cert = gen.certify(t, w, R, beta, tol)
        assert cert["passes"], cert["failures"]
        built.update(R=R, beta=beta, n=len(t), kappa0=cert["kappa0"])
        return {"t": t / delta_min, "h": np.real(w) / delta_min,
                "n_nodes": len(t), "family": point["family"],
                "max_error": cert["real_part_max_error"] / delta_min,
                "delta_min": float(delta_min),
                "delta_max": float(delta_max), "target_error": tol}

    evaluator.existing_kernel_rule = from_table
    try:
        values, cost = evaluator.evaluate_samples(
            plan, delta, weight, rel_tol=tol, kernel_target_error=tol)
    finally:
        evaluator.existing_kernel_rule = original

    values = np.asarray(values)
    k = evaluator.damped_kernel(np.asarray(z)[:, None], delta)
    exact = np.tensordot(k, weight, axes=(1, 0))
    sample_error = float(np.max(np.abs(values - exact)))

    worst_eval = worst_synth = cond = 0.0
    for ch in range(weight.shape[1]):
        om_e, b_e, diag = pade_fit.fit_mpa_poles(values[:, ch], z, n_p)
        om_s, b_s, _ = pade_fit.fit_mpa_poles(exact[:, ch], z, n_p)
        worst_eval = max(worst_eval,
                         float(np.max(np.abs(np.asarray(om_e) - delta))),
                         float(np.max(np.abs(np.asarray(b_e)
                                             - weight[:, ch]))))
        worst_synth = max(worst_synth,
                          float(np.max(np.abs(np.asarray(om_s) - delta))),
                          float(np.max(np.abs(np.asarray(b_s)
                                              - weight[:, ch]))))
        cond = max(cond, float(diag["cond_pade"]))

    print(f"[imag tables] fit floor: imag cell served by a composite "
          f"table (R={built['R']:.3f}, beta={built['beta']:.3f}, "
          f"N={built['n']}, kappa0={built['kappa0']:.4f}); sample error "
          f"{sample_error:.3e}, recovery {worst_eval:.3e} against a "
          f"synthesis baseline of {worst_synth:.3e} at Pade condition "
          f"{cond:.2e}; {cost['kernel_nodes']} kernel nodes")
    assert sample_error < 1.0e-11
    # RELATIVE TO THE CELL'S OWN SYNTHESIS BASELINE, and only that.  The
    # absolute 1e-8 that used to sit here was below `worst_synth` on this
    # host and above it on another; see the docstring.  2.0 covers the
    # measured spread of 0.87 to 1.15 with ~1.7x of margin.
    assert worst_eval < 2.0 * worst_synth, (
        f"recovery {worst_eval:.3e} is {worst_eval / worst_synth:.2f}x the "
        f"synthesis baseline {worst_synth:.3e}; the sampled fit has not "
        f"reached the exact fit's floor")


def test_the_fit_stage_tier_is_served_from_the_catalog_through_the_door():
    """The same floor, but with the shipped catalog and selector in it.

    The cell above generates its rule inline, which proves the physics
    and nothing about the bundle.  This one asks the beta selector for a
    ``1e-12`` table the way a driver would, gets one, and runs the fit
    on it -- so what is measured is the artefact that ships.

    THE FIXTURE'S REQUEST IS NOT ON THE GRID, AND THAT IS THE POINT.
    The toy spectrum's transition span gives ``R = 11.67`` and the far
    line gives ``beta = 3.33``, which is no deck's omega-hat and is
    below every tabulated one.  It is served anyway, by the downward
    rounding a ``positive_composite`` entry carries: those nodes were
    chosen without reference to beta and beta enters only as the phase
    ``e^{i beta t_l}``, so re-phasing an entry swept at a larger beta to
    this one is the rule that route would have built here.  That is the
    property the 1e-12 tier was generated to exploit, and this is it
    being exploited.
    """

    # The selector moved into the service with the bundle it reads;
    # reached off the DOOR, which is the production import edge.
    import minimax as _mm                               # noqa: PLC0415
    bs = _mm.beta_selector
    from gw.mpa import evaluator, pade_fit, sample_plan  # noqa: PLC0415

    n_p, tol = 8, gen.FIT_STAGE_ERROR_BOUND
    rng = np.random.default_rng(3)
    delta = np.linspace(0.6, 7.0, n_p)
    weight = 0.2 + 0.6 * rng.random((n_p, 3))
    plan = sample_plan.mpa_plan(n_p, float(delta.max()), energy_unit="Ry")
    z = sample_plan.plan_z(plan)
    original = evaluator.existing_kernel_rule
    picked = {}

    def from_catalog(point, *, delta_min, delta_max, target_error,
                     max_nodes=64, use_shipped_tables=True):
        if point["family"] != "exponential_sum_imag":
            return original(point, delta_min=delta_min,
                            delta_max=delta_max,
                            target_error=target_error,
                            max_nodes=max_nodes,
                            use_shipped_tables=use_shipped_tables)
        R = float(delta_max) / float(delta_min)
        beta = float(point["varpi"]) / float(delta_min)
        got = bs.select(range_value=R, beta=beta, beta_clause=bs.HEIGHT,
                        target_error=tol, max_nodes=100000)
        assert isinstance(got, bs.TableSelection), getattr(
            got, "message", got)
        picked.update(entry=got.entry, n=got.node_count,
                      err=got.certified_error, line=got.one_line())
        return {"t": got.tau / delta_min,
                "h": np.real(got.alpha) / delta_min,
                "n_nodes": got.node_count, "family": point["family"],
                "max_error": got.certified_error / delta_min,
                "delta_min": float(delta_min),
                "delta_max": float(delta_max), "target_error": tol}

    evaluator.existing_kernel_rule = from_catalog
    try:
        values, _cost = evaluator.evaluate_samples(
            plan, delta, weight, rel_tol=tol, kernel_target_error=tol)
    finally:
        evaluator.existing_kernel_rule = original

    values = np.asarray(values)
    k = evaluator.damped_kernel(np.asarray(z)[:, None], delta)
    exact = np.tensordot(k, weight, axes=(1, 0))

    worst_eval = worst_synth = 0.0
    for ch in range(weight.shape[1]):
        om_e, b_e, _ = pade_fit.fit_mpa_poles(values[:, ch], z, n_p)
        om_s, b_s, _ = pade_fit.fit_mpa_poles(exact[:, ch], z, n_p)
        worst_eval = max(worst_eval,
                         float(np.max(np.abs(np.asarray(om_e) - delta))),
                         float(np.max(np.abs(np.asarray(b_e)
                                             - weight[:, ch]))))
        worst_synth = max(worst_synth,
                          float(np.max(np.abs(np.asarray(om_s) - delta))),
                          float(np.max(np.abs(np.asarray(b_s)
                                              - weight[:, ch]))))

    print(f"[imag tables] fit floor THROUGH THE DOOR: {picked['line']}")
    print(f"[imag tables]   recovery {worst_eval:.3e} against a "
          f"synthesis baseline of {worst_synth:.3e}")
    assert picked["entry"]["error_bound"] == tol
    assert picked["entry"]["rule"] == "positive_composite"
    assert picked["entry"]["beta_axis"] == "exact_phase_on_fixed_nodes"
    assert picked["err"] <= tol
    # Relative, for the reason the sibling cell's docstring gives at
    # length: the baseline is host conditioning and moves by a decade
    # between machines, while the ratio does not.
    assert worst_eval < 2.0 * worst_synth, (
        f"recovery {worst_eval:.3e} is {worst_eval / worst_synth:.2f}x the "
        f"synthesis baseline {worst_synth:.3e}")


def test_the_catalog_carries_both_tiers_and_says_who_asks_for_each():
    """Production has one tier and the fit stage has another.

    C sec 2.6 measured that no in-tree request has ever asked for a tier
    other than 1e-6, and build note V measured that the fit stage's floor
    lives at 1e-12.  Both are in the catalog and they are not
    interchangeable: the 1e-6 rows are minimax, because every node there
    is one chi0 build repeated per q and per band pair, and the 1e-12
    rows are composite, because the fit stage evaluates 2n_p samples once
    and buys the beta envelope with the nodes it saves nothing on.
    """

    by_tier = {}
    for entry in _entries():
        by_tier.setdefault(entry["error_bound"], []).append(entry)
    assert set(by_tier) == {1.0e-6, gen.FIT_STAGE_ERROR_BOUND}
    for tier, rows in sorted(by_tier.items()):
        rules = {r["rule"] for r in rows}
        nodes = sorted(r["node_count"] for r in rows)
        print(f"[imag tables] tier {tier:.0e}: {len(rows)} entries, "
              f"rules {sorted(rules)}, nodes {nodes[0]}-{nodes[-1]}, "
              f"kappa0 <= {max(r['kappa0'] for r in rows):.4f}")
        assert all(r["certified"] for r in rows)
        assert all(r["kappa0"] <= gen.KAPPA_EXCEPTION for r in rows)
    # Every 1e-12 row is an envelope: it reaches down to beta = 0, which
    # is what lets an off-grid fit-stage request be served at all.
    for row in by_tier[gen.FIT_STAGE_ERROR_BOUND]:
        assert row["rule"] == "positive_composite"
        assert row["beta_min"] == 0.0
        assert row["beta_axis"] == "exact_phase_on_fixed_nodes"
        assert row["kappa0"] == pytest.approx(1.0, abs=1.0e-6)


# ---------------------------------------------------------------------------
# 7. The generator is deterministic
# ---------------------------------------------------------------------------

def test_the_composite_route_regenerates_byte_identically():
    """No seed, no local optimiser -- so the bytes must repeat exactly.

    The census's most uncomfortable finding is that the same host gave
    two answers to one request four months apart.  A route whose whole
    definition is fixed Gauss-Legendre panels on analytically placed
    edges cannot do that, and this is the assertion of it.
    """

    args = (46.41588833612777, 16.006, 1.0e-6)
    t1, w1, _ = gen.composite_rule(*args)
    t2, w2, _ = gen.composite_rule(*args)
    assert gen.payload_digest(t1, w1) == gen.payload_digest(t2, w2)
    assert hashlib.sha256(t1.tobytes()).digest() == \
        hashlib.sha256(t2.tobytes()).digest()
