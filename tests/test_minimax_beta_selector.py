"""The beta axis of the runtime selector, and the three ways it says no.

WHAT THIS SUITE IS FOR.  The ``complex_laplace`` catalog was certified
and then deliberately NOT wired, because the shipped selection rule keys
on ``(R, error bound, node count)`` and nothing else, and a door with no
beta axis will happily hand out a table fitted to a different function.
``GATE0_IMAG_ENVELOPE.md`` sec 9 registered that as a landing row and
``MINIMAX_REQUEST_CENSUS.md`` sec 3.1 stated the reason in one sentence
that this file exists to enforce: *a table at omega-hat' != omega-hat is
not conservative, it is wrong*.

So the assertions come in pairs.  The two requests the census measured
refusing are now served -- named, with their real parameters, at the
precision the deck produces rather than the precision the census
displays.  And every way the door could serve something it should not is
a red twin here: the neighbouring beta, the wrong envelope clause, the
unreadable bundle, the entry whose beta rule it does not recognise.  A
selector that cannot refuse is not a selector, it is a lookup with
opinions.
"""

import json

import numpy as np
import pytest

from common import minimax_beta_selector as bs
from gw import minimax_screening as ms


pytestmark = pytest.mark.skipif(
    bs.load_catalog()[0] is None,
    reason="complex_laplace catalog not staged in this checkout")


#: The two requests ``MINIMAX_REQUEST_CENSUS.md`` sec 2.2 measured as
#: production misses, in the deck's own numbers: ``(label, x_min, R,
#: omega_p)``.  ``omega_hat`` is DERIVED from them here rather than
#: transcribed, because the census displays it to three decimals and
#: three decimals is not the specification -- Gate 0 sec 9 says so, and
#: the deck-sweep test below shows what happens to the decks where the
#: rounding does not happen to fit.
CENSUS_REFUSALS = (
    ("gnppm", 0.124953, 42.6814, 2.0),
    ("bispinor", 0.122161, 38.7749, 2.0),
)


@pytest.fixture(autouse=True)
def _quiet_announcements():
    bs.reset_announcements()
    yield
    bs.reset_announcements()


def _entries():
    doc, _ = bs.load_catalog()
    return doc["tables"]


# ---------------------------------------------------------------------------
# 1. The claim: the two measured refusals are served
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("label,x_min,R,omega_p", CENSUS_REFUSALS,
                         ids=[r[0] for r in CENSUS_REFUSALS])
def test_the_two_census_requests_are_now_served_from_certified_tables(
        label, x_min, R, omega_p):
    """The arming row, through the production door and not around it.

    This calls what ``build_imag_quadrature`` calls, with the deck's own
    interval, and asserts that what comes back is a catalog entry rather
    than four hundred milliseconds of scipy.  The error assertion is
    against the tier the caller asked for -- which is the interesting
    part, because the runtime solve these two requests used to get did
    NOT meet it: the census measured N = 6 rules whose scaled error is
    6.7e-07 and 8.8e-07 but whose amplification is 8.8e+04 and 2.6e+03,
    and it is the amplification that made this a correctness hole rather
    than a cost one.
    """

    x_max = x_min * R
    quad = ms.solve_laplace_minimax_imag_interval(
        x_min, x_max, omega_p, target_error=1.0e-6, max_nodes=64)

    assert ms.LAST_IMAG_TABLE_REFUSAL is None, (
        f"{label} was refused: "
        f"{ms.LAST_IMAG_TABLE_REFUSAL and ms.LAST_IMAG_TABLE_REFUSAL.message}")
    assert bs.PROVENANCE_LOG, "a served table must announce itself"
    line = bs.PROVENANCE_LOG[-1]
    assert "SHIPPED TABLE" in line
    assert "certified=True" in line

    # The physical error convention: scaled error over x_min.  The table
    # meets the requested tier, at the request's own beta.
    assert quad.max_error * x_min <= 1.0e-6
    assert quad.node_count == len(quad.tau) == len(quad.alpha)


@pytest.mark.parametrize("label,x_min,R,omega_p", CENSUS_REFUSALS,
                         ids=[r[0] for r in CENSUS_REFUSALS])
def test_the_served_table_beats_the_solve_it_replaces_on_amplification(
        label, x_min, R, omega_p):
    """Why serving these two is an improvement and not merely a change.

    The functional is ``kappa_0 = max_u u * sum_l |w_l| e^{-t_l u}``, the
    same one the certification harness and ``mpa.evaluator`` use.  The
    census measured the runtime solves at 2.6e+03 and 8.8e+04 against
    shipped ``noncrossing`` tables at 1.00-1.02; the entries these
    requests now land on are stamped at 1.08-1.09.
    """

    picked = bs.select(range_value=R, beta=omega_p / x_min,
                       beta_clause=bs.HEIGHT, target_error=1.0e-6,
                       max_nodes=64)
    assert isinstance(picked, bs.TableSelection)
    assert picked.entry["kappa0"] <= 2.0
    assert picked.entry["kappa0_tier"] == "normal"
    u = np.geomspace(1.0, R, 4001)
    kappa = float(np.max(u * (np.exp(-np.outer(u, picked.tau))
                              @ np.abs(picked.alpha))))
    assert kappa == pytest.approx(picked.entry["kappa0"], rel=1e-3)
    assert kappa < 2.0


def test_the_announcement_is_pinned_and_happens_exactly_once():
    """R2, as a fixed string rather than a hope.

    The provenance line is the whole of what a driver sees, so its
    content is asserted field by field: which rule, whether certified,
    which catalog entry, the request beside the entry it landed on, the
    error re-measured HERE, and how much of the stamped beta band the
    request spent.  And it is announced ONCE -- the door is called per
    rank and per Sigma pass, and a line printed eighty times is a line
    nobody reads.
    """

    printed = []
    for _ in range(4):
        ms.solve_laplace_minimax_imag_interval(
            0.124953, 0.124953 * 42.6814, 2.0,
            target_error=1.0e-6, max_nodes=64,
            print_fn=printed.append)

    assert len(printed) == 1, printed
    assert bs.PROVENANCE_LOG == printed
    line = printed[0]
    for token in (
            "minimax[complex_laplace]: SHIPPED TABLE",
            # The deck's OWN omega-hat, twelve decimals of it, not the
            # census's displayed 16.006.  Before the tables campaign
            # regenerated at full precision this request was served by
            # the rounded entry at 14.7% of its band; it is now served
            # by an entry measured at the request itself.
            "complex_laplace/complex_laplace_R_46p415888_b_16p006018262867"
            "_eps_1p0em06.npz",
            "rule=btv_minimax",
            "certified=True",
            "N=10",
            "[height clause]",
            "beta=16.00601826286684",
            "request at 0.0% of it",
            "re-measured HERE at this beta",
            "exact-match axis; no rounding",
            "catalog complex_laplace/v1 schema 2",
    ):
        assert token in line, f"{token!r} missing from:\n{line}"
    # One line, not a paragraph: this is what a driver prints.
    assert "\n" not in line


# ---------------------------------------------------------------------------
# 2. Red twin -- the neighbouring beta
# ---------------------------------------------------------------------------

def test_red_twin_a_near_miss_refuses_and_names_the_nearest_certified_beta():
    """The refusal the whole staging exists for.

    A beta a few band-widths off a certified entry is the dangerous
    request, not an obviously silly one: it is close enough that a rule
    keyed on R and the tier alone would serve it without a murmur, and
    far enough that the served table is fitted to a different function.
    The FALSE case is directly below: move the same request inside the
    band and it serves.
    """

    entry = min(_entries(), key=lambda e: e["range_max"])
    band = entry["beta_tolerance"]
    R = entry["range_max"]

    refused = bs.select(range_value=R, beta=entry["beta"] + 4.0 * band,
                        beta_clause=bs.HEIGHT, target_error=1.0e-6,
                        max_nodes=64)
    assert isinstance(refused, bs.TableRefusal)
    assert refused.code == "BetaNearMiss"
    assert refused.nearest_beta == entry["beta"]
    assert refused.nearest_band == band
    assert "is not conservative, it is wrong" in refused.message
    assert "reachable by" in refused.message
    assert bs.GENERATOR in refused.message
    # It named the neighbour; it did not serve it.
    assert not isinstance(refused, bs.TableSelection)

    served = bs.select(range_value=R, beta=entry["beta"] + 0.5 * band,
                       beta_clause=bs.HEIGHT, target_error=1.0e-6,
                       max_nodes=64)
    assert isinstance(served, bs.TableSelection)
    assert served.band_fraction == pytest.approx(0.5, rel=1e-9)


def test_the_near_miss_is_a_real_error_and_not_a_bookkeeping_rule():
    """Why the band is narrow: the error really does move.

    Four bands off, the same bytes score above the tier they are
    certified at.  Without this the beta axis would be a convention
    rather than a measurement, and a reviewer would be right to ask why
    the door does not simply round.
    """

    entry = min(_entries(), key=lambda e: e["range_max"])
    picked = bs.select(range_value=entry["range_max"], beta=entry["beta"],
                       beta_clause=bs.HEIGHT, target_error=1.0e-6,
                       max_nodes=64)
    assert isinstance(picked, bs.TableSelection)
    at_entry, _ = bs.measure_at(picked.tau, picked.alpha,
                                entry["range_max"], entry["beta"])
    off, _ = bs.measure_at(picked.tau, picked.alpha, entry["range_max"],
                           entry["beta"] + 4.0 * entry["beta_tolerance"])
    assert at_entry <= entry["error_bound"]
    assert off > entry["error_bound"]


#: The census's x_min column, and the R each deck asks for.  omega-hat is
#: DERIVED as ``2 Ry / x_min`` rather than transcribed, for the reason
#: this file's header gives: three decimals is a display, not a
#: specification.
DECK_SWEEP = {
    "hbn": (0.342664, 14.19), "cohsex": (0.141617, 37.12),
    "gnppm": (0.124953, 42.68), "bispinor": (0.122161, 38.77),
    "si_fast": (0.050916, 33.18), "si_test": (0.050916, 72.75),
    "si_bse": (0.049734, 74.48),
}


def test_the_full_precision_deck_sweep_is_served_end_to_end():
    """Gate 0 sec 9's warning, executable -- and now discharged.

    THE HISTORY IS THE POINT AND IS KEPT HERE ON PURPOSE.  The catalog's
    first beta grid was generated at the census's DISPLAYED three
    decimals.  Scored at ``2 Ry / x_min``, three decks landed inside
    their entry's stamped band and four missed: hbn by 32 bands, cohsex
    by 4.2, si_fast and si_test by 2.9.  Nothing was wrong with the
    selector then -- it was making visible exactly what serving the
    neighbour would have hidden, which is what a band is for.

    The tables campaign then regenerated at the deck's own omega-hat,
    and this cell is the other end of that: all seven decks served, each
    by an entry whose beta IS the request's to the last digit the census
    carries, so every band fraction is ~0 rather than the 98.7% that
    ``bispinor`` was spending against the rounded grid.
    """

    served, refused = {}, []
    for name, (x_min, R) in sorted(DECK_SWEEP.items()):
        got = bs.select(range_value=R, beta=2.0 / x_min,
                        beta_clause=bs.HEIGHT, target_error=1.0e-6,
                        max_nodes=64)
        if isinstance(got, bs.TableRefusal):
            refused.append((name, got.code, got.message.splitlines()[0]))
            continue
        served[name] = got
        print(f"[beta selector] {name:9s} R={R:7.2f} "
              f"omega_hat={2.0 / x_min!r} -> {got.entry['file'].split('/')[-1]}"
              f" N={got.node_count:3d} band_frac={got.band_fraction:.3%} "
              f"Re E={got.certified_error:.3e} <= {got.error_bound:.0e}")
    assert not refused, refused
    assert set(served) == set(DECK_SWEEP)
    for name, got in served.items():
        # An entry generated AT the request's beta spends essentially
        # none of its band; anything else means the sweep rounded.
        assert got.band_fraction < 0.02, (name, got.band_fraction)
        assert got.certified_error <= got.error_bound
        assert got.node_count <= 64


def test_the_two_census_production_requests_are_still_the_easy_case():
    """gnppm and bispinor were served before the regeneration too.

    They are the two C sec 2.2 measured refusing at runtime, and they
    are the reason the campaign existed.  Keeping them as their own cell
    means a regression that broke only them could not hide inside the
    seven-deck sweep above.
    """

    for name in ("gnppm", "bispinor"):
        x_min, R = DECK_SWEEP[name]
        got = bs.select(range_value=R, beta=2.0 / x_min,
                        beta_clause=bs.HEIGHT, target_error=1.0e-6,
                        max_nodes=64)
        assert isinstance(got, bs.TableSelection), name
        assert got.entry["rule"] == "btv_minimax"
        assert got.entry["kappa0"] <= 2.0


# ---------------------------------------------------------------------------
# 3. Red twin -- the wrong clause
# ---------------------------------------------------------------------------

def test_the_catalog_clause_stamp_is_proved_against_the_bytes():
    """``CATALOG_CLAUSE`` is a claim about the sweep, so it is checked.

    Every shipped beta must sit inside the height clause's envelope and
    outside the width clause's.  If a later sweep adds width-clause
    entries to this same file, this assertion fails and the stamp has to
    be reconsidered -- which is the point of proving it rather than
    declaring it.
    """

    doc, _ = bs.load_catalog()
    assert bs.CATALOG_CLAUSE[bs.catalog_version(doc)] == bs.HEIGHT
    height = bs.BETA_CLAUSES[bs.HEIGHT]
    width = bs.BETA_CLAUSES[bs.WIDTH]
    for entry in doc["tables"]:
        assert 0.0 <= entry["beta"] <= height.beta_max
        assert entry["beta"] > width.beta_max


def test_red_twin_a_width_clause_request_refuses_a_height_clause_table():
    """The trap: numerically it matches, physically it is a different ask.

    ``beta = 1/3`` is a qualified point of the WIDTH clause, and at
    ``R = 50`` with a generous node budget it falls squarely inside the
    composite entry's ``[beta_min, beta] = [0, 39.284]`` coverage -- so a
    selector that checked only the arithmetic would serve it.  It must
    not.  The composite entry was swept as a sampling line height; a pole
    width over a window edge is a different physical quantity, and the
    two clauses were separated precisely because merging them tabulates
    one consumer's range at the other's bandwidth (Gate 0 sec 2).

    The FALSE case is the same request under the clause it was actually
    swept for, which serves.
    """

    ask = dict(range_value=50.0, beta=1.0 / 3.0, target_error=1.0e-6,
               max_nodes=600)

    # BOTH clauses serve this number now -- the width campaign shipped --
    # and that is a sharper test of discrimination than a refusal was:
    # one arithmetic request, two clauses, two DIFFERENT catalogs, and
    # neither entry is reachable from the other's clause.
    as_width = bs.select(beta_clause=bs.WIDTH, **ask)
    as_height = bs.select(beta_clause=bs.HEIGHT, **ask)
    assert isinstance(as_width, bs.TableSelection)
    assert isinstance(as_height, bs.TableSelection)
    assert as_width.entry["file"] != as_height.entry["file"]
    assert bs.CATALOG_FILES["width"].split(".")[0].endswith("width")
    assert as_width.catalog_version == "complex_laplace_width/v1"
    assert as_height.catalog_version == "complex_laplace/v1"
    # The width entry is swept AT the clause's own edge; the height one
    # is a line-height sweep that merely happens to cover 1/3.
    assert as_width.entry["beta"] == 1.0
    assert as_height.entry["beta"] > 1.0
    assert as_width.node_count < as_height.node_count


def test_a_clause_with_no_catalog_refuses_by_naming_the_one_that_has(
        monkeypatch):
    """The state the MPA-Sigma routing audit found, kept as a red twin.

    Before the width campaign this was the shipped behaviour for every
    MPA Laplace window: ``CATALOG_CLAUSE`` stamped the only catalog as
    the HEIGHT clause end to end, so a width request refused on an empty
    family whether its beta was 0.01 or 0.99 -- 81 432 576 poles, 62.25%
    of the |B| mass, all of them.  The width catalog exists now, so the
    state has to be simulated to keep the refusal tested; what must
    never happen is that a missing width sweep silently serves a height
    entry, because the two ranges overlap here.

    Order matters too, and is asserted with it: the clause is settled
    before the beta match, so a caller is told the important of its two
    problems rather than the one whose obvious fix (generate this beta)
    would produce a table that still could not legitimately serve it.
    """

    real = bs.load_catalog

    def only_height(clause=bs.HEIGHT):
        if clause == bs.WIDTH:
            return None, "no width catalog staged (simulated)"
        return real(clause)

    monkeypatch.setattr(bs, "load_catalog", only_height)
    refused = bs.select(range_value=50.0, beta=0.6, beta_clause=bs.WIDTH,
                        target_error=1.0e-6, max_nodes=64)
    assert refused.code == "WrongClause"
    assert "Gamma_p / x_min" in refused.message
    assert "varpi / x_min" in refused.message
    assert bs.GENERATOR in refused.message
    # Not a beta complaint, and not an absent family.
    assert "no certified table AT beta" not in refused.message
    assert refused.code != "CatalogUnavailable"


def test_a_beta_outside_its_own_clause_refuses_by_name():
    """Both envelopes have edges, and the metals rung is beyond one.

    With ``x_min`` clamped at 1e-12 a gapless deck asks for beta = 2e+12.
    That is the clamp showing through and not a tabulation target, and
    the census sec 4.2 measured what happens today when nobody refuses
    it: silent garbage.  A named refusal is the whole improvement.
    """

    metals = bs.select(range_value=3.7e12, beta=2.0e12,
                       beta_clause=bs.HEIGHT, target_error=1.0e-6,
                       max_nodes=64)
    assert metals.code == "OutsideEnvelope"
    assert "1e-12 clamp" in metals.message

    too_wide = bs.select(range_value=50.0, beta=1.5, beta_clause=bs.WIDTH,
                         target_error=1.0e-6, max_nodes=64)
    assert too_wide.code == "OutsideEnvelope"
    assert "width clause" in too_wide.message

    unknown = bs.select(range_value=50.0, beta=1.0, beta_clause="depth",
                        target_error=1.0e-6, max_nodes=64)
    assert unknown.code == "UnknownClause"


# ---------------------------------------------------------------------------
# 4. Off catalog, and the generator pointer
# ---------------------------------------------------------------------------

def test_the_fit_stage_tier_is_now_tabulated_and_the_budget_is_the_gate():
    """The refusal this cell used to assert has been generated away.

    It read: Gate 0 sec 3 registered the tier axis as three values and
    not two, only 1e-6 was generated, so the fit stage's imaginary cell
    refused and was told what to run.  The tables campaign ran it.  The
    1e-12 tier now exists as positive-composite entries -- 238 to 1420
    nodes at kappa_0 = 1.0000 -- and the refusal that remains is the
    honest one: at a 64-node budget nothing that accurate can fit, and
    the door says node counts rather than pretending the tier is absent.

    Raise the budget and the same request is served, by an entry swept
    at a LARGER beta and re-phased down to this one, which is the
    composite route's envelope doing the work it was generated for: the
    fixture's ``beta = 3.33`` is no deck's omega-hat and is below every
    tabulated one.
    """

    ask = dict(range_value=11.666, beta=3.3333333333333335,
               beta_clause=bs.HEIGHT, target_error=1.0e-12)

    refused = bs.select(max_nodes=64, **ask)
    assert refused.code == "NoCertifiedTable"
    assert "node counts on hand" in refused.message
    assert "the tier rounds DOWN" in refused.message
    assert bs.GENERATOR in refused.message

    served = bs.select(max_nodes=4000, **ask)
    assert isinstance(served, bs.TableSelection), served.message
    assert served.entry["error_bound"] == 1.0e-12
    assert served.entry["rule"] == "positive_composite"
    assert served.entry["beta_min"] == 0.0
    assert served.entry["beta"] > 3.3333333333333335
    assert served.certified_error <= 1.0e-12

    over_R = bs.select(range_value=1.0e4, beta=16.006,
                       beta_clause=bs.HEIGHT, target_error=1.0e-6,
                       max_nodes=64)
    assert over_R.code == "NoCertifiedTable"
    assert "R rounds UP" in over_R.message


def test_the_node_budget_is_a_ceiling_and_says_so():
    """566 composite nodes against a 64-node budget is a refusal.

    Every node on the production chi0 path is one time-domain build
    repeated per q and per band pair, so the budget is a physics cost and
    not a formality.  ``beta = 1.0`` at ``R = 50`` is covered by exactly
    one entry -- the composite, whose interval reaches down to zero --
    and by no minimax entry, so under the default budget the request
    refuses on beta with the composite never in the running, and under a
    budget that can afford it the composite serves.
    """

    ask = dict(range_value=50.0, beta=1.0, beta_clause=bs.HEIGHT,
               target_error=1.0e-6)

    tight = bs.select(max_nodes=64, **ask)
    assert isinstance(tight, bs.TableRefusal)
    assert tight.code == "BetaNearMiss"
    assert tight.nearest_beta == 5.836        # a minimax entry, not the
    assert tight.nearest_band < 1.0e-4        # 566-node composite

    loose = bs.select(max_nodes=600, **ask)
    assert isinstance(loose, bs.TableSelection)
    assert loose.node_count == 566
    assert loose.entry["rule"] == "positive_composite"


def test_the_ranking_prefers_the_cheap_certified_rule_over_the_composite():
    """Where both cover the request, the ten-node rule wins.

    At ``R = 100`` and ``beta = 16.006`` the composite entry's
    ``[0, 39.284]`` coverage includes the request and so does an exact
    minimax entry.  Fifty-six times the node count for the same certified
    tier is not a tie.
    """

    picked = bs.select(range_value=99.0, beta=16.006, beta_clause=bs.HEIGHT,
                       target_error=1.0e-6, max_nodes=600)
    assert isinstance(picked, bs.TableSelection)
    assert picked.entry["rule"] == "btv_minimax"
    assert picked.node_count == 12


# ---------------------------------------------------------------------------
# 5. The composite's downward rounding, and its red twin
# ---------------------------------------------------------------------------

def test_the_composite_entry_is_rephased_at_the_request_not_served_raw():
    """The one axis that rounds, and the reason it is not a shortcut.

    A composite entry's weights are ``h_l e^{i beta_entry t_l}`` on nodes
    chosen without reference to beta.  Serving those bytes at a smaller
    beta would be the very thing this module refuses minimax entries for;
    what rounds is the RULE, so the magnitudes are recovered and the
    unit-modulus phase re-evaluated at the request.  The red twin is the
    shortcut: the same nodes with the shipped phase left alone, which
    misses by orders.
    """

    beta = 5.0
    picked = bs.select(range_value=99.0, beta=beta, beta_clause=bs.HEIGHT,
                       target_error=1.0e-6, max_nodes=600)
    assert isinstance(picked, bs.TableSelection)
    assert picked.entry["beta_axis"] == "exact_phase_on_fixed_nodes"
    assert picked.entry["beta"] != beta

    good, _ = bs.measure_at(picked.tau, picked.alpha,
                            picked.entry["range_max"], beta)
    assert good <= picked.entry["error_bound"]
    assert picked.modulus_error == pytest.approx(good, rel=1e-12)

    raw = np.abs(picked.alpha) * np.exp(1j * picked.entry["beta"]
                                        * picked.tau)
    bad, _ = bs.measure_at(picked.tau, raw, picked.entry["range_max"],
                           beta)
    assert bad > 100.0 * good

    # And positivity -- the property the composite's kappa_0 = 1 rests on
    # -- survives the re-phasing, because only the phase moved.
    assert np.all(np.abs(picked.alpha) > 0.0)
    assert float(np.max(np.abs(np.abs(picked.alpha)
                               - np.abs(raw)))) < 1.0e-15


def test_the_composite_does_not_round_upward():
    """Downward only.  Above its own beta it is an ordinary near miss."""

    doc, _ = bs.load_catalog()
    comp = next(e for e in doc["tables"]
                if e["beta_axis"] == "exact_phase_on_fixed_nodes")
    above = comp["beta"] + 10.0 * comp["beta_tolerance"]
    got = bs.select(range_value=comp["range_max"], beta=above,
                    beta_clause=bs.HEIGHT, target_error=1.0e-6,
                    max_nodes=600)
    assert isinstance(got, bs.TableRefusal)
    assert got.code == "BetaNearMiss"


# ---------------------------------------------------------------------------
# 6. Failing closed on a corrupt or unknown catalog
# ---------------------------------------------------------------------------

def _fake_catalog(mutate):
    doc, _ = bs.load_catalog()
    doc = json.loads(json.dumps(doc))
    mutate(doc)
    return doc


def test_an_unknown_beta_axis_raises_rather_than_guessing(monkeypatch):
    """A rule this selector does not know is not a rule it may ignore."""

    doc = _fake_catalog(lambda d: d["tables"].__setitem__(
        0, {**d["tables"][0], "beta_axis": "envelope_v3"}))
    monkeypatch.setattr(bs, "load_catalog",
                        lambda *a, **k: (doc, None))
    with pytest.raises(bs.CatalogCorrupt, match="beta_axis"):
        bs.select(range_value=doc["tables"][0]["range_max"],
                  beta=doc["tables"][0]["beta"], beta_clause=bs.HEIGHT,
                  target_error=1.0e-6, max_nodes=64)


def test_a_missing_band_field_refuses_the_catalog_not_the_entry(monkeypatch):
    """R2's per-entry row: a malformed entry means a corrupt artifact.

    Skipping it -- which is what the shipped selector's ``continue`` does
    -- would silently narrow the catalog, and a narrowed catalog is
    exactly how a neighbouring beta gets served.
    """

    def drop(d):
        entry = dict(d["tables"][0])
        entry.pop("beta_tolerance")
        d["tables"][0] = entry

    doc = _fake_catalog(drop)
    monkeypatch.setattr(bs, "load_catalog",
                        lambda *a, **k: (doc, None))
    with pytest.raises(bs.CatalogCorrupt, match="beta_tolerance"):
        bs.select(range_value=doc["tables"][0]["range_max"],
                  beta=doc["tables"][0]["beta"], beta_clause=bs.HEIGHT,
                  target_error=1.0e-6, max_nodes=64)


def test_an_unreadable_payload_refuses_by_name(monkeypatch):
    def repoint(d):
        d["tables"][0] = {**d["tables"][0],
                          "file": "complex_laplace/not_a_table.npz"}

    doc = _fake_catalog(repoint)
    monkeypatch.setattr(bs, "load_catalog",
                        lambda *a, **k: (doc, None))
    got = bs.select(range_value=doc["tables"][0]["range_max"],
                    beta=doc["tables"][0]["beta"], beta_clause=bs.HEIGHT,
                    target_error=1.0e-6, max_nodes=64)
    assert got.code == "TableUnreadable"
    assert "not_a_table.npz" in got.message


def test_a_stamped_band_that_does_not_hold_refuses(monkeypatch):
    """The band is a measurement, so it has a FALSE case at runtime too.

    Widen an entry's stamped band by four orders and the request now
    "matches" a beta the rule cannot serve.  The per-request re-measure
    catches it and the table is not served -- which is what stops a
    mis-stamped catalog from becoming a wrong answer.
    """

    def widen(d):
        d["tables"][0] = {**d["tables"][0],
                          "beta_tolerance": 1.0e4 * d["tables"][0][
                              "beta_tolerance"]}

    doc = _fake_catalog(widen)
    monkeypatch.setattr(bs, "load_catalog",
                        lambda *a, **k: (doc, None))
    entry = doc["tables"][0]
    got = bs.select(range_value=entry["range_max"],
                    beta=entry["beta"] + 0.5 * entry["beta_tolerance"],
                    beta_clause=bs.HEIGHT, target_error=1.0e-6,
                    max_nodes=64)
    assert got.code == "BandViolated"
    assert "the band is a measurement" in got.message


def test_a_missing_catalog_refuses_with_the_path_it_tried(monkeypatch):
    """No catalog for EITHER clause -- which is a different failure from
    no catalog for THIS clause, and gets a different name.  The one-clause
    case is ``WrongClause`` and is covered above; this is the family being
    absent altogether."""

    monkeypatch.setattr(bs, "load_catalog",
                        lambda *a, **k: (None,
                                         "cannot read /nowhere: ENOENT"))
    got = bs.select(range_value=42.0, beta=16.006, beta_clause=bs.HEIGHT,
                    target_error=1.0e-6, max_nodes=64)
    assert got.code == "CatalogUnavailable"
    assert "/nowhere" in got.message


# ---------------------------------------------------------------------------
# 7. A/B -- everything the selector refuses is byte-identical to before
# ---------------------------------------------------------------------------

def test_a_refused_request_is_byte_identical_to_the_runtime_solve():
    """The A/B claim, pinned where it can never drift.

    "Everything else unchanged" is asserted against the DEFINITION of
    the pre-change behaviour -- ``noncrossing_imag_grids`` on the
    rounded keys, rescaled by ``x_min`` -- rather than against a
    recorded snapshot, so it keeps meaning something after the fixture
    ages.

    THE FIXTURE HAD TO MOVE ONCE, AND THAT IS WORTH SAYING.  It used to
    be hBN's own request, which missed its entry's band by 32 widths
    back when the catalog's beta grid was the census's three-decimal
    display.  The tables campaign generated at the decks' own
    omega-hats, so hBN is served now and is no longer a refusal to
    measure.  The request here is deliberately BETWEEN two tabulated
    betas instead -- far outside every band, and no deck's -- because
    what this cell pins is the refusal PATH, and that path must keep
    working for requests the catalog was never swept for.
    """

    x_min, R, omega_p = 0.342664, 14.19, 2.0 * (1.0 + 3.0e-3)
    quad = ms.solve_laplace_minimax_imag_interval(
        x_min, x_min * R, omega_p, target_error=1.0e-6, max_nodes=64)
    assert ms.LAST_IMAG_TABLE_REFUSAL is not None
    assert ms.LAST_IMAG_TABLE_REFUSAL.code == "BetaNearMiss"
    assert not bs.PROVENANCE_LOG

    x_max = max(x_min * R, x_min * (1.0 + 1.0e-9))
    tau_hat, w_hat, err_hat = ms._solve_noncrossing_imag_scaled_cached(
        round(float(np.log(x_max / x_min)), 12),
        round(omega_p / x_min, 12), round(1.0e-6, 14), 64)

    assert np.array_equal(quad.tau, tau_hat / x_min)
    assert np.array_equal(quad.alpha, w_hat / x_min)
    assert quad.max_error == err_hat / x_min


def test_the_escape_hatch_still_reaches_the_runtime_solve():
    """``use_shipped_tables=False`` bypasses the catalog entirely.

    R1's staging is untouched by this commit: the hatch that already
    governs the static and crossing paths now governs this one too, and
    with it clear the served request goes back to being solved.
    """

    x_min, R, omega_p = 0.124953, 42.6814, 2.0
    served = ms.solve_laplace_minimax_imag_interval(
        x_min, x_min * R, omega_p, target_error=1.0e-6, max_nodes=64)
    solved = ms.solve_laplace_minimax_imag_interval(
        x_min, x_min * R, omega_p, target_error=1.0e-6, max_nodes=64,
        use_shipped_tables=False)
    assert served.node_count == 10
    assert solved.node_count == 6
    assert not np.array_equal(served.tau, solved.tau)


def test_the_static_and_crossing_selectors_are_untouched():
    """The beta axis is additive: the old door still keys on three fields.

    ``_find_shipped_table_entry`` is what serves ``noncrossing`` and
    ``crossing``, and it must not have learned about beta -- those
    families have no beta and a signature change there would be a
    behaviour change on every default run.
    """

    entry = ms._find_shipped_table_entry(
        "noncrossing", range_value=42.68, target_error=1.0e-6,
        max_nodes=64)
    assert entry is not None
    assert "beta" not in entry
    assert entry["family"] == "noncrossing"
