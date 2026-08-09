"""The ``damped_line`` catalog, and the certificate that admits it.

WHAT A CERTIFICATE IS FOR, AND WHY IT IS TESTED THIS WAY.  The sibling
harness makes the argument in full and it is the same argument here:
the request census measured one host answering the same uncertified
solve two different ways four months apart, under one cache key, so a
generator that merely reports its own numbers reproduces that failure
with extra steps.  Every check below is therefore re-derived HERE, from
the shipped bytes, without asking the generator what it thinks it
produced -- and every check is paired with a red twin, a deliberately
mis-certified rule that the same check must refuse.  A certificate
nothing can fail is not a certificate.

WHAT IS NEW IN THIS FAMILY, and it is what the extra sections are for.
A ``complex_laplace`` entry serves one ``(R, beta)`` request.  A
``damped_line`` entry is ONE node set serving BOTH sampling lines and
EVERY ``n_p``, so it carries two claims with no analogue there -- and
they are exactly the two the shipping rule rests on:

* the FAR line rides the NEAR line's nodes (section 2, check 2), which
  is what makes one entry per cell honest rather than optimistic;
* the shipped weight table is ``n_p``-PROOF (section 4): every
  partition from ``n_p = 2`` to ``N_P_MAX``, on both alpha sets,
  selects rows that are already certified -- no new solve, and no
  interpolation anywhere.

THE CHECKS, AND WHAT EACH ONE IS ACTUALLY GUARDING.

``held_out_error``
    Sup error in complex modulus on a half-cell-offset ``Delta`` grid
    the solver never sampled, BOTH BAND EDGES INCLUDED, at the shipped
    rows' own omegas.  Guards fitting the grid instead of the
    function; the band edge is in it because that is where a sine
    sum's residual is largest and where gate zero measured an early
    midpoint-gridded pass certifying 1.9x over budget.
``far_line``
    The same, restricted to the far line.  One node set, two lines --
    this is the check that says so instead of assuming it.
``kappa0``
    ``varpi * sum_l |w_l| / 2``, re-measured from the shipped weights:
    the amplification an error in the per-node chi0 build would get.
    Gate zero ran this same selection uncapped and produced rules at
    406, 620 and 1267 that MEET their sup-norm tolerance, which is the
    census's cancelling-pair pathology reproduced in this family, so
    section 3 carries one of those specimens as a twin.
``moment``
    ``|int_0^Dmax (fit - target) dDelta|`` with both sides in closed
    form.  The only check that touches no grid at all, so it is the
    one no choice of sample points can game.  Its twin is NOT the
    sibling's: on a SINE basis a node at ``t -> 0`` contributes
    ``~ w Delta t`` and not a constant, so the perturbation that hides
    from the sup norm and shows up in the integral is the one at
    ``Delta_max * t = 2.3311``, where ``(1 - cos(Delta_max t))/t`` --
    the moment picked up per unit of sup norm -- is maximal.
``rescale``
    The family is EXACTLY scale free in ``varpi``, so the catalog's
    ``tau = t/varpi_1``, ``w = w_hat/varpi_1`` convention is an
    identity and this check is correspondingly sharp.
``positive_nodes``, ``positive_weights``
    A negative time node is not a rule of this family at all, and a
    composite entry with a non-positive weight has lost the positivity
    that IS its ``kappa0 = 1`` argument.
``payload_sha256``
    Byte identity of the TABLE.  The cross-platform claim is about the
    shipped bytes and never about re-running the solve; the census is
    the argument for that distinction.
"""

import importlib.util
import json
import os
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_ASSETS = _REPO / "src" / "common" / "minimax_assets"
_CATALOG = _ASSETS / "catalog_damped_line.json"

#: Every failure token ``gen.certify`` can raise.  A twin names the one
#: it means to trip and the ones it must leave silent, because a check
#: that fires for everything is not a check.
_TOKENS = ("held_out_error", "far_line", "kappa0", "moment", "rescale",
           "positive_nodes", "positive_weights")


def _load_generator():
    name = "lorrax_tools_generate_damped_line_assets"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(str(_REPO), "tools",
                           "generate_damped_line_assets.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


gen = _load_generator()

pytestmark = pytest.mark.skipif(
    not _CATALOG.exists(),
    reason="damped_line catalog not staged in this checkout")


def _catalog():
    return json.loads(_CATALOG.read_text(encoding="utf-8"))


def _entries():
    return _catalog()["tables"]


def _load(entry):
    return gen.read_payload(_ASSETS / entry["file"])


def _ids(entries):
    return [f"A{e['A']:.0f}_eps{e['error_bound']:.0e}_{e['rule']}"
            for e in entries]


ENTRIES = _entries() if _CATALOG.exists() else []


def _rows_from_payload(data, a_dim):
    """The certificate's row list, rebuilt FROM THE SHIPPED BYTES.

    The payload stores each row's line height and its exact fraction of
    ``omega_max`` as an integer pair, which is the whole selection rule
    in two arrays: ``omega = (num/den) * omega_max`` reconstructs the
    sample point without consulting the generator at all.  Every
    re-derivation below starts here, so that what is being scored is
    the table as a consumer would read it.
    """

    omega_max = 0.5 * float(a_dim)
    rows = []
    for varpi, num, den in zip(data["row_varpi"],
                               data["row_fraction_num"],
                               data["row_fraction_den"]):
        frac = Fraction(int(num), int(den))
        rows.append({
            "line": "near" if float(varpi) == 1.0 else "far",
            "varpi": float(varpi),
            "fraction": frac,
            "z": complex(float(frac) * omega_max, float(varpi)),
        })
    return rows


def _row_index(rows):
    """``(varpi, exact fraction) -> row position``.  No interpolation."""

    return {(r["varpi"], r["fraction"]): i for i, r in enumerate(rows)}


def _first(rule=None):
    for entry in _entries():
        if rule is None or entry["rule"] == rule:
            return entry
    pytest.skip(f"no {rule} entry in the catalog")


def _first_composite():
    """A composite entry, preferring one from the composite-only tier.

    ``COMPOSITE_ONLY_TIERS`` is where the composite route ships by
    declaration rather than by losing a race, so an entry from it is
    guaranteed to exist whatever the sparse route managed elsewhere --
    and its truncation deficit is the tier's own ``rel_tol/2``, which
    is what lets the alpha-waiver test below quote ``kappa0 = 1``
    to nine figures.
    """

    for entry in _entries():
        if (entry["rule"] == "positive_composite"
                and entry["error_bound"] in gen.COMPOSITE_ONLY_TIERS):
            return entry
    return _first("positive_composite")


def _assert_refuses(cert, token, silent):
    """A twin fails for ITS reason and is silent on the named others.

    Specificity is the whole content of a red twin.  A perturbation
    that trips every check at once proves only that the certificate
    noticed SOMETHING, which is not what any of these checks claim.
    """

    assert not cert["certified"], cert["failures"]
    assert token in cert["failures"], cert["failures"]
    for other in silent:
        assert other in _TOKENS
        assert other not in cert["failures"], (token, cert["failures"])


# ---------------------------------------------------------------------------
# 1. The catalog's own shape
# ---------------------------------------------------------------------------

def test_the_catalog_covers_the_span_and_tier_ladder():
    """Eight spans x four tiers, every cell certified.

    The A ladder is not a sweep for its own sake: it is C sec 7's deck
    stand-ins, build note VI's enlarged silicon deck (A = 99.1, served
    exactly by the A = 100 rung, which is why rounding UP is the
    selection rule) and P sec E's ``fit_line_global`` rungs.  Asserting
    the product is complete is asserting that no cell was quietly
    dropped when its solve got expensive -- the failure mode a ledger
    of attempts makes easy and a shipped catalog must not have.
    """

    doc = _catalog()
    assert doc["schema_version"] == 2
    assert doc["family"] == "damped_line"
    got_a = sorted({e["A"] for e in doc["tables"]})
    got_e = sorted({e["error_bound"] for e in doc["tables"]})
    assert got_a == sorted(gen.SPANS)
    assert got_e == sorted(gen.TIERS)
    assert len(doc["tables"]) == len(got_a) * len(got_e)
    assert all(e["certified"] for e in doc["tables"])
    assert doc["sweep"]["entries"] == len(doc["tables"])
    assert doc["sweep"]["certified"] == len(doc["tables"])
    assert len(doc["sweep"]["ledger"]) == len(doc["tables"])


def test_every_entry_declares_a_known_rule_and_a_shipping_tier():
    """The vocabulary of the schema, and the one tier that is a policy.

    ``sample_axis`` is the field a consumer dispatches on, so it may
    only carry the two tokens the selection rule defines: a sparse
    entry serves EXACT partition fractions, a composite entry serves
    any omega in closed form.  And ``COMPOSITE_ONLY_TIERS`` is a
    declaration rather than an outcome -- the sparse instrument's wall
    is at 1e-10 and the composite is closed form, self-certifying and
    costs seconds -- so at 1e-12 the rule is fixed in advance and this
    asserts the catalog kept that promise.
    """

    for entry in _entries():
        assert entry["family"] == "damped_line"
        assert entry["range_param"] == "A"
        assert entry["rule"] in ("btv_minimax", "positive_composite")
        assert entry["kappa0_tier"] in ("normal", "versioned_exception")
        assert entry["sample_axis"] in ("exact_fraction_match",
                                        "closed_form_any_omega")
        assert entry["line_height_ratio"] == gen.LINE_RATIO == 10.0
        assert entry["kappa0"] <= entry["kappa0_bound"] == gen.KAPPA_MAX
        if entry["error_bound"] in gen.COMPOSITE_ONLY_TIERS:
            assert entry["rule"] == "positive_composite"
        if entry["rule"] == "btv_minimax":
            assert entry["sample_axis"] == "exact_fraction_match"
            assert entry["n_p_max"] == gen.N_P_MAX
            assert entry["alpha_sets"] == list(gen.ALPHAS)
        else:
            assert entry["sample_axis"] == "closed_form_any_omega"


def test_clause_iii_of_the_shipping_rule_holds_on_every_entry():
    """A sparse entry ships only if it BEATS the rules it replaces.

    Clause (iii) is what makes a stalled prune harmless: the fallback
    is a certified rule of the same family at the same cell, so the
    catalog can never be worse than what the pipeline spends today.
    The baseline is the honest one -- ONE COMPOSITE RULE PER LINE,
    which is what ``evaluate_samples(batching='per-line')`` calls --
    and the composite entries are checked against their own node count
    rather than against the stamp, because a composite entry ships the
    NEAR line's rule and the far line then rides it.
    """

    for entry in _entries():
        a, tol = entry["A"], entry["error_bound"]
        assert entry["composite_node_count"] == (
            entry["composite_near_nodes"] + entry["composite_far_nodes"])
        if entry["rule"] == "btv_minimax":
            assert entry["node_count"] < entry["composite_node_count"]
            assert entry["compression_vs_composite"] > 1.0
        else:
            near = gen.composite_entry_rule(a, tol)
            assert entry["node_count"] == near["n_nodes"]
            assert entry["node_count"] == entry["composite_near_nodes"]
        assert entry["compression_vs_composite"] == pytest.approx(
            entry["composite_node_count"] / entry["node_count"],
            rel=1.0e-12)


# ---------------------------------------------------------------------------
# 2. The checks, re-derived from the shipped bytes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("entry", ENTRIES, ids=_ids(ENTRIES))
def test_shipped_entry_recertifies_from_its_own_bytes(entry):
    """Every stamped number, recomputed off disk and compared.

    The rows are rebuilt from the payload's own fraction pairs and
    then cross-checked against ``sample_rows``, so a table whose
    weights had drifted out of step with its row labels -- the one
    corruption a digest over ``(t, w)`` alone would not catch, because
    the row labels are not in the digest -- fails here.  The digest is
    compared EXACTLY and the measured floats to ``rel=1e-9``: the
    bytes are a claim about identity, the reductions over them are a
    claim about arithmetic.
    """

    data = _load(entry)
    rows = _rows_from_payload(data, entry["A"])
    expect = gen.sample_rows(entry["A"])
    assert len(rows) == len(expect) == entry["n_row"]
    for got, want in zip(rows, expect):
        assert got["line"] == want["line"]
        assert got["varpi"] == want["varpi"]
        assert got["fraction"] == Fraction(want["fraction_num"],
                                           want["fraction_den"])
        assert got["z"] == want["z"]

    cert = gen.certify(data["t"], data["w"], rows, entry["A"],
                       entry["error_bound"], rule=entry["rule"],
                       h=data.get("h"))
    assert cert["certified"], cert["failures"]
    assert cert["node_count"] == entry["node_count"]
    assert cert["n_rows"] == entry["n_row"]
    assert cert["held_out_points"] == entry["held_out_points"]
    assert cert["payload_sha256"] == entry["payload_sha256"]
    assert cert["max_error"] == pytest.approx(entry["max_error"],
                                              rel=1.0e-9)
    assert cert["kappa0"] == pytest.approx(entry["kappa0"], rel=1.0e-9)
    assert cert["moment_residual"] == pytest.approx(
        entry["moment_residual"], rel=1.0e-9)
    assert cert["rescale_max_error_ratio"] == pytest.approx(
        entry["rescale_max_error_ratio"], rel=1.0e-9)
    assert cert["max_error_over_budget"] <= 1.0
    assert cert["moment_residual"] <= cert["moment_ceiling"]
    assert np.all(data["t"] > 0.0)


@pytest.mark.parametrize("entry", ENTRIES, ids=_ids(ENTRIES))
def test_the_far_line_rides_the_same_node_set(entry):
    """Check 2: one node set, two lines, measured on the far one alone.

    This is the analogue of the sibling's real-part check, and it is
    the load-bearing one for this family's whole economy.  A composite
    entry ships the NEAR line's rule -- its truncation is sized at
    ``varpi_1``, so on the far line the tail is ``e^{-10 t_max}``
    instead of ``e^{-t_max}``, ten times the decades it needs, while
    its panels are sized on ``max|omega| + Delta_max``, the same
    angular bandwidth on both lines.  That is an argument, not a
    measurement, so here is the measurement: the far rows are scored
    on their OWN budget (``rel_tol/varpi``, ten times tighter in
    absolute terms than the near line's) with the near line's nodes.
    """

    data = _load(entry)
    rows = _rows_from_payload(data, entry["A"])
    far = [i for i, r in enumerate(rows) if r["line"] == "far"]
    near = [i for i, r in enumerate(rows) if r["line"] == "near"]
    assert far and near

    only_far = gen.certify(data["t"], data["w"][far],
                           [rows[i] for i in far], entry["A"],
                           entry["error_bound"], rule=entry["rule"],
                           h=data.get("h"))
    assert only_far["far_line_max_over_budget"] == pytest.approx(
        entry["far_line_max_over_budget"], rel=1.0e-9)
    assert entry["far_line_max_over_budget"] <= 1.0
    assert only_far["max_error_over_budget"] <= 1.0
    assert "far_line" not in only_far["failures"]
    # ...and the near line is inside its own budget on the same nodes,
    # which is what makes the pair of numbers a statement about ONE
    # rule rather than two coincidences.
    assert entry["near_line_max_over_budget"] <= 1.0


@pytest.mark.parametrize("entry", ENTRIES, ids=_ids(ENTRIES))
def test_the_amplification_metric_agrees_with_the_shipped_catalog(entry):
    """kappa0, recomputed from the weight table with no generator help.

    ``kappa0 = varpi * sum_l |w_l| / 2`` is normalised against the
    continuum kernel's own L1 mass ``2/varpi``, the same normalisation
    ``gw.mpa.evaluator.damped_line_rule`` reports, so ``kappa0 = 1`` is
    the exact rule and ``kappa0 <= 2`` means "at most twice the
    amplification the kernel itself has".  It is computed here as a
    row-wise reduction over the shipped complex weights -- three lines
    of numpy against a stamped number -- because the failure it guards
    is a rule that passes every sup-norm test while amplifying the
    per-node chi0 error by three orders.
    """

    data = _load(entry)
    varpi = np.asarray(data["row_varpi"], dtype=np.float64)
    sum_abs = np.sum(np.abs(np.asarray(data["w"])), axis=1)
    kappa = 0.5 * varpi * sum_abs
    assert float(kappa.max()) == pytest.approx(entry["kappa0"], rel=1e-9)
    assert float(kappa.min()) == pytest.approx(entry["kappa0_min"],
                                               rel=1.0e-9)
    assert float(sum_abs.max()) == pytest.approx(entry["sum_abs_w"],
                                                 rel=1.0e-9)
    assert float(kappa.max()) <= gen.KAPPA_MAX == 2.0
    if entry["rule"] == "positive_composite":
        # Positivity is this route's entire stability argument: the
        # weights are a unit-modulus phase on positive quadrature
        # weights, so |w| is the quadrature weight and kappa0 is the
        # rule's own integral of the damping envelope, which is 1
        # minus the truncated tail and cannot exceed 1.
        assert np.all(data["h"] > 0.0)
        assert float(kappa.max()) <= 1.0 + 1.0e-12


# ---------------------------------------------------------------------------
# 3. The red twins.  One per check.
# ---------------------------------------------------------------------------

def _cancelling_pair(t, w_rows, rows, kappa_target):
    """Two nodes at all but the same ``t``, carrying opposite weights.

    The perturbation the census named: a pair whose contributions
    cancel to a part in ``1e6`` on every sample, so the FIT is
    untouched, while ``sum_l |w_l|`` gains ``2 * bump`` on every row.
    The bump is calibrated against the WORST line height, since
    ``kappa0`` maximises over rows and ``varpi`` multiplies it.
    """

    varpi_max = max(r["varpi"] for r in rows)
    bump = float(kappa_target) / varpi_max
    t_a = 1.0e-8
    bad_t = np.concatenate([[t_a, t_a * (1.0 + 1.0e-6)],
                            np.asarray(t, dtype=np.float64)])
    col = np.full((np.shape(w_rows)[0], 1), bump)
    bad_w = np.concatenate([col, -col, np.asarray(w_rows)], axis=1)
    return bad_t, bad_w


def test_red_twin_inflated_weight_fails_the_amplification_check():
    """A weight pair that cancels on the target still amplifies.

    ``+c`` and ``-c`` at two nodes a part in ``1e6`` apart move the fit
    by nothing a sup norm can see and add ``2c`` to the L1 mass of
    every row, which is precisely the shape of the near-cancellation
    the census measured at ``sum|w| = 8.8e4``.  The twin is scored at a
    modest ``kappa0 ~ 12`` deliberately: the check is a THRESHOLD at 2,
    not a detector of the outrageous, and a twin six times over the
    line is the one that proves the threshold rather than the sign.
    """

    entry = _first()
    data = _load(entry)
    rows = _rows_from_payload(data, entry["A"])
    bad_t, bad_w = _cancelling_pair(data["t"], data["w"], rows, 12.0)
    cert = gen.certify(bad_t, bad_w, rows, entry["A"],
                       entry["error_bound"])
    _assert_refuses(cert, "kappa0",
                    silent=("held_out_error", "far_line", "moment",
                            "rescale", "positive_nodes"))
    assert cert["kappa0"] > gen.KAPPA_MAX
    assert cert["kappa0"] == pytest.approx(12.0, rel=0.5)
    # Specificity, stated as a number: the fit really is unchanged.
    assert cert["max_error"] == pytest.approx(entry["max_error"],
                                              rel=1.0e-3)


def test_red_twin_moved_node_fails_the_held_out_error_check():
    """A five percent shift in one node, and the sup norm sees it.

    The node set is what the ``G_c(tau) G_v(tau)`` product is built on,
    so a table whose nodes disagreed with the build's would be exactly
    this perturbation.  The specificity claim is the interesting half:
    the WEIGHTS are untouched, so ``sum_l |w_l|`` is untouched, and
    ``kappa0`` stays silent -- the two checks are measuring different
    things, which is why both are shipped.
    """

    entry = _first()
    data = _load(entry)
    rows = _rows_from_payload(data, entry["A"])
    bad_t = np.asarray(data["t"], dtype=np.float64).copy()
    bad_t[bad_t.size // 2] *= 1.05
    cert = gen.certify(bad_t, data["w"], rows, entry["A"],
                       entry["error_bound"])
    _assert_refuses(cert, "held_out_error",
                    silent=("kappa0", "positive_nodes"))
    assert cert["max_error_over_budget"] > 10.0


def test_red_twin_flipped_byte_fails_the_digest():
    """The stamp is over the object that SHIPS, in both its arrays."""

    entry = _first()
    data = _load(entry)
    shipped = (data["h"] if entry["rule"] == "positive_composite"
               else data["w"])
    assert gen.payload_digest(data["t"], shipped) == \
        entry["payload_sha256"]
    bad = np.asarray(shipped).copy()
    head = np.unravel_index(0, bad.shape)
    bad[head] = (np.nextafter(bad[head].real, np.inf)
                 + 1j * bad[head].imag) if np.iscomplexobj(bad) else \
        np.nextafter(bad[head], np.inf)
    assert gen.payload_digest(data["t"], bad) != entry["payload_sha256"]
    # And the digest is over BOTH arrays, not just the weights: a node
    # set that had drifted in its last ulp is a different rule.
    bad_t = np.asarray(data["t"], dtype=np.float64).copy()
    bad_t[-1] = np.nextafter(bad_t[-1], np.inf)
    assert gen.payload_digest(bad_t, shipped) != entry["payload_sha256"]


def test_certify_refuses_a_negative_node():
    """``t < 0`` is not a rule of this family at any error.

    ``e^{i z t}`` with ``Im z > 0`` grows on the negative axis, so a
    negative node is not a badly placed node -- it is a divergent
    integrand, and the check is a type check rather than a tolerance.
    Specificity: the weights are untouched, so ``kappa0`` is silent.
    """

    entry = _first()
    data = _load(entry)
    rows = _rows_from_payload(data, entry["A"])
    bad_t = np.asarray(data["t"], dtype=np.float64).copy()
    bad_t[0] = -bad_t[0]
    cert = gen.certify(bad_t, data["w"], rows, entry["A"],
                       entry["error_bound"])
    _assert_refuses(cert, "positive_nodes", silent=("kappa0",))


def test_red_twin_constant_offset_fails_the_moment_identity():
    """The grid-free check catches what a grid-blind rule hides.

    THE SIBLING'S TWIN DOES NOT PORT, and the reason is worth stating.
    There, a node at ``t -> 0`` contributes an almost constant
    ``epsilon`` across the interval, invisible in the sup norm and
    lethal in the integral.  Here the atoms are ``sin(Delta t)``, so a
    node at ``t -> 0`` contributes ``~ epsilon Delta t`` -- it vanishes
    in BOTH.  The perturbation that hides from the sup norm and shows
    in the integral is the one maximising the moment picked up per unit
    of sup norm, ``(1 - cos(Delta_max t))/t`` against ``max_Delta
    |sin(Delta t)| = 1``, whose maximum sits at ``Delta_max t =
    2.3311`` and is worth ``0.7246 * Delta_max`` per unit weight.  At
    ``epsilon = 0.4`` of each row's own budget that is ``1.16`` times
    the moment ceiling while the sup norm is at ``0.41`` of budget --
    which is the entire point: the ONLY check refusing it integrates.

    The base rule is the composite one built two decades TIGHTER than
    the budget it is scored against, so the twin is a controlled
    experiment -- the same nodes certify, and adding one atom is the
    only difference between the two calls below.
    """

    entry = _first()
    a, tol = entry["A"], entry["error_bound"]
    rows = gen.sample_rows(a)
    z = np.asarray([r["z"] for r in rows], dtype=np.complex128)
    rule = gen.composite_entry_rule(a, tol / 100.0)
    w = np.stack([gen.composite_weights(rule["t"], rule["h"], zz)
                  for zz in z], axis=0)
    control = gen.certify(rule["t"], w, rows, a, tol)
    assert control["certified"], control["failures"]

    eps = 0.4 * tol / z.imag                  # 0.4 of each row's budget
    bad_t = np.concatenate([[2.3311 / (0.5 * a)], rule["t"]])
    bad_w = np.concatenate([eps[:, None], w], axis=1)
    cert = gen.certify(bad_t, bad_w, rows, a, tol)
    _assert_refuses(cert, "moment",
                    silent=("held_out_error", "far_line", "kappa0",
                            "rescale", "positive_nodes"))
    assert cert["max_error_over_budget"] <= 0.6
    assert cert["moment_residual"] > cert["moment_ceiling"]
    assert cert["moment_residual"] < 2.0 * cert["moment_ceiling"]


def test_red_twin_from_the_gate_zero_specimen():
    """The census's cancelling-pair pathology, at gate zero's numbers.

    THIS IS THE TWIN THE CAP EXISTS FOR.  ``GATE0_DAMPED_LINE_LP.md``
    ran this same selection with no amplification cap and produced
    REAL rules -- not constructions -- at ``kappa0`` of 406, 620 and
    1267 which MEET their sup-norm tolerance and would amplify any
    error in the per-node chi0 build by three orders.  They are the
    census's cancelling-pair pathology reproduced in this family, and
    they are why every entry carries a ``kappa0`` re-measured from its
    own shipped weights instead of a node count and a sup norm.

    The specimen is synthesised faithfully: a near-cancelling pair
    scaled so the L1 mass lands in gate zero's range, added to a rule
    that certifies.  What must be true of it is BOTH halves -- the sup
    norm still passes (``held_out_error`` silent, and the measured
    error equal to the shipped entry's to a part in a thousand), and
    the certificate refuses it anyway.  A harness that only asserted
    the refusal would not have shown that the refusal is the only
    thing standing between this rule and the pipeline.
    """

    entry = _first()
    data = _load(entry)
    rows = _rows_from_payload(data, entry["A"])
    bad_t, bad_w = _cancelling_pair(data["t"], data["w"], rows, 620.0)
    cert = gen.certify(bad_t, bad_w, rows, entry["A"],
                       entry["error_bound"])
    _assert_refuses(cert, "kappa0",
                    silent=("held_out_error", "far_line", "moment",
                            "rescale", "positive_nodes"))
    assert 400.0 < cert["kappa0"] < 1300.0
    assert cert["max_error"] == pytest.approx(entry["max_error"],
                                              rel=1.0e-3)
    assert cert["max_error_over_budget"] <= 1.0
    print(f"[damped_line] gate-zero specimen at A={entry['A']:.0f}, "
          f"eps={entry['error_bound']:.0e}: kappa0 "
          f"{cert['kappa0']:.1f} (shipped {entry['kappa0']:.4f}), sup "
          f"error {cert['max_error_over_budget']:.4f} of budget -- "
          f"refused on {cert['failures']}")


# ---------------------------------------------------------------------------
# 4. The family's own two checks: n_p-proofness and the alpha waiver
# ---------------------------------------------------------------------------

_N_P = tuple(range(2, gen.N_P_MAX + 1))


@pytest.mark.parametrize("n_p", _N_P, ids=[f"np{n}" for n in _N_P])
def test_n_p_proofness_from_the_bytes(n_p):
    """Every partition the schedules ask for is ALREADY certified.

    This is the claim that makes one entry per cell serve a whole
    ``n_p`` scan: the published partition nests strictly, so
    ``partition_fractions(n)`` is a subset of
    ``partition_fractions(n+1)``, and a table shipped at ``n_p_max``
    serves every smaller ``n_p`` by SELECTING ROWS.  The test is in two
    halves and both matter.  First, membership: every fraction this
    ``n_p`` needs, on both alpha sets and both lines, is present in the
    shipped row labels as an EXACT ``Fraction`` -- no nearest match, no
    interpolation, which is what the selection rule promises and what
    P sec B's Cu schedule at ``n_p = 12`` with a scan to 15 will
    exercise.  Second, error: the certificate is re-measured on those
    rows ALONE, because "the union is inside budget" would not imply
    it for a subset if the shipped weights were ever refit per scan.
    """

    entry = _first("btv_minimax")
    data = _load(entry)
    rows = _rows_from_payload(data, entry["A"])
    index = _row_index(rows)
    for alpha in gen.ALPHAS:
        picked = []
        for frac in gen.partition_fractions(n_p):
            f_alpha = frac ** int(alpha)
            for varpi in (1.0, float(gen.LINE_RATIO)):
                if varpi == 1.0 and f_alpha == 0:
                    continue        # z = 0 is the STATIC cell, not this
                key = (varpi, f_alpha)
                assert key in index, (
                    f"n_p={n_p} alpha={alpha} needs fraction "
                    f"{f_alpha} on varpi={varpi} and the shipped table "
                    f"has no such row; the partition is supposed to "
                    f"nest inside n_p_max={gen.N_P_MAX}")
                picked.append(index[key])
        assert len(picked) == len(set(picked))
        cert = gen.certify(data["t"], data["w"][picked],
                           [rows[i] for i in picked], entry["A"],
                           entry["error_bound"])
        assert cert["certified"], (n_p, alpha, cert["failures"])
        assert cert["max_error_over_budget"] <= 1.0
        assert cert["node_count"] == entry["node_count"]


def _n_p_rows(rows, n_p, alpha):
    """Row positions the ``(n_p, alpha)`` partition selects, exactly."""

    index = _row_index(rows)
    picked = []
    for frac in gen.partition_fractions(n_p):
        f_alpha = frac ** int(alpha)
        for varpi in (1.0, float(gen.LINE_RATIO)):
            if varpi == 1.0 and f_alpha == 0:
                continue
            picked.append(index[(varpi, f_alpha)])
    return picked


def test_red_twin_a_partition_row_may_not_be_interpolated():
    """The n_p check's twin: round the omega axis and it collapses.

    ``n_p``-proofness is a claim about ROWS BEING PRESENT, so its twin
    has to be the thing a consumer would do if they were not.  Both
    ways of rounding are scored here -- take the adjacent fraction's
    weights, or linearly interpolate the two that bracket the row --
    against the row's own target, on the row's own nodes.  Each lands
    five orders over budget, which is what "EXACT partition-fraction
    match, never interpolated" is worth as a number, and it is why the
    strict nesting of the published partition is load-bearing rather
    than decorative: the n_p = 8 rows have to BE in the n_p = 16 table,
    not be reconstructible from it.

    The control matters as much as the twin.  The same nodes, with the
    row's OWN weights, certify -- so what fails is the rounding and
    not the table.
    """

    entry = _first("btv_minimax")
    a, tol = entry["A"], entry["error_bound"]
    data = _load(entry)
    rows = _rows_from_payload(data, a)
    near = [i for i, r in enumerate(rows) if r["line"] == "near"]
    assert len(near) >= 3
    lo, mid, hi = near[len(near) // 2 - 1:len(near) // 2 + 2]
    o_lo, o_mid, o_hi = (rows[lo]["z"].real, rows[mid]["z"].real,
                         rows[hi]["z"].real)
    assert o_lo < o_mid < o_hi
    silent = ("far_line", "kappa0", "rescale", "positive_nodes")

    control = gen.certify(data["t"], data["w"][[mid]], [rows[mid]],
                          a, tol)
    assert control["certified"], control["failures"]

    lam = (o_mid - o_lo) / (o_hi - o_lo)
    w_int = ((1.0 - lam) * data["w"][lo] + lam * data["w"][hi])
    twin = gen.certify(data["t"], w_int[None, :], [rows[mid]], a, tol)
    _assert_refuses(twin, "held_out_error", silent=silent)
    assert twin["max_error_over_budget"] > 1.0e3

    swap = gen.certify(data["t"], data["w"][[lo]], [rows[mid]], a, tol)
    _assert_refuses(swap, "held_out_error", silent=silent)
    assert swap["max_error_over_budget"] > 1.0e3
    print(f"[damped_line] rounding the omega axis at A={a:.0f}, "
          f"eps={tol:.0e}: nearest fraction "
          f"{swap['max_error_over_budget']:.2e} x budget, linear "
          f"interpolation {twin['max_error_over_budget']:.2e} x "
          f"budget, own weights {control['max_error_over_budget']:.3f}")


def test_a_partition_only_rule_would_not_have_been_a_cheaper_rule():
    """Why the certificate is taken ONCE, on the whole rectangle.

    The tempting economy is to solve each ``n_p`` scan point's own
    partition and certify that, which is one certificate per point of
    every scan on every deck.  Gate zero priced solving the rectangle
    instead at between -1% and +14% of the partition's node count, and
    this is that finding as an executable statement: on a node set six
    percent smaller than the shipped one, weights fit by least squares
    AGAINST THE n_p = 8 PARTITION ALONE are already over budget on
    that partition -- and no worse on the full continuum sweep than
    they are on the fifteen rows they were fit to.

    So there is no rule that serves one partition and fails the
    continuum: with per-sample weights each row is its own fit, and the
    achievable error is a property of the NODE SET, which is why
    ``n_p``-proofness costs nothing and why the shipped set is
    near-minimal (drop six percent of it and both numbers leave budget
    together, by orders).
    """

    entry = _first("btv_minimax")
    a, tol = entry["A"], entry["error_bound"]
    data = _load(entry)
    rows = _rows_from_payload(data, a)
    t = np.asarray(data["t"], dtype=np.float64)
    keep = np.unique(np.linspace(0, t.size - 1, int(0.94 * t.size))
                     .round().astype(int))
    sub = t[keep]
    assert 0 < sub.size < t.size

    delta = gen.delta_grid(a, 1200)
    d_ev = gen.heldout_delta(a, 2400)
    phi_fit = gen.sine_matrix(delta, sub)
    phi_ev = gen.sine_matrix(d_ev, sub)

    def worst_over_budget(z):
        """Best weights those nodes admit, scored held out."""

        g = gen.damped_kernel(z[:, None], delta[None, :])
        w = np.linalg.lstsq(phi_fit, g.T, rcond=None)[0].T
        resid = w @ phi_ev.T - gen.damped_kernel(z[:, None],
                                                 d_ev[None, :])
        per_row = np.max(np.abs(resid), axis=1) / (tol / z.imag)
        return float(np.max(per_row))

    z_8 = np.asarray([rows[i]["z"] for i in _n_p_rows(rows, 8, 1)],
                     dtype=np.complex128)
    z_cont = gen.solve_points(a)
    on_partition = worst_over_budget(z_8)
    on_continuum = worst_over_budget(z_cont)
    print(f"[damped_line] A={a:.0f}, eps={tol:.0e}: dropping "
          f"{t.size - sub.size} of {t.size} nodes costs "
          f"{on_partition:.3e} x budget on the n_p=8 partition and "
          f"{on_continuum:.3e} x budget on the {z_cont.size}-point "
          f"continuum sweep")
    assert on_partition > 1.0
    assert on_continuum > 1.0
    assert on_continuum <= 10.0 * on_partition


def test_the_composite_alpha_waiver():
    """What licenses ``alpha: IGNORED`` for the composite route.

    A composite entry's nodes are chosen without reference to omega
    and its weights are ``w_l(z) = -2 h_l e^{i z t_l}``, an exact
    unit-modulus phase on those fixed nodes.  So re-phasing to an
    omega the catalog never tabulated is not an approximation -- it is
    the rule that route would have built there.  This test picks
    omegas that are deliberately NOT partition fractions of anything,
    on both lines, and certifies the result.

    ``kappa0`` is the sharp part.  Positivity makes it the rule's own
    integral of the damping envelope, ``1 - e^{-varpi t_max}``, and the
    truncation is sized so that deficit is ``rel_tol/2`` on the near
    line -- so at the composite-only tier this is 1.0 to within 5e-13,
    independent of omega and of node count.  That invariance is the
    waiver: there is no alpha axis to match on because there is no
    quantity that depends on it.
    """

    entry = _first_composite()
    a, tol = entry["A"], entry["error_bound"]
    data = _load(entry)
    assert "h" in data
    omega_max = 0.5 * a
    made = []
    for frac in (0.1234567, 0.4999, 0.7071067811, 0.98765):
        for varpi in (1.0, float(gen.LINE_RATIO)):
            made.append({"line": "near" if varpi == 1.0 else "far",
                         "varpi": varpi,
                         "z": complex(frac * omega_max, varpi)})
    w = np.stack([gen.composite_weights(data["t"], data["h"], r["z"])
                  for r in made], axis=0)
    cert = gen.certify(data["t"], w, made, a, tol,
                       rule="positive_composite", h=data["h"])
    assert cert["certified"], cert["failures"]
    assert cert["node_count"] == entry["node_count"]
    band = max(1.0e-9, tol)
    assert cert["kappa0"] == pytest.approx(1.0, abs=band)
    assert cert["kappa0_min"] == pytest.approx(1.0, abs=band)
    assert cert["kappa0"] <= 1.0 + 1.0e-12
    assert cert["payload_sha256"] == entry["payload_sha256"]


def test_red_twin_a_sparse_entry_carries_no_phase_table():
    """...and the same re-phasing is impossible for the sparse route.

    The waiver above is not a property of the family, it is a property
    of ONE route, and the payload says which: a composite entry ships
    ``h``, the positive quadrature weights the phase is applied to,
    and a ``btv_minimax`` entry ships only the solved complex weights
    ``w``.  There is nothing to re-phase, which is why that route's
    ``sample_axis`` is ``exact_fraction_match`` and its ``alpha_sets``
    are stamped.  The asymmetry is the same one ``complex_laplace``
    already carries on beta.
    """

    sparse = _first("btv_minimax")
    data = _load(sparse)
    assert "h" not in data
    assert sparse["sample_axis"] == "exact_fraction_match"
    composite = _first_composite()
    assert "h" in _load(composite)
    assert composite["sample_axis"] == "closed_form_any_omega"


# ---------------------------------------------------------------------------
# 5. The port is not trusted
# ---------------------------------------------------------------------------

#: ``(varpi, freq_max, rel_tol)`` triples spanning the shipped ladder,
#: including both line heights: the generator builds its entry rule at
#: ``varpi = 1`` and its per-line baseline at ``varpi = LINE_RATIO``,
#: and ``freq_max`` is the span ``A`` in both cases.
_PORT_CELLS = (
    (1.0, 20.0, 1.0e-6),
    (10.0, 20.0, 1.0e-6),
    (1.0, 60.0, 1.0e-8),
    (1.0, 100.0, 1.0e-10),
    (10.0, 200.0, 1.0e-12),
)


def _evaluator():
    """``gw.mpa.evaluator`` FROM THIS CHECKOUT, not from the install.

    The editable install points at whichever tree was installed, and
    the claim being tested is about the two files in THIS one, so the
    checkout's ``src`` goes on the path first.
    """

    src = str(_REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from gw.mpa import evaluator                       # noqa: PLC0415
    return evaluator


@pytest.mark.parametrize("varpi,freq_max,tol", _PORT_CELLS,
                         ids=[f"v{v:.0f}_A{f:.0f}_e{t:.0e}"
                              for v, f, t in _PORT_CELLS])
def test_the_ported_composite_is_the_shipped_evaluator_rule(
        varpi, freq_max, tol):
    """The duplication is a CHECKED fact, not a hope.

    ``damped_line_rule`` exists twice on purpose: once in
    ``gw.mpa.evaluator``, which is what production calls, and once in
    the generator, which is ported so the tool stays free of jax and of
    ``src``.  The sibling generator has the same property for the same
    reason.  What licenses that duplication is this test and nothing
    else -- and it demands BIT identity rather than agreement to a
    tolerance, because the composite entries' shipped bytes ARE the
    output of this function and a digest does not round.
    """

    evaluator = _evaluator()
    ours = gen.damped_line_rule(varpi, freq_max, rel_tol=tol)
    theirs = evaluator.damped_line_rule(varpi, freq_max, rel_tol=tol)
    assert ours["n_nodes"] == theirs["n_nodes"]
    assert ours["n_panels"] == theirs["n_panels"]
    assert ours["orders"] == theirs["orders"]
    assert np.array_equal(ours["t"], theirs["t"])
    assert np.array_equal(ours["h"], theirs["h"])
    assert ours["t_max"] == theirs["t_max"]
    assert ours["a_dim"] == theirs["a_dim"]
    assert ours["kappa0"] == theirs["kappa0"]
    assert gen.payload_digest(ours["t"], ours["h"]) == \
        gen.payload_digest(theirs["t"], theirs["h"])


# ---------------------------------------------------------------------------
# 6. The generator is deterministic
# ---------------------------------------------------------------------------

def test_the_generator_is_deterministic():
    """One cheap cell, rebuilt, byte for byte.

    The census's most uncomfortable finding is that the same host gave
    two answers to one request four months apart.  The cell rebuilt
    here is the composite-only tier at the smallest span: fixed
    Gauss-Legendre panels on analytically placed edges, closed-form
    weights, no seed and no local optimiser, so it CANNOT do that and
    this is the assertion of it.  A sparse cell is deliberately not
    rebuilt -- it is minutes of solve for a claim this makes in a
    second, and the claim the catalog actually makes is about the
    shipped BYTES, which every entry's digest above already carries.
    """

    tol = gen.COMPOSITE_ONLY_TIERS[0]
    a = float(min(gen.SPANS))
    shipped = next((e for e in _entries()
                    if e["A"] == a and e["error_bound"] == tol), None)
    assert shipped is not None, (a, tol)
    entry = gen.build_entry(a, tol, verbose=False)[0]
    assert entry["rule"] == "positive_composite" == shipped["rule"]
    assert entry["node_count"] == shipped["node_count"]
    assert entry["payload_sha256"] == shipped["payload_sha256"]
    assert entry["max_error"] == pytest.approx(shipped["max_error"],
                                               rel=1.0e-12)
    assert entry["kappa0"] == pytest.approx(shipped["kappa0"],
                                            rel=1.0e-12)
