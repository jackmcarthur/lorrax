"""Which axes of a shipped quadrature catalog round, and which do not.

WHY THIS MODULE EXISTS.  ``beta_selector`` was written for one
family and encodes that family's answer to this question in its control
flow: ``R`` rounds up, the error bound rounds down, the node count is a
ceiling, and ``beta`` does not round at all.  Those four sentences are
not properties of the selector.  They are properties of
``complex_laplace``, and the moment a second family arrives they have to
be asked again -- ``damped_line`` has no ``beta`` at all, has an axis
(``line_height_ratio``) that must match exactly for a reason unrelated
to any envelope, and has two axes (``n_p`` and ``alpha``) whose rules
DIFFER BY ROUTE within the same family.

So the answer becomes a record rather than control flow.  Each family
declares its axes and the direction each one rounds; the selectors read
the record and do the arithmetic.  A new family is then a declaration
and a payload reader, not a new set of hard-coded comparisons that have
to be re-derived by whoever reads the diff.

WHAT A DIRECTION MEANS, precisely, because "rounds up" is ambiguous
until you say which side is conservative:

``ROUND_UP``
    Serve the smallest tabulated value ``>=`` the request.  Legal only
    where a larger tabulated value certifies a SUPERSET of what was
    asked for -- ``R`` on the interval families (a rule on ``[1, R']``
    with ``R' >= R`` is valid on the requested subinterval) and ``A`` on
    ``damped_line`` (a rule at larger ``A`` certifies a strictly larger
    rectangle, and the request's target is its restriction).
``ROUND_DOWN``
    Serve the largest tabulated value ``<=`` the request.  The error
    bound, and only the error bound: a table certified tighter than
    asked is always admissible.
``CEILING``
    Not an axis to match but a budget to respect: the node count.
``EXACT``
    No rounding is conservative here, so a mismatch refuses.  Two very
    different reasons reach this in the two families and the record
    keeps them apart: ``beta`` because the TARGET IS A DIFFERENT
    FUNCTION at every value, and ``line_height_ratio`` because the far
    line's rows are what the node set was certified against.
``BY_RULE``
    The axis's direction depends on the entry's ``rule``.  This is not a
    special case invented for ``damped_line``: ``complex_laplace``
    already carries it on ``beta`` (exact for ``btv_minimax``, rounding
    down for ``positive_composite``, whose beta dependence is an exact
    unit-modulus phase on beta-independent nodes).  Writing it into the
    record is what stops the second occurrence becoming a second
    special case.

This module holds no numbers of its own and reads no files.  It is the
vocabulary; the selectors are the readers.

WHERE IT LIVES.  This module and the two selectors that read it were
written under ``src/common/``, against a bundle that lived at
``common/minimax_assets``.  The service extraction moved the bundle into
``services/minimax/``, and a rule cannot live one import edge away from
the bytes it governs, so the whole family layer moved with it.  The
public names are ``minimax.family_axes``, ``minimax.beta_selector`` and
``minimax.damped_line_selector``; there are no shims at the old paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- the directions -------------------------------------------------------

ROUND_UP = "round_up"
ROUND_DOWN = "round_down"
CEILING = "ceiling"
EXACT = "exact"
BY_RULE = "by_rule"

DIRECTIONS = (ROUND_UP, ROUND_DOWN, CEILING, EXACT, BY_RULE)


class CatalogCorrupt(ValueError):
    """A shipped entry is malformed, so the artifact is not trustworthy.

    An exception and not a refusal, on purpose (``DESIGN_minimax.md``
    R2): a missing or unparseable field does not mean the entry is
    absent, it means the bundle on disk is not the one that was
    certified.  Skipping such an entry would silently narrow the catalog
    and could serve a neighbour in its place.
    """


@dataclass(frozen=True)
class Axis:
    """One axis of one family, and the direction it rounds."""

    name: str
    direction: str
    why: str
    #: For ``BY_RULE`` axes: the per-rule direction, keyed by the entry's
    #: ``rule`` string.  Empty for every other direction.
    by_rule: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.direction not in DIRECTIONS:
            raise ValueError(
                f"GATE axis_direction: axis {self.name!r} declares "
                f"direction={self.direction!r}, which is not one of "
                f"{DIRECTIONS}. FALSE case: a declared direction, "
                "because a selector that guesses a direction serves a "
                "neighbour.")
        if (self.direction == BY_RULE) != bool(self.by_rule):
            raise ValueError(
                f"GATE axis_by_rule: axis {self.name!r} has "
                f"direction={self.direction!r} and by_rule="
                f"{self.by_rule!r}. FALSE case: by_rule is populated if "
                "and only if the direction is BY_RULE.")


@dataclass(frozen=True)
class FamilyAxes:
    """One family's catalog identity and its complete axis record."""

    family: str
    catalog_file: str
    generator: str
    payload_keys: tuple
    axes: tuple

    def axis(self, name: str) -> Axis:
        for a in self.axes:
            if a.name == name:
                return a
        raise KeyError(
            f"GATE unknown_axis: family {self.family!r} has no axis "
            f"{name!r}. FALSE case: one of "
            f"{[a.name for a in self.axes]}.")

    def direction_for(self, name: str, rule: str | None = None) -> str:
        """The direction, resolved against an entry's ``rule``."""

        a = self.axis(name)
        if a.direction != BY_RULE:
            return a.direction
        if rule is None:
            raise ValueError(
                f"GATE by_rule_needs_rule: axis {name!r} of family "
                f"{self.family!r} rounds differently per route, so it "
                "cannot be resolved without the entry's rule. FALSE "
                f"case: rule is one of {sorted(a.by_rule)}.")
        try:
            return a.by_rule[str(rule)]
        except KeyError:
            raise CatalogCorrupt(
                f"{self.catalog_file}: axis {name!r} has no direction "
                f"for rule={rule!r}; the record knows "
                f"{sorted(a.by_rule)}. This selector will not guess.")


#: ``complex_laplace``, transcribed from the behaviour
#: ``beta_selector`` already implements.  Nothing here changes
#: what that selector does; it writes down what it does.
COMPLEX_LAPLACE = FamilyAxes(
    family="complex_laplace",
    catalog_file="catalog_complex_laplace.json",
    generator="tools/generate_imag_minimax_assets.py",
    payload_keys=("tau", "alpha"),
    axes=(
        Axis("range_max", ROUND_UP,
             "a rule fitted on [1, R'] with R' >= R is still valid on "
             "the requested subinterval"),
        Axis("error_bound", ROUND_DOWN,
             "a table certified tighter than asked is admissible"),
        Axis("node_count", CEILING,
             "the caller's budget, not a property of the table"),
        Axis("beta", BY_RULE,
             "the target 1/(u - i beta) is a DIFFERENT FUNCTION at "
             "every beta, so serving a neighbour is not conservative, "
             "it is wrong -- except on the composite route, where the "
             "beta dependence is an exact unit-modulus phase on "
             "beta-independent nodes",
             by_rule={"btv_minimax": EXACT,
                      "positive_composite": ROUND_DOWN}),
    ),
)


#: ``damped_line``.  Gate zero (``GATE0_DAMPED_LINE_LP.md``) and the
#: campaign plan (``WP9_CAMPAIGN_REPLAN.md`` section 4) are where each of
#: these directions comes from.
DAMPED_LINE = FamilyAxes(
    family="damped_line",
    catalog_file="catalog_damped_line.json",
    generator="tools/generate_damped_line_assets.py",
    payload_keys=("t", "w"),
    axes=(
        Axis("A", ROUND_UP,
             "a rule at larger A certifies a strictly larger "
             "(z-line x Delta-band) rectangle, and the request's target "
             "is its restriction"),
        Axis("error_bound", ROUND_DOWN,
             "a table certified tighter than asked is admissible"),
        Axis("node_count", CEILING,
             "the caller's budget, not a property of the table"),
        Axis("line_height_ratio", EXACT,
             "the far line's rows are what the shared node set was "
             "certified against; a different ratio is a different "
             "rectangle, and nothing in the certificate covers it"),
        Axis("n_p", CEILING,
             "bounded by the entry's n_p_max; the partition nests "
             "strictly, so a smaller n_p is a ROW SUBSET of a shipped "
             "weight table and is never interpolated"),
        Axis("alpha", BY_RULE,
             "the sparse route ships one weight row per (line, exact "
             "partition fraction), and the fraction depends on alpha; "
             "the composite route's nodes are omega-independent and its "
             "weights are an exact unit-modulus phase, so one composite "
             "entry serves every partition and every alpha",
             by_rule={"btv_minimax": EXACT,
                      "positive_composite": "ignored"}),
    ),
)


FAMILIES = {f.family: f for f in (COMPLEX_LAPLACE, DAMPED_LINE)}


def family_axes(name: str) -> FamilyAxes:
    """The axis record for one family, by catalog family name."""

    try:
        return FAMILIES[str(name)]
    except KeyError:
        raise KeyError(
            f"GATE unknown_family: {name!r} has no axis record. FALSE "
            f"case: one of {sorted(FAMILIES)} -- a family without a "
            "record has no declared rounding directions, and a selector "
            "that proceeds anyway is guessing.")


def covers(direction: str, entry_value: float, request: float, *,
           tol: float = 0.0) -> bool:
    """Does a tabulated value on this axis cover a request?

    The one place the words ROUND_UP and ROUND_DOWN turn into a
    comparison.  Both selectors call it, so the two doors cannot drift
    apart on the direction of an inequality -- which is the specific bug
    a per-family record exists to make impossible.  ``EXACT`` is a
    tolerance-band equality, and ``CEILING`` reads the other way round
    (the tabulated value must not EXCEED the request, because the
    request is a budget rather than a demand).
    """

    e, r, t = float(entry_value), float(request), float(tol)
    if direction == ROUND_UP:
        return e + t >= r
    if direction == ROUND_DOWN:
        return e - t <= r
    if direction == CEILING:
        return e <= r
    if direction == EXACT:
        return abs(e - r) <= t
    raise ValueError(
        f"GATE covers_direction: {direction!r} is not a comparable "
        f"direction. FALSE case: one of {(ROUND_UP, ROUND_DOWN, CEILING, EXACT)}"
        " -- BY_RULE must be resolved with FamilyAxes.direction_for "
        "before it reaches a comparison.")
