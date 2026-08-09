"""The beta axis of the shipped quadrature catalog, and its refusals.

WHY THIS MODULE EXISTS AT ALL.  ``gw.minimax_screening``'s selection rule
rounds two axes and rounds them both safely: the interval ratio ``R``
rounds UP, because a rule fitted on ``[1, R']`` with ``R' >= R`` is still
valid on the requested subinterval, and the error bound rounds DOWN, so
the served table is at least as accurate as the caller asked for.  The
``complex_laplace`` catalog adds a third axis that does neither.  Its
target is ``1/(u - i beta)``, which is a DIFFERENT FUNCTION at every
``beta``; a table fitted at ``beta'`` and served at ``beta`` is not a
conservative approximation, it is a wrong one, and it fails quietly --
the census (``MINIMAX_REQUEST_CENSUS.md`` sec 3.1) is the document that
first said so and the certification harness
(``tests/test_minimax_imag_tables.py``) is where it is an executable
fact.  So the catalog shipped in its own schema-v2 file rather than in
``catalog.json``: the old door had no beta axis, and giving it these
entries without one would have let it serve a table fitted to a
different function.  This module is that axis.

WHAT A REQUEST IS.  Three numbers and a word: the interval ratio ``R``,
the dimensionless line parameter ``beta``, the error tier, and the CLAUSE
the ``beta`` came from.  The clause is the part that is easy to miss and
the part that does the most work, so it is stated here rather than
implied.  ``beta`` is a ratio, and ``GATE0_IMAG_ENVELOPE.md`` sec 2
measured that two physically unrelated numerators reach this family
through it:

* the **width** clause -- ``Gamma_p / x_min``, a fitted pole width over a
  window edge, which is small compared with the interval it sits in and
  which the theory plan bounds at ``2/3``;
* the **height** clause -- ``varpi / x_min``, a sampling line height over
  a band gap, which is not small at all, because the protocol puts the
  deep line at 1 Ha whatever the material while Si's gap is 0.0497 Ry.
  The census measured 0.58 ... 40.2 there.

Two orders of separation between them is not an error in either number;
it is what happens when a dimensionless ratio is formed from two
unrelated numerators.  The envelope therefore has two clauses and not one
wider one, the two OVERLAP in ``beta`` around 0.6, and a request cannot
be assigned to a clause by looking at its ``beta``.  The caller says
which one it is, because only the caller knows what its numerator was.

WHAT THIS MODULE NEVER DOES.  It never solves, it never falls back, and
it never serves a neighbour.  Every path out of ``select`` is either a
``TableSelection`` carrying one certified entry and its provenance, or a
``TableRefusal`` naming what was asked, what is nearest, and the two
levers that would make the refusal go away -- the physics one and the
generator one.  A refusal that does not tell you how to make it go away
is a crash with better manners (``DESIGN_minimax.md`` sec 3.2).

WHAT THE CALLER DOES WITH A REFUSAL is the caller's business and is
deliberately not decided here.  Today ``solve_laplace_minimax_imag_interval``
falls back to the runtime solve it has always used, so a refusal costs
nothing but an explanation; when R1 stage 2 arms, the same refusal
becomes the error.  That staging is why this module returns a value
rather than raising.

WHERE IT LIVES, AND WHY IT MOVED.  This module was written as
``common.minimax_beta_selector``, against a bundle that then lived at
``common/minimax_assets``.  The service extraction moved the bundle to
this package, and a selection rule cannot live one import edge away from
the bytes it selects from: the two would be free to disagree about which
catalog is the catalog.  So it moved with them, and it resolves its
catalog through ``_catalog``'s own ``ASSET_PACKAGE``/``ASSET_DIR``
rather than through a second copy of the path -- one asset root for the
whole service, named once.  The public name is ``minimax.beta_selector``;
there is no shim at the old path, per the phase's ratchet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import importlib.resources as importlib_resources
import json

import numpy as np

from minimax import family_axes as _axes
from minimax._catalog import ASSET_DIR, ASSET_PACKAGE


#: The catalog's family name, its file, and the campaign that made it.
#: This family's AXIS RECORD.  ``FAMILY``/``CATALOG_FILE``/``GENERATOR``
#: are read off it rather than restated, so which catalog this door opens
#: and which way each of its axes rounds are declared in ONE place
#: (``minimax.family_axes``) for every family, rather than once
#: per door where the second copy drifts.
_AXES = _axes.COMPLEX_LAPLACE
FAMILY = _AXES.family
CATALOG_FILE = _AXES.catalog_file
GENERATOR = _AXES.generator

#: The two clause names.  They are module constants rather than bare
#: strings at the call sites so that a typo is an ImportError and not a
#: silent refusal.
WIDTH = "width"
HEIGHT = "height"


class CatalogCorrupt(ValueError):
    """A shipped entry is malformed, so the artifact is not trustworthy.

    This is an exception and not a refusal on purpose (``DESIGN_minimax.md``
    R2, the per-entry field row): a missing or unparseable field does not
    mean the entry is absent, it means the bundle on disk is not the one
    that was certified.  Skipping such an entry -- which is what the
    ``continue`` in the old selector did -- would silently narrow the
    catalog and could serve a neighbour in its place.
    """


@dataclass(frozen=True)
class BetaClause:
    """One clause of the two-clause envelope, as data.

    ``beta_max`` is the edge of the domain the campaign certified for
    this clause, NOT a bound on what the physics may ask for.  A request
    beyond it is refused with the clause named, which is the difference
    between "we did not tabulate this" and "this cannot be computed".
    """

    name: str
    numerator: str
    beta_max: float
    qualified_at: tuple[float, ...]
    measured_range: tuple[float, float]
    source: str


#: The envelope, from ``GATE0_IMAG_ENVELOPE.md`` sec 2.
#:
#: The width clause is unchanged from the theory plan's section E and is
#: qualified at 1/6, 1/3 and 2/3.  The height clause is the second clause
#: Gate 0 added: a geometric ladder ``{1, 2, 4, 8, 16, 32, 64}`` whose top
#: covers the measured 0.58 ... 40.2 with headroom for the narrower gaps a
#: larger-N_cond deck will produce.  The metals rung is deliberately
#: OUTSIDE both: with ``x_min`` clamped at 1e-12 a metal's beta is 2e12,
#: which is the clamp showing through rather than a tabulation target, and
#: refusing it by name is the whole value of having an envelope at all.
BETA_CLAUSES: dict[str, BetaClause] = {
    WIDTH: BetaClause(
        name=WIDTH,
        numerator="Gamma_p / x_min -- a fitted pole width over a window "
                  "edge (the Sigma-stage Laplace complement)",
        beta_max=2.0 / 3.0,
        qualified_at=(1.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0),
        measured_range=(1.0 / 6.0, 2.0 / 3.0),
        source="MPA_THEORY_PLAN.md T-E, unchanged by GATE0 sec 2",
    ),
    HEIGHT: BetaClause(
        name=HEIGHT,
        numerator="varpi / x_min -- a sampling line height over a band "
                  "gap (the chi0 probe and the MPA fit stage)",
        beta_max=64.0,
        qualified_at=(1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
        measured_range=(0.58, 40.214),
        source="GATE0_IMAG_ENVELOPE.md sec 2, against "
               "MINIMAX_REQUEST_CENSUS.md sec 7.2",
    ),
}

#: Which clause the shipped catalog's beta grid belongs to, keyed by the
#: catalog's own ``target.version``.  This is a stamp on the CATALOG and
#: not on each entry, because the sweep was one sweep: every one of the
#: eighteen betas is a census ``omega_hat = omega_p / x_min`` at
#: ``omega_p = 2 Ry``, which is a sampling line height and nothing else.
#: ``tests/test_minimax_beta_selector.py`` proves the stamp rather than
#: trusting it -- every entry's beta must sit inside the height clause and
#: outside the width clause -- so this table cannot drift away from the
#: bytes it describes.
CATALOG_CLAUSE: dict[str, str] = {"complex_laplace/v1": HEIGHT}


@dataclass(frozen=True)
class TableRefusal:
    """No certified table was served, and this says why in prose.

    ``code`` is for tests and logs; ``message`` is for the human whose run
    just did something slower or less accurate than they expected.  The
    fields after it are the machine-readable form of what the message
    says, so a caller can build its own message without parsing this one.
    """

    code: str
    message: str
    beta: float | None = None
    nearest_beta: float | None = None
    nearest_band: float | None = None
    clause: str | None = None

    def one_line(self) -> str:
        return self.message.strip().splitlines()[0]


@dataclass(frozen=True)
class TableSelection:
    """One certified table, the request it answers, and its provenance.

    Three error numbers, and they are three because they answer three
    different questions.  ``certified_error`` is what this rule ACHIEVES
    on ``Re`` at the REQUEST's beta, re-measured here from the shipped
    bytes -- it is what the consumer of ``alpha.real`` actually gets, and
    it is the one the door reports.  ``modulus_error`` is the same
    quantity in complex modulus, which dominates it and which is the
    metric the catalog's certificate is written in.  ``real_part_max_error``
    is the entry's own stamped number, measured at the ENTRY's beta by the
    generator; it is kept for comparison and is deliberately not the one
    served, because a number measured somewhere else is not a measurement
    of this request.
    """

    tau: np.ndarray
    alpha: np.ndarray
    entry: dict
    clause: str
    beta: float
    range_value: float
    certified_error: float
    modulus_error: float
    error_bound: float
    real_part_max_error: float
    band_fraction: float
    catalog_version: str
    _line: str = field(default="", repr=False)

    @property
    def node_count(self) -> int:
        return int(self.tau.size)

    def one_line(self) -> str:
        """What the driver prints, once, per ``DESIGN_minimax.md`` R2."""

        return self._line


# ---------------------------------------------------------------------------
# The catalog, read once and never repaired
# ---------------------------------------------------------------------------

def _catalog_path():
    return importlib_resources.files(ASSET_PACKAGE).joinpath(
        ASSET_DIR, CATALOG_FILE)


@lru_cache(maxsize=1)
def load_catalog() -> tuple[dict | None, str | None]:
    """Return ``(document, None)`` or ``(None, why_not)``.

    The failure is a VALUE and not a swallowed exception (R2): a run that
    cannot find its own catalog should say which path it tried and what
    the OS said, because the alternative -- today's ``return None`` -- is
    indistinguishable from a request that legitimately has no table.
    """

    try:
        path = _catalog_path()
    except Exception as exc:                       # pragma: no cover
        return None, f"cannot resolve the minimax_assets package: {exc!r}"
    try:
        with path.open("r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except OSError as exc:
        return None, f"cannot read {path}: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"{path} is not valid JSON: {exc}"
    if not isinstance(doc, dict) or doc.get("family") != FAMILY:
        return None, (f"{path} does not declare family={FAMILY!r} "
                      f"(found {doc.get('family')!r} )")
    if int(doc.get("schema_version", 0)) < 2:
        return None, (f"{path} is schema_version="
                      f"{doc.get('schema_version')!r}; the beta axis needs "
                      "2 or later")
    return doc, None


def catalog_version(doc: dict) -> str:
    return str(doc.get("target", {}).get("version", "unknown"))


def _need(entry: dict, index: int, key: str, cast):
    if key not in entry:
        raise CatalogCorrupt(
            f"{CATALOG_FILE} entry {index} ({entry.get('file')!r}) has no "
            f"{key!r} field. A schema-v2 entry carries it; an entry that "
            "does not is a corrupt artifact, not an absent table.")
    try:
        return cast(entry[key])
    except (TypeError, ValueError) as exc:
        raise CatalogCorrupt(
            f"{CATALOG_FILE} entry {index} ({entry.get('file')!r}) has "
            f"{key}={entry[key]!r}, which is not a {cast.__name__}: "
            f"{exc}") from None


def beta_covers(entry: dict, index: int, beta: float) -> bool:
    """Does this entry's certificate reach ``beta``?  The whole rule.

    Two answers, and which one applies is stamped on the entry rather
    than inferred from its numbers:

    ``exact_entry_only``
        A ``btv_minimax`` rule.  Its nodes and weights were solved
        against ``1/(u - i beta)`` at one beta and have no structure in
        beta at all, so the certificate reaches exactly as far as the
        entry's own measured band -- ``beta_tolerance``, which is about
        ``error_bound * (1 + beta**2)`` and is 1.2e-04 at beta = 16 on the
        1e-6 tier.  Scoring such an entry at its NEIGHBOUR's beta raises
        the error by more than two orders, which is the red twin the
        certification harness carries.

    ``exact_phase_on_fixed_nodes``
        A ``positive_composite`` rule.  Its nodes were chosen without
        reference to beta -- they resolve ``e^{-R t}`` near zero and the
        truncation at ``T`` -- and beta enters only as the unit-modulus
        phase ``e^{i beta t_l}``.  Re-phasing it to any SMALLER beta is
        not an approximation; it is the rule that route would have built
        there, measured at kappa_0 = 1.000000 at every beta in its band.
        So this clause rounds DOWN, to ``beta_min``, and only downward.

    Anything else fails closed, because a beta axis whose rule is unknown
    is the exact situation this module was written to prevent.
    """

    axis = str(_need(entry, index, "beta_axis", str))
    band = _need(entry, index, "beta_tolerance", float)
    beta_entry = _need(entry, index, "beta", float)
    if band < 0.0:
        raise CatalogCorrupt(
            f"{CATALOG_FILE} entry {index} has beta_tolerance={band!r}; a "
            "band is a nonnegative width.")
    if axis == "exact_entry_only":
        return abs(float(beta) - beta_entry) <= band
    if axis == "exact_phase_on_fixed_nodes":
        beta_min = _need(entry, index, "beta_min", float)
        return (beta_min - band) <= float(beta) <= (beta_entry + band)
    raise CatalogCorrupt(
        f"{CATALOG_FILE} entry {index} ({entry.get('file')!r}) declares "
        f"beta_axis={axis!r}, which this selector does not know. It will "
        "not guess: the FALSE case is one of 'exact_entry_only' or "
        "'exact_phase_on_fixed_nodes'.")


def measure_at(tau: np.ndarray, alpha: np.ndarray, range_max: float,
               beta: float, n: int = 2001) -> tuple[float, float]:
    """L-infinity error of this rule AT THIS beta, complex and real part.

    The stamped band is a promise that the certificate survives a small
    move in beta.  This is that promise CHECKED, per request, against the
    shipped bytes -- which is the same discipline the certification
    harness applies at generation time and for the same reason: a number
    a generator reports about itself is not evidence, and the census's
    most uncomfortable finding was one host giving two answers to one
    request four months apart.

    The grid is log-spaced with a half-cell offset plus both endpoints, so
    it is disjoint from any grid the solver saw; it is written here rather
    than imported from ``tools/`` because a check that shares code with
    the thing it checks is checking less than it looks.
    """

    lo, hi = 0.0, float(np.log(float(range_max)))
    edges = np.linspace(lo, hi, int(n) + 1)
    u = np.concatenate([[1.0], np.exp(0.5 * (edges[:-1] + edges[1:])),
                        [float(range_max)]])
    fit = np.exp(-np.outer(u, np.asarray(tau, dtype=np.float64))) @ alpha
    want = 1.0 / (u - 1j * float(beta))
    return (float(np.max(np.abs(want - fit))),
            float(np.max(np.abs(np.real(want) - np.real(fit)))))


def _read_payload(entry: dict) -> tuple[np.ndarray, np.ndarray] | str:
    rel = entry.get("file")
    if not isinstance(rel, str) or not rel:
        return "the entry carries no 'file' field"
    try:
        path = importlib_resources.files(ASSET_PACKAGE).joinpath(
            ASSET_DIR, rel)
        with path.open("rb") as fh:
            with np.load(fh, allow_pickle=False) as data:
                keys = set(data.files)
                if not {"tau", "alpha"} <= keys:
                    return (f"{rel} has npz keys {sorted(keys)}; a "
                            "complex_laplace table carries 'tau' and "
                            "'alpha'")
                tau = np.asarray(data["tau"], dtype=np.float64)
                alpha = np.asarray(data["alpha"], dtype=np.complex128)
    except OSError as exc:
        return f"cannot read {rel}: {exc}"
    if tau.shape != alpha.shape or tau.ndim != 1:
        return (f"{rel} has tau{tau.shape} and alpha{alpha.shape}; a table "
                "is one node vector and one weight vector of equal length")
    return tau, alpha


# ---------------------------------------------------------------------------
# The selection rule
# ---------------------------------------------------------------------------

def _generator_pointer(range_value: float, beta: float,
                       target_error: float) -> str:
    return (f"  or generate: {GENERATOR} --range-max {range_value:.6g} "
            f"--omega-hat {beta!r} --error-bound {target_error:.0e}\n"
            "               (the whole 18-entry certified sweep cost 23 "
            "minutes of one core, offline)")


def _refuse(code, message, **kw) -> TableRefusal:
    return TableRefusal(code=code, message=message, **kw)


def select(
    *, range_value: float, beta: float, beta_clause: str,
    target_error: float, max_nodes: int,
) -> TableSelection | TableRefusal:
    """Serve one certified ``complex_laplace`` table, or refuse in prose.

    The rule in three sentences.  A request is ``(R, beta, clause, tier,
    node budget)``; ``R`` rounds up and the tier rounds down exactly as
    the shipped selector has always rounded them, and the node budget is
    a hard ceiling.  ``beta`` does not round: the request must land inside
    a certified entry's own stamped tolerance band, which is a measured
    two-sided width of about ``error_bound * (1 + beta**2)`` and not a
    display precision, and the one exception -- a ``positive_composite``
    entry, whose beta dependence is an exact phase on beta-independent
    nodes -- rounds DOWNWARD only, to its own ``beta_min``.  The clause
    must match the catalog's before any of that is consulted, because a
    pole width and a sampling line height are different physical
    quantities that happen to share a dimensionless name, and an entry
    swept for one of them certifies nothing about the other.
    """

    doc, why_not = load_catalog()
    if doc is None:
        return _refuse(
            "CatalogUnavailable",
            f"minimax[{FAMILY}]: no catalog, so no certified table for "
            f"R={range_value:.6g} at beta={beta:.9g}.\n"
            f"  reason: {why_not}\n"
            f"  the run continues on the uncertified runtime solve; "
            f"nothing below has been checked.",
            beta=float(beta), clause=str(beta_clause))

    clause = str(beta_clause)
    if clause not in BETA_CLAUSES:
        return _refuse(
            "UnknownClause",
            f"minimax[{FAMILY}]: beta_clause={clause!r} is not a clause of "
            "the envelope.\n"
            f"  the envelope has exactly two: {sorted(BETA_CLAUSES)}. A "
            "caller must say which numerator its beta came from, because "
            "beta alone cannot say -- the two clauses overlap near 0.6.",
            beta=float(beta), clause=clause)

    spec = BETA_CLAUSES[clause]
    if not (0.0 <= float(beta) <= spec.beta_max):
        return _refuse(
            "OutsideEnvelope",
            f"minimax[{FAMILY}]: beta={beta:.9g} is outside the {clause} "
            f"clause, which is certified on [0, {spec.beta_max:.6g}].\n"
            f"  that clause's beta is {spec.numerator}\n"
            f"  measured range: {spec.measured_range[0]:.6g} ... "
            f"{spec.measured_range[1]:.6g}   ({spec.source})\n"
            + ("  a beta this large usually means x_min hit the 1e-12 "
               "clamp -- a gapless deck has no well-posed quadrature "
               "request to tabulate, and this refusal is the clamp "
               "becoming visible rather than a missing table.\n"
               if float(beta) > 1.0e6 else "")
            + _generator_pointer(range_value, beta, target_error),
            beta=float(beta), clause=clause)

    version = catalog_version(doc)
    catalog_clause = CATALOG_CLAUSE.get(version)
    if catalog_clause is None:
        return _refuse(
            "UnknownCatalogClause",
            f"minimax[{FAMILY}]: catalog target version {version!r} is not "
            "stamped with an envelope clause in CATALOG_CLAUSE.\n"
            "  a catalog whose clause is unknown cannot be matched against "
            "a request's clause, and guessing is the failure this module "
            "exists to prevent.",
            beta=float(beta), clause=clause)
    if catalog_clause != clause:
        other = BETA_CLAUSES[catalog_clause]
        return _refuse(
            "WrongClause",
            f"minimax[{FAMILY}]: this is a {clause}-clause request and the "
            f"shipped catalog ({version}) is a {catalog_clause}-clause "
            "sweep.\n"
            f"  you asked with beta = {spec.numerator}\n"
            f"  the tables were swept over  {other.numerator}\n"
            "  the two are different physical quantities wearing one "
            "dimensionless name, so an entry certified for one of them "
            "certifies nothing about the other -- and because the "
            "clauses overlap in beta, some of these entries would "
            "otherwise MATCH this request numerically. That near-match is "
            "the reason this check runs before the beta match and not "
            "after it.\n"
            + _generator_pointer(range_value, beta, target_error),
            beta=float(beta), clause=clause)

    entries = doc.get("tables")
    if not isinstance(entries, list) or not entries:
        return _refuse(
            "CatalogUnavailable",
            f"minimax[{FAMILY}]: the catalog carries no 'tables' list.",
            beta=float(beta), clause=clause)

    # Pass one: the axes that round.  R up, tier down, nodes a ceiling --
    # unchanged from the shipped selector, and stated in its terms.
    in_shape: list[tuple[int, dict]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CatalogCorrupt(
                f"{CATALOG_FILE} entry {index} is {type(entry).__name__}, "
                "not an object.")
        if str(entry.get("family", FAMILY)) != FAMILY:
            continue
        entry_range = _need(entry, index, "range_max", float)
        entry_err = _need(entry, index, "error_bound", float)
        node_count = _need(entry, index, "node_count", int)
        if not bool(entry.get("certified", False)):
            continue
        # The three directions come from the family's AXIS RECORD rather
        # than from three inequalities written out here, so this door and
        # the damped_line door cannot drift apart on which way an axis
        # rounds.  The comparisons themselves are unchanged: R up, tier
        # down, nodes a ceiling.
        if not _axes.covers(_AXES.direction_for("range_max"),
                            entry_range, float(range_value),
                            tol=1.0e-12):
            continue
        if not _axes.covers(_AXES.direction_for("error_bound"),
                            entry_err, float(target_error),
                            tol=1.0e-18):
            continue
        if not _axes.covers(_AXES.direction_for("node_count"),
                            node_count, int(max_nodes)):
            continue
        in_shape.append((index, entry))

    if not in_shape:
        betas = sorted({float(e.get("beta", float("nan")))
                        for e in entries})
        ranges = sorted({float(e.get("range_max", float("nan")))
                         for e in entries})
        tiers = sorted({float(e.get("error_bound", float("nan")))
                        for e in entries})
        nodes = sorted({int(e.get("node_count", 0)) for e in entries})
        return _refuse(
            "NoCertifiedTable",
            f"minimax[{FAMILY}]: no certified table for R={range_value:.6g} "
            f"at tier {float(target_error):.0e} within {int(max_nodes)} "
            "nodes.\n"
            f"  tabulated R buckets: {ranges} (R rounds UP, so the request "
            "must be at or below one of them)\n"
            f"  tabulated tiers:     {tiers} (the tier rounds DOWN, so a "
            "request tighter than every tabulated bound has no table)\n"
            f"  node counts on hand: {nodes[0]} ... {nodes[-1]}\n"
            f"  tabulated betas:     {betas}\n"
            + _generator_pointer(range_value, beta, target_error),
            beta=float(beta), clause=clause)

    # Pass two: the axis that does not round.
    matched = [(i, e) for i, e in in_shape if beta_covers(e, i, beta)]
    if not matched:
        nearest_i, nearest = min(
            in_shape,
            key=lambda item: abs(_need(item[1], item[0], "beta", float)
                                 - float(beta)))
        nb = _need(nearest, nearest_i, "beta", float)
        band = _need(nearest, nearest_i, "beta_tolerance", float)
        delta = abs(nb - float(beta))
        return _refuse(
            "BetaNearMiss",
            f"minimax[{FAMILY}]: no certified table AT beta={beta!r} "
            f"(R={range_value:.6g}, tier {float(target_error):.0e}).\n"
            f"  nearest certified beta: {nb!r} in "
            f"{nearest.get('file')}\n"
            f"  the request misses it by {delta:.6e}, and that entry's "
            f"stamped band is {band:.6e} "
            f"({delta / band if band else float('inf'):.1%} of it)\n"
            "  beta is NOT rounded and the neighbour is NOT served: "
            "1/(u - i beta) is a different function at every beta, so a "
            "table at beta' != beta is not conservative, it is wrong "
            "(MINIMAX_REQUEST_CENSUS.md sec 3.1, measured at >2 orders in "
            "tests/test_minimax_imag_tables.py).\n"
            f"  reachable by: beta = {nb!r} exactly, i.e. varpi = "
            f"{nb!r} * x_min -- the probe frequency is a free convention "
            "and x_min is not\n"
            + _generator_pointer(range_value, beta, target_error),
            beta=float(beta), nearest_beta=nb, nearest_band=band,
            clause=clause)

    # Rank: nearest larger range, then the loosest acceptable tier, then
    # the fewest nodes, then the closest beta.  The node term is what
    # keeps a 566-node composite from displacing a 10-node minimax entry
    # that covers the same request -- every node here is one chi0 build
    # in the consumer.
    #
    # THE LAST TERM IS NOT DECORATION.  The catalog carries both the
    # census's DISPLAYED beta grid and the decks' own full-precision
    # omega-hats, and near a rounded value the two entries can cost the
    # same node count and both cover the request.  Without this term the
    # tie went to file order, so gnppm was served by the rounded 16.006
    # entry at 14.7% of its band while an entry measured AT
    # 16.00601826286684 sat beside it spending none.  Among equally
    # cheap certified rules, the one whose certificate was measured
    # closest to the request is the one to serve; nothing else in the
    # ranking can express that.
    matched.sort(key=lambda item: (
        _need(item[1], item[0], "range_max", float),
        -_need(item[1], item[0], "error_bound", float),
        _need(item[1], item[0], "node_count", int),
        abs(_need(item[1], item[0], "beta", float) - float(beta)),
    ))
    index, entry = matched[0]

    payload = _read_payload(entry)
    if isinstance(payload, str):
        return _refuse(
            "TableUnreadable",
            f"minimax[{FAMILY}]: entry {index} matched the request "
            f"(R={range_value:.6g}, beta={beta!r}) but its payload will "
            "not load.\n"
            f"  reason: {payload}\n"
            "  a matched-but-unreadable entry is a broken bundle, not an "
            "absent table.",
            beta=float(beta), clause=clause)
    tau, alpha = payload

    beta_entry = _need(entry, index, "beta", float)
    entry_range = _need(entry, index, "range_max", float)
    band = _need(entry, index, "beta_tolerance", float)
    bound = _need(entry, index, "error_bound", float)
    measured = _need(entry, index, "real_part_max_error", float)
    exact = float(beta) == beta_entry

    # How the beta axis was satisfied, said in the terms of the rule that
    # satisfied it.  A minimax entry is a POINT with a measured band and
    # the useful number is how much of that band the request spent; a
    # composite entry is an INTERVAL that rounds downward and a "fraction
    # of the band" would be a category error there.
    axis = str(entry.get("beta_axis"))
    if axis == "exact_phase_on_fixed_nodes" and not exact:
        # THE RE-PHASING, AND WHY IT IS NOT OPTIONAL.  A composite entry's
        # weights are ``h_l * e^{i beta_entry t_l}`` with ``h_l > 0`` chosen
        # without reference to beta at all.  Serving those bytes at a
        # smaller beta would be serving a table fitted to a different
        # function -- exactly what this module refuses the minimax entries
        # for.  What rounds downward is the RULE, not the payload: recover
        # the beta-independent magnitudes and re-evaluate the unit-modulus
        # phase at the request.  That is the rule this route would have
        # built at the request's beta, and the harness measures it at
        # kappa_0 = 1.000000 at every beta in the band.
        beta_min = _need(entry, index, "beta_min", float)
        alpha = np.abs(alpha) * np.exp(1j * float(beta) * tau)
        frac = 0.0
        beta_note = (f"beta rounds DOWN on this rule: certified on "
                     f"[{beta_min:.6g}, {beta_entry!r}], phase "
                     f"RE-EVALUATED at the request")
    elif axis == "exact_phase_on_fixed_nodes":
        beta_min = _need(entry, index, "beta_min", float)
        frac = 0.0
        beta_note = (f"beta rounds DOWN on this rule: certified on "
                     f"[{beta_min:.6g}, {beta_entry!r}], request sits on "
                     "the entry's own beta so the shipped bytes serve "
                     "unchanged")
    else:
        frac = (abs(beta_entry - float(beta)) / band) if band else 0.0
        beta_note = (f"beta band {band:.3e}, request at {frac:.1%} of it "
                     "(exact-match axis; no rounding)")

    # The band, checked rather than trusted.  If the entry's certificate
    # does not in fact survive the move to this beta, that is a refusal
    # and not a served table -- the FALSE case of the band being a
    # measurement at all.
    modulus_error, real_error = measure_at(tau, alpha, entry_range, beta)
    if modulus_error > bound:
        return _refuse(
            "BandViolated",
            f"minimax[{FAMILY}]: {entry.get('file')} matched "
            f"beta={beta!r} inside its stamped band {band:.6e}, but "
            "re-measured on a held-out grid at that beta it reaches "
            f"{modulus_error:.6e} against a bound of {bound:.0e}.\n"
            "  the band is a measurement, and this is the measurement "
            "disagreeing with it. The table is NOT served: a stamped "
            "band that does not hold is worse than no band at all.\n"
            + _generator_pointer(range_value, beta, target_error),
            beta=float(beta), nearest_beta=beta_entry, nearest_band=band,
            clause=clause)
    certified_error = real_error

    line = (
        f"minimax[{FAMILY}]: SHIPPED TABLE {entry.get('file')} "
        f"rule={entry.get('rule')} certified={bool(entry.get('certified'))} "
        f"N={int(tau.size)} kappa0={_need(entry, index, 'kappa0', float):.4f}"
        f" (tier {entry.get('kappa0_tier')}) "
        f"| request R={range_value:.6g} beta={beta:.9g} "
        f"[{clause} clause] tier {float(target_error):.0e} "
        f"-> entry R={entry_range:.6g} "
        f"beta={beta_entry!r} bound={bound:.0e} "
        f"| re-measured HERE at this beta: |E|={modulus_error:.3e}, "
        f"Re E={certified_error:.3e} "
        f"| {beta_note} "
        f"| sha256:{str(entry.get('payload_sha256', ''))[:12]} "
        f"catalog {catalog_version(doc)} schema {doc.get('schema_version')}"
    )

    return TableSelection(
        tau=tau,
        alpha=alpha,
        entry=entry,
        clause=clause,
        beta=float(beta),
        range_value=float(range_value),
        certified_error=float(certified_error),
        modulus_error=float(modulus_error),
        error_bound=float(bound),
        real_part_max_error=float(measured),
        band_fraction=float(frac),
        catalog_version=catalog_version(doc),
        _line=line,
    )


# ---------------------------------------------------------------------------
# Loud provenance (DESIGN_minimax.md R2)
# ---------------------------------------------------------------------------

#: Every provenance line this process has emitted, in order.  Tests read
#: it; so can a driver that would rather log than print.
PROVENANCE_LOG: list[str] = []
_ANNOUNCED: set[str] = set()


def announce(selection: TableSelection, *, print_fn=None) -> bool:
    """Say once, out loud, where a served table came from.

    R2's finding is that ``build_static_quadrature`` has always printed
    ``nodes=`` and ``fit_err~`` without ever saying whether that came from
    a shipped table or from four minutes of scipy, and that this single
    omission is why a solver pathology sat unnoticed for months.  So a
    served table announces its rule, its certification and its catalog
    entry.

    Once, though: the door is called per rank and per Sigma pass, and a
    line repeated eighty times is noise that gets filtered, which is the
    same as not printing it.  Returns True if this call is the one that
    printed.
    """

    line = selection.one_line()
    if line in _ANNOUNCED:
        return False
    _ANNOUNCED.add(line)
    PROVENANCE_LOG.append(line)
    (print_fn or print)(line)
    return True


def reset_announcements() -> None:
    """Forget what has been announced.  For tests, and for long drivers."""

    _ANNOUNCED.clear()
    PROVENANCE_LOG.clear()
