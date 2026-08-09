"""The ``damped_line`` axis of the shipped quadrature catalog.

WHAT THIS SERVES.  The fourth cell of ``DESIGN_minimax.md`` R4's 2x2
sampling table -- ``z = omega + i*varpi`` with both parts nonzero -- and
therefore the door the MPA fit stage is currently standing outside,
where ``gw/screening.py`` says *"complex-axis omega ... not supported"*.
One entry is one dimensionless certified rule for

    K_z(Delta) = -2 * Delta / (Delta**2 - z**2)

on the rectangle ``(z on either sampling line) x (Delta in
[0, Delta_max])``, in units of the near line's height.

WHY IT IS A SEPARATE DOOR FROM ``minimax_beta_selector``.  Not because
the arithmetic differs -- ``minimax_family_axes`` now holds both
families' rounding directions as one record, and both doors read it --
but because the two catalogs answer different QUESTIONS.  The
``complex_laplace`` door is asked for a table on an interval ``[1, R]``
at a line parameter ``beta``; this one is asked for a table on a
bandwidth at a dimensionless span ``A``, and it must also answer "which
row of the weight table is my sample".  A single ``select()`` taking the
union of both families' parameters would be a function whose arguments
half apply, which is the shape that produces silent wrong answers.

WHAT IS DIFFERENT ABOUT THIS FAMILY, and it is worth stating because
each difference removes a failure mode the sibling had to build
machinery for:

* **No envelope axis.**  The family is exactly scale free in ``varpi``,
  ``K_z(Delta) = (1/varpi) Khat(Delta/varpi, omega/varpi)``, so an entry
  is dimensionless.  There is no ``beta`` to round, no clause to
  declare, and no way to serve a table fitted to a different function.
* **One entry, both lines.**  The node set is certified against the near
  line AND the far line at once, which is why ``line_height_ratio`` is
  an EXACT-match axis: the far line's rows are the certificate.
* **One entry, every n_p.**  The nested partition means a smaller
  ``n_p`` is a ROW SUBSET of the shipped weight table.  Selecting a row
  is an exact fraction match on ``(numerator, denominator)`` and never
  an interpolation, so an ``n_p`` scan costs one lookup.

WHAT THIS MODULE NEVER DOES.  It never solves, never falls back, and
never serves a neighbour.  Every path out of ``select`` is either a
``DampedLineSelection`` carrying one certified entry and its provenance,
or a ``TableRefusal`` naming what was asked, what is nearest, and the
generator invocation that would close it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from functools import lru_cache
import importlib.resources as importlib_resources
import json

import numpy as np

from common import minimax_family_axes as _axes
from common.minimax_family_axes import CatalogCorrupt

AXES = _axes.DAMPED_LINE
FAMILY = AXES.family
CATALOG_FILE = AXES.catalog_file
GENERATOR = AXES.generator

#: The published sampling geometry: varpi_2/varpi_1 = 1 Ha / 0.1 Ha.
PUBLISHED_LINE_RATIO = 10.0


@dataclass(frozen=True)
class TableRefusal:
    """A refusal that names its own cure.

    Mirrors ``minimax_beta_selector.TableRefusal`` field for field on the
    two fields that are family-generic, and carries this family's axes
    on the rest.  A refusal that does not tell you how to make it go
    away is a crash with better manners (``DESIGN_minimax.md`` 3.2).
    """

    code: str
    message: str
    a_dim: float | None = None
    nearest_a: float | None = None
    error_bound: float | None = None
    line_height_ratio: float | None = None
    n_p: int | None = None

    def one_line(self) -> str:
        return self.message.strip().splitlines()[0]


@dataclass(frozen=True)
class DampedLineSelection:
    """One certified entry, with the rows the caller's plan asked for."""

    t: np.ndarray
    w: np.ndarray
    rows: tuple
    entry: dict
    rule: str
    a_dim: float
    error_bound: float
    certified_error: float
    kappa0: float
    composite_node_count: int
    catalog_version: str
    _line: str = field(default="", repr=False)

    @property
    def node_count(self) -> int:
        return int(self.t.size)

    @property
    def compression(self) -> float:
        """Nodes saved against ONE COMPOSITE RULE PER LINE, which is what
        ``evaluate_samples(batching='per-line')`` spends today."""

        return float(self.composite_node_count) / max(self.node_count, 1)

    def one_line(self) -> str:
        return self._line


def _catalog_path():
    return importlib_resources.files("common").joinpath(
        "minimax_assets", CATALOG_FILE)


@lru_cache(maxsize=1)
def load_catalog() -> tuple[dict | None, str | None]:
    """The catalog document, or a reason it is unusable.

    Zero-argument and cached exactly like the sibling's, so the two
    doors are monkeypatched the same way in tests.
    """

    path = _catalog_path()
    try:
        if not path.is_file():
            return None, f"{path} is not present in this checkout"
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"{path} could not be read: {exc}"
    if not isinstance(doc, dict) or doc.get("family") != FAMILY:
        return None, (f"{path} does not declare family={FAMILY!r} "
                      f"(found {doc.get('family')!r})")
    if int(doc.get("schema_version", 0)) < 2:
        return None, (f"{path} is schema_version="
                      f"{doc.get('schema_version')!r}; this door needs 2 "
                      "or later")
    return doc, None


def catalog_version(doc: dict) -> str:
    return str(doc.get("target", {}).get("version", "unknown"))


def _need(entry, index, key, kind):
    if key not in entry:
        raise CatalogCorrupt(
            f"{CATALOG_FILE} entry {index} ({entry.get('file')!r}) has "
            f"no {key!r} field. A missing field does not mean the entry "
            "is absent; it means the bundle on disk is not the one that "
            "was certified.")
    try:
        return kind(entry[key])
    except (TypeError, ValueError) as exc:
        raise CatalogCorrupt(
            f"{CATALOG_FILE} entry {index} ({entry.get('file')!r}) has "
            f"{key}={entry[key]!r}, which is not {kind.__name__}: {exc}")


def _generator_pointer(a_dim, error_bound):
    return (f"  or generate: {GENERATOR} --spans {a_dim:.6g} "
            f"--tiers {error_bound:.0e}\n"
            "               (the whole 32-entry certified sweep cost "
            "about an hour of one laptop, offline)")


def _refuse(code, message, **kw) -> TableRefusal:
    return TableRefusal(code=code, message=message, **kw)


def _read_payload(entry, index):
    rel = _need(entry, index, "file", str)
    path = importlib_resources.files("common").joinpath(
        "minimax_assets", rel)
    try:
        with path.open("rb") as fh:
            with np.load(fh, allow_pickle=False) as data:
                keys = set(data.files)
                if not set(AXES.payload_keys) <= keys:
                    return None, (
                        f"{rel} has npz keys {sorted(keys)}; a "
                        f"{FAMILY} table carries "
                        f"{list(AXES.payload_keys)}")
                out = {k: np.asarray(data[k]) for k in data.files}
    except (OSError, ValueError) as exc:
        return None, f"{rel} could not be read: {exc}"
    return out, None


def row_index(payload, *, line_varpi, fraction):
    """The row for one sample, by EXACT fraction match.

    The partition nests strictly, so a request at any ``n_p`` names
    fractions the shipped table already carries; matching them exactly
    -- as integer numerator and denominator, not as floats -- is what
    makes "no interpolation anywhere" a property of the code rather than
    a claim in a document.
    """

    frac = Fraction(fraction)
    num = np.asarray(payload["row_fraction_num"])
    den = np.asarray(payload["row_fraction_den"])
    vp = np.asarray(payload["row_varpi"])
    hit = np.flatnonzero(
        (num == frac.numerator) & (den == frac.denominator)
        & (np.abs(vp - float(line_varpi)) <= 1.0e-12 * max(
            float(line_varpi), 1.0)))
    if hit.size != 1:
        raise CatalogCorrupt(
            f"{FAMILY}: the shipped table has {hit.size} rows at "
            f"varpi={line_varpi!r}, fraction={frac}. FALSE case: exactly "
            "one -- the row list is the union over alpha of the nested "
            f"partition to n_p_max, and every entry of it is unique.")
    return int(hit[0])


def select(*, a_dim: float, target_error: float,
           line_height_ratio: float = PUBLISHED_LINE_RATIO,
           n_p: int | None = None, alpha: int | None = None,
           max_nodes: int = 1 << 30):
    """Return one certified entry for this request, or a refusal.

    ``a_dim`` is ``(max|omega| + Delta_max)/varpi_1`` -- the evaluator's
    own ``A``, which on the MPA protocol is ``2*Delta_max/varpi_1``
    because ``omega_m`` IS the top transition.  ``n_p`` and ``alpha``
    are the sampling plan's, and are only consulted on the sparse route:
    a composite entry's nodes are omega-independent, so it serves every
    partition and every alpha, which is what the family's axis record
    declares and what the harness measures.
    """

    doc, why_not = load_catalog()
    if doc is None:
        return _refuse(
            "CatalogUnavailable",
            f"minimax[{FAMILY}]: no usable catalog.\n  {why_not}\n"
            + _generator_pointer(float(a_dim), float(target_error)),
            a_dim=float(a_dim), error_bound=float(target_error))

    ratio = float(line_height_ratio)
    entries = doc.get("tables", [])

    # Pass one: the axes that round.  A up, tier down, nodes a ceiling.
    in_shape = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CatalogCorrupt(
                f"{CATALOG_FILE} entry {index} is "
                f"{type(entry).__name__}, not an object.")
        if str(entry.get("family", FAMILY)) != FAMILY:
            continue
        entry_a = _need(entry, index, "A", float)
        entry_err = _need(entry, index, "error_bound", float)
        entry_ratio = _need(entry, index, "line_height_ratio", float)
        node_count = _need(entry, index, "node_count", int)
        if not bool(entry.get("certified", False)):
            continue
        if entry_a + 1.0e-12 < float(a_dim):
            continue                     # A ROUNDS UP
        if entry_err - 1.0e-18 > float(target_error):
            continue                     # TIER ROUNDS DOWN
        if abs(entry_ratio - ratio) > 1.0e-12:
            continue                     # LINE RATIO IS EXACT
        if node_count > int(max_nodes):
            continue
        in_shape.append((index, entry))

    if not in_shape:
        spans = sorted({float(e.get("A", float("nan")))
                        for e in entries})
        tiers = sorted({float(e.get("error_bound", float("nan")))
                        for e in entries})
        ratios = sorted({float(e.get("line_height_ratio", float("nan")))
                         for e in entries})
        bigger = [s for s in spans if s >= float(a_dim)]
        return _refuse(
            "NoCertifiedTable",
            f"minimax[{FAMILY}]: nothing certified covers A="
            f"{float(a_dim):.6g} at error_bound<={float(target_error):.0e}"
            f" with line_height_ratio={ratio:.6g}.\n"
            f"  tabulated A:                 {spans}\n"
            f"  tabulated tiers:             {tiers}\n"
            f"  tabulated line ratios:       {ratios}\n"
            f"  max_nodes budget:            {int(max_nodes)}\n"
            + ("  A is above the top of the ladder; there is no "
               "extrapolation and there is no runtime solve.\n"
               if not bigger else "")
            + _generator_pointer(float(a_dim), float(target_error)),
            a_dim=float(a_dim),
            nearest_a=(min(bigger) if bigger else
                       (max(spans) if spans else None)),
            error_bound=float(target_error), line_height_ratio=ratio)

    # Rank: the tightest span first (least wasted rectangle), then the
    # loosest bound that still qualifies, then the fewest nodes.
    in_shape.sort(key=lambda ie: (
        float(ie[1]["A"]), -float(ie[1]["error_bound"]),
        int(ie[1]["node_count"])))
    index, entry = in_shape[0]
    rule = _need(entry, index, "rule", str)

    # The alpha axis, resolved through the family record rather than
    # through an `if` -- this is the asymmetry complex_laplace already
    # carries on beta, and writing it once is the point of the record.
    alpha_direction = AXES.direction_for("alpha", rule)
    if alpha_direction == _axes.EXACT and alpha is not None:
        shipped = [int(a) for a in entry.get("alpha_sets", [])]
        if shipped and int(alpha) not in shipped:
            return _refuse(
                "AlphaNotShipped",
                f"minimax[{FAMILY}]: entry {entry.get('file')!r} ships "
                f"alpha_sets={shipped} and the plan asked for "
                f"alpha={int(alpha)}.\n"
                "  the sparse route ships one weight row per (line, "
                "exact partition fraction) and the fraction depends on "
                "alpha, so there is nothing to serve here.\n"
                + _generator_pointer(float(a_dim), float(target_error)),
                a_dim=float(a_dim), error_bound=float(target_error))

    if rule == "btv_minimax" and n_p is not None:
        n_p_max = _need(entry, index, "n_p_max", int)
        if int(n_p) > n_p_max:
            return _refuse(
                "NpAboveTable",
                f"minimax[{FAMILY}]: entry {entry.get('file')!r} ships "
                f"rows to n_p_max={n_p_max} and the plan asked for "
                f"n_p={int(n_p)}.\n"
                "  the partition nests, so a LARGER n_p adds sample "
                "points the shipped weight table has no rows for. It is "
                "not an interpolation away; it is a re-solve.\n"
                + _generator_pointer(float(a_dim), float(target_error)),
                a_dim=float(a_dim), n_p=int(n_p),
                error_bound=float(target_error))

    payload, why_bad = _read_payload(entry, index)
    if payload is None:
        return _refuse(
            "TableUnreadable",
            f"minimax[{FAMILY}]: the selected entry's payload is not "
            f"usable.\n  {why_bad}\n"
            + _generator_pointer(float(a_dim), float(target_error)),
            a_dim=float(a_dim), error_bound=float(target_error))

    t = np.asarray(payload["t"], dtype=np.float64)
    w = np.asarray(payload["w"], dtype=np.complex128)
    rows = tuple(zip(
        np.asarray(payload["row_varpi"], dtype=np.float64).tolist(),
        np.asarray(payload["row_fraction_num"], dtype=np.int64).tolist(),
        np.asarray(payload["row_fraction_den"], dtype=np.int64).tolist()))

    entry_a = float(entry["A"])
    comp = _need(entry, index, "composite_node_count", int)
    line = (f"minimax[{FAMILY}]: SHIPPED TABLE {entry.get('file')} "
            f"rule={rule} A={entry_a:g} (asked {float(a_dim):g}) "
            f"eps={float(entry['error_bound']):.0e} N={t.size} "
            f"kappa0={float(entry['kappa0']):.4f} "
            f"vs per-line composite {comp} "
            f"({comp / max(int(t.size), 1):.2f}x) "
            f"catalog {catalog_version(doc)} schema "
            f"{int(doc['schema_version'])}")
    return DampedLineSelection(
        t=t, w=w, rows=rows, entry=entry, rule=rule, a_dim=entry_a,
        error_bound=float(entry["error_bound"]),
        certified_error=float(entry["max_error"]),
        kappa0=float(entry["kappa0"]), composite_node_count=comp,
        catalog_version=catalog_version(doc), _line=line)
