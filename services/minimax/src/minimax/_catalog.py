"""The shipped bundle: read it, parse it strictly, select from it, load it.

UNDERSCORE-PRIVATE BY NECESSITY, not by taste.  The design gives the door
a function called ``catalog()`` (§4.3), and a module called
``minimax.catalog`` beside it would be shadowed by that function the
moment ``__init__`` binds the name -- so ``from minimax import catalog``
would hand a caller the function on one import order and the module on
another.  Everything here that a consumer needs is re-exported on the
door: :func:`catalog_view`, :func:`select_entry`, :func:`parse_catalog`,
:func:`parse_entry`, :func:`nearest_below`, :func:`load_catalog_dict`,
:func:`load_table`, :func:`provenance_for` and :class:`CatalogEntry`.

This module is where four of R2's six silent handlers used to live.  The
replacement rule is not "narrow the exception" but **the failure becomes a
value the caller can see**:

===========================  =====================  =========================
site                         before                 after
===========================  =====================  =========================
catalog path / read          ``return None`` (×2)   :class:`CatalogUnavailable`
table load                   ``return None``        :class:`TableUnreadable`
per-entry field parse        ``continue``           :class:`CatalogCorrupt`
``eps_q`` compare            ``continue``           :class:`CatalogCorrupt`
===========================  =====================  =========================

The third and fourth rows are the ones that changed MEANING rather than
volume.  A malformed entry used to be treated as ABSENT — the loop skipped
it and the selection rule went on to pick something else, or to find
nothing and fall through to a four-minute uncertified solve.  A malformed
entry means the ARTIFACT IS CORRUPT.  So parsing is separated from
selecting here: :func:`parse_entry` turns raw JSON into a typed
:class:`CatalogEntry` and refuses by index and field name, and
:func:`select_entry` then runs on already-typed entries with no
``try``/``except`` anywhere in it.

THE SELECTION RULE ITSELF IS CARRIED VERBATIM, and that is deliberate.
Every comparison, every tolerance (``+1.0e-12`` on the range, ``-1.0e-18``
on the error bound), the ordering key ``(range, -error_bound, node_count)``
and the "first after sort" tie-break are byte-for-byte what
``gw.minimax_screening._find_shipped_table_entry`` did before the
extraction.  The WP1 census measured 54 distinct requests across the frozen
decks, the campaign decks and the suite; all 51 that were served must be
served by the same table afterwards, and the only way to promise that is to
move the rule rather than rewrite it.

    COORDINATION NOTE (2026-08-08).  ``feat/minimax-beta-selector-2026-08-08``
    is adding a beta axis to this rule so the eighteen staged
    ``complex_laplace`` entries become selectable.  That work was launched
    against the rule's pre-extraction location and lands as a diff to the
    body of :func:`select_entry` — which is why the body here is a
    line-for-line carry rather than a tidier rewrite: a rewrite would have
    turned a replayable patch into a merge argument.  Whichever branch
    lands second replays the other's change here.
"""

from __future__ import annotations

import hashlib
import importlib.resources as importlib_resources
import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Mapping

import numpy as np

from minimax.records import CatalogView, Provenance
from minimax.refusals import (
    CatalogCorrupt,
    CatalogUnavailable,
    TableUnreadable,
)

#: The asset directory inside this package.  It used to be
#: ``src/common/minimax_assets/`` reached through
#: ``importlib.resources.files("common")``, which is the single production
#: import edge the extraction cut.
ASSET_PACKAGE = "minimax"
ASSET_DIR = "minimax_assets"


# ---------------------------------------------------------------------------
#  1.  Reading the catalog.  R2 rows 1 and 2.
# ---------------------------------------------------------------------------

def _asset_root():
    """The bundle root.  Refuses rather than returning ``None``."""
    try:
        return importlib_resources.files(ASSET_PACKAGE).joinpath(ASSET_DIR)
    except (ImportError, ModuleNotFoundError, TypeError) as exc:
        raise CatalogUnavailable(
            f"minimax: cannot resolve the shipped asset bundle "
            f"({ASSET_PACKAGE}/{ASSET_DIR}): {type(exc).__name__}: {exc}.  "
            f"This is an INSTALL defect, not a missing table: the bundle is "
            f"package data declared in services/minimax/pyproject.toml."
        ) from exc


@lru_cache(maxsize=4)
def load_catalog_dict(name: str = "catalog.json") -> dict[str, Any]:
    """The raw catalog JSON.  Refuses; never returns ``None``.

    Every failure names the RESOLVED PATH and the underlying error, because
    "no catalog" and "catalog present but unreadable" used to be the same
    event to every caller and they are not the same defect.

    CACHED, as it was before the extraction (``@lru_cache(maxsize=1)`` on
    ``_load_shipped_minimax_catalog``).  Not cosmetic: one Σ build issues
    2 crossing + 6 noncrossing window requests, times P ranks, times the
    MPA pass plan's fourteen — re-reading and re-parsing the bundle on
    each would be a real cost for no information.  ``maxsize=4`` because
    the bundle now carries more than one catalog file.
    :func:`clear_caches` is the test hook.
    """
    path = _asset_root().joinpath(name)
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError as exc:
        raise CatalogUnavailable(
            f"minimax: the shipped catalog {name!r} is not in the bundle "
            f"(resolved to {str(path)!r}): {exc}"
        ) from exc
    except OSError as exc:
        raise CatalogUnavailable(
            f"minimax: cannot read the shipped catalog {str(path)!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CatalogUnavailable(
            f"minimax: the shipped catalog {str(path)!r} is not valid JSON: "
            f"{exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise CatalogUnavailable(
            f"minimax: the shipped catalog {str(path)!r} is a "
            f"{type(raw).__name__}, not an object with a 'tables' array.")
    return raw


# ---------------------------------------------------------------------------
#  2.  Parsing an entry.  R2 rows 3 and 4 — a malformed entry refuses the
#      CATALOG rather than being silently skipped.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CatalogEntry:
    """One typed row of the catalog."""

    index: int
    family: str
    range_max: float
    error_bound: float
    node_count: int
    file: str
    range_param: str
    target_kind: str | None
    eps_q: float | None
    #: The catalog's CLAIM about the achieved error.  The served value comes
    #: off the payload, not from here — see :func:`load_table`.
    claimed_max_error: float | None
    kappa0: float | None
    certified: bool
    catalog_name: str
    raw: Mapping[str, Any] = field(repr=False, default_factory=dict)


def _required(raw, index, key, caster, catalog_name):
    if key not in raw:
        raise CatalogCorrupt(
            f"minimax: catalog {catalog_name!r} entry {index} has no {key!r} "
            f"field.  A malformed entry means the artifact is corrupt, not "
            f"that the entry is absent, so the catalog refuses rather than "
            f"skipping the row.  Entry keys present: {sorted(raw)}")
    try:
        return caster(raw[key])
    except (TypeError, ValueError) as exc:
        raise CatalogCorrupt(
            f"minimax: catalog {catalog_name!r} entry {index} field {key!r} "
            f"is {raw[key]!r}, which is not a {caster.__name__}: "
            f"{type(exc).__name__}: {exc}") from exc


def _optional(raw, index, key, caster, catalog_name):
    """``None`` when the field is absent; a REFUSAL when it is present and bad.

    Absence is legitimate — a ``noncrossing`` entry carries no ``eps_q`` and
    never did.  A present-but-unparseable value is the corruption case, and
    it is exactly the one the old ``except Exception: continue`` at the
    ``eps_q`` comparison used to swallow.
    """
    if key not in raw or raw[key] is None:
        return None
    try:
        return caster(raw[key])
    except (TypeError, ValueError) as exc:
        raise CatalogCorrupt(
            f"minimax: catalog {catalog_name!r} entry {index} field {key!r} "
            f"is {raw[key]!r}, which is not a {caster.__name__}: "
            f"{type(exc).__name__}: {exc}") from exc


def parse_entry(raw: Any, index: int, *, catalog_name: str,
                default_family: str | None = None) -> CatalogEntry:
    """Type one raw JSON row, refusing by index and field name."""
    if not isinstance(raw, dict):
        raise CatalogCorrupt(
            f"minimax: catalog {catalog_name!r} entry {index} is a "
            f"{type(raw).__name__}, not an object.")
    family = raw.get("family", default_family)
    if not isinstance(family, str) or not family:
        raise CatalogCorrupt(
            f"minimax: catalog {catalog_name!r} entry {index} has no usable "
            f"'family' ({family!r}).")
    rel = raw.get("file")
    if not isinstance(rel, str) or not rel:
        raise CatalogCorrupt(
            f"minimax: catalog {catalog_name!r} entry {index} names no "
            f"payload file ('file' is {rel!r}).")
    return CatalogEntry(
        index=index,
        family=family,
        range_max=_required(raw, index, "range_max", float, catalog_name),
        error_bound=_required(raw, index, "error_bound", float, catalog_name),
        node_count=_required(raw, index, "node_count", int, catalog_name),
        file=rel,
        range_param=str(raw.get("range_param") or "range"),
        target_kind=_optional(raw, index, "target_kind", str, catalog_name),
        eps_q=_optional(raw, index, "eps_q", float, catalog_name),
        claimed_max_error=_optional(raw, index, "max_error", float,
                                    catalog_name),
        kappa0=_optional(raw, index, "kappa0", float, catalog_name),
        certified=bool(raw.get("certified", False)),
        catalog_name=catalog_name,
        raw=raw)


def parse_catalog(raw: Mapping[str, Any], *, catalog_name: str
                  ) -> tuple[CatalogEntry, ...]:
    """Every row, typed.  One bad row refuses the whole catalog."""
    tables = raw.get("tables", [])
    if not isinstance(tables, list):
        raise CatalogCorrupt(
            f"minimax: catalog {catalog_name!r} has a 'tables' field that is "
            f"a {type(tables).__name__}, not a list.")
    default_family = raw.get("family") if isinstance(raw.get("family"), str) \
        else None
    return tuple(parse_entry(e, i, catalog_name=catalog_name,
                             default_family=default_family)
                 for i, e in enumerate(tables))


@lru_cache(maxsize=4)
def catalog_view(name: str = "catalog.json") -> CatalogView:
    """The enumerable, assertable view.  No solve, ever.

    Cached for the same reason :func:`load_catalog_dict` is, and by the
    same hook.  The view is frozen and its entries are frozen, so sharing
    one instance across callers is safe by construction rather than by
    convention.
    """
    raw = load_catalog_dict(name)
    entries = parse_catalog(raw, catalog_name=name)
    return CatalogView(
        schema_version=int(raw.get("schema_version", 0)),
        entries=entries,
        shipping_rule=dict(raw.get("shipping_rule", {})),
        source_path=str(_asset_root().joinpath(name)))


# ---------------------------------------------------------------------------
#  3.  The selection rule.  CARRIED VERBATIM — see the module docstring.
# ---------------------------------------------------------------------------

def select_entry(
    entries,
    family: str,
    *,
    range_value: float,
    target_error: float,
    max_nodes: int,
    target_kind: str | None = None,
    eps_q: float | None = None,
) -> CatalogEntry | None:
    """The best shipped entry for a request, or ``None`` if there is none.

    Conservative by construction, and the convention is load-bearing: the
    requested interval is rounded UP to the next tabulated range and the
    requested error target is rounded DOWN to the nearest stricter shipped
    bound, so the loaded table is at least as accurate as the caller asked
    for under the same absolute-error convention the solver uses.  Using a
    table fitted on a LARGER interval is safe because the requested
    interval is a subset of the tabulated one.

    Preference order: nearest larger range, then the least strict
    acceptable error, then the fewest nodes.
    """
    candidates: list[tuple[tuple[float, float, int], CatalogEntry]] = []
    for entry in entries:
        if entry.family != family:
            continue
        if entry.range_max + 1.0e-12 < float(range_value):
            continue
        if entry.error_bound - 1.0e-18 > float(target_error):
            continue
        if entry.node_count > int(max_nodes):
            continue
        if target_kind is not None and \
                str(entry.target_kind) != str(target_kind):
            continue
        if eps_q is not None:
            # An entry that declares no eps_q cannot match a request that
            # does.  That is ABSENCE, and it stays a skip; the corruption
            # case (a present-but-unparseable eps_q) was already refused in
            # `parse_entry`, which is the R2 row this split implements.
            if entry.eps_q is None:
                continue
            if abs(entry.eps_q - float(eps_q)) > 1.0e-12:
                continue
        key = (entry.range_max, -entry.error_bound, entry.node_count)
        candidates.append((key, entry))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def nearest_below(
    entries,
    family: str,
    *,
    range_value: float,
    target_error: float,
    target_kind: str | None = None,
    eps_q: float | None = None,
) -> CatalogEntry | None:
    """The largest certified range at or under the request.  Never raises.

    This is what a refusal OFFERS.  It deliberately ignores ``max_nodes``
    and the error tier: the point of the message is "here is the edge of
    the certified region, and here is what it cost", so filtering it by
    the same constraints that produced the miss would usually print
    "nearest: none" and teach nobody anything.
    """
    best = None
    for entry in entries:
        if entry.family != family:
            continue
        if entry.range_max > float(range_value) + 1.0e-12:
            continue
        if target_kind is not None and \
                str(entry.target_kind) != str(target_kind):
            continue
        if eps_q is not None and entry.eps_q is not None:
            if abs(entry.eps_q - float(eps_q)) > 1.0e-12:
                continue
        if best is None or (entry.range_max, -entry.error_bound) > \
                (best.range_max, -best.error_bound):
            best = entry
    return best


# ---------------------------------------------------------------------------
#  4.  Loading a payload.  R2 row 2.
# ---------------------------------------------------------------------------

_HASH_CACHE: dict[str, str] = {}


def payload_sha256(tau: np.ndarray, alpha: np.ndarray) -> str:
    """SHA-256 over the canonical little-endian numerical payload bytes."""
    digest = hashlib.sha256()
    digest.update(np.asarray(tau, dtype="<f8").tobytes())
    alpha_dtype = "<c16" if np.iscomplexobj(alpha) else "<f8"
    digest.update(np.asarray(alpha, dtype=alpha_dtype).tobytes())
    return digest.hexdigest()


def _sha256_of(path) -> str:
    key = str(path)
    cached = _HASH_CACHE.get(key)
    if cached is not None:
        return cached
    with path.open("rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    stamp = f"sha256:{digest[:16]}"
    _HASH_CACHE[key] = stamp
    return stamp


def load_table(entry: CatalogEntry) -> tuple[np.ndarray, np.ndarray, float,
                                             float | None, str]:
    """``(tau, alpha, max_error, kappa0, table_hash)`` — or a refusal.

    BIT-IDENTITY LIVES HERE.  ``tau`` and ``alpha`` are exactly what
    ``np.load`` returns from the shipped ``.npz``, cast with the same
    ``np.asarray(..., dtype=np.float64)`` the pre-extraction loader used,
    and ``max_error`` is read off the PAYLOAD rather than off the catalog's
    claim about it.  Nothing between the file and the caller rounds,
    rescales or re-solves.
    """
    path = _asset_root().joinpath(entry.file)
    try:
        table_hash = _sha256_of(path)
        with path.open("rb") as fh:
            with np.load(fh, allow_pickle=False) as data:
                found = tuple(data.files)
                missing = [k for k in ("tau", "alpha", "max_error")
                           if k not in found]
                if missing:
                    raise TableUnreadable(
                        f"minimax: shipped table {entry.file!r} (catalog "
                        f"{entry.catalog_name!r} entry {entry.index}) is "
                        f"missing {missing}; the npz key set found was "
                        f"{list(found)}.")
                tau = np.asarray(data["tau"], dtype=np.float64)
                alpha = np.asarray(data["alpha"])
                err = float(data["max_error"][()])
                kappa0 = (float(data["kappa0"][()]) if "kappa0" in found
                          else None)
    except TableUnreadable:
        raise
    except (OSError, ValueError, KeyError, EOFError) as exc:
        raise TableUnreadable(
            f"minimax: cannot read shipped table {entry.file!r} "
            f"(catalog {entry.catalog_name!r} entry {entry.index}, resolved "
            f"to {str(path)!r}): {type(exc).__name__}: {exc}"
        ) from exc
    if not np.iscomplexobj(alpha):
        # The real families' loader has always cast to float64 explicitly;
        # keeping that cast keeps the served bytes identical.  The strip
        # families ship complex128 alpha and must NOT be flattened to real.
        alpha = np.asarray(alpha, dtype=np.float64)
    expected_payload = entry.raw.get("payload_sha256")
    if expected_payload is not None:
        actual_payload = payload_sha256(tau, alpha)
        if str(expected_payload) != actual_payload:
            raise TableUnreadable(
                f"minimax: shipped table {entry.file!r} payload SHA-256 "
                f"is {actual_payload}, but catalog {entry.catalog_name!r} "
                f"entry {entry.index} stamps {expected_payload!r}.")
    if entry.kappa0 is not None:
        kappa0 = entry.kappa0
    return tau, alpha, err, kappa0, table_hash


def provenance_for(entry: CatalogEntry, table_hash: str,
                   catalog_raw: Mapping[str, Any] | None) -> Provenance:
    """The provenance block a shipped table carries.

    The v1 catalog records NO generator commit and NO backend, and this is
    where that is said out loud rather than filled in with a plausible
    guess.  ``'unrecorded (catalog schema v1)'`` is a measurement of the
    bundle, and printing it on every serve is what makes WP6's absence
    visible in a log instead of only in a design document.
    """
    entry_prov = entry.raw.get("provenance")
    if entry_prov is not None and not isinstance(entry_prov, Mapping):
        raise CatalogCorrupt(
            f"minimax: catalog {entry.catalog_name!r} entry {entry.index} "
            f"has non-object provenance {entry_prov!r}.")
    prov = (entry_prov if entry_prov is not None
            else (catalog_raw or {}).get("provenance") or {})
    tool = prov.get("tool")
    commit = prov.get("generator_commit") or prov.get("tool_sha256")
    if commit and tool:
        commit = f"{tool}@{str(commit)[:12]}"
    elif commit:
        commit = str(commit)[:12]
    else:
        commit = "unrecorded (catalog schema v1)"
    backend_bits = [b for b in (
        f"numpy-{prov['numpy']}" if "numpy" in prov else None,
        f"scipy-{prov['scipy']}" if "scipy" in prov else None,
        f"python-{prov['python']}" if "python" in prov else None,
        f"id-{str(prov['backend_sha256'])[:12]}"
        if "backend_sha256" in prov else None,
    ) if b]
    backend = ("cpu:" + "/".join(backend_bits) if backend_bits
               else "unrecorded (catalog schema v1)")
    return Provenance(
        source="shipped",
        catalog_entry=entry.file,
        table_hash=table_hash,
        generator_commit=commit,
        generation_backend=backend,
        certified=bool(entry.certified))


def clear_caches() -> None:
    """Forget every cached catalog and payload hash.

    The test hook, and the only supported way to make the service re-read
    a bundle inside one process.  Production never calls it: the bundle is
    package data and does not change under a running job.
    """
    load_catalog_dict.cache_clear()
    catalog_view.cache_clear()
    _HASH_CACHE.clear()
