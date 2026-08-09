"""The refusals — every gap named, with the lever that closes it.

``DESIGN_minimax.md`` §3.2 lists six, none of which existed before the
extraction: the module they came from turned every one of them into a
silent ``None`` and a minutes-long uncertified solve that nothing announced
(survey §2.3).  Each refusal here ships with the case where it returns
FALSE, in ``services/minimax/tests/`` — no exceptions
(``SERVICE_FORM.md:46``).

THE MESSAGE SHAPE IS THE POINT.  A refusal that does not tell you how to
make it go away is a crash with better manners, so
:func:`no_certified_table_text` prints three things: what was asked, the
nearest certified artifact BELOW the request (in the rank-quantization
idiom the phase already uses), and the two levers — the physics lever
(move the request into the catalog) and the generator lever (extend the
catalog to cover the request).

All six derive from :class:`MinimaxRefusal`, so a caller that wants to
catch "the service said no" without enumerating the taxonomy can, and a
caller that wants to distinguish "no table" from "corrupt bundle" also
can.  They are ``RuntimeError``s rather than ``ValueError``s because none
of them is a grammar error on the caller's part -- the caller asked a
well-formed question the service cannot answer.
"""

from __future__ import annotations


class MinimaxRefusal(RuntimeError):
    """Base of the taxonomy.  Never raised directly."""


class NoCertifiedTable(MinimaxRefusal):
    """F1 — the request lands outside the shipped catalog."""


class AmplificationCap(MinimaxRefusal):
    """F2 — the matched table's certified κ₀ exceeds the declared cap.

    The cap is DATA, read off the catalog's own ``shipping_rule`` block
    (``normal`` / ``versioned_exception`` / ``rejected_above``), not a
    constant in this file: the theory plan owns the number and the
    generator stamps it, so a service-side constant would be a second
    opinion nobody asked for.
    """


class UnknownTarget(MinimaxRefusal):
    """F3 — ``target=`` outside the declared vocabulary."""


class CatalogUnavailable(MinimaxRefusal):
    """F4a — the catalog file could not be resolved or read.

    R2's first row.  This used to be ``except Exception: return None`` at
    two sites, which meant a missing bundle and a healthy one were the
    same event to every caller.
    """


class TableUnreadable(MinimaxRefusal):
    """F4b — the catalog names a payload the loader cannot read.

    R2's second row.  The message names the file AND the ``.npz`` key set
    that was actually found, because "unreadable" and "readable but
    missing ``alpha``" are different defects with the same old symptom.
    """


class CatalogCorrupt(MinimaxRefusal):
    """F4c — an entry in the catalog is malformed.

    R2's third and fourth rows, and the one place the replacement rule is
    a genuine semantic change rather than a louder version of the same
    thing: a malformed entry used to ``continue``, i.e. be treated as
    ABSENT.  A malformed entry means the ARTIFACT IS CORRUPT, not that
    the entry is missing, so the whole catalog refuses and names the
    offending index and field.
    """


class UncertifiedSolveRefused(MinimaxRefusal):
    """F5 — a runtime solve was required and the escape hatch is closed.

    R1 ships in two stages and this refusal is stage 2's whole surface.
    In stage 1 (today) ``LORRAX_MINIMAX_ALLOW_RUNTIME_SOLVE`` defaults to
    ``1``, so this is raised only when a deck or a test closes the hatch
    on purpose — which is exactly the FALSE case the suite ships.
    """


class SamplingUnsupported(MinimaxRefusal):
    """F6 — a sampling point whose analytic character has no live family.

    R4's 2×2: three cells are filled and the fourth (both parts of z
    nonzero — the strip, where MPA lives) is the line
    ``gw/screening.py`` refuses at today.  The refusal is raised from the
    declarative family table, before any physics runs.
    """


# ---------------------------------------------------------------------------
#  Message shaping
# ---------------------------------------------------------------------------

def _fmt_entry(entry) -> str:
    """One line describing a catalog entry, for a refusal's "nearest" row.

    ``claimed_max_error`` and not ``max_error``, and the name is the point:
    this is the catalog's CLAIM about the entry, printed without opening
    the payload, because a refusal must not become a second I/O path that
    can itself fail.  The served value always comes off the payload.
    """
    kappa = ("kappa0 unrecorded" if entry.kappa0 is None
             else f"kappa0 {entry.kappa0:.3g}")
    claimed = ("max_err unrecorded" if entry.claimed_max_error is None
               else f"max_err {entry.claimed_max_error:.2g}")
    return (f"{entry.range_param}={entry.range_max:g} "
            f"({entry.node_count} nodes, {claimed}, {kappa})")


def no_certified_table_text(
    *,
    family: str,
    target: str,
    range_param: str,
    range_value: float,
    error_bound: float,
    n_max: int,
    nearest_below,
    range_lever: str,
    generator_hint: str,
    extra: str = "",
) -> str:
    """F1's message, in the rank-quantization idiom (design §3.2).

    ``nearest_below`` is a :class:`~minimax._catalog.CatalogEntry` or
    ``None``; the "reachable by" row is the family's DECLARED physics
    lever, carried as data on the :class:`~minimax.targets.FamilySpec`
    rather than computed here — the service does not own the physics that
    turns a deck into a range parameter, it owns the fact that the
    parameter has a ceiling.
    """
    lines = [
        f"minimax: no certified {family} table for {range_param}="
        f"{range_value:g} at target {error_bound:.0e} "
        f"(target={target!r}, n_max={n_max}).",
    ]
    if extra:
        lines.append(f"  {extra}")
    if nearest_below is not None:
        lines.append(f"  nearest certified below: {_fmt_entry(nearest_below)}")
    else:
        lines.append(f"  nearest certified below: none — the {family} family "
                     f"ships no entry at or under this request")
    lines.append(f"  reachable by: {range_lever}")
    lines.append(f"  or generate:  {generator_hint}")
    return "\n".join(lines)
