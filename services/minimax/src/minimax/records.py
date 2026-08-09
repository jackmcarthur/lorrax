"""What the door hands back: a quadrature that knows where it came from.

R2, in one sentence: *the return type is not an array pair; it is a record
whose provenance block is mandatory and whose printed form is what the
driver logs.*

Before the extraction, ``build_static_quadrature`` printed ``nodes=`` and
``fit_err~`` and never said whether that came from a shipped table or four
minutes of SciPy, and ``_build_windows_for_branch`` printed
``err<{target_error:.0e}`` — the bound that was REQUESTED, not the one that
was ACHIEVED.  Between them, the system had a way to silently substitute an
uncertified computation for a certified artifact and a log line that
reported the substitution as success.  Everything downstream of that was a
number believed for the wrong reason.

:meth:`Provenance.one_line` is the fix, and it is deliberately one line:
it is emitted once per distinct request, into a log a human reads, and a
paragraph would be scrolled past.

CERTIFIED IS NOT THE SAME AS SHIPPED, and this file is where that
distinction is kept honest.  ``source`` says which ARTIFACT answered
(the shipped bundle, the disk cache, or a runtime solve).  ``certified``
says whether that artifact carries a CERTIFICATION RECORD — a measured
held-out error and a measured amplification, stamped by the generator.
The v1 catalog carries neither, so every table in it is
``source='shipped', certified=False``: it is a real artifact whose claim
about itself has never been checked (survey §2.9 measured that zero test
cells validate a shipped table against its own ``error_bound``).  The v2
bundle carries both.  Printing that difference on every serve is what
makes the certification tier (WP6) visible as missing instead of assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


#: The three artifact kinds, plus the one that is an artifact only by
#: courtesy.  ``'cache-legacy'`` is separated from ``'cache'`` on purpose:
#: the pre-extraction disk cache keyed on ``{solver, logR, target, max_nodes}``
#: with NO solver version, NO scipy/BLAS version and NO machine tag, so a
#: shared ``$HOME`` served one platform's quadrature to another under an
#: identical key.  The WP1 census measured what that means in
#: practice: the G2 gate's "runtime solve" on a warm host reads a
#: 2026-04-09 cache entry, and re-solving the same request on the same
#: machine today yields a different object (Sigma|w| 1.90e5 -> 4.22e4).
#: A legacy hit is archaeology, and it now says so.
SOURCES = ("shipped", "cache", "cache-legacy", "runtime-uncertified")


@dataclass(frozen=True)
class Provenance:
    """Where a served rule came from.  MANDATORY on a Quadrature."""

    #: One of :data:`SOURCES`.
    source: str
    #: The payload path, verbatim, relative to the asset bundle — or the
    #: cache file, or ``None`` for a solve that touched no file.
    catalog_entry: str | None
    #: ``'sha256:...'`` over the payload bytes.  This is the promise the
    #: service can actually keep across machines: not "any host would
    #: compute this", but "this is table X and its bytes hash to H".
    table_hash: str
    #: The commit that generated the artifact, in the ``kin_ion`` /
    #: ``qirr_store`` stamp idiom — ``'unrecorded'`` where the schema
    #: predates the stamp, which is a fact about the bundle and not a
    #: placeholder to be tidied away.
    generator_commit: str
    #: e.g. ``'cpu:scipy-1.17.1/numpy-2.4.3'``, measured AT GENERATION.
    generation_backend: str
    #: Does the artifact carry a certification record (measured held-out
    #: error + measured amplification, stamped)?  See the module docstring.
    certified: bool

    def one_line(self) -> str:
        """The provenance, as the driver prints it."""
        if self.source == "shipped":
            what = f"shipped {self.catalog_entry}"
        elif self.source == "cache":
            what = f"cache {self.catalog_entry}"
        elif self.source == "cache-legacy":
            what = (f"cache LEGACY-UNVERSIONED {self.catalog_entry} "
                    f"(no solver/backend key; provenance unknowable)")
        else:
            what = "runtime solve, no artifact"
        cert = "CERTIFIED" if self.certified else "UNCERTIFIED"
        return (f"{what} {self.table_hash} gen {self.generator_commit} "
                f"backend {self.generation_backend} {cert}")


#: The provenance a runtime solve carries.  Its ``table_hash`` is over the
#: bytes the solve produced, so two hosts that disagree say so by hash.
def runtime_provenance(table_hash: str, backend: str) -> Provenance:
    return Provenance(
        source="runtime-uncertified",
        catalog_entry=None,
        table_hash=table_hash,
        generator_commit="n/a (solved in-process)",
        generation_backend=backend,
        certified=False)


@dataclass(frozen=True)
class Quadrature:
    """A rule, in the SCALED units the catalog tabulates.

    THE RESCALE STAYS WITH THE CALLER, deliberately (design §3.3): the
    three ``solve_*_interval`` / ``solve_phase_*`` wrappers in
    ``gw.minimax_screening`` rescale into Rydberg and name *windows*, and
    they stay behind.  So this object carries the table exactly as it sits
    on disk — ``tau`` on ``[1, R]`` or ``[0, A]``, ``alpha`` beside it —
    and the physical error is ``max_error / x_min`` at the caller's
    convention.  That is what makes the extraction bit-identical BY
    CONSTRUCTION rather than by tolerance: the bytes the door hands out
    are the bytes ``np.load`` returned, and the arithmetic that used to
    follow them still follows them, in the same module, unchanged.
    """

    nodes: np.ndarray        # float64
    weights: np.ndarray      # float64, or complex128 for strip families
    family: str
    target: str
    range_param: str
    range_value: float
    #: What was ASKED.
    error_bound: float
    #: What was MEASURED at certification (or by the solver, for a solve).
    max_error: float
    #: Amplification.  ``None`` where the artifact does not record one —
    #: the v1 catalog does not, and inventing a number here would be
    #: exactly the kind of unearned confidence R2 exists to end.
    kappa0: float | None
    #: Node/phase sensitivity.  Reserved; no shipped schema records it yet.
    kappa1: float | None
    provenance: Provenance

    @property
    def node_count(self) -> int:
        return int(np.asarray(self.nodes).shape[0])

    def one_line(self) -> str:
        """The whole serve, as one log line.  R2's announcement payload."""
        kappa = ("kappa0 unrecorded" if self.kappa0 is None
                 else f"kappa0 {self.kappa0:.3g}")
        return (f"minimax: served {self.family}/{self.target} "
                f"{self.range_param}={self.range_value:g} "
                f"target {self.error_bound:.0e} -> {self.node_count} nodes, "
                f"max_err {self.max_error:.3g}, {kappa} | "
                f"{self.provenance.one_line()}")


@dataclass(frozen=True)
class CatalogView:
    """An enumerable, assertable view of the bundle.  No solve, ever."""

    schema_version: int
    #: ``tuple[CatalogEntry, ...]`` — kept untyped here to avoid a cycle
    #: with :mod:`minimax._catalog`, which imports this module.
    entries: tuple
    #: The ``shipping_rule`` block, if the schema carries one.
    shipping_rule: dict
    source_path: str

    def __len__(self) -> int:
        return len(self.entries)

    def families(self) -> tuple[str, ...]:
        return tuple(sorted({e.family for e in self.entries}))

    def for_family(self, family: str) -> tuple:
        return tuple(e for e in self.entries if e.family == family)
