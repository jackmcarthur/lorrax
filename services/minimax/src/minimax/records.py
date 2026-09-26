"""What the door hands back: a quadrature that knows where it came from.

The return type is a record, not an array pair, and its provenance block is
mandatory.  :meth:`Provenance.one_line` is what the driver logs, once per
distinct request.  ``source`` says which artifact answered (the disk cache
or a runtime solve); ``certified`` says whether that artifact carries a
measured certification record, which no runtime solve does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


#: The two cache kinds and the runtime solve.  ``'cache-legacy'`` is separated from ``'cache'`` on purpose:
#: the pre-extraction disk cache keyed on ``{solver, logR, target, max_nodes}``
#: with NO solver version, NO scipy/BLAS version and NO machine tag, so a
#: shared ``$HOME`` served one platform's quadrature to another under an
#: identical key.  The WP1 census measured what that means in
#: practice: the G2 gate's "runtime solve" on a warm host reads a
#: 2026-04-09 cache entry, and re-solving the same request on the same
#: machine today yields a different object (Sigma|w| 1.90e5 -> 4.22e4).
#: A legacy hit is archaeology, and it now says so.
SOURCES = ("cache", "cache-legacy", "runtime-uncertified")


@dataclass(frozen=True)
class Provenance:
    """Where a served rule came from.  MANDATORY on a Quadrature."""

    #: One of :data:`SOURCES`.
    source: str
    #: The cache file, or ``None`` for a solve that touched no file.
    catalog_entry: str | None
    #: ``'sha256:...'`` over the payload bytes.  This is the promise the
    #: service can actually keep across machines: not "any host would
    #: compute this", but "this is table X and its bytes hash to H".
    table_hash: str
    #: The commit that generated the artifact, in the ``kin_ion`` /
    #: ``qirr_store`` stamp idiom.
    generator_commit: str
    #: e.g. ``'cpu:scipy-1.17.1/numpy-2.4.3'``, measured AT GENERATION.
    generation_backend: str
    #: Does the artifact carry a certification record (measured held-out
    #: error + measured amplification, stamped)?  See the module docstring.
    certified: bool

    def one_line(self) -> str:
        """The provenance, as the driver prints it."""
        if self.source == "cache":
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
    """A rule, in the SCALED units the solver works in.

    THE RESCALE STAYS WITH THE CALLER (design §3.3): the three
    ``solve_*_interval`` / ``solve_phase_*`` wrappers in
    ``gw.minimax_screening`` rescale into Rydberg and name *windows*.  This
    object carries the rule as solved — ``tau`` on ``[1, R]`` or
    ``[0, A]``, ``alpha`` beside it — and the physical error is
    ``max_error / x_min`` at the caller's convention.
    """

    nodes: np.ndarray        # float64
    weights: np.ndarray      # float64, or complex128 for strip families
    family: str
    target: str
    range_param: str
    range_value: float
    #: What was ASKED.
    error_bound: float
    #: What the solver MEASURED.
    max_error: float
    #: Amplification, measured on the rule.
    kappa0: float | None
    #: Node/phase sensitivity.  Reserved; nothing records it yet.
    kappa1: float | None
    provenance: Provenance

    @property
    def node_count(self) -> int:
        return int(np.asarray(self.nodes).shape[0])
