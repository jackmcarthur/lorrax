"""What :func:`serve` hands back: a quadrature that knows where it came from.

The return type is a record, not an array pair, and its provenance block is
mandatory.  :meth:`Provenance.one_line` is what the driver logs, once per
distinct request.  Every rule is solved in process at run time and nothing
is stored across processes (owner, 2026-09-28), so ``source`` is always
``'runtime-uncertified'``; ``certified`` says whether the rule carries a
measured certification record, which no runtime solve does.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import numpy as np


def backend_tag() -> str:
    """The numerics backend a runtime solve ran on, for its provenance.

    numpy and scipy versions plus the machine: node positions can differ in
    the last digits between hosts because the solve goes through the host's
    LAPACK.  scipy is imported lazily and tolerated absent: the numpy-only
    solvers (``levelled``) never need it.
    """
    try:
        import scipy                                   # noqa: PLC0415
        scipy_v = scipy.__version__
    except Exception:                                  # pragma: no cover
        # Genuinely broad, and it is not a demotion: this value is a
        # provenance LABEL, so "scipy could not be interrogated" must produce
        # a distinct, stable tag rather than an exception that would take
        # down a solve that does not need scipy at all.
        scipy_v = "absent"
    return (f"cpu:numpy-{np.__version__}/scipy-{scipy_v}/"
            f"{os.uname().machine if hasattr(os, 'uname') else 'unknown'}")


def payload_hash(tau: np.ndarray, w: np.ndarray) -> str:
    """``'sha256:...'`` over a rule's node and weight bytes."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(tau, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(w).tobytes())
    return f"sha256:{h.hexdigest()[:16]}"


@dataclass(frozen=True)
class Provenance:
    """Where a served rule came from.  MANDATORY on a Quadrature."""

    #: ``'runtime-uncertified'``: solved in this process.
    source: str
    #: ``'sha256:...'`` over the payload bytes, so two hosts that disagree
    #: say so by hash.
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
        what = "runtime solve, no artifact"
        cert = "CERTIFIED" if self.certified else "UNCERTIFIED"
        return (f"{what} {self.table_hash} gen {self.generator_commit} "
                f"backend {self.generation_backend} {cert}")


def runtime_provenance(tau: np.ndarray, w: np.ndarray) -> Provenance:
    """The provenance a runtime solve of ``(tau, w)`` carries."""
    return Provenance(
        source="runtime-uncertified",
        table_hash=payload_hash(tau, w),
        generator_commit="n/a (solved in-process)",
        generation_backend=backend_tag(),
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
