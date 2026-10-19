"""The refusals — every gap named.

Each refusal ships with the case where it returns FALSE, in
``services/minimax/tests/``.  All derive from :class:`MinimaxRefusal`, so a
caller can catch "the service said no" without enumerating the taxonomy.
They are ``RuntimeError``s rather than ``ValueError``s because none of them
is a grammar error on the caller's part: the caller asked a well-formed
question the service cannot answer.
"""

from __future__ import annotations


class MinimaxRefusal(RuntimeError):
    """Base of the taxonomy.  Never raised directly."""


class UnknownTarget(MinimaxRefusal):
    """F3 — ``target=``, ``family=`` or a selector outside the declared
    vocabulary."""


class UncertifiedSolveRefused(MinimaxRefusal):
    """F5 — the requested family has no in-process solver.

    ``serve`` computes every rule at run time, so a declared family with no
    solver (``complex_laplace``, ``damped_line``) refuses by name.
    """


class SamplingUnsupported(MinimaxRefusal):
    """F6 — a sampling point whose analytic character has no live family.

    R4's 2×2: three cells are filled and the fourth (both parts of z
    nonzero — the strip, where MPA lives) has no family.  The refusal is
    raised from the declarative family table, before any physics runs.
    """
