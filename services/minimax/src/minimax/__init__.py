"""Numerical quadrature service for LORRAX frequency kernels.

The package is the public door: consumers import ``minimax`` and never its
submodules. Importing the package loads no JAX or SciPy. Rule constructors
load optional numerical libraries only when called.

The target is part of every rule contract. ``serve`` computes the static,
imaginary-probe, and regularized crossing screening rules in process. Sigma's
complex denominator rectangles use ``build_uniform_rule``. The shared-pole W
bank uses ``response_group_rules`` because it needs current-frequency value
and derivative projections; the GN-PPM imaginary probe uses
``response_laplace_rule``. ``response_bank_rule`` has no production caller. Finite-temperature
KMS response has ``matsubara_response_rule``. The GN-PPM odd kernel shares
its even rule's times through ``augment_odd_laplace``. MPA's damped line and
rectangle time rules are built through the four ``damped_*_rule`` names.

Three analytic reciprocal constructors cover only their stated
one-dimensional domains: positive real intervals, the gapped odd real axis,
and a normalized fixed-height line. Sigma PPM requests the line for crossing
windows with real poles. Their validity does not transfer to a different
regularized kernel or to a complex box by matching bandwidth alone.
See ``docs/services/minimax.md`` for units, error currencies and certificates.
"""

from __future__ import annotations

from minimax.door import (
    family_for_character,
    noncrossing_kappa0,
    reset_announcements,
    serve,
    solve_uncertified,
)
from minimax.records import (
    SOURCES,
    Provenance,
    Quadrature,
    runtime_provenance,
)
from minimax.refusals import (
    MinimaxRefusal,
    SamplingUnsupported,
    UncertifiedSolveRefused,
    UnknownTarget,
)
from minimax.targets import (
    CHARACTERS,
    FAMILIES,
    TARGETS,
    FamilySpec,
    TargetSpec,
    families_for_character,
)

#: The solver half, reached lazily: the names the door itself calls.
#: Lazy because :mod:`minimax.solver` is the service's ONLY scipy consumer,
#: so ``import minimax`` never imports an optimiser.
_SOLVER_NAMES = (
    # target functions
    "G_hgl", "G_fermi", "tau_max_hgl", "tau_max_fermi",
    # the two grid drivers the door reaches
    "noncrossing_imag_grids", "crossing_grids",
    # the solvers under them
    "solve_noncrossing_imag", "solve_crossing",
)

# Positive minimax fitting for complex-frequency resolvents, over the
# analytic sinc rule in ``sector``.  A production surface: the pane/slab
# Laplace route reaches it as ``minimax.fit_damped_reciprocal``
# (``gw/mpa/sigma_windows.py::_laplace_nodes``).  SciPy, so lazy.
_FREQUENCY_FIT_NAMES = (
    "DampedReciprocalFit", "fit_damped_reciprocal",
)

# Uniform denominator-box rules are a production service surface.  Keep the
# numerical module lazy, like every other SciPy-backed builder below, so a
# bare import still pays no solver dependency.
_UNIFORM_RULE_NAMES = (
    "UniformRule", "box_samples", "boundary_samples", "build_uniform_rule",
    "rule_roundoff_amplification", "rule_sup_error",
    "uniform_rule_solver_identity",
)

# Levelled (minimax-optimal) noncrossing rules: NumPy only, but lazy like
# every builder so a bare import pays nothing.
_LEVELLED_NAMES = ("noncrossing_levelled", "certify_noncrossing")

# Experimental constructors remain lazy: a bare import stays NumPy-only.
_ANALYTIC_NAMES = ("positive_reciprocal", "odd_reciprocal",
                   "damped_line_reciprocal", "analytic_line_box_rule")
_ODD_LAPLACE_NAMES = ("augment_odd_laplace",)
_DAMPED_RULE_NAMES = (
    "DEFAULT_DAMPED_REL_TOL", "DEFAULT_WAVELENGTHS_PER_PANEL",
    "damped_line_rule", "damped_rectangle_rule",
    "damped_rectangle_gauss_rule", "damped_rectangle_positive_rule",
)


_RESPONSE_RULE_NAMES = ("RESPONSE_RULE_CAPACITY", "RESPONSE_NODE_CAPACITY", "response_bank_rule", "response_laplace_rule", "response_group_rules")

# Finite-temperature bosonic Matsubara rules (KMS-bounded tau correlations).
# SciPy (pivoted QR), so lazy.
_MATSUBARA_RULE_NAMES = ("matsubara_response_rule",)


def __getattr__(name: str):
    """PEP 562 lazy door for the solver half.

    ``from minimax import crossing_grids`` imports scipy at that
    moment and not before.  ``import minimax`` never does.
    """
    if name in _RESPONSE_RULE_NAMES:
        from minimax import response_rules as _response
        return getattr(_response, name)
    if name in _MATSUBARA_RULE_NAMES:
        from minimax import matsubara_rules as _matsubara   # noqa: PLC0415
        return getattr(_matsubara, name)
    if name in _SOLVER_NAMES:
        from minimax import solver as _solver          # noqa: PLC0415
        return getattr(_solver, name)
    if name in _FREQUENCY_FIT_NAMES:
        from minimax import frequency_fit as _fit      # noqa: PLC0415
        return getattr(_fit, name)
    if name in _UNIFORM_RULE_NAMES:
        from minimax import uniform_rule as _uniform   # noqa: PLC0415
        return getattr(_uniform, name)
    if name in _LEVELLED_NAMES:
        from minimax import levelled as _levelled  # noqa: PLC0415
        return getattr(_levelled, name)
    if name in _ANALYTIC_NAMES:
        from minimax import analytic as _analytic
        return getattr(_analytic, name)
    if name in _ODD_LAPLACE_NAMES:
        from minimax import odd_laplace as _odd_laplace
        return getattr(_odd_laplace, name)
    if name in _DAMPED_RULE_NAMES:
        from minimax import damped_rules as _damped_rules
        if name == "DEFAULT_DAMPED_REL_TOL":
            return _damped_rules.DEFAULT_REL_TOL
        return getattr(_damped_rules, name)
    raise AttributeError(f"module 'minimax' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_SOLVER_NAMES)
                  | set(_FREQUENCY_FIT_NAMES)
                  | set(_UNIFORM_RULE_NAMES)
                  | set(_LEVELLED_NAMES) | set(_RESPONSE_RULE_NAMES)
                  | set(_MATSUBARA_RULE_NAMES) | set(_ANALYTIC_NAMES)
                  | set(_ODD_LAPLACE_NAMES) | set(_DAMPED_RULE_NAMES))


__all__ = [
    # --- the vocabulary, as data -------------------------------------------
    "TARGETS", "FAMILIES", "CHARACTERS", "TargetSpec", "FamilySpec",
    "families_for_character", "family_for_character",
    # --- what you get back -------------------------------------------------
    "Quadrature", "Provenance", "SOURCES", "runtime_provenance",
    # --- the door ----------------------------------------------------------
    "serve", "solve_uncertified", "reset_announcements",
    "noncrossing_kappa0",
    # --- the refusals ------------------------------------------------------
    "MinimaxRefusal", "UnknownTarget", "UncertifiedSolveRefused",
    "SamplingUnsupported",
    # --- complex-frequency resolvent fitting (lazy; scipy) -----------------
    *_FREQUENCY_FIT_NAMES,
    # --- uniform denominator-box rules (lazy; scipy) -----------------------
    *_UNIFORM_RULE_NAMES,
    # --- levelled noncrossing rules (lazy; numpy) --------------------------
    *_LEVELLED_NAMES,
    # --- exploratory reciprocal constructors (lazy; scipy/mpmath) ---------
    *_ANALYTIC_NAMES,
    *_ODD_LAPLACE_NAMES,
    *_DAMPED_RULE_NAMES,
    # --- the offline solvers (lazy; scipy) ---------------------------------
    *_SOLVER_NAMES,
    *_RESPONSE_RULE_NAMES,
    *_MATSUBARA_RULE_NAMES,
]
