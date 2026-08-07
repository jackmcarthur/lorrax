"""Re-export shim — the TRS measurement moved to
``symmetry_maps.density_symmetry_check``.

``services/symmetry_maps/`` is the standalone service now (charter wave 1);
this module only forwards to it so lorrax stays green while the call sites
migrate.  New code imports the top-level ``symmetry_maps`` door directly.

DELETION GATE: this file is deleted by the phase-wide cleanup commit after
all four wave-1 branches land — WAVE1_BRIEF ruling 2.

ONE SIGNATURE CHANGE CAME WITH THE MOVE, and it is a compat default
(WAVE1_BRIEF ruling 4): ``check_density_symmetries`` and
``cached_density_symmetry_check`` grew keyword-only ``valence_density_fn``
and ``spin_degeneracy_fn``.  Both default to ``None``, which performs
EXACTLY the lazy ``psp.get_DFT_mtxels`` imports the module always did, so
every call site through this shim — ``file_io/wfn_loader.py:327`` is the
only live one — is unchanged.  Adoption of the explicit form is registered
to the wfn_loader orchestrator.

FORWARDING IS BY ``__getattr__`` (PEP 562), not by re-export alone, so the
module constants and private helpers that the density-check tests pin
(``TOL_TRS``, ``TOL_SPATIAL``, ``MAX_K_DEFAULT``, ``_CACHE``,
``_env_int``, ``_negation_pairs``, …) keep resolving at this path.  The
four names in the implementation's ``__all__`` are ALSO bound at module
scope, the way distrib_la's shims bound theirs.
"""
from __future__ import annotations

from ffi import _services

_services.ensure_on_path()

from symmetry_maps import (                                    # noqa: E402
    DensitySymmetryReport,
    cached_density_symmetry_check,
    check_density_symmetries,
    trs_check_mode,
)
from symmetry_maps import density_symmetry_check as _impl      # noqa: E402

__all__ = [
    "DensitySymmetryReport",
    "cached_density_symmetry_check",
    "check_density_symmetries",
    "trs_check_mode",
]


def __getattr__(name):
    """Anything else — constants, private helpers — from the service."""
    try:
        return getattr(_impl, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}; it is a "
            f"forwarding shim for 'symmetry_maps.density_symmetry_check', "
            f"which has no such attribute either") from None


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
