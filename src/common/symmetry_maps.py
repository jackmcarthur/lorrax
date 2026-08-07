"""Re-export shim — the symmetry tables moved to ``symmetry_maps.maps``.

``services/symmetry_maps/`` is the standalone service now (charter wave 1);
this module only forwards to it so lorrax stays green while the call sites
migrate.  New code imports the top-level ``symmetry_maps`` door directly.

DELETION GATE: this file is deleted by the phase-wide cleanup commit after
all four wave-1 branches land — WAVE1_BRIEF ruling 2.  Unlike distrib_la
(whose replumb deleted its own shims), wave-1 services are cross-consumed
by files on sibling branches, so shim deletion is ONE commit after the
owner merges all four, and branch land-order stops mattering.

FORWARDING IS BY ``__getattr__`` (PEP 562), not by re-export alone,
because the private names are load-bearing at real call sites:
``tests/test_symmetry_unfold.py`` imports ``_I_SIGMA_Y``, the star suite
and the re-derivation script import ``_star_row_order`` /
``_star_conj_flags``, and ``misc/`` and ``tests/bench/`` reach in for
others.  A shim that forwarded only ``__all__`` would look complete and
break exactly the uncollected callers nobody runs before a merge.  The
public names are ALSO bound at module scope, the way distrib_la's shims
bound theirs, so ``from common.symmetry_maps import SymMaps`` resolves
without going through the hook and ``dir()`` is honest either way.
"""
from __future__ import annotations

from ffi import _services

_services.ensure_on_path()

from symmetry_maps import (                                    # noqa: E402
    KStarMap,
    SymMaps,
    find_irreducible_bz_points,
    kgrid_shift_map,
    slice_q_full_to_ibz,
    star_broadcast,
    star_select,
    star_spread,
    tau_phase_row,
    trs_augment_U,
    unfold_psi,
    unfold_v_q,
    unfold_v_q_bispinor_lorentz,
)
from symmetry_maps import maps as _impl                        # noqa: E402

__all__ = [
    "KStarMap",
    "SymMaps",
    "find_irreducible_bz_points",
    "kgrid_shift_map",
    "slice_q_full_to_ibz",
    "star_broadcast",
    "star_select",
    "star_spread",
    "tau_phase_row",
    "trs_augment_U",
    "unfold_psi",
    "unfold_v_q",
    "unfold_v_q_bispinor_lorentz",
]


def __getattr__(name):
    """Anything else — including the private names — from the service."""
    try:
        return getattr(_impl, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}; it is a "
            f"forwarding shim for 'symmetry_maps.maps', which has no such "
            f"attribute either") from None


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
