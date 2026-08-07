"""Re-export shim — the orbit machinery moved to ``symmetry_maps.orbit_syms``.

``services/symmetry_maps/`` is the standalone service now (charter wave 1);
this module only forwards to it so lorrax stays green while the call sites
migrate.  New code imports the top-level ``symmetry_maps`` door directly.

DELETION GATE: this file is deleted by the phase-wide cleanup commit after
all four wave-1 branches land — WAVE1_BRIEF ruling 2.

THE ONE EDGE THAT DID NOT WAIT.  ``centroid/kmeans_isdf.py:581`` — the
sanctioned layering exception **R3** — was rewired to the door in the same
commit that moved this file, because ratchet deletion in the extraction
commit is what makes R3 dissolved rather than relabelled (WAVE1_BRIEF
ruling 5).  Its ``_L2_UPWARD_EXCEPTIONS`` entry is gone from
``tests/test_layering.py`` and ``docs/architecture/layers.md`` says so.

FORWARDING IS BY ``__getattr__`` (PEP 562), not by re-export alone, so the
private names (``_CANON_INV``, ``_orbit_lex_winner``, and the helpers the
centroid tests reach for) keep resolving.  The public names are ALSO bound
at module scope, the way distrib_la's shims bound theirs.
"""
from __future__ import annotations

from ffi import _services

_services.ensure_on_path()

from symmetry_maps import (                                    # noqa: E402
    build_real_space_syms,
    canonicalize_orbit,
    compute_centroid_sym_perm,
    compute_rgrid_sym_perm,
    orbit_images,
    recover_symmorphic_density_point_group,
    unfold_orbit_unique_with_id,
)
from symmetry_maps import orbit_syms as _impl                  # noqa: E402

__all__ = [
    "build_real_space_syms",
    "canonicalize_orbit",
    "compute_centroid_sym_perm",
    "compute_rgrid_sym_perm",
    "orbit_images",
    "recover_symmorphic_density_point_group",
    "unfold_orbit_unique_with_id",
]


def __getattr__(name):
    """Anything else — including the private names — from the service."""
    try:
        return getattr(_impl, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}; it is a "
            f"forwarding shim for 'symmetry_maps.orbit_syms', which has no "
            f"such attribute either") from None


def __dir__():
    return sorted(set(globals()) | set(dir(_impl)))
