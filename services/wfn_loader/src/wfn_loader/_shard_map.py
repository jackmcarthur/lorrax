"""``shard_map`` — the one way this package enters manual-sharding mode.

PROVENANCE.  Copied VERBATIM (this docstring aside) from
``services/distrib_la/src/distrib_la/_shard_map.py`` at e9340d1, which was
itself ported from LORRAX ``src/common/shard_map.py`` at 96a6399 with the
resolution made LAZY (see "Two changes" below).  It exists HERE, rather
than being imported from lorrax or from a sibling service, because
coordination ruling 3 says so: a standalone package cannot reach into the
monorepo for its jax version boundary, and one service reaching into
another for it would only move the coupling.  Consolidating the two copies
into :mod:`lxkit` is registered as post-wave work; until then the
duplication is deliberate and the two files are byte-comparable on
purpose.

WHAT IS A RENAME AND WHAT IS NOT — MEASURED, not assumed.  Probed
in-container on Perlmutter, jax 0.5.3 (``nvcr.io/nvidia/jax:25.04-py3``)
and jax 0.7.0 (``ghcr.io/nvidia/jax:jax-2025-07-21``), CPU backend forced
to 4 devices:

===================  ==========================  ==========================
                     jax 0.5.3                   jax 0.7.0
===================  ==========================  ==========================
``jax.shard_map``    ABSENT                      present, a distinct object
``check_rep=``       accepted (experimental)     accepted (experimental only)
``check_vma=``       ``TypeError``               accepted (modern only)
===================  ==========================  ==========================

**``check_rep`` → ``check_vma`` is a PURE RENAME** — not merely equivalent
by observation, the same variable.  The 0.7.0 experimental wrapper's body,
read out of the running container, ends ``jshmap._shard_map(..., check_vma=
check_rep, ...)``: the value is handed unmodified to the same internal
``_shard_map`` that ``jax.shard_map`` calls, so there is no code path on
which the two can differ.  This package has exactly ONE ``shard_map``
site (:func:`wfn_loader.loader._phdf5_unfold_kernel`) and it passes
``check_vma=False``, which is why it does not need
:func:`lxkit.jax_compat.mark_varying` — VMA checking is off for that
``shard_map`` and its carries are exempt.

``auto=`` → ``axis_names=`` is an inversion (``axis_names =
frozenset(mesh.axis_names) - auto``), and this package has ZERO sites
passing either — the one site inherits "manual over all mesh axes", which
is what both defaults mean.  Neither kwarg is exposed here.  If a site
ever needs partial-manual mode, add it HERE, with its census.

Two changes from the lorrax original
------------------------------------
1. **The resolution is lazy.**  lorrax resolves at IMPORT and exports the
   answer as the constant ``SHARD_MAP_MODE``; that is fine for a module
   nobody imports early, but it is the property :mod:`lxkit.jax_compat`
   had to give up for the same reason — an import-time jax probe in a
   foundation package initializes nothing here today and everything
   tomorrow.  Resolved and memoized on first :func:`shard_map` call.
2. **``select_mode`` stays PURE** (as in lorrax), so the whole decision
   table — including the branch this container cannot exhibit and the
   refusal — is testable by construction on any jax, including none.  The
   compatibility path nobody can reach is exactly the one that produced
   the defect ``lxkit.jax_compat`` was written to remove.

This is NOT the generic wrapper ``docs/architecture/layers.md`` §2 forbids.
It abstracts nothing: every call site still spells its own ``mesh``,
``in_specs`` and ``out_specs`` in full, at the site.  What it owns is a
jax API VERSION decision, which has nowhere else to live.
"""

from __future__ import annotations

__all__ = ["shard_map", "shard_map_mode", "select_mode",
           "MODERN_SHARD_MAP_SINCE", "ShardMapSupportError"]


class ShardMapSupportError(RuntimeError):
    """This jax exposes no ``shard_map`` this package knows how to call."""


#: First jax exposing the modern ``jax.shard_map``.  Recorded for the
#: refusal message ONLY — the decision is made by probing for the SYMBOL,
#: never by comparing versions, because every NVIDIA container jax is a
#: source build that restamps ``__version__`` with the date it was probed.
MODERN_SHARD_MAP_SINCE = (0, 7, 0)


def select_mode(has_modern: bool, has_experimental: bool, version_info) -> str:
    """Which ``shard_map`` spelling applies, or raise.  PURE — no jax state."""
    if has_modern:                       # jax >= 0.7.0
        return "jax.shard_map"
    if has_experimental:                 # jax <= 0.6.x
        return "jax.experimental.shard_map"
    raise ShardMapSupportError(
        "jax %s exposes neither jax.shard_map (added at %s) nor "
        "jax.experimental.shard_map, so wfn_loader cannot enter manual "
        "sharding mode at all.  Refusing rather than degrading: the phdf5 "
        "unfold kernel in this package is written in shard_map, so there "
        "is no meaningful fallback.  Fix: add the new spelling to "
        "wfn_loader/_shard_map.py, or run a jax that has one."
        % (".".join(map(str, version_info)),
           ".".join(map(str, MODERN_SHARD_MAP_SINCE))))


_RESOLVED: tuple[str, tuple, object] | None = None
_ANNOUNCED = False


def _resolve() -> tuple[str, tuple, object]:
    """``(mode, version_info, experimental_module_or_None)``, memoized."""
    global _RESOLVED
    if _RESOLVED is None:
        import jax
        try:
            import jax.experimental.shard_map as _exp
            has_experimental = hasattr(_exp, "shard_map")
        except ImportError:
            _exp, has_experimental = None, False
        ver = tuple(jax.version.__version_info__)
        _RESOLVED = (select_mode(hasattr(jax, "shard_map"),
                                 has_experimental, ver), ver, _exp)
    return _RESOLVED


def shard_map_mode() -> str:
    """Which spelling this process resolved to.  For tests and for logs."""
    mode, ver, _ = _resolve()
    if mode == "jax.shard_map":
        return mode
    return "jax.experimental.shard_map (jax %s has no jax.shard_map)" % (
        ".".join(map(str, ver)),)


def shard_map(f, *, mesh, in_specs, out_specs, check_vma=True):
    """Map ``f`` over shards of its arguments.  This package's only spelling.

    ``mesh`` / ``in_specs`` / ``out_specs`` are REQUIRED and keyword-only,
    stricter than either jax spelling and for a measured reason: at 0.7.0
    ``jax.shard_map`` has ``in_specs=None`` and by the current released
    docs it has become ``in_specs=jax.sharding.Infer``, so a site that
    omitted it would change meaning between two jaxes that both "have
    ``jax.shard_map``".
    """
    global _ANNOUNCED
    mode, ver, exp = _resolve()
    if not _ANNOUNCED:
        _ANNOUNCED = True
        print("  [wfn_loader] shard_map via %s (jax %s)"
              % (shard_map_mode(), ".".join(map(str, ver))), flush=True)
    if mode == "jax.shard_map":
        import jax
        return jax.shard_map(f, mesh=mesh, in_specs=in_specs,
                             out_specs=out_specs, check_vma=check_vma)
    # check_vma IS check_rep under a different name -- upstream's own
    # 0.7.0 wrapper passes ``check_vma=check_rep`` verbatim.
    return exp.shard_map(f, mesh, in_specs, out_specs, check_rep=check_vma)
