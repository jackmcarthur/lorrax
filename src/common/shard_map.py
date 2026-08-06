"""``shard_map`` — the one way this tree enters manual-sharding mode.

Why this module exists
----------------------
``shard_map`` has moved and changed shape.  Both spellings coexist on jax
0.7.0, which is this tree's migration target, and they are **not the same
object** --- measured in-container on 0.7.0 (see MEASURED table below)::

    jax.experimental.shard_map.shard_map(
        f, mesh, in_specs, out_specs, check_rep=True, auto=frozenset())

    jax.shard_map(
        f=None, /, *, out_specs, axis_names=set(),
        in_specs=None, mesh=None, check_vma=True)

``jax.experimental.shard_map`` is **deprecated as of jax 0.8.0** (upstream's
own words: "jax.experimental.shard_map is deprecated in v0.8.0. Use
jax.shard_map instead."), and it does not support nesting.  So the modern
spelling is where this tree has to end up.  But ``jax.shard_map`` **does not
exist at all on 0.5.3**, which is the jax the production container still
ships, so the tree cannot simply be renamed across.

One module therefore decides, once, which symbol and which kwarg spelling this
jax wants; everything else imports the decided ``shard_map``.  This is the same
contract as ``common.vma``, and for the same reason: the alternative --- a
``try: from jax import shard_map / except ImportError`` in each file --- was
already live in ``bse/bse_ring_comm.py`` and ``bse/bse_stack_matvec.py``, in
two copies, and **neither copy translated the kwarg**.  Those two shims happen
to work only because their 14 call sites pass no check kwarg at all; the first
site to add ``check_vma=False`` under them would have raised ``TypeError`` on
0.5.3 with nothing to say why.

What is a RENAME and what is NOT — MEASURED, not assumed
--------------------------------------------------------
Probed in-container on Perlmutter, jax 0.5.3 (``nvcr.io/nvidia/jax:25.04-py3``)
and jax 0.7.0 (``ghcr.io/nvidia/jax:jax-2025-07-21``), CPU backend forced to 4
devices so the device count is deterministic:

===================  ==========================  ==========================
                     jax 0.5.3                   jax 0.7.0
===================  ==========================  ==========================
``jax.shard_map``    ABSENT                      present, a distinct object
``check_rep=``       accepted (experimental)     accepted (experimental only)
``check_vma=``       ``TypeError``               accepted (modern only)
``lax.pvary``        absent                      present
===================  ==========================  ==========================

**1. ``check_rep`` -> ``check_vma`` is a PURE RENAME.**  Not merely equivalent
by observation --- the same variable.  The 0.7.0 experimental wrapper's entire
body, read out of the running container, is::

    axis_names = frozenset(mesh.axis_names) - auto
    return jshmap._shard_map(
        f, mesh=mesh, in_specs=in_specs, out_specs=out_specs,
        check_vma=check_rep, axis_names=axis_names, _skip_mesh_check=True)

``check_vma=check_rep``: the value is handed unmodified to the same internal
``_shard_map`` that ``jax.shard_map`` calls.  There is no code path on which
the two can differ.  Confirmed behaviourally as well: with an ``out_specs``
that claims replication the body does not deliver, both ``check_rep=True`` and
``check_vma=True`` raise the same ``ValueError``, both ``=False`` return
**bit-identical** results (sha256 ``32064687fdcfa677`` --- the same bytes on
0.5.3's ``check_rep=False``), and for an unmarked ``fori_loop`` carry both
defaults raise the same VMA carry ``TypeError`` while both ``=False`` pass.

So the ~70 sites in this tree that pass ``check_rep=False`` are exempt from VMA
marking for a reason that survives the rename exactly.  None of them needed
marking instead of a kwarg swap.

**2. ``auto=`` -> ``axis_names=`` IS an inversion --- and this tree has zero of
them.**  The wrapper line above is upstream's own translation:
``axis_names = frozenset(mesh.axis_names) - auto``.  ``auto`` named the axes
left AUTOMATIC; ``axis_names`` names the axes that are MANUAL.  An AST census
of all 94 ``shard_map`` construction sites under ``src/`` and ``tests/`` found
**no site passing ``auto=`` at all**, so every site inherits ``auto=frozenset()``
--- and the two defaults coincide: ``axis_names=frozenset()`` is documented as
"manual over all mesh axes", which is what ``frozenset(mesh.axis_names) - {}``
evaluates to.  Measured: ``axis_names=frozenset()`` and
``axis_names={'x','y'}`` behave identically on a 2x2 mesh, while a strict
subset ``axis_names={'x'}`` raises.  The inversion is real; it simply never
bites here.  **This module deliberately does not expose either kwarg.**  If a
site ever needs partial-manual mode, add it HERE, with its census.

**3. The one difference that is NOT a rename: ``_skip_mesh_check``.**  The
experimental wrapper passes ``_skip_mesh_check=True``; ``jax.shard_map`` has no
such parameter and the internal default is ``False``.  It governs whether an
explicit ``mesh=`` must agree with the ambient mesh context.  Measured on
0.7.0: with no ambient context --- which is this tree's situation everywhere,
since ``src/`` contains **zero** uses of ``jax.set_mesh``/``jax.sharding.use_mesh``
--- both spellings accept the same call.  Under a deliberately mismatched
``use_mesh`` context both spellings fail, the modern one earlier and with a
better message.  No case was found where the experimental spelling succeeds and
the modern one fails.  This is the residual risk of the conversion and it is
named here so the next reader does not have to rediscover it.

This is NOT the generic wrapper layers.md rule 2 forbids
---------------------------------------------------
``docs/architecture/layers.md`` §2 refuses a "generic ``shard_map`` wrapper",
on the grounds that ``in_specs``/``out_specs`` **are** the distributed
algorithm and a wrapper taking them as arguments "has abstracted nothing and
put a layer between the reader and the only thing they came to read".  That
refusal stands, and this module is not an instance of it.

It abstracts nothing.  It introduces no named pattern, hides no spec, and
changes no call site's shape: every site still spells its own ``mesh``,
``in_specs`` and ``out_specs`` in full, at the site, exactly as before.  The
entire diff at a call site is the import path.  What it owns is a **version
decision** --- which symbol exists on this jax and what its check kwarg is
called --- which is the same thing ``common.vma`` owns for ``lax.pvary`` /
``lax.pcast``, and which has nowhere else to live: the alternative is 36 files
each deciding for themselves, which is the state this replaced, in two
already-divergent copies.

The distinction that matters: rule 2 is about abstracting a *distributed
algorithm*.  This abstracts a *jax API version*.  If a future edit starts
defaulting specs, naming patterns, or accepting an algorithm choice here, that
edit is what rule 2 forbids --- not this one.

Why the three specs are REQUIRED keyword-only here
--------------------------------------------------
Stricter than either jax spelling, deliberately.  All 94 sites already pass
``mesh``/``in_specs``/``out_specs`` by keyword and always pass all three, so
nothing is lost --- and it means this tree can never silently inherit a jax
default that moves underneath it.  That is not hypothetical: at 0.7.0
``jax.shard_map`` has ``in_specs=None``, and by the current released docs it
has become ``in_specs=jax.sharding.Infer``.  A site that omitted ``in_specs``
would change meaning between two jaxes that both "have ``jax.shard_map``".
"""
from __future__ import annotations

import jax

__all__ = ["shard_map", "shard_map_mode", "select_mode", "SHARD_MAP_MODE",
           "MODERN_SHARD_MAP_SINCE", "ShardMapSupportError"]


class ShardMapSupportError(RuntimeError):
    """This jax exposes no ``shard_map`` this tree knows how to call."""


#: First jax exposing the modern ``jax.shard_map``.  Recorded for the refusal
#: message only --- the decision is made by probing for the SYMBOL, never by
#: comparing versions, because every NVIDIA container jax is a source build
#: that restamps ``__version__`` with the date it was probed (CLAIMS 111/127).
MODERN_SHARD_MAP_SINCE = (0, 7, 0)

_VER = tuple(jax.version.__version_info__)


def select_mode(has_modern: bool, has_experimental: bool, version_info) -> str:
    """Which ``shard_map`` spelling applies, or raise.  PURE — no jax state.

    Split out from the module body, exactly as ``common.vma.select_mode`` is,
    so the whole decision table is testable by construction on any jax ---
    including the branch the running container cannot exhibit and the refusal,
    which is the branch that most needs a test precisely because nobody can
    reach it by accident.
    """
    if has_modern:                     # jax >= 0.7.0
        return "jax.shard_map"
    if has_experimental:               # jax <= 0.6.x
        return "jax.experimental.shard_map"
    raise ShardMapSupportError(
        "jax %s exposes neither jax.shard_map (added at %s) nor "
        "jax.experimental.shard_map, so this tree cannot enter manual "
        "sharding mode at all.  Refusing rather than degrading: every "
        "distributed kernel in src/ is written in shard_map, so there is no "
        "meaningful fallback.  Fix: add the new spelling to "
        "src/common/shard_map.py, or run a jax that has one."
        % (".".join(map(str, version_info)),
           ".".join(map(str, MODERN_SHARD_MAP_SINCE))))


def _probe():
    has_modern = hasattr(jax, "shard_map")
    try:
        import jax.experimental.shard_map as _exp
        has_experimental = hasattr(_exp, "shard_map")
    except ImportError:
        _exp, has_experimental = None, False
    return has_modern, has_experimental, _exp


_HAS_MODERN, _HAS_EXPERIMENTAL, _EXP = _probe()
_MODE = select_mode(_HAS_MODERN, _HAS_EXPERIMENTAL, _VER)

if _MODE == "jax.shard_map":
    SHARD_MAP_MODE = "jax.shard_map"

    def _apply(f, mesh, in_specs, out_specs, check_vma):
        return jax.shard_map(f, mesh=mesh, in_specs=in_specs,
                             out_specs=out_specs, check_vma=check_vma)

else:
    SHARD_MAP_MODE = "jax.experimental.shard_map (jax %s has no jax.shard_map)" % (
        ".".join(map(str, _VER)),)

    def _apply(f, mesh, in_specs, out_specs, check_vma):
        # check_vma is check_rep under a different name -- upstream's own 0.7.0
        # wrapper passes `check_vma=check_rep` verbatim.  `auto` is left at its
        # default because no site in this tree passes it; see the module
        # docstring's census.
        return _EXP.shard_map(f, mesh, in_specs, out_specs,
                              check_rep=check_vma)


_announced = False


def _announce() -> None:
    global _announced
    if _announced:
        return
    _announced = True
    print("  [shard_map] using %s (jax %s)"
          % (SHARD_MAP_MODE, ".".join(map(str, _VER))), flush=True)


def shard_map_mode() -> str:
    """Which spelling this process resolved to.  For tests and for logs."""
    return SHARD_MAP_MODE


def shard_map(f, *, mesh, in_specs, out_specs, check_vma=True):
    """Map ``f`` over shards of its arguments.  The tree's only spelling.

    ``check_vma=False`` switches off varying-manual-axes checking for this
    ``shard_map``, which also exempts its loop carries from
    ``common.vma.mark_varying``.  On a jax without ``jax.shard_map`` it is
    passed through as ``check_rep``, which upstream proves is the same value.

    ``mesh``/``in_specs``/``out_specs`` are required and keyword-only on
    purpose --- see the module docstring.
    """
    _announce()
    return _apply(f, mesh, in_specs, out_specs, check_vma)
