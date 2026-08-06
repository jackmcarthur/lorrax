"""``mark_varying`` — the one way this tree marks a loop carry as device-varying.

What the marking is for
-----------------------
From **jax 0.7.0** onward, ``shard_map`` tracks *varying manual axes* (VMA):
every value inside the body carries the set of mesh axes over which it may
differ between devices.  ``scan`` / ``fori_loop`` / ``while_loop`` then require
the carry's VMA set to be **equal** at input and output.  An accumulator built
by ``jnp.zeros(...)`` has an EMPTY set, so the first ``A = A + <sharded data>``
makes the output varying and the loop is rejected::

    TypeError: scan body function carry input and carry output must have
    equal types, but they differ:
      ... complex128[1,2,399,9] vs complex128[1,2,399,9]{x,y}
    This might be fixed by applying `jax.lax.pvary(..., ('x', 'y'))` to the
    initial carry value ...

``mark_varying(init, axes)`` is that fix.  It emits no communication; it only
declares that the value is allowed to differ per device.

Which jax spells it how — MEASURED, not assumed
-----------------------------------------------
Four container tags imported on a Perlmutter A100 (2026-08-06, JID 56405158 /
56405696, and re-confirmed here on 0.7.0 under JID 56405696):

    jax     VMA tracked?   lax.pvary            lax.pcast
    0.5.3   no             absent               absent
    0.7.0   YES            present              absent
    0.7.2   YES            present              absent
    0.9.0   YES            present (deprecated) present

Two facts this module exists to encode:

1. **VMA tracking starts at 0.7.0**, not 0.9.  A guard that assumes "only 0.9
   needs this" is wrong over the whole 0.7-0.8 range.
2. **0.7.x has no ``lax.pcast`` at all.**  So ``try: lax.pcast / except
   AttributeError: identity`` installs a NO-OP on exactly the versions that
   enforce the rule.  That defect was live in ``common/cholesky_2d.py`` and is
   the reason this module is a shared helper rather than a per-file shim.

So: probe for the CAPABILITY, newest spelling first, and fall back to the
identity **only** on a jax that genuinely does not track VMA — where the
identity is not merely safe but the only possible behaviour.  A jax that DOES
track VMA and has neither spelling is a case nobody has seen; this module
**refuses at import** rather than silently degrading, per the same contract as
``ffi.gate`` and ``common.jax_support``: an operation that cannot be honoured
says so and names the fix.

Two rules for CALLERS, both measured
------------------------------------
* **Do not blanket-mark.**  Marking is not free of consequence: a carry that is
  invariant by construction (fed only by ``psum`` / ``pmax`` results) and that
  leaves the ``shard_map`` through a REPLICATED ``out_specs`` entry FAILS if it
  is marked --- measured on 0.7.0::

      ValueError: shard_map applied to the function '...' was given out_specs
      which require replication which can't be statically verified

  Mark the axes the error message names, which are the axes the body actually
  introduces --- not "all the mesh axes, to be safe".

* **A ``shard_map`` with the check OFF needs no marking at all.**  Also
  measured on 0.7.0, and the two spellings are NOT interchangeable::

      jax.shard_map(...)                      takes check_vma=
      jax.experimental.shard_map.shard_map()  takes check_rep=

  Passing the wrong one is a ``TypeError``, not a silently ignored kwarg.
  ``check_rep=False`` / ``check_vma=False`` disables VMA checking for that
  ``shard_map``, so its carries are exempt.

  **Since 2026-08-06 no caller has to make that choice.**  Every
  ``shard_map`` in this tree comes from ``common.shard_map``, which owns the
  symbol-and-kwarg decision the way this module owns the marking decision,
  and whose kwarg is always the modern ``check_vma``.  The equivalence above
  is upstream's own: the 0.7.0 experimental wrapper forwards
  ``check_vma=check_rep`` verbatim to the same internal ``_shard_map``.

Sites in this tree that need the mark are enumerated by
``tests/test_vma_marking.py``, which re-derives them from the AST so a new
unmarked accumulator is caught without a GPU.
"""
from __future__ import annotations

import jax
from jax import lax

__all__ = ["mark_varying", "vma_mode", "select_mode", "VMA_MODE",
           "VMA_TRACKING_SINCE", "VmaSupportError"]


class VmaSupportError(RuntimeError):
    """This jax tracks varying manual axes but offers no way to mark them."""


# The version at which shard_map began tracking varying manual axes.  Read from
# ``__version_info__``, never the display string: every NVIDIA container jax is
# a source build that restamps ``__version__`` with the date it was probed
# (CLAIMS 111), so the string is evidence of nothing.
_VMA_TRACKING_SINCE = VMA_TRACKING_SINCE = (0, 7, 0)

_VER = tuple(jax.version.__version_info__)


def select_mode(has_pcast: bool, has_pvary: bool, version_info) -> str:
    """Which marking spelling applies, or raise.  PURE — no jax state.

    Split out from the module body so the whole decision table can be tested
    by construction on any jax, including the two cases the running container
    cannot exhibit: the >=0.9 spelling and the refusal.  The compatibility
    path that nobody can see is exactly what produced the defect this module
    replaces, so it is the part that most needs a test.
    """
    if has_pcast:                                   # jax >= 0.9
        return "lax.pcast"
    if has_pvary:                                   # jax 0.7 - 0.8
        return "lax.pvary"
    if tuple(version_info) < _VMA_TRACKING_SINCE:   # jax <= 0.6
        return "identity"
    # A jax inside the tracking window with NEITHER spelling.  Degrading to the
    # identity here is precisely the bug this module was written to remove: the
    # carry would go unmarked on a jax that enforces marking, and every
    # shard_map with a zeros-initialised accumulator would fail at trace time
    # with a message about "varying manual axes" and no hint that a
    # compatibility path had chosen to do nothing.
    raise VmaSupportError(
        "jax %s tracks varying manual axes inside shard_map (tracking starts "
        "at %s) but exposes neither lax.pcast nor lax.pvary, so this tree "
        "cannot mark a loop carry as device-varying.  Refusing rather than "
        "installing a no-op: an unmarked carry fails at trace time with "
        "'the varying manual axes do not match'.  Fix: add the new spelling "
        "to src/common/vma.py (see the measured table in its docstring), or "
        "run a jax that has one."
        % (".".join(map(str, version_info)),
           ".".join(map(str, _VMA_TRACKING_SINCE))))


_MODE = select_mode(hasattr(lax, "pcast"), hasattr(lax, "pvary"), _VER)

if _MODE == "lax.pcast":
    VMA_MODE = "lax.pcast"

    def _mark(x, axes):
        return lax.pcast(x, axes, to="varying")

elif _MODE == "lax.pvary":
    VMA_MODE = "lax.pvary"

    def _mark(x, axes):
        return lax.pvary(x, axes)

else:
    VMA_MODE = "identity (jax %s does not track varying manual axes)" % (
        ".".join(map(str, _VER)),)

    def _mark(x, axes):
        return x


_announced = False


def _announce() -> None:
    global _announced
    if _announced:
        return
    _announced = True
    print("  [vma] marking loop carries via %s (jax %s)"
          % (VMA_MODE, ".".join(map(str, _VER))), flush=True)


def vma_mode() -> str:
    """Which spelling this process resolved to.  For tests and for logs."""
    return VMA_MODE


def mark_varying(x, axes):
    """Declare ``x`` may differ per device over ``axes``.

    ``axes`` is a tuple of mesh axis names, e.g. ``("x", "y")``.  Use the axes
    the jax error message names --- see the CALLERS rules in the module
    docstring; over-marking is not safe.

    Inserts no communication.  On a jax with no VMA tracking this is the
    identity, which is the only behaviour that exists there.
    """
    _announce()
    return _mark(x, axes)
