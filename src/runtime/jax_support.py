"""``jax_support`` — the STARTUP refusal for an unsupported JAX.

Why this module exists
----------------------
``pyproject.toml`` declared ``jax>=0.9.0`` and nothing checked it, so on
2026-08-06 the two production legs were measured to be running *different JAX
generations*:

    Frontera CPU   jax 0.9.1                (release; ``_release_version='0.9.1'``)
    Perlmutter GPU jax 0.5.3.dev20260806    (nvcr.io/nvidia/jax:25.04-py3, built
                                             from source; ``_release_version=None``)

The GPU leg ran four declared-minor-versions BELOW the project's own floor,
and it was nobody's decision.  This module is the missing teeth, written to the
same contract as :mod:`src.ffi.gate`: *an explicit request that cannot be
honored REFUSES, naming the fix — it never silently downgrades.*

Both halves of that skew are now closed, and in opposite directions.  The GPU
leg moved UP, to ``ghcr.io/nvidia/jax:jax-2025-07-21`` (jax 0.7.0, the last
CUDA-12 image in the family).  The declared floor moved DOWN, from 0.9.0 to
0.7.0, because 0.9.0 was unreachable on CUDA 12 by construction — see
:data:`SUPPORTED_MIN`.  The two legs are now jax 0.7.0 and jax 0.9.1, one
declared window contains both, and every ``jax._src`` shape this tree patches
was MEASURED identical on them.

Why a version-number check ALONE would be the wrong instrument
--------------------------------------------------------------
Two measured facts make the version string untrustworthy as the sole evidence:

1. **The string need not describe the API.**  NVIDIA date-stamps its
   source-built containers.  ``0.5.3.dev20260806`` is stamped the same day
   0.9.1 shipped, so a naive ``>=`` on the stamp is meaningless; the honest
   reading is ``jax.version.__version_info__ == (0, 5, 3)`` with
   ``_release_version is None``, i.e. a dev build off the 0.5.3 line.
2. **What actually breaks is arity, not the number.**
   ``common/jax_compile_cache.py`` monkeypatches four ``jax._src`` privates.
   Two of them changed shape between the two generations, so the patched
   function is called with the wrong number of arguments and the run dies on
   its FIRST ``jit`` compile (measured; see :data:`REQUIRED_PRIVATE_ARITY`).

So the gate asserts BOTH: the declared support window, *and* the specific
private-API shapes the tree is written against.  A future container that
reports a blessed version but carries a different ``jax._src`` still refuses.

Where this sits in the startup order
------------------------------------
:func:`enforce` is called from ``runtime.initialize_communicator_stack`` as
step 5b — after ``bootstrap()``, whose last act is the first ``jax.devices()``
(so the backend exists and ``jax._src`` is fully populated), and before
``prepare_mesh``, which performs the first ``jit``.  Same position and same
reason as ``ffi.gate.Gate.enforce`` one step later: refuse before anything
compiles, never in the middle of a run.  It costs one ``inspect.signature``
per hook.

WIRED IN 2026-08-06 — and why it was right to wait until then
--------------------------------------------------------------
This module spent a day deliberately unwired, for three reasons that were all
correct at the time and are all gone now.  Recording them because "wire the
gate" was recommended, then refused, then taken, and the difference each time
was a measurement, not an opinion.

1. **It was unsatisfiable on every image.**  Two required private symbols
   (``VerificationCache``, ``compilation_cache_check_contents``) exist on no
   NVIDIA container at any tag — ten probed, 0.5.3 through 0.9.1 — so wiring
   it would have refused a correct stack for a reason having nothing to do
   with that stack.  Removed 2026-08-06; see
   :data:`REQUIRED_PRIVATE_SYMBOLS`.

2. **The floor was unreachable.**  ``SUPPORTED_MIN`` was ``0.9.0`` while the
   GPU leg ran 0.5.3 and no CUDA-12 image above 0.7.0 exists, so every GPU run
   would have refused with no reachable remedy.  The leg is now on jax 0.7.0
   and the floor is 0.7.0: MEASURED satisfiable, not asserted.

3. **Every condition it tests was already handled elsewhere.**  While
   ``common/jax_compile_cache.py`` carried five compatibility shims, the
   arity/symbol clauses duplicated work the shims did quietly, so a refusal
   keyed on them would have stopped runs that worked (that was the standing
   ruling, and it was right).  **Four of those five shims are now deleted**
   with jax 0.5.3 support.  That inverts the argument: the conditions are no
   longer handled anywhere else, and unhandled they surface as a ``TypeError``
   on the first ``jit`` — or, worse, as the silent variant, where
   ``ensure_jax_compile_cache`` reported ``enabled=True`` over a cache writing
   ZERO entries.  This gate is now the only thing standing between that class
   of failure and a named startup refusal.

So the gate and the shim removal are one decision, not two: the shims were
per-call-site accommodation, this is a once-per-process assertion, and keeping
both would be paying twice for one guarantee.

MEASURED on both containers before wiring: on jax 0.7.0
:func:`check_private_arity` and :func:`check_private_symbols` each return
EMPTY and :func:`enforce` is clean; on jax 0.5.3 it refuses honestly, naming
the version below the floor, the two wrong arities, and the absent
``backend_compile_and_load``.

WHY IT LIVES IN ``runtime/`` — moved from ``common/`` 2026-08-06
----------------------------------------------------------------
This module is a **startup fact collector**, and its only caller in ``src/``
is ``runtime.initialize_communicator_stack`` (step 5b, above).  It landed in
``common/`` on ``agent/jax-070-land``, where the layer map's default put it at
**L1 — physics** — the most permissive level in the tree — and ``runtime``
(L3) then had to reach *up* into it.  That import was written
function-locally, which made the direction invisible in the file header
without making it legal: ``tests/test_layering.py`` reports lazy edges too.

The fix is the one this tree already made for the same shape, and recorded as
numbered request **R9** (``docs/architecture/layers.md`` §7, and the LAYERING
NOTE at ``runtime/__init__.py``'s allocator corroboration): the startup facts
``runtime`` needs become a **sibling module inside** ``runtime/``, so the
direction is right by construction rather than by exception.  ``runtime.
xla_memory`` moved out of ``gw/gw_config.py`` for exactly this reason.

The move also removes the *reason* the import had to be lazy, which is worth
stating because it is a property of the packages, not a style preference:
``src/common/__init__.py`` imports ``.wfn_transforms`` and ``.cholesky_2d`` at
package scope, and both import ``jax``.  So ``from common.jax_support import
enforce`` — at ANY point in a file — pulls jax in through
``common/__init__.py``, and ``runtime`` is the one module in the tree that
must be importable *before* jax reads its environment.  This module itself
imports only the standard library at module scope (jax is imported inside
:func:`describe` and ``importlib`` inside the two private probes), so as
``runtime.jax_support`` it can be — and now is — imported at ``runtime``'s
module scope with no jax anywhere in the chain.
"""
from __future__ import annotations

import inspect
import os
from typing import Any

__all__ = [
    "JaxSupportError",
    "describe",
    "enforce",
    "check_version",
    "check_private_arity",
    "check_private_symbols",
]

# --------------------------------------------------------------------------
# The declared support window.  ONE place, and it is the thing pyproject.toml
# means.  Widening it is a decision that should show up in a diff.
#
# FLOOR 0.7.0, lowered from 0.9.0 on 2026-08-06.  Not a relaxation of
# standards — the opposite.  0.9.0 was a floor NO REACHABLE PERLMUTTER IMAGE
# COULD MEET: the device FFI .so links CUDA 12, and no NVIDIA JAX image has
# both jax >= 0.9 and CUDA 12 (ten tags probed; the CUDA 12 -> 13 flip happens
# three minors before jax reaches 0.9).  A floor nothing can satisfy is not
# enforcement, it is a permanent override, which is exactly why this gate sat
# unwired.  0.7.0 is the floor the tree actually needs and can actually run:
#
#   * 0.5.3 has NO ``jax.shard_map`` and NO ``lax.pvary`` — the owner's
#     ruling, and measured in-container.
#   * varying-manual-axes tracking inside ``shard_map`` starts AT 0.7.0, and
#     ``common.vma`` marks carries from there up.
#   * both jax._src arities this tree patches reach their current shape at
#     0.7.0 (see the table below) — 0.7.0 and 0.9.1 are the same shape.
#
# CEILING 0.10.0, unchanged, and it is a statement about what has been
# measured rather than about what is broken: the two production legs are jax
# 0.7.0 (Perlmutter container) and jax 0.9.1 (Frontera venv), both inside the
# window.  Nothing at 0.10+ has been run.  One thing to check before raising
# it: ``jax.experimental.shard_map`` is imported by ~24 files, ~60 of whose
# call sites pass ``check_rep=False`` and are exempt from VMA marking BECAUSE
# of it.  If that symbol is ever removed, those sites lose the exemption and
# the import in the same release.
# --------------------------------------------------------------------------
SUPPORTED_MIN = (0, 7, 0)
SUPPORTED_MAX_EXCLUSIVE = (0, 10, 0)

#: Escape hatch for deliberately running an unsupported stack (e.g. to
#: reproduce a container bug).  Named, not a bare truthy string, so it cannot
#: be set by accident and always leaves a line in the log.
OVERRIDE_ENV = "LORRAX_JAX_UNSUPPORTED_OK"

# --------------------------------------------------------------------------
# The private-API shapes this tree is WRITTEN AGAINST.
#
# Measured 2026-08-06 on both legs (JID 56389339 / Frontera job 7890771), and
# re-measured the same day by importing jax._src inside FOUR container tags on
# a Perlmutter A100 (JID 56405158) — which corrected two rows:
#
#   hook                              0.5.3   0.7.0   0.7.2   0.9.0   0.9.1
#   --------------------------------  ------  ------  ------  ------  -----
#   _hash_accelerator_config          3       2       2       2       2
#   _hash_serialized_compile_options  3       3       3       3       3
#   get_executable_and_time           3       4       4       4       4
#   is_executable_in_cache            2       2       2       2       2
#   backend_compile_and_load          ABSENT  pres.   pres.   pres.   pres.
#   compilation_cache_check_contents  ABSENT  ABSENT  ABSENT  ABSENT  pres.†
#   VerificationCache                 ABSENT  ABSENT  ABSENT  ABSENT  pres.†
#
#   † the 0.9.1 column is the Frontera venv, measured earlier and NOT
#     re-measured here.  Every container column IS re-measured, and the
#     containers are what the GPU leg runs.
#
# The ``backend_compile_and_load`` ABSENT is why the try/except around
# ``_install_compile_counter`` degrades SILENTLY on the 0.5.3 GPU leg: the
# counter never installs, so the compile-storm telemetry the docs promise is
# simply not collected, with no announcement.
#
# The last two rows are the correction, and they are why this gate could not
# be wired: see :data:`REQUIRED_PRIVATE_SYMBOLS`.
#
# READ THE 0.7.0 AND 0.9.1 COLUMNS TOGETHER: they are identical on every row
# except the two verification symbols.  That is what made it possible to
# delete four of ``jax_compile_cache``'s five compatibility shims — and it is
# what makes this table load-bearing rather than documentary.  Each entry
# below is now the ONLY thing checking a condition that used to have an
# in-line shim:
#
#   _hash_accelerator_config    2  <- was shim 1 (0.5.3 passed a 3rd arg)
#   get_executable_and_time     4  <- was shims 2 AND 5.  The 4th parameter is
#                                     ``executable_devices``; without it a
#                                     peer CANNOT bind a cached executable to
#                                     its own devices, so a shared cache dir
#                                     at P>1 makes rank 1 name rank 0's entry,
#                                     fetch it and die loading it (measured,
#                                     2 GPUs, rc 70).  That whole degradation
#                                     branch is gone; this row is its
#                                     replacement, and it refuses at startup
#                                     instead of degrading mid-run.
#   backend_compile_and_load       <- was shim 4, in REQUIRED_PRIVATE_SYMBOLS
#
# So do not "simplify" this dict by dropping rows whose arity happens to match
# on the two stacks you can reach today.  Matching is the point; the row is
# what turns a future mismatch into a named refusal rather than a TypeError on
# the first jit.
# --------------------------------------------------------------------------
REQUIRED_PRIVATE_ARITY: dict[tuple[str, str], int] = {
    ("jax._src.cache_key", "_hash_accelerator_config"): 2,
    ("jax._src.cache_key", "_hash_serialized_compile_options"): 3,
    ("jax._src.compilation_cache", "get_executable_and_time"): 4,
    ("jax._src.compilation_cache", "is_executable_in_cache"): 2,
}

#: Private symbols that must merely EXIST for the compile-cache patches to
#: resolve at call time.  Absence here is the silent class, so it refuses too.
#:
#: This list used to also demand ``compilation_cache.VerificationCache`` and
#: ``config.compilation_cache_check_contents``.  Both are REMOVED, because
#: neither exists on any NVIDIA JAX container at any tag — MEASURED
#: 2026-08-06 on 0.5.3, 0.7.0, 0.7.2 and 0.9.0, absent on all four, including
#: the 0.9 container that the table above was previously read as promising.
#: A gate is only teeth if something can pass it: demanding a symbol no
#: reachable image provides would have made :func:`enforce` refuse EVERY run
#: the moment it was wired, on a stack that is otherwise fine.  That is worse
#: than the silence it was written to replace, because it would have been read
#: as "this JAX is unsupported" rather than "this check is unsatisfiable".
#:
#: ``common/jax_compile_cache.py`` reads both symbols with ``getattr(...,
#: None)`` and skips content verification when they are absent, which is
#: byte-for-byte what JAX's own ``get_file_cache`` does — so their absence is
#: not a defect to refuse over.  If a future JAX does ship them, add them back
#: as a measured capability, not as an inference from a version number.
REQUIRED_PRIVATE_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("jax._src.compiler", "backend_compile_and_load"),
    ("jax._src.api", "clean_up"),
    ("jax._src.distributed", "global_state"),
)

RULE_UNSUPPORTED_VERSION = "jax-support.version"
RULE_PRIVATE_ARITY = "jax-support.private-arity"
RULE_PRIVATE_MISSING = "jax-support.private-missing"

_DOC = "docs/dev/jax_support.md"


class JaxSupportError(RuntimeError):
    """Raised at startup when the running JAX is not the one we are written for."""


def _refuse(rule: str, got: str, want: str, fix: str) -> None:
    """The standard LORRAX refusal — every field mandatory, same as ffi.gate."""
    raise JaxSupportError(
        "\n"
        f"+{'-' * 78}+\n"
        f"| REFUSED: {rule}\n"
        f"+{'-' * 78}+\n"
        f"| got   {got}\n"
        f"| want  {want}\n"
        f"| fix   {fix}\n"
        f"| doc   {_DOC}\n"
        f"+{'-' * 78}+"
    )


def _fmt(v: tuple[int, ...]) -> str:
    return ".".join(str(x) for x in v)


def describe() -> dict[str, Any]:
    """Everything needed to identify this JAX, WITHOUT trusting its version string."""
    import jax
    import jax.version as jv

    return {
        "version": jax.__version__,
        "version_info": tuple(getattr(jv, "__version_info__", ())),
        "release_version": getattr(jv, "_release_version", None),
        "is_dev_build": getattr(jv, "_release_version", None) is None,
        "file": jax.__file__,
        "shard_map_top_level": hasattr(jax, "shard_map"),
    }


def check_version() -> list[str]:
    """Return a list of refusal-reason strings (empty when supported)."""
    info = describe()
    vi = info["version_info"][:3]
    if not vi:
        return [f"jax.version.__version_info__ is empty (version string "
                f"{info['version']!r}) — cannot establish a generation"]
    if not (SUPPORTED_MIN <= vi < SUPPORTED_MAX_EXCLUSIVE):
        return [f"jax {info['version']} (__version_info__={vi}, "
                f"{'dev build' if info['is_dev_build'] else 'release'}) "
                f"from {info['file']}"]
    return []


def check_private_arity() -> list[str]:
    """Verify the ``jax._src`` shapes this tree monkeypatches. Returns problems."""
    import importlib

    problems: list[str] = []
    for (modname, attr), want_n in REQUIRED_PRIVATE_ARITY.items():
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{modname} is not importable ({type(exc).__name__})")
            continue
        fn = getattr(mod, attr, None)
        if fn is None:
            problems.append(f"{modname}.{attr} is ABSENT (expected {want_n} params)")
            continue
        try:
            got_n = len(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            continue  # not introspectable; not evidence of a mismatch
        if got_n != want_n:
            problems.append(
                f"{modname}.{attr} takes {got_n} parameters, this tree patches "
                f"it with {want_n}")
    return problems


def check_private_symbols() -> list[str]:
    """Verify privates that must merely exist. Returns problems."""
    import importlib

    problems: list[str] = []
    for modname, attr in REQUIRED_PRIVATE_SYMBOLS:
        try:
            mod = importlib.import_module(modname)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{modname} is not importable ({type(exc).__name__})")
            continue
        if not hasattr(mod, attr):
            problems.append(f"{modname}.{attr} is ABSENT")
    return problems


def enforce(*, announce=None) -> None:
    """Refuse at startup if the running JAX is not the one this tree targets.

    Call once, from ``runtime.initialize_communicator_stack``, right beside
    ``ffi.gate`` enforcement.  ``announce`` is an optional rank-0 print hook.

    Honors :data:`OVERRIDE_ENV`, which downgrades every refusal to a single
    announced line — the ONE declared silence, in the sense of
    ``Gate.silent_platform_demote``.
    """
    version_problems = check_version()
    arity_problems = check_private_arity()
    symbol_problems = check_private_symbols()
    all_problems = version_problems + arity_problems + symbol_problems

    if not all_problems:
        return

    if os.environ.get(OVERRIDE_ENV) == "1":
        if announce is not None:
            announce(
                f"*** {OVERRIDE_ENV}=1: running an UNSUPPORTED JAX on purpose. "
                f"{len(all_problems)} problem(s): " + "; ".join(all_problems)
                + " ***")
        return

    if version_problems:
        _refuse(
            RULE_UNSUPPORTED_VERSION,
            got=version_problems[0],
            want=f"jax >= {_fmt(SUPPORTED_MIN)}, < {_fmt(SUPPORTED_MAX_EXCLUSIVE)} "
                 f"(same window as pyproject.toml; see SUPPORTED_MIN for why "
                 f"the floor is {_fmt(SUPPORTED_MIN)} and not higher)",
            fix=f"on Perlmutter, load a module whose image is "
                f"ghcr.io/nvidia/jax:jax-2025-07-21 (config/perlmutter/"
                f"site_config.sh); elsewhere install a jax in the window; or "
                f"set {OVERRIDE_ENV}=1 to run anyway and own the consequences",
        )
    if symbol_problems:
        _refuse(
            RULE_PRIVATE_MISSING,
            got="; ".join(symbol_problems),
            want="the jax._src symbols common/jax_compile_cache.py resolves at "
                 "call time",
            fix=f"use a jax whose private surface matches this tree, or set "
                f"{OVERRIDE_ENV}=1 and expect the compile cache to be wrong",
        )
    _refuse(
        RULE_PRIVATE_ARITY,
        got="; ".join(arity_problems),
        want="the jax._src arities common/jax_compile_cache.py patches against "
             "(see REQUIRED_PRIVATE_ARITY)",
        fix=f"use a supported jax, or set {OVERRIDE_ENV}=1 together with "
            f"ISDF_JAX_CACHE_DIR=\"\" (the compile cache is what breaks)",
    )
