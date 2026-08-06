"""``jax_support`` — the STARTUP refusal for an unsupported JAX.

Why this module exists
----------------------
``pyproject.toml`` declares ``jax[cuda13]>=0.9.0``.  Nothing checked it, so on
2026-08-06 the two production legs were measured to be running *different JAX
generations*:

    Frontera CPU   jax 0.9.1                (release; ``_release_version='0.9.1'``)
    Perlmutter GPU jax 0.5.3.dev20260806    (nvcr.io/nvidia/jax:25.04-py3, built
                                             from source; ``_release_version=None``)

The GPU leg runs four declared-minor-versions BELOW the project's own floor,
and it was nobody's decision.  This module is the missing teeth, written to the
same contract as :mod:`src.ffi.gate`: *an explicit request that cannot be
honored REFUSES, naming the fix — it never silently downgrades.*

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

Where this belongs in the startup order
---------------------------------------
:func:`enforce` is designed to be called from
``runtime.initialize_communicator_stack``, in the same place and for the same
reason as ``ffi.gate.Gate.enforce``: after the backend exists, before any real
work compiles.  It costs one ``inspect.signature`` per hook.

NOT WIRED IN YET — deliberately.  Turning this on today would refuse every
Perlmutter GPU run, which is the correct behavior for an unsupported stack but
is the OWNER's call to make, not an agent's.  Wiring is one line; see
``docs`` in :data:`RULE_UNSUPPORTED_VERSION`.
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
]

# --------------------------------------------------------------------------
# The declared support window.  ONE place, and it is the thing pyproject.toml
# means.  Widening it is a decision that should show up in a diff.
# --------------------------------------------------------------------------
SUPPORTED_MIN = (0, 9, 0)
SUPPORTED_MAX_EXCLUSIVE = (0, 10, 0)

#: Escape hatch for deliberately running an unsupported stack (e.g. to
#: reproduce a container bug).  Named, not a bare truthy string, so it cannot
#: be set by accident and always leaves a line in the log.
OVERRIDE_ENV = "LORRAX_JAX_UNSUPPORTED_OK"

# --------------------------------------------------------------------------
# The private-API shapes this tree is WRITTEN AGAINST.
#
# Measured 2026-08-06 on both legs (JID 56389339 / Frontera job 7890771):
#
#   hook                              jax 0.5.3          jax 0.9.1
#   --------------------------------  -----------------  ------------------
#   _hash_accelerator_config          3 (…, backend)     2               <-- differs
#   _hash_serialized_compile_options  3                  3
#   get_executable_and_time           3                  4 (…, devices)  <-- differs
#   is_executable_in_cache            2                  2
#   compilation_cache_check_contents  ABSENT             present         <-- differs
#   VerificationCache                 ABSENT             present         <-- differs
#   backend_compile_and_load          ABSENT             present         <-- differs
#
# The three ABSENT rows are why the existing try/except around
# ``_install_compile_counter`` degrades SILENTLY on the GPU leg: the counter
# never installs, so the compile-storm telemetry the docs promise is simply
# not collected, with no announcement.
# --------------------------------------------------------------------------
REQUIRED_PRIVATE_ARITY: dict[tuple[str, str], int] = {
    ("jax._src.cache_key", "_hash_accelerator_config"): 2,
    ("jax._src.cache_key", "_hash_serialized_compile_options"): 3,
    ("jax._src.compilation_cache", "get_executable_and_time"): 4,
    ("jax._src.compilation_cache", "is_executable_in_cache"): 2,
}

#: Private symbols that must merely EXIST for the compile-cache patches to
#: resolve at call time.  Absence here is the silent class, so it refuses too.
REQUIRED_PRIVATE_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("jax._src.compilation_cache", "VerificationCache"),
    ("jax._src.config", "compilation_cache_check_contents"),
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
                 f"(the floor pyproject.toml already declares)",
            fix=f"install a supported jax in this environment, or set "
                f"{OVERRIDE_ENV}=1 to run anyway and own the consequences",
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
