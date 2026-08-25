"""Startup refusal for a JAX/JAXLIB stack outside the 0.9 series.

This is the in-process backstop for the package constraints and the tracked
``tools/require_jax09.py`` launch preflight.  All three must agree.  It checks
both JAX and JAXLIB after backend initialization but before the first physics
``jit``, then checks the private-API shapes used by the persistent compile
cache.  There is no unsupported-version escape hatch: production and developer
driver invocations run the same 0.9 contract.

The stricter policy was restored on 2026-08-25 after a staged CUDA-12/JAX-0.7
launcher survived an earlier environment consolidation.  The current
Perlmutter lane is bare-host CUDA 13.2 with JAX/JAXLIB 0.9.1, so the former
CUDA-12 reachability argument no longer applies.  Historical measurements of
0.5/0.7 remain useful below only as negative controls for the API-shape gate.

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

Why the private-surface checks remain
-------------------------------------
The first version gate was wired in 2026-08-06 after four compatibility shims
were removed from ``common/jax_compile_cache.py``.  The old 0.5 and 0.7
measurements below are retained as negative controls: they prove that a
version-stamped but API-incompatible build is rejected before compilation.
They no longer define a supported production lane; only 0.9 does.

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
``src/common/__init__.py`` re-exports from ``.meta`` and ``.wfn_transforms``
at package scope, and both import ``jax``.  So ``from common.jax_support
import enforce`` — at ANY point in a file — pulls jax in through
``common/__init__.py``, and ``runtime`` is the one module in the tree that
must be importable *before* jax reads its environment.

(The third package-scope importer used to be ``.cholesky_2d``; the
distrib_la replumb deleted it and its re-export block, and the conclusion
above is unchanged because it never rested on that one module — ``import
common`` still imports jax, and ``tests/test_layering.py`` is what keeps
that from being a claim nobody re-checks.)  This module itself
imports only the standard library at module scope (jax is imported inside
:func:`describe` and ``importlib`` inside the two private probes), so as
``runtime.jax_support`` it can be — and now is — imported at ``runtime``'s
module scope with no jax anywhere in the chain.
"""
from __future__ import annotations

import inspect
import re
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
# The owner requires one generation everywhere.  0.9.1 is the currently
# deployed and locked release; the declared series leaves patch upgrades
# possible while refusing 0.8 and 0.10 at startup.
# --------------------------------------------------------------------------
SUPPORTED_MIN = (0, 9, 0)
SUPPORTED_MAX_EXCLUSIVE = (0, 10, 0)

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


def _parse_version_info(version: str) -> tuple[int, ...]:
    """Numeric release prefix, independent of local/dev suffixes."""
    match = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", str(version))
    if match is None:
        return ()
    return tuple(int(v) for v in match.groups(default="0"))


def describe() -> dict[str, Any]:
    """Everything needed to identify this JAX/JAXLIB pair."""
    import jax
    import jaxlib
    import jax.version as jv

    return {
        "version": jax.__version__,
        "version_info": tuple(getattr(jv, "__version_info__", ())),
        "release_version": getattr(jv, "_release_version", None),
        "is_dev_build": getattr(jv, "_release_version", None) is None,
        "file": jax.__file__,
        "jaxlib_version": jaxlib.__version__,
        "jaxlib_version_info": _parse_version_info(jaxlib.__version__),
        "jaxlib_file": jaxlib.__file__,
        "shard_map_top_level": hasattr(jax, "shard_map"),
    }


def check_version() -> list[str]:
    """Return a list of refusal-reason strings (empty when supported)."""
    info = describe()
    problems: list[str] = []
    for package, vi, version, path in (
        ("jax", tuple(info.get("version_info", ()))[:3],
         info.get("version"), info.get("file")),
        ("jaxlib", tuple(info.get("jaxlib_version_info", ()))[:3],
         info.get("jaxlib_version"), info.get("jaxlib_file")),
    ):
        if not vi:
            problems.append(
                f"{package} has no parseable version tuple "
                f"(version string {version!r})")
        elif not (SUPPORTED_MIN <= vi < SUPPORTED_MAX_EXCLUSIVE):
            problems.append(
                f"{package} {version} (version_info={vi}) from {path}")
    return problems


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

    ``announce`` is retained for call-signature stability; unsupported
    versions are never downgraded or silenced.
    """
    version_problems = check_version()
    arity_problems = check_private_arity()
    symbol_problems = check_private_symbols()
    all_problems = version_problems + arity_problems + symbol_problems

    if not all_problems:
        return

    if version_problems:
        _refuse(
            RULE_UNSUPPORTED_VERSION,
            got="; ".join(version_problems),
            want=f"jax and jaxlib >= {_fmt(SUPPORTED_MIN)}, "
                 f"< {_fmt(SUPPORTED_MAX_EXCLUSIVE)} (same window as "
                 f"pyproject.toml and tools/require_jax09.py)",
            fix="on Perlmutter select LX_BASE_MODULE=lorrax_A; elsewhere "
                "install matching jax and jaxlib 0.9.x packages",
        )
    if symbol_problems:
        _refuse(
            RULE_PRIVATE_MISSING,
            got="; ".join(symbol_problems),
            want="the jax._src symbols common/jax_compile_cache.py resolves at "
                 "call time",
            fix="use the certified JAX/JAXLIB 0.9 environment whose private "
                "surface matches this tree",
        )
    _refuse(
        RULE_PRIVATE_ARITY,
        got="; ".join(arity_problems),
        want="the jax._src arities common/jax_compile_cache.py patches against "
             "(see REQUIRED_PRIVATE_ARITY)",
        fix="use the certified JAX/JAXLIB 0.9 environment; the compile-cache "
            "patches cannot safely run against this private surface",
    )
