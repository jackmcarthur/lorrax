"""``lxkit`` — the shared foundation every LORRAX service stands on.

Six things that standalone services need and must not own private copies of:
the env-dial :class:`~lxkit.gate.Gate` (grammar, rank discipline,
announce-or-refuse), the ABSENT-vs-BROKEN probe vocabulary
(:mod:`lxkit.probe`), the loop-carry marking
(:func:`mark_varying`), process-local array placement
(:mod:`lxkit.placement`), native-provider selection/attestation policy
(:mod:`lxkit.native_provider`), and the per-user persistent-cache root
(:mod:`lxkit.cache`).  :mod:`lxkit.testing` ships the pytest harness
as a ``pytest11`` plugin.

PINNED PROPERTY — STDLIB-ONLY AT IMPORT
---------------------------------------
``import lxkit`` pulls in the standard library and NOTHING else.  jax is an
OPTIONAL, LAZY dependency: every jax import in this package sits inside a
function body, never at module scope.  ``lxkit`` therefore declares **zero**
runtime dependencies and installs into a bare interpreter.

This is a property, not a coincidence, and it is enforced by a subprocess
test (``tests/test_import_isolation.py``).  It exists because
:meth:`Gate.enabled` is a KERNEL-CACHE KEY read at factory time, before
``jax.distributed.initialize``: anything in this package that touched the
jax backend at import would initialize it too early and destroy that
promise (see the "Two tiers" section of :mod:`lxkit.gate`).  The same
property is what lets the shape-algebra test tier run on a laptop with no
jax at all.

lxkit carries POLICY, never TABLES.  The FFI symbol tables, library search
paths and backend preference rows belong to the service that owns them;
what lives here is the decision procedure they all share.
"""

from __future__ import annotations

from lxkit.gate import (
    FFI_PLATFORM_MAP,
    Gate,
    MODE_HELP,
    MODE_SPELLINGS,
    announce_once,
    device_vendor,
    dial_key,
    mesh_ffi_platform,
    platform_from_env,
    rank0,
    rank_id,
    reset_gate_state,
)
from lxkit import native_provider
from lxkit.cache import user_cache_dir
from lxkit.placement import device_put_process_local, device_put_process_tiles
from lxkit.probe import (
    AVAILABLE,
    LibraryNotBuilt,
    LibraryUnusable,
    ProbeResult,
    missing_symbol,
    not_loadable,
    unknown_target,
)


def mark_varying(x, axes):
    """Declare ``x`` device-varying over the mesh ``axes`` inside ``shard_map``
    (a loop carry built from zeros); no communication.  Mark only the axes the
    body introduces: a marked carry that leaves through a replicated
    ``out_specs`` is an error."""
    from jax import lax
    return lax.pcast(x, axes, to="varying")

__all__ = [
    # gate
    "Gate", "MODE_SPELLINGS", "MODE_HELP", "FFI_PLATFORM_MAP",
    "rank_id", "rank0", "announce_once", "reset_gate_state",
    "device_vendor", "mesh_ffi_platform", "platform_from_env", "dial_key",
    # probe
    "ProbeResult", "AVAILABLE", "LibraryNotBuilt", "LibraryUnusable",
    "unknown_target", "not_loadable", "missing_symbol",
    # JAX utilities
    "device_put_process_local", "device_put_process_tiles", "mark_varying",
    # native-provider policy (the tables remain with each caller)
    "native_provider",
    # the per-user persistent-cache root
    "user_cache_dir",
]
