"""``lxkit`` — the shared foundation every LORRAX service stands on.

Nine things that standalone services need and must not own private copies of:
the env-dial :class:`~lxkit.gate.Gate` (grammar, rank discipline,
announce-or-refuse) and the device-vendor platform key
(:func:`device_vendor`, :func:`mesh_ffi_platform`), the platform legs of the
native-provider ABI (:mod:`lxkit.legs`), the ABSENT-vs-BROKEN probe vocabulary
(:mod:`lxkit.probe`), the loop-carry marking (:func:`mark_varying`),
process-local array placement (:mod:`lxkit.placement`), placement-independent
rank-order reductions (:mod:`lxkit.collectives`: :func:`ordered_psum` and
friends), the device-memory model that prices, admits and receipts every
stage (:mod:`lxkit.memory`), native-provider selection/attestation policy
(:mod:`lxkit.native_provider`), and the per-user persistent-cache root
(:mod:`lxkit.cache`).  :mod:`lxkit.testing` ships the pytest harness
as a ``pytest11`` plugin; :mod:`lxkit.deck_doctor` (input-file preflight) and
:mod:`lxkit.launcher_policy` (the ``lx`` launcher's pure policy) are
command-line modules and are not re-exported here.

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
from lxkit import legs, memory, native_provider
from lxkit.legs import (
    LEGS,
    PlatformLeg,
    accelerator_leg_to_open_first,
    accelerator_legs,
    leg_device_present,
    leg_for_jax_platform,
    leg_for_platform,
    leg_for_vendor,
    installed_plugin_names,
    installed_plugin_leg,
    process_accelerator_leg,
    startup_gpu_leg,
)
from lxkit.cache import atomic_write_bytes, user_cache_dir, user_cache_root
from lxkit.callbacks import host_debug_callback, host_io_callback, host_pure_callback
from lxkit.placement import (device_put_process_local, device_put_process_local_cached,
                             device_put_process_tiles, placement_cache_stats)
from lxkit.collectives import (install_reduction_guard, install_reduction_guard_from_env,
                                ordered_norm, ordered_psum, ordered_psum_scatter, ordered_reduce,
                                ordered_sum, ordered_transient_bytes, reduction_trace)
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


def varying_axes(x):
    """The mesh axes ``x`` is device-varying over inside ``shard_map`` (empty
    outside one): ``jax.typeof(x).manual_axis_type.varying`` (JAX 0.10 removed
    ``ShapedArray.vma``, which held the same set).  JAX 0.9.1, still accepted
    on the ROCm leg (``runtime.jax_support.SUPPORTED_WINDOWS``), has only
    ``vma``; a type with neither raises rather than reading as empty."""
    import jax
    aval = jax.typeof(x)
    manual = getattr(aval, "manual_axis_type", None)
    return frozenset(manual.varying if manual is not None else aval.vma)

__all__ = [
    # gate
    "Gate", "MODE_SPELLINGS", "MODE_HELP", "FFI_PLATFORM_MAP",
    "rank_id", "rank0", "announce_once", "reset_gate_state",
    "device_vendor", "mesh_ffi_platform", "platform_from_env", "dial_key",
    # probe
    "ProbeResult", "AVAILABLE", "LibraryNotBuilt", "LibraryUnusable",
    "unknown_target", "not_loadable", "missing_symbol",
    # JAX utilities
    "device_put_process_local", "device_put_process_local_cached",
    "device_put_process_tiles", "placement_cache_stats", "mark_varying", "varying_axes",
    "ordered_psum", "ordered_psum_scatter", "ordered_reduce", "ordered_sum", "ordered_norm",
    "ordered_transient_bytes", "install_reduction_guard",
    "host_pure_callback", "host_io_callback", "host_debug_callback",
    "install_reduction_guard_from_env", "reduction_trace",
    # the platform legs of the native-provider ABI
    "legs", "memory", "LEGS", "PlatformLeg", "accelerator_leg_to_open_first",
    "accelerator_legs", "leg_device_present", "leg_for_jax_platform", "leg_for_platform", "leg_for_vendor",
    "process_accelerator_leg", "installed_plugin_names", "installed_plugin_leg", "startup_gpu_leg",
    # native-provider policy (the tables remain with each caller)
    "native_provider",
    # the per-user persistent-cache root
    "atomic_write_bytes", "user_cache_dir", "user_cache_root",
]
