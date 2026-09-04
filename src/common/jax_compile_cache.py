"""JAX persistent compile cache — SAFE and EFFECTIVE at ``process_count() > 1``.

DRIVERS MUST NOT CALL :func:`ensure_jax_compile_cache`.  Arming it is
owned by ``runtime.initialize_communicator_stack`` (step 7), which every
driver already runs at module scope — above its own ``import jax`` and
therefore above every jit in the process.  That ordering is the whole
requirement: arming late means the expensive early compiles miss the
cache.  A driver-local second call cannot advance that moment, so it buys
nothing and its error handling is unreachable; a driver that needs the
cache earlier needs the startup call earlier, not a second arming.

Two callers are deliberately NOT drivers and are not covered by the
above: kernel factories that arm the cache for callers arriving without
a driver (``gw.w_isdf``, ``gw.ppm_tau_kernel``), and the drivers that run
bare ``runtime.bootstrap()`` rather than the full startup call
(``psp.run_nscf``, ``psp.run_sternheimer``, ``bse.bse_feast``), where
nothing else arms it.

Knobs (all optional):

  ``ISDF_JAX_CACHE_DIR=/some/path``    — override cache location.
  ``ISDF_JAX_CACHE_DIR=""``             — opt out entirely.
  ``LORRAX_RUN_DIR=/some/run``         — when the cache knob is unset, share
                                          ``.lorrax_jax_cache`` only among
                                          processes/drivers in this workflow.
  ``LORRAX_JAX_CACHE_MULTIPROCESS=0``   — restore the scorecard-AG refusal
                                          (no cache at all when P > 1).
  ``LORRAX_JAX_CACHE_AGREE_TIMEOUT_S``  — agreement timeout, default 300.
  ``LORRAX_JAX_COMPILE_AGREEMENT=0``    — UNSAFE bisect-only opt-out from the
                                          per-module cross-rank compile-key
                                          refusal (default on at P > 1).
  ``LORRAX_JAX_COMPILE_AGREE_TIMEOUT_S`` — bounded per-module agreement
                                          timeout, default 60 seconds.
  ``LORRAX_JAX_CACHE_STRICT=0``         — on an agreed entry that then fails
                                          to load, warn instead of aborting
                                          (UNSAFE on GPU: can hang).
  ``LORRAX_JAX_CACHE_FORCE_DIVERGE=N``  — TEST HOOK (positive control): every
                                          rank != 0 pretends its N
                                          alphabetically-last cache entries
                                          are missing, forcing the agreement
                                          to drop them.
  ``LORRAX_JAX_CACHE_NO_AGREE=1``       — TEST HOOK: shared dir with the
                                          agreement layer DISABLED, i.e. the
                                          naive shared-dir design.  This is
                                          the deadlock reproducer; never use
                                          it in production.
  ``JAX_EXPLAIN_CACHE_MISSES=1``        — opt in to JAX cache-miss explanations.
                                          This is intentionally independent of
                                          ``LORRAX_DEBUG_PRINT``: explanation
                                          construction is diagnostic work, not
                                          ordinary stage logging.
  ``LORRAX_JAX_CACHE_KEYDUMP=<dir>``    — every rank writes the SET of
                                          persistent-cache keys it asked
                                          about to ``<dir>/rank{i}_of{N}.json``
                                          at exit.  This is what makes the
                                          key-symmetry invariant falsifiable:
                                          JAX cache-miss logging names
                                          only the keys that MISSED, so on a
                                          healthy warm run it prints nothing
                                          and two ranks asking about
                                          different programs look identical.
  ``LORRAX_JAX_CACHE_SHARD_SLICE=0``    — TEST HOOK (red twin): leave JAX's
                                          ``ArrayImpl._multi_slice`` alone, so
                                          each rank bakes its own shard
                                          offsets into the jit signature and
                                          gets its own ``jit__multi_slice``
                                          cache key.  That is the divergent
                                          hit/miss pattern; never in production.
  ``LORRAX_JAX_CACHE_INVARIANT_KEY=0``  — TEST HOOK: do NOT make the cache key
                                          process-invariant.  At P > 1 this
                                          also switches the cache OFF, because
                                          hitting it would be unsafe (only
                                          process 0 could).
  ``LORRAX_JAX_CACHE_PREFETCH``         — pull the agreed entries into the
                                          page cache from a thread pool right
                                          after the agreement (default in
                                          ``_PREFETCH_DEFAULT``), with
                                          ``LORRAX_JAX_CACHE_PREFETCH_THREADS``
                                          workers (16).
  ``JAX_COMPILATION_CACHE_MAX_SIZE``    — JAX's byte cap. ``0`` disables the
                                          cache; a positive cap is supported
                                          only at P=1. Live LRU eviction is
                                          refused at P>1 because it can
                                          invalidate the agreed startup set.
  ``JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS`` — standard JAX write
                                          threshold (default 1 second).  LORRAX
                                          does not lower it: cheap compiles are
                                          cheaper than a Lustre entry.

Default policy (``ISDF_JAX_CACHE_DIR`` unset) prefers the workflow: when
``LORRAX_RUN_DIR`` is set, entries live in
``$LORRAX_RUN_DIR/.lorrax_jax_cache/np{N_proc}/``.  Launchers without an
explicit run directory retain the legacy ``$SCRATCH/lorrax_jax_cache`` (or
``$XDG_CACHE_HOME``/``~/.cache``) fallback until the required P=4 default-flip
A/B is available.  The workflow path avoids a cross-material cache that grows
by tens of thousands of small files for the usual one-shot calculation while
allowing kmeans/dipole/kin-ion/GW processes in one named workflow to reuse
compatible entries.  An explicit ``ISDF_JAX_CACHE_DIR`` still overrides or
opts out.  JAX's in-process executable cache is active in every case.

Within an enabled base, ``np{N_proc}`` is ONE directory shared by every rank
of that world size (the old ``rank{i}/`` partitioning is gone; see below).

The ``ISDF_*`` env-var naming is legacy — historically this was for ISDF
kernels only, now it caches the whole run.  Left as-is for backward
compat with existing user shell aliases and run scripts.

**Expect a wall of scary-looking XLA:CPU log lines on every warm run, and
ignore them.**  ``cpu_aot_loader.cc`` compares the feature list the entry was
compiled with against the host's and shouts

    E cpu_aot_loader.cc:220] Loading XLA:CPU AOT result. Target machine
    feature +prefer-no-gather is not supported on the host machine ...
    This could lead to execution errors such as SIGILL.

``prefer-no-gather`` / ``prefer-no-scatter`` are LLVM *cost-model pseudo-
features*: they exist at compile time and never appear in a runtime CPU
feature list, so this comparison mismatches on every load on every machine.
It is a ``LOG(ERROR)``, not a rejection — MEASURED on the same run that emits
738 of these lines: 369/373 cache hits and 4 compiles; and on htransform, 304
lines with 152/152 hits and ZERO compiles.  ADVICE section 4's "each forcing a
recompile" is not true of jax 0.9.1.

===========================================================================
WHY THIS FILE IS COMPLICATED: XLA:GPU COMPILATION IS A COLLECTIVE
===========================================================================
Scorecard AG root-caused the ``load_centroid_wfns`` hang that blocked every
multi-process GPU run on Frontera's ``rtx`` queue, from a live C-level stack:

    xla::gpu::AutotunerPass::RunImpl
      xla::Autotuner::Autotune(HloModule*, ..., MultiProcessKeyValueStore&)
        xla::DistributedKeyValueStore::Get
          xla::CoordinationServiceAgent::GetKeyValue    <-- blocks forever

``AutotunerPass`` shards autotuning across processes and exchanges the
results through the JAX coordination service.  **A process that skips
compilation never publishes its share, so every peer blocks forever.**
Therefore the hit/miss pattern of the persistent cache must be IDENTICAL
on every rank, for every module, or the job hangs.

The old layout made that impossible on purpose: entries were nested under
``{base}/np{P}/rank{i}/`` while JAX writes cache entries from process 0
only (``jax/_src/compiler.py::_cache_write``, unconditional::

      # Only write cache entries from the first process. Otherwise we
      # create problems with contention for writes on some filesystems
      if distributed.global_state.process_id != 0:
        return

), so ``np4/rank0`` accumulated 882 entries while ``np4/rank{1,2,3}``
stayed empty forever.  AG's fix was to REFUSE the cache at P > 1.  This
file replaces that refusal with a working implementation.

===========================================================================
THE DESIGN (workstream AH): SNAPSHOT AGREEMENT OVER A SHARED DIRECTORY
===========================================================================
1. **One shared directory per world size**, ``{base}/np{P}/``, plus a patch
   that makes the cache KEY process-invariant.

   AG.7 is right that a bare shared directory buys nothing: MEASURED at 4 CPU
   ranks, the ranks compute DIFFERENT keys for the same SPMD module, so
   process 0 — the only writer — is the only rank that ever hits, and its
   peers recompile everything.  jax/_src/cache_key.py strips the device
   assignment from the hashed compile options only for GPU

       strip_device_assignment=(backend.platform == "gpu")
       # In case of GPU multi-process tasks we need to strip device
       # assignment to use cache key as invariant between processes.

   and hashes the accelerator config as a serialized topology blob that also
   carries process-local content.  :func:`_install_invariant_key_patch` does
   for every platform what JAX already does for GPU (see its docstring).
   That is what turns the cache from *safe* into *effective*.

2. **Hit/miss agreement, taken once per run as a snapshot.**  Right here in
   :func:`ensure_jax_compile_cache`, after ``jax.distributed.initialize``:

   * process 0 lists the shared dir and publishes the sorted list of cache
     keys it can see, through the coordination-service KV store;
   * every rank fetches that list, checks which of those entries it can
     itself see, and publishes a presence BITMASK (one bit per key — 882
     entries is 111 bytes, so this scales to any world size);
   * process 0 ANDs the masks and publishes the result; every rank fetches
     it and a barrier commits the decision.

   The intersection is the set of entries the run is allowed to use.  Every
   subsequent cache probe is answered from that frozen set, so **hit/miss
   is identical on every rank by construction** — including for entries
   process 0 writes *during* this run, which are deliberately invisible
   until the next run.  That closes the within-run race that makes a naive
   shared directory unsafe: JAX's ``LRUCache.put`` is a plain
   ``cache_path.write_bytes(val)`` (NOT tmp+rename — verified in
   jax/_src/lru_cache.py 0.9.1), so a peer reading an entry the writer is
   still writing gets a truncated file, its read raises, ``_cache_read``
   swallows it into a MISS, and the ranks diverge.

3. **Writes stay process-0-only** (JAX's own rule).  Under SPMD process 0
   compiles the same module set as everyone else, so its writes cover the
   whole set; all-rank writes would only add Lustre contention.  We do
   however make the write ATOMIC (temp file + ``os.replace``) so a
   concurrently-running job can never observe a torn entry.

4. **JAX's auto-enabled XLA sub-caches are turned OFF at P > 1, and whenever
   the LORRAX persistent cache is explicitly off.**  When the
   persistent cache is on, ``jax/_src/compiler.py::get_compile_options``
   also points XLA at ``{cache_dir}/xla_gpu_per_fusion_autotune_cache_dir``
   with ``AutotuneCacheMode.UPDATE`` on process 0 and ``READ`` on the peers.
   That is a *second*, rank-asymmetric cache that changes the set of fusions
   each process still has to autotune — which is exactly the input to
   AutotunerPass' modulo-P work split.  Different sets on different ranks =
   the same deadlock one level down.  We set
   ``jax_persistent_cache_enable_xla_caches=""``.

5. **A key-environment fingerprint** rides along with each rank's bitmask
   (:func:`_key_env_fingerprint`).  The agreement decides which cache ENTRIES
   may be used; it cannot see that two ranks would compute different KEYS for
   the same module because they were launched with different ``XLA_FLAGS``.
   Any mismatch turns the cache off on every rank, loudly.

6. **Graceful degradation, never a hang.**  If the coordination client is
   missing, or any KV/barrier step fails or times out, or the key cannot be
   made process-invariant, the agreed set is EMPTY: every rank misses
   everything and compiles, which is the pre-AH behaviour and is always
   correct.  The reason is printed on every rank.  If an entry that *was*
   agreed then fails to load (corrupt file, deserialisation error) the ranks
   would diverge, so that path aborts the process loudly
   (``LORRAX_JAX_CACHE_STRICT=0`` downgrades it to a warning).

The load-bearing-ness of (2) was A/B'd on GPU with a node-local cache
directory, which JAX's process-0-only write leaves populated on one node and
empty on the others: without the agreement the warm run HANGS (rc=124 at
600 s); with it the same run completes in 48 s after dropping all 287
unshared entries.  With a genuinely shared directory and (1) in place there is
nothing left to diverge, so the agreement is insurance — but it is the
difference between a loud message and a dead job.

Considered and rejected: ``jax_share_binary_between_hosts`` — it is only
reached on a cache MISS (``compile_or_get_cached`` returns early on a hit),
so when process 0 hits and its peers miss, the peers wait forever on a
publication that never happens: the same deadlock.  Worse on CPU, where it
keys the broadcast on the per-rank cache key, so the peers would block on a
key process 0 never sets even on a symmetric cold miss.  (It also has exactly
one process compile, which reads as incompatible with a sharded autotuner —
but that is inference from the sources, not something measured here.)  It is
orthogonal to, not a substitute for, the agreement step.

===========================================================================
WHICH JAX THIS FILE IS WRITTEN AGAINST
===========================================================================
The four patches above reach into ``jax._src``.  Both production legs now run
a generation with the same shapes: Frontera's venv has the released jax 0.9.1,
Perlmutter's GPU container (``ghcr.io/nvidia/jax:jax-2025-07-21``) has jax
0.7.0.  Every hook this file patches was MEASURED identical on the two — see
the table in the ``jax._src surface this file patches`` block below.

Until 2026-08-06 the GPU container was ``nvcr.io/nvidia/jax:25.04-py3`` (jax
0.5.3), which differed on four of them, and this file carried five named
compatibility shims for that.  Four are DELETED with the 0.5.3 support: the
detection duty they served now belongs to ``runtime.jax_support``, which
asserts those arities and symbols once at startup and refuses by name.  The
FIFTH survives and is not a version shim at all — ``VerificationCache`` and
``compilation_cache_check_contents`` are absent from every NVIDIA container at
every tag, including the 0.9 ones, and present only in the released wheel.

Do not read a container's ``jax.__version__``: NVIDIA re-stamps it with the
build date, so ten images all print ``.dev<today>`` regardless of which line
they were cut from.  ``jax.version.__version_info__`` is the honest tuple.
"""
from __future__ import annotations

import atexit
import hashlib
import json
import os
import sys
import threading
import time
import uuid
import warnings
from pathlib import Path

_COMPILATION_CACHE_READY = False

# jax/_src/lru_cache.py
_CACHE_SUFFIX = "-cache"

# Coordination-service key namespace.  The KV store lives and dies with one
# `jax.distributed` session (one srun step), so a fixed namespace is safe.
_KV_NS = "lorrax/compile_cache/v1"

# Per-module compile fingerprints use a separate protocol.  Unlike the
# startup cache snapshot above, this one executes before EVERY backend
# compile and refuses a rank-divergent module before XLA can enter collective
# GPU autotuning and wait forever.
_COMPILE_KV_NS = "lorrax/compile_agreement/v2"
_COMPILE_AGREEMENT_TIMEOUT_DEFAULT_S = 60.0

# Parallel page-cache prefetch of the agreed entries (see _prefetch_agreed).
# ON: at 606 centroids / P=16 the SERIAL reads of 169 entries cost 29 s on one
# rank and 8.8 s on another, against the ~4.5 s of XLA compile they replace —
# i.e. without this the cache is a net LOSS on a cold-read CPU run.  876 kB of
# payload, so it is pure per-file Lustre latency under 16-way concurrency.
_PREFETCH_DEFAULT = "1"


class _CacheState:
    """Per-process bookkeeping for the agreed cache (also the atexit report)."""

    def __init__(self) -> None:
        self._write_lock = threading.Lock()
        self._compile_event_lock = threading.Lock()
        self.enabled = False
        self.dir = ""
        self.n_proc = 1
        self.proc_idx = 0
        self.n_seen = 0        # entries process 0 advertised
        self.n_agreed = 0      # entries every rank could see
        self.probes = 0        # persistent-cache lookups JAX asked for
        self.hits = 0          # lookups served from disk
        self.blocked = 0       # lookups vetoed by the agreement
        self.compiles = 0      # actual XLA compiles (backend_compile_and_load)
        self.compile_secs = 0.0
        self.read_secs = 0.0   # time spent loading executables from disk
        self.prefetch_secs = 0.0
        self.compile_agreement_configured = False
        self.compile_agreement_enabled = False
        self.compile_agreement_reason = "not installed"
        self.compile_agreement_timeout_s = 0.0
        self.compile_agreement_checks = 0
        self.compile_fingerprint_secs = 0.0
        self.compile_agreement_secs = 0.0
        self._compile_client = None
        self._compile_sequence = 0
        # JAX calls the file-cache writer on process 0 only.  Keep these
        # explicitly process-local: summing them across ranks would turn one
        # physical write into a fictitious P writes.  The atomic, unlimited
        # path below is the only path whose successful writes and exact
        # payload bytes we can observe without changing JAX's LRU policy.
        self.write_metrics_available = False
        self.reset_write_metrics()
        self.agreed: frozenset[str] = frozenset()
        # THE KEY SET.  Every persistent-cache key this rank asked about,
        # hit or miss.  The counters above cannot express the invariant the
        # cache contract is actually about: two ranks can both report
        # ``xla_compiles=0 vetoed=0`` while asking about DIFFERENT programs,
        # which is the state that precedes the collective-compile deadlock.
        # A set of keys is the only observable that separates those.
        self.probe_keys: set[str] = set()

    def reset_write_metrics(self) -> None:
        """Reset this process's successful-write receipt deterministically."""
        with self._write_lock:
            self.local_writes = 0
            self.local_write_bytes = 0
            self.local_write_secs = 0.0

    def set_write_metrics_available(self, available: bool) -> None:
        """Record whether this cache policy has an observable write boundary."""
        with self._write_lock:
            self.write_metrics_available = bool(available)

    def record_write(self, nbytes: int, elapsed_s: float) -> None:
        """Record one completed local write; callers hold no state lock."""
        with self._write_lock:
            # The actual instrumented boundary is stronger evidence than an
            # earlier setup-time guess (JAX may have constructed its cache
            # lazily before LORRAX initialization).
            self.write_metrics_available = True
            self.local_writes += 1
            self.local_write_bytes += int(nbytes)
            self.local_write_secs += max(0.0, float(elapsed_s))

    def write_metrics(self) -> dict:
        """Snapshot process-local write metrics without torn counter reads."""
        with self._write_lock:
            available = self.write_metrics_available
            return {
                "write_metrics_available": available,
                "local_writes": self.local_writes if available else None,
                "local_write_bytes": (
                    self.local_write_bytes if available else None),
                "local_write_secs": (
                    self.local_write_secs if available else None),
            }


_STATE = _CacheState()


class _KeyEnvMismatch(RuntimeError):
    """The ranks would compute different cache keys — see _key_env_fingerprint."""


class UnsafeCachePolicy(RuntimeError):
    """A requested disk-cache lifecycle would violate the cache contract."""


class CompileAgreementError(RuntimeError):
    """The ranks did not present the same module to the compile boundary."""


def _positive_float_env(name: str, default: float) -> float:
    """Read a positive finite duration, refusing an unbounded spelling."""
    raw = os.environ.get(name)
    try:
        value = default if raw is None or not raw.strip() else float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a positive number of seconds, got {raw!r}") \
            from exc
    if not (value > 0.0 and value < float("inf")):
        raise ValueError(
            f"{name} must be a finite positive number of seconds, "
            f"got {raw!r}")
    return value


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "", "0", "false", "no", "off")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip())
    except (TypeError, ValueError):
        return default


def _cache_size_policy(n_proc: int, max_size: int) -> bool:
    """Whether the persistent cache may run under JAX's size policy.

    JAX's ``0`` spelling is an explicit cache-off request.  A positive limit
    enables live LRU eviction: safe at P=1, but unsafe after the P>1 agreement
    freezes the entries every rank is allowed to read.  Rank 0 evicting one
    of those files before a peer's first lookup invalidates that snapshot and
    creates the hit/miss divergence this module exists to prevent.

    Returns ``False`` only for the standard cache-off spelling.  Every unsafe
    or invalid spelling refuses before a cache object is armed.
    """
    n_proc = int(n_proc)
    max_size = int(max_size)
    if max_size < -1:
        raise UnsafeCachePolicy(
            "JAX_COMPILATION_CACHE_MAX_SIZE must be -1 (unlimited), 0 "
            f"(off), or a positive byte count; got {max_size}.")
    if max_size == 0:
        return False
    if n_proc > 1 and max_size > 0:
        raise UnsafeCachePolicy(
            "JAX_COMPILATION_CACHE_MAX_SIZE enables live LRU eviction, which "
            f"is unsafe at P={n_proc}: LORRAX freezes an all-rank readable-"
            "entry set at startup, and rank-0 eviction can remove an agreed "
            "entry before a peer reads it. Use ISDF_JAX_CACHE_DIR=\"\" for "
            "a one-shot run, or an explicit run-local cache directory that "
            "the outer launcher removes after every rank has exited.")
    return True


def _say(msg: str) -> None:
    print(f"  [compile-cache] {msg}", flush=True)


def _debug_say(msg: str) -> None:
    """Healthy cache telemetry follows the driver's one debug switch."""
    try:
        from runtime import debug_print_enabled
        enabled = debug_print_enabled()
    except Exception:                                      # noqa: BLE001
        enabled = False
    if enabled:
        _say(msg)


# ---------------------------------------------------------------------------
# jax._src surface this file patches
# ---------------------------------------------------------------------------
# This file monkeypatches five ``jax._src`` privates.  The fifth,
# ``ArrayImpl._multi_slice`` (see :func:`_install_shard_slice_patch`), is not
# about the cache LAYER at all — it is about the ranks compiling the same
# module — but it lives here because it defends the same invariant as
# everything else in this file and would be invisible anywhere else.
#
# It used to carry FIVE
# named compatibility shims so that one source ran on both the jax 0.5.3 line
# (Perlmutter's old ``nvcr.io/nvidia/jax:25.04-py3``) and the jax 0.9 line.
# **Four of the five are gone**: the GPU leg moved to
# ``ghcr.io/nvidia/jax:jax-2025-07-21`` (jax 0.7.0) and the owner ruled against
# keeping a permanent compatibility layer for a version being abandoned.
#
# Re-measured in-container on a Perlmutter A100, both images, one srun step
# each, reading ``jax.version.__version_info__`` and ``inspect.signature`` —
# capability-probed, never inferred from a version string (every container jax
# is a dev build that restamps ``__version__`` to the run date; both of these
# printed ``.dev20260806``):
#
#   hook                                       0.5.3    0.7.0    0.9.1†
#   -----------------------------------------  -------  -------  -------
#   cache_key._hash_accelerator_config          3        2        2
#   cache_key._hash_serialized_compile_options  3        3        3
#   compilation_cache.get_executable_and_time   3        4        4
#   compilation_cache.is_executable_in_cache    2        2        2
#   compiler.backend_compile_and_load           ABSENT   present  present
#   compilation_cache.VerificationCache         ABSENT   ABSENT   present
#   config.compilation_cache_check_contents     ABSENT   ABSENT   present
#   lru_cache.LRUCache.put                      write_bytes, no rename, all three
#
#   † the 0.9.1 column is the released wheel in the Frontera venv, measured
#     2026-08-06 and NOT re-measured here.  Both CONTAINER columns are
#     re-measured, and a container is what the GPU leg runs.
#
# Read the 0.7.0 column against the 0.9.1 one: on every row except the two
# verification symbols they are the SAME SHAPE.  So shims 1 (accelerator-config
# arity), 2 (lookup arity), 4 (compile entry point) and 5 (the P>1 degradation
# for a jax that cannot rebind a cached executable to the reading process's
# devices) had nothing left to bridge and were deleted.  Their detection duty
# did not vanish with them: ``runtime.jax_support`` asserts exactly these
# arities and symbols at startup and REFUSES by name, which is a better
# instrument than an in-line branch — it fires once, before anything compiles,
# instead of shaping every call site forever.
#
# **ONE shim survives, and it is NOT a 0.5.3 shim.**  ``VerificationCache`` and
# ``compilation_cache_check_contents`` are ABSENT on the 0.7.0 container just
# as they were on 0.5.3 — and, per CLAIMS 112, on all TEN NVIDIA JAX images
# probed at any tag, INCLUDING the 0.9.0/0.9.1 ones.  They exist only in the
# released wheel.  So this is a container-vs-released-wheel difference, not a
# generation one, and removing its guard would restore the CLAIMS 114 defect
# verbatim on the new image: every cache read raising ``AttributeError`` inside
# JAX's own swallowing read path, zero entries written, ``enabled=True``
# reported.  It stays, renamed for what it actually is.
#
# What we deliberately do NOT do is wrap the installers in a blanket
# ``try/except``: that is what hid this for months.  Installing a patch is an
# attribute ASSIGNMENT and never raises, so an ``except`` around installation
# catches nothing — the error fires later, when JAX CALLS the hook, long after
# that scope has exited.
#
# The surviving shim announces.  A compatibility path nobody can see in the log
# is indistinguishable from the bug it replaced.
_COMPAT_SAID: set[str] = set()


def _compat(key: str, msg: str) -> None:
    """Announce once, on rank 0, that a patch took its compatibility path.

    Rank 0 speaks alone because the private-API generation is a property of
    the image, not of the process: every rank of one launch imports the same
    ``jax._src``.  Ranks that somehow did NOT are caught by a different
    instrument — :func:`_key_env_fingerprint` folds ``jaxlib``'s version
    string into the digest the agreement compares, and a mismatch turns the
    cache off on every rank, loudly.
    """
    if key in _COMPAT_SAID:
        return
    _COMPAT_SAID.add(key)
    if _STATE.proc_idx == 0:
        _say(f"jax-compat: {msg}")


class _JaxSurfaceUnsupported(RuntimeError):
    """No shim covers this ``jax._src`` — named so a caller can report it."""


def _jax_generation() -> str:
    """``x.y.z`` from ``__version_info__``, NEVER the display string.

    Every NVIDIA container JAX is a source build that re-stamps
    ``__version__`` with the date it was PROBED: all ten images on the shelf
    print ``.dev20260806`` today, including the ones actually built from the
    0.5.3 line.  ``jax.version.__version_info__`` is the tuple that survives
    that, so it is what these announcements quote.
    """
    try:
        import jax.version as _jv

        vi = tuple(getattr(_jv, "__version_info__", ()))[:3]
        kind = "release" if getattr(_jv, "_release_version", None) else "dev build"
        return f"{'.'.join(str(x) for x in vi)} {kind}" if vi else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# local view of the shared cache directory
# ---------------------------------------------------------------------------
def _local_entry_keys(cache_path: Path) -> list[str]:
    """Cache keys this rank can see on disk, sorted.

    An entry counts only if the file is non-empty; our atomic ``put`` makes
    torn files impossible within a job, and this also screens out anything a
    pre-AH (non-atomic) writer may have left behind.
    """
    out: list[str] = []
    try:
        names = os.listdir(cache_path)
    except OSError:
        return out
    for name in names:
        if not name.endswith(_CACHE_SUFFIX) or name.startswith("."):
            continue
        try:
            if os.path.getsize(os.path.join(cache_path, name)) <= 0:
                continue
        except OSError:
            continue
        out.append(name[: -len(_CACHE_SUFFIX)])
    out.sort()
    return out


def _mask_bytes(n: int) -> bytearray:
    return bytearray((n + 7) // 8)


def _bit(mask, i: int) -> bool:
    return bool(mask[i >> 3] & (1 << (i & 7)))


def _set_bit(mask: bytearray, i: int) -> None:
    mask[i >> 3] |= 1 << (i & 7)


# ---------------------------------------------------------------------------
# the agreement
# ---------------------------------------------------------------------------
def _forced_divergence_hidden(keys: list[str], proc_idx: int) -> set[str]:
    """TEST HOOK — positive control for the agreement layer.

    ``LORRAX_JAX_CACHE_FORCE_DIVERGE=N`` makes every rank != 0 pretend the N
    alphabetically-last advertised entries are missing.  A correct agreement
    layer must drop them (all ranks then compile those modules) and say so;
    a broken one lets process 0 hit them alone and the job hangs.
    """
    n = _int_env("LORRAX_JAX_CACHE_FORCE_DIVERGE", 0)
    if n <= 0 or proc_idx == 0 or not keys:
        return set()
    return set(keys[-min(n, len(keys)):])


_HOST_TARGET_ID: str | None = None


def _host_target_id() -> str:
    """Rank-invariant identity of the machine the code is compiled FOR.

    :func:`_install_invariant_key_patch` replaces JAX's serialized-topology
    hash (which carries process-local content) with a canonical string; this
    is the part of the topology that actually matters for whether a cached
    executable is *valid* here.  XLA:CPU bakes host CPU features into the AOT
    result — loading an entry built for another CPU model produces the
    "Target machine feature ... is not supported on the host machine"
    reload-reject storm this cluster hit before (ADVICE section 4).  Folding
    the CPU model into the key keeps entries from different machine types
    apart; folding it into the env fingerprint additionally turns the cache
    off (loudly) if one job somehow spans two machine types.
    """
    global _HOST_TARGET_ID
    if _HOST_TARGET_ID is not None:
        return _HOST_TARGET_ID
    import platform as _platform

    bits = [_platform.machine()]
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    bits.append(line.split(":", 1)[1].strip())
                    break
    except OSError:
        pass
    _HOST_TARGET_ID = "|".join(bits)
    return _HOST_TARGET_ID


#: PER-PROCESS ENVIRONMENT THAT CHANGES THE EMITTED MODULE, by name.
#:
#: Everything else in :func:`_key_env_fingerprint` is about the cache KEY of
#: the same module.  These are different and worse: they change the module
#: ITSELF.  ``LORRAX_FFT_FFI_FUSED`` picks between a fused host-FFI
#: ``ffi_call`` and a native three-FFT ``jnp`` chain inside
#: ``gw.ppm_tau_kernel`` / ``gw.cohsex_sigma`` / ``gw.w_isdf``, so a rank
#: launched with a different value emits different HLO, compiles a different
#: program, and misses where its peers hit — ``jit__multi_slice``'s
#: divergence (FIX_multislice_cachekey.md §6.1, sibling 5) arriving through
#: the environment rather than through a shard offset.
#:
#: These are also the ones the GEMM/FFT autotuner actually sees: the affected
#: kernels are Σ_kij, Σ_τ, the cohsex Σ chain and χ⁰, i.e. the modules with
#: the largest autotune candidate sets in the tree.  A divergent autotune set
#: is the deadlock one level below the cache (module docstring, above).
#:
#: THE SOURCE OF TRUTH IS ``src/ffi/__init__.py::FFI_DIAL_ENV`` and this list
#: mirrors it; ``tests/cache_key_lint.py``'s ``env-dial`` rule fails when the
#: two disagree.  Mirrored rather than imported because this function runs
#: during the agreement, on every rank, on machines with no FFI library
#: present — an ImportError here would turn the cache off for a reason that
#: has nothing to do with the cache.
RANK_FINGERPRINT_ENV = (
    "LORRAX_FFT_FFI",
    "LORRAX_FFT_FFI_FUSED",
    "LORRAX_BANDS_GEMM_FFI",
    "LORRAX_CONV_KMINOR_FFI",
    "LORRAX_CONV_KLEAD_FFI",
    "LORRAX_CONV_KPAIR_FFI",
)


def _key_env_fingerprint() -> bytes:
    """32-byte digest of everything OUTSIDE the module that feeds the cache key.

    The agreement below decides which cache ENTRIES may be used; it cannot see
    that two ranks would compute different KEYS for the same module.  The one
    realistic way that happens once the key is process-invariant is a rank that
    was launched with different key-affecting environment — the classic case
    being a harness that sets ``XLA_FLAGS`` on rank 0 only for an HLO dump
    (workstream Y's probe does exactly that).  Process 0 would then MISS while
    its peers HIT: the AG deadlock with the roles reversed.

    So every rank fingerprints that environment, the fingerprints are compared
    during the agreement, and any mismatch turns the cache off for everyone
    with a printed reason.  Flags JAX itself excludes from the cache key
    (``--xla_dump_*`` and friends) are excluded here too.
    """
    import hashlib

    h = hashlib.sha256()
    try:
        from jax._src import cache_key as _ck
        excluded = set(_ck.xla_flags_to_exclude_from_cache_key)
        prefixes = tuple(_ck.get_flag_prefixes())
    except Exception:
        excluded, prefixes = set(), ()
    flags = []
    for tok in os.environ.get("XLA_FLAGS", "").split():
        name = tok.split("=", 1)[0]
        if name in excluded:
            continue
        flags.append(tok)
    h.update(("|".join(sorted(flags))).encode("utf-8"))
    h.update(("|".join(sorted(prefixes))).encode("utf-8"))
    h.update(_host_target_id().encode("utf-8"))
    # The per-process dials that change the emitted MODULE, not just its key.
    # Unset and empty are folded to the same token deliberately: the gates
    # (``ffi/gate.py::Gate.mode``) treat "" as "take the default", so two
    # ranks that differ only in whether the variable exists are NOT
    # divergent and must not be reported as such.
    for name in RANK_FINGERPRINT_ENV:
        val = os.environ.get(name, "").strip().lower()
        h.update(f"{name}={val};".encode("utf-8"))
    try:
        from jax._src.lib import version_str as _jaxlib_version_str
        h.update(_jaxlib_version_str.encode("utf-8"))
    except Exception:
        pass
    try:
        import jax as _jax
        for knob in ("jax_persistent_cache_min_compile_time_secs",
                     "jax_persistent_cache_enable_xla_caches",
                     "jax_compilation_cache_include_metadata_in_key",
                     "jax_enable_x64"):
            h.update(f"{knob}={getattr(_jax.config, knob, None)!r};"
                     .encode("utf-8"))
    except Exception:
        pass
    return h.digest()


def _agree_on_entries(cache_path: Path, n_proc: int, proc_idx: int,
                      timeout_s: float, client=None
                      ) -> tuple[int, frozenset[str]]:
    """Return ``(n_advertised, agreed_keys)``; raises on any failure.

    Protocol (coordination-service KV store, O(1) RPCs per rank):
      p0   set   {ns}/keylist  = sorted keys p0 can see
      all  get   {ns}/keylist
      p!=0 set   {ns}/mask/{p} = presence bitmask over that list
      p0   get   {ns}/mask/{1..P-1}, AND them into its own, set {ns}/final
      p!=0 get   {ns}/final
      all  barrier {ns}/commit          <- collective commit point

    ``client`` defaults to the live ``jax.distributed`` coordination client;
    the tests inject a fake one.
    """
    if client is None:
        from jax._src import distributed as _dist
        client = _dist.global_state.client
    if client is None:
        raise RuntimeError("jax.distributed coordination client is None "
                           "(was jax.distributed.initialize() called?)")
    tmo = int(timeout_s * 1000)

    local = _local_entry_keys(cache_path)

    if proc_idx == 0:
        payload = b"K" + "\n".join(local).encode("utf-8")
        client.key_value_set_bytes(f"{_KV_NS}/keylist", payload)
    else:
        payload = client.blocking_key_value_get_bytes(
            f"{_KV_NS}/keylist", tmo)
    keys = [k for k in payload[1:].decode("utf-8").split("\n") if k]

    hidden = _forced_divergence_hidden(keys, proc_idx)
    local_set = set(local)
    mask = _mask_bytes(len(keys))
    for i, k in enumerate(keys):
        if k in local_set and k not in hidden:
            _set_bit(mask, i)

    fp = _key_env_fingerprint()
    if proc_idx != 0:
        client.key_value_set_bytes(f"{_KV_NS}/mask/{proc_idx}",
                                   b"M" + fp + bytes(mask))
        payload = client.blocking_key_value_get_bytes(f"{_KV_NS}/final", tmo)
        status, final = payload[1:2], payload[2:]
    else:
        acc = bytearray(mask)
        status = b"\x01"
        for p in range(1, n_proc):
            other = client.blocking_key_value_get_bytes(
                f"{_KV_NS}/mask/{p}", tmo)[1:]
            other_fp, other_mask = other[:32], other[32:]
            if other_fp != fp:
                status = b"\x00"
            if len(other_mask) != len(acc):
                raise RuntimeError(
                    f"rank {p} returned a {len(other_mask)}-byte mask, "
                    f"expected {len(acc)} — cache-key list disagreement")
            for i in range(len(acc)):
                acc[i] &= other_mask[i]
        final = bytes(acc)
        client.key_value_set_bytes(f"{_KV_NS}/final", b"F" + status + final)

    # Collective commit: if ANY rank failed to get this far the barrier fails
    # on ALL of them, so nobody is left believing the cache is usable.
    client.wait_at_barrier(f"{_KV_NS}/commit", timeout_in_ms=tmo)

    if status != b"\x01":
        # Symmetric on every rank (the verdict was broadcast), so all of them
        # degrade to cache-off together — which is always correct.
        raise _KeyEnvMismatch(
            "the ranks were launched with DIFFERENT key-affecting environment "
            "(XLA_FLAGS / jaxlib version / jax cache config / the per-process "
            f"FFI dials {list(RANK_FINGERPRINT_ENV)}), so they would compute "
            "different cache keys — and in the FFI-dial case a different "
            "emitted MODULE — for the same computation, and their hit/miss "
            "patterns would diverge: the scorecard-AG deadlock. Make the "
            "environment identical on every rank (a rank-0-only XLA_FLAGS for "
            "an HLO dump is the usual cause; a per-node FFI dial is the other)")

    agreed = frozenset(k for i, k in enumerate(keys) if _bit(final, i))
    return len(keys), agreed


def _prefetch_agreed(cache_path: Path, agreed, n_threads: int) -> float:
    """Pull the agreed entries into the page cache, in parallel.

    MEASURED (wk_AH job 3, CPU P=8 fixture): a warm run's 368 cache reads cost
    2-3 s on the node that wrote them and **12 s** on the second node, against
    the 9.6 s of XLA compile they replace — i.e. on Lustre the serial
    open()+read() latency of a few hundred tiny files can eat the whole win.
    The files total ~1.5 MB, so this is pure per-file latency, and issuing the
    reads from a thread pool hides it.  Reads are discarded; the point is the
    client page cache, which JAX's own read then hits.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not agreed:
        return 0.0
    t0 = time.monotonic()

    def _rd(key: str) -> None:
        try:
            with open(os.path.join(cache_path, key + _CACHE_SUFFIX), "rb") as fh:
                while fh.read(1 << 20):
                    pass
        except OSError:
            pass

    try:
        with ThreadPoolExecutor(max_workers=max(1, n_threads)) as pool:
            list(pool.map(_rd, agreed))
    except Exception:
        pass
    return time.monotonic() - t0


# ---------------------------------------------------------------------------
# the monkeypatches
# ---------------------------------------------------------------------------
def _fatal(cache_key: str, why: str) -> None:
    msg = (f"  [compile-cache] FATAL: entry '{cache_key[:48]}...' was agreed "
           f"readable by every rank but this rank ({_STATE.proc_idx}) cannot "
           f"load it ({why}).  Continuing would make the ranks' hit/miss "
           f"patterns diverge, which deadlocks XLA:GPU's cross-process "
           f"autotune exchange (scorecard AG).  Aborting instead of hanging. "
           f"Delete {_STATE.dir} and re-run; set LORRAX_JAX_CACHE_STRICT=0 to "
           f"downgrade this to a warning (UNSAFE on GPU).")
    if _truthy("LORRAX_JAX_CACHE_STRICT", "1"):
        print(msg, file=sys.stderr, flush=True)
        print(msg, flush=True)
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(70)
    warnings.warn(msg)


def _install_invariant_key_patch() -> None:
    """Make the persistent-cache key IDENTICAL on every rank.

    MEASURED (wk_AH keyprobe, 4 CPU ranks): out of the box the ranks compute
    DIFFERENT keys for the same SPMD module, so with a shared directory only
    process 0 — the only writer — ever hits.  jax/_src/cache_key.py strips the
    device assignment from the hashed compile options only when
    ``backend.platform == "gpu"``::

        # In case of GPU multi-process tasks we need to strip device
        # assignment to use cache key as invariant between processes.
        strip_device_assignment=(backend.platform == "gpu")

    and hashes the accelerator config as
    ``get_topology_for_devices(devices).serialize()``, which carries
    process-local content.  We do for every platform what JAX already does for
    GPU: force the strip, and replace the topology blob with a canonical
    ``platform:count:device_kinds:host_target`` string (see
    :func:`_host_target_id` for why the host target belongs in there).
    Everything else in the key — the module IR, the jaxlib version, the backend
    version, XLA flags, the remaining compile options — is identical across
    ranks of one SPMD program by construction, so the key becomes
    process-invariant.

    This is what turns the cache from merely *safe* into *effective*: with it,
    ranks 1..P-1 hit process 0's entries instead of recompiling every module
    (scorecard D's storm).  It is also load-bearing for SAFETY:
    ``LORRAX_JAX_CACHE_INVARIANT_KEY=0`` therefore switches the cache OFF at
    P > 1 rather than leaving process 0 hitting alone.

    Not applied at P == 1, where there is nothing to make invariant.
    """
    from jax._src import cache_key as _ck

    if getattr(_ck, "_lorrax_invariant_key_installed", False):
        return
    _orig_opts = _ck._hash_serialized_compile_options

    def _stripped(hash_obj, compile_options_obj, strip_device_assignment=False):
        return _orig_opts(hash_obj, compile_options_obj,
                          strip_device_assignment=True)

    # Two parameters, matching ``_hash_accelerator_config`` on every jax this
    # tree supports (0.7.0 container and 0.9.1 wheel, both MEASURED).  It used
    # to end in ``*_compat_tail`` to swallow the third positional jax 0.5.3
    # passed (``backend``, never read); that shim is gone with 0.5.3, and
    # ``runtime.jax_support`` asserts the arity at startup instead, so a jax
    # that reintroduces a third argument is a named refusal rather than a
    # silently discarded one.
    def _canonical_accelerator(hash_obj, accelerators):
        devs = list(accelerators.flat)
        plat = getattr(devs[0], "platform", "?") if devs else "?"
        kinds = sorted(str(getattr(d, "device_kind", "?")) for d in devs)
        _ck._hash_string(
            hash_obj,
            f"lorrax-canon:{plat}:{len(devs)}:{','.join(kinds)}:"
            f"{_host_target_id()}")

    _ck._hash_serialized_compile_options = _stripped
    _ck._hash_accelerator_config = _canonical_accelerator
    _ck._lorrax_invariant_key_installed = True


# ---------------------------------------------------------------------------
# the shard-slice patch: one ``jit__multi_slice`` program for every rank
# ---------------------------------------------------------------------------
# Everything above makes the ranks compute the same KEY for the same MODULE.
# This one is the other half of the same safety property: it makes them
# compile the same MODULE in the first place, for the one JAX-internal jit
# whose program is built out of per-rank shard offsets.
_CANON_SLICE_JIT = None


def _canon_slice_jit():
    """The rank-invariant slicer: shard SIZES static, shard OFFSETS dynamic.

    Built once, lazily, because importing jax at this module's import time is
    something the rest of the file is careful not to do.
    """
    global _CANON_SLICE_JIT
    if _CANON_SLICE_JIT is not None:
        return _CANON_SLICE_JIT

    import jax

    # NAMED, not `_body`: the jit's name becomes the XLA module name and the
    # cache-key prefix, and `jit__multi_slice` being distinctive is the only
    # reason this defect was ever findable in an explain log.  A module called
    # `jit__body` would hide the next one.
    def _lorrax_canonical_shard_slice(self, sizes, removed_dims, starts):
        out = []
        for sz, rm, st in zip(sizes, removed_dims, starts):
            sliced = jax.lax.dynamic_slice(self, st, sz)
            if rm:
                sliced = jax.lax.squeeze(sliced, rm)
            out.append(sliced)
        return out

    _CANON_SLICE_JIT = jax.jit(_lorrax_canonical_shard_slice,
                               static_argnums=(1, 2))
    return _CANON_SLICE_JIT


def _slice_index_dtype(shape):
    """Index dtype for the dynamic offsets — chosen from the GLOBAL shape.

    It has to be picked from something every rank agrees on.  ``shape`` is the
    global array's shape and is identical on every rank; the OFFSETS are not,
    so sizing the dtype to ``max(local offsets)`` would put the rank back in
    the signature through the back door.
    """
    import numpy as np

    return np.int64 if (shape and max(shape) >= 2 ** 31) else np.int32


def _install_shard_slice_patch() -> None:
    """Make JAX's device-array resharding compile ONE program on every rank.

    THE DEFECT.  When a **single-device, fully addressable** ``jax.Array`` is
    handed to a consumer with a multi-device sharding — a jit with
    ``in_shardings``, or a bare ``jax.device_put(arr, NamedSharding(...))`` —
    ``jax/_src/array.py::_array_shard_arg`` takes its resharding path::

        indices = sharding.addressable_devices_indices_map(x.shape).values()
        if dispatch.is_single_device_sharding(x.sharding):
          results.append(shard_device_array(x, devices, indices, sharding))

    and ``shard_device_array`` turns those indices into slice bounds and
    passes them to ``ArrayImpl._multi_slice``, which is::

        @api.jit(static_argnums=(1,2,3))
        def _multi_slice(self, start_indices, limit_indices, removed_dims):

    ``addressable_devices_indices_map`` is ADDRESSABLE — on the production
    one-GPU-per-process launch it is this rank's single shard.  So the shard
    OFFSETS are baked into the jit signature as static arguments, rank r
    compiles ``slice(x, [r*n/P, 0], [(r+1)*n/P, m])``, and the four ranks of a
    P=4 run compile four DIFFERENT modules and therefore hold four different
    persistent-cache keys.  Writes are process-0-only
    (``jax/_src/compiler.py::_cache_write``), so on a warm run rank 0 HITS its
    own key while ranks 1..P-1 MISS and compile.

    That divergent hit/miss pattern is the scorecard-AG deadlock condition
    this whole file exists to prevent, arriving by a route the agreement layer
    structurally cannot see: the agreement makes hit/miss identical for a
    GIVEN key, and here the ranks are asking about genuinely different
    programs.  MEASURED warm at P=4 (``FIX_warmcache.md`` §1.2): rank 0
    ``xla_compiles=1 hits=36`` against ranks 1,2,3 ``xla_compiles=2 hits=35``,
    with three distinct ``jit__multi_slice-*`` keys in the explain log.

    THE CANONICALIZATION.  Rebind ``ArrayImpl._multi_slice`` to a form that
    keeps the shard SIZES static — they are what the output shapes are made
    of, and they are equal on every rank whenever the mesh axis divides the
    array evenly — and passes the shard OFFSETS as ordinary dynamic operands
    to ``lax.dynamic_slice``.  One program, one key, every rank.

    Bit-identity: ``lax.dynamic_slice`` with in-bounds start indices and unit
    strides is exactly ``lax.slice`` — both are a copy, neither does
    arithmetic on the values — and the offsets are in bounds by construction
    (they came from a sharding's own index map).  The ``lax.squeeze`` of
    ``removed_dims`` is unchanged.

    RESIDUAL, stated rather than hidden: what is left in the signature is the
    tuple of shard SHAPES.  On a ragged sharding (a mesh axis that does not
    divide the array evenly) those still differ across ranks and the keys
    still diverge — strictly no worse than before this patch, but not fixed
    by it.  No single program can serve differently-shaped outputs; that case
    needs padding at the call site, not a patch here.

    Not applied at P == 1, where there is nothing to make invariant — which is
    also why this cannot perturb a single-process run.
    """
    from jax._src.array import ArrayImpl

    if getattr(ArrayImpl, "_lorrax_shard_slice_installed", False):
        return

    orig = ArrayImpl.__dict__.get("_multi_slice")
    if orig is None:
        _compat("shard-slice-absent",
                f"jax {_jax_generation()} has no ArrayImpl._multi_slice; the "
                f"rank-dependent shard-slice key cannot arise here and the "
                f"patch is not installed.")
        return

    def _canonical_multi_slice(self, start_indices, limit_indices,
                               removed_dims):
        import numpy as np

        try:
            sizes = tuple(
                tuple(int(hi) - int(lo) for lo, hi in zip(st, li))
                for st, li in zip(start_indices, limit_indices))
            removed = tuple(tuple(int(d) for d in rm) for rm in removed_dims)
            idx_dtype = _slice_index_dtype(tuple(self.shape))
            starts = tuple(
                tuple(np.array(int(v), idx_dtype) for v in st)
                for st in start_indices)
            return _canon_slice_jit()(self, sizes, removed, starts)
        except Exception as exc:                                # noqa: BLE001
            # Correctness first: any shape we did not anticipate goes back to
            # JAX's own slicer, which is right and merely rank-dependent.  It
            # ANNOUNCES, because a compatibility path nobody can see in the
            # log is indistinguishable from the bug it replaced.
            _compat("shard-slice-fallback",
                    f"the canonical shard slicer declined "
                    f"({type(exc).__name__}: {exc}); falling back to JAX's "
                    f"rank-dependent ArrayImpl._multi_slice.  Expect one "
                    f"jit__multi_slice cache key PER RANK.")
            return orig(self, start_indices, limit_indices, removed_dims)

    ArrayImpl._multi_slice = _canonical_multi_slice
    ArrayImpl._lorrax_shard_slice_orig = orig
    ArrayImpl._lorrax_shard_slice_installed = True


def _install_lookup_patch(*, enforce_agreement: bool) -> None:
    """Instrument cache lookups, optionally enforcing the all-rank set.

    ``P == 1`` needs the counters but not the policy: JAX's answer is returned
    unchanged, including an ordinary cache miss.  ``P > 1`` additionally
    vetoes keys outside :attr:`_CacheState.agreed` and treats disappearance of
    an agreed entry as fatal.  Keeping both modes in this one wrapper prevents
    the observation-only path from drifting away from the lookup surface whose
    multi-process twin it is measuring.
    """
    from jax._src import compilation_cache as _cc

    marker = ("_lorrax_agreement_installed" if enforce_agreement
              else "_lorrax_observer_installed")
    other = ("_lorrax_observer_installed" if enforce_agreement
             else "_lorrax_agreement_installed")
    if getattr(_cc, marker, False):
        return
    if getattr(_cc, other, False):
        raise RuntimeError(
            "the JAX persistent-cache lookup hook is already installed in "
            f"{'observation' if enforce_agreement else 'agreement'} mode; "
            "jax.process_count() cannot change within one process")
    _orig_get = _cc.get_executable_and_time
    _orig_in_cache = _cc.is_executable_in_cache

    # ``get_executable_and_time`` is ``(cache_key, compile_options, backend,
    # executable_devices)`` on the supported 0.9.1 wheel (and was the same on
    # the historical 0.7.0 container — MEASURED).  The arity PROBE and its
    # announcement, which existed
    # to report jax 0.5.3's 3-parameter form, are gone; ``runtime.jax_support``
    # asserts the 4 at startup.
    #
    # ``*passthrough`` is NOT a leftover compatibility branch and is kept
    # deliberately: this wrapper reads the cache key and nothing else, so
    # naming three arguments it never interprets would be a claim about their
    # meaning that this file has no reason to make.  It forwards them
    # untouched, which is exact for any arity.
    def _observed_get(cache_key, *passthrough):
        _STATE.probes += 1
        _STATE.probe_keys.add(cache_key)
        if enforce_agreement and cache_key not in _STATE.agreed:
            _STATE.blocked += 1
            return None, None
        t0 = time.monotonic()
        try:
            executable, compile_time = _orig_get(cache_key, *passthrough)
        except BaseException as exc:  # noqa: BLE001 - deliberate
            if enforce_agreement:
                _fatal(cache_key, f"{type(exc).__name__}: {exc}")
                return None, None
            # Observation at P=1 is transparent: JAX still owns the error
            # policy, so an instrument must not turn a failing read into a
            # cache miss (or vice versa).
            raise
        finally:
            _STATE.read_secs += time.monotonic() - t0
        if executable is None:
            if enforce_agreement:
                _fatal(cache_key, "entry disappeared between agreement and read")
                return None, None
            return executable, compile_time
        _STATE.hits += 1
        return executable, compile_time

    def _observed_in_cache(backend, cache_key):
        _STATE.probe_keys.add(cache_key)
        if enforce_agreement and cache_key not in _STATE.agreed:
            return False
        return _orig_in_cache(backend, cache_key)

    _cc.get_executable_and_time = _observed_get
    _cc.is_executable_in_cache = _observed_in_cache
    setattr(_cc, marker, True)


def _install_observation_patch() -> None:
    """Count P=1 cache probes/hits without changing JAX's decision."""
    _install_lookup_patch(enforce_agreement=False)


def _install_agreement_patch() -> None:
    """Answer every P>1 persistent-cache lookup from the agreed set."""
    _install_lookup_patch(enforce_agreement=True)


def _install_atomic_put_patch() -> None:
    """Make cache writes atomic (tmp + rename).

    jax 0.9.1's ``LRUCache.put`` does ``cache_path.write_bytes(val)`` with no
    rename, so a reader in another process can observe a truncated entry.
    Only the eviction-disabled path (``jax_compilation_cache_max_size=-1``,
    the default) is replaced; if eviction is on, JAX takes a file lock and we
    defer to it.

    ``LRUCache.put`` writes with ``write_bytes`` and no rename, and
    ``eviction_enabled``/``path`` are instance attributes, on 0.5.3, 0.7.0 and
    0.9.1 alike (MEASURED), so the subclass below needs no shim.  What is NOT
    uniform is the content-verification wrapper — see THE SURVIVING SHIM in
    the body.
    """
    from jax._src import compilation_cache as _cc
    from jax._src import config as _cfg
    from jax._src import lru_cache as _lru

    if getattr(_cc, "_lorrax_atomic_put_installed", False):
        return

    class _AtomicLRUCache(_lru.LRUCache):
        def put(self, key: str, val: bytes) -> None:
            if self.eviction_enabled:
                # Stock LRUCache.put returns None for both a write and a
                # pre-existing entry, and may evict under its file lock.  Do
                # not manufacture write/eviction numbers for that opaque P=1
                # policy path.
                return super().put(key, val)
            if not key:
                raise ValueError("key cannot be empty")
            final = self.path / f"{key}{_lru._CACHE_SUFFIX}"
            if final.exists():
                return
            tmp = self.path / (f".{key}{_lru._CACHE_SUFFIX}.tmp."
                               f"{os.getpid()}.{uuid.uuid4().hex[:8]}")
            t0 = time.monotonic()
            try:
                tmp.write_bytes(val)
                os.replace(str(tmp), str(final))
            except BaseException:
                try:
                    os.unlink(str(tmp))
                except OSError:
                    pass
                raise
            # ``val`` is the serialized/compressed payload JAX hands to the
            # file cache.  Count only after the atomic rename succeeds: a
            # failed write remains an exception and never becomes a receipt.
            _STATE.record_write(len(val), time.monotonic() - t0)

    # THE SURVIVING SHIM — and the reason it survives is NOT a jax version.
    #
    # ``VerificationCache`` and the ``compilation_cache_check_contents`` flag
    # that selects it are absent from every NVIDIA JAX CONTAINER at every tag
    # probed — ten of them, 0.5.3 through 0.9.1, CLAIMS 112 — and are present
    # only in the released wheel (the Frontera venv's 0.9.1).  Re-confirmed
    # here on the new GPU image: on jax 0.7.0 both are still ABSENT.  So this
    # is a container-vs-wheel difference, and moving the GPU leg off 0.5.3
    # does nothing to it.
    #
    # Deleting this guard with the other four would have restored the CLAIMS
    # 114 defect verbatim on the new image: both names were read at CALL time
    # by the body below, so on the GPU leg EVERY cache read raised
    # ``AttributeError`` inside JAX's own swallowing read path — zero entries
    # written, ``enabled=True`` reported, one ``UserWarning`` per jit the only
    # trace.
    #
    # Resolve them ONCE, here, where absence is a fact we can name.  Skipping
    # verification on a jax that has no such feature is not a downgrade: it is
    # exactly what that jax's own ``get_file_cache`` does (MEASURED — its
    # source is ``return LRUCache(path, max_size=max_size), path``, with no
    # branch).  The atomic write, which is the whole point of this patch, is
    # unaffected either way.
    _verification_cache = getattr(_cc, "VerificationCache", None)
    _check_contents = getattr(_cfg, "compilation_cache_check_contents", None)
    if _verification_cache is None or _check_contents is None:
        _missing = ", ".join(
            n for n, o in (("compilation_cache.VerificationCache",
                            _verification_cache),
                           ("config.compilation_cache_check_contents",
                            _check_contents)) if o is None)
        _compat(
            "compilation_cache.verification",
            f"jax._src has no {_missing} on this jax ({_jax_generation()}); "
            f"cache-content verification is a jax 0.9 feature.  Writes stay "
            f"atomic; reads are unverified, which is this jax's own "
            f"behaviour.  Set jax_compilation_cache_check_contents on a jax "
            f">= 0.9 to get verification back.")

    def _atomic_get_file_cache(path: str):
        cache = _AtomicLRUCache(
            path, max_size=_cfg.compilation_cache_max_size.value)
        _STATE.set_write_metrics_available(not cache.eviction_enabled)
        if (_verification_cache is not None and _check_contents is not None
                and _check_contents.value):
            return _verification_cache(cache), path
        return cache, path

    _cc.get_file_cache = _atomic_get_file_cache
    _cc._lorrax_atomic_put_installed = True


#: The ``jax._src.compiler`` entry point a real XLA compile goes through.
#: The supported 0.9 series routes compiles through
#: ``backend_compile_and_load`` (MEASURED on the 0.9.1 wheel; also present on
#: the historical 0.7.0 container), and it
#: itself calls ``backend_compile`` — so exactly ONE of them may be patched or
#: every compile is counted twice.
#:
#: This used to be a two-element preference tuple with a fallback to
#: ``backend_compile``, because jax 0.5.3 had no ``backend_compile_and_load``
#: at all.  The fallback went with 0.5.3.
_COMPILE_ENTRY_POINT = "backend_compile_and_load"


def _compile_module_identity(module) -> tuple[str, str, float]:
    """Return ``(module_name, stable_mlir_sha256, fingerprint_seconds)``.

    This hashes the binary MLIR handed to the backend, before XLA compilation
    starts.  Debug locations are excluded so source-path metadata cannot make
    otherwise identical rank programs disagree.  The digest is the cold-path
    equivalent of a persistent-cache key: it remains available when the disk
    cache is disabled, and it needs neither a backend compile nor a device
    assignment to compute.
    """
    t0 = time.monotonic()
    operation = getattr(module, "operation", None)
    name = "<unnamed-module>"
    if operation is not None:
        try:
            attr = operation.attributes["sym_name"]
            name = str(getattr(attr, "value", attr)).strip('"')
        except Exception:                                  # noqa: BLE001
            pass
    try:
        mlir = operation.get_asm(binary=True, enable_debug_info=False)
    except Exception as exc:                               # noqa: BLE001
        raise CompileAgreementError(
            "GATE cross_rank_compile_agreement: REFUSED before compiling "
            f"module {name!r}: its stable MLIR fingerprint could not be "
            f"computed ({type(exc).__name__}: {exc}).") from exc
    if isinstance(mlir, str):
        mlir = mlir.encode("utf-8")
    digest = hashlib.sha256(bytes(mlir)).hexdigest()
    return name, digest, time.monotonic() - t0


def _compile_event_prefix(occurrence: int) -> str:
    """Name one all-rank compile slot by global order, not module name.

    Keying by each module name's local occurrence permits two concurrently
    lowered modules to be approved in opposite orders on different ranks.
    Their later collective executions can then deadlock even though each
    individual fingerprint agreed.  A single global slot turns that ordering
    difference into the same bounded, rank-by-rank refusal as a shape change.
    """
    return f"{_COMPILE_KV_NS}/{int(occurrence)}"


def _decode_compile_record(payload: bytes, rank: int) -> dict:
    try:
        record = json.loads(payload.decode("utf-8"))
    except Exception as exc:                               # noqa: BLE001
        raise CompileAgreementError(
            f"rank {rank} published a malformed compile-agreement record "
            f"({type(exc).__name__}: {exc})") from exc
    if not isinstance(record, dict) or "key" not in record:
        raise CompileAgreementError(
            f"rank {rank} published an incomplete compile-agreement record: "
            f"{record!r}")
    return record


def _snapshot_compile_records(client, prefix: str, n_proc: int,
                              local_rank: int, local_record: dict) -> list:
    """Best-effort all-rank snapshot for a bounded-time refusal message."""
    records: list[dict | None] = [None] * n_proc
    records[local_rank] = local_record
    for rank in range(n_proc):
        if records[rank] is not None:
            continue
        try:
            payload = client.blocking_key_value_get_bytes(
                f"{prefix}/rank/{rank}", 1)
            records[rank] = _decode_compile_record(payload, rank)
        except Exception:                                  # noqa: BLE001
            pass
    return records


def _format_compile_refusal(verdict: dict) -> str:
    module_name = verdict.get("module", "<unknown-module>")
    occurrence = verdict.get("occurrence", "?")
    reason = verdict.get("reason", "compile-key disagreement")
    records = verdict.get("records") or []
    rank_lines = []
    for rank, record in enumerate(records):
        if record is None:
            rank_lines.append(f"rank {rank}: <not-arrived>")
        else:
            rank_lines.append(
                f"rank {rank}: key={record.get('key', '<missing>')} "
                f"module={record.get('module', '<missing>')!r}")
    return (
        "GATE cross_rank_compile_agreement: REFUSED before XLA execution.\n"
        f"  got: {reason}; stalled module {module_name!r}, occurrence "
        f"{occurrence}.\n"
        f"  rank keys: {'; '.join(rank_lines)}.\n"
        "  want: every rank to present the same stable MLIR/HLO key before "
        "any rank enters backend compilation.\n"
        "  why: a rank-divergent GPU compile can enter collective autotuning "
        "on only part of the world and hang silently.\n"
        "  fix: remove rank-conditional shapes/jits or make the emitted "
        "module identical; LORRAX_JAX_COMPILE_AGREEMENT=0 is an UNSAFE "
        "bisect-only opt-out.")


def _agree_before_module_compile(module_name: str, key: str, occurrence: int,
                                 *, client=None, n_proc: int | None = None,
                                 proc_idx: int | None = None,
                                 timeout_s: float | None = None) -> None:
    """Exchange one compile fingerprint and refuse divergence or absence."""
    s = _STATE
    client = s._compile_client if client is None else client
    n_proc = int(s.n_proc if n_proc is None else n_proc)
    proc_idx = int(s.proc_idx if proc_idx is None else proc_idx)
    timeout_s = float(
        s.compile_agreement_timeout_s if timeout_s is None else timeout_s)
    prefix = _compile_event_prefix(occurrence)
    record = {
        "rank": proc_idx,
        "module": module_name,
        "occurrence": occurrence,
        "key": key,
    }
    client.key_value_set_bytes(
        f"{prefix}/rank/{proc_idx}",
        json.dumps(record, sort_keys=True).encode("utf-8"))

    t0 = time.monotonic()
    if proc_idx == 0:
        deadline = t0 + timeout_s
        records: list[dict | None] = [None] * n_proc
        records[0] = record
        reason = ""
        for rank in range(1, n_proc):
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            if time.monotonic() >= deadline:
                reason = f"deadline expired after {timeout_s:g} seconds"
                break
            try:
                payload = client.blocking_key_value_get_bytes(
                    f"{prefix}/rank/{rank}", remaining_ms)
                records[rank] = _decode_compile_record(payload, rank)
            except Exception as exc:                        # noqa: BLE001
                reason = (
                    f"rank {rank} did not arrive within {timeout_s:g} "
                    f"seconds ({type(exc).__name__})")
                break
        if any(item is None for item in records):
            records = _snapshot_compile_records(
                client, prefix, n_proc, proc_idx, record)
            if not reason:
                reason = f"deadline expired after {timeout_s:g} seconds"
        keys = {item["key"] for item in records if item is not None}
        modules = {item.get("module") for item in records if item is not None}
        passed = len(records) == n_proc and None not in records \
            and len(keys) == 1 and modules == {module_name}
        if not passed and not reason:
            reason = "ranks published different stable MLIR/HLO keys"
        verdict = {
            "passed": passed,
            "module": module_name,
            "occurrence": occurrence,
            "reason": reason,
            "records": records,
        }
        client.key_value_set_bytes(
            f"{prefix}/verdict",
            json.dumps(verdict, sort_keys=True).encode("utf-8"))
    else:
        # A peer may reach this slot nearly one full deadline before rank 0;
        # rank 0 may then legitimately consume its own full deadline waiting
        # for the last rank.  Therefore an early peer needs two intervals plus
        # a small handoff allowance.  Anything shorter can time out a peer
        # milliseconds before rank 0 publishes a passing verdict, leaving the
        # remaining ranks to enter a collective without it (measured on the Si
        # MPA P4 path, JID 57909046.123).
        handoff_s = min(2.0, max(0.1, timeout_s * 0.1))
        peer_wait_s = 2.0 * timeout_s + handoff_s
        try:
            payload = client.blocking_key_value_get_bytes(
                f"{prefix}/verdict", int(peer_wait_s * 1000))
            verdict = json.loads(payload.decode("utf-8"))
        except Exception as exc:                            # noqa: BLE001
            records = _snapshot_compile_records(
                client, prefix, n_proc, proc_idx, record)
            verdict = {
                "passed": False,
                "module": module_name,
                "occurrence": occurrence,
                "reason": (
                    f"rank 0 published no verdict within "
                    f"{peer_wait_s:g} seconds "
                    f"({type(exc).__name__})"),
                "records": records,
            }

    s.compile_agreement_checks += 1
    s.compile_agreement_secs += time.monotonic() - t0
    if not verdict.get("passed"):
        raise CompileAgreementError(_format_compile_refusal(verdict))


def _configure_compile_agreement() -> None:
    """Resolve the default-on agreement once, after JAX coordination setup."""
    s = _STATE
    if s.compile_agreement_configured:
        return
    s.compile_agreement_configured = True
    try:
        import jax
        s.n_proc = int(jax.process_count())
        s.proc_idx = int(jax.process_index())
    except Exception:                                      # noqa: BLE001
        s.n_proc = 1
        s.proc_idx = 0

    if s.n_proc <= 1:
        s.compile_agreement_reason = "no-op: process_count=1"
        if s.proc_idx == 0:
            _say("cross-rank compile agreement no-op: process_count=1.")
        return

    # Install the known rank-local shard-offset canonicalization before the
    # mesh warmup can compile ``jit__multi_slice``.  ``ensure_jax_compile_cache``
    # repeats this idempotently later for direct cache callers, but doing it
    # only there would put the new refusal in front of the existing repair.
    if _truthy("LORRAX_JAX_CACHE_SHARD_SLICE", "1"):
        _install_shard_slice_patch()

    from runtime.env_flags import env_bool
    requested = env_bool(
        "LORRAX_JAX_COMPILE_AGREEMENT", True, print_fn=_say)
    if not requested:
        s.compile_agreement_reason = (
            "disabled by LORRAX_JAX_COMPILE_AGREEMENT=0")
        if s.proc_idx == 0:
            _say("*** cross-rank compile agreement DISABLED by "
                 "LORRAX_JAX_COMPILE_AGREEMENT=0. This is an UNSAFE "
                 "bisect-only mode: a rank-divergent compile may hang. ***")
        return

    from jax._src import distributed as _dist
    client = _dist.global_state.client
    if client is None:
        s.compile_agreement_reason = (
            "no-op: jax.distributed coordination client is not initialized")
        if s.proc_idx == 0:
            _say("cross-rank compile agreement no-op: jax.distributed "
                 "coordination client is not initialized.")
        return

    timeout_s = _positive_float_env(
        "LORRAX_JAX_COMPILE_AGREE_TIMEOUT_S",
        _COMPILE_AGREEMENT_TIMEOUT_DEFAULT_S)
    s._compile_client = client
    s.compile_agreement_timeout_s = timeout_s
    s.compile_agreement_enabled = True
    s.compile_agreement_reason = "enabled"
    if s.proc_idx == 0:
        _debug_say(
            "cross-rank compile agreement enabled before backend compile "
            f"with a {timeout_s:g}-second deadline.")


def install_compile_agreement() -> None:
    """Install the compile counter and default-on all-rank module refusal.

    The runtime calls this after ``jax.distributed.initialize`` and before
    mesh warmup.  Direct library/test paths are safe: P=1 or an absent
    coordination client makes agreement an announced no-op, while the
    compile counter remains useful.
    """
    _configure_compile_agreement()
    _install_compile_counter()


def _install_compile_counter() -> None:
    """Count real XLA compiles so the storm is measurable, warm vs cold.

    Raises :class:`_JaxSurfaceUnsupported` when
    :data:`_COMPILE_ENTRY_POINT` is absent, so the caller can report that the
    storm telemetry is OFF rather than leave
    ``compile_cache_stats()['compiles']`` reading a confident 0.  That is a
    refusal, not a compatibility branch: there is no second entry point left
    to silently prefer.
    """
    from jax._src import compiler as _compiler

    if getattr(_compiler, "_lorrax_compile_counter_installed", False):
        return

    name = _COMPILE_ENTRY_POINT
    if getattr(_compiler, name, None) is None:
        raise _JaxSurfaceUnsupported(
            f"jax._src.compiler has no {name} on this jax "
            f"({_jax_generation()}) — no entry point left to count real XLA "
            f"compiles at.  jax 0.5.3 spelled it backend_compile; support for "
            f"that line was dropped when the GPU leg moved to jax 0.7.0.")
    _orig = getattr(_compiler, name)

    def _counting(*args, **kwargs):
        if _STATE.compile_agreement_enabled:
            module = args[1] if len(args) > 1 else kwargs.get("module")
            module_name, key, fingerprint_secs = _compile_module_identity(
                module)
            _STATE.compile_fingerprint_secs += fingerprint_secs
            # JAX may ask host threads to lower independent modules at once.
            # Keep each process's exchange *and backend entry* in one order;
            # otherwise a later local thread can overtake a module whose
            # all-rank agreement just passed.  The slot is global across
            # module names, so another rank choosing a different first module
            # refuses with both names instead of approving both out of order.
            with _STATE._compile_event_lock:
                occurrence = _STATE._compile_sequence
                _STATE._compile_sequence += 1
                _agree_before_module_compile(module_name, key, occurrence)
                t0 = time.monotonic()
                try:
                    return _orig(*args, **kwargs)
                finally:
                    _STATE.compiles += 1
                    _STATE.compile_secs += time.monotonic() - t0
        t0 = time.monotonic()
        try:
            return _orig(*args, **kwargs)
        finally:
            _STATE.compiles += 1
            _STATE.compile_secs += time.monotonic() - t0

    setattr(_compiler, name, _counting)
    _compiler._lorrax_compile_counter_installed = True


def _report() -> None:
    try:
        _report_impl()
    except Exception:
        pass
    try:
        _dump_keys()
    except Exception:
        pass


#: Filename a rank writes under ``LORRAX_JAX_CACHE_KEYDUMP``.  Spelled once,
#: here, because the contract gate globs for it and a launcher that renamed
#: it would give the gate an empty directory to be green about.
def keydump_name(proc_idx: int, n_proc: int) -> str:
    return f"rank{int(proc_idx):03d}_of{int(n_proc):03d}.json"


def _dump_keys() -> None:
    """Write this rank's cache-key set, when ``LORRAX_JAX_CACHE_KEYDUMP`` asks.

    THE POINT.  ``xla_compiles`` and ``vetoed`` are per-rank COUNTS, and the
    defect class this dump exists for is invisible to counts: four ranks
    that each compiled a different program report the same four numbers as
    four ranks that shared one.  What separates them is WHICH keys each rank
    named, so that is what gets written.

    Not the explain log.  ``jax_explain_cache_misses`` prints only the keys
    that MISSED; a run where every rank hits every one of its own private
    entries prints nothing at all on any rank, which is exactly the state
    this dump has to be able to fail on.
    """
    dest = (os.environ.get("LORRAX_JAX_CACHE_KEYDUMP") or "").strip()
    if not dest:
        return
    s = _STATE
    writes = s.write_metrics()
    path = Path(dest)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "proc_idx": s.proc_idx,
        "n_proc": s.n_proc,
        "enabled": s.enabled,
        "dir": s.dir,
        "bound_dir": bound_cache_dir(),
        "xla_compiles": s.compiles,
        "compile_agreement_enabled": s.compile_agreement_enabled,
        "compile_agreement_reason": s.compile_agreement_reason,
        "compile_agreement_checks": s.compile_agreement_checks,
        "compile_fingerprint_secs": s.compile_fingerprint_secs,
        "compile_agreement_secs": s.compile_agreement_secs,
        "probes": s.probes,
        "hits": s.hits,
        "vetoed": s.blocked,
        "n_seen": s.n_seen,
        "n_agreed": s.n_agreed,
        # These are THIS PROCESS'S completed writes.  JAX invokes its
        # persistent-cache writer on process 0 only, so peers correctly
        # report zero rather than duplicating p0's work.
        **writes,
        "is_cache_writer": s.proc_idx == 0,
        "write_scope": "process-local; JAX writes on process 0 only",
        # SORTED, so a reader diffing two ranks' files sees the divergence
        # and not an iteration order.
        "keys": sorted(s.probe_keys),
    }
    # Same tmp+rename discipline as the cache writes themselves: four ranks
    # write into one directory and the gate globs it, so a reader must never
    # be able to observe a half-written file and call it a short key set.
    final = path / keydump_name(s.proc_idx, s.n_proc)
    tmp = path / f".{final.name}.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True))
    os.replace(str(tmp), str(final))


def bound_cache_dir() -> str:
    """The directory JAX's cache object is ACTUALLY bound to, or ``""``.

    NOT the same question as :attr:`_CacheState.dir`, which records the
    directory this module ASKED for.  ``jax._src.compilation_cache._cache``
    is built once, lazily, at the first compile that consults the cache,
    from ``jax_compilation_cache_dir`` AS IT READ THEN — and a later
    ``config.update`` does not rebind it.  So the two can disagree, and
    when they do every symptom is silent: the agreement lists the
    directory we asked for while JAX reads and writes another one, so
    every probe is vetoed, every write lands on a key that already exists
    somewhere else, and the summary line reports a healthy-looking
    ``enabled=True`` with a directory nothing used.  Reporting the bound
    directory is what makes that state visible instead of inferable.
    """
    try:
        from jax._src import compilation_cache as _cc
        cache = getattr(_cc, "_cache", None)
        if cache is None:
            return ""
        path = getattr(cache, "path", None) or getattr(cache, "_path", None)
        return str(path) if path is not None else ""
    except Exception:                                          # noqa: BLE001
        return ""


def _reset_bound_cache_for_atomic_writer(cache_path: Path,
                                         proc_idx: int) -> None:
    """Rebuild any live stock-JAX cache through LORRAX's patched factory.

    Patching ``get_file_cache`` only affects future cache objects.  Even when
    JAX is already bound to the requested path, keeping that object would keep
    stock non-atomic writes and make write telemetry unavailable.  Reset the
    object unconditionally; persistent files at the path are not removed.
    """
    previous = bound_cache_dir()
    if not previous:
        return

    same_path = (os.path.realpath(previous)
                 == os.path.realpath(str(cache_path)))
    try:
        from jax._src import compilation_cache as _cc_rebind
        _cc_rebind.reset_cache()
    except Exception as exc:                                   # noqa: BLE001
        raise RuntimeError(
            "LORRAX patched JAX's cache factory for atomic writes but could "
            f"not reset the already-bound cache object at {previous}; "
            "continuing would silently retain stock non-atomic writes"
        ) from exc

    if proc_idx == 0:
        if same_path:
            _debug_say(
                f"rebuilt JAX's cache object at {cache_path} through the "
                f"atomic LORRAX writer; persistent entries were retained.")
        else:
            _debug_say(
                f"rebound JAX's compile cache from {previous} "
                f"(inherited through JAX_COMPILATION_CACHE_DIR) to "
                f"{cache_path}.  The inherited directory is not "
                f"per-world-size, so the P>1 agreement cannot use it.")


def _report_impl() -> None:
    s = _STATE
    writes = s.write_metrics()
    bound = bound_cache_dir()
    # Only spelled out when it DISAGREES with what we asked for: the
    # agreement, the veto and the writes all key off the asked-for
    # directory, so a disagreement means the cache is inert in a way no
    # other number on this line shows.
    where = "" if (not bound or bound == s.dir) else f" BOUND-ELSEWHERE={bound}"
    if writes["write_metrics_available"]:
        write_receipt = (
            f"cache_writes_local={writes['local_writes']} "
            f"bytes={writes['local_write_bytes']} "
            f"({writes['local_write_secs']:.2f}s; JAX p0-only)  ")
    else:
        write_receipt = (
            "cache_writes_local=unmeasured "
            "(cache off or capped-LRU path; JAX p0-only)  ")
    msg = (f"rank {s.proc_idx}/{s.n_proc} summary: "
           f"xla_compiles={s.compiles} ({s.compile_secs:.2f}s)  "
           f"compile_agreement={s.compile_agreement_checks} "
           f"({s.compile_fingerprint_secs:.3f}s fingerprint + "
           f"{s.compile_agreement_secs:.3f}s exchange; "
           f"{s.compile_agreement_reason})  "
           f"cache_probes={s.probes} hits={s.hits} "
           f"({s.read_secs:.2f}s) vetoed={s.blocked}  "
           f"{write_receipt}"
           f"agreed={s.n_agreed}/{s.n_seen} "
           f"prefetch={s.prefetch_secs:.2f}s  enabled={s.enabled}{where}")
    # A healthy per-rank performance receipt is forensic detail.  Binding to
    # a different directory is a broken agreement contract and remains loud
    # in production.
    (_say if where else _debug_say)(msg)


def compile_cache_stats() -> dict:
    """Snapshot this rank's cache counters; unavailable writes are ``None``."""
    s = _STATE
    writes = s.write_metrics()
    return {
        "enabled": s.enabled, "dir": s.dir, "n_proc": s.n_proc,
        "proc_idx": s.proc_idx, "n_seen": s.n_seen, "n_agreed": s.n_agreed,
        "probes": s.probes, "hits": s.hits, "vetoed": s.blocked,
        "compiles": s.compiles, "compile_secs": s.compile_secs,
        "compile_agreement_configured": s.compile_agreement_configured,
        "compile_agreement_enabled": s.compile_agreement_enabled,
        "compile_agreement_reason": s.compile_agreement_reason,
        "compile_agreement_timeout_s": s.compile_agreement_timeout_s,
        "compile_agreement_checks": s.compile_agreement_checks,
        "compile_fingerprint_secs": s.compile_fingerprint_secs,
        "compile_agreement_secs": s.compile_agreement_secs,
        "read_secs": s.read_secs, "prefetch_secs": s.prefetch_secs,
        **writes,
        "is_cache_writer": s.proc_idx == 0,
        "write_scope": "process-local; JAX writes on process 0 only",
        "bound_dir": bound_cache_dir(),
        "keys": sorted(s.probe_keys),
    }


def _resolve_cache_base_dir() -> tuple[str, str]:
    """Resolve the one persistent-cache owner without inventing global state.

    ``ISDF_JAX_CACHE_DIR`` is the explicit expert control, including its empty
    opt-out.  Otherwise ``LORRAX_RUN_DIR`` scopes reuse to one workflow.  With
    neither set, retain the legacy scratch/home fallback until the separately
    required P=4 default-flip experiment can be run.
    """
    explicit = os.environ.get("ISDF_JAX_CACHE_DIR")
    if explicit is not None:
        return explicit.strip(), "explicit"

    run_dir = os.environ.get("LORRAX_RUN_DIR", "").strip()
    if run_dir:
        return os.path.join(run_dir, ".lorrax_jax_cache"), "LORRAX_RUN_DIR"

    scratch = os.environ.get("SCRATCH", "").strip()
    if scratch:
        return os.path.join(scratch, "lorrax_jax_cache"), "$SCRATCH fallback"

    base_cache = os.environ.get(
        "XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.join(base_cache, "isdf_jax_compilation"), "home fallback"


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def ensure_jax_compile_cache() -> None:
    """Enable the JAX persistent compile cache once per process.

    At ``jax.process_count() == 1`` this simply arms JAX's persistent cache
    at ``{base}/np1/``.

    At ``jax.process_count() > 1`` it additionally installs the LORRAX
    hit/miss AGREEMENT layer described in the module docstring, so that a
    shared cache directory is safe: every rank uses exactly the same set of
    cache entries, so no rank can skip a compile its peers are performing.
    Set ``LORRAX_JAX_CACHE_MULTIPROCESS=0`` to fall back to the scorecard-AG
    behaviour (no cache at all when P > 1).

    ONE caller owns this: ``runtime.initialize_communicator_stack`` step 7.
    Idempotence is not a licence for a second call site — it is what makes a
    stray re-entry through an unlucky import order harmless rather than a
    second, differently-configured cache.  See the module docstring for the
    two non-driver exceptions.
    """
    global _COMPILATION_CACHE_READY
    if _COMPILATION_CACHE_READY:
        return
    _COMPILATION_CACHE_READY = True

    try:
        import jax as _jax
        n_proc = _jax.process_count()
        proc_idx = _jax.process_index()
    except Exception:
        n_proc = 1
        proc_idx = 0
    _STATE.n_proc = n_proc
    _STATE.proc_idx = proc_idx
    _STATE.reset_write_metrics()
    _STATE.set_write_metrics_available(False)

    # Legacy shared caches (pre-AH) can still be on disk with entries whose
    # device binding does not match; JAX warns once per primitive per jit.
    # Non-fatal (JAX recompiles) but noisy.  Suppressed ONLY at P > 1, where
    # any agreed entry that fails to load is already reported loudly by
    # ``_fatal``.  At P == 1 no agreement layer is installed, so this
    # warning is the only observable that distinguishes "cache warm" from
    # "cache rotting" (torn pre-AH write, scratch bit rot) — a blanket
    # filter made healthy and corrupt caches log identically there
    # (QUALITY_PATTERNS #7 addendum; release audit 2026-07-28).
    if n_proc > 1:
        warnings.filterwarnings(
            "ignore",
            message=r"Error reading persistent compilation cache entry .*",
            category=UserWarning,
        )

    # The compile counter goes in on EVERY path, including cache-off, so that
    # "compiles with the cache" and "compiles without it" are the same
    # measurement (scorecard D's storm number, per rank, per run).
    #
    # This except used to be a bare ``pass``, and that is how the counter came
    # to be silently absent on the whole GPU leg for months: its jax-0.9 target
    # does not exist on the 0.5.3 line, the exception was swallowed here, and
    # every run then reported ``xla_compiles=0`` with total confidence.  A
    # counter that is not installed must SAY it is not installed — the number
    # it stops producing is the one the docstring above promises.
    try:
        install_compile_agreement()
        atexit.register(_report)
    except Exception as exc:  # noqa: BLE001
        if proc_idx == 0:
            _say(f"compile-storm telemetry OFF: the XLA compile counter did "
                 f"not install ({type(exc).__name__}: {exc}).  The cache "
                 f"itself is unaffected, but xla_compiles / compile_secs "
                 f"will read 0 no matter what this run compiles — do not "
                 f"read that as a cache hit.")

    cache_dir, cache_source = _resolve_cache_base_dir()
    if n_proc > 1 or not cache_dir:
        # This setting belongs before every early return below.  Otherwise an
        # explicit cache opt-out leaves JAX's process-0 UPDATE / peer READ
        # per-fusion cache active but gives it no real base directory.  Rank 0
        # then fails alone while its peers continue towards a collective.
        _jax.config.update("jax_persistent_cache_enable_xla_caches", "")
    if not cache_dir:  # only the explicit empty/whitespace opt-out reaches here
        if proc_idx == 0:
            _say(f"persistent compile cache OFF (ISDF_JAX_CACHE_DIR=\"\" "
                 f"opt-out). JAX's in-process "
                 f"executable cache remains active. For reuse among "
                 f"sequential drivers in one workflow, set LORRAX_RUN_DIR; "
                 f"for a deliberate restart campaign, set "
                 f"ISDF_JAX_CACHE_DIR to a rank-visible directory.")
        return

    if cache_source != "explicit" and proc_idx == 0:
        # A derived location must be visible in the log (quality-pattern #8).
        label = ("workflow-local" if cache_source == "LORRAX_RUN_DIR"
                 else "legacy fallback")
        _debug_say(
            f"cache dir ({label}): {cache_dir} ({cache_source}; "
            f"ISDF_JAX_CACHE_DIR overrides, \"\" opts out).")

    # ---- back-compat escape hatch: the scorecard-AG refusal --------------
    if n_proc > 1 and not _truthy("LORRAX_JAX_CACHE_MULTIPROCESS", "1"):
        if proc_idx == 0:
            _say(f"DISABLED at {n_proc} processes by "
                 f"LORRAX_JAX_CACHE_MULTIPROCESS=0 (scorecard-AG refusal). "
                 f"Every rank compiles from scratch; correct, ~1 min slower. "
                 f"Would have used {cache_dir}/np{n_proc}.")
        return

    from jax._src import config as _jax_config
    _max_size = int(_jax_config.compilation_cache_max_size.value)
    if not _cache_size_policy(n_proc, _max_size):
        if proc_idx == 0:
            _say("persistent compile cache OFF "
                 "(JAX_COMPILATION_CACHE_MAX_SIZE=0).")
        return
    # ONE directory per world size, shared by every rank (see docstring).
    cache_path = Path(cache_dir).expanduser() / f"np{n_proc}"
    _STATE.dir = str(cache_path)
    try:
        cache_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        if proc_idx == 0:
            _say(f"DISABLED: cannot create {cache_path} ({exc}). "
                 f"Every rank compiles from scratch.")
        return

    try:
        import jax as _jax
        _jax.config.update("jax_compilation_cache_dir", str(cache_path))
        # Keep JAX's standard one-second write threshold (or the user's
        # JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS override).  Forcing zero
        # here made every persistable compilation eligible regardless of its
        # compile time, creating a Lustre-file storm of cheap entries.
        if n_proc > 1:
            # See docstring §4: JAX would otherwise auto-enable XLA's own
            # per-fusion autotune cache in UPDATE(p0)/READ(peers) mode, which
            # desynchronises AutotunerPass' modulo-P work split.
            _jax.config.update("jax_persistent_cache_enable_xla_caches", "")
        # Cache-miss explanations are opt-in via JAX_EXPLAIN_CACHE_MISSES=1
        # only; LORRAX_DEBUG_PRINT deliberately does NOT imply them (the
        # coupling produced 56.6% log spam — io-overhead audit 2026-08-30).
    except Exception as exc:
        if proc_idx == 0:
            _say(f"DISABLED: jax.config.update failed ({exc}).")
        return

    _install_atomic_put_patch()

    # ---- make JAX's cache OBJECT follow the directory we just chose -------
    # MEASURED, Perlmutter 2026-08-07, the centroid deck at P=4: without this
    # the run reports `cache_probes=149 hits=0 vetoed=149 agreed=622/622` and
    # never warms up, on consecutive runs, forever.
    #
    # `jax._src.compilation_cache._cache` is built ONCE, lazily, at the first
    # compile that consults the cache, from `jax_compilation_cache_dir` AS IT
    # READ THEN — and `reset_cache()` is the only way to rebind it; a later
    # `config.update` does not.  Older deployed modulefiles exported
    # `JAX_COMPILATION_CACHE_DIR=$SCRATCH/.jax_cache` into the Shifter
    # container, JAX picked that up at import, and the mesh warm-up in
    # `runtime.initialize_communicator_stack` compiles before this function is
    # reached.  So by the time we get here the cache is already bound to
    # `.jax_cache` (19950 entries, actively written) while everything in THIS
    # file — the directory listing, the agreement, the veto — is about
    # `{base}/np{P}`.
    #
    # The two failure modes that produces are both silent:
    #   * every probe is vetoed, because `_STATE.agreed` was built by listing
    #     a directory JAX is not reading;
    #   * every write lands in `.jax_cache` on a key that is usually already
    #     there, so `LRUCache.put`'s `if cache_path.exists(): return` makes it
    #     a no-op and NOTHING appears to be written anywhere.
    # and the summary line still says `enabled=True`.  At P == 1 there is no
    # agreement patch, so JAX reads `.jax_cache` unimpeded and HITS — which is
    # exactly why the 1-rank leg warms up (12.5 s -> 6.0 s) and the 4-rank leg
    # does not.  The asymmetry read like a P>1 cache policy and was not one.
    #
    # Rebinding rather than adopting the inherited directory: `{base}/np{P}`
    # is per-world-size BY DESIGN (the whole agreement rests on every rank
    # seeing the same set), and `.jax_cache` is one flat directory shared by
    # every world size, so adopting it would put P=1 and P=16 entries in one
    # namespace and hand the agreement a set that changes under it.
    _reset_bound_cache_for_atomic_writer(cache_path, proc_idx)

    if n_proc == 1:
        # There is no all-rank agreement to enforce, but the exit receipt and
        # optional key dump still promise real probe/hit counts.  The old early
        # return left those counters at their initial zeros even on a warm run
        # that deserialized hundreds of executables.  This wrapper delegates
        # every decision unchanged and observes only JAX's actual answer.
        _install_observation_patch()
        _STATE.enabled = True
        return

    # Make the ranks compile the SAME MODULE before making them agree on which
    # keys they may use.  This is not part of the agreement and does not
    # depend on its outcome: the agreement equalises hit/miss for a given key,
    # and `jit__multi_slice` diverges one level below that, by building a
    # different program per rank out of that rank's own shard offsets.  See
    # `_install_shard_slice_patch` for the mechanism and the measurement.
    # It is installed on every P>1 path, including the degraded ones, because
    # fewer distinct modules is never the wrong direction.
    if _truthy("LORRAX_JAX_CACHE_SHARD_SLICE", "1"):
        _install_shard_slice_patch()
    elif proc_idx == 0:
        _say("LORRAX_JAX_CACHE_SHARD_SLICE=0: JAX's rank-dependent "
             "ArrayImpl._multi_slice is left in place.  Expect one "
             "jit__multi_slice cache key PER RANK — this is the red twin of "
             "the shard-slice canonicalization, not a supported mode.")

    # NOTE for anyone re-reading the P>1 path.  There used to be a fifth
    # compatibility shim here: a whole-cache degradation for a jax whose
    # ``get_executable_and_time`` has no ``executable_devices`` parameter.
    # Sharing a cache across processes needs the reading rank to bind the
    # deserialized executable to ITS OWN devices, and jax 0.5.3 had no way to
    # be told — so a process-invariant key was ACTIVELY HARMFUL there: rank 1
    # named rank 0's entry, fetched it, and died loading it (MEASURED on one
    # node, 2 GPUs, container 25.04, warm: rank 0 hit 9/9, rank 1 raised
    # ``XlaRuntimeError: INVALID_ARGUMENT: Device assignment ... does not have
    # any local devices`` and ``_fatal`` aborted the job, rc 70).
    #
    # The supported 0.9.1 wheel HAS the parameter (the historical 0.7.0
    # container also measured 4 parameters with ``executable_devices`` named),
    # so the branch was unreachable and is deleted rather than left as a
    # permanent compatibility layer for a version being abandoned.  The
    # condition itself is not unguarded: ``runtime.jax_support`` requires
    # ``compilation_cache.get_executable_and_time`` to take 4 parameters and
    # refuses at startup, by name, on a jax that does not.

    # A process-invariant key is load-bearing for SAFETY, not just for the
    # win: without it process 0 hits the entries it wrote while every peer
    # computes a different key, misses, and compiles — which is precisely the
    # divergent hit/miss pattern that deadlocks XLA:GPU's collective autotune.
    # (MEASURED on CPU: rank 0 hits 7/7, ranks 1-3 hit 0/7 and compile 7.)
    # So if we cannot install it, the cache must not be USED at all.
    invariant_key = False
    if _truthy("LORRAX_JAX_CACHE_INVARIANT_KEY", "1"):
        try:
            _install_invariant_key_patch()
            invariant_key = True
        except Exception as exc:
            _say(f"rank {proc_idx}: could not make the cache key "
                 f"process-invariant ({type(exc).__name__}: {exc}).")
    if not invariant_key and not _truthy("LORRAX_JAX_CACHE_NO_AGREE"):
        _STATE.agreed = frozenset()
        _install_agreement_patch()   # veto everything -> symmetric miss
        if proc_idx == 0:
            _say(f"DEGRADED TO CACHE-OFF at {n_proc} processes: the cache key "
                 f"is not process-invariant here, so process 0 would hit "
                 f"entries its peers cannot even name — the scorecard-AG "
                 f"divergence.  Every rank compiles from scratch, which is "
                 f"correct.  (Set LORRAX_JAX_CACHE_INVARIANT_KEY=1, the "
                 f"default, to get the cache back.)")
        return

    # ---------------- P > 1: the hit/miss agreement -----------------------
    if _truthy("LORRAX_JAX_CACHE_NO_AGREE"):
        _STATE.enabled = True
        if proc_idx == 0:
            _say("*** LORRAX_JAX_CACHE_NO_AGREE=1: naive shared directory, "
                 "NO hit/miss agreement.  This is the DEADLOCK REPRODUCER, "
                 "not a supported mode. ***")
        return

    t0 = time.monotonic()
    timeout_s = float(_int_env("LORRAX_JAX_CACHE_AGREE_TIMEOUT_S", 300))
    try:
        n_seen, agreed = _agree_on_entries(
            cache_path, n_proc, proc_idx, timeout_s)
    except BaseException as exc:  # noqa: BLE001 - never let this hang a run
        _STATE.agreed = frozenset()
        _STATE.n_seen = 0
        _STATE.n_agreed = 0
        _install_agreement_patch()   # veto everything -> symmetric miss
        tail = (f"Entries are still being written to {cache_path} for next "
                f"time.")
        if isinstance(exc, _KeyEnvMismatch):
            # Writing here would populate a SECOND key space that no
            # correctly-launched run can ever hit — pure bloat.  Turn the
            # persistent cache off outright (nothing has compiled yet).
            try:
                import jax as _jax
                _jax.config.update("jax_compilation_cache_dir", None)
                tail = "Nothing is being written either (it would be unusable)."
            except Exception:
                pass
        _say(f"rank {proc_idx}: DEGRADED TO CACHE-OFF — the hit/miss "
             f"agreement failed ({type(exc).__name__}: {exc}). Every rank "
             f"compiles from scratch this run, which is correct and slower. "
             f"{tail}")
        return

    _STATE.n_seen = n_seen
    _STATE.n_agreed = len(agreed)
    _STATE.agreed = agreed
    _STATE.enabled = True
    _install_agreement_patch()

    if _truthy("LORRAX_JAX_CACHE_PREFETCH", _PREFETCH_DEFAULT):
        _STATE.prefetch_secs = _prefetch_agreed(
            cache_path, agreed,
            _int_env("LORRAX_JAX_CACHE_PREFETCH_THREADS", 16))

    dropped = n_seen - len(agreed)
    if proc_idx == 0:
        _debug_say(
            f"ARMED at {n_proc} processes, shared dir {cache_path} "
            f"({n_seen} entries advertised, {len(agreed)} agreed by all "
            f"ranks; agree+prefetch {time.monotonic() - t0:.2f}s of which "
            f"prefetch {_STATE.prefetch_secs:.2f}s).")
        if dropped:
            _say(f"*** {dropped} entr{'y' if dropped == 1 else 'ies'} DROPPED "
                 f"— at least one rank could not see them, so NO rank will "
                 f"use them and every rank will compile those modules.  This "
                 f"is the agreement doing its job (a divergent hit/miss "
                 f"pattern is what deadlocks XLA:GPU autotuning). ***")
        if n_seen == 0:
            _debug_say(
                "cold cache: nothing to reuse this run; process 0 will "
                "populate it for the next one.")
