"""JAX persistent compile cache — SAFE and EFFECTIVE at ``process_count() > 1``.

Every LORRAX driver that does a meaningful amount of ``jax.jit``
compile (gw.gw_jax, psp.run_nscf, centroid.kmeans_isdf, bse.bse_jax,
bandstructure.htransform, ...) should call
:func:`ensure_jax_compile_cache` near the top of its ``main()`` — right
after ``jax.distributed.initialize`` (i.e. after ``runtime.bootstrap()``)
and BEFORE any jit.  Calling it late means the expensive compiles that
happen early in the pipeline miss the cache.

Knobs (all optional):

  ``ISDF_JAX_CACHE_DIR=/some/path``    — override cache location.
  ``ISDF_JAX_CACHE_DIR=""``             — opt out entirely.
  ``LORRAX_JAX_CACHE_MULTIPROCESS=0``   — restore the scorecard-AG refusal
                                          (no cache at all when P > 1).
  ``LORRAX_JAX_CACHE_AGREE_TIMEOUT_S``  — agreement timeout, default 300.
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
  ``LORRAX_JAX_CACHE_EXPLAIN=1``        — turn on ``jax_explain_cache_misses``.
  ``LORRAX_JAX_CACHE_KEYDUMP=<dir>``    — every rank writes the SET of
                                          persistent-cache keys it asked
                                          about to ``<dir>/rank{i}_of{N}.json``
                                          at exit.  This is what makes the
                                          key-symmetry invariant falsifiable:
                                          ``LORRAX_JAX_CACHE_EXPLAIN`` names
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

Default location (``ISDF_JAX_CACHE_DIR`` unset) is
``$SCRATCH/lorrax_jax_cache`` when ``$SCRATCH`` is set, else
``$XDG_CACHE_HOME``/``~/.cache`` + ``/isdf_jax_compilation`` — the resolved
default is announced once at init.  Entries live in ``{base}/np{N_proc}/``
— ONE directory shared by every rank of a world size (the old ``rank{i}/``
partitioning is gone; see below).

**Why ``$SCRATCH`` comes first.**  One entry is one small file, and a
populated world size is several hundred of them (301 for the gw fixture at
P=8, 886 measured for a big deck), so a default under ``/home1`` eats inodes
on the filesystem whose *inode* quota locks you out of the machine (ADVICE
section 2).  The cache is enabled by default, so the safe location has to be
the code's default rather than shell-alias advice.  Correctness does not
depend on this — the directory only has to be visible to every rank, which
``$SCRATCH`` and ``$WORK`` both are and ``/tmp`` is NOT.

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

4. **JAX's auto-enabled XLA sub-caches are turned OFF at P > 1.**  When the
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
import json
import os
import sys
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

# Parallel page-cache prefetch of the agreed entries (see _prefetch_agreed).
# ON: at 606 centroids / P=16 the SERIAL reads of 169 entries cost 29 s on one
# rank and 8.8 s on another, against the ~4.5 s of XLA compile they replace —
# i.e. without this the cache is a net LOSS on a cold-read CPU run.  876 kB of
# payload, so it is pure per-file Lustre latency under 16-way concurrency.
_PREFETCH_DEFAULT = "1"


class _CacheState:
    """Per-process bookkeeping for the agreed cache (also the atexit report)."""

    def __init__(self) -> None:
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
        self.agreed: frozenset[str] = frozenset()
        # THE KEY SET.  Every persistent-cache key this rank asked about,
        # hit or miss.  The counters above cannot express the invariant the
        # cache contract is actually about: two ranks can both report
        # ``xla_compiles=0 vetoed=0`` while asking about DIFFERENT programs,
        # which is the state that precedes the collective-compile deadlock.
        # A set of keys is the only observable that separates those.
        self.probe_keys: set[str] = set()


_STATE = _CacheState()


class _KeyEnvMismatch(RuntimeError):
    """The ranks would compute different cache keys — see _key_env_fingerprint."""


def _truthy(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() not in (
        "", "0", "false", "no", "off")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip())
    except (TypeError, ValueError):
        return default


def _say(msg: str) -> None:
    print(f"  [compile-cache] {msg}", flush=True)


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


def _install_agreement_patch() -> None:
    """Answer every persistent-cache lookup from the agreed set."""
    from jax._src import compilation_cache as _cc

    if getattr(_cc, "_lorrax_agreement_installed", False):
        return
    _orig_get = _cc.get_executable_and_time
    _orig_in_cache = _cc.is_executable_in_cache

    # ``get_executable_and_time`` is ``(cache_key, compile_options, backend,
    # executable_devices)`` on both supported jaxes (0.7.0 container, 0.9.1
    # wheel — MEASURED).  The arity PROBE and its announcement, which existed
    # to report jax 0.5.3's 3-parameter form, are gone; ``runtime.jax_support``
    # asserts the 4 at startup.
    #
    # ``*passthrough`` is NOT a leftover compatibility branch and is kept
    # deliberately: this wrapper reads the cache key and nothing else, so
    # naming three arguments it never interprets would be a claim about their
    # meaning that this file has no reason to make.  It forwards them
    # untouched, which is exact for any arity.
    def _agreed_get(cache_key, *passthrough):
        _STATE.probes += 1
        _STATE.probe_keys.add(cache_key)
        if cache_key not in _STATE.agreed:
            _STATE.blocked += 1
            return None, None
        t0 = time.monotonic()
        try:
            executable, compile_time = _orig_get(cache_key, *passthrough)
        except BaseException as exc:  # noqa: BLE001 - deliberate
            _fatal(cache_key, f"{type(exc).__name__}: {exc}")
            return None, None
        finally:
            _STATE.read_secs += time.monotonic() - t0
        if executable is None:
            _fatal(cache_key, "entry disappeared between agreement and read")
            return None, None
        _STATE.hits += 1
        return executable, compile_time

    def _agreed_in_cache(backend, cache_key):
        _STATE.probe_keys.add(cache_key)
        if cache_key not in _STATE.agreed:
            return False
        return _orig_in_cache(backend, cache_key)

    _cc.get_executable_and_time = _agreed_get
    _cc.is_executable_in_cache = _agreed_in_cache
    _cc._lorrax_agreement_installed = True


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
                return super().put(key, val)
            if not key:
                raise ValueError("key cannot be empty")
            final = self.path / f"{key}{_lru._CACHE_SUFFIX}"
            if final.exists():
                return
            tmp = self.path / (f".{key}{_lru._CACHE_SUFFIX}.tmp."
                               f"{os.getpid()}.{uuid.uuid4().hex[:8]}")
            try:
                tmp.write_bytes(val)
                os.replace(str(tmp), str(final))
            except BaseException:
                try:
                    os.unlink(str(tmp))
                except OSError:
                    pass
                raise

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
        if (_verification_cache is not None and _check_contents is not None
                and _check_contents.value):
            return _verification_cache(cache), path
        return cache, path

    _cc.get_file_cache = _atomic_get_file_cache
    _cc._lorrax_atomic_put_installed = True


#: The ``jax._src.compiler`` entry point a real XLA compile goes through.
#: Both supported jaxes route compiles through ``backend_compile_and_load``
#: (MEASURED: present on the 0.7.0 container and the 0.9.1 wheel), and it
#: itself calls ``backend_compile`` — so exactly ONE of them may be patched or
#: every compile is counted twice.
#:
#: This used to be a two-element preference tuple with a fallback to
#: ``backend_compile``, because jax 0.5.3 had no ``backend_compile_and_load``
#: at all.  The fallback went with 0.5.3.
_COMPILE_ENTRY_POINT = "backend_compile_and_load"


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
    path = Path(dest)
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "proc_idx": s.proc_idx,
        "n_proc": s.n_proc,
        "enabled": s.enabled,
        "dir": s.dir,
        "bound_dir": bound_cache_dir(),
        "xla_compiles": s.compiles,
        "probes": s.probes,
        "hits": s.hits,
        "vetoed": s.blocked,
        "n_seen": s.n_seen,
        "n_agreed": s.n_agreed,
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


def _report_impl() -> None:
    s = _STATE
    bound = bound_cache_dir()
    # Only spelled out when it DISAGREES with what we asked for: the
    # agreement, the veto and the writes all key off the asked-for
    # directory, so a disagreement means the cache is inert in a way no
    # other number on this line shows.
    where = "" if (not bound or bound == s.dir) else f" BOUND-ELSEWHERE={bound}"
    _say(f"rank {s.proc_idx}/{s.n_proc} summary: xla_compiles={s.compiles} "
         f"({s.compile_secs:.2f}s)  cache_probes={s.probes} hits={s.hits} "
         f"({s.read_secs:.2f}s) vetoed={s.blocked}  "
         f"agreed={s.n_agreed}/{s.n_seen} prefetch={s.prefetch_secs:.2f}s  "
         f"enabled={s.enabled}{where}")


def compile_cache_stats() -> dict:
    """Snapshot of this rank's compile/cache counters (for tests + probes)."""
    s = _STATE
    return {
        "enabled": s.enabled, "dir": s.dir, "n_proc": s.n_proc,
        "proc_idx": s.proc_idx, "n_seen": s.n_seen, "n_agreed": s.n_agreed,
        "probes": s.probes, "hits": s.hits, "vetoed": s.blocked,
        "compiles": s.compiles, "compile_secs": s.compile_secs,
        "read_secs": s.read_secs, "prefetch_secs": s.prefetch_secs,
        "bound_dir": bound_cache_dir(),
        "keys": sorted(s.probe_keys),
    }


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

    Idempotent: safe to call multiple times, and from several drivers in the
    same process.
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
        _install_compile_counter()
        atexit.register(_report)
    except Exception as exc:  # noqa: BLE001
        if proc_idx == 0:
            _say(f"compile-storm telemetry OFF: the XLA compile counter did "
                 f"not install ({type(exc).__name__}: {exc}).  The cache "
                 f"itself is unaffected, but xla_compiles / compile_secs "
                 f"will read 0 no matter what this run compiles — do not "
                 f"read that as a cache hit.")

    cache_dir = os.environ.get("ISDF_JAX_CACHE_DIR")
    default_note = None
    if cache_dir is None:
        # Default chain (announced once, rank 0, below): $SCRATCH first.
        # A populated world size is several hundred small files (886
        # measured for a big deck), and this module's own docstring has
        # always said the default under /home1 eats the inode quota that
        # locks a user out of the machine.  With the cache now enabled by
        # default at all P, the safe location must be the CODE's default,
        # not a docstring plea (release audit 2026-07-28).
        scratch = os.environ.get("SCRATCH", "").strip()
        if scratch:
            cache_dir = os.path.join(scratch, "lorrax_jax_cache")
            default_note = "$SCRATCH default"
        else:
            base_cache = os.environ.get(
                "XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            cache_dir = os.path.join(base_cache, "isdf_jax_compilation")
            default_note = ("no $SCRATCH in the environment — falling back "
                            "under the home filesystem; several hundred "
                            "small files per world size, WATCH THE INODE "
                            "QUOTA or set ISDF_JAX_CACHE_DIR")
    else:
        # A whitespace-only value is the ""-opt-out, not a real path.
        cache_dir = cache_dir.strip()
    if not cache_dir:  # explicit opt-out via ISDF_JAX_CACHE_DIR=""
        # Announce the opt-out (workstream AT).  This used to be silent, and
        # the production harnesses kept exporting ISDF_JAX_CACHE_DIR="" for a
        # reason that STOPPED being true when the AH agreement layer landed
        # ("the shared cache deadlocks at P>1" — it no longer can, by
        # construction).  An env var that silently disables a measured win is
        # exactly the env-coupled-behavior class (quality-pattern #8): say it
        # once, on rank 0, so a stale harness line is visible in every log.
        if proc_idx == 0:
            _say(f"persistent compile cache OFF (ISDF_JAX_CACHE_DIR=\"\" "
                 f"opt-out): every rank compiles every module, every run. "
                 f"Safe at any P since the AH agreement layer; to enable, "
                 f"unset ISDF_JAX_CACHE_DIR (the default resolves under "
                 f"$SCRATCH) or point it at any rank-visible directory.")
        return

    if default_note is not None and proc_idx == 0:
        # Announce the resolved default exactly once — a location the user
        # never chose must be visible in the log (quality-pattern #8).
        _say(f"cache dir (default): {cache_dir} ({default_note}; override "
             f"with ISDF_JAX_CACHE_DIR, \"\" opts out).")

    # ---- back-compat escape hatch: the scorecard-AG refusal --------------
    if n_proc > 1 and not _truthy("LORRAX_JAX_CACHE_MULTIPROCESS", "1"):
        if proc_idx == 0:
            _say(f"DISABLED at {n_proc} processes by "
                 f"LORRAX_JAX_CACHE_MULTIPROCESS=0 (scorecard-AG refusal). "
                 f"Every rank compiles from scratch; correct, ~1 min slower. "
                 f"Would have used {cache_dir}/np{n_proc}.")
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
        _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        if n_proc > 1:
            # See docstring §4: JAX would otherwise auto-enable XLA's own
            # per-fusion autotune cache in UPDATE(p0)/READ(peers) mode, which
            # desynchronises AutotunerPass' modulo-P work split.
            _jax.config.update("jax_persistent_cache_enable_xla_caches", "")
        if _truthy("LORRAX_JAX_CACHE_EXPLAIN"):
            _jax.config.update("jax_explain_cache_misses", True)
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
    # `config.update` does not.  The `lorrax_J070` modulefile exports
    # `JAX_COMPILATION_CACHE_DIR=$SCRATCH/.jax_cache` and passes it into the
    # Shifter container (`0.1.0.lua:312` and the `--env=` at `:252`), JAX picks
    # that up at import, and the mesh warm-up in
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
    _prev_bound = bound_cache_dir()
    if _prev_bound and (os.path.realpath(_prev_bound)
                        != os.path.realpath(str(cache_path))):
        try:
            from jax._src import compilation_cache as _cc_rebind
            _cc_rebind.reset_cache()
        except Exception as exc:                               # noqa: BLE001
            if proc_idx == 0:
                _say(f"could not rebind JAX's compile cache off "
                     f"{_prev_bound} ({type(exc).__name__}: {exc}); it will "
                     f"keep reading and writing there while this module's "
                     f"agreement is about {cache_path}.  Expect hits=0 at "
                     f"P>1.  Unset JAX_COMPILATION_CACHE_DIR to avoid it.")
        else:
            if proc_idx == 0:
                _say(f"rebound JAX's compile cache from {_prev_bound} "
                     f"(inherited, usually JAX_COMPILATION_CACHE_DIR from the "
                     f"module) to {cache_path}.  The inherited directory is "
                     f"not per-world-size, so the P>1 agreement cannot be "
                     f"taken over it.")

    if n_proc == 1:
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
    # Both supported jaxes HAVE the parameter (0.7.0 container measured at 4
    # parameters with ``executable_devices`` named, same as the 0.9.1 wheel),
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
        _say(f"ARMED at {n_proc} processes, shared dir {cache_path} "
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
            _say("cold cache: nothing to reuse this run; process 0 will "
                 "populate it for the next one.")
