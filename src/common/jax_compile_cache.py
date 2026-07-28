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
"""
from __future__ import annotations

import atexit
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
            "(XLA_FLAGS / jaxlib version / jax cache config), so they would "
            "compute different cache keys for the same module and their "
            "hit/miss patterns would diverge — the scorecard-AG deadlock. "
            "Make the environment identical on every rank (a rank-0-only "
            "XLA_FLAGS for an HLO dump is the usual cause)")

    agreed = frozenset(k for i, k in enumerate(keys) if _bit(final, i))
    return len(keys), agreed


def _single_stripe(cache_path: Path) -> None:
    """Ask Lustre for one stripe per cache entry (best effort, rank 0, once).

    Cache entries are TINY — 140 of them are 876 kB at 606 centroids / P=16 —
    and reading them is pure per-file latency, not bandwidth.  A multi-stripe
    default multiplies the RPCs per open for no benefit at all at this size.
    Only affects files created afterwards, and only on Lustre; anywhere else
    the command is missing or fails and we move on.
    """
    import shutil
    import subprocess

    if not shutil.which("lfs"):
        return
    try:
        subprocess.run(["lfs", "setstripe", "-c", "1", str(cache_path)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=20, check=False)
    except Exception:
        pass


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


def _install_agreement_patch() -> None:
    """Answer every persistent-cache lookup from the agreed set."""
    from jax._src import compilation_cache as _cc

    if getattr(_cc, "_lorrax_agreement_installed", False):
        return
    _orig_get = _cc.get_executable_and_time
    _orig_in_cache = _cc.is_executable_in_cache

    def _agreed_get(cache_key, compile_options, backend, executable_devices):
        _STATE.probes += 1
        if cache_key not in _STATE.agreed:
            _STATE.blocked += 1
            return None, None
        t0 = time.monotonic()
        try:
            executable, compile_time = _orig_get(
                cache_key, compile_options, backend, executable_devices)
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

    def _atomic_get_file_cache(path: str):
        cache = _AtomicLRUCache(
            path, max_size=_cfg.compilation_cache_max_size.value)
        if _cfg.compilation_cache_check_contents.value:
            return _cc.VerificationCache(cache), path
        return cache, path

    _cc.get_file_cache = _atomic_get_file_cache
    _cc._lorrax_atomic_put_installed = True


def _install_compile_counter() -> None:
    """Count real XLA compiles so the storm is measurable, warm vs cold."""
    from jax._src import compiler as _compiler

    if getattr(_compiler, "_lorrax_compile_counter_installed", False):
        return
    _orig = _compiler.backend_compile_and_load

    def _counting(*args, **kwargs):
        t0 = time.monotonic()
        try:
            return _orig(*args, **kwargs)
        finally:
            _STATE.compiles += 1
            _STATE.compile_secs += time.monotonic() - t0

    _compiler.backend_compile_and_load = _counting
    _compiler._lorrax_compile_counter_installed = True


def _report() -> None:
    try:
        _report_impl()
    except Exception:
        pass


def _report_impl() -> None:
    s = _STATE
    _say(f"rank {s.proc_idx}/{s.n_proc} summary: xla_compiles={s.compiles} "
         f"({s.compile_secs:.2f}s)  cache_probes={s.probes} hits={s.hits} "
         f"({s.read_secs:.2f}s) vetoed={s.blocked}  "
         f"agreed={s.n_agreed}/{s.n_seen} prefetch={s.prefetch_secs:.2f}s  "
         f"enabled={s.enabled}")


def compile_cache_stats() -> dict:
    """Snapshot of this rank's compile/cache counters (for tests + probes)."""
    s = _STATE
    return {
        "enabled": s.enabled, "dir": s.dir, "n_proc": s.n_proc,
        "proc_idx": s.proc_idx, "n_seen": s.n_seen, "n_agreed": s.n_agreed,
        "probes": s.probes, "hits": s.hits, "vetoed": s.blocked,
        "compiles": s.compiles, "compile_secs": s.compile_secs,
        "read_secs": s.read_secs, "prefetch_secs": s.prefetch_secs,
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
    try:
        _install_compile_counter()
        atexit.register(_report)
    except Exception:
        pass

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
        fresh = not cache_path.exists()
        cache_path.mkdir(parents=True, exist_ok=True)
        if fresh and proc_idx == 0:
            _single_stripe(cache_path)
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

    if n_proc == 1:
        _STATE.enabled = True
        return

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
